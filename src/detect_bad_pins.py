import cv2
import numpy as np
import os
import glob
import math

# ================= CONFIGURATION =================
SOURCE_DIR = r'./runs/detect/predict'
OUTPUT_DIR = r'./analyzed_pins'
TARGET_HEX = "052aff"

# --- LOCAL DIFFERENCE THRESHOLDS (Percentage) ---
# Width Logic: WIDE pins fail easily (strict), THIN pins fail less likely (loose)
MAX_DIFF_W_WIDE_PER = 15.0    # Width "wide" threshold (Strict: Internal damage + Color Fail if > this)
MAX_DIFF_W_THIN_PER = 15.0   # Width "thin" threshold (Loose: Only Color Fail if thinner than -this)

# Height Logic: SHORT pins fail easily (strict), LONG pins fail less likely (loose)
MAX_DIFF_H_SHORT_PER = 5.5   # Height "short" threshold (Strict: Internal damage + Color Fail if > this)
MAX_DIFF_H_LONG_PER = 10.0   # Height "long" threshold (Loose: Only Color Fail if taller than this)

MAX_DIFF_A_PER = 20.0        # Area symmetric
MAX_DIFF_AR_PER = 100        # Aspect Ratio (Height/Width) symmetric

# --- ASPECT RATIO OUTLIER DETECTION ---
# Detects crooked pins by comparing aspect ratio to the median of all pins in the image
# Asymmetric thresholds: crooked pins typically appear shorter & wider
MAX_AR_OUTLIER_SHORT_WIDE_PER = 14.0   # Strict: pins with lower AR (shorter & wider) than median
MAX_AR_OUTLIER_TALL_THIN_PER = 30.0   # Loose: pins with higher AR (taller & thinner) than median
MIN_PINS_FOR_OUTLIER = 3              # Minimum number of pins needed to perform outlier detection
# =================================================

REASON_MAP = {
    'Width': 'width',
    'Height': 'length',
    'Area': 'area',
    'AspectRatio': 'aspect_ratio',
    'AspectRatioOutlier': 'crooked'
}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def get_closest_to_median_index(areas):
    if not areas:
        return None
    median = np.median(areas)
    return np.argmin(np.abs(np.array(areas) - median))

def replace_blue_with_red(img, bbox, lower_bound, upper_bound, pad=4, red=(0,0,255),
                          close_kernel=(3,3), dilate_kernel=(5,5), dilate_iters=1):
    """
    Replace all blue-outline pixels in bbox with red.

    - Uses the same color bounds (lower_bound/upper_bound) to find blue pixels inside an expanded ROI.
    - Performs a morphological close to bridge anti-aliased gaps, then a dilation to ensure the
      entire outline (including overlapping edges) is covered.
    - Does NOT change detection masks/contours used elsewhere; it's purely per-ROI recolor.
    """
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return

    # 1) base mask: exact blue pixels in ROI using your existing bounds
    mask_roi = cv2.inRange(roi, lower_bound, upper_bound)

    # If nothing obvious found, still proceed (may be tiny); but guard:
    if cv2.countNonZero(mask_roi) == 0:
        # Try a tiny extra in-range tolerance step (non-destructive, only locally)
        # Build a looser mask in HSV to catch slightly different blues without affecting detection.
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # convert target_bgr from outer scope is not available here; skip — keep conservative
        # If nothing found, continue with mask_roi (will do nothing)
        pass

    # 2) morphological close to fill small gaps/anti-alias holes (kernel small)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    mask_closed = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, ck, iterations=1)

    # 3) dilate to ensure full coverage of the outline and overlapping edges
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel)
    mask_dilated = cv2.dilate(mask_closed, dk, iterations=dilate_iters)

    # 4) Apply mask: set any selected pixel to red
    roi[mask_dilated > 0] = red
    img[y0:y1, x0:x1] = roi


def process_directory(source_dir=None, output_dir=None):
    src = source_dir if source_dir else SOURCE_DIR
    out = output_dir if output_dir else OUTPUT_DIR

    if not os.path.exists(src):
        print(f"Error: Source directory '{src}' does not exist.")
        return

    os.makedirs(out, exist_ok=True)

    types = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_files = []
    for files in types:
        image_files.extend(glob.glob(os.path.join(src, files)))

    print(f"Found {len(image_files)} images in {src}...")

    target_bgr = hex_to_bgr(TARGET_HEX)
    tolerance = 15
    lower_bound = np.array([max(0, c - tolerance) for c in target_bgr])
    upper_bound = np.array([min(255, c + tolerance) for c in target_bgr])

    # Local comparison thresholds (in decimals)
    percentage_thresholds = {
        'Width_wide': MAX_DIFF_W_WIDE_PER / 100.0,   # Strict
        'Width_thin': MAX_DIFF_W_THIN_PER / 100.0,   # Loose
        'Height_short': MAX_DIFF_H_SHORT_PER / 100.0, # Strict
        'Height_long': MAX_DIFF_H_LONG_PER / 100.0,   # Loose
        'Area': MAX_DIFF_A_PER / 100.0,
        'AspectRatio': MAX_DIFF_AR_PER / 100.0,
    }

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"-> Processing: {filename}")

        img = cv2.imread(img_path)
        if img is None:
            print("   Warning: Could not read image. Skipping.")
            continue

        mask = cv2.inRange(img, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_data = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            standard_width = min(w, h)
            standard_height = max(w, h)
            aspect_ratio = standard_height / standard_width if standard_width > 0 else 0

            box_data.append({
                'Width': standard_width,
                'Height': standard_height,
                'Area': w * h,
                'AspectRatio': aspect_ratio,
                'cnt': cnt,
                'bbox': (x, y, w, h),
                'is_damaged': False
            })

        if len(box_data) < 2:
            print("   Not enough pins detected for comparison. Skipping.")
            cv2.imwrite(os.path.join(out, filename), img)
            continue

        # Baseline logic
        areas = [b['Area'] for b in box_data]
        baseline_index = get_closest_to_median_index(areas)

        # --- ASPECT RATIO OUTLIER DETECTION ---
        # Calculate median aspect ratio for the image
        aspect_ratios = [b['AspectRatio'] for b in box_data]
        median_ar = np.median(aspect_ratios)
        ar_outlier_short_wide_th = MAX_AR_OUTLIER_SHORT_WIDE_PER / 100.0  # Strict
        ar_outlier_tall_thin_th = MAX_AR_OUTLIER_TALL_THIN_PER / 100.0    # Loose

        # Mark aspect ratio outliers (only if we have enough pins)
        if len(box_data) >= MIN_PINS_FOR_OUTLIER:
            for pin in box_data:
                if median_ar > 0:
                    ar_diff = pin['AspectRatio'] - median_ar
                    ar_deviation = abs(ar_diff) / median_ar

                    # Asymmetric thresholds: stricter for short/wide, looser for tall/thin
                    if ar_diff < 0:  # Pin is shorter & wider than median (likely crooked)
                        pin['is_ar_outlier'] = ar_deviation > ar_outlier_short_wide_th
                    else:  # Pin is taller & thinner than median
                        pin['is_ar_outlier'] = ar_deviation > ar_outlier_tall_thin_th
                else:
                    pin['is_ar_outlier'] = False
        else:
            for pin in box_data:
                pin['is_ar_outlier'] = False

        # Sort top-to-bottom for consistent neighbor comparison
        box_data.sort(key=lambda x: x['bbox'][1])

        damage_found = False
        image_failure_metrics = set()

        # Find baseline pin index after sorting (match by area)
        baseline_pin_area = areas[baseline_index]
        # In case of duplicates, pick the first matching entry
        baseline_pin_index = next(i for i, b in enumerate(box_data) if b['Area'] == baseline_pin_area)

        reference_pin_data = box_data[baseline_pin_index]

        metrics_to_check = ['Width', 'Height', 'Area', 'AspectRatio']

        for i in range(len(box_data)):
            current_pin = box_data[i]

            # --- CHECK ASPECT RATIO OUTLIER FIRST ---
            if current_pin['is_ar_outlier']:
                current_pin['is_damaged'] = True
                damage_found = True
                image_failure_metrics.add('AspectRatioOutlier')

                # Replace blue pixels with red in this pin's bbox (no drawing of boxes)
                replace_blue_with_red(img, current_pin['bbox'], lower_bound, upper_bound)
                x, y, w, h = current_pin['bbox']
                label = f"FAIL (CROOKED)"
                cv2.putText(img, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue  # Skip other checks for this pin

            # Skip baseline pin itself
            if i == baseline_pin_index:
                continue

            # --- Determine Reference Pin (R) with look-backwards skipping damaged pins ---
            ref_pin = None
            if i > 0 and not box_data[i-1]['is_damaged']:
                ref_pin = box_data[i-1]
            if ref_pin is None and i > 0:
                for j in range(i - 2, -1, -1):
                    if not box_data[j]['is_damaged']:
                        ref_pin = box_data[j]
                        break
            if ref_pin is None:
                ref_pin = reference_pin_data

            internal_reasons = []  # metrics that cause internal damaged marking (excluded from refs)
            color_reasons = []     # metrics that cause visual (colored) FAIL

            for metric in metrics_to_check:

                # --- HEIGHT LOGIC (Existing) ---
                if metric == 'Height':
                    short_th = percentage_thresholds['Height_short']
                    long_th = percentage_thresholds['Height_long']
                    curr = current_pin['Height']
                    refv = ref_pin['Height']
                    if refv == 0: continue

                    signed_diff = (curr - refv) / refv  # negative => shorter, positive => longer

                    # Internal marking if absolute diff > short threshold
                    if abs(signed_diff) > short_th:
                        internal_reasons.append(metric)

                    # Visual coloring rules:
                    #  - shorter than allowed => color (fail)
                    #  - only color longer pins if they exceed the LONG threshold
                    if signed_diff < -short_th:
                        color_reasons.append(metric)
                    elif signed_diff > long_th:
                        color_reasons.append(metric)
                    continue

                # --- WIDTH LOGIC (New) ---
                if metric == 'Width':
                    wide_th = percentage_thresholds['Width_wide'] # Strict (e.g. 5.5%)
                    thin_th = percentage_thresholds['Width_thin'] # Loose (e.g. 15.0%)
                    curr = current_pin['Width']
                    refv = ref_pin['Width']
                    if refv == 0: continue

                    signed_diff = (curr - refv) / refv # negative => thinner, positive => wider

                    # Internal marking if absolute diff > WIDE (strict) threshold
                    # This prevents slightly wide or slightly thin pins from being used as references
                    if abs(signed_diff) > wide_th:
                        internal_reasons.append(metric)

                    # Visual coloring rules:
                    # - Wider than allowed (Strict) => color (fail)
                    # - Only color thinner pins if they exceed the THIN (Loose) threshold
                    if signed_diff > wide_th:
                        color_reasons.append(metric)
                    elif signed_diff < -thin_th:
                        color_reasons.append(metric)
                    continue

                # --- SYMMETRIC LOGIC (Area, Aspect Ratio) ---
                threshold = percentage_thresholds.get(metric, 0.0)
                curr = current_pin[metric]
                refv = ref_pin[metric]
                if refv == 0:
                    continue
                percent_diff = abs((curr - refv) / refv)
                if percent_diff > threshold:
                    internal_reasons.append(metric)
                    color_reasons.append(metric)

            # If internal reasons exist -> mark internally damaged and don't allow it to be used as future ref
            if internal_reasons:
                current_pin['is_damaged'] = True
                image_failure_metrics.update(internal_reasons)

            # Draw colored fail only if color_reasons present
            if color_reasons:
                damage_found = True
                # Replace blue pixels with red for this pin only (no outline drawing)
                replace_blue_with_red(img, current_pin['bbox'], lower_bound, upper_bound)
                x, y, w, h = current_pin['bbox']
                label = f"FAIL ({len(color_reasons)})"
                cv2.putText(img, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save image
        output_path = os.path.join(out, filename)
        cv2.imwrite(output_path, img)

        status = "DAMAGED PINS FOUND" if damage_found else "ALL OK"
        print(f"   Status: {status} | Saved to {out}")

        # Print summary reasons (internal reasons)
        if image_failure_metrics:
            for metric in sorted(image_failure_metrics):
                print(f"                 {REASON_MAP[metric]} fail")


if __name__ == "__main__":
    process_directory()
