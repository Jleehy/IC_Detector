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
# Maximum allowed percentage difference between a pin and its neighbor.
MAX_DIFF_W_PER = 15.0   # Width (shorter side)
MAX_DIFF_H_PER = 5.5   # Height (longer side / length)
MAX_DIFF_A_PER = 20.0   # Area
MAX_DIFF_AR_PER = 100  # NEW: Aspect Ratio (Height/Width)
# =================================================

# Mapping for user-friendly console output
REASON_MAP = {
    'Width': 'width',
    'Height': 'length',
    'Area': 'area',
    'AspectRatio': 'aspect_ratio' # NEW reason
}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def get_closest_to_median_index(areas):
    """
    Finds the index of the pin whose area is closest to the median area.
    """
    if not areas:
        return None
    median = np.median(areas)
    return np.argmin(np.abs(np.array(areas) - median))

def process_directory():
    # Setup Directories
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get list of images
    types = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_files = []
    for files in types:
        image_files.extend(glob.glob(os.path.join(SOURCE_DIR, files)))

    print(f"Found {len(image_files)} images in {SOURCE_DIR}...")

    # Prepare Color Bounds
    target_bgr = hex_to_bgr(TARGET_HEX)
    tolerance = 15
    lower_bound = np.array([max(0, c - tolerance) for c in target_bgr])
    upper_bound = np.array([min(255, c + tolerance) for c in target_bgr])

    # Local comparison thresholds (in decimals)
    percentage_thresholds = {
        'Width': MAX_DIFF_W_PER / 100.0,
        'Height': MAX_DIFF_H_PER / 100.0,
        'Area': MAX_DIFF_A_PER / 100.0,
        'AspectRatio': MAX_DIFF_AR_PER / 100.0, # NEW THRESHOLD
    }

    # Process Loop
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"-> Processing: {filename}")

        img = cv2.imread(img_path)
        if img is None:
            print("   Warning: Could not read image. Skipping.")
            continue

        mask = cv2.inRange(img, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Harvest Data and standardize W/H
        box_data = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Standardize Length and Width
            standard_width = min(w, h)
            standard_height = max(w, h)

            # Calculate Aspect Ratio (Height / Width)
            aspect_ratio = standard_height / standard_width if standard_width > 0 else 0

            box_data.append({
                'Width': standard_width,
                'Height': standard_height,
                'Area': w * h,
                'AspectRatio': aspect_ratio, # NEW DATA POINT
                'cnt': cnt,
                'bbox': (x, y, w, h),
                'is_damaged': False
            })

        if len(box_data) < 2:
            print("   Not enough pins detected for comparison. Skipping.")
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
            continue

        # --- BASELINE LOGIC ---

        areas = [b['Area'] for b in box_data]
        baseline_index = get_closest_to_median_index(areas)

        box_data.sort(key=lambda x: x['bbox'][1])

        damage_found = False
        image_failure_metrics = set()

        # Find the index of the baseline pin *after* sorting
        baseline_pin_area = areas[baseline_index]
        baseline_pin_index = next(i for i, b in enumerate(box_data) if b['Area'] == baseline_pin_area)

        # Set the reference pin to the median pin
        reference_pin_data = box_data[baseline_pin_index]

        # --- COMPARISON LOOP ---
        # Metrics to check now include AspectRatio
        metrics_to_check = ['Width', 'Height', 'Area', 'AspectRatio']

        for i in range(len(box_data)):
            current_pin = box_data[i]

            # --- Determine Reference Pin (R) ---
            ref_pin = None

            if i == baseline_pin_index:
                 # Skip comparison for the baseline pin itself
                 continue

            # Priority 1: Adjacent Pin (i-1)
            if i > 0 and not box_data[i-1]['is_damaged']:
                ref_pin = box_data[i-1]

            # Priority 2: Contingency - Nearest non-failing pin
            if ref_pin is None and i > 0:
                for j in range(i - 2, -1, -1):
                    if not box_data[j]['is_damaged']:
                        ref_pin = box_data[j]
                        break

            # Tertiary Fallback: Globally-calculated median pin
            if ref_pin is None:
                ref_pin = reference_pin_data


            # 2. Perform comparison against ref_pin
            reasons = []

            for metric in metrics_to_check:
                threshold = percentage_thresholds[metric]
                current_value = current_pin[metric]
                ref_value = ref_pin[metric]

                if ref_value == 0: continue

                percent_diff = abs((current_value - ref_value) / ref_value)

                if percent_diff > threshold:
                    reasons.append(metric)

            if reasons:
                damage_found = True
                current_pin['is_damaged'] = True # Mark current pin as damaged
                image_failure_metrics.update(reasons)

                # Draw RED box and Label
                cv2.drawContours(img, [current_pin['cnt']], -1, (0, 0, 255), 3)
                x, y, w, h = current_pin['bbox']
                label = f"FAIL ({len(reasons)})"
                cv2.putText(img, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save to Output Directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, img)

        status = "DAMAGED PINS FOUND" if damage_found else "ALL OK"
        print(f"   Status: {status} | Saved to {OUTPUT_DIR}")

        # Print detailed failure reasons
        if image_failure_metrics:
            for metric in sorted(image_failure_metrics):
                print(f"                 {REASON_MAP[metric]} fail")


if __name__ == "__main__":
    process_directory()
