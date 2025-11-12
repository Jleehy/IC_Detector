#!/usr/bin/env python3
"""
grayscale_converter.py

Batch crop "pin rows" from images in a directory.

Usage:
    python3 grayscale_converter.py --input-dir /path/to/images --output-dir ./output [--show]

Output filenames:
    IC{image_index}_pin{pin_index}.png
    (image_index starts at 1 for the first input file)
"""

import os
import argparse
import glob
import cv2
import numpy as np
from typing import Tuple, List


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def clamp_rect(x, y, w, h, max_w, max_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(w)
    h = int(h)
    if x + w > max_w:
        w = max_w - x
    if y + h > max_h:
        h = max_h - y
    w = max(0, w)
    h = max(0, h)
    return x, y, w, h


def find_largest_contour_bbox(binary_mask: np.ndarray):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:  # tiny area -> ignore
        return None
    return cv2.boundingRect(largest)  # x, y, w, h


def crop_pin_rows(input_path: str, output_dir: str, ic_index: int = 1, show: bool = False) -> int:
    """
    Process a single image path, detect inner body and outer area and save
    pin-row crops to `output_dir` named IC{ic_index}_pin{n}.png.
    Returns number of crops saved.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return 0

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)

    # --- Outer detection using adaptive threshold (robust to lighting) ---
    # adaptive block size must be odd and >=3
    block_size = 11
    if block_size >= min(h_img, w_img):
        block_size = max(3, (min(h_img, w_img) // 2) | 1)
    outer_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
    )

    # morphology to close small gaps (kernel size relative to image size)
    k = max(3, int(min(h_img, w_img) / 200))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # find largest outer bbox
    outer_bbox = find_largest_contour_bbox(outer_mask)
    if outer_bbox is None:
        # fallback: try Otsu on whole image
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        outer_bbox = find_largest_contour_bbox(otsu_mask)
        if outer_bbox is None:
            print(f"[WARN] No outer contour found for {input_path}")
            return 0

    ox, oy, ow, oh = outer_bbox
    ox, oy, ow, oh = clamp_rect(ox, oy, ow, oh, w_img, h_img)

    # --- Inner detection inside outer box using Otsu (likely IC body) ---
    roi = gray[oy:oy+oh, ox:ox+ow]
    inner_bbox = None
    if roi.size > 0:
        _, inner_mask_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # opening to remove speckle
        ik = max(3, int(min(roi.shape[:2]) / 200))
        if ik % 2 == 0:
            ik += 1
        inner_kernel = np.ones((ik, ik), np.uint8)
        inner_mask_roi = cv2.morphologyEx(inner_mask_roi, cv2.MORPH_OPEN, inner_kernel, iterations=1)
        bbox_roi = find_largest_contour_bbox(inner_mask_roi)
        if bbox_roi is not None:
            rx, ry, rw, rh = bbox_roi
            # translate to full image coords
            inner_bbox = (ox + rx, oy + ry, rw, rh)

    # fallback: if no inner found, shrink outer bbox by margin
    if inner_bbox is None:
        margin_x = int(ow * 0.10)  # 10% margin
        margin_y = int(oh * 0.10)
        ix = ox + margin_x
        iy = oy + margin_y
        iw = max(1, ow - 2 * margin_x)
        ih = max(1, oh - 2 * margin_y)
        inner_bbox = clamp_rect(ix, iy, iw, ih, w_img, h_img)
        print(f"[INFO] Inner body fallback used for {input_path}")

    ix, iy, iw, ih = inner_bbox
    ix, iy, iw, ih = clamp_rect(ix, iy, iw, ih, w_img, h_img)

    # Prepare pin region candidates (top, bottom, left, right)
    regions = []
    # Top row: outer top -> inner top
    top_h = iy - oy
    if top_h > 5:
        regions.append(("top", ox, oy, ow, top_h))
    # Bottom row: inner bottom -> outer bottom
    bottom_y = iy + ih
    bottom_h = (oy + oh) - bottom_y
    if bottom_h > 5:
        regions.append(("bottom", ox, bottom_y, ow, bottom_h))
    # Left row: outer left -> inner left
    left_w = ix - ox
    if left_w > 5:
        regions.append(("left", ox, oy, left_w, oh))
    # Right row: inner right -> outer right
    right_x = ix + iw
    right_w = (ox + ow) - right_x
    if right_w > 5:
        regions.append(("right", right_x, oy, right_w, oh))

    # Save crops with numbering IC{ic_index}_pin{n}.png
    base_prefix = f"IC{ic_index}_pin"
    saved = 0
    ensure_dir(output_dir)

    for idx, (name, x, y, w, h) in enumerate(regions, start=1):
        x, y, w, h = clamp_rect(x, y, w, h, w_img, h_img)
        if w <= 0 or h <= 0:
            continue
        crop = image[y:y+h, x:x+w]
        out_name = f"{base_prefix}{idx}.png"
        out_path = os.path.join(output_dir, out_name)
        success = cv2.imwrite(out_path, crop)
        if success:
            saved += 1
            print(f"[SAVED] {out_path} ({w}x{h}) region={name}")
        else:
            print(f"[ERROR] Failed to write {out_path}")

        if show:
            cv2.imshow(out_name, crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if saved == 0:
        print(f"[INFO] No pin rows cropped for {input_path}")
    return saved


def find_images_in_dir(input_dir: str, exts: List[str] = None) -> List[str]:
    if exts is None:
        exts = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    files = sorted(set(files))
    return files


def main():
    parser = argparse.ArgumentParser(description="Batch crop pin rows from images in a directory.")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing images to process.")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory to save crops.")
    parser.add_argument("--show", action="store_true", help="Show each crop with cv2.imshow (blocks).")
    parser.add_argument("--exts", nargs="+", default=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                        help="Image extensions to search for (space separated).")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    show = args.show
    exts = [e.lstrip(".").lower() for e in args.exts]

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    ensure_dir(output_dir)
    image_files = find_images_in_dir(input_dir, exts)
    if not image_files:
        print(f"[INFO] No images found in {input_dir} with extensions {exts}")
        return

    total_saved = 0
    for i, path in enumerate(image_files, start=1):
        print(f"\n[PROCESS] ({i}/{len(image_files)}) {path}")
        saved = crop_pin_rows(path, output_dir, ic_index=i, show=show)
        print(f"[RESULT] saved {saved} crops for image {i}")
        total_saved += saved

    print(f"\n[SUMMARY] Processed {len(image_files)} images, total crops saved: {total_saved}")


if __name__ == "__main__":
    main()
