import cv2
import sys
import matplotlib.pyplot as plt

import numpy as np


def convert_to_grayscale(input_path, output_path):
    """Convert image to grayscale"""
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply high contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray)
    
    # Save result
    cv2.imshow('Grayscale High Contrast', gray_contrast)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, gray_contrast)
    print(f"Converted: {input_path} -> {output_path}")
    return True



def crop_pin_rows(input_path, output_dir="output"):
    """Crop each row/side of pins and save as separate images."""
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return False
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray)
    
    # --- Inner black body detection ---
    inner_thresh_value = 87
    _, inner_thresh = cv2.threshold(gray_contrast, inner_thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    inner_thresh = cv2.morphologyEx(inner_thresh, cv2.MORPH_OPEN, kernel)
    
    inner_contours, _ = cv2.findContours(inner_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not inner_contours:
        print("No inner contours found")
        return False
    
    inner_largest = max(inner_contours, key=cv2.contourArea)
    ix, iy, iw, ih = cv2.boundingRect(inner_largest)
    
    # --- Outer detection including pins ---
    outer_thresh_value = 215
    _, outer_thresh = cv2.threshold(gray_contrast, outer_thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    outer_kernel = np.ones((3,3), np.uint8)
    outer_thresh = cv2.morphologyEx(outer_thresh, cv2.MORPH_CLOSE, outer_kernel)
    
    outer_contours, _ = cv2.findContours(outer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not outer_contours:
        print("No outer contours found")
        return False
    
    outer_largest = max(outer_contours, key=cv2.contourArea)
    ox, oy, ow, oh = cv2.boundingRect(outer_largest)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Calculate and crop pin row regions
    cropped_count = 0
    
    # Top row: from outer top to inner top
    top_row = (ox, oy, ow, iy - oy)
    if top_row[3] > 5:  # height > 5 pixels (has pins)
        top_crop = image[top_row[1]:top_row[1]+top_row[3], top_row[0]:top_row[0]+top_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_top_pins.png")
        cv2.imwrite(output_path, top_crop)
        print(f"Saved top row: {output_path} (size: {top_row[2]}x{top_row[3]})")
        cropped_count += 1
    
    # Bottom row: from inner bottom to outer bottom
    bottom_row = (ox, iy + ih, ow, (oy + oh) - (iy + ih))
    if bottom_row[3] > 5:
        bottom_crop = image[bottom_row[1]:bottom_row[1]+bottom_row[3], bottom_row[0]:bottom_row[0]+bottom_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_bottom_pins.png")
        cv2.imwrite(output_path, bottom_crop)
        print(f"Saved bottom row: {output_path} (size: {bottom_row[2]}x{bottom_row[3]})")
        cropped_count += 1
    
    # Left row: from outer left to inner left
    left_row = (ox, oy, ix - ox, oh)
    if left_row[2] > 5:  # width > 5 pixels (has pins)
        left_crop = image[left_row[1]:left_row[1]+left_row[3], left_row[0]:left_row[0]+left_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_left_pins.png")
        cv2.imwrite(output_path, left_crop)
        print(f"Saved left row: {output_path} (size: {left_row[2]}x{left_row[3]})")
        cropped_count += 1
    
    # Right row: from inner right to outer right
    right_row = (ix + iw, oy, (ox + ow) - (ix + iw), oh)
    if right_row[2] > 5:
        right_crop = image[right_row[1]:right_row[1]+right_row[3], right_row[0]:right_row[0]+right_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_right_pins.png")
        cv2.imwrite(output_path, right_crop)
        print(f"Saved right row: {output_path} (size: {right_row[2]}x{right_row[3]})")
        cropped_count += 1
    
    print(f"Total cropped pin rows: {cropped_count}")
    return True


def detect_single_pin(input_path, output_path):
    """Detect a single pin in the image"""
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray)
    cv2.imshow('Grayscale High Contrast', gray_contrast)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, gray_contrast)
    print(f"Converted: {input_path} -> {output_path}")
    return True

if __name__ == "__main__":
    
    input_path = 'pin2.png'
    #input_path = "Pin defect/A-D-64QFP-14B-SM.png"
    #input_path = "Pin defect/A-D-64QFP-15B-SM.png"
    #input_path = "Pin defect/A-J-28SOP-01B-SM.png"
    #input_path = "Pin defect/C-T-28SOP-04F-SM.png"
    output_path = "pin2_detected.png"
    #crop_pin_rows(input_path, output_dir="output")
    #convert_to_grayscale(input_path, output_path)
    detect_single_pin(input_path, output_path)