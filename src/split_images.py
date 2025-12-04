import cv2
import numpy as np

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
    
    # Convert to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances local contrast, making the IC body and pins more distinguishable
    # clipLimit=3.0 prevents over-amplification of noise
    # tileGridSize=(8,8) divides image into 8x8 tiles for localized enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_contrast = clahe.apply(gray)
    
    # Detect the inner dark IC body (without pins)
    # Use a lower threshold (87) to detect the darker IC chip body
    inner_thresh_value = 87
    _, inner_thresh = cv2.threshold(gray_contrast, inner_thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological opening to remove noise (erosion followed by dilation)
    kernel = np.ones((5,5), np.uint8)
    inner_thresh = cv2.morphologyEx(inner_thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours of the dark IC body
    inner_contours, _ = cv2.findContours(inner_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not inner_contours:
        print("No inner contours found")
        return False
    
    # Get the largest contour (the IC body) and its bounding rectangle
    inner_largest = max(inner_contours, key=cv2.contourArea)
    # ix, iy: top-left corner coordinates of inner body
    # iw, ih: width and height of inner body
    ix, iy, iw, ih = cv2.boundingRect(inner_largest)
    
    # Detect the outer boundary including the pins
    # Use a higher threshold (215) to capture the lighter-colored pins
    outer_thresh_value = 215
    _, outer_thresh = cv2.threshold(gray_contrast, outer_thresh_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological closing to fill gaps between pins (dilation followed by erosion)
    # This creates a continuous boundary around the entire chip including pins
    outer_kernel = np.ones((3,3), np.uint8)
    outer_thresh = cv2.morphologyEx(outer_thresh, cv2.MORPH_CLOSE, outer_kernel)
    
    # Find contours of the entire chip (body + pins)
    outer_contours, _ = cv2.findContours(outer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not outer_contours:
        print("No outer contours found")
        return False
    
    # Get the largest contour and its bounding rectangle
    outer_largest = max(outer_contours, key=cv2.contourArea)
    # ox, oy: top-left corner coordinates of outer boundary
    # ow, oh: width and height of outer boundary
    ox, oy, ow, oh = cv2.boundingRect(outer_largest)
    
    # Add padding around the outer boundary to ensure we don't cut off any pins
    padding = 30
    image_height, image_width = image.shape[:2]
    # Expand outer box by padding, but ensure we stay within image boundaries
    ox = max(0, ox - padding)
    oy = max(0, oy - padding)
    ow = min(image_width - ox, ow + 2 * padding)
    oh = min(image_height - oy, oh + 2 * padding)
    
    # Get base filename without extension for naming output files
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Crop each pin row (space between inner and outer boundaries)
    cropped_count = 0
    
    # Top row: Region from outer top edge to inner top edge
    # top_row format: (x, y, width, height)
    #   [0] = x-coordinate (ox)
    #   [1] = y-coordinate (oy)
    #   [2] = width (ow)
    #   [3] = height (iy - oy) <- The vertical distance from outer to inner
    top_row = (ox, oy, ow, iy - oy)
    if top_row[3] > 5:  # Only save if height > 5 pixels (indicates pins are present)
        # Slice image using: image[y_start:y_end, x_start:x_end]
        top_crop = image[top_row[1]:top_row[1]+top_row[3], top_row[0]:top_row[0]+top_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_top_pins.png")
        cv2.imwrite(output_path, top_crop)
        print(f"Saved top row: {output_path} (size: {top_row[2]}x{top_row[3]})")
        cropped_count += 1
    
    # Bottom row: Region from inner bottom edge to outer bottom edge
    # bottom_row[3] = height from inner bottom (iy + ih) to outer bottom (oy + oh)
    bottom_row = (ox, iy + ih, ow, (oy + oh) - (iy + ih))
    if bottom_row[3] > 5:  # Only save if height > 5 pixels
        bottom_crop = image[bottom_row[1]:bottom_row[1]+bottom_row[3], bottom_row[0]:bottom_row[0]+bottom_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_bottom_pins.png")
        cv2.imwrite(output_path, bottom_crop)
        print(f"Saved bottom row: {output_path} (size: {bottom_row[2]}x{bottom_row[3]})")
        cropped_count += 1
    
    # Left row: Region from outer left edge to inner left edge
    # left_row[2] = width from outer left (ox) to inner left (ix)
    left_row = (ox, oy, ix - ox, oh)
    if left_row[2] > 5:  # Only save if width > 5 pixels (indicates pins are present)
        left_crop = image[left_row[1]:left_row[1]+left_row[3], left_row[0]:left_row[0]+left_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_left_pins.png")
        cv2.imwrite(output_path, left_crop)
        print(f"Saved left row: {output_path} (size: {left_row[2]}x{left_row[3]})")
        cropped_count += 1
    
    # Right row: Region from inner right edge to outer right edge
    # right_row format: (x, y, width, height)
    #   [0] = x-coordinate (ix + iw) - starts at the inner right x
    #   [1] = y-coordinate (oy) - starts at the outer top y
    #   [2] = width ((ox + ow) - (ix + iw)) - The horizontal distance from inner right (ix+iw) to outer right (ox+ow)
    #   [3] = height (oh) - spans the full outer height
    right_row = (ix + iw, oy, (ox + ow) - (ix + iw), oh)
    if right_row[2] > 5:  # Only save if width > 5 pixels
        right_crop = image[right_row[1]:right_row[1]+right_row[3], right_row[0]:right_row[0]+right_row[2]]
        output_path = os.path.join(output_dir, f"{base_name}_right_pins.png")
        cv2.imwrite(output_path, right_crop)
        print(f"Saved right row: {output_path} (size: {right_row[2]}x{right_row[3]})")
        cropped_count += 1
    
    print(f"Total cropped pin rows: {cropped_count}")
    return True
