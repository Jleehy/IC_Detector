import cv2
import sys


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


if __name__ == "__main__":
    input_path = "Pin defect/C-T-28SOP-04F-SM.png"
    output_path = "output_gray.png"
    
    convert_to_grayscale(input_path, output_path)
