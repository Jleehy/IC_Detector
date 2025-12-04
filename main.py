import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, "./src")

from split_images import crop_pin_rows
from marker import Marker
from pin_detection import detect_pins
from detect_bad_pins import process_directory as analyze_pins


def process_one_image(image_path: str, temp_dir: str = "temp") -> None:

    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print(f"Processing: {image_path}")

    dummy_temp = os.path.join(temp_dir, "dummy.png")

    m = Marker(image_path, dummy_temp)
    m.detect_text()


def main():
    input_dir = "IC marking"
    temp_dir = "temp"
    output_dir = "output"
    to_crop = "Pin defect"
    final_output_dir = "output_shrunk"
    analyzed_output_dir = "analyzed_pins"

    # get all .png files in "IC marking" (sorted)
    png_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ]
    png_files.sort()

    if not png_files:
        print(f"No PNG files found under: {input_dir}")
        return 0

    for img_path in png_files:
        process_one_image(img_path, temp_dir=temp_dir)

    # crop the images in "Pin defect"
    if os.path.exists(to_crop):
        pin_files = [
            os.path.join(to_crop, f)
            for f in os.listdir(to_crop)
            if f.lower().endswith(".png") or f.lower().endswith(".jpg")
        ]
        pin_files.sort()
        
        if not pin_files:
            print(f"No images found in {to_crop} to crop.")
        else:
            print(f"Cropping {len(pin_files)} images from {to_crop}...")
            for pin_img in pin_files:
                crop_pin_rows(pin_img, output_dir=output_dir)
            
            # Run pin detection on the cropped images
            print(f"Running pin detection on {output_dir}...")
            detect_pins(input_dir=output_dir, output_dir=final_output_dir)
            
            # Analyze the detected pins for defects
            print(f"Analyzing pins in {final_output_dir}...")
            analyze_pins(source_dir=final_output_dir, output_dir=analyzed_output_dir)

    else:
        print(f"Directory {to_crop} does not exist.")

    return 0


if __name__ == "__main__":
    main()