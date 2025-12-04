import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, "./src")

from marker import Marker


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

    return 0


if __name__ == "__main__":
    main()