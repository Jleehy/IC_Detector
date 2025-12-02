import sys
import os

# make sure we can import marker from ./src if needed
sys.path.insert(0, "./src")

from marker import Marker


def process_one_image(image_path: str, temp_dir: str = "temp") -> None:
    """
    Run the 2-step OCR pipeline for a single image:
    1) full image -> ori_* outputs
    2) cropped image -> new_* outputs
    """
    os.makedirs(temp_dir, exist_ok=True)

    # e.g. image_path = "IC marking/A-J-28SOP-03F-SM.png"
    base_name = os.path.basename(image_path)              # "A-J-28SOP-03F-SM.png"
    name_no_ext, _ = os.path.splitext(base_name)          # "A-J-28SOP-03F-SM"

    # build all output paths
    ori_edited = os.path.join(temp_dir, f"{name_no_ext}_ori_edited.png")
    ori_box    = os.path.join(temp_dir, f"{name_no_ext}_ori_box.png")
    new_img    = os.path.join(temp_dir, f"{name_no_ext}_new.png")
    new_edited = os.path.join(temp_dir, f"{name_no_ext}_new_edited.png")
    new_box    = os.path.join(temp_dir, f"{name_no_ext}_new_box.png")

    print("=" * 60)
    print(f"Processing: {image_path}")

    # ---------- First pass: full image ----------
    m1 = Marker(image_path, ori_edited)
    m1.tesseract(
        print_result=False,    # do not print first-pass OCR result
        box_output=ori_box,    # full original with red box
        crop_output=new_img,   # cropped original
        do_crop=True,
    )

    # if cropping failed (no text, no new.png), skip second pass
    if not os.path.exists(new_img):
        print(f"Skip second pass for {image_path}: cropped image not created.")
        return

    # ---------- Second pass: cropped image ----------
    m2 = Marker(new_img, new_edited)
    m2.tesseract(
        print_result=True,     # print final OCR result
        box_output=new_box,    # cropped image with red box
        crop_output=None,      # no further cropping
        do_crop=False,
    )


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
