import sys
sys.path.insert(0, "./src")

import sys
sys.path.insert(0, "./src")

from marker import Marker

def main():

    image_file = "IC marking/test.png"
    m1 = Marker(image_file, "temp/ori_edited.png")
    m1.tesseract(
        print_result=False,
        box_output="temp/ori_box.png",
        crop_output="temp/new.png",
        do_crop=True,
    )

    cut_file = "temp/new.png"
    m2 = Marker(cut_file, "temp/new_edited.png")
    m2.tesseract(
        print_result=True,
        box_output="temp/new_box.png",
        crop_output=None,   # no second-level cropping
        do_crop=False,
    )

    return 0


if __name__ == "__main__":
    main()
