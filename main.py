import sys
sys.path.insert(0, "./src")

from marker import Marker

def main():
    # First pass: run OCR on the whole image (do NOT print result)
    image_file = "IC marking/test.png"
    m1 = Marker(image_file, "temp/test_post.png")
    m1.tesseract(print_result=False)

    # Second pass: run OCR again on the cropped region (print result)
    cut_file = "temp/test_post_box_cut.png"
    m2 = Marker(cut_file, "temp/test_post_box_cut_proc.png")
    m2.tesseract()  # default print_result=True

    return 0

if __name__ == "__main__":
    main()
