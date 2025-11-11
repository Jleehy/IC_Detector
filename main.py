import sys
sys.path.insert(0, "./src")

from marker import Marker

def main():
    image_file = "IC marking/test.png"
    marked_image = Marker(image_file, "temp/test_post.png")
    marked_image.tesseract()
    return 0

main()