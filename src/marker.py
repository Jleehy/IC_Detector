import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


class Marker:
    def __init__(self, filename, temppath):
        self.input_name = filename      # path to input image
        self.tempdir = temppath         # path to processed (binary) image
        self.image = None
        self.processed_image = None
        self.markings = None
        self.custom_config = (
            r'--oem 3 --psm 11 '
            r'-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
        )

    def tesseract(self, print_result=True):
        # 1. Preprocess input image and save to self.tempdir
        self.cvprocess()

        # 2. OCR on the processed image
        pil_img = Image.open(self.tempdir)
        self.markings = pytesseract.image_to_string(
            pil_img, config=self.custom_config
        )

        # only print if requested
        if print_result:
            print(self.markings)

        # 3. Get bounding boxes of each word/character
        data = pytesseract.image_to_data(
            pil_img,
            config=self.custom_config,
            output_type=Output.DICT
        )

        xs, ys, xes, yes = [], [], [], []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if data["text"][i].strip() == "":
                continue  # skip empty results

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            xs.append(x)
            ys.append(y)
            xes.append(x + w)
            yes.append(y + h)

        if not xs:
            print("No text detected, skip drawing/cropping.")
            return

        # 4. Merge all small boxes into one big bounding box
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xes), max(yes)

        # Add small padding to make the box slightly larger
        pad = 25
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(pil_img.width - 1, x_max + pad)
        y_max = min(pil_img.height - 1, y_max + pad)

        # 5. Draw red rectangle on original image -> test_post_box.png
        orig = cv2.imread(self.input_name)
        img_box = orig.copy()
        cv2.rectangle(
            img_box, (x_min, y_min), (x_max, y_max),
            (0, 0, 255), 15  # thick red line
        )
        cv2.imwrite("temp/test_post_box.png", img_box)

        # 6. Crop the same region from original image -> test_post_box_cut.png
        cut = orig[y_min:y_max, x_min:x_max]
        cv2.imwrite("temp/test_post_box_cut.png", cut)

    def cvprocess(self):
        image = cv2.imread(self.input_name)#Read image file
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Grayscale
        blurred_image = cv2.GaussianBlur(gray_image, (29,29), 0) #Gaussian blur
        bw_image = cv2.threshold(blurred_image, 63, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imwrite(self.tempdir, bw_image)
