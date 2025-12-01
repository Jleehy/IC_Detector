import cv2 #requirements.txt, needs libgl1 on Linux
import numpy as np #installed as dependency to cv2
import pytesseract
from PIL import Image

class Marker:
    def __init__(self, filename, temppath):
        self.input_name = filename
        self.tempdir = temppath
        self.image = None
        self.processed_image = None
        self.markings = None
        self.custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'


    def tesseract(self):
        self.cvprocess()
        image = Image.open(self.tempdir)
        self.markings = pytesseract.image_to_string(image, config=self.custom_config)
        print(self.markings)

    def cvprocess(self):
        image = cv2.imread(self.input_name)#Read image file
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Grayscale
        blurred_image = cv2.GaussianBlur(gray_image, (29,29), 0) #Gaussian blur
        bw_image = cv2.threshold(blurred_image, 67, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imwrite(self.tempdir, bw_image)
