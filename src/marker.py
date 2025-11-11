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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Grayscale
        blurred_image = cv2.GaussianBlur(gray_image, (7,7), 0) #Gaussian blur
        threshold, bw_image = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY_INV)
        #noiseless_image = self.noise_removal(inverted_image)
        kernel = np.ones((4,4),np.uint8)
        dilated_image = cv2.dilate(bw_image, kernel, iterations=1)
        cv2.imwrite(self.tempdir, dilated_image)
    
    def noise_removal(self, image):
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)
