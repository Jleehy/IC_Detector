import re
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

    def detect_text(self, path: str | None = None) -> str:
        from google.cloud import vision

        if path is None:
            path = self.input_name

        client = vision.ImageAnnotatorClient()

        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                f"{response.error.message}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors"
            )

        if not texts:
            print("No text detected.")
            self.markings = ""
            return ""

        raw_full_text = texts[0].description

        lines = raw_full_text.splitlines()
        clean_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue

            no_space = re.sub(r"\s+", "", s)

            if len(no_space) >= 3 and len(set(no_space)) == 1:
                continue

            clean_lines.append(s)

        full_text = "\n".join(clean_lines)
        self.markings = full_text

        print("Full text:")
        print(full_text)
