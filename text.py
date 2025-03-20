import fitz      # PYMUPDF
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR


# INITIALIZE OCR MODEL

ocr = PaddleOCR(use_angle_cls = True, lang = "en")


def extract_text_from_invoice_pdf(pdf_path):

    docs = fitz.open(pdf_path)
    invoice_text = ""

    for page in docs:
        invoice_text += page.get_text("text") + "\n"

    return invoice_text


def extract_text_with_positions(pdf_path):

    image = convert_from_path(pdf_path)
    all_text = []

    for img in image:
        ocr_result = ocr.ocr(img, cls = True)  # RUN OCR ON IMAGE TO EXTRACT TEXT
        for line in ocr_result:
            for word in line:
                text, (x1, y1, x2, y2) = word[1][0], word[0][0], word[0][1], word[0][2], word[0][3]  # Extracts text along with its bounding box coordinates (x1, y1, x2, y2).
                all_text.append({"text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2})  # STORE TEXT WITH POSTION DATA IN A LIST OF DICTIONARIES
    
    return all_text