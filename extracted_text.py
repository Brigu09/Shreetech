import fitz  # PyMuPDF
import pandas as pd

def extract_text_from_invoice_pdf(pdf_path):
    docs = fitz.open(pdf_path)
    invoice_text = ""
    
    for page in docs:
        invoice_text += page.get_text("text") + "\n"
    
    return invoice_text

def extract_text_with_positions(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_width, page_height = page.rect.width, page.rect.height  # Get page size
        

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if text.strip():  # Only add non-empty text
                            x1, y1, x2, y2 = span["bbox"]
                            all_text.append({
                                "text": text,
                                "page": page_num + 1,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                            })
    
    return all_text, page_width, page_height