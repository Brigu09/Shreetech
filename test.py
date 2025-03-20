from extracted_text import extract_text_from_invoice_pdf, extract_text_with_positions
from prediction import predict_invoice_fields

pdf_path = "Invoice_data_test.pdf"


raw_text = extract_text_from_invoice_pdf(pdf_path)
layout_text, page_width, page_height = extract_text_with_positions(pdf_path)
prediction = predict_invoice_fields(layout_text, page_width, page_height)


# print("Extracted raw text: \n", raw_text)
# print("Extracted layout text: \n", layout_text)

print("Predicted invoice fields: \n", prediction)
