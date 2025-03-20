from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
import torch




# load transformer based pretrained model

model_name = "microsoft/layoutlmv3-base"
processor = AutoProcessor.from_pretrained(model_name, apply_ocr = False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)


# Normalize bounding boxes to a 0-1000 range required by LayoutLMv3.

def normalize_bounding_box(layout_text, page_width, page_height):

    normalized_bbox = []
    words = []

    for item in layout_text:
        words.append(item["text"])  # store words

        x1_norm = int(item["x1"] * 1000 / page_width)
        y1_norm = int(item["y1"] * 1000 / page_height)
        x2_norm = int(item["x2"] * 1000 / page_width)
        y2_norm = int(item["y2"] * 1000 / page_height)

        normalized_bbox.append([x1_norm, y1_norm, x2_norm, y2_norm])

    return words, normalized_bbox


def predict_invoice_fields(layout_text, page_width, page_height):
    # Normalize bounding boxes
    words, boxes = normalize_bounding_box(layout_text, page_width, page_height)

    # Prepare inputs for the model
    encoding = processor.tokenizer(
        text=words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions


