import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import fitz  # PyMuPDF


class InvoiceDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=512):
        self.data = dataframe
        self.processor = processor
        self.max_length = max_length
        
        # Create label mapping
        self.label_map = self.create_label_mapping(dataframe)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
    
    def create_label_mapping(self, dataframe):
        # Create a mapping of unique labels to integers
        unique_labels = dataframe['label'].unique()
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get a single item from the dataframe
        item = self.data.iloc[idx]
        
        # Prepare words and labels
        words = item['text'].split()
        labels = [self.label_map.get(item['label'], 0) for _ in words]
        
        # Prepare bounding boxes (normalize if needed)
        boxes = item['boxes']
        
        # Process using LayoutLMv3 processor
        encoding = self.processor(
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        
        # Pad labels to match input length
        labels = labels[:self.max_length] + [0] * (self.max_length - len(labels))
        labels = torch.tensor(labels)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bbox': encoding['bbox'].squeeze(),
            'labels': labels
        }

def extract_text_with_positions(pdf_path):
    """Extract text with bounding box positions from PDF"""
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if text.strip():
                            x1, y1, x2, y2 = span["bbox"]
                            all_text.append({
                                "text": text,
                                "page": page_num + 1,
                                "boxes": [x1, y1, x2, y2]
                            })
    
    return all_text

def prepare_training_data(pdf_paths):
    """Prepare training data from multiple PDF invoices"""
    all_data = []
    
    for pdf_path in pdf_paths:
        layout_text = extract_text_with_positions(pdf_path)
        
        for item in layout_text:
            text = item['text']
            label = 'other'
            
            # Custom labeling for extracted key invoice fields
            if 'Invoice No' in text or 'Invoice Number' in text:
                label = 'invoice_number'
            elif 'Date' in text:
                label = 'invoice_date'
            elif 'Buyer' in text or 'Billed To' in text:
                label = 'buyer_name'
            elif 'Address' in text:
                label = 'billing_address'
            elif 'Product' in text or 'Item' in text or 'Description' in text:
                label = 'product_name'
            elif 'Qty' in text or 'Quantity' in text:
                label = 'quantity'
            elif 'Total' in text:
                label = 'total_amount'
            elif 'Taxable Amount' in text:
                label = 'taxable_amount'
            elif 'Tax' in text:
                label = 'tax_amount'
            
            all_data.append({
                'text': text,
                'label': label,
                'boxes': item['boxes']
            })
    
    return pd.DataFrame(all_data)


def fine_tune_and_evaluate(pdf_paths, model_name="microsoft/layoutlmv3-base"):
    # Prepare training data
    training_data = prepare_training_data(pdf_paths)
    
    # Initialize processor and model
    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    num_labels = len(training_data['label'].unique())
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )
    
    # Create dataset
    dataset = InvoiceDataset(training_data, processor)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    num_epochs = 10
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bbox = batch['bbox'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                # Flatten predictions and labels
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {total_train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {total_val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_invoice_model.pth")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=dataset.dataset.label_map.keys()
    ))
    
    return model, processor, best_val_accuracy

# Example usage
if __name__ == "__main__":
    # List of PDF paths
    pdf_paths = [
        "Invoice_data_test.PDF",
        # Add paths to other invoice PDFs
        # "path/to/invoice2.pdf",
        # "path/to/invoice3.pdf",
    ]
    
    # Fine-tune and evaluate
    model, processor, accuracy = fine_tune_and_evaluate(pdf_paths)
    
    print(f"\nBest Validation Accuracy: {accuracy:.4f}")
    
    # Save final model and processor
    model.save_pretrained("./final_invoice_model")
    processor.save_pretrained("./final_invoice_processor")