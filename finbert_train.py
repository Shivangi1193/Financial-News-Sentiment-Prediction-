import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# 1. PARAMETERS & CONFIGURATIONS
# ==========================================
TRAIN_PATH = r"C:\Users\ASUS\Downloads\GUvi_projects\Sentiment_analysis\sent_train.csv"
VALID_PATH = r"C:\Users\ASUS\Downloads\GUvi_projects\Sentiment_analysis\sent_valid.csv"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. TRANSFORMERS DATASET WRAPPER
# ==========================================
class FinBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. TRAINING ENGINE LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Using compute hardware: {DEVICE}")
    
    # Load and clean DataFrame inputs
    df_train = pd.read_csv(TRAIN_PATH).dropna(subset=['text', 'label'])
    df_valid = pd.read_csv(VALID_PATH).dropna(subset=['text', 'label'])
    
    # Initialize Tokenizer and Sequence Classification footprint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True)
    model = model.to(DEVICE)
    
    train_dataset = FinBERTDataset(df_train['text'].values, df_train['label'].values, tokenizer, MAX_LEN)
    valid_dataset = FinBERTDataset(df_valid['text'].values, df_valid['label'].values, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    print("\n🚀 Fine-tuning FinBERT transformer parameters...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} -> Batch Loss Average: {total_loss / len(train_loader):.4f}")
        
    # Evaluate model parameters
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n📊 FinBERT Validation Performance:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Macro F1 Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    
    # Save the optimized token model parameter outputs
    model.save_pretrained("./finbert_sentiment_model")
    tokenizer.save_pretrained("./finbert_sentiment_model")
    print("Transformer configurations exported successfully.")
