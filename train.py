import os
import re
import string
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# 1. PREPROCESSING & CLEANING UTILITIES
# ==========================================
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Strip URLs
    text = re.sub(r'@\S+', '', text)                  # Strip Mentions
    valid_chars = f"a-zA-Z0-9\s\$\%\.\-\+\,\:\(\)\'\""
    text = re.sub(f"[^{valid_chars}]", "", text)       # Keep tickers/financial expressions
    return text.strip().lower()

class Vocabulary:
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = {}
        
    def build_vocab(self, texts):
        for text in texts:
            for word in text.split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
                
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_vocab_size - 2]
        
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def numericalize(self, text):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]

class FinancialTweetsDataset(Dataset):
    def __init__(self, sequences, labels, max_len=64):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 2. NEURAL NETWORK TOPO ENGINE
# ==========================================
class FinancialRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, model_type="lstm"):
        super(FinancialRNN, self).__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if self.model_type == "simplernn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        if self.model_type == "lstm":
            out, (hidden, _) = self.rnn(embedded)
        else:
            out, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])

# ==========================================
# 3. TRAINING & EVALUATION ENGINES
# ==========================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, macro_f1, cm

def train_baseline(model_type, train_loader, val_loader, vocab_size, device, epochs=5):
    print(f"\n🚀 Training Architecture: Embedding + {model_type.upper()}")
    model = FinancialRNN(vocab_size, embed_dim=128, hidden_dim=128, output_dim=3, model_type=model_type)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    acc, f1, cm = evaluate_model(model, val_loader, device)
    print(f"📊 Validation Performance -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    torch.save(model.state_dict(), f"{model_type}_sentiment.pt")
    return acc, f1, cm

# ==========================================
# 4. RUN ALL PIPELINES
# ==========================================
if __name__ == "__main__":
    # Define absolute path signatures safely from workspace context map
    train_path = r"C:\Users\ASUS\Downloads\GUvi_projects\Sentiment_analysis\sent_train.csv"
    valid_path = r"C:\Users\ASUS\Downloads\GUvi_projects\Sentiment_analysis\sent_valid.csv"
    
    print("--- Loading Pre-Split Datasets ---")
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    
    # Process text arrays uniformly
    df_train['cleaned_text'] = df_train['text'].apply(clean_tweet)
    df_valid['cleaned_text'] = df_valid['text'].apply(clean_tweet)
    
    # Build vocabulary footprint strictly from training partition metadata context mapping
    vocab = Vocabulary()
    vocab.build_vocab(df_train['cleaned_text'])
    
    # Numericalize sequence index boundaries
    train_sequences = [vocab.numericalize(text) for text in df_train['cleaned_text']]
    valid_sequences = [vocab.numericalize(text) for text in df_valid['cleaned_text']]
    
    # Wrap tokenized tensors into structured matrix pipeline blocks
    train_dataset = FinancialTweetsDataset(train_sequences, df_train['label'].values)
    val_dataset = FinancialTweetsDataset(valid_sequences, df_valid['label'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab.word2idx)
    
    # Run structural benchmark optimization iterations
    rnn_acc, rnn_f1, rnn_cm = train_baseline("simplernn", train_loader, val_loader, vocab_size, device)
    lstm_acc, lstm_f1, lstm_cm = train_baseline("lstm", train_loader, val_loader, vocab_size, device)
    gru_acc, gru_f1, gru_cm = train_baseline("gru", train_loader, val_loader, vocab_size, device)
    
    # Cache mapping lookup assets for local application deployment instance requirements
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        
    print("\nProcessing Pipeline Concluded successfully.")
