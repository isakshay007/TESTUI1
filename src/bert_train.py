# 📁 src/bert_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

# ✅ Model class
def get_model(num_tags):
    class MiniTagTransformer(nn.Module):
        def __init__(self, num_tags):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            return self.classifier(cls_output)

    return MiniTagTransformer(num_tags)

# ✅ Dataset class
class TagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = torch.from_numpy(np.array(labels)).float()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }

# ✅ Main training function
def train_and_save_model(csv_path, save_dir, top_k=2500, batch_size=16, epochs=10):
    df = pd.read_csv(csv_path)
    df['tags'] = df['tags'].apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['tags'])
    texts = df['text'].tolist()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, y, test_size=0.1, random_state=42)

    train_dataset = TagDataset(train_texts, train_labels, tokenizer)
    val_dataset = TagDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_tags=len(mlb.classes_))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=" Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=" Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📉 Train Loss: {avg_train_loss:.4f} | 📊 Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'trained_model.pt'))
            with open(os.path.join(save_dir, 'mlb.pkl'), 'wb') as f:
                pickle.dump(mlb, f)
            print("💾 Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    train_and_save_model(
        csv_path="/content/drive/MyDrive/hmm_model/train_split.csv",
        save_dir="/content/drive/MyDrive/hmm_model"
    )
