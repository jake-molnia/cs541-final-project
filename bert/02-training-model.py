import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os

# Set random seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 8e-6  # Lower LR to stabilize training
WARMUP_STEPS_RATIO = 0.1  # Warmup for smoother optimization
DROPOUT_RATE = 0.3  # Apply dropout for regularization

# Load dataset
data_path = '/multilingual_customer_support_tickets.csv'
df = pd.read_csv(data_path)
df.rename(columns={'urgency': 'priority'}, inplace=True)

# Compute class weights dynamically
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['priority']), y=df['priority'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Split into training and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed_val, stratify=df['priority'])

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Custom dataset class
class SupportTicketDataset(Dataset):
    def __init__(self, texts, priorities, tokenizer, max_len):
        self.texts = texts
        self.priorities = priorities
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        priority = self.priorities[idx]

        # Apply domain-specific keyword augmentation
        priority_keywords = {
            0: ["non-urgent", "later", "optional"],
            1: ["important", "attention", "needs review"],
            2: ["urgent", "critical", "immediate"]
        }
        keyword = random.choice(priority_keywords[priority])
        text = f"{text} {keyword}"

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len, padding='max_length',
            return_attention_mask=True, return_tensors='pt', truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'priority': torch.tensor(priority, dtype=torch.long)
        }

# Create datasets
train_dataset = SupportTicketDataset(train_df['text'].values, train_df['priority'].values, tokenizer, MAX_LEN)
val_dataset = SupportTicketDataset(val_df['text'].values, val_df['priority'].values, tokenizer, MAX_LEN)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

# Initialize model with dropout
class EnhancedBERTModel(torch.nn.Module):
    def __init__(self, dropout_rate):
        super(EnhancedBERTModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.logits)  # logits already (batch_size, num_labels)
        return x 

model = EnhancedBERTModel(DROPOUT_RATE).to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_STEPS_RATIO), num_training_steps=total_steps)

# Training function with misclassification tracking
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Training phase
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['priority'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions, true_labels = [], []

        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['priority'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        print(classification_report(true_labels, predictions, target_names=['low', 'medium', 'high']))

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_bert_priority_model.pt')
            print("Saved best model!")

        # Track misclassified samples
        misclassified_df = pd.DataFrame({'text': val_df['text'], 'true': true_labels, 'predicted': predictions})
        misclassified_df.to_csv(f'misclassified_samples_epoch_{epoch+1}.csv', index=False)

# Run training
train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
print("Training complete!")