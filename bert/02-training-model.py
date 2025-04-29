import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os

# Set random seeds for reproducibility
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0

# Create a custom dataset class
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
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'priority': torch.tensor(priority, dtype=torch.long)
        }

# Training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    print("Starting training...")
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print('-' * 10)
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['priority'].to(device)
            
            model.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['priority'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                
                total_val_loss += loss.item()
                
                # Get predictions
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Print metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Priority Classification Accuracy: {accuracy:.4f}")
        
        print("\nPriority Classification Report:")
        print(classification_report(true_labels, predictions, 
                                    target_names=['low', 'medium', 'high']))
        
        # Plot confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['low', 'medium', 'high'],
                    yticklabels=['low', 'medium', 'high'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
        
        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_bert_priority_model.pt')
            print("Saved best model!")

# Main training function
def main():
    # Path to your dataset
    data_path = 'multilingual_customer_support_tickets.csv'
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found. Please run data preparation script first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Priority/Urgency distribution:\n{df['urgency'].value_counts()}")
    
    # For the priority model, we'll use the 'urgency' column from the preprocessed data
    # Rename for clarity in this context
    df = df.rename(columns={'urgency': 'priority'})
    
    # Split into training and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed_val, stratify=df['priority'])
    
    print("Training set distribution:")
    print(train_df['priority'].value_counts(normalize=True))
    print("\nValidation set distribution:")
    print(val_df['priority'].value_counts(normalize=True))
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Create datasets
    train_dataset = SupportTicketDataset(
        texts=train_df['text'].values,
        priorities=train_df['priority'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = SupportTicketDataset(
        texts=val_df['text'].values,
        priorities=val_df['priority'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE
    )
    
    # Initialize model - BERT for sequence classification with 3 priority classes
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=3,  # 0=low, 1=medium, 2=high
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Train the model
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
    
    print("Training complete!")

if __name__ == "__main__":
    main()