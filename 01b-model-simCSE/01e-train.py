#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

# Configuration E: Smaller, faster model
DATA_PATH = "data/emails.csv"
TEXT_COLUMN = "message"
OUTPUT_DIR = "./email-simcse-model-e"
MAX_LENGTH = 192  # Decreased from 256
BATCH_SIZE = 96  # Increased from 64
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5  # Increased from 2e-5
TEMPERATURE = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 4
VALIDATION_SPLIT = 0.1
WANDB_PROJECT = "cs541-final-project"
WANDB_RUN_NAME = "EmailSimCSE-E"
# Modified parameters
HIDDEN_SIZE = 512  # Decreased from 768
NUM_HIDDEN_LAYERS = 4  # Decreased from 6

class EmailDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class EmailSimCSE(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        # Initialize a BERT model from scratch with random weights
        config = BertConfig(
            vocab_size=30522,  # Standard BERT vocab size
            hidden_size=768,
            num_hidden_layers=6,  # Smaller model for email domain
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.encoder = BertModel(config)
        self.temperature = temperature

    def forward(self, inputs):
        emb1 = self.encoder(**inputs).last_hidden_state[:, 0]  # CLS token
        emb2 = self.encoder(**inputs).last_hidden_state[:, 0]  # Same input, different dropout

        return emb1, emb2

    def compute_loss(self, emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        cos_sim = torch.matmul(emb1, emb2.T) / self.temperature
        labels = torch.arange(emb1.size(0), device=cos_sim.device)
        return F.cross_entropy(cos_sim, labels)

def train():
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading data...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA_PATH)
    texts = df[TEXT_COLUMN].tolist()
    train_texts, val_texts = train_test_split(texts, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Just for tokenization
    model = EmailSimCSE(temperature=TEMPERATURE)
    model.to(device)

    train_dataset = EmailDataset(train_texts)
    val_dataset = EmailDataset(val_texts)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for batch_texts in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            batch_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)

            # Forward pass
            emb1, emb2 = model(batch_inputs)
            loss = model.compute_loss(emb1, emb2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            # Update metrics
            train_loss += loss.item()
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0]
            })

        train_loss /= len(train_dataloader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_texts in tqdm(val_dataloader, desc="Validation"):
                batch_inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                ).to(device)

                emb1, emb2 = model(batch_inputs)
                loss = model.compute_loss(emb1, emb2)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)

            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "email_simcse_model.pt"))
            wandb.log({"best_val_loss": val_loss})

    wandb.finish()
    print(f"Training completed. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
