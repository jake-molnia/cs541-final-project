#!/usr/bin/env python3
import os
import logging
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_CSV = os.path.expanduser("./data/emails.csv")
OUTPUT_CSV = os.path.expanduser("./data/enron_tagged.csv")
BATCH_SIZE = 256  # Optimized for A30 GPU memory (24GB)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'message' not in df.columns:
        raise ValueError("CSV must contain a 'message' column")
    return df

def setup_model():
    """Set up model with optimizations for A30 GPUs"""
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPUs")

    if gpu_count <= 1:
        logger.info("Using single GPU with optimized settings")
        # Use mixed precision (FP16) for better performance
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None
        )

    logger.info(f"Multiple GPUs detected, using primary GPU with optimized settings")
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0,
        torch_dtype=torch.float16
    )

def classify_messages(texts, classifier, labels):
    results = classifier(texts, labels, multi_label=False)
    return [(r['labels'][0], r['scores'][0]) for r in results]

def main():
    logger.info("Starting classification job")

    if torch.cuda.is_available():
        logger.info("CUDA available, setting optimization flags")
        torch.backends.cudnn.benchmark = True  # Use cuDNN autotuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 when available

    classifier = setup_model()
    candidate_labels = ["very negative", "moderately negative", "negative",
                        "neutral", "positive", "moderately positive", "very positive"]

    emotion_mapping = {'very negative': -1, 'moderately negative': -0.5, 'negative': -0.25,
                      'neutral': 0, 'positive': 0.25, 'moderately positive': 0.5, 'very positive': 1}
    df = load_data(INPUT_CSV)
    logger.info(f"Loaded {len(df)} records")
    predictions = []
    scores = []

    # Process in chunks to avoid memory pressure
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Classifying"):
        end = min(start + BATCH_SIZE, len(df))
        batch_texts = df.loc[start:end-1, 'message'].tolist()

        if start % (BATCH_SIZE * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        preds = classify_messages(batch_texts, classifier, candidate_labels)
        batch_labels, batch_scores = zip(*preds)
        predictions.extend(batch_labels)
        scores.extend(batch_scores)

    df['predicted_emotion'] = predictions
    df['emotion_score'] = df['predicted_emotion'].map(emotion_mapping)  # Map text labels to numerical scores
    df['confidence'] = scores
    logger.info(f"Classification complete, saving results")
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
