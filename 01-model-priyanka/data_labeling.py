import pandas as pd
import torch
import logging
import os
import time
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# Configure logging by default
os.makedirs('./logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join('./logs', f"data_labeling_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set device - use GPU by default if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("GPU not available, using CPU instead")

def batch_predict_sentiment(model, tokenizer, texts, batch_size=32):
    """Predict sentiment in batches to handle large datasets efficiently"""
    model.to(device)
    model.eval()

    all_predictions = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting sentiment"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the texts
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                          return_tensors='pt', max_length=128)

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted sentiment
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)

    return all_predictions

# Log start time
start_time = time.time()
logger.info("Starting data labeling process")

# Input and output paths
input_file = './data/dataset-tickets-multi-lang-4-20k.csv'
output_file = './data/customer_service_with_sentiment_predictions.csv'
batch_size = 32

try:
    # Load the customer service dataset
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded dataset with {len(df)} rows")

    # Preprocess the data
    logger.info("Preprocessing data...")
    df = df[df['language'] == 'en']
    logger.info(f"Filtered to {len(df)} English rows")

    df['text'] = df['subject'] + ' ' + df['body']
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['text'], inplace=True)
    logger.info(f"After preprocessing: {len(df)} rows")

    # Emotion Mapping for sentiment labels
    emotion_mapping1 = {'very negative': -1, 'moderately negative': -0.5, 'negative': -0.25,
                        'neutral': 0, 'positive': 0.25, 'moderately positive': 0.5, 'very positive': 1}
    priority_mapping2 = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Load pre-trained model for sentiment analysis
    logger.info("Loading pre-trained DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3  # For 3-class sentiment
    )

    # Get sentiment predictions
    logger.info(f"Predicting sentiment for {len(df)} texts with batch size {batch_size}...")
    texts = df['text'].tolist()
    df['sentiment_predicted'] = batch_predict_sentiment(
        model,
        tokenizer,
        texts,
        batch_size=batch_size
    )

    # Map predictions to emotion categories
    logger.info("Mapping predictions to emotion categories...")
    df['emotion'] = df['sentiment_predicted'].map({0: 'negative', 1: 'neutral', 2: 'positive'})

    # Map sentiment to emotion numeric scale
    df['emotion_numeric'] = df['emotion'].map(emotion_mapping1)

    # Save the predictions to a new file
    logger.info(f"Saving labeled data to {output_file}")
    df.to_csv(output_file, index=False)

    # Log sample of the results
    logger.info("Sample of labeled data:")
    logger.info(df[['text', 'sentiment_predicted', 'emotion', 'emotion_numeric']].head())

    # Log completion time
    end_time = time.time()
    logger.info(f"Data labeling completed successfully in {end_time - start_time:.2f} seconds")

except Exception as e:
    logger.error(f"Error during data labeling: {str(e)}", exc_info=True)
    raise

logger.info("Data labeling script completed")
