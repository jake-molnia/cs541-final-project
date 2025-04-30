import pandas as pd
import torch
import logging
import os
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from accelerate import Accelerator

# Configure logging
os.makedirs('./logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join('./logs', f"model_training_{timestamp}.log")

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

# Initialize Accelerator
accelerator = Accelerator()
logger.info("Accelerator initialized")

try:
    # Load the preprocessed dataset with sentiment labels
    logger.info("Loading preprocessed dataset")
    df = pd.read_csv('./data/customer_service_with_sentiment_predictions.csv')
    logger.info(f"Loaded dataset with {len(df)} rows")

    # Check if text column exists and is valid
    if 'text' not in df.columns:
        logger.error("'text' column not found in the dataset")
        raise ValueError("'text' column not found in the dataset")

    # Ensure text column contains string values
    logger.info("Validating text column")
    df['text'] = df['text'].astype(str)

    # Display some data statistics and sample
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Sample text: {df['text'].iloc[0][:100]}...")

    # Check for NaN values
    nan_count = df['text'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in text column. Dropping them.")
        df = df.dropna(subset=['text'])

    # Check for empty strings
    empty_count = (df['text'] == '').sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty strings in text column. Dropping them.")
        df = df[df['text'] != '']

    # Tokenizer initialization for DistilBERT
    logger.info("Loading DistilBERT tokenizer")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Map the labels to appropriate scales
    logger.info("Setting up emotion mappings")
    emotion_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    # Prepare numeric labels for training
    logger.info("Preparing labels for training")
    if 'emotion' not in df.columns:
        logger.error("'emotion' column not found in the dataset")
        raise ValueError("'emotion' column not found in the dataset")

    # Get unique emotion values to understand what we're working with
    logger.info(f"Unique values in emotion column: {df['emotion'].unique()}")

    # For classification training, convert labels to integers
    # Create label columns for both models
    # For three-class model (classification): 0 = negative, 1 = neutral, 2 = positive
    df['label_three'] = df['emotion'].map(emotion_mapping)

    # For gradient model (regression): -1 to 1 scale
    emotion_scale = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}
    df['label_grad'] = df['emotion'].map(emotion_scale)

    # Handle any NaN values in labels
    if df['label_three'].isna().any() or df['label_grad'].isna().any():
        logger.warning("Found NaN values in labels, filling with defaults")
        df['label_three'].fillna(1, inplace=True)  # Default to neutral
        df['label_grad'].fillna(0.0, inplace=True)  # Default to neutral

    # Convert to numeric types
    df['label_three'] = df['label_three'].astype(int)
    df['label_grad'] = df['label_grad'].astype(float)

    logger.info(f"Label distribution for three-class model: {df['label_three'].value_counts().to_dict()}")
    logger.info(f"Label range for gradient model: min={df['label_grad'].min()}, max={df['label_grad'].max()}")

    # Initialize models
    logger.info("Initializing DistilBERT models")
    model_grad = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
    model_three = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    # Define tokenize function with label handling
    logger.info("Defining tokenization function")
    def tokenize_and_add_labels(examples):
        # Ensure we're dealing with strings
        texts = [str(text) for text in examples['text']]

        # Tokenize the texts
        tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=128)

        # Add labels for three-class model
        tokenized['labels'] = examples['label_three']

        return tokenized

    def tokenize_and_add_labels_grad(examples):
        # Ensure we're dealing with strings
        texts = [str(text) for text in examples['text']]

        # Tokenize the texts
        tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=128)

        # Add labels for gradient model (regression)
        tokenized['labels'] = examples['label_grad']

        return tokenized

    # Convert to Hugging Face Dataset
    logger.info("Converting to Hugging Face Dataset")
    dataset_three = Dataset.from_pandas(df[['text', 'label_three']])
    dataset_grad = Dataset.from_pandas(df[['text', 'label_grad']])

    # Tokenize the datasets
    logger.info("Tokenizing datasets")
    tokenized_dataset_three = dataset_three.map(tokenize_and_add_labels, batched=True)
    tokenized_dataset_grad = dataset_grad.map(tokenize_and_add_labels_grad, batched=True)

    # Split the datasets into train and test
    logger.info("Splitting datasets into train and test sets")
    train_test_split_three = tokenized_dataset_three.train_test_split(test_size=0.2)
    train_test_split_grad = tokenized_dataset_grad.train_test_split(test_size=0.2)

    train_dataset_three = train_test_split_three['train']
    test_dataset_three = train_test_split_three['test']

    train_dataset_grad = train_test_split_grad['train']
    test_dataset_grad = train_test_split_grad['test']

    # Verify that labels are present
    logger.info(f"Train dataset three features: {train_dataset_three.features}")
    logger.info(f"Train dataset grad features: {train_dataset_grad.features}")

    # Training Arguments
    logger.info("Setting up training arguments")
    common_args = {
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 8,
        'num_train_epochs': 3,
        'weight_decay': 0.01,
        'logging_dir': './logs',
        'logging_steps': 10,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'load_best_model_at_end': True,
        'report_to': 'none'  # Avoid sending data to external logging services
    }

    # Initialize trainers for both models
    logger.info("Initializing model trainers")

    # For three-class classification model
    training_args_three = TrainingArguments(
        output_dir='./data/results_three',
        **common_args
    )

    trainer_three = Trainer(
        model=model_three,
        args=training_args_three,
        train_dataset=train_dataset_three,
        eval_dataset=test_dataset_three,
    )

    # For gradient regression model
    training_args_grad = TrainingArguments(
        output_dir='./data/results_grad',
        **common_args
    )

    trainer_grad = Trainer(
        model=model_grad,
        args=training_args_grad,
        train_dataset=train_dataset_grad,
        eval_dataset=test_dataset_grad,
    )

    # Fine-tuning the models
    logger.info("Fine-tuning Three-class Sentiment model...")
    trainer_three.train()

    logger.info("Fine-tuning Gradient-style Sentiment model...")
    trainer_grad.train()

    # Save both models
    logger.info("Saving trained models")
    model_three.save_pretrained('./data/model_three')
    model_grad.save_pretrained('./data/model_grad')

    # Evaluate the models
    logger.info("Evaluating Three-class Sentiment model...")
    three_eval = trainer_three.evaluate()
    logger.info(f"Three-class model evaluation results: {three_eval}")

    logger.info("Evaluating Gradient-style Sentiment model...")
    grad_eval = trainer_grad.evaluate()
    logger.info(f"Gradient model evaluation results: {grad_eval}")

    # Prediction function for sentiment classification
    def predict_sentiment(model, tokenizer, texts):
        # Ensure texts are strings
        texts = [str(text) for text in texts]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits.cpu().detach().numpy()
        return predictions

    # Example predictions with both models
    logger.info("Making example predictions")
    example_texts = df['text'][:5].tolist()  # Get first 5 texts for prediction example

    logger.info("Three-class Sentiment predictions:")
    three_predictions = predict_sentiment(model_three, tokenizer, example_texts)
    # Convert logits to class predictions
    three_class_preds = three_predictions.argmax(axis=1)
    logger.info(f"Raw logits: {three_predictions}")
    logger.info(f"Class predictions: {three_class_preds}")

    logger.info("Gradient-style Sentiment predictions:")
    grad_predictions = predict_sentiment(model_grad, tokenizer, example_texts)
    logger.info(f"Sentiment scores: {grad_predictions.flatten()}")

    logger.info("Model training completed successfully")

except Exception as e:
    logger.error(f"Error during model training: {str(e)}", exc_info=True)
    raise
