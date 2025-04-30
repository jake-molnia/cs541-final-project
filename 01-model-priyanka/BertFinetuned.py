import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()

# Load the customer service dataset
df = pd.read_csv('dataset-tickets-multi-lang-4-20k.csv')

# Preprocess the data: Filter for English language, handle NaNs, and create a combined text column
df = df[df['language'] == 'en']
df['text'] = df['subject'] + ' ' + df['body']
df['text'] = df['text'].astype(str)
df.dropna(subset=['text'], inplace=True)

# Emotion Mapping for sentiment labels
emotion_mapping1 = {'very negative': -1, 'moderately negative': -0.5, 'negative': -0.25,
                   'neutral': 0, 'positive': 0.25, 'moderately positive': 0.5, 'very positive': 1}
priority_mapping2 = {'negative': 0, 'neutral': 1, 'positive': 2}

# 1. Add sentiment predictions to the dataframe using DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # For 3-class sentiment

def predict_sentiment(model, tokenizer, texts):
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted sentiment
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).numpy()  # Get the class with the highest score (negative=0, neutral=1, positive=2)
    return predictions

# Get sentiment predictions for customer service dataset
df['sentiment_predicted'] = predict_sentiment(model, tokenizer, df['text'].tolist())

# Map the sentiment predictions to emotion scale
df['emotion'] = df['sentiment_predicted'].map({0: 'negative', 1: 'neutral', 2: 'positive'})

# Now, map the sentiment to the emotion numeric scale as defined earlier
df['emotion_numeric'] = df['emotion'].map(emotion_mapping1)

# Save the predictions to a new file (after sentiment prediction)
df.to_csv('customer_service_with_sentiment_predictions.csv', index=False)

# Verify the first few rows of the DataFrame with added sentiment predictions
print(df[['text', 'sentiment_predicted', 'emotion', 'emotion_numeric']].head())

# 2. Now proceed with training models after sentiment prediction has been added
# Map the labels to a continuous scale for Gradient-style sentiment
df['emotion'] = df['emotion'].map(emotion_mapping1)
df['emotions'] = df['priority'].map(priority_mapping2)

# Tokenizer and model initialization for DistilBERT
model_grad = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)  # Gradient-style sentiment
model_three = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # Three-class sentiment

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'emotion', 'priority']])

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test
train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()

# Training Arguments
training_args = {
    'output_dir': './results',
    'evaluation_strategy': "epoch",
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0.01
}

# Initialize trainers for both models
trainer_grad = Trainer(
    model=model_grad,
    args=TrainingArguments(**training_args, output_dir='./results_grad'),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer_three = Trainer(
    model=model_three,
    args=TrainingArguments(**training_args, output_dir='./results_three'),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tuning the models
print("Fine-tuning Gradient-style Sentiment model...")
trainer_grad.train()

print("Fine-tuning Three-class Sentiment model...")
trainer_three.train()

# Save both models
model_grad.save_pretrained('./model_grad')
model_three.save_pretrained('./model_three')

# Evaluate the models
print("\nEvaluating Gradient-style Sentiment model...")
trainer_grad.evaluate()

print("\nEvaluating Three-class Sentiment model...")
trainer_three.evaluate()

# Prediction function for emotion classification
def predict_emotion(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits.detach().numpy()
    return predictions

# Example predictions with both models
example_texts = df['text'][:5]  # Get first 5 texts for prediction example

print("\nGradient-style Sentiment predictions:")
emotion_predictions = predict_emotion(model_grad, tokenizer, example_texts)
print(emotion_predictions)

print("\nThree-class Sentiment predictions:")
class_predictions = predict_emotion(model_three, tokenizer, example_texts)
print(class_predictions)
