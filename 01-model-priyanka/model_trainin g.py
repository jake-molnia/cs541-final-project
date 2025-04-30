import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

accelerator = Accelerator()

df = pd.read_csv('./data/customer_service_with_sentiment_predictions.csv')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

emotion_mapping1 = {'very negative': -1, 'moderately negative': -0.5, 'negative': -0.25,
                   'neutral': 0, 'positive': 0.25, 'moderately positive': 0.5, 'very positive': 1}
priority_mapping2 = {'negative': 0, 'neutral': 1, 'positive': 2}

df['emotion'] = df['emotion'].map(emotion_mapping1)
df['emotions'] = df['priority'].map(priority_mapping2)

model_grad = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)  # Gradient-style sentiment
model_three = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # Three-class sentiment

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

dataset = Dataset.from_pandas(df[['text', 'emotion', 'priority']])
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset, test_dataset = tokenized_datasets.train_test_split(test_size=0.2).values()
training_args = {
    'output_dir': './data/results',
    'evaluation_strategy': "epoch",
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0.01
}

trainer_grad = Trainer(
    model=model_grad,
    args=TrainingArguments(**training_args, output_dir='./data/results_grad'),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer_three = Trainer(
    model=model_three,
    args=TrainingArguments(**training_args, output_dir='./data/results_three'),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Fine-tuning Gradient-style Sentiment model...")
trainer_grad.train()

print("Fine-tuning Three-class Sentiment model...")
trainer_three.train()

model_grad.save_pretrained('./data/model_grad')
model_three.save_pretrained('./data/model_three')

print("\nEvaluating Gradient-style Sentiment model...")
trainer_grad.evaluate()

print("\nEvaluating Three-class Sentiment model...")
trainer_three.evaluate()

def predict_emotion(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits.detach().numpy()
    return predictions

example_texts = df['text'][:5]  # Get first 5 texts for prediction example

print("\nGradient-style Sentiment predictions:")
emotion_predictions = predict_emotion(model_grad, tokenizer, example_texts)
print(emotion_predictions)

print("\nThree-class Sentiment predictions:")
class_predictions = predict_emotion(model_three, tokenizer, example_texts)
print(class_predictions)
