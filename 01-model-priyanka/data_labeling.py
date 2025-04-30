import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

df = pd.read_csv('./data/dataset-tickets-multi-lang-4-20k.csv')
df = df[df['language'] == 'en']
df['text'] = df['subject'] + ' ' + df['body']
df['text'] = df['text'].astype(str)
df.dropna(subset=['text'], inplace=True)

emotion_mapping1 = {'very negative': -1, 'moderately negative': -0.5, 'negative': -0.25,
                   'neutral': 0, 'positive': 0.25, 'moderately positive': 0.5, 'very positive': 1}
priority_mapping2 = {'negative': 0, 'neutral': 1, 'positive': 2}

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # For 3-class sentiment

def predict_sentiment(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).numpy()  # Get the class with the highest score (negative=0, neutral=1, positive=2)
    return predictions

df['sentiment_predicted'] = predict_sentiment(model, tokenizer, df['text'].tolist())
df['emotion'] = df['sentiment_predicted'].map({0: 'negative', 1: 'neutral', 2: 'positive'})
df['emotion_numeric'] = df['emotion'].map(emotion_mapping1)
df.to_csv('./data/customer_service_with_sentiment_predictions.csv', index=False)
print(df[['text', 'sentiment_predicted', 'emotion', 'emotion_numeric']].head())
