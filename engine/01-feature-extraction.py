import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

def load_preprocessed_data(data_dir):
    """Load the preprocessed data from the previous step"""
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    with open(os.path.join(data_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Also reload original data to access raw text
    enron_df, spam_df = load_original_datasets(data_dir)
    combined_df = create_combined_df(enron_df, spam_df)
    
    return X_train, X_test, y_train, y_test, tokenizer, combined_df

def load_original_datasets(data_dir):
    """Load the original dataset files"""
    enron_csv_path = os.path.join(data_dir, "enron_emails.csv")
    spam_csv_path = os.path.join(data_dir, "spam_emails.csv")
    
    enron_df = pd.read_csv(enron_csv_path)
    spam_df = pd.read_csv(spam_csv_path)
    
    return enron_df, spam_df

def extract_subject_from_message(message):
    """Extract subject from email message if available"""
    if not isinstance(message, str):
        return ""
    
    # Try to find Subject line in email headers
    match = re.search(r'Subject: (.*?)(?:\r?\n|\r|$)', message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def create_combined_df(enron_df, spam_df):
    """Create a combined dataframe with the original content"""
    # Process Enron emails
    # Extract subject from message for Enron dataset
    enron_subjects = enron_df['message'].apply(extract_subject_from_message)
    
    enron_processed = pd.DataFrame({
        'content': enron_df['message'].fillna(''),
        'subject': enron_subjects,
        'is_spam': False
    })
    
    # Process spam emails
    # For spam, check if the subject is already in the message or extract it
    spam_subjects = spam_df['text'].apply(extract_subject_from_message)
    
    spam_processed = pd.DataFrame({
        'content': spam_df['text'].fillna(''),
        'subject': spam_subjects,
        'is_spam': True
    })

    # Combine the datasets
    return pd.concat([enron_processed, spam_processed], ignore_index=True)

def extract_sentiment_features(df):
    """Extract sentiment-related features from email content"""
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to content
    print("Analyzing content sentiment...")
    sentiment_scores = df['content'].apply(lambda x: sia.polarity_scores(str(x)))
    
    df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    
    # Also analyze subject sentiment if available
    print("Analyzing subject sentiment...")
    df['subject_sentiment'] = df['subject'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    
    return df

def extract_urgency_features(df):
    """Extract features that indicate urgency"""
    # Urgency keywords in subject or content
    urgency_keywords = ['urgent', 'immediately', 'asap', 'deadline', 'important', 
                         'critical', 'priority', 'attention', 'emergency', 'quick',
                         'soon', 'today', 'tomorrow', 'needed', 'required', 'fast']
    
    # Check for urgency keywords in subject
    print("Checking urgency in subjects...")
    df['urgent_subject'] = df['subject'].apply(
        lambda x: any(keyword in str(x).lower() for keyword in urgency_keywords)
    )
    
    # Check for urgency keywords in content
    print("Checking urgency in content...")
    df['urgent_content'] = df['content'].apply(
        lambda x: any(keyword in str(x).lower() for keyword in urgency_keywords)
    )
    
    # Check for exclamation marks (often indicate urgency)
    df['exclamation_count'] = df['content'].apply(
        lambda x: str(x).count('!')
    )
    
    # Check for question marks (often indicate requests)
    df['question_count'] = df['content'].apply(
        lambda x: str(x).count('?')
    )
    
    # Check for ALL CAPS words (often indicate emphasis)
    df['caps_word_count'] = df['content'].apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 2)
    )
    
    # Count time-related words indicating imminent deadlines
    time_indicators = ['today', 'tomorrow', 'tonight', 'morning', 'afternoon', 'evening',
                       'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'asap']
    
    df['time_indicator_count'] = df['content'].apply(
        lambda x: sum(str(x).lower().count(indicator) for indicator in time_indicators)
    )
    
    return df

def create_priority_labels(df):
    """
    Create priority labels based on sentiment and urgency features
    Priority levels: 0 (Low), 1 (Medium), 2 (High)
    """
    # Start with a base priority score
    df['priority_score'] = 0
    
    # Increase priority for negative sentiment (often indicates problems/issues)
    df.loc[df['sentiment_compound'] < -0.2, 'priority_score'] += 1
    df.loc[df['sentiment_compound'] < -0.5, 'priority_score'] += 1
    
    # Increase priority for urgency indicators
    df.loc[df['urgent_subject'], 'priority_score'] += 2
    df.loc[df['urgent_content'], 'priority_score'] += 1
    df.loc[df['exclamation_count'] > 2, 'priority_score'] += 1
    df.loc[df['caps_word_count'] > 3, 'priority_score'] += 1
    df.loc[df['time_indicator_count'] > 2, 'priority_score'] += 1
    
    # Convert scores to priority levels
    df['priority'] = pd.cut(
        df['priority_score'], 
        bins=[-1, 1, 3, 100], 
        labels=[0, 1, 2]  # Low, Medium, High
    )
    
    # Convert to integer for model training
    df['priority'] = df['priority'].astype(int)
    
    # Spam emails should always be low priority
    df.loc[df['is_spam'], 'priority'] = 0
    
    print(f"Priority distribution: {df['priority'].value_counts()}")
    
    return df

def save_feature_data(data_dir, df):
    """Save the extended feature dataframe"""
    # Save the full dataframe with features
    df.to_csv(os.path.join(data_dir, 'email_features.csv'), index=False)
    
    # Save just the feature matrix and priority labels for modeling
    X_features = df[[
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'subject_sentiment', 'urgent_subject', 'urgent_content',
        'exclamation_count', 'question_count', 'caps_word_count', 'time_indicator_count'
    ]].values
    
    y_priority = df['priority'].values
    
    np.save(os.path.join(data_dir, 'X_features.npy'), X_features)
    np.save(os.path.join(data_dir, 'y_priority.npy'), y_priority)
    
    print(f"Features extracted and saved. Shape: {X_features.shape}")
    print(f"Priority distribution: {np.bincount(y_priority)}")

def main():
    """Main function to execute the feature extraction pipeline"""
    data_dir = "data"
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, tokenizer, combined_df = load_preprocessed_data(data_dir)
    
    # Extract features
    print("Extracting sentiment features...")
    combined_df = extract_sentiment_features(combined_df)
    
    print("Extracting urgency features...")
    combined_df = extract_urgency_features(combined_df)
    
    print("Creating priority labels...")
    combined_df = create_priority_labels(combined_df)
    
    print("Saving feature data...")
    save_feature_data(data_dir, combined_df)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()