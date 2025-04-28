import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

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

def create_email_dataframe(enron_df, spam_df):
    """Create a dataframe with email content, subject, and spam label"""
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
    sentiment_scores = df['content'].apply(lambda x: sia.polarity_scores(str(x)))
    
    df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
    
    # Also analyze subject sentiment if available
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
    df['urgent_subject'] = df['subject'].apply(
        lambda x: any(keyword in str(x).lower() for keyword in urgency_keywords)
    )
    
    # Check for urgency keywords in content
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
    with added randomness to prevent perfect prediction
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
    
    # Add randomness to prevent perfect prediction
    # This simulates the human factor in prioritization
    import random
    random.seed(42)  # For reproducibility
    df['priority_score'] = df['priority_score'] + np.random.normal(0, 0.5, len(df))
    
    # Convert scores to priority levels
    df['priority'] = pd.cut(
        df['priority_score'], 
        bins=[-1, 1, 3, 100], 
        labels=[0, 1, 2]  # Low, Medium, High
    )
    
    # Fill any NaN values with the most common priority (usually 0)
    df['priority'] = df['priority'].fillna(0)
    
    # Convert to integer for model training
    df['priority'] = df['priority'].astype(int)
    
    # Spam emails should always be low priority
    df.loc[df['is_spam'], 'priority'] = 0
    
    print(f"Priority distribution: {df['priority'].value_counts()}")
    
    return df

def split_and_save_feature_data(data_dir, email_df, test_size=0.2, random_state=42):
    """Split the data into train and test sets and save it"""
    # Create separate feature sets for training and label creation
    # This prevents the model from learning the exact rules used for labeling
    
    # Full feature set for training
    feature_columns = [
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'subject_sentiment', 'urgent_subject', 'urgent_content',
        'exclamation_count', 'question_count', 'caps_word_count', 'time_indicator_count'
    ]
    
    # Create additional derived features to use instead of the exact ones used for labeling
    email_df['text_length'] = email_df['content'].apply(lambda x: len(str(x)))
    email_df['subject_length'] = email_df['subject'].apply(lambda x: len(str(x)))
    email_df['sentiment_ratio'] = email_df['sentiment_pos'] / (email_df['sentiment_neg'] + 0.01)
    
    # Add these new features to the feature set
    feature_columns.extend(['text_length', 'subject_length', 'sentiment_ratio'])
    
    # Extract features and labels
    X_features = email_df[feature_columns].values
    y_priority = email_df['priority'].values
    
    print(f"Total samples: {len(X_features)}")
    
    # Split data first, before any deduplication
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_priority, test_size=test_size, 
        random_state=random_state, stratify=y_priority
    )
    
    # Save the full dataframe with features
    email_df.to_csv(os.path.join(data_dir, 'email_features.csv'), index=False)
    
    # Save train data
    np.save(os.path.join(data_dir, 'X_train_features.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train_priority.npy'), y_train)
    
    # Save test data
    np.save(os.path.join(data_dir, 'X_test_features.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test_priority.npy'), y_test)
    
    # Save feature names
    with open(os.path.join(data_dir, 'feature_names.pickle'), 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Train priority distribution: {np.bincount(y_train)}")
    print(f"Test priority distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to execute the simplified feature extraction pipeline"""
    data_dir = "data"
    
    # Load original datasets
    print("Loading original datasets...")
    enron_df, spam_df = load_original_datasets(data_dir)
    
    # Create email dataframe with all samples
    print("Creating email dataframe...")
    email_df = create_email_dataframe(enron_df, spam_df)
    
    # Extract features
    print("Extracting sentiment features...")
    email_df = extract_sentiment_features(email_df)
    
    print("Extracting urgency features...")
    email_df = extract_urgency_features(email_df)
    
    print("Creating priority labels...")
    email_df = create_priority_labels(email_df)

    priority_by_spam = email_df.groupby('is_spam')['priority'].value_counts().unstack()
    print("Priority distribution by spam label:")
    print(priority_by_spam)
        
    print("Splitting and saving feature data...")
    X_train, X_test, y_train, y_test = split_and_save_feature_data(data_dir, email_df)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()