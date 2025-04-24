import os
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data if needed
nltk.download('vader_lexicon')

def load_model_and_dependencies(data_dir):
    """Load the trained model and necessary dependencies"""
    # Load priority model
    with open(os.path.join(data_dir, 'priority_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    return model

def extract_email_features(email_content, email_subject=""):
    """Extract features from a single email"""
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(email_content)
    subject_sentiment = sia.polarity_scores(email_subject)
    
    # Define urgency keywords
    urgency_keywords = ['urgent', 'immediately', 'asap', 'deadline', 'important', 
                        'critical', 'priority', 'attention', 'emergency', 'quick',
                        'soon', 'today', 'tomorrow', 'needed', 'required', 'fast']
    
    # Check for urgency indicators
    urgent_subject = any(keyword in email_subject.lower() for keyword in urgency_keywords)
    urgent_content = any(keyword in email_content.lower() for keyword in urgency_keywords)
    exclamation_count = email_content.count('!')
    question_count = email_content.count('?')
    caps_word_count = sum(1 for word in email_content.split() if word.isupper() and len(word) > 2)
    
    # Count time indicators
    time_indicators = ['today', 'tomorrow', 'tonight', 'morning', 'afternoon', 'evening',
                       'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'asap']
    time_indicator_count = sum(email_content.lower().count(indicator) for indicator in time_indicators)
    
    # Create feature vector
    features = np.array([
        sentiment_scores['neg'],
        sentiment_scores['neu'],
        sentiment_scores['pos'],
        sentiment_scores['compound'],
        subject_sentiment['compound'],
        int(urgent_subject),
        int(urgent_content),
        exclamation_count,
        question_count,
        caps_word_count,
        time_indicator_count
    ]).reshape(1, -1)  # Reshape for single prediction
    
    return features

def predict_priority(model, features):
    """Predict priority level for an email"""
    priority_level = model.predict(features)[0]
    priority_probs = model.predict_proba(features)[0]
    
    priority_names = ['Low', 'Medium', 'High']
    priority_name = priority_names[priority_level]
    
    return priority_level, priority_name, priority_probs

def display_results(priority_level, priority_name, priority_probs):
    """Display prediction results"""
    print(f"\nPredicted Priority: {priority_name} (Level {priority_level})")
    print(f"Confidence scores: Low: {priority_probs[0]:.2f}, "
          f"Medium: {priority_probs[1]:.2f}, High: {priority_probs[2]:.2f}")

def main():
    """Main function for email priority prediction"""
    data_dir = "data"
    model = load_model_and_dependencies(data_dir)
    
    print("Email Priority Prediction")
    print("------------------------")
    print("Enter 'q' to quit when prompted for the subject")
    
    while True:
        email_subject = input("\nEnter email subject: ")
        if email_subject.lower() == 'q':
            break
            
        email_content = input("Enter email content: ")
        features = extract_email_features(email_content, email_subject)
        priority_level, priority_name, priority_probs = predict_priority(model, features)
        display_results(priority_level, priority_name, priority_probs)

if __name__ == "__main__":
    main()