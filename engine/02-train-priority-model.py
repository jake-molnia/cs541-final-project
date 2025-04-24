import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_data(data_dir):
    """Load the feature data created in the previous step"""
    X_features = np.load(os.path.join(data_dir, 'X_features.npy'), allow_pickle=True)
    y_priority = np.load(os.path.join(data_dir, 'y_priority.npy'), allow_pickle=True)
    feature_df = pd.read_csv(os.path.join(data_dir, 'email_features.csv'))
    
    return X_features, y_priority, feature_df

def split_feature_data(X_features, y_priority, test_size=0.2, random_state=42):
    """Split the feature data into training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_priority, test_size=test_size, 
        random_state=random_state, stratify=y_priority
    )
    
    return X_train, X_test, y_train, y_test

def train_priority_model(X_train, y_train):
    """Train a model to predict email priority"""
    # Random Forest often works well for this type of classification
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    
    # If you have time for hyperparameter tuning:
    '''
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    '''
    
    print("Training priority model...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    
    print("\nModel Evaluation:")
    print("Priority Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Low', 'Medium', 'High']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('data/confusion_matrix.png')
    
    return y_pred

def get_feature_importance(model, feature_names):
    """Extract and plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = model.best_estimator_.feature_importances_
        
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance for Priority Classification')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    
    # Print importance ranking
    print("\nFeature Importance Ranking:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def save_model(model, data_dir):
    """Save the trained model"""
    with open(os.path.join(data_dir, 'priority_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {os.path.join(data_dir, 'priority_model.pkl')}")

def main():
    """Main function to train the priority model"""
    data_dir = "data"
    
    # Load feature data
    X_features, y_priority, feature_df = load_feature_data(data_dir)
    
    # Get feature names for importance analysis
    feature_names = [
        'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
        'subject_sentiment', 'urgent_subject', 'urgent_content',
        'exclamation_count', 'question_count', 'caps_word_count', 'time_indicator_count'
    ]
    
    # Split data
    X_train, X_test, y_train, y_test = split_feature_data(X_features, y_priority)
    
    # Train model
    model = train_priority_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Analyze feature importance
    get_feature_importance(model, feature_names)
    
    # Save model
    save_model(model, data_dir)

if __name__ == "__main__":
    main()