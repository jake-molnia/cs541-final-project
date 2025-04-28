import os
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data if needed
nltk.download('vader_lexicon', quiet=True)

def load_model_and_dependencies(data_dir):
    """Load the trained model and necessary dependencies"""
    # Load priority model
    with open(os.path.join(data_dir, 'priority_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    X_test_features = np.load(os.path.join(data_dir, 'X_features.npy'), allow_pickle=True)
    y_test_priority = np.load(os.path.join(data_dir, 'y_priority.npy'), allow_pickle=True)
    feature_df = pd.read_csv(os.path.join(data_dir, 'email_features.csv'))
    
    # Get test set indices - assuming we're using the same train/test split from script 02
    from sklearn.model_selection import train_test_split
    _, X_test_indices = train_test_split(
        np.arange(len(y_test_priority)), 
        test_size=0.2, 
        random_state=42, 
        stratify=y_test_priority
    )
    
    # Extract test set
    X_test = X_test_features[X_test_indices]
    y_test = y_test_priority[X_test_indices]
    
    # Extract corresponding email content and subjects for test set
    test_df = feature_df.iloc[X_test_indices]
    
    return model, X_test, y_test, test_df

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    print("\nModel Evaluation on Test Data:")
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
    plt.savefig('data/test_confusion_matrix.png')
    
    return y_pred

def show_example_predictions(model, test_df, y_test, y_pred, num_examples=5):
    """Show example predictions from the test set"""
    priority_names = ['Low', 'Medium', 'High']
    
    # Get indices of correct and incorrect predictions
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    print("\n--- CORRECTLY CLASSIFIED EXAMPLES ---")
    for i in range(min(num_examples, len(correct_indices))):
        idx = correct_indices[i]
        print(f"\nExample {i+1}:")
        print(f"Subject: {test_df.iloc[idx]['subject']}")
        print(f"Content excerpt: {test_df.iloc[idx]['content'][:100]}...")
        print(f"True Priority: {priority_names[y_test[idx]]}")
        print(f"Predicted Priority: {priority_names[y_pred[idx]]}")
        print("-" * 50)
    
    print("\n--- MISCLASSIFIED EXAMPLES ---")
    for i in range(min(num_examples, len(incorrect_indices))):
        idx = incorrect_indices[i]
        print(f"\nExample {i+1}:")
        print(f"Subject: {test_df.iloc[idx]['subject']}")
        print(f"Content excerpt: {test_df.iloc[idx]['content'][:100]}...")
        print(f"True Priority: {priority_names[y_test[idx]]}")
        print(f"Predicted Priority: {priority_names[y_pred[idx]]}")
        print("-" * 50)

def main():
    """Main function for testing email priority model against test data"""
    data_dir = "data"
    
    print("Email Priority Prediction - Test Evaluation")
    print("------------------------------------------")
    
    # Load model and test data
    model, X_test, y_test, test_df = load_model_and_dependencies(data_dir)
    
    # Evaluate model on test data
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Show example predictions
    show_example_predictions(model, test_df, y_test, y_pred)
    
    print("\nTest evaluation complete!")

if __name__ == "__main__":
    main()