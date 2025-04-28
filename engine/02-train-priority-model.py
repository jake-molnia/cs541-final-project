import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_data(data_dir):
    """Load the feature data created in the previous step"""
    # Load train data
    X_train = np.load(os.path.join(data_dir, 'X_train_features.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(data_dir, 'y_train_priority.npy'), allow_pickle=True)
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test_features.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(data_dir, 'y_test_priority.npy'), allow_pickle=True)
    
    # Load feature names
    with open(os.path.join(data_dir, 'feature_names.pickle'), 'rb') as f:
        feature_names = pickle.load(f)
    
    # Also load the full dataframe for reference
    feature_df = pd.read_csv(os.path.join(data_dir, 'email_features.csv'))
    
    print(f"Loaded train features: {X_train.shape}")
    print(f"Loaded test features: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names, feature_df

def train_priority_model(X_train, y_train):
    """Train a model to predict email priority with cross-validation"""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, f1_score
    
    # Random Forest with more conservative parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=15,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    # Perform cross-validation to check model performance
    f1_scorer = make_scorer(f1_score, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
    
    # Train the final model
    print("Training priority model...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model with more detailed metrics"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    y_pred = model.predict(X_test)
    
    # Probability predictions (for ROC curves)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        
        # Calculate ROC AUC for multiclass
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            print(f"ROC AUC: {roc_auc:.4f}")
        except:
            print("Could not calculate ROC AUC")
    
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
    plt.savefig('confusion_matrix.png')
    
    # Analyze misclassifications
    misclassified = X_test[y_test != y_pred]
    print(f"Number of misclassified samples: {len(misclassified)}")
    
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
    
    # Load feature data with pre-defined splits
    print("Loading feature data...")
    X_train, X_test, y_train, y_test, feature_names, feature_df = load_feature_data(data_dir)
    
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