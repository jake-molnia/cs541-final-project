import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import zipfile

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def download_and_prepare_data(kaggle_dataset_url):
    """
    Downloads the dataset from Kaggle and prepares it for training.
    
    Note: You need to have Kaggle API credentials set up in your environment.
    """
    # Extract dataset info from URL
    dataset_owner, dataset_name = kaggle_dataset_url.split('/')[-2:]
    
    # Use Kaggle API to download the dataset
    print(f"Downloading dataset from {kaggle_dataset_url}...")
    os.system(f"kaggle datasets download -d {dataset_owner}/{dataset_name}")
    
    # Unzip the downloaded file using Python's zipfile module
    zip_file = f"{dataset_name}.zip"
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_name)  # Extract into a folder named after the dataset
        print(f"Dataset extracted successfully to {dataset_name}/")
    else:
        print(f"Error: Downloaded file {zip_file} not found.")
        return None

    return dataset_name

def explore_data(df):

    print(df)

    """
    Explores the dataset and prints useful statistics.
    """
    print(f"Dataset shape: {df.shape}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    print("\nSample rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe(include='all').T)
    
    # If there are categorical columns, show their value counts
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols[:5]:  # Limit to first 5 categorical columns to avoid too much output
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts().head(10))
    
    return df

def clean_text(text):
    """
    Cleans text data by removing special characters, URLs, etc.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers, keeping spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data(df, text_column, sentiment_column=None, urgency_column=None):
    """
    Preprocesses the dataset by cleaning text and preparing labels.

    Args:
        df: Pandas DataFrame containing the dataset
        text_column: Column name containing the text data
        sentiment_column: Column name containing sentiment labels (if available)
        urgency_column: Column name containing urgency labels (if available)

    Returns:
        Preprocessed DataFrame ready for model training
    """
    print("Preprocessing data...")

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Clean text
    tqdm.pandas(desc="Cleaning text")
    processed_df['text'] = processed_df[text_column].progress_apply(clean_text)

    # Remove rows with empty text after cleaning
    processed_df = processed_df[processed_df['text'].str.strip() != ""]

    # Handle sentiment labels if provided
    if sentiment_column and sentiment_column in processed_df.columns:
        if processed_df[sentiment_column].dtype == 'object':
            sentiment_mapping = {
                'negative': 0, 'neg': 0, 'not satisfied': 0, 'complaint': 0,
                'neutral': 1, 'neu': 1, 'neither': 1,
                'positive': 2, 'pos': 2, 'satisfied': 2, 'praise': 2
            }

            # Standardize text labels
            processed_df['sentiment_raw'] = processed_df[sentiment_column].str.lower()
            processed_df['sentiment'] = processed_df['sentiment_raw'].map(sentiment_mapping)

            # Handle missing values safely
            if processed_df['sentiment'].isnull().any():
                most_common = processed_df['sentiment'].dropna().mode()
                if not most_common.empty:
                    processed_df['sentiment'] = processed_df['sentiment'].fillna(most_common.iloc[0])
                else:
                    processed_df['sentiment'] = processed_df['sentiment'].fillna(1)  # Default neutral sentiment

            processed_df['sentiment'] = processed_df['sentiment'].replace([np.inf, -np.inf], processed_df['sentiment'].median())
            processed_df['sentiment'] = processed_df['sentiment'].astype(int)

        else:
            processed_df['sentiment'] = processed_df[sentiment_column]
    else:
        print("Sentiment column not provided, generating synthetic labels...")
        processed_df['sentiment'] = np.random.choice([0, 1, 2], size=len(processed_df))

    # Handle urgency labels if provided
    if urgency_column and urgency_column in processed_df.columns:
        if processed_df[urgency_column].dtype == 'object':
            urgency_mapping = {
                'low': 0, 'normal': 0, 'medium': 1, 'standard': 1,
                'high': 2, 'urgent': 2, 'critical': 2
            }

            # Standardize text labels
            processed_df['urgency_raw'] = processed_df[urgency_column].str.lower()
            processed_df['urgency'] = processed_df['urgency_raw'].map(urgency_mapping)

            # Handle missing values safely
            if processed_df['urgency'].isnull().any():
                most_common = processed_df['urgency'].dropna().mode()
                if not most_common.empty:
                    processed_df['urgency'] = processed_df['urgency'].fillna(most_common.iloc[0])
                else:
                    processed_df['urgency'] = processed_df['urgency'].fillna(1)  # Default medium urgency

            processed_df['urgency'] = processed_df['urgency'].replace([np.inf, -np.inf], processed_df['urgency'].median())
            processed_df['urgency'] = processed_df['urgency'].astype(int)

        else:
            processed_df['urgency'] = processed_df[urgency_column]
    else:
        print("Urgency column not provided, generating synthetic labels...")
        processed_df['urgency'] = np.random.choice([0, 1, 2], size=len(processed_df))

    # Print label distributions for debugging
    print("\nSentiment distribution:")
    print(processed_df['sentiment'].value_counts())

    print("\nUrgency distribution:")
    print(processed_df['urgency'].value_counts())

    # Keep only the columns we need
    final_df = processed_df[['text', 'sentiment', 'urgency']]

    print(f"Preprocessing complete. Final dataset shape: {final_df.shape}")

    return final_df

def plot_data_distribution(df):
    """
    Plots distribution of labels in the dataset.
    """
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment (0=negative, 1=neutral, 2=positive)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='urgency', data=df)
    plt.title('Urgency Distribution')
    plt.xlabel('Urgency (0=low, 1=medium, 2=high)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.show()

def plot_text_length_distribution(df):
    """
    Plots distribution of text lengths in the dataset.
    """
    df['text_length'] = df['text'].apply(len)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=50)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.axvline(x=512, color='red', linestyle='--', label='BERT max length (512 tokens)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('text_length_distribution.png')
    plt.show()
    
    # Print statistics about text length
    print("\nText length statistics:")
    print(df['text_length'].describe())
    
    # Show percentage of texts longer than BERT's max length
    pct_long = (df['text_length'] > 512).mean() * 100
    print(f"\n{pct_long:.2f}% of texts are longer than 512 characters")

def save_processed_data(df, output_file='multilingual_customer_support_tickets.csv'):
    """
    Saves the processed data to a CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

def main():
    """
    Main function to run the data preparation pipeline.
    """
    # URL for the Kaggle dataset
    kaggle_dataset_url = 'https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets'
    
    # Download and load the dataset
    folder = download_and_prepare_data(kaggle_dataset_url)
    
    if folder is None:
        print("Failed to download or load the dataset.")
        print("As an alternative, you can manually download the dataset from Kaggle and place the CSV file in the current directory.")
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        if csv_files:
            print(f"Found CSV files: {csv_files}")
            filename = input("Enter the name of the CSV file to use: ")
            if os.path.exists(filename):
                df = pd.read_csv(filename)
            else:
                print(f"File {filename} not found.")
                return
        else:
            print("No CSV files found in the current directory.")
            return
    
    df1 = pd.read_csv('multilingual-customer-support-tickets/dataset-tickets-multi-lang-4-20k.csv')
    df2 = pd.read_csv('multilingual-customer-support-tickets/dataset-tickets-multi-lang3-4k.csv')

    df = pd.concat([df1, df2], ignore_index=True)

    # Explore the raw data
    #explore_data(df)
        
    text_column = 'body'  
    sentiment_column = 'answer'  
    urgency_column = 'priority'  
    
    # Preprocess the data
    processed_df = preprocess_data(df, text_column, sentiment_column, urgency_column)
    
    # Plot data distributions
    plot_data_distribution(processed_df)
    plot_text_length_distribution(processed_df)
    
    # Save the processed data
    save_processed_data(processed_df)
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()