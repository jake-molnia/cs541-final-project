import os
from typing import Tuple, List, Dict, Any

def load_datasets(data_dir: str) -> Tuple[Any, Any]:
    """
    Load Enron and spam email datasets.

    Args:
        data_dir: Directory containing the dataset files

    Returns:
        Tuple of (enron_df, spam_df) pandas DataFrames
    """
    import pandas as pd

    enron_csv_path = os.path.join(data_dir, "enron_emails.csv")
    spam_csv_path = os.path.join(data_dir, "spam_emails.csv")

    print(f"Loading Enron emails from {enron_csv_path}...")
    enron_df = pd.read_csv(enron_csv_path)

    print(f"Loading Spam emails from {spam_csv_path}...")
    spam_df = pd.read_csv(spam_csv_path)

    print(f"Enron: {enron_df.shape}, columns: {enron_df.columns.tolist()}")
    print(f"Spam: {spam_df.shape}, columns: {spam_df.columns.tolist()}")

    return enron_df, spam_df

def create_standardized_dataframes(enron_df: Any, spam_df: Any) -> Any:
    """
    Create standardized dataframes with consistent columns.

    Args:
        enron_df: DataFrame with Enron emails
        spam_df: DataFrame with spam emails

    Returns:
        Combined DataFrame with balanced classes
    """
    import pandas as pd

    enron_processed = pd.DataFrame({
        'content': enron_df['message'].fillna(''),
        'is_spam': False
    })

    spam_processed = pd.DataFrame({
        'content': spam_df['text'].fillna(''),
        'is_spam': True
    })

    combined_df = pd.concat([enron_processed, spam_processed], ignore_index=True)

    print(f"Combined dataset: {combined_df.shape}")
    print(f"Class distribution: {combined_df['is_spam'].value_counts()}")

    return combined_df

def clean_text(text: str) -> str:
    """
    Clean and normalize email text.

    Args:
        text: Input text string

    Returns:
        Cleaned text string
    """
    import re

    text = re.sub(r'<.*?>', ' ', str(text))  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)     # Remove punctuation
    text = re.sub(r'\s+', ' ', text)         # Remove extra spaces
    return text.lower().strip()

def preprocess_text(combined_df: Any) -> Any:
    """
    Clean and preprocess the text in the dataframe.

    Args:
        combined_df: DataFrame with email content

    Returns:
        DataFrame with added cleaned_content column
    """
    combined_df['cleaned_content'] = combined_df['content'].apply(clean_text)
    print("Text cleaning complete")
    return combined_df

def tokenize_text(cleaned_texts: List[str], max_words: int, max_sequence_length: int) -> Tuple[Any, Any, Dict[str, int]]:
    """
    Tokenize and pad text sequences.

    Args:
        cleaned_texts: List of cleaned text strings
        max_words: Maximum number of words in the vocabulary
        max_sequence_length: Maximum length of padded sequences

    Returns:
        Tuple of (padded_sequences, tokenizer, word_index)
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    word_index = tokenizer.word_index

    print(f"Found {len(word_index)} unique tokens")

    data = pad_sequences(sequences, maxlen=max_sequence_length)
    return data, tokenizer, word_index

def split_data(data: Any, labels: Any, test_size: float, random_state: int) -> Tuple[Any, Any, Any, Any]:
    """
    Split data into training and test sets.

    Args:
        data: Preprocessed text data
        labels: Target labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(data_dir: str, X_train: Any, X_test: Any,
                         y_train: Any, y_test: Any, tokenizer: Any) -> None:
    """
    Save preprocessed data and tokenizer.

    Args:
        data_dir: Directory to save files
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        tokenizer: Fitted tokenizer
    """
    import numpy as np
    import pickle

    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    with open(os.path.join(data_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Preprocessing complete. Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Files saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy, tokenizer.pickle")

def main() -> None:
    """Main function to execute the email preprocessing pipeline."""
    data_dir = os.path.join("scratch", "data")

    ###############################Parameters#################################
    max_words = 10000  # Dictionary size
    max_sequence_length = 1000  # Max email length
    test_size = 0.2
    random_state = 42
    ##########################################################################

    enron_df, spam_df = load_datasets(data_dir)
    combined_df = create_standardized_dataframes(enron_df, spam_df)
    combined_df = preprocess_text(combined_df)
    data, tokenizer, word_index = tokenize_text(
        combined_df['cleaned_content'], max_words, max_sequence_length
    )
    labels = combined_df['is_spam'].values
    X_train, X_test, y_train, y_test = split_data(
        data, labels, test_size, random_state
    )
    save_preprocessed_data(data_dir, X_train, X_test, y_train, y_test, tokenizer)

if __name__ == "__main__":
    main()
