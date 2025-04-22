import os
import sys
import numpy as np
import pandas as pd

def setup_kaggle_credentials():
    """
    Set up Kaggle credentials for API access
    Either from environment variables or by prompting the user
    """
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        print("Kaggle credentials not found in environment variables.")
        print("You need a Kaggle account to download the dataset.")
        print("Get your API key from https://www.kaggle.com/<username>/account")

        kaggle_username = input("Enter your Kaggle username: ")
        kaggle_key = input("Enter your Kaggle API key: ")
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
    print("Kaggle credentials set up successfully.")

def download_datasets(output_dir="data", enron_filename="enron_emails.csv", spam_filename="spam_emails.csv"):
    """
    Download the Enron email dataset and Spam email dataset from Kaggle

    Args:
        output_dir (str): Directory to save the datasets
        enron_filename (str): Desired filename for the Enron dataset
        spam_filename (str): Desired filename for the Spam dataset

    Returns:
        tuple: Paths to the downloaded CSV files
    """
    import kaggle
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Downloading Enron email dataset to {output_dir}...")

    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'wcukierski/enron-email-dataset',
            path=output_dir,
            unzip=True
        )
        os.rename(os.path.join(output_dir, "emails.csv"), os.path.join(output_dir, enron_filename))
        kaggle.api.dataset_download_files(
            "jackksoncsie/spam-email-dataset",
            path=output_dir,
            unzip=True
        )
        os.rename(os.path.join(output_dir, "emails.csv"), os.path.join(output_dir, spam_filename))
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

def load_emails_from_csv(csv_path):
    """
    Load Enron emails from the downloaded CSV file

    Args:
        csv_path (str): Path to the emails CSV file

    Returns:
        pandas.DataFrame: DataFrame containing emails
    """
    print(f"Loading emails from {csv_path}...")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.fillna("")

    print(f"Total emails loaded: {len(df)}")
    return df

def main():
    print("Starting Enron email dataset processing")

    setup_kaggle_credentials()
    data_dir = os.path.join(os.getcwd(), "data")
    spam_csv_path = os.path.join(data_dir, "spam_emails.csv")
    enron_csv_path = os.path.join(data_dir, "enron_emails.csv")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    download_datasets(data_dir, "enron_emails.csv", "spam_emails.csv")

    # Load and save Enron dataset
    enron_emails_df = load_emails_from_csv(enron_csv_path)
    enron_pickle_path = os.path.join(data_dir, "enron_emails_df.pkl")
    enron_emails_df.to_pickle(enron_pickle_path)
    print(f"Enron emails saved to {enron_pickle_path}")

    # Load and save Spam dataset
    spam_emails_df = load_emails_from_csv(spam_csv_path)
    spam_pickle_path = os.path.join(data_dir, "spam_emails_df.pkl")
    spam_emails_df.to_pickle(spam_pickle_path)
    print(f"Spam emails saved to {spam_pickle_path}")

if __name__ == "__main__":
    main()
