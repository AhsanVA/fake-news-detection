
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor

def run_eda():
    loader = DataLoader()
    processor = TextPreprocessor()
    
    print("Loading Data...")
    train_df, val_df, test_df = loader.load_liar_dataset()
    
    if train_df is None:
        print("Failed to load dataset. Please check internet connection or manual file placement.")
        return

    print(f"Training Set Shape: {train_df.shape}")
    print(f"Validation Set Shape: {val_df.shape}")
    print(f"Test Set Shape: {test_df.shape}")
    
    # LIAR Label mapping (simplification for binary classification)
    # 0: false, 1: half-true, 2: mostly-true, 3: true, 4: barely-true, 5: pants-fire
    # Approximation: 
    # Real: true, mostly-true, half-true
    # Fake: false, barely-true, pants-fire
    # Note: HuggingFace 'liar' dataset labels might be integers.
    # Let's inspect the first few rows to confirm label format if we could, 
    # but based on documentation, they are often class integers 0-5.
    
    print("\nInitial Label Distribution (Train):")
    print(train_df['label'].value_counts())
    
    # Check if 'statement' column exists (input text)
    if 'statement' not in train_df.columns:
        print("Error: 'statement' column missing.")
        return

    print("\nSample Preprocessing:")
    sample_text = train_df.iloc[0]['statement']
    print(f"Original: {sample_text}")
    print(f"Processed: {processor.preprocess(sample_text)}")
    
    # Save a small sample to processed to verify write permissions
    print("\nSaving sample processed data...")
    train_df['clean_text'] = train_df['statement'].apply(processor.preprocess)
    loader.save_to_processed(train_df.head(100), "train_sample_processed.csv")
    print("EDA Complete. Sample saved to data/processed/train_sample_processed.csv")

if __name__ == "__main__":
    run_eda()
