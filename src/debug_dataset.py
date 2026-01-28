
import logging
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)

def debug_labels():
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_social_media_dataset()
    
    if train_df is None:
        print("Failed to load dataset.")
        return

    print("\n--- Raw Label Counts ---")
    print(train_df['label'].value_counts())
    
    print("\n--- Sample Raw Data ---")
    print(train_df[['label', 'statement']].head(10))

    def map_label(label):
        # Handle existing integer 0/1
        if isinstance(label, (int, np.integer)):
            return int(label)
        # Fallback for strings if any issues
        if str(label).lower() in ['fake', '0', 'false']: return 0
        return 1

    train_df['binary_label'] = train_df['label'].apply(map_label)
    
    print("\n--- Mapped Label Counts ---")
    print(train_df['binary_label'].value_counts())
    
    print("\n--- Sample Mapped Data ---")
    print(train_df[['label', 'binary_label']].head(10))

if __name__ == "__main__":
    debug_labels()
