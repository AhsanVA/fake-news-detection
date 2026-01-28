
import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.features import FeatureEngineer
from src.models import ModelFactory

logging.basicConfig(level=logging.INFO)

def train(model_name='logistic_regression'):
    logging.info(f"Starting training pipeline for {model_name}...")
    
    # 1. Load Data
    loader = DataLoader()
    # Switch to Social Media Dataset for better real-world performance
    train_df, val_df, test_df = loader.load_social_media_dataset()
    
    if train_df is None:
        logging.error("Could not load data. Exiting.")
        return

    # Check labels
    # If using gonzaloa/fake_news: labels are likely 0 (Fake) and 1 (Real) already.
    # We should verify mapping.
    
    def map_label(label):
        # Handle existing integer 0/1
        if isinstance(label, (int, np.integer)):
            return int(label)
        # Fallback for strings if any issues
        if str(label).lower() in ['fake', '0', 'false']: return 0
        return 1

    train_df['binary_label'] = train_df['label'].apply(map_label)
    val_df['binary_label'] = val_df['label'].apply(map_label)
    test_df['binary_label'] = test_df['label'].apply(map_label)
    
    # DEBUG: Check Class Balance
    logging.info(f"Train Label Distribution:\n{train_df['binary_label'].value_counts()}")
    logging.info(f"Sample Train Data (Text + Label):\n{train_df[['statement', 'binary_label']].head()}")
    
    # Preprocessing
    processor = TextPreprocessor()
    logging.info("Preprocessing text...")
    # For speed in demo, maybe sample? No, let's try full.
    train_texts = train_df['statement'].apply(processor.preprocess)
    val_texts = val_df['statement'].apply(processor.preprocess)
    test_texts = test_df['statement'].apply(processor.preprocess)
    
    # Feature Engineering
    logging.info("Extracting features...")
    # Select method based on model type
    fe_method = 'lstm' if model_name == 'lstm' else 'tfidf'
    fe = FeatureEngineer(method=fe_method, max_features=50000)
    
    X_train, _ = fe.fit_transform(train_texts)
    X_val = fe.transform(val_texts)
    X_test = fe.transform(test_texts)
    
    y_train = train_df['binary_label']
    y_val = val_df['binary_label']
    y_test = test_df['binary_label']
    
    # Model Training
    logging.info(f"Training {model_name}...")
    
    if model_name in ['logistic_regression', 'random_forest', 'svm']:
        model = ModelFactory.get_model(model_name)
        model.fit(X_train, y_train)
        
        # Validation
        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        logging.info(f"Validation Accuracy: {acc:.4f}")
        
    elif model_name == 'lstm':
        logging.info("Training LSTM Neural Network...")
        # X_train is now a dense numpy array of padded sequences
        
        # Verify shape
        vocab_size = fe.max_features
        max_length = X_train.shape[1]
        logging.info(f"LSTM Config: Vocab={vocab_size}, MaxLen={max_length}")
        
        model = ModelFactory.get_model('lstm', vocab_size=vocab_size, max_length=max_length)
        
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        logging.info(f"Computed Class Weights: {class_weight_dict}")

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=8,  # Increased to 8 to allow convergence
            batch_size=64,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Validation Eval
        val_preds_prob = model.predict(X_val)
        val_preds = (val_preds_prob > 0.5).astype(int)
        acc = accuracy_score(y_val, val_preds)
        logging.info(f"Validation Accuracy: {acc:.4f}")
        
        # Save Keras Model
        model_path = os.path.join("models", "lstm_model.h5")
        model.save(model_path)
        logging.info(f"LSTM Model saved to {model_path}")
        
        # Save Tokenizer
        with open(os.path.join("models", "tokenizer.pkl"), 'wb') as f:
            pickle.dump(fe.vectorizer, f)
            
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='logistic_regression', help='Model to train')
    args = parser.parse_args()
    train(args.model)
