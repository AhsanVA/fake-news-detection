import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import logging
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)

def evaluate(model_name='lstm'):
    logging.info(f"Evaluating {model_name}...")
    
    loader = DataLoader()
    # Use social media dataset as that's what we trained on mostly
    _, _, test_df = loader.load_social_media_dataset()
    
    if test_df is None:
        logging.error("Could not load test data.")
        return

    # Process Labels
    def map_label(label):
        if isinstance(label, (int, np.integer)): return int(label)
        if str(label).lower() in ['fake', '0', 'false']: return 0
        return 1
        
    test_df['binary_label'] = test_df['label'].apply(map_label)
    processor = TextPreprocessor()
    test_texts = test_df['statement'].apply(processor.preprocess)
    y_test = test_df['binary_label']
    
    # Load Model
    if model_name == 'lstm':
        model_path = os.path.join("models", "lstm_model.h5")
        tok_path = os.path.join("models", "tokenizer.pkl")
        
        if not os.path.exists(model_path):
            logging.error("LSTM model not found.")
            return
            
        model = load_keras_model(model_path)
        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Vectorize
        seq = tokenizer.texts_to_sequences(test_texts)
        X_test = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        
        # Predict
        probs = model.predict(X_test)
        preds = (probs > 0.5).astype(int)
        
    else:
        # Sklearn models
        model_path = os.path.join("models", f"{model_name}.pkl")
        vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
        
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(vec_path, 'rb') as f: vectorizer = pickle.load(f)
        
        X_test = vectorizer.transform(test_texts)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    # Metrics
    acc = accuracy_score(y_test, preds)
    logging.info(f"Test Accuracy: {acc:.4f}")
    
    report = classification_report(y_test, preds, target_names=['Fake', 'Real'])
    print("Classification Report:")
    print(report)
    
    # Save Classification Report
    with open(f"models/{model_name}_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")
    
    try:
        roc = roc_auc_score(y_test, probs)
        logging.info(f"ROC AUC: {roc:.4f}")
    except:
        pass
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"models/{model_name}_confusion_matrix.png")
    logging.info(f"Confusion matrix saved to models/{model_name}_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm', help='Model to evaluate')
    args = parser.parse_args()
    evaluate(args.model)
