
import logging
import sys
import os
import pickle
import pandas as pd
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)

def diagnose():
    print("\n=== DIAGNOSING COMBINED DATASET BALANCE ===")
    loader = DataLoader()
    train_df, val_df, test_df = loader.load_social_media_dataset()
    
    if train_df is None:
        print("Failed to load dataset.")
        return

    # Check distribution
    print("\n--- Final Training Set Distribution ---")
    counts = train_df['label'].value_counts()
    print(counts)
    
    total = len(train_df)
    fake_count = counts.get(0, 0)
    real_count = counts.get(1, 0)
    print(f"Total: {total}")
    print(f"Fake (0): {fake_count} ({fake_count/total:.2%})")
    print(f"Real (1): {real_count} ({real_count/total:.2%})")
    
    print("\n=== DIAGNOSING PREDICTION PROBABILITIES ===")
    model_path = os.path.join("models", "random_forest.pkl")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    processor = TextPreprocessor()
    
    # User examples that were flagged as fake
    inputs = [
        "Apple unveiled its latest iPhone model featuring improved battery efficiency and on-device AI processing during its annual product launch event.", 
        "Scientists at ISRO successfully tested a reusable launch vehicle landing experiment, marking a significant step toward reducing future space mission costs.",
        "The World Health Organization reported a decline in global COVID-19 related deaths in 2024 due to increased vaccination coverage and improved treatments."
    ]
    
    print("\n--- Model Predictions on User Queries ---")
    for text in inputs:
        print(f"\nInput: {text[:80]}...")
        proc = processor.preprocess(text)
        vec = vectorizer.transform([proc])
        
        # Check for Empty Vector (Vocabulary Miss)
        print(f"Non-Zero Features: {vec.nnz} / {vec.shape[1]}")
        if vec.nnz == 0:
            print("WARNING: Vector is empty! No words found in vocabulary.")
        
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        
        label = "REAL" if pred == 1 else "FAKE"
        print(f"Prediction: {label} (Label {pred})")
        print(f"Confidence: FAKE={prob[0]:.4f}, REAL={prob[1]:.4f}")

if __name__ == "__main__":
    diagnose()
