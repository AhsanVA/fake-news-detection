
import logging
import sys
import os
import pickle
import pandas as pd
from datasets import load_dataset
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import TextPreprocessor

def inspect():
    # 1. Inspect Dataset Labels
    print("\n--- DATASET LABEL INSPECTION ---")
    try:
        ds = load_dataset('gonzaloa/fake_news', split='train')
        df = pd.DataFrame(ds)
        
        print("Scaning 5 samples of LABEL 0:")
        for text in df[df['label'] == 0]['title'].head(5):
            print(f" [0] {text[:100]}...")
            
        print("\nScaning 5 samples of LABEL 1:")
        for text in df[df['label'] == 1]['title'].head(5):
            print(f" [1] {text[:100]}...")
    except Exception as e:
        print(f"Error loading dataset: {e}")

    # 2. Inspect Vocabulary
    print("\n--- VOCABULARY INSPECTION ---")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        vocab = vectorizer.vocabulary_
        keywords = ["apple", "iphone", "isro", "space", "who", "covid", "vaccine", "science"]
        print(f"Vocabulary Size: {len(vocab)}")
        print("Keyword Check:")
        for kw in keywords:
            print(f"  '{kw}': {'FOUND' if kw in vocab else 'MISSING'}")
    else:
        print("Vectorizer not found.")

    # 3. Predict on User Examples
    print("\n--- PREDICTION DEBUG ---")
    model_path = os.path.join("models", "random_forest.pkl")
    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        processor = TextPreprocessor()
        examples = [
            "Apple unveiled its latest iPhone model featuring improved battery efficiency.",
            "Scientists at ISRO successfully tested a reusable launch vehicle.",
            "The World Health Organization reported a decline in global COVID-19 related deaths."
        ]
        
        for text in examples:
            print(f"\nInput: {text}")
            proc = processor.preprocess(text)
            print(f"Processed: {proc}")
            vec = vectorizer.transform([proc])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            print(f"Prediction: {pred} (0=Fake?, 1=Real?)")
            print(f"Probabilities: {prob}")
            
if __name__ == "__main__":
    inspect()
