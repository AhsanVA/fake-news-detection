import os
import pickle
import json
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import TextPreprocessor

def main():
    try:
        processor = TextPreprocessor()
        
        # Load resources
        lstm_path = os.path.join("models", "lstm_model.h5")
        tok_path = os.path.join("models", "tokenizer.pkl")
        
        if not os.path.exists(lstm_path) or not os.path.exists(tok_path):
            print("Models not found.")
            return

        model = load_model(lstm_path)
        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Test Cases
        test_cases = [
            {"text": "Apple releases new iPhone with holographic display.", "expected": "Fake"},
            {"text": "The World Health Organization explicitly stated that COVID-19 vaccines are safe.", "expected": "Real"},
            {"text": "Aliens found in Mars by NASA rover.", "expected": "Fake"},
            {"text": "ISRO successfully launches Chandrayaan-3 into orbit.", "expected": "Real"},
            {"text": "Government announces free gold for everyone.", "expected": "Fake"}
        ]
        
        results = []
        for case in test_cases:
            text = case['text']
            processed_text = processor.preprocess(text)
            # Tokenize
            seq = tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
            
            # Predict
            prob = float(model.predict(padded, verbose=0)[0][0])
            label = "Real" if prob > 0.5 else "Fake"
            
            results.append({
                "input": text,
                "probability": prob,
                "prediction": label,
                "expected": case['expected']
            })
            
        with open("debug_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print("Done.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
