import os
import pickle
import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)

def debug_lstm():
    # Load resources
    lstm_path = os.path.join("models", "lstm_model.h5")
    tok_path = os.path.join("models", "tokenizer.pkl")
    
    if not os.path.exists(lstm_path) or not os.path.exists(tok_path):
        print("Models not found.")
        return

    print("Loading model and tokenizer...")
    model = load_model(lstm_path)
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
        
    print(f"Tokenizer Vocab Size: {len(tokenizer.word_index)}")
    
    # Test Cases
    test_cases = [
        "Apple releases new iPhone with holographic display.", # likely fake/tech
        "The World Health Organization explicitly stated that COVID-19 vaccines are safe.", # Real
        "Aliens found in Mars by NASA rover.", # Fake
        "ISRO successfully launches Chandrayaan-3 into orbit.", # Real
        "Government announces free gold for everyone.", # Fake
    ]
    
    for text in test_cases:
        # Tokenize
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        
        # Predict
        prob = model.predict(padded)[0][0]
        label = "REAL" if prob > 0.5 else "FAKE"
        
        print(f"\nInput: {text}")
        print(f"Tokens: {seq[0][:10]}...") # Show first few tokens
        print(f"Prob: {prob:.4f} => {label}")

if __name__ == "__main__":
    debug_lstm()
