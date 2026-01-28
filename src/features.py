
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(level=logging.INFO)

class FeatureEngineer:
    """
    Class to handle feature extraction from text data.
    """
    def __init__(self, method='tfidf', max_features=50000):
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3))
        elif method == 'lstm':
            # Limit vocab size to max_features
            self.vectorizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
        else:
            # Placeholder for Word2Vec/GloVe or BERT embeddings
            logging.warning(f"Method {method} not fully implemented yet. Defaulting to TF-IDF logic where applicable.")

    def fit_transform(self, texts):
        if self.method == 'tfidf':
            return self.vectorizer.fit_transform(texts), self.vectorizer
        elif self.method == 'lstm':
            self.vectorizer.fit_on_texts(texts)
            sequences = self.vectorizer.texts_to_sequences(texts)
            # Default max length 200 for now
            return pad_sequences(sequences, maxlen=200, padding='post', truncating='post'), self.vectorizer
        return None, None

    def transform(self, texts):
        """
        Transforms texts using the fitted vectorizer.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
            
        if self.method == 'tfidf':
            return self.vectorizer.transform(texts)
        elif self.method == 'lstm':
            sequences = self.vectorizer.texts_to_sequences(texts)
            return pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        return None

    def save_vectorizer(self, path):
        """
        Saves the vectorizer to disk.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logging.info(f"Vectorizer saved to {path}")

    def load_vectorizer(self, path):
        """
        Loads the vectorizer from disk.
        """
        with open(path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        logging.info(f"Vectorizer loaded from {path}")
