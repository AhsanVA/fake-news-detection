
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional
import logging

logging.basicConfig(level=logging.INFO)

class ModelFactory:
    """
    Factory class to create ML and DL models.
    """
    @staticmethod
    def get_model(model_name, input_dim=None, vocab_size=None, embedding_dim=100, max_length=100):
        if model_name == 'logistic_regression':
            return LogisticRegression(max_iter=1000)
        elif model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        elif model_name == 'svm':
            return SVC(probability=True)
        elif model_name == 'lstm':
            if vocab_size is None:
                raise ValueError("vocab_size required for LSTM")
            return ModelFactory._build_lstm(vocab_size, embedding_dim, max_length)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    @staticmethod
    def _build_lstm(vocab_size, embedding_dim, max_length):
        """
        Builds a Bi-Directional LSTM model.
        """
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        logging.info("Built LSTM model.")
        return model
