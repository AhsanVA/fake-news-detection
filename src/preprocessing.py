
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    # Pre-downloaded or permission issues - assume present or manual install needed
    pass 
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)

class TextPreprocessor:
    """
    Class to handle text cleaning, tokenization, and normalization.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Basic text cleaning: remove URLs, special chars, lowercase.
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#' from hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers (optional, depending on context, keeping for now might be noise)
        text = re.sub(r'\d+', '', text)
        
        return text.strip()

    def preprocess(self, text):
        """
        Full preprocessing pipeline: Clean -> Tokenize -> Remove Stopwords -> Lemmatize
        Fallback to simple split if NLTK fails.
        """
        cleaned = self.clean_text(text)
        
        try:
            tokens = nltk.word_tokenize(cleaned)
            # Remove stopwords and lemmatize
            processed_tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in self.stop_words
            ]
            return " ".join(processed_tokens)
        except Exception:
            # Fallback for when NLTK data is missing or broken
            # logging.warning("NLTK tokenization failed. Using simple split.")
            return cleaned

if __name__ == "__main__":
    processor = TextPreprocessor()
    sample = "Check this out! https://example.com @user #FakeNews 123"
    print(f"Original: {sample}")
    print(f"Processed: {processor.preprocess(sample)}")
