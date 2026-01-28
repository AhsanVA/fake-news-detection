# Fake News Detection System

## Overview
This project is an industry-grade AI system designed to detect fake news using Natural Language Processing (NLP) and Machine Learning. Ideally suited for social media data, it leverages datasets like LIAR and ISOT to train robust classifiers.

## Architecture
The project follows a modular architecture:
- **`src.data_loader`**: Handles data ingestion from Hugging Face or local CSVs.
- **`src.preprocessing`**: Cleans text (lower-casing, URL removal, stop-word removal, lemmatization).
- **`src.features`**: vectorizes text using TF-IDF (extensible to Word2Vec/GloVe).
- **`src.models`**: Contains model definitions (Logistic Regression, Random Forest, SVM, **Bi-Directional LSTM**).
- **`src.train`**: Orchestrates the training pipeline (including class balancing and oversampling).
- **`app.app`**: Streamlit-based web interface with **Model Selector** (Random Forest vs LSTM).

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download datasets if not using the automated loader.

## Usage

### 1. Training
To train the baseline model (Logistic Regression):
```bash
python src/train.py --model logistic_regression
```

To train other models:
```bash
python src/train.py --model random_forest
python src/train.py --model lstm
```

### 2. Frontend Application
To launch the web interface (with Dark Mode & Model Selector):
```bash
streamlit run app/app.py
```

## Folder Structure
- `data/`: Raw and processed data.
- `models/`: Saved model artifacts.
- `notebooks/`: EDA and experimentation.
- `src/`: Core source code.

## Disclaimer
This tool uses statistical patterns to predict the likelihood of a news item being real or fake. It does not perform fact-checking against a knowledge base.
