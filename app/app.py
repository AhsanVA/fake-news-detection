
import streamlit as st
import pickle
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import TextPreprocessor

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="centered"
)

# Title and Style
st.title("📱 Social Media Rumor Detector")
st.markdown("""
<style>
    .main {
        background: #f0f2f5; 
        font-family: 'Helvetica', sans-serif;
    }
    .post-container {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #ccc;
        display: inline-block;
        margin-right: 10px;
    }
    .user-name {
        font-weight: bold;
        color: #333;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

st.write("### Verify content from your feed")

# Load Resources
@st.cache_resource
def load_resources():
    models = {}
    
    # Load RF
    rf_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest.pkl')
    vec_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
    
    if os.path.exists(rf_path) and os.path.exists(vec_path):
        with open(rf_path, 'rb') as f: models['rf_model'] = pickle.load(f)
        with open(vec_path, 'rb') as f: models['rf_vec'] = pickle.load(f)
            
    # Load LSTM
    lstm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.h5')
    tok_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tokenizer.pkl')
    
    if os.path.exists(lstm_path) and os.path.exists(tok_path):
        from tensorflow.keras.models import load_model
        try:
            models['lstm_model'] = load_model(lstm_path)
            with open(tok_path, 'rb') as f: 
                if os.path.getsize(tok_path) > 0:
                    models['lstm_tok'] = pickle.load(f)
                else:
                    print("Tokenizer file is empty.")
        except Exception as e:
            print(f"Error loading LSTM model/tokenizer: {e}")
            # If load fails, just don't add them to models
            pass
        
    return models

resources = load_resources()

if not resources:
    st.error("⚠️ No models found! Please run 'python main.py train' first.")
    st.stop()

col1, col2 = st.columns([1, 4])
with col1:
    source = st.selectbox("Source", ["Twitter/X", "Facebook", "WhatsApp", "Reddit", "News Site"])
    
    # Model Selector
    available_models = []
    if 'rf_model' in resources: available_models.append("Random Forest (Fast)")
    if 'lstm_model' in resources: available_models.append("LSTM Neural Net (Accurate)")
    
    model_choice = st.selectbox("AI Model", available_models, index=len(available_models)-1)

with col2:
    st.info("Paste the text, headline, or message you want to verify.")

# Input simulated as a post
news_text = st.text_area("Post Content", height=150, placeholder="e.g. 'Breaking: Scientists discover that drinking water causes memory loss...'")

if st.button("🔍 Verify Credibility"):
    if not news_text:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Scanning {source} database and analyzing linguistic patterns..."):
            # Contextualize
            full_input = f"{news_text}"  
            
            # Preprocess
            processor = TextPreprocessor()
            processed_text = processor.preprocess(full_input)
            
            prediction = 0
            proba = [0.5, 0.5]
            
            if "Random Forest" in model_choice:
                features = resources['rf_vec'].transform([processed_text])
                prediction = resources['rf_model'].predict(features)[0]
                proba = resources['rf_model'].predict_proba(features)[0]
                
            elif "LSTM" in model_choice:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                tokenizer = resources['lstm_tok']
                seq = tokenizer.texts_to_sequences([processed_text])
                features = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                
                # Predict (returns float 0..1)
                pred_float = resources['lstm_model'].predict(features)[0][0]
                prediction = 1 if pred_float > 0.5 else 0
                proba = [1 - pred_float, pred_float] # [Fake, Real]
            
            # Display Result
            st.markdown("<div class='post-container'>", unsafe_allow_html=True)
            col_a, col_b = st.columns([1, 4])
            with col_a:
                if prediction == 1:
                    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Twitter_Verified_Badge.svg/640px-Twitter_Verified_Badge.svg.png", width=50)
                else:
                    st.markdown("⚠️", unsafe_allow_html=True)
            with col_b:
                if prediction == 1:
                    st.success("**LIKELY REAL**")
                    st.markdown(f"Our AI is **{proba[1]*100:.1f}%** confident this content is authentic.")
                else:
                    st.error("**LIKELY FAKE / MISLEADING**")
                    st.markdown(f"Our AI is **{proba[0]*100:.1f}%** confident this content is fabricated.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualization
            st.caption(f"Analysis performed by: {model_choice}")
