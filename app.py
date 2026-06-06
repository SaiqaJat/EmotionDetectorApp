# ============================================
# EMOTION DETECTION - IMPROVED STREAMLIT APP
# ============================================

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD ARTIFACTS
# ============================================
@st.cache_resource
def load_models():
    """Load model, vectorizer, and labels with caching"""
    try:
        model = joblib.load("emotion_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        emotion_labels = joblib.load("emotion_labels.pkl")
        return model, vectorizer, emotion_labels
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure these files are in the same directory:")
        st.code("- emotion_model.pkl\n- tfidf_vectorizer.pkl\n- emotion_labels.pkl")
        st.stop()

model, vectorizer, emotion_labels = load_models()

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def preprocess_text(text):
    """
    Match the preprocessing pipeline from training
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 3. Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    
    # 4. Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================
# UI LAYOUT
# ============================================
st.title("🎭 Emotion Detection System")
st.write("Detect the emotion expressed in any text using Machine Learning")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses a **Logistic Regression** model trained on text data 
    to classify emotions into 6 categories:
    - 😢 Sadness
    - 😠 Anger
    - 💕 Love
    - 😮 Surprise
    - 😨 Fear
    - 😊 Joy
    """)
    
    st.header("📊 Model Info")
    st.metric("Accuracy", "89.5%")
    st.metric("Training Samples", "16K+")
    st.metric("Emotion Classes", "6")

# ============================================
# MAIN INPUT
# ============================================
st.subheader("Enter Your Text")

user_input = st.text_area(
    "Type a sentence or paragraph:",
    placeholder="Example: I am very happy and excited about this opportunity!",
    height=100
)

# ============================================
# PREDICTION
# ============================================
if st.button("🔍 Predict Emotion", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Show input being processed
        with st.spinner("Processing... 🤔"):
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # Vectorize
            text_vector = vectorizer.transform([processed_text])
            
            # Predict
            prediction = model.predict(text_vector)[0]
            
            # Get prediction probabilities
            probabilities = model.predict_proba(text_vector)[0]
            
            # Get emotion name
            emotion_name = emotion_labels[prediction]
            confidence = probabilities[prediction] * 100
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Main prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.markdown("")
        with col2:
            emotion_emoji = {
                'sadness': '😢',
                'anger': '😠',
                'love': '💕',
                'surprise': '😮',
                'fear': '😨',
                'joy': '😊'
            }
            emoji = emotion_emoji.get(emotion_name, '😶')
            st.markdown(f"""
            <h1 style='text-align: center; color: #FF6B6B;'>
            {emoji} {emotion_name.upper()}
            </h1>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("")
        
        # Confidence
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show all probabilities
        st.subheader("Probability Distribution")
        
        prob_df = pd.DataFrame({
            'Emotion': [emotion_labels[i] for i in range(len(probabilities))],
            'Probability': probabilities * 100
        }).sort_values('Probability', ascending=False)
        
        # Bar chart
        st.bar_chart(
            prob_df.set_index('Emotion')['Probability'],
            use_container_width=True
        )
        
        # Detailed probabilities
        st.subheader("Detailed Breakdown")
        for idx, row in prob_df.iterrows():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{row['Emotion'].capitalize()}**")
            with col2:
                st.write(f"{row['Probability']:.2f}%")
            st.progress(row['Probability'] / 100)
        
        # Show processed text
        with st.expander("📝 View Preprocessed Text"):
            st.write("Original text:")
            st.code(user_input)
            st.write("After preprocessing:")
            st.code(processed_text)

# ============================================
# EXAMPLE SENTENCES
# ============================================
st.divider()
st.subheader("Try These Examples")

examples = {
    "😊 Joy": "I just got the best news ever! I'm so excited and happy!",
    "😢 Sadness": "I feel so lonely and depressed. Nothing seems to matter anymore.",
    "😠 Anger": "I can't believe what you did! This is absolutely unacceptable!",
    "💕 Love": "I love spending time with you. You mean everything to me.",
    "😮 Surprise": "Oh my goodness! I never expected this at all!",
    "😨 Fear": "I'm terrified and anxious about what might happen."
}

cols = st.columns(3)
for idx, (label, text) in enumerate(examples.items()):
    col = cols[idx % 3]
    if col.button(f"Try: {label}"):
        st.session_state.example_text = text
        st.rerun()

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<p style='text-align: center; color: gray; font-size: 12px;'>
Emotion Detection System | Built with Streamlit & Scikit-learn
</p>
""", unsafe_allow_html=True)