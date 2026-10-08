import streamlit as st
import joblib
import os

# Train on the fly if model doesn't exist
if not os.path.exists('model/tech_classifier.joblib'):
    import train

# Clean, minimal page config without emojis
st.set_page_config(page_title="TECHi", layout="centered")

# Minimal Header
st.title("TECHi")
st.markdown("A machine learning text classifier for technology headlines.")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('model/tech_classifier.joblib')

model = load_model()

# User Input - Hiding the label for a cleaner, modern look
user_input = st.text_input(
    "Headline input", 
    placeholder="Paste a tech headline here (e.g., Apple unveils new MacBook Pro with M4 chip)", 
    label_visibility="collapsed"
)

# Output Section
if user_input:
    prediction = model.predict([user_input])[0]
    probabilities = model.predict_proba([user_input])[0]
    classes = model.classes_
    
    st.markdown(f"### Predicted Category: **{prediction}**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Confidence Scores")
    
    # Clean confidence bars
    for cls, prob in zip(classes, probabilities):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"<span style='color: gray;'>{cls}</span>", unsafe_allow_html=True)
        with col2:
            st.progress(float(prob))