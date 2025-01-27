import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
from transformers import pipeline

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load dataset
def load_data():
    return pd.read_excel("mental_health_assistant_dataset.xlsx", engine='openpyxl')

data = load_data()
data['Processed_Text'] = data['User Query'].apply(preprocess_text)

# Feature extraction and model training
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Processed_Text']).toarray()
label_encoder = LabelEncoder()
data['Emotional_State_Encoded'] = label_encoder.fit_transform(data['Emotion'])

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, data['Emotional_State_Encoded'])

# Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Predict emotion
def predict_emotion(text_input):
    processed_input = preprocess_text(text_input)
    vectorized_input = vectorizer.transform([processed_input]).toarray()
    prediction = model.predict(vectorized_input)
    predicted_emotion = label_encoder.inverse_transform(prediction)[0]
    return predicted_emotion

# Generate response suggestions
def generate_response(emotion):
    responses = {
        "Happy": "That's great to hear! Keep up the positivity!",
        "Sad": "I'm sorry you're feeling this way. Would you like to talk more about it?",
        "Anxious": "It sounds like you're feeling anxious. Have you tried deep breathing exercises?",
        "Overwhelmed": "Take a deep breath. It’s okay to feel overwhelmed. Breaking tasks into smaller steps might help."
    }
    return responses.get(emotion, "I'm here to help. Tell me more about how you're feeling.")

# Streamlit app UI
st.title("Personalized Mental Health Assistant")

user_input = st.text_input("How are you feeling today?", "I feel like I’m drowning in work and can’t take it anymore.")
if user_input:
    emotion = predict_emotion(user_input)
    sentiment = sentiment_analyzer(user_input)
    response = generate_response(emotion)

    st.subheader("Results")
    st.write(f"**Predicted Emotion:** {emotion}")
    st.write(f"**Sentiment Analysis:** {sentiment[0]['label']} ({sentiment[0]['score']:.2f})")
    st.write(f"**Suggested Response:** {response}")
