import streamlit as st
import joblib
import nltk
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

# Load pre-trained model and label encoders
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('mental_health_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return pipeline, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'mental_health_model.pkl' and 'label_encoders.pkl' are in the same directory.")
        return None, None

# Preprocessing function (same as in the original script)
def preprocess_text(text):
    """
    Preprocess text data for NLP
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Clean text
    text = re.sub(r'http\S+', '', str(text))  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Response generation function (similar to original script)
def generate_llm_response(emotion, severity, urgency, query):
    """
    Generate a customized response based on the predicted parameters
    """
    # Base responses for each emotion
    emotion_responses = {
        'Happy': [
            "It's wonderful to hear you're feeling positive! This kind of emotional state is great for building resilience.",
            "I'm glad you're in good spirits! These moments of happiness can be really valuable.",
            "It's great that you're feeling happy. These positive emotions help strengthen our mental wellbeing."
        ],
        'Sad': [
            "I'm sorry to hear you're feeling down. It's completely normal to experience sadness at times.",
            "I can sense you're feeling sad. Remember that it's okay to not be okay sometimes.",
            "I understand you're feeling sad right now. These emotions, while difficult, are a natural part of life."
        ],
        # ... (rest of the emotion_responses dictionary from the original script)
    }

    # Severity modifiers
    severity_modifiers = {
        'Low': [
            "This seems to be a mild experience for you.",
            "From what you've shared, this isn't severely impacting your daily life.",
            "It sounds like you're managing this well overall."
        ],
        'Medium': [
            "This seems to be having a noticeable impact on you.",
            "I can tell this is affecting your wellbeing to some degree.",
            "This sounds like it's creating some significant challenges for you."
        ],
        'High': [
            "This appears to be having a substantial impact on your wellbeing.",
            "I can see this is significantly affecting your daily functioning.",
            "This sounds like it's creating major challenges for you right now."
        ]
    }

    # Urgency response components
    urgency_responses = {
        'Low': [
            "There's no rush to address this immediately, but it's good to be aware of.",
            "This isn't an emergency situation, but it's still worth your attention.",
            "While not urgent, this is still important to acknowledge."
        ],
        'Medium': [
            "It would be beneficial to address this relatively soon.",
            "This deserves your attention in the near future.",
            "Consider making this a priority in your self-care."
        ],
        'High': [
            "I recommend addressing this as soon as possible.",
            "This situation deserves immediate attention.",
            "It's important that you take care of this right away."
        ]
    }

    # Select random components for variety
    emotion_response = np.random.choice(emotion_responses[emotion])
    severity_modifier = np.random.choice(severity_modifiers[severity])
    urgency_response = np.random.choice(urgency_responses[urgency])

    # Construct the complete response
    response = f"{emotion_response} {severity_modifier} {urgency_response}"

    return response

# Action suggestion function
def get_action_suggestion(emotion, severity, urgency):
    """
    Generate action suggestions based on predicted parameters
    """
    # Comprehensive mapping of parameter combinations to suggestions
    action_mapping = {
        # Comprehensive action suggestions (similar to original script)
        ('Happy', 'Low', 'Low'): [
            "Continue engaging in activities that bring you joy.",
            "Consider sharing your positive experiences with others.",
            "Take note of what's contributing to your happiness for future reference."
        ],
        # ... (rest of the action_mapping dictionary from the original script)
    }

    # Default suggestions based on severity
    default_suggestions = {
        'Low': [
            "Continue monitoring your feelings and practice regular self-care.",
            "Consider keeping a mood journal to track patterns in your emotions.",
            "Maintain healthy habits like regular sleep, nutrition, and exercise."
        ],
        'Medium': [
            "Reach out to your support network to discuss what you're experiencing.",
            "Consider implementing structured self-care activities into your routine.",
            "Look into resources like books or podcasts about emotional wellbeing."
        ],
        'High': [
            "It would be beneficial to consult with a mental health professional.",
            "Consider whether your symptoms warrant a conversation with your doctor.",
            "Look into local mental health resources or support groups."
        ]
    }

    # Get appropriate suggestions
    key = (emotion, severity, urgency)
    if key in action_mapping:
        suggestions = action_mapping[key]
    else:
        suggestions = default_suggestions[severity]

    # Return a random suggestion for variety
    return np.random.choice(suggestions)

# Main Streamlit app
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Mental Health Assistant", page_icon="🧠")

    # Title and description
    st.title("Personalized Mental Health Assistant 🧠")
    st.write("Share your thoughts, and I'll provide personalized insights and support.")

    # Load the model
    pipeline, label_encoders = load_model()

    if pipeline is None or label_encoders is None:
        st.error("Could not load the mental health model. Please check the model files.")
        return

    # User input
    user_query = st.text_area("Enter your thoughts or feelings:", 
                               help="The more specific you are, the more accurate the analysis.")

    # Process button
    if st.button("Analyze My Feelings"):
        if not user_query:
            st.warning("Please enter a query before analyzing.")
            return

        # Preprocess the query
        processed_query = preprocess_text(user_query)

        # Make predictions
        predictions = pipeline.predict([processed_query])[0]

        # Decode predictions
        emotion = label_encoders['Emotion'].inverse_transform([predictions[0]])[0]
        severity = label_encoders['Severity'].inverse_transform([predictions[1]])[0]
        urgency = label_encoders['Urgency'].inverse_transform([predictions[2]])[0]

        # Generate response and action suggestions
        response = generate_llm_response(emotion, severity, urgency, user_query)
        action = get_action_suggestion(emotion, severity, urgency)

        # Display results
        st.subheader("Analysis Results")
        
        # Analysis details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emotion", emotion)
        with col2:
            st.metric("Severity", severity)
        with col3:
            st.metric("Urgency", urgency)

        # Response section
        st.subheader("Supportive Response")
        st.write(response)

        # Action suggestions
        st.subheader("Suggested Actions")
        st.write(action)

        # Disclaimer
        st.info("Note: This is an AI-powered assistant and does not replace professional mental health care. " 
                "If you're experiencing a mental health crisis, please contact a qualified professional.")

# Run the app
if __name__ == "__main__":
    main()
