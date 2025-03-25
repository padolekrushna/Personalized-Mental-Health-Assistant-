import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

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

# Load the dataset
df = pd.read_excel("mental_health_assistant_dataset.xlsx")

# Preprocess the text
df['Processed_Text'] = df['User_Query'].apply(preprocess_text)

# Encode target labels
label_encoders = {}
for column in ['Emotion', 'Severity', 'Urgency']:
    le = LabelEncoder()
    df[f'{column}_Encoded'] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and multiple targets
X = df['Processed_Text']
y = df[['Emotion_Encoded', 'Severity_Encoded', 'Urgency_Encoded']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and multi-output classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# Train the model
print("\nTraining the multi-output model...")
pipeline.fit(X_train, y_train)

# Save the model and encoders
print("\nSaving the model and resources...")
joblib.dump(pipeline, 'mental_health_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\nModel and resources saved successfully!")
