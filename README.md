# Personalized Mental Health Assistant

#Project link : https://xdp85jx27efpurucd23ni8.streamlit.app/

## Overview
This is a Streamlit-based Mental Health Assistant that uses machine learning to analyze user input and provide personalized emotional support and action suggestions.

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps
1. Clone the repository
```bash
git clone https://your-repository-url.git
cd mental-health-assistant
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK resources
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

5. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

## Files in the Project
- `streamlit_app.py`: Main Streamlit application
- `requirements.txt`: Python package dependencies
- `mental_health_model.pkl`: Pre-trained machine learning model
- `label_encoders.pkl`: Label encoders for model predictions

## How to Use
1. Enter your thoughts or feelings in the text area
2. Click "Analyze My Feelings"
3. Receive personalized insights about your emotional state
4. Get supportive responses and suggested actions

## Disclaimer
This is an AI-powered assistant and does not replace professional mental health care. If you're experiencing a mental health crisis, please contact a qualified professional.

## Model Training
The underlying model was trained on a synthetic dataset to classify emotions, severity, and urgency of mental health queries.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Specify your license here]
