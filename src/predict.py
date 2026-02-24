import sys
import os
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text

model=joblib.load('models/phishing_model.pkl')
vectorizer=joblib.load('models/vectorizer.pkl')
def predict_email(text):
    cleaned_text = clean_text(text)
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]   
    probability = model.predict_proba(vector)[0].max()

    if prediction == 1:
        return "Email is Not Safe", round(probability * 100, 2)
    else:
        return "Email is Safe", round(probability * 100, 2)