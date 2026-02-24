import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/phishing_dataset.csv")

# Rename columns
df.columns = ['text', 'label']

# Clean text
df['text'] = df['text'].astype(str).apply(clean_text)

# Vectorization (FIT happens here)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])   # 🔴 THIS LINE FITS THE VECTORIZER
y = df['label']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save both
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model and vectorizer trained & saved")
