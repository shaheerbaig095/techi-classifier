import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

print("Loading data...")
df = pd.read_csv("data/headlines.csv")

X = df['headline']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model (TF-IDF + Logistic Regression)...")
# Pipeline creates TF-IDF vectors and feeds them to Logistic Regression
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

model_pipeline.fit(X_train, y_train)
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model trained successfully! Accuracy on test set: {accuracy * 100:.2f}%")

os.makedirs('model', exist_ok=True)
joblib.dump(model_pipeline, 'model/tech_classifier.joblib')
print("✅ Model saved to model/tech_classifier.joblib")