import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. LOAD DATA
# =========================

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Add labels
fake["label"] = 1
real["label"] = 0

# Combine datasets
data = pd.concat([fake, real], ignore_index=True)

# Remove missing values
data = data.dropna(subset=["text"])

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# 2. FEATURES & LABEL
# =========================

X = data["text"]
y = data["label"]


# =========================
# 3. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. TF-IDF + LOGISTIC REGRESSION
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# =========================
# 5. TRAIN
# =========================

print("Training Fake News model...")

model.fit(X_train, y_train)


# =========================
# 6. EVALUATE
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("FAKE NEWS MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Real", "Fake"]
))


# =========================
# 7. SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/fake_news_model.pkl"
)

print("\nModel saved successfully!")
print("models/fake_news_model.pkl")