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

data = pd.read_csv("data/clickbait_data.csv")

print("Dataset loaded:", data.shape)
print("Columns:", data.columns.tolist())


# =========================
# 2. REMOVE MISSING VALUES
# =========================

data = data.dropna(subset=["headline", "clickbait"])

# Make sure label is integer
data["clickbait"] = data["clickbait"].astype(int)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# 3. FEATURES & LABEL
# =========================

X = data["headline"]
y = data["clickbait"]


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. TF-IDF + LOGISTIC REGRESSION
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# =========================
# 6. TRAIN
# =========================

print("\nTraining Clickbait model...")

model.fit(X_train, y_train)


# =========================
# 7. EVALUATE
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("CLICKBAIT MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Non-Clickbait", "Clickbait"]
))


# =========================
# 8. SAVE MODEL
# =========================

joblib.dump(
    model,
    "models/clickbait_model.pkl"
)

print("\nModel saved successfully!")
print("models/clickbait_model.pkl")