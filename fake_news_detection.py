import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# Download NLTK data
# -------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -------------------------
# Text Cleaning Function
# -------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -------------------------
# Load dataset
# -------------------------
df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

# Add labels
df_fake["label"] = 0
df_true["label"] = 1

# Combine title + text (IMPORTANT)
df_fake["content"] = df_fake["title"] + " " + df_fake["text"]
df_true["content"] = df_true["title"] + " " + df_true["text"]

df_fake = df_fake[["content", "label"]]
df_true = df_true[["content", "label"]]

df = pd.concat([df_fake, df_true])

# -------------------------
# Add extra data
# -------------------------
df_extra = pd.read_csv("data/extra_news.csv")
df_extra.columns = ["content", "label"]

df = pd.concat([df, df_extra])

# -------------------------
# Clean text
# -------------------------
df["content"] = df["content"].apply(clean_text)

# -------------------------
# Split data
# -------------------------
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------
# Create pipeline
# -------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("model", LogisticRegression())
])

# -------------------------
# Train model
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# Save model
# -------------------------
with open("fake_news_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved!")