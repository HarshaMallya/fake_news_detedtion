from flask import Flask, request, render_template
import pickle
import re
import nltk
import webbrowser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# 0. Download required NLTK data
# -------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  # ✅ important fix

# -------------------------------
# 1. Define TextCleaner (same as training)
# -------------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return " ".join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(t) for t in X]

# -------------------------------
# 2. Load trained pipeline
# -------------------------------
try:
    with open("fake_news_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print("Error: fake_news_pipeline.pkl not found. Run fake_news_detection.py first.")
    exit()

# -------------------------------
# 3. Initialize Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 4. Home Route
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form["news"]

        if news_text.strip() != "":
            pred = pipeline.predict([news_text])[0]
            prob = pipeline.predict_proba([news_text])[0]
            confidence = max(prob) * 100
            prediction = "✅ Real News" if pred == 1 else "⚠️ Fake News"
        else:
            prediction = "Please enter some text."

    return render_template("index.html", prediction=prediction, news=news_text)

# -------------------------------
# 5. Run Server
# -------------------------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000/"
    webbrowser.open(url)
    app.run(debug=True)