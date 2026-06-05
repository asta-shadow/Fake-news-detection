import os
import pickle
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from flask_cors import CORS
from scraper import get_latest_news
from preprocess import clean_text

app = Flask(__name__)
CORS(app)

# API KEY
GROQ_API_KEY = "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"

MODEL_PATH = "models/fakenews_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# ---------------- LLM VERIFICATION ---------------- #

def llm_verify(news_content):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional fact checker. Determine if the news is REAL or FAKE and explain briefly."
            },
            {
                "role": "user",
                "content": news_content
            }
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM verification failed: {str(e)}"


# ---------------- URL ARTICLE SCRAPER ---------------- #

def extract_article(url):

    try:
        page = requests.get(url, timeout=10)

        soup = BeautifulSoup(page.text, "html.parser")

        paragraphs = soup.find_all("p")

        article = " ".join([p.get_text() for p in paragraphs])

        return article

    except Exception as e:

        print("URL extraction error:", e)

        return ""


# ---------------- HOME PAGE ---------------- #

@app.route("/")
def home():

    trending = get_latest_news()

    return render_template(
        "index.html",
        trending=trending,
        active_tab="scan"
    )


# ---------------- TEXT ANALYSIS ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    news_input = request.form.get("news", "").strip()

    if not news_input:

        return render_template(
            "index.html",
            trending=get_latest_news(),
            error="Please enter news text.",
            active_tab="scan"
        )

    ml_prediction = "UNKNOWN"

    try:

        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):

            model = pickle.load(open(MODEL_PATH, "rb"))

            vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

            cleaned = clean_text(news_input)

            vector = vectorizer.transform([cleaned])

            pred = model.predict(vector)[0]

            ml_prediction = "REAL" if str(pred) in ["1", "True", "REAL"] else "FAKE"

    except Exception as e:

        print("ML Error:", e)

    llm_result = llm_verify(news_input)

    return render_template(
        "index.html",
        news_text=news_input,
        ml_result=ml_prediction,
        result=llm_result,
        trending=get_latest_news(),
        active_tab="scan"
    )


# ---------------- URL ANALYSIS ---------------- #

@app.route("/predict_url", methods=["POST"])
def predict_url():

    news_url = request.form.get("url", "").strip()

    if not news_url:

        return render_template(
            "index.html",
            trending=get_latest_news(),
            error="Please enter a URL.",
            active_tab="url"
        )

    article_text = extract_article(news_url)

    if article_text == "":
        url_result = "Could not extract article text from this URL."
    else:
        url_result = llm_verify(article_text)

    return render_template(
        "index.html",
        url_input=news_url,
        url_result=url_result,
        trending=get_latest_news(),
        active_tab="url"
    )


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True, port=5000)