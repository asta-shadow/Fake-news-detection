import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from preprocess import clean_text

# Load datasets
data1 = pd.read_csv("dataset/WELFake_Dataset.csv")
data2 = pd.read_csv("dataset/CoAID.csv")

data = pd.concat([data1, data2])

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

x_train_vec = tfidf.fit_transform(x_train)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_vec, y_train)

pickle.dump(model, open("models/fakenews_model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf_vectorizer.pkl", "wb"))

print("Model Trained Successfully")