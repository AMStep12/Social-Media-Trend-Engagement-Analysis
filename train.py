import os, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

DATA = "data/synthetic/posts.csv"
ART = "artifacts"; os.makedirs(ART, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA)
    df["engagement"] = df["likes"] + df["comments"] + df["shares"]
    X_text = df["text"].fillna("")
    y = df["engagement"].values
    vec = TfidfVectorizer(min_df=5, ngram_range=(1,2))
    X = vec.fit_transform(X_text)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"R2 train={model.score(X_train,y_train):.3f}  R2 test={model.score(X_test,y_test):.3f}")
    joblib.dump(vec, os.path.join(ART, "tfidf.joblib"))
    joblib.dump(model, os.path.join(ART, "linreg.joblib"))
    print("Saved vectorizer and model to artifacts/")
