"""
Baseline tree‑based classifier for the DJIA news‑sentiment dataset.

Run:
    python random_forest_model.py
"""
import pandas as pd
import numpy as np
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from pca import PCA                      # local file

# ------------------------------------------------------------------
# 1.  DATA LOADING & FEATURE ENGINEERING
# ------------------------------------------------------------------
def load_and_preprocess():
    db_news   = pd.read_csv("Combined_News_DJIA.csv")
    db_prices = pd.read_csv("DJIA_table.csv")  # <- make sure name matches disk
    df = db_news.merge(db_prices, how="inner", on="Date")

    # ----- sentiment from 25 headlines combined -----
    analyzer = VS()
    headlines = [" ".join(map(str, df.iloc[i, 2:27])) for i in range(len(df))]

    df["subjectivity"] = [tb(t).sentiment.subjectivity for t in headlines]
    df["polarity"]     = [tb(t).sentiment.polarity     for t in headlines]

    vs_scores = [analyzer.polarity_scores(t) for t in headlines]
    df["compound"] = [s["compound"] for s in vs_scores]
    df["negative"] = [s["neg"]      for s in vs_scores]
    df["positive"] = [s["pos"]      for s in vs_scores]
    df["neutral"]  = [s["neu"]      for s in vs_scores]

    y = df.pop("Label")                       # binary target (0 / 1)

    X = df[["Open", "High", "Low", "Volume",
            "subjectivity", "polarity",
            "compound", "negative", "positive", "neutral"]]

    # scale ➔ PCA ➔ scale (again) to match LSTM pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    X_reduced = pca.transform_rv(X_scaled, retained_variance=0.95)

    X_final = StandardScaler().fit_transform(X_reduced)
    return X_final, y


# ------------------------------------------------------------------
# 2.  RANDOM‑FOREST TRAIN / TEST
# ------------------------------------------------------------------
def main():
    X, y = load_and_preprocess()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    print("\n=== Random‑Forest results ===")
    print("Accuracy :", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))
    print("F1 :", f1_score(y_te, y_pred))
    print("Precision :", precision_score(y_te, y_pred))
    print("Recall :", recall_score(y_te, y_pred))


if __name__ == "__main__":
    main()

