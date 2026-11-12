"""
Lightweight 1‑D CNN classifier for the DJIA news‑sentiment dataset.

Run:
    python cnn1d_model.py
"""
import pandas as pd
import numpy as np
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                            f1_score, precision_score, recall_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout,
                                    Flatten, Dense)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from pca import PCA                      # local file


# ------------------------------------------------------------------
# 1.  DATA LOADING & FEATURE ENGINEERING  (same as RF)
# ------------------------------------------------------------------
def load_and_preprocess():
    db_news   = pd.read_csv("Combined_News_DJIA.csv")
    db_prices = pd.read_csv("DJIA_table.csv")
    df = db_news.merge(db_prices, how="inner", on="Date")

    analyzer = VS()
    headlines = [" ".join(map(str, df.iloc[i, 2:27])) for i in range(len(df))]

    df["subjectivity"] = [tb(t).sentiment.subjectivity for t in headlines]
    df["polarity"]     = [tb(t).sentiment.polarity     for t in headlines]

    vs_scores = [analyzer.polarity_scores(t) for t in headlines]
    df["compound"] = [s["compound"] for s in vs_scores]
    df["negative"] = [s["neg"]      for s in vs_scores]
    df["positive"] = [s["pos"]      for s in vs_scores]
    df["neutral"]  = [s["neu"]      for s in vs_scores]

    y = df.pop("Label")

    X = df[["Open", "High", "Low", "Volume",
            "subjectivity", "polarity",
            "compound", "negative", "positive", "neutral"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    X_reduced = pca.transform_rv(X_scaled, retained_variance=0.95)

    X_final = StandardScaler().fit_transform(X_reduced)
    return X_final, y


def plot_history(history, test_loss, test_acc):
    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]
    epochs     = range(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss, label="Validation loss")
    ax1.axhline(test_loss, color="red", linestyle=":",
                label=f"Final test loss ({test_loss:.3f})")
    ax1.axvline(best_epoch, color="gray", linestyle="--",
                label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary cross-entropy loss")
    ax1.set_title("Loss per epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"], label="Train accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    ax2.axhline(test_acc, color="red", linestyle=":",
                label=f"Final test accuracy ({test_acc:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# 2.  1‑D CNN TRAIN / TEST
# ------------------------------------------------------------------
def main():
    X, y = load_and_preprocess()

    # reshape:  (samples, timesteps, channels)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 1) hold out the final test set (20%)
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 2) split the rest into train (60% of total) and validation (20% of total)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp
    )
    print(f"Train: {len(X_tr)}  Validation: {len(X_val)}  Test: {len(X_te)}")

    cnn = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=X.shape[1:]),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Flatten(),
        Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        # Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    cnn.compile(tf.keras.optimizers.Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    cnn.summary()

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = cnn.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),   # validation, NOT the test set
        callbacks=[es],
        verbose=1
    )

    # Final, untouched test evaluation (uses the best weights restored by EarlyStopping)
    test_loss, test_acc = cnn.evaluate(X_te, y_te, verbose=0)
    plot_history(history, test_loss, test_acc)

    y_pred = (cnn.predict(X_te) > 0.5).astype(int)

    print("\n=== 1‑D CNN results (held-out test set) ===")
    print("Test loss :", test_loss)
    print("Accuracy :", accuracy_score(y_te, y_pred))
    print(classification_report(y_te, y_pred))
    print("F1 :", f1_score(y_te, y_pred))
    print("Precision :", precision_score(y_te, y_pred))
    print("Recall :", recall_score(y_te, y_pred))


if __name__ == "__main__":
    main()

