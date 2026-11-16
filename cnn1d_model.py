"""
Lightweight 1-D CNN classifier for the DJIA news-sentiment dataset.
 - Predicts TOMORROW's direction (label shifted by one day).
 - Chronological train/val/test split, scaler + PCA fit on train only.
 - Repeats training over several seeds and reports mean +/- std.

Run:
    python cnn1d_model.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                            precision_score, recall_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout,
                                    Flatten, Dense)
from tensorflow.keras.callbacks import EarlyStopping

from pca import PCA                      # local file

SEEDS = [0, 1, 2, 3, 4]


# ------------------------------------------------------------------
# 1.  DATA  (target shifted: today's features -> tomorrow's label)
# ------------------------------------------------------------------
def load_data():
    db_news   = pd.read_csv("Combined_News_DJIA.csv")
    db_prices = pd.read_csv("DJIA_table.csv")
    df = db_news.merge(db_prices, how="inner", on="Date")
    df = df.sort_values("Date").reset_index(drop=True)   # chronological order

    analyzer = VS()
    headlines = [" ".join(map(str, df.iloc[i, 2:27])) for i in range(len(df))]

    df["subjectivity"] = [tb(t).sentiment.subjectivity for t in headlines]
    df["polarity"]     = [tb(t).sentiment.polarity     for t in headlines]

    vs_scores = [analyzer.polarity_scores(t) for t in headlines]
    df["compound"] = [s["compound"] for s in vs_scores]
    df["negative"] = [s["neg"]      for s in vs_scores]
    df["positive"] = [s["pos"]      for s in vs_scores]
    df["neutral"]  = [s["neu"]      for s in vs_scores]

    df["target"] = df["Label"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    feats = ["Open", "High", "Low", "Volume",
            "subjectivity", "polarity",
            "compound", "negative", "positive", "neutral"]
    return df[feats].values, df["target"].values.astype("float32")


def prepare_splits(X, y):
    """Chronological 60/20/20 split. Scalers and PCA are fit on TRAIN only."""
    n = len(X)
    i_tr, i_val = int(n * 0.6), int(n * 0.8)

    sc1 = StandardScaler().fit(X[:i_tr])
    X_s = sc1.transform(X)

    pca = PCA()
    pca.fit(X_s[:i_tr])
    X_red = pca.transform_rv(X_s, retained_variance=0.95)

    X_f = StandardScaler().fit(X_red[:i_tr]).transform(X_red)
    X_f = X_f.reshape((X_f.shape[0], X_f.shape[1], 1)).astype("float32")

    return (X_f[:i_tr], y[:i_tr],
            X_f[i_tr:i_val], y[i_tr:i_val],
            X_f[i_val:], y[i_val:])


# ------------------------------------------------------------------
# 2.  MODEL
# ------------------------------------------------------------------
def build_model(input_shape):
    cnn = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Flatten(),
        Dense(32, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        Dense(1, activation="sigmoid")
    ])
    cnn.compile(tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="binary_crossentropy", metrics=["accuracy"])
    return cnn


# ------------------------------------------------------------------
# 3.  ONE RUN
# ------------------------------------------------------------------
def run_once(seed, splits):
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    cnn = build_model(X_tr.shape[1:])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = cnn.fit(X_tr, y_tr, epochs=50, batch_size=32,
                validation_data=(X_val, y_val),
                # callbacks=[es],
                verbose=0)

    test_loss, test_acc = cnn.evaluate(X_te, y_te, verbose=0)
    y_pred = (cnn.predict(X_te, verbose=0) > 0.5).astype(int).ravel()

    metrics = {
        "seed": seed,
        "epochs_run": len(hist.history["loss"]),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "accuracy": accuracy_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "frac_pred_up": y_pred.mean(),   # ~0 or ~1 means a one-class model
    }
    return hist.history, metrics


# ------------------------------------------------------------------
# 4.  AGGREGATE PLOT (mean curve + min/max band across seeds)
# ------------------------------------------------------------------
def stack(histories, key):
    max_len = max(len(h[key]) for h in histories)
    arr = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        arr[i, :len(h[key])] = h[key]
    return arr


def band(ax, arr, label, color):
    x = np.arange(1, arr.shape[1] + 1)
    ax.plot(x, np.nanmean(arr, axis=0), color=color, label=label)
    ax.fill_between(x, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                    color=color, alpha=0.2)


def plot_runs(histories, test_losses, test_accs, y_te):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    n = len(histories)

    band(ax1, stack(histories, "loss"), "Train loss (mean)", "C0")
    band(ax1, stack(histories, "val_loss"), "Validation loss (mean)", "C1")
    ax1.axhline(np.mean(test_losses), color="red", linestyle=":",
                label=f"Test loss ({np.mean(test_losses):.3f} ± {np.std(test_losses, ddof=1):.3f})")
    ax1.axhline(np.log(2), color="black", linewidth=0.8, label="Chance (ln 2)")
    ax1.set(xlabel="Epoch", ylabel="Binary cross-entropy loss",
            title=f"Loss per epoch ({n} seeds, band = min/max)")
    ax1.legend(); ax1.grid(alpha=0.3)

    band(ax2, stack(histories, "accuracy"), "Train accuracy (mean)", "C0")
    band(ax2, stack(histories, "val_accuracy"), "Validation accuracy (mean)", "C1")
    ax2.axhline(np.mean(test_accs), color="red", linestyle=":",
                label=f"Test acc ({np.mean(test_accs):.3f} ± {np.std(test_accs, ddof=1):.3f})")
    base = max(y_te.mean(), 1 - y_te.mean())
    ax2.axhline(base, color="black", linewidth=0.8,
                label=f"Majority-class baseline ({base:.3f})")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy per epoch")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("cnn_training_curves.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# 5.  MAIN
# ------------------------------------------------------------------
def main():
    X, y = load_data()
    splits = prepare_splits(X, y)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits
    print(f"Train: {len(X_tr)}  Validation: {len(X_val)}  Test: {len(X_te)}")
    print(f"Input shape: {X_tr.shape[1:]}")
    print(f"Majority-class baseline on test: "
        f"{max(y_te.mean(), 1 - y_te.mean()):.3f}\n")

    histories, rows = [], []
    for seed in SEEDS:
        print(f"--- run with seed {seed} ---")
        h, m = run_once(seed, splits)
        histories.append(h)
        rows.append(m)
        print(f"  epochs={m['epochs_run']}  test acc={m['accuracy']:.3f}  "
            f"test loss={m['test_loss']:.3f}")

    results = pd.DataFrame(rows).set_index("seed")
    print("\n=== Per-seed results ===")
    print(results.round(4))

    summary = results.drop(columns="epochs_run").agg(["mean", "std"]).T
    print(f"\n=== 1-D CNN, held-out test set: mean ± std over {len(SEEDS)} seeds ===")
    for name, r in summary.iterrows():
        print(f"{name:13s}: {r['mean']:.4f} ± {r['std']:.4f}")

    results.to_csv("cnn_results.csv")
    plot_runs(histories, results["test_loss"].values,
            results["accuracy"].values, y_te)


if __name__ == "__main__":
    main()
