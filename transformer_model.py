"""
Transformer classifier for the DJIA news-sentiment dataset.
 - Predicts TOMORROW's direction from a window of the last WINDOW days.
 - Repeats training over several seeds and reports mean +/- std.

Run:
    python transformer_model.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from textblob import TextBlob as tb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

SEEDS = [0, 1, 2, 3, 4, 5, 6]   # number of repeated runs
WINDOW = 15               # days of history per sample
D_MODEL = 32
NUM_HEADS = 2
FF_DIM = 64
NUM_BLOCKS = 2
DROPOUT = 0.2


# ------------------------------------------------------------------
# 1. DATA  (target is shifted: today's window -> tomorrow's label)
# ------------------------------------------------------------------
def load_data():
    db_news = pd.read_csv("Combined_News_DJIA.csv")
    db_prices = pd.read_csv("DJIA_table.csv")
    df = db_news.merge(db_prices, how="inner", on="Date")
    df = df.sort_values("Date").reset_index(drop=True)

    analyzer = VS()
    headlines = [" ".join(map(str, df.iloc[i, 2:27])) for i in range(len(df))]

    df["subjectivity"] = [tb(t).sentiment.subjectivity for t in headlines]
    df["polarity"] = [tb(t).sentiment.polarity for t in headlines]
    vs = [analyzer.polarity_scores(t) for t in headlines]
    df["compound"] = [s["compound"] for s in vs]
    df["negative"] = [s["neg"] for s in vs]
    df["positive"] = [s["pos"] for s in vs]
    df["neutral"] = [s["neu"] for s in vs]

    df["target"] = df["Label"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    feats = ["Open", "High", "Low", "Volume", "subjectivity", "polarity",
            "compound", "negative", "positive", "neutral"]
    return df[feats].values, df["target"].values.astype("float32")


def make_windows(X, y, window):
    Xs, ys = [], []
    for i in range(window - 1, len(X)):
        Xs.append(X[i - window + 1: i + 1])
        ys.append(y[i])
    return np.array(Xs, dtype="float32"), np.array(ys, dtype="float32")


def prepare_splits(X, y):
    """Chronological 60/20/20 split; scaler fit on train rows only."""
    n = len(X)
    i_tr, i_val = int(n * 0.6), int(n * 0.8)
    X = StandardScaler().fit(X[:i_tr]).transform(X)

    X_tr, y_tr = make_windows(X[:i_tr], y[:i_tr], WINDOW)
    X_val, y_val = make_windows(X[i_tr:i_val], y[i_tr:i_val], WINDOW)
    X_te, y_te = make_windows(X[i_val:], y[i_val:], WINDOW)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ------------------------------------------------------------------
# 2. MODEL
# ------------------------------------------------------------------
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.proj = layers.Dense(d_model)
        self.pos = layers.Embedding(seq_len, d_model)

    def call(self, x):
        return self.proj(x) + self.pos(tf.range(self.seq_len))


def encoder_block(x):
    attn = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=D_MODEL // NUM_HEADS, dropout=DROPOUT)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(DROPOUT)(attn))
    ff = layers.Dense(FF_DIM, activation="relu")(x)
    ff = layers.Dense(D_MODEL)(layers.Dropout(DROPOUT)(ff))
    return layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(DROPOUT)(ff))


def build_model(seq_len, n_features):
    inp = layers.Input(shape=(seq_len, n_features))
    x = PositionalEmbedding(seq_len, D_MODEL)(inp)
    for _ in range(NUM_BLOCKS):
        x = encoder_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ------------------------------------------------------------------
# 3. ONE RUN
# ------------------------------------------------------------------
def run_once(seed, splits):
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_model(WINDOW, X_tr.shape[2])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    hist = model.fit(X_tr, y_tr, epochs=100, batch_size=64,
                    validation_data=(X_val, y_val),
                    callbacks=[rlr], verbose=0)

    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).ravel()

    metrics = {
        "seed": seed,
        "epochs_run": len(hist.history["loss"]),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "accuracy": accuracy_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "frac_pred_up": y_pred.mean(),   # ~1.0 or ~0.0 means a degenerate model
    }
    return hist.history, metrics


# ------------------------------------------------------------------
# 4. AGGREGATE PLOT (mean curve + min/max band across seeds)
# ------------------------------------------------------------------
def stack(histories, key):
    max_len = max(len(h[key]) for h in histories)
    arr = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        arr[i, :len(h[key])] = h[key]      # early stopping -> runs differ in length
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
    plt.savefig("transformer_training_curves.png", dpi=150)
    plt.show()


# ------------------------------------------------------------------
# 5. MAIN
# ------------------------------------------------------------------
def main():
    X, y = load_data()
    splits = prepare_splits(X, y)
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits
    print(f"Train: {len(X_tr)}  Validation: {len(X_val)}  Test: {len(X_te)}")
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
    print("\n=== Transformer, held-out test set: mean ± std over "
          f"{len(SEEDS)} seeds ===")
    for name, r in summary.iterrows():
        print(f"{name:13s}: {r['mean']:.4f} ± {r['std']:.4f}")

    results.to_csv("transformer_results.csv")
    plot_runs(histories, results["test_loss"].values,
              results["accuracy"].values, y_te)


if __name__ == "__main__":
    main()