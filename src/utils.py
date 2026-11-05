import numpy as np, pandas as pd

def load_task(task):  # task = "img" or "sent"
    csv = "../data/image_train.csv" if task == "img" else "../data/sentiment_train.csv"
    df = pd.read_csv(csv)
    X = df.filter(like="feature_").values
    y = df["label"].values
    tr = np.load(f"../splits/{task}_split_train.npy")
    va = np.load(f"../splits/{task}_split_val.npy")
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]

    # normalize using TRAIN statistics only
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-8
    return (X_tr - mean) / std, y_tr, (X_va - mean) / std, y_va