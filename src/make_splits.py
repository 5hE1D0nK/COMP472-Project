import numpy as np, pandas as pd, random, torch
from utils import load_task

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def stratified_split(y, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        val_idx.extend(idx[:int(len(idx) * val_frac)])
    val_idx = np.array(val_idx)
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
    return train_idx, val_idx

# Load datasets
img = pd.read_csv("../data/image_train.csv")
sent = pd.read_csv("../data/sentiment_train.csv")

# Image dataset
X_img= img.filter(like="feature_").values
y_img = img["label"].values

tr_img, va_img = stratified_split(y_img)

# sent dataset
X_sent = sent.filter(like="feature_").values
y_sent = sent["label"].values

tr_sent, va_sent = stratified_split(y_sent)

# Save splits
np.save("../splits/img_split_train.npy", tr_img)
np.save("../splits/img_split_val.npy", va_img)

np.save("../splits/sent_split_train.npy", tr_sent)
np.save("../splits/sent_split_val.npy", va_sent)

# Quick check 
for task in ["img", "sent"]:
    X_tr, y_tr, X_va, y_va = load_task(task)
    print(task, X_tr.shape, X_va.shape)
    print("  class counts (train):", np.bincount(y_tr)[:5], "...")
    print("  class counts (val):  ", np.bincount(y_va)[:5], "...")
    print("  train mean ~0, std ~1:", X_tr.mean().round(3), X_tr.std().round(3))