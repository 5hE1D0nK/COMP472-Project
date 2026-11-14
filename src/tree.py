import numpy as np
import random
from collections import Counter
from utils import load_task

def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())
    
def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    if n == 0:
        return 0.0
    w_left  = len(y_left)  / n
    w_right = len(y_right) / n
    return entropy(y_parent) - (w_left * entropy(y_left) + w_right * entropy(y_right))

class TreeNode:
    def __init__(self):
        self.feature_dix = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = None

    def is_leaf(self):
        return self.label is not None

def find_best_split(X, y, n_features=None):
    n, n_feats = X.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    onehot = np.zeros((n, len(classes)))
    onehot[np.arange(n), y_idx] = 1
    total = onehot.sum(0)
    p = total[total > 0] / n
    parent_h = -np.sum(p * np.log2(p))

    feats = (random.sample(range(n_feats), min(n_features, n_feats))
             if n_features is not None else range(n_feats))
    best_gain, best_feat, best_thresh = -1, None, None
    nl = np.arange(1, n)[:, None]      # left sizes 1..n-1
    nr = n - nl

    for fi in feats:
        col = X[:, fi]
        order = np.argsort(col, kind="stable")
        sv = col[order]
        left = np.cumsum(onehot[order], axis=0)[:-1]   # class counts left of each split
        right = total - left
        with np.errstate(divide="ignore", invalid="ignore"):
            pl, pr = left / nl, right / nr
            hl = -np.nansum(np.where(pl > 0, pl * np.log2(pl), 0), axis=1)
            hr = -np.nansum(np.where(pr > 0, pr * np.log2(pr), 0), axis=1)
        gain = parent_h - (nl[:, 0] / n) * hl - (nr[:, 0] / n) * hr
        gain[sv[:-1] == sv[1:]] = -1                    # no split between equal values
        j = int(np.argmax(gain))
        if gain[j] > best_gain:
            best_gain, best_feat = gain[j], fi
            best_thresh = (sv[j] + sv[j + 1]) / 2
    return best_feat, best_thresh, best_gain

def build_tree(X, y, max_depth=None, min_samples=1, depth=0, n_features=None):
    node = TreeNode()

    if (len(set(y)) == 1 or len(y) <= min_samples
            or (max_depth is not None and depth >= max_depth)):
        node.label = Counter(y.tolist()).most_common(1)[0][0]
        return node

    f, t, g = find_best_split(X, y, n_features)
    if f is None or g <= 0:
        node.label = Counter(y.tolist()).most_common(1)[0][0]
        return node

    node.feature_idx, node.threshold = f, t
    m = X[:, f] <= t
    node.left  = build_tree(X[m],  y[m],  max_depth, min_samples, depth + 1, n_features)
    node.right = build_tree(X[~m], y[~m], max_depth, min_samples, depth + 1, n_features)
    return node

def predict_one(tree, x):
    if tree.is_leaf():
        return tree.label
    if x[tree.feature_idx] <= tree.threshold:
        return predict_one(tree.left,  x)
    else:
        return predict_one(tree.right, x)

def predict(tree, X):
    return [predict_one(tree, x) for x in X]

class DecisionTreeClassifier:
        def __init__(self, max_depth=None, min_samples=1, n_features=None):
            self.max_depth   = max_depth
            self.min_samples = min_samples
            self.n_features  = n_features
            self.root        = None
    
        def fit(self, X, y):
            self.root = build_tree(X, y, self.max_depth, self.min_samples,
                                   n_features=self.n_features)
            return self
    
        def predict(self, X):
            return predict(self.root, X)
    
        def score(self, X, y):
            preds = self.predict(X)
            return sum(p == t for p, t in zip(preds, y)) / len(y)

for task in ["img", "sent"]:
    X_tr, y_tr, X_va, y_va = load_task(task)
    print(f"\nDataset: {task}")
    for depth in [5, 10, 20, None]:
        if task == "img":
            clf = DecisionTreeClassifier(max_depth=depth, n_features=22).fit(X_tr, y_tr)
        else:
            clf = DecisionTreeClassifier(max_depth=depth, n_features=28).fit(X_tr, y_tr)
        print(f"max_depth={depth}: Validation Accuracy={clf.score(X_va, y_va):.2%}")