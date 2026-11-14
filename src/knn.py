import numpy as np
from collections import Counter
from utils import load_task

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

class KNNClassifier:
    def __init__(self, k, distance_fn=euclidean_distance):
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        return self
        
    def _predict_one(self, x):
        # Calculate distances between x and all training samples
        dists = self.distance_fn(self.X_train, x)
        # Find the k nearest neighbors
        nearest = np.argpartition(dists, self.k - 1)[:self.k]
        # Majority voting
        return Counter(self.y_train[nearest]).most_common(1)[0][0]

    def predict(self, X_test, batch=500):
        X_test = np.asarray(X_test)
        # Square each feature, add them up per row
        train_sq = (self.X_train ** 2).sum(1)
        preds = []
        # Process 500 test points at a time
        for s in range(0, len(X_test), batch):
            # Extract the current batch
            chunk = X_test[s:s + batch]
            # Calculate the squared Euclidean distances
            d2 = (chunk ** 2).sum(1)[:, None] - 2 * chunk @ self.X_train.T + train_sq
            # Find the k nearest neighbors
            nearest = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
            # Majority voting
            preds.extend(Counter(self.y_train[row]).most_common(1)[0][0] for row in nearest)
        return preds

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        # number of correct predictions / total
        return sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
        

for task in ["img", "sent"]:
    X_tr, y_tr, X_va, y_va = load_task(task)
    print(f"\nDataset: {task}")
    for k in [1, 3, 5, 10, 20]:
        clf = KNNClassifier(k=k)
        clf.fit(X_tr, y_tr)
        accuracy = clf.score(X_va, y_va)
        print(f"K={k}: Validation Accuracy={accuracy:.2%}")