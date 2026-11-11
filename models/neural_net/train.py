import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure Python can find src/ and models/ regardless of where you execute from
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "models" / "neural_net"))

# Change working directory to src so utils.py relative paths ('../data') work
import os
os.chdir(ROOT / "src")

from utils import load_task
from model import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_task(task: str, in_dim: int, num_classes: int, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3):
    print(f"\n--- Training {task.upper()} ({in_dim} features -> {num_classes} classes) ---")
    
    # 1. Load data directly using your existing utils.py
    X_tr, y_tr, X_va, y_va = load_task(task)
    
    # 2. Wrap into PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    
    # 3. Model setup
    model = NeuralNetwork(in_features=in_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                
        val_acc = correct / len(val_loader.dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} (Best: {best_acc:.4f})")
            
    return best_acc

if __name__ == "__main__":
    img_acc = train_one_task("img", in_dim=500, num_classes=50, epochs=50, batch_size=128, lr=1e-3)
    sent_acc = train_one_task("sent", in_dim=768, num_classes=3, epochs=35, batch_size=64, lr=5e-4)
    
    print("\n" + "="*40)
    print(f"Best Image Val Acc:     {img_acc:.4f}")
    print(f"Best Sentiment Val Acc: {sent_acc:.4f}")
    print(f"Combined Local Score:   {((img_acc + sent_acc) / 2):.4f}")
    print("="*40)