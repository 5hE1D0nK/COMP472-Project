import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.3):
        super().__init__()
        
        # Sequential chains layers together in order
        self.network = nn.Sequential(
            # First hidden block
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden block
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Final output layer: maps directly to class count (50 or 3)
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a batch of embeddings and returns raw logits."""
        return self.network(x)
