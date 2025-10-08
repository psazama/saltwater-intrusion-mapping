"""
Deep learning model for salinity classification/regression.

Implements a simple CNN inspired by Garajeh et al. (2022).
Can be extended to U-Net or other architectures.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Model definition
# -----------------------


class SalinityCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(SalinityCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Flatten and FC layers
        # adjust depending on input patch size!
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------
# Training & evaluation helpers
# -----------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Train the CNN with cross-entropy loss.
    Returns best validation metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {total_loss/len(train_loader):.4f}, "
            f"Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_salinitycnn.pt")

    return {"val_acc": best_val_acc}


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str = "best_salinitycnn.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Evaluate CNN on a test set.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for inputs, y in test_loader:
            inputs, y = inputs.to(device), y.to(device)
            outputs = model(inputs)
            y_pred = torch.argmax(outputs, dim=1)
            preds.extend(y_pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# -----------------------
# Example Dataset stub
# -----------------------


class SalinityPatchDataset(Dataset):
    """
    Simple dataset wrapper for salinity patches.
    Expects numpy arrays X (N, C, H, W) and y (N,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
