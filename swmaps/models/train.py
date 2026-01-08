import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from swmaps.models.dataset import SaltwaterSegDataset
from swmaps.models.model import FarSegModel


def train(train_samples, val_samples, num_classes, epochs=20, lr=1e-4):
    # create datasets
    train_ds = SaltwaterSegDataset(train_samples)
    # val_ds = SaltwaterSegDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    # model
    model = FarSegModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} train loss: {running_loss / len(train_loader)}")
