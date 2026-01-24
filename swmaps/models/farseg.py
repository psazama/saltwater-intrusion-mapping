from pathlib import Path

import rasterio
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchgeo.models import FarSeg
from tqdm import tqdm

from swmaps.models.base import BaseSegModel


class FarSegDataset(Dataset):
    """Internal helper to load aligned GeoTIFF pairs."""

    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

        with rasterio.open(img_path) as src:
            # Read all bands, scale to [0, 1] if uint16
            img = src.read().astype("float32")
            if img.max() > 1.0:
                img /= 65535.0 if src.dtypes[0] == "uint16" else 255.0

        with rasterio.open(mask_path) as src:
            # Mask is usually single band
            mask = src.read(1).astype("int64")

        h, w = img.shape[1], img.shape[2]
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32

        pad_h = new_h - h
        pad_w = new_w - w

        # img shape: [C, H, W] -> pad is (left, right, top, bottom)
        img_tensor = torch.from_numpy(img)
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        mask_tensor = torch.from_numpy(mask)
        mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return img_tensor, mask_tensor


class FarSegModel(BaseSegModel):
    """
    Wrapper around TorchGeo's FarSeg for semantic segmentation.
    https://arxiv.org/pdf/2011.09766
    """

    def __init__(
        self,
        backbone="resnet50",
        num_classes=16,
        backbone_pretrained=True,
        in_channels=6,
    ):
        super().__init__(num_classes)
        self.model = FarSeg(
            backbone=backbone,
            classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

        # If input channels != 3, we must replace the first conv layer
        if in_channels != 3:
            # For ResNet backbones, the first layer is usually self.model.backbone.conv1
            old_conv = self.model.backbone.conv1
            self.model.backbone.conv1 = torch.nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

    def forward(self, x):
        return self.model(x)

    def train_model(
        self, data_pairs, out_dir, epochs=10, batch_size=4, lr=1e-4, **kwargs
    ):
        """
        Executes the FarSeg-specific training loop.
        """
        device = self.device
        self.to(device)

        # 1. Prepare Data
        dataset = FarSegDataset(data_pairs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Setup Optimizer & Loss
        # FarSeg benefits from Adam or SGD with momentum
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Standard CrossEntropy is used, but FarSeg's internal
        # Relation-aware module handles the 'far-field' context during forward()
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        print(f"Starting FarSeg training on {device}...")

        self.train()
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = self.forward(images)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")

        # 3. Save weights
        out_path = Path(out_dir) / "farseg_final.pth"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_path)
        print(f"Model saved to {out_path}")
