from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swmaps.models.dataset import SaltwaterSegDataset
from swmaps.models.model_factory import get_model


def run_segmentation(
    mosaics,
    out_dir,
    model_name="farseg",
    num_classes=2,
    batch_size=4,
    save_png=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    samples = [{"image_path": p, "mask_path": None} for p in mosaics]
    dataset = SaltwaterSegDataset(samples, inference_only=True)
    loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, num_classes=num_classes)
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu()
            preds.extend(pred)

    # Write outputs as GeoTIFFs and optionally png
    for mosaic_path, mask in zip(mosaics, preds):
        stem = Path(mosaic_path).stem
        tif_path = out_dir / f"{stem}_segmentation.tif"

        _write_mask_like(mosaic_path, mask, tif_path)

        if save_png:
            png_path = out_dir / f"{stem}_segmentation.png"
            print(mask.min().item(), mask.max().item(), torch.unique(mask)[:10])
            _write_png(mask, png_path)

    return out_dir


def _write_mask_like(ref_raster, mask_tensor, out_path):
    import rasterio

    with rasterio.open(ref_raster) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_tensor.numpy().astype("uint8"), 1)


def _write_png(mask_tensor, out_path):
    import numpy as np
    from PIL import Image

    arr = mask_tensor.numpy().astype("uint8")
    h, w = arr.shape

    unique_classes = np.unique(arr)

    # Generate a random color for each class
    rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
    class_colors = {
        cls: rng.integers(0, 256, size=3, dtype=np.uint8) for cls in unique_classes
    }

    # Create empty RGB image
    rgb_arr = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in class_colors.items():
        mask = arr == cls
        rgb_arr[mask] = color

    img = Image.fromarray(rgb_arr, mode="RGB")
    img.save(out_path)
