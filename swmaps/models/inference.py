from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swmaps.models.dataset import SaltwaterSegDataset
from swmaps.models.model import FarSegModel


def run_segmentation(
    mosaics,
    out_dir,
    num_classes=2,
    batch_size=4,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    samples = [{"image_path": p, "mask_path": None} for p in mosaics]
    dataset = SaltwaterSegDataset(samples, inference_only=True)
    loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FarSegModel(num_classes=num_classes)
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1).cpu()
            preds.extend(pred)

    # Write outputs as GeoTIFFs
    for mosaic_path, mask in zip(mosaics, preds):
        out_path = out_dir / f"{Path(mosaic_path).stem}_segmentation.tif"
        _write_mask_like(mosaic_path, mask, out_path)

    return out_dir


def _write_mask_like(ref_raster, mask_tensor, out_path):
    import rasterio

    with rasterio.open(ref_raster) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_tensor.numpy().astype("uint8"), 1)
