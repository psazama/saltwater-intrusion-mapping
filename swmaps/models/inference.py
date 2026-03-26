"""Segmentation model inference utilities.

This module provides :func:`run_segmentation`, the primary entry point for
running batch inference with a trained segmentation model. It handles model
loading, checkpoint restoration, dataset construction, and output writing.

Output formats
--------------
- GeoTIFF - georeferenced prediction raster aligned to the source mosaic.
- PNG (optional) - RGB visualisation with a random but reproducible colour
  per class, useful for quick visual inspection.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swmaps.models.dataset import (
    SegDataset,
    mission_from_path,
    satellite_id_from_mission,
)
from swmaps.models.farseg import FarSegModel
from swmaps.models.model_factory import MODEL_REGISTRY, get_model


def _write_mask_like(ref_raster, mask_tensor, out_path):
    """Write a predicted mask as a GeoTIFF matching a reference raster.

    Copies the CRS, transform, and spatial extent from *ref_raster* so the
    output is correctly georeferenced.

    Args:
        ref_raster: Path to the source mosaic GeoTIFF used as a spatial
            reference.
        mask_tensor: 2-D integer tensor of shape ``(H, W)`` containing
            predicted class indices.
        out_path: Destination path for the output GeoTIFF.

    Returns:
        None
    """
    import rasterio

    with rasterio.open(ref_raster) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_tensor.numpy().astype("uint8"), 1)


def _write_png(mask_tensor, out_path):
    """Write a predicted mask as an RGB PNG with per-class colours.

    Each unique class value is assigned a random colour using a fixed seed
    (42) for reproducibility across runs.

    Args:
        mask_tensor: 2-D integer tensor of shape ``(H, W)`` containing
            predicted class indices.
        out_path: Destination path for the output PNG.

    Returns:
        None
    """
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


def run_segmentation(
    mosaics,
    out_dir,
    model_name="farseg",
    batch_size=4,
    save_png=False,
    weights_path=None,
):
    """Run batch segmentation inference on a list of mosaic GeoTIFFs.

    Loads a model from the registry, optionally restores weights from a
    checkpoint, runs inference on each mosaic, and writes georeferenced
    prediction rasters alongside the source files.

    Args:
        mosaics: List of paths to multiband GeoTIFF mosaics.
        out_dir: Directory where prediction rasters will be written.
        model_name: Model architecture name - must be a key in
            :data:`~swmaps.models.model_factory.MODEL_REGISTRY`.
            Defaults to ``"farseg"``.
        batch_size: Number of mosaics to process per forward pass.
            Defaults to ``4``.
        save_png: Whether to write an RGB PNG preview alongside each
            GeoTIFF prediction. Defaults to ``False``.
        weights_path: Optional path to a ``.pth`` checkpoint file. When
            provided the model type and kwargs are read from the checkpoint
            metadata and the model is rebuilt before loading weights.

    Returns:
        Path: The *out_dir* path after inference completes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the Model via Factory
    if model_name == "farseg":
        model = FarSegModel()
    else:
        model = get_model(model_name)

    # 2. Load Weights if provided
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)

        model_type = ckpt["model_type"]
        model_kwargs = ckpt["model_kwargs"]

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        model_cls = MODEL_REGISTRY[model_type]
        model = model_cls(**model_kwargs)

        model.load_state_dict(ckpt["state_dict"], strict=False)

    model.to(device)
    model.eval()

    # 3. Use the existing Dataset loader
    samples = [
        (p, None, satellite_id_from_mission(mission_from_path(p))) for p in mosaics
    ]
    dataset = SegDataset(samples, inference_only=True)
    loader = DataLoader(dataset, batch_size=batch_size)

    # 4. Run Inference using the model's internal predict()
    preds = []
    with torch.no_grad():
        for imgs, _, sat_ids in loader:
            imgs = imgs.to(device)
            sat_ids = sat_ids.to(device)
            pred = model.predict(imgs, sat_ids)
            preds.extend(pred.cpu())

    # 5. Output management (Geotiff/PNG)
    for mosaic_path, mask in zip(mosaics, preds):
        stem = Path(mosaic_path).stem
        tif_path = out_dir / f"{stem}_segmentation.tif"
        _write_mask_like(mosaic_path, mask, tif_path)

        if save_png:
            _write_png(mask, out_dir / f"{stem}_segmentation.png")

    return out_dir
