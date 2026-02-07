#!/usr/bin/env python3

"""
Workflow runner for the saltwater intrusion mapping pipeline (GEE version).
Updated: Site-specific subdirectories (1, 2, 3...) and global validation.
"""

import argparse
import logging
import shutil
import tomllib
from pathlib import Path

from swmaps.config import data_path
from swmaps.datasets.cdl import CDL_TO_BINARY_CLASS
from swmaps.models.inference import run_segmentation
from swmaps.models.salinity_heuristic import SalinityHeuristicModel
from swmaps.pipeline.download import download_cdl, download_data
from swmaps.pipeline.masks import generate_masks
from swmaps.pipeline.salinity import salinity_pipeline
from swmaps.pipeline.trend import trend_heatmap

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def ensure_list(val):
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEE Saltwater Intrusion Pipeline")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--loss_function", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.loss_function:
        cfg["loss_function"] = args.loss_function
        # Highly recommended: update the output dir so they don't overwrite each other
        cfg["segmentation_model_dir"] = (
            f"{cfg['segmentation_model_dir']}_{args.loss_function}"
        )

    for dir_str in [
        "out_dir",
        "segmentation_model_dir",
        "val_dir",
        "val_cdl_out",
        "segmentation_weights_path",
        "segmentation_out_dir",
    ]:
        path = Path(cfg.get(dir_str, None))
        if path.exists() and path.is_dir():
            logging.info(f"Clearing existing data in: {path}")
            shutil.rmtree(path)

    base_out_dir = Path(cfg.get("out_dir", data_path("mosaics")))
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Step 2 — Download imagery and datasets
    # -----------------------------------------------------------
    if cfg.get("skip_download", False):
        logging.info("skip_download is True: Skipping GEE download.")
    else:
        # 1. Training Sites (Numbered Subdirectories)
        lats = ensure_list(cfg.get("latitude"))
        lons = ensure_list(cfg.get("longitude"))

        if lats and lons:
            for i, (lat, lon) in enumerate(zip(lats, lons), start=1):
                site_dir = base_out_dir / str(i)
                site_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"--- Downloading Training Site {i}: {lat}, {lon} ---")

                p_cfg = cfg.copy()
                p_cfg.update(
                    {"latitude": lat, "longitude": lon, "out_dir": str(site_dir)}
                )
                download_data(p_cfg)

                if cfg.get("download_cdl", False):
                    # Save CDL specifically as cdl_site_X.tif for pairing logic
                    p_cfg["cdl_out"] = str(site_dir / f"cdl_site_{i}.tif")
                    download_cdl(p_cfg)

        # 2. Global Validation Site (Run Once)
        if cfg.get("do_val", False):
            val_dir = Path(cfg.get("val_dir", base_out_dir / "validation"))
            val_dir.mkdir(parents=True, exist_ok=True)
            logging.info("--- Downloading Global Validation Site ---")

            v_cfg = cfg.copy()
            v_cfg.update(
                {
                    "latitude": cfg.get("val_latitude"),
                    "longitude": cfg.get("val_longitude"),
                    "val_dir": str(val_dir),
                    "val_cdl_out": str(val_dir / "val_cdl.tif"),
                }
            )
            download_data(v_cfg, val=True)
            if cfg.get("download_cdl", False):
                download_cdl(v_cfg, val=True)

    # -----------------------------------------------------------
    # Step 2.5 — Segmentation (Training & Label Alignment)
    # -----------------------------------------------------------
    if cfg.get("train_segmentation", False):
        logging.info("Preparing CDL labels and training FarSeg")
        from swmaps.datasets.cdl import align_cdl_to_imagery

        model_type = cfg.get("model_type", "farseg").lower()
        if model_type == "farseg":
            from swmaps.models.farseg import FarSegModel

            ModelClass = FarSegModel
        # elif model_type == "unet":
        #     from swmaps.models.unet import UNetModel
        #     ModelClass = UNetModel
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # 1. Collect Training Pairs (Skip 'validation' folder)
        training_pairs = []
        all_mosaics = sorted(base_out_dir.rglob("*_multiband.tif"))

        logging.info(f"Training segmentation model with {len(all_mosaics)} mosaics")
        for mosaic in all_mosaics:
            mosaic_str = str(mosaic)
            if any(
                x in mosaic_str for x in ["validation", "val", "aligned_cdl", "mask"]
            ):
                continue

            # Match imagery with the CDL in its specific subdirectory
            site_cdls = list(mosaic.parents[1].glob("cdl_site_*.tif"))
            if site_cdls:
                label_path = mosaic.with_name(f"aligned_cdl_{mosaic.name}")
                if not label_path.exists():
                    align_cdl_to_imagery(site_cdls[0], mosaic, label_path)
                training_pairs.append((mosaic, label_path))
            else:
                logging.warning(f"No CDL label found in folder: {mosaic.parent}")

        # 2. Collect Validation Pairs (From global validation folder)
        validation_pairs = []
        if cfg.get("do_val", False):
            val_root = Path(cfg.get("val_dir", base_out_dir / "validation"))
            val_mosaics = sorted(val_root.rglob("*_multiband.tif"))
            val_cdl_file = val_root / "val_cdl.tif"

            for v_mosaic in val_mosaics:
                mosaic_str = str(v_mosaic)
                if any(x in mosaic_str for x in ["aligned_cdl", "mask"]):
                    continue
                if val_cdl_file.exists():
                    v_label = v_mosaic.with_name(f"aligned_cdl_{v_mosaic.name}")
                    if not v_label.exists():
                        align_cdl_to_imagery(val_cdl_file, v_mosaic, v_label)
                    validation_pairs.append((v_mosaic, v_label))
                else:
                    logging.warning(f"No CDL label found in folder: {v_mosaic.parent}")

        if training_pairs:
            logging.info(
                f"Training segmentation model with {len(training_pairs)} pairs"
            )
            seg_model_dir = Path(cfg["segmentation_model_dir"])
            seg_model_dir.mkdir(parents=True, exist_ok=True)

            model_instance = ModelClass(
                num_classes=cfg.get("segmentation_num_classes", 256),
                in_channels=cfg.get("in_channels", 6),
            )

            model_instance.train_model(
                data_pairs=training_pairs,
                out_dir=seg_model_dir,
                label_map=CDL_TO_BINARY_CLASS,
                val_pairs=validation_pairs if validation_pairs else None,
                epochs=cfg.get("epochs", 50),
                lr=cfg.get("lr", 1e-4),
                batch_size=cfg.get("batch_size", 4),
                loss_type=cfg.get("loss_function", "ce"),
                lr_patience=cfg.get("lr_patience", 5),
                stopping_patience=cfg.get("stopping_patience", 15),
            )

        if validation_pairs:
            logging.info("Running inference on validation set for visual inspection...")
            val_out = seg_model_dir / "val_predictions"
            val_out.mkdir(parents=True, exist_ok=True)

            # Extract only the mosaic paths from the pairs
            val_mosaics_only = [p[0] for p in validation_pairs]

            run_segmentation(
                mosaics=val_mosaics_only,
                out_dir=val_out,
                model_name="farseg",
                weights_path=str(seg_model_dir / "best_model.pth"),
                save_png=True,  # Helpful for visual debugging
            )

    # -----------------------------------------------------------
    # Inference & Post-Processing
    # -----------------------------------------------------------
    if cfg.get("run_segmentation", False):
        mosaics = [
            m
            for m in sorted(base_out_dir.rglob("*_multiband.tif"))
            if "aligned_cdl" not in str(m)
        ]
        if mosaics:
            seg_out = Path(cfg["segmentation_out_dir"])
            seg_out.mkdir(parents=True, exist_ok=True)
            run_segmentation(
                mosaics=mosaics,
                out_dir=seg_out,
                model_name=cfg.get("segmentation_model", "farseg"),
                weights_path=cfg.get("segmentation_weights_path"),
                save_png=bool(cfg.get("segmentation_png", False)),
            )

    if cfg.get("run_salinity_pipeline", False):
        salinity_pipeline(
            truth_download_list=cfg.get("truth_download_list"),
            truth_dir=cfg.get("truth_dir"),
            truth_file=cfg.get("truth_file"),
        )

    if cfg.get("run_salinity_classification", False):
        for mosaic_path in sorted(base_out_dir.rglob("*.tif")):
            if any(
                x in mosaic_path.name for x in ["_score", "_class", "_mask", "aligned"]
            ):
                continue
            if "landsat" in mosaic_path.name.lower():
                SalinityHeuristicModel().estimate_salinity_from_mosaic(
                    mosaic_path=mosaic_path,
                    water_threshold=cfg.get("water_threshold", 0.2),
                )

    if cfg.get("run_water_trend", False):
        generate_masks(input_dir=base_out_dir)
        trend_heatmap(output_dir=cfg.get("trend_output_dir", base_out_dir))

    logging.info("Workflow complete!")


if __name__ == "__main__":
    main()
