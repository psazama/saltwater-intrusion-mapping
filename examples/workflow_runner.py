"""Workflow runner for the saltwater intrusion mapping pipeline.

This script is the primary CLI entry point. It loads a TOML config file,
constructs a typed :class:`~swmaps.schema.WorkflowConfig`, and calls each
enabled pipeline stage in order.

No model classes are imported here - all model access is encapsulated
inside the pipeline layer.

Usage::

    python examples/workflow_runner.py --config examples/quickstart_train.toml
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tomllib
from pathlib import Path

from swmaps.datasets.cdl import run_cdl_download
from swmaps.models.dataset import mission_from_path, satellite_id_from_mission
from swmaps.models.inference import run_segmentation
from swmaps.models.model_factory import get_model
from swmaps.pipeline.download import run_download
from swmaps.pipeline.masks import run_water_masks
from swmaps.pipeline.salinity import run_salinity_classification, run_salinity_pipeline
from swmaps.pipeline.trend import run_trend_heatmap
from swmaps.schema import DownloadConfig, WorkflowConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _log_result(stage: str, result) -> None:
    """Log a PipelineResult at INFO or WARNING level.

    Args:
        stage: Name of the pipeline stage for the log prefix.
        result: A :class:`~swmaps.schema.PipelineResult` instance.
    """
    if result.is_ok:
        logger.info("[%s] ok - %d files written", stage, len(result.output_paths))
    elif result.status == "skipped":
        logger.info("[%s] skipped - %s", stage, result.error or "")
    else:
        logger.warning("[%s] error - %s", stage, result.error)


def main() -> None:
    parser = argparse.ArgumentParser(description="swmaps workflow runner")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument(
        "--loss_function",
        default=None,
        help="Override loss function (e.g. 'dice', 'focal', 'ce', 'hybrid')",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        raw = tomllib.load(f)

    if args.loss_function:
        raw["loss"] = args.loss_function

    cfg = WorkflowConfig.from_dict(raw)
    dl = cfg.download
    seg = cfg.segmentation
    sal = cfg.salinity
    trend = cfg.trend

    base_out_dir = Path(dl.out_dir) if dl.out_dir else Path("data/outputs").resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe output directories if configured
    if raw.get("wipe_data_dir", False):
        for dir_str in ["out_dir", "val_dir", "val_cdl_out"]:
            path = raw.get(dir_str)
            if path:
                path = Path(path)
                if path.exists() and path.is_dir():
                    logger.info("Clearing existing data in: %s", path)
                    shutil.rmtree(path)

    # ------------------------------------------------------------------
    # 1. Download imagery
    # ------------------------------------------------------------------
    dl_result = run_download(dl)
    _log_result("download", dl_result)

    if seg.do_val and seg.val_start_date:
        val_cfg = DownloadConfig(
            start_date=seg.val_start_date,
            end_date=seg.val_end_date,
            mission=dl.mission,
            geometry=seg.val_region,
            out_dir=seg.val_dir,
            buffer_km=dl.buffer_km,
            cloud_filter=dl.cloud_filter,
            days_before=dl.days_before,
            days_after=dl.days_after,
            samples_per_date=dl.samples_per_date,
            save_png=dl.save_png,
        )
        val_dl_result = run_download(val_cfg)
        _log_result("download_val", val_dl_result)

    cdl_result = run_cdl_download(dl)
    _log_result("cdl", cdl_result)

    # ------------------------------------------------------------------
    # 2. Segmentation training
    # ------------------------------------------------------------------
    if seg.train_segmentation:

        # Clear existing segmentation dirs if present
        for dir_str in ["segmentation_model_dir", "segmentation_out_dir"]:
            path = raw.get(dir_str)
            if path:
                path = Path(path)
                if path.exists() and path.is_dir():
                    logger.info("Clearing existing data in: %s", path)
                    shutil.rmtree(path)

        model = get_model(
            seg.model_type,
            num_classes=seg.segmentation_num_classes,
            segmentation_num_classes=seg.segmentation_num_classes,
        )

        all_mosaics = sorted(base_out_dir.rglob("*_multiband.tif"))
        training_pairs = []
        val_pairs = []

        for mosaic in all_mosaics:
            mosaic_str = str(mosaic)
            if any(
                x in mosaic_str for x in ["validation", "val", "aligned_cdl", "mask"]
            ):
                continue
            label_path = mosaic.with_name(f"aligned_cdl_{mosaic.name}")
            if not label_path.exists():
                continue
            sat_id = satellite_id_from_mission(mission_from_path(mosaic))
            training_pairs.append((mosaic, label_path, sat_id))

        if seg.do_val:
            val_dir = Path(seg.val_dir) if seg.val_dir else base_out_dir / "validation"
            for mosaic in sorted(val_dir.rglob("*_multiband.tif")):
                if any(x in str(mosaic) for x in ["aligned_cdl", "mask"]):
                    continue
                label_path = mosaic.with_name(f"aligned_cdl_{mosaic.name}")
                if label_path.exists():
                    sat_id = satellite_id_from_mission(mission_from_path(mosaic))
                    val_pairs.append((mosaic, label_path, sat_id))

        if training_pairs:
            seg_model_dir = Path(
                seg.segmentation_model_dir or str(base_out_dir / "model")
            )
            seg_model_dir.mkdir(parents=True, exist_ok=True)
            model.train_model(
                data_pairs=training_pairs,
                out_dir=seg_model_dir,
                val_pairs=val_pairs or None,
                epochs=seg.epochs,
                batch_size=seg.batch_size,
                lr=seg.learning_rate,
                loss_type=seg.loss,
            )
            logger.info("[train] model saved to %s", seg_model_dir)
        else:
            logger.warning("[train] no training pairs found - skipping.")

    # ------------------------------------------------------------------
    # 3. Segmentation inference
    # ------------------------------------------------------------------
    if seg.run_segmentation:
        mosaics = [
            m
            for m in sorted(base_out_dir.rglob("*_multiband.tif"))
            if "aligned_cdl" not in str(m)
        ]
        if mosaics:
            seg_out = Path(
                seg.segmentation_out_dir or str(base_out_dir / "segmentation")
            )
            seg_out.mkdir(parents=True, exist_ok=True)
            run_segmentation(
                mosaics=mosaics,
                out_dir=seg_out,
                model_name=seg.model_type,
                weights_path=seg.segmentation_weights_path,
                save_png=seg.segmentation_png,
            )
            logger.info("[inference] outputs written to %s", seg_out)
        else:
            logger.warning("[inference] no mosaics found - skipping.")

    # ------------------------------------------------------------------
    # 4. Salinity ground-truth pipeline
    # ------------------------------------------------------------------
    sal_pipeline_result = run_salinity_pipeline(sal)
    _log_result("salinity_pipeline", sal_pipeline_result)

    # ------------------------------------------------------------------
    # 5. Per-mosaic salinity classification
    # ------------------------------------------------------------------
    sal_class_result = run_salinity_classification(sal, base_out_dir)
    _log_result("salinity_classification", sal_class_result)

    # ------------------------------------------------------------------
    # 6. Water masks + trend heatmap
    # ------------------------------------------------------------------
    if trend.run_water_trend:
        mask_result = run_water_masks(base_out_dir)
        _log_result("water_masks", mask_result)

    trend_result = run_trend_heatmap(trend, output_dir=base_out_dir)
    _log_result("trend_heatmap", trend_result)

    logger.info("Workflow complete.")


if __name__ == "__main__":
    main()
