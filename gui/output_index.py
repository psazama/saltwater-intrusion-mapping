from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg"}
TIFF_EXTS = {".tif", ".tiff"}


def _is_imagery_file(p: Path) -> bool:
    n = p.name.lower()
    return "_multiband." in n and "_multiband_segmentation" not in n


def _is_segmentation_file(p: Path) -> bool:
    return "_multiband_segmentation." in p.name.lower()


def _scene_id_from_imagery(p: Path) -> str:
    return p.name.rsplit("_multiband", 1)[0]


def _scene_id_from_segmentation(p: Path) -> str:
    return p.name.rsplit("_multiband_segmentation", 1)[0]


@dataclass(frozen=True)
class Scene:
    root: Path
    run: str
    site: str
    mission: str
    scene_id: str
    imagery_png: Optional[Path]
    imagery_tif: Optional[Path]
    seg_png: Optional[Path]
    seg_tif: Optional[Path]


def index_run_outputs(run_root: Path) -> List[Scene]:
    if not run_root.exists():
        return []

    seg_by_scene: Dict[str, Dict[str, Path]] = {}
    for p in run_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (IMG_EXTS | TIFF_EXTS):
            continue
        if not _is_segmentation_file(p):
            continue

        scene_id = _scene_id_from_segmentation(p)
        entry = seg_by_scene.setdefault(scene_id, {})
        if p.suffix.lower() in IMG_EXTS:
            entry["png"] = p
        else:
            entry["tif"] = p

    scenes: List[Scene] = []
    for p in run_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (IMG_EXTS | TIFF_EXTS):
            continue
        if not _is_imagery_file(p):
            continue

        rel = p.relative_to(run_root)
        if len(rel.parts) < 3:
            continue

        site = rel.parts[0]
        mission = rel.parts[1]
        scene_id = _scene_id_from_imagery(p)

        imagery_png = (
            p
            if p.suffix.lower() in IMG_EXTS
            else p.with_name(f"{scene_id}_multiband.png")
        )
        if not imagery_png.exists():
            imagery_png = None

        imagery_tif = (
            p
            if p.suffix.lower() in TIFF_EXTS
            else p.with_name(f"{scene_id}_multiband.tif")
        )
        if not imagery_tif.exists():
            imagery_tif = None

        seg = seg_by_scene.get(scene_id, {})
        scenes.append(
            Scene(
                root=run_root,
                run=run_root.name,
                site=site,
                mission=mission,
                scene_id=scene_id,
                imagery_png=imagery_png,
                imagery_tif=imagery_tif,
                seg_png=seg.get("png"),
                seg_tif=seg.get("tif"),
            )
        )

    scenes.sort(key=lambda s: (s.site, s.mission, s.scene_id))
    return scenes
