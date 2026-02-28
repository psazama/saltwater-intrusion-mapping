from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from output_index import Scene, index_run_outputs


def _pick_run_root(repo_root: Path) -> Optional[Path]:
    outputs_root = repo_root / "data" / "outputs"
    if not outputs_root.exists():
        st.warning(f"Outputs root not found: {outputs_root}")
        return None

    runs = sorted([p for p in outputs_root.iterdir() if p.is_dir()])
    if not runs:
        st.warning(f"No run directories found under {outputs_root}")
        return None

    return st.selectbox("Outputs run", runs, format_func=lambda p: p.name)


def render_outputs_viewer(repo_root: Path) -> None:
    st.subheader("Outputs viewer (raw vs processed)")

    run_root = _pick_run_root(repo_root)
    if run_root is None:
        return

    scenes = index_run_outputs(run_root)
    if not scenes:
        st.info("No imagery scenes found under this run.")
        return

    sites = sorted({s.site for s in scenes})
    missions = sorted({s.mission for s in scenes})

    c1, c2 = st.columns(2)
    with c1:
        site = st.selectbox("Site", sites)
    with c2:
        mission = st.selectbox("Mission", missions)

    filtered = [s for s in scenes if s.site == site and s.mission == mission]
    if not filtered:
        st.info("No scenes for that selection.")
        return

    labels = [s.scene_id for s in filtered]
    idx = st.selectbox("Scene", range(len(filtered)), format_func=lambda i: labels[i])
    scene: Scene = filtered[idx]

    st.markdown(f"### `{scene.scene_id}`")

    left, right = st.columns(2)

    with left:
        st.markdown("**Raw imagery**")
        if scene.imagery_png:
            st.image(
                str(scene.imagery_png),
                caption=str(scene.imagery_png.relative_to(repo_root)),
            )
        elif scene.imagery_tif:
            st.write(str(scene.imagery_tif.relative_to(repo_root)))
            st.download_button(
                "Download imagery GeoTIFF",
                data=scene.imagery_tif.read_bytes(),
                file_name=scene.imagery_tif.name,
            )
        else:
            st.write("No imagery file found.")

    with right:
        st.markdown("**Processed: segmentation**")
        if scene.seg_png:
            st.image(
                str(scene.seg_png), caption=str(scene.seg_png.relative_to(repo_root))
            )
        elif scene.seg_tif:
            st.write(str(scene.seg_tif.relative_to(repo_root)))
            st.download_button(
                "Download segmentation GeoTIFF",
                data=scene.seg_tif.read_bytes(),
                file_name=scene.seg_tif.name,
            )
        else:
            st.write("No segmentation output found for this scene.")
