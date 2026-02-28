from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from view_outputs import render_outputs_viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
TEMP_DIR = REPO_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def list_example_tomls() -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    # change to rglob("*.toml") if you want nested configs
    return sorted(EXAMPLES_DIR.glob("*.toml"))


def save_uploaded_toml(upload) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    dst = TEMP_DIR / f"{ts}-{upload.name}"
    dst.write_bytes(upload.getvalue())
    return dst


def run_and_stream(config_path: Path) -> int:
    cmd = [sys.executable, "examples/workflow_runner.py", "--config", str(config_path)]

    latest = st.empty()  # most recent log message
    log_box = st.empty()  # scrolling log tail
    logs: list[str] = []

    latest.info("Starting workflow...")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        logs.append(line)
        latest.info(line)
        log_box.code("\n".join(logs[-400:]))

    return proc.wait()


st.set_page_config(page_title="Saltwater Intrusion Mapping", layout="wide")
st.title("Saltwater Intrusion Mapping")

tomls = list_example_tomls()
labels = [p.name for p in tomls]

source = st.radio("Config source", ["examples/", "upload"], horizontal=True)

config_path: Path | None = None

if source == "examples/":
    if not tomls:
        st.error("No TOML files found in examples/.")
    else:
        selected = st.selectbox("Select a TOML file:", labels)
        config_path = EXAMPLES_DIR / selected
        with st.expander("Preview TOML", expanded=False):
            st.code(config_path.read_text(), language="toml")
else:
    uploaded = st.file_uploader("Upload a TOML config", type="toml")
    if uploaded is not None:
        config_path = save_uploaded_toml(uploaded)
        st.caption(f"Saved to: {config_path.relative_to(REPO_ROOT)}")

if st.button("Run Workflow", type="primary", disabled=(config_path is None)):
    assert config_path is not None
    with st.spinner("Running the workflow... (latest log line + full log below)"):
        rc = run_and_stream(config_path)

    if rc == 0:
        st.success("Workflow executed successfully!")
    else:
        st.error(f"Workflow failed (exit code {rc}).")

st.divider()
render_outputs_viewer(REPO_ROOT)
