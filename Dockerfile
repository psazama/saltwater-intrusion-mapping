FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    tini \
    wget \
    git \
    libgdal-dev \
    libgeos-dev \
    g++ \
    python3-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /app

# 3. Create the Conda environment
COPY environment.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -p /opt/conda/envs/swmaps_env -f environment.yml && \
    conda clean -afy

# Put the env on PATH so plain `python` / `uvicorn` resolve for CMD overrides
ENV PATH="/opt/conda/envs/swmaps_env/bin:${PATH}"

# 4. Copy project files
COPY . .

# 5. Install the package in non-editable mode
RUN pip install .

# 6. Entrypoint is just tini; the default command can be overridden by
#    `docker run ... <cmd>` or a compose `command:` without being appended
#    to workflow_runner.py.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "examples/workflow_runner.py"]
