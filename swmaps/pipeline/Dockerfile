FROM ubuntu:22.04

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    tini \
    wget \
    ca-certificates \
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
COPY swmaps/pipeline/environment.pipeline.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -p /opt/conda/envs/swmaps_pipeline_env -f environment.pipeline.yml && \
    conda clean -afy

# 4. Copy project files
COPY . .

# 5. Install the package in non-editable mode
RUN /opt/conda/envs/swmaps_pipeline_env/bin/pip install .

# 6. Set the Entrypoint
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/conda/envs/swmaps_pipeline_env/bin/python", "swmaps/pipeline/run_pipeline.py"]