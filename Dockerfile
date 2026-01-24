# 1. Use a GPU-ready base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 2. Install Miniconda & System dependencies
RUN apt-get update && apt-get install -y \
    wget git libgdal-dev libgeos-dev g++ python3-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# 3. Setup Working Directory
WORKDIR /app

# 4. Create your environment
# Copy ONLY the environment file first to leverage Docker caching
COPY environment.yml .

# Accept ToS AND create the environment in ONE step
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml && \
    conda clean -afy

# 5. Copy the rest of your code
# (We do this AFTER the env is built so code changes don't trigger a full reinstall)
COPY . .

# 6. Set the entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "swmaps_env", "python", "examples/workflow_runner.py"]