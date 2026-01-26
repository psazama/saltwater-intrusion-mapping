FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget git libgdal-dev libgeos-dev g++ python3-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /app

COPY environment.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -p /opt/conda/envs/swmaps_env -f environment.yml && \
    conda clean -afy


COPY . .

RUN conda run --no-capture-output -p /opt/conda/envs/swmaps_env pip install -e .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-p", "/opt/conda/envs/swmaps_env", "python", "examples/workflow_runner.py"]