# ./Dockerfile  (project root)
FROM python:3.11-slim

# 1. System deps (GDAL, proj, etc.)
RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy project + install
WORKDIR /app
COPY pyproject.toml .
COPY swmaps/ ./swmaps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 3. Dagster entry-point (for gRPC server)
EXPOSE 4000
ENTRYPOINT ["dagster", "api", "grpc", "-m", "swmaps.gke_defs"]

