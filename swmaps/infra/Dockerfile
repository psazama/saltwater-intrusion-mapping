FROM ubuntu:22.04

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    tini \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3-pip \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Flask
RUN pip3 install flask pydantic google-cloud-run tomli gunicorn

# 3. Copy project files
COPY . .

# 4. Set the Entrypoint
ENTRYPOINT ["/usr/bin/tini", "--", "gunicorn", "--bind", "0.0.0.0:8080", "swmaps.infra.trigger:app"]