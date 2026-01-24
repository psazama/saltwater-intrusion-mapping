#!/bin/bash

# Exit on any error
set -e

# Navigate to the PROJECT ROOT (one level up from /deploy)
cd "$(dirname "$0")/.."

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="swmaps-repo"
IMAGE_NAME="farseg-train"
TAG="v1"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"

echo "Step 1: Ensuring Artifact Registry exists..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for FarSeg training" || true

echo "Step 2: Building Image via Cloud Build: $IMAGE_URI..."
gcloud builds submit --tag $IMAGE_URI .

echo "Step 3: Submitting Job to Vertex AI..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="farseg-$(date +%Y%m%d-%H%M%S)" \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=$IMAGE_URI \
    --args="--config=examples/cdl_farseg_finetune_cloud.toml"

echo "------------------------------------------------"
echo "Job Submitted! Check progress here:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"