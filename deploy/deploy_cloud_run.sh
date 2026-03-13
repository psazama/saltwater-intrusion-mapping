#!/bin/bash
set -e

# --- Argument Parsing ---
EE_PROJECT=""
TASK=""
SCENE_IDS_FILE=""

usage() { echo "Usage: $0 -p <earthengine_project_id> -t <task> -s <scene_ids_json>"; exit 1; }

while getopts "p:t:s:" opt; do
    case "$opt" in
        p) EE_PROJECT=$OPTARG ;;
        t) TASK=$OPTARG ;;
        s) SCENE_IDS_FILE=$OPTARG ;;
        *) usage ;;
    esac
done

if [ -z "$EE_PROJECT" ] || [ -z "$TASK" ]; then usage; fi

# --- Main Script ---
cd "$(dirname "$0")/.."

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="swmaps-repo"
IMAGE_NAME="swmaps-pipeline"
TAG="v1"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"

echo "Step 1: Ensuring Artifact Registry exists..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker --location=$REGION || true

echo "Step 2: Building and pushing image..."
gcloud builds submit --tag $IMAGE_URI .

echo "Step 3: Deploying to Cloud Run..."
gcloud run jobs create swmaps-${TASK}-$(date +%Y%m%d-%H%M%S) \
    --image=$IMAGE_URI \
    --region=$REGION \
    --set-env-vars="EARTHENGINE_PROJECT=$EE_PROJECT,TASK=$TASK" \
    --set-secrets="DB_PASSWORD=swmaps-db-password:latest" \
    --memory=4Gi \
    --cpu=2 \
    --max-retries=1 \
    --execute-now