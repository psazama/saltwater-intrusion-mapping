#!/bin/bash
set -e

# --- Argument Parsing ---
EE_PROJECT=""
usage() { echo "Usage: $0 -p <earthengine_project_id>"; exit 1; }

while getopts "p:" opt; do
    case "$opt" in
        p) EE_PROJECT=$OPTARG ;;
        *) usage ;;
    esac
done

if [ -z "$EE_PROJECT" ]; then usage; fi

# --- Main Script ---
cd "$(dirname "$0")/.."

PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='get(projectNumber)')
REGION="us-central1"
REPO_NAME="swmaps-repo"
IMAGE_NAME="farseg-train"
TAG="v1"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"

echo "Step 0: Authorizing Service Account..."
SVC_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

echo "Step 1: Ensuring Artifact Registry exists..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker --location=$REGION || true

echo "Step 2: Building Image..."
gcloud builds submit --tag $IMAGE_URI .

echo "Step 3: Generating Hyperparameter Tuning Config..."
CONFIG_FILE=$(mktemp /tmp/vertex-hpt-XXXXXX.yaml)
trap 'rm -f "$CONFIG_FILE"; echo "Cleanup: Sensitive config deleted."' EXIT

cat <<EOF > "$CONFIG_FILE"
# This tells Vertex AI what we want to test
studySpec:
  metrics:
    - metricId: val_miou
      goal: MAXIMIZE
  parameters:
    - parameterId: loss_function
      categoricalValueSpec:
        values: ["ce", "focal", "dice", "hybrid"]
  algorithm: GRID_SEARCH  # Runs every value in the list exactly once

# This tells Vertex AI how to run it
trialJobSpec:
  workerPoolSpecs:
    machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: $IMAGE_URI
      # IMPORTANT: Use args instead of command to let Vertex pass the --loss_function flag
      args:
        - conda
        - run
        - --no-capture-output
        - -p
        - /opt/conda/envs/swmaps_env
        - python
        - examples/workflow_runner.py
        - --config=examples/cdl_farseg_finetune_cloud.toml
      env:
        - name: EARTHENGINE_PROJECT
          value: $EE_PROJECT
  scheduling:
    timeout: 604800s
    strategy: SPOT
EOF

# --- Step 4: Submitting HP Tuning Job ---
echo "Step 4: Submitting Hyperparameter Tuning Job (Sequential Queueing)..."
gcloud ai hp-tuning-jobs create \
    --region=$REGION \
    --display-name="farseg-multi-loss-$(date +%Y%m%d)" \
    --config="$CONFIG_FILE" \
    --max-trial-count=4 \
    --parallel-trial-count=1 \
    --service-account="$SVC_ACCOUNT"