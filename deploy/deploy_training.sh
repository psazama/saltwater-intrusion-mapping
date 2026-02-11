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

echo "Step 3: Generating temporary YAML configuration..."
CONFIG_FILE=$(mktemp /tmp/vertex-config-XXXXXX.yaml)
trap 'rm -f "$CONFIG_FILE"; echo "Cleanup: Sensitive config deleted."' EXIT
cat <<EOF > "$CONFIG_FILE"
workerPoolSpecs:
  machineSpec:
    machineType: a2-highgpu-1g
    acceleratorType: NVIDIA_TESLA_A100
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: $IMAGE_URI
    command:
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
  restartJobOnWorkerRestart: false
  strategy: SPOT
EOF

echo "Step 4: Submitting Job to Vertex AI..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="farseg-$(date +%Y%m%d-%H%M%S)" \
    --config="$CONFIG_FILE" \
    --service-account="$SVC_ACCOUNT"
