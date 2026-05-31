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
CLOUD_SQL_INSTANCE="ml-dev-env-460700:us-central1:swmaps-db"

echo "Step 0: Authorizing Service Account..."
SVC_ACCOUNT="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SVC_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

echo "Step 1: Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    pubsub.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com

echo "Step 2: Ensuring Artifact Registry exists..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker --location=$REGION 2>/dev/null || true

PIPELINE_IMAGE_NAME="swmaps-pipeline"
PIPELINE_TAG="v5"
PIPELINE_IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$PIPELINE_IMAGE_NAME:$PIPELINE_TAG"

TRIGGER_IMAGE_NAME="swmaps-trigger"
TRIGGER_TAG="v5"
TRIGGER_IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$TRIGGER_IMAGE_NAME:$TRIGGER_TAG"

TRIGGER_SERVICE="swmaps-trigger"
PIPELINE_JOB="swmaps-pipeline"

API_IMAGE_NAME="swmaps-api"
API_TAG="v1"
API_IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$API_IMAGE_NAME:$API_TAG"
API_SERVICE="swmaps-api"

echo "Step 3: Building frontend..."
cd frontend
if ! command -v npm &>/dev/null; then
    echo "ERROR: npm not found. Install Node.js 18+ before deploying."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'.' -f1 | tr -d 'v')
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "ERROR: Node.js 18+ required, found $(node --version)."
    echo "Run: nvm install 18 && nvm use 18"
    exit 1
fi

npm install --silent
npm run build
cd ..
echo "Frontend built successfully."

echo "Step 4: Building Image..."
gcloud builds submit \
    --config=deploy/cloudbuild.pipeline.yaml \
    --substitutions="_IMAGE_URI=$PIPELINE_IMAGE_URI" .

gcloud builds submit \
    --config=deploy/cloudbuild.trigger.yaml \
    --substitutions="_IMAGE_URI=$TRIGGER_IMAGE_URI" .

gcloud builds submit \
    --config=deploy/cloudbuild.trigger.yaml \
    --substitutions="_IMAGE_URI=$API_IMAGE_URI" .

echo "Step 5: Deploying to Cloud Run..."
if gcloud run jobs describe $PIPELINE_JOB --region=$REGION &>/dev/null; then
    gcloud run jobs update $PIPELINE_JOB \
        --image=$PIPELINE_IMAGE_URI \
        --region=$REGION \
        --set-env-vars="EARTHENGINE_PROJECT=$EE_PROJECT,DB_HOST=/cloudsql/$CLOUD_SQL_INSTANCE,DB_PORT=5432,DB_NAME=swmaps,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,REGION=$REGION" \
        --add-cloudsql-instances=$CLOUD_SQL_INSTANCE
else
    gcloud run jobs create $PIPELINE_JOB \
        --image=$PIPELINE_IMAGE_URI \
        --region=$REGION \
        --set-env-vars="EARTHENGINE_PROJECT=$EE_PROJECT,DB_HOST=/cloudsql/$CLOUD_SQL_INSTANCE,DB_PORT=5432,DB_NAME=swmaps,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,REGION=$REGION" \
        --set-secrets="DB_PASSWORD=swmaps-db-password:latest,DB_USER=swmaps-db-user:latest" \
        --memory=4Gi \
        --cpu=2 \
        --max-retries=1 \
        --add-cloudsql-instances=$CLOUD_SQL_INSTANCE
fi

gcloud run deploy $TRIGGER_SERVICE \
    --image=$TRIGGER_IMAGE_URI \
    --region=$REGION \
    --no-allow-unauthenticated \
    --set-env-vars="PIPELINE_JOB=swmaps-pipeline,REGION=$REGION,GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --service-account=$SVC_ACCOUNT

echo "Deploying API service..."
gcloud run deploy $API_SERVICE \
    --image=$API_IMAGE_URI \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars="DB_HOST=/cloudsql/$CLOUD_SQL_INSTANCE,DB_PORT=5432,DB_NAME=swmaps,GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --set-secrets="DB_PASSWORD=swmaps-db-password:latest,DB_USER=swmaps-db-user:latest" \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=5 \
    --add-cloudsql-instances=$CLOUD_SQL_INSTANCE \
    --service-account=$SVC_ACCOUNT


echo "Step 6: Setting up Pub/Sub topic"
gcloud pubsub topics create swmaps-new-scenes 2>/dev/null || true

echo "Step 7: Wiring Pub/Sub to trigger service"
TRIGGER_URL=$(gcloud run services describe $TRIGGER_SERVICE \
    --region=$REGION \
    --format='get(status.url)')

PUBSUB_SA="swmaps-pubsub-invoker@$PROJECT_ID.iam.gserviceaccount.com"
gcloud iam service-accounts create swmaps-pubsub-invoker \
    --display-name="SwMaps Pub/Sub Invoker" 2>/dev/null || true

gcloud run services add-iam-policy-binding $TRIGGER_SERVICE \
    --region=$REGION \
    --member="serviceAccount:$PUBSUB_SA" \
    --role="roles/run.invoker"

gcloud pubsub subscriptions delete swmaps-new-scenes-sub 2>/dev/null || true
gcloud pubsub subscriptions create swmaps-new-scenes-sub \
    --topic=swmaps-new-scenes \
    --push-endpoint="$TRIGGER_URL/trigger" \
    --push-auth-service-account=$PUBSUB_SA

API_URL=$(gcloud run services describe $API_SERVICE \
    --region=$REGION \
    --format='get(status.url)')

echo "---------------------------------------------------"
echo "Deployment complete."
echo "Pipeline job:     $PIPELINE_JOB"
echo "Trigger service:  $TRIGGER_URL"
echo "API service:      $API_URL"
echo "Science viewer:   $API_URL"
echo "Swagger UI:       $API_URL/docs"
echo "Pub/Sub topic:    swmaps-new-scenes"
echo "---------------------------------------------------"
echo "To test manually:"
echo "gcloud pubsub topics publish swmaps-new-scenes --message='{\"scene_id\":\"test\",\"sensor\":\"landsat-7\",\"acquisition_date\":\"1999-08-06\"}'"