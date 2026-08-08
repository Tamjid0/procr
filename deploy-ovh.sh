#!/bin/bash
# Procr - OVH AI Deploy Script
# Usage: bash deploy-ovh.sh [build|push|deploy|all]

set -euo pipefail

# === CONFIG ===
IMAGE_NAME="procr"
IMAGE_TAG="latest"
OVH_REGISTRY="${OVH_REGISTRY:-}"  # Set this or login to shared registry
APP_NAME="procr-ocr"
GPU_FLAVOR="L4-1-gpu"  # L4-1-gpu, L40s-1-gpu, etc.
HTTP_PORT=8080
REGION="gra"  # gra = Gravelines (best GPU availability)

# === FUNCTIONS ===

build() {
    echo "🔨 Building Docker image..."
    docker build \
        --platform linux/amd64 \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -f Dockerfile \
        .
    echo "✅ Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
}

push() {
    if [ -z "$OVH_REGISTRY" ]; then
        echo "📦 Using OVH shared registry..."
        # Get shared registry address
        SHARED_REG=$(ovhai registry list 2>/dev/null | grep "ovhcloud/ai" | head -1 | awk '{print $1}')
        if [ -z "$SHARED_REG" ]; then
            echo "❌ Could not find shared registry. Login first: ovhai auth login"
            exit 1
        fi
        OVH_REGISTRY="$SHARED_REG"
    fi

    echo "📦 Pushing to ${OVH_REGISTRY}..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${OVH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${OVH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
    echo "✅ Push complete"
}

deploy() {
    if [ -z "$OVH_REGISTRY" ]; then
        SHARED_REG=$(ovhai registry list 2>/dev/null | grep "ovhcloud/ai" | head -1 | awk '{print $1}')
        OVH_REGISTRY="$SHARED_REG"
    fi

    echo "🚀 Deploying to OVH AI Deploy..."
    
    # Check if app already exists
    APP_ID=$(ovhai app list 2>/dev/null | grep "$APP_NAME" | awk '{print $1}' | head -1)
    
    if [ -n "$APP_ID" ]; then
        echo "⚠️  App ${APP_NAME} already exists (ID: ${APP_ID}). Updating..."
        ovhai app stop "$APP_ID" 2>/dev/null || true
        sleep 5
        ovhai app delete "$APP_ID" 2>/dev/null || true
        sleep 5
    fi

    ovhai app run ${OVH_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
        --gpu 1 \
        --flavor ${GPU_FLAVOR} \
        --default-http-port ${HTTP_PORT} \
        --name ${APP_NAME} \
        --region ${REGION} \
        --unsecure-http \
        --probe-path /diagnostic

    echo "✅ Deploy complete!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Get the app URL: ovhai app list | grep ${APP_NAME}"
    echo "   2. Set PYTHON_OCR_URL on VPS to the app URL"
    echo "   3. Test: curl <APP_URL>/diagnostic"
}

status() {
    echo "📊 App status:"
    ovhai app list | grep "$APP_NAME" || echo "App not found"
}

logs() {
    APP_ID=$(ovhai app list 2>/dev/null | grep "$APP_NAME" | awk '{print $1}' | head -1)
    if [ -n "$APP_ID" ]; then
        ovhai app logs "$APP_ID"
    else
        echo "App not found"
    fi
}

stop() {
    APP_ID=$(ovhai app list 2>/dev/null | grep "$APP_NAME" | awk '{print $1}' | head -1)
    if [ -n "$APP_ID" ]; then
        echo "⏹️  Stopping ${APP_NAME}..."
        ovhai app stop "$APP_ID"
        echo "✅ Stopped"
    else
        echo "App not found"
    fi
}

cleanup() {
    APP_ID=$(ovhai app list 2>/dev/null | grep "$APP_NAME" | awk '{print $1}' | head -1)
    if [ -n "$APP_ID" ]; then
        echo "🗑️  Deleting ${APP_NAME}..."
        ovhai app stop "$APP_ID" 2>/dev/null || true
        sleep 5
        ovhai app delete "$APP_ID"
        echo "✅ Deleted"
    else
        echo "App not found"
    fi
}

# === MAIN ===
case "${1:-all}" in
    build)   build ;;
    push)    push ;;
    deploy)  deploy ;;
    all)     build; push; deploy ;;
    status)  status ;;
    logs)    logs ;;
    stop)    stop ;;
    cleanup) cleanup ;;
    *)
        echo "Usage: $0 [build|push|deploy|all|status|logs|stop|cleanup]"
        exit 1
        ;;
esac
