#!/bin/bash
# Pre-push smoke test — catches import/syntax errors before pushing
# Usage: bash test-local.sh

set -e

echo "🔍 Testing Python imports..."
python -c "
from app.core.model_manager import _get_gpu_config, ModelManager
from app.services.adapter import MinerUAdapter
from app.main import app
print('✅ All imports OK')
print(f'   Routes: {[r.path for r in app.routes if hasattr(r, \"path\")]}')
"

echo ""
echo "🔍 Checking Docker build (no GPU needed)..."
docker build --platform linux/amd64 -t procr:test -f Dockerfile . 2>&1 | tail -5

echo ""
echo "🔍 Smoke test in Docker (import check)..."
docker run --rm procr:test python -c "
from app.core.model_manager import _get_gpu_config, ModelManager
from app.services.adapter import MinerUAdapter
from app.main import app
print('✅ Docker imports OK')
"

echo ""
echo "✅ All checks passed. Safe to push."
