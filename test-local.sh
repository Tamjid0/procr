#!/bin/bash
# Pre-push smoke test — catches import/syntax errors before pushing
# Usage: bash test-local.sh

set -e

echo "🔍 Testing Python imports (no GPU needed)..."
python -c "
from app.services.adapter import MinerUAdapter
from app.main import app, OCRRequest
print('✅ All imports OK')
routes = [r.path for r in app.routes if hasattr(r, 'path')]
print(f'   Routes: {routes}')
"

echo ""
echo "✅ Safe to push."
