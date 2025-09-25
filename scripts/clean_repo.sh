#!/bin/bash
set -euo pipefail

echo "🧹 Cleaning repository artifacts..."

rm -rf logs || true
rm -rf sessions || true
rm -f ui_run.log jarvis_run.log jarvis_live.log ui.pid || true

echo "✅ Cleaned logs and sessions."
echo "Tip: Ensure you commit with a sanitized .env (use .env.example)."


