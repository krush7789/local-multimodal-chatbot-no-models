#!/usr/bin/env bash
set -euo pipefail

if [[ "${BOOTSTRAP_MODELS:-1}" == "1" ]]; then
  python scripts/bootstrap_models.py
fi

PORT="${PORT:-8501}"

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
