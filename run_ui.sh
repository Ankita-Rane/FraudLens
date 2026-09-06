#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIRECTORY"

export HF_HOME="$SCRIPT_DIRECTORY/.cache/huggingface"
export MPLCONFIGDIR="/private/tmp/thesis_mpl"
export TOKENIZERS_PARALLELISM="false"

exec streamlit run ui/streamlit_app.py \
  --server.headless=true \
  --server.address=127.0.0.1 \
  --server.port=8501 \
  --browser.gatherUsageStats=false
