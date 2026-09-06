#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Load non-browser Jira/runtime configuration from the ignored local file. Values are
# exported only to the local API process and are never printed by this launcher.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Prefer the macOS Keychain for the API token. On first launch, prompt once and store
# it under a repository-specific service name; subsequent launches retrieve it without
# putting the token in .env, shell history, logs, Node, or the browser.
JIRA_KEYCHAIN_SERVICE="${JIRA_KEYCHAIN_SERVICE:-thesis-fraud-jira-api-token}"
if [[ -z "${JIRA_API_TOKEN:-}" && -n "${JIRA_USER_EMAIL:-}" ]] && command -v security >/dev/null 2>&1; then
  JIRA_API_TOKEN="$(security find-generic-password \
    -a "$JIRA_USER_EMAIL" \
    -s "$JIRA_KEYCHAIN_SERVICE" \
    -w 2>/dev/null || true)"
  export JIRA_API_TOKEN
fi

if [[ -z "${JIRA_API_TOKEN:-}" && -n "${JIRA_USER_EMAIL:-}" && -t 0 ]] && command -v security >/dev/null 2>&1; then
  read -r -s -p "Atlassian API token (will be stored in macOS Keychain): " JIRA_API_TOKEN
  echo
  if [[ -n "$JIRA_API_TOKEN" ]]; then
    security add-generic-password \
      -a "$JIRA_USER_EMAIL" \
      -s "$JIRA_KEYCHAIN_SERVICE" \
      -w "$JIRA_API_TOKEN" \
      -U >/dev/null
    export JIRA_API_TOKEN
  fi
fi

JIRA_REQUIRED=(JIRA_BASE_URL JIRA_USER_EMAIL JIRA_API_TOKEN JIRA_PROJECT_KEY)
JIRA_MISSING=()
for name in "${JIRA_REQUIRED[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    JIRA_MISSING+=("$name")
  fi
done
if (( ${#JIRA_MISSING[@]} > 0 )); then
  echo "Jira analytics disabled; missing: ${JIRA_MISSING[*]}" >&2
  echo "Configure the ignored .env file and rerun this launcher." >&2
else
  echo "Jira analytics configured for project ${JIRA_PROJECT_KEY}."
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 -m uvicorn src.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

API_READY=false
for _ in {1..50}; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" || true
    echo "The new API did not start. Stop the process already using port 8000, then rerun ./run_node_ui.sh." >&2
    exit 2
  fi
  if curl --silent --fail --max-time 1 http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
    API_READY=true
    break
  fi
  sleep 0.1
done
if [[ "$API_READY" != "true" ]]; then
  echo "The local API did not become healthy within five seconds." >&2
  exit 2
fi

node web/server.mjs
