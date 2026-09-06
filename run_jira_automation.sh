#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# An ignored .env is supported for a local sandbox. A shell session or password
# manager injection is preferable. Values are never printed by this launcher.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

required=(
  JIRA_BASE_URL
  JIRA_USER_EMAIL
  JIRA_API_TOKEN
  JIRA_PROJECT_KEY
  JIRA_ULB_EPIC_KEY
  JIRA_SPARKOV_EPIC_KEY
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required Jira setting: $name" >&2
    exit 2
  fi
done

if [[ "${JIRA_AUTO_REFERRAL_ENABLED:-false}" != "true" ]]; then
  echo "JIRA_AUTO_REFERRAL_ENABLED must be true." >&2
  exit 2
fi

mkdir -p log
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 tools/run_jira_agent.py \
  --project-key "$JIRA_PROJECT_KEY" automation-doctor \
  2>&1 | tee -a log/application.log

echo "Starting automatic Jira referral/monitoring API on http://127.0.0.1:8000"
exec python3 -m uvicorn src.api:app --host 127.0.0.1 --port 8000
