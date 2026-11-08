#!/usr/bin/env bash
# Launches the dashboard at http://localhost:8501 (PORT=... to change it). Run from anywhere.
# Uses the experiment's virtual environment if present. Stop it with Ctrl+C.
#
# It only listens on this machine (localhost), so it's never exposed to the network.
# On a cloud VM, open it from your laptop through an SSH tunnel:
#   gcloud compute ssh <VM> --zone <ZONE> -- -N -L 8501:localhost:8501
# then browse http://localhost:8501 on the laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
PORT="${PORT:-8501}"
URL="http://localhost:${PORT}"
echo "Dashboard: ${URL}   (Ctrl+C to stop)"
# On a Mac, open the browser once the server answers.
if [ "$(uname)" = "Darwin" ]; then
  ( for _ in $(seq 1 30); do
      curl -s "${URL}/_stcore/health" >/dev/null 2>&1 && { open "${URL}"; break; }
      sleep 1
    done ) &
fi
exec "$PY" -m streamlit run dashboard/app.py \
  --server.headless true --server.address localhost --server.port "${PORT}" \
  --browser.gatherUsageStats false "$@"
