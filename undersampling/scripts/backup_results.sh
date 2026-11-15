#!/usr/bin/env bash
# Copies results/ to a Cloud Storage bucket every N minutes (default 10), so that if the VM is deleted,
# at most N minutes of work are lost. Run it on the VM next to main.py; stop it with Ctrl+C (or kill).
#
#   nohup ./scripts/backup_results.sh gs://<bucket>/undersampling 10 > backup.out 2>&1 &
#
# The live Optuna databases are not copied directly (they may be mid-write): each one is first copied
# with SQLite's own backup into optuna_snapshot.db, which is what gets uploaded. To restore on a new
# machine, see README ("If the VM dies").
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:?usage: ./scripts/backup_results.sh gs://<bucket>/<path> [minutes]}"
MINUTES="${2:-10}"
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

while true; do
  "$PY" - <<'EOF'
import pathlib
import sqlite3

for db in pathlib.Path('results').glob('*/optuna.db'):
    src, dst = sqlite3.connect(db), sqlite3.connect(db.with_name('optuna_snapshot.db'))
    src.backup(dst)
    dst.close()
    src.close()
EOF
  gcloud storage rsync --recursive --exclude='(^|/)optuna\.db$' results "${DEST}/results"
  echo "$(date '+%Y-%m-%d %H:%M:%S') results copied to ${DEST}/results"
  sleep "$((MINUTES * 60))"
done
