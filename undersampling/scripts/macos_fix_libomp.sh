#!/usr/bin/env bash
# macOS only. LightGBM needs the OpenMP library (libomp), which macOS doesn't ship.
# Official fix: `brew install libomp`. This script is a contained alternative: it points LightGBM,
# inside this folder's .venv only, to the libomp that scikit-learn bundles. Not needed on Linux.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB=$(.venv/bin/python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('lightgbm').origin).parent / 'lib' / 'lib_lightgbm.dylib')")
if otool -l "$LIB" | grep -q "sklearn/.dylibs"; then
  echo "Already fixed: $LIB"
else
  install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB"
  codesign --force --sign - "$LIB"     # re-sign after editing (required on Apple Silicon)
  echo "Fixed: $LIB"
fi
.venv/bin/python -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'loads correctly')"
