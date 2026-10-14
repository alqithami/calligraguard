#!/usr/bin/env bash
# Portable setup only; does not start a study or change an existing locked run.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -e EXPERIMENT_LOCK.json || -e PROCESS.lock ]]; then
  echo 'Existing study detected. Preserve its environment; use a fresh clone for setup.' >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
command -v git >/dev/null
command -v nvidia-smi >/dev/null
"$PYTHON_BIN" - <<'PY'
import ctypes
for name in ['libcairo.so.2', 'libEGL.so.1', 'libGL.so.1']:
    ctypes.CDLL(name)
print('SYSTEM_LIBRARY_IMPORTS_PASS')
PY
sha256sum -c SOURCE_SHA256SUMS
if [[ ! -x .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi
printf 'torch==2.6.0+cu124\ntorchvision==0.21.0+cu124\nnumpy==1.26.4\n' > constraints.txt
.venv/bin/python -m pip install -c constraints.txt torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -c constraints.txt -r requirements.txt
mkdir -p vendor logs
ensure_vendor() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "vendor/$name" ]]; then
    git clone "$url" "vendor/$name"
    git -C "vendor/$name" checkout --detach "$revision"
  fi
  [[ "$(git -C "vendor/$name" rev-parse HEAD)" == "$revision" ]] || { echo "Unexpected revision: $name" >&2; exit 1; }
  [[ -z "$(git -C "vendor/$name" status --porcelain --untracked-files=no)" ]] || { echo "Modified tracked vendor files: $name" >&2; exit 1; }
}
ensure_vendor patchcore https://github.com/amazon-science/patchcore-inspection.git fcaa92f124fb1ad74a7acf56726decd4b27cbcad
ensure_vendor dinomaly https://github.com/guojiajeremy/Dinomaly.git 1f252be03a918789b19848f0ca37166e7c28dada
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import torch, torchvision, segmentation_models_pytorch, skia, pathops, cairosvg, faiss, tabulate, colorama
assert torch.cuda.is_available(), 'CUDA unavailable; refusing silent CPU training'
print('DEPENDENCY_IMPORTS_PASS', torch.__version__, torchvision.__version__)
PY
.venv/bin/python -m py_compile *.py
.venv/bin/python -m pip freeze > installed_environment.txt
printf 'PUBLIC_SETUP_COMPLETE: no training has been launched.\n'
