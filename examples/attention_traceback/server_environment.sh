#!/usr/bin/env bash
# Read-only server inventory. Activate the existing ORSD environment before running.
set -euo pipefail
python3 --version
nvidia-smi
python3 -m pip show torch
python3 -m pip show verl || true
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('available:', torch.cuda.is_available())
print('devices:', torch.cuda.device_count())
PY
