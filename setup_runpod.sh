#!/bin/bash
# One-time pod setup: check GPU, install pinned deps, verify torch sees CUDA.
set -e
cd "$(dirname "$0")"

fail() { echo "SETUP FAILED: $1"; exit 1; }

nvidia-smi > /dev/null 2>&1 || fail "no CUDA GPU detected (nvidia-smi)"

# Model downloads go to the persistent volume so they survive pod stops
export HF_HOME=/workspace/hf_cache

pip install --break-system-packages -r requirements-pod.txt || fail "pip install requirements-pod.txt"

python -c "import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/2**30,1), 'GB')" || fail "torch CUDA check"

echo "Setup OK. Run 'export HF_HOME=/workspace/hf_cache' in new shells too."
