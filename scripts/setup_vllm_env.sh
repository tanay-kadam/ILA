#!/usr/bin/env bash
# Separate venv for vLLM evaluation. The server driver is CUDA 12.7, so use an older vLLM
# with cu126 torch. Usage: bash scripts/setup_vllm_env.sh [vllm_version]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
VER="${1:-0.10.2}"
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python "vllm==$VER" --torch-backend=cu126 pyyaml numpy
.venv-vllm/bin/python -c 'import torch, vllm; print("torch", torch.__version__, torch.version.cuda, "vllm", vllm.__version__)'
echo SETUP_VLLM_DONE
