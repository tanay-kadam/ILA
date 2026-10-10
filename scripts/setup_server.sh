#!/usr/bin/env bash
# One-time environment setup on a Linux GPU server without admin rights or pip.
# Creates .venv (Python 3.12 via uv) with vLLM and the training stack.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 .venv
# vLLM first: it pins compatible torch/transformers versions.
uv pip install --python .venv/bin/python vllm
uv pip install --python .venv/bin/python peft datasets accelerate pyyaml pytest
.venv/bin/python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
echo SETUP_DONE
