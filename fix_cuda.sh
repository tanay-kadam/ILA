set -uo pipefail
cd ~/ILA
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
VLLM_VER=$($PY -c "import importlib.metadata as m; print(m.version('vllm'))")
TORCH_VER=$($PY -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
echo "vllm=$VLLM_VER torch=$TORCH_VER"

gpu_ok() {
  CUDA_VISIBLE_DEVICES=1 $PY -c "import torch; x=torch.randn(2048,2048,device=torch.device(0),dtype=torch.bfloat16); print('GPU_OK', torch.__version__, torch.version.cuda, (x@x).float().abs().mean().item())"
}

# 1) vLLM CUDA-12.x build from the GitHub release (bundles matching torch pin).
for CU in cu129 cu128 cu126; do
  URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}+${CU}-cp38-abi3-manylinux_2_31_x86_64.whl"
  if curl -sfIL "$URL" >/dev/null; then
    echo "found vllm wheel for $CU"
    uv pip install --python $PY "$URL" --torch-backend="$CU" --reinstall-package vllm --reinstall-package torch \
      --reinstall-package torchvision --reinstall-package torchaudio && break
  else
    echo "no vllm wheel for $CU"
  fi
done

if ! gpu_ok; then
  echo "vLLM CUDA-12 path failed; installing torch cu128 for HF backend"
  uv pip install --python $PY "torch==${TORCH_VER}" --torch-backend=cu128 --reinstall-package torch
  gpu_ok || { uv pip install --python $PY torch --torch-backend=cu126 --reinstall-package torch; gpu_ok; }
fi

CUDA_VISIBLE_DEVICES=1 $PY - <<'EOF'
try:
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-0.5B", dtype="bfloat16", gpu_memory_utilization=0.3, enforce_eager=True)
    out = llm.generate(["Question: What is 2+3?\nAnswer:\n"], SamplingParams(temperature=0, max_tokens=16))
    print("VLLM_OK", repr(out[0].outputs[0].text))
except Exception as e:
    print("VLLM_FAIL", type(e).__name__, str(e)[:300])
EOF
echo FIX_DONE
