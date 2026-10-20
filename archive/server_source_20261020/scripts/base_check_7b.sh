cd ~/ILA
for pair in "q7:Qwen/Qwen2.5-7B-Instruct:2" "d7:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:3"; do
  IFS=: read tag model gpu <<< "$pair"
  sed -e "s#^model: .*#model: $model#" -e "s#^results_dir: .*#results_dir: results/stage1_$tag#" configs/stage1.yaml > configs/stage1_$tag.yaml
  mkdir -p results/stage1_$tag/logs
  (
    export CUDA_VISIBLE_DEVICES=$gpu ILA_VLLM_UTIL=0.6
    HF_HUB_OFFLINE=0 .venv-vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('$model', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*'])" > results/stage1_$tag/logs/download.log 2>&1
    export HF_HUB_OFFLINE=1
    for es in gsm8k_test math500; do
      .venv-vllm/bin/python scripts/evaluate.py --config configs/stage1_$tag.yaml --style _base --seed 0 --base --eval-set $es > results/stage1_$tag/logs/base_$es.log 2>&1
    done
    echo DONE > results/stage1_$tag/base.flag
  ) &
done
wait
