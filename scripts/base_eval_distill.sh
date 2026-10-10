cd ~/ILA
export HF_HUB_OFFLINE=1 ILA_VLLM_UTIL=0.45
for es in gsm8k_test math500; do
  .venv-vllm/bin/python scripts/evaluate.py --config configs/stage1_distill.yaml --style _base --seed 0 --eval-set $es --base > results/stage1_distill/logs/base_$es.log 2>&1
done
echo DONE > results/stage1_distill/base.flag
