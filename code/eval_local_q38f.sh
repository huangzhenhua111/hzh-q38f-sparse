#!/usr/bin/env bash
# 本地单卡版官方评测: bash eval_local.sh <dataset.jsonl> <model_dir> [seq_len] [batch] [gpu]
set -euo pipefail
d=$(cd "$(dirname "$0")" && pwd); ds=$1; m=$2; seq=${3:-2048}; bs=${4:-8}; gpu=${5:-0}
out=$m/../eval_$(basename "$ds" .jsonl)_seq${seq}.json; rm -f "$out"
CUDA_VISIBLE_DEVICES=$gpu /home/hzh/scicomp26/.venv_q35/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  "$d/eval_q38f.py" --dataset-path "$ds" --model-path "$m" --batch-size "$bs" --sequence-length "$seq" --output "$out"
echo "RESULT_FILE $out"; cat "$out"
