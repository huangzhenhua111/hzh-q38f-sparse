#!/bin/bash
# 训练已完成、只缺评测的点: eval + flops + RUN_ALL_DONE
U=$1; RUN=$2; NG=$3; S=/home/hzh/scicomp26; Q=$S/q38f; PY=$S/.venv_q35/bin/python; RB=$S/runs
cd $Q/code; export CUDA_VISIBLE_DEVICES=$U PYTHONUNBUFFERED=1
bash $S/eval/eval_local_q38f.sh $S/data/dolma3_mix_100M_a.jsonl $RB/$RUN/hf 2048 8 $U 2>&1 | tee $RB/$RUN/eval_100M_a.log || { echo "EVAL_FAILED Traceback" >> $RB/$RUN/train.log; exit 1; }
T=$($PY -c "import json;print(json.load(open(\"$RB/$RUN/summary.json\"))[\"tokens_trained\"])") && $PY count_flops_q38f.py --model $RB/$RUN/hf --seq 2048 --tokens $T --ngram_rows $NG > $RB/$RUN/flops.json && echo RUN_ALL_DONE >> $RB/$RUN/train.log
