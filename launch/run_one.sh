#!/bin/bash
# 用法: run_one.sh <GPU-UUID> <h> <L> <lr> <micro_bs> <ngram_rows|0> [max_steps(冒烟)] [E K] [额外 train 参数...]  训练 → 官方 eval(seq 2048, 100M_a 全量) → FLOPs 核算 → RUN_ALL_DONE
U=$1; H=$2; L=$3; LR=$4; MB=$5; NG=$6; MS=${7:-0}; E=${8:-256}; K=${9:-5}; EXTRA="${@:10}"; S=/home/hzh/scicomp26; Q=$S/q38f; PY=$S/.venv_q35/bin/python
RUN=q38f_${H}x${L}; [ "$NG" = 0 ] && RUN=q38f_${H}x${L}_nong; [ "$MS" != 0 ] && RUN=smoke_${RUN}_mb${MB}${SMOKE_TAG}
RB=$S/runs; [ "$MS" != 0 ] && RB=$Q/runs     # 正式点写 $S/runs (collect.sh 的约定), 冒烟写 $Q/runs
cd $Q/code; mkdir -p $RB/$RUN
read NK NV NQ FE TOK <<<$($PY - <<PY
import sys; sys.path.insert(0,'.'); from plan_q38f import cfg, tokens
c=cfg($H,$L,$E,$K)
print(c['linear_num_key_heads'],c['linear_num_value_heads'],c['num_attention_heads'],c['moe_intermediate_size'],tokens(c, ng=($NG!=0)))
PY
)
export CUDA_VISIBLE_DEVICES=$U PYTHONUNBUFFERED=1
echo "$(date '+%F %T') $RUN gpu=$U h=$H L=$L nk=$NK nv=$NV nq=$NQ fe=$FE E=$E K=$K ngram_rows=$NG lr=$LR mb=$MB tokens=$TOK max_steps=$MS extra=$EXTRA" >> $Q/runs.log
$PY train_q38f.py --run_dir $RB/$RUN --gpu 0 --data $S/data/corpus_bpe16k_big/train.bin --tokenizer $S/tokenizer/bpe16k_v2 \
  --layers $L --hidden $H --heads $NQ --kv_heads 2 --head_dim 64 --lin_k_heads $NK --lin_v_heads $NV --lin_head_dim 64 --experts $E --topk $K --moe_ffn $FE --shared_ffn $FE \
  --aux_coef 0.001 --rope_theta 10000000.0 --partial_rotary 0.25 --experts_impl grouped_mm --seq 2048 --tokens $TOK --batch_tokens 524288 --micro_bs $MB --ngram_rows $NG \
  --lr $LR --min_lr_ratio 0.1 --warmup 0 --wd 0.1 --clip 1.0 --seed 0 --log_every 10 --eval_every 500 --eval_tokens 2000000 --ckpt_every 200 --resume 1 --max_steps $MS $EXTRA \
  2>&1 | tee -a $RB/$RUN/train.log
[ "$MS" != 0 ] && exit 0
grep -q '^DONE' $RB/$RUN/train.log || { echo "TRAIN_FAILED Traceback" >> $RB/$RUN/train.log; exit 1; }
mkdir -p $Q/eval; sed 's#eval_q35.py#../q38f/code/eval_q38f.py#' $S/eval/eval_local_q35.sh > $S/eval/eval_local_q38f.sh
bash $S/eval/eval_local_q38f.sh $S/data/dolma3_mix_100M_a.jsonl $RB/$RUN/hf 2048 8 $U 2>&1 | tee $RB/$RUN/eval_100M_a.log || { echo "EVAL_FAILED Traceback" >> $RB/$RUN/train.log; exit 1; }
T=$($PY -c "import json;print(json.load(open('$RB/$RUN/summary.json'))['tokens_trained'])") && $PY count_flops_q38f.py --model $RB/$RUN/hf --seq 2048 --tokens $T --ngram_rows $NG > $RB/$RUN/flops.json && echo RUN_ALL_DONE >> $RB/$RUN/train.log
