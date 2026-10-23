#!/bin/bash
# 用法: worker.sh <gpu序号>  从 queue.txt 取点(flock)依次跑; 队列空: 若 control_pending 存在则等(不占 GPU), 否则退出(卡交还)。无占位进程。
G=$1; D=/home/hzh/scicomp26/q38f; cd $D
U=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
while :; do
  [ -f worker_stop ] && break
  LINE=$(flock queue.lock bash -c 'l=$(grep -v "^#" queue.txt | grep . | head -1); [ -n "$l" ] && { grep -vxF "$l" queue.txt > queue.tmp; mv queue.tmp queue.txt; echo "$l"; }')
  if [ -z "$LINE" ]; then [ -f control_pending ] && { sleep 60; continue; }; break; fi
  echo "$(date '+%F %T') g$G start $LINE" >> worker.log
  bash run_one.sh $U $LINE; echo "$(date '+%F %T') g$G end $LINE rc=$?" >> worker.log
done
echo "$(date '+%F %T') g$G queue empty, exit" >> worker.log
