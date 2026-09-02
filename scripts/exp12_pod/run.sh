#!/usr/bin/env bash
# 实验 1/2 上云：并行启动（nohup 脱离；日志在 /workspace/logs；全程 --resume 可续）。
# 用法（pod 上）：bash /workspace/wp/CLS/scripts/exp12_pod/run.sh
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"; LOG=/workspace/logs; cd "$REPO"; mkdir -p "$LOG"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nohup python scripts/30_e_degradation_diag.py --train-r1 --diag --bypass --resume --verdict \
      > "$LOG/exp1.log" 2> "$LOG/exp1.err" &
echo "exp1 pid=$!"
nohup python scripts/31_soft_label_tclsc.py --noise-model --train --resume --only H0 S1 Cshared F10 F20 \
      > "$LOG/exp2_A.log" 2> "$LOG/exp2_A.err" &
echo "exp2A pid=$!"
nohup python scripts/31_soft_label_tclsc.py --train --resume --only S2 S1e F30 F40 X \
      > "$LOG/exp2_B.log" 2> "$LOG/exp2_B.err" &
echo "exp2B pid=$!"
sleep 5; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
