#!/usr/bin/env bash
# WP17-ACTS UT-ENS 全自动流水（实验室 8k-4090；他人占卡；同时在跑的进程 ≤ MAXP）：
#   解包 → 输入核对 → 输出 symlink 到根盘 /tmp/acts_utens_out → pytest →
#   逐轮：剩余作业按成本 LPT 分给 MAXP 个进程（空闲显存最多的卡）→ 直到 33 个作业都有结果 → 打包 upload/out_acts_utens.tar.gz（不含表示缓存）。
#   MAXP=4 setsid -f bash .../scripts/pa_pod/acts_lab_run_utens.sh > /tmp/acts_utens_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_utens_out"; LOG="$T/logs"; UP="$REPO/upload"; MAXP="${MAXP:-4}"
OUT="results/acts/utens"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export AUTOCLS_EVAL_TIMEOUT_S=1800                                       # = UT
tar --no-same-owner -xf "$UP/code_head_utens.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_utens_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_utens.txt")"
[ -f results/acts/ut/md/meta.json ] || { echo "!! UT meta 缺"; exit 3; }
[ -f "$OUT/plan.json" ] || { echo "!! plan.json 缺"; exit 3; }
mkdir -p "$T/utens"; if [ -e "$OUT" ] && [ ! -L "$OUT" ]; then cp -r "$OUT/." "$T/utens/" && rm -rf "$OUT"; fi; [ -L "$OUT" ] || ln -s "$T/utens" "$OUT"
python -m pytest tests/test_acts_utens.py tests/test_acts_ut.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
top_gpus() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2-$3, $1}' | sort -rn | head -n $1 | awk '{print $2}'; }
remaining() { python -c "
import json, os
P = json.load(open('$OUT/plan.json')); print(sum(not os.path.exists(f'$OUT/results/{j}.json') for j in P['costs']))"; }
for pass in $(seq 1 10); do
  left=$(remaining); [ "$left" -eq 0 ] && break
  n=$MAXP; [ $n -gt $left ] && n=$left
  mapfile -t G < <(top_gpus $n); pids=()
  echo "== [$(date -u +%H:%M:%S)] 轮 $pass：剩余作业 $left；卡 ${G[*]}"
  for ((i=0; i<n; i++)); do
    CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 python scripts/119_acts_utens.py --run --worker $i --workers $n --device cuda >> "$LOG/run_p${pass}_w$i.log" 2>&1 < /dev/null & pids+=($!); sleep 3
  done
  for p in "${pids[@]}"; do wait $p; done
  sleep 30
done
echo "== 剩余作业 $(remaining)"
tar -chzf "$UP/out_acts_utens.tar.gz" $OUT/plan.json $OUT/results "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_utens.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts utens done"
