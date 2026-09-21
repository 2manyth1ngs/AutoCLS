#!/usr/bin/env bash
# WP17-ACTS UCR-ADM 在实验室 8k-4090 跑准入阶段 M（2,400 格 + 120 探针）：
#   83 --run --phase M 按 GPUS 列表分进程（按未完成格均衡分片，各占一张卡，共用 runs.jsonl）→ 内置守护（某片进程非零退出 →
#   等磁盘 ≥ 1 GB 后按同片 --resume 重启，最多 RESTARTS 次；共享盘写满是 S1″ 两次停机的原因）→ --probe0 → --verify
#   （不过则单进程 --resume 补跑）→ 打包 upload/out_acts_ucradm_M.tar.gz。不自停（非计费机器）。
#   GPUS="0 1 2" RESTARTS=6 [LOG=/tmp/ucradm_out/logs] nohup bash .../scripts/pa_pod/acts_lab_run_ucradm.sh > .../logs/acts_ucradm_M.log 2>&1 &
#   共享数据盘写满（S1″ 两次、本轮 13:23 UTC 一次）→ 输出文件（runs / probe0 / inflight / oom）symlink 到根盘 /tmp/ucradm_out，LOG 同去；tar -h 解引用。
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
LOG="${LOG:-$REPO/logs}"; UP="$REPO/upload"; OUT="results/acts/ucradm"   # LOG 可指到根盘（/tmp/...）：共享数据盘写满时日志 / 输出经 symlink 落根盘
PHASE="M"; GPUS="${GPUS:-0 1}"; RESTARTS="${RESTARTS:-6}"
GPU0=$(echo $GPUS | cut -d" " -f1)
NP=$(echo $GPUS | wc -w)
COMMON="--profile ucradm --outdir $OUT --phase $PHASE"
cd "$REPO"; mkdir -p "$LOG" "$UP" "$OUT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g python - <<'PY' || { echo "CUDA gate FAILED on GPU $g"; exit 5; }
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA", torch.cuda.get_device_name(0))
PY
done
free_gb() { df -BG --output=avail "$LOG" | tail -1 | tr -dc '0-9'; }   # 看输出所在盘（LOG 与 runs.jsonl symlink 同盘）
launch() {  # launch <part_index> <gpu> → LAST_PID（不用 $(...)：子 shell 里起的进程不是主 shell 的子进程，wait 拿不到退出码）
  CUDA_VISIBLE_DEVICES=$2 nohup nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON --parts $NP --part $1 >> "$LOG/acts_ucradm_${PHASE}_p$1.log" 2>&1 < /dev/null &
  LAST_PID=$!
}
echo "== [$(date -u +%H:%M:%S)] phase $PHASE start ($NP procs, GPUs $GPUS, free $(free_gb) GB)"
declare -a PIDS GPUL RS DONE
i=0
for g in $GPUS; do launch $i $g; PIDS[$i]=$LAST_PID; GPUL[$i]=$g; RS[$i]=0; DONE[$i]=0; i=$((i+1)); sleep 20; done
echo "pids ${PIDS[*]}"
# 守护循环：某片进程非零退出（磁盘写满 / 崩溃）→ 等磁盘 ≥ 1 GB 后按同片 --resume 重启（最多 RESTARTS 次）；正常退出（rc 0）记 DONE
while :; do
  alive=0
  for ((j=0; j<NP; j++)); do
    [ "${DONE[$j]}" = 1 ] && continue
    if kill -0 "${PIDS[$j]}" 2>/dev/null; then alive=1; continue; fi
    wait "${PIDS[$j]}"; rc=$?
    if [ $rc = 0 ] || [ "${RS[$j]}" -ge "$RESTARTS" ]; then
      echo "== [$(date -u +%H:%M:%S)] part $j exited rc=$rc (restarts ${RS[$j]}) → done"; DONE[$j]=1; continue
    fi
    while [ "$(free_gb)" -lt 1 ]; do echo "== [$(date -u +%H:%M:%S)] disk $(free_gb) GB < 1 GB, waiting"; sleep 300; done
    RS[$j]=$((RS[$j]+1)); echo "== [$(date -u +%H:%M:%S)] part $j exited rc=$rc → restart #${RS[$j]} on GPU ${GPUL[$j]}"
    launch $j ${GPUL[$j]}; PIDS[$j]=$LAST_PID; alive=1; sleep 20
  done
  [ $alive = 1 ] || break
  sleep 120
done
echo "== [$(date -u +%H:%M:%S)] all exited; rows $(wc -l < "$OUT/runs.jsonl")"
CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON > "$LOG/acts_ucradm_${PHASE}_probe0.log" 2>&1
python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_ucradm_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_ucradm_${PHASE}_verify.log"
if ! grep -q "→ OK" "$LOG/acts_ucradm_${PHASE}_verify.log"; then
  echo "== verify NOT OK → 单进程 --resume 补跑"
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --run --resume --device cuda $COMMON >> "$LOG/acts_ucradm_${PHASE}_fix.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU0 nice -n 10 python scripts/83_acts_corpus.py --probe0 --resume --device cuda $COMMON >> "$LOG/acts_ucradm_${PHASE}_fix.log" 2>&1
  python scripts/83_acts_corpus.py --verify --device cpu $COMMON > "$LOG/acts_ucradm_${PHASE}_verify.log" 2>&1; cat "$LOG/acts_ucradm_${PHASE}_verify.log"
fi
tar -chzf "$UP/out_acts_ucradm_${PHASE}.tar.gz" "$OUT/runs.jsonl" "$OUT/probe0.jsonl" $( [ -f "$OUT/oom_failures.jsonl" ] && echo "$OUT/oom_failures.jsonl" ) "$LOG"/acts_ucradm_*.log
ls -la "$UP/out_acts_ucradm_${PHASE}.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts ucradm phase $PHASE done"
