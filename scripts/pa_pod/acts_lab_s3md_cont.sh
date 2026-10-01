#!/usr/bin/env bash
# S3-MD 接管补跑（2026-10-01 实际从 upload/s3md_cont.sh 运行，任务列表脚本为 upload/s3md_tasks.py = scripts/pa_pod/acts_s3md_tasks.py）
# S3-MD 接管（主流水的父进程已停，原分片 2 / 3 的 83 进程照跑）：
#   CV 格：每轮清空 oom_failures.jsonl（83 会剔除 OOM ≥ 2 次的格），按任务分给空闲显存最多的卡，总进程数 ≤ 4（含仍在跑的原分片）→ 直到全部完成
#   → select → plan final → 最终格同法补全 → verify → 打包（与主流水同名同内容）。
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3md_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3md"
cd "$REPO"; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
R83="python scripts/83_acts_corpus.py --resume --device cuda --profile s1pp --outdir $OUT/runs --phase W"
D="python scripts/115_acts_s3md.py --outdir $OUT --models /tmp/acts_s3z_out/models"
MAXP=4
orig_alive() { pgrep -f "outdir $OUT/runs --phase W --run --parts 4 --part $1" > /dev/null; }
top_gpus() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2-$3, $1}' | sort -rn | head -n $1 | awk '{print $2}'; }
fill() {   # $1 = 标签；直到 plan_W 全部完成
  local tag=$1 pass=0
  while :; do
    pass=$((pass+1)); [ $pass -gt 300 ] && { echo "!! $tag 超过 300 轮，停止补跑"; break; }; local ex=""; local alive=0
    for k in 0 1 2 3; do if orig_alive $k; then ex="$ex $k"; alive=$((alive+1)); fi; done
    mapfile -t ROWS < <(python "$UP/s3md_tasks.py" $ex)
    if [ ${#ROWS[@]} -eq 0 ]; then
      [ $alive -eq 0 ] && break
      echo "== [$(date -u +%H:%M:%S)] $tag 轮 $pass：可接管任务 0，原分片仍在跑（$ex），等 120 s"; sleep 120; continue
    fi
    local n=$((MAXP - alive)); [ $n -gt ${#ROWS[@]} ] && n=${#ROWS[@]}; [ $n -lt 1 ] && { sleep 120; continue; }
    [ -f $OUT/runs/oom_failures.jsonl ] && mv $OUT/runs/oom_failures.jsonl "$T/oom_failures_${tag}_$pass.jsonl"
    mapfile -t G < <(top_gpus $n)
    local groups=(); for ((i=0; i<n; i++)); do groups[$i]=""; done
    for ((j=0; j<${#ROWS[@]}; j++)); do t=${ROWS[$j]%% *}; groups[$((j % n))]="${groups[$((j % n))]} $t"; done
    echo "== [$(date -u +%H:%M:%S)] $tag 轮 $pass：未完成 $(printf '%s\n' "${ROWS[@]}" | awk '{s+=$2} END {print s}') 格 / ${#ROWS[@]} 任务；原分片在跑:${ex:- 无}；接管卡 ${G[*]}"
    local pids=()
    for ((i=0; i<n; i++)); do
      CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $R83 --run --tasks ${groups[$i]} >> "$LOG/cont_${tag}_g${G[$i]}.log" 2>&1 < /dev/null & pids+=($!); sleep 5
    done
    for p in "${pids[@]}"; do wait $p; done
    sleep 60
  done
  python - <<PY | tee "$LOG/runs_${tag}_verify.log"
import json
p = json.load(open("$OUT/runs/plan_W.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
for l in open("$OUT/runs/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify $tag: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
}
echo "== [$(date -u +%H:%M:%S)] 接管开始"
fill cv
$D --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$D --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
fill final
tar -chzf "$UP/out_acts_s3md.tar.gz" $OUT/s0src.npz $OUT/ea_targets.npz $OUT/experts.npz $OUT/fit.json $OUT/search $OUT/runs/plan_W.json $OUT/runs/runs.jsonl \
  $OUT/select.json "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3md.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3md done"
