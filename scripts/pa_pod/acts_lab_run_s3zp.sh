#!/usr/bin/env bash
# WP17-ACTS S3-Z′ 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3zp_out）：
#   解包 → 核对输入（计划 / 元数据 / 预置 S3-Z 已跑格）→ symlink → pytest → 分槽 → 83 phase W（分片同时启动 + 补跑 + verify）→ 打包。
#   不训练、不搜索：计划在本地由 113 --prep 生成（决赛 4 条 × 两次重复 3 折 CV + 官方 3 种子）。
#   GPUS="0 1 2 3 4 5 6 7" setsid -f bash .../scripts/pa_pod/acts_lab_run_s3zp.sh > /tmp/acts_s3zp_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3zp_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3zp"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-6}"; SLOT_MB="${SLOT_MB:-2500}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
tar --no-same-owner -xf "$UP/code_head_s3zp.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3zp_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_s3zp.txt")"
for f in meta.json finalists.json runs/plan_W.json runs/runs.jsonl; do [ -f "$OUT/$f" ] || { echo "!! $OUT/$f 缺"; exit 3; }; done
d=runs; mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
python -m pytest tests/test_acts_s3zp.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
SLOTS=()
for g in $GPUS; do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $g | awk -F', ' '{print $1-$2}')
  n=$(( (free - 1500) / SLOT_MB )); [ $n -gt 2 ] && n=2; [ $n -lt 0 ] && n=0
  for ((k=0; k<n; k++)); do SLOTS+=($g); done
  echo "== GPU $g free ${free} MiB → $n 槽"
done
NS=${#SLOTS[@]}; echo "== 共 $NS 槽：${SLOTS[*]}"; [ $NS -ge 1 ] || { echo "!! 无可用槽"; exit 5; }
retry() { local g=$1 lf=$2; shift 2
  for ((r=0; r<=RETRIES; r++)); do CUDA_VISIBLE_DEVICES=$g nice -n 10 "$@" >> "$lf" 2>&1 < /dev/null && return 0; echo "== [$(date -u +%H:%M:%S)] GPU $g retry #$((r+1)): $*" >> "$lf"; sleep 60; done
  echo "== FAILED after $RETRIES retries: $*" >> "$lf"; return 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
R83="python scripts/83_acts_corpus.py --resume --device cuda --profile s1pp --outdir $OUT/runs --phase W"
echo "== [$(date -u +%H:%M:%S)] 跑格"
pids=()
for ((s=0; s<NS; s++)); do ( retry ${SLOTS[$s]} "$LOG/runs_p$s.log" $R83 --run --parts $NS --part $s ) & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
retry ${SLOTS[0]} "$LOG/runs_fix.log" $R83 --run
python - <<PY | tee "$LOG/runs_verify.log"
import json
p = json.load(open("$OUT/runs/plan_W.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
for l in open("$OUT/runs/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
tar -chzf "$UP/out_acts_s3zp.tar.gz" $OUT/runs/plan_W.json $OUT/runs/runs.jsonl "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3zp.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3zp done"
