#!/usr/bin/env bash
# WP17-ACTS S3-X 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3x_out）：
#   等 S3-W 收官（不在 S3-W 运行中改动代码）→ 解包 → 核对输入 → symlink → pytest → 分槽 →
#   ① 7 个比较器池内打分 ② 83 phase W 跑 21 条 × （官方种子 0 / 1 / 2 + D_val）中未跑的格 → verify → 打包 upload/out_acts_s3x.tar.gz。
#   GPUS="..." setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3x.sh > /tmp/acts_s3x_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3x_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3x"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-6}"; SLOT_MB="${SLOT_MB:-2500}"; WAIT_W="${WAIT_W:-1}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
if [ "$WAIT_W" = 1 ]; then
  echo "== [$(date -u +%H:%M:%S)] 等 S3-W 收官"
  for k in $(seq 1 1440); do grep -q "acts s3w done" /tmp/acts_s3w_out/logs/run.log 2>/dev/null && break; pgrep -f acts_lab_run_s3w.sh > /dev/null || break; sleep 30; done
  echo "== [$(date -u +%H:%M:%S)] S3-W 已结束：$(tail -1 /tmp/acts_s3w_out/logs/run.log)"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head_s3x.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3x_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_s3x.txt")"
for f in pool.json meta.json runs/plan_W.json runs/runs.jsonl; do [ -f "$OUT/$f" ] || { echo "!! $OUT/$f 缺"; exit 3; }; done
nm=$(ls results/acts/s3rp/models/*_s*_*.pt 2>/dev/null | wc -l); echo "== S3-R′ 模型 $nm / 36"
for d in abl_mean abl_raw abl_noshared; do echo "== s3v/$d 模型 $(ls results/acts/s3v/$d/models/*.pt 2>/dev/null | wc -l) / 18"; done
echo "== S3-C 模型 A/D $(ls results/acts/s3c/models/{A,D}_*.pt 2>/dev/null | wc -l) / 12；靶端 E $(ls results/acts/s3cd/E/*.npz 2>/dev/null | wc -l) / 6；原始窗口 $(ls results/acts/s3v/raw/*_whole_official.npz 2>/dev/null | wc -l) / 6"
for d in runs scores; do
  mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
python -m pytest tests/test_acts_s3x.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
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
run_jobs() {
  local stage=$1; shift; local JOBS=("$@"); local pids=()
  for ((s=0; s<NS; s++)); do local g=${SLOTS[$s]}
    ( for ((j=s; j<${#JOBS[@]}; j+=NS)); do retry $g "$LOG/${stage}_s$s.log" ${JOBS[$j]}; done ) & pids+=($!); sleep 2; done
  for p in "${pids[@]}"; do wait $p; done; }
X="python scripts/110_acts_s3x.py --outdir $OUT"
# ① 池内打分（失败不阻断跑格，可事后补）
echo "== [$(date -u +%H:%M:%S)] 池内打分 7"
RETRIES=1; J=(); for a in FULL WO_SETTF WO_TS2VEC WO_SHARED BLIND A D; do J+=("$X --score --arm $a --device cuda --resume"); done
run_jobs score "${J[@]}"; RETRIES="${RETRIES_RUN:-6}"
echo "== [$(date -u +%H:%M:%S)] 打分完：$(ls $OUT/scores/*.json 2>/dev/null | wc -l) / 7"
# ② 83 phase W（各分片同时启动 + 单槽补跑 + verify）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
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
tar -chzf "$UP/out_acts_s3x.tar.gz" $OUT/scores $OUT/runs/plan_W.json $OUT/runs/runs.jsonl "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3x.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3x done"
