#!/usr/bin/env bash
# WP17-ACTS S3-M 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3m_out）：
#   解包 → 核对 S3-C 数据 / S3-E 轴坐标 / S3-G 扩充 / S3-I 模型 → symlink → pytest → 按各卡空闲显存分槽 →
#   扩充搜索 6 折（S3-I 集成、E 坐标）→ plan-aug → 83 真跑（seed 0）→ I′ 30 训练 → 考卷搜索 6 折（O′ / A′ / AX′）→ plan-runs → 83 真跑 → verify →
#   打包 upload/out_acts_s3j.tar.gz（train / search / search_aug / runs / runs_aug / logs，不含权重）。
#   83 的各分片同时启动（S3-F 教训：按启动时未完成格切分，错开启动会错位）。
#   GPUS="0 1 2 3 4 5 6 7" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3m.sh > /tmp/acts_s3m_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3m_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3m"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-20}"; SLOT_MB="${SLOT_MB:-8000}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3m_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
[ -f results/acts/s3c/data.npz ] && [ -f results/acts/s3c/runs/runs.jsonl ] || { echo "!! S3-C 数据缺"; exit 3; }
[ -f results/acts/s3e/axis_targets.npz ] || { echo "!! 轴坐标缺"; exit 3; }
[ -f results/acts/s3m/axes.npz ] && [ -f results/acts/s3m/prep.json ] && [ -f results/acts/s3k/analysis.json ] || { echo "!! axes / prep / s3k 缺"; exit 3; }
na=$(ls results/acts/s3j/search_aug/*.json 2>/dev/null | wc -l); [ "$na" -ge 6 ] || { echo "!! S3-J 扩充缺"; exit 3; }
[ -f results/acts/s3g/runs_aug/runs.jsonl ] && [ -f results/acts/s3j/runs_aug/runs.jsonl ] && [ -f results/acts/s3j/runs/runs.jsonl ] || { echo "!! 旧 runs 缺"; exit 3; }
mkdir -p "$OUT"
for d in models train search runs; do mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"; done
python -m pytest tests/test_acts_s3m.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
# 分槽
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
P="python scripts/100_acts_s3m.py --outdir $OUT --device cuda --resume"
run_jobs() {  # run_jobs <stage> <job 参数串...>：作业轮转到 NS 个槽
  local stage=$1; shift; local JOBS=("$@"); local pids=()
  for ((s=0; s<NS; s++)); do local g=${SLOTS[$s]}
    ( for ((j=s; j<${#JOBS[@]}; j+=NS)); do retry $g "$LOG/${stage}_s$s.log" $P ${JOBS[$j]}; done ) & pids+=($!); sleep 2; done
  for p in "${pids[@]}"; do wait $p; done; }
run83() {  # run83 <子目录> <日志前缀>：各分片同时启动 + 单槽补跑 + verify
  local sub=$1 tag=$2
  local R83="python scripts/83_acts_corpus.py --run --resume --device cuda --profile s2 --outdir $OUT/$sub --phase X"
  ( export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
    export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
    local pids=(); for ((s=0; s<NS; s++)); do ( retry ${SLOTS[$s]} "$LOG/${tag}_p$s.log" $R83 --parts $NS --part $s ) & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    retry ${SLOTS[0]} "$LOG/${tag}_fix.log" $R83 )
  python - <<PY | tee "$LOG/${tag}_verify.log"
import json
p = json.load(open("$OUT/$sub/plan_X.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
try:
    for l in open("$OUT/$sub/runs.jsonl"):
        r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
except FileNotFoundError:
    pass
print(f"verify $sub: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
}
FOLDS="Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1"
echo "== [$(date -u +%H:%M:%S)] EP / ET 训练"
J=(); for k in EP ET; do for sd in 0 1 2 3 4; do for f in $FOLDS; do J+=("--train --kind $k --fold $f --seed $sd"); done; done; done; run_jobs train "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 训练完：$(ls $OUT/train/*_s*_*.npz 2>/dev/null | wc -l) / 60"
CUDA_VISIBLE_DEVICES="" python scripts/100_acts_s3m.py --outdir $OUT --rank --device cpu > "$LOG/rank.log" 2>&1; tail -1 "$LOG/rank.log"
cp $OUT/rank.json "$LOG/rank.json" 2>/dev/null
G2=$(python -c "import json; print(int(json.load(open('$OUT/rank.json'))['gate2']))" 2>/dev/null || echo 0)
if [ "$G2" = "1" ]; then
  echo "== [$(date -u +%H:%M:%S)] 排序门通过 → 搜索"
  J=(); for f in $FOLDS; do J+=("--search --fold $f"); done; run_jobs search "${J[@]}"
  echo "== [$(date -u +%H:%M:%S)] 考卷搜索完：$(ls $OUT/search/*.json 2>/dev/null | wc -l) / 6"
  python scripts/100_acts_s3m.py --outdir $OUT --plan-runs --device cpu > "$LOG/plan_runs.log" 2>&1; tail -1 "$LOG/plan_runs.log"
  echo "== [$(date -u +%H:%M:%S)] 考卷真跑"; run83 runs exam
  tar -chzf "$UP/out_acts_s3m.tar.gz" $OUT/train $OUT/search $OUT/rank.json $OUT/runs/plan_X.json $OUT/runs/runs.jsonl "$LOG"/*.log 2>/dev/null
else
  echo "== [$(date -u +%H:%M:%S)] 排序门未过 → 停（S3M-RANK-NULL）"
  tar -chzf "$UP/out_acts_s3m.tar.gz" $OUT/train $OUT/rank.json "$LOG"/*.log 2>/dev/null
fi
ls -la "$UP/out_acts_s3m.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3m done"
