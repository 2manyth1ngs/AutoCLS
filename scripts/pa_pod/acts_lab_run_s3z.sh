#!/usr/bin/env bash
# WP17-ACTS S3-Z 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3z_out）：
#   解包 → 核对输入 → symlink → pytest → 分槽 →
#   ① 复制 S3-Y 种子 0–4 的模型 + 新训种子 5–9（2 臂 × 6 折 × 5 = 60 次，设置同 S3-Y）② 按分数搜索 NEW / NEW_B × 6 集（11 个搜索者并集）→
#   ③ U 每条 3 折 CV + 官方种子 0 → select（CV 均值选 1）→ 选中者与纯比较器第 1 名的官方种子 1 / 2 → verify → 打包 upload/out_acts_s3z.tar.gz。
#   GPUS="0 1 2 3 4 5 6 7" setsid -f bash .../scripts/pa_pod/acts_lab_run_s3z.sh > /tmp/acts_s3z_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3z_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3z"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-6}"; SLOT_MB="${SLOT_MB:-2500}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head_s3z.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3z_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_s3z.txt")"
[ -f results/acts/s3c/data.npz ] && [ -f results/acts/s3c/data.json ] || { echo "!! S3-C 数据缺"; exit 3; }
npt=$(ls results/acts/s3b/E_pt/*.npz | grep -vc _means); echo "== E_pt $npt / 176"; [ "$npt" -ge 176 ] || exit 3
[ -f results/acts/s3r/labels.npz ] || { echo "!! labels.npz 缺"; exit 3; }
ne=$(ls results/acts/s3cd/E/*.npz 2>/dev/null | wc -l); echo "== S3-C-D 靶端 E $ne / 6"; [ "$ne" -ge 6 ] || exit 3
[ -f results/acts/s3x/pool.json ] || { echo "!! S3-X 池缺"; exit 3; }
[ -f "$OUT/meta.json" ] && [ -f "$OUT/runs/runs.jsonl" ] || { echo "!! $OUT/meta.json 或预置 runs 缺"; exit 3; }
ny=$(ls /tmp/acts_s3y_out/models/*_s[0-4]_*.pt 2>/dev/null | wc -l); echo "== S3-Y 种子 0–4 模型 $ny / 60"; [ "$ny" -ge 60 ] || exit 3
for d in models train search runs; do
  mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
cp -n /tmp/acts_s3y_out/models/*_s[0-4]_*.pt "$T/models/"; cp -n /tmp/acts_s3y_out/train/*_s[0-4]_*.npz "$T/train/"
echo "== 复制后 models $(ls $T/models/*.pt | wc -l) / train $(ls $T/train/*.npz | wc -l)"
python -m pytest tests/test_acts_s3z.py tests/test_acts_s3y.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
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
FOLDS="Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1"
Y="python scripts/112_acts_s3z.py --outdir $OUT"
# ① 训练（S3-R′ 制度 + field 头 + 顶部排序权重；E 臂较慢，交错排队）
echo "== [$(date -u +%H:%M:%S)] 训练 60（种子 5–9）"
TR="python scripts/105_acts_s3r.py --train --stop acc --device cuda --resume --head field --loss lambda --outdir $OUT"
J=(); for sd in 5 6 7 8 9; do for f in $FOLDS; do for arm in E B; do J+=("$TR --arm $arm --fold $f --seed $sd"); done; done; done
run_jobs train "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 训练完：E $(ls $OUT/models/E_*.pt 2>/dev/null | wc -l) / B $(ls $OUT/models/B_*.pt 2>/dev/null | wc -l)（各需 60）"
python - <<PY > "$LOG/trainsum.log" 2>&1
import json, glob, numpy as np
out = {}
for f in sorted(glob.glob("$OUT/train/*.npz")):
    z = np.load(f); out[f.split("/")[-1][:-4]] = {"best_epoch": int(z["best_epoch"]), "hist": json.loads(str(z["hist"])), "head": str(z["head"]), "loss": str(z["loss"])}
json.dump(out, open("$OUT/trainsum.json", "w"), ensure_ascii=False)
print(len(out), "个训练记录；最优 epoch 中位", float(np.median([v["best_epoch"] for v in out.values()])))
PY
cat "$LOG/trainsum.log"
# ② 搜索
echo "== [$(date -u +%H:%M:%S)] 搜索 12"
J=(); for a in NEW NEW_B; do for f in $FOLDS; do J+=("$Y --search --arm $a --dataset $f --device cuda --resume"); done; done
run_jobs search "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 搜索完：$(ls $OUT/search/*.json 2>/dev/null | wc -l) / 12"
grep -h "\[search\]" $LOG/search_s*.log | sort > "$LOG/search_summary.log"; cat "$LOG/search_summary.log"
# ③ 83 phase W：CV 格 + 官方种子 0 → 选择 → 最终格
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
R83="python scripts/83_acts_corpus.py --resume --device cuda --profile s1pp --outdir $OUT/runs --phase W"
run83() {
  local tag=$1; local pids=()
  for ((s=0; s<NS; s++)); do ( retry ${SLOTS[$s]} "$LOG/runs_${tag}_p$s.log" $R83 --run --parts $NS --part $s ) & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  retry ${SLOTS[0]} "$LOG/runs_${tag}_fix.log" $R83 --run
  python - <<PY | tee "$LOG/runs_${tag}_verify.log"
import json
p = json.load(open("$OUT/runs/plan_W.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
for l in open("$OUT/runs/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify $tag: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
}
echo "== [$(date -u +%H:%M:%S)] CV 格"
$Y --plan cv > "$LOG/plan_cv.log" 2>&1; tail -1 "$LOG/plan_cv.log"
run83 cv
$Y --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$Y --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
run83 final
tar -chzf "$UP/out_acts_s3z.tar.gz" $OUT/trainsum.json $OUT/search $OUT/runs/plan_W.json $OUT/runs/runs.jsonl $OUT/select.json "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3z.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3z done"
