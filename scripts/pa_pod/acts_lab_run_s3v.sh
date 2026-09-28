#!/usr/bin/env bash
# WP17-ACTS S3-V 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3v_out）：
#   解包 → 核对输入（S3-C 数据 / E_pt 窗口 / labels.npz / S3-R′ 模型 36 / S3-C-D 靶端 E 6）→ symlink → pytest → 分槽 →
#   ① 原始窗口（CPU）② 三个消融臂 54 次训练 ③ 5 臂 × 6 集搜索 ④ D_val 选择格 → 83 phase W ⑤ select → 最终格 → 83 phase W → verify →
#   打包 upload/out_acts_s3v.tar.gz（search / runs / select / 消融臂 train npz / logs，不含权重）。
#   83 的各分片同时启动（S3-F 教训）。
#   GPUS="0 1 2 3 4 5 6 7" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3v.sh > /tmp/acts_s3v_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3v_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3v"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-6}"; SLOT_MB="${SLOT_MB:-2500}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3v_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
[ -f results/acts/s3c/data.npz ] && [ -f results/acts/s3c/data.json ] || { echo "!! S3-C 数据缺"; exit 3; }
npt=$(ls results/acts/s3b/E_pt/*.npz | grep -vc _means); echo "== E_pt $npt / 176"; [ "$npt" -ge 176 ] || exit 3
[ -f results/acts/s3r/labels.npz ] || { echo "!! labels.npz 缺"; exit 3; }
nm=$(ls results/acts/s3rp/models/*_s*_*.pt 2>/dev/null | wc -l); echo "== S3-R′ 模型 $nm / 36"; [ "$nm" -ge 36 ] || exit 3
ne=$(ls results/acts/s3cd/E/*.npz 2>/dev/null | wc -l); echo "== S3-C-D 靶端 E $ne / 6"; [ "$ne" -ge 6 ] || exit 3
[ -f "$OUT/meta.json" ] || { echo "!! meta.json 缺"; exit 3; }
for d in raw search runs abl_mean abl_raw abl_noshared; do
  mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
python -m pytest tests/test_acts_s3v.py tests/test_acts_s3r.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
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
run_jobs() {  # run_jobs <stage> <完整命令串...>：作业轮转到 NS 个槽
  local stage=$1; shift; local JOBS=("$@"); local pids=()
  for ((s=0; s<NS; s++)); do local g=${SLOTS[$s]}
    ( for ((j=s; j<${#JOBS[@]}; j+=NS)); do retry $g "$LOG/${stage}_s$s.log" ${JOBS[$j]}; done ) & pids+=($!); sleep 2; done
  for p in "${pids[@]}"; do wait $p; done; }
FOLDS="Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1"
V="python scripts/108_acts_s3v.py --outdir $OUT"
# ① 原始窗口（w/o TS2Vec 臂的输入）
echo "== [$(date -u +%H:%M:%S)] prep-raw"
$V --prep-raw --resume > "$LOG/prep_raw.log" 2>&1 || { echo "!! prep-raw 失败"; tail -5 "$LOG/prep_raw.log"; exit 4; }
tail -1 "$LOG/prep_raw.log"
# ② 三个消融臂训练（S3-R′ 制度：--stop acc）
echo "== [$(date -u +%H:%M:%S)] 训练 54 次"
TR="python scripts/105_acts_s3r.py --train --arm E --stop acc --device cuda --resume"
J=()
for sd in 0 1 2; do for f in $FOLDS; do
  J+=("$TR --fold $f --seed $sd --e-mode raw --raw-dir $OUT/raw --outdir $OUT/abl_raw")
  J+=("$TR --fold $f --seed $sd --e-mode mean --outdir $OUT/abl_mean")
  J+=("$TR --fold $f --seed $sd --no-shared --outdir $OUT/abl_noshared")
done; done
run_jobs train "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 训练完：raw $(ls $OUT/abl_raw/train/*.npz 2>/dev/null | wc -l) / mean $(ls $OUT/abl_mean/train/*.npz 2>/dev/null | wc -l) / noshared $(ls $OUT/abl_noshared/train/*.npz 2>/dev/null | wc -l)（各需 18）"
# ③ 搜索 5 臂 × 6 集
echo "== [$(date -u +%H:%M:%S)] 搜索 30"
J=(); for a in FULL WO_SETTF WO_TS2VEC WO_SHARED BLIND; do for f in $FOLDS; do J+=("$V --search --arm $a --dataset $f --device cuda --resume"); done; done
run_jobs search "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 搜索完：$(ls $OUT/search/*.json 2>/dev/null | wc -l) / 30"
# ④⑤ 83 phase W：先 D_val 选择格，再最终格
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
R83="python scripts/83_acts_corpus.py --resume --device cuda --profile s1pp --outdir $OUT/runs --phase W"
run83() {  # run83 <日志前缀>：各分片同时启动 + 单槽补跑 + verify
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
echo "== [$(date -u +%H:%M:%S)] D_val 选择格"
$V --plan val > "$LOG/plan_val.log" 2>&1; tail -1 "$LOG/plan_val.log"
run83 val
$V --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$V --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
run83 final
tar -chzf "$UP/out_acts_s3v.tar.gz" $OUT/search $OUT/runs/plan_W.json $OUT/runs/runs.jsonl $OUT/select.json $OUT/abl_raw/train $OUT/abl_mean/train $OUT/abl_noshared/train "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3v.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3v done"
