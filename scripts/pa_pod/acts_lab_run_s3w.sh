#!/usr/bin/env bash
# WP17-ACTS S3-W 全自动流水（实验室 8k-4090；输出 symlink 到根盘 /tmp/acts_s3w_out）：
#   解包 → 核对输入 → symlink → pytest → 分槽 →
#   ① 保守式比较器 2 臂 × 3 α × 6 折 × 3 种子 = 108 次训练 ② 扩充池评估 8 组（含 S3-R′ 参照）③ α 选择 →
#   ④ 搜索 CONS / CONS_B × 6 集 ⑤ D_val 选择格 → select → 最终格 → verify → 打包 upload/out_acts_s3w.tar.gz。
#   GPUS="0 1 2 3 4 5 6 7" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3w.sh > /tmp/acts_s3w_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3w_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3w"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; RETRIES="${RETRIES:-6}"; SLOT_MB="${SLOT_MB:-2500}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3w_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
[ -f results/acts/s3c/data.npz ] && [ -f results/acts/s3c/data.json ] || { echo "!! S3-C 数据缺"; exit 3; }
npt=$(ls results/acts/s3b/E_pt/*.npz | grep -vc _means); echo "== E_pt $npt / 176"; [ "$npt" -ge 176 ] || exit 3
[ -f results/acts/s3r/labels.npz ] || { echo "!! labels.npz 缺"; exit 3; }
nm=$(ls results/acts/s3rp/models/*_s*_*.pt 2>/dev/null | wc -l); echo "== S3-R′ 模型 $nm / 36"; [ "$nm" -ge 36 ] || exit 3
ne=$(ls results/acts/s3cd/E/*.npz 2>/dev/null | wc -l); echo "== S3-C-D 靶端 E $ne / 6"; [ "$ne" -ge 6 ] || exit 3
for f in meta.json augpool.npz augpool.json; do [ -f "$OUT/$f" ] || { echo "!! $OUT/$f 缺"; exit 3; }; done
for d in a0.3 a1 a3 ext sel_E sel_B search runs; do
  mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
python -m pytest tests/test_acts_s3w.py tests/test_acts_s3r.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
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
W="python scripts/109_acts_s3w.py --outdir $OUT"
# ① 保守式比较器训练（S3-R′ 制度 + --cons α）
echo "== [$(date -u +%H:%M:%S)] 训练 108 次"
TR="python scripts/105_acts_s3r.py --train --stop acc --device cuda --resume"
J=()
for sd in 0 1 2; do for f in $FOLDS; do for a in 0.3 1 3; do for arm in E B; do
  J+=("$TR --arm $arm --fold $f --seed $sd --cons $a --outdir $OUT/a$a")
done; done; done; done
run_jobs train "${J[@]}"
for a in 0.3 1 3; do echo "== a$a：E $(ls $OUT/a$a/train/E_*.npz 2>/dev/null | wc -l) / B $(ls $OUT/a$a/train/B_*.npz 2>/dev/null | wc -l)（各需 18）"; done
echo "== [$(date -u +%H:%M:%S)] 训练完"
# ② 扩充池评估（含 α = 0 参照 = S3-R′ 模型）
J=("$W --eval-ext --src results/acts/s3rp --tag ref --arm E --device cuda --resume" "$W --eval-ext --src results/acts/s3rp --tag ref --arm B --device cuda --resume")
for a in 0.3 1 3; do for arm in E B; do J+=("$W --eval-ext --src $OUT/a$a --tag a$a --arm $arm --device cuda --resume"); done; done
run_jobs ext "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 扩充池评估：$(ls $OUT/ext/*.json 2>/dev/null | wc -l) / 8"
grep -h "eval-ext" $LOG/ext_s*.log | sort > "$LOG/ext_summary.log"
# ③ α 选择
$W --select-alpha > "$LOG/select_alpha.log" 2>&1; tail -1 "$LOG/select_alpha.log"
[ -f "$OUT/alpha.json" ] || { echo "!! α 选择失败"; tail -20 "$LOG/select_alpha.log"; exit 6; }
# 训练记录摘要（全部 108 次，供本地读数；不打包权重）
python - <<PY > "$LOG/trainsum.log" 2>&1
import json, glob, numpy as np
out = {}
for f in sorted(glob.glob("$OUT/a*/train/*.npz")):
    z = np.load(f); tag = f.split("/")[-3] + "/" + f.split("/")[-1][:-4]
    out[tag] = {"best_epoch": int(z["best_epoch"]), "hist": json.loads(str(z["hist"])), "cons_hist": json.loads(str(z["cons_hist"]))}
json.dump(out, open("$OUT/trainsum.json", "w"), ensure_ascii=False)
print(len(out), "个训练记录")
PY
cat "$LOG/trainsum.log"
# ④ 搜索
echo "== [$(date -u +%H:%M:%S)] 搜索 12"
J=(); for a in CONS CONS_B; do for f in $FOLDS; do J+=("$W --search --arm $a --dataset $f --device cuda --resume"); done; done
run_jobs search "${J[@]}"
echo "== [$(date -u +%H:%M:%S)] 搜索完：$(ls $OUT/search/*.json 2>/dev/null | wc -l) / 12"
# ⑤ 83 phase W：先 D_val 选择格，再最终格
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
$W --plan val > "$LOG/plan_val.log" 2>&1; tail -1 "$LOG/plan_val.log"
run83 val
$W --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$W --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
run83 final
tar -chzf "$UP/out_acts_s3w.tar.gz" $OUT/ext $OUT/alpha.json $OUT/trainsum.json $OUT/search $OUT/runs/plan_W.json $OUT/runs/runs.jsonl $OUT/select.json \
  $OUT/sel_E/train $OUT/sel_B/train "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3w.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3w done"
