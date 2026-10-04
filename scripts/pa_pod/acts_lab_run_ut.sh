#!/usr/bin/env bash
# WP17-ACTS UT 全自动流水（实验室 8k-4090；他人占卡；同时在跑的 GPU 进程 ≤ MAXP）：
#   解包 → 输入核对 → 输出 symlink 到根盘 /tmp/acts_ut_out → pytest →
#   靶 E_pt（MAXP 路）→ E_A → s0src → 成员模型链接 → fit → 115 搜索 ME / UNI / BLIND × 11 → plan cv → fill W → select
#   → plan final → fill W → 打包 upload/out_acts_ut.tar.gz。
#   MAXP=4 setsid -f bash .../scripts/pa_pod/acts_lab_run_ut.sh > /tmp/acts_ut_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_ut_out"; LOG="$T/logs"; UP="$REPO/upload"
MODELS="/tmp/acts_s3z_out/models"; MAXP="${MAXP:-4}"
UT="results/acts/ut"; MD="results/acts/ut/md"
TARGETS="AtrialFibrillation BasicMotions DuckDuckGeese Epilepsy FaceDetection FingerMovements HandMovementDirection Heartbeat MotorImagery SelfRegulationSCP2 StandWalkJump"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export AUTOCLS_EVAL_TIMEOUT_S=1800                                       # FaceDetection（TRAIN 5,890）SVM 网格在占卡服务器上可能 > 480 s
tar --no-same-owner -xf "$UP/code_head_ut.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_ut_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_ut.txt")"
for n in $TARGETS; do [ -f "data/uea_cache/$n/${n}_merged.npz" ] || { echo "!! 数据缺 $n"; exit 3; }; done
for f in $MD/meta.json results/acts/xs/md/s0src.npz results/acts/xs/new.json results/acts/xs/new.npz results/acts/xs/E_pt/_means.npz results/acts/xs/ea_new.npz \
         results/acts/s3c/data.npz results/acts/s3r/labels.npz results/acts/s3b/E_pt/_means.npz results/acts/s3n/ea.npz; do
  [ -e "$f" ] || { echo "!! 输入缺 $f"; exit 3; }; done
nb=$(ls $MODELS/B_s[0-9]_*.pt 2>/dev/null | wc -l); echo "== S3-Z NEW_B 模型 $nb / 60"; [ "$nb" -ge 60 ] || exit 3
mkdir -p "$T/ut"; if [ -e "$UT" ] && [ ! -L "$UT" ]; then cp -r "$UT/." "$T/ut/" && rm -rf "$UT"; fi; [ -L "$UT" ] || ln -s "$T/ut" "$UT"
python -m pytest tests/test_acts_ut.py tests/test_acts_xs.py tests/test_acts_s3md.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
top_gpus() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2-$3, $1}' | sort -rn | head -n $1 | awk '{print $2}'; }
fill() {   # $1 标签 $2 outdir $3 phase $4 profile：直到 plan_<phase> 全部完成（每轮清 OOM 记录、按空闲显存分卡）
  local tag=$1 od=$2 ph=$3 prof=$4 pass=0
  local R83="python scripts/83_acts_corpus.py --resume --device cuda --profile $prof --outdir $od --phase $ph"
  while :; do
    pass=$((pass+1)); [ $pass -gt 300 ] && { echo "!! $tag 超过 300 轮，停止补跑"; break; }
    mapfile -t ROWS < <(python scripts/pa_pod/acts_fill_tasks.py "$od" "$ph")
    [ ${#ROWS[@]} -eq 0 ] && break
    local n=$MAXP; [ $n -gt ${#ROWS[@]} ] && n=${#ROWS[@]}
    [ -f "$od/oom_failures.jsonl" ] && mv "$od/oom_failures.jsonl" "$T/oom_failures_${tag}_$pass.jsonl"
    mapfile -t G < <(top_gpus $n)
    local groups=(); for ((i=0; i<n; i++)); do groups[$i]=""; done
    for ((j=0; j<${#ROWS[@]}; j++)); do t=${ROWS[$j]%% *}; groups[$((j % n))]="${groups[$((j % n))]} $t"; done
    echo "== [$(date -u +%H:%M:%S)] $tag 轮 $pass：未完成 $(printf '%s\n' "${ROWS[@]}" | awk '{s+=$2} END {print s}') 格 / ${#ROWS[@]} 任务；卡 ${G[*]}"
    local pids=()
    for ((i=0; i<n; i++)); do
      CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $R83 --run --tasks ${groups[$i]} >> "$LOG/${tag}_g${G[$i]}.log" 2>&1 < /dev/null & pids+=($!); sleep 5
    done
    for p in "${pids[@]}"; do wait $p; done
    sleep 30
  done
  python - <<PY | tee "$LOG/${tag}_verify.log"
import json
p = json.load(open("$od/plan_$ph.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
for l in open("$od/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify $tag: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
}
pack() {
  tar -chzf "$UP/out_acts_ut.tar.gz" $UT/E $MD/meta.json $MD/s0src.npz $MD/ea_targets.npz $MD/experts.npz $MD/fit.json $MD/search \
    $MD/runs/plan_W.json $MD/runs/runs.jsonl $MD/select.json "$LOG"/*.log 2>/dev/null
  ls -la "$UP/out_acts_ut.tar.gz"; }
X="python scripts/118_acts_ut.py"
# ---------------- 靶 E
echo "== [$(date -u +%H:%M:%S)] 靶 E_pt（11 个，$MAXP 路）"
for r in 1 2 3; do                                           # 显存不足时整轮重来（已算的跳过）
  mapfile -t G < <(top_gpus $MAXP); pids=()
  for ((i=0; i<${#G[@]}; i++)); do CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $X --embed --parts ${#G[@]} --part $i --device cuda >> "$LOG/embed_$i.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  ne=$(ls $UT/E/*.npz 2>/dev/null | wc -l); echo "== E_pt 第 $r 轮后 $ne / 11"; [ "$ne" -ge 11 ] && break; sleep 60
done
[ "$(ls $UT/E/*.npz 2>/dev/null | wc -l)" -ge 11 ] || { echo "!! E_pt 未齐"; grep -h "Error" $LOG/embed_*.log | tail -5; pack; exit 8; }
$X --prep-ea > "$LOG/prep_ea.log" 2>&1; tail -2 "$LOG/prep_ea.log"
$X --s0 > "$LOG/s0.log" 2>&1; tail -1 "$LOG/s0.log"
$X --models --models-src $MODELS > "$LOG/models.log" 2>&1; tail -1 "$LOG/models.log"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $X --fit > "$LOG/fit.log" 2>&1 || { echo "!! fit 失败"; tail -20 "$LOG/fit.log"; pack; exit 8; }
grep "\[fit\]" "$LOG/fit.log"
# ---------------- 部署
mkdir -p "$T/md_search" "$T/md_runs"
for d in search runs; do if [ -e "$MD/$d" ] && [ ! -L "$MD/$d" ]; then cp -r "$MD/$d/." "$T/md_$d/" && rm -rf "$MD/$d"; fi; [ -L "$MD/$d" ] || ln -s "$T/md_$d" "$MD/$d"; done
echo "== [$(date -u +%H:%M:%S)] 搜索 33"
J=(); for a in ME UNI BLIND; do for f in $TARGETS; do J+=("$a $f"); done; done
for ((j=0; j<${#J[@]}; j+=MAXP)); do
  mapfile -t G < <(top_gpus $MAXP); pids=()
  for ((i=0; i<MAXP && j+i<${#J[@]}; i++)); do set -- ${J[$((j+i))]}
    CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $X --search --arm $1 --dataset $2 --device cuda --resume >> "$LOG/search.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
done
grep "\[search\]" "$LOG/search.log" | sort; echo "== 搜索完：$(ls $MD/search/*.json 2>/dev/null | wc -l) / 33"
echo "== [$(date -u +%H:%M:%S)] CV 格"
$X --plan cv > "$LOG/plan_cv.log" 2>&1; tail -1 "$LOG/plan_cv.log"
fill cv $MD/runs W s1pp
$X --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$X --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
fill final $MD/runs W s1pp
pack
echo "== [$(date -u +%H:%M:%S)] acts ut done"
