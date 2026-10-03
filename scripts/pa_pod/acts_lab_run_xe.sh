#!/usr/bin/env bash
# WP17-ACTS XE 全自动流水（实验室 8k-4090；他人占卡；同时在跑的 GPU 进程 ≤ MAXP）：
#   解包 → 输入核对 → 输出 symlink 到根盘 /tmp/acts_xe_out → pytest →
#   E_T（32 新任务 + 6 靶，MAXP 路）→ merge-et → select（XE-SAME → 打包退出）→ prep-md → fit
#   → 115 --search ME × 6 → plan cv → fill W → select → plan final → fill W → 打包 upload/out_acts_xe.tar.gz。
#   MAXP=4 setsid -f bash .../scripts/pa_pod/acts_lab_run_xe.sh > /tmp/acts_xe_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_xe_out"; LOG="$T/logs"; UP="$REPO/upload"
MODELS="/tmp/acts_s3z_out/models"; MAXP="${MAXP:-4}"
XE="results/acts/xe"; MD="results/acts/xe/md"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
tar --no-same-owner -xf "$UP/code_head_xe.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_xe_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_xe.txt")"
for f in results/acts/s3o/et.npz results/acts/xs/new.json results/acts/xs/new.npz results/acts/xs/tasks.json results/acts/xs/E_pt/_means.npz results/acts/xs/ea_new.npz \
         results/acts/xs/md/s0src.npz results/acts/xs/md/experts.npz results/acts/xs/md/fit.json results/acts/xs/md/runs/runs.jsonl results/acts/s3md/ea_targets.npz \
         results/acts/s2/labels.jsonl results/acts/s2/analysis_S.json results/acts/s2/catalog.json results/acts/s3y/meta.json results/acts/s3x/pool.json; do
  [ -e "$f" ] || { echo "!! 输入缺 $f"; exit 3; }; done
nu=$(ls results/acts/xs/md/search/UNI_*.json 2>/dev/null | wc -l); echo "== XS UNI 搜索 $nu / 6"; [ "$nu" -ge 6 ] || exit 3
nb=$(ls $MODELS/B_s[0-9]_*.pt 2>/dev/null | wc -l); echo "== S3-Z NEW_B 模型 $nb / 60"; [ "$nb" -ge 60 ] || exit 3
mkdir -p "$T/xe"; if [ -e "$XE" ] && [ ! -L "$XE" ]; then cp -r "$XE/." "$T/xe/" && rm -rf "$XE"; fi; [ -L "$XE" ] || ln -s "$T/xe" "$XE"
python -m pytest tests/test_acts_xe.py tests/test_acts_xs.py tests/test_acts_s3md.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
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
  tar -chzf "$UP/out_acts_xe.tar.gz" $XE/et $XE/et_new.npz $XE/select.json $XE/select.md $XE/form.npz $MD/s0src.npz $MD/experts.npz $MD/fit.json $MD/search \
    $MD/runs/plan_W.json $MD/runs/runs.jsonl $MD/select.json "$LOG"/*.log 2>/dev/null
  ls -la "$UP/out_acts_xe.tar.gz"; }
X="python scripts/117_acts_xe.py"; D="python scripts/115_acts_s3md.py --outdir $MD --models $MODELS"
# ---------------- E_T
echo "== [$(date -u +%H:%M:%S)] E_T（38 个，$MAXP 路）"
for r in 1 2 3; do                                           # 显存不足时整轮重来（--resume 跳过已算）
  mapfile -t G < <(top_gpus $MAXP); pids=()
  for ((i=0; i<${#G[@]}; i++)); do CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $X --compute-et --parts ${#G[@]} --part $i --device cuda >> "$LOG/et_$i.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  $X --merge-et > "$LOG/merge_et.log" 2>&1 && break; echo "== E_T 第 $r 轮未齐：$(tail -1 $LOG/merge_et.log)"; sleep 60
done
tail -1 "$LOG/merge_et.log"; [ -f "$XE/et_new.npz" ] || { echo "!! E_T 未齐"; pack; exit 8; }
# ---------------- 选形式
$X --select > "$LOG/select_form.log" 2>&1; cat "$LOG/select_form.log"
python -c "import json,sys; sys.exit(0 if json.load(open('$XE/select.json'))['code'] != 'XE-SAME' else 9)" || { echo "== XE-SAME：不部署"; pack; exit 0; }
# ---------------- 部署
$X --prep-md > "$LOG/prep_md.log" 2>&1; tail -1 "$LOG/prep_md.log"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $X --fit > "$LOG/fit.log" 2>&1 || { echo "!! fit 失败"; tail -20 "$LOG/fit.log"; pack; exit 8; }
grep "\[fit\]" "$LOG/fit.log"
mkdir -p "$T/md_search" "$T/md_runs"
for d in search runs; do if [ -e "$MD/$d" ] && [ ! -L "$MD/$d" ]; then cp -r "$MD/$d/." "$T/md_$d/" && rm -rf "$MD/$d"; fi; [ -L "$MD/$d" ] || ln -s "$T/md_$d" "$MD/$d"; done
echo "== [$(date -u +%H:%M:%S)] 搜索 ME × 6"
J=(Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1)
for ((j=0; j<${#J[@]}; j+=MAXP)); do
  mapfile -t G < <(top_gpus $MAXP); pids=()
  for ((i=0; i<MAXP && j+i<${#J[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $D --search --arm ME --dataset ${J[$((j+i))]} --device cuda --resume >> "$LOG/search.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
done
grep "\[search\]" "$LOG/search.log" | sort; echo "== 搜索完：$(ls $MD/search/ME_*.json 2>/dev/null | wc -l) / 6"
echo "== [$(date -u +%H:%M:%S)] CV 格"
$D --plan cv > "$LOG/plan_cv.log" 2>&1; tail -1 "$LOG/plan_cv.log"
fill cv $MD/runs W s1pp
$D --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$D --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
fill final $MD/runs W s1pp
pack
echo "== [$(date -u +%H:%M:%S)] acts xe done"
