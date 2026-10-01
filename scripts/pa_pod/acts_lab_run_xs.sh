#!/usr/bin/env bash
# WP17-ACTS XS 全自动流水（实验室 8k-4090；他人占卡；同时在跑的 83 / GPU 进程 ≤ MAXP，用户 2026-10-01 指定四卡并行）：
#   解包 → 数据 / 输入核对 → 输出 symlink 到根盘 /tmp/acts_xs_out → pytest →
#   A 准入：xsadm 切片 + plan_M（与本地 plan 键集逐位核对）→ fill M → 探针 → 83 --analyze-adm
#   → 116 --select（无选中 → 打包退出）→ B 补全：116 --prep-s → fill S → 探针 → 116 --gate（无保留 → 打包退出）
#   → 116 --labels → E_pt（MAXP 路并行）→ pack → E_A → score-new → merge → fit
#   → 115 --search ME / UNI × 6 → plan cv → fill W → select → plan final → fill W → 打包 upload/out_acts_xs.tar.gz。
#   fill = 接管式补跑：每轮清空 oom_failures.jsonl（83 剔除 OOM ≥ 2 次的格）、未完成任务按空闲显存分卡、直到全部完成。
#   MAXP=4 setsid -f bash .../scripts/pa_pod/acts_lab_run_xs.sh > /tmp/acts_xs_out/logs/run.log 2>&1 < /dev/null
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_xs_out"; LOG="$T/logs"; UP="$REPO/upload"
MODELS="/tmp/acts_s3z_out/models"; MAXP="${MAXP:-4}"
ADM="results/acts/xsadm"; XS="results/acts/xs"; MD="results/acts/xs/md"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
tar --no-same-owner -xf "$UP/code_head_xs.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_xs_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD_xs.txt")"
for n in ArticularyWordRecognition ERing FingerMovements Heartbeat NATOPS PEMS-SF; do
  [ -f "data/uea_cache/$n/${n}_merged.npz" ] || { echo "!! 数据缺 $n"; exit 3; }; done
for f in results/acts/s1pp/catalog.json results/acts/s1pp/labels.jsonl results/acts/s1pp/posthoc_s1pp.json results/acts/s2/labels.jsonl results/acts/s2/analysis_S.json \
         results/acts/s3c/data.npz results/acts/s3r/labels.npz results/acts/s3b/E_pt/_means.npz results/acts/s3n/ea.npz \
         results/acts/s3md/s0src.npz results/acts/s3md/ea_targets.npz results/acts/s3md/runs/runs.jsonl results/acts/s3y/meta.json results/acts/s3x/pool.json; do
  [ -e "$f" ] || { echo "!! 输入缺 $f"; exit 3; }; done
nb=$(ls $MODELS/B_s[0-9]_*.pt 2>/dev/null | wc -l); echo "== S3-Z NEW_B 模型 $nb / 60"; [ "$nb" -ge 60 ] || exit 3
for d in "$ADM" "$XS"; do                                   # 语料目录整体放根盘（共享盘会写满）
  name=$(basename "$d"); mkdir -p "$T/$name"
  if [ -e "$d" ] && [ ! -L "$d" ]; then cp -r "$d/." "$T/$name/" && rm -rf "$d"; fi; [ -L "$d" ] || ln -s "$T/$name" "$d"
done
python -m pytest tests/test_acts_xs.py tests/test_acts_s3md.py tests/test_acts_s3me.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
top_gpus() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $2-$3, $1}' | sort -rn | head -n $1 | awk '{print $2}'; }
fill() {   # $1 标签 $2 outdir $3 phase $4 profile：直到 plan_<phase> 全部完成
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
probe() {  # $1 outdir $2 phase $3 profile
  CUDA_VISIBLE_DEVICES=$(top_gpus 1) nice -n 10 python scripts/83_acts_corpus.py --resume --device cuda --profile $3 --outdir $1 --phase $2 --probe0 > "$LOG/probe_$2.log" 2>&1
  tail -1 "$LOG/probe_$2.log"; }
pack() {
  tar -chzf "$UP/out_acts_xs.tar.gz" $ADM/tasks.json $ADM/plan_M.json $ADM/catalog.json $ADM/runs.jsonl $ADM/probe0.jsonl $ADM/labels.jsonl $ADM/analysis_M.json $ADM/tables_M.md \
    $XS/selected.json $XS/tasks.json $XS/plan_S.json $XS/catalog.json $XS/runs.jsonl $XS/probe0.jsonl $XS/gate.json $XS/new.json $XS/new.npz $XS/ea_new.npz $XS/E_pt/_means.npz \
    $MD/s0new.npz $MD/s0src.npz $MD/experts.npz $MD/fit.json $MD/search $MD/runs/plan_W.json $MD/runs/runs.jsonl $MD/select.json "$LOG"/*.log 2>/dev/null
  ls -la "$UP/out_acts_xs.tar.gz"; }
X="python scripts/116_acts_xs.py"; D="python scripts/115_acts_s3md.py --outdir $MD --models $MODELS"
# ---------------- A 准入
echo "== [$(date -u +%H:%M:%S)] A：切片 + plan_M"
python scripts/83_acts_corpus.py --profile xsadm --outdir $ADM --slice-all --plan-adm > "$LOG/plan_M.log" 2>&1; tail -1 "$LOG/plan_M.log"
python - <<PY || exit 4
import json, hashlib
a = sorted(c["key"] for c in json.load(open("$ADM/plan_M.json"))["cells"]); b = sorted(c["key"] for c in json.load(open("$UP/xs_plan_M_local.json"))["cells"])
h = lambda x: hashlib.md5("|".join(x).encode()).hexdigest()
print(f"C0 plan_M 键集：服务器 {len(a)} {h(a)[:8]} / 本地 {len(b)} {h(b)[:8]} → {'OK' if a == b else 'MISMATCH'}"); raise SystemExit(0 if a == b else 4)
PY
fill adm $ADM M xsadm
probe $ADM M xsadm
python scripts/83_acts_corpus.py --profile xsadm --outdir $ADM --analyze-adm > "$LOG/analyze_adm.log" 2>&1; grep -h "准入" "$LOG/analyze_adm.log" | head -3
$X --select > "$LOG/select_src.log" 2>&1; tail -1 "$LOG/select_src.log"
python -c "import json,sys; sys.exit(0 if json.load(open('$XS/selected.json'))['selected'] else 9)" || { echo "== XS-NOSRC（无选中源）"; pack; exit 0; }
# ---------------- B 补全
echo "== [$(date -u +%H:%M:%S)] B：prep-s"
$X --prep-s > "$LOG/prep_s.log" 2>&1; tail -1 "$LOG/prep_s.log"
fill xs $XS S xs
probe $XS S xs
$X --gate > "$LOG/gate.log" 2>&1; cat "$LOG/gate.log"
python -c "import json,sys; sys.exit(0 if json.load(open('$XS/gate.json'))['kept'] else 9)" || { echo "== XS-NOSRC（8 行门后无保留）"; pack; exit 0; }
# ---------------- 并入
echo "== [$(date -u +%H:%M:%S)] labels / E_pt / E_A / score-new / merge / fit"
$X --labels > "$LOG/labels.log" 2>&1; tail -1 "$LOG/labels.log"
mapfile -t G < <(top_gpus $MAXP); pids=()
for ((i=0; i<${#G[@]}; i++)); do CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $X --extract-pt --parts ${#G[@]} --part $i --device cuda > "$LOG/extract_pt_$i.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
$X --pack-pt > "$LOG/pack_pt.log" 2>&1; tail -1 "$LOG/pack_pt.log"
$X --ea > "$LOG/ea.log" 2>&1; tail -1 "$LOG/ea.log"
CUDA_VISIBLE_DEVICES=$(top_gpus 1) $X --score-new --models $MODELS --device cuda > "$LOG/score_new.log" 2>&1; tail -1 "$LOG/score_new.log"
$X --merge > "$LOG/merge.log" 2>&1; tail -1 "$LOG/merge.log"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $X --fit > "$LOG/fit.log" 2>&1 || { echo "!! fit 失败"; tail -20 "$LOG/fit.log"; pack; exit 8; }
grep "\[fit\]" "$LOG/fit.log"
# ---------------- 部署
mkdir -p "$T/md_search" "$T/md_runs"
for d in search runs; do if [ -e "$MD/$d" ] && [ ! -L "$MD/$d" ]; then cp -r "$MD/$d/." "$T/md_$d/" && rm -rf "$MD/$d"; fi; [ -L "$MD/$d" ] || ln -s "$T/md_$d" "$MD/$d"; done
echo "== [$(date -u +%H:%M:%S)] 搜索 12"
J=(); for a in ME UNI; do for f in Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1; do J+=("$a $f"); done; done
for ((j=0; j<${#J[@]}; j+=MAXP)); do
  mapfile -t G < <(top_gpus $MAXP); pids=()
  for ((i=0; i<MAXP && j+i<${#J[@]}; i++)); do set -- ${J[$((j+i))]}
    CUDA_VISIBLE_DEVICES=${G[$i]} nice -n 10 $D --search --arm $1 --dataset $2 --device cuda --resume >> "$LOG/search.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
done
grep "\[search\]" "$LOG/search.log" | sort; echo "== 搜索完：$(ls $MD/search/*.json 2>/dev/null | wc -l) / 12"
echo "== [$(date -u +%H:%M:%S)] CV 格"
$D --plan cv > "$LOG/plan_cv.log" 2>&1; tail -1 "$LOG/plan_cv.log"
fill cv $MD/runs W s1pp
$D --select > "$LOG/select.log" 2>&1; tail -1 "$LOG/select.log"
echo "== [$(date -u +%H:%M:%S)] 最终格"
$D --plan final > "$LOG/plan_final.log" 2>&1; tail -1 "$LOG/plan_final.log"
fill final $MD/runs W s1pp
pack
echo "== [$(date -u +%H:%M:%S)] acts xs done"
