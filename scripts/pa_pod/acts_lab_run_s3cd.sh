#!/usr/bin/env bash
# WP17-ACTS S3-C-D 部署口径（6 个 UEA 靶，官方 train / test）在实验室 8k-4090 跑；输出 symlink 到根盘 /tmp/acts_s3cd_out。
#   STAGE=A：解包代码 / 输入 → 核对 S3-C 模型 → symlink → pytest → embed（首卡）→ 搜索 A / D（12 作业分槽）→ plan-a → 83 phase W 真跑（分片）+ iters=0 探针 → 打包
#   STAGE=B：（本地 score-probe / plan-b 后已上传新 plan_W.json）83 phase W --resume 补跑 ④ 前 3 → 打包
#   STAGE=A GPUS="6 0 5 3" setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3cd.sh > /tmp/acts_s3cd_out/logs/run_A.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3cd_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3cd"
GPUS="${GPUS:-0 1}"; RETRIES="${RETRIES:-3}"; NP=$(echo $GPUS | wc -w); G0=$(echo $GPUS | cut -d" " -f1); STAGE="${STAGE:-A}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
retry() { local g=$1 lf=$2; shift 2
  for ((r=0; r<=RETRIES; r++)); do CUDA_VISIBLE_DEVICES=$g nice -n 10 "$@" >> "$lf" 2>&1 < /dev/null && return 0; echo "== [$(date -u +%H:%M:%S)] GPU $g retry #$((r+1)): $*" >> "$lf"; sleep 30; done
  echo "== FAILED after $RETRIES retries: $*" >> "$lf"; return 1; }
P="python scripts/90_acts_s3cd.py --outdir $OUT --device cuda --resume"
R83="python scripts/83_acts_corpus.py --resume --device cuda --profile s1pp --outdir $OUT/runs --phase W"
run83() {
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
  export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
  local i=0 pids=()
  for g in $GPUS; do ( retry $g "$LOG/runs_${STAGE}_p$i.log" $R83 --run --parts $NP --part $i ) & pids+=($!); i=$((i+1)); sleep 15; done
  for p in "${pids[@]}"; do wait $p; done
  retry $G0 "$LOG/runs_${STAGE}_fix.log" $R83 --run
  python - <<PY | tee "$LOG/verify_${STAGE}.log"
import json
p = json.load(open("$OUT/runs/plan_W.json")); keys = {c["key"] for c in p["cells"]}; done = set(); err = 0
for l in open("$OUT/runs/runs.jsonl"):
    r = json.loads(l); done.add(r["key"]); err += bool(r.get("error"))
print(f"verify: plan {len(keys)} / done {len(keys & done)} / error {err} → {'OK' if keys <= done else 'MISSING ' + str(len(keys - done))}")
PY
}
if [ "$STAGE" = A ]; then
  tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
  find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
  tar --no-same-owner -xzf "$UP/acts_s3cd_inputs.tar.gz" -C "$REPO"
  echo "== HEAD $(cat "$UP/HEAD.txt")"
  n=$(ls results/acts/s3c/models/{A,D}_*.pt 2>/dev/null | wc -l); echo "== S3-C models A/D: $n / 12"; [ "$n" -ge 12 ] || exit 3
  mkdir -p "$OUT"
  for d in E search runs; do mkdir -p "$T/$d"; if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi; [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"; done
  python -m pytest tests/test_acts_s3cd.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
  echo "== [$(date -u +%H:%M:%S)] embed"; retry $G0 "$LOG/embed.log" $P --embed
  echo "== [$(date -u +%H:%M:%S)] search"
  JOBS=(); for arm in A D; do for d in Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration SelfRegulationSCP1; do JOBS+=("$arm $d"); done; done
  pids=()
  for ((s=0; s<NP; s++)); do g=$(echo $GPUS | cut -d" " -f$((s+1)))
    ( for ((j=s; j<${#JOBS[@]}; j+=NP)); do set -- ${JOBS[$j]}; retry $g "$LOG/search_s$s.log" $P --search --arm $1 --dataset $2; done ) & pids+=($!); sleep 5; done
  for p in "${pids[@]}"; do wait $p; done
  echo "== [$(date -u +%H:%M:%S)] search done: $(ls $OUT/search/*.json 2>/dev/null | wc -l) / 12"
  python scripts/90_acts_s3cd.py --outdir $OUT --plan-a --device cpu > "$LOG/plan_a.log" 2>&1; tail -1 "$LOG/plan_a.log"
  echo "== [$(date -u +%H:%M:%S)] runs"; run83
  echo "== [$(date -u +%H:%M:%S)] probe0"; retry $G0 "$LOG/probe0.log" $R83 --probe0
else
  echo "== [$(date -u +%H:%M:%S)] runs (stage B)"; run83
fi
tar -chzf "$UP/out_acts_s3cd_${STAGE}.tar.gz" $OUT/search $OUT/runs/plan_W.json $OUT/runs/runs.jsonl $( [ -f $OUT/runs/probe0.jsonl ] && echo $OUT/runs/probe0.jsonl ) "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3cd_${STAGE}.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3cd stage $STAGE done"
