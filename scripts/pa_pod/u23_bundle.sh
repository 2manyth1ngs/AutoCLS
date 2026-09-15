#!/usr/bin/env bash
# WP16 U 检验上云（4090）：本地打包（Git Bash）。pod 上跑：72 --features（23 靶官方 TRAIN 探针 E_T）→ 72 --simulate → 45 --finals。
#   bash scripts/pa_pod/u23_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD）、u23_inputs.tar.gz（all8 模型 6 个 .pt + results/uea30/{runs.jsonl,candidates.json,analysis.json}
#       + results/tclsc_s1et_uea30/PROTOCOL.md + results/mx3_pq 三件（72 --features 不用，--simulate 只读 uea30；带上以防 45 号导入期读）、
#       md5_uea_ts.txt（23 靶 <Name>_{TRAIN,TEST}.ts 原始字节 md5，pod 侧 uea30_fetch.py 对账 / 缺则 Zenodo 直下）、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /d/Anaconda/envs/pytorch/python.exe ]; then PY=/d/Anaconda/envs/pytorch/python.exe; else PY=python; fi
fi
mkdir -p "$B"
for s in 0 1 2; do for a in blind_diff_hard ft60_hard; do
  f="results/tclsc_s1et_uea30/models/all8/${a}_s$s.pt"; [ -f "$f" ] || { echo "MISSING $f（先 72 --train-all8）"; exit 2; }
done; done
INPUTS="results/tclsc_s1et_uea30/models/all8 results/tclsc_s1et_uea30/PROTOCOL.md \
results/uea30/runs.jsonl results/uea30/candidates.json results/uea30/analysis.json \
results/mx3_pq/rprime_cache.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/u23_inputs.tar.gz" $INPUTS
TARGETS=$("$PY" - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/uea30.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"
: > "$B/md5_uea_ts.txt"
for d in $TARGETS; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
