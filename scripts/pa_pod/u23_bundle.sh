#!/usr/bin/env bash
# WP16 U 检验上云（4090）：本地打包（Git Bash）。只上云 finals（新冠军 × seeds {1,2}，官方口径），特征 / 排名 / SH 模拟在本地已完成。
#   bash scripts/pa_pod/u23_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD）、u23_inputs.tar.gz（results/tclsc_s1et_uea30/{picks.json,candidates.json}）、
#       md5_uea_ts.txt（picks 里涉及的靶 <Name>_{TRAIN,TEST}.ts 原始字节 md5，pod 侧 uea30_fetch.py 对账 / 缺则 Zenodo 直下）、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /d/Anaconda/envs/pytorch/python.exe ]; then PY=/d/Anaconda/envs/pytorch/python.exe; else PY=python; fi
fi
mkdir -p "$B"
[ -f results/tclsc_s1et_uea30/picks.json ] || { echo "MISSING results/tclsc_s1et_uea30/picks.json（先 72 --simulate）"; exit 2; }
[ -f results/tclsc_s1et_uea30/candidates.json ] || { echo "MISSING results/tclsc_s1et_uea30/candidates.json"; exit 2; }
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/u23_inputs.tar.gz" results/tclsc_s1et_uea30/picks.json results/tclsc_s1et_uea30/candidates.json
TARGETS=$("$PY" - <<'PY'
import json
p = json.load(open("results/tclsc_s1et_uea30/picks.json", encoding="utf-8"))
print(" ".join(sorted(d for d, v in p.items() if v)))
PY
)
echo "== finals targets ($(echo $TARGETS | wc -w)): $TARGETS"
: > "$B/md5_uea_ts.txt"
for d in $TARGETS; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
