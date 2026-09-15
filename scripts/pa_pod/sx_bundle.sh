#!/usr/bin/env bash
# WP16 SX 检验上云（5090）：本地打包（Git Bash）。pod 上跑：74 --slice（对账）→ --rprime（42 切片 × 21 条轨迹）→ --baseline → --features → --train S9/S12。
#   bash scripts/pa_pod/sx_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD）、sx_inputs.tar.gz（mx3_pq 四件 + 探针目录 + ctx_uea + tclsc_slices/{PROTOCOL,tasks,slice_check} + UniMiB npz）、
#       md5_uea_ts.txt（6 个 UEA 源数据集 <Name>_{TRAIN,TEST}.ts md5，pod 侧 uea30_fetch.py 对账 / 缺则 Zenodo 直下）、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json \
results/route_a_probe/probe_cls_catalog.json results/tclsc_s1et/uea/ctx_uea.jsonl \
results/tclsc_slices/PROTOCOL.md results/tclsc_slices/tasks.json results/tclsc_slices/slice_check.json data/candidates/UniMiB-SHAR.npz"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/sx_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
