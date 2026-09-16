#!/usr/bin/env bash
# WP16 K3 检验上云（5090）：本地打包（Git Bash）。pod 上跑：75 --rprime（shared-10 × 87 切片 × rep {1,2}，3 进程分数据集）→ 合并 → --verify → 打包。
#   bash scripts/pa_pod/k3_bundle.sh <bundle_dir>
# 产物：code_head.tar、k3_inputs.tar.gz（mx3_pq 五件含 noise_probe + tclsc_slices/{tasks,cls_catalog} + 探针目录 + PROTOCOL + UniMiB npz）、md5_uea_ts.txt、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/noise_probe.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json \
results/tclsc_slices/tasks.json results/tclsc_slices/cls_catalog.json results/route_a_probe/probe_cls_catalog.json results/tclsc_k3/PROTOCOL.md data/candidates/UniMiB-SHAR.npz"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/k3_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
