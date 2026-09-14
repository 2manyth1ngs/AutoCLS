#!/usr/bin/env bash
# WP16-TCLSC 阶段 1 真 E 臂上云：本地打包（仓库根目录，Git Bash）。
#   bash scripts/pa_pod/s1e_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD：autocls scripts configs tests）、data_s1e.tar.gz（mx3_pq 四件 + 探针目录 + 本协议）、
#       pq_data.tar.gz（8 源数据集：7 UEA .ts + UniMiB npz/json ≈ 200 MB；pod 卷已有则 setup 按 md5 跳过）、
#       md5_text_local.txt / md5_data_local.txt / md5_pqdata_local.txt、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json \
results/route_a_probe/probe_cls_catalog.json results/tclsc_s1e/PROTOCOL.md results/tclsc_s1phi/analysis.json"
for f in $DATA; do [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; done
tar --force-local -czf "$B/data_s1e.tar.gz" $DATA
PQDATA=""
for d in Handwriting UWaveGestureLibrary PhonemeSpectra EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
  for s in TRAIN TEST; do f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; PQDATA="$PQDATA $f"; done
done
for f in data/candidates/UniMiB-SHAR.npz data/candidates/UniMiB-SHAR.json; do [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; PQDATA="$PQDATA $f"; done
tar --force-local -czf "$B/pq_data.tar.gz" $PQDATA
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $DATA; do printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_data_local.txt"; done
: > "$B/md5_pqdata_local.txt"
for f in $PQDATA; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_pqdata_local.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
