#!/usr/bin/env bash
# 全量协议对齐上云：本地打包（在仓库根目录、Git Bash 里跑）。
#   bash scripts/pa_pod/make_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD，LF）、data_pa.tar.gz（6 个 UEA 缓存 + mx3_pq 的
#       rankings/cls_catalog + protocol_alignment 的 REPORT/picks）、md5_text_local.txt、md5_data_local.txt。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="data/uea_cache/Handwriting data/uea_cache/UWaveGestureLibrary data/uea_cache/PhonemeSpectra \
data/uea_cache/EthanolConcentration data/uea_cache/Cricket data/uea_cache/HandMovementDirection \
results/mx3_pq/rankings.json results/mx3_pq/cls_catalog.json \
results/protocol_alignment/REPORT.md results/protocol_alignment/picks.json"
tar -czf "$B/data_pa.tar.gz" $DATA
# 文本清单：按 LF 归一化 md5；数据清单：.ts 原样 md5、json/md 按 LF
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $(find $DATA -type f | sort); do
  case "$f" in
    *.ts) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  printf '%s  %s\n' "$h" "$f" >> "$B/md5_data_local.txt"
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
