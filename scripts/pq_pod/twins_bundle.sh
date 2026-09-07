#!/usr/bin/env bash
# MX-4 WP6 S1b+S2 上云（5090）：本地打包（仓库根目录 = wp6-mask 工作树，Git Bash）。
#   bash scripts/pq_pod/twins_bundle.sh <bundle_dir> [--with-uea]
# 产物：code_head.tar（git archive HEAD）、data_twins.tar.gz（mx3_pq 缓存 + 探针目录 + 基线缓存 + 定义漂移参照 +
#       UniMiB 注册表 + mx4_mask 协议）、可选 data_uea.tar.gz（7 个 UEA 缓存，卷上已有则不打）、md5 清单、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"; WITH_UEA="${2:-}"
mkdir -p "$B"
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/tasks.json \
results/mx3_pq/cls_catalog.json results/route_a_probe/probe_cls_catalog.json \
results/mx4_mask/PROTOCOL.md results/mx4_deploy/runs.jsonl \
data/candidates/UniMiB-SHAR.npz data/candidates/UniMiB-SHAR.json \
results/labels_workpoint/tasks.json"
OPT="results/baseline_probe/baseline_cache.jsonl results/candidate_survey/baseline/baseline_cache.jsonl \
results/stage1_rerun_20260804/results/step0_snr_gate/noise_cache.jsonl"
for f in $OPT; do [ -f "$f" ] && DATA="$DATA $f"; done
DIRS=""
[ -d results/stage1_rerun_20260804/results/step0_snr_gate/tasks_meta.d ] && DIRS="results/stage1_rerun_20260804/results/step0_snr_gate/tasks_meta.d"
tar --force-local -czf "$B/data_twins.tar.gz" $DATA $DIRS
if [ "$WITH_UEA" = "--with-uea" ]; then
  tar --force-local -czf "$B/data_uea.tar.gz" data/uea_cache/Handwriting data/uea_cache/PhonemeSpectra \
      data/uea_cache/UWaveGestureLibrary data/uea_cache/EthanolConcentration data/uea_cache/Cricket \
      data/uea_cache/HandMovementDirection data/uea_cache/SelfRegulationSCP2
fi
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $DATA; do
  case "$f" in
    *.npz|*.pt|*.ts|*.gz) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  printf '%s  %s\n' "$h" "$f" >> "$B/md5_data_local.txt"
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
