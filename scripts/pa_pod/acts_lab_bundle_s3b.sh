#!/usr/bin/env bash
# WP17-ACTS S3-B → 实验室 8k-4090（部署路径 /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：
# 代码（HEAD）+ 输入（S2 tasks / analysis_S、S3-B PROTOCOL / targets.npz）+ 19 源数据 md5 清单（数据服务器已有，自 S2）。
#   bash scripts/pa_pod/acts_lab_bundle_s3b.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/acts/s2/tasks.json results/acts/s2/analysis_S.json results/acts/s3b/PROTOCOL.md results/acts/s3b/targets.npz results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
DATA=""
for d in Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration HandMovementDirection SelfRegulationSCP1 SelfRegulationSCP2; do
  f="data/uea_cache/$d/${d}_merged.npz"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; DATA="$DATA $f"
done
for d in UniMiB-SHAR PTB-XL SleepEDF PAMAP2 NonInvasiveFetalECGThorax1 ShapesAll LargeKitchenAppliances Adiac FordA SemgHandMovementCh2 CinCECGTorso; do
  f="data/candidates/$d.npz"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; DATA="$DATA $f"
done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_s3b_inputs.tar.gz" $INPUTS
: > "$B/md5_data.txt"
for f in $DATA; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_data.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
