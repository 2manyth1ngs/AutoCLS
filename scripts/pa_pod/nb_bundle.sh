#!/usr/bin/env bash
# WP16 NB1 检验上云（5090）：本地打包（Git Bash）。pod 上跑：76 --rprime（870 条邻域轨迹，3 进程分数据集）→ 合并 → --verify → 打包。
#   bash scripts/pa_pod/nb_bundle.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/tclsc_nb/nb_catalog.json results/tclsc_nb/PROTOCOL.md results/mx3_pq/rprime_cache.jsonl results/mx3_pq/noise_probe.jsonl results/mx3_pq/delta_calibration.json \
results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json results/tclsc_slices/tasks.json results/tclsc_slices/cls_catalog.json results/route_a_probe/probe_cls_catalog.json \
results/tclsc_k3/labels_k3.jsonl data/candidates/UniMiB-SHAR.npz"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/nb_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
