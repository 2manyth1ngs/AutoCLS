#!/usr/bin/env bash
# WP17-ACTS S1′ → 实验室 8k-4090（部署路径固定 /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：
# 代码（HEAD）+ 输入（S1′ 协议 / tasks / plan_A / catalog + S1 precheck（G-A0 沿用）+ probe 目录）+ 9 个候选源的 npz
# （8 个 UEA *_merged.npz + UniMiB-SHAR candidates npz）。
#   bash scripts/pa_pod/acts_lab_bundle_s1p.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/acts/s1p/PROTOCOL.md results/acts/s1p/tasks.json results/acts/s1p/plan_A.json results/acts/s1p/catalog.json results/acts/s1/precheck.json results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
NPZ="data/candidates/UniMiB-SHAR.npz"
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2 SelfRegulationSCP1; do
  f="data/uea_cache/$d/${d}_merged.npz"; [ -f "$f" ] || { echo "MISSING $f（先导出 npz）"; exit 2; }
  NPZ="$NPZ $f"
done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_inputs.tar.gz" $INPUTS
tar --force-local -czf "$B/acts_npz.tar.gz" $NPZ
: > "$B/md5_npz.txt"
for f in $NPZ; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_npz.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
