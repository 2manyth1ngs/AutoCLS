#!/usr/bin/env bash
# WP17-ACTS S3-C → 实验室 8k-4090。本地打包（Git Bash）：代码（HEAD）+ 输入（S3-C PROTOCOL / data.json / data.npz、S2 tasks / analysis_S）。
# 源数据与 S3-B 的 E 窗口（/tmp/acts_s3b_out）服务器已有，setup 只核对。
#   bash scripts/pa_pod/acts_lab_bundle_s3c.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"; mkdir -p "$B"
INPUTS="results/acts/s3c/PROTOCOL.md results/acts/s3c/data.json results/acts/s3c/data.npz results/acts/s2/tasks.json results/acts/s2/analysis_S.json results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_s3c_inputs.tar.gz" $INPUTS
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
