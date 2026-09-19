#!/usr/bin/env bash
# WP17-ACTS S1 试点语料上云（5090）：本地打包（Git Bash）。pod 上跑：83 --run（1,008 条官方口径切片轨迹，2 进程按任务估时分片）
# → --probe0（72 次 iters=0 探针）→ --verify（不过补跑）→ 打包。分析在本地做。
#   bash scripts/pa_pod/acts_bundle.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/acts/s1/PROTOCOL.md results/acts/s1/tasks.json results/acts/s1/plan.json results/acts/s1/catalog.json results/acts/s1/precheck.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in PhonemeSpectra HandMovementDirection Cricket; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
