#!/usr/bin/env bash
# WP17-ACTS UCR-ADM → 实验室 8k-4090（部署路径固定 /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：
# 代码（HEAD）+ 输入（协议 / tasks / plan_M / catalog）+ 10 个 UCR 源 npz（含 .json 元数据）+ md5。
#   bash scripts/pa_pod/acts_lab_bundle_ucradm.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
OUT="results/acts/ucradm"
INPUTS="$OUT/PROTOCOL.md $OUT/tasks.json $OUT/plan_M.json $OUT/catalog.json results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
NPZ=""
for d in NonInvasiveFetalECGThorax1 CinCECGTorso SemgHandMovementCh2 EOGHorizontalSignal ShapesAll MixedShapesSmallTrain LargeKitchenAppliances Adiac FiftyWords FordA; do
  for ext in npz json; do f="data/candidates/$d.$ext"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; NPZ="$NPZ $f"; done
done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_inputs.tar.gz" $INPUTS
tar --force-local -czf "$B/acts_npz.tar.gz" $NPZ
: > "$B/md5_npz.txt"
for f in $NPZ; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_npz.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
