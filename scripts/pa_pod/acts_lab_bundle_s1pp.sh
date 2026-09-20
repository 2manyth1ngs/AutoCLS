#!/usr/bin/env bash
# WP17-ACTS S1″ → 实验室 8k-4090（部署路径固定 /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：
# 代码（HEAD）+ 输入（S1″ 协议 / tasks / plan_W / catalog_W（+ 若已生成的 plan_C / catalog.json）+ S1 precheck + probe 目录）
# + 4 个新外部源 npz（PTB-XL / SleepEDF / PAMAP2 / Opportunity，含 .json 元数据）。旧 9 源的 npz 服务器已有（S1′ setup）。
#   bash scripts/pa_pod/acts_lab_bundle_s1pp.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/acts/s1pp/PROTOCOL.md results/acts/s1pp/tasks.json results/acts/s1pp/plan_W.json results/acts/s1pp/catalog_W.json results/acts/s1/precheck.json results/route_a_probe/probe_cls_catalog.json"
for f in results/acts/s1pp/plan_C.json results/acts/s1pp/catalog.json; do [ -e "$f" ] && INPUTS="$INPUTS $f"; done
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
NPZ=""
for d in PTB-XL SleepEDF PAMAP2 Opportunity; do
  for ext in npz json; do f="data/candidates/$d.$ext"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; NPZ="$NPZ $f"; done
done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_inputs.tar.gz" $INPUTS
tar --force-local -czf "$B/acts_npz.tar.gz" $NPZ
: > "$B/md5_npz.txt"
for f in $NPZ; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_npz.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
