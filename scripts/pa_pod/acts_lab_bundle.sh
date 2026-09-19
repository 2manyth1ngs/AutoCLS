#!/usr/bin/env bash
# WP17-ACTS S1 → 实验室 8k-4090 服务器（用户 2026-09-19 拍板：选项 2，数据转 npz 上传绕开 aeon；部署路径固定
# /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：代码（HEAD）+ 输入（协议 / tasks / plan / catalog / precheck）
# + 3 个试点数据集的 *_merged.npz（由 `python -c "from autocls.data.uea import export_uea_merged_npz; ..."` 导出，与本地 load_uea_merged 逐位同）。
#   bash scripts/pa_pod/acts_lab_bundle.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/acts/s1/PROTOCOL.md results/acts/s1/tasks.json results/acts/s1/plan.json results/acts/s1/catalog.json results/acts/s1/precheck.json results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
NPZ=""
for d in PhonemeSpectra HandMovementDirection Cricket; do
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
