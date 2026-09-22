#!/usr/bin/env bash
# WP17-ACTS S2-CORPUS → 实验室 8k-4090（部署路径固定 /home/b311/data/wangwenjie/kejunyu/AutoCLS）。本地打包（Git Bash）：
# 代码（HEAD）+ 输入（协议 / tasks / plan_S / catalog）+ 缓存种子（S1″ 与 UCR-ADM 的 runs / probe0）+ 19 源数据文件的 md5 清单
# （数据本身服务器已有：UEA *_merged.npz 自 S1 / S1′ / S1″，候选 npz 自 S1″ / UCR-ADM；setup 只核对，缺了再单独传）。
#   bash scripts/pa_pod/acts_lab_bundle_s2.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
OUT="results/acts/s2"
INPUTS="$OUT/PROTOCOL.md $OUT/tasks.json $OUT/plan_S.json $OUT/catalog.json results/route_a_probe/probe_cls_catalog.json"
SEEDS="results/acts/s1pp/runs.jsonl results/acts/s1pp/probe0.jsonl results/acts/ucradm/runs.jsonl results/acts/ucradm/probe0.jsonl"
for f in $INPUTS $SEEDS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
DATA=""
for d in Handwriting PhonemeSpectra UWaveGestureLibrary Cricket EthanolConcentration HandMovementDirection SelfRegulationSCP1 SelfRegulationSCP2; do
  f="data/uea_cache/$d/${d}_merged.npz"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; DATA="$DATA $f"
done
for d in UniMiB-SHAR PTB-XL SleepEDF PAMAP2 NonInvasiveFetalECGThorax1 ShapesAll LargeKitchenAppliances Adiac FordA SemgHandMovementCh2 CinCECGTorso; do
  f="data/candidates/$d.npz"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }; DATA="$DATA $f"
done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/acts_inputs.tar.gz" $INPUTS
tar --force-local -czf "$B/acts_seeds.tar.gz" $SEEDS
: > "$B/md5_data.txt"
for f in $DATA; do printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_data.txt"; done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
