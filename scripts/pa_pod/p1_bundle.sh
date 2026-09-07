#!/usr/bin/env bash
# WP9-P1 路径一上云（4090）：本地打包（仓库根目录 = wp9-p1 主目录，Git Bash）。
#   bash scripts/pa_pod/p1_bundle.sh <bundle_dir> [--with-uea]
# 产物：code_head.tar（git archive HEAD：autocls scripts configs tests）、data_p1.tar.gz（mx3_pq 三件 + 探针目录 +
#       WP3 candidates/runs/analysis + WP8 runs + WP6 runs + 本包 PROTOCOL/pool.json/c0.json + configs/p1.yaml）、
#       md5_text_local.txt / md5_data_local.txt（去 CR 后 md5）、md5_uea_ts.txt（6 靶 <Name>_{TRAIN,TEST}.ts 原始字节 md5）、HEAD.txt。
# --with-uea 另打 uea_p1.tar.gz（6 靶 .ts 合一包，≈ 数百 MB）；默认不打：pod 卷 tudo4ntlot 上 WP3/WP6/UEA30 已留 6 靶缓存，
# setup 用 md5 对账，缺则 uea30_fetch.py 从 Zenodo 直下。
# 6 个靶从 configs/p1.yaml 读（order 序），不在 shell 里硬编码。
set -euo pipefail
B="${1:?bundle dir}"
WITH_UEA=0
[ "${2:-}" = "--with-uea" ] && WITH_UEA=1
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x /d/Anaconda/envs/pytorch/python.exe ]; then PY=/d/Anaconda/envs/pytorch/python.exe; else PY=python; fi
fi
mkdir -p "$B"

TARGETS=$("$PY" - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["targets"].items())))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"

git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/rankings.json results/mx3_pq/cls_catalog.json \
results/route_a_probe/probe_cls_catalog.json \
results/mx4_deploy/candidates.json results/mx4_deploy/runs.jsonl results/mx4_deploy/analysis.json \
results/mx4_bo/runs.jsonl results/mx4_bo/stage0.json results/mx4_mask/runs.jsonl \
results/mx4_p1/PROTOCOL.md results/mx4_p1/pool.json results/mx4_p1/c0.json configs/p1.yaml"
for f in $DATA; do [ -f "$f" ] || { echo "MISSING $f（协议锁产物未落盘？）"; exit 2; }; done
tar --force-local -czf "$B/data_p1.tar.gz" $DATA
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $DATA; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_data_local.txt"
done
: > "$B/md5_uea_ts.txt"
UEA=""
for d in $TARGETS; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"
    [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_uea_ts.txt"
    UEA="$UEA $f"
  done
done
if [ "$WITH_UEA" = 1 ]; then
  tar --force-local -czf "$B/uea_p1.tar.gz" $UEA
fi
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
