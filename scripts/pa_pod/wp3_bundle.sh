#!/usr/bin/env bash
# MX-4 WP3 上云：本地打包（仓库根目录、Git Bash）。
#   bash scripts/pa_pod/wp3_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD：autocls scripts configs tests）、data_wp3.tar.gz（mx3_pq 三件 +
#       protocol_alignment runs + mx4_deploy 协议/候选 + 探针目录）、md5 清单、HEAD.txt。
# UEA 六个缓存已在卷 tudo4ntlot 的 /workspace/wp/CLS/data/uea_cache（wp3_setup.sh 会核对，缺则报错）。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/rankings.json results/mx3_pq/cls_catalog.json \
results/protocol_alignment/runs.jsonl results/mx4_deploy/PROTOCOL.md results/mx4_deploy/candidates.json \
results/route_a_probe/probe_cls_catalog.json"
tar --force-local -czf "$B/data_wp3.tar.gz" $DATA
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $DATA; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_data_local.txt"
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
