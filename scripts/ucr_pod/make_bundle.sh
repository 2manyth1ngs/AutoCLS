#!/usr/bin/env bash
# WP7-UCR 上云：本地打包（仓库根目录、Git Bash 里跑）。
#   bash scripts/ucr_pod/make_bundle.sh <bundle_dir>
# 产物：code_head.tar（git archive HEAD：autocls/scripts/configs/tests，LF）、
#       data_ucr.tar.gz（configs/ucr_pool.yaml 里 84 个数据集的 .ts 缓存 + shared 目录 + tasks.json）、
#       md5_text_local.txt / md5_data_local.txt、HEAD.txt。
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
PY="${PY:-D:/Anaconda/envs/pytorch/python.exe}"      # Git Bash 里裸 python 可能不是 pytorch 环境（无 yaml）
NAMES=$("$PY" - <<'PYEOF'
import yaml
cfg = yaml.safe_load(open("configs/ucr_pool.yaml", encoding="utf-8"))
print(" ".join(e["name"] for e in cfg["source_pool"]))
PYEOF
)
[ -n "$NAMES" ] || { echo "ucr_pool.yaml 名单为空（python/yaml 不可用？）"; exit 2; }
echo "datasets: $(echo $NAMES | wc -w)"
DATA="results/route_a_probe/probe_cls_catalog.json results/ucr_rprime/tasks.json"
for n in $NAMES; do DATA="$DATA data/candidates/raw/ucr_cache/$n"; done
tar --force-local -czf "$B/data_ucr.tar.gz" $DATA
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $(find $DATA -type f | sort); do
  case "$f" in
    *.ts) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  printf '%s  %s\n' "$h" "$f" >> "$B/md5_data_local.txt"
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
