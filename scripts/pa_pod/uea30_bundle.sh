#!/usr/bin/env bash
# WP10-UEA30 上云（4090）：本地打包（仓库根目录 = wp10-uea30 工作树，Git Bash）。
#   bash scripts/pa_pod/uea30_bundle.sh <bundle_dir> [--with-uea]
# 产物：code_head.tar（git archive HEAD：autocls scripts configs tests）、data_uea30.tar.gz（mx3_pq 三件 + 探针目录 +
#       uea30 协议/候选）、md5_text_local.txt / md5_data_local.txt（去 CR 后 md5，同 wp6）、
#       md5_uea_ts.txt（23 靶 <Name>_{TRAIN,TEST}.ts 的**原始字节** md5，不去 CR；pod 侧 uea30_fetch.py / uea30_setup.sh 对账用）、
#       HEAD.txt。--with-uea 另打 uea_<Name>.tar.gz（FaceDetection / MotorImagery / PEMS-SF / DuckDuckGeese / EigenWorms
#       各一包）+ uea_small.tar.gz（其余 18 靶合一包）。默认不打：pod 侧 uea30_fetch.py 从 Zenodo 直下（同一记录，
#       Zenodo API 校验和与本地缓存逐位同，2026-09-06 实测 RacketSports / EigenWorms）。
# 23 个靶从 configs/uea30.yaml 读（run: true 按 order），不在 shell 里硬编码。
# Git Bash 里 `python` 是 WindowsApps 存根，读 yaml 默认走 Anaconda pytorch 环境（memory env-pytorch-python-path）；
# 可用 PYTHON=<解释器> 覆盖。
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
cfg = yaml.safe_load(open("configs/uea30.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"

git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
DATA="results/mx3_pq/rprime_cache.jsonl results/mx3_pq/rankings.json results/mx3_pq/cls_catalog.json \
results/route_a_probe/probe_cls_catalog.json \
results/uea30/PROTOCOL.md results/uea30/candidates.json"
for f in $DATA; do [ -f "$f" ] || { echo "MISSING $f（协议锁产物未落盘？）"; exit 2; }; done
tar --force-local -czf "$B/data_uea30.tar.gz" $DATA
: > "$B/md5_text_local.txt"
git ls-files autocls scripts configs tests | while read -r f; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_text_local.txt"
done
: > "$B/md5_data_local.txt"
for f in $DATA; do
  printf '%s  %s\n' "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" "$f" >> "$B/md5_data_local.txt"
done
# 23 靶 .ts 原始字节 md5（不去 CR：.ts 是数据不是文本源码，pod 侧直下 / 解包后逐位对账）
: > "$B/md5_uea_ts.txt"
for d in $TARGETS; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"
    [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$f" >> "$B/md5_uea_ts.txt"
  done
done
if [ "$WITH_UEA" = 1 ]; then
  BIG="FaceDetection MotorImagery PEMS-SF DuckDuckGeese EigenWorms"
  SMALL=""
  for d in $TARGETS; do
    case " $BIG " in
      *" $d "*) tar --force-local -czf "$B/uea_$d.tar.gz" "data/uea_cache/$d/${d}_TRAIN.ts" "data/uea_cache/$d/${d}_TEST.ts" ;;
      *) SMALL="$SMALL data/uea_cache/$d/${d}_TRAIN.ts data/uea_cache/$d/${d}_TEST.ts" ;;
    esac
  done
  tar --force-local -czf "$B/uea_small.tar.gz" $SMALL
fi
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
