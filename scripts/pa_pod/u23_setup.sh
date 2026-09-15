#!/usr/bin/env bash
# WP16 U 检验上云（4090）：pod 侧一次性准备——依赖按 pin、解包代码（--no-same-owner + LF）、解包输入（all8 模型 / uea30 目录 / mx3_pq）、
# 磁盘闸、23 靶 UEA .ts 就位（已在卷上且 md5 一致则跳过，否则 uea30_fetch.py 从 Zenodo 直下）、CUDA 闸、导入自检。
#   bash /workspace/wp/CLS/scripts/pa_pod/u23_setup.sh /workspace/upload_u23
# 退出码：3 = UEA 缓存缺 / md5 不对；4 = /workspace 空闲 < 4 GB；5 = CUDA 不可用；其它 = set -e 处的命令失败。
set -euo pipefail
UP="${1:-/workspace/upload_u23}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
python - <<'PY' || NEED=1
import scipy, yaml, sklearn, pytest, joblib, aeon, requests, psutil  # noqa
print("deps present")
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" requests psutil 2>&1 | tail -2
fi

tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/u23_inputs.tar.gz" -C "$REPO"
find results/uea30 results/tclsc_s1et_uea30 results/mx3_pq -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== all8 models: $(ls results/tclsc_s1et_uea30/models/all8 | wc -l)"
echo "== HEAD $(cat "$UP/HEAD.txt")"

df -h /workspace | tail -1
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 4 ] || { echo "FREE ${AVAIL_GB:-?} GB < 4 GB on /workspace"; exit 4; }

TARGETS=$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/uea30.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"
mkdir -p data/uea_cache
FETCH_FAIL=""
for d in $TARGETS; do
  python scripts/pa_pod/uea30_fetch.py "$d" --cache-dir data/uea_cache --md5 "$UP/md5_uea_ts.txt" || FETCH_FAIL="$FETCH_FAIL $d"
done
[ -z "$FETCH_FAIL" ] || { echo "!! fetch / 对账失败：$FETCH_FAIL"; exit 3; }
fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  g="data/uea_cache/$f"
  if [ ! -f "$g" ]; then echo "MISSING uea $g"; fail=1; continue; fi
  h=$(md5sum "$g" | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH uea $g"; fail=1; }
done < "$UP/md5_uea_ts.txt"
echo "== uea .ts md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ] || exit 3

python - <<'PY' || exit 5
import torch, sys
assert torch.cuda.is_available(), "CUDA 不可用（NVML 掉线？）"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK")
PY
OMP_NUM_THREADS=1 python -c "import importlib; importlib.import_module('scripts.45_uea30_deploy'); importlib.import_module('scripts.72_tclsc_s1et_uea30'); print('import 45 / 72 OK')"
echo "== setup done"
