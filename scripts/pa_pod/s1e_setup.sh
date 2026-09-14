#!/usr/bin/env bash
# WP16-TCLSC 阶段 1 真 E 臂：pod 侧一次性准备。
#   bash /workspace/wp/CLS/scripts/pa_pod/s1e_setup.sh /workspace/upload_s1e
# 依赖 pin → 解包（--no-same-owner、转 LF）→ md5 对账 → 8 源数据集就位（卷上有则 md5 对账，缺则解 pq_data.tar.gz）
# → 变体目录 results/tclsc_s1e/binned 只复制 mx3_pq 输入 → CUDA 闸 → 单测子集。
# 退出码：3 = 数据缺 / md5 不对；4 = /workspace 空闲 < 6 GB（折特征 ≈ 4 GB）；6 = CUDA 不可用。
set -euo pipefail
UP="${1:-/workspace/upload_s1e}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
}
echo "== torch fingerprint (before)"; tfp
python - <<'PY' || NEED=1
import scipy, yaml, sklearn, pytest, joblib, aeon  # noqa
print("deps present: scipy", scipy.__version__, "sklearn", sklearn.__version__, "aeon", aeon.__version__)
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" 2>&1 | tail -3
fi
echo "== torch fingerprint (after)"; tfp

tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/data_s1e.tar.gz" -C "$REPO"
find results/mx3_pq results/route_a_probe results/tclsc_s1e results/tclsc_s1phi -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'

df -h /workspace
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 1 ] || { echo "FREE ${AVAIL_GB:-?} GB < 1 GB on /workspace"; exit 4; }
AVAIL_ROOT=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
[ "${AVAIL_ROOT:-0}" -ge 6 ] || { echo "FREE ${AVAIL_ROOT:-?} GB < 6 GB on /（折特征 ≈ 2.5 GB 放容器盘）"; exit 4; }

# 8 源数据集：卷上已有且 md5 对得上 → 跳过；否则解上传包
missing=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ] || [ "$(md5sum "$f" | cut -d' ' -f1)" != "$sum" ]; then missing=1; fi
done < "$UP/md5_pqdata_local.txt"
if [ $missing = 1 ]; then
  [ -f "$UP/pq_data.tar.gz" ] || { echo "数据集缺且 $UP/pq_data.tar.gz 不在：先上传"; exit 3; }
  echo "== unpack pq_data.tar.gz"; tar --no-same-owner -xzf "$UP/pq_data.tar.gz" -C "$REPO"
fi
fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING pqdata $f"; fail=1; continue; fi
  [ "$(md5sum "$f" | cut -d' ' -f1)" = "$sum" ] || { echo "MISMATCH pqdata $f"; fail=1; }
done < "$UP/md5_pqdata_local.txt"
echo "== pq data md5: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"; [ $fail = 0 ] || exit 3

fail=0
for list in md5_text_local.txt md5_data_local.txt; do
  while read -r sum f; do
    [ -n "$f" ] || continue
    if [ ! -f "$f" ]; then echo "MISSING $f"; fail=1; continue; fi
    [ "$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)" = "$sum" ] || { echo "MISMATCH $f"; fail=1; }
  done < "$UP/$list"
done
echo "== code/data md5: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"; [ $fail = 0 ] || exit 3

# 变体目录放容器盘（30 GB，折特征 ≈ 2.5 GB；网络卷只 10 GB）：只复制 mx3_pq 输入（56 号 prepare_variant_dir）
OUT="${OUT:-/root/s1e/binned}"
mkdir -p "$OUT"
OUT="$OUT" python - <<'PY'
import importlib, os, sys
sys.path.insert(0, ".")
from pathlib import Path
g56 = importlib.import_module("scripts.56_tclsc_l1")
print("prepare:", g56.prepare_variant_dir(Path("results/mx3_pq"), Path(os.environ["OUT"])))
PY
df -h / | tail -1
# 旧折特征（MX-3-PQ 留在卷上的 /workspace/wp/CLS/results/mx3_pq/folds）若在，报告但不复用——协议要求折内 TS2Vec 现算（20 s/折）
ls results/mx3_pq/folds 2>/dev/null | head -3 || true

python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 6)" || { echo "CUDA 不可用"; exit 6; }
OMP_NUM_THREADS=1 python -m pytest tests/test_tclsc_s1e.py tests/test_tclsc_l3.py tests/test_tclsc_model.py -q -p no:cacheprovider 2>&1 | tail -2
echo "== setup done"
