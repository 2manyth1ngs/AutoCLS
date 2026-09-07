#!/usr/bin/env bash
# WP9-P1 路径一上云（4090）：pod 侧一次性准备（依赖按 pin、解包 --no-same-owner 转 LF、磁盘闸、6 靶 UEA 缓存就位
# （卷上已有 → md5 对账；缺 → 上传包或 uea30_fetch.py 直下）、code/data md5 对账、单测子集、--simulate --c0 逐位、冒烟、dry-run）。
#   bash /workspace/wp/CLS/scripts/pa_pod/p1_setup.sh /workspace/upload_p1
# 退出码：3 = UEA 缓存缺 / md5 不对；4 = /workspace 空闲 < 3 GB；5 = C0 不过；其它 = set -e 处的命令失败。
set -euo pipefail
UP="${1:-/workspace/upload_p1}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
}
echo "== torch fingerprint (before)"; tfp
python - <<'PY' || NEED=1
import scipy, yaml, sklearn, pytest, joblib, aeon, requests, psutil  # noqa
print("deps present: scipy", scipy.__version__, "sklearn", sklearn.__version__, "aeon", aeon.__version__,
      "requests", requests.__version__, "psutil", psutil.__version__)
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" requests psutil 2>&1 | tail -3
fi
echo "== torch fingerprint (after)"; tfp

tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/data_p1.tar.gz" -C "$REPO"
find results/mx3_pq results/route_a_probe results/mx4_deploy results/mx4_bo results/mx4_mask results/mx4_p1 configs -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.yaml' \) -print0 | xargs -0 sed -i 's/\r$//'

df -h /workspace
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 3 ] || { echo "FREE ${AVAIL_GB:-?} GB < 3 GB on /workspace"; exit 4; }

TARGETS=$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["targets"].items())))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"
[ -f "$UP/md5_uea_ts.txt" ] || { echo "MISSING $UP/md5_uea_ts.txt（bundle 产物）"; exit 3; }
mkdir -p data/uea_cache
if [ -f "$UP/uea_p1.tar.gz" ]; then echo "== unpack $UP/uea_p1.tar.gz"; tar --no-same-owner -xzf "$UP/uea_p1.tar.gz" -C "$REPO"; fi
FETCH_FAIL=""
for d in $TARGETS; do
  # 两文件已在且 md5 与清单一致 → fetch 跳过；否则 Zenodo 直下（.ts 是原始字节，绝不过 sed 转 LF）
  python scripts/pa_pod/uea30_fetch.py "$d" --cache-dir data/uea_cache --md5 "$UP/md5_uea_ts.txt" \
    || FETCH_FAIL="$FETCH_FAIL $d"
done
[ -z "$FETCH_FAIL" ] || echo "!! Zenodo 直下 / 对账失败：$FETCH_FAIL —— 本地 bash scripts/pa_pod/p1_bundle.sh <dir> --with-uea 后把 uea_p1.tar.gz 传到 $UP 再重跑本脚本"

fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING uea $f"; fail=1; continue; fi
  h=$(md5sum "$f" | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH uea $f"; fail=1; }
done < "$UP/md5_uea_ts.txt"
echo "== uea .ts md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ] || exit 3

fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING text $f"; fail=1; continue; fi
  h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH text $f"; fail=1; }
done < "$UP/md5_text_local.txt"
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING data $f"; fail=1; continue; fi
  h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH data $f"; fail=1; }
done < "$UP/md5_data_local.txt"
echo "== md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ]

# 单测限 OMP=1；代理金标（pred_z 1e-9）是本机浮点，pod 上 deselect；test_p1_pool 其余含 C0 金标（WP3 / 掩码扫臂逐位）
OMP_NUM_THREADS=1 python -m pytest tests/test_p1_pool.py tests/test_sh_deploy.py tests/test_cls_space.py -q \
    --deselect tests/test_p1_pool.py::test_c0_surrogate_reproduces_wp3_pred_z 2>&1 | tail -3
# C0：pod 上重算 c0.json 必须与本地协议锁产物逐位（先把本地版挪开再重算）
cp results/mx4_p1/c0.json results/mx4_p1/c0_local.json
OMP_NUM_THREADS=1 python scripts/46_p1_pool.py --simulate --c0 --outdir results/mx4_p1 2>&1 | tail -2
python - <<'PY' || exit 5
import json
a = json.load(open("results/mx4_p1/c0_local.json", encoding="utf-8"))["c0"]
b = json.load(open("results/mx4_p1/c0.json", encoding="utf-8"))["c0"]
assert a["_pass"] and b["_pass"], (a["_pass"], b["_pass"])
for d in a:
    if d.startswith("_"): continue
    for arm in a[d]:
        assert a[d][arm]["pick"] == b[d][arm]["pick"] and abs(a[d][arm]["acc"] - b[d][arm]["acc"]) < 1e-12, (d, arm)
print("C0 pod == local: PASS")
PY
# 冒烟用与 p1_run.sh 同一套线程 / 分块环境
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/46_p1_pool.py --smoke --datasets Handwriting --outdir results/mx4_p1 --joblib-backend loky
nvidia-smi --query-gpu=memory.used --format=csv
python scripts/46_p1_pool.py --dry-run
echo "== setup done"
