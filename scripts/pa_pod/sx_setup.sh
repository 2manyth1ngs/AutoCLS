#!/usr/bin/env bash
# WP16 SX 检验上云（5090）：pod 侧一次性准备——依赖按 pin、解包代码（--no-same-owner + LF）、解包输入、磁盘闸、6 个 UEA 源数据集 .ts 就位
# （卷上已有且 md5 一致则跳过，否则 uea30_fetch.py 从 Zenodo 直下）、CUDA 闸、导入自检、74 --slice 与本地 tasks.json 逐字段对账。
#   bash /workspace/wp/CLS/scripts/pa_pod/sx_setup.sh /workspace/upload_sx
# 退出码：3 = UEA 缓存缺 / md5 不对；4 = /workspace 空闲 < 6 GB；5 = CUDA 不可用；6 = 切片对账不一致；其它 = set -e 处的命令失败。
set -euo pipefail
UP="${1:-/workspace/upload_sx}"
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
tar --no-same-owner -xzf "$UP/sx_inputs.tar.gz" -C "$REPO"
find results/mx3_pq results/tclsc_slices results/tclsc_s1et/uea results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
cp results/tclsc_slices/tasks.json "$UP/tasks_local.json"
echo "== HEAD $(cat "$UP/HEAD.txt")"

df -h /workspace | tail -1
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 6 ] || { echo "FREE ${AVAIL_GB:-?} GB < 6 GB on /workspace"; exit 4; }

mkdir -p data/uea_cache
FETCH_FAIL=""
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection; do
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
[ -f data/candidates/UniMiB-SHAR.npz ] || { echo "MISSING data/candidates/UniMiB-SHAR.npz"; exit 3; }

python - <<'PY' || exit 5
import torch, sys
assert torch.cuda.is_available(), "CUDA 不可用（NVML 掉线？）"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK")
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c "import importlib; importlib.import_module('scripts.74_tclsc_slices'); print('import 74 OK')"
python scripts/74_tclsc_slices.py --slice > /workspace/logs/sx_slice.log 2>&1 || { tail -5 /workspace/logs/sx_slice.log; exit 6; }
python - "$UP/tasks_local.json" results/tclsc_slices/tasks.json <<'PY' || exit 6
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]; b = json.load(open(sys.argv[2], encoding="utf-8"))["tasks"]
bad = [k for k in a if k not in b or any(a[k][f] != b[k][f] for f in ("classes", "channels", "sample_indices", "n_train", "n_eval"))]
assert set(a) == set(b) and not bad, f"切片对账不一致：{bad[:5]} / {sorted(set(a) ^ set(b))[:5]}"
print(f"== slice check: {len(a)} 切片逐字段一致")
PY
echo "== setup done"
