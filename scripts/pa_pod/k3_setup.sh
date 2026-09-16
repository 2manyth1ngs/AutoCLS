#!/usr/bin/env bash
# WP16 K3 检验上云（5090）：pod 侧一次性准备——依赖按 pin、解包代码（--no-same-owner + LF）、解包输入、磁盘闸、7 个 UEA 源数据集 .ts 就位
# （卷上已有且 md5 一致则跳过，否则 uea30_fetch.py 从 Zenodo 直下）、CUDA 闸、导入自检、网格计数 + 扩展切片与本地 tasks.json 逐字段对账。
#   bash /workspace/wp/CLS/scripts/pa_pod/k3_setup.sh /workspace/upload_k3
# 退出码：3 = 数据缺 / md5 不对；4 = /workspace 空闲 < 6 GB；5 = CUDA 不可用；6 = 网格 / 切片对账不一致；其它 = set -e 处的命令失败。
set -euo pipefail
UP="${1:-/workspace/upload_k3}"
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
  for attempt in 1 2 3; do
    pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
        "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" requests psutil 2>&1 | tail -2 && break
    echo "pip attempt $attempt failed, retry"; sleep 10
  done
  python -c "import scipy, yaml, sklearn, pytest, joblib, aeon, requests, psutil; print('deps present after install')"
fi

tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/k3_inputs.tar.gz" -C "$REPO"
find results/mx3_pq results/tclsc_slices results/tclsc_k3 results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== HEAD $(cat "$UP/HEAD.txt")"

df -h /workspace | tail -1
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 6 ] || { echo "FREE ${AVAIL_GB:-?} GB < 6 GB on /workspace"; exit 4; }

mkdir -p data/uea_cache
FETCH_FAIL=""
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
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
python - <<'PY' || exit 6
import importlib, json, sys
sys.path.insert(0, ".")
g = importlib.import_module("scripts.75_tclsc_k3")
cells = g.k3_cells(g.all_slices()); assert len(cells) == 87 * 10 * 2, len(cells)
ds = g.load_k3_datasets()
ref = json.load(open("results/tclsc_slices/tasks.json", encoding="utf-8"))["tasks"]
ref0 = json.load(open("results/mx3_pq/tasks.json", encoding="utf-8"))["tasks"]
bad = []
for d in ds.values():
    for t in d.slices:
        r = ref.get(t.task_id) or ref0.get(t.task_id)
        if r is None or list(t.classes) != list(r["classes"]) or list(t.channels) != list(r["channels"]) or list(t.sample_indices) != list(r["sample_indices"]):
            bad.append(t.task_id)
assert not bad, f"切片对账不一致：{bad[:5]}"
print(f"== grid {len(cells)} cells, {sum(len(d.slices) for d in ds.values())} slices 逐字段一致")
PY
echo "== setup done"
