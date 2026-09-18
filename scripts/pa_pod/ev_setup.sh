#!/usr/bin/env bash
# WP16 EV1 检验上云（5090）：pod 侧一次性准备——依赖按 pin（失败重试）、解包代码（LF）、解包输入、磁盘闸（折特征 ≈ 5 GB）、7 个 UEA 源数据集 .ts md5（缺则 Zenodo 直下）、CUDA 闸、
# 导入自检（14 切片 / 信任域 11 / es_field_hard 臂 / NB1 训练标签行数）。
#   bash /workspace/wp/CLS/scripts/pa_pod/ev_setup.sh /workspace/upload_ev
# 退出码：3 = 数据缺 / md5 不对；4 = /workspace 空闲 < 12 GB；5 = CUDA 不可用；6 = 自检不一致。
set -euo pipefail
UP="${1:-/workspace/upload_ev}"
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
tar --no-same-owner -xzf "$UP/ev_inputs.tar.gz" -C "$REPO"
find results/mx3_pq results/tclsc_slices results/tclsc_nb results/tclsc_k3 results/tclsc_ev results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== HEAD $(cat "$UP/HEAD.txt")"

df -h /workspace | tail -1
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 12 ] || { echo "FREE ${AVAIL_GB:-?} GB < 12 GB on /workspace"; exit 4; }

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
import importlib, sys
sys.path.insert(0, ".")
g = importlib.import_module("scripts.77_tclsc_ev")
sp = g._g56.space()
assert len(g.ev_slices()) == 14 and len(g.trust_entries(sp)) == 11
s = g._m33.arm_spec("es_field_hard"); assert (s.task_branch, s.encoder, s.head) == ("E+ctx", "settf", "field")
rows, dk = g._g76.load_rows(g.NB, "train"); assert len(rows) == 5220, len(rows)
ds = g._g75.load_k3_datasets(); n = sum(len(d.slices) for d in ds.values()); assert n == 87, n
print(f"== self-check OK: 14 slices, trust 11, es arm, train rows {len(rows)}, {n} slices")
PY
echo "== setup done"
