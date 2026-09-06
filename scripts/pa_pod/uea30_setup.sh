#!/usr/bin/env bash
# WP10-UEA30 上云（4090）：pod 侧一次性准备（依赖按 pin、解包 --no-same-owner 转 LF、磁盘闸、23 靶 UEA 缓存就位
# （上传分包优先，否则 uea30_fetch.py 从 Zenodo 直下）+ .ts 原始字节 md5 对账、code/data md5 对账、单测、--check、
# 冒烟、dry-run）。
#   bash /workspace/wp/CLS/scripts/pa_pod/uea30_setup.sh /workspace/upload_uea30
# 退出码：3 = UEA 缓存缺 / md5 不对；4 = /workspace 空闲 < 4 GB；其它 = set -e 处的命令失败。
set -euo pipefail
UP="${1:-/workspace/upload_uea30}"
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
tar --no-same-owner -xzf "$UP/data_uea30.tar.gz" -C "$REPO"
find results/mx3_pq results/route_a_probe results/uea30 -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'

# 磁盘闸（网络卷 10 GB：23 靶 .ts ≈ 2.4 GB，另加上传分包）
df -h /workspace
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 4 ] || { echo "FREE ${AVAIL_GB:-?} GB < 4 GB on /workspace"; exit 4; }

# 23 靶清单：configs/uea30.yaml 里 run: true 按 order（与 45 号 run_targets 同规则，不硬编码）
TARGETS=$(python - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/uea30.yaml", encoding="utf-8"))
print(" ".join(d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))))
PY
)
echo "== targets ($(echo $TARGETS | wc -w)): $TARGETS"
[ -f "$UP/md5_uea_ts.txt" ] || { echo "MISSING $UP/md5_uea_ts.txt（bundle 产物）"; exit 3; }
mkdir -p data/uea_cache
SMALL_DONE=0
FETCH_FAIL=""
for d in $TARGETS; do
  # 上传分包优先：uea_<Name>.tar.gz（五个大靶各一包）或 uea_small.tar.gz（其余合一包，只解一次）；没有才 Zenodo 直下
  if [ -f "$UP/uea_$d.tar.gz" ]; then
    echo "== unpack $UP/uea_$d.tar.gz"; tar --no-same-owner -xzf "$UP/uea_$d.tar.gz" -C "$REPO"
  elif [ -f "$UP/uea_small.tar.gz" ] && [ "$SMALL_DONE" = 0 ]; then
    echo "== unpack $UP/uea_small.tar.gz"; tar --no-same-owner -xzf "$UP/uea_small.tar.gz" -C "$REPO"; SMALL_DONE=1
  fi
  # 两文件已在且 md5 与清单一致 → fetch 跳过；否则 Zenodo 直下（.ts 是原始字节，绝不过 sed 转 LF）
  python scripts/pa_pod/uea30_fetch.py "$d" --cache-dir data/uea_cache --md5 "$UP/md5_uea_ts.txt" \
    || FETCH_FAIL="$FETCH_FAIL $d"
done
[ -z "$FETCH_FAIL" ] || echo "!! Zenodo 直下 / 对账失败：$FETCH_FAIL —— 本地 bash scripts/pa_pod/uea30_bundle.sh <dir> --with-uea 后把对应 uea_<Name>.tar.gz / uea_small.tar.gz 传到 $UP 再重跑本脚本"

fail=0
for d in $TARGETS; do
  [ -d "data/uea_cache/$d" ] || { echo "MISSING data/uea_cache/$d"; fail=1; }
done
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING uea $f"; fail=1; continue; fi
  h=$(md5sum "$f" | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH uea $f"; fail=1; }
done < "$UP/md5_uea_ts.txt"
echo "== uea .ts md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ] || exit 3
du -sh data/uea_cache; df -h /workspace | tail -1

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

# 单测限 OMP=1（pod 上 pytest 不限线程会把 vCPU 打满，见 memory mx4-diagnosis）
# 先验金标（test_build_candidates_prior_golden）是本机折内数值，跨机浮点可能误报，本机已过，pod 上不跑
OMP_NUM_THREADS=1 python -m pytest tests/test_uea30_deploy.py tests/test_sh_deploy.py tests/test_uea.py tests/test_cls_space.py -q \
    --deselect tests/test_uea30_deploy.py::test_build_candidates_prior_golden --deselect tests/test_uea30_deploy.py::test_uea30_arms_golden_six_targets 2>&1 | tail -3
python scripts/45_uea30_deploy.py --check
# 冒烟用与 uea30_run.sh 同一套线程 / 分块环境（评估 CPU-bound，不限 OMP 慢 4×；见 memory route-a-throughput-omp1）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864 AUTOCLS_INFONCE_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/45_uea30_deploy.py --smoke \
    --datasets JapaneseVowels CharacterTrajectories DuckDuckGeese PEMS-SF MotorImagery EigenWorms PenDigits \
    --outdir results/uea30 --joblib-backend loky
nvidia-smi --query-gpu=memory.used --format=csv
python scripts/45_uea30_deploy.py --dry-run
echo "== setup done"
