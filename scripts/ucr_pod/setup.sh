#!/usr/bin/env bash
# WP7-UCR 上云：pod 侧一次性准备（沿用 pq_pod/setup.sh 的三步固化）。
#   bash /workspace/wp/CLS/scripts/ucr_pod/setup.sh /workspace/upload_ucr
# 前置：$UP 下有 make_bundle.sh 的四件产物。
set -euo pipefail
UP="${1:-/workspace/upload_ucr}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()[-1] if torch.cuda.is_available() else None)
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
tar --no-same-owner -xzf "$UP/data_ucr.tar.gz" -C "$REPO"

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
  case "$f" in
    *.ts) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  [ "$h" = "$sum" ] || { echo "MISMATCH data $f"; fail=1; }
done < "$UP/md5_data_local.txt"
echo "== md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/test_ucr_rprime.py tests/test_labeling.py tests/test_oom_guard.py -q 2>&1 | tail -3

# 装载自检：pod 上重算 tasks.json 必须与上传的逐字节相同（canonical 子采样可复现）
cp results/ucr_rprime/tasks.json /tmp/tasks_uploaded.json
python scripts/41_ucr_rprime.py --tasks 2>&1 | tail -2
python - <<'PY'
import json
a = json.load(open("/tmp/tasks_uploaded.json", encoding="utf-8")); b = json.load(open("results/ucr_rprime/tasks.json", encoding="utf-8"))
assert a["tasks"] == b["tasks"], "tasks.json 与本地不一致（子采样不可复现？）"
print("tasks.json 与本地逐项一致：", len(b["tasks"]), "任务")
PY
echo "== setup done"
