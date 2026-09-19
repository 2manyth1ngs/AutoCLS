#!/usr/bin/env bash
# WP17-ACTS S1 实验室服务器（8k-4090）一次性准备。**部署路径固定**（用户要求，不得改）：
#   REPO=/home/b311/data/wangwenjie/kejunyu/AutoCLS；上传包在 $REPO/upload。
# 步骤：conda activate pytorch（现成 py3.8 / torch 2.4.1 env）→ 解包代码（LF）/ 输入 / npz → npz md5 → 磁盘闸（≥ 2 GB）→ 只补 pytest
#   → CUDA 闸 → 83 --check（npz 重建的切片划分 md5 / n_train / majority 与 tasks.json 逐任务一致 + 本机重建 plan 的 key 序一致）→ 单测。
#   bash /home/b311/data/wangwenjie/kejunyu/AutoCLS/scripts/pa_pod/acts_lab_setup.sh
# 退出码：3 = npz 缺 / md5 不对；4 = 磁盘 < 2 GB；5 = CUDA 不可用；6 = --check 不过；7 = 单测不过。
set -euo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
UP="$REPO/upload"
cd "$REPO"; mkdir -p logs results/acts/s1
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
python - <<'PY'
import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "n_gpu", torch.cuda.device_count())
PY

tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_inputs.tar.gz" -C "$REPO"
tar --no-same-owner -xzf "$UP/acts_npz.tar.gz" -C "$REPO"
find results/acts/s1 results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== HEAD $(cat "$UP/HEAD.txt")"

fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING npz $f"; fail=1; continue; fi
  h=$(md5sum "$f" | cut -d' ' -f1); [ "$h" = "$sum" ] || { echo "MISMATCH npz $f"; fail=1; }
done < "$UP/md5_npz.txt"
echo "== npz md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ] || exit 3

df -h "$REPO" | tail -1
AVAIL_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 2 ] || { echo "FREE ${AVAIL_GB:-?} GB < 2 GB（数据盘几乎满，先清理再来）"; exit 4; }

python -c "import pytest" 2>/dev/null || pip install -q pytest 2>&1 | tail -1
python -c "import scipy, yaml, sklearn, joblib, pytest, numpy; print('deps', 'sklearn', sklearn.__version__, 'scipy', scipy.__version__, 'joblib', joblib.__version__, 'numpy', numpy.__version__)"

python - <<'PY' || exit 5
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK", torch.cuda.get_device_name(0))
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
python scripts/83_acts_corpus.py --check --device cpu 2>&1 | tail -8 | tee logs/acts_check.log
grep -q "→ OK" logs/acts_check.log || { echo "!! --check NOT OK"; exit 6; }
python -m pytest tests/test_acts_s1.py -q 2>&1 | tail -3 | tee logs/acts_pytest.log
grep -q "passed" logs/acts_pytest.log && ! grep -q "failed\|error" logs/acts_pytest.log || { echo "!! pytest NOT OK"; exit 7; }
echo "== setup done"
