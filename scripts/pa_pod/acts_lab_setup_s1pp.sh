#!/usr/bin/env bash
# WP17-ACTS S1″ 实验室服务器准备（部署路径固定）：解包代码 / 输入 / 新源 npz → npz md5 → 磁盘闸 → 缓存种子（s1p → s1pp）→ CUDA 闸 →
#   83 --check --profile s1pp（本机 npz 重建的 104 任务 5 折划分 md5 与 tasks.json 逐任务一致）→ S1″ 单测。
#   bash /home/b311/data/wangwenjie/kejunyu/AutoCLS/scripts/pa_pod/acts_lab_setup_s1pp.sh
# 退出码同 acts_lab_setup.sh（3 npz / 4 磁盘 / 5 CUDA / 6 check / 7 pytest）。
set -euo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
UP="$REPO/upload"
cd "$REPO"; mkdir -p logs results/acts/s1pp
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
python - <<'PY'
import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "n_gpu", torch.cuda.device_count())
PY
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_inputs.tar.gz" -C "$REPO"
tar --no-same-owner -xzf "$UP/acts_npz.tar.gz" -C "$REPO"
find results/acts/s1pp results/acts/s1 results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
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
[ "${AVAIL_GB:-0}" -ge 2 ] || { echo "FREE ${AVAIL_GB:-?} GB < 2 GB"; exit 4; }
# 缓存种子：S1′ 的 runs / probe0 复用（同 key；只在 s1pp 缓存不存在时拷贝）
[ -f results/acts/s1pp/runs.jsonl ] || cp results/acts/s1p/runs.jsonl results/acts/s1pp/runs.jsonl
[ -f results/acts/s1pp/probe0.jsonl ] || cp results/acts/s1p/probe0.jsonl results/acts/s1pp/probe0.jsonl
echo "== cache seed: $(wc -l < results/acts/s1pp/runs.jsonl) rows / $(wc -l < results/acts/s1pp/probe0.jsonl) probes"
python -c "import pytest" 2>/dev/null || pip install -q pytest 2>&1 | tail -1
python - <<'PY' || exit 5
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK", torch.cuda.get_device_name(0))
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
python scripts/83_acts_corpus.py --check --profile s1pp --outdir results/acts/s1pp --device cpu 2>&1 | grep -v "^\[data\]" | tail -6 | tee logs/acts_s1pp_check.log
grep -q "→ OK" logs/acts_s1pp_check.log || { echo "!! --check NOT OK"; exit 6; }
python -m pytest tests/test_acts_s1pp.py -q 2>&1 | tail -2 | tee logs/acts_s1pp_pytest.log
grep -q "passed" logs/acts_s1pp_pytest.log && ! grep -q "failed\|error" logs/acts_s1pp_pytest.log || { echo "!! pytest NOT OK"; exit 7; }
echo "== setup done"
