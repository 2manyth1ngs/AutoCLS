#!/usr/bin/env bash
# WP17-ACTS S2-CORPUS 实验室服务器准备（部署路径固定）：解包代码 / 输入 / 缓存种子 → 19 源数据文件 md5 核对（数据服务器已有）→ 磁盘闸（根盘）→
#   输出文件 symlink 到根盘 /tmp/acts_s2_out（共享盘写满不再打到我们）→ --merge-seeds（S1″ + UCR-ADM 去重合并）→ CUDA 闸 →
#   83 --check --profile s2（本机重建 184 任务 5 折划分 md5 + plan_S key 序逐位一致）→ S2 单测。
#   bash /home/b311/data/wangwenjie/kejunyu/AutoCLS/scripts/pa_pod/acts_lab_setup_s2.sh
# 退出码：3 数据缺 / md5 不对；4 根盘 < 5 GB；5 CUDA；6 check；7 pytest。
set -euo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
UP="$REPO/upload"
OUT="results/acts/s2"; T="/tmp/acts_s2_out"
cd "$REPO"; mkdir -p logs "$OUT" "$T/logs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
python - <<'PY'
import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "n_gpu", torch.cuda.device_count())
PY
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_inputs.tar.gz" -C "$REPO"
tar --no-same-owner -xzf "$UP/acts_seeds.tar.gz" -C "$REPO"
find "$OUT" results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== HEAD $(cat "$UP/HEAD.txt")"
fail=0
while read -r sum f; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then echo "MISSING data $f"; fail=1; continue; fi
  h=$(md5sum "$f" | cut -d' ' -f1); [ "$h" = "$sum" ] || { echo "MISMATCH data $f"; fail=1; }
done < "$UP/md5_data.txt"
echo "== data md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ] || exit 3
df -h "$REPO" /tmp | tail -2
AVAIL_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 5 ] || { echo "ROOT FREE ${AVAIL_GB:-?} GB < 5 GB"; exit 4; }
# 输出文件 symlink 到根盘（已是 symlink 则跳过；普通文件先搬走）
for f in runs.jsonl probe0.jsonl inflight.txt oom_failures.jsonl; do
  if [ -e "$OUT/$f" ] && [ ! -L "$OUT/$f" ]; then mv "$OUT/$f" "$T/$f"; fi
  [ -L "$OUT/$f" ] || ln -s "$T/$f" "$OUT/$f"
done
ls -la "$OUT" | grep -E "runs|probe0|inflight|oom"
python -c "import pytest" 2>/dev/null || pip install -q pytest 2>&1 | tail -1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8
python scripts/83_acts_corpus.py --merge-seeds --profile s2 --outdir "$OUT" --device cpu 2>&1 | tail -1 | tee logs/acts_s2_merge.log
python - <<'PY' || exit 5
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK", torch.cuda.get_device_name(0))
PY
python scripts/83_acts_corpus.py --check --profile s2 --outdir "$OUT" --device cpu 2>&1 | grep -v "^\[data\]" | tail -6 | tee logs/acts_s2_check.log
grep -q "→ OK" logs/acts_s2_check.log || { echo "!! --check NOT OK"; exit 6; }
python -m pytest tests/test_acts_s2.py -q 2>&1 | tail -2 | tee logs/acts_s2_pytest.log
grep -q "passed" logs/acts_s2_pytest.log && ! grep -q "failed\|error" logs/acts_s2_pytest.log || { echo "!! pytest NOT OK"; exit 7; }
echo "== setup done (seeds: $(wc -l < "$T/runs.jsonl") runs / $(wc -l < "$T/probe0.jsonl") probes)"
