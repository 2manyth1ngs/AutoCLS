#!/usr/bin/env bash
# WP17-ACTS S3-B 实验室服务器准备：解包代码 / 输入 → 19 源数据 md5 核对 → 根盘闸（≥ 60 GB：E_sh 18 折 × 176 任务）→
#   E_pt / E_sh / settf 目录 symlink 到根盘 /tmp/acts_s3b_out → CUDA 闸 → S3-B 单测 → 冒烟（--smoke，单独 outdir）。
#   bash /home/b311/data/wangwenjie/kejunyu/AutoCLS/scripts/pa_pod/acts_lab_setup_s3b.sh
# 退出码：3 数据；4 根盘；5 CUDA；7 pytest；8 smoke。
set -euo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"
UP="$REPO/upload"; OUT="results/acts/s3b"; T="/tmp/acts_s3b_out"
cd "$REPO"; mkdir -p logs "$OUT" "$T/logs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'n_gpu', torch.cuda.device_count())"
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3b_inputs.tar.gz" -C "$REPO"
find results/acts/s2 "$OUT" results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
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
[ "${AVAIL_GB:-0}" -ge 60 ] || { echo "ROOT FREE ${AVAIL_GB:-?} GB < 60 GB"; exit 4; }
for d in E_pt E_sh settf; do
  mkdir -p "$T/$d"
  if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi
  [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
ls -la "$OUT"
python -c "import pytest" 2>/dev/null || pip install -q pytest 2>&1 | tail -1
python - <<'PY' || exit 5
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK", torch.cuda.get_device_name(0))
PY
python -m pytest tests/test_acts_s3b.py -q 2>&1 | tail -2 | tee logs/acts_s3b_pytest.log
grep -q "passed" logs/acts_s3b_pytest.log && ! grep -q "failed\|error" logs/acts_s3b_pytest.log || { echo "!! pytest NOT OK"; exit 7; }
SM="$T/smoke"; rm -rf "$SM"; mkdir -p "$SM"; cp "$OUT/targets.npz" "$SM/"
CUDA_VISIBLE_DEVICES=${GPU:-0} python scripts/86_acts_s3b.py --smoke --outdir "$SM" --device cuda 2>&1 | grep -v "^\[data\]" | tail -4 | tee logs/acts_s3b_smoke.log
grep -q "\[smoke\] OK" logs/acts_s3b_smoke.log || { echo "!! smoke NOT OK"; exit 8; }
echo "== setup done"
