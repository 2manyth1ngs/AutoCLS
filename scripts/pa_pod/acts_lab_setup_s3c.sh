#!/usr/bin/env bash
# WP17-ACTS S3-C 实验室服务器准备：解包 → 核对 S3-B 的 E 窗口（E_pt 176 + E_sh 18 折）→ 输出目录 symlink 到根盘 /tmp/acts_s3c_out → CUDA 闸 → 单测 → 冒烟。
#   GPU=<空闲卡> bash /home/b311/data/wangwenjie/kejunyu/AutoCLS/scripts/pa_pod/acts_lab_setup_s3c.sh
# 退出码：3 E 窗口缺；4 根盘；5 CUDA；7 pytest；8 smoke。
set -euo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; UP="$REPO/upload"; OUT="results/acts/s3c"; T="/tmp/acts_s3c_out"
cd "$REPO"; mkdir -p logs "$OUT" "$T/logs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3c_inputs.tar.gz" -C "$REPO"
find results/acts/s2 "$OUT" results/route_a_probe -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'
echo "== HEAD $(cat "$UP/HEAD.txt")"
npt=$(ls results/acts/s3b/E_pt/*.npz 2>/dev/null | grep -vc _means || true); nsh=$(ls -d results/acts/s3b/E_sh/*/ 2>/dev/null | wc -l)
echo "== E_pt $npt / 176, E_sh folds $nsh / 18"
[ "$npt" -ge 176 ] && [ "$nsh" -ge 18 ] || exit 3
AVAIL_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'); [ "${AVAIL_GB:-0}" -ge 20 ] || { echo "ROOT FREE ${AVAIL_GB} GB < 20"; exit 4; }
for d in models train search runs; do
  mkdir -p "$T/$d"
  if [ -e "$OUT/$d" ] && [ ! -L "$OUT/$d" ]; then cp -r "$OUT/$d/." "$T/$d/" && rm -rf "$OUT/$d"; fi
  [ -L "$OUT/$d" ] || ln -s "$T/$d" "$OUT/$d"
done
ls -la "$OUT"
CUDA_VISIBLE_DEVICES=${GPU:-0} python - <<'PY' || exit 5
import torch; assert torch.cuda.is_available(); x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); torch.cuda.synchronize(); print("CUDA gate OK", torch.cuda.get_device_name(0))
PY
python -m pytest tests/test_acts_s3c.py -q 2>&1 | tail -2 | tee logs/acts_s3c_pytest.log
grep -q "passed" logs/acts_s3c_pytest.log && ! grep -q "failed\|error" logs/acts_s3c_pytest.log || { echo "!! pytest NOT OK"; exit 7; }
SM="$T/smoke"; rm -rf "$SM"; mkdir -p "$SM"; cp "$OUT/data.json" "$OUT/data.npz" "$SM/"
CUDA_VISIBLE_DEVICES=${GPU:-0} python scripts/89_acts_s3c.py --smoke --outdir "$SM" --device cuda 2>&1 | grep -E "done|smoke|Error|Traceback" | tail -12 | tee logs/acts_s3c_smoke.log
grep -q "\[smoke\] OK" logs/acts_s3c_smoke.log || { echo "!! smoke NOT OK"; exit 8; }
echo "== setup done"
