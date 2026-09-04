#!/usr/bin/env bash
# MX-4 WP6 S1 上云（4090）：pod 侧一次性准备（依赖按 pin、解包 --no-same-owner 转 LF、md5 对账、单测、dry-run）。
#   bash /workspace/wp/CLS/scripts/pa_pod/wp6_setup.sh /workspace/upload_wp6
set -euo pipefail
UP="${1:-/workspace/upload_wp6}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
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
tar --no-same-owner -xzf "$UP/data_wp6.tar.gz" -C "$REPO"
find results/mx3_pq results/protocol_alignment results/mx4_deploy results/mx4_mask results/mx4_prior_sh \
     results/mx4_surrogate results/route_a_probe -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'

for d in Handwriting UWaveGestureLibrary PhonemeSpectra EthanolConcentration Cricket HandMovementDirection; do
  [ -d "data/uea_cache/$d" ] || { echo "MISSING data/uea_cache/$d（卷上无缓存，需上传）"; exit 3; }
done
echo "== uea caches present"

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
OMP_NUM_THREADS=1 python -m pytest tests/test_cls_space.py tests/test_cls_encoder.py tests/test_labeling_route_a.py \
    tests/test_sh_deploy.py tests/test_mask_field_sim.py -q 2>&1 | tail -3
# C0 已在本机逐位通过（含 GBDT 折内数值，跨机浮点可能误报）；pod 上只用零 GBDT 的 --s0 做 40 号脚本冒烟
python scripts/40_mask_field_sim.py --s0 --outdir results/mx4_mask
python scripts/39_sh_deploy.py --dry-run --twins input_mask=binomial --twin-only --outdir results/mx4_mask \
    --legacy-runs results/mx4_deploy/runs.jsonl
echo "== setup done"
