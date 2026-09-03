#!/usr/bin/env bash
# MX-4 WP3 上云：pod 侧一次性准备（依赖按 pin、解包 --no-same-owner 转 LF、md5 对账、单测、dry-run）。
#   bash /workspace/wp/CLS/scripts/pa_pod/wp3_setup.sh /workspace/upload_wp3
set -euo pipefail
UP="${1:-/workspace/upload_wp3}"
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

# 代码：覆盖 autocls/scripts/configs/tests，转 LF
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
# 数据 + 协议产物
tar --no-same-owner -xzf "$UP/data_wp3.tar.gz" -C "$REPO"
find results/mx3_pq results/protocol_alignment results/mx4_deploy results/route_a_probe -maxdepth 1 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 | xargs -0 sed -i 's/\r$//'

# UEA 缓存必须已在卷上
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

python -m pytest tests/test_cl_trainer.py tests/test_sh_deploy.py tests/test_surrogate.py tests/test_prior_sh_sim.py -q 2>&1 | tail -3
python scripts/39_sh_deploy.py --dry-run --outdir results/mx4_deploy
echo "== setup done"
