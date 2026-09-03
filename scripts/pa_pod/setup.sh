#!/usr/bin/env bash
# 全量协议对齐上云：pod 侧一次性准备（重开 pod 检查单三步固化：依赖按 pin、解包转 LF、md5 对账）。
#   bash /workspace/wp/CLS/scripts/pa_pod/setup.sh /workspace/upload_pa
set -euo pipefail
UP="${1:-/workspace/upload_pa}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
}
echo "== torch fingerprint (before)"; tfp

# 1) 依赖：只装增量、逐版本 pin（永远不跑 pip install -r requirements.txt）
python - <<'PY' || NEED=1
import scipy, yaml, sklearn, pytest, joblib, aeon  # noqa
print("deps present: scipy", scipy.__version__, "sklearn", sklearn.__version__, "aeon", aeon.__version__)
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" 2>&1 | tail -3
fi
echo "== torch fingerprint (after)"; tfp

# 2) 代码：HEAD 归档只解 autocls/scripts/configs/tests（覆盖卷上旧版本），转 LF
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'

# 3) 数据 + 参照
tar --no-same-owner -xzf "$UP/data_pa.tar.gz" -C "$REPO"

# 4) md5 对账（文本 LF 归一化；.ts 原样）
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

# 5) 单测：金标 + 新开关 + 35 号纯函数
python -m pytest tests/test_cl_trainer.py tests/test_swa.py tests/test_augmentations.py tests/test_encoders.py \
                 tests/test_linear_eval.py tests/test_uea.py tests/test_protocol_alignment.py -q 2>&1 | tail -3

# 6) 零 GPU 自检：工单数
python scripts/35_protocol_alignment.py --dry-run --outdir results/protocol_alignment
echo "== setup done"
