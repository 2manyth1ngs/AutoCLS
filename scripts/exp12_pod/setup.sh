#!/usr/bin/env bash
# 实验 1/2 上云：pod 侧一次性准备（重开 pod 检查单 runpod-pod-reprovision-checklist 三步固化）。
# 用法（pod 上）：bash /workspace/wp/CLS/scripts/exp12_pod/setup.sh /workspace/upload
# 前置：本地已把 code_head.tar / code_overlay.tar / data_wp.tar.gz / md5_text_local.txt /
#       md5_data_local.txt 传到 $UP（默认 /workspace/upload）。
set -euo pipefail
UP="${1:-/workspace/upload}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()[-1] if torch.cuda.is_available() else None)
PY
}
echo "== torch fingerprint (before)"; tfp

# 1) 依赖：只装增量、逐版本 pin（永远不跑 pip install -r requirements.txt）
python - <<'PY' || NEED=1
import scipy, yaml, sklearn, pytest, joblib  # noqa
print("deps present: scipy", scipy.__version__, "sklearn", sklearn.__version__)
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" 2>&1 | tail -3
fi
echo "== torch fingerprint (after)"; tfp

# 2) 代码：HEAD 归档只解 autocls/scripts/configs/tests，再叠加本会话覆盖层（均 LF）
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
tar --no-same-owner -xf "$UP/code_overlay.tar" -C "$REPO"
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'

# 3) 数据：labels_workpoint（含 28 任务特征）+ 两个基线缓存
tar --no-same-owner -xzf "$UP/data_wp.tar.gz" -C "$REPO"

# 4) 一致性校验：文本按 LF 归一化 md5，二进制原样
fail=0
while read -r sum f; do
  h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1)
  [ "$h" = "$sum" ] || { echo "MISMATCH text $f"; fail=1; }
done < "$UP/md5_text_local.txt"
while read -r sum f; do
  case "$f" in
    *.npz|*.pt) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  [ "$h" = "$sum" ] || { echo "MISMATCH data $f"; fail=1; }
done < "$UP/md5_data_local.txt"
echo "== md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ]

# 5) 单测（新增三件 + 引擎/模型）
python -m pytest tests/test_soft_labels.py tests/test_mean_pool_encoder.py tests/test_e_diagnostics.py \
                 tests/test_tclsc_engine.py tests/test_tclsc_model.py -q 2>&1 | tail -2
echo "== setup done"
