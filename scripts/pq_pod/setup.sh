#!/usr/bin/env bash
# MX-3-PQ 上云：pod 侧一次性准备（重开 pod 检查单 runpod-pod-reprovision-checklist 三步固化）。
# 用法（pod 上）：bash /workspace/wp/CLS/scripts/pq_pod/setup.sh /workspace/upload
#
# 前置：本地把下列文件传到 $UP（默认 /workspace/upload）——
#   code_head.tar        git archive HEAD（只解 autocls/scripts/configs/tests）
#   code_overlay.tar     本会话未提交的改动（32 号脚本 + pq_common + Task 1/2/3 改动）
#   data_pq.tar.gz       仓库相对路径打包，必须含：
#       data/uea_cache/{Handwriting,PhonemeSpectra,UWaveGestureLibrary,
#                       EthanolConcentration,Cricket,HandMovementDirection,
#                       SelfRegulationSCP2}      —— 7 个 UEA 源
#       data/candidates/UniMiB-SHAR.{npz,json}   —— 候选池注册表那一个
#       results/route_a_probe/probe_cls_catalog.json          shared 10 的唯一真源
#       results/baseline_probe/baseline_cache.jsonl           δ_nd 的 strict 判据（UEA）
#       results/candidate_survey/baseline/baseline_cache.jsonl  同上（UniMiB-SHAR）
#       results/stage1_rerun_20260804/results/step0_snr_gate/tasks_meta.d/
#       results/stage1_rerun_20260804/results/step0_snr_gate/noise_cache.jsonl
#                                                 定义漂移闸 + @290 k3 一致性对账
#       results/labels_workpoint/tasks.json       UniMiB-SHAR 的切片定义参照
#       results/source_pool_v2/rprime_cache.jsonl   复现校验参照（全 5090，PROTOCOL §8）
#       results/wide_panel/rprime_cache.jsonl       复现校验参照（4060/5090 混机，同上）
#                                                 —— finish.sh 的 --reproduce-check 读这两份；
#                                                    缺了整条收尾会判红（零重叠不算通过）
#   md5_text_local.txt / md5_data_local.txt   本地清单（文本按 LF 归一化后的 md5）
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
import scipy, yaml, sklearn, pytest, joblib, aeon  # noqa
print("deps present: scipy", scipy.__version__, "sklearn", sklearn.__version__,
      "aeon", aeon.__version__)
PY
if [ "${NEED:-0}" = "1" ]; then
  pip install --break-system-packages "scikit-learn==1.7.2" "pyyaml==6.0.3" "pytest==9.1.1" \
      "scipy==1.15.3" "joblib==1.5.3" "threadpoolctl==3.6.0" "aeon==1.3.0" 2>&1 | tail -3
fi
# 装完复看指纹：pip 拉进一个新 torch 就是换了另一台机器的 R'（blackwell-5090 闸门口径）
echo "== torch fingerprint (after)"; tfp

# 2) 代码：HEAD 归档只解 autocls/scripts/configs/tests，再叠加本会话覆盖层（均 LF）
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
tar --no-same-owner -xf "$UP/code_overlay.tar" -C "$REPO"
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'

# 3) 数据 + 参照落盘
tar --no-same-owner -xzf "$UP/data_pq.tar.gz" -C "$REPO"

# 3b) bundle 期望：复现校验的两份旧缓存必须在场。md5 清单本来也会查它们，
#     但只有 make_bundle.sh 打的包才带；这里显式点名，免得换个方式打包时
#     一路跑到 finish.sh 最后一步才发现"零重叠"。
for f in results/source_pool_v2/rprime_cache.jsonl results/wide_panel/rprime_cache.jsonl; do
  [ -f "$f" ] || { echo "MISSING repro-reference $f（PROTOCOL §8 复现校验参照）"; exit 2; }
done

# 4) 一致性校验：文本按 LF 归一化 md5，二进制原样
# 清单里的文件没解出来 = 漏了一份数据，必须报 MISSING 并判败；直接跑
# md5sum 会在 set -e 下让整个脚本静默终止，看不出是哪一份（fix 轮 1 之 12）。
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
    *.npz|*.pt|*.ts|*.gz) h=$(md5sum "$f" | cut -d' ' -f1) ;;
    *) h=$(sed 's/\r$//' "$f" | md5sum | cut -d' ' -f1) ;;
  esac
  [ "$h" = "$sum" ] || { echo "MISMATCH data $f"; fail=1; }
done < "$UP/md5_data_local.txt"
echo "== md5 check: $( [ $fail = 0 ] && echo ALL MATCH || echo FAILED )"
[ $fail = 0 ]

# 5) 单测（Task 1 的三件 + PQ 公共装载 + 32 号脚本自测）
python -m pytest tests/test_pq_common.py tests/test_labeling.py tests/test_cl_trainer.py \
                 tests/test_oom_guard.py tests/test_pq_generate.py -q 2>&1 | tail -3

# 6) 装载自检：45 切片 + 定义漂移闸（跑 --tasks，零 GPU，约 15 s）
python scripts/32_pq_generate_rprime.py --tasks --outdir results/mx3_pq
echo "== setup done"
