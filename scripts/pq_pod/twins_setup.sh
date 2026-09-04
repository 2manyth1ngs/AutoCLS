#!/usr/bin/env bash
# MX-4 WP6 S1b+S2 上云（5090）：pod 侧一次性准备。
#   bash /workspace/wp/CLS/scripts/pq_pod/twins_setup.sh /workspace/upload_twins
# 与 pq_pod/setup.sh 同套路（依赖按 pin、LF 归一、md5 对账），不需要 code_overlay，也不跑复现校验参照。
set -euo pipefail
UP="${1:-/workspace/upload_twins}"
REPO="${REPO:-/workspace/wp/CLS}"
mkdir -p "$REPO" /workspace/logs
cd "$REPO"

tfp() { python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list()[-1] if torch.cuda.is_available() else None,
      "dev", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
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
tar --no-same-owner -xzf "$UP/data_twins.tar.gz" -C "$REPO"
[ -f "$UP/data_uea.tar.gz" ] && tar --no-same-owner -xzf "$UP/data_uea.tar.gz" -C "$REPO"
# 解出的 json/jsonl/md 归一化为 LF（tar 里的 mtime 是打包前的，不能用 -newer 挑）
for d in results/mx3_pq results/mx4_mask results/mx4_deploy results/route_a_probe results/labels_workpoint \
         results/baseline_probe results/candidate_survey/baseline results/stage1_rerun_20260804/results/step0_snr_gate; do
  [ -d "$d" ] && find "$d" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' \) -print0 \
    | xargs -0 -r sed -i 's/\r$//'
done
[ -d results/stage1_rerun_20260804/results/step0_snr_gate/tasks_meta.d ] && \
  find results/stage1_rerun_20260804/results/step0_snr_gate/tasks_meta.d -type f -name '*.json' -print0 | xargs -0 -r sed -i 's/\r$//'

for d in Handwriting UWaveGestureLibrary PhonemeSpectra EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
  [ -d "data/uea_cache/$d" ] || { echo "MISSING data/uea_cache/$d（卷上无缓存，需 twins_bundle.sh --with-uea）"; exit 3; }
done
[ -f data/candidates/UniMiB-SHAR.npz ] || { echo "MISSING data/candidates/UniMiB-SHAR.npz"; exit 3; }
echo "== source caches present"

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

OMP_NUM_THREADS=1 python -m pytest tests/test_cls_space.py tests/test_labeling.py tests/test_pq_generate.py -q 2>&1 | tail -3
# 计数自检：全量孪生 655 格（--verify 全缺即红，正常）、S1b 前缀 66 格
python scripts/32_pq_generate_rprime.py --verify --twins input_mask=binomial --twin-only \
    --outdir results/mx4_mask/source_twins | head -4 || true
echo "== setup done"
