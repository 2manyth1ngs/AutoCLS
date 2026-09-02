#!/usr/bin/env bash
# MX-3-PQ 上云收尾：worker=1 串行重扫 → δ 标定并回填 → 收官校验 → 复现校验。
#
# 用法（pod 上，run.sh 全部结束后）：
#   bash /workspace/wp/CLS/scripts/pq_pod/finish.sh
#
# 四步各自的理由：
# 1) 串行重扫（--on-oom record）：并发档里因显存跳过的格（oom_guard 规矩 2：不落盘、
#    保持"未跑"）在这一遍独占整卡重试；仍不够显存的写进 oom_failures.jsonl，
#    成为一份可查清单而不是缓存里的空洞（规矩 4）。
# 2) --calibrate --apply-delta：按 (数据集, 预算) 标定 δ 与 δ_nd，**回填
#    configs/labels.yaml 的 pq.delta_per_dataset_budget**。33 号读 δ 的唯一真源
#    就是 labels.yaml 这张表（`delta_calibration.json` 只在场时作 ±1e-9 等值对账，
#    不是真源）——**回填的是 pod 上那份 labels.yaml，收工前务必把它连同
#    results/mx3_pq/ 一起拉回本地**，否则 33 号配对会因空表 fail-fast；
#    只拉回产物目录而漏了 labels.yaml，33 号同样拒跑（这正是这道设计要防的事）。
# 3) --verify：缺格 / 半条轨迹 / 重复键（主 + 噪声）/ null 无 error / inflight 非空 /
#    **毒轨迹（整条 error 行，那几格永远没有 R′）** 任一为真即非零退出。
#    毒轨迹两条出路：排查后 `--fill --ggs --noise-probe --resume --retry-terminal`
#    （剔旧行并备份后重跑，永不自动发生）；确认不可恢复就
#    `--verify --allow-terminal` 显式放行。
# 4) --reproduce-check（PROTOCOL §8 预注册容差）：本次 @290 的 canonical 行 vs
#    results/source_pool_v2（全 5090：≥90% 在 0.02 内且配对中位 |Δ| ≤ 0.01）与
#    results/wide_panel（混机：≥85% 在 0.03 内），按 device_name 逐对分桶。
#    不过（含**零重叠**）→ 退出码 5：停下诊断，**不得进比较器阶段**；
#    人工确认无碍才加 --allow-reproduction-breach。
#
# ⚠️ 重跑 setup.sh 会从 code_head.tar 重新解出 configs/，把本脚本回填的
#    pq.delta_per_dataset_budget 块**还原成空表**。回填后若还要重跑 setup，
#    先把 configs/labels.yaml 拷走，或 setup 后重跑一次 --calibrate --apply-delta。
set -euo pipefail
REPO="${REPO:-/workspace/wp/CLS}"
LOG="${LOG:-/workspace/logs}"
OUT="${OUT:-results/mx3_pq}"
cd "$REPO"; mkdir -p "$LOG" "$OUT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export AUTOCLS_TRIPLET_CHUNK_ELEMS=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="Handwriting UniMiB-SHAR PhonemeSpectra UWaveGestureLibrary \
EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2"

echo "== 收尾串行重扫（worker=1，--on-oom record）"
for ds in $DATASETS; do
  echo "  -- $ds"
  python scripts/32_pq_generate_rprime.py --fill --ggs --noise-probe --resume \
        --on-oom record --datasets "$ds" --outdir "$OUT" \
        --inflight "$OUT/inflight_finish.txt" --oom-log "$OUT/oom_failures.jsonl" \
        >> "$LOG/pq_finish.log" 2>> "$LOG/pq_finish.err" \
    || echo "  ⚠️ $ds 非零退出（见 $LOG/pq_finish.err），继续下一个"
done

echo "== δ 标定 + 回填 configs/labels.yaml + 收官校验"
if python scripts/32_pq_generate_rprime.py --calibrate --apply-delta --verify \
        --outdir "$OUT" 2>&1 | tee -a "$LOG/pq_finish.log"; then
  echo "== verify 通过"
else
  echo "== ❌ verify 未通过：按上面的清单补跑"
  echo "   缺格        → 重跑 run.sh（--resume）"
  echo "   半条轨迹    → --fill --ggs --noise-probe --resume --redo-partial"
  echo "   重复键      → 主/噪声缓存都查（同键行必须唯一），剔重后重跑 --verify"
  echo "   毒轨迹      → 先看上面列出的 trajectory_id 与 error，排查后"
  echo "                  --fill --ggs --noise-probe --resume --retry-terminal；"
  echo "                  确认不可恢复才用 --verify --allow-terminal 放行"
  echo "   inflight 非空 → 确认无进程在跑后清空 $OUT/inflight*.txt"
  exit 1
fi

echo "== 复现校验（PROTOCOL §8：新 @290 vs source_pool_v2 / wide_panel 旧缓存）"
if python scripts/32_pq_generate_rprime.py --reproduce-check \
        --outdir "$OUT" 2>&1 | tee -a "$LOG/pq_finish.log"; then
  echo "== finish done：verify + 复现校验均通过"
  echo "   拉回本地：$OUT/（含 reproduction_check.json）与 configs/labels.yaml"
  echo "   （δ 表在后者的 pq.delta_per_dataset_budget —— 33 号读 δ 的唯一真源）"
else
  echo "== ❌ 复现校验未通过（§8）：**停下诊断，不得进比较器阶段**"
  echo "   零重叠      → 参照缓存没上传，或切片/CLS/权重种子口径漂了"
  echo "   超差        → 看 $OUT/reproduction_check.json 的分桶明细（命中率/中位 |Δ|）"
  echo "   人工确认无碍才用 --reproduce-check --allow-reproduction-breach 放行"
  exit 1
fi
