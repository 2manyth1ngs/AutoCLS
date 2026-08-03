#!/usr/bin/env bash
# =====================================================================
# scripts/cloud_run.sh —— 云上脱离进程 + 看门狗（Part A-3；方案 B-3）
#
# 职责（进程级崩溃恢复；per-(task,cls) 异常已由 evaluate_rprime 内 NaN 化处理）：
# - worker 非零退出（被杀/OOM/CUDA 崩/spot 抢占残留）→ 带 --resume 自动重拉直到 exit 0
# - 每次重启换日志名 worker_r<N>.log（不覆盖，事后可逐次排查）
# - 毒条目隔离：同一 inflight (task,cls) 连崩 ≥FAIL_LIMIT 次 → 向 R' 缓存追加
#   r_prime=null 的永久隔离记录（--resume 按缓存键跳过，防"崩→重拉→又崩"死循环）
# - 总重启 ≥MAX_RESTARTS → 停并告警（需人工介入）
#
# ⚠️ spot 抢占：抢占=整机回收，本进程连同看门狗一起消失——恢复靠"实例重启后再次
#   拉起本脚本"（systemd unit / 云商 startup-script / tmux 自启均可）。本脚本幂等：
#   任意时刻重新拉起都从缓存 --resume 无损续跑（这正是敢用 spot 的前提）。挂法：
#     tmux new -d -s route_a 'STAGE=step0 STEP0_DATASETS="..." bash scripts/cloud_run.sh'
#   或 nohup：
#     STAGE=step0 STEP0_DATASETS="..." nohup bash scripts/cloud_run.sh \
#         > results/step0_snr_gate/watchdog.log 2>&1 &
#     STAGE=labels nohup bash scripts/cloud_run.sh \
#         > results/labels_route_a/watchdog.log 2>&1 &
#
# 已知边界：step0 的噪声种子(s≥1)崩溃无法用 canonical 缓存键隔离（inflight 不含
# seed）——隔离记录仍写入 canonical 缓存令 s=0 跳过，s≥1 反复崩靠 MAX_RESTARTS
# 封顶；step0 单条仅 ~1-2 min，实际风险低。
#
# 环境变量：STAGE=step0|labels（必填，除非自给 CMD+OUT）；PY（默认 python）；
#   STEP0_DATASETS（step0 必填）；EXTRA_ARGS（追加给 worker，如 --device cuda:1）；
#   MAX_RESTARTS（默认 50）；FAIL_LIMIT（默认 3）；SLEEP（重启间隔秒，默认 10）；
#   CMD/OUT/CACHE（测试/自定义时整体覆盖）。
# 多卡分片（B-3）：每卡一份本脚本，EXTRA_ARGS 带各自 --device 与 --datasets 分片，
#   共享同一 rprime_cache.jsonl（行追加并发安全）；OUT 各卡独立目录时 STATE/日志互不干扰。
# =====================================================================
set -u

PY="${PY:-python}"
STAGE="${STAGE:-}"

# ---------------------------------------------------------------------------
# 线程环境（GATE_5090_REPORT §3.1：**硬性启动条件**）
# 54 核容器里 torch 默认 192 线程，对 128×51×24 这种小张量 OpenMP 同步开销压倒计算
# → 53 s/条（0.63×，比 4060 笔记本还慢）；置 1 后 12.6 s/条（2.67×）。
# 唯一真源是 configs/runtime.yaml 的 env 段；读不到（无 python/无 yaml）时兜底硬编码，
# 绝不允许"没设成 1"这一状态存在。已显式 export 的同名变量不覆盖（便于临时试验）。
# ---------------------------------------------------------------------------
_repo_root="$(cd "$(dirname "$0")/.." && pwd)"
_env_exports="$(REPO_ROOT="$_repo_root" $PY - <<'PYEOF' 2>/dev/null
import os
import sys
sys.path.insert(0, os.environ["REPO_ROOT"])
try:
    from autocls.cloud_plan import load_runtime, worker_env
    for k, v in worker_env(load_runtime()).items():
        print(f"export {k}=${{{k}:-{v}}}")
except Exception:
    pass
PYEOF
)"
if [ -n "$_env_exports" ]; then
  eval "$_env_exports"
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  echo "[watchdog] ⚠️ 读不到 configs/runtime.yaml，线程数用兜底值 OMP=$OMP_NUM_THREADS" >&2
fi
echo "[watchdog] 线程环境: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
MAX_RESTARTS="${MAX_RESTARTS:-50}"
FAIL_LIMIT="${FAIL_LIMIT:-3}"
SLEEP="${SLEEP:-10}"

if [ -z "${CMD:-}" ]; then
  case "$STAGE" in
    step0)
      CMD="$PY scripts/step0_snr_gate.py --run --resume --datasets ${STEP0_DATASETS:?STAGE=step0 需 STEP0_DATASETS} ${EXTRA_ARGS:-}"
      OUT="${OUT:-results/step0_snr_gate}"
      CACHE="${CACHE:-results/labels_route_a/rprime_cache.jsonl}"   # step0 canonical 并入全量缓存
      ;;
    labels)
      CMD="$PY scripts/2b_generate_labels_route_a.py --resume ${EXTRA_ARGS:-}"
      OUT="${OUT:-results/labels_route_a}"
      ;;
    *)
      echo "用法: STAGE=step0|labels [PY=...] bash scripts/cloud_run.sh（或 CMD=... OUT=... 自定义）" >&2
      exit 64
      ;;
  esac
fi
OUT="${OUT:?自定义 CMD 时需同时给 OUT}"
CACHE="${CACHE:-$OUT/rprime_cache.jsonl}"
INFLIGHT="$OUT/inflight.txt"
STATE="$OUT/watchdog_state.tsv"

mkdir -p "$OUT" "$(dirname "$CACHE")"
touch "$STATE"

# 日志名带看门狗启动时间戳：同一轮内按 r<N> 区分，跨看门狗重启（spot 抢占后
# 重新拉起）也不覆盖上一轮日志；watchdog_state.tsv 追加制，毒条目计数跨重启累计
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
restarts=0
while [ "$restarts" -lt "$MAX_RESTARTS" ]; do
  wlog="$OUT/worker_${RUN_TAG}_r${restarts}.log"
  echo "[watchdog] $(date -u +%FT%TZ) 拉起 #$restarts: $CMD → $wlog"
  $CMD >"$wlog" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[watchdog] $(date -u +%FT%TZ) worker 正常退出 rc=0，完成（共重启 $restarts 次）"
    exit 0
  fi
  echo "[watchdog] $(date -u +%FT%TZ) 非零退出 rc=$rc（被杀/崩溃/抢占），将 --resume 续跑"
  if [ -s "$INFLIGHT" ]; then
    poison="$(cat "$INFLIGHT")"
    tid="$(printf '%s' "$poison" | cut -f1)"
    key="$(printf '%s' "$poison" | cut -f2)"
    it="$(printf '%s' "$poison" | cut -f3)"
    printf '%s\n' "$poison" >>"$STATE"
    n="$(grep -cxF "$poison" "$STATE")"
    echo "[watchdog] 崩溃时在跑: $tid/$key iters=$it（同条目第 $n 次）"
    if [ "$n" -ge "$FAIL_LIMIT" ]; then
      echo "[watchdog] 毒条目 $tid/$key 连崩 $n 次 → 写 r_prime=null 永久隔离（--resume 跳过，防死循环）"
      printf '{"task_id":"%s","cls_key":"%s","iters":%s,"r_prime":null,"error":"watchdog: quarantined after %s crashes"}\n' \
        "$tid" "$key" "$it" "$n" >>"$CACHE"
    fi
  fi
  restarts=$((restarts + 1))
  sleep "$SLEEP"
done

echo "[watchdog] 达重启上限 $MAX_RESTARTS，停止并告警（人工介入：查 $OUT/worker_r*.log 与 $STATE）"
exit 1
