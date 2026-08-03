"""显存/内存不足（OOM）防护——阶段1 多 worker 并发跑 label 的唯一安全网。

**为什么必须单独一层**（GATE_5090_REPORT §4 第 3 条）：`evaluate_rprime` 原先
`except Exception` 把一切异常吞成 `r_prime=null` 写缓存，而 `--resume` 按缓存键
跳过、看门狗还会把连崩条目主动隔离 —— 于是"显存不够"这种**纯配置问题**会表现为
**永久丢 label 且零报错**。中档（L~300-500）worker=2 是未实测的保守值，
长序列端 L=1152 单进程峰值就 15.8 GB（`losses.py:91` 的 triplet (n,n) 矩阵），
所以这条防线必须扎实。

四条规矩（缺一条都会静默毁数据）：
1. **分清 OOM 与普通失败**：训练发散/数据坏是"这条 CLS 的属性"，记 NaN 落盘是对的；
   显存不足与这条 (task,cls) 的好坏无关，绝不能落盘成结论。`is_oom_error` 是唯一判据。
2. **OOM 不写缓存**：该条保持"未跑"，`--resume` 下一轮自然重试（降 worker 的串行
   收尾扫描即是那一轮）。
3. **OOM 清 inflight**：看门狗（`scripts/cloud_run.sh`）只在 `[ -s $INFLIGHT ]` 为真
   时累计毒条目，连崩 FAIL_LIMIT 次就写永久隔离记录。OOM 后留着 inflight = 把配置
   问题当成毒数据永久隔离。
4. **worker=1 仍 OOM 要看得见**：`on_oom=record` 写 `oom_failures.jsonl`，
   让"哪些条目因显存跑不了"是一份可查的清单，而不是缓存里的一个空洞。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

ON_OOM_SKIP = "skip"      # 并发档第一遍：跳过不落盘，留给串行收尾重试
ON_OOM_RECORD = "record"  # 已降到 worker=1 的收尾扫描：仍 OOM 即终局失败，落盘可见

# 消息级判据。旧版 torch、cuBLAS/cuDNN 句柄创建、CPU 分配器都不抛
# torch.cuda.OutOfMemoryError，只能按消息认——漏认一条 = 漏掉一条 label。
_OOM_MARKERS = (
    "out of memory",
    "outofmemoryerror",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cudnn_status_not_initialized",   # 显存耗尽时 cuDNN 句柄创建的典型表现
    "can't allocate memory",
    "cannot allocate memory",
    "bad_alloc",
)


class RprimeOOM(RuntimeError):
    """一条 (task, cls) 的 R' 因显存/内存不足未跑成——**绝不写缓存**。

    与"训练发散记 NaN"严格区分：NaN 是关于这条 CLS 的结论，OOM 只是这次没跑成。
    """


def is_oom_error(exc: BaseException) -> bool:
    """异常是否属于"资源不足"类（显存或宿主内存）。

    严格宁漏勿滥：误判成 OOM 的条目永远不写缓存 → 每轮 resume 都重跑 → 真死循环。
    故只认异常类型明确是 OOM 的，或消息里出现分配失败标志的。
    """
    if isinstance(exc, MemoryError):   # 宿主内存耗尽（8 worker 并发时可能）
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:      # noqa: BLE001 —— 无 torch / 老版本无该类，退回消息判据
        pass
    if type(exc).__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _OOM_MARKERS)


def append_oom_failure(path: str | Path, rec: dict) -> None:
    """追加一条"显存跑不了"的终局记录（append-only，与 R' 缓存同风格）。"""
    from .labeling import append_jsonl

    append_jsonl(path, rec)


def guarded_run(
    evaluate: Callable[[], dict],
    make_row: Callable[[dict], dict],
    *,
    cache_path: str | Path,
    inflight: str | Path,
    task_id: str,
    cls_key: str,
    iters: int,
    on_oom: str = ON_OOM_SKIP,
    oom_log: Optional[str | Path] = None,
    context: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> Optional[dict]:
    """跑一条 R' 并落盘（step0 与 2b 共用的循环体）。返回落盘行；OOM 返回 None。

    正常路径与改动前逐字节等价：开跑前 `mark_inflight` 打点（看门狗协议）→ 跑 →
    `make_row(rec)` 合成缓存行 → 追加落盘 → 清 inflight。
    OOM 路径：清 inflight（规矩 3）、不落盘（规矩 2）、按 on_oom 决定是否记
    oom_failures.jsonl（规矩 4），返回 None——**调用方必须原样跳过，不得自行补写缓存**。
    非 OOM 异常照常向上抛（进程级崩溃由看门狗 + 毒条目机制处理，本层不插手）。
    """
    from .labeling import append_jsonl, clear_inflight, mark_inflight

    mark_inflight(inflight, task_id, cls_key, iters)
    try:
        rec = evaluate()
    except RprimeOOM as e:
        clear_inflight(inflight)          # 规矩 3：不留 inflight → 不计入毒条目
        who = f"{task_id}/{cls_key[:8]} iters={iters}"
        if on_oom == ON_OOM_RECORD:
            failure = {"task_id": task_id, "cls_key": cls_key, "iters": iters,
                       **(context or {}), "error": str(e)}
            append_oom_failure(oom_log or Path(cache_path).with_name("oom_failures.jsonl"),
                               failure)
            log(f"❌ OOM（已降到 worker=1 仍不够显存）{who} → 记入 oom_failures.jsonl，"
                f"该条无 R'：{e}")
        else:
            log(f"⚠️ OOM {who} → 不写缓存（保持未跑），留待 worker=1 串行收尾重试：{e}")
        return None                        # 规矩 2：缓存里没有这条 → resume 会重试
    row = make_row(rec)
    append_jsonl(cache_path, row)
    clear_inflight(inflight)
    return row
