"""显存/内存不足（OOM）防护——阶段1 多 worker 并发跑 label 的唯一安全网。

**为什么必须单独一层**（GATE_5090_REPORT §4 第 3 条）：`evaluate_rprime` 原先
`except Exception` 把一切异常吞成 `r_prime=null` 写缓存，而 `--resume` 按缓存键
跳过、看门狗还会把连崩条目主动隔离 —— 于是"显存不够"这种**纯配置问题**会表现为
**永久丢 label 且零报错**。中档（L~300-500）worker=2 是未实测的保守值，
长序列端 L=1152 单进程峰值就 15.8 GB（`_masked_triplet` 的 (P, n) hinge），
所以这条防线必须扎实。

阶段1 实证了它值多少：69 条 (task, cls) 因显存没跑成，因为有这层，它们是
`oom_failures.jsonl` 里一份可查的清单、且 `--resume` 认得出"未跑"，而不是缓存里
69 个无声的空洞。修复与补跑见 `docs/TRIPLET_OOM_FIX.md`。

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


def guarded_run_multi(
    fn: Callable[..., dict],
    make_row: Callable[[int, dict], dict],
    *,
    cache_path: str | Path,
    inflight_path: str | Path,
    task_id: str,
    cls_key: str,
    iters_list: tuple[int, ...],
    on_oom: str = ON_OOM_SKIP,
    oom_log_path: Optional[str | Path] = None,
    context: Optional[dict] = None,
    log: Callable[[str], None] = print,
    **fn_kwargs,
) -> Optional[list[dict]]:
    """跑一条**多截点轨迹**并把它的 N 行原子落盘（MX-3-PQ 的 guarded_run 版）。

    `fn(**fn_kwargs)` 返回 `{iters: rec}`（evaluate_rprime_trajectory 的形状），
    `make_row(iters, rec)` 由调用方合成该 budget 的完整缓存行（dataset/tag/cls
    等元数据它自己知道）；本函数只保证两件事写死在行上：`iters` 与
    `checkpoint_iters`——它们是"这行属于哪条轨迹的哪个 budget"的唯一凭据，
    交给调用方就会被漏写。返回落盘的行（按 iters 升序）；OOM 返回 None。

    **原子性是本函数存在的理由**：N 行先全部合成完，再由 append_jsonl_many 一次
    open + 一次 write 整段落盘。若逐行 append（N 次 open/write/close），进程在两次
    追加之间被杀就留下"60/120 有、290 没有"的缓存，--resume 只补 290 → 补出来的行
    来自另一条权重轨迹，却与前两行共享同一个 trajectory_id，budget 间比较就此
    掺进轨迹噪声（δ_nd 0.030 量级）。make_row 自己抛错时同样一行不写。

    **budget 对账**：`sorted(recs)` 必须与 `iters_list` 完全相等才落盘——多返回的
    budget 被静默丢弃 = 那条 R' 白跑，少返回的 = 静默少一行；`iters_list` 自身重复
    则会把同一行写两遍。两者一律 ValueError。每行的 `checkpoint_iters` 保留
    evaluate_rprime_trajectory 署名的那份（它才知道这条轨迹实际打了哪些截点），
    本函数只核对不改写——改写等于把不一致抹平成谎言。

    与 guarded_run 的其余语义逐条一致：开跑前 mark_inflight 打点、OOM 清 inflight
    且一行不写（规矩 2/3）、on_oom=record 时记 oom_failures.jsonl（规矩 4，
    **每个 budget 各一条**，让"哪些格子没有 R'"逐格可查）、非 OOM 异常照常上抛。
    inflight 打点的 iters 写轨迹的 max_iters：cloud_run.sh 把该字段裸拼进 JSON，
    只能是单个整数；毒条目隔离因此只隔离 max budget 那一行，其余 budget 下一轮
    resume 仍会重试整条轨迹（比写坏 JSON 让看门狗静默失效强）。
    """
    from .labeling import append_jsonl_many, clear_inflight, mark_inflight

    iters_list = tuple(int(b) for b in iters_list)
    if not iters_list:
        raise ValueError("iters_list 不能为空")
    if len(set(iters_list)) != len(iters_list):
        raise ValueError(f"iters_list={iters_list} 有重复 budget——同一行会被写两遍")
    iters_list = tuple(sorted(iters_list))

    mark_inflight(inflight_path, task_id, cls_key, max(iters_list))
    try:
        recs = fn(**fn_kwargs)
    except RprimeOOM as e:
        clear_inflight(inflight_path)      # 规矩 3：不留 inflight → 不计入毒条目
        who = f"{task_id}/{cls_key[:8]} iters={list(iters_list)}"
        if on_oom == ON_OOM_RECORD:
            path = oom_log_path or Path(cache_path).with_name("oom_failures.jsonl")
            for b in iters_list:
                append_oom_failure(path, {
                    "task_id": task_id, "cls_key": cls_key, "iters": b,
                    "checkpoint_iters": list(iters_list),
                    **(context or {}), "error": str(e)})
            log(f"❌ OOM（已降到 worker=1 仍不够显存）{who} → 记入 oom_failures.jsonl，"
                f"该轨迹全部 budget 无 R'：{e}")
        else:
            log(f"⚠️ OOM {who} → 不写缓存（整条轨迹保持未跑），留待 worker=1 串行"
                f"收尾重试：{e}")
        return None                        # 规矩 2：缓存里没有这条 → resume 会重试
    if sorted(recs) != list(iters_list):
        raise ValueError(
            f"{task_id}/{cls_key[:8]}: 轨迹 budget 与 iters_list 不符——fn 返回 "
            f"{sorted(recs)}，期望 {list(iters_list)}（多的会被静默丢弃、少的会"
            "静默少写一行，两种都不允许）")
    rows = []
    for b in iters_list:
        row = {**make_row(b, recs[b]), "iters": b}
        if row.get("checkpoint_iters") != list(iters_list):
            raise ValueError(
                f"{task_id}/{cls_key[:8]} iters={b}: 行自带的 checkpoint_iters="
                f"{row.get('checkpoint_iters')} 与 iters_list={list(iters_list)} 不符"
                "——本函数只核对不改写（改写等于把不一致抹平成谎言）")
        rows.append(row)
    append_jsonl_many(cache_path, rows)    # 全部合成成功后一次写入（原子性）
    clear_inflight(inflight_path)
    return rows
