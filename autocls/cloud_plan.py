"""单卡阶段1 的排程计划层（configs/runtime.yaml 的可测加载/推导层）。

职责边界：只回答"哪些数据集、什么顺序、几个同时跑、大概多久"，不碰任何 R' 数值口径
（那是 configs/train.yaml + 铁律 2）。GATE_5090_REPORT §1.9 标定推翻了原 4 卡前提——
本负载 CPU-bound（增强在 CPU 上做，GPU 只跑 encoder 前向，单 worker GPU 利用率 5.7%），
**1 张卡 8 worker（15.2 条/分）> 4 张卡各 1（11.6 条/分）**，故阶段1 走单卡多进程。

并行粒度是**数据集**（一个数据集一个进程）：
- 天花板加难重探（snr_gate.ceiling_escalation）在各自进程内闭环，跨进程零竞争；
- 分档表在此的语义随之是"同一档的数据集最多几个同时在跑"；
- 跨档回填（长序列旁边塞短数据集）由显存预算把关——长序列独占整卡是纯浪费。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "configs" / "runtime.yaml"


def load_runtime(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_RUNTIME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 分档与资源模型
# ---------------------------------------------------------------------------


def tier_index(l_enc: int, cfg: dict) -> int:
    """编码长度落在第几档（tiers 按 max_l_enc 升序，末档 max_l_enc=null 兜底）。"""
    for i, t in enumerate(cfg["workers"]["tiers"]):
        cap = t.get("max_l_enc")
        if cap is None or l_enc <= cap:
            return i
    return len(cfg["workers"]["tiers"]) - 1


def workers_for(l_enc: int, cfg: dict) -> int:
    """该长度档允许的同时进程数上限（GATE §3.3 实测表）。"""
    return int(cfg["workers"]["tiers"][tier_index(l_enc, cfg)]["workers"])


def estimate_peak_gib(l_enc: int, cfg: dict) -> float:
    """单进程显存峰值估计（两实测点线性外推，见 runtime.yaml 注释）。

    启动前预检用它：若"分档 worker 数 × 峰值"超预算，宁可启动前降档，
    也不要跑到一半才 OOM（哪怕有 oom_guard 兜底，重试也要重跑整条）。
    """
    g = cfg["gpu"]
    return float(g["peak_gib_base"] + g["peak_gib_per_len"] * l_enc)


def estimate_item_seconds(l_enc: int, cfg: dict) -> float:
    """单 worker 下一条 (task, cls, seed) → R' 的秒数估计（early_iters=290 口径）。"""
    t = cfg["throughput"]
    return float(t["seconds_base"] + t["seconds_per_len"] * l_enc)


def budget_gib(cfg: dict) -> float:
    return float(cfg["gpu"]["budget_gib"])


def worker_env(cfg: dict) -> dict[str, str]:
    """worker 进程必带的线程环境（§3.1：不置 1 慢 4.2×）。"""
    return {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}


# ---------------------------------------------------------------------------
# 并发准入
# ---------------------------------------------------------------------------


def can_admit(running_lens: list[int], candidate_len: int, cfg: dict) -> tuple[bool, str]:
    """在已跑 running_lens（各进程的 l_enc）之上，还能不能再起一个 candidate_len。

    三道闸，任一不过即拒（返回原因供日志）：
    ① 全局进程数上限；② 同档进程数上限（分档表）；③ 显存预算。
    """
    cap = int(cfg["workers"]["max_concurrent"])
    if len(running_lens) + 1 > cap:
        return False, f"并发进程数已达上限 {cap}"

    tier = tier_index(candidate_len, cfg)
    same = sum(1 for x in running_lens if tier_index(x, cfg) == tier) + 1
    tier_cap = workers_for(candidate_len, cfg)
    if same > tier_cap:
        return False, f"L={candidate_len} 所在档同时最多 {tier_cap} 个（已有 {same - 1} 个）"

    used = sum(estimate_peak_gib(x, cfg) for x in running_lens)
    need = estimate_peak_gib(candidate_len, cfg)
    if used + need > budget_gib(cfg) + 1e-9:
        return False, (f"显存预算不足：已占 {used:.1f} + 需 {need:.1f} > "
                       f"{budget_gib(cfg):.1f} GiB")
    return True, ""


def admission_order(items: list, cfg: dict, length_of=lambda x: x) -> list:
    """准入顺序：长的先起。长序列是墙钟长杆且互斥，越早起越好；短的在剩余显存里回填。

    items 可以是长度列表（默认），也可以是数据集名 + length_of 取长度。
    """
    return sorted(items, key=lambda x: -length_of(x))


def simulate_wall_clock(jobs: dict[str, tuple[int, int]], cfg: dict) -> float:
    """按 can_admit 规则模拟排程，返回墙钟秒估计。jobs: {名称: (l_enc, 待跑条数)}。

    只作 ETA 与"长杆在哪"的判断依据（真实调度由 scripts/run_stage1.py 执行同一规则）。
    """
    c = float(cfg["throughput"].get("contention_factor", 1.0))
    pending = sorted(jobs.items(), key=lambda kv: -kv[1][0])
    running: dict[str, tuple[int, float]] = {}   # 名称 → (l_enc, 结束时刻)
    t = 0.0
    while pending or running:
        progressed = True
        while progressed:
            progressed = False
            for name, (l_enc, n) in list(pending):
                lens = [x for x, _ in running.values()]
                ok, _ = can_admit(lens, l_enc, cfg)
                if ok:
                    running[name] = (l_enc, t + n * estimate_item_seconds(l_enc, cfg) * c)
                    pending.remove((name, (l_enc, n)))
                    progressed = True
        if not running:
            break     # 单个任务都塞不下（预算配错）：避免死循环
        t = min(end for _, end in running.values())
        for name in [k for k, (_, end) in running.items() if end <= t]:
            del running[name]
    return t


# ---------------------------------------------------------------------------
# 阶段1 跑哪些数据集
# ---------------------------------------------------------------------------


def stage1_datasets(registry, *, include_reserve: bool = True) -> list[str]:
    """阶段1 的 step0 SNR 闸对象：未实测源（tested=false）+ reserve（checklist C1
    要求 13 源 + 3 reserve 全覆盖——源池任务数下限闸不过时要靠 reserve 回填，
    回填前必须已有它们的闸结论）。按 l_enc 降序（准入顺序，长杆先起）。
    """
    names = [e.name for e in registry.source if not e.tested]
    if include_reserve:
        names += [e.name for e in registry.reserve]
    return sorted(names, key=lambda n: -(registry.get(n).l_enc or 0))


def preflight(names: list[str], registry, cfg: dict) -> list[dict]:
    """启动前显存预检表：逐数据集给出 l_enc、分档 worker 数、单进程峰值估计与告警。

    告警条件 = 该档的 worker 数满载时会超预算（说明分档表与显存模型不自洽，
    应该以显存为准降档）——宁可启动前降，也不要跑到一半才 OOM。
    """
    rows = []
    for n in names:
        e = registry.get(n)
        l_enc = e.l_enc or 0
        w = workers_for(l_enc, cfg)
        peak = estimate_peak_gib(l_enc, cfg)
        safe_w = max(1, int(budget_gib(cfg) // peak)) if peak > 0 else w
        rows.append({
            "dataset": n, "l_enc": l_enc, "tier_workers": w,
            "peak_gib": round(peak, 2), "vram_safe_workers": safe_w,
            "workers": min(w, safe_w),
            "warn": None if safe_w >= w else
            f"分档 {w} 个 × {peak:.1f} GiB > 预算 {budget_gib(cfg):.1f} GiB，降到 {safe_w}",
        })
    return rows


def plan_jobs(names: list[str], registry, cfg: dict,
              n_items: Optional[dict[str, int]] = None) -> dict[str, tuple[int, int]]:
    """{数据集: (l_enc, 待跑条数)}——simulate_wall_clock 的输入。

    n_items 缺省用 step0 的满额条数（由调用方传入实际待跑数才准，续跑时差别很大）。
    """
    out: dict[str, tuple[int, int]] = {}
    for n in names:
        e = registry.get(n)
        out[n] = (e.l_enc or 0, int((n_items or {}).get(n, 0)))
    return out
