"""Gate v2：用阶段1 的 1040 条 R' 缓存按新判据重评 13 个数据集（纯离线，不跑 GPU）。

**判据定义与每个阈值的独立理由见 `results/gate_v2/CRITERION.md`，该文件先于本脚本
的任何输出提交（commit 1136cfb）**——这是"不为凑够 100 而设计判据"的可验证机制。
本脚本只执行那份定义，不在此重新论证阈值。

一句话判据：pass ⟺ ∃δ 使得「δ 过滤后的方向正确率 A(δ) ≥ 0.95」且「产量 Y(δ) ≥ 1/3」。
- A(δ)：canonical(seed 0) 单种子造出的 label 方向 vs seed 1/2/3 均值的共识方向。
  两份缓存的种子不重叠 → label 不给自己投票（天然留一），且每个种子换的是
  weight_seed **和** split_seed，共识是独立划分上的复现检验。
- Y(δ)：δ 过滤后存活配对 / 候选配对（口径逐字对齐 autocls.labeling.make_pairs）。

旧判据（autocls/snr_gate.py: SNR≥1 ∧ pair_sign_stability≥0.7）的两个结构性缺陷见
results/probe_sensitivity/REPORT.md：① 循环性（Pearson(退化 CLS 占比, 任务数闸)=0.751）；
② 分母含 δ 本就丢弃的近平局对，与下游口径不一致。

用法：
  python scripts/gate_v2_eval.py             # 全部分析 → results/gate_v2/
  python scripts/gate_v2_eval.py --quick     # 跳过 637 子集穷举（只留一法）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import (  # noqa: E402
    delta_for,
    load_registry,
    task_count_gate,
)
from autocls.labeling import load_probe_cls  # noqa: E402
from autocls.snr_gate import compute_snr_stats  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402

# --- 判据常量（CRITERION.md §2；改动等于改判据，须先改那份文档并说明理由）---
A_FLOOR = 0.95          # 探针 §2.2 标定 δ 时用的"达 ≥95% 正确"判据，原样继承
Y_FLOOR = 1.0 / 3.0     # 项目已接受的最低产量点（Cricket δ=0.08 保留 36.4%）
DELTA_GRID = tuple(round(0.005 * i, 3) for i in range(1, 41))   # 0.005 … 0.200

CANON_SEED = 0                    # canonical 缓存（造 label 用的那一条）
CONSENSUS_SEEDS = (1, 2, 3)       # 噪声缓存（独立 weight_seed + split_seed）

SNAPSHOT = ROOT / "results" / "stage1_rerun_20260804" / "results"
CANON_CACHE = SNAPSHOT / "labels_route_a" / "rprime_cache.jsonl"
NOISE_CACHE = SNAPSHOT / "step0_snr_gate" / "noise_cache.jsonl"
META_DIR = SNAPSHOT / "step0_snr_gate" / "tasks_meta.d"
BASE_ANALYSIS = SNAPSHOT / "step0_snr_gate" / "analysis.json"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
LABELS_CFG = ROOT / "configs" / "labels.yaml"

OUTDIR = ROOT / "results" / "gate_v2"
N_PROBE_CLS = 10


# ===========================================================================
# 判据核心（纯函数，tests/test_gate_v2.py 覆盖）
# ===========================================================================


@dataclass(frozen=True)
class PairObs:
    """一个任务上一对 CLS 的观测：canonical 造出的 label + 独立种子的共识方向。

    label_a_better=None → canonical R' 严格相等，造不出 label（make_pairs 的
    n_tie_dropped）；仍计入候选数，永不计入存活数。
    cons_a_better=None → 共识为平局或共识种子缺失 → 方向无定义，不计入正确率分母。
    """

    task_id: str
    a: str
    b: str
    d_canon: float                  # R'_a − R'_b（canonical 种子，带号）
    label_a_better: Optional[bool]
    cons_a_better: Optional[bool]


def _direction(ra: float, rb: float, task_type: str) -> Optional[bool]:
    """方向唯一合法出口（铁律 8）：a 优→True，b 优→False，严格平局→None。"""
    if is_better(ra, rb, task_type):
        return True
    if is_better(rb, ra, task_type):
        return False
    return None


def pair_observations(
    task_map: dict,
    keys: Sequence[str],
    *,
    task_type: str = TASK_CLASSIFICATION,
    canonical_seed: int = CANON_SEED,
    consensus_seeds: Sequence[int] = CONSENSUS_SEEDS,
) -> list[PairObs]:
    """{task: {cls: {seed: R'}}} → 配对观测列表（候选集口径同 make_pairs）。

    候选对 = 两侧 canonical R' 均有限的无序对（非有限的 CLS 整体剔除）。
    共识只由 consensus_seeds 给出——canonical 混进共识就是 label 给自己投票，
    正确率被系统性抬高且从结果上看不出来（tests 有专门用例挡这条）。
    """
    out: list[PairObs] = []
    for tid, clsmap in task_map.items():
        avail = [k for k in keys
                 if k in clsmap and np.isfinite(clsmap[k].get(canonical_seed, np.nan))]
        for a, b in combinations(avail, 2):
            ra, rb = clsmap[a][canonical_seed], clsmap[b][canonical_seed]
            shared = [s for s in consensus_seeds
                      if np.isfinite(clsmap[a].get(s, np.nan))
                      and np.isfinite(clsmap[b].get(s, np.nan))]
            cons = None
            if shared:
                cons = _direction(float(np.mean([clsmap[a][s] for s in shared])),
                                  float(np.mean([clsmap[b][s] for s in shared])),
                                  task_type)
            out.append(PairObs(task_id=tid, a=a, b=b, d_canon=float(ra - rb),
                               label_a_better=_direction(ra, rb, task_type),
                               cons_a_better=cons))
    return out


def yield_accuracy(obs: Sequence[PairObs], delta: float) -> dict:
    """给定 δ 的产量与方向正确率（δ 过滤逐字对齐 make_pairs：|Δ|<δ 弃样）。"""
    n_cand = len(obs)
    kept = [o for o in obs
            if abs(o.d_canon) >= delta and o.label_a_better is not None]
    comp = [o for o in kept if o.cons_a_better is not None]
    n_agree = sum(o.label_a_better == o.cons_a_better for o in comp)
    return {
        "delta": delta,
        "n_candidates": n_cand,
        "n_kept": len(kept),
        "yield_frac": len(kept) / n_cand if n_cand else None,
        "n_comparable": len(comp),
        "n_agree": n_agree,
        "accuracy": n_agree / len(comp) if comp else None,
    }


@dataclass(frozen=True)
class Verdict2:
    passed: bool
    label: str                      # pass | fail_noisy | fail_unreliable
    delta_star: Optional[float]     # 满足 A≥floor ∧ Y≥floor 的最小 δ
    accuracy: Optional[float]
    yield_frac: Optional[float]
    n_kept: int
    n_comparable: int
    best_clean_yield: Optional[float]   # 达 A≥floor 的 δ 中最高产量（fail_noisy 用）
    reason: str


def gate_v2_decide(
    obs: Sequence[PairObs],
    *,
    grid: Sequence[float] = DELTA_GRID,
    a_floor: float = A_FLOOR,
    y_floor: float = Y_FLOOR,
) -> Verdict2:
    """pass ⟺ ∃δ∈grid：A(δ)≥a_floor ∧ Y(δ)≥y_floor；δ* 取最小（Y 单调降 → 产量最高）。

    不通过分两型（只是失败原因标注）：
    - fail_noisy：有干净 δ 但那时产量不够（干净得起，label 太少）；
    - fail_unreliable：任何 δ 都达不到正确率下限（方向本身是噪声，PenDigits 模式）。
    """
    clean: list[dict] = []
    for d in grid:
        st = yield_accuracy(obs, d)
        if st["accuracy"] is not None and st["accuracy"] >= a_floor:
            clean.append(st)
            if st["yield_frac"] is not None and st["yield_frac"] >= y_floor:
                return Verdict2(
                    True, "pass", d, st["accuracy"], st["yield_frac"],
                    st["n_kept"], st["n_comparable"], st["yield_frac"],
                    f"δ*={d}：方向正确率 {st['accuracy']:.3f}≥{a_floor} 且产量 "
                    f"{st['yield_frac']:.3f}≥{y_floor:.3f}（{st['n_kept']}/"
                    f"{st['n_candidates']} 条 label）",
                )
    if clean:
        best = max(clean, key=lambda s: s["yield_frac"] or 0.0)
        return Verdict2(
            False, "fail_noisy", None, best["accuracy"], best["yield_frac"],
            best["n_kept"], best["n_comparable"], best["yield_frac"],
            f"存在干净 δ（最优 δ={best['delta']}，正确率 {best['accuracy']:.3f}）"
            f"但产量仅 {best['yield_frac']:.3f}<{y_floor:.3f}"
            f"（{best['n_kept']}/{best['n_candidates']} 条）：抬 δ 抬得干净，label 不够",
        )
    st0 = yield_accuracy(obs, grid[0])
    return Verdict2(
        False, "fail_unreliable", None, st0["accuracy"], st0["yield_frac"],
        st0["n_kept"], st0["n_comparable"], None,
        f"任何 δ∈[{grid[0]}, {grid[-1]}] 都达不到正确率 {a_floor}"
        f"（δ={grid[0]} 处 {st0['accuracy']}）：δ 过滤后方向仍是噪声",
    )


def wilson_lower(k: int, n: int, *, conf: float = 0.90) -> Optional[float]:
    """比例的 Wilson 分数下界（单侧 conf）——n 小时比正态近似保守，仅作诊断。"""
    if n <= 0:
        return None
    z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}[conf]
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


# ===========================================================================
# 向量化重放（敏感性检验要跑 637 子集 × 13 数据集；与上面的纯函数对拍后使用）
# ===========================================================================


@dataclass(frozen=True)
class PairTable:
    """一个数据集的全部候选对，列式存放，供子集重放快速筛选。"""

    a: np.ndarray          # CLS 下标
    b: np.ndarray
    absd: np.ndarray       # |ΔR'| canonical
    has_label: np.ndarray  # canonical 非严格平局
    comparable: np.ndarray # 共识方向有定义
    agree: np.ndarray      # label 方向 == 共识方向


def build_table(obs: Sequence[PairObs], key_index: dict[str, int]) -> PairTable:
    return PairTable(
        a=np.array([key_index[o.a] for o in obs], dtype=int),
        b=np.array([key_index[o.b] for o in obs], dtype=int),
        absd=np.array([abs(o.d_canon) for o in obs], dtype=float),
        has_label=np.array([o.label_a_better is not None for o in obs]),
        comparable=np.array([o.cons_a_better is not None for o in obs]),
        agree=np.array([o.label_a_better is not None and o.cons_a_better is not None
                        and o.label_a_better == o.cons_a_better for o in obs]),
    )


def decide_table(
    tbl: PairTable,
    rows: np.ndarray,
    *,
    grid: Sequence[float] = DELTA_GRID,
    a_floor: float = A_FLOOR,
    y_floor: float = Y_FLOOR,
) -> tuple[bool, str, Optional[float]]:
    """decide 的向量化等价（rows = 参与本次重放的行掩码）。"""
    n_cand = int(rows.sum())
    if n_cand == 0:
        return False, "fail_unreliable", None
    absd, lab = tbl.absd[rows], tbl.has_label[rows]
    comp, agr = tbl.comparable[rows], tbl.agree[rows]
    any_clean = False
    for d in grid:
        keep = (absd >= d) & lab
        n_comp = int((keep & comp).sum())
        if n_comp == 0:
            continue
        acc = int((keep & agr).sum()) / n_comp
        if acc < a_floor:
            continue
        any_clean = True
        if int(keep.sum()) / n_cand >= y_floor:
            return True, "pass", d
    return False, "fail_noisy" if any_clean else "fail_unreliable", None


# ===========================================================================
# 数据加载
# ===========================================================================


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def load_index() -> tuple[dict, dict, list[str], dict]:
    """→ (idx[ds][task][cls][seed]=R', tasks_meta, cls_keys, har_rank)。"""
    meta = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(META_DIR.glob("*.json"))}
    keep = {ds: {t["task_id"] for t in m["tasks"]} for ds, m in meta.items()}
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for rec in read_jsonl(CANON_CACHE) + read_jsonl(NOISE_CACHE):
        ds = rec.get("dataset")
        if ds not in keep or rec["task_id"] not in keep[ds]:
            continue
        rp = rec["r_prime"]
        idx[ds][rec["task_id"]][rec["cls_key"]][rec["seed"]] = (
            float(rp) if rp is not None else float("nan"))
    idx = {ds: {t: {k: dict(sm) for k, sm in cm.items()} for t, cm in tm.items()}
           for ds, tm in idx.items()}
    _, har_rank = load_probe_cls(PROBE_CATALOG, N_PROBE_CLS)
    return idx, meta, list(har_rank), har_rank


# ===========================================================================
# 诊断（CRITERION.md §3.3：落盘但**不参与判定**）
# ===========================================================================


def tail_cls(idx: dict, keys: list[str]) -> list[str]:
    """数据驱动的"尾部/退化 CLS"：跨数据集平均 |稳健 z| > 2（口径同
    probe_sensitivity §5.2，沿用以便与旧分析逐条对照）。仅用于诊断 D1 与
    敏感性检验的剂量-反应，**不从评估集剔除任何 CLS**（CRITERION.md §3.1）。"""
    zs: dict[str, list[float]] = defaultdict(list)
    for tm in idx.values():
        means = {k: np.nanmean([v for t in tm.values() if k in t
                                for v in t[k].values()]) for k in keys}
        arr = np.array([means[k] for k in keys])
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) * 1.4826
        for k in keys:
            zs[k].append(abs((means[k] - med) / mad) if mad > 0 else 0.0)
    return sorted(k for k in keys if float(np.mean(zs[k])) > 2.0)


def trivial_share(obs: Sequence[PairObs], delta: float, tails: set[str]) -> Optional[float]:
    """D1：δ 存活 label 中"涉及尾部 CLS"的比例（高 = label 多是平凡对比）。"""
    kept = [o for o in obs if abs(o.d_canon) >= delta and o.label_a_better is not None]
    if not kept:
        return None
    return sum((o.a in tails) or (o.b in tails) for o in kept) / len(kept)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def task_specificity(idx: dict, keys: list[str]) -> dict:
    """D2：Spearman(本数据集 CLS 排名, 全局平均排名)。≈1 = 偏好就是全局偏好，
    比较器不需要任务嵌入也能拟合（与 M3 路线 B「E 消融零下降」同源的诊断）。"""
    rbar = {}
    for ds, tm in idx.items():
        rbar[ds] = np.array([np.nanmean([v for t in tm.values() if k in t
                                         for v in t[k].values()]) for k in keys])
    ranks = {ds: np.argsort(np.argsort(v)).astype(float) for ds, v in rbar.items()}
    glob = np.mean([ranks[ds] for ds in ranks], axis=0)
    return {ds: round(spearman(rbar[ds], glob), 3) for ds in rbar}


def holdout_delta_bias(idx_ds: dict, keys: list[str], *, grid=DELTA_GRID) -> Optional[dict]:
    """CRITERION.md §5.4：δ* 与 A(δ*) 在同一批配对上选/报 → A 乐观偏置。
    任务级留一（在一个任务上选 δ*，在另一个任务上报正确率）量化这个偏置。"""
    tids = sorted(idx_ds)
    if len(tids) < 2:
        return None
    rows = []
    for fit, ev in ((tids[0], tids[1]), (tids[1], tids[0])):
        v = gate_v2_decide(pair_observations({fit: idx_ds[fit]}, keys), grid=grid)
        d = v.delta_star
        if d is None:
            rows.append({"fit_task": fit, "eval_task": ev, "delta_star": None,
                         "eval_accuracy": None, "eval_yield": None})
            continue
        st = yield_accuracy(pair_observations({ev: idx_ds[ev]}, keys), d)
        rows.append({"fit_task": fit, "eval_task": ev, "delta_star": d,
                     "eval_accuracy": st["accuracy"], "eval_yield": st["yield_frac"]})
    accs = [r["eval_accuracy"] for r in rows if r["eval_accuracy"] is not None]
    return {"rounds": rows,
            "mean_eval_accuracy": round(float(np.mean(accs)), 4) if accs else None}


# ===========================================================================
# 主分析
# ===========================================================================


def evaluate_datasets(idx, keys, registry, labels_cfg, tails, *,
                      canonical_seed=CANON_SEED, consensus_seeds=CONSENSUS_SEEDS) -> dict:
    """13 个数据集的 gate v2 判定 + 诊断 + δ 曲线。"""
    out = {}
    for ds in sorted(idx):
        obs = pair_observations(idx[ds], keys, canonical_seed=canonical_seed,
                                consensus_seeds=consensus_seeds)
        v = gate_v2_decide(obs)
        reg_delta = delta_for(ds, labels_cfg)
        reg = yield_accuracy(obs, reg_delta)
        sug = compute_snr_stats(idx[ds]).suggested_delta          # 本批实测 3σ
        sug_row = yield_accuracy(obs, sug)
        curve = [yield_accuracy(obs, d) for d in DELTA_GRID]
        out[ds] = {
            "verdict": v.label, "passed": v.passed, "reason": v.reason,
            "delta_star": v.delta_star,
            "accuracy": None if v.accuracy is None else round(v.accuracy, 4),
            "yield": None if v.yield_frac is None else round(v.yield_frac, 4),
            "n_kept": v.n_kept, "n_comparable": v.n_comparable,
            "n_candidates": len(obs),
            "wilson90_lo": (None if v.n_comparable == 0 else
                            round(wilson_lower(round((v.accuracy or 0) * v.n_comparable),
                                               v.n_comparable), 4)),
            "best_clean_yield": v.best_clean_yield,
            "registry_delta": {"delta": reg_delta,
                               "accuracy": None if reg["accuracy"] is None else round(reg["accuracy"], 4),
                               "yield": None if reg["yield_frac"] is None else round(reg["yield_frac"], 4),
                               "n_kept": reg["n_kept"]},
            "suggested_3sigma_delta": {"delta": sug,
                                       "accuracy": None if sug_row["accuracy"] is None else round(sug_row["accuracy"], 4),
                                       "yield": None if sug_row["yield_frac"] is None else round(sug_row["yield_frac"], 4),
                                       "n_kept": sug_row["n_kept"]},
            "diagnostics": {
                "trivial_share_at_delta_star": (
                    None if v.delta_star is None else
                    round(trivial_share(obs, v.delta_star, set(tails)) or 0.0, 4)),
                "trivial_share_at_registry_delta": (
                    None if trivial_share(obs, reg_delta, set(tails)) is None else
                    round(trivial_share(obs, reg_delta, set(tails)), 4)),
                "holdout_delta_bias": holdout_delta_bias(idx[ds], keys),
            },
            "curve": [{k: (round(x, 4) if isinstance(x, float) else x)
                       for k, x in c.items()} for c in curve],
        }
    return out


def gate_count(passed: set[str], registry) -> dict:
    src = {e.name for e in registry.source}
    untested_pass = {d for d in passed if d in src}
    g = task_count_gate(registry, passed_untested=untested_pass)
    bonus = sum(e.planned_tasks for e in registry.reserve if e.name in passed)
    return {"count": g.count, "ok": g.ok, "floor": g.floor,
            "passed_source": sorted(untested_pass),
            "passed_reserve": sorted(d for d in passed if d not in src),
            "reserve_bonus": bonus}


def sensitivity(idx, keys, registry, tails, *, quick: bool) -> dict:
    """留一法 + k=5..9 子集穷举下新判据的稳定性（口径同 probe_sensitivity §1/§2）。"""
    key_index = {k: i for i, k in enumerate(keys)}
    tables = {ds: build_table(pair_observations(idx[ds], keys), key_index)
              for ds in sorted(idx)}

    def replay(subset: tuple[str, ...]) -> dict:
        sel = {key_index[k] for k in subset}
        passed, labels = set(), {}
        for ds, tbl in tables.items():
            rows = np.isin(tbl.a, list(sel)) & np.isin(tbl.b, list(sel))
            ok, lab, _ = decide_table(tbl, rows)
            labels[ds] = lab
            if ok:
                passed.add(ds)
        return {"subset": list(subset), "gate": gate_count(passed, registry)["count"],
                "passed": sorted(passed), "labels": labels}

    loo = []
    for drop in keys:
        sub = tuple(k for k in keys if k != drop)
        r = replay(sub)
        loo.append({"dropped": drop, "gate": r["gate"], "passed": r["passed"],
                    "labels": r["labels"]})

    subsets: dict[str, list] = {}
    dose: list[tuple[float, int]] = []
    per_ds_pass: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    if not quick:
        for k in range(5, 10):
            rows = []
            for sub in combinations(keys, k):
                r = replay(sub)
                rows.append({"gate": r["gate"], "passed": r["passed"],
                             "n_tail": sum(x in tails for x in sub)})
                dose.append((sum(x in tails for x in sub) / k, r["gate"]))
                for ds in tables:
                    per_ds_pass[ds][k].append(int(ds in r["passed"]))
            subsets[str(k)] = rows

    gates = [r["gate"] for rows in subsets.values() for r in rows]
    dose_stats = None
    if dose:
        f = np.array([d[0] for d in dose])
        g = np.array([d[1] for d in dose])
        by_n: dict[int, list[int]] = defaultdict(list)
        for rows in subsets.values():
            for r in rows:
                by_n[r["n_tail"]].append(r["gate"])
        dose_stats = {
            "pearson_tailshare_vs_gate": round(float(np.corrcoef(f, g)[0, 1]), 3)
            if g.std() > 0 else 0.0,
            "by_n_tail": {str(n): {"n_subsets": len(v), "mean_gate": round(float(np.mean(v)), 1),
                                   "median_gate": int(np.median(v)),
                                   "range": [int(min(v)), int(max(v))]}
                          for n, v in sorted(by_n.items())},
        }
    return {
        "leave_one_out": loo,
        "loo_gate_values": sorted({r["gate"] for r in loo}),
        "subset_gate_stats": (None if not gates else {
            "n_subsets": len(gates), "mean": round(float(np.mean(gates)), 1),
            "median": int(np.median(gates)), "min": int(min(gates)), "max": int(max(gates)),
            "p10": int(np.percentile(gates, 10)), "p90": int(np.percentile(gates, 90)),
            "p_ge_100": round(float(np.mean(np.array(gates) >= 100)), 3)}),
        "per_dataset_pass_rate": {
            ds: {str(k): round(float(np.mean(v)), 3) for k, v in ks.items()}
            for ds, ks in per_ds_pass.items()},
        "dose_response": dose_stats,
        "tail_cls": sorted(tails),
    }


def threshold_grid(idx, keys, registry) -> list[dict]:
    """阈值网格（**只报告不改判据**）：A_floor × y_floor → 任务数闸。"""
    key_index = {k: i for i, k in enumerate(keys)}
    tables = {ds: build_table(pair_observations(idx[ds], keys), key_index)
              for ds in sorted(idx)}
    allrows = np.ones(len(next(iter(tables.values())).absd), dtype=bool)
    out = []
    for a_floor in (1.0, 0.95, 0.90, 0.85):
        for y_floor in (0.5, 1 / 3, 0.25, 0.1):
            passed = set()
            for ds, tbl in tables.items():
                rows = np.ones(len(tbl.absd), dtype=bool)
                ok, _, _ = decide_table(tbl, rows, a_floor=a_floor, y_floor=y_floor)
                if ok:
                    passed.add(ds)
            g = gate_count(passed, registry)
            out.append({"a_floor": a_floor, "y_floor": round(y_floor, 3),
                        "gate": g["count"], "passed": sorted(passed)})
    del allrows
    return out


def seed_rotation(idx, keys, registry, labels_cfg, tails) -> list[dict]:
    """把 canonical 换成另一个种子（其余三个当共识）——判定对"哪个种子造 label"
    的敏感性。R' 在 5090 上有 ±1 样本抖动（记忆），这一项是必要的。"""
    out = []
    for canon in (0, 1, 2, 3):
        cons = tuple(s for s in (0, 1, 2, 3) if s != canon)
        ev = evaluate_datasets(idx, keys, registry, labels_cfg, tails,
                               canonical_seed=canon, consensus_seeds=cons)
        passed = {ds for ds, r in ev.items() if r["passed"]}
        out.append({"canonical_seed": canon, "gate": gate_count(passed, registry)["count"],
                    "passed": sorted(passed),
                    "verdicts": {ds: r["verdict"] for ds, r in ev.items()}})
    return out


def consensus_variant_majority(idx, keys, registry) -> dict:
    """共识改用逐种子多数票（而非均值）——判据对共识定义的敏感性。"""
    passed = set()
    verdicts = {}
    for ds in sorted(idx):
        obs = []
        for tid, clsmap in idx[ds].items():
            avail = [k for k in keys if k in clsmap
                     and np.isfinite(clsmap[k].get(CANON_SEED, np.nan))]
            for a, b in combinations(avail, 2):
                ra, rb = clsmap[a][CANON_SEED], clsmap[b][CANON_SEED]
                votes = [_direction(clsmap[a][s], clsmap[b][s], TASK_CLASSIFICATION)
                         for s in CONSENSUS_SEEDS
                         if np.isfinite(clsmap[a].get(s, np.nan))
                         and np.isfinite(clsmap[b].get(s, np.nan))]
                votes = [v for v in votes if v is not None]
                cons = None
                if votes:
                    n_a = sum(votes)
                    cons = None if n_a * 2 == len(votes) else (n_a * 2 > len(votes))
                obs.append(PairObs(tid, a, b, float(ra - rb),
                                   _direction(ra, rb, TASK_CLASSIFICATION), cons))
        v = gate_v2_decide(obs)
        verdicts[ds] = v.label
        if v.passed:
            passed.add(ds)
    return {"gate": gate_count(passed, registry)["count"],
            "passed": sorted(passed), "verdicts": verdicts}


def decoupling_preview(idx, keys, registry, tails) -> dict:
    """CRITERION.md §3.2 预登记项的加强版：把两条尾部 CLS 从**评估集**拿掉后重判。

    这不是改判据（判据仍在全 10 条上给出，见 §3.1 为何不剔除），而是回答
    "若探针 CLS 集合真的解耦（从 space.yaml 重采常规 CLS，需 GPU），闸门大概会给
    什么"——是解耦后结果的下界预览。
    """
    sub = tuple(k for k in keys if k not in tails)
    per, passed = {}, set()
    for ds in sorted(idx):
        obs = pair_observations(idx[ds], sub)
        v = gate_v2_decide(obs)
        per[ds] = {"verdict": v.label, "passed": v.passed, "delta_star": v.delta_star,
                   "accuracy": None if v.accuracy is None else round(v.accuracy, 4),
                   "yield": None if v.yield_frac is None else round(v.yield_frac, 4),
                   "n_kept": v.n_kept, "n_candidates": len(obs)}
        if v.passed:
            passed.add(ds)
    g = gate_count(passed, registry)

    # 同一批常规 CLS 上重放**旧判据**作对照（snr_gate.decide 逐字复用）——报告里
    # "旧判据解耦后 = 56" 这个数必须可复现，不能是从 0-tail 子集分布里目测的近似。
    from autocls.snr_gate import decide, is_range_maxed
    old_per, old_passed = {}, set()
    for ds in sorted(idx):
        st = compute_snr_stats({t: {k: sm for k, sm in cm.items() if k in sub}
                                for t, cm in idx[ds].items()})
        e = registry.get(ds)
        v = decide(st, range_maxed=is_range_maxed(tuple(e.n_classes_range), e.n_classes))
        old_per[ds] = {"verdict": v.label, "passed": v.passed,
                       "snr": round(st.snr, 3),
                       "pair_sign_stability": round(st.pair_sign_stability, 3)}
        if v.passed:
            old_passed.add(ds)
    return {"keys": list(sub), "n_cls": len(sub), "gate": g["count"],
            "passed": sorted(passed), "datasets": per,
            "old_criterion_same_subset": {
                "gate": gate_count(old_passed, registry)["count"],
                "passed": sorted(old_passed), "datasets": old_per}}


PROBE_CACHE = ROOT / "results" / "route_a_probe" / "rprime_cache.jsonl"


def probe_consistency() -> Optional[dict]:
    """CRITERION.md §4 的一致性核对：把 gate v2 套回探针那 4 个数据集，应复现
    探针自己的判定（SAD/PhonemeSpectra/Cricket 可用、PenDigits 剔除）。

    **这不是独立验证**（两个阈值就取自那张表），只是端到端跑通 + 不自相矛盾的检查。
    探针缓存是另一批 (8 CLS × 2–3 任务 × 4–5 seed)，canonical 取 seed 0，其余作共识。
    """
    if not PROBE_CACHE.exists():
        return None
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for rec in read_jsonl(PROBE_CACHE):
        rp = rec["r_prime"]
        idx[rec["dataset"]][rec["task_id"]][rec["cls_key"]][rec["seed"]] = (
            float(rp) if rp is not None else float("nan"))
    expect = {"SpokenArabicDigits": True, "PhonemeSpectra": True,
              "Cricket": True, "PenDigits": False}
    out = {}
    for ds, tm in idx.items():
        tm = {t: {k: dict(sm) for k, sm in cm.items()} for t, cm in tm.items()}
        keys = sorted({k for cm in tm.values() for k in cm})
        seeds = sorted({s for cm in tm.values() for sm in cm.values() for s in sm})
        obs = pair_observations(tm, keys, canonical_seed=seeds[0],
                                consensus_seeds=tuple(seeds[1:]))
        v = gate_v2_decide(obs)
        out[ds] = {"verdict": v.label, "passed": v.passed, "delta_star": v.delta_star,
                   "accuracy": None if v.accuracy is None else round(v.accuracy, 4),
                   "yield": None if v.yield_frac is None else round(v.yield_frac, 4),
                   "n_kept": v.n_kept, "n_candidates": len(obs),
                   "n_tasks": len(tm), "n_cls": len(keys), "seeds": seeds,
                   "probe_report_verdict": expect.get(ds),
                   "agrees_with_probe": v.passed == expect.get(ds)}
    return {"datasets": out,
            "all_agree": all(r["agrees_with_probe"] for r in out.values())}


def selfcheck(idx, keys) -> dict:
    """向量化重放必须与纯函数判定逐条一致（否则敏感性检验测的是另一个判据）。"""
    key_index = {k: i for i, k in enumerate(keys)}
    bad = []
    for ds in sorted(idx):
        obs = pair_observations(idx[ds], keys)
        v = gate_v2_decide(obs)
        tbl = build_table(obs, key_index)
        ok, lab, d = decide_table(tbl, np.ones(len(obs), dtype=bool))
        if (ok, lab, d) != (v.passed, v.label, v.delta_star):
            bad.append({"dataset": ds, "pure": [v.passed, v.label, v.delta_star],
                        "table": [ok, lab, d]})
    return {"n_datasets": len(idx), "mismatches": bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过 637 子集穷举")
    args = ap.parse_args()

    idx, meta, keys, har_rank = load_index()
    registry = load_registry()
    labels_cfg = yaml.safe_load(LABELS_CFG.read_text(encoding="utf-8"))
    base = json.loads(BASE_ANALYSIS.read_text(encoding="utf-8"))

    n_rec = sum(len(sm) for tm in idx.values() for cm in tm.values()
                for sm in cm.values())
    print(f"[载入] {len(idx)} 数据集 / {sum(len(t) for t in idx.values())} 任务 / "
          f"{len(keys)} 探针 CLS / {n_rec} 条 R'")

    chk = selfcheck(idx, keys)
    assert not chk["mismatches"], f"向量化重放与纯函数不一致：{chk['mismatches']}"
    print("[自检] 向量化重放 = 纯函数判定 ✓")

    tails = tail_cls(idx, keys)
    print(f"[诊断] 数据驱动尾部 CLS（仅诊断，不剔除）：{tails}")

    ev = evaluate_datasets(idx, keys, registry, labels_cfg, tails)
    passed = {ds for ds, r in ev.items() if r["passed"]}
    gate = gate_count(passed, registry)

    old_pass = {ds for ds, r in base["datasets"].items() if r["passed"]}
    result = {
        "criterion": {"a_floor": A_FLOOR, "y_floor": Y_FLOOR,
                      "delta_grid": list(DELTA_GRID),
                      "canonical_seed": CANON_SEED,
                      "consensus_seeds": list(CONSENSUS_SEEDS),
                      "doc": "results/gate_v2/CRITERION.md"},
        "datasets": ev,
        "gate": gate,
        "old_gate": base["gate"],
        "old_verdicts": {ds: {"verdict": r["verdict"], "passed": r["passed"]}
                         for ds, r in base["datasets"].items()},
        "flips": {ds: {"old": base["datasets"][ds]["verdict"], "new": ev[ds]["verdict"]}
                  for ds in ev if (ds in old_pass) != (ds in passed)},
        "task_specificity": task_specificity(idx, keys),
        "decoupling_preview": decoupling_preview(idx, keys, registry, set(tails)),
        "probe_consistency": probe_consistency(),
        "threshold_grid": threshold_grid(idx, keys, registry),
        "seed_rotation": seed_rotation(idx, keys, registry, labels_cfg, tails),
        "consensus_majority_vote": consensus_variant_majority(idx, keys, registry),
        "sensitivity": sensitivity(idx, keys, registry, set(tails), quick=args.quick),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "gate_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'数据集'.ljust(28)}{'旧判定'.ljust(16)}{'新判定'.ljust(16)}"
          f"δ*     产量    正确率   n_kept")
    for ds in sorted(ev, key=lambda d: (not ev[d]["passed"], d)):
        r, o = ev[ds], base["datasets"][ds]["verdict"]
        ds_star = "—" if r["delta_star"] is None else f"{r['delta_star']:.3f}"
        y = "—" if r["yield"] is None else f"{r['yield']:.3f}"
        a = "—" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        print(f"{ds.ljust(28)}{o.ljust(16)}{r['verdict'].ljust(16)}"
              f"{ds_star:<7}{y:<8}{a:<9}{r['n_kept']}/{r['n_candidates']}")
    print(f"\n任务数闸：{gate['count']}（旧 {base['gate']['count']}）"
          f"，floor={gate['floor']}，通过源={gate['passed_source']}"
          f"，reserve 通过={gate['passed_reserve']}(bonus {gate['reserve_bonus']})")
    s = result["sensitivity"]
    print(f"留一法任务数取值：{s['loo_gate_values']}")
    if s["subset_gate_stats"]:
        g = s["subset_gate_stats"]
        print(f"子集穷举（{g['n_subsets']}）：中位 {g['median']}，"
              f"范围 [{g['min']}, {g['max']}]，P(≥100)={g['p_ge_100']}")
    if s["dose_response"]:
        print(f"剂量-反应 Pearson(尾部 CLS 占比, 任务数)="
              f"{s['dose_response']['pearson_tailshare_vs_gate']}（旧判据 0.751）")
    dp = result["decoupling_preview"]
    print(f"解耦预览（去掉 {len(tails)} 条尾部 CLS，仅 {dp['n_cls']} 条常规）："
          f"任务数 {dp['gate']}，通过={dp['passed']}"
          f"　｜同口径旧判据：{dp['old_criterion_same_subset']['gate']}，"
          f"通过={dp['old_criterion_same_subset']['passed']}")
    pc = result["probe_consistency"]
    if pc:
        print(f"探针 4 数据集一致性核对：{'全部一致 ✓' if pc['all_agree'] else '有分歧 ✗'}"
              f" — " + "; ".join(f"{d}:{r['verdict']}" for d, r in pc["datasets"].items()))
    print(f"\n→ {OUTDIR / 'gate_v2.json'}")


if __name__ == "__main__":
    main()
