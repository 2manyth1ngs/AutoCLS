"""MX-4 WP5：学习曲线外推器驱动的逐次减半淘汰——零 GPU 缓存模拟。

协议：results/mx4_lc_extrapolator/PROTOCOL.md（§0 声明头顶读数已看过；臂表 §2、判读 §3、预测 §4
先于正式跑写死）。数据：results/mx3_pq/{rprime_cache.jsonl, noise_probe.jsonl, delta_calibration.json}（只读）；
对照：results/mx4_prior_sh/analysis.json（WP1，C0 一致性）。

思路：WP1 的 SH 在 @60 / @120 淘汰时按"当前截点真 R'"留前 k；先验 top-6 内 @60→@290 的秩相关中位只有
0.36。本脚本把"留谁"抽象成可换的评分器（`sh_path_scored`），比较六种淘汰规则：
  cur（现行）/ lc_hgb（主臂：源池轨迹上学的增益回归器）/ lc_ridge / lc_hgb_nocls（消融：不给 CLS 编码）/
  blend（z@当前 + λ·先验，λ 训练侧选）/ slope（两点对数线性外推）。
warm start 全部 = 先验排名（WP1 头条臂口径），成本口径与 36 号逐位一致。

复用 36 号的纯函数（zscore_within / prior_scores / rank_by / sh_pick / path_cost_* / gate / strata 等），
不修改 36（WP3 会话在用）。模块：autocls/search/learning_curve.py（任务无关，铁律 7）。

用法：
  python scripts/38_lc_sh_sim.py --analyze                       # 7 折 → analysis.json + tables.md
  python scripts/38_lc_sh_sim.py --analyze --folds Handwriting   # 单折调试
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.learning_curve import LearningCurveExtrapolator, slope_extrapolate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")

# ---------------------------------------------------------------- 协议常量（PROTOCOL §2–§3）
OUT = ROOT / "results" / "mx4_lc_extrapolator"
WP1_ANALYSIS = ROOT / "results" / "mx4_prior_sh" / "analysis.json"
MX3 = pq.OUT
BUDGETS = pq.BUDGETS                                   # (60, 120, 290)：同一轨迹的截点
SH_CONFIGS = ((6, 3, 3), (6, 3, 1), (4, 2, 2), (4, 2, 1), (10, 4, 3), (10, 4, 1))
RULES = ("cur", "lc_hgb", "lc_ridge", "lc_hgb_nocls", "blend", "slope")
MODEL_SPECS = (((60,), 290), ((60,), 120), ((60, 120), 290))   # (观测截点, b_final)
MAIN_RULE = "lc_hgb"
REF_ARM = "SH6-3-3_cur"                                # = WP1 SH6-3-3_prior（C0）
MAIN_ARM = f"SH6-3-3_{MAIN_RULE}"
CHEAP_TIERS = ("SH4-2-2", "SH6-3-1")                   # L2 候选档位
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
ALPHA = 0.05
K_TOP = 3
TOP_N = 6                                              # 机制门看先验 top-6
FRESH_SLICES = ("00", "01")
FRESH_REPS = (1, 2)

Scorer = Callable[[Sequence[str], Sequence[int], int], Mapping[str, float]]


# ---------------------------------------------------------------- 可换评分器的逐次减半
def sh_path_scored(ranked: Sequence[str], ks: Sequence[int], budgets: Sequence[int],
                   scorer: Scorer) -> list[list[str]]:
    """path[i] = 第 i 级在训的候选；第 i≥1 级 = 上一级幸存者按 scorer(幸存者, 已观测截点, b_final) 降序留前 k_i。

    scorer 返回 {key: 分}；并列按 key 升序（逐位确定）。scorer = 当前截点真值时与 36 号 sh_path 逐位相同。
    """
    if len(ks) != len(budgets) or not ks:
        raise ValueError(f"ks {ks} 与 budgets {budgets} 必须等长且非空")
    if list(budgets) != sorted(budgets):
        raise ValueError("budgets 必须升序")
    b_final = int(budgets[-1])
    cur = list(ranked)[:int(ks[0])]
    path = [list(cur)]
    for i in range(1, len(budgets)):
        obs = tuple(int(b) for b in budgets[:i])
        sc = scorer(list(cur), obs, b_final)
        cur = sorted(cur, key=lambda k: (-float(sc[k]), k))[:int(ks[i])]
        path.append(list(cur))
    return path


def scorer_current(traj: Mapping[str, Mapping[int, float]]) -> Scorer:
    """现行规则：按最后一个已观测截点的真 R'。"""
    def f(survivors, observed_budgets, b_final):
        b = int(observed_budgets[-1])
        return {k: float(traj[k][b]) for k in survivors}
    return f


def scorer_slope(traj: Mapping[str, Mapping[int, float]]) -> Scorer:
    """两点对数线性外推；只有一个观测截点时退化为 cur。"""
    def f(survivors, observed_budgets, b_final):
        obs = [int(b) for b in observed_budgets]
        M = np.array([[float(traj[k][b]) for b in obs] for k in survivors], dtype=float)
        pred = slope_extrapolate(M, obs, int(b_final))
        return {k: float(v) for k, v in zip(survivors, pred)}
    return f


def scorer_blend(traj: Mapping[str, Mapping[int, float]], prior: Mapping[str, float], lam: float) -> Scorer:
    """z（在当前幸存者内算，部署可得）+ λ·先验；先验缺席记 0；幸存者同分时 z 恒 0。"""
    def f(survivors, observed_budgets, b_final):
        b = int(observed_budgets[-1])
        v = np.array([float(traj[k][b]) for k in survivors], dtype=float)
        sd = float(v.std())
        z = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
        return {k: float(zk + float(lam) * float(prior.get(k, 0.0))) for k, zk in zip(survivors, z)}
    return f


def scorer_lc(models: Mapping[tuple, object], traj: Mapping[str, Mapping[int, float]],
              cls_vec: Mapping[str, np.ndarray], slice_extra) -> Scorer:
    """外推器规则：按 (已观测截点, b_final) 选模型，预测 R'@b_final；切片常量广播到每个候选。"""
    extra = np.asarray(slice_extra, dtype=float).reshape(-1)

    def f(survivors, observed_budgets, b_final):
        key = (tuple(int(b) for b in observed_budgets), int(b_final))
        if key not in models:
            raise KeyError(f"缺 (观测截点, b_final)={key} 的外推器；已有 {sorted(models)}")
        M = np.array([[float(traj[k][b]) for b in key[0]] for k in survivors], dtype=float)
        C = np.stack([np.asarray(cls_vec[k], dtype=float) for k in survivors])
        E = np.tile(extra, (len(survivors), 1))
        pred = models[key].predict(M, cls_X=C, extra=E)
        return {k: float(v) for k, v in zip(survivors, pred)}
    return f


# ---------------------------------------------------------------- 实例装配
def instance_row_lc(keys: Sequence[str], traj: Mapping[str, Mapping[int, float]],
                    prior_rank: Sequence[str], scorers: Mapping[str, Scorer], *,
                    final_budget: int, budgets: Sequence[int],
                    seconds: Optional[Mapping[str, Mapping[int, float]]],
                    k_top: int = K_TOP, top_n: int = TOP_N) -> dict:
    """一个搜索实例：参考臂（oracle / prior1 / prior3 / prior6_oracle / random1 / bottom）+
    SH 档位 × 淘汰规则臂 + all{n}@b0→top{1,k} × 规则；成本口径同 WP1。

    另记 top6（先验 top-6）与 stage1_scores[rule]（各规则在第一次淘汰时对 top-6 的打分，机制门用）、
    finals[arm]（最后一级幸存者，冠军存活率用）。
    """
    if "cur" not in scorers:
        raise KeyError("scorers 必须含 'cur'（现行规则，C0 一致性的锚）")
    b = int(final_budget)
    stages = [int(x) for x in budgets]
    if stages[-1] != b:
        raise ValueError(f"budgets {budgets} 必须以 final_budget {b} 结尾")
    keys = list(keys)
    truth = {k: float(traj[k][b]) for k in keys}
    vals = list(truth.values())
    oracle = max(vals)
    ranked = list(prior_rank)
    top6 = ranked[:top_n]
    arms: dict[str, float] = {"oracle": oracle, "prior1": truth[ranked[0]],
                              "prior3": max(truth[k] for k in ranked[:k_top]),
                              "prior6_oracle": max(truth[k] for k in top6),
                              "random1": float(np.mean(vals)), "bottom": float(min(vals))}
    cost: dict[str, int] = {"prior1": b, "prior3": k_top * b, "random1": b, "catalog_direct": len(keys) * b}
    secs: Optional[dict[str, float]] = {} if seconds is not None else None
    picks: dict[str, str] = {"prior1": ranked[0],
                             "prior3": sorted(ranked[:k_top], key=lambda k: (-truth[k], k))[0]}
    finals: dict[str, list[str]] = {}
    if secs is not None:
        secs["prior1"] = float(seconds[ranked[0]][b])
        secs["prior3"] = float(sum(float(seconds[k][b]) for k in ranked[:k_top]))
        secs["random1"] = float(np.mean([float(seconds[k][b]) for k in keys]))
        secs["catalog_direct"] = float(sum(float(seconds[k][b]) for k in keys))

    def add(name: str, path: Sequence[Sequence[str]]) -> None:
        pick = _g36.sh_pick(path[-1], traj, b)
        arms[name] = truth[pick]
        picks[name] = pick
        finals[name] = list(path[-1])
        cost[name] = _g36.path_cost_iters(path, stages)
        if secs is not None:
            secs[name] = _g36.path_cost_seconds(path, stages, seconds)

    stage1: dict[str, dict[str, float]] = {}
    if len(stages) >= 2:
        for cfg in SH_CONFIGS:
            ks = tuple(cfg[:len(stages)])
            label = _g36.sh_label(ks)
            for rule, sc in scorers.items():
                add(f"{label}_{rule}", sh_path_scored(ranked, ks, stages, sc))
        n = len(keys)
        for top in (1, k_top):
            ks_all = (n,) + (top,) * (len(stages) - 1)
            for rule, sc in scorers.items():
                add(f"all{n}@{stages[0]}->top{top}_{rule}", sh_path_scored(keys, ks_all, stages, sc))
        stage1 = {rule: {k: float(v) for k, v in sc(list(top6), (stages[0],), b).items()}
                  for rule, sc in scorers.items()}

    regret = {k: oracle - v for k, v in arms.items()}
    hits = {k: bool(_g36.bottom_quartile_hit(truth, p)) for k, p in picks.items()}
    return {"arms": arms, "regret": regret, "cost_iters": cost, "cost_seconds": secs,
            "picks": picks, "finals": finals, "bottom_quartile_hit": hits,
            "top6": list(top6), "stage1_scores": stage1}


def tune_lambda(trajs: Sequence[Mapping[str, Mapping[int, float]]], prior: Mapping[str, float],
                grid: Sequence[float], ks: Sequence[int], budgets: Sequence[int]) -> float:
    """训练侧选 λ：各切片按先验 warm start 跑 SH(ks) + blend 淘汰，取 mean regret 最小的 λ（并列取小）。"""
    b = int(budgets[-1])
    best: Optional[tuple[float, float]] = None
    for lam in grid:
        regs = []
        for traj in trajs:
            keys = sorted(traj)
            ranked = _g36.rank_by(prior, keys)
            path = sh_path_scored(ranked, ks, budgets, scorer_blend(traj, prior, float(lam)))
            pick = _g36.sh_pick(path[-1], traj, b)
            regs.append(max(float(traj[k][b]) for k in keys) - float(traj[pick][b]))
        m = float(np.mean(regs))
        if best is None or m < best[0] - 1e-12:
            best = (m, float(lam))
    assert best is not None
    return best[1]


def mechanism_metrics(keys: Sequence[str], truth: Mapping[str, float],
                      scores: Mapping[str, Mapping[str, float]], delta: float) -> dict[str, dict]:
    """top-k 内各规则打分 vs 真值：Spearman（常量打分记 0）与 |Δ真值| ≥ δ 的无序对配对正确数。"""
    keys = list(keys)
    t = np.array([float(truth[k]) for k in keys], dtype=float)
    pairs = [(i, j) for i in range(len(keys)) for j in range(i + 1, len(keys)) if abs(t[i] - t[j]) >= float(delta)]
    out = {}
    for rule, sc in scores.items():
        s = np.array([float(sc[k]) for k in keys], dtype=float)
        if len(keys) < 3 or s.std() == 0 or t.std() == 0:
            rho = 0.0
        else:
            rho = float(spearmanr(s, t).correlation)
        correct = sum(1 for i, j in pairs if (s[i] - s[j]) * (t[i] - t[j]) > 0)
        out[rule] = {"spearman": rho, "pair_correct": int(correct), "pair_total": len(pairs)}
    return out


# ---------------------------------------------------------------- 门 / 读数码
def consistency_check(rows: Sequence[dict], wp1_rows: Sequence[dict], tol: float = 1e-9) -> dict:
    """C0：本脚本 SH6-3-3_cur @290 与 WP1 SH6-3-3_prior @290 逐实例（按 slice）相同。"""
    ref = {r["slice"]: float(r["regret"]["SH6-3-3_prior"]) for r in wp1_rows if int(r["budget"]) == 290}
    diffs, missing = [], []
    for r in rows:
        if int(r["budget"]) != 290:
            continue
        if r["slice"] not in ref:
            missing.append(r["slice"])
            continue
        diffs.append(abs(float(r["regret"][REF_ARM]) - ref[r["slice"]]))
    mx = max(diffs) if diffs else float("inf")
    return {"n": len(diffs), "max_abs_diff": float(mx), "missing": missing,
            "pass": bool(diffs and not missing and mx <= tol)}


def run_gates(rows290: Sequence[dict], fresh_rows: Sequence[dict], mech: Mapping[str, Mapping[str, float]],
              c0_pass: bool) -> dict:
    """PROTOCOL §3：C0 / L1（主）/ L2（次，两档位取存在）/ L3（机制）/ L4（新鲜种子，描述性）。"""
    g: dict = {"C0": {"pass": bool(c0_pass)}}
    g["L1"] = _g36.gate(rows290, MAIN_ARM, REF_ARM)
    ref = np.array([float(r["regret"][REF_ARM]) for r in rows290])
    l2: dict = {"tiers": {}, "pass": False, "winner": None, "mean_ref": float(ref.mean())}
    for tier in CHEAP_TIERS:
        arm = f"{tier}_{MAIN_RULE}"
        x = np.array([float(r["regret"][arm]) for r in rows290])
        p_worse = float(_g36.paired_wilcoxon_less(ref, x))           # H1：ref < x，即该档位更差
        ok = bool(x.mean() <= ref.mean() and p_worse >= ALPHA)
        l2["tiers"][arm] = {"mean": float(x.mean()), "median": float(np.median(x)), "p_worse": p_worse,
                            "n_better": int(np.sum(x < ref)), "n_worse": int(np.sum(x > ref)),
                            "n_tie": int(np.sum(x == ref)), "pass": ok}
        if ok and l2["winner"] is None:
            l2["winner"], l2["pass"] = arm, True
    g["L2"] = l2
    a, c = mech[MAIN_RULE], mech["cur"]
    g["L3"] = {"spearman_median": {MAIN_RULE: float(a["spearman_median"]), "cur": float(c["spearman_median"])},
               "pair_acc": {MAIN_RULE: float(a["pair_acc"]), "cur": float(c["pair_acc"])},
               "pass": bool(a["spearman_median"] > c["spearman_median"] and a["pair_acc"] > c["pair_acc"])}
    if fresh_rows:
        x = np.array([float(r["regret"][MAIN_ARM]) for r in fresh_rows])
        y = np.array([float(r["regret"][REF_ARM]) for r in fresh_rows])
        g["L4"] = {"n": len(fresh_rows), "descriptive": True, "mean_a": float(x.mean()), "mean_b": float(y.mean()),
                   "p": float(_g36.paired_wilcoxon_less(x, y)), "pass": bool(x.mean() <= y.mean())}
    else:
        g["L4"] = {"n": 0, "descriptive": True, "mean_a": None, "mean_b": None, "p": None, "pass": None}
    return g


def verdict_code(g: Mapping[str, dict]) -> str:
    if not g["C0"]["pass"]:
        return "INVALID"
    l1, l2, l3 = (bool(g[k].get("pass")) for k in ("L1", "L2", "L3"))
    if l1 and l3:
        return "LC-WIN"
    if (not l1) and l2 and l3:
        return "LC-CHEAP"
    return "LC-DEAD"


# ---------------------------------------------------------------- 装载 / 特征 / 拟合
def load_rows(mx3: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(全部无错行含 GGS, 目录行 = 非 GGS, 噪声探针行)。"""
    rows = [r for r in read_jsonl(mx3 / "rprime_cache.jsonl") if r.get("error") is None]
    cat = [r for r in rows if r["origin"] != "ggs"]
    probe_path = mx3 / "noise_probe.jsonl"
    probe = ([r for r in read_jsonl(probe_path) if r.get("error") is None] if probe_path.exists() else [])
    return rows, cat, probe


def slice_extra(row: Mapping) -> np.ndarray:
    """切片常量：[log10 n_train, log10 n_eval, 类数]（PROTOCOL §2）。"""
    return np.array([math.log10(int(row["n_train"])), math.log10(int(row["n_eval"])), len(row["classes"])], dtype=float)


def group_trajectories(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    """(task_id, cls_key) → {"r": {iters: R'}, "cls": CLS 字典, "extra": 切片常量}。"""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["task_id"], r["cls_key"])
        rec = out.setdefault(key, {"r": {}, "cls": r["cls"], "extra": slice_extra(r)})
        rec["r"][int(r["iters"])] = float(r["r_prime"])
    return out


def fit_models(train_rows: Sequence[dict], encoder: CLSEncoder, *, kind: str, use_cls: bool,
               seed: int) -> dict[tuple, LearningCurveExtrapolator]:
    """每个 (观测截点, b_final) 一个外推器，只用训练侧轨迹（铁律 6 由调用方保证）。"""
    trajs = group_trajectories(train_rows)
    models = {}
    for obs, b_final in MODEL_SPECS:
        need = set(obs) | {b_final}
        recs = [t for t in trajs.values() if need <= set(t["r"])]
        M = np.array([[t["r"][b] for b in obs] for t in recs], dtype=float)
        y = np.array([t["r"][b_final] for t in recs], dtype=float)
        C = np.stack([encoder.encode(CLS(**t["cls"])) for t in recs]) if use_cls else None
        E = np.stack([t["extra"] for t in recs])
        models[(tuple(obs), int(b_final))] = LearningCurveExtrapolator(
            kind=kind, use_cls=use_cls, seed=seed).fit(M, y, cls_X=C, extra=E)
    return models


def fold_instances(target: str, all_rows: Sequence[dict], cat_z: Sequence[dict], probe: Sequence[dict],
                   encoder: CLSEncoder, delta: Mapping[tuple[str, int], float], *, seed: int = 0):
    """一折：外推器 / 先验 / λ 只见训练侧；靶切片 × b_final ∈ {120, 290} 各出一个实例；新鲜种子行另出。"""
    pq.fold_split(target)                                        # 合法性（SCP2 不可作靶）
    train_all = [r for r in all_rows if r["dataset"] != target]  # 含 GGS 行：819 条轨迹
    train_cat = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior = _g36.prior_scores([r for r in train_cat if r["origin"] == ORIGIN_SHARED])
    models = {"lc_hgb": fit_models(train_all, encoder, kind="hgb", use_cls=True, seed=seed),
              "lc_ridge": fit_models(train_all, encoder, kind="ridge", use_cls=True, seed=seed),
              "lc_hgb_nocls": fit_models(train_all, encoder, kind="hgb", use_cls=False, seed=seed)}
    train_trajs = []
    for s in sorted({r["task_id"] for r in train_cat}):
        traj, _ = _g36.trajectories([r for r in train_cat if r["task_id"] == s])
        train_trajs.append(traj)
    lam = tune_lambda(train_trajs, prior, LAMBDA_GRID, (6, 3, 3), BUDGETS)

    def make_scorers(traj, cls_vec, extra) -> dict[str, Scorer]:
        sc = {"cur": scorer_current(traj)}
        for rule in ("lc_hgb", "lc_ridge", "lc_hgb_nocls"):
            sc[rule] = scorer_lc(models[rule], traj, cls_vec, extra)
        sc["blend"] = scorer_blend(traj, prior, lam)
        sc["slope"] = scorer_slope(traj)
        return sc

    out = []
    for s in sorted({r["task_id"] for r in test}):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, secs = _g36.trajectories(rows_s)
        keys = sorted(traj)
        cls_vec = {r["cls_key"]: encoder.encode(CLS(**r["cls"])) for r in rows_s}
        extra = slice_extra(rows_s[0])
        scorers = make_scorers(traj, cls_vec, extra)
        ranked = _g36.rank_by(prior, keys)
        for b in BUDGETS[1:]:
            stages = tuple(x for x in BUDGETS if x <= b)
            row = instance_row_lc(keys, traj, ranked, scorers, final_budget=b, budgets=stages, seconds=secs)
            row.update({"fold": target, "dataset": target, "slice": s, "budget": int(b),
                        "n_candidates": len(keys), "lambda": lam})
            truth = {k: float(traj[k][b]) for k in keys}
            row["mechanism"] = mechanism_metrics(row["top6"], truth, row["stage1_scores"], delta[(target, int(b))])
            best6 = max(row["top6"], key=lambda k: (truth[k], k))
            label = _g36.sh_label(tuple((6, 3, 3)[:len(stages)]))
            row["winner_survived"] = {rule: bool(best6 in row["finals"][f"{label}_{rule}"]) for rule in scorers}
            out.append(row)
    fresh = fresh_instances(target, cat_z, probe, prior, models, encoder)
    return out, fresh, {"prior": prior, "prior_top": _g36.rank_by(prior, list(prior))[:6], "lambda": lam}


def fresh_instances(target: str, cat_z: Sequence[dict], probe: Sequence[dict], prior: Mapping[str, float],
                    models: Mapping[str, Mapping], encoder: CLSEncoder) -> list[dict]:
    """L4：切片 00/01 × shared-10；选择只用 canonical 轨迹（cur 与 lc_hgb 同规则），真值 = 探针 rep 均值 @290。"""
    b = max(BUDGETS)
    stages = tuple(BUDGETS)
    by_cell: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in probe:
        if r["dataset"] == target and int(r["iters"]) == b:
            by_cell[(r["task_id"], r["cls_key"])][int(r["seed"])] = float(r["r_prime"])
    out = []
    for suffix in FRESH_SLICES:
        s = f"{target}_{suffix}"
        rows_s = [r for r in cat_z if r["task_id"] == s and r["origin"] == ORIGIN_SHARED]
        if not rows_s:
            continue
        traj, _ = _g36.trajectories(rows_s)
        keys = sorted(k for k in traj if all(rep in by_cell.get((s, k), {}) for rep in FRESH_REPS))
        if len(keys) < 2:
            continue
        fresh = {k: float(np.mean([by_cell[(s, k)][rep] for rep in FRESH_REPS])) for k in keys}
        oracle = max(fresh.values())
        cls_vec = {r["cls_key"]: encoder.encode(CLS(**r["cls"])) for r in rows_s}
        scorers = {"cur": scorer_current(traj),
                   MAIN_RULE: scorer_lc(models[MAIN_RULE], traj, cls_vec, slice_extra(rows_s[0]))}
        ranked = _g36.rank_by(prior, keys)
        arms = {"oracle": oracle, "prior1": fresh[ranked[0]],
                "prior3": fresh[_g36.sh_pick(ranked[:K_TOP], traj, b)],
                "random1": float(np.mean(list(fresh.values())))}
        same = {k: float(traj[k][b]) for k in keys}
        arms_same = {"oracle": max(same.values()), "prior1": same[ranked[0]],
                     "prior3": same[_g36.sh_pick(ranked[:K_TOP], traj, b)]}
        for rule, sc in scorers.items():
            path = sh_path_scored(ranked, (6, 3, 3), stages, sc)
            pick = _g36.sh_pick(path[-1], traj, b)
            name = f"SH6-3-3_{rule}"
            arms[name] = fresh[pick]
            arms[name + "_fresh_max"] = max(fresh[k] for k in path[-1])
            arms_same[name] = same[pick]
        out.append({"fold": target, "slice": s, "n_keys": len(keys), "arms": arms,
                    "regret": {k: oracle - v for k, v in arms.items()},
                    "regret_same_trajectory": {k: arms_same["oracle"] - v for k, v in arms_same.items()}})
    return out


# ---------------------------------------------------------------- 汇总
def aggregate_mechanism(rows290: Sequence[dict]) -> dict[str, dict]:
    """逐规则：Spearman 逐实例中位 / 均值、δ 存活对 pooled 配对准确率、top-6 冠军存活率（SH6-3-3）。"""
    acc: dict[str, dict] = defaultdict(lambda: {"rho": [], "correct": 0, "total": 0, "survived": []})
    for r in rows290:
        for rule, m in r["mechanism"].items():
            a = acc[rule]
            a["rho"].append(float(m["spearman"]))
            a["correct"] += int(m["pair_correct"])
            a["total"] += int(m["pair_total"])
            a["survived"].append(bool(r["winner_survived"][rule]))
    return {rule: {"spearman_median": float(np.median(a["rho"])), "spearman_mean": float(np.mean(a["rho"])),
                   "pair_acc": (a["correct"] / a["total"]) if a["total"] else float("nan"),
                   "pair_correct": a["correct"], "pair_total": a["total"],
                   "winner_survival": float(np.mean(a["survived"])), "n": len(a["rho"])}
            for rule, a in sorted(acc.items())}


def predictions(m290: Mapping[str, float], mech: Mapping[str, dict]) -> dict:
    """PROTOCOL §4 P1–P6 对账。"""
    p = {}
    v = mech[MAIN_RULE]["spearman_median"]
    p["P1"] = {"claim": "top-6 内 lc_hgb Spearman 中位 ∈ [0.40, 0.60]", "value": v, "hit": bool(0.40 <= v <= 0.60)}
    v = m290[MAIN_ARM]
    p["P2"] = {"claim": "SH6-3-3_lc_hgb @290 ∈ [0.013, 0.019]", "value": v, "hit": bool(0.013 <= v <= 0.019)}
    v = m290[f"SH6-3-1_{MAIN_RULE}"]
    p["P3"] = {"claim": "SH6-3-1_lc_hgb @290 < 0.030", "value": v, "hit": bool(v < 0.030)}
    gain_lc = m290["SH6-3-1_cur"] - m290[f"SH6-3-1_{MAIN_RULE}"]
    gain_bl = m290["SH6-3-1_cur"] - m290["SH6-3-1_blend"]
    p["P4"] = {"claim": "SH6-3-1 上 blend 拿到 lc_hgb 增益的 ≥50%", "gain_lc": gain_lc, "gain_blend": gain_bl,
               "hit": bool(gain_lc > 0 and gain_bl >= 0.5 * gain_lc),
               "note": None if gain_lc > 0 else "lc_hgb 在 SH6-3-1 无增益，P4 不适用（记 ✗）"}
    d = abs(m290["SH6-3-3_lc_hgb_nocls"] - m290[REF_ARM])
    p["P5"] = {"claim": "SH6-3-3_lc_hgb_nocls 与 cur |Δ| < 0.003", "value": d, "hit": bool(d < 0.003)}
    a, c = m290[f"all20@60->top3_{MAIN_RULE}"], m290["all20@60->top3_cur"]
    p["P6"] = {"claim": "all20@60->top3_lc_hgb ≤ all20@60->top3_cur", "lc": a, "cur": c, "hit": bool(a <= c)}
    return p


def markdown_tables(analysis: dict) -> str:
    L = []
    for budget in ("290", "120"):
        strat = analysis["by_budget"].get(budget)
        if not strat:
            continue
        m, md, c = strat["mean_regret"], strat["median_regret"], analysis["cost"][budget]
        aw, n = analysis["avoid_worst"], strat["n"]
        L += [f"## b_final={budget}：淘汰规则臂（行 = SH 档位 / 参考臂，列 = 规则）mean regret（n={n}）", ""]
        rules = [r for r in RULES if any(f"SH6-3_{r}" in m or f"SH6-3-3_{r}" in m for _ in (0,))]
        L.append("| 臂 | iters/实例 | " + " | ".join(rules) + " |")
        L.append("|---|---:|" + "|".join(["---:"] * len(rules)) + "|")
        labels = []
        nst = 3 if budget == "290" else 2
        for cfg in SH_CONFIGS:
            lab = _g36.sh_label(cfg[:nst])
            if lab not in labels:
                labels.append(lab)
        labels += [f"all20@60->top{t}" for t in (1, K_TOP)]
        for lab in labels:
            it = c["iters"].get(f"{lab}_cur")
            L.append(f"| {lab} | {(it / n) if it else float('nan'):.0f} | " +
                     " | ".join(f"{m[f'{lab}_{r}']:.4f} (med {md[f'{lab}_{r}']:.3f})" if f"{lab}_{r}" in m else "—"
                                for r in rules) + " |")
        L += ["", "参考臂：" + "；".join(f"{a} {m[a]:.4f}" for a in ("oracle", "prior1", "prior3", "prior6_oracle", "random1")
                                        if a in m), ""]
        L += [f"垫底四分位命中（{budget}）：" + "；".join(
            f"{a} {aw[a]['hits']}/{aw[a]['n']}" for a in (REF_ARM, MAIN_ARM, f"SH6-3-1_cur", f"SH6-3-1_{MAIN_RULE}",
                                                       f"SH4-2-2_cur", f"SH4-2-2_{MAIN_RULE}") if a in aw), ""]
    mech = analysis["mechanism"]
    L += ["## 机制（@290，先验 top-6 内，第一次淘汰时的打分 vs R'@290）", "",
          "| 规则 | Spearman 中位 | Spearman 均值 | δ 存活对配对 acc | 对数 | top-6 冠军存活率（SH6-3-3） |",
          "|---|---:|---:|---:|---:|---:|"]
    for rule, a in mech.items():
        L.append(f"| {rule} | {a['spearman_median']:.3f} | {a['spearman_mean']:.3f} | {a['pair_acc']:.3f} | "
                 f"{a['pair_correct']}/{a['pair_total']} | {a['winner_survival']:.3f} |")
    L += ["", "## 分数据集 @290：SH6-3-3 cur vs lc_hgb vs blend（mean regret）", "",
          "| 数据集 | n | cur | lc_hgb | blend | prior6_oracle | λ |", "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, strat in analysis["by_dataset_290"].items():
        m = strat["mean_regret"]
        L.append(f"| {ds} | {strat['n']} | {m[REF_ARM]:.4f} | {m[MAIN_ARM]:.4f} | {m['SH6-3-3_blend']:.4f} | "
                 f"{m['prior6_oracle']:.4f} | {analysis['fold_meta'][ds]['lambda']} |")
    g = analysis["gates"]
    L += ["", "## 门（PROTOCOL §3）", "", "| 门 | 内容 | 读数 | 过 |", "|---|---|---|---|"]
    c0 = analysis["consistency"]
    L.append(f"| C0 | SH6-3-3_cur ≡ WP1 SH6-3-3_prior @290 | n={c0['n']} max|Δ|={c0['max_abs_diff']:.2e} | {'✓' if c0['pass'] else '✗'} |")
    l1 = g["L1"]
    L.append(f"| L1（主） | {MAIN_ARM} < {REF_ARM} | {l1['mean_a']:.4f} vs {l1['mean_b']:.4f}，p={l1['p']:.3g}，"
             f"{l1['n_better']}胜/{l1['n_worse']}负/{l1['n_tie']}平 | {'✓' if l1['pass'] else '✗'} |")
    for arm, t in g["L2"]["tiers"].items():
        L.append(f"| L2 候选 | {arm} ≤ {REF_ARM} 且不显著更差 | {t['mean']:.4f} vs {g['L2']['mean_ref']:.4f}，"
                 f"p_worse={t['p_worse']:.3g}，{t['n_better']}胜/{t['n_worse']}负/{t['n_tie']}平 | {'✓' if t['pass'] else '✗'} |")
    L.append(f"| L2（次） | 任一档位 | winner={g['L2']['winner']} | {'✓' if g['L2']['pass'] else '✗'} |")
    l3 = g["L3"]
    L.append(f"| L3（机制） | Spearman 中位 & 配对 acc 都高于 cur | ρ {l3['spearman_median'][MAIN_RULE]:.3f} vs "
             f"{l3['spearman_median']['cur']:.3f}；acc {l3['pair_acc'][MAIN_RULE]:.3f} vs {l3['pair_acc']['cur']:.3f} | "
             f"{'✓' if l3['pass'] else '✗'} |")
    l4 = g["L4"]
    l4s = (f"{l4['mean_a']:.4f} vs {l4['mean_b']:.4f}，p={l4['p']:.2f}，n={l4['n']}" if l4["n"] else "无新鲜种子行")
    L.append(f"| L4（描述性） | 新鲜种子真值 {MAIN_ARM} ≤ {REF_ARM} | {l4s} | {'✓' if l4['pass'] else ('—' if l4['pass'] is None else '✗')} |")
    L += ["", f"读数码：**{analysis['verdict']}**", ""]
    L += ["## 预测对账（PROTOCOL §4）", "", "| 预测 | 内容 | 读数 | 命中 |", "|---|---|---|---|"]
    for k, p in analysis["predictions"].items():
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in p.items() if kk not in ("claim", "hit")}
        L.append(f"| {k} | {p['claim']} | {detail} | {'✓' if p['hit'] else '✗'} |")
    fr = analysis["fresh"]
    if fr["rows"]:
        L += ["", "## 新鲜种子复核（L4，n=%d）：mean regret" % len(fr["rows"]), "",
              "| 口径 | prior1 | prior3 | SH6-3-3_cur | SH6-3-3_lc_hgb | cur 幸存集乐观上界 | lc 幸存集乐观上界 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        a, s = fr["mean_regret"], fr["mean_regret_same_trajectory"]
        L.append(f"| 同轨迹真值 | {s['prior1']:.4f} | {s['prior3']:.4f} | {s[REF_ARM]:.4f} | {s[MAIN_ARM]:.4f} | — | — |")
        L.append(f"| 新鲜种子真值 | {a['prior1']:.4f} | {a['prior3']:.4f} | {a[REF_ARM]:.4f} | {a[MAIN_ARM]:.4f} | "
                 f"{a[REF_ARM + '_fresh_max']:.4f} | {a[MAIN_ARM + '_fresh_max']:.4f} |")
    return "\n".join(L) + "\n"


def cmd_analyze(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0) -> dict:
    all_rows, cat, probe = load_rows(mx3)
    cat_z = _g36.zscore_within(cat)
    delta = pq.load_delta_calibration(mx3 / "delta_calibration.json")
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space, mode="binned")                  # PROTOCOL §2
    rows: list[dict] = []
    fresh: list[dict] = []
    fold_meta = {}
    for target in folds:
        inst, fr, meta = fold_instances(target, all_rows, cat_z, probe, encoder, delta, seed=seed)
        rows += inst
        fresh += fr
        fold_meta[target] = {"prior_top6": meta["prior_top"], "lambda": meta["lambda"]}
        r290 = [r for r in inst if r["budget"] == 290]
        print(f"[fold {target}] {len(inst)} 实例；λ={meta['lambda']}；@290 {REF_ARM} "
              f"{np.mean([r['regret'][REF_ARM] for r in r290]):.4f} vs {MAIN_ARM} "
              f"{np.mean([r['regret'][MAIN_ARM] for r in r290]):.4f}", flush=True)
    rows290 = [r for r in rows if r["budget"] == 290]
    by_budget = _g36.strata(rows, "budget")
    by_dataset_290 = _g36.strata(rows290, "dataset")
    mech = aggregate_mechanism(rows290)
    wp1 = json.loads(WP1_ANALYSIS.read_text(encoding="utf-8"))["per_instance"] if WP1_ANALYSIS.exists() else []
    c0 = consistency_check(rows, wp1) if wp1 else {"n": 0, "max_abs_diff": float("inf"), "missing": [],
                                                    "pass": False, "note": "WP1 analysis.json 缺失"}
    gates = run_gates(rows290, fresh, mech, c0["pass"])
    analysis = {
        "protocol": "results/mx4_lc_extrapolator/PROTOCOL.md",
        "source": str(mx3.relative_to(ROOT)) if mx3.is_relative_to(ROOT) else str(mx3),
        "folds": list(folds), "n_instances": len(rows), "n_instances_290": len(rows290), "n_fresh": len(fresh),
        "mean_regret": _g36.mean_regret(rows), "median_regret": _g36.median_regret(rows),
        "by_budget": by_budget, "by_dataset_290": by_dataset_290,
        "mechanism": mech, "consistency": c0, "gates": gates, "verdict": verdict_code(gates),
        "fresh": {"rows": fresh, "mean_regret": _g36.mean_regret(fresh),
                  "mean_regret_same_trajectory": _g36.mean_regret(
                      [{"regret": r["regret_same_trajectory"]} for r in fresh])},
        "cost": _g36.cost_summary(rows), "avoid_worst": _g36.avoid_worst(rows),
        "fold_meta": fold_meta,
        "per_instance": [{k: v for k, v in r.items() if k not in ("stage1_scores",)} for r in rows],
    }
    analysis["predictions"] = predictions(by_budget["290"]["mean_regret"], mech)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(analysis), encoding="utf-8")
    m = by_budget["290"]["mean_regret"]
    print(f"verdict {analysis['verdict']}；C0 {'✓' if c0['pass'] else '✗'}；@290 {REF_ARM} {m[REF_ARM]:.4f} / "
          f"{MAIN_ARM} {m[MAIN_ARM]:.4f} / SH4-2-2_lc {m['SH4-2-2_' + MAIN_RULE]:.4f} / "
          f"SH6-3-1_lc {m['SH6-3-1_' + MAIN_RULE]:.4f}")
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--mx3", default=str(MX3))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.analyze:
        cmd_analyze(Path(args.mx3), Path(args.outdir), args.folds, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
