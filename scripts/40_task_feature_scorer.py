"""MX-4 WP6-TF：任务特征对零样本 CLS 排序的影响——现构造（打分器 + 成绩单）上的零 GPU 验证。

计划：~/.claude/plans/ts2vec-e-quirky-bubble.md（用户 2026-09-04 拍板：家族 A 预测型 + B 转导型；
起跑名单可自适应、淘汰规则不动；主判据 = oracle 入榜率 + regret@3；只偷看阶段 0 再锁 A/B 协议）。
数据：results/mx3_pq/{rprime_cache.jsonl, noise_probe.jsonl}（只读）；对照：results/mx4_prior_sh/analysis.json（WP1）。

阶段 0（本文件当前实现；先跑先看，结果给用户后再锁 A/B 协议）：
  C0 复现闸  ——本脚本重算 prior / gbdt / hybrid 名单与 SH6-3-3_prior @290 regret，须与 WP1 逐实例逐位相同；
             同时落盘"oracle ∉ hybrid top-6"的实例数、来源（shared/random）与数据集分布（0.0118 的拆账）。
  C1 上界    ——数据集身份 oracle：靶数据集内 leave-one-slice-out，shared 按同数据集其余切片 z 均值、
             random 按 hgb ⊕ onehot(dataset)；这是"任何数据集级特征能做到的最好"。
  0b 交互定位——逐数据集 ridge_binned 字段效应的跨数据集方差 / 切片 bootstrap 方差之比；逐数据集
             shared-10 z 均值 vs LODO 先验的 Spearman；shared-10 × 45 切片 R'@290 的嵌套方差分解
             （任务 / CLS / 数据集级交互 / 切片级交互）与噪声探针 σ²。
统一记法：s0 = 现行打分器（shared → 先验 z 均值、random → 任务盲 hgb 于 scalar 编码，与 36 号逐位同）；
hybrid = s0 + HYBRID_BIAS·[shared]（36 号 hybrid）；s0z = 无偏置版（阶段 0 新增的廉价对照）。
复用 36 号纯函数（importlib，不改 36/37/39 与 autocls/search/surrogate.py——铁律 7：模块只见向量）。

用法：
  python scripts/40_task_feature_scorer.py --stage0                 # 7 折 → results/mx4_task_feature/stage0.{json,md}
  python scripts/40_task_feature_scorer.py --stage0 --folds Handwriting --n-boot 5
"""
from __future__ import annotations

import argparse
import importlib
import json
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
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")

# ---------------------------------------------------------------- 常量
OUT = ROOT / "results" / "mx4_task_feature"
WP1_ANALYSIS = ROOT / "results" / "mx4_prior_sh" / "analysis.json"
MX3 = pq.OUT
BUDGETS = pq.BUDGETS                      # (60, 120, 290)：同一轨迹的截点
DATASETS = tuple(pq.PQ_DATASETS)          # onehot(dataset) 的列序（8）
TOP_N = 6                                 # 起跑名单长度（SH6-3-3 第一级）
K_TOP = 3
SH_MAIN = (6, 3, 3)
SH_PLUS = (8, 3, 3)                       # SH6+2-3-3 ≡ SH8-3-3（成本 1170）
HYBRID_BIAS = _g36.HYBRID_BIAS
S0_ARMS = ("prior", "gbdt", "hybrid", "s0z")
C1_ARMS = ("c1_pure_z", "c1_pure_hyb", "c1_blend_z", "c1_blend_hyb")
C1_BLEND = 0.5
RIDGE_ALPHA = 3.0                         # 与 37 号 field_effects 同
C0_TOL = 1e-9


# ---------------------------------------------------------------- 现行打分器 s0
def fit_s0(train_rows: Sequence[dict], enc: CLSEncoder, *, seed: int = 0) -> tuple[dict, FieldSurrogate]:
    """(先验 z 均值 dict, 任务盲 hgb) —— 与 36 号 fold_instances 的 prior / model 逐位同构。"""
    prior = _g36.prior_scores([r for r in train_rows if r["origin"] == ORIGIN_SHARED])
    blind = FieldSurrogate(kind="hgb", seed=seed).fit(
        _g36.features(train_rows, enc, max(BUDGETS)), np.array([float(r["z"]) for r in train_rows]))
    return prior, blind


def s0_scores(rows_sb: Sequence[dict], prior: Mapping[str, float], blind, enc: CLSEncoder) -> dict[str, float]:
    """z 尺度：shared 键取先验，其余取任务盲预测（按 rows_sb 顺序）。"""
    pred, _ = blind.predict(_g36.features(rows_sb, enc, max(BUDGETS)))
    return {r["cls_key"]: (float(prior[r["cls_key"]]) if r["cls_key"] in prior else float(pred[i]))
            for i, r in enumerate(rows_sb)}


def with_bias(scores: Mapping[str, float], rows_sb: Sequence[dict], bias: float = HYBRID_BIAS) -> dict[str, float]:
    """hybrid：shared 置顶（36 号 HYBRID_BIAS 语义）。"""
    shared = {r["cls_key"] for r in rows_sb if r["origin"] == ORIGIN_SHARED}
    return {k: float(v) + (bias if k in shared else 0.0) for k, v in scores.items()}


# ---------------------------------------------------------------- 名单指标
def oracle_key(truth: Mapping[str, float]) -> str:
    return sorted(truth, key=lambda k: (-float(truth[k]), k))[0]


def list_metrics(keys: Sequence[str], truth: Mapping[str, float], traj: Mapping[str, Mapping[int, float]],
                 stages: Sequence[int], ranked: Sequence[str], *, top_n: int = TOP_N, k_top: int = K_TOP,
                 sh_ks: Sequence[int] = SH_MAIN) -> dict:
    """一条起跑名单的全部读数：oracle 入前 N、regret@1/@k、前 N 内最好、SH（起点 = ranked）regret 与成本。"""
    ranked = list(ranked)
    oracle = max(float(v) for v in truth.values())
    ok = oracle_key(truth)
    top = ranked[:top_n]
    out = {"oracle_in_top": ok in top,
           "regret1": oracle - float(truth[ranked[0]]),
           "regret3": oracle - max(float(truth[k]) for k in ranked[:k_top]),
           "regret_top": oracle - max(float(truth[k]) for k in top),
           "top": top, "sh_regret": None, "sh_cost": None, "sh_pick": None}
    stages = [int(b) for b in stages]
    if len(stages) >= 2:
        ks = tuple(int(k) for k in sh_ks[:len(stages)])
        path = _g36.sh_path(ranked, traj, ks, stages)
        pick = _g36.sh_pick(path[-1], traj, stages[-1])
        out["sh_regret"] = oracle - float(truth[pick])
        out["sh_cost"] = _g36.path_cost_iters(path, stages)
        out["sh_pick"] = pick
    return out


# ---------------------------------------------------------------- 折驱动：s0 四臂
def fold_s0_instances(target: str, cat_z: Sequence[dict], enc: CLSEncoder, *, seed: int = 0) -> list[dict]:
    """一折（靶 = target）：训练侧 = 其余全部行；每 (切片, 预算) 一个实例，含 prior / gbdt / hybrid / s0z 四臂。"""
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior, blind = fit_s0(train, enc, seed=seed)
    out = []
    for s in sorted({r["task_id"] for r in test}):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _secs = _g36.trajectories(rows_s)
        for b in BUDGETS:
            rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
            keys = [r["cls_key"] for r in rows_sb]
            origin = {r["cls_key"]: r["origin"] for r in rows_sb}
            s0z = s0_scores(rows_sb, prior, blind, enc)
            gbdt = dict(zip(keys, blind.predict(_g36.features(rows_sb, enc, max(BUDGETS)))[0].tolist()))
            ranks = {"prior": _g36.rank_by(prior, keys), "gbdt": _g36.rank_by(gbdt, keys),
                     "hybrid": _g36.rank_by(with_bias(s0z, rows_sb), keys), "s0z": _g36.rank_by(s0z, keys)}
            truth = {k: float(traj[k][b]) for k in keys}
            stages = [x for x in BUDGETS if x <= b]
            ok = oracle_key(truth)
            out.append({"fold": target, "dataset": target, "slice": s, "budget": b, "keys": keys,
                        "traj": {k: {int(bb): float(v) for bb, v in traj[k].items()} for k in keys},
                        "truth": truth, "oracle": ok, "oracle_origin": origin[ok], "ranks": ranks,
                        "scores": {"s0z": s0z, "prior": {k: prior[k] for k in keys if k in prior}},
                        "metrics": {arm: list_metrics(keys, truth, traj, stages, rk) for arm, rk in ranks.items()}})
    return out


# ---------------------------------------------------------------- C0：与 WP1 逐实例比对
_C0_MAP = {"prior3": ("prior", "regret3"), "gbdt3": ("gbdt", "regret3"), "hybrid3": ("hybrid", "regret3"),
           "SH6-3-3_prior": ("prior", "sh_regret")}


def c0_compare(instances: Sequence[dict], wp1: Mapping, *, tol: float = C0_TOL) -> dict:
    """prior3 / gbdt3 / hybrid3（全预算）与 SH6-3-3_prior（@290）逐实例 |Δ| ≤ tol；均值也对账。"""
    ref = {(r["slice"], int(r["budget"])): r["regret"] for r in wp1["per_instance"]}
    max_abs: dict[str, float] = {k: 0.0 for k in _C0_MAP}
    mismatches, n = [], 0
    sh_ours = []
    for r in instances:
        key = (r["slice"], int(r["budget"]))
        if key not in ref:
            continue
        n += 1
        for arm, (ours_arm, field) in _C0_MAP.items():
            if arm not in ref[key]:
                continue
            ours = r["metrics"][ours_arm][field]
            if ours is None:
                continue
            d = abs(float(ours) - float(ref[key][arm]))
            max_abs[arm] = max(max_abs[arm], d)
            if d > tol:
                mismatches.append({"slice": r["slice"], "budget": r["budget"], "arm": arm,
                                   "ours": float(ours), "wp1": float(ref[key][arm]), "abs": d})
        if int(r["budget"]) == 290 and r["metrics"]["prior"]["sh_regret"] is not None:
            sh_ours.append(float(r["metrics"]["prior"]["sh_regret"]))
    wp1_mean = float(wp1["by_budget"]["290"]["mean_regret"]["SH6-3-3_prior"])
    mean_ours = float(np.mean(sh_ours)) if sh_ours else None
    mean_ok = mean_ours is None or abs(mean_ours - wp1_mean) <= tol
    return {"pass": bool(not mismatches and mean_ok and n > 0), "n_compared": n, "tol": tol,
            "max_abs": max_abs, "mismatches": mismatches[:20], "n_mismatch": len(mismatches),
            "mean_sh290": {"ours": mean_ours, "wp1": wp1_mean}}


def oracle_miss_table(instances: Sequence[dict], *, arm: str = "hybrid") -> dict:
    """按预算：oracle ∉ 该臂前 N 的实例数、漏网 oracle 的来源与数据集分布、前 N 内最好的平均 regret。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in instances:
        groups[str(int(r["budget"]))].append(r)
        groups["pooled"].append(r)
    out = {}
    for b, rows in groups.items():
        miss = [r for r in rows if not r["metrics"][arm]["oracle_in_top"]]
        by_origin: dict[str, int] = defaultdict(int)
        by_dataset: dict[str, int] = defaultdict(int)
        for r in miss:
            by_origin[r["oracle_origin"]] += 1
            by_dataset[r["dataset"]] += 1
        out[b] = {"n": len(rows), "n_miss": len(miss), "by_origin": dict(sorted(by_origin.items())),
                  "by_dataset": dict(sorted(by_dataset.items())),
                  "regret_top_mean": float(np.mean([r["metrics"][arm]["regret_top"] for r in rows])),
                  "miss_slices": [f"{r.get('slice', '?')}@{r['budget']}:{r['oracle_origin']}" for r in miss]}
    return out


# ---------------------------------------------------------------- C1：数据集身份上界
def identity_features(rows: Sequence[dict], enc: CLSEncoder, datasets: Sequence[str]) -> np.ndarray:
    """36 号 features（编码 ⊕ b/290）⊕ onehot(dataset)，one-hot 在最尾部。"""
    X = _g36.features(rows, enc, max(BUDGETS))
    oh = np.array([[1.0 if r["dataset"] == d else 0.0 for d in datasets] for r in rows], dtype=float)
    return np.hstack([X, oh])


def ceiling_scores(target: str, slice_id: str, cat_z: Sequence[dict], enc: CLSEncoder,
                   datasets: Sequence[str], *, seed: int = 0,
                   fit_identity: Optional[Callable[[Sequence[dict]], object]] = None) -> dict[int, dict[str, float]]:
    """{预算: {cls_key: 分}}：shared = 同数据集**其余**切片的 z 均值（全预算）；random = 身份 hgb 预测。

    训练行 = 除 slice_id 外全部行（含其他数据集 + 同数据集其余切片，后者携带 onehot(target)）。
    """
    rows_s = [r for r in cat_z if r["task_id"] == slice_id]
    if not rows_s:
        raise KeyError(f"缓存里没有切片 {slice_id}")
    others_same = [r for r in cat_z if r["dataset"] == target and r["task_id"] != slice_id
                   and r["origin"] == ORIGIN_SHARED]
    in_ds_prior = _g36.prior_scores(others_same)
    train = [r for r in cat_z if r["task_id"] != slice_id]
    if fit_identity is None:
        model = FieldSurrogate(kind="hgb", seed=seed).fit(
            identity_features(train, enc, datasets), np.array([float(r["z"]) for r in train]))
    else:
        model = fit_identity(train)
    out: dict[int, dict[str, float]] = {}
    for b in sorted({int(r["iters"]) for r in rows_s}):
        rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
        pred, _ = model.predict(identity_features(rows_sb, enc, datasets))
        out[b] = {r["cls_key"]: (float(in_ds_prior[r["cls_key"]]) if r["origin"] == ORIGIN_SHARED
                                 else float(pred[i])) for i, r in enumerate(rows_sb)}
    return out


def fold_c1_metrics(target: str, cat_z: Sequence[dict], enc: CLSEncoder, datasets: Sequence[str],
                    pooled_prior: Mapping[str, float], *, seed: int = 0) -> dict[tuple[str, int], dict]:
    """{(切片, 预算): {c1 臂: list_metrics}}；blend = C1_BLEND·同数据集先验 + (1−C1_BLEND)·LODO 先验（shared）。"""
    test = [r for r in cat_z if r["dataset"] == target]
    out: dict[tuple[str, int], dict] = {}
    for s in sorted({r["task_id"] for r in test}):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        pure = ceiling_scores(target, s, cat_z, enc, datasets, seed=seed)
        for b in BUDGETS:
            rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
            keys = [r["cls_key"] for r in rows_sb]
            shared = {r["cls_key"] for r in rows_sb if r["origin"] == ORIGIN_SHARED}
            sc_pure = pure[b]
            sc_blend = {k: (C1_BLEND * sc_pure[k] + (1 - C1_BLEND) * float(pooled_prior[k]) if k in shared
                            else sc_pure[k]) for k in keys}
            truth = {k: float(traj[k][b]) for k in keys}
            stages = [x for x in BUDGETS if x <= b]
            ranks = {"c1_pure_z": _g36.rank_by(sc_pure, keys),
                     "c1_pure_hyb": _g36.rank_by(with_bias(sc_pure, rows_sb), keys),
                     "c1_blend_z": _g36.rank_by(sc_blend, keys),
                     "c1_blend_hyb": _g36.rank_by(with_bias(sc_blend, rows_sb), keys)}
            out[(s, b)] = {arm: list_metrics(keys, truth, traj, stages, rk) for arm, rk in ranks.items()}
    return out


# ---------------------------------------------------------------- 0b：交互定位
def per_dataset_field_effects(rows: Sequence[dict], enc: CLSEncoder, space: CLSSpace, *,
                              seed: int = 0) -> dict[str, np.ndarray]:
    """ridge_binned（alpha=3，37 号同式）逐字段档内去均值系数向量；bool 字段单值。"""
    X = _g36.features(rows, enc, max(BUDGETS))
    y = np.array([float(r["z"]) for r in rows])
    coef = FieldSurrogate("ridge", seed=seed, alpha=RIDGE_ALPHA).fit(X, y)._models[0].coef_
    out = {}
    for name, sl in enc.field_slices.items():
        if space.field_spec(name)["type"] == "bool":
            out[name] = np.array([float(coef[sl.start])])
        else:
            c = np.asarray(coef[sl], dtype=float)
            out[name] = c - c.mean()
    return out


def field_variance_ratio(cat_z: Sequence[dict], enc: CLSEncoder, space: CLSSpace, *,
                         n_boot: int = 50, seed: int = 0) -> dict:
    """各字段效应向量：跨数据集方差（between）/ 数据集内切片 bootstrap 方差（within）→ 交互字段排名。"""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in cat_z:
        by_ds[r["dataset"]].append(r)
    datasets = sorted(by_ds)
    effects = {d: per_dataset_field_effects(by_ds[d], enc, space, seed=seed) for d in datasets}
    fields = list(enc.field_slices)
    between = {f: float(np.mean(np.var(np.stack([effects[d][f] for d in datasets]), axis=0))) for f in fields}
    rng = np.random.default_rng(seed)
    within_by_ds: dict[str, dict[str, float]] = {}
    for d in datasets:
        slices = sorted({r["task_id"] for r in by_ds[d]})
        rows_by_slice = {s: [r for r in by_ds[d] if r["task_id"] == s] for s in slices}
        reps: dict[str, list[np.ndarray]] = defaultdict(list)
        for _ in range(int(n_boot)):
            pick = rng.choice(len(slices), size=len(slices), replace=True)
            boot_rows = [r for i in pick for r in rows_by_slice[slices[i]]]
            eff = per_dataset_field_effects(boot_rows, enc, space, seed=seed)
            for f in fields:
                reps[f].append(eff[f])
        within_by_ds[d] = {f: float(np.mean(np.var(np.stack(reps[f]), axis=0))) for f in fields}
    within = {f: float(np.mean([within_by_ds[d][f] for d in datasets])) for f in fields}
    eps = 1e-12
    ratio = {f: between[f] / max(within[f], eps) for f in fields}
    rank = sorted(fields, key=lambda f: -ratio[f])
    return {"between": between, "within": within, "ratio": ratio, "rank": rank, "top4": rank[:4],
            "n_boot": int(n_boot), "datasets": datasets,
            "per_dataset_effects": {d: {f: effects[d][f].tolist() for f in fields} for d in datasets}}


def dataset_prior_spearman(cat_z: Sequence[dict]) -> dict[str, float]:
    """逐数据集：shared 键的数据集内 z 均值 vs 其余数据集的先验（LODO）之 Spearman。"""
    shared_rows = [r for r in cat_z if r["origin"] == ORIGIN_SHARED]
    datasets = sorted({r["dataset"] for r in shared_rows})
    out = {}
    for d in datasets:
        inside = _g36.prior_scores([r for r in shared_rows if r["dataset"] == d])
        lodo = _g36.prior_scores([r for r in shared_rows if r["dataset"] != d])
        keys = sorted(set(inside) & set(lodo))
        out[d] = float(spearmanr([inside[k] for k in keys], [lodo[k] for k in keys]).correlation)
    return out


# ---------------------------------------------------------------- 方差分解与噪声
def nested_decomposition(M: np.ndarray, groups: Sequence[str]) -> dict:
    """M (n 切片 × J CLS) 两因素分解，交互再按数据集拆成"数据集级"与"切片级"。

    M_ij = μ + a_i + b_j + R_ij；R_ij = Rbar_{g(i)j}（数据集级）+ (R_ij − Rbar_{g(i)j})（切片级）。
    df：task n−1、cls J−1、数据集级 (G−1)(J−1)、切片级 (n−G)(J−1)。
    """
    M = np.asarray(M, dtype=float)
    groups = list(groups)
    n, J = M.shape
    gnames = sorted(set(groups))
    G = len(gnames)
    grand = M.mean()
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    ss_task = float(((row - grand) ** 2).sum() * J)
    ss_cls = float(((col - grand) ** 2).sum() * n)
    R = M - row - col + grand
    ss_inter = float((R ** 2).sum())
    ss_inter_ds = 0.0
    Rg = {}
    for g in gnames:
        idx = [i for i, x in enumerate(groups) if x == g]
        Rg[g] = R[idx].mean(axis=0)
        ss_inter_ds += float(len(idx) * (Rg[g] ** 2).sum())
    ss_inter_slice = float(sum(((R[i] - Rg[groups[i]]) ** 2).sum() for i in range(n)))
    total = ss_task + ss_cls + ss_inter
    df = {"df_task": n - 1, "df_cls": J - 1, "df_inter_dataset": (G - 1) * (J - 1),
          "df_inter_slice": (n - G) * (J - 1)}
    ms = {"ms_task": ss_task / max(df["df_task"], 1), "ms_cls": ss_cls / max(df["df_cls"], 1),
          "ms_inter_dataset": ss_inter_ds / max(df["df_inter_dataset"], 1),
          "ms_inter_slice": ss_inter_slice / max(df["df_inter_slice"], 1)}
    return {"n_slices": n, "n_cls": J, "n_datasets": G, "grand": float(grand),
            "ss_task": ss_task, "ss_cls": ss_cls, "ss_inter": ss_inter,
            "ss_inter_dataset": ss_inter_ds, "ss_inter_slice": ss_inter_slice,
            "share_task": ss_task / total if total else None,
            "share_cls": ss_cls / total if total else None,
            "share_inter": ss_inter / total if total else None,
            "share_inter_dataset": ss_inter_ds / ss_inter if ss_inter else None,
            "share_inter_slice": ss_inter_slice / ss_inter if ss_inter else None,
            **df, **ms}


def noise_variance(cat_rows: Sequence[dict], probe_rows: Sequence[dict], budget: int) -> dict:
    """同 (切片, CLS, 预算) 的 canonical + 探针种子 → 格内方差（ddof=1）均值 = 单跑噪声 σ²。"""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in cat_rows:
        if int(r["iters"]) == int(budget):
            cells[(r["task_id"], r["cls_key"])].append(float(r["r_prime"]))
    for r in probe_rows:
        if int(r["iters"]) == int(budget):
            cells[(r["task_id"], r["cls_key"])].append(float(r["r_prime"]))
    vs = [float(np.var(v, ddof=1)) for v in cells.values() if len(v) >= 2]
    sigma2 = float(np.mean(vs)) if vs else None
    return {"budget": int(budget), "n_cells": len(vs), "sigma2": sigma2,
            "sd": float(np.sqrt(sigma2)) if sigma2 is not None else None}


def shared_matrix(cat_z: Sequence[dict], budget: int, *, field: str = "r_prime") -> tuple[np.ndarray, list[str], list[str]]:
    """shared-10 × 切片 完备格 (n 切片 × 10) 的 field 矩阵，返回 (M, 切片列表, 数据集列表)。"""
    rows = [r for r in cat_z if r["origin"] == ORIGIN_SHARED and int(r["iters"]) == int(budget)]
    keys = sorted({r["cls_key"] for r in rows})
    slices = sorted({r["task_id"] for r in rows})
    cell = {(r["task_id"], r["cls_key"]): float(r[field]) for r in rows}
    M = np.array([[cell[(s, k)] for k in keys] for s in slices], dtype=float)
    ds = [next(r["dataset"] for r in rows if r["task_id"] == s) for s in slices]
    return M, slices, ds


# ---------------------------------------------------------------- 汇总
def _rate(xs: Sequence[bool]) -> Optional[float]:
    return float(np.mean([bool(x) for x in xs])) if len(xs) else None


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def summarize_arms(instances: Sequence[dict], arms: Sequence[str]) -> dict:
    """pooled / 按预算 / 按数据集@290：oracle 入榜率、mean regret@1/@3/前 N 内最好/SH。"""
    def block(rows: Sequence[dict]) -> dict:
        out = {"n": len(rows)}
        for a in arms:
            ms = [r["metrics"][a] for r in rows if a in r["metrics"]]
            if not ms:
                continue
            out[a] = {"oracle_in_top": _rate([m["oracle_in_top"] for m in ms]),
                      "n_in_top": int(sum(bool(m["oracle_in_top"]) for m in ms)),
                      "regret1": _mean([m["regret1"] for m in ms]), "regret3": _mean([m["regret3"] for m in ms]),
                      "regret_top": _mean([m["regret_top"] for m in ms]),
                      "sh_regret": _mean([m["sh_regret"] for m in ms]),
                      "sh_regret_median": (float(np.median([m["sh_regret"] for m in ms if m["sh_regret"] is not None]))
                                           if any(m["sh_regret"] is not None for m in ms) else None)}
        return out
    by_budget = {str(b): block([r for r in instances if int(r["budget"]) == b]) for b in BUDGETS}
    r290 = [r for r in instances if int(r["budget"]) == 290]
    by_dataset_290 = {d: block([r for r in r290 if r["dataset"] == d]) for d in sorted({r["dataset"] for r in r290})}
    return {"pooled": block(instances), "by_budget": by_budget, "by_dataset_290": by_dataset_290}


def paired_tests(instances: Sequence[dict], pairs: Sequence[tuple[str, str]]) -> dict:
    """配对单侧 Wilcoxon（H1: a 的 regret < b）：regret3 pooled；sh_regret 与 oracle 入榜数差 @290。"""
    out = {}
    r290 = [r for r in instances if int(r["budget"]) == 290]
    for a, b in pairs:
        rows = [r for r in instances if a in r["metrics"] and b in r["metrics"]]
        x = np.array([r["metrics"][a]["regret3"] for r in rows])
        y = np.array([r["metrics"][b]["regret3"] for r in rows])
        rows290 = [r for r in r290 if a in r["metrics"] and b in r["metrics"]]
        xs = np.array([r["metrics"][a]["sh_regret"] for r in rows290], dtype=float)
        ys = np.array([r["metrics"][b]["sh_regret"] for r in rows290], dtype=float)
        out[f"{a}_vs_{b}"] = {
            "n": len(rows), "regret3_mean_a": float(x.mean()), "regret3_mean_b": float(y.mean()),
            "regret3_p": float(_g36.paired_wilcoxon_less(x, y)),
            "regret3_n_better": int(np.sum(x < y)), "regret3_n_worse": int(np.sum(x > y)),
            "n_290": len(rows290),
            "in_top_a_290": int(sum(bool(r["metrics"][a]["oracle_in_top"]) for r in rows290)),
            "in_top_b_290": int(sum(bool(r["metrics"][b]["oracle_in_top"]) for r in rows290)),
            "sh_mean_a_290": float(xs.mean()) if xs.size else None,
            "sh_mean_b_290": float(ys.mean()) if ys.size else None,
            "sh_p_290": float(_g36.paired_wilcoxon_less(xs, ys)) if xs.size else None}
    return out


def c1_reference_line(tests: Mapping[str, dict], *, gain_top: int = 4, gain_regret3: float = 0.005,
                      alpha: float = 0.05) -> dict:
    """计划 §4.2 参考线（阶段 0 已偷看，只作事后核对）：上界比 hybrid oracle 入前 6 率 ≥ +gain_top/42 @290
    或 regret@3 改善 ≥ gain_regret3 且 p<alpha（n=126）。逐 C1 臂报。"""
    out = {}
    for arm in C1_ARMS:
        t = tests.get(f"{arm}_vs_hybrid")
        if not t:
            continue
        d_top = t["in_top_a_290"] - t["in_top_b_290"]
        d_r3 = t["regret3_mean_b"] - t["regret3_mean_a"]
        out[arm] = {"delta_in_top_290": d_top, "delta_regret3": d_r3, "regret3_p": t["regret3_p"],
                    "pass": bool(d_top >= gain_top or (d_r3 >= gain_regret3 and t["regret3_p"] < alpha))}
    out["any_pass"] = bool(any(v["pass"] for k, v in out.items() if k != "any_pass"))
    return out


def stage0_markdown(a: Mapping) -> str:
    L = ["# 阶段 0 简报（C0 复现 / oracle 漏网拆账 / C1 数据集身份上界 / 0b 交互定位）", ""]
    c0 = a["c0"]
    L.append(f"- **C0** {'✓ 逐位复现' if c0['pass'] else '✗ 不一致'}：n={c0['n_compared']}，max|Δ| "
             + ", ".join(f"{k} {v:.1e}" for k, v in c0["max_abs"].items())
             + f"；SH6-3-3_prior @290 均值 ours {c0['mean_sh290']['ours']:.6f} vs WP1 {c0['mean_sh290']['wp1']:.6f}")
    L += ["", "## oracle ∉ hybrid 前 6（漏网拆账）", "", "| 预算 | n | 漏网 | 来源 | 数据集 | 前 6 内最好 mean regret |", "|---|---:|---:|---|---|---:|"]
    for b, t in a["oracle_miss"]["hybrid"].items():
        L.append(f"| {b} | {t['n']} | {t['n_miss']} | {t['by_origin']} | {t['by_dataset']} | {t['regret_top_mean']:.4f} |")
    L += ["", "## 臂表（pooled 126 / @290 42）", "", "| 臂 | 入前6率 pooled | 入前6 @290 | regret@1 | regret@3 | 前6最好 | SH @290 mean / median |", "|---|---:|---:|---:|---:|---:|---:|"]
    s = a["summary"]
    n290 = s["by_budget"]["290"]["n"]
    for arm in list(S0_ARMS) + list(C1_ARMS):
        p, q = s["pooled"].get(arm), s["by_budget"]["290"].get(arm)
        if not p or not q:
            continue
        sh = f"{q['sh_regret']:.4f} / {q['sh_regret_median']:.4f}" if q.get("sh_regret") is not None else "—"
        L.append(f"| {arm} | {p['oracle_in_top']:.3f} | {q['n_in_top']}/{n290} | {p['regret1']:.4f} | {p['regret3']:.4f} | {p['regret_top']:.4f} | {sh} |")
    L += ["", "## 配对检验（H1: a < b）", "", "| 对 | n | regret3 a/b | p | 胜/负 | 入前6 @290 a/b | SH@290 a/b | SH p |", "|---|---:|---|---:|---|---|---|---:|"]
    for k, t in a["tests"].items():
        L.append(f"| {k} | {t['n']} | {t['regret3_mean_a']:.4f}/{t['regret3_mean_b']:.4f} | {t['regret3_p']:.2e} | "
                 f"{t['regret3_n_better']}/{t['regret3_n_worse']} | {t['in_top_a_290']}/{t['in_top_b_290']} | "
                 f"{t['sh_mean_a_290']:.4f}/{t['sh_mean_b_290']:.4f} | {t['sh_p_290']:.2e} |")
    L += ["", "## C1 参考线（事后核对）", "", f"`{json.dumps(a['c1_reference'], ensure_ascii=False)}`"]
    L += ["", "## 0b 交互定位", "", "字段 ratio（跨数据集方差 / 切片 bootstrap 方差）排名：",
          ", ".join(f"{f} {a['field_variance']['ratio'][f]:.2f}" for f in a["field_variance"]["rank"]), "",
          "逐数据集 shared-10 z 均值 vs LODO 先验 Spearman：" + json.dumps({k: round(v, 3) for k, v in a["prior_spearman"].items()}, ensure_ascii=False), ""]
    for b, dcp in a["decomposition"].items():
        L.append(f"- @{b}：任务 {dcp['share_task']:.3f} / CLS {dcp['share_cls']:.3f} / 交互 {dcp['share_inter']:.3f}"
                 f"（交互中数据集级 {dcp['share_inter_dataset']:.3f} / 切片级 {dcp['share_inter_slice']:.3f}）；"
                 f"MS 数据集级交互 / σ² = {dcp.get('ms_inter_dataset_over_sigma2', float('nan')):.1f}，"
                 f"切片级 / σ² = {dcp.get('ms_inter_slice_over_sigma2', float('nan')):.1f}，"
                 f"σ = {dcp.get('noise_sd', float('nan')):.4f}（{dcp.get('noise_cells', 0)} 格）")
    return "\n".join(L) + "\n"


def cmd_stage0(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0, n_boot: int = 50,
               wp1_path: Path = WP1_ANALYSIS) -> dict:
    cat, _ggs, probe, _rankings = _g36.load_inputs(mx3)
    cat_z = _g36.zscore_within(cat)
    space = CLSSpace.from_yaml()
    enc_s = CLSEncoder(space)                       # scalar：s0 / C1（与 36 号同）
    enc_b = CLSEncoder(space, mode="binned")        # binned：0b 字段效应（与 37 号同）
    instances: list[dict] = []
    for target in folds:
        inst = fold_s0_instances(target, cat_z, enc_s, seed=seed)
        pooled_prior = _g36.prior_scores([r for r in cat_z if r["dataset"] != target and r["origin"] == ORIGIN_SHARED])
        c1 = fold_c1_metrics(target, cat_z, enc_s, DATASETS, pooled_prior, seed=seed)
        for r in inst:
            r["metrics"].update(c1[(r["slice"], int(r["budget"]))])
        instances += inst
        miss = sum(not r["metrics"]["hybrid"]["oracle_in_top"] for r in inst if int(r["budget"]) == 290)
        print(f"[fold {target}] {len(inst)} 实例；@290 oracle ∉ hybrid 前 6：{miss}/6", flush=True)
    wp1 = json.loads(Path(wp1_path).read_text(encoding="utf-8"))
    c0 = c0_compare(instances, wp1)
    print(f"C0 {'PASS' if c0['pass'] else 'FAIL'}：n={c0['n_compared']} max_abs={c0['max_abs']}", flush=True)
    summary = summarize_arms(instances, list(S0_ARMS) + list(C1_ARMS))
    tests = paired_tests(instances, [(a, "hybrid") for a in ("s0z", *C1_ARMS)] + [("c1_pure_z", "s0z")])
    fv = field_variance_ratio(cat_z, enc_b, space, n_boot=n_boot, seed=seed)
    rho = dataset_prior_spearman(cat_z)
    decomp = {}
    for b in BUDGETS:
        M, _slices, ds = shared_matrix(cat_z, b)
        d = nested_decomposition(M, ds)
        nz = noise_variance(cat, probe, b)
        if nz["sigma2"]:
            d["ms_inter_dataset_over_sigma2"] = d["ms_inter_dataset"] / nz["sigma2"]
            d["ms_inter_slice_over_sigma2"] = d["ms_inter_slice"] / nz["sigma2"]
            d["ms_task_over_sigma2"] = d["ms_task"] / nz["sigma2"]
            d["ms_cls_over_sigma2"] = d["ms_cls"] / nz["sigma2"]
        d["noise_sd"] = nz["sd"]
        d["noise_cells"] = nz["n_cells"]
        decomp[str(b)] = d
    analysis = {
        "plan": "~/.claude/plans/ts2vec-e-quirky-bubble.md §4.2（阶段 0：先跑先看，A/B 协议其后锁定）",
        "source": str(mx3.relative_to(ROOT)) if mx3.is_relative_to(ROOT) else str(mx3),
        "folds": list(folds), "n_instances": len(instances), "seed": seed,
        "c0": c0, "oracle_miss": {arm: oracle_miss_table(instances, arm=arm) for arm in ("hybrid", "s0z", "gbdt")},
        "summary": summary, "tests": tests, "c1_reference": c1_reference_line(tests),
        "field_variance": {k: v for k, v in fv.items() if k != "per_dataset_effects"},
        "field_effects_per_dataset": fv["per_dataset_effects"],
        "prior_spearman": rho, "decomposition": decomp,
        "per_instance": [{k: v for k, v in r.items() if k not in ("traj", "truth", "scores")} for r in instances],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "stage0.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "stage0.md").write_text(stage0_markdown(analysis), encoding="utf-8")
    print(stage0_markdown(analysis))
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--mx3", default=str(MX3))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=50)
    args = ap.parse_args(argv)
    if args.stage0:
        cmd_stage0(Path(args.mx3), Path(args.outdir), args.folds, seed=args.seed, n_boot=args.n_boot)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
