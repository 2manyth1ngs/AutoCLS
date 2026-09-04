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

阶段 1/2（协议 results/mx4_task_feature/PROTOCOL.md，锁 6b3c610）：
  --features  45 切片 φ（φ_meta 4 ⊕ φ_base 2 ⊕ φ_sig 6，只用训练划分）→ task_features.jsonl
  --stage1a   家族 A：残差修正 s = ŝ0_z + g(CLS_binned ⊕ b ⊕ φ)，7 折 LODO 126 实例，臂表 §3
  --stage1b   家族 B：前 6 条 @60 实测分 → φ_B → 追加 2 条（SH6+2-3-3 ≡ SH8-3-3），42 实例
  --stage2    官方口径重放（results/mx4_deploy，6 靶，描述性）
  --analyze   门 A1/A2/B1/B2/B3 + 读数码 + 预测对账 → analysis.json / tables.md

用法：
  python scripts/40_task_feature_scorer.py --stage0                 # 7 折 → results/mx4_task_feature/stage0.{json,md}
  python scripts/40_task_feature_scorer.py --stage0 --folds Handwriting --n-boot 5
  python scripts/40_task_feature_scorer.py --features --stage1a --stage1b --stage2 --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import kurtosis, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g37 = importlib.import_module("scripts.37_surrogate_loo")

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

# ---- 阶段 1/2（PROTOCOL §2.3–§2.6、§3）
PHI_COLS = ("log10_T", "log10_C", "k", "log10_N", "base_mean", "base_band",
            "spec_conc", "hf_ratio", "acf_tau", "ex_kurt", "chan_cv", "samp_cv")
PHI_META = PHI_COLS[:4]
PHI_BASE = PHI_COLS[4:6]
PHI_SIG = PHI_COLS[6:]
PHI_SLICE_ONLY = ("k", "log10_N") + PHI_BASE + PHI_SIG      # 排除数据集级常量 T / C
INT_FIELDS = ("cross_scale", "pool", "norm", "emb_mask")   # 预注册：0b 交互字段前 4（PROTOCOL §0）
N_SHUFFLE = 5
A_MAIN = "hgb_phi_all"
B_MAIN = "B_resid"
B_CONTROL = "+2_prior78"                                   # ≡ SH8-3-3_prior（同成本 1170）
PHI_FILE = OUT / "task_features.jsonl"
PA_RUNS = ROOT / "results" / "protocol_alignment" / "runs.jsonl"
DEPLOY = ROOT / "results" / "mx4_deploy"
OFFICIAL_TARGETS = ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra",
                    "EthanolConcentration", "Cricket", "HandMovementDirection")
ALPHA = 0.05
A1_TOP_GAIN = 3
B1_TOP_GAIN = 3
B3_TOL = 0.005


# ---------------------------------------------------------------- 现行打分器 s0
def fit_s0(train_rows: Sequence[dict], enc: CLSEncoder, *, seed: int = 0) -> tuple[dict, FieldSurrogate]:
    """(先验 z 均值 dict, 任务盲 hgb) —— 与 36 号 fold_instances 的 prior / model 逐位同构。"""
    prior = _g36.prior_scores([r for r in train_rows if r["origin"] == ORIGIN_SHARED])
    blind = FieldSurrogate(kind="hgb", seed=seed).fit(
        _g36.features(train_rows, enc, max(BUDGETS)), np.array([float(r["z"]) for r in train_rows]))
    return prior, blind


def s0_rowwise(rows: Sequence[dict], prior: Mapping[str, float], blind, enc: CLSEncoder) -> np.ndarray:
    """逐行 ŝ0_z：shared 键取先验，其余取任务盲预测（random 的预测随 b 变，故按行）。"""
    pred, _ = blind.predict(_g36.features(rows, enc, max(BUDGETS)))
    return np.array([float(prior[r["cls_key"]]) if r["cls_key"] in prior else float(pred[i])
                     for i, r in enumerate(rows)], dtype=float)


def s0_scores(rows_sb: Sequence[dict], prior: Mapping[str, float], blind, enc: CLSEncoder) -> dict[str, float]:
    """一个 (切片, 预算) 组的 {cls_key: ŝ0_z}（组内 cls_key 唯一）。"""
    v = s0_rowwise(rows_sb, prior, blind, enc)
    return {r["cls_key"]: float(v[i]) for i, r in enumerate(rows_sb)}


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


# ================================================================ 阶段 1/2：任务特征 φ（PROTOCOL §2.3）
def signal_stats(X: np.ndarray) -> dict[str, float]:
    """(n, T, C) → 六个信号统计量：谱集中度 / 高频能量比 / 自相关衰减 τ/T / 超额峭度 / 通道方差 CV / 样本方差 CV。

    逐 (样本, 通道) 去均值后计算，前四项对零方差行跳过后取均值；chan_cv 逐样本跨通道 CV 再平均（C=1 → 0）；
    samp_cv = 样本总方差跨样本 CV。序列等长、无填充（PQ 8 数据集与官方 6 靶均如此）。
    """
    X = np.asarray(X, dtype=float)
    n, T, C = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    R = np.transpose(Xc, (0, 2, 1)).reshape(n * C, T)
    ok = R.var(axis=1) > 1e-12
    P = np.abs(np.fft.rfft(R, axis=1)) ** 2
    P = P[:, 1:]                                                   # 去 DC
    tot = P.sum(axis=1)
    nb = P.shape[1]
    ktop = max(1, int(np.ceil(0.1 * nb)))
    top = np.sort(P, axis=1)[:, -ktop:].sum(axis=1)
    safe = np.maximum(tot, 1e-300)
    spec = np.where(tot > 0, top / safe, 0.0)
    hf = np.where(tot > 0, P[:, nb // 2:].sum(axis=1) / safe, 0.0)
    F = np.fft.rfft(R, n=2 * T, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n=2 * T, axis=1)[:, :T]
    ac0 = np.maximum(ac[:, :1], 1e-300)
    below = (ac / ac0) < np.exp(-1.0)
    first = np.where(below.any(axis=1), below.argmax(axis=1), T)
    tau = first / float(T)
    kt = kurtosis(R, axis=1, fisher=True, bias=True)
    kt = np.where(np.isfinite(kt), kt, 0.0)
    v_nc = Xc.var(axis=1)                                          # (n, C)
    if C > 1:
        m = v_nc.mean(axis=1)
        chan_cv = np.where(m > 0, v_nc.std(axis=1) / np.maximum(m, 1e-300), 0.0)
    else:
        chan_cv = np.zeros(n)
    v_n = (Xc ** 2).mean(axis=(1, 2))
    samp_cv = float(v_n.std() / v_n.mean()) if v_n.mean() > 0 else 0.0

    def _m(a: np.ndarray) -> float:
        return float(np.mean(a[ok])) if ok.any() else 0.0

    return {"spec_conc": _m(spec), "hf_ratio": _m(hf), "acf_tau": _m(tau), "ex_kurt": _m(kt),
            "chan_cv": float(np.mean(chan_cv)), "samp_cv": samp_cv}


def phi_meta(T: int, C: int, k: int, N: int) -> dict[str, float]:
    return {"log10_T": float(np.log10(T)), "log10_C": float(np.log10(C)), "k": float(k), "log10_N": float(np.log10(N))}


def phi_base(mean: float, band: float) -> dict[str, float]:
    return {"base_mean": float(mean), "base_band": float(band)}


def phi_vec(phi: Mapping[str, float], cols: Sequence[str]) -> np.ndarray:
    return np.array([float(phi[c]) for c in cols], dtype=float)


def phi_stats(phi_by_slice: Mapping[str, Mapping], slices: Sequence[str], cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """训练侧切片的 φ 均值 / 标准差（sd=0 → 1，防除零；铁律 6：只喂训练侧切片）。"""
    M = np.stack([phi_vec(phi_by_slice[s], cols) for s in slices])
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def build_X(rows: Sequence[dict], enc_b: CLSEncoder, phi_by_slice: Mapping[str, Mapping], cols: Sequence[str],
            *, mu: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None) -> np.ndarray:
    """36 号 features（编码 ⊕ b/290）⊕ φ[cols]（按 task_id 广播，拼在最尾部）；cols 空 → 只有编码。"""
    Xb = _g36.features(rows, enc_b, max(BUDGETS))
    cols = tuple(cols)
    if not cols:
        return Xb
    Phi = np.stack([phi_vec(phi_by_slice[r["task_id"]], cols) for r in rows])
    if mu is not None:
        Phi = (Phi - np.asarray(mu, dtype=float)) / np.asarray(sd, dtype=float)
    return np.hstack([Xb, Phi])


def derange_by_dataset(phi_by_slice: Mapping[str, Mapping], seed: int) -> dict[str, dict]:
    """φ 置换对照：数据集级 derangement（无自供体），切片按下标对供体切片数取模；返回副本（含供体 dataset 字段）。"""
    by_ds: dict[str, list[str]] = defaultdict(list)
    for s in sorted(phi_by_slice):
        by_ds[phi_by_slice[s]["dataset"]].append(s)
    ds = sorted(by_ds)
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(ds))
        if all(int(perm[i]) != i for i in range(len(ds))):
            break
    out: dict[str, dict] = {}
    for i, d in enumerate(ds):
        donors = by_ds[ds[int(perm[i])]]
        for j, s in enumerate(by_ds[d]):
            out[s] = dict(phi_by_slice[donors[j % len(donors)]])
    return out


def interaction_columns(Xb: np.ndarray, Phi: np.ndarray, field_slices: Mapping[str, slice],
                        fields: Sequence[str]) -> np.ndarray:
    """字段 one-hot 档 × φ 的乘积列（ridge 交互臂）：列序 = 字段顺序 × 档顺序 × φ 顺序。"""
    blocks = []
    for f in fields:
        sl = field_slices[f]
        for j in range(sl.start, sl.stop):
            blocks.append(Xb[:, j:j + 1] * Phi)
    return np.hstack(blocks)


def oof_s0(train_rows: Sequence[dict], enc: CLSEncoder, *, seed: int = 0,
           fit_fn: Optional[Callable[[Sequence[dict]], tuple]] = None) -> np.ndarray:
    """交叉拟合的 ŝ0_z：训练侧内层 LODO（按数据集），每行的 ŝ0 由不含其数据集的先验 / 任务盲 hgb 给出。"""
    out = np.full(len(train_rows), np.nan)
    for d in sorted({r["dataset"] for r in train_rows}):
        idx = [i for i, r in enumerate(train_rows) if r["dataset"] == d]
        inner = [r for r in train_rows if r["dataset"] != d]
        prior, blind = fit_fn(inner) if fit_fn is not None else fit_s0(inner, enc, seed=seed)
        out[idx] = s0_rowwise([train_rows[i] for i in idx], prior, blind, enc)
    return out


def fit_resid(X: np.ndarray, r: np.ndarray, *, kind: str = "hgb", seed: int = 0, **params) -> FieldSurrogate:
    return FieldSurrogate(kind, seed=seed, **params).fit(X, r)


def load_phi(path: Path = PHI_FILE) -> dict[str, dict]:
    from scripts.e_ablation_common import read_jsonl
    return {r["task_id"]: r for r in read_jsonl(path)}


def cmd_features(mx3: Path, outdir: Path, *, resume: bool = True) -> dict[str, dict]:
    """45 切片 φ → task_features.jsonl（铁律 3：落盘、--resume）。只用 canonical 训练划分（与 baseline / proxy 同）。"""
    from autocls.data.task_slicer import materialize_task
    from autocls.labeling import run_seed_for, split_task_train_eval
    path = outdir / "task_features.jsonl"
    tasks = pq.load_tasks_json(mx3 / "tasks.json")
    done = load_phi(path) if (resume and path.exists()) else {}
    if len(done) == len(tasks):
        print(f"[features] 已有 {len(done)} 切片，跳过", flush=True)
        return done
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = pq.load_pq_datasets(pq.PQ_DATASETS)
    with open(path, "a", encoding="utf-8") as fh:
        for name, ds in datasets.items():
            bands = pq.baseline_bands(name)
            for t in ds.slices:
                if t.task_id in done:
                    continue
                spec = tasks[t.task_id]
                Xt, yt = materialize_task(t, ds.X, ds.y)
                tr, _ev = split_task_train_eval(yt, t.task_type, eval_ratio=0.3,
                                                seed=run_seed_for(t.task_id, "train_eval_split"))
                assert len(tr) == int(spec["n_train"]), f"{t.task_id}: 训练划分 {len(tr)} ≠ tasks.json n_train {spec['n_train']}"
                mean, band = bands[t.task_id]
                row = {"task_id": t.task_id, "dataset": name, "n_train_split": int(len(tr)), "T": int(Xt.shape[1]),
                       **phi_meta(Xt.shape[1], Xt.shape[2], len(spec["classes"]), len(tr)),
                       **phi_base(mean, band), **signal_stats(Xt[tr])}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[t.task_id] = row
                print(f"[features] {t.task_id}: " + ", ".join(f"{c}={row[c]:.3f}" for c in PHI_COLS), flush=True)
    return done


# ================================================================ 阶段 1-A：预测型（PROTOCOL §2.4 / §3）
def _resid_scores(rows_sb: Sequence[dict], s0z: Mapping[str, float], model, X: np.ndarray) -> dict[str, float]:
    rhat, _ = model.predict(X)
    return {r["cls_key"]: float(s0z[r["cls_key"]]) + float(rhat[i]) for i, r in enumerate(rows_sb)}


def fold_stage1a(target: str, cat_z: Sequence[dict], phi_by_slice: Mapping[str, Mapping], enc_s: CLSEncoder,
                 enc_b: CLSEncoder, delta: Mapping, *, seed: int = 0, n_shuffle: int = N_SHUFFLE) -> tuple[list[dict], dict]:
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior, blind = fit_s0(train, enc_s, seed=seed)
    r_tgt = np.array([float(r["z"]) for r in train]) - oof_s0(train, enc_s, seed=seed)
    train_slices = sorted({r["task_id"] for r in train})
    mu, sd = phi_stats(phi_by_slice, train_slices, PHI_COLS)
    fs = enc_b.field_slices
    Xb_tr = build_X(train, enc_b, {}, ())
    X_all = build_X(train, enc_b, phi_by_slice, PHI_COLS)
    Phi_std = (np.stack([phi_vec(phi_by_slice[r["task_id"]], PHI_COLS) for r in train]) - mu) / sd
    X_int = np.hstack([Xb_tr, Phi_std, interaction_columns(Xb_tr, Phi_std, fs, INT_FIELDS)])
    models = {
        "resid_nophi": fit_resid(Xb_tr, r_tgt, seed=seed),
        "hgb_phi_all": fit_resid(X_all, r_tgt, seed=seed),
        "hgb_phi_meta": fit_resid(build_X(train, enc_b, phi_by_slice, PHI_META + PHI_BASE), r_tgt, seed=seed),
        "hgb_phi_slice": fit_resid(build_X(train, enc_b, phi_by_slice, PHI_SLICE_ONLY), r_tgt, seed=seed),
        "ridge_int_top4": fit_resid(X_int, r_tgt, kind="ridge", seed=seed, alpha=RIDGE_ALPHA),
    }
    shuffles = [derange_by_dataset(phi_by_slice, seed=1000 * seed + 17 * i + 1) for i in range(n_shuffle)]
    base = enc_b.dim + 1
    groups = {"cls": slice(0, enc_b.dim), "budget": slice(enc_b.dim, base)}
    for name, cols in (("phi_meta", PHI_META), ("phi_base", PHI_BASE), ("phi_sig", PHI_SIG)):
        i0 = base + PHI_COLS.index(cols[0])
        groups[name] = slice(i0, i0 + len(cols))
    importance = _g37.grouped_permutation_importance(models["hgb_phi_all"], X_all, r_tgt, groups, seed=seed)
    out = []
    for s in sorted({r["task_id"] for r in test}):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        for b in BUDGETS:
            rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
            keys = [r["cls_key"] for r in rows_sb]
            truth = {k: float(traj[k][b]) for k in keys}
            stages = [x for x in BUDGETS if x <= b]
            s0z = s0_scores(rows_sb, prior, blind, enc_s)
            Xb_sb = build_X(rows_sb, enc_b, {}, ())
            Phi_sb = (np.stack([phi_vec(phi_by_slice[s], PHI_COLS)] * len(rows_sb)) - mu) / sd
            sc = {"hybrid": with_bias(s0z, rows_sb), "s0z": dict(s0z),
                  "resid_nophi": _resid_scores(rows_sb, s0z, models["resid_nophi"], Xb_sb),
                  "hgb_phi_all": _resid_scores(rows_sb, s0z, models["hgb_phi_all"],
                                               build_X(rows_sb, enc_b, phi_by_slice, PHI_COLS)),
                  "hgb_phi_meta": _resid_scores(rows_sb, s0z, models["hgb_phi_meta"],
                                                build_X(rows_sb, enc_b, phi_by_slice, PHI_META + PHI_BASE)),
                  "hgb_phi_slice": _resid_scores(rows_sb, s0z, models["hgb_phi_slice"],
                                                 build_X(rows_sb, enc_b, phi_by_slice, PHI_SLICE_ONLY)),
                  "ridge_int_top4": _resid_scores(rows_sb, s0z, models["ridge_int_top4"],
                                                  np.hstack([Xb_sb, Phi_sb, interaction_columns(Xb_sb, Phi_sb, fs, INT_FIELDS)]))}
            for i, shp in enumerate(shuffles):
                sc[f"hgb_phi_shuffle_{i}"] = _resid_scores(rows_sb, s0z, models["hgb_phi_all"],
                                                           build_X(rows_sb, enc_b, shp, PHI_COLS))
            sc["hgb_phi_all_hyb"] = with_bias(sc["hgb_phi_all"], rows_sb)
            dl = float(delta[(target, b)])
            metrics = {}
            for arm, v in sc.items():
                m = list_metrics(keys, truth, traj, stages, _g36.rank_by(v, keys))
                m["pair_acc"] = _g37.pair_accuracy(v, truth, dl)[0]
                metrics[arm] = m
            ok = oracle_key(truth)
            origin = {r["cls_key"]: r["origin"] for r in rows_sb}
            out.append({"fold": target, "dataset": target, "slice": s, "budget": b, "oracle": ok,
                        "oracle_origin": origin[ok], "metrics": metrics})
    meta = {"importance": importance, "phi_mu": mu.tolist(), "phi_sd": sd.tolist(),
            "prior_top6": _g36.rank_by(prior, list(prior))[:TOP_N],
            "shuffle_donors": [{s: shp[s]["dataset"] for s in sorted({r["task_id"] for r in test})} for shp in shuffles]}
    return out, meta


# ================================================================ 阶段 1-B：转导型（PROTOCOL §2.5 / §3）
def phi_b_of(values_by_key: Mapping[str, float], top6: Sequence[str]) -> np.ndarray:
    """前 6 条实测分 → 6 条内 z（sd=0 → 零向量）。"""
    v = np.array([float(values_by_key[k]) for k in top6], dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def source_phi_b(rows: Sequence[dict], top6: Sequence[str], budget: int = 60) -> dict[str, np.ndarray]:
    """源切片在同 6 列 @budget 的 φ_B（缺列的切片不返回）。"""
    want = set(top6)
    by_slice: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if int(r["iters"]) == int(budget) and r["cls_key"] in want:
            by_slice[r["task_id"]][r["cls_key"]] = float(r["r_prime"])
    return {s: phi_b_of(v, top6) for s, v in by_slice.items() if len(v) == len(top6)}


def sh8_metrics(top6: Sequence[str], added: Sequence[str], traj: Mapping[str, Mapping[int, float]],
                truth: Mapping[str, float], *, budgets: Sequence[int] = BUDGETS) -> dict:
    """SH6+2-3-3：起跑 8 条（前 6 + 追加 2），之后按截点实测留 3 → 留 3 → 取最好（≡ sh_path(…, (8,3,3))）。"""
    ranked8 = list(top6) + list(added)
    stages = [int(b) for b in budgets]
    path = _g36.sh_path(ranked8, traj, SH_PLUS, stages)
    pick = _g36.sh_pick(path[-1], traj, stages[-1])
    oracle = max(float(v) for v in truth.values())
    return {"oracle_in_8": oracle_key(truth) in ranked8, "regret": oracle - float(truth[pick]),
            "cost": _g36.path_cost_iters(path, stages), "pick": pick, "added": list(added)}


def expected_plus_random(top6: Sequence[str], unstarted: Sequence[str], traj: Mapping, truth: Mapping,
                         *, budgets: Sequence[int] = BUDGETS) -> dict:
    """+2_random：全部 C(n,2) 对的精确期望（regret 与 oracle 入 8 率）。"""
    ms = [sh8_metrics(top6, list(p), traj, truth, budgets=budgets) for p in combinations(list(unstarted), 2)]
    return {"regret": float(np.mean([m["regret"] for m in ms])),
            "oracle_in_8": float(np.mean([m["oracle_in_8"] for m in ms])), "n_pairs": len(ms)}


def cf_knn_scores(src_phi: Mapping[str, np.ndarray], src_tgt: Mapping[str, Mapping[str, float]],
                  q: np.ndarray, keys: Sequence[str], *, k: int = 3) -> dict[str, float]:
    """协同过滤：按 φ_B 的 L2 距离取 k 个最近源切片，目标列 = 它们的 z@290 均值（并列按切片名）。"""
    names = sorted(n for n in src_phi if all(key in src_tgt.get(n, {}) for key in keys))
    q = np.asarray(q, dtype=float)
    order = sorted(names, key=lambda n: (float(np.sum((np.asarray(src_phi[n]) - q) ** 2)), n))[:k]
    return {key: float(np.mean([src_tgt[n][key] for n in order])) for key in keys}


def fold_stage1b(target: str, cat_z: Sequence[dict], enc_s: CLSEncoder, enc_b: CLSEncoder, probe: Sequence[dict],
                 *, seed: int = 0, n_shuffle: int = N_SHUFFLE) -> tuple[list[dict], dict]:
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior, blind = fit_s0(train, enc_s, seed=seed)
    r_tgt = np.array([float(r["z"]) for r in train]) - oof_s0(train, enc_s, seed=seed)
    top6 = _g36.rank_by(prior, list(prior))[:TOP_N]               # 折内常量（hybrid 前 6 = 先验前 6）
    m_nophi = fit_resid(build_X(train, enc_b, {}, ()), r_tgt, seed=seed)
    src_phi = source_phi_b(train, top6)
    keep = [i for i, r in enumerate(train) if r["cls_key"] not in set(top6)]   # 自泄漏：top-6 的 z@60 就在 φ_B 里
    train_b = [train[i] for i in keep]
    X_b = np.hstack([build_X(train_b, enc_b, {}, ()), np.stack([src_phi[r["task_id"]] for r in train_b])])
    m_b = fit_resid(X_b, r_tgt[keep], seed=seed)
    src_tgt: dict[str, dict[str, float]] = defaultdict(dict)
    for r in train:
        if int(r["iters"]) == BUDGETS[-1] and r["origin"] == ORIGIN_SHARED:
            src_tgt[r["task_id"]][r["cls_key"]] = float(r["z"])
    ds_of = {r["task_id"]: r["dataset"] for r in cat_z}
    b = BUDGETS[-1]
    by_cell: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in probe:
        if r["dataset"] == target and int(r["iters"]) == b:
            by_cell[(r["task_id"], r["cls_key"])][int(r["seed"])] = float(r["r_prime"])
    slices = sorted({r["task_id"] for r in test})
    phi_dicts = {s: {"dataset": ds_of[s], "v": v.tolist()} for s, v in src_phi.items()}
    out = []
    target_phi = {}
    for s in slices:
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        target_phi[s] = phi_b_of({k: traj[k][BUDGETS[0]] for k in top6}, top6)
        phi_dicts[s] = {"dataset": target, "v": target_phi[s].tolist()}
    shuffles = [derange_by_dataset(phi_dicts, seed=1000 * seed + 17 * i + 1) for i in range(n_shuffle)]
    for s in slices:
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
        keys = [r["cls_key"] for r in rows_sb]
        truth = {k: float(traj[k][b]) for k in keys}
        s0z = s0_scores(rows_sb, prior, blind, enc_s)
        hybrid_rank = _g36.rank_by(with_bias(s0z, rows_sb), keys)
        assert hybrid_rank[:TOP_N] == top6, f"{s}: hybrid 前 6 {hybrid_rank[:TOP_N]} ≠ 先验前 6 {top6}"
        unstarted = [k for k in keys if k not in top6]
        rows_u = [r for r in rows_sb if r["cls_key"] in set(unstarted)]
        keys_u = [r["cls_key"] for r in rows_u]
        q = target_phi[s]
        Xu = build_X(rows_u, enc_b, {}, ())

        def score_b(qv: np.ndarray) -> dict[str, float]:
            rhat, _ = m_b.predict(np.hstack([Xu, np.repeat(qv[None, :], len(rows_u), axis=0)]))
            return {k: s0z[k] + float(rhat[i]) for i, k in enumerate(keys_u)}

        added: dict[str, list[str]] = {
            "B_resid": _g36.rank_by(score_b(q), keys_u)[:2],
            "B_nophi": _g36.rank_by(_resid_scores(rows_u, s0z, m_nophi, Xu), keys_u)[:2],
            B_CONTROL: hybrid_rank[TOP_N:TOP_N + 2],
            "+2_gbdt": _g36.rank_by({k: s0z[k] for k in keys_u}, keys_u)[:2],
        }
        keys4 = [k for k in keys_u if k in prior]
        cf = cf_knn_scores(src_phi, src_tgt, q, keys4, k=3)
        added["B_cf"] = _g36.rank_by(cf, keys4)[:2]
        for i, shp in enumerate(shuffles):
            added[f"B_shuffle_{i}"] = _g36.rank_by(score_b(np.asarray(shp[s]["v"], dtype=float)), keys_u)[:2]
        metrics = {arm: sh8_metrics(top6, a, traj, truth) for arm, a in added.items()}
        rnd = expected_plus_random(top6, keys_u, traj, truth)
        metrics["+2_random"] = {"oracle_in_8": rnd["oracle_in_8"], "regret": rnd["regret"], "cost": 1170,
                                "pick": None, "added": None, "n_pairs": rnd["n_pairs"]}
        for name, ks in (("SH6-3-3_prior", SH_MAIN), ("SH10-4-3_prior", (10, 4, 3))):
            path = _g36.sh_path(hybrid_rank, traj, ks, BUDGETS)
            pick = _g36.sh_pick(path[-1], traj, b)
            metrics[name] = {"oracle_in_8": oracle_key(truth) in path[0], "regret": max(truth.values()) - truth[pick],
                             "cost": _g36.path_cost_iters(path, BUDGETS), "pick": pick, "added": None}
        fresh = {k: float(np.mean([by_cell[(s, k)][rep] for rep in _g36.FRESH_REPS])) for k in keys
                 if all(rep in by_cell.get((s, k), {}) for rep in _g36.FRESH_REPS)}
        fresh_regret = None
        if len(fresh) >= 2:
            o_f = max(fresh.values())
            fresh_regret = {arm: (o_f - fresh[m["pick"]]) if m.get("pick") in fresh else None
                            for arm, m in metrics.items()}
        ok = oracle_key(truth)
        origin = {r["cls_key"]: r["origin"] for r in rows_sb}
        out.append({"fold": target, "dataset": target, "slice": s, "budget": b, "oracle": ok,
                    "oracle_origin": origin[ok], "top6": list(top6), "metrics": metrics,
                    "fresh_regret": fresh_regret, "phi_b": q.tolist()})
    return out, {"top6": list(top6), "n_train_b": len(train_b), "n_src_slices": len(src_phi),
                 "shuffle_donors": [{s: shp[s]["dataset"] for s in slices} for shp in shuffles]}


# ================================================================ 阶段 2：官方口径重放（PROTOCOL §2.6，描述性）
def trajectory_from_rows(rows: Sequence[dict]):
    """runs 行 → {cls_key: {iters: acc}}（与 39 号 trajectory_from_rows 同式，本地重写以免 import torch）。"""
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r.get("error") or not r.get("checkpoint_results"):
            continue
        for c in r["checkpoint_results"]:
            traj[r["cls_key"]][int(c["iters"])] = float(c["acc"])
    return dict(traj)


def official_phi(dataset: str, pa_rows: Sequence[dict]) -> dict:
    """官方 TRAIN 划分上的 φ：meta 取自数组，base 取 protocol_alignment 的 audit_iters0（3 种子），sig 同 signal_stats。"""
    from autocls.data.uea import load_uea_pair
    Xtr, ytr, _Xte, _yte = load_uea_pair(dataset, scaler="train")
    rows = [r for r in pa_rows if r.get("dataset") == dataset and r.get("arm") == "audit_iters0" and r.get("error") is None]
    if not rows:
        raise KeyError(f"{dataset}: protocol_alignment 无 audit_iters0 行")
    acc = np.array([float(r["acc"]) for r in rows])
    n_test = int(rows[0]["n_test"])
    band = 2.0 * max(float(acc.std(ddof=1)) if len(acc) > 1 else 0.0, 1.0 / n_test)
    return {"task_id": dataset, "dataset": dataset, "T": int(Xtr.shape[1]), "n_train_split": int(Xtr.shape[0]),
            **phi_meta(Xtr.shape[1], Xtr.shape[2], int(len(np.unique(ytr))), Xtr.shape[0]),
            **phi_base(float(acc.mean()), band), **signal_stats(Xtr)}


def replay_official(d: str, cand: Mapping, s1_rows: Sequence[dict], phi_d: Mapping, phi_by_slice: Mapping,
                    cat_z: Sequence[dict], ggs_rows: Sequence[dict], enc_s: CLSEncoder, enc_b: CLSEncoder,
                    *, seed: int = 0) -> dict:
    traj = trajectory_from_rows(s1_rows)
    budgets = sorted({b for v in traj.values() for b in v})
    bf = budgets[-1]
    keys = sorted(k for k in traj if all(b in traj[k] for b in budgets))
    cls_of = {e["cls_key"]: e["cls"] for e in list(cand["shared"]) + [cand["ggs"]] + list(cand.get("nominated", []))}
    prior = {k: float(v) for k, v in cand["prior"].items()}
    nom = {e["cls_key"]: float(e["pred_z"]) for e in cand.get("nominated", [])}
    s0 = {k: (prior[k] if k in prior else nom[k]) for k in keys}
    hybrid = {k: s0[k] + (HYBRID_BIAS if k in prior else 0.0) for k in keys}
    truth = {k: float(traj[k][bf]) for k in keys}
    pseudo = [{"task_id": d, "dataset": d, "cls_key": k, "cls": cls_of[k], "iters": max(BUDGETS),
               "origin": ORIGIN_SHARED if k in prior else "random"} for k in keys]
    train = [r for r in cat_z if r["dataset"] != d]
    r_tgt = np.array([float(r["z"]) for r in train]) - oof_s0(train, enc_s, seed=seed)
    phi_map = {**phi_by_slice, d: phi_d}
    m_all = fit_resid(build_X(train, enc_b, phi_by_slice, PHI_COLS), r_tgt, seed=seed)
    m_nophi = fit_resid(build_X(train, enc_b, {}, ()), r_tgt, seed=seed)
    rhat_all, _ = m_all.predict(build_X(pseudo, enc_b, phi_map, PHI_COLS))
    rhat_no, _ = m_nophi.predict(build_X(pseudo, enc_b, {}, ()))
    sc = {"hybrid": hybrid, "s0z": dict(s0),
          "resid_nophi": {k: s0[k] + float(rhat_no[i]) for i, k in enumerate(keys)},
          "hgb_phi_all": {k: s0[k] + float(rhat_all[i]) for i, k in enumerate(keys)}}
    sc["hgb_phi_all_hyb"] = {k: sc["hgb_phi_all"][k] + (HYBRID_BIAS if k in prior else 0.0) for k in keys}
    a_metrics = {arm: list_metrics(keys, truth, traj, budgets, _g36.rank_by(v, keys)) for arm, v in sc.items()}
    # B：前 6 的 25% 截点实测 → φ_B → 追加 2 → SH8-3-3
    top6 = _g36.rank_by(hybrid, keys)[:TOP_N]
    ggs_key = cand["ggs"]["cls_key"]
    train_aug = list(train) + [{"task_id": r["task_id"], "dataset": r["dataset"], "cls_key": r["cls_key"],
                                "iters": r["iters"], "r_prime": r["r_prime"], "origin": "ggs"}
                               for r in ggs_rows if r["dataset"] != d]
    src_phi = source_phi_b(train_aug, top6, budget=BUDGETS[0])
    keep = [i for i, r in enumerate(train) if r["cls_key"] not in set(top6) and r["task_id"] in src_phi]
    train_b = [train[i] for i in keep]
    X_b = np.hstack([build_X(train_b, enc_b, {}, ()), np.stack([src_phi[r["task_id"]] for r in train_b])])
    m_b = fit_resid(X_b, r_tgt[keep], seed=seed)
    q = phi_b_of({k: traj[k][budgets[0]] for k in top6}, top6)
    unstarted = [k for k in keys if k not in top6]
    pseudo_u = [p for p in pseudo if p["cls_key"] in set(unstarted)]
    Xu = build_X(pseudo_u, enc_b, {}, ())
    rhat_b, _ = m_b.predict(np.hstack([Xu, np.repeat(q[None, :], len(pseudo_u), axis=0)]))
    rhat_bn, _ = m_nophi.predict(Xu)
    s_b = {p["cls_key"]: s0[p["cls_key"]] + float(rhat_b[i]) for i, p in enumerate(pseudo_u)}
    s_bn = {p["cls_key"]: s0[p["cls_key"]] + float(rhat_bn[i]) for i, p in enumerate(pseudo_u)}
    hybrid_rank = _g36.rank_by(hybrid, keys)
    added = {"B_resid": _g36.rank_by(s_b, unstarted)[:2], "B_nophi": _g36.rank_by(s_bn, unstarted)[:2],
             B_CONTROL: hybrid_rank[TOP_N:TOP_N + 2],
             "+2_gbdt": _g36.rank_by({k: s0[k] for k in unstarted}, unstarted)[:2]}
    b_metrics = {}
    for arm, a in added.items():
        m = sh8_metrics(top6, a, traj, truth, budgets=budgets)
        m["ggs_added"] = ggs_key in a
        m["acc"] = float(truth[m["pick"]])
        b_metrics[arm] = m
    return {"dataset": d, "budgets": budgets, "n_candidates": len(keys), "oracle": oracle_key(truth),
            "oracle_acc": max(truth.values()), "ggs_key": ggs_key, "ggs_acc": truth.get(ggs_key),
            "top6": top6, "ggs_in_top6": ggs_key in top6, "phi_b": q.tolist(),
            "A": {arm: {**m, "acc": float(truth[m["sh_pick"]]) if m["sh_pick"] else None} for arm, m in a_metrics.items()},
            "B": b_metrics, "n_train_b": len(train_b)}


# ================================================================ 门、读数码、预测、表（PROTOCOL §4 / §5）
def _mean_metric(rows: Sequence[dict], arm: str, field: str) -> float:
    return float(np.mean([float(r["metrics"][arm][field]) for r in rows]))


def _wilcoxon(rows: Sequence[dict], a: str, b: str, field: str) -> dict:
    x = np.array([float(r["metrics"][a][field]) for r in rows])
    y = np.array([float(r["metrics"][b][field]) for r in rows])
    return {"n": len(rows), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "p": float(_g36.paired_wilcoxon_less(x, y)), "n_better": int(np.sum(x < y)), "n_worse": int(np.sum(x > y))}


def gates(inst_a: Sequence[dict], inst_b: Sequence[dict], *, alpha: float = ALPHA) -> dict:
    a290 = [r for r in inst_a if int(r["budget"]) == 290]
    top_main = sum(bool(r["metrics"][A_MAIN]["oracle_in_top"]) for r in a290)
    top_hyb = sum(bool(r["metrics"]["hybrid"]["oracle_in_top"]) for r in a290)
    w1 = _wilcoxon(inst_a, A_MAIN, "hybrid", "regret3")
    A1 = {"in_top_main_290": top_main, "in_top_hybrid_290": top_hyb, "need": top_hyb + A1_TOP_GAIN, **{f"r3_{k}": v for k, v in w1.items()},
          "pass": bool(top_main >= top_hyb + A1_TOP_GAIN and w1["p"] < alpha)}
    w2 = _wilcoxon(inst_a, A_MAIN, "resid_nophi", "regret3")
    shuffle_arms = sorted(k for k in inst_a[0]["metrics"] if k.startswith("hgb_phi_shuffle_"))
    gain_real = w1["mean_b"] - w1["mean_a"]
    gain_shuf = [w1["mean_b"] - _mean_metric(inst_a, s, "regret3") for s in shuffle_arms]
    A2 = {"vs_nophi": w2, "gain_real": gain_real, "gain_shuffle": gain_shuf, "gain_shuffle_mean": float(np.mean(gain_shuf)),
          "pass": bool(w2["p"] < alpha and gain_real >= 2 * float(np.mean(gain_shuf)))}
    in8_main = sum(bool(r["metrics"][B_MAIN]["oracle_in_8"]) for r in inst_b)
    in8_ctl = sum(bool(r["metrics"][B_CONTROL]["oracle_in_8"]) for r in inst_b)
    wb1 = _wilcoxon(inst_b, B_MAIN, B_CONTROL, "regret")
    B1 = {"in8_main": in8_main, "in8_control": in8_ctl, "need": in8_ctl + B1_TOP_GAIN, **{f"r_{k}": v for k, v in wb1.items()},
          "pass": bool(in8_main >= in8_ctl + B1_TOP_GAIN and wb1["p"] < alpha)}
    wb2 = _wilcoxon(inst_b, B_MAIN, "B_nophi", "regret")
    b_shuf = sorted(k for k in inst_b[0]["metrics"] if k.startswith("B_shuffle_"))
    gb_real = wb1["mean_b"] - wb1["mean_a"]
    gb_shuf = [wb1["mean_b"] - _mean_metric(inst_b, s, "regret") for s in b_shuf]
    B2 = {"vs_nophi": wb2, "gain_real": gb_real, "gain_shuffle": gb_shuf, "gain_shuffle_mean": float(np.mean(gb_shuf)),
          "pass": bool(wb2["p"] < alpha and gb_real >= 2 * float(np.mean(gb_shuf)))}
    m_main = _mean_metric(inst_b, B_MAIN, "regret")
    m_sh10 = _mean_metric(inst_b, "SH10-4-3_prior", "regret")
    B3 = {"mean_main": m_main, "mean_sh10": m_sh10, "tol": B3_TOL, "descriptive": True,
          "pass": bool(m_main <= m_sh10 + B3_TOL)}
    return {"A1": A1, "A2": A2, "B1": B1, "B2": B2, "B3": B3}


def verdict_code(g: Mapping[str, Mapping]) -> str:
    a = bool(g["A1"]["pass"]) and bool(g["A2"]["pass"])
    b = bool(g["B1"]["pass"]) and bool(g["B2"]["pass"])
    if a and b:
        return "TF-WIN"
    if a:
        return "TF-A-WIN"
    if b:
        return "TF-B-WIN"
    return "TF-DEAD"


def predictions(g: Mapping, inst_a: Sequence[dict], inst_b: Sequence[dict], stage2: Mapping, meta_a: Mapping) -> dict:
    p = {}
    p["P2"] = {"claim": "A1 ∧ A2", "prior": 0.25, "hit": bool(g["A1"]["pass"] and g["A2"]["pass"])}
    p["P3"] = {"claim": "B1 ∧ B2", "prior": 0.45, "hit": bool(g["B1"]["pass"] and g["B2"]["pass"])}
    p["P4"] = {"claim": "B3：SH6+2-3-3_B ≤ SH10-4-3_prior + 0.005", "prior": 0.50, "hit": bool(g["B3"]["pass"])}
    hw = stage2.get("per_target", {}).get("Handwriting")
    p["P5"] = {"claim": "官方口径 HW 上 B 把 GGS 追加进 8 条起跑名单", "prior": 0.40,
               "hit": bool(hw and hw["B"][B_MAIN]["ggs_added"]) if hw else None}
    w = _wilcoxon(inst_a, "resid_nophi", "hybrid", "regret3")
    p["P7"] = {"claim": "resid_nophi 单独已显著优于 hybrid3", "prior": 0.40, "p": w["p"], "hit": bool(w["p"] < ALPHA)}
    imp = meta_a.get("importance_mean", {})
    p["P8"] = {"claim": "φ 分组置换重要性 sig > meta", "prior": 0.55, "phi_sig": imp.get("phi_sig"), "phi_meta": imp.get("phi_meta"),
               "hit": bool(imp.get("phi_sig", 0) > imp.get("phi_meta", 0))}
    d = _mean_metric(inst_a, "hgb_phi_slice", "regret3") - _mean_metric(inst_a, A_MAIN, "regret3")
    p["P9"] = {"claim": "hgb_phi_slice 与 hgb_phi_all 的 regret@3 之差 < 0.003", "prior": 0.60, "delta": d, "hit": bool(abs(d) < 0.003)}
    return p


def summarize_a(inst_a: Sequence[dict]) -> dict:
    arms = list(inst_a[0]["metrics"])
    out = {"pooled": {}, "by_budget": {}, "by_dataset_290": {}}

    def block(rows):
        res = {"n": len(rows)}
        for a in arms:
            ms = [r["metrics"][a] for r in rows]
            res[a] = {"oracle_in_top": _rate([m["oracle_in_top"] for m in ms]),
                      "n_in_top": int(sum(bool(m["oracle_in_top"]) for m in ms)),
                      "regret1": _mean([m["regret1"] for m in ms]), "regret3": _mean([m["regret3"] for m in ms]),
                      "regret_top": _mean([m["regret_top"] for m in ms]), "sh_regret": _mean([m["sh_regret"] for m in ms]),
                      "pair_acc": _mean([m.get("pair_acc") for m in ms])}
        return res
    out["pooled"] = block(inst_a)
    for b in BUDGETS:
        out["by_budget"][str(b)] = block([r for r in inst_a if int(r["budget"]) == b])
    r290 = [r for r in inst_a if int(r["budget"]) == 290]
    for d in sorted({r["dataset"] for r in r290}):
        out["by_dataset_290"][d] = block([r for r in r290 if r["dataset"] == d])
    return out


def summarize_b(inst_b: Sequence[dict]) -> dict:
    arms = list(inst_b[0]["metrics"])
    out = {"pooled": {"n": len(inst_b)}, "by_dataset": {}}

    def block(rows):
        res = {"n": len(rows)}
        for a in arms:
            ms = [r["metrics"][a] for r in rows]
            fr = [r["fresh_regret"][a] for r in rows if r.get("fresh_regret") and r["fresh_regret"].get(a) is not None]
            res[a] = {"oracle_in_8": _mean([float(m["oracle_in_8"]) for m in ms]),
                      "n_in_8": float(sum(float(m["oracle_in_8"]) for m in ms)),
                      "regret": _mean([m["regret"] for m in ms]),
                      "regret_median": float(np.median([m["regret"] for m in ms])),
                      "cost": _mean([m["cost"] for m in ms]),
                      "fresh_regret": _mean(fr), "n_fresh": len(fr)}
        return res
    out["pooled"] = block(inst_b)
    for d in sorted({r["dataset"] for r in inst_b}):
        out["by_dataset"][d] = block([r for r in inst_b if r["dataset"] == d])
    return out


def markdown_tables(a: Mapping) -> str:
    L = [f"# WP6-TF 结果表（读数码 **{a['verdict']}**）", ""]
    sa = a["summary_a"]
    L += ["## A：起跑名单臂（pooled 126 / @290 42）", "",
          "| 臂 | 入前6率 pooled | 入前6 @290 | regret@1 | regret@3 | 前6最好 | 配对acc | SH6-3-3 @290 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    n290 = sa["by_budget"]["290"]["n"]
    for arm in sa["pooled"]:
        if arm == "n":
            continue
        p, q = sa["pooled"][arm], sa["by_budget"]["290"][arm]
        L.append(f"| {arm} | {p['oracle_in_top']:.3f} | {q['n_in_top']}/{n290} | {p['regret1']:.4f} | {p['regret3']:.4f} | "
                 f"{p['regret_top']:.4f} | {p['pair_acc']:.3f} | {q['sh_regret']:.4f} |")
    L += ["", "## A：按数据集 @290（hybrid / hgb_phi_all：入前6 · regret@3 · SH）", "", "| 数据集 | n | hybrid | hgb_phi_all | resid_nophi |", "|---|---:|---|---|---|"]
    for d, blk in sa["by_dataset_290"].items():
        cells = [f"{blk[x]['n_in_top']} · {blk[x]['regret3']:.3f} · {blk[x]['sh_regret']:.3f}" for x in ("hybrid", A_MAIN, "resid_nophi")]
        L.append(f"| {d} | {blk['n']} | " + " | ".join(cells) + " |")
    sb = a["summary_b"]
    L += ["", "## B：追加 2 条（42 实例 @290；成本 iters）", "", "| 臂 | oracle 入 8 | mean regret | median | 成本 | 新鲜种子 regret (n) |", "|---|---:|---:|---:|---:|---|"]
    for arm, v in sb["pooled"].items():
        if arm == "n":
            continue
        fr = f"{v['fresh_regret']:.4f} ({v['n_fresh']})" if v["fresh_regret"] is not None else "—"
        L.append(f"| {arm} | {v['n_in_8']:.1f}/{sb['pooled']['n']} | {v['regret']:.4f} | {v['regret_median']:.4f} | {v['cost']:.0f} | {fr} |")
    L += ["", "## 门（PROTOCOL §4）", "", "| 门 | 内容 | 读数 | 过 |", "|---|---|---|---|"]
    g = a["gates"]
    L.append(f"| A1 | 入前6 @290 ≥ hybrid+{A1_TOP_GAIN} ∧ regret@3 < hybrid3 | {g['A1']['in_top_main_290']} vs 需 {g['A1']['need']}；r3 {g['A1']['r3_mean_a']:.4f}/{g['A1']['r3_mean_b']:.4f} p={g['A1']['r3_p']:.2e} | {'✓' if g['A1']['pass'] else '✗'} |")
    L.append(f"| A2 | < resid_nophi ∧ 增益 ≥ 2×shuffle | p={g['A2']['vs_nophi']['p']:.2e}；gain {g['A2']['gain_real']:.4f} vs shuffle 均 {g['A2']['gain_shuffle_mean']:.4f} | {'✓' if g['A2']['pass'] else '✗'} |")
    L.append(f"| B1 | 入 8 ≥ 对照+{B1_TOP_GAIN} ∧ regret < +2_prior78 | {g['B1']['in8_main']} vs 需 {g['B1']['need']}；r {g['B1']['r_mean_a']:.4f}/{g['B1']['r_mean_b']:.4f} p={g['B1']['r_p']:.2e} | {'✓' if g['B1']['pass'] else '✗'} |")
    L.append(f"| B2 | < B_nophi ∧ 增益 ≥ 2×shuffle | p={g['B2']['vs_nophi']['p']:.2e}；gain {g['B2']['gain_real']:.4f} vs shuffle 均 {g['B2']['gain_shuffle_mean']:.4f} | {'✓' if g['B2']['pass'] else '✗'} |")
    L.append(f"| B3（描述性） | SH6+2-3-3_B ≤ SH10-4-3_prior + 0.005 | {g['B3']['mean_main']:.4f} vs {g['B3']['mean_sh10']:.4f} | {'✓' if g['B3']['pass'] else '✗'} |")
    L += ["", "## 预测对账（PROTOCOL §5）", "", "| 预测 | 内容 | 先验 | 读数 | 命中 |", "|---|---|---:|---|---|"]
    for k, p in a["predictions"].items():
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in p.items() if kk not in ("claim", "hit", "prior")}
        L.append(f"| {k} | {p['claim']} | {p['prior']:.2f} | {detail} | {'✓' if p['hit'] else ('✗' if p['hit'] is not None else '—')} |")
    s2 = a.get("stage2", {}).get("per_target", {})
    if s2:
        L += ["", "## 阶段 2：官方口径重放（seed 0 acc@100%）", "", "| 靶 | oracle | hybrid SH6-3-3 | hgb_phi_all SH6-3-3 | 入前6 hyb/φ | +2_prior78 SH8 | B_resid SH8 | GGS 入榜（hyb前6 / B追加） |", "|---|---:|---:|---:|---|---:|---:|---|"]
        for d, t in s2.items():
            L.append(f"| {d} | {t['oracle_acc']:.3f} | {t['A']['hybrid']['acc']:.3f} | {t['A'][A_MAIN]['acc']:.3f} | "
                     f"{'✓' if t['A']['hybrid']['oracle_in_top'] else '✗'}/{'✓' if t['A'][A_MAIN]['oracle_in_top'] else '✗'} | "
                     f"{t['B'][B_CONTROL]['acc']:.3f} | {t['B'][B_MAIN]['acc']:.3f} | {t['ggs_in_top6']} / {t['B'][B_MAIN]['ggs_added']} |")
    L += ["", "φ 分组置换重要性（7 折均值）：" + json.dumps({k: round(v, 4) for k, v in a["meta_a"]["importance_mean"].items()}, ensure_ascii=False), ""]
    return "\n".join(L) + "\n"


# ================================================================ 子命令
def _load_common(mx3: Path):
    cat, ggs, probe, _rk = _g36.load_inputs(mx3)
    cat_z = _g36.zscore_within(cat)
    space = CLSSpace.from_yaml()
    return cat, cat_z, probe, CLSEncoder(space), CLSEncoder(space, mode="binned")


def cmd_stage1a(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0, n_shuffle: int = N_SHUFFLE) -> dict:
    _cat, cat_z, _probe, enc_s, enc_b = _load_common(mx3)
    phi = load_phi(outdir / "task_features.jsonl")
    delta = pq.load_delta_calibration(mx3 / "delta_calibration.json")
    inst, meta = [], {}
    for target in folds:
        rows, m = fold_stage1a(target, cat_z, phi, enc_s, enc_b, delta, seed=seed, n_shuffle=n_shuffle)
        inst += rows
        meta[target] = m
        r290 = [r for r in rows if int(r["budget"]) == 290]
        print(f"[1a {target}] 入前6 @290 hybrid {sum(r['metrics']['hybrid']['oracle_in_top'] for r in r290)}/6 "
              f"φ {sum(r['metrics'][A_MAIN]['oracle_in_top'] for r in r290)}/6；regret@3 pooled hybrid "
              f"{_mean_metric(rows, 'hybrid', 'regret3'):.4f} φ {_mean_metric(rows, A_MAIN, 'regret3'):.4f}", flush=True)
    imp_keys = list(next(iter(meta.values()))["importance"])
    meta["importance_mean"] = {k: float(np.mean([meta[t]["importance"][k] for t in folds])) for k in imp_keys}
    out = {"protocol": "results/mx4_task_feature/PROTOCOL.md §3 A", "folds": list(folds), "seed": seed,
           "n_instances": len(inst), "meta": meta, "per_instance": inst}
    (outdir / "stage1a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_stage1b(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0, n_shuffle: int = N_SHUFFLE) -> dict:
    _cat, cat_z, probe, enc_s, enc_b = _load_common(mx3)
    inst, meta = [], {}
    for target in folds:
        rows, m = fold_stage1b(target, cat_z, enc_s, enc_b, probe, seed=seed, n_shuffle=n_shuffle)
        inst += rows
        meta[target] = m
        print(f"[1b {target}] 入8 B_resid {sum(r['metrics'][B_MAIN]['oracle_in_8'] for r in rows)}/6 "
              f"对照 {sum(r['metrics'][B_CONTROL]['oracle_in_8'] for r in rows)}/6；regret B {_mean_metric(rows, B_MAIN, 'regret'):.4f} "
              f"对照 {_mean_metric(rows, B_CONTROL, 'regret'):.4f} SH6 {_mean_metric(rows, 'SH6-3-3_prior', 'regret'):.4f}", flush=True)
    out = {"protocol": "results/mx4_task_feature/PROTOCOL.md §3 B", "folds": list(folds), "seed": seed,
           "n_instances": len(inst), "meta": meta, "per_instance": inst}
    (outdir / "stage1b.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_stage2(mx3: Path, outdir: Path, *, seed: int = 0, targets: Sequence[str] = OFFICIAL_TARGETS) -> dict:
    from scripts.e_ablation_common import read_jsonl
    cat, cat_z, _probe, enc_s, enc_b = _load_common(mx3)
    ggs_rows = [r for r in read_jsonl(mx3 / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] == "ggs"]
    phi = load_phi(outdir / "task_features.jsonl")
    pa_rows = read_jsonl(PA_RUNS)
    cands = json.loads((DEPLOY / "candidates.json").read_text(encoding="utf-8"))
    runs = [r for r in read_jsonl(DEPLOY / "runs.jsonl") if r.get("stage") == "s1_catalog" and int(r.get("seed", 0)) == 0
            and r.get("error") is None]
    per_target, phi_official = {}, {}
    for d in targets:
        if d not in cands:
            continue
        phi_official[d] = official_phi(d, pa_rows)
        s1 = [r for r in runs if r["dataset"] == d]
        per_target[d] = replay_official(d, cands[d], s1, phi_official[d], phi, cat_z, ggs_rows, enc_s, enc_b, seed=seed)
        t = per_target[d]
        print(f"[2 {d}] oracle {t['oracle_acc']:.3f}；hybrid SH {t['A']['hybrid']['acc']:.3f} φ SH {t['A'][A_MAIN]['acc']:.3f}；"
              f"+2_prior78 {t['B'][B_CONTROL]['acc']:.3f} B_resid {t['B'][B_MAIN]['acc']:.3f}（GGS 追加 {t['B'][B_MAIN]['ggs_added']}）", flush=True)
    out = {"protocol": "results/mx4_task_feature/PROTOCOL.md §2.6（描述性）", "seed": seed,
           "phi_official": phi_official, "per_target": per_target}
    (outdir / "stage2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_analyze(outdir: Path) -> dict:
    s1a = json.loads((outdir / "stage1a.json").read_text(encoding="utf-8"))
    s1b = json.loads((outdir / "stage1b.json").read_text(encoding="utf-8"))
    s2p = outdir / "stage2.json"
    s2 = json.loads(s2p.read_text(encoding="utf-8")) if s2p.exists() else {}
    inst_a, inst_b = s1a["per_instance"], s1b["per_instance"]
    g = gates(inst_a, inst_b)
    analysis = {"protocol": "results/mx4_task_feature/PROTOCOL.md", "n_a": len(inst_a), "n_b": len(inst_b),
                "gates": g, "verdict": verdict_code(g),
                "summary_a": summarize_a(inst_a), "summary_b": summarize_b(inst_b),
                "meta_a": {"importance_mean": s1a["meta"]["importance_mean"],
                           "prior_top6": {t: s1a["meta"][t]["prior_top6"] for t in s1a["folds"]}},
                "meta_b": {t: {k: v for k, v in s1b["meta"][t].items() if k != "shuffle_donors"} for t in s1b["folds"]},
                "stage2": {"per_target": s2.get("per_target", {}), "phi_official": s2.get("phi_official", {})}}
    analysis["predictions"] = predictions(g, inst_a, inst_b, analysis["stage2"], analysis["meta_a"])
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown_tables(analysis)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md)
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage0", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--stage1a", action="store_true")
    ap.add_argument("--stage1b", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--mx3", default=str(MX3))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    mx3, outdir = Path(args.mx3), Path(args.outdir)
    ran = False
    if args.stage0:
        cmd_stage0(mx3, outdir, args.folds, seed=args.seed, n_boot=args.n_boot); ran = True
    if args.features:
        cmd_features(mx3, outdir, resume=not args.no_resume); ran = True
    if args.stage1a:
        cmd_stage1a(mx3, outdir, args.folds, seed=args.seed, n_shuffle=args.n_shuffle); ran = True
    if args.stage1b:
        cmd_stage1b(mx3, outdir, args.folds, seed=args.seed, n_shuffle=args.n_shuffle); ran = True
    if args.stage2:
        cmd_stage2(mx3, outdir, seed=args.seed); ran = True
    if args.analyze:
        cmd_analyze(outdir); ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
