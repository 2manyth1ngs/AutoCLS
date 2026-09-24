"""WP17-ACTS S3-A（results/acts/s3a/PROTOCOL.md）：低秩交互模型 z(t,c) = μ(c) + Σ_{k≤K} a_k(t)·b_k(c) + 零 GPU 特征 LOSO 预测 a(t)。

零 GPU、只读 S2 缓存。每个 LOSO 折（留一源）只在训练源上拟合 μ / B / ridge / α / K；留出源只考。
纯函数（pool_tasks / meta_features / lowrank_fit / residual / project / align_sign / scores_from / regret_at_k / ridge_* /
choose_alpha / loso_evaluate / cluster_bootstrap_ci / source_wilcoxon / verdict_s3a）不含数据集概念；源分组只在脚本层（铁律 7）。

用法：python scripts/84_acts_s3a.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import rankdata, spearmanr, wilcoxon  # noqa: E402

S2_DIR = ROOT / "results" / "acts" / "s2"
OUT = ROOT / "results" / "acts" / "s3a"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
KS = (1, 2, 3)
ELIGIBLE = ("meta", "stat25", "land25", "probe")          # PROTOCOL §1.5：进门的四组（可部署）
COST = {"meta": 0.0, "stat25": 7.5, "stat_diag": 60.0, "land25": 7.5, "probe": 1.0, "all": 7.5}   # 目标侧全跑当量（§1.3）

Feature = Union[np.ndarray, Callable[[Mapping], np.ndarray]]


# ---------------------------------------------------------------- 数据选择 / 元特征
def pool_tasks(probe_tasks: Sequence[str], meta: Mapping[str, Mapping], pool: Sequence[str], demoted_to_8: Sequence[str]) -> list[str]:
    """训练池任务：源 ∈ pool；降 8 片的源只取 a00–a07（S2 G-R）。"""
    out = []
    for t in probe_tasks:
        d = meta[t]["dataset"]
        if d not in pool:
            continue
        if d in demoted_to_8 and int(t.rsplit("_a", 1)[1]) >= 8:
            continue
        out.append(t)
    return out


def meta_features(m: Mapping) -> np.ndarray:
    """① meta：log n_train、log T、log C、log k、majority、切片档位 c / w / n（tag 'c{i}w{j}n{k}'）。"""
    tag = str(m["tag"])
    c, w, n = int(tag[1]), int(tag[3]), int(tag[5])
    return np.array([np.log(float(m["n_train"])), np.log(float(m["T"])), np.log(float(m["C"])), np.log(float(m["k"])),
                     float(m["majority"]), c, w, n], dtype=float)


# ---------------------------------------------------------------- 低秩模型
def residual(Z: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """R = Z − μ(c) 再按行去均值（去任务主效应，不改任务内排序）。"""
    R = np.asarray(Z, float) - np.asarray(mu, float)[None, :]
    return R - R.mean(1, keepdims=True)


def lowrank_fit(Z: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """μ = 列均值；B = 残差 SVD 前 K 个右奇异向量（列正交，30 × K）。"""
    Z = np.asarray(Z, float)
    mu = Z.mean(0)
    _U, _S, Vt = np.linalg.svd(residual(Z, mu), full_matrices=False)
    return mu, Vt[:K].T.copy()


def project(Z: np.ndarray, mu: np.ndarray, B: np.ndarray) -> np.ndarray:
    """a(t) = R(t)·B（B 列正交 → 最小二乘）。"""
    return residual(Z, mu) @ B


def align_sign(B: np.ndarray, A: np.ndarray, b_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """第一分量与参照同向（点积 ≥ 0），否则 b₁ 与 a₁ 同时翻转；其余分量不动。"""
    B, A = np.array(B, float), np.array(A, float)
    if float(np.dot(B[:, 0], b_ref)) < 0:
        B[:, 0] *= -1; A[:, 0] *= -1
    return B, A


def scores_from(mu: np.ndarray, B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ŝ(t,c) = μ(c) + a(t)·Bᵀ。"""
    return np.asarray(mu, float)[None, :] + np.asarray(A, float) @ np.asarray(B, float).T


def regret_at_k(S: np.ndarray, end0: np.ndarray, end1: np.ndarray, k: int) -> np.ndarray:
    """S2 H_cat 口径：oracle = argmax end0（seed 0 选），pick = 按 S 排前 k 里 end0 最高；regret = end1[oracle] − end1[pick]。"""
    S, end0, end1 = np.asarray(S, float), np.asarray(end0, float), np.asarray(end1, float)
    out = np.empty(S.shape[0])
    for i in range(S.shape[0]):
        top = np.argsort(-S[i], kind="stable")[:k]
        pick = int(top[int(np.argmax(end0[i, top]))])
        out[i] = end1[i, int(np.argmax(end0[i]))] - end1[i, pick]
    return out


# ---------------------------------------------------------------- ridge
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """标准化特征（常数列 sd → 1）后的闭式 ridge，截距不罚。"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m, s = X.mean(0), X.std(0)
    s = np.where(s > 1e-12, s, 1.0)
    Xs = (X - m) / s
    ym = float(y.mean())
    d = Xs.shape[1]
    w = np.linalg.solve(Xs.T @ Xs + float(alpha) * np.eye(d), Xs.T @ (y - ym))
    return {"m": m, "s": s, "w": w, "b": ym}


def ridge_predict(model: Mapping, X: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, float) - model["m"]) / model["s"]) @ model["w"] + model["b"]


def choose_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float]) -> float:
    """按组留一（LOSO）最小化预测 MSE 选 α；平局取大 α（更保守）。"""
    alphas = list(alphas)
    if len(alphas) == 1:
        return float(alphas[0])
    groups = np.asarray(groups)
    err = np.zeros(len(alphas))
    for gname in np.unique(groups):
        te = groups == gname; tr = ~te
        for j, a in enumerate(alphas):
            err[j] += float(np.sum((ridge_predict(ridge_fit(X[tr], y[tr], a), X[te]) - y[te]) ** 2))
    best = np.min(err)
    return float(max(a for a, e in zip(alphas, err) if e <= best * (1 + 1e-9) + 1e-12))


# ---------------------------------------------------------------- LOSO
N_BASIS = 3                                               # 折内基底至少 3 个分量（③ 投到 B₃，与 K 无关）


def _fold(Z: np.ndarray, tr: np.ndarray, Kmax: int, b_ref: np.ndarray) -> dict:
    mu, B = lowrank_fit(Z[tr], max(N_BASIS, Kmax))
    A_all = project(Z, mu, B)
    B, A_all = align_sign(B, A_all, b_ref)
    return {"mu": mu, "B": B, "A": A_all, "train_idx": tr}


def _features(feat: Feature, ctx: Mapping) -> np.ndarray:
    X = feat(ctx) if callable(feat) else feat
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


def _predict_a(X: np.ndarray, A: np.ndarray, groups: np.ndarray, tr: np.ndarray, te: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """每分量独立 ridge（α 各自在训练侧 LOSO 选）；返回 te 行的 â。"""
    out = np.zeros((int(te.sum()) if te.dtype == bool else len(te), A.shape[1]))
    for k in range(A.shape[1]):
        a = choose_alpha(X[tr], A[tr, k], groups[tr], alphas)
        out[:, k] = ridge_predict(ridge_fit(X[tr], A[tr, k], a), X[te])
    return out


def _choose_k(Z, groups, end0, end1, feat, tr_idx, Ks, alphas, b_ref, k_eval, oracle: bool) -> int:
    """外层训练源上的内层 LOSO：最小化内层任务加权 regret@k_eval 选 K（平局取小 K）。"""
    Kmax = max(Ks)
    tr_groups = groups[tr_idx]
    tot = {K: [] for K in Ks}
    for s2 in np.unique(tr_groups):
        te2 = tr_idx[tr_groups == s2]
        tr2 = tr_idx[tr_groups != s2]
        ctx = _fold(Z, tr2, Kmax, b_ref)
        if oracle:
            Ah = ctx["A"][te2, :Kmax]
        else:
            X = _features(feat, ctx)
            Ah = _predict_a(X, ctx["A"][:, :Kmax], groups, tr2, te2, alphas)
        for K in Ks:
            S = scores_from(ctx["mu"], ctx["B"][:, :K], Ah[:, :K])
            tot[K].extend(regret_at_k(S, end0[te2], end1[te2], k_eval).tolist())
    means = {K: float(np.mean(v)) for K, v in tot.items()}
    best = min(means.values())
    return int(min(K for K, v in means.items() if v <= best + 1e-12))


def _summ(groups, a_hat, a_true, r1, r3, r3_blind, extra: Mapping) -> dict:
    srcs = list(dict.fromkeys(groups.tolist()))
    sm_h = [float(np.mean(a_hat[groups == s, 0])) for s in srcs]
    sm_t = [float(np.mean(a_true[groups == s, 0])) for s in srcs]
    wh = a_hat[:, 0] - np.array([np.mean(a_hat[groups == g, 0]) for g in groups])
    wt = a_true[:, 0] - np.array([np.mean(a_true[groups == g, 0]) for g in groups])
    return {"a_hat": a_hat, "a_true": a_true, "regret1": r1, "regret3": r3,
            "rho_a1": float(spearmanr(a_hat[:, 0], a_true[:, 0])[0]) if np.std(a_hat[:, 0]) > 0 else 0.0,
            "rho_src": float(spearmanr(sm_h, sm_t)[0]) if np.std(sm_h) > 0 else 0.0,
            "rho_within": float(spearmanr(wh, wt)[0]) if np.std(wh) > 0 else 0.0,
            "mean_regret1": float(np.mean(r1)), "mean_regret3": float(np.mean(r3)),
            "delta_regret3": float(np.mean(r3_blind - r3)), **extra}


def loso_evaluate(Z: np.ndarray, groups: Sequence, end0: np.ndarray, end1: np.ndarray, feats: Mapping[str, Feature], *,
                  Ks: Sequence[int] = KS, alphas: Sequence[float] = ALPHAS, inner: bool = True, b_ref: Optional[np.ndarray] = None,
                  k_eval: int = 3) -> dict:
    """PROTOCOL §1.2–1.4。返回 {组名: 读数}，另含 '_blind'（μ 排序）与 '_oracle'（真 a 代回，K 同样内层选）。
    特征可为静态矩阵或 callable(ctx)（ctx = 折内 μ / B / A / train_idx，只含训练侧拟合量）。
    inner=False → 不做内层选择，K = max(Ks)。"""
    Z, end0, end1 = np.asarray(Z, float), np.asarray(end0, float), np.asarray(end1, float)
    groups = np.asarray(groups)
    Ks = tuple(sorted(Ks)); Kmax = max(Ks)
    if b_ref is None:
        b_ref = lowrank_fit(Z, Kmax)[1][:, 0]
    n = len(Z)
    names = list(feats) + ["_oracle"]
    a_hat = {k: np.zeros((n, Kmax)) for k in names}
    a_true = np.zeros((n, Kmax))
    r1 = {k: np.zeros(n) for k in names}; r3 = {k: np.zeros(n) for k in names}
    rfix = {k: {K: np.zeros(n) for K in Ks} for k in names}
    kch = {k: [] for k in names}
    b1 = np.zeros(n); b3 = np.zeros(n)
    for s in list(dict.fromkeys(groups.tolist())):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        ctx = _fold(Z, tr, Kmax, b_ref)
        a_true[te] = ctx["A"][te, :Kmax]
        blind = np.tile(ctx["mu"], (len(te), 1))
        b1[te] = regret_at_k(blind, end0[te], end1[te], 1); b3[te] = regret_at_k(blind, end0[te], end1[te], k_eval)
        for name in names:
            oracle = name == "_oracle"
            if oracle:
                Ah = ctx["A"][te, :Kmax]
            else:
                X = _features(feats[name], ctx)
                Ah = _predict_a(X, ctx["A"][:, :Kmax], groups, tr, te, alphas)
            a_hat[name][te] = Ah
            K = _choose_k(Z, groups, end0, end1, None if oracle else feats[name], tr, Ks, alphas, b_ref, k_eval, oracle) if (inner and len(Ks) > 1) else Kmax
            kch[name].append(K)
            for Kf in Ks:
                rfix[name][Kf][te] = regret_at_k(scores_from(ctx["mu"], ctx["B"][:, :Kf], Ah[:, :Kf]), end0[te], end1[te], k_eval)
            S = scores_from(ctx["mu"], ctx["B"][:, :K], Ah[:, :K])
            r1[name][te] = regret_at_k(S, end0[te], end1[te], 1); r3[name][te] = regret_at_k(S, end0[te], end1[te], k_eval)
    out = {}
    for name in names:
        out[name] = _summ(groups, a_hat[name], a_true, r1[name], r3[name], b3,
                          {"k_chosen": kch[name], "regret3_fixed": {K: rfix[name][K] for K in Ks},
                           "mean_regret3_fixed": {K: float(np.mean(rfix[name][K])) for K in Ks},
                           "delta_regret3_fixed": {K: float(np.mean(b3 - rfix[name][K])) for K in Ks}})
    out["_blind"] = {"regret1": b1, "regret3": b3, "mean_regret1": float(np.mean(b1)), "mean_regret3": float(np.mean(b3))}
    return out


# ---------------------------------------------------------------- 统计
def cluster_bootstrap_ci(x: np.ndarray, groups: Sequence, *, n_boot: int = 10000, seed: int = 0,
                         stat: Optional[Callable[[np.ndarray], float]] = None, level: float = 0.95) -> tuple[float, float]:
    """按组（源）有放回重抽的 bootstrap 分位 CI；stat 默认均值（任务加权）。"""
    x = np.asarray(x, float); groups = np.asarray(groups)
    stat = stat or (lambda r: float(np.mean(r)))
    uniq = list(dict.fromkeys(groups.tolist()))
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx[uniq[j]] for j in pick])
        vals.append(stat(x[rows]))
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def source_wilcoxon(diff: np.ndarray, groups: Sequence) -> float:
    """源均值差的 Wilcoxon 符号秩（双侧）；全零 → 1.0。"""
    diff = np.asarray(diff, float); groups = np.asarray(groups)
    m = np.array([np.mean(diff[groups == g]) for g in dict.fromkeys(groups.tolist())])
    if np.all(np.abs(m) < 1e-15):
        return 1.0
    return float(wilcoxon(m).pvalue)


def verdict_s3a(res: Mapping[str, Mapping], eligible: Sequence[str]) -> str:
    """G-S3A（§1.5）：同一组 ρ ≥ 0.7 ∧ Δ ≥ 0.010 → GO；否则任一组 ρ ≥ 0.5 → FIELD；否则 STOP。只看 eligible。"""
    el = [res[k] for k in eligible if k in res]
    if any(r["rho_a1"] >= 0.70 and r["delta_regret3"] >= 0.010 for r in el):
        return "S3A-GO"
    if any(r["rho_a1"] >= 0.50 for r in el):
        return "S3A-FIELD"
    return "S3A-STOP"


# ---------------------------------------------------------------- 数据装配（读真实缓存）
def load_s2(s2_dir: Path = S2_DIR) -> dict:
    ac = importlib.import_module("scripts.acts_common")
    g83 = importlib.import_module("scripts.83_acts_corpus")
    cfg = ac.load_acts_config(profile="s2")
    plan = json.loads((s2_dir / "plan_S.json").read_text(encoding="utf-8"))
    cat = json.loads((s2_dir / "catalog.json").read_text(encoding="utf-8"))
    an = json.loads((s2_dir / "analysis_S.json").read_text(encoding="utf-8"))
    rows = g83._g35.load_done(s2_dir / "runs.jsonl")
    meta = plan["tasks_meta"]
    q = float(cfg["scale"]["clip_quantile"]); tf = float(cfg["scale"]["noise_top_frac"])
    frames = g83._frames_c(plan, rows, cat, cfg)
    raw = g83._raw_seed_frames(plan, rows, cat, cfg)
    floors = g83.floors_cv(g83.load_probe0(s2_dir), meta)
    n_sh = len(cat["shared"]); keys = g83.cat_keys(cat)
    pool = pool_tasks(plan["probe_tasks"], meta, an["pool_after_gates"], an["reverse_demoted_to_8"])
    extra = [t for t in plan["probe_tasks"] if meta[t]["dataset"] in cfg["sources"]["probation"]]

    def block(tasks):
        Z = np.array([ac.clip_z(frames[t]["frame_vals"], q)[0][:n_sh] for t in tasks])
        M = g83._fill_pairs({"end_0": np.array([raw[t][0] for t in tasks]), "end_1": np.array([raw[t][1] for t in tasks])})
        q25 = np.array([[g83._ckpt(rows, plan, t, k, 0, 0) for k in keys] for t in tasks])
        acc25 = g83._fill_single(q25, M["end_0"])
        maj = np.array([float(meta[t]["majority"]) for t in tasks])
        pfloor = np.array([float(np.mean(floors[t]["probe_accs"])) if floors[t]["probe_accs"] else float(meta[t]["majority"]) for t in tasks])
        s2floor = np.array([floors[t]["floor"] for t in tasks])
        Z25 = np.array([ac.clip_z(v, q)[0] for v in acc25])
        meta_X = np.array([meta_features(meta[t]) for t in tasks])
        stat25 = []
        for i in range(len(tasks)):
            v = acc25[i]; top = np.sort(v)[len(v) // 2:]
            stat25.append([pfloor[i] - maj[i], float(np.median(top)) - maj[i], float(np.mean(v <= pfloor[i])), float(np.subtract(*np.percentile(v, [75, 25])))])
        diag = []
        for i, t in enumerate(tasks):
            a0, a1 = raw[t]
            h = ac.chance_guard({t: (a0, a1)}, {t: maj[i]}, headroom_min=0.10)["h"]
            sig = ac.top_half_sigma(a0, a1, top_frac=tf)[0]
            diag.append([h[0] if h else np.nan, float(np.mean(frames[t]["m"] <= s2floor[i])), sig])
        return {"tasks": tasks, "groups": np.array([meta[t]["dataset"] for t in tasks]), "Z": Z, "end0": M["end_0"], "end1": M["end_1"],
                "acc25": acc25, "Z25": Z25, "maj": maj, "pfloor": pfloor, "meta_X": meta_X, "stat25": np.array(stat25), "diag": np.array(diag)}

    return {"pool": block(pool), "extra": block(extra), "keys": keys, "cfg": cfg}


def land25_feature(Z25: np.ndarray) -> Callable[[Mapping], np.ndarray]:
    """③：@25% 钉底 z 画像 − 训练折 @25% 列均值、行去均值 → 投到折内 B₃。"""
    def f(ctx):
        mu25 = Z25[ctx["train_idx"]].mean(0)
        return residual(Z25, mu25) @ ctx["B"]
    return f


def probe_configs(ctx: Mapping) -> list[int]:
    """④ 的四条：折内 μ 前 2（并列冠军）+ b₁ 载荷最大 / 最小（轴两端）。"""
    top = [int(j) for j in np.argsort(-ctx["mu"], kind="stable")[:2]]
    return top + [int(np.argmax(ctx["B"][:, 0])), int(np.argmin(ctx["B"][:, 0]))]


def probe_feature(acc25: np.ndarray, pfloor: np.ndarray, maj: np.ndarray) -> Callable[[Mapping], np.ndarray]:
    def f(ctx):
        p = probe_configs(ctx)
        v = acc25[:, p]
        d1, d2 = v[:, 0] - v[:, 1], v[:, 2] - v[:, 3]
        lvl = v.mean(1)
        den = np.maximum(0.01, lvl - pfloor)
        return np.column_stack([d1, d2, d1 / den, d2 / den, lvl - maj])
    return f


def all_feature(parts: Sequence[Feature]) -> Callable[[Mapping], np.ndarray]:
    return lambda ctx: np.column_stack([_features(p, ctx) for p in parts])


def m4_scores(acc25: np.ndarray, ctx: Mapping) -> np.ndarray:
    """M4：④ 四条里 @25% 最好 + 任务盲前 2（去重后补足 3 条）→ 构造分数让这 3 条排前。"""
    p = probe_configs(ctx)
    S = np.tile(ctx["mu"], (acc25.shape[0], 1)).astype(float)
    order = np.argsort(-ctx["mu"], kind="stable")
    for i in range(acc25.shape[0]):
        best = p[int(np.argmax(acc25[i, p]))]
        chosen = [best] + [int(j) for j in order if int(j) != best][:2]
        S[i, chosen] = 1e6 - np.arange(3)
    return S


def comparators(D: Mapping, b_ref: np.ndarray) -> dict:
    """同 160 任务 LOSO：H_cat 先验（训练源平均秩）、M25（直接按 @25%）、M4。"""
    groups, end0, end1 = D["groups"], D["end0"], D["end1"]
    n = len(groups)
    out = {k: {"regret1": np.zeros(n), "regret3": np.zeros(n)} for k in ("prior_rank", "M25", "M4")}
    for s in list(dict.fromkeys(groups.tolist())):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        ctx = _fold(D["Z"], tr, 3, b_ref)
        mr = np.array([rankdata(-end0[j]) for j in tr]).mean(0)
        for name, S in (("prior_rank", np.tile(-mr, (len(te), 1))), ("M25", D["acc25"][te]), ("M4", m4_scores(D["acc25"][te], ctx))):
            out[name]["regret1"][te] = regret_at_k(S, end0[te], end1[te], 1)
            out[name]["regret3"][te] = regret_at_k(S, end0[te], end1[te], 3)
    for v in out.values():
        v["mean_regret1"] = float(np.mean(v["regret1"])); v["mean_regret3"] = float(np.mean(v["regret3"]))
    return out


def representation_extra(D: Mapping, E: Mapping, feats_pool: Mapping[str, Feature], feats_extra: Mapping[str, Feature], b_ref: np.ndarray) -> dict:
    """§1.6：17 源全训 → HMD / SCP2 的 â₁ 与真 a₁（投到全训 B），对照训练任务 a₁ 的下四分位。"""
    Zall = np.vstack([D["Z"], E["Z"]])
    n_p = len(D["Z"])
    tr = np.arange(n_p)
    ctx = _fold(Zall, tr, 3, b_ref)
    q25 = float(np.quantile(ctx["A"][:n_p, 0], 0.25))
    out = {"train_a1_q25": q25, "true": {}, "pred": {}}
    for s in dict.fromkeys(E["groups"].tolist()):
        m = np.where(E["groups"] == s)[0] + n_p
        out["true"][s] = float(np.mean(ctx["A"][m, 0]))
    gall = np.concatenate([D["groups"], E["groups"]])
    for name in feats_pool:
        Xp = _features(feats_pool[name], {**ctx, "train_idx": tr})
        Xe = _features(feats_extra[name], {**ctx, "train_idx": tr})
        X = np.vstack([Xp, Xe])
        te = np.arange(n_p, len(Zall))
        Ah = _predict_a(X, ctx["A"], gall, tr, te, ALPHAS)
        out["pred"][name] = {s: float(np.mean(Ah[E["groups"] == s, 0])) for s in dict.fromkeys(E["groups"].tolist())}
    return out


def _f(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def cmd_analyze(outdir: Path = OUT) -> dict:
    data = load_s2()
    D, E = data["pool"], data["extra"]
    groups = D["groups"]
    b_ref = lowrank_fit(D["Z"], 3)[1][:, 0]
    if "SemgHandMovementCh2" in set(groups.tolist()):             # 定向：与 S2 轴同号（Semg 在负端）
        A0 = project(D["Z"], D["Z"].mean(0), b_ref[:, None])[:, 0]
        if np.mean(A0[groups == "SemgHandMovementCh2"]) > 0:
            b_ref = -b_ref
    fp = {"meta": D["meta_X"], "stat25": D["stat25"], "stat_diag": D["diag"], "land25": land25_feature(D["Z25"]),
          "probe": probe_feature(D["acc25"], D["pfloor"], D["maj"])}
    fp["all"] = all_feature([fp["meta"], fp["stat25"], fp["land25"], fp["probe"]])
    res = loso_evaluate(D["Z"], groups, D["end0"], D["end1"], fp, b_ref=b_ref)
    comp = comparators(D, b_ref)
    blind = res["_blind"]
    code = verdict_s3a(res, ELIGIBLE)
    # 统计
    summ = {}
    for name in list(fp) + ["_oracle"]:
        r = res[name]
        diff = blind["regret3"] - r["regret3"]
        summ[name] = {"rho_a1": r["rho_a1"], "rho_a1_ci": cluster_bootstrap_ci(np.column_stack([r["a_hat"][:, 0], r["a_true"][:, 0]]), groups,
                                                                                 stat=lambda x: float(spearmanr(x[:, 0], x[:, 1])[0]) if np.std(x[:, 0]) > 0 else 0.0),
                      "rho_src": r["rho_src"], "rho_within": r["rho_within"],
                      "regret1": r["mean_regret1"], "regret3": r["mean_regret3"], "delta_regret3": r["delta_regret3"],
                      "delta_ci": cluster_bootstrap_ci(diff, groups), "p_src": source_wilcoxon(diff, groups),
                      "k_chosen": r["k_chosen"], "regret3_fixed": r["mean_regret3_fixed"], "delta_fixed": r["delta_regret3_fixed"],
                      "cost": COST.get(name), "eligible": name in ELIGIBLE}
    comp_s = {k: {"regret1": v["mean_regret1"], "regret3": v["mean_regret3"], "delta_vs_blind": float(np.mean(blind["regret3"] - v["regret3"])),
                  "p_src_vs_blind": source_wilcoxon(blind["regret3"] - v["regret3"], groups)} for k, v in comp.items()}
    # 每源
    srcs = list(dict.fromkeys(groups.tolist()))
    per_src = {s: {"n": int(np.sum(groups == s)), "a1_true": float(np.mean(res["_oracle"]["a_true"][groups == s, 0])),
                   "blind": float(np.mean(blind["regret3"][groups == s])),
                   **{name: float(np.mean(res[name]["regret3"][groups == s])) for name in list(fp) + ["_oracle"]},
                   "M25": float(np.mean(comp["M25"]["regret3"][groups == s]))} for s in srcs}
    # §1.6 HMD / SCP2
    fe = {"meta": E["meta_X"], "stat25": E["stat25"], "stat_diag": E["diag"], "land25": None, "probe": None}
    Z25all = np.vstack([D["Z25"], E["Z25"]]); acc25all = np.vstack([D["acc25"], E["acc25"]])
    pf_all = np.concatenate([D["pfloor"], E["pfloor"]]); maj_all = np.concatenate([D["maj"], E["maj"]])
    n_p = len(D["Z"])
    fl = land25_feature(Z25all); fpr = probe_feature(acc25all, pf_all, maj_all)
    feats_pool = {"meta": D["meta_X"], "stat25": D["stat25"], "land25": (lambda c: fl(c)[:n_p]), "probe": (lambda c: fpr(c)[:n_p])}
    feats_extra = {"meta": E["meta_X"], "stat25": E["stat25"], "land25": (lambda c: fl(c)[n_p:]), "probe": (lambda c: fpr(c)[n_p:])}
    rep = representation_extra(D, E, feats_pool, feats_extra, b_ref)
    out = {"verdict": code, "n_tasks": int(len(groups)), "n_sources": len(srcs), "summary": summ, "blind": {"regret1": blind["mean_regret1"], "regret3": blind["mean_regret3"]},
           "comparators": comp_s, "per_source": per_src, "representation_hmd_scp2": rep,
           "a_true_within_share": float(sum(np.sum((res["_oracle"]["a_true"][groups == s, 0] - res["_oracle"]["a_true"][groups == s, 0].mean()) ** 2) for s in srcs)
                                        / np.sum((res["_oracle"]["a_true"][:, 0] - res["_oracle"]["a_true"][:, 0].mean()) ** 2))}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return out


def tables_markdown(o: Mapping) -> str:
    lab = {"meta": "① meta", "stat25": "②′ stat25", "stat_diag": "② stat-diag（诊断）", "land25": "③ land25", "probe": "④ probe", "all": "all（副）", "_oracle": "oracle-a（天花板）"}
    L = [f"# S3-A 读数（{o['verdict']}）", "", f"{o['n_tasks']} 任务 / {o['n_sources']} 源；任务盲 μ regret@1 {_f(o['blind']['regret1'])} / regret@3 {_f(o['blind']['regret3'])}；a₁ 源内份额 {_f(o['a_true_within_share'], 2)}", "",
         "| 组 | 进门 | 成本 | ρ_a1 [95% CI] | ρ 源均值 | ρ 源内 | regret@1 | regret@3 | Δregret@3 [95% CI] | p 源 | K 选择 | Δ K=1 / 2 / 3 |",
         "|---|---|---:|---|---:|---:|---:|---:|---|---:|---|---|"]
    for k, v in o["summary"].items():
        ks = v["k_chosen"]; kc = {K: ks.count(K) for K in (1, 2, 3)}
        L.append(f"| {lab.get(k, k)} | {'✓' if v['eligible'] else '—'} | {v['cost'] if v['cost'] is not None else '—'} | {_f(v['rho_a1'], 2)} [{_f(v['rho_a1_ci'][0], 2)}, {_f(v['rho_a1_ci'][1], 2)}] | "
                 f"{_f(v['rho_src'], 2)} | {_f(v['rho_within'], 2)} | {_f(v['regret1'])} | {_f(v['regret3'])} | {_f(v['delta_regret3'])} [{_f(v['delta_ci'][0])}, {_f(v['delta_ci'][1])}] | "
                 f"{_f(v['p_src'], 3)} | {kc[1]}/{kc[2]}/{kc[3]} | {' / '.join(_f(v['delta_fixed'][K]) for K in (1, 2, 3))} |")
    L += ["", "| 对照 | regret@1 | regret@3 | Δ vs 任务盲 | p 源 |", "|---|---:|---:|---:|---:|"]
    for k, v in o["comparators"].items():
        L.append(f"| {k} | {_f(v['regret1'])} | {_f(v['regret3'])} | {_f(v['delta_vs_blind'])} | {_f(v['p_src_vs_blind'], 3)} |")
    names = [k for k in o["summary"]]
    L += ["", "| 源 | n | 真 a₁ | 任务盲 | " + " | ".join(lab.get(k, k) for k in names) + " | M25 |", "|---|---:|---:|---:|" + "---:|" * (len(names) + 1)]
    for s, v in sorted(o["per_source"].items(), key=lambda kv: kv[1]["a1_true"]):
        L.append(f"| {s} | {v['n']} | {_f(v['a1_true'], 2)} | {_f(v['blind'])} | " + " | ".join(_f(v[k]) for k in names) + f" | {_f(v['M25'])} |")
    r = o["representation_hmd_scp2"]
    L += ["", f"HMD / SCP2 表征（17 源全训；训练任务 a₁ 下四分位 {_f(r['train_a1_q25'], 2)}）：真 a₁ {', '.join(f'{s} {_f(v, 2)}' for s, v in r['true'].items())}"]
    for k, v in r["pred"].items():
        L.append(f"- {lab.get(k, k)} â₁：{', '.join(f'{s} {_f(x, 2)}' for s, x in v.items())}")
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        cmd_analyze(Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
