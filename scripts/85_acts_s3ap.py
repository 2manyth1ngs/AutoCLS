"""WP17-ACTS S3-A′（results/acts/s3ap/PROTOCOL.md）：零 GPU 三项——任务 3 探针数曲线 / 任务 1 配置侧归纳 / 任务 2 名单外净空归属。

复用 84 的低秩模型与 LOSO（每折只在训练源拟合；留出源只考）。纯函数不含数据集概念（铁律 7）。
用法：python scripts/85_acts_s3ap.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import norm, rankdata, spearmanr  # noqa: E402

g84 = importlib.import_module("scripts.84_acts_s3a")
OUT = ROOT / "results" / "acts" / "s3ap"
S2_DIR = ROOT / "results" / "acts" / "s2"
K_LIST = (2, 4, 6, 8, 12)
N_GREEDY = 12
GREEDY_ALPHA = 10.0
SPACE_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)


# ================================================================ 纯函数
def expected_max(n: int) -> float:
    """n 个独立标准正态最大值的期望（同 83.expected_max_normal；n=3 → 0.846）。"""
    n = int(n)
    if n <= 1:
        return 0.0
    x = np.linspace(-9.0, 9.0, 36001)
    pdf = n * norm.pdf(x) * norm.cdf(x) ** (n - 1)
    return float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(x * pdf, x))


def probe_X(acc: np.ndarray, P: Sequence[int], maj: np.ndarray) -> np.ndarray:
    """探针特征：[acc[P] − mean(acc[P]), mean(acc[P]) − majority]（k + 1 维）。"""
    v = np.asarray(acc, float)[:, list(P)]
    m = v.mean(1, keepdims=True)
    return np.column_stack([v - m, m[:, 0] - np.asarray(maj, float)])


def _ridge_multi(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    m, s = Xtr.mean(0), Xtr.std(0)
    s = np.where(s > 1e-12, s, 1.0)
    A, B = (Xtr - m) / s, (Xte - m) / s
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return B @ W + ym


def _cv_err(X: np.ndarray, Y: np.ndarray, grp: np.ndarray, alpha: float) -> float:
    err = 0.0
    for s in np.unique(grp):
        te = grp == s
        err += float(np.sum((_ridge_multi(X[~te], Y[~te], X[te], alpha) - Y[te]) ** 2))
    return err


def greedy_order(acc: np.ndarray, A: np.ndarray, groups: np.ndarray, idx: np.ndarray, *, n_max: int = N_GREEDY,
                 alpha: float = GREEDY_ALPHA, maj: Optional[np.ndarray] = None) -> list[int]:
    """前向选择：只用 idx 行；每步加入使 探针特征 → A 的 ridge 组留一平方误差和下降最多的配置。"""
    idx = np.asarray(idx)
    acc_i, A_i, g_i = np.asarray(acc, float)[idx], np.asarray(A, float)[idx], np.asarray(groups)[idx]
    maj_i = np.zeros(len(idx)) if maj is None else np.asarray(maj, float)[idx]
    chosen: list[int] = []
    for _ in range(int(n_max)):
        best, best_e = None, np.inf
        for j in range(acc_i.shape[1]):
            if j in chosen:
                continue
            e = _cv_err(probe_X(acc_i, chosen + [j], maj_i), A_i, g_i, alpha)
            if e < best_e - 1e-12:
                best, best_e = j, e
        chosen.append(int(best))
    return chosen


def sh_select(order: Sequence[int], q25: np.ndarray, q50: np.ndarray, end0: np.ndarray, n0: int, n1: int, n2: int) -> tuple[int, dict]:
    """逐次减半（S2 _headroom_four 同规则）：前 n0 @25% → 按 q25 留 n1 @50% → 按 q50 留 n2 到终点 → 按 end0 选。返回 (pick, 每配置走到的截点分数)。"""
    c0 = [int(j) for j in list(order)[:n0]]
    c1 = sorted(c0, key=lambda j: -q25[j])[:n1]
    c2 = sorted(c1, key=lambda j: -q50[j])[:n2]
    reach = {j: 0.25 for j in c0}
    reach.update({j: 0.5 for j in c1})
    reach.update({j: 1.0 for j in c2})
    return int(max(c2, key=lambda j: end0[j])), reach


def plain_select(order: Sequence[int], end0: np.ndarray, k: int = 3) -> tuple[int, dict]:
    top = [int(j) for j in list(order)[:k]]
    return int(max(top, key=lambda j: end0[j])), {j: 1.0 for j in top}


def task_cost(reach: Mapping[int, float], probes: Sequence[int], probe_frac: float) -> float:
    """轨迹续跑累计：每配置按其走到的最远截点分数计，同一配置不重复计。"""
    r = {int(j): float(probe_frac) for j in probes}
    for j, f in reach.items():
        r[int(j)] = max(r.get(int(j), 0.0), float(f))
    return float(sum(r.values()))


def mk_order_a(acc: np.ndarray, P: Sequence[int], mu: np.ndarray) -> list[int]:
    """Mk-a：探针里 acc 最好 1 条 + μ 序其余。"""
    best = int(max(P, key=lambda j: acc[j]))
    return [best] + [int(j) for j in np.argsort(-np.asarray(mu), kind="stable") if int(j) != best]


def mk_order_b(acc: np.ndarray, P: Sequence[int], mu: np.ndarray) -> list[int]:
    """Mk-b：探针按 acc 降序，再接 μ 序其余。"""
    ps = sorted([int(j) for j in P], key=lambda j: -acc[j])
    return ps + [int(j) for j in np.argsort(-np.asarray(mu), kind="stable") if int(j) not in ps]


def dominated(arm: tuple[float, float], baselines: Sequence[tuple[float, float]]) -> bool:
    c, r = arm
    return any(bc <= c + 1e-12 and br <= r + 1e-12 for bc, br in baselines)


def verdict_probe(rows: Sequence[Mapping]) -> str:
    ok = [r for r in rows if r["rho"] >= 0.70 and r["undominated"]]
    if not ok:
        return "PROBE-DOMINATED"
    r = min(ok, key=lambda x: (x["cost_rank"], x["k"]))
    code = "PROBE-K" + str(r["label"])[1:]
    return code + ("+MODEL" if r["model_delta"] > 0 and r["model_p"] < 0.05 else "")


def verdict_space(S: float, S_mu: float, ub: float) -> str:
    if S >= 0.50:
        return "SPACE-OK" if ub <= 0.005 else "SPACE-RANK-ONLY"
    return "SPACE-MU-FAIL" if S_mu < 0.50 else "SPACE-B-FAIL"


def verdict_headroom(enr: float, p: float) -> str:
    if enr >= 2.0:
        return "HEADROOM-NEG" if p < 0.05 else "HEADROOM-NEG-WEAK"
    return "HEADROOM-UNIFORM"


def o_adj(end0: np.ndarray, end1: np.ndarray, R0: np.ndarray, sigma: float) -> float:
    o_cat = float(end1[int(np.argmax(end0))])
    R0 = np.asarray(R0, float)
    return max(o_cat, float(R0.max()) - expected_max(len(R0)) * sigma) if len(R0) else o_cat


def regret_30_50(pred_sh: np.ndarray, pred_rd: np.ndarray, end0: np.ndarray, end1: np.ndarray, R0: np.ndarray, sigma: float, *, k: int = 3) -> tuple[float, float]:
    """共同分母 o_adj。30 候选：shared 前 k 按 end0 选、end1 评。50 候选：union 前 k 按 seed-0 值选；random 终选按 R0 − κ_kσ 计值。"""
    o = o_adj(end0, end1, R0, sigma)
    top = np.argsort(-np.asarray(pred_sh), kind="stable")[:k]
    r30 = o - float(end1[int(top[int(np.argmax(end0[top]))])])
    n_sh = len(pred_sh)
    allp = np.concatenate([np.asarray(pred_sh, float), np.asarray(pred_rd, float)])
    top = np.argsort(-allp, kind="stable")[:k]
    s0 = np.concatenate([np.asarray(end0, float), np.asarray(R0, float)])
    j = int(top[int(np.argmax(s0[top]))])
    val = float(end1[j]) if j < n_sh else float(R0[j - n_sh]) - expected_max(k) * sigma
    return float(r30), float(o - val)


def task_spearman(pred: np.ndarray, actual: np.ndarray, *, min_n: int = 10, top_half: bool = False) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    if len(pred) < min_n:
        return float("nan")
    if top_half:
        keep = actual >= np.median(actual)
        pred, actual = pred[keep], actual[keep]
    if np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    return float(spearmanr(pred, actual)[0])


def enrichment(E: np.ndarray, neg: np.ndarray) -> float:
    E, neg = np.asarray(E, bool), np.asarray(neg, bool)
    base = E.mean()
    return float(E[neg].mean() / base) if base > 0 and neg.any() else float("nan")


def perm_test_sources(frac: Mapping[str, float], neg: Sequence[str]) -> float:
    """源级精确置换（单侧）：统计量 = 负端源平均份额 − 其余源平均份额；枚举全部 C(n, |neg|) 分组。"""
    names = list(frac)
    v = np.array([frac[s] for s in names], float)
    k = len(neg)
    obs_mask = np.array([s in set(neg) for s in names])
    def stat(mask):
        return v[mask].mean() - v[~mask].mean()
    obs = stat(obs_mask)
    cnt = tot = 0
    for comb in itertools.combinations(range(len(names)), k):
        m = np.zeros(len(names), bool); m[list(comb)] = True
        tot += 1
        cnt += stat(m) >= obs - 1e-12
    return cnt / tot


# ================================================================ 数据装配
def load_all() -> dict:
    """同 84.load_s2 的装配 + random 臂（R0 / z / φ）、@50%、σ、全 184 任务（C0 用）。"""
    ac = importlib.import_module("scripts.acts_common")
    g83 = importlib.import_module("scripts.83_acts_corpus")
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    cfg = ac.load_acts_config(profile="s2")
    plan = json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8"))
    cat = json.loads((S2_DIR / "catalog.json").read_text(encoding="utf-8"))
    an = json.loads((S2_DIR / "analysis_S.json").read_text(encoding="utf-8"))
    rows = g83._g35.load_done(S2_DIR / "runs.jsonl")
    meta = plan["tasks_meta"]
    q = float(cfg["scale"]["clip_quantile"])
    frames = g83._frames_c(plan, rows, cat, cfg)
    raw = g83._raw_seed_frames(plan, rows, cat, cfg)
    floors = g83.floors_cv(g83.load_probe0(S2_DIR), meta)
    n_sh = len(cat["shared"]); keys = g83.cat_keys(cat)
    enc = CLSEncoder(ac.acts_space(cfg), mode="binned")
    phi_sh = np.array([enc.encode(CLS(**s["cls"])) for s in cat["shared"]], float)
    rcls = {tid: {x["cls_key"]: x["cls"] for x in lst} for tid, lst in cat["random"].items()}

    def block(tasks):
        Z = np.array([ac.clip_z(frames[t]["frame_vals"], q)[0][:n_sh] for t in tasks])
        M = g83._fill_pairs({"end_0": np.array([raw[t][0] for t in tasks]), "end_1": np.array([raw[t][1] for t in tasks])})
        acc25 = g83._fill_single(np.array([[g83._ckpt(rows, plan, t, k, 0, 0) for k in keys] for t in tasks]), M["end_0"])
        acc50 = g83._fill_single(np.array([[g83._ckpt(rows, plan, t, k, 0, 1) for k in keys] for t in tasks]), M["end_0"])
        maj = np.array([float(meta[t]["majority"]) for t in tasks])
        pfloor = np.array([float(np.mean(floors[t]["probe_accs"])) if floors[t]["probe_accs"] else float(meta[t]["majority"]) for t in tasks])
        rnd = []
        for t in tasks:
            f = frames[t]
            zf = ac.clip_z(f["frame_vals"], q)[0]
            rk = f["random_keys"]
            rnd.append({"R0": np.asarray(f["frame_vals"], float)[n_sh:], "Z": zf[n_sh:],
                        "phi": np.array([enc.encode(CLS(**rcls[t][k])) for k in rk], float).reshape(len(rk), enc.dim)})
        sigma = np.std(M["end_0"] - M["end_1"], axis=1) / np.sqrt(2.0)
        return {"tasks": tasks, "groups": np.array([meta[t]["dataset"] for t in tasks]), "Z": Z, "end0": M["end_0"], "end1": M["end_1"],
                "acc25": acc25, "acc50": acc50, "Z25": np.array([ac.clip_z(v, q)[0] for v in acc25]), "maj": maj, "pfloor": pfloor,
                "rand": rnd, "sigma": sigma}

    pool = g84.pool_tasks(plan["probe_tasks"], meta, an["pool_after_gates"], an["reverse_demoted_to_8"])
    return {"pool": block(pool), "all": block(plan["probe_tasks"]), "phi_sh": phi_sh, "reverse": list(cfg["sources"]["reverse"]),
            "s2_adj_frac": float(an["headroom"]["oracle_ext"]["adj_frac"])}


def orient_bref(D: Mapping) -> np.ndarray:
    groups = D["groups"]
    b_ref = g84.lowrank_fit(D["Z"], 3)[1][:, 0]
    A0 = g84.project(D["Z"], D["Z"].mean(0), b_ref[:, None])[:, 0]
    if np.mean(A0[groups == "SemgHandMovementCh2"]) > 0:
        b_ref = -b_ref
    return b_ref


def _fold_list(groups: np.ndarray) -> list[str]:
    return list(dict.fromkeys(groups.tolist()))


def _ci(x, groups):
    return g84.cluster_bootstrap_ci(np.asarray(x, float), groups)


# ================================================================ 任务 3
def task3(D: Mapping, b_ref: np.ndarray) -> dict:
    groups = D["groups"]
    cache: dict = {}

    def greedy_feat(acc, n, ck):
        def f(ctx):
            key = (ck, tuple(int(i) for i in ctx["train_idx"]))
            if key not in cache:
                cache[key] = greedy_order(acc, ctx["A"][:, :3], groups, ctx["train_idx"], n_max=N_GREEDY if ck == 25 else 4, alpha=GREEDY_ALPHA, maj=D["maj"])
            return probe_X(acc, cache[key][:n], D["maj"])
        return f

    specs = {f"k{n}": (D["acc25"], n, 25, 0.25) for n in K_LIST}
    specs["k4@50"] = (D["acc50"], 4, 50, 0.5)
    feats = {lab: greedy_feat(acc, n, ck) for lab, (acc, n, ck, _fr) in specs.items()}
    feats["hand4"] = g84.probe_feature(D["acc25"], D["pfloor"], D["maj"])
    feats["land25"] = g84.land25_feature(D["Z25"])
    res = g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], feats, b_ref=b_ref)
    folds = _fold_list(groups)
    n = len(groups)
    q25, q50, e0, e1 = D["acc25"], D["acc50"], D["end0"], D["end1"]
    arms: dict[str, dict] = {}

    def add(name, i, pick, cost):
        a = arms.setdefault(name, {"regret3": np.zeros(n), "cost": np.zeros(n)})
        a["regret3"][i] = e1[i, int(np.argmax(e0[i]))] - e1[i, pick]
        a["cost"][i] = cost

    probe_sets: dict[str, dict] = {}
    for fi, s in enumerate(folds):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        mu_order = [int(j) for j in np.argsort(-ctx["mu"], kind="stable")]
        mr = np.array([rankdata(-e0[j]) for j in tr]).mean(0)
        pr_order = [int(j) for j in np.argsort(mr, kind="stable")]
        for i in te:
            # 基线
            p, r = plain_select(mu_order, e0[i]); add("blind_top3", i, p, task_cost(r, [], 0))
            for nm, od in (("mu", mu_order), ("prior", pr_order)):
                p, r = sh_select(od, q25[i], q50[i], e0[i], 6, 3, 3); add(f"SH6_{nm}", i, p, task_cost(r, [], 0))
                p, r = sh_select(od, q25[i], q50[i], e0[i], 10, 4, 3); add(f"SH10_{nm}", i, p, task_cost(r, [], 0))
            od = [int(j) for j in np.argsort(-q25[i], kind="stable")]
            p, r = plain_select(od, e0[i]); add("M25", i, p, task_cost(r, range(30), 0.25))
        for lab in list(specs) + ["hand4", "land25"]:
            K = res[lab]["k_chosen"][fi]
            S = g84.scores_from(ctx["mu"], ctx["B"][:, :K], res[lab]["a_hat"][te][:, :K])
            if lab == "land25":
                P, frac, acc = list(range(30)), 0.25, q25
            elif lab == "hand4":
                P, frac, acc = g84.probe_configs(ctx), 0.25, q25
            else:
                acc, nk, ck, frac = specs[lab]
                P = cache[(ck, tuple(int(x) for x in tr))][:nk]
            probe_sets.setdefault(lab, {})[s] = [int(x) for x in P]
            for ii, i in enumerate(te):
                od = [int(j) for j in np.argsort(-S[ii], kind="stable")]
                p, r = plain_select(od, e0[i]); add(f"{lab}|plain", i, p, task_cost(r, P, frac))
                if lab == "land25":
                    continue
                p, r = sh_select(od, q25[i], q50[i], e0[i], 6, 3, 3); add(f"{lab}|SH6", i, p, task_cost(r, P, frac))
                p, r = sh_select(od, q25[i], q50[i], e0[i], 10, 4, 3); add(f"{lab}|SH10", i, p, task_cost(r, P, frac))
                for mk, fn in (("Mk-a", mk_order_a), ("Mk-b", mk_order_b)):
                    if mk == "Mk-b" and len(P) < 3:
                        continue
                    p, r = plain_select(fn(acc[i], P, ctx["mu"]), e0[i]); add(f"{lab}|{mk}", i, p, task_cost(r, P, frac))
    summ = {k: {"regret3": float(v["regret3"].mean()), "cost": float(v["cost"].mean()), "ci": _ci(v["regret3"], groups)} for k, v in arms.items()}
    base_names = ["blind_top3", "SH6_mu", "SH10_mu", "SH6_prior", "SH10_prior", "M25", "land25|plain"]
    base = [(summ[b]["cost"], summ[b]["regret3"]) for b in base_names]
    rows = []
    for lab in list(specs) + ["hand4"]:
        frac = specs[lab][3] if lab in specs else 0.25
        k = specs[lab][1] if lab in specs else 4
        mks = [m for m in ("Mk-a", "Mk-b") if f"{lab}|{m}" in arms]
        mk = min(mks, key=lambda m: summ[f"{lab}|{m}"]["regret3"])
        d = arms[f"{lab}|{mk}"]["regret3"] - arms[f"{lab}|plain"]["regret3"]
        armv = {a: (summ[f"{lab}|{a}"]["cost"], summ[f"{lab}|{a}"]["regret3"]) for a in ("plain", "SH6", "SH10")}
        und = {a: not dominated(v, base) for a, v in armv.items()}
        # 与"成本不高于它的最贵基线"的源级配对差
        pair = {}
        for a, (c, _r) in armv.items():
            cands = [b for b in base_names if summ[b]["cost"] <= c + 1e-12]
            if cands:
                bn = max(cands, key=lambda b: summ[b]["cost"])
                dd = arms[bn]["regret3"] - arms[f"{lab}|{a}"]["regret3"]
                pair[a] = {"vs": bn, "delta": float(dd.mean()), "ci": _ci(dd, groups), "p": g84.source_wilcoxon(dd, groups)}
        rows.append({"label": lab, "k": k, "cost_rank": k * frac, "rho": res[lab]["rho_a1"], "rho_ci": g84.cluster_bootstrap_ci(
                        np.column_stack([res[lab]["a_hat"][:, 0], res[lab]["a_true"][:, 0]]), groups, stat=lambda x: float(spearmanr(x[:, 0], x[:, 1])[0]) if np.std(x[:, 0]) > 0 else 0.0),
                     "regret1": res[lab]["mean_regret1"], "regret3_loso": res[lab]["mean_regret3"],
                     "arms": {a: {"cost": armv[a][0], "regret3": armv[a][1], "undominated": und[a]} for a in armv},
                     "undominated": any(und.values()), "mk": mk, "mk_regret3": summ[f"{lab}|{mk}"]["regret3"],
                     "model_delta": float(d.mean()), "model_ci": _ci(d, groups), "model_p": g84.source_wilcoxon(d, groups), "pair_vs_base": pair,
                     "k_chosen": res[lab]["k_chosen"]})
    code = verdict_probe([r for r in rows if r["label"] != "hand4"])
    return {"verdict": code, "rows": rows, "arms": summ, "baselines": base_names, "probe_sets": probe_sets,
            "a_hat_hand4": res["hand4"]["a_hat"], "a_true": res["hand4"]["a_true"], "res": res,
            "c0": {"hand4_regret3": res["hand4"]["mean_regret3"], "hand4_rho": res["hand4"]["rho_a1"],
                   "SH6_prior": summ["SH6_prior"]["regret3"], "SH10_prior": summ["SH10_prior"]["regret3"]}}


# ================================================================ 任务 1
def _space_rows(D, phi_sh, idx, a1):
    """行 = idx 任务 × (shared-30 ∪ random)；返回 φ、a₁、z、任务号。"""
    P, A, Y, T = [], [], [], []
    for i in idx:
        r = D["rand"][i]
        ph = np.vstack([phi_sh, r["phi"]])
        P.append(ph); A.append(np.full(len(ph), a1[i])); Y.append(np.concatenate([D["Z"][i], r["Z"]])); T.append(np.full(len(ph), i))
    return np.vstack(P), np.concatenate(A), np.concatenate(Y), np.concatenate(T)


def _design(phi, a, kind):
    if kind == "mu":
        return phi
    if kind in ("L", "MLP"):
        return np.column_stack([phi, phi * a[:, None]])
    return np.column_stack([phi, a])          # GBDT


def _fit_space(kind, X, y, groups_rows):
    if kind in ("L", "mu"):
        a = g84.choose_alpha(X, y, groups_rows, SPACE_ALPHAS)
        m = g84.ridge_fit(X, y, a)
        return lambda Xn: g84.ridge_predict(m, Xn)
    if kind == "GBDT":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, y)
        return m.predict
    from sklearn.neural_network import MLPRegressor
    m = MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, random_state=0, max_iter=500).fit(X, y)
    return m.predict


def task1(D: Mapping, phi_sh: np.ndarray, b_ref: np.ndarray, a_hat_heldout: np.ndarray, a_true_heldout: np.ndarray) -> dict:
    groups = D["groups"]
    n = len(groups)
    kinds = ("L", "mu", "GBDT", "MLP")
    out = {f"{k}|{src}": {"sp": np.full(n, np.nan), "sp_top": np.full(n, np.nan), "r30": np.full(n, np.nan), "r50": np.full(n, np.nan),
                          "cal_rd": np.full(n, np.nan), "cal_sh": np.full(n, np.nan)}
           for k in kinds for src in ("hat", "true") if not (k == "mu" and src == "true")}
    for s in _fold_list(groups):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        a_tr = ctx["A"][:, 0]
        Ptr, Atr, Ytr, Ttr = _space_rows(D, phi_sh, tr, a_tr)
        grow = groups[Ttr]
        for kind in kinds:
            f = _fit_space(kind, _design(Ptr, Atr, kind), Ytr, grow)
            for src, av in (("hat", a_hat_heldout[:, 0]), ("true", a_true_heldout[:, 0])):
                key = f"{kind}|{src}"
                if key not in out:
                    continue
                for i in te:
                    r = D["rand"][i]
                    ph = np.vstack([phi_sh, r["phi"]])
                    pred = f(_design(ph, np.full(len(ph), av[i]), kind))
                    ps, pr = pred[:30], pred[30:]
                    o = out[key]
                    o["sp"][i] = task_spearman(pr, r["R0"], min_n=10)
                    o["sp_top"][i] = task_spearman(pr, r["R0"], min_n=10, top_half=True)
                    if len(pr):
                        o["r30"][i], o["r50"][i] = regret_30_50(ps, pr, D["end0"][i], D["end1"][i], r["R0"], float(D["sigma"][i]))
                        t3 = np.argsort(-pr, kind="stable")[:3]
                        o["cal_rd"][i] = float(np.mean(pr[t3] - r["Z"][t3]))
                    t3 = np.argsort(-ps, kind="stable")[:3]
                    o["cal_sh"][i] = float(np.mean(ps[t3] - D["Z"][i][t3]))
    summ = {}
    for key, o in out.items():
        ok = np.isfinite(o["sp"])
        d50 = o["r50"] - o["r30"]
        fin = np.isfinite(d50)
        summ[key] = {"sp_median": float(np.nanmedian(o["sp"])), "sp_mean": float(np.nanmean(o["sp"])), "sp_ci": _ci(o["sp"][ok], groups[ok]),
                     "sp_top_median": float(np.nanmedian(o["sp_top"])), "n_sp": int(ok.sum()),
                     "r30": float(np.nanmean(o["r30"])), "r50": float(np.nanmean(o["r50"])), "d50": float(np.nanmean(d50)), "d50_ci": _ci(d50[fin], groups[fin]),
                     "cal_rd": float(np.nanmean(o["cal_rd"])), "cal_sh": float(np.nanmean(o["cal_sh"]))}
    L, M = summ["L|hat"], summ["mu|hat"]
    code = verdict_space(L["sp_median"], M["sp_median"], L["d50_ci"][1])
    return {"verdict": code, "summary": summ}


# ================================================================ 任务 2
def ext_adj_vector(B: Mapping, sigma_override: Optional[float] = None) -> np.ndarray:
    out = []
    for i in range(len(B["groups"])):
        R0 = B["rand"][i]["R0"]
        sig = float(B["sigma"][i]) if sigma_override is None else float(sigma_override)
        o_cat = float(B["end1"][i, int(np.argmax(B["end0"][i]))])
        out.append(float(R0.max()) - expected_max(len(R0)) * sig - o_cat if len(R0) else -np.inf)
    return np.array(out)


def task2(D: Mapping, reverse: Sequence[str], b_ref: np.ndarray) -> dict:
    groups = D["groups"]
    ext = ext_adj_vector(D)
    sig_med = float(np.median(D["sigma"]))
    ext_s = ext_adj_vector(D, sigma_override=sig_med)
    neg = np.isin(groups, list(reverse))
    a1 = g84.project(D["Z"], D["Z"].mean(0), b_ref[:, None])[:, 0]
    srcs = _fold_list(groups)
    res = {}
    for nm, E in (("E0", ext > 0), ("E2", ext >= 2 * D["sigma"]), ("E0_sigmed", ext_s > 0)):
        frac = {s: float(E[groups == s].mean()) for s in srcs}
        enr = enrichment(E, neg)
        res[nm] = {"share": float(E.mean()), "p_neg": float(E[neg].mean()), "p_pos": float(E[~neg].mean()), "enrichment": enr,
                   "enr_ci": g84.cluster_bootstrap_ci(np.column_stack([E, neg]).astype(float), groups,
                                                     stat=lambda x: enrichment(x[:, 0] > 0.5, x[:, 1] > 0.5) if (x[:, 0] > 0.5).any() and (x[:, 1] > 0.5).any() else 0.0),
                   "perm_p": perm_test_sources(frac, [s for s in srcs if s in reverse]), "per_source": frac}
    sm_a = [float(a1[groups == s].mean()) for s in srcs]; sm_e = [float(ext[groups == s].mean()) for s in srcs]
    out = {"verdict": verdict_headroom(res["E0"]["enrichment"], res["E0"]["perm_p"]), "sets": res, "ceiling": float(len(groups) / neg.sum()),
           "rho_task": float(spearmanr(a1, ext)[0]), "rho_src": float(spearmanr(sm_a, sm_e)[0]),
           "gain": {g: {"mean_ext": float(ext[m].mean()), "mean_pos_ext": float(np.maximum(0, ext[m]).mean()), "sigma_median": float(np.median(D["sigma"][m]))}
                    for g, m in (("neg", neg), ("pos", ~neg), ("all", np.ones(len(groups), bool)))}}
    return out


# ================================================================ 汇总
def _f(x, nd=3):
    try:
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def cmd_analyze(outdir: Path = OUT) -> dict:
    data = load_all()
    D = data["pool"]
    b_ref = orient_bref(D)
    # C0：任务 2 在 184 任务上的 E0 份额 = S2 adj_frac
    ext_all = ext_adj_vector(data["all"])
    c0_t2 = float(np.mean(ext_all > 0))
    t3 = task3(D, b_ref)
    s3a = json.loads((ROOT / "results/acts/s3a/analysis.json").read_text(encoding="utf-8"))
    ph = json.loads((ROOT / "results/acts/s3a/posthoc_sh.json").read_text(encoding="utf-8"))
    c0 = {"hand4_regret3": (t3["c0"]["hand4_regret3"], s3a["summary"]["probe"]["regret3"]), "hand4_rho": (t3["c0"]["hand4_rho"], s3a["summary"]["probe"]["rho_a1"]),
          "SH6_prior": (t3["c0"]["SH6_prior"], ph["sh633"]), "SH10_prior": (t3["c0"]["SH10_prior"], ph["sh1043"]), "E0_share_184": (c0_t2, data["s2_adj_frac"])}
    c0_ok = all(abs(a - b) < 1e-9 for a, b in c0.values())
    t1 = task1(D, data["phi_sh"], b_ref, t3["a_hat_hand4"], t3["a_true"])
    t2 = task2(D, data["reverse"], b_ref)
    out = {"c0": {k: list(v) for k, v in c0.items()}, "c0_ok": c0_ok, "task3": {k: v for k, v in t3.items() if k not in ("res", "a_hat_hand4", "a_true")},
           "task1": t1, "task2": t2, "codes": {"task3": t3["verdict"], "task1": t1["verdict"], "task2": t2["verdict"]}}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return out


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-A′ 读数：任务 3 {o['codes']['task3']} ｜ 任务 1 {o['codes']['task1']} ｜ 任务 2 {o['codes']['task2']}", "",
         f"C0 {'✓' if o['c0_ok'] else '✗'}：" + "；".join(f"{k} {_f(a, 4)} vs {_f(b, 4)}" for k, (a, b) in o["c0"].items()), "",
         "## 表 1 任务 3：探针数曲线（regret@3，30 候选；成本 = 轨迹续跑累计全跑当量）", "",
         "| 探针 | ρ_a1 [CI] | plain 成本 / regret@3 | SH6 叠加 | SH10 叠加 | 不被支配 | Mk（同成本测量） | Δ_model [CI] p | K=1/2/3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in o["task3"]["rows"]:
        a = r["arms"]; kc = [r["k_chosen"].count(K) for K in (1, 2, 3)]
        L.append(f"| {r['label']} | {_f(r['rho'], 2)} [{_f(r['rho_ci'][0], 2)}, {_f(r['rho_ci'][1], 2)}] | {_f(a['plain']['cost'], 2)} / {_f(a['plain']['regret3'])} | "
                 f"{_f(a['SH6']['cost'], 2)} / {_f(a['SH6']['regret3'])} | {_f(a['SH10']['cost'], 2)} / {_f(a['SH10']['regret3'])} | "
                 f"{'/'.join(n for n, v in a.items() if v['undominated']) or '—'} | {r['mk']} {_f(r['mk_regret3'])} | {_f(r['model_delta'])} [{_f(r['model_ci'][0])}, {_f(r['model_ci'][1])}] {_f(r['model_p'], 3)} | {kc[0]}/{kc[1]}/{kc[2]} |")
    L += ["", "| 基线 | 成本 | regret@3 [CI] |", "|---|---:|---|"]
    for b in o["task3"]["baselines"]:
        v = o["task3"]["arms"][b]
        L.append(f"| {b} | {_f(v['cost'], 2)} | {_f(v['regret3'])} [{_f(v['ci'][0])}, {_f(v['ci'][1])}] |")
    L += ["", "与成本不高于它的最贵基线的配对差（正 = 探针臂更好）："]
    for r in o["task3"]["rows"]:
        L.append(f"- {r['label']}：" + "；".join(f"{a} vs {v['vs']} {_f(v['delta'])} [{_f(v['ci'][0])}, {_f(v['ci'][1])}] p {_f(v['p'], 3)}" for a, v in r["pair_vs_base"].items()))
    L += ["", "## 表 2 任务 1：配置侧归纳（留出 random 上的 Spearman 用原始 R0；regret 共同分母 o_adj）", "",
          "| 臂 | 中位 Spearman | 均值 [CI] | 前一半中位 | regret@3 30 候选 | 50 候选 | Δ50 [CI] | 校准 random / shared |", "|---|---:|---|---:|---:|---:|---|---|"]
    for k, v in o["task1"]["summary"].items():
        L.append(f"| {k} | {_f(v['sp_median'], 2)} | {_f(v['sp_mean'], 2)} [{_f(v['sp_ci'][0], 2)}, {_f(v['sp_ci'][1], 2)}] | {_f(v['sp_top_median'], 2)} | {_f(v['r30'])} | {_f(v['r50'])} | "
                 f"{_f(v['d50'])} [{_f(v['d50_ci'][0])}, {_f(v['d50_ci'][1])}] | {_f(v['cal_rd'], 2)} / {_f(v['cal_sh'], 2)} |")
    t2 = o["task2"]
    L += ["", "## 表 3 任务 2：名单外净空归属（负端 = S2 反向四源；富集比上限 " + _f(t2["ceiling"], 2) + "）", "",
          "| 集合 | 全池份额 | 负端 | 正端 | 富集比 [CI] | 源级置换 p |", "|---|---:|---:|---:|---|---:|"]
    for k, v in t2["sets"].items():
        L.append(f"| {k} | {_f(v['share'], 3)} | {_f(v['p_neg'], 3)} | {_f(v['p_pos'], 3)} | {_f(v['enrichment'], 2)} [{_f(v['enr_ci'][0], 2)}, {_f(v['enr_ci'][1], 2)}] | {_f(v['perm_p'], 3)} |")
    L += ["", f"Spearman(a₁, ext_adj)：任务级 {_f(t2['rho_task'], 2)}、源级 {_f(t2['rho_src'], 2)}；" +
          "；".join(f"{g} mean ext {_f(v['mean_ext'])} / mean max(0,ext) {_f(v['mean_pos_ext'])} / σ 中位 {_f(v['sigma_median'])}" for g, v in t2["gain"].items()),
          "", "每源 E0 份额：" + "，".join(f"{s} {_f(v, 2)}" for s, v in t2["sets"]["E0"]["per_source"].items())]
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
