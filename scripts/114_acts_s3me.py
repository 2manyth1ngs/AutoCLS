"""WP17-ACTS S3-ME（results/acts/s3me/PROTOCOL.md）：源专家混合（E 只做相似度）——零 GPU 线性版。

s(c) = s₀(c) + γ · Σ_j π_j · w_j·x_c；s₀ = 全训练任务岭回归，w_j = 源 j 残差岭回归，π = softmax(E 相似度 / τ)；
(α_w, τ, γ) 在训练源内层留一源按 Spearman 选。17 源留一源评估 M1 regret@3 / M2 全配置 Spearman / M3 random-20 Spearman / M4 残差 Spearman。
用法：--run（本地，约数分钟）
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

OUT = ROOT / "results" / "acts" / "s3me"
S3C = ROOT / "results" / "acts" / "s3c"
N_SH = 30
ALPHA0 = 10.0
ALPHA_W = (1.0, 10.0, 100.0)
TAUS = (0.05, 0.1, 0.2, 0.5, 1.0)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("BLIND", "UNIFORM", "ME_pt", "ME_A", "ME_ptA", "ORACLE")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def softmax_weights(sims: np.ndarray, tau: float) -> np.ndarray:
    s = np.asarray(sims, float) / float(tau)
    e = np.exp(s - s.max())
    return e / e.sum()


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """带截距岭回归（中心化后闭式解）。"""
    X = np.asarray(X, float); y = np.asarray(y, float)
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + float(alpha) * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - xm @ w)


def mixture_predict(X: np.ndarray, base: tuple, experts: tuple, pi: np.ndarray, gamma: float) -> np.ndarray:
    (b0, c0), (W, cw) = base, experts
    s = np.asarray(X, float) @ b0 + c0
    if gamma == 0:
        return s
    return s + float(gamma) * (np.asarray(X, float) @ (np.asarray(pi) @ W) + float(np.asarray(pi) @ cw))


def source_E(E: np.ndarray, groups: np.ndarray, sources: Sequence[str]) -> np.ndarray:
    return np.stack([np.asarray(E, float)[np.asarray(groups) == s].mean(0) for s in sources])


def select_grid(scores: Mapping[tuple, float]) -> tuple:
    """(α_w, τ, γ) → 分数；取最高，平局依次取较小 γ、较大 τ、较大 α_w。"""
    return max(scores, key=lambda k: (round(float(scores[k]), 12), -k[2], k[1], k[0]))


def verdict_s3me(me: Mapping[str, float], uni: Mapping[str, float], blind: Mapping[str, float]) -> str:
    safe = me["M1"] <= blind["M1"] + 0.002 and me["M2"] >= blind["M2"] - 0.005
    e_ok = me["M2"] > uni["M2"] and me["M4"] > uni["M4"]
    if not safe:
        return "S3ME-HURT"
    if not e_ok:
        return "S3ME-E-NULL"
    return "S3ME-GO" if me["M1"] < blind["M1"] else "S3ME-SAFE"


def _cos_rows(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)


# ================================================================ 运行（本地）
def load_all() -> dict:
    g89 = _g("89_acts_s3c")
    meta, arr = g89.load_data(S3C); tasks = meta["tasks"]; groups = np.array(meta["groups"])
    L = np.load(ROOT / "results" / "acts" / "s3r" / "labels.npz")
    X = [np.asarray(arr[f"{t}/vec"], float) for t in tasks]
    Z = []
    for t in tasks:
        r = np.asarray(L[f"{t}/r"], float); Z.append((r - r.mean()) / (r.std() + 1e-9))
    with np.load(ROOT / "results" / "acts" / "s3b" / "E_pt" / "_means.npz", allow_pickle=True) as z:
        d = dict(zip([str(t) for t in z["tasks"]], z["E_mean"]))
    Ept = np.array([d[t] for t in tasks], float)
    with np.load(ROOT / "results" / "acts" / "s3n" / "ea.npz", allow_pickle=True) as z:
        d = dict(zip([str(t) for t in z["tasks"]], z["EA"]))
    EA = np.array([d[t] for t in tasks], float)
    return {"tasks": tasks, "groups": groups, "X": X, "Z": Z, "Ept": Ept, "EA": EA, "end0": arr["end0"], "end1": arr["end1"],
            "R0": [np.asarray(arr[f"{t}/R0"], float) for t in tasks], "order": list(dict.fromkeys(groups.tolist()))}


def _std(E: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
    return (E - mu) / sd


def _fit(data: Mapping, train_src: Sequence[str]) -> tuple:
    """s₀ 与各源专家（每个 α_w 一套）。"""
    g = data["groups"]; idx = [i for i in range(len(g)) if g[i] in set(train_src)]
    Xa = np.vstack([data["X"][i] for i in idx]); Za = np.concatenate([data["Z"][i] for i in idx])
    b0, c0 = ridge_fit(Xa, Za, ALPHA0)
    experts = {}
    for aw in ALPHA_W:
        W, cw = [], []
        for s in train_src:
            ii = [i for i in idx if g[i] == s]
            Xs = np.vstack([data["X"][i] for i in ii]); Rs = np.concatenate([data["Z"][i] - (data["X"][i] @ b0 + c0) for i in ii])
            w, c = ridge_fit(Xs, Rs, aw); W.append(w); cw.append(c)
        experts[aw] = (np.array(W), np.array(cw))
    return (b0, c0), experts


def _sims(data: Mapping, arm: str, i: int, train_src: Sequence[str], Es: Mapping[str, np.ndarray], base: tuple) -> Optional[np.ndarray]:
    if arm == "UNIFORM":
        return np.zeros(len(train_src))
    if arm == "ORACLE":
        from scipy.stats import spearmanr
        g = data["groups"]; b0, c0 = base
        rt = data["Z"][i][:N_SH] - (data["X"][i][:N_SH] @ b0 + c0)
        out = []
        for s in train_src:
            rs = np.mean([data["Z"][j][:N_SH] - (data["X"][j][:N_SH] @ b0 + c0) for j in range(len(g)) if g[j] == s], axis=0)
            out.append(spearmanr(rt, rs).correlation)
        return np.nan_to_num(np.array(out))
    E = Es[arm]; g = data["groups"]
    return _cos_rows(E[i], source_E(E, g, train_src))


def _metrics(data: Mapping, i: int, pred: np.ndarray, base_pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    g84 = _g("84_acts_s3a")
    z = data["Z"][i]
    m1 = float(g84.regret_at_k(pred[:N_SH][None], data["end0"][[i]], data["end1"][[i]], 3)[0])
    m2 = float(spearmanr(pred, z).correlation)
    m3 = float(spearmanr(pred[N_SH:], data["R0"][i]).correlation) if len(data["R0"][i]) > 2 else np.nan
    d = pred - base_pred
    m4 = float(spearmanr(d, z - base_pred).correlation) if np.std(d) > 1e-12 else np.nan
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4}


def _E_sets(data: Mapping, tr_mask: np.ndarray) -> dict:
    Ept, EA = _std(data["Ept"], tr_mask), _std(data["EA"], tr_mask)
    return {"ME_pt": Ept, "ME_A": EA, "ME_ptA": np.hstack([Ept, EA])}


def cmd_run(args) -> dict:
    from scipy.stats import spearmanr
    data = load_all(); g = data["groups"]; order = data["order"]; outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(ALPHA_W, TAUS, GAMMAS))
    res = {a: {} for a in ARMS}; chosen = {a: {} for a in ARMS}
    for h in order:
        train_src = [s for s in order if s != h]
        Es = _E_sets(data, g != h)
        # ---- 内层留一源选超参
        inner = {a: {k: [] for k in grid} for a in ARMS if a != "BLIND"}
        for ih in train_src:
            src_in = [s for s in train_src if s != ih]
            base, experts = _fit(data, src_in)
            for i in np.where(g == ih)[0]:
                bp = data["X"][i] @ base[0] + base[1]
                for a in inner:
                    sims = _sims(data, a, i, src_in, Es, base)
                    for aw, tau, gam in grid:
                        pred = mixture_predict(data["X"][i], base, experts[aw], softmax_weights(sims, tau), gam)
                        inner[a][(aw, tau, gam)].append(spearmanr(pred, data["Z"][i]).correlation if gam > 0 else spearmanr(bp, data["Z"][i]).correlation)
        base, experts = _fit(data, train_src)
        for a in ARMS:
            k = (ALPHA_W[1], TAUS[0], 0.0) if a == "BLIND" else select_grid({kk: float(np.nanmean(v)) for kk, v in inner[a].items()})
            chosen[a][h] = k
            for i in np.where(g == h)[0]:
                bp = data["X"][i] @ base[0] + base[1]
                sims = _sims(data, a, i, train_src, Es, base) if a != "BLIND" else np.zeros(len(train_src))
                pred = mixture_predict(data["X"][i], base, experts[k[0]], softmax_weights(sims, k[1]), k[2])
                res[a][int(i)] = _metrics(data, int(i), pred, bp)
        print(f"[s3me] {h} 完成；选中 γ：" + "，".join(f"{a} {chosen[a][h][2]:g}" for a in ARMS if a != "BLIND"), flush=True)
    # ---- 汇总
    g84 = _g("84_acts_s3a")
    idx = sorted(res["BLIND"]); src = g[idx]
    M = {a: {m: float(np.nanmean([res[a][i][m] for i in idx])) for m in ("M1", "M2", "M3", "M4")} for a in ARMS}
    ci = {}
    for a in ("ME_ptA", "ME_pt", "ME_A", "UNIFORM", "ORACLE"):
        for ref in ("BLIND", "UNIFORM"):
            if a == ref:
                continue
            for m in ("M1", "M2"):
                dlt = np.array([res[a][i][m] - res[ref][i][m] for i in idx])
                ci[f"{a}-{ref}:{m}"] = {"d": float(np.nanmean(dlt)), "ci": g84.cluster_bootstrap_ci(np.nan_to_num(dlt), src, n_boot=5000),
                                         "src_pos": int(sum(np.nanmean(dlt[src == s]) > 0 for s in order))}
    verdict = verdict_s3me(M["ME_ptA"], M["UNIFORM"], M["BLIND"])
    preds = {"P1": M["ME_ptA"]["M2"] > M["BLIND"]["M2"], "P2": M["ME_ptA"]["M1"] < M["BLIND"]["M1"], "P3": M["ME_ptA"]["M2"] > M["UNIFORM"]["M2"],
             "P4": M["ORACLE"]["M2"] - M["BLIND"]["M2"] >= 0.05, "P5": M["ME_ptA"]["M2"] >= max(M["ME_pt"]["M2"], M["ME_A"]["M2"]),
             "P6": sum(chosen["ME_ptA"][h][2] > 0 for h in order) >= 9}
    per_src = {a: {s: {m: float(np.nanmean([res[a][i][m] for i in idx if g[i] == s])) for m in ("M1", "M2")} for s in order} for a in ARMS}
    o = {"means": M, "ci": ci, "verdict": verdict, "predictions": preds, "chosen": {a: {h: list(k) for h, k in v.items()} for a, v in chosen.items()},
         "per_source": per_src, "n_tasks": len(idx)}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    Lr = ["# S3-ME 读数（17 源留一源，160 任务；线性源专家混合）", "", "| 臂 | M1 regret@3 | M2 全配置 Spearman | M3 random-20 Spearman | M4 残差 Spearman |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        Lr.append(f"| {a} | {M[a]['M1']:.4f} | {M[a]['M2']:.3f} | {M[a]['M3']:.3f} | {M[a]['M4']:+.3f} |")
    Lr += ["", "| 对比 | Δ 均值 | 源级 95% CI | 源为正数 |", "|---|---:|---|---:|"]
    for k, v in ci.items():
        Lr.append(f"| {k} | {v['d']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['src_pos']}/17 |")
    Lr += ["", f"读数码：**{verdict}**；预测 {sum(preds.values())}/6",
           "", "内层选中 γ（ME_ptA 逐折）：" + "，".join(f"{h[:6]} {chosen['ME_ptA'][h][2]:g}" for h in order)]
    (outdir / "tables.md").write_text("\n".join(Lr) + "\n", encoding="utf-8")
    print("\n".join(Lr), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run:
        cmd_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
