"""WP16 GEO-E 检验：E_T 能否预测目标的损失几何家族（loss_type × sim_func × norm）。协议 results/tclsc_geo/PROTOCOL.md。零 GPU、只读既有数据。

  python scripts/81_tclsc_geo.py --run     # → analysis.json / tables.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.mech_features import rep_cols_at  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
OUT = ROOT / "results" / "tclsc_geo"
SRC_RUNS = (ROOT / "results" / "uea30" / "runs.jsonl", ROOT / "results" / "mx4_deploy" / "runs.jsonl")
SRC_ET = (ROOT / "results" / "tclsc_s1et_deploy" / "et_official.jsonl", ROOT / "results" / "tclsc_s1et_uea30" / "et_uea30.jsonl")
ET60, ET290 = tuple(rep_cols_at(60)), tuple(rep_cols_at(290))
PHI = tuple(c for c in _m33.PHI_COLS if c not in ("base_mean", "base_band"))     # 官方靶无 iters0 基线：φ 去掉两列（23 靶可用；六靶无 φ）
ARMS = {"et60": ET60, "et290": ET290, "phi": PHI, "et60+phi": ET60 + PHI}
PRIMARY = "et60"
PRIOR_TOP1_FAMILY = ("triplet", "dot", "none")          # 9092e36eefe1 的家族
MIN_CONFIGS = 10
ALPHAS = (0.1, 1.0, 10.0)
G1_GAIN, G2_MARGIN, ALPHA_P = 0.005, 0.10, 0.05
EEG_LIKE = {"SelfRegulationSCP1", "SelfRegulationSCP2", "HandMovementDirection", "MotorImagery", "FaceDetection", "PEMS-SF", "FingerMovements", "EthanolConcentration"}
SEED = 0


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


# ---------------------------------------------------------------- 标签
def family_of(cls: Mapping) -> tuple:
    return (cls["loss_type"], cls["sim_func"], cls["norm"])


def catalog_rows() -> list[dict]:
    rows = []
    for p in SRC_RUNS:
        rows += [r for r in read(p) if r["stage"] == "s1_catalog" and r["seed"] == 0 and r.get("acc") is not None and r["arm"] in ("shared", "ggs")
                 and r["cls"].get("input_mask") in (None, "none")]
    cnt = Counter(r["dataset"] for r in rows)
    return [r for r in rows if cnt[r["dataset"]] >= MIN_CONFIGS]


def utility_table(rows: Sequence[dict]) -> tuple[np.ndarray, list[str], list[tuple]]:
    """U[d, f] = 家族 f 最好 acc − 靶 d 最好 acc（≤ 0；目录里没有的家族 = nan）。"""
    best: dict[str, dict[tuple, float]] = defaultdict(dict)
    for r in rows:
        f = family_of(r["cls"])
        best[r["dataset"]][f] = max(best[r["dataset"]].get(f, -np.inf), float(r["acc"]))
    ds = sorted(best)
    fams = sorted({f for d in best.values() for f in d})
    U = np.full((len(ds), len(fams)), np.nan)
    for i, d in enumerate(ds):
        top = max(best[d].values())
        for j, f in enumerate(fams):
            if f in best[d]:
                U[i, j] = best[d][f] - top
    return U, ds, fams


def champion(U: np.ndarray, fams: Sequence[tuple]) -> list[tuple]:
    return [fams[int(np.nanargmax(row))] for row in U]


def fixed_best(U: np.ndarray, *, train_idx: Sequence[int]) -> int:
    """训练折上均值 u 最高的家族（nan 视为该靶最差家族的值，保守）。"""
    sub = U[list(train_idx)]
    filled = np.where(np.isnan(sub), np.nanmin(sub, axis=1, keepdims=True), sub)
    return int(np.argmax(filled.mean(axis=0)))


def noise_by_dataset() -> dict[str, float]:
    sig = {}
    for p in SRC_RUNS:
        rows = read(p)
        s1 = {(r["dataset"], r["cls_key"]): float(r["acc"]) for r in rows if r["stage"] == "s1_catalog" and r["seed"] == 0 and r.get("acc") is not None}
        fin = defaultdict(list)
        for r in rows:
            if r["stage"] == "s2_finals" and r.get("acc") is not None and r["arm"].startswith("final_SH") and "prior" in r["arm"] and r["seed"] in (1, 2):
                fin[(r["dataset"], r["cls_key"])].append(float(r["acc"]))
        for (d, c), v in fin.items():
            if (d, c) in s1 and len(v) >= 2:
                sig.setdefault(d, []).append(float(np.std([s1[(d, c)]] + v, ddof=1)))
    return {d: float(np.mean(v)) for d, v in sig.items()}


# ---------------------------------------------------------------- 特征
def feature_table() -> dict[str, dict]:
    t = {}
    for p in SRC_ET:
        for r in read(p):
            t[r["dataset"]] = r
    return t


def feature_matrix(table: Mapping[str, Mapping], ds: Sequence[str], cols: Sequence[str]) -> tuple[np.ndarray, list[int]]:
    """行 = 靶；缺列 / None → nan；返回全列可用的靶下标。"""
    X = np.full((len(ds), len(cols)), np.nan)
    for i, d in enumerate(ds):
        row = table.get(d, {})
        for j, c in enumerate(cols):
            v = row.get(c)
            if v is not None:
                X[i, j] = float(v)
    ok = [i for i in range(len(ds)) if not np.isnan(X[i]).any()]
    return X, ok


# ---------------------------------------------------------------- 预测
def _ridge_fit_predict(Xtr, Ytr, Xte, alpha):
    mu, sd = Xtr.mean(axis=0), np.maximum(Xtr.std(axis=0), 1e-6)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    ym = Ytr.mean(axis=0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return B @ W + ym


def lodo_predict(X: np.ndarray, U: np.ndarray, *, alphas: Sequence[float] = ALPHAS, seed: int = SEED) -> dict:
    """留一靶：训练折内层 LOO 选 α（按 mean regret），预测 u 向量取 argmax（nan 家族不可选）。"""
    n = X.shape[0]
    Uf = np.where(np.isnan(U), np.nanmin(U, axis=1, keepdims=True), U)
    picks = np.zeros(n, dtype=int); fixed = np.zeros(n, dtype=int); alpha_used = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_a, best_r = alphas[0], -np.inf
        for a in alphas:
            reg = []
            for k in tr:
                tr2 = [j for j in tr if j != k]
                pred = _ridge_fit_predict(X[tr2], Uf[tr2], X[[k]], a)[0]
                pred = np.where(np.isnan(U[k]), -np.inf, pred)
                reg.append(U[k, int(np.argmax(pred))])
            if np.mean(reg) > best_r:
                best_r, best_a = float(np.mean(reg)), a
        pred = _ridge_fit_predict(X[tr], Uf[tr], X[[i]], best_a)[0]
        pred = np.where(np.isnan(U[i]), -np.inf, pred)
        picks[i] = int(np.argmax(pred)); fixed[i] = fixed_best(U, train_idx=tr); alpha_used.append(best_a)
    champ = np.nanargmax(U, axis=1)
    return {"picks": picks, "fixed_by_fold": fixed, "hit_rate": float(np.mean(picks == champ)), "alphas": alpha_used}


def evaluate_picks(U: np.ndarray, picks: np.ndarray, *, fixed_by_fold: np.ndarray) -> dict:
    reg = np.array([U[i, picks[i]] for i in range(len(picks))])
    fx = np.array([U[i, fixed_by_fold[i]] for i in range(len(picks))])
    return {"mean_regret": float(reg.mean()), "fixed_regret": float(fx.mean()), "gain_vs_fixed": float((reg - fx).mean()),
            "wins": int(np.sum(reg > fx)), "losses": int(np.sum(reg < fx)), "per_dataset": (reg - fx).tolist()}


def permutation_both(X: np.ndarray, U: np.ndarray, *, n_perm: int = 2000, seed: int = SEED, alphas: Sequence[float] = ALPHAS) -> dict:
    """打乱靶 ↔ 特征行，同一轮置换同时重算 gain 与 hit 两个 LODO 统计量，单侧 p。"""
    def _stats(Xp):
        r = lodo_predict(Xp, U, alphas=alphas, seed=seed)
        return evaluate_picks(U, r["picks"], fixed_by_fold=r["fixed_by_fold"])["gain_vs_fixed"], r["hit_rate"]
    og, oh = _stats(X)
    rng = np.random.default_rng([seed, 8101])
    cg = ch = 0
    for _ in range(int(n_perm)):
        g, h = _stats(X[rng.permutation(X.shape[0])])
        cg += g >= og; ch += h >= oh
    return {"gain": {"observed": float(og), "p": (cg + 1) / (n_perm + 1)}, "hit": {"observed": float(oh), "p": (ch + 1) / (n_perm + 1)}, "n_perm": int(n_perm)}


def permutation_p(X: np.ndarray, U: np.ndarray, *, stat: str, n_perm: int = 2000, seed: int = SEED, alphas: Sequence[float] = ALPHAS) -> dict:
    r = permutation_both(X, U, n_perm=n_perm, seed=seed, alphas=alphas)
    return {"observed": r[stat]["observed"], "p": r[stat]["p"], "n_perm": int(n_perm)}


# ---------------------------------------------------------------- 门
def gate_g1(m: Mapping) -> bool:
    return bool(m["gain_vs_fixed"] >= G1_GAIN and m["p_gain"] < ALPHA_P)


def gate_g2(m: Mapping) -> bool:
    return bool(m["hit_rate"] >= m["majority"] + G2_MARGIN and m["p_hit"] < ALPHA_P)


def verdict_geo(g: Mapping[str, bool]) -> str:
    if g["G-G1"]:
        return "GEO-E-GO"
    return "GEO-E-WEAK" if g["G-G2"] else "GEO-E-DEAD"


# ---------------------------------------------------------------- 主流程
def cmd_run(outdir: Path, *, n_perm: int = 2000) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = catalog_rows()
    U, ds, fams = utility_table(rows)
    champ = champion(U, fams)
    majority = max(Counter(champ).values()) / len(ds)
    sig = noise_by_dataset()
    table = feature_table()
    a: dict = {"protocol": "results/tclsc_geo/PROTOCOL.md", "n_datasets": len(ds), "families": [list(f) for f in fams], "champion_counts": {str(k): v for k, v in Counter(champ).items()},
               "majority": majority, "family_oracle_minus_fixed_all": None, "arms": {}, "datasets": ds}
    fx_all = fixed_best(U, train_idx=list(range(len(ds))))
    a["family_oracle_minus_fixed_all"] = float(np.mean([0.0 - U[i, fx_all] for i in range(len(ds))]))
    a["fixed_family_all"] = list(fams[fx_all])
    prior_j = fams.index(PRIOR_TOP1_FAMILY) if PRIOR_TOP1_FAMILY in fams else None
    a["prior_top1_family_regret"] = float(np.nanmean(U[:, prior_j])) if prior_j is not None else None
    for arm, cols in ARMS.items():
        X, ok = feature_matrix(table, ds, cols)
        if len(ok) < 10:
            a["arms"][arm] = {"n": len(ok), "skipped": True}
            continue
        Xa, Ua = X[ok], U[ok]
        res = lodo_predict(Xa, Ua)
        ev = evaluate_picks(Ua, res["picks"], fixed_by_fold=res["fixed_by_fold"])
        pb = permutation_both(Xa, Ua, n_perm=n_perm)
        pg, ph = pb["gain"], pb["hit"]
        champ_a = [champ[i] for i in ok]
        maj_a = max(Counter(champ_a).values()) / len(ok)
        per = {ds[i]: {"pick": list(fams[res["picks"][k]]), "champion": list(champ[i]), "fixed": list(fams[res["fixed_by_fold"][k]]), "regret_pick": float(Ua[k, res["picks"][k]]),
                       "gain_vs_fixed": ev["per_dataset"][k], "sigma": sig.get(ds[i])} for k, i in enumerate(ok)}
        eeg = [per[d]["gain_vs_fixed"] for d in per if d in EEG_LIKE]; non = [per[d]["gain_vs_fixed"] for d in per if d not in EEG_LIKE]
        a["arms"][arm] = {"n": len(ok), "hit_rate": res["hit_rate"], "majority": maj_a, "mean_regret": ev["mean_regret"], "fixed_regret": ev["fixed_regret"], "gain_vs_fixed": ev["gain_vs_fixed"],
                          "wins": ev["wins"], "losses": ev["losses"], "p_gain": pg["p"], "p_hit": ph["p"], "alphas": dict(Counter(res["alphas"])), "per_dataset": per,
                          "gain_eeg_like": float(np.mean(eeg)) if eeg else None, "gain_other": float(np.mean(non)) if non else None}
        print(f"[{arm}] n {len(ok)} hit {res['hit_rate']:.2f} (maj {maj_a:.2f}, p {ph['p']:.3f}) gain vs fixed {ev['gain_vs_fixed']:+.4f} (p {pg['p']:.3f}) regret {ev['mean_regret']:+.4f} fixed {ev['fixed_regret']:+.4f}", flush=True)
    pa = a["arms"].get(PRIMARY, {})
    g = {"G-G1": bool(pa and not pa.get("skipped") and gate_g1(pa)), "G-G2": bool(pa and not pa.get("skipped") and gate_g2(pa))}
    a["gates"], a["verdict"] = g, verdict_geo(g)
    phi = a["arms"].get("phi", {})
    a["predictions"] = {"P1": {"desc": "G-G1", "hit": g["G-G1"]}, "P2": {"desc": "G-G2", "hit": g["G-G2"]},
                        "P3": {"desc": "et60 增益优于 phi", "hit": bool(pa.get("gain_vs_fixed") is not None and phi.get("gain_vs_fixed") is not None and pa["gain_vs_fixed"] > phi["gain_vs_fixed"])},
                        "P4": {"desc": "增益集中在 EEG / 传感器类靶", "value": (pa.get("gain_eeg_like"), pa.get("gain_other")),
                               "hit": bool(pa.get("gain_eeg_like") is not None and pa.get("gain_other") is not None and pa["gain_eeg_like"] > pa["gain_other"])}}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], g, flush=True)
    return a


def markdown_tables(a: dict) -> str:
    f = lambda v, d=4: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    L = [f"# GEO-E 检验 tables：E_T → 损失几何家族（{a['n_datasets']} 个官方靶）", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", "",
         f"家族 {len(a['families'])} 个；冠军家族分布 {a['champion_counts']}；猜多数 {a['majority']:.2f}；固定最好家族（全数据）{a['fixed_family_all']}，家族 oracle − 固定最好 {a['family_oracle_minus_fixed_all']:+.4f}；先验 top-1 家族 regret {f(a['prior_top1_family_regret'])}", "",
         "| 特征臂 | n | 命中率 (猜多数) | p_hit | mean regret（选） | regret（固定最好） | 增益 vs 固定 | 胜:负 | p_gain | EEG 类增益 / 其余 |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for arm, m in a["arms"].items():
        if m.get("skipped"):
            L.append(f"| {arm} | {m['n']} | 跳过 | | | | | | | |"); continue
        L.append(f"| {arm} | {m['n']} | {m['hit_rate']:.2f} ({m['majority']:.2f}) | {m['p_hit']:.3f} | {m['mean_regret']:+.4f} | {m['fixed_regret']:+.4f} | **{m['gain_vs_fixed']:+.4f}** | {m['wins']}:{m['losses']} | {m['p_gain']:.3f} | {f(m['gain_eeg_like'])} / {f(m['gain_other'])} |")
    pa = a["arms"].get("et60", {})
    if pa and not pa.get("skipped"):
        L += ["", "## et60 逐靶", "", "| 靶 | 冠军家族 | 选 | 固定 | regret（选） | 增益 vs 固定 | σ |", "|---|---|---|---|---:|---:|---:|"]
        for d, v in sorted(pa["per_dataset"].items(), key=lambda kv: -kv[1]["gain_vs_fixed"]):
            L.append(f"| {d} | {'/'.join(v['champion'])} | {'/'.join(v['pick'])} | {'/'.join(v['fixed'])} | {v['regret_pick']:+.3f} | {v['gain_vs_fixed']:+.3f} | {f(v['sigma'], 3)} |")
    L += ["", f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.run:
        cmd_run(args.outdir, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
