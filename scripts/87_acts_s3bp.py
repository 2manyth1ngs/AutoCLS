"""WP17-ACTS S3-B′（results/acts/s3bp/PROTOCOL.md）：field 结构直接吃 E（零 GPU）。

z(t, c) ≈ wᵀφ(c) + Σ_k u_k(t)·v_kᵀφ(c)，u = 训练折标准化 + PCA 的 E_mean（E_pt / 折 E_sh）。
ridge 用按源累加的充分统计量 + 特征分解一次解出全部 α（内层 LOSO 选 α），每个 LOSO 折只在训练源拟合。
C0 参照（μ-only / field-④）走 85 的原代码路径以逐位复现 S3-A′。
用法：python scripts/87_acts_s3bp.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3bp"
S3B = ROOT / "results" / "acts" / "s3b"
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4)
RS = (4, 8, 16)
R_MAIN = 8


# ================================================================ 纯函数
def pca_fit(E: np.ndarray, r: int) -> dict:
    E = np.asarray(E, float)
    m, s = E.mean(0), E.std(0)
    s = np.where(s > 1e-12, s, 1.0)
    Zs = (E - m) / s
    _U, _S, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    return {"m": m, "s": s, "V": Vt[:r].T, "c": Zs.mean(0)}


def pca_apply(f: Mapping, E: np.ndarray) -> np.ndarray:
    return ((np.asarray(E, float) - f["m"]) / f["s"] - f["c"]) @ f["V"]


def field_design(phi: np.ndarray, U: np.ndarray) -> np.ndarray:
    """[φ, u₁φ, …, u_rφ]（逐行；U 与 phi 同行数）。"""
    phi, U = np.asarray(phi, float), np.asarray(U, float)
    return np.concatenate([phi] + [phi * U[:, [k]] for k in range(U.shape[1])], axis=1)


def verdict_fe(gates: Mapping[str, Mapping[str, bool]]) -> str:
    names = [("F1", "CONTENT"), ("F2", "VALUE"), ("F3", "REPLACE"), ("F4", "SPACE")]
    hit = [lab for g, lab in names if any(v[g] for v in gates.values())]
    return "FE-" + "+".join(hit) if hit else "FE-INERT"


# ================================================================ 充分统计量 ridge
def _stats(X: np.ndarray, y: np.ndarray) -> tuple:
    return (len(y), X.sum(0), float(y.sum()), X.T @ X, X.T @ y)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _solve_all(st: tuple, alphas: Sequence[float]) -> list[tuple[np.ndarray, float]]:
    n, sx, sy, Sxx, sxy = st
    m, ym = sx / n, sy / n
    C = Sxx - n * np.outer(m, m)
    c = sxy - n * m * ym
    lam, V = np.linalg.eigh((C + C.T) / 2)
    Vc = V.T @ c
    out = []
    for a in alphas:
        w = V @ (Vc / (np.maximum(lam, 0) + a))
        out.append((w, float(ym - m @ w)))
    return out


# ================================================================ 分析
def _fold_blocks(D: Mapping, phi_sh: np.ndarray, U: np.ndarray, col_m: np.ndarray, col_s: np.ndarray, idx: Sequence[int]) -> dict:
    """每任务标准化设计块与标签（shared 行在前）。"""
    out = {}
    for i in idx:
        r = D["rand"][i]
        ph = np.vstack([phi_sh, r["phi"]])
        X = (field_design(ph, np.repeat(U[[i]], len(ph), axis=0)) - col_m) / col_s
        out[i] = (X, np.concatenate([D["Z"][i], r["Z"]]))
    return out


def run_arm(D: Mapping, phi_sh: np.ndarray, E_of: callable, r: int, order: Sequence[str], cross_map: Sequence[int]) -> dict:
    """E_of(fold) → (n_tasks, 256) E_mean（该折编码）。返回逐任务 shared 预测、random 预测（原 E 与 cross-E）。"""
    groups = D["groups"]; n = len(groups)
    pred_sh = np.zeros((n, 30)); pred_rd = [None] * n; pred_sh_x = np.zeros((n, 30)); pred_rd_x = [None] * n; alphas = {}
    for s in order:
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        E = E_of(s)
        f = pca_fit(E[tr], r)
        U = pca_apply(f, E)
        # 列标准化：用外层训练行的均值 / 标准差（内层 α 选择沿用）
        Xs = [field_design(np.vstack([phi_sh, D["rand"][i]["phi"]]), np.repeat(U[[i]], 30 + len(D["rand"][i]["phi"]), axis=0)) for i in tr]
        allX = np.vstack(Xs)
        col_m, col_s = allX.mean(0), allX.std(0); col_s = np.where(col_s > 1e-12, col_s, 1.0)
        del allX
        blk = _fold_blocks(D, phi_sh, U, col_m, col_s, tr)
        per_g = {}
        for i in tr:
            st = _stats(*blk[i])
            per_g[groups[i]] = st if groups[i] not in per_g else _add(per_g[groups[i]], st)
        tot = None
        for st in per_g.values():
            tot = st if tot is None else _add(tot, st)
        err = np.zeros(len(ALPHAS))
        for g2, st in per_g.items():
            sols = _solve_all(_sub(tot, st), ALPHAS)
            for i in tr[groups[tr] == g2]:
                X, y = blk[i]
                for j, (w, b) in enumerate(sols):
                    err[j] += float(np.sum((X @ w + b - y) ** 2))
        best = err.min(); ai = max(j for j in range(len(ALPHAS)) if err[j] <= best * (1 + 1e-9))
        alphas[s] = ALPHAS[ai]
        w, b = _solve_all(tot, [ALPHAS[ai]])[0]
        for i in te:
            for ux, (ps, pr) in ((U[[i]], (pred_sh, pred_rd)), (U[[cross_map[i]]], (pred_sh_x, pred_rd_x))):
                ph = np.vstack([phi_sh, D["rand"][i]["phi"]])
                X = (field_design(ph, np.repeat(ux, len(ph), axis=0)) - col_m) / col_s
                p = X @ w + b
                ps[i] = p[:30]; pr[i] = p[30:]
    return {"pred_sh": pred_sh, "pred_rd": pred_rd, "pred_sh_x": pred_sh_x, "pred_rd_x": pred_rd_x, "alpha": alphas}


def score_arm(D: Mapping, P: Mapping, key_sh: str, key_rd: str) -> dict:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    n = len(D["groups"])
    sp = np.full(n, np.nan); r3 = g84.regret_at_k(P[key_sh], D["end0"], D["end1"], 3); r1 = g84.regret_at_k(P[key_sh], D["end0"], D["end1"], 1)
    d50 = np.full(n, np.nan)
    for i in range(n):
        R0 = D["rand"][i]["R0"]; pr = P[key_rd][i]
        sp[i] = g85.task_spearman(pr, R0, min_n=10)
        if len(pr):
            a, b = g85.regret_30_50(P[key_sh][i], pr, D["end0"][i], D["end1"][i], R0, float(D["sigma"][i]))
            d50[i] = b - a
    return {"sp": sp, "r3": r3, "r1": r1, "d50": d50}


def c0_refs(D: Mapping, phi_sh: np.ndarray, b_ref: np.ndarray, a_hat_probe: np.ndarray) -> dict:
    """S3-A′ 任务 1 的 μ-only / L|hat（ridge [φ, a₁φ]）两臂，逐位同 85.task1 的代码路径。"""
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    groups = D["groups"]; n = len(groups)
    out = {k: {"sp": np.full(n, np.nan), "pred_sh": np.zeros((n, 30)), "pred_rd": [None] * n} for k in ("mu", "L")}
    for s in list(dict.fromkeys(groups.tolist())):
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        Ptr, Atr, Ytr, Ttr = g85._space_rows(D, phi_sh, tr, ctx["A"][:, 0])
        for kind in ("L", "mu"):
            f = g85._fit_space(kind, g85._design(Ptr, Atr, kind), Ytr, groups[Ttr])
            for i in te:
                r = D["rand"][i]; ph = np.vstack([phi_sh, r["phi"]])
                pred = f(g85._design(ph, np.full(len(ph), a_hat_probe[i, 0]), kind))
                out[kind]["pred_sh"][i] = pred[:30]; out[kind]["pred_rd"][i] = pred[30:]
                out[kind]["sp"][i] = g85.task_spearman(pred[30:], r["R0"], min_n=10)
    return out


def _src_means(x, groups):
    ok = np.isfinite(x)
    srcs = list(dict.fromkeys(groups.tolist()))
    return np.array([np.nanmean(x[(groups == s) & ok]) for s in srcs])


def _wil(d, groups):
    return importlib.import_module("scripts.84_acts_s3a").source_wilcoxon(np.nan_to_num(d), groups)


def cmd_analyze(outdir: Path = OUT) -> dict:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap"); g86 = importlib.import_module("scripts.86_acts_s3b")
    data = g85.load_all(); D = data["pool"]; phi_sh = data["phi_sh"]
    groups = D["groups"]; n = len(groups); tasks = list(D["tasks"])
    order = list(dict.fromkeys(groups.tolist()))
    b_ref = g85.orient_bref(D)
    T = g86._load_targets(S3B)
    assert [str(t) for t in T["tasks"]][:n] == tasks
    all_tasks = [str(t) for t in T["tasks"]]
    em_pt = g86._e_means(S3B, "pt", all_tasks, order)[""][:n]
    em_sh = {f: v[:n] for f, v in g86._e_means(S3B, "sh", all_tasks, order).items()}
    cross = g86.cross_assign(tasks, groups, order)
    # 参照：任务盲 / ④ / ③（S3-A 口径）+ C0
    ref = g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], {"probe": g84.probe_feature(D["acc25"], D["pfloor"], D["maj"]), "land25": g84.land25_feature(D["Z25"])}, b_ref=b_ref)
    blind3, probe3 = ref["_blind"]["regret3"], ref["probe"]["regret3"]
    c0r = c0_refs(D, phi_sh, b_ref, ref["probe"]["a_hat"])
    s3ap = json.loads((ROOT / "results/acts/s3ap/analysis.json").read_text(encoding="utf-8"))["task1"]["summary"]
    c0 = {"mu_sp_median": (float(np.nanmedian(c0r["mu"]["sp"])), s3ap["mu|hat"]["sp_median"]), "L_sp_median": (float(np.nanmedian(c0r["L"]["sp"])), s3ap["L|hat"]["sp_median"]),
          "blind3": (float(blind3.mean()), 0.02131), "probe3": (float(probe3.mean()), 0.00984), "land3": (float(ref["land25"]["mean_regret3"]), 0.00611)}
    c0_ok = abs(c0["mu_sp_median"][0] - c0["mu_sp_median"][1]) < 1e-12 and abs(c0["L_sp_median"][0] - c0["L_sp_median"][1]) < 1e-12 and all(abs(a - b) < 5e-5 for k, (a, b) in c0.items() if k.endswith("3"))
    mu_sp_src = _src_means(c0r["mu"]["sp"], groups)
    rows, gates = {}, {}
    refs_rows = {}
    for nm, kind in (("μ-only", "mu"), ("field-④", "L")):
        P = {"pred_sh": c0r[kind]["pred_sh"], "pred_rd": c0r[kind]["pred_rd"]}
        sc = score_arm(D, P, "pred_sh", "pred_rd")
        refs_rows[nm] = {"sp_median": float(np.nanmedian(sc["sp"])), "sp_mean": float(np.nanmean(sc["sp"])), "regret3": float(sc["r3"].mean()), "regret1": float(sc["r1"].mean()),
                         "d50": float(np.nanmean(sc["d50"]))}
    for arm in ("pt", "sh"):
        for r in RS:
            E_of = (lambda s: em_pt) if arm == "pt" else (lambda s, em=em_sh: em[s])
            P = run_arm(D, phi_sh, E_of, r, order, cross)
            sc = score_arm(D, P, "pred_sh", "pred_rd"); sx = score_arm(D, P, "pred_sh_x", "pred_rd_x")
            key = f"{arm}_r{r}"
            dsp = _src_means(sc["sp"], groups) - mu_sp_src
            from scipy.stats import wilcoxon
            p_sp = float(wilcoxon(dsp).pvalue) if np.any(np.abs(dsp) > 1e-15) else 1.0
            dcross = sx["r3"] - sc["r3"]; dblind = blind3 - sc["r3"]; dprobe = sc["r3"] - probe3
            row = {"sp_median": float(np.nanmedian(sc["sp"])), "sp_mean": float(np.nanmean(sc["sp"])),
                   "sp_ci": g84.cluster_bootstrap_ci(sc["sp"][np.isfinite(sc["sp"])], groups[np.isfinite(sc["sp"])]),
                   "sp_vs_mu_src": float(np.mean(dsp)), "p_sp_vs_mu": p_sp, "sp_cross_median": float(np.nanmedian(sx["sp"])),
                   "regret1": float(sc["r1"].mean()), "regret3": float(sc["r3"].mean()),
                   "vs_blind": float(dblind.mean()), "vs_blind_ci": g84.cluster_bootstrap_ci(dblind, groups), "p_vs_blind": _wil(dblind, groups),
                   "vs_probe": float(dprobe.mean()), "vs_probe_ci": g84.cluster_bootstrap_ci(dprobe, groups),
                   "cross_d": float(dcross.mean()), "cross_ci": g84.cluster_bootstrap_ci(dcross, groups), "p_cross": _wil(dcross, groups),
                   "d50": float(np.nanmean(sc["d50"])), "alpha": P["alpha"]}
            rows[key] = row
            if r == R_MAIN:
                gates[arm] = {"F1": bool((row["cross_d"] > 0 and row["p_cross"] < 0.05) or (row["sp_vs_mu_src"] > 0 and p_sp < 0.05)),
                              "F2": bool(row["vs_blind"] >= 0.010 and row["p_vs_blind"] < 0.05),
                              "F3": bool(row["vs_probe_ci"][1] <= 0.005), "F4": bool(row["sp_median"] >= 0.50)}
            print(f"[s3bp] {key} sp {row['sp_median']:.3f} r3 {row['regret3']:.4f} cross {row['cross_d']:+.4f}", flush=True)
    code = verdict_fe(gates)
    out = {"verdict": code, "c0": {k: list(v) for k, v in c0.items()}, "c0_ok": c0_ok, "gates": gates, "rows": rows, "refs": refs_rows,
           "blind3": float(blind3.mean()), "probe3": float(probe3.mean()), "land3": float(ref["land25"]["mean_regret3"])}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return out


def _f(x, nd=3):
    try:
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-B′ 读数（{o['verdict']}）", "", f"C0 {'✓' if o['c0_ok'] else '✗'}：" + "；".join(f"{k} {_f(a, 4)} vs {_f(b, 4)}" for k, (a, b) in o["c0"].items()), "",
         f"参照（shared-30，S3-A 口径）：任务盲 regret@3 {_f(o['blind3'])}；④ {_f(o['probe3'])}；③ {_f(o['land3'])}", "",
         "| 臂 | random-20 中位 Spearman | 均值 [CI] | 逐源 vs μ-only p | cross-E 中位 | regret@1 | regret@3 | vs 任务盲 [CI] p | E − ④ [CI] | cross-E Δregret@3 [CI] p | Δ50 |",
         "|---|---:|---|---|---:|---:|---:|---|---|---|---:|"]
    for k, v in o["refs"].items():
        L.append(f"| {k}（参照） | {_f(v['sp_median'], 2)} | {_f(v['sp_mean'], 2)} | — | — | {_f(v['regret1'])} | {_f(v['regret3'])} | — | — | — | {_f(v['d50'])} |")
    for k, v in o["rows"].items():
        L.append(f"| {k} | {_f(v['sp_median'], 2)} | {_f(v['sp_mean'], 2)} [{_f(v['sp_ci'][0], 2)}, {_f(v['sp_ci'][1], 2)}] | {_f(v['sp_vs_mu_src'], 3)} p {_f(v['p_sp_vs_mu'], 3)} | {_f(v['sp_cross_median'], 2)} | "
                 f"{_f(v['regret1'])} | {_f(v['regret3'])} | {_f(v['vs_blind'])} [{_f(v['vs_blind_ci'][0])}, {_f(v['vs_blind_ci'][1])}] {_f(v['p_vs_blind'], 3)} | "
                 f"{_f(v['vs_probe'])} [{_f(v['vs_probe_ci'][0])}, {_f(v['vs_probe_ci'][1])}] | {_f(v['cross_d'])} [{_f(v['cross_ci'][0])}, {_f(v['cross_ci'][1])}] {_f(v['p_cross'], 3)} | {_f(v['d50'])} |")
    L += ["", "门（主 r = 8）：" + "；".join(f"{a}: " + ", ".join(f"{g} {'✓' if b else '✗'}" for g, b in v.items()) for a, v in o["gates"].items())]
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
