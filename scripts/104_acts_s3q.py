"""WP17-ACTS S3-Q（results/acts/s3q/PROTOCOL.md）：E 任务检索（E-kNN），零 GPU。

目标任务的 E → 在训练折任务里找 k 个最近邻（标准化余弦）→ 近邻偏好画像当该任务的打分（目录层 shared-30）；
目录外（random-20）用近邻任务上拟合的 binned-φ 岭回归（SUR-KNN）对照全训练折拟合的同款（SUR-BLIND）。
k 与 α 都在训练折内层留一个源选；换 E 对照 = 88.cross_maps 偏移 1–16。
用法：python scripts/104_acts_s3q.py --analyze   |   --smoke（合成数据跑通）
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import rankdata, spearmanr  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3q"
S3B = ROOT / "results" / "acts" / "s3b"
S3C = ROOT / "results" / "acts" / "s3c"
S3N = ROOT / "results" / "acts" / "s3n"
S3O = ROOT / "results" / "acts" / "s3o"
KS = (3, 5, 10, 20, 40)
ALPHAS = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
OFFSETS = tuple(range(1, 17))
UEA6 = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1")
E_NAMES = ("E_pt", "E_sh", "E_A", "E_TF")
N_SH = 30


# ================================================================ 纯函数
def std_cos(Xtr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """按训练折每维均值 / 标准差标准化后，x 与 Xtr 各行的余弦。"""
    Xtr = np.asarray(Xtr, float); x = np.asarray(x, float)
    m, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A = (Xtr - m) / sd; b = (x - m) / sd
    return A @ b / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(b), 1e-12)


def knn_profile(Z: np.ndarray, sim: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    nb = np.argsort(-np.asarray(sim, float), kind="stable")[:int(k)]
    return np.asarray(Z, float)[nb].mean(0), nb


def choose_k(inner: Mapping[int, float]) -> int:
    best = min(inner.values())
    return int(max(k for k, v in inner.items() if v <= best + 1e-12))


def choose_alpha_sp(inner: Mapping[float, float]) -> float:
    best = max(inner.values())
    return float(max(a for a, v in inner.items() if v >= best - 1e-12))


def signflip_p(d: np.ndarray, n_mc: int = 200_000, seed: int = 0) -> float:
    """源均值差的符号翻转检验（双侧）；n ≤ 20 精确枚举。"""
    d = np.asarray(d, float)
    if np.all(np.abs(d) < 1e-15):
        return 1.0
    n = len(d); obs = abs(d.mean())
    if n <= 20:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_mc, n))
    return float(np.mean(np.abs(signs @ d) / n >= obs - 1e-12))


def spearman_rows(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    out = []
    for p, t in zip(np.asarray(P, float), np.asarray(T, float)):
        r = spearmanr(p, t)[0] if np.ptp(p) > 0 and np.ptp(t) > 0 else 0.0
        out.append(0.0 if not np.isfinite(r) else float(r))
    return np.array(out)


def verdict_s3q(g1: Mapping, g2: Mapping, g3: Mapping) -> str:
    v = g1["d"] >= 0.003 and g1["p"] < 0.05
    c = g2["d"] > 0 and g2["p"] < 0.05
    code = {(True, True): "S3Q-RETR-WIN", (True, False): "S3Q-RETR-VALUE", (False, True): "S3Q-RETR-CONTENT",
            (False, False): "S3Q-RETR-NULL"}[(v, c)]
    return code + ("+SPACE" if g3["d"] >= 0.03 and g3["p"] < 0.05 else "")


def holm(ps: Sequence[float]) -> list[float]:
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def ridge_multi(X: np.ndarray, y: np.ndarray, alphas: Sequence[float]) -> dict:
    """= 84.ridge_fit（列标准化、截距不罚）对多个 α 共用一次 XᵀX。"""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m, s = X.mean(0), X.std(0)
    s = np.where(s > 1e-12, s, 1.0)
    Xs = (X - m) / s; ym = float(y.mean())
    G = Xs.T @ Xs; b = Xs.T @ (y - ym); I = np.eye(G.shape[0])
    return {a: {"m": m, "s": s, "w": np.linalg.solve(G + float(a) * I, b), "b": ym} for a in alphas}


def ridge_pred(model: Mapping, X: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, float) - model["m"]) / model["s"]) @ model["w"] + model["b"]


def rank_sim(z: np.ndarray, Ztr: np.ndarray) -> np.ndarray:
    """z 与 Ztr 各行的 Spearman（秩的 Pearson）。"""
    r = rankdata(z); R = np.apply_along_axis(rankdata, 1, Ztr)
    r = r - r.mean(); R = R - R.mean(1, keepdims=True)
    return R @ r / np.maximum(np.linalg.norm(R, axis=1) * np.linalg.norm(r), 1e-12)


# ================================================================ 折内计算
def _regret(S: np.ndarray, e0: np.ndarray, e1: np.ndarray, k: int = 3) -> np.ndarray:
    g84 = importlib.import_module("scripts.84_acts_s3a")
    return g84.regret_at_k(np.atleast_2d(S), np.atleast_2d(e0), np.atleast_2d(e1), k)


def run_E(D: Mapping, X_of: Callable[[str], np.ndarray], cmaps: Sequence[Sequence[int]], *, ks: Sequence[int] = KS,
          alphas: Sequence[float] = ALPHAS, sur: bool = True, oracle: bool = True) -> dict:
    """一个 E 的全部 LOSO 读数。返回逐任务数组（regret、Spearman、高估）与逐折 k / α / 近邻源。"""
    ac = importlib.import_module("scripts.acts_common")
    G = np.asarray(D["groups"]); Z, e0, e1 = D["Z"], D["end0"], D["end1"]
    phi, Y, R0 = D["phi"], D["Y"], D["R0"]
    n = len(G); nr = R0.shape[1]
    arms3 = ("BLIND", "KNN", "KNN10", "KNNX", "ORACLE")
    r3 = {a: np.full(n, np.nan) for a in arms3}
    r1 = {a: np.full(n, np.nan) for a in arms3}
    sp = {a: np.full(n, np.nan) for a in ("SUR-BLIND", "SUR-KNN", "SUR-X")}
    sp_top = {a: np.full(n, np.nan) for a in ("SUR-BLIND", "SUR-KNN")}
    sur_r3 = {a: np.full(n, np.nan) for a in ("SUR-BLIND", "SUR-KNN")}
    over = {a: np.full(n, np.nan) for a in ("SUR-BLIND", "SUR-KNN")}
    kfold, afold, nbsrc = {}, {}, {}

    def sur_readout(model, i, arm, top=True):
        pr = ridge_pred(model, phi[i, N_SH:])
        sp[arm][i] = spearman_rows(pr[None], R0[i][None])[0]
        if top and arm in sp_top:
            hi = np.argsort(-R0[i], kind="stable")[: nr // 2]
            sp_top[arm][i] = spearman_rows(pr[hi][None], R0[i][hi][None])[0]
            sur_r3[arm][i] = _regret(ridge_pred(model, phi[i, :N_SH]), e0[i], e1[i])[0]
            t3 = np.argsort(-pr, kind="stable")[:3]
            over[arm][i] = float(np.mean(pr[t3] - Y[i, N_SH:][t3]))

    for s in dict.fromkeys(G.tolist()):
        te = np.where(G == s)[0]; tr = np.where(G != s)[0]
        X = np.asarray(X_of(s), float)
        # ---- 内层选 k（目录层 regret@3，源加权）
        inner = {k: [] for k in ks}
        for u in dict.fromkeys(G[tr].tolist()):
            tu = tr[G[tr] != u]; eu = tr[G[tr] == u]
            S = {k: [] for k in ks}
            for i in eu:
                o = np.argsort(-std_cos(X[tu], X[i]), kind="stable")
                for k in ks:
                    S[k].append(Z[tu[o[:k]]].mean(0))
            for k in ks:
                inner[k].append(float(np.mean(_regret(np.array(S[k]), e0[eu], e1[eu]))))
        kf = choose_k({k: float(np.mean(v)) for k, v in inner.items()})
        kfold[s] = kf
        # ---- 目录层外层
        mu = Z[tr].mean(0)
        SB = np.tile(mu, (len(te), 1)); SK, S10, SO = [], [], []
        for i in te:
            o = np.argsort(-std_cos(X[tr], X[i]), kind="stable")
            SK.append(Z[tr[o[:kf]]].mean(0)); S10.append(Z[tr[o[:10]]].mean(0))
            nbsrc[int(i)] = G[tr[o[:kf]]].tolist()
            if oracle:
                zi = ac.clip_z(e0[i], 0.5)[0]
                so = np.argsort(-rank_sim(zi, Z[tr]), kind="stable")[:10]
                SO.append(Z[tr[so]].mean(0))
        for arm, S in (("BLIND", SB), ("KNN", np.array(SK)), ("KNN10", np.array(S10))) + ((("ORACLE", np.array(SO)),) if oracle else ()):
            r3[arm][te] = _regret(S, e0[te], e1[te], 3); r1[arm][te] = _regret(S, e0[te], e1[te], 1)
        xr3, xr1 = np.zeros(len(te)), np.zeros(len(te))
        for cm in cmaps:
            SX = []
            for i in te:
                o = np.argsort(-std_cos(X[tr], X[cm[i]]), kind="stable")
                SX.append(Z[tr[o[:kf]]].mean(0))
            xr3 += _regret(np.array(SX), e0[te], e1[te], 3); xr1 += _regret(np.array(SX), e0[te], e1[te], 1)
        r3["KNNX"][te] = xr3 / len(cmaps); r1["KNNX"][te] = xr1 / len(cmaps)
        if not sur:
            continue
        # ---- 目录外：α 内层（random-20 平均 Spearman 最大）
        inB = {a: [] for a in alphas}; inK = {a: [] for a in alphas}
        for u in dict.fromkeys(G[tr].tolist()):
            tu = tr[G[tr] != u]; eu = tr[G[tr] == u]
            MB = ridge_multi(phi[tu].reshape(-1, phi.shape[2]), Y[tu].reshape(-1), alphas)
            vk = {a: [] for a in alphas}
            for a in alphas:
                P = np.array([ridge_pred(MB[a], phi[i, N_SH:]) for i in eu])
                inB[a].append(float(np.mean(spearman_rows(P, R0[eu]))))
            for i in eu:
                o = np.argsort(-std_cos(X[tu], X[i]), kind="stable"); nb = tu[o[:kf]]
                MK = ridge_multi(phi[nb].reshape(-1, phi.shape[2]), Y[nb].reshape(-1), alphas)
                for a in alphas:
                    vk[a].append(spearman_rows(ridge_pred(MK[a], phi[i, N_SH:])[None], R0[i][None])[0])
            for a in alphas:
                inK[a].append(float(np.mean(vk[a])))
        aB = choose_alpha_sp({a: float(np.mean(v)) for a, v in inB.items()})
        aK = choose_alpha_sp({a: float(np.mean(v)) for a, v in inK.items()})
        afold[s] = {"blind": aB, "knn": aK}
        MB = ridge_multi(phi[tr].reshape(-1, phi.shape[2]), Y[tr].reshape(-1), (aB,))[aB]
        for i in te:
            sur_readout(MB, i, "SUR-BLIND")
            o = np.argsort(-std_cos(X[tr], X[i]), kind="stable"); nb = tr[o[:kf]]
            sur_readout(ridge_multi(phi[nb].reshape(-1, phi.shape[2]), Y[nb].reshape(-1), (aK,))[aK], i, "SUR-KNN")
            vx = []
            for cm in cmaps:
                o = np.argsort(-std_cos(X[tr], X[cm[i]]), kind="stable"); nb = tr[o[:kf]]
                mk = ridge_multi(phi[nb].reshape(-1, phi.shape[2]), Y[nb].reshape(-1), (aK,))[aK]
                vx.append(spearman_rows(ridge_pred(mk, phi[i, N_SH:])[None], R0[i][None])[0])
            sp["SUR-X"][i] = float(np.mean(vx))
    return {"r3": r3, "r1": r1, "sp": sp, "sp_top": sp_top, "sur_r3": sur_r3, "over": over,
            "k": kfold, "alpha": afold, "nbsrc": nbsrc}


# ================================================================ 统计
def compare(d_task: np.ndarray, G: Sequence, *, n_boot: int = 10000) -> dict:
    """d_task = 逐任务差（正 = 后者更好）；门用源加权均值 + 精确符号翻转。"""
    g84 = importlib.import_module("scripts.84_acts_s3a")
    G = np.asarray(G); d_task = np.asarray(d_task, float)
    srcs = list(dict.fromkeys(G.tolist()))
    dm = np.array([d_task[G == s].mean() for s in srcs])
    lo, hi = g84.cluster_bootstrap_ci(d_task, G, n_boot=n_boot)
    return {"d": float(dm.mean()), "d_task": float(d_task.mean()), "p": signflip_p(dm), "ci": [lo, hi],
            "n_pos": int((dm > 0).sum()), "n_neg": int((dm < 0).sum()), "n_src": len(srcs),
            "per_src": {s: float(v) for s, v in zip(srcs, dm)}}


def summarize(res: Mapping, G: Sequence) -> dict:
    G = np.asarray(G)
    r3, r1, sp = res["r3"], res["r1"], res["sp"]
    out = {"regret3": {a: float(np.nanmean(v)) for a, v in r3.items()},
           "regret3_src": {a: float(np.mean([np.nanmean(v[G == s]) for s in dict.fromkeys(G.tolist())])) for a, v in r3.items()},
           "regret1": {a: float(np.nanmean(v)) for a, v in r1.items()},
           "G1": compare(r3["BLIND"] - r3["KNN"], G), "G2": compare(r3["KNNX"] - r3["KNN"], G),
           "G1_k10": compare(r3["BLIND"] - r3["KNN10"], G), "r1_blind_minus_knn": compare(r1["BLIND"] - r1["KNN"], G),
           "k": res["k"], "alpha": res["alpha"]}
    if np.isfinite(sp["SUR-KNN"]).any():
        out["G3"] = compare(sp["SUR-KNN"] - sp["SUR-BLIND"], G)
        out["G3x"] = compare(sp["SUR-KNN"] - sp["SUR-X"], G)
        out["spearman"] = {a: float(np.nanmean(v)) for a, v in sp.items()}
        out["spearman_median"] = {a: float(np.nanmedian(v)) for a, v in sp.items()}
        out["spearman_top"] = {a: float(np.nanmean(v)) for a, v in res["sp_top"].items()}
        out["sur_regret3"] = {a: float(np.nanmean(v)) for a, v in res["sur_r3"].items()}
        out["overest"] = {a: float(np.nanmean(v)) for a, v in res["over"].items()}
    else:
        out["G3"] = {"d": 0.0, "p": 1.0}
    out["verdict"] = verdict_s3q(out["G1"], out["G2"], out["G3"])
    uea = [s for s in UEA6 if s in out["G1"]["per_src"]]
    out["uea6_knn_better"] = int(sum(out["G1"]["per_src"][s] > 0 for s in uea))
    return out


# ================================================================ 数据
def load_data() -> dict:
    g84 = importlib.import_module("scripts.84_acts_s3a"); ac = importlib.import_module("scripts.acts_common")
    S = g84.load_s2(); P = S["pool"]
    meta = json.loads((S3C / "data.json").read_text(encoding="utf-8"))
    with np.load(S3C / "data.npz") as z:
        arr = {k: z[k] for k in z.files}
    tasks = list(P["tasks"])
    assert tasks == meta["tasks"], "s3c 任务序与 84 池不一致"
    assert np.allclose(arr["end0"], P["end0"]) and np.allclose(arr["end1"], P["end1"])
    assert len(S["keys"]) == N_SH and all(list(meta["keys"][t][:N_SH]) == list(S["keys"]) for t in tasks), "shared 列序不一致"
    phi = np.stack([arr[f"{t}/vec"] for t in tasks]).astype(float)
    R0 = np.stack([arr[f"{t}/R0"] for t in tasks]).astype(float)
    Y = np.stack([ac.clip_z(np.r_[P["end0"][i], R0[i]], 0.5)[0] for i in range(len(tasks))])
    return {"tasks": tasks, "groups": np.asarray(P["groups"]), "Z": np.asarray(P["Z"], float), "end0": P["end0"],
            "end1": P["end1"], "phi": phi, "R0": R0, "Y": Y, "order": meta["order"]}


def load_E(tasks: Sequence[str]) -> dict[str, Callable[[str], np.ndarray]]:
    def rows(path: Path, key: str) -> np.ndarray:
        with np.load(path, allow_pickle=True) as z:
            d = dict(zip([str(t) for t in z["tasks"]], z[key]))
        return np.array([d[t] for t in tasks], float)
    pt = rows(S3B / "E_pt" / "_means.npz", "E_mean")
    ea = rows(S3N / "ea.npz", "EA")
    et = rows(S3O / "et.npz", "ETF")
    sh: dict[str, np.ndarray] = {}

    def e_sh(fold: str) -> np.ndarray:
        if fold not in sh:
            sh[fold] = rows(S3B / "E_sh" / fold / "_means.npz", "E_mean")
        return sh[fold]
    return {"E_pt": lambda f: pt, "E_sh": e_sh, "E_A": lambda f: ea, "E_TF": lambda f: et}


# ================================================================ 命令
def _nb_composition(nbsrc: Mapping[int, list], G: np.ndarray) -> dict:
    out = {}
    for s in dict.fromkeys(G.tolist()):
        c: dict[str, int] = {}
        for i in np.where(G == s)[0]:
            for g in nbsrc[int(i)]:
                c[g] = c.get(g, 0) + 1
        tot = sum(c.values())
        out[s] = [(g, round(v / tot, 2)) for g, v in sorted(c.items(), key=lambda x: -x[1])[:3]]
    return out


def cmd_analyze(outdir: Path = OUT) -> dict:
    g88 = importlib.import_module("scripts.88_acts_s3bpp")
    D = load_data(); G = D["groups"]
    cmaps = g88.cross_maps(D["tasks"], G, D["order"], OFFSETS)
    Es = load_E(D["tasks"])
    o: dict = {"n_tasks": len(G), "n_src": len(set(G.tolist())), "E": {}}
    for name in E_NAMES:
        print(f"[s3q] {name} …", flush=True)
        res = run_E(D, Es[name], cmaps, oracle=(name == "E_pt"))
        sm = summarize(res, G)
        sm["nb_composition"] = _nb_composition(res["nbsrc"], G)
        o["E"][name] = sm
        print(f"[s3q] {name}: {sm['verdict']}  regret@3 BLIND {sm['regret3']['BLIND']:.4f} KNN {sm['regret3']['KNN']:.4f} "
              f"KNNX {sm['regret3']['KNNX']:.4f}  G1 d {sm['G1']['d']:+.4f} p {sm['G1']['p']:.3f}  "
              f"G2 d {sm['G2']['d']:+.4f} p {sm['G2']['p']:.3f}  G3 d {sm['G3']['d']:+.3f} p {sm['G3']['p']:.3f}", flush=True)
    hp = holm([o["E"][n]["G1"]["p"] for n in E_NAMES])
    for n, p in zip(E_NAMES, hp):
        o["E"][n]["G1"]["p_holm4"] = p
    o["verdict"] = o["E"]["E_pt"]["verdict"]
    pt = o["E"]["E_pt"]
    ks = list(pt["k"].values())
    o["predictions"] = {
        "P1": pt["G1"]["d"] >= 0.003 and pt["G1"]["p"] < 0.05,
        "P2": pt["G2"]["d"] > 0 and pt["G2"]["p"] < 0.05,
        "P3": pt["G3"]["d"] >= 0.03 and pt["G3"]["p"] < 0.05,
        "P4": sum(k == 10 for k in ks) >= 9,
        "P5": all(pt["regret3"]["KNN"] <= o["E"][n]["regret3"]["KNN"] for n in ("E_sh", "E_A", "E_TF")),
        "P6": pt["uea6_knn_better"] >= 4,
        "P7": pt["regret3"]["ORACLE"] < 0.5 * pt["regret3"]["BLIND"],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(tables_markdown(o), encoding="utf-8")
    print(f"[s3q] verdict {o['verdict']}  predictions {sum(o['predictions'].values())}/7", flush=True)
    return o


def _f(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-Q 表（{o['n_tasks']} 任务 / {o['n_src']} 源 LOSO；regret 为 84.regret_at_k 口径）", "",
         "## 目录层 regret@3（任务均值 / 源均值）", "",
         "| E | BLIND | KNN | KNN10 | KNN-X | ORACLE | regret@1 BLIND / KNN |", "|---|---|---|---|---|---|---|"]
    for n, sm in o["E"].items():
        a, b = sm["regret3"], sm["regret3_src"]
        L.append(f"| {n} | {_f(a['BLIND'])} / {_f(b['BLIND'])} | {_f(a['KNN'])} / {_f(b['KNN'])} | {_f(a['KNN10'])} | "
                 f"{_f(a['KNNX'])} | {_f(a['ORACLE'])} | {_f(sm['regret1']['BLIND'])} / {_f(sm['regret1']['KNN'])} |")
    L += ["", "## 门（源加权差，精确符号翻转）", "",
          "| E | G1 BLIND−KNN | p（Holm4） | 源 +/− | G2 KNNX−KNN | p | G3 Sp KNN−BLIND | p | 读数码 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, sm in o["E"].items():
        g1, g2, g3 = sm["G1"], sm["G2"], sm["G3"]
        L.append(f"| {n} | {g1['d']:+.4f} [{g1['ci'][0]:+.4f}, {g1['ci'][1]:+.4f}] | {g1['p']:.3f}（{g1.get('p_holm4', 1):.3f}） | "
                 f"{g1['n_pos']}/{g1['n_neg']} | {g2['d']:+.4f} | {g2['p']:.3f} | {g3['d']:+.3f} | {g3['p']:.3f} | {sm['verdict']} |")
    L += ["", "## 目录外（random-20，Spearman 任务均值 / 中位 / 前一半；代理 shared regret@3；前 3 高估）", "",
          "| E | SUR-BLIND | SUR-KNN | SUR-X | 前一半 BLIND / KNN | 代理 regret@3 BLIND / KNN | 高估 BLIND / KNN |",
          "|---|---|---|---|---|---|---|"]
    for n, sm in o["E"].items():
        if "spearman" not in sm:
            continue
        s, md, tp = sm["spearman"], sm["spearman_median"], sm["spearman_top"]
        L.append(f"| {n} | {s['SUR-BLIND']:.3f} / {md['SUR-BLIND']:.3f} | {s['SUR-KNN']:.3f} / {md['SUR-KNN']:.3f} | {s['SUR-X']:.3f} | "
                 f"{tp['SUR-BLIND']:.3f} / {tp['SUR-KNN']:.3f} | {_f(sm['sur_regret3']['SUR-BLIND'])} / {_f(sm['sur_regret3']['SUR-KNN'])} | "
                 f"{sm['overest']['SUR-BLIND']:+.3f} / {sm['overest']['SUR-KNN']:+.3f} |")
    pt = o["E"]["E_pt"]
    L += ["", "## E_pt 逐源（regret@3 源均值差 BLIND−KNN；k；近邻来源前 3）", "", "| 源 | BLIND−KNN | KNNX−KNN | k | 近邻来源 |", "|---|---|---|---|---|"]
    for s, v in pt["G1"]["per_src"].items():
        L.append(f"| {s}{' *' if s in UEA6 else ''} | {v:+.4f} | {pt['G2']['per_src'][s]:+.4f} | {pt['k'][s]} | "
                 + ", ".join(f"{g} {w}" for g, w in pt["nb_composition"][s]) + " |")
    L += ["", "（* = UEA 折源）", "", "## 预测", ""]
    L += [f"- {k}: {'✓' if v else '✗'}" for k, v in o["predictions"].items()]
    return "\n".join(L) + "\n"


def cmd_smoke() -> None:
    """合成数据跑通 run_E + summarize（4 源 × 4 任务）。"""
    rng = np.random.default_rng(0)
    G = np.repeat(["a", "b", "c", "d"], 4); n = len(G)
    Z = rng.normal(size=(n, N_SH)); e0 = rng.uniform(0.3, 0.9, size=(n, N_SH)); e1 = e0 + rng.normal(0, 0.01, size=(n, N_SH))
    D = {"groups": G, "Z": Z, "end0": e0, "end1": e1, "phi": rng.integers(0, 2, size=(n, 50, 12)).astype(float),
         "Y": rng.normal(size=(n, 50)), "R0": rng.uniform(0.3, 0.9, size=(n, 20))}
    X = rng.normal(size=(n, 5))
    tasks = [f"{g}_a{j:02d}" for g, j in zip(G, np.tile(np.arange(4), 4))]
    cm = importlib.import_module("scripts.88_acts_s3bpp").cross_maps(tasks, G, ["a", "b", "c", "d"], (1, 2))
    res = run_E(D, lambda f: X, cm, ks=(2, 3), alphas=(1.0, 10.0))
    sm = summarize(res, G)
    print("[smoke] ok", sm["verdict"], {k: round(v, 4) for k, v in sm["regret3"].items()})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        cmd_smoke()
    if args.analyze:
        cmd_analyze(Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
