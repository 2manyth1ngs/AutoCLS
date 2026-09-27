"""WP17-ACTS S3-S（results/acts/s3s/PROTOCOL.md）：偏好对齐的 E 检索，零 GPU。

每个外层折只用训练源学线性投影 W：跨源软近邻 p_ij = softmax_j(−‖u_i − u_j‖²)（j 限其他源），
损失 = −mean_i Σ_j p_ij s_ij + λ‖W‖²，s_ij = Z 画像 Spearman；(r, λ) 内层留一个源按 regret@3 选。
留出任务投影后取 10 个最近训练任务，Z 均值作打分；对照 RAW（原始余弦 10 近邻 = S3-Q KNN10）/ BLIND / ORACLE。
用法：python scripts/106_acts_s3s.py --analyze   |   --smoke
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

from scipy.stats import rankdata  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3s"
K = 10
RS = (8, 32)
LAMS = (1e-3, 1e-2, 1e-1)
STEPS, LR_W = 300, 0.01


# ================================================================ 纯函数
def pref_sim(Z: np.ndarray) -> np.ndarray:
    """行间 Spearman 矩阵（秩的 Pearson）。"""
    R = np.apply_along_axis(rankdata, 1, np.asarray(Z, float))
    R = R - R.mean(1, keepdims=True)
    R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    return R @ R.T


def fit_align(X: np.ndarray, S: np.ndarray, groups: Sequence, *, r: int, lam: float, steps: int = STEPS, seed: int = 0,
              lr: float = LR_W) -> np.ndarray:
    """学 W（d × r）：软近邻只在其他源里取，最大化检索分布下的期望偏好相似度。X 须已标准化。"""
    import torch
    X = np.asarray(X, float); groups = np.asarray(groups)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    r = int(min(r, Vt.shape[0]))
    W0 = Vt[:r].T / np.sqrt(r)                                   # 主成分初始化
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    St = torch.as_tensor(np.asarray(S, float), dtype=torch.float32)
    mask = torch.as_tensor(groups[:, None] != groups[None, :])
    W = torch.as_tensor(W0, dtype=torch.float32).clone().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(int(steps)):
        U = Xt @ W
        sq = (U * U).sum(1)
        Dm = sq[:, None] + sq[None, :] - 2.0 * U @ U.T               # 平方欧氏距离（避开 cdist 在 0 处的梯度）
        P = torch.softmax((-Dm).masked_fill(~mask, float("-inf")), dim=1)
        loss = -(P * St).sum(1).mean() + float(lam) * (W * W).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().numpy().astype(float)


def knn_dist(Utr: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    d = ((np.asarray(Utr, float) - np.asarray(u, float)) ** 2).sum(1)
    return np.argsort(d, kind="stable")[:int(k)]


def choose_hp(vals: Mapping[tuple, float]) -> tuple:
    best = min(vals.values())
    c = [hp for hp, v in vals.items() if v <= best + 1e-12]
    return sorted(c, key=lambda x: (-x[1], x[0]))[0]


def verdict_s3s(g1: Mapping, g2: Mapping) -> str:
    if not (g1["d"] >= 0.002 and g1["p"] < 0.05):
        return "S3S-ALIGN-NULL"
    return "S3S-ALIGN-WIN" if (g2["d"] >= 0.003 and g2["p"] < 0.05) else "S3S-ALIGN-HELPS"


def _std(Xtr: np.ndarray, X: np.ndarray) -> np.ndarray:
    m, sd = Xtr.mean(0), Xtr.std(0)
    return (X - m) / np.where(sd > 1e-12, sd, 1.0)


# ================================================================ 主计算
def run_E(D: Mapping, X: np.ndarray, *, rs=RS, lams=LAMS, steps: int = STEPS, k: int = K, oracle: bool = True) -> dict:
    q = importlib.import_module("scripts.104_acts_s3q"); ac = importlib.import_module("scripts.acts_common")
    G = np.asarray(D["groups"]); Z, e0, e1 = np.asarray(D["Z"], float), D["end0"], D["end1"]
    X = np.asarray(X, float); n = len(G)
    S_all = pref_sim(Z)
    arms = ("BLIND", "RAW", "ALIGN", "ORACLE")
    r3 = {a: np.full(n, np.nan) for a in arms}
    nbsim = {a: np.full(n, np.nan) for a in ("RAW", "ALIGN", "ORACLE")}
    hps, inner_tab = {}, {}

    def reg(rows, idx):
        return q._regret(np.array(rows), e0[idx], e1[idx], 3)

    for s in dict.fromkeys(G.tolist()):
        te = np.where(G == s)[0]; tr = np.where(G != s)[0]
        vals = {}
        for r in rs:
            for lam in lams:
                regs = []
                for u in dict.fromkeys(G[tr].tolist()):
                    tu = tr[G[tr] != u]; eu = tr[G[tr] == u]
                    Xu = _std(X[tu], X)
                    W = fit_align(Xu[tu], S_all[np.ix_(tu, tu)], G[tu], r=r, lam=lam, steps=steps)
                    U = Xu @ W
                    regs.append(float(np.mean(reg([Z[tu[knn_dist(U[tu], U[i], k)]].mean(0) for i in eu], eu))))
                vals[(r, lam)] = float(np.mean(regs))
        hp = choose_hp(vals); hps[s] = hp; inner_tab[s] = {f"{a}_{b}": v for (a, b), v in vals.items()}
        Xs = _std(X[tr], X)
        W = fit_align(Xs[tr], S_all[np.ix_(tr, tr)], G[tr], r=hp[0], lam=hp[1], steps=steps)
        U = Xs @ W
        rowsA, rowsR, rowsO = [], [], []
        for i in te:
            nb = tr[knn_dist(U[tr], U[i], k)]; rowsA.append(Z[nb].mean(0)); nbsim["ALIGN"][i] = S_all[i, nb].mean()
            o = np.argsort(-q.std_cos(X[tr], X[i]), kind="stable")[:k]; nr_ = tr[o]
            rowsR.append(Z[nr_].mean(0)); nbsim["RAW"][i] = S_all[i, nr_].mean()
            if oracle:
                zi = ac.clip_z(e0[i], 0.5)[0]
                no = tr[np.argsort(-q.rank_sim(zi, Z[tr]), kind="stable")[:k]]
                rowsO.append(Z[no].mean(0)); nbsim["ORACLE"][i] = S_all[i, no].mean()
        r3["ALIGN"][te] = reg(rowsA, te); r3["RAW"][te] = reg(rowsR, te)
        r3["BLIND"][te] = reg([Z[tr].mean(0)] * len(te), te)
        if oracle:
            r3["ORACLE"][te] = reg(rowsO, te)
        print(f"[s3s] {s[:18]:18s} hp r={hp[0]} λ={hp[1]:g}  ALIGN {np.mean(r3['ALIGN'][te]):.4f} RAW {np.mean(r3['RAW'][te]):.4f} "
              f"BLIND {np.mean(r3['BLIND'][te]):.4f}", flush=True)
    return {"r3": r3, "nbsim": nbsim, "hp": hps, "inner": inner_tab}


def summarize(res: Mapping, G: np.ndarray) -> dict:
    q = importlib.import_module("scripts.104_acts_s3q")
    r3 = res["r3"]
    out = {"regret3": {a: float(np.nanmean(v)) for a, v in r3.items()},
           "regret3_src": {a: float(np.nanmean([np.nanmean(v[G == s]) for s in dict.fromkeys(G.tolist())])) for a, v in r3.items()},
           "nbsim": {a: float(np.nanmean(v)) for a, v in res["nbsim"].items()},
           "G1": q.compare(r3["RAW"] - r3["ALIGN"], G), "G2": q.compare(r3["BLIND"] - r3["ALIGN"], G),
           "raw_vs_blind": q.compare(r3["BLIND"] - r3["RAW"], G),
           "hp": {s: list(v) for s, v in res["hp"].items()}}
    u = np.isin(G, q.UEA6)
    out["uea6"] = {a: float(np.nanmean(v[u])) for a, v in r3.items()}
    out["verdict"] = verdict_s3s(out["G1"], out["G2"])
    return out


def cmd_analyze(outdir: Path = OUT) -> dict:
    q = importlib.import_module("scripts.104_acts_s3q")
    D = q.load_data(); G = D["groups"]; tasks = D["tasks"]
    Es = q.load_E(tasks)
    pt = Es["E_pt"](None); tf = Es["E_TF"](None)
    variants = {"E_pt": pt, "E_pt+E_TF": np.c_[pt, tf]}
    o: dict = {"n_tasks": len(G), "k": K, "E": {}}
    for name, X in variants.items():
        print(f"[s3s] ===== {name} =====", flush=True)
        res = run_E(D, X, oracle=(name == "E_pt"))
        sm = summarize(res, G); o["E"][name] = sm
        print(f"[s3s] {name}: {sm['verdict']}  ALIGN {sm['regret3']['ALIGN']:.4f} RAW {sm['regret3']['RAW']:.4f} BLIND {sm['regret3']['BLIND']:.4f}"
              f"  G1 {sm['G1']['d']:+.4f} p {sm['G1']['p']:.3f}  G2 {sm['G2']['d']:+.4f} p {sm['G2']['p']:.3f}", flush=True)
    pt_ = o["E"]["E_pt"]
    o["verdict"] = pt_["verdict"]
    o["c0_raw_equals_s3q_knn10"] = pt_["regret3"]["RAW"]
    o["predictions"] = {
        "P1": pt_["G1"]["d"] >= 0.002 and pt_["G1"]["p"] < 0.05,
        "P2": pt_["G2"]["d"] >= 0.003 and pt_["G2"]["p"] < 0.05,
        "P3": pt_["nbsim"]["ALIGN"] > pt_["nbsim"]["RAW"],
        "P4": pt_["uea6"]["ALIGN"] <= pt_["uea6"]["RAW"],
        "P5": sum(v[0] == 8 for v in pt_["hp"].values()) >= 9,
        "P6": o["E"]["E_pt+E_TF"]["regret3"]["ALIGN"] <= pt_["regret3"]["ALIGN"],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[s3s] verdict {o['verdict']}  predictions {sum(o['predictions'].values())}/6", flush=True)
    return o


def cmd_smoke() -> None:
    rng = np.random.default_rng(0)
    G = np.repeat(np.array(["a", "b", "c", "d"]), 4); n = len(G)
    D = {"groups": G, "Z": rng.normal(size=(n, 30)), "end0": rng.uniform(0.3, 0.9, size=(n, 30))}
    D["end1"] = D["end0"] + rng.normal(0, 0.01, size=(n, 30))
    res = run_E(D, rng.normal(size=(n, 12)), rs=(2,), lams=(1e-2,), steps=20, k=3)
    print("[smoke] ok", summarize(res, G)["verdict"])


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
