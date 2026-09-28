"""WP17-ACTS S3-U（results/acts/s3u/PROTOCOL.md）：源数学习曲线，零 GPU。

每个留出源 s，从其余 16 个源随机抽 k ∈ {4, 8, 12, 16} 个作训练池（k < 16 各 30 子集），子集内算
BLIND（Z 均值）/ RAW（E 标准化余弦 10 近邻）/ ORACLE（目标种子 0 画像 Spearman 10 近邻，描述）的 regret@3；
G(k) = BLIND − RAW，C(k) = BLIND − ORACLE，源 bootstrap 给 G(16) − G(8)、C(16) − C(8)。
用法：python scripts/107_acts_s3u.py --analyze   |   --smoke
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

OUT = ROOT / "results" / "acts" / "s3u"
KS = (4, 8, 12, 16)
NSUB = 30
K_NN = 10
N_BOOT = 10000


# ================================================================ 纯函数
def source_subsets(pool: Sequence[str], k: int, n: int, fold: str) -> list:
    pool = sorted(pool)
    if int(k) >= len(pool):
        return [pool]
    from autocls.labeling import run_seed_for
    rng = np.random.default_rng(run_seed_for(fold, f"s3u_k{int(k)}"))
    return [sorted(rng.choice(pool, size=int(k), replace=False).tolist()) for _ in range(int(n))]


def verdict_s3u(*, g8: float, g12: float, g16: float, p_g: float, dc: float, p_c: float) -> str:
    rising = (g16 - g8 >= 0.002) and (g16 - g12 > 0) and (p_g >= 0.9)
    return ("S3U-RISING" if rising else "S3U-FLAT") + ("+COV" if (dc >= 0.002 and p_c >= 0.9) else "")


def _rank_rows(Z: np.ndarray) -> np.ndarray:
    R = np.apply_along_axis(rankdata, 1, np.asarray(Z, float))
    R = R - R.mean(1, keepdims=True)
    return R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)


# ================================================================ 主计算
def run_curve(D: Mapping, Xs: Mapping[str, np.ndarray], *, ks=KS, nsub: int = NSUB, k_nn: int = K_NN) -> dict:
    """返回 res[k][arm] = 逐任务 regret@3（子集均值）；arm ∈ BLIND / ORACLE / RAW:<E 名>。"""
    q = importlib.import_module("scripts.104_acts_s3q"); ac = importlib.import_module("scripts.acts_common")
    G = np.asarray(D["groups"]); Z = np.asarray(D["Z"], float); e0, e1 = D["end0"], D["end1"]
    n = len(G); srcs = list(dict.fromkeys(G.tolist()))
    RZ = _rank_rows(Z)
    R0 = _rank_rows(np.array([ac.clip_z(e0[i], 0.5)[0] for i in range(n)]))        # 目标种子 0 画像的秩（ORACLE 用）
    arms = ["BLIND", "ORACLE"] + [f"RAW:{nm}" for nm in Xs]
    res = {k: {a: np.full(n, np.nan) for a in arms} for k in ks}
    nsubs = {}
    for s in srcs:
        te = np.where(G == s)[0]; pool = [o for o in srcs if o != s]
        for k in ks:
            subs = source_subsets(pool, k, nsub, s); nsubs[(s, k)] = len(subs)
            acc = {a: np.zeros(len(te)) for a in arms}
            for sub in subs:
                tr = np.where(np.isin(G, sub))[0]
                rows = {a: [] for a in arms}
                mu = Z[tr].mean(0)
                for i in te:
                    rows["BLIND"].append(mu)
                    o = np.argsort(-(RZ[tr] @ R0[i]), kind="stable")[:k_nn]; rows["ORACLE"].append(Z[tr[o]].mean(0))
                    for nm, X in Xs.items():
                        o = np.argsort(-q.std_cos(X[tr], X[i]), kind="stable")[:k_nn]; rows[f"RAW:{nm}"].append(Z[tr[o]].mean(0))
                for a in arms:
                    acc[a] += q._regret(np.array(rows[a]), e0[te], e1[te], 3)
            for a in arms:
                res[k][a][te] = acc[a] / len(subs)
        print(f"[s3u] {s[:18]:18s} " + "  ".join(f"k{k} G {np.mean(res[k]['BLIND'][te] - res[k]['RAW:E_pt'][te]) if 'RAW:E_pt' in res[k] else 0:+.4f}" for k in ks), flush=True)
    return {"res": res, "nsub": {f"{s}|{k}": v for (s, k), v in nsubs.items()}, "srcs": srcs}


def summarize(out: Mapping, G: np.ndarray, *, e_main: str = "E_pt", n_boot: int = N_BOOT, seed: int = 0) -> dict:
    q = importlib.import_module("scripts.104_acts_s3q")
    res, srcs = out["res"], out["srcs"]; ks = sorted(res)
    per = {k: {a: np.array([np.nanmean(v[G == s]) for s in srcs]) for a, v in res[k].items()} for k in ks}
    curve = {k: {a: float(v.mean()) for a, v in per[k].items()} for k in ks}
    Gk = {k: per[k]["BLIND"] - per[k][f"RAW:{e_main}"] for k in ks}
    Ck = {k: per[k]["BLIND"] - per[k]["ORACLE"] for k in ks}
    rng = np.random.default_rng(seed); m = len(srcs)
    bg, bc = [], []
    for _ in range(int(n_boot)):
        j = rng.integers(0, m, size=m)
        bg.append(Gk[16][j].mean() - Gk[8][j].mean()); bc.append(Ck[16][j].mean() - Ck[8][j].mean())
    bg, bc = np.array(bg), np.array(bc)
    g = {k: float(Gk[k].mean()) for k in ks}; c = {k: float(Ck[k].mean()) for k in ks}
    uea = [j for j, s in enumerate(srcs) if s in q.UEA6]
    o = {"curve": {str(k): v for k, v in curve.items()}, "G": {str(k): v for k, v in g.items()}, "C": {str(k): v for k, v in c.items()},
         "G_other": {nm.split(":", 1)[1]: {str(k): float((per[k]["BLIND"] - per[k][nm]).mean()) for k in ks} for nm in res[ks[0]] if nm.startswith("RAW:")},
         "G16_minus_G8": {"d": g[16] - g[8], "p_pos": float(np.mean(bg > 0)), "ci": [float(np.quantile(bg, 0.025)), float(np.quantile(bg, 0.975))]},
         "C16_minus_C8": {"d": c[16] - c[8], "p_pos": float(np.mean(bc > 0)), "ci": [float(np.quantile(bc, 0.025)), float(np.quantile(bc, 0.975))]},
         "uea6": {"G": {str(k): float(Gk[k][uea].mean()) for k in ks}, "C": {str(k): float(Ck[k][uea].mean()) for k in ks},
                  "BLIND": {str(k): float(per[k]["BLIND"][uea].mean()) for k in ks}},
         "per_src_G": {s: {str(k): float(Gk[k][j]) for k in ks} for j, s in enumerate(srcs)}}
    o["verdict"] = verdict_s3u(g8=g[8], g12=g[12], g16=g[16], p_g=o["G16_minus_G8"]["p_pos"], dc=c[16] - c[8], p_c=o["C16_minus_C8"]["p_pos"])
    return o


def cmd_analyze(outdir: Path = OUT) -> dict:
    q = importlib.import_module("scripts.104_acts_s3q")
    D = q.load_data(); G = np.asarray(D["groups"])
    Es = q.load_E(D["tasks"])
    out = run_curve(D, {"E_pt": Es["E_pt"](None), "E_TF": Es["E_TF"](None)})
    o = summarize(out, G)
    b = o["curve"]
    o["predictions"] = {
        "P1": o["verdict"].startswith("S3U-RISING"),
        "P2": o["verdict"].endswith("+COV"),
        "P3": o["G"]["16"] - o["G"]["12"] > 0,
        "P4": abs(b["16"]["BLIND"] - b["8"]["BLIND"]) < 0.002,
        "P5": o["uea6"]["G"]["16"] > o["uea6"]["G"]["8"],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: o[k] for k in ("verdict", "G", "C", "G_other", "G16_minus_G8", "C16_minus_C8", "uea6", "predictions")}, ensure_ascii=False, indent=1, default=float))
    print("curve", json.dumps(o["curve"], indent=1, default=float))
    return o


def cmd_smoke() -> None:
    rng = np.random.default_rng(0)
    G = np.repeat(np.array([f"s{j}" for j in range(17)]), 2); n = len(G)
    D = {"groups": G, "Z": rng.normal(size=(n, 30)), "end0": rng.uniform(0.3, 0.9, size=(n, 30))}
    D["end1"] = D["end0"] + rng.normal(0, 0.01, size=(n, 30))
    out = run_curve(D, {"E_pt": rng.normal(size=(n, 8))}, nsub=3, k_nn=3)
    print("[smoke] ok", summarize(out, G, n_boot=200)["verdict"])


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
