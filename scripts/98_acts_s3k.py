"""WP17-ACTS S3-K（results/acts/s3k/PROTOCOL.md）：需要多准的坐标——S3-J 的 I′ 集成在 8 种坐标条件下的排序读数（零 GPU）。

  --analyze   本地（先把服务器 /tmp/acts_s3j_out/models 拉到 results/acts/s3j/models）
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

OUT = ROOT / "results" / "acts" / "s3k"
DIRS = ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h", "s3i", "s3j")
CONDS = ("Z", "P", "P1", "T1", "T1+P23", "P1+T23", "T12", "T")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def coords(cond: str, a: np.ndarray, ah: np.ndarray) -> np.ndarray:
    a, ah = np.asarray(a, float), np.asarray(ah, float)
    z = np.zeros(3)
    return {"Z": z, "P": ah, "P1": np.array([ah[0], 0, 0]), "T1": np.array([a[0], 0, 0]), "T1+P23": np.array([a[0], ah[1], ah[2]]),
            "P1+T23": np.array([ah[0], a[1], a[2]]), "T12": np.array([a[0], a[1], 0]), "T": a}[cond]


def pool_regret(wins: np.ndarray, v0: np.ndarray, v1: np.ndarray, k: int = 3) -> float:
    """胜场前 k（并列按下标）→ seed 0 选 1 → 报 seed 1；oracle = 池内 seed 0 最好者的 seed 1。"""
    wins, v0, v1 = np.asarray(wins, float), np.asarray(v0, float), np.asarray(v1, float)
    top = sorted(range(len(wins)), key=lambda j: (-wins[j], j))[:k]
    pick = top[int(np.argmax(v0[top]))]
    return float(v1[int(np.argmax(v0))] - v1[pick])


def realized(z: float, c: float, t: float, lower_better: bool) -> float:
    if abs(t - z) < 1e-12:
        return float("nan")
    return float((z - c) / (z - t)) if lower_better else float((c - z) / (t - z))


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    import torch
    from scipy.stats import spearmanr, wilcoxon
    from autocls.search.rank import symmetrize, wins_from_scores
    from autocls.space.cls_space import CLS
    g84, g89, g91, g93, g94, g96 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "91_acts_s3e", "93_acts_s3f", "94_acts_s3g", "96_acts_s3i"))
    D = {k: ROOT / "results" / "acts" / k for k in DIRS}
    meta, arr, _cfg, _space, enc = g94._ctx(argparse.Namespace(s3c=str(D["s3c"])))
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    val = g94._val_map([D[k] for k in DIRS])
    cls_of = {}
    for d in [ROOT / "results" / "acts" / "s2"] + [D[k] for k in DIRS]:
        for sub, name in (("", "plan_S.json"), ("runs", "plan_X.json"), ("runs_aug", "plan_X.json")):
            p = (d / sub / name) if sub else (d / name)
            if p.exists():
                for c in json.loads(p.read_text(encoding="utf-8"))["cells"]:
                    cls_of[(c["task_id"], c["cls_key"])] = c["cls"]
    dev = "cpu"
    ens = {}
    for s in g89.FOLDS:
        ms, ah = [], []
        for sd in g91.ENS_SEEDS:
            m = g96.AxisComparator(enc.dim); m.load_state_dict(torch.load(D["s3j"] / "models" / f"I_s{sd}_{s}.pt", map_location=dev)); m.eval(); ms.append(g96.AxisScorer(m, dev))
            with np.load(D["s3j"] / "train" / f"I_s{sd}_{s}.npz") as z:
                ah.append(np.asarray(z["Ahat"], float))
        with np.load(D["s3e"] / "axis_targets.npz") as z:
            at = np.asarray(z[f"aux/{s}"], float)[:, :3]
        ens[s] = (ms, ah, at)

    def wins(i: int, V: np.ndarray, cond: str) -> np.ndarray:
        ms, ah, at = ens[groups[i]]; W = []
        for sc, A in zip(ms, ah):
            T = torch.as_tensor(coords(cond, at[i], A[i]), dtype=torch.float32)
            P = sc.pmat(V, T); np.fill_diagonal(P, 0.5)
            W.append(wins_from_scores(symmetrize(P)).astype(float))
        return np.mean(W, axis=0)

    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); eg = np.array([groups[tasks.index(t)] for t in ex])
    R1 = {c: np.zeros(len(held)) for c in CONDS}; R3 = {c: np.full(len(held), np.nan) for c in CONDS}
    for jj, i in enumerate(held):
        t = tasks[i]; Vs = arr[f"{t}/vec"][:N]; R0 = arr[f"{t}/R0"]; Vr = arr[f"{t}/vec"][N:N + len(R0)]
        for c in CONDS:
            R1[c][jj] = pool_regret(wins(i, Vs, c), end0[i], end1[i])
            if len(R0) >= 10:
                R3[c][jj] = float(spearmanr(wins(i, Vr, c), R0)[0])
    R2, R2s, npool = {c: np.zeros(len(ex)) for c in CONDS}, {c: np.zeros(len(ex)) for c in CONDS}, []
    for jj, t in enumerate(ex):
        i = tasks.index(t)
        keys = sorted({k for (tt, k, s) in val if tt == t and s == 0 and (t, k, 1) in val and (t, k) in cls_of})
        V = np.stack([enc.encode(CLS(**cls_of[(t, k)])) for k in keys]).astype(np.float32)
        v0 = np.array([val[(t, k, 0)] for k in keys]); v1 = np.array([val[(t, k, 1)] for k in keys]); npool.append(len(keys))
        for c in CONDS:
            w = wins(i, V, c)
            R2[c][jj] = pool_regret(w, v0, v1); R2s[c][jj] = float(spearmanr(w, (v0 + v1) / 2)[0])
    summ = {c: {"R1_regret3": float(R1[c].mean()), "R2_regret3": float(R2[c].mean()), "R2_sp": float(np.mean(R2s[c])), "R3_sp": float(np.nanmedian(R3[c]))} for c in CONDS}
    real = {c: {"R1": realized(summ["Z"]["R1_regret3"], summ[c]["R1_regret3"], summ["T"]["R1_regret3"], True),
                "R2": realized(summ["Z"]["R2_regret3"], summ[c]["R2_regret3"], summ["T"]["R2_regret3"], True),
                "R2_sp": realized(summ["Z"]["R2_sp"], summ[c]["R2_sp"], summ["T"]["R2_sp"], False),
                "R3": realized(summ["Z"]["R3_sp"], summ[c]["R3_sp"], summ["T"]["R3_sp"], False)} for c in CONDS}

    def pair(a, b, R, grp):
        d = R[a] - R[b]
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "ci": g84.cluster_bootstrap_ci(d, grp)}
    pairs = {f"{a}−{b}": {"R2_regret": pair(a, b, R2, eg), "R2_sp": pair(a, b, R2s, eg), "R1_regret": pair(a, b, R1, hg)}
             for a, b in (("T1", "T"), ("T12", "T"), ("T1+P23", "T1"), ("P1", "P"), ("P1+T23", "P"), ("T1", "P"), ("P", "Z"))}
    rT1, rT12 = real["T1"]["R2"], real["T12"]["R2"]
    if rT1 >= 0.70 and (rT12 - rT1) < 0.20:
        code = "S3K-A1-DOMINANT"
    elif rT12 >= 0.80:
        code = "S3K-A1A2"
    else:
        code = "S3K-ALL3"
    per_src = {c: {s: float(R2[c][eg == s].mean()) for s in g89.FOLDS} for c in CONDS}
    out = {"verdict": code, "summary": summ, "realized": real, "pairs": pairs, "pool_size_median": float(np.median(npool)), "per_source_R2": per_src}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    pc = lambda x: "—" if not np.isfinite(x) else f"{100 * x:.0f}%"  # noqa: E731
    L = [f"# S3-K 读数（{code}）", "", f"候选池（48 任务）中位 {out['pool_size_median']:.0f} 条（shared-30 + 两种子都有真值的提名）", "",
         "| 条件 | R1 目录 regret@3 | R2 池内 regret@3 | R2 池内 Sp | R3 random-20 Sp | 兑现 R1 | 兑现 R2 | 兑现 R2 Sp | 兑现 R3 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDS:
        s, r = summ[c], real[c]
        L.append(f"| {c} | {f(s['R1_regret3'])} | {f(s['R2_regret3'])} | {f(s['R2_sp'])} | {f(s['R3_sp'])} | {pc(r['R1'])} | {pc(r['R2'])} | {pc(r['R2_sp'])} | {pc(r['R3'])} |")
    L += ["", "| 配对 | R2 regret 差 [CI] p | R2 Sp 差 p | R1 regret 差 p |", "|---|---|---|---|"]
    for k, v in pairs.items():
        L.append(f"| {k} | {v['R2_regret']['d']:+.4f} [{f(v['R2_regret']['ci'][0])}, {f(v['R2_regret']['ci'][1])}] {v['R2_regret']['p']:.3f} | {v['R2_sp']['d']:+.3f} {v['R2_sp']['p']:.3f} | {v['R1_regret']['d']:+.4f} {v['R1_regret']['p']:.3f} |")
    L += ["", "逐源 R2 池内 regret@3：", "", "| 源 | " + " | ".join(CONDS) + " |", "|---|" + "---:|" * len(CONDS)]
    for s_ in g89.FOLDS:
        L.append(f"| {s_} | " + " | ".join(f(per_src[c][s_]) for c in CONDS) + " |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
