"""WP17-ACTS S3-P（results/acts/s3p/PROTOCOL.md）：任务盲集成 B′ 搜索 → 候选池（最终种群 ∪ 分块前 20）→ 带 E 的比较器重排取前 3。

阶段：
  --search        服务器：--fold → search/<fold>.json（B′-pool / RR-EP / RR-EPX / RR-A′）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1；去重至 S3-M）
  --analyze       本地
  --smoke         本地冒烟
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3p"
PREV = ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h", "s3i", "s3j", "s3l", "s3m")
N_CHUNK = 20
ARMS = ("B′-pool", "RR-EP", "RR-EPX", "RR-A′")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def build_pool(final: Sequence[str], chunk: Sequence[str], n_chunk: int = N_CHUNK) -> list[str]:
    """最终种群（按 B′ 适应度序）∪ 分块筛选名单前 n_chunk，去重保序。"""
    return list(dict.fromkeys(list(final) + list(chunk)[:n_chunk]))


def rerank(W: np.ndarray, keys: Sequence[str], lam: float = 1.0, k: int = 3) -> list[str]:
    o = _g("91_acts_s3e").ens_order(W, list(keys), lam)
    return [keys[i] for i in o[:k]]


def verdict_s3p(dR_m3: Mapping, dR_p1: Mapping, dX_m3: Mapping) -> str:
    if dR_m3["d"] >= 0.010 and dR_m3["p"] < 0.05 and dR_m3["npos"] >= 4 and dR_p1["d"] > 0:
        code = "S3P-E-WIN"
    elif dR_m3["d"] <= -0.010 and dR_m3["p"] < 0.05:
        code = "S3P-E-HURT"
    else:
        code = "S3P-E-NULL"
    if dX_m3["d"] >= 0.005 and dX_m3["p"] < 0.05 and dX_m3["npos"] >= 4:
        code += "+CONTENT"
    return code


# ================================================================ 搜索与重排（服务器）
def _search_pool(ens, Ts, t: str, space, enc, cfg, ks: int, gens: int, lam: float) -> tuple[list, list]:
    """= 92._search_task 的单模式流程，额外返回：进化最终种群（按适应度序）与分块 400 条（按适应度序）。"""
    ac, g89, g91 = _g("acts_common"), _g("89_acts_s3c"), _g("91_acts_s3e")
    from autocls.labeling import run_seed_for
    cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(t, "s3c_ks")), ks, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
    fin = []
    for s in range(0, len(cands), g89.CHUNK):
        idx = list(range(s, min(len(cands), s + g89.CHUNK)))
        o = g91.ens_order(ens.wins(Vc[idx], Ts), [keys[i] for i in idx], lam)
        fin += [idx[j] for j in o[:g89.PER_CHUNK]]
    o = g91.ens_order(ens.wins(Vc[fin], Ts), [keys[i] for i in fin], lam)
    chunk_ranked = [cands[fin[j]][0] for j in o]
    final = g91._evolve(ens, Ts, space, enc, cfg, chunk_ranked[:g89.KP], np.random.default_rng(run_seed_for(t, "s3c_evo")), lam, g89.KP, gens, g89.P_CROSS, g89.P_MUT)
    return final, chunk_ranked


def cmd_search(args) -> None:
    import torch
    g89, g91, g92, g93, g94, g96 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g", "96_acts_s3i"))
    from autocls.labeling import cls_key
    meta, arr, cfg, space, enc = g94._ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS; n = len(seeds)
    ens_b = g94._ens("D", [Path(args.s3g) / "models" / f"blind_s{sd}_{fold}.pt" for sd in seeds], enc.dim, dev)
    with np.load(Path(args.s3m) / "axes.npz") as z:
        C_ep = torch.as_tensor(np.asarray(z[f"{fold}/EP_fit"]), dtype=torch.float32, device=dev)
    K = C_ep.shape[1]
    ms = []
    for sd in seeds:
        m = g96.AxisComparator(enc.dim, k=K).to(dev); m.load_state_dict(torch.load(Path(args.s3m) / "models" / f"EP_s{sd}_{fold}.pt", map_location=dev)); m.eval(); ms.append(m)
    ens_ep = g96.AxisEnsemble(ms, dev)
    ms, Ah = [], []
    for sd in seeds:
        m = g96.AxisComparator(enc.dim).to(dev); m.load_state_dict(torch.load(Path(args.s3j) / "models" / f"I_s{sd}_{fold}.pt", map_location=dev)); m.eval(); ms.append(m)
        with np.load(Path(args.s3j) / "train" / f"I_s{sd}_{fold}.npz") as z:
            Ah.append(torch.as_tensor(z["Ahat"], dtype=torch.float32, device=dev))
    ens_a = g96.AxisEnsemble(ms, dev)
    tasks = meta["tasks"]; groups = meta["groups"]; cx = g92.cross_index(tasks, groups, meta["order"])
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    res, t0 = {}, time.time()
    for t in g93.exam_tasks(tasks, groups, [fold]):
        i = tasks.index(t)
        final, chunk = _search_pool(ens_b, [None] * n, t, space, enc, cfg, ks, gens, g91.LAM)
        byk = {cls_key(c): c for c in list(final) + list(chunk)}
        pool = build_pool([cls_key(c) for c in final], [cls_key(c) for c in chunk], N_CHUNK)
        Vp = np.stack([enc.encode(byk[k]) for k in pool]).astype(np.float32)
        picks = {"B′-pool": [cls_key(c) for c in final[:3]],
                 "RR-EP": rerank(ens_ep.wins(Vp, [C_ep[i]] * n), pool),
                 "RR-EPX": rerank(ens_ep.wins(Vp, [C_ep[cx[i]]] * n), pool),
                 "RR-A′": rerank(ens_a.wins(Vp, [A[i] for A in Ah]), pool)}
        res[t] = {"pool": pool, "x_task": tasks[cx[i]], **{a: [{"cls_key": k, "cls": byk[k].to_dict()} for k in ks_] for a, ks_ in picks.items()}}
        print(f"[search] {fold} {t} pool {len(pool)} " + " ".join(f"{a} {v}" for a, v in picks.items()) + f" {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def _dirs(args) -> list[Path]:
    return [Path(getattr(args, k)) for k in PREV] + [Path(args.outdir)]


def cmd_plan_runs(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in ARMS}
            for a in ARMS:
                byk.update({e["cls_key"]: e["cls"] for e in rec[a]})
    _g("94_acts_s3g")._plan(args, noms, byk, (0, 1), "runs", dirs=_dirs(args))


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, wilcoxon
    g84, g89, g93, g94 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "93_acts_s3f", "94_acts_s3g"))
    outdir = Path(args.outdir); P_ = {k: Path(getattr(args, k)) for k in PREV}
    meta, arr = g89.load_data(P_["s3c"]); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]
    val = g94._val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    sp = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}; sgg = {s: js(P_["s3g"] / "search" / f"{s}.json") for s in g89.FOLDS}
    arms = ARMS + ("B′-lcb(S3-G)",)
    noms = {t: {**{a: sp[s][t][a] for a in ARMS}, "B′-lcb(S3-G)": sgg[s][t]["B′-lcb"]} for t, s in zip(ex, sg)}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    cat_p1, cat_m3, overlap = [], [], {a: 0 for a in ARMS}
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i]))
        base = {x["cls_key"] for x in noms[t]["B′-pool"]}
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
            if a in overlap:
                overlap[a] += len(base & set(ks_))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"] = np.array(cat_p1), np.array(cat_m3)

    def cmp(x, y):
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    pairs = (("RR-EP", "B′-pool"), ("RR-EP", "RR-EPX"), ("RR-A′", "B′-pool"), ("RR-EP", "cat3"), ("B′-pool", "cat3"), ("B′-pool", "B′-lcb(S3-G)"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    code = verdict_s3p(C["RR-EP−B′-pool"]["m3"], C["RR-EP−B′-pool"]["p1"], C["RR-EP−RR-EPX"]["m3"])
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "missing": miss, "overlap_with_B′pool_top3": {a: v / (3 * len(ex)) for a, v in overlap.items()}, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-P 读数（{code}）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 | 与 B′-pool 前 3 重合 |", "|---|---:|---:|---:|---:|"]
    for a in list(arms) + ["cat3"]:
        ov = out["overlap_with_B′pool_top3"].get(a)
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'][a])} | {miss.get(a, 0)} | {'—' if ov is None else f'{100 * ov:.0f}%'} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89, g92, g100 = _g("89_acts_s3c"), _g("92_acts_s3ea"), _g("100_acts_s3m")
    od = Path(args.outdir); sg, sm = od / "_s3g", od / "_s3m"
    for d in (sg, sm):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results/acts/s3m/axes.npz", sm / "axes.npz")
    fold = g89.FOLDS[0]
    base = dict(vars(args)); base.update({"max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "seeds": [0, 1], "synthetic": True, "s3g": str(sg), "s3m": str(sm)})
    for sd in (0, 1):
        g92.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sg), "fold": fold, "seed": sd}))
        g100.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sm), "kind": "EP", "fold": fold, "seed": sd}))
    ns = types.SimpleNamespace(**{**base, "fold": fold}); cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in ARMS) and all(len(v["pool"]) >= 10 for v in r.values())
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    for k in PREV:
        ap.add_argument(f"--{k}", default=str(ROOT / "results" / "acts" / k))
    ap.add_argument("--s3b", default=str(ROOT / "results" / "acts" / "s3b"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("smoke", cmd_smoke), ("search", cmd_search), ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
