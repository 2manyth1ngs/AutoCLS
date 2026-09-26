"""WP17-ACTS S3-L（results/acts/s3l/PROTOCOL.md）：E → 坐标映射的筛选（零 GPU）+ M★ 换进 S3-J 的 I′ 上一次搜索。

阶段：
  --screen        本地：8 个候选（4 种 E × MSE / DS 映射）→ 训练侧下游读数选 M★ → 留出 R1 / R2 / R3 → ahat.npz（M★ 各成员坐标）
  --search        服务器：--fold → search/<fold>.json（A″-lcb / AX″-lcb，I′ 模型不重训）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1）
  --analyze       本地
  --smoke         本地冒烟（只走搜索路径）
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3l"
PREV = ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h", "s3i", "s3j")
E_VARS = ("sh4", "sh8", "pt4", "pt4+sh4")
MAPS = ("MSE", "DS")
ALPHAS = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
EXAM = ("A″-lcb", "AX″-lcb")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def wins_linear(H: np.ndarray, c: np.ndarray) -> np.ndarray:
    """轴瓶颈比较器按构造 P(a≻b) = σ(s_a − s_b)，s = h·[1, c]；Round-Robin 胜场 = 严格低于自己的条数。"""
    s = np.asarray(H, float)[:, 0] + np.asarray(H, float)[:, 1:] @ np.asarray(c, float)
    return (s[None, :] < s[:, None]).sum(1)


def calib_slope(a: np.ndarray, ah: np.ndarray) -> float:
    """过原点回归斜率 Σ a·â / Σ â²，截断到 [0, sd(a)/sd(â)]（校准后幅度不超过真实坐标）。"""
    a, ah = np.asarray(a, float), np.asarray(ah, float)
    sd = ah.std()
    if sd < 1e-12:
        return 0.0
    return float(np.clip((a * ah).sum() / (ah * ah).sum(), 0.0, a.std() / sd))


def select_best(crit: Mapping[tuple, float]) -> tuple:
    best = max(crit.values())
    return min(k for k, v in crit.items() if v >= best - 1e-12)


def verdict_s3l(dA_m3: Mapping, dA_p1: Mapping, dM_m3: Mapping, dX_m3: Mapping) -> str:
    sfx = lambda m3: m3["d"] >= 0.005 and m3["p"] < 0.05 and m3["npos"] >= 4  # noqa: E731
    if dA_m3["d"] >= 0.010 and dA_m3["p"] < 0.05 and dA_m3["npos"] >= 4 and dA_p1["d"] > 0:
        code = "S3L-E-WIN"
    elif dA_m3["d"] <= -0.010 and dA_m3["p"] < 0.05:
        code = "S3L-E-HURT"
    else:
        code = "S3L-E-NULL"
    return code + ("+MAP" if sfx(dM_m3) else "") + ("+CONTENT" if sfx(dX_m3) else "")


# ================================================================ E 特征与映射
def e_features(var: str, fold: str, groups: np.ndarray, em_sh: Mapping, em_pt: np.ndarray) -> np.ndarray:
    g87 = _g("87_acts_s3bp"); tr = groups != fold

    def pca(E, r):
        f = g87.pca_fit(E[tr], r); Z = g87.pca_apply(f, E); m, s = Z[tr].mean(0), Z[tr].std(0)
        return (Z - m) / np.where(s > 1e-12, s, 1.0)
    return {"sh4": lambda: pca(em_sh[fold], 4), "sh8": lambda: pca(em_sh[fold], 8), "pt4": lambda: pca(em_pt, 4),
            "pt4+sh4": lambda: np.hstack([pca(em_pt, 4), pca(em_sh[fold], 4)])}[var]()


def _ridge(U, A, rows, alphas_k):
    g84 = _g("84_acts_s3a")
    return [g84.ridge_fit(U[rows], A[rows, k], alphas_k[k]) for k in range(A.shape[1])]


def _pred(models, U):
    g84 = _g("84_acts_s3a")
    return np.stack([g84.ridge_predict(m, U) for m in models], 1)


def _boot_rows(groups: np.ndarray, tr: np.ndarray, seed: Optional[int]) -> np.ndarray:
    idx = np.where(tr)[0]
    if seed is None:
        return idx
    srcs = np.array(sorted(set(groups[idx].tolist())))
    pick = np.random.default_rng(int(seed)).choice(srcs, len(srcs), replace=True)
    return np.concatenate([idx[groups[idx] == s] for s in pick])


# ================================================================ 筛选（本地）
def _load_heads(fold: str, enc):
    import torch
    g91, g96 = _g("91_acts_s3e"), _g("96_acts_s3i")
    ms = []
    for sd in g91.ENS_SEEDS:
        m = g96.AxisComparator(enc.dim); m.load_state_dict(torch.load(ROOT / "results/acts/s3j/models" / f"I_s{sd}_{fold}.pt", map_location="cpu")); m.eval(); ms.append(m)

    def heads(V):
        with torch.no_grad():
            return [m.heads(torch.as_tensor(np.asarray(V, np.float32))).numpy() for m in ms]
    return heads


def cmd_screen(args) -> None:
    from scipy.stats import spearmanr
    from autocls.space.cls_space import CLS
    g84, g86, g89, g93, g94, g98 = (_g(m) for m in ("84_acts_s3a", "86_acts_s3b", "89_acts_s3c", "93_acts_s3f", "94_acts_s3g", "98_acts_s3k"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    meta, arr, _cfg, _space, enc = g94._ctx(args)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    s3b = ROOT / "results/acts/s3b"
    em_sh = g86._e_means(s3b, "sh", tasks, list(g89.FOLDS)); em_pt = g86._e_means(s3b, "pt", tasks, [""])[""]
    sm = (ROOT / "results/acts/s3e/axis_targets.npz")
    with np.load(sm) as z:
        AX = {s: np.asarray(z[f"aux/{s}"], float)[:, :3] for s in g89.FOLDS}
    # 候选池（同 S3-K）
    D = {k: ROOT / "results/acts" / k for k in PREV}
    val = g94._val_map([D[k] for k in PREV]); cls_of = {}
    for d in [ROOT / "results/acts/s2"] + [D[k] for k in PREV]:
        for sub, name in (("", "plan_S.json"), ("runs", "plan_X.json"), ("runs_aug", "plan_X.json")):
            p = (d / sub / name) if sub else (d / name)
            if p.exists():
                for c in json.loads(p.read_text(encoding="utf-8"))["cells"]:
                    cls_of[(c["task_id"], c["cls_key"])] = c["cls"]
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS)
    pool = {}
    for t in ex:
        keys = sorted({k for (tt, k, s) in val if tt == t and s == 0 and (t, k, 1) in val and (t, k) in cls_of})
        pool[t] = (np.stack([enc.encode(CLS(**cls_of[(t, k)])) for k in keys]).astype(np.float32),
                   np.array([val[(t, k, 0)] for k in keys]), np.array([val[(t, k, 1)] for k in keys]))
    crit, held, ahat_store, mag = {}, {}, {}, {}
    t0 = time.time()
    for fold in g89.FOLDS:
        heads = _load_heads(fold, enc); A = AX[fold]; tr = groups != fold
        trn = np.where(tr)[0]; srcs = sorted(set(groups[trn].tolist()))
        Hs = {i: heads(arr[f"{tasks[i]}/vec"][:N]) for i in trn}
        target = {i: (end0[i] + end1[i]) / 2 for i in trn}

        def criterion(C: np.ndarray) -> float:
            return float(np.mean([spearmanr(np.mean([wins_linear(H, C[i]) for H in Hs[i]], 0), target[i])[0] for i in trn]))
        for var in E_VARS:
            U = e_features(var, fold, groups, em_sh, em_pt)
            # MSE：逐分量 α（嵌套于内层留一个源）
            oof = np.zeros((len(tasks), 3))
            for s_ in srcs:
                inner = tr & (groups != s_)
                al = [float(g84.choose_alpha(U[inner], A[inner, k], groups[inner], ALPHAS)) for k in range(3)]
                te_ = groups == s_
                oof[te_] = _pred(_ridge(U, A, np.where(inner)[0], al), U[te_])
            crit.setdefault((var, "MSE"), []).append(criterion(oof))
            al_mse = [float(g84.choose_alpha(U[tr], A[tr, k], groups[tr], ALPHAS)) for k in range(3)]
            # DS：共用 α（按下游读数选）+ 逐分量幅度校准
            best = (-np.inf, None, None)
            for a_ in ALPHAS:
                oof = np.zeros((len(tasks), 3))
                for s_ in srcs:
                    inner = tr & (groups != s_); te_ = groups == s_
                    oof[te_] = _pred(_ridge(U, A, np.where(inner)[0], [a_] * 3), U[te_])
                beta = np.array([calib_slope(A[tr, k], oof[tr, k]) for k in range(3)])
                v = criterion(oof * beta)
                if v > best[0]:
                    best = (v, a_, beta)
            crit.setdefault((var, "DS"), []).append(best[0])
            # 部署：成员 bootstrap 重拟合 → 全体任务坐标
            for mp in MAPS:
                mem = []
                for sd in _g("91_acts_s3e").ENS_SEEDS:
                    rows = _boot_rows(groups, tr, sd)
                    if mp == "MSE":
                        mem.append(_pred(_ridge(U, A, rows, al_mse), U))
                    else:
                        mem.append(_pred(_ridge(U, A, rows, [best[1]] * 3), U) * best[2])
                ahat_store[(var, mp, fold)] = np.array(mem)
                mag.setdefault((var, mp), []).append((np.mean(mem, 0)[tr].std(0) / A[tr].std(0)).tolist())
                held.setdefault((var, mp), {})[fold] = {"alpha": al_mse if mp == "MSE" else best[1], "beta": None if mp == "MSE" else best[2].tolist()}
        print(f"[screen] {fold} done {time.time() - t0:.0f}s", flush=True)
    crit_m = {k: float(np.mean(v)) for k, v in crit.items()}
    star = select_best(crit_m)
    # 留出读数
    heads_f = {s: _load_heads(s, enc) for s in g89.FOLDS}
    hi = [i for i, g in enumerate(groups) if g in g89.FOLDS]
    res = {}
    for key in crit_m:
        r1, r3, rho = [], [], []
        for i in hi:
            s = groups[i]; mem = ahat_store[(key[0], key[1], s)][:, i]; hs = heads_f[s](arr[f"{tasks[i]}/vec"][:N])
            w = np.mean([wins_linear(H, c) for H, c in zip(hs, mem)], 0)
            r1.append(g98.pool_regret(w, end0[i], end1[i]))
            R0 = arr[f"{tasks[i]}/R0"]
            if len(R0) >= 10:
                hr = heads_f[s](arr[f"{tasks[i]}/vec"][N:N + len(R0)])
                r3.append(float(spearmanr(np.mean([wins_linear(H, c) for H, c in zip(hr, mem)], 0), R0)[0]))
        ah_mean = np.array([ahat_store[(key[0], key[1], groups[i])][:, i].mean(0) for i in hi]); at = np.array([AX[groups[i]][i] for i in hi])
        rho = [float(spearmanr(ah_mean[:, k], at[:, k])[0]) for k in range(3)]
        r2, r2s = [], []
        for t in ex:
            i = tasks.index(t); s = groups[i]; mem = ahat_store[(key[0], key[1], s)][:, i]
            V, v0, v1 = pool[t]; hs = heads_f[s](V)
            w = np.mean([wins_linear(H, c) for H, c in zip(hs, mem)], 0)
            r2.append(g98.pool_regret(w, v0, v1)); r2s.append(float(spearmanr(w, (v0 + v1) / 2)[0]))
        res[f"{key[0]}|{key[1]}"] = {"crit": crit_m[key], "R1": float(np.mean(r1)), "R2": float(np.mean(r2)), "R2_sp": float(np.mean(r2s)), "R3": float(np.nanmedian(r3)),
                                     "rho_heldout": rho, "mag_train": np.mean(mag[key], 0).tolist(), "fits": held[key]}
    k3 = json.loads((ROOT / "results/acts/s3k/analysis.json").read_text(encoding="utf-8"))["summary"]
    for v in res.values():
        v["real"] = {"R1": g98.realized(k3["Z"]["R1_regret3"], v["R1"], k3["T"]["R1_regret3"], True),
                     "R2": g98.realized(k3["Z"]["R2_regret3"], v["R2"], k3["T"]["R2_regret3"], True),
                     "R2_sp": g98.realized(k3["Z"]["R2_sp"], v["R2_sp"], k3["T"]["R2_sp"], False),
                     "R3": g98.realized(k3["Z"]["R3_sp"], v["R3"], k3["T"]["R3_sp"], False)}
    np.savez_compressed(outdir / "ahat.npz", **{f"{s}/m{m}": ahat_store[(star[0], star[1], s)][m].astype(np.float32) for s in g89.FOLDS for m in range(5)})
    out = {"star": list(star), "search": star != ("sh4", "MSE"), "candidates": res}
    (outdir / "screen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    pc = lambda x: "—" if not np.isfinite(x) else f"{100 * x:.0f}%"  # noqa: E731
    L = [f"# S3-L 筛选（M★ = {star[0]} / {star[1]}；{'上搜索' if out['search'] else '= 现行，不上搜索'}）", "",
         "| 候选 | 训练侧下游读数 | R1 regret@3 | R2 regret@3 | R2 Sp | R3 Sp | 兑现 R2 Sp | 兑现 R3 | 兑现 R1 | 留出 ρ(â,a) a₁/a₂/a₃ | 幅度比 a₁/a₂/a₃ |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["crit"]):
        L.append(f"| {k} | {v['crit']:.4f} | {f(v['R1'])} | {f(v['R2'])} | {f(v['R2_sp'])} | {f(v['R3'])} | {pc(v['real']['R2_sp'])} | {pc(v['real']['R3'])} | {pc(v['real']['R1'])} | "
                 + "/".join(f"{x:+.2f}" for x in v["rho_heldout"]) + " | " + "/".join(f"{x:.2f}" for x in v["mag_train"]) + " |")
    L += ["", f"参照（S3-K）：Z R2 Sp {k3['Z']['R2_sp']:.3f} / T {k3['T']['R2_sp']:.3f}；现行 P {k3['P']['R2_sp']:.3f}"]
    (outdir / "screen.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 搜索（服务器）
def cmd_search(args) -> None:
    import torch
    g89, g91, g92, g93, g94, g96 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g", "96_acts_s3i"))
    from autocls.space.cls_space import CLS
    meta, arr, cfg, space, enc = g94._ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS
    models, Ah = [], []
    with np.load(outdir / "ahat.npz") as z:
        for sd in seeds:
            m = g96.AxisComparator(enc.dim).to(dev); m.load_state_dict(torch.load(Path(args.s3j) / "models" / f"I_s{sd}_{fold}.pt", map_location=dev)); m.eval(); models.append(m)
            Ah.append(torch.as_tensor(z[f"{fold}/m{sd}"], device=dev))
    ens = g96.AxisEnsemble(models, dev)
    tasks = meta["tasks"]; groups = meta["groups"]; cx = g92.cross_index(tasks, groups, meta["order"])
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t in g93.exam_tasks(tasks, groups, [fold]):
        i = tasks.index(t); Ta = [A[i] for A in Ah]; Tx = [A[cx[i]] for A in Ah]; Vsh = arr[f"{t}/vec"][:g89.N_SH]
        rec = {"x_task": tasks[cx[i]], "a_hat": [A[i].tolist() for A in Ah]}
        rec.update(g92._search_task(ens, Ta, t, lcb("A″-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens, Tx, t, lcb("AX″-lcb"), space, enc, cfg, ks, gens))
        for e in rec["A″-lcb"]:
            e["pred_beat"] = g94._pred_beat(ens, Ta, enc.encode(CLS(**e["cls"])), Vsh)
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in EXAM) + f" {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def _dirs(args) -> list[Path]:
    return [Path(getattr(args, k)) for k in PREV] + [Path(args.outdir)]


def cmd_plan_runs(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in EXAM}
            for a in EXAM:
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
    sl = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}; sj = {s: js(P_["s3j"] / "search" / f"{s}.json") for s in g89.FOLDS}
    sgg = {s: js(P_["s3g"] / "search" / f"{s}.json") for s in g89.FOLDS}; sf = {s: js(P_["s3f"] / "search" / f"{s}.json") for s in g89.FOLDS}
    sc_ = {s: js(P_["s3c"] / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("A″-lcb", "AX″-lcb", "A′-lcb", "O′-lcb", "B′-lcb", "R")
    noms = {t: {"A″-lcb": sl[s][t]["A″-lcb"], "AX″-lcb": sl[s][t]["AX″-lcb"], "A′-lcb": sj[s][t]["A-lcb"], "O′-lcb": sj[s][t]["O-lcb"],
                "B′-lcb": sgg[s][t]["B′-lcb"], "R": sf[s][t]["R"] if sf[s][t]["new_row"] else sc_[s][t]["random"]} for t, s in zip(ex, sg)}
    P1, M3, miss, over = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}, []
    cat_p1, cat_m3 = [], []
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        for e in sl[s][t]["A″-lcb"]:
            over.append(e["pred_beat"] - g94.true_beat((val.get((t, e["cls_key"], 0), fl) + val.get((t, e["cls_key"], 1), fl)) / 2, ref))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"] = np.array(cat_p1), np.array(cat_m3)

    def cmp(x, y):
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    pairs = (("A″-lcb", "B′-lcb"), ("A″-lcb", "A′-lcb"), ("A″-lcb", "AX″-lcb"), ("O′-lcb", "A″-lcb"), ("A″-lcb", "cat3"), ("O′-lcb", "B′-lcb"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    code = verdict_s3l(C["A″-lcb−B′-lcb"]["m3"], C["A″-lcb−B′-lcb"]["p1"], C["A″-lcb−A′-lcb"]["m3"], C["A″-lcb−AX″-lcb"]["m3"])
    gain = {"m3": (M3["A″-lcb"].mean() - M3["B′-lcb"].mean()) / (M3["O′-lcb"].mean() - M3["B′-lcb"].mean()),
            "p1": (P1["A″-lcb"].mean() - P1["B′-lcb"].mean()) / (P1["O′-lcb"].mean() - P1["B′-lcb"].mean())}
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    scr = js(outdir / "screen.json")
    out = {"verdict": code, "star": scr["star"], "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "share_of_O′_gain": gain, "missing": miss, "over_A″": float(np.mean(over)), "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-L 读数（{code}；M★ = {scr['star'][0]} / {scr['star'][1]}）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'][a])} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", f"A″ 占 O′ − B′ 增益：3 条平均 {100 * gain['m3']:.0f}% / 3 选 1 {100 * gain['p1']:.0f}%；A″ 自提名高估 {out['over_A″']:+.3f}（A′ +0.151 / O′ +0.098）", "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地：只走搜索路径）
def cmd_smoke(args) -> None:
    g89 = _g("89_acts_s3c")
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    fold = g89.FOLDS[0]; rng = np.random.default_rng(0)
    np.savez_compressed(od / "ahat.npz", **{f"{fold}/m{m}": rng.normal(size=(160, 3)).astype(np.float32) for m in range(2)})
    ns = types.SimpleNamespace(**{**vars(args), "fold": fold, "ks": 1000, "gens": 2, "resume": False, "seeds": [0, 1]})
    cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in EXAM)
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("screen", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    for k in PREV:
        ap.add_argument(f"--{k}", default=str(ROOT / "results" / "acts" / k))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("screen", cmd_screen), ("smoke", cmd_smoke), ("search", cmd_search), ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
