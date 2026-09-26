"""WP17-ACTS S3-I（results/acts/s3i/PROTOCOL.md）：轴瓶颈 T-CLSC——E 只通过 3 维偏好轴坐标 â(t) 进入比较器。

logit(a, b | t) = [s₀(a) − s₀(b)] + Σₖ âₖ(t)[gₖ(a) − gₖ(b)]（论文未给低秩轴头，按 S3-A 低秩模型 z = μ(c) + Σ aₖ(t)bₖ(c) 语义补全）。
第一段用已知轴坐标 a(t) 学 s₀ / gₖ；第二段 â = W·u(E_sh PCA4)，逐分量岭回归（α 按留一个源选，集成成员用源 bootstrap）。

阶段：
  --train         服务器：--fold --seed → models/I_s<seed>_<fold>.pt、train/I_s<seed>_<fold>.npz（含 â、P_O / P_A / X_A）
  --search        服务器：--fold → search/<fold>.json（48 任务考卷 O-lcb / A-lcb / AX-lcb + 自提名预测胜率）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1；去重至 S3-H）
  --analyze       本地
  --smoke         本地冒烟
"""
from __future__ import annotations

import argparse
import copy
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

OUT = ROOT / "results" / "acts" / "s3i"
S3C, S3E, S3EA, S3F, S3G, S3H = (ROOT / "results" / "acts" / d for d in ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h"))
K_AX = 3
ALPHAS = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
EXAM_ARMS = ("O-lcb", "A-lcb", "AX-lcb")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 比较器（任务无关：只见坐标向量）
def _torch_nn():
    import torch
    return torch, torch.nn


def AxisComparator(cls_dim: int, k: int = K_AX, emb: int = 128, fz: int = 128):
    """工厂：返回 nn.Module。主干宽度同 TCLSC 的 CLS 支路（embedding_dim = fz = 128，← ahc_engine.py L97-126）。"""
    torch, nn = _torch_nn()

    class _AxisComparator(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = int(k)
            self.cls_mlp = nn.Sequential(nn.Linear(cls_dim, emb), nn.ReLU(), nn.Linear(emb, emb))
            self.head = nn.Sequential(nn.Linear(emb, fz), nn.ReLU(), nn.Linear(fz, 1 + self.k))

        def heads(self, x):
            return self.head(self.cls_mlp(x))

        def forward(self, a, b, t):
            h = self.heads(a) - self.heads(b)
            return torch.sigmoid(h[:, 0] + (h[:, 1:] * t).sum(1))

    return _AxisComparator()


class AxisScorer:
    """与 89.Scorer 同接口（forward / pmat / dev），任务条件 = 3 维坐标张量。"""

    def __init__(self, model, dev):
        self.m, self.dev = model, dev

    def forward(self, va, vb, T):
        return self.m(va, vb, T)

    def pmat(self, V: np.ndarray, T, batch: int = 65536) -> np.ndarray:
        import torch
        n = len(V); P = np.full((n, n), 0.5)
        ii, jj = np.where(~np.eye(n, dtype=bool))
        Vt = torch.as_tensor(np.asarray(V, np.float32), device=self.dev)
        with torch.no_grad():
            for s in range(0, len(ii), batch):
                a, b = ii[s:s + batch], jj[s:s + batch]
                P[a, b] = self.forward(Vt[a], Vt[b], T.unsqueeze(0).expand(len(a), -1)).float().cpu().numpy()
        return P


class AxisEnsemble:
    def __init__(self, models, dev):
        self.sc = [AxisScorer(m, dev) for m in models]

    wins = _g("91_acts_s3e").Ensemble.wins


# ================================================================ E → 坐标
def fit_axis_map(U: np.ndarray, A: np.ndarray, groups: Sequence, tr: np.ndarray, alphas: Sequence[float] = ALPHAS, boot_seed: Optional[int] = None) -> dict:
    """逐分量岭回归 U → A，只用训练侧行；α 按留一个源选（全样本），boot_seed 给定时在训练源上有放回重采样后重拟合。"""
    g84 = _g("84_acts_s3a")
    U, A, groups, tr = np.asarray(U, float), np.asarray(A, float), np.asarray(groups), np.asarray(tr, bool)
    Xtr, Atr, gtr = U[tr], A[tr], groups[tr]
    al = [float(g84.choose_alpha(Xtr, Atr[:, k], gtr, alphas)) for k in range(A.shape[1])]
    rows = np.arange(len(Xtr))
    if boot_seed is not None:
        srcs = np.array(sorted(set(gtr.tolist())))
        pick = np.random.default_rng(int(boot_seed)).choice(srcs, len(srcs), replace=True)
        rows = np.concatenate([np.where(gtr == s)[0] for s in pick])
    return {"alpha": al, "models": [g84.ridge_fit(Xtr[rows], Atr[rows, k], al[k]) for k in range(A.shape[1])]}


def predict_axis(fit: Mapping, U: np.ndarray) -> np.ndarray:
    g84 = _g("84_acts_s3a")
    return np.stack([g84.ridge_predict(m, U) for m in fit["models"]], 1)


def verdict_s3i(dO_m3: Mapping, dO_p1: Mapping, dA_m3: Mapping, dA_p1: Mapping, dX_m3: Mapping) -> str:
    ok = lambda m3, p1: m3["d"] >= 0.010 and m3["p"] < 0.05 and m3["npos"] >= 4 and p1["d"] > 0  # noqa: E731
    code = "S3I-ARCH-OK" if ok(dO_m3, dO_p1) else "S3I-ARCH-NULL"
    if ok(dA_m3, dA_p1):
        code += "+E-WIN"
    elif dA_m3["d"] <= -0.010 and dA_m3["p"] < 0.05:
        code += "+E-HURT"
    else:
        code += "+E-NULL"
    if dX_m3["d"] >= 0.005 and dX_m3["p"] < 0.05 and dX_m3["npos"] >= 4:
        code += "+CONTENT"
    return code


# ================================================================ 公用
def _axis(args, fold: str) -> np.ndarray:
    with np.load(Path(args.s3e) / "axis_targets.npz") as z:
        return np.asarray(z[f"aux/{fold}"], np.float32)[:, :K_AX]


def _dirs(args) -> list[Path]:
    return [Path(getattr(args, k)) for k in ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h")] + [Path(args.outdir)]


# ================================================================ 训练（服务器）
def cmd_train(args) -> None:
    import torch
    g88, g89, g91, g94, g95 = (_g(m) for m in ("88_acts_s3bpp", "89_acts_s3c", "91_acts_s3e", "94_acts_s3g", "95_acts_s3h"))
    outdir = Path(args.outdir)
    meta, arr, _cfg, _space, enc = g94._ctx(args)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, seed, dev = args.fold, int(args.seed), args.device
    tag = f"I_s{seed}_{fold}"
    for d in ("train", "models"):
        (outdir / d).mkdir(parents=True, exist_ok=True)
    op = outdir / "train" / f"{tag}.npz"
    if op.exists() and args.resume:
        print(f"[train] skip {tag}"); return
    vs = meta["val"][fold]
    tr = [i for i, g in enumerate(groups) if g not in (fold, vs)]
    va = [i for i, g in enumerate(groups) if g == vs]
    te = [i for i, g in enumerate(groups) if g == fold]
    V = [arr[f"{t}/vec"] for t in tasks]
    R, NS, SIG = g91._labels(arr, tasks, "L4")
    aug = g94.load_aug(types.SimpleNamespace(**{**vars(args), "outdir": args.s3g}), fold, meta, enc)
    V, R, NS = g94.merge_aug(V, R, NS, aug)
    print(f"[aug] {fold}: {len(aug)} 任务、{sum(len(v[1]) for v in aug.values())} 条扩充标签", flush=True)
    Aax = _axis(args, fold)
    U = g95._ctx_of(arr, groups, fold)
    fitW = fit_axis_map(U, Aax, groups, groups != fold, ALPHAS, boot_seed=seed)
    Ahat = predict_axis(fitW, U).astype(np.float32)
    Tt = torch.as_tensor(Aax, device=dev); Th = torch.as_tensor(Ahat, device=dev)
    torch.manual_seed(seed)
    model = AxisComparator(V[0].shape[1]).to(dev)
    sc = AxisScorer(model, dev)
    opt = torch.optim.Adam(model.parameters(), lr=g89.LR)
    rng = np.random.default_rng(seed)

    def weighted(i, trip):
        out = []
        for a, b, y in trip:
            w = g91.l4_weight(R[i][a], R[i][b], SIG[i], NS[i][a], NS[i][b])
            if w > 0:
                out.append((i, a, b, y, w))
        return out

    val_pairs = [p for i in va for p in weighted(i, g89.all_ordered_pairs(R[i]))]

    def batch_loss(batch):
        va_ = torch.as_tensor(np.stack([V[b[0]][b[1]] for b in batch]), dtype=torch.float32, device=dev)
        vb_ = torch.as_tensor(np.stack([V[b[0]][b[2]] for b in batch]), dtype=torch.float32, device=dev)
        y = torch.as_tensor([float(b[3]) for b in batch], dtype=torch.float32, device=dev)
        w = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32, device=dev)
        p = sc.forward(va_, vb_, Tt[[b[0] for b in batch]]).clamp(1e-6, 1 - 1e-6)     # 第一段：已知坐标（teacher forcing）
        return -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(args.max_epochs or g89.MAX_EPOCHS):
        model.train()
        pairs = [p for i in tr for p in weighted(i, g89.linear_pairs(R[i], rng))]
        perm = rng.permutation(len(pairs)); tot, nb = 0.0, 0
        for s in range(0, len(perm), g89.BATCH):
            loss = batch_loss([pairs[j] for j in perm[s:s + g89.BATCH]])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]
                vl += float(batch_loss(b)) * len(b); vn += len(b)
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl, "sec": round(time.time() - t0, 1)})
        if vl < best - 1e-9:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= g89.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    cmaps = g88.cross_maps(tasks, groups, order)
    out = {"hist": json.dumps(hist), "te": np.array(te), "va": np.array(va), "Ahat": Ahat, "alpha": np.array(fitW["alpha"]),
           "ridge": json.dumps([{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in m.items()} for m in fitW["models"]])}
    for i in te + va:
        out[f"P/{tasks[i]}"] = sc.pmat(V[i], Tt[i]).astype(np.float32)                     # 真实坐标（上界）
    for i in te:
        out[f"PA/{tasks[i]}"] = sc.pmat(V[i], Th[i]).astype(np.float32)                    # E 读出的坐标
        out[f"XA/{tasks[i]}"] = np.stack([sc.pmat(V[i][:g89.N_SH], Th[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best val {best:.4f} epochs {len(hist)} alpha {fitW['alpha']} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 考卷搜索（服务器）
def cmd_search(args) -> None:
    import torch
    g89, g91, g92, g93, g94 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g"))
    from autocls.space.cls_space import CLS
    meta, arr, cfg, space, enc = g94._ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS
    models, Ahats = [], []
    for sd in seeds:
        m = AxisComparator(enc.dim).to(dev); m.load_state_dict(torch.load(outdir / "models" / f"I_s{sd}_{fold}.pt", map_location=dev)); m.eval(); models.append(m)
        with np.load(outdir / "train" / f"I_s{sd}_{fold}.npz") as z:
            Ahats.append(torch.as_tensor(z["Ahat"], device=dev))
    ens = AxisEnsemble(models, dev)
    Aax = torch.as_tensor(_axis(args, fold), device=dev)
    tasks = meta["tasks"]; groups = meta["groups"]; cx = g92.cross_index(tasks, groups, meta["order"])
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t in g93.exam_tasks(tasks, groups, [fold]):
        i = tasks.index(t); Vsh = arr[f"{t}/vec"][:g89.N_SH]
        To = [Aax[i]] * len(seeds); Ta = [A[i] for A in Ahats]; Tx = [A[cx[i]] for A in Ahats]
        rec = {"x_task": tasks[cx[i]], "a_true": Aax[i].tolist(), "a_hat": [A[i].tolist() for A in Ahats]}
        rec.update(g92._search_task(ens, To, t, lcb("O-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens, Ta, t, lcb("A-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens, Tx, t, lcb("AX-lcb"), space, enc, cfg, ks, gens))
        for arm, Ts in (("O-lcb", To), ("A-lcb", Ta)):
            for e in rec[arm]:
                e["pred_beat"] = g94._pred_beat(ens, Ts, enc.encode(CLS(**e["cls"])), Vsh)
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in EXAM_ARMS) + f" {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def cmd_plan_runs(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in EXAM_ARMS}
            for a in EXAM_ARMS:
                byk.update({e["cls_key"]: e["cls"] for e in rec[a]})
    _g("94_acts_s3g")._plan(args, noms, byk, (0, 1), "runs", dirs=_dirs(args))


# ================================================================ analyze（本地）
def _rank(zs: Mapping[str, list], key: str, meta: Mapping, arr: Mapping, held: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import spearmanr
    from autocls.search.rank import symmetrize, wins_from_scores
    g89 = _g("89_acts_s3c"); N = g89.N_SH; tasks = meta["tasks"]; groups = meta["groups"]; end0, end1 = arr["end0"], arr["end1"]

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    r3, sp = np.full(len(held), np.nan), np.full(len(held), np.nan)
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; zl = zs[groups[i]]
        Wf = np.mean([wins_of(z[f"{key}/{t}"]) for z in zl], axis=0); Ws = np.mean([wins_of(z[f"{key}/{t}"][:N, :N]) for z in zl], axis=0)
        o = sorted(range(N), key=lambda j: (-Ws[j], keys[j]))[:3]
        r3[jj] = end1[i, int(np.argmax(end0[i]))] - end1[i, o[int(np.argmax(end0[i, o]))]]
        R0 = arr[f"{t}/R0"]
        if len(R0) >= 10:
            sp[jj] = float(spearmanr(Wf[N:N + len(R0)], R0)[0])
    return r3, sp


def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, spearmanr, wilcoxon
    from autocls.search.rank import symmetrize, wins_from_scores
    g84, g89, g91, g92, g93, g94 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g"))
    outdir, s3c, s3ea, s3f, s3g, s3h = (Path(getattr(args, k)) for k in ("outdir", "s3c", "s3ea", "s3f", "s3g", "s3h"))
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    val = g94._val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    si = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}; sgg = {s: js(s3g / "search" / f"{s}.json") for s in g89.FOLDS}
    shh = {s: js(s3h / "search" / f"{s}.json") for s in g89.FOLDS}; sf = {s: js(s3f / "search" / f"{s}.json") for s in g89.FOLDS}
    sa = {s: js(s3ea / "search" / f"{s}.json") for s in g89.FOLDS}; sc_ = {s: js(s3c / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("O-lcb", "A-lcb", "AX-lcb", "B′-lcb", "H-lcb", "F′-lcb", "R")
    noms = {}
    for t, s in zip(ex, sg):
        f = sf[s][t]
        noms[t] = {"O-lcb": si[s][t]["O-lcb"], "A-lcb": si[s][t]["A-lcb"], "AX-lcb": si[s][t]["AX-lcb"], "B′-lcb": sgg[s][t]["B′-lcb"],
                   "F′-lcb": sgg[s][t]["F′-lcb"], "H-lcb": shh[s][t]["H-lcb"], "R": f["R"] if f["new_row"] else sc_[s][t]["random"]}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    cat_p1, cat_m3, orc = [], [], []
    over = {"O": [], "A": []}
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        for arm, key in (("O-lcb", "O"), ("A-lcb", "A")):
            for e in si[s][t][arm]:
                over[key].append(e["pred_beat"] - g94.true_beat((val.get((t, e["cls_key"], 0), fl) + val.get((t, e["cls_key"], 1), fl)) / 2, ref))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
        orc.append(float(end1[i, int(np.argmax(end0[i]))]))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"], P1["oracle"] = np.array(cat_p1), np.array(cat_m3), np.array(orc)

    def cmp(x, y):
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    pairs = (("O-lcb", "B′-lcb"), ("A-lcb", "B′-lcb"), ("A-lcb", "AX-lcb"), ("O-lcb", "A-lcb"), ("A-lcb", "H-lcb"), ("O-lcb", "cat3"), ("A-lcb", "cat3"), ("B′-lcb", "cat3"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    code = verdict_s3i(C["O-lcb−B′-lcb"]["m3"], C["O-lcb−B′-lcb"]["p1"], C["A-lcb−B′-lcb"]["m3"], C["A-lcb−B′-lcb"]["p1"], C["A-lcb−AX-lcb"]["m3"])
    # ---- 训练 / E → 坐标 / 排序
    zi = {s: [np.load(outdir / "train" / f"I_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    ahat = np.zeros((len(held), K_AX)); atrue = np.zeros((len(held), K_AX))
    for jj, i in enumerate(held):
        s = groups[i]; ahat[jj] = np.mean([z["Ahat"][i] for z in zi[s]], axis=0); atrue[jj] = _axis(args, s)[i]
    rho = [float(spearmanr(ahat[:, k], atrue[:, k])[0]) for k in range(K_AX)]
    rho_src = [float(spearmanr([ahat[hg == s, 0].mean() for s in g89.FOLDS], [atrue[hg == s, 0].mean() for s in g89.FOLDS])[0])]
    zb2 = {s: [np.load(s3g / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    rO, spO = _rank(zi, "P", meta, arr, held); rA, spA = _rank(zi, "PA", meta, arr, held); rB, spB = g92._rank_readout(zb2, meta, arr, held)

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    rx = np.zeros(len(held))
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; xs = []
        for k in range(16):
            Wx = np.mean([wins_of(z[f"XA/{t}"][k]) for z in zi[groups[i]]], axis=0)
            ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
            xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
        rx[jj] = float(np.mean(xs))
    be = [int(np.argmin([h["val_loss"] for h in json.loads(str(z["hist"]))])) for s in g89.FOLDS for z in zi[s]]
    alphas = [z["alpha"].tolist() for s in g89.FOLDS for z in zi[s]]
    rank = {"O": {"regret3": float(rO.mean()), "sp": float(np.nanmedian(spO))}, "A": {"regret3": float(rA.mean()), "sp": float(np.nanmedian(spA))},
            "B′": {"regret3": float(rB.mean()), "sp": float(np.nanmedian(spB))},
            "O−B′": {"d": float((rO - rB).mean()), "p_source": g84.source_wilcoxon(rO - rB, hg)}, "A−B′": {"d": float((rA - rB).mean()), "p_source": g84.source_wilcoxon(rA - rB, hg)},
            "A cross": {"d": float((rx - rA).mean()), "ci": g84.cluster_bootstrap_ci(rx - rA, hg), "p_source": g84.source_wilcoxon(rx - rA, hg), "npos": g89.n_sources_positive(rx - rA, hg)}}
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "n_tasks": len(ex), "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "missing": miss, "over": {k: float(np.mean(v)) for k, v in over.items()}, "axis_rho": rho, "axis_rho_source_a1": rho_src[0],
           "rank": rank, "best_epoch_median": float(np.median(be)), "alphas": alphas, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-I 读数（{code}）", "", f"## 考卷（{len(ex)} 任务）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3", "oracle"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'].get(a))} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", f"自提名高估：O {out['over']['O']:+.3f} / A {out['over']['A']:+.3f}（S3-G B′ +0.111 / F′ +0.203，S3-H H +0.132）",
          f"E → 坐标（56 留出任务，成员均值 â vs 真实 a）：Spearman a₁ {rho[0]:+.2f} / a₂ {rho[1]:+.2f} / a₃ {rho[2]:+.2f}；源级 a₁（6 源均值）{rho_src[0]:+.2f}", "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    L += ["", "## 排序（56 留出任务，λ = 0 胜场均值集成）", "",
          f"- regret@3：O {f(rank['O']['regret3'])} / A {f(rank['A']['regret3'])} / B′ {f(rank['B′']['regret3'])}；O − B′ {rank['O−B′']['d']:+.4f}（源级 p {rank['O−B′']['p_source']:.3f}）；A − B′ {rank['A−B′']['d']:+.4f}（p {rank['A−B′']['p_source']:.3f}）",
          f"- random-20 中位 Sp：O {f(rank['O']['sp'])} / A {f(rank['A']['sp'])} / B′ {f(rank['B′']['sp'])}",
          f"- A 换源 Δ（16 偏移）{rank['A cross']['d']:+.4f} [{f(rank['A cross']['ci'][0])}, {f(rank['A cross']['ci'][1])}]，源级 p {rank['A cross']['p_source']:.3f}，同号 {rank['A cross']['npos']}/6",
          f"- 第一段最好 epoch 中位 {out['best_epoch_median']}；岭回归 α（逐分量，30 个成员）示例 {alphas[:3]}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89 = _g("89_acts_s3c")
    od = Path(args.outdir); sg = od / "_s3g"; (sg / "search_aug").mkdir(parents=True, exist_ok=True)
    fold = g89.FOLDS[0]
    (sg / "search_aug" / f"{fold}.json").write_text("{}", encoding="utf-8")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "s3g": str(sg), "seeds": [0, 1]})
    for sd in (0, 1):
        cmd_train(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    ns = types.SimpleNamespace(**{**base, "fold": fold})
    cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in EXAM_ARMS)
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("train", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    for k, v in (("s3c", S3C), ("s3e", S3E), ("s3ea", S3EA), ("s3f", S3F), ("s3g", S3G), ("s3h", S3H)):
        ap.add_argument(f"--{k}", default=str(v))
    ap.add_argument("--s3b", default=str(ROOT / "results" / "acts" / "s3b"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("smoke", cmd_smoke), ("train", cmd_train), ("search", cmd_search), ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
