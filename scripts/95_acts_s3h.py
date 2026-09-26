"""WP17-ACTS S3-H（results/acts/s3h/PROTOCOL.md）：不学习的 E（共享 TS2Vec 任务均值 PCA 4 维 = S3-C 的 ctx）进残差 field 项，起点 S3-G 的 B′。

阶段：
  --train         服务器：--fold --seed → models/H_s<seed>_<fold>.pt、train/H_s<seed>_<fold>.npz
  --search        服务器：--fold → search/<fold>.json（48 任务考卷 H-lcb / HX-lcb + 自提名预测胜率）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1；去重 S2 / S3-C / E / EA / F / G 已有格）
  --analyze       本地
  --smoke         本地冒烟
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3h"
S3C = ROOT / "results" / "acts" / "s3c"
S3E = ROOT / "results" / "acts" / "s3e"
S3EA = ROOT / "results" / "acts" / "s3ea"
S3F = ROOT / "results" / "acts" / "s3f"
S3G = ROOT / "results" / "acts" / "s3g"
K_CTX = 4
LR_E = 3e-5
EVALS_PER_EPOCH, PAT_EVALS, MAX_EPOCHS_H = 4, 8, 30
EXAM_ARMS = ("H-lcb", "HX-lcb")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def standardize_ctx(C: np.ndarray, groups: Sequence, fold: str) -> np.ndarray:
    """按折内训练侧（留出源以外）的均值 / 标准差逐维标准化。"""
    C = np.asarray(C, float); tr = np.asarray(groups) != fold
    m, s = C[tr].mean(0), C[tr].std(0)
    return (C - m) / np.where(s > 1e-12, s, 1.0)


def build_field_ctx(cls_dim: int):
    from autocls.comparator.tclsc_model import TCLSC
    return TCLSC(cls_dim, None, ctx_dim=K_CTX, task_branch="ctx", head="field")


def verdict_s3h(dH_m3: Mapping, dH_p1: Mapping, dX_m3: Mapping, cat_ok: bool, n_gain: int) -> str:
    if n_gain == 0:
        code = "S3H-VAL-INERT"
    elif dH_m3["d"] >= 0.010 and dH_m3["p"] < 0.05 and dH_m3["npos"] >= 4 and dH_p1["d"] > 0:
        code = "S3H-E-CONTRIB"
    elif dH_m3["d"] <= -0.010 and dH_m3["p"] < 0.05:
        code = "S3H-E-HURT"
    else:
        code = "S3H-E-NULL"
    if n_gain > 0 and dX_m3["d"] >= 0.005 and dX_m3["p"] < 0.05 and dX_m3["npos"] >= 4:
        code += "+CONTENT"
    if cat_ok:
        code += "+CAT-PAR"
    return code


def _ctx_of(arr: Mapping, groups, fold: str) -> np.ndarray:
    return standardize_ctx(np.asarray(arr[f"ctx/{fold}"])[:, :K_CTX], groups, fold).astype(np.float32)


def _gns(args) -> types.SimpleNamespace:
    """指向 S3-G 目录的参数（读 B′ 模型与扩充标签）。"""
    return types.SimpleNamespace(**{**vars(args), "outdir": args.s3g})


def _dirs(args) -> list[Path]:
    return [Path(args.s3c), Path(args.s3e), Path(args.s3ea), Path(args.s3f), Path(args.s3g), Path(args.outdir)]


# ================================================================ 训练（服务器）
def cmd_train(args) -> None:
    import torch
    g88, g89, g91, g93, g94 = (_g(m) for m in ("88_acts_s3bpp", "89_acts_s3c", "91_acts_s3e", "93_acts_s3f", "94_acts_s3g"))
    outdir = Path(args.outdir)
    meta, arr, _cfg, _space, enc = g94._ctx(args)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, seed, dev = args.fold, int(args.seed), args.device
    tag = f"H_s{seed}_{fold}"
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
    aug = g94.load_aug(_gns(args), fold, meta, enc)
    V, R, NS = g94.merge_aug(V, R, NS, aug)
    print(f"[aug] {fold}: {len(aug)} 任务、{sum(len(v[1]) for v in aug.values())} 条扩充标签", flush=True)
    Ct = torch.as_tensor(_ctx_of(arr, groups, fold), device=dev)
    torch.manual_seed(seed)
    model = build_field_ctx(V[0].shape[1]).to(dev)
    blind_sd = torch.load(Path(args.s3g) / "models" / f"blind_s{seed}_{fold}.pt", map_location=dev)
    opt = torch.optim.Adam(g93.init_residual(model, blind_sd), lr=LR_E)
    sc = g89.Scorer(model, "C", dev)
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
        p = sc.forward(va_, vb_, Ct[[b[0] for b in batch]]).clamp(1e-6, 1 - 1e-6)
        return -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)

    def val_loss():
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]
                vl += float(batch_loss(b)) * len(b); vn += len(b)
        return vl / max(vn, 1)

    t0 = time.time()
    start = best = val_loss()
    best_state, best_at, bad = copy.deepcopy(model.state_dict()), (-1, 0), 0
    hist = [{"epoch": -1, "q": 0, "train_loss": None, "val_loss": start}]
    stop = False
    for ep in range(args.max_epochs or MAX_EPOCHS_H):
        model.train()
        pairs = [p for i in tr for p in weighted(i, g89.linear_pairs(R[i], rng))]
        perm = rng.permutation(len(pairs)); nb = math.ceil(len(perm) / g89.BATCH)
        marks = {max(0, int(round(nb * k / EVALS_PER_EPOCH)) - 1): k for k in range(1, EVALS_PER_EPOCH + 1)}
        tot, cnt = 0.0, 0
        for bi, s in enumerate(range(0, len(perm), g89.BATCH)):
            loss = batch_loss([pairs[j] for j in perm[s:s + g89.BATCH]])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
            if bi in marks:
                vl = val_loss(); model.train()
                hist.append({"epoch": ep, "q": marks[bi], "train_loss": tot / max(cnt, 1), "val_loss": vl, "sec": round(time.time() - t0, 1)})
                if vl < best - 1e-9:
                    best, bad, best_at, best_state = vl, 0, (ep, marks[bi]), copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                    if bad >= PAT_EVALS:
                        stop = True; break
        if stop:
            break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    cmaps = g88.cross_maps(tasks, groups, order)
    out = {"hist": json.dumps(hist), "te": np.array(te), "va": np.array(va), "best_at": np.array(best_at), "start_val": start, "best_val": best}
    for i in te + va:
        out[f"P/{tasks[i]}"] = sc.pmat(V[i], Ct[i]).astype(np.float32)
    for i in te:
        out[f"X/{tasks[i]}"] = np.stack([sc.pmat(V[i][:g89.N_SH], Ct[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done start {start:.4f} best {best:.4f} at {best_at} evals {len(hist) - 1} {time.time() - t0:.0f}s", flush=True)


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
    ens = g92.ArmEnsemble(g93._load_models(lambda: build_field_ctx(enc.dim), [outdir / "models" / f"H_s{sd}_{fold}.pt" for sd in seeds], dev), "C", dev)
    tasks = meta["tasks"]; groups = meta["groups"]
    Ct = torch.as_tensor(_ctx_of(arr, groups, fold), device=dev)
    cx = g92.cross_index(tasks, groups, meta["order"])
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t in g93.exam_tasks(tasks, groups, [fold]):
        i = tasks.index(t); Ts = [Ct[i]] * len(seeds); Tx = [Ct[cx[i]]] * len(seeds); Vsh = arr[f"{t}/vec"][:g89.N_SH]
        rec = {"x_task": tasks[cx[i]]}
        rec.update(g92._search_task(ens, Ts, t, lcb("H-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens, Tx, t, lcb("HX-lcb"), space, enc, cfg, ks, gens))
        for e in rec["H-lcb"]:
            e["pred_beat"] = g94._pred_beat(ens, Ts, enc.encode(CLS(**e["cls"])), Vsh)
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in EXAM_ARMS) + f" ctx←{rec['x_task']} {time.time() - t0:.0f}s", flush=True)
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
def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, wilcoxon
    g84, g89, g91, g92, g93, g94 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g"))
    from autocls.search.rank import symmetrize, wins_from_scores
    outdir, s3c, s3ea, s3f, s3g = Path(args.outdir), Path(args.s3c), Path(args.s3ea), Path(args.s3f), Path(args.s3g)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    val = g94._val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    sh = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}; sgg = {s: js(s3g / "search" / f"{s}.json") for s in g89.FOLDS}
    sf = {s: js(s3f / "search" / f"{s}.json") for s in g89.FOLDS}; sa = {s: js(s3ea / "search" / f"{s}.json") for s in g89.FOLDS}
    sc_ = {s: js(s3c / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("H-lcb", "HX-lcb", "B′-lcb", "F′-lcb", "B-lcb", "R")
    noms = {}
    for t, s in zip(ex, sg):
        f = sf[s][t]; new = f["new_row"]
        noms[t] = {"H-lcb": sh[s][t]["H-lcb"], "HX-lcb": sh[s][t]["HX-lcb"], "B′-lcb": sgg[s][t]["B′-lcb"], "F′-lcb": sgg[s][t]["F′-lcb"],
                   "B-lcb": f["B-lcb"] if new else sa[s][t]["B-lcb"], "R": f["R"] if new else sc_[s][t]["random"]}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    cat_p1, cat_m3, orc, over = [], [], [], []
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        for e in sh[s][t]["H-lcb"]:
            over.append(e["pred_beat"] - g94.true_beat((val.get((t, e["cls_key"], 0), fl) + val.get((t, e["cls_key"], 1), fl)) / 2, ref))
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
    pairs = (("H-lcb", "B′-lcb"), ("H-lcb", "HX-lcb"), ("H-lcb", "F′-lcb"), ("H-lcb", "cat3"), ("B′-lcb", "cat3"), ("H-lcb", "B-lcb"), ("H-lcb", "R"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    zh = {s: [np.load(outdir / "train" / f"H_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    ba = [tuple(int(x) for x in z["best_at"]) for s in g89.FOLDS for z in zh[s]]
    n_gain = int(sum(b[0] >= 0 for b in ba))
    gain = {s: float(np.mean([float(z["start_val"]) - float(z["best_val"]) for z in zh[s]])) for s in g89.FOLDS}
    code = verdict_s3h(C["H-lcb−B′-lcb"]["m3"], C["H-lcb−B′-lcb"]["p1"], C["H-lcb−HX-lcb"]["m3"],
                       bool(C["H-lcb−cat3"]["m3"]["d"] >= 0 and C["H-lcb−cat3"]["p1"]["d"] >= 0), n_gain)
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    zb2 = {s: [np.load(s3g / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    rh, sph = g92._rank_readout(zh, meta, arr, held); rb2, spb2 = g92._rank_readout(zb2, meta, arr, held)

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    rx = np.zeros(len(held))
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; xs = []
        for k in range(16):
            Wx = np.mean([wins_of(z[f"X/{t}"][k]) for z in zh[groups[i]]], axis=0)
            ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
            xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
        rx[jj] = float(np.mean(xs))
    g_over = js(s3g / "analysis.json")["over_exam"]
    rank = {"H": {"regret3": float(rh.mean()), "sp": float(np.nanmedian(sph))}, "B′": {"regret3": float(rb2.mean()), "sp": float(np.nanmedian(spb2))},
            "H−B′": {"d": float((rh - rb2).mean()), "p_source": g84.source_wilcoxon(rh - rb2, hg)},
            "cross": {"d": float((rx - rh).mean()), "ci": g84.cluster_bootstrap_ci(rx - rh, hg), "p_source": g84.source_wilcoxon(rx - rh, hg), "npos": g89.n_sources_positive(rx - rh, hg)}}
    train = {"n_gain": n_gain, "best_at": ba, "per_fold_gain": gain, "best_epoch_median": float(np.median([b[0] for b in ba]))}
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "n_tasks": len(ex), "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "missing": miss, "over_H_own": float(np.mean(over)), "over_ref_s3g": g_over, "train": train, "rank": rank, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-H 读数（{code}）", "", f"## 考卷（{len(ex)} 任务）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3", "oracle"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'].get(a))} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", f"自提名高估：H {out['over_H_own']:+.3f}；S3-G 参照 B′ {g_over['B′ own']['mean']:+.3f} / F′ {g_over['F′ own']['mean']:+.3f}", "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    L += ["", "## 训练与排序（56 留出任务）", "",
          f"- 验证损失低于起点 {n_gain}/30；逐折下降 " + "，".join(f"{s} {v:+.4f}" for s, v in gain.items()) + f"；最好 epoch 中位 {train['best_epoch_median']}",
          f"- regret@3：H {f(rank['H']['regret3'])} / B′ {f(rank['B′']['regret3'])}（H − B′ {rank['H−B′']['d']:+.4f}，源级 p {rank['H−B′']['p_source']:.3f}）；random-20 中位 Sp：H {f(rank['H']['sp'])} / B′ {f(rank['B′']['sp'])}",
          f"- 换源 Δ（16 偏移）{rank['cross']['d']:+.4f} [{f(rank['cross']['ci'][0])}, {f(rank['cross']['ci'][1])}]，源级 p {rank['cross']['p_source']:.3f}，同号 {rank['cross']['npos']}/6"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89, g92 = _g("89_acts_s3c"), _g("92_acts_s3ea")
    od = Path(args.outdir); sg = od / "_s3g"; (sg / "search_aug").mkdir(parents=True, exist_ok=True)
    fold = g89.FOLDS[0]
    (sg / "search_aug" / f"{fold}.json").write_text("{}", encoding="utf-8")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "s3g": str(sg), "seeds": [0, 1]})
    for sd in (0, 1):
        g92.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sg), "fold": fold, "seed": sd}))
        cmd_train(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    ns = types.SimpleNamespace(**{**base, "fold": fold})
    cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in EXAM_ARMS) and all("pred_beat" in e for v in r.values() for e in v["H-lcb"])
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
    for k, v in (("s3c", S3C), ("s3e", S3E), ("s3ea", S3EA), ("s3f", S3F), ("s3g", S3G)):
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
