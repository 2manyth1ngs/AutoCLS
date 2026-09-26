"""WP17-ACTS S3-F（results/acts/s3f/PROTOCOL.md）：残差式 field-E（载入 S3-EA 任务盲模型、冻结 s₀、只训 E 通路）+ 48 任务搜索考卷。

阶段：
  --train         服务器：--fold <源> --seed s → models/F_s<seed>_<fold>.pt、train/F_s<seed>_<fold>.npz
  --search        服务器：--fold <源> → search/<fold>.json（8 任务 F-lcb / FX-lcb；偶数行 4 任务另出 B-lcb / E-lcb / R）
  --plan-runs     服务器：提名去重（S2 / S3-C / S3-E / S3-EA 已有格复用）→ runs/plan_X.json（83 phase X）
  --analyze       本地
  --smoke         本地冒烟（合成窗口）
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3f"
S3C = ROOT / "results" / "acts" / "s3c"
S3E = ROOT / "results" / "acts" / "s3e"
S3EA = ROOT / "results" / "acts" / "s3ea"
S2_DIR = ROOT / "results" / "acts" / "s2"
LR_E = 3e-5
AUX_BETA_F = 0.1
EVALS_PER_EPOCH = 4
PAT_EVALS = 8
MAX_EPOCHS_F = 30
N_ROWS = 8                      # 每个折源取 OA 行 a00–a07
ARMS = ("F-lcb", "FX-lcb", "B-lcb", "E-lcb", "R")
S0_PREFIX = ("cls_mlp", "score_mlp")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def exam_tasks(tasks: Sequence[str], groups: Sequence, folds: Sequence[str]) -> list[str]:
    """考卷：每个折源的 OA 行 a00–a07（按折源顺序、行号顺序）。"""
    row = _g("86_acts_s3b")._row
    out = []
    for s in folds:
        out += sorted((t for t, g in zip(tasks, groups) if g == s and row(t) < N_ROWS), key=row)
    return out


def build_field_e(cls_dim: int):
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.task_embed import TaskSetEncoder
    return TCLSC(cls_dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head="field")


def init_residual(model, blind_state: Mapping) -> list:
    """载入任务盲模型的 s₀ 通路并冻结；field_w 末层置零（起点输出 = 任务盲模型）；返回可训参数（E 通路）。"""
    import torch
    res = model.load_state_dict(blind_state, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert not any(k.startswith(S0_PREFIX) for k in res.missing_keys), res.missing_keys
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(S0_PREFIX))
    with torch.no_grad():
        model.field_w[-1].weight.zero_()
    return [p for p in model.parameters() if p.requires_grad]


def readouts(t: str, keys: Sequence[str], val: Mapping, floor: float) -> tuple[float, float]:
    """(3 选 1：seed 0 选、seed 1 评；3 条平均：两种子均值的平均)。缺格按任务地板补。"""
    v0 = [val.get((t, k, 0), floor) for k in keys]; v1 = [val.get((t, k, 1), floor) for k in keys]
    b = int(np.argmax(v0))
    return float(v1[b]), float(np.mean([(a + c) / 2.0 for a, c in zip(v0, v1)]))


def verdict_s3f(dF_m3: Mapping, dF_p1: Mapping, dX_m3: Mapping, cat_ok: bool, n_gain: int) -> str:
    if n_gain == 0:
        code = "S3F-VAL-INERT"
    elif dF_m3["d"] >= 0.010 and dF_m3["p"] < 0.05 and dF_m3["npos"] >= 4 and dF_p1["d"] > 0:
        code = "S3F-E-CONTRIB"
    elif dF_m3["d"] <= -0.010 and dF_m3["p"] < 0.05:
        code = "S3F-E-HURT"
    else:
        code = "S3F-E-NULL"
    if n_gain > 0 and dX_m3["d"] >= 0.005 and dX_m3["p"] < 0.05 and dX_m3["npos"] >= 4:
        code += "+CONTENT"
    if cat_ok:
        code += "+CAT-PAR"
    return code


# ================================================================ 训练（服务器）
def cmd_train(args) -> None:
    import torch
    g88, g89, g91 = _g("88_acts_s3bpp"), _g("89_acts_s3c"), _g("91_acts_s3e")
    outdir = Path(args.outdir)
    meta, arr = g89.load_data(Path(args.s3c))
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, seed, dev = args.fold, int(args.seed), args.device
    tag = f"F_s{seed}_{fold}"
    for d in ("train", "models"):
        (outdir / d).mkdir(parents=True, exist_ok=True)
    op = outdir / "train" / f"{tag}.npz"
    if op.exists() and args.resume:
        print(f"[train] skip {tag}"); return
    with np.load(Path(args.s3e) / "axis_targets.npz") as z:
        aux_np = z[f"aux/{fold}"]
    vs = meta["val"][fold]
    tr = [i for i, g in enumerate(groups) if g not in (fold, vs)]
    va = [i for i, g in enumerate(groups) if g == vs]
    te = [i for i, g in enumerate(groups) if g == fold]
    V = [arr[f"{t}/vec"] for t in tasks]
    R, NS, SIG = g91._labels(arr, tasks, "L4")
    W = g89.load_windows("A", fold, tasks, Path(args.s3b), synthetic=args.synthetic)
    torch.manual_seed(seed)
    model = build_field_e(V[0].shape[1]).to(dev)
    aux_head = torch.nn.Linear(256, 3).to(dev)
    blind_sd = torch.load(Path(args.s3ea) / "models" / f"blind_s{seed}_{fold}.pt", map_location=dev)
    params = init_residual(model, blind_sd) + list(aux_head.parameters())
    opt = torch.optim.Adam(params, lr=LR_E)
    aux_t = torch.as_tensor(aux_np, dtype=torch.float32, device=dev)
    sc = g89.Scorer(model, "A", dev)
    rng = np.random.default_rng(seed)

    def weighted(i, trip):
        out = []
        for a, b, y in trip:
            w = g91.l4_weight(R[i][a], R[i][b], SIG[i], NS[i][a], NS[i][b])
            if w > 0:
                out.append((i, a, b, y, w))
        return out

    val_pairs = [p for i in va for p in weighted(i, g89.all_ordered_pairs(R[i]))]

    def batch_loss(batch, sub, with_aux):
        tids = sorted({b[0] for b in batch}); pos = {t: j for j, t in enumerate(tids)}
        T = torch.stack([sc.task_vec(W[i], None, rng, sub) for i in tids])
        va_ = torch.as_tensor(np.stack([V[b[0]][b[1]] for b in batch]), dtype=torch.float32, device=dev)
        vb_ = torch.as_tensor(np.stack([V[b[0]][b[2]] for b in batch]), dtype=torch.float32, device=dev)
        y = torch.as_tensor([float(b[3]) for b in batch], dtype=torch.float32, device=dev)
        w = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32, device=dev)
        p = sc.forward(va_, vb_, T[[pos[b[0]] for b in batch]]).clamp(1e-6, 1 - 1e-6)
        loss = -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)
        if with_aux:
            loss = loss + AUX_BETA_F * ((aux_head(T) - aux_t[tids]) ** 2).mean()
        return loss

    def val_loss():
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]
                vl += float(batch_loss(b, False, False)) * len(b); vn += len(b)
        return vl / max(vn, 1)

    t0 = time.time()
    start = best = val_loss()
    best_state, best_at, bad = copy.deepcopy(model.state_dict()), (-1, 0), 0
    hist = [{"epoch": -1, "q": 0, "train_loss": None, "val_loss": start, "sec": round(time.time() - t0, 1)}]
    print(f"[train] {tag} start (= blind) val {start:.4f}", flush=True)
    stop = False
    for ep in range(args.max_epochs or MAX_EPOCHS_F):
        model.train()
        pairs = [p for i in tr for p in weighted(i, g89.linear_pairs(R[i], rng))]
        perm = rng.permutation(len(pairs))
        nb = math.ceil(len(perm) / g89.BATCH)
        marks = {max(0, int(round(nb * k / EVALS_PER_EPOCH)) - 1): k for k in range(1, EVALS_PER_EPOCH + 1)}
        tot, cnt = 0.0, 0
        for bi, s in enumerate(range(0, len(perm), g89.BATCH)):
            loss = batch_loss([pairs[j] for j in perm[s:s + g89.BATCH]], True, True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
            if bi in marks:
                vl = val_loss(); model.train()
                hist.append({"epoch": ep, "q": marks[bi], "train_loss": tot / max(cnt, 1), "val_loss": vl, "sec": round(time.time() - t0, 1)})
                print(f"[train] {tag} ep {ep} q {marks[bi]} train {tot / max(cnt, 1):.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
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
    with torch.no_grad():
        Tall = torch.stack([sc.task_vec(W[i]) for i in range(len(tasks))])
        out["T_all"] = Tall.float().cpu().numpy()
        for i in te + va:
            out[f"P/{tasks[i]}"] = sc.pmat(V[i], Tall[i]).astype(np.float32)
        for i in te:
            out[f"X/{tasks[i]}"] = np.stack([sc.pmat(V[i][:g89.N_SH], Tall[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done start {start:.4f} best {best:.4f} at {best_at} evals {len(hist) - 1} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 搜索（服务器）
def _load_models(build, paths: Sequence[Path], dev):
    import torch
    out = []
    for p in paths:
        m = build().to(dev); m.load_state_dict(torch.load(p, map_location=dev)); m.eval(); out.append(m)
    return out


def cmd_search(args) -> None:
    import torch
    ac, g89, g91, g92 = _g("acts_common"), _g("89_acts_s3c"), _g("91_acts_s3e"), _g("92_acts_s3ea")
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.labeling import run_seed_for
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.task_embed import TaskSetEncoder
    outdir, s3e, s3ea = Path(args.outdir), Path(args.s3e), Path(args.s3ea)
    meta, _arr = g89.load_data(Path(args.s3c)); tasks = meta["tasks"]; groups = meta["groups"]; fold, dev = args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    star = args.star or json.loads((s3e / "select.json").read_text(encoding="utf-8"))["star"]
    head, _a, _l = g91.parse_arm(star)
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    seeds = args.seeds or g91.ENS_SEEDS
    ens_f = g92.ArmEnsemble(_load_models(lambda: build_field_e(enc.dim), [outdir / "models" / f"F_s{sd}_{fold}.pt" for sd in seeds], dev), "A", dev)
    ens_b = g92.ArmEnsemble(_load_models(lambda: g92.build_blind(enc.dim), [s3ea / "models" / f"blind_s{sd}_{fold}.pt" for sd in seeds], dev), "D", dev)
    ens_e = g92.ArmEnsemble(_load_models(lambda: TCLSC(enc.dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head=head),
                                         [s3e / "models" / f"{star}_s{sd}_{fold}.pt" for sd in seeds], dev), "A", dev)
    ex = exam_tasks(tasks, groups, [fold]); old = set(meta["search"][fold])
    cx = g92.cross_index(tasks, groups, meta["order"]); xsrc = [tasks[cx[tasks.index(t)]] for t in ex]
    Wo = g89.load_windows("A", fold, ex, Path(args.s3b), synthetic=args.synthetic)
    Wx = g89.load_windows("A", fold, xsrc, Path(args.s3b), synthetic=args.synthetic)
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t, xs, w, wx in zip(ex, xsrc, Wo, Wx):
        with torch.no_grad():
            Tf = [sc.task_vec(w) for sc in ens_f.sc]; Tfx = [sc.task_vec(wx) for sc in ens_f.sc]
        rec = {"x_task": xs, "new_row": t not in old}
        rec.update(g92._search_task(ens_f, Tf, t, lcb("F-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens_f, Tfx, t, lcb("FX-lcb"), space, enc, cfg, ks, gens))
        if t not in old:
            with torch.no_grad():
                Te = [sc.task_vec(w) for sc in ens_e.sc]
            rec.update(g92._search_task(ens_b, [None] * len(ens_b.sc), t, lcb("B-lcb"), space, enc, cfg, ks, gens))
            rec.update(g92._search_task(ens_e, Te, t, lcb("E-lcb"), space, enc, cfg, ks, gens))
            cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(t, "s3c_ks")), ks, set())
            rec["R"] = [{"cls_key": k, "cls": c.to_dict()} for c, k in cands[:3]]
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in ARMS if a in rec) + f" xE←{xs} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def cmd_plan_runs(args) -> None:
    ac, g35, g89, g92 = _g("acts_common"), _g("35_protocol_alignment"), _g("89_acts_s3c"), _g("92_acts_s3ea")
    from autocls.space.cls_space import CLS
    outdir, s3c = Path(args.outdir), Path(args.s3c)
    meta, _arr = g89.load_data(s3c)
    cfg = ac.load_acts_config(profile="s2"); protos = g35.load_protocols()
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8")); tm = tj["tasks"]
    done = g92._done_tags(meta, [s3c, Path(args.s3e), Path(args.s3ea), outdir])
    noms, byk = {}, {}
    for f in sorted((outdir / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in ARMS if a in rec}
            for a in ARMS:
                for e in rec.get(a, []):
                    byk[e["cls_key"]] = e["cls"]
    cells = []
    for tag in g92.new_tags(noms, done):
        t, k, sd = tag.split("|")
        cells.append(ac.make_cell(t, tm[t], ac.ARM_RANDOM, CLS(**byk[k]), k, int(sd), protos, cfg))
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    st = list(noms)
    plan = {"phase": "X", "profile": cfg.get("profile"), "n_cells": len(cells), "probe_tasks": st, "protocol": cfg["protocol"],
            "tasks_meta": {t: {k: tm[t][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in tm[t]} for t in st},
            "cells": cells}
    (rd / "plan_X.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (rd / "tasks.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    print(f"[plan-runs] {len(cells)} new cells（复用 S2 / S3-C / S3-E / S3-EA 已有格）", flush=True)


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, spearmanr, wilcoxon
    from autocls.search.rank import symmetrize, wins_from_scores
    g83, g84, g86, g89, g91, g92 = (_g(m) for m in ("83_acts_corpus", "84_acts_s3a", "86_acts_s3b", "89_acts_s3c", "91_acts_s3e", "92_acts_s3ea"))
    outdir, s3c, s3e, s3ea = Path(args.outdir), Path(args.s3c), Path(args.s3e), Path(args.s3ea)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    star = json.loads((s3e / "select.json").read_text(encoding="utf-8"))["star"]
    # ---- 真值
    rows = dict(g83._g35.load_done(S2_DIR / "runs.jsonl")); plans = [json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8"))]
    for d in (s3c, s3e, s3ea, outdir):
        if (d / "runs" / "runs.jsonl").exists():
            rows.update(g83._g35.load_done(d / "runs" / "runs.jsonl"))
        if (d / "runs" / "plan_X.json").exists():
            plans.append(json.loads((d / "runs" / "plan_X.json").read_text(encoding="utf-8")))
    val = {}
    for pl in plans:
        for c in pl["cells"]:
            r = rows.get(c["key"])
            if r and not r.get("error") and r.get("checkpoint_results"):
                val[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
    # ---- 提名
    ex = exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    sf = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}
    se = {s: js(s3e / "search" / f"{s}.json") for s in g89.FOLDS}
    sa = {s: js(s3ea / "search" / f"{s}.json") for s in g89.FOLDS}
    sc_ = {s: js(s3c / "search" / f"A_{s}.json") for s in g89.FOLDS}
    noms = {}
    for t, s in zip(ex, sg):
        f = sf[s][t]; new = f["new_row"]
        noms[t] = {"F-lcb": f["F-lcb"], "FX-lcb": f["FX-lcb"],
                   "B-lcb": f["B-lcb"] if new else sa[s][t]["B-lcb"], "E-lcb": f["E-lcb"] if new else se[s][t]["lcb"],
                   "R": f["R"] if new else sc_[s][t]["random"]}
    P1, M3, miss = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: 0 for a in ARMS}
    cat_p1, cat_m3, orc = [], [], []
    for t in ex:
        i = tasks.index(t); fl = float(np.nanmin(end0[i]))
        for a in ARMS:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
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
    pairs = (("F-lcb", "B-lcb"), ("F-lcb", "FX-lcb"), ("F-lcb", "E-lcb"), ("E-lcb", "B-lcb"), ("F-lcb", "cat3"), ("B-lcb", "cat3"), ("E-lcb", "cat3"),
             ("F-lcb", "R"), ("B-lcb", "R"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    old_rows = np.array([not sf[s][t]["new_row"] for t, s in zip(ex, sg)])
    C_half = {h: {f"{a}−{b}": cmp(M3[a][m], M3[b][m]) for a, b in (("F-lcb", "B-lcb"), ("E-lcb", "B-lcb"))} for h, m in (("odd", old_rows), ("even", ~old_rows))}
    # ---- 训练
    zf = {s: [np.load(outdir / "train" / f"F_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    ba = [tuple(int(x) for x in z["best_at"]) for s in g89.FOLDS for z in zf[s]]
    gain = [float(z["start_val"]) - float(z["best_val"]) for s in g89.FOLDS for z in zf[s]]
    n_gain = int(sum(b[0] >= 0 for b in ba))
    train = {"n_gain": n_gain, "best_at": ba, "val_gain": gain, "val_gain_mean": float(np.mean(gain)),
             "per_fold_gain": {s: float(np.mean([float(z["start_val"]) - float(z["best_val"]) for z in zf[s]])) for s in g89.FOLDS},
             "best_epoch_median": float(np.median([b[0] for b in ba]))}
    code = verdict_s3f(C["F-lcb−B-lcb"]["m3"], C["F-lcb−B-lcb"]["p1"], C["F-lcb−FX-lcb"]["m3"],
                       bool(C["F-lcb−cat3"]["m3"]["d"] >= 0 and C["F-lcb−cat3"]["p1"]["d"] >= 0), n_gain)
    # ---- 排序（56 留出任务）
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    zb = {s: [np.load(s3ea / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    ze = {s: [np.load(s3e / "train" / f"{star}_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    rf, spf = g92._rank_readout(zf, meta, arr, held); rb, spb = g92._rank_readout(zb, meta, arr, held); re_, spe = g92._rank_readout(ze, meta, arr, held)

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    rx = np.zeros(len(held))
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; xs = []
        for k in range(16):
            Wx = np.mean([wins_of(z[f"X/{t}"][k]) for z in zf[groups[i]]], axis=0)
            ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
            xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
        rx[jj] = float(np.mean(xs))
    E = np.vstack([zf[s][0]["T_all"][[i]] for s in g89.FOLDS for i in held if groups[i] == s])
    eg = np.array([groups[i] for s in g89.FOLDS for i in held if groups[i] == s])
    cw, cb = g86.cosine_within_between(E, eg)
    rank = {"F": {"regret3": float(rf.mean()), "sp": float(np.nanmedian(spf))}, "B": {"regret3": float(rb.mean()), "sp": float(np.nanmedian(spb))},
            "★": {"regret3": float(re_.mean()), "sp": float(np.nanmedian(spe))},
            "F−B": {"d": float((rf - rb).mean()), "p_source": g84.source_wilcoxon(rf - rb, hg)},
            "cross": {"d": float((rx - rf).mean()), "ci": g84.cluster_bootstrap_ci(rx - rf, hg), "p_source": g84.source_wilcoxon(rx - rf, hg), "npos": g89.n_sources_positive(rx - rf, hg)},
            "cos_within": cw, "cos_between": cb}
    per_src = {a: {s: {"p1": float(P1[a][sg == s].mean()), "m3": float(M3[a][sg == s].mean()) if a in M3 else None} for s in g89.FOLDS} for a in list(ARMS) + ["cat3", "oracle"]}
    out = {"verdict": code, "star": star, "n_tasks": len(ex), "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "cmp_half": C_half, "missing": miss, "train": train, "rank": rank, "per_source": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-F 读数（{code}）", "", f"## 搜索（{len(ex)} 任务；3 选 1 = seed 0 选、seed 1 评；3 条平均 = 两种子均值的平均）", "",
         "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(ARMS) + ["cat3", "oracle"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'].get(a))} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"]
    for k, v in C.items():
        L.append(f"| {k} | {g(v['m3'])} | {g(v['p1'])} |")
    L += ["", "奇 / 偶行分开（3 条平均）：" + "；".join(f"{h} " + "，".join(f"{k} {v['d']:+.3f}（p {v['p']:.2f}）" for k, v in d.items()) for h, d in C_half.items()), "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(ARMS) + ["cat3"]) + " |", "|---|" + "---:|" * (len(ARMS) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]["m3"]) for a in list(ARMS) + ["cat3"]) + " |")
    L += ["", "## 训练（残差式 field-E）", "",
          f"- 最好一步不在起点：{n_gain}/30；验证损失下降均值 {train['val_gain_mean']:+.4f}；最好 epoch 中位 {train['best_epoch_median']}",
          "- 逐折验证损失下降：" + "，".join(f"{s} {v:+.4f}" for s, v in train["per_fold_gain"].items()), "",
          "## 排序（56 留出任务，λ = 0 胜场均值集成）", "",
          f"- regret@3：F {f(rank['F']['regret3'])} / B {f(rank['B']['regret3'])} / ★ {f(rank['★']['regret3'])}；F − B {rank['F−B']['d']:+.4f}（源级 p {rank['F−B']['p_source']:.3f}）",
          f"- random-20 中位 Sp：F {f(rank['F']['sp'])} / B {f(rank['B']['sp'])} / ★ {f(rank['★']['sp'])}",
          f"- 换源 Δ（16 偏移）{rank['cross']['d']:+.4f} [{f(rank['cross']['ci'][0])}, {f(rank['cross']['ci'][1])}]，源级 p {rank['cross']['p_source']:.3f}，同号 {rank['cross']['npos']}/6；E 余弦 集内 {f(cw)} / 跨集 {f(cb)}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    import types
    g89, g91, g92 = _g("89_acts_s3c"), _g("91_acts_s3e"), _g("92_acts_s3ea")
    od = Path(args.outdir); se, sa = od / "_s3e", od / "_s3ea"
    for d in (se, sa):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(S3E / "axis_targets.npz", se / "axis_targets.npz")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "s3e": str(se), "s3ea": str(sa), "star": "field-on-L4", "seeds": [0, 1]})
    fold = g89.FOLDS[0]
    for sd in (0, 1):
        g91.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(se), "arm": "field-on-L4", "fold": fold, "seed": sd}))
        g92.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sa), "fold": fold, "seed": sd}))
        cmd_train(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    cmd_search(types.SimpleNamespace(**{**base, "fold": fold}))
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == N_ROWS and sum(v["new_row"] for v in r.values()) == 4
    assert all(len(v[a]) == 3 for v in r.values() for a in ("F-lcb", "FX-lcb")) and all(len(v[a]) == 3 for v in r.values() if v["new_row"] for a in ("B-lcb", "E-lcb", "R"))
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("train", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--star", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--s3c", default=str(S3C))
    ap.add_argument("--s3e", default=str(S3E))
    ap.add_argument("--s3ea", default=str(S3EA))
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
