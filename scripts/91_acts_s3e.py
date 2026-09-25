"""WP17-ACTS S3-E（results/acts/s3e/PROTOCOL.md）：提升 T-CLSC——diff / field 头 × E 辅助监督 × L0 / L4 标签筛选，★ 臂 5 种子集成 + 胜场 LCB 搜索。

阶段：
  --prep          本地：辅助目标（每折低秩轴坐标 a(t)，训练侧标准化）→ axis_targets.npz（其余数据复用 S3-C 的 data.json / data.npz）
  --train         服务器：--arm <head>-<on|off>-<L0|L4> --fold <源> --seed s → models/、train/<arm>_s<seed>_<fold>.npz
  --select        服务器：验证源 regret@3 → select.json（★ 臂）
  --search        服务器：--fold <源> → search/<fold>.json（E-mean / E-lcb / E-lcb-div 三组提名）
  --plan-runs     服务器：提名去重（S2 / S3-C 已有格复用）→ runs/plan_X.json（83 phase X）
  --analyze       本地
  --smoke         本地冒烟（合成窗口）
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3e"
S3C = ROOT / "results" / "acts" / "s3c"
S2_DIR = ROOT / "results" / "acts" / "s2"
HEADS = ("diff", "field")
ARMS = tuple(f"{h}-{a}-{l}" for h in HEADS for a in ("on", "off") for l in ("L4", "L0"))
ENS_SEEDS = (0, 1, 2, 3, 4)
AUX_BETA = 1.0
LAM = 1.0


# ================================================================ 纯函数
def parse_arm(arm: str) -> tuple[str, bool, str]:
    h, a, l = arm.split("-")
    return h, a == "on", l


def l4_weight(ra: float, rb: float, sigma: float, na: int, nb: int) -> float:
    d = abs(float(ra) - float(rb))
    if d == 0:
        return 0.0
    s = float(sigma) * math.sqrt(1.0 / na + 1.0 / nb)
    return 1.0 if s <= 0 else float(min(d / (2.0 * s), 1.0))


def ens_fitness(W: np.ndarray, lam: float) -> np.ndarray:
    W = np.asarray(W, float)
    return W.mean(0) - float(lam) * W.std(0)


def ens_order(W: np.ndarray, keys: Sequence[str], lam: float) -> list[int]:
    f = ens_fitness(W, lam)
    return sorted(range(len(keys)), key=lambda i: (-float(f[i]), keys[i]))


def diversified_top(keys: Sequence[str], family: Mapping[str, tuple], k: int = 3) -> list[str]:
    out, seen = [], set()
    for x in keys:
        if family[x] not in seen:
            out.append(x); seen.add(family[x])
        if len(out) == k:
            return out
    for x in keys:
        if x not in out:
            out.append(x)
        if len(out) == k:
            break
    return out


def select_arm(val_regret: Mapping[str, float]) -> str:
    def tie(a):
        h, on, l = parse_arm(a)
        return (h != "diff", not on, l != "L4")
    best = min(val_regret.values())
    return min((a for a, v in val_regret.items() if v <= best + 1e-12), key=tie)


def verdict_s3e(S: Mapping[str, bool], R: Mapping[str, bool]) -> str:
    base = "S3E-SEARCH-LOSS" if not S["S2"] else ("S3E-SEARCH-WIN" if S["S1"] and S["S3"] else "S3E-CATALOG-PAR")
    return base + "".join(f"+{k}" for k in ("R1", "R2", "R3") if R.get(k))


def family_of(cls: Mapping) -> tuple:
    return (cls["loss_type"], cls["sim_func"], cls["norm"])


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap"); g89 = importlib.import_module("scripts.89_acts_s3c")
    data = g85.load_all(); D = data["pool"]; groups = D["groups"]; b_ref = g85.orient_bref(D)
    meta, _arr = g89.load_data(S3C)
    assert [str(t) for t in D["tasks"]] == meta["tasks"]
    out = {}
    for s in g89.FOLDS:
        tr = np.where(groups != s)[0]
        A = g84._fold(D["Z"], tr, 3, b_ref)["A"][:, :3]
        m, sd = A[tr].mean(0), A[tr].std(0)
        out[f"aux/{s}"] = ((A - m) / np.where(sd > 1e-12, sd, 1.0)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "axis_targets.npz", **out)
    print(f"[prep] aux targets for {len(g89.FOLDS)} folds → axis_targets.npz", flush=True)


# ================================================================ 训练（服务器）
def _labels(arr: Mapping, tasks: Sequence[str], kind: str) -> tuple[list, list, list]:
    """每任务 (r, n_seeds, σ)。L0 = best-of-3 seed 0（S3-C 原样）；L4 = shared 两种子终点均值 + random seed-0 终点。"""
    ac = importlib.import_module("scripts.acts_common"); g89 = importlib.import_module("scripts.89_acts_s3c")
    N = g89.N_SH
    R, NS, SIG = [], [], []
    for i, t in enumerate(tasks):
        e0, e1 = arr["end0"][i], arr["end1"][i]
        sig = ac.top_half_sigma(e0, e1)[0]
        if kind == "L0":
            r = arr[f"{t}/r"]; n = np.ones(len(r), int)
        else:
            R0 = arr[f"{t}/R0"]
            r = np.concatenate([(e0 + e1) / 2.0, R0]); n = np.array([2] * N + [1] * len(R0))
        R.append(np.asarray(r, float)); NS.append(n); SIG.append(float(sig) if np.isfinite(sig) else 0.0)
    return R, NS, SIG


def cmd_train(args) -> None:
    import torch
    g88 = importlib.import_module("scripts.88_acts_s3bpp"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.task_embed import TaskSetEncoder
    outdir, s3c = Path(args.outdir), Path(args.s3c)
    meta, arr = g89.load_data(s3c)
    with np.load(outdir / "axis_targets.npz") as z:
        aux_all = {k: z[k] for k in z.files}
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, arm, seed, dev = args.fold, args.arm, int(args.seed), args.device
    head, aux_on, lab = parse_arm(arm)
    tag = f"{arm}_s{seed}_{fold}"
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
    R, NS, SIG = _labels(arr, tasks, lab)
    W = g89.load_windows("A", fold, tasks, Path(args.s3b), synthetic=args.synthetic)
    torch.manual_seed(seed)
    model = TCLSC(V[0].shape[1], TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head=head).to(dev)
    aux_head = torch.nn.Linear(256, 3).to(dev) if aux_on else None
    aux_t = torch.as_tensor(aux_all[f"aux/{fold}"], dtype=torch.float32, device=dev)
    sc = g89.Scorer(model, "A", dev)
    params = list(model.parameters()) + (list(aux_head.parameters()) if aux_head is not None else [])
    opt = torch.optim.Adam(params, lr=g89.LR)
    rng = np.random.default_rng(seed)

    def pairs_of(i, rs):
        out = []
        for a, b, y in g89.linear_pairs(R[i], rs):
            w = 1.0 if lab == "L0" else l4_weight(R[i][a], R[i][b], SIG[i], NS[i][a], NS[i][b])
            if w > 0:
                out.append((i, a, b, y, w))
        return out

    val_pairs = []
    for i in va:
        for a, b, y in g89.all_ordered_pairs(R[i]):
            w = 1.0 if lab == "L0" else l4_weight(R[i][a], R[i][b], SIG[i], NS[i][a], NS[i][b])
            if w > 0:
                val_pairs.append((i, a, b, y, w))

    def batch_loss(batch, sub, with_aux):
        tids = sorted({b[0] for b in batch}); pos = {t: j for j, t in enumerate(tids)}
        T = torch.stack([sc.task_vec(W[i], None, rng, sub) for i in tids])
        va_ = torch.as_tensor(np.stack([V[b[0]][b[1]] for b in batch]), dtype=torch.float32, device=dev)
        vb_ = torch.as_tensor(np.stack([V[b[0]][b[2]] for b in batch]), dtype=torch.float32, device=dev)
        y = torch.as_tensor([float(b[3]) for b in batch], dtype=torch.float32, device=dev)
        w = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32, device=dev)
        p = sc.forward(va_, vb_, T[[pos[b[0]] for b in batch]]).clamp(1e-6, 1 - 1e-6)
        bce = -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)
        if with_aux and aux_head is not None:
            bce = bce + AUX_BETA * ((aux_head(T) - aux_t[tids]) ** 2).mean()
        return bce

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(args.max_epochs or g89.MAX_EPOCHS):
        model.train()
        pairs = [p for i in tr for p in pairs_of(i, rng)]
        perm = rng.permutation(len(pairs))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), g89.BATCH):
            loss = batch_loss([pairs[j] for j in perm[s:s + g89.BATCH]], True, True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]
                vl += float(batch_loss(b, False, False)) * len(b); vn += len(b)
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl, "sec": round(time.time() - t0, 1)})
        print(f"[train] {tag} ep {ep} train {tot / max(nb, 1):.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(aux_head.state_dict()) if aux_head is not None else None)
        else:
            bad += 1
            if bad >= g89.PATIENCE:
                break
    model.load_state_dict(best_state[0]); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    cmaps = g88.cross_maps(tasks, groups, order)
    out = {"hist": json.dumps(hist), "te": np.array(te), "va": np.array(va)}
    with torch.no_grad():
        Tall = torch.stack([sc.task_vec(W[i]) for i in range(len(tasks))])
        out["T_all"] = Tall.float().cpu().numpy()
        for i in te + va:
            out[f"P/{tasks[i]}"] = sc.pmat(V[i], Tall[i]).astype(np.float32)
        for i in te:
            out[f"X/{tasks[i]}"] = np.stack([sc.pmat(V[i][:g89.N_SH], Tall[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best val {best:.4f} epochs {len(hist)} {time.time() - t0:.0f}s", flush=True)


def _rank_regret(P: np.ndarray, keys: Sequence[str], e0: np.ndarray, e1: np.ndarray, k: int = 3) -> float:
    g89 = importlib.import_module("scripts.89_acts_s3c")
    N = g89.N_SH
    o = g89.rr_order(P[:N, :N], list(keys)[:N])[:k]
    pick = o[int(np.argmax(e0[o]))]
    return float(e1[int(np.argmax(e0))] - e1[pick])


def cmd_select(args) -> None:
    g89 = importlib.import_module("scripts.89_acts_s3c")
    outdir = Path(args.outdir); meta, arr = g89.load_data(Path(args.s3c))
    tasks = meta["tasks"]
    val = {}
    for arm in ARMS:
        per_fold = []
        for s in g89.FOLDS:
            with np.load(outdir / "train" / f"{arm}_s0_{s}.npz") as z:
                va = z["va"].tolist()
                regs = [_rank_regret(z[f"P/{tasks[i]}"], meta["keys"][tasks[i]], arr["end0"][i], arr["end1"][i]) for i in va]
            per_fold.append(float(np.mean(regs)))
        val[arm] = float(np.mean(per_fold))
    star = select_arm(val)
    (outdir / "select.json").write_text(json.dumps({"val_regret3": val, "star": star}, indent=1), encoding="utf-8")
    print(f"[select] ★ = {star}；验证源 regret@3 {json.dumps({k: round(v, 4) for k, v in val.items()})}", flush=True)


# ================================================================ 集成搜索（服务器）
class Ensemble:
    def __init__(self, models, dev):
        g89 = importlib.import_module("scripts.89_acts_s3c")
        self.sc = [g89.Scorer(m, "A", dev) for m in models]

    def wins(self, V: np.ndarray, Ts: Sequence) -> np.ndarray:
        from autocls.search.rank import symmetrize, wins_from_scores
        out = []
        for sc, T in zip(self.sc, Ts):
            P = sc.pmat(V, T); np.fill_diagonal(P, 0.5)
            out.append(wins_from_scores(symmetrize(P)))
        return np.array(out, float)


def _evolve(ens: Ensemble, Ts, space, enc, cfg, init: Sequence, rng: np.random.Generator, lam: float, kp: int, gens: int, p_cross: float, p_mut: float):
    ac = importlib.import_module("scripts.acts_common")
    from autocls.labeling import cls_key
    from autocls.search.evolution import _make_offspring
    accept = lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg)  # noqa: E731
    pop = list(init)

    def rank(cs):
        V = np.stack([enc.encode(c) for c in cs]).astype(np.float32); keys = [cls_key(c) for c in cs]
        o = ens_order(ens.wins(V, Ts), keys, lam)
        return [cs[i] for i in o]

    pop = rank(pop)[:kp]
    for _g in range(gens):
        seen = {cls_key(c) for c in pop}; kids = []
        for _ in range(kp):
            ch = _make_offspring(pop, rng, space, p_cross, p_mut, accept=accept)
            if ch is None or cls_key(ch) in seen:
                continue
            seen.add(cls_key(ch)); kids.append(ch)
        pop = rank(pop + kids)[:kp]
    return rank(pop)


def cmd_search(args) -> None:
    import torch
    ac = importlib.import_module("scripts.acts_common"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.labeling import cls_key, run_seed_for
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.task_embed import TaskSetEncoder
    outdir, s3c = Path(args.outdir), Path(args.s3c)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; fold, dev = args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    star = args.arm or json.loads((outdir / "select.json").read_text(encoding="utf-8"))["star"]
    head, _aux, _lab = parse_arm(star)
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    models = []
    for sd in (args.seeds or ENS_SEEDS):
        m = TCLSC(enc.dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head=head).to(dev)
        m.load_state_dict(torch.load(outdir / "models" / f"{star}_s{sd}_{fold}.pt", map_location=dev)); m.eval(); models.append(m)
    ens = Ensemble(models, dev)
    W = g89.load_windows("A", fold, [t for t in meta["search"][fold]], Path(args.s3b), synthetic=args.synthetic)
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    res = {}
    t0 = time.time()
    for t, w in zip(meta["search"][fold], W):
        with torch.no_grad():
            Ts = [sc.task_vec(w) for sc in ens.sc]
        cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(t, "s3c_ks")), ks, set())
        Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
        rec = {"star": star}
        for mode, lam in (("mean", 0.0), ("lcb", LAM)):
            fin = []
            for s in range(0, len(cands), g89.CHUNK):
                idx = list(range(s, min(len(cands), s + g89.CHUNK)))
                o = ens_order(ens.wins(Vc[idx], Ts), [keys[i] for i in idx], lam)
                fin += [idx[j] for j in o[:g89.PER_CHUNK]]
            o = ens_order(ens.wins(Vc[fin], Ts), [keys[i] for i in fin], lam)
            init = [cands[fin[j]][0] for j in o[:g89.KP]]
            final = _evolve(ens, Ts, space, enc, cfg, init, np.random.default_rng(run_seed_for(t, "s3c_evo")), lam, g89.KP, gens, g89.P_CROSS, g89.P_MUT)
            fk = [cls_key(c) for c in final]; fam = {cls_key(c): family_of(c.to_dict()) for c in final}
            rec[mode] = [{"cls_key": cls_key(c), "cls": c.to_dict()} for c in final[:3]]
            if mode == "lcb":
                div = diversified_top(fk, fam, 3); byk = {cls_key(c): c for c in final}
                rec["lcbdiv"] = [{"cls_key": k, "cls": byk[k].to_dict()} for k in div]
        res[t] = rec
        print(f"[search] {fold} {t} mean {[x['cls_key'] for x in rec['mean']]} lcb {[x['cls_key'] for x in rec['lcb']]} div {[x['cls_key'] for x in rec['lcbdiv']]} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def cmd_plan_runs(args) -> None:
    ac = importlib.import_module("scripts.acts_common"); g35 = importlib.import_module("scripts.35_protocol_alignment"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.space.cls_space import CLS
    outdir, s3c = Path(args.outdir), Path(args.s3c)
    meta, _arr = g89.load_data(s3c)
    cfg = ac.load_acts_config(profile="s2"); protos = g35.load_protocols()
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8")); tm = tj["tasks"]
    done = set(meta["s2_done"])
    p = s3c / "runs" / "runs.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if not r.get("error"):
                done.add(f"{r['task_id']}|{r['cls_key']}|{int(r['seed'])}")
    cells, seen = [], set()
    for f in sorted((outdir / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            for mode in ("mean", "lcb", "lcbdiv"):
                for e in rec[mode]:
                    for sd in (0, 1):
                        tag = f"{t}|{e['cls_key']}|{sd}"
                        if tag in done or tag in seen:
                            continue
                        seen.add(tag)
                        cells.append(ac.make_cell(t, tm[t], ac.ARM_RANDOM, CLS(**e["cls"]), e["cls_key"], sd, protos, cfg))
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    st = [t for v in meta["search"].values() for t in v]
    plan = {"phase": "X", "profile": cfg.get("profile"), "n_cells": len(cells), "probe_tasks": st, "protocol": cfg["protocol"],
            "tasks_meta": {t: {k: tm[t][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in tm[t]} for t in st},
            "cells": cells}
    (rd / "plan_X.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (rd / "tasks.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    print(f"[plan-runs] {len(cells)} new cells（复用 S2 / S3-C 已有格）", flush=True)


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import spearmanr, wilcoxon
    g83 = importlib.import_module("scripts.83_acts_corpus"); g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    g86 = importlib.import_module("scripts.86_acts_s3b"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.search.rank import symmetrize, wins_from_scores
    outdir, s3c = Path(args.outdir), Path(args.s3c)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"]); N = g89.N_SH
    sel = json.loads((outdir / "select.json").read_text(encoding="utf-8")); star = sel["star"]
    end0, end1 = arr["end0"], arr["end1"]
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    s3c_rank = json.loads((s3c / "per_task.json").read_text(encoding="utf-8"))

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)

    def readout(models: Sequence[str]):
        """models = [arm_s<seed>] 列表；多个时按胜场均值（λ = 0）集成。"""
        r3, sp, rx = np.full(len(held), np.nan), np.full(len(held), np.nan), np.full(len(held), np.nan)
        Es = []
        zs = {s: [np.load(outdir / "train" / f"{m}_{s}.npz") for m in models] for s in g89.FOLDS}
        for jj, i in enumerate(held):
            t = tasks[i]; s = groups[i]; keys = meta["keys"][t]
            Wf = np.mean([wins_of(z[f"P/{t}"]) for z in zs[s]], axis=0)
            Ws = np.mean([wins_of(z[f"P/{t}"][:N, :N]) for z in zs[s]], axis=0)
            o = sorted(range(N), key=lambda j: (-Ws[j], keys[j]))[:3]
            r3[jj] = end1[i, int(np.argmax(end0[i]))] - end1[i, o[int(np.argmax(end0[i, o]))]]
            R0 = arr[f"{t}/R0"]
            if len(R0) >= 10:
                sp[jj] = float(spearmanr(Wf[N:N + len(R0)], R0)[0])
            xs = []
            for k in range(16):
                Wx = np.mean([wins_of(z[f"X/{t}"][k]) for z in zs[s]], axis=0)
                ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
                xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
            rx[jj] = float(np.mean(xs))
        E = np.vstack([zs[s][0]["T_all"][[i]] for s in g89.FOLDS for i in held if groups[i] == s])
        eg = np.array([groups[i] for s in g89.FOLDS for i in held if groups[i] == s])
        cw, cb = g86.cosine_within_between(E, eg)
        d = rx - r3
        return {"regret3": float(np.mean(r3)), "sp_median": float(np.nanmedian(sp)), "cross": float(np.mean(d)), "cross_ci": g84.cluster_bootstrap_ci(d, hg),
                "cross_p": g84.source_wilcoxon(d, hg), "cross_npos": g89.n_sources_positive(d, hg), "cos_within": cw, "cos_between": cb, "_r3": r3}

    ranks = {arm: readout([f"{arm}_s0"]) for arm in ARMS}
    ranks["★ensemble"] = readout([f"{star}_s{s}" for s in ENS_SEEDS])
    A_r3 = np.array([s3c_rank["rank"][tasks[i]]["A"] for i in held])
    blind_r3 = np.array([s3c_rank["rank"][tasks[i]]["blind"] for i in held])
    for k, r in ranks.items():
        d = A_r3 - r["_r3"]; r["vs_s3cA"] = float(d.mean()); r["vs_s3cA_p"] = g84.source_wilcoxon(d, hg)
        r["vs_blind"] = float((blind_r3 - r["_r3"]).mean())
    e = ranks["★ensemble"]
    R = {"R1": bool(e["cross"] >= 0.005 and e["cross_npos"] == 6), "R2": bool(e["vs_s3cA"] >= 0.003), "R3": bool(e["sp_median"] >= 0.35)}
    # 搜索
    rows = dict(g83._g35.load_done(S2_DIR / "runs.jsonl"))
    for p in (s3c / "runs" / "runs.jsonl", outdir / "runs" / "runs.jsonl"):
        if p.exists():
            rows.update(g83._g35.load_done(p))
    val = {}
    plans = [json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8")), json.loads((s3c / "runs" / "plan_X.json").read_text(encoding="utf-8"))]
    if (outdir / "runs" / "plan_X.json").exists():
        plans.append(json.loads((outdir / "runs" / "plan_X.json").read_text(encoding="utf-8")))
    for pl in plans:
        for c in pl["cells"]:
            r = rows.get(c["key"])
            if r and not r.get("error") and r.get("checkpoint_results"):
                val[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
    srch_old = s3c_rank["search"]
    stasks = list(srch_old)
    final = {m: [] for m in ("mean", "lcb", "lcbdiv")}
    miss = 0
    for t in stasks:
        s = srch_old[t]["source"]
        rec = json.loads((outdir / "search" / f"{s}.json").read_text(encoding="utf-8"))[t]
        for m in final:
            ks_ = [x["cls_key"] for x in rec[m]]
            v0 = {k: val.get((t, k, 0), np.nan) for k in ks_}; v1 = {k: val.get((t, k, 1), np.nan) for k in ks_}
            if any(not np.isfinite(v) for v in list(v0.values()) + list(v1.values())):
                miss += 1; i = tasks.index(t); fl = float(np.nanmin(end0[i]))
                v0 = {k: (v if np.isfinite(v) else fl) for k, v in v0.items()}; v1 = {k: (v if np.isfinite(v) else fl) for k, v in v1.items()}
            final[m].append(g89.pick_final(ks_, v0, v1)[1])
    sg = np.array([srch_old[t]["source"] for t in stasks])
    ref = {k: np.array([srch_old[t][k] for t in stasks]) for k in ("A", "D", "R", "cat3", "oracle")}

    def cmp(a, b):
        d = np.array(a) - np.array(b)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    search = {m: {"final": float(np.mean(v)), "vs_R": cmp(v, ref["R"]), "vs_cat3": cmp(v, ref["cat3"]), "vs_s3cA": cmp(v, ref["A"]), "vs_s3cD": cmp(v, ref["D"]),
                  "vs_oracle": cmp(v, ref["oracle"]), "per_source": {s: float(np.mean([v[j] for j in range(len(stasks)) if sg[j] == s])) for s in g89.FOLDS}}
              for m, v in final.items()}
    L = search["lcb"]
    S = {"S1": bool(L["vs_R"]["d"] >= 0.010 and L["vs_R"]["p"] < 0.05 and L["vs_R"]["npos"] >= 5), "S2": bool(L["vs_cat3"]["d"] >= 0),
         "S3": bool(L["vs_s3cA"]["d"] >= 0.010 and L["vs_s3cA"]["p"] < 0.05)}
    code = verdict_s3e(S, R)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    out = {"verdict": code, "star": star, "val_regret3": sel["val_regret3"], "rank": {k: strip(v) for k, v in ranks.items()}, "R": R, "S": S, "search": search,
           "refs": {k: float(v.mean()) for k, v in ref.items()}, "refs_per_source": {k: {s: float(np.mean([ref[k][j] for j in range(len(stasks)) if sg[j] == s])) for s in g89.FOLDS} for k in ref},
           "missing_tasks": miss}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"
    Lm = [f"# S3-E 读数（{code}）；★ 臂 = {star}", "", "## 排序（56 个留出任务；S3-C A 0.012 / D 0.015 / 任务盲 μ 0.016）", "",
          "| 臂 | 验证源 regret@3（选臂用） | 留出 regret@3 | vs S3-C A（p） | 换源 Δ [CI] p / 同号源 | random-20 中位 Sp | 余弦 集内 / 跨集 |", "|---|---:|---:|---|---|---:|---|"]
    for k, r in ranks.items():
        vr = sel["val_regret3"].get(k)
        Lm.append(f"| {k} | {f(vr)} | {f(r['regret3'])} | {r['vs_s3cA']:+.3f}（{r['vs_s3cA_p']:.3f}） | {r['cross']:+.3f} [{f(r['cross_ci'][0])}, {f(r['cross_ci'][1])}] {r['cross_p']:.3f} / {r['cross_npos']} | {f(r['sp_median'])} | {f(r['cos_within'])} / {f(r['cos_between'])} |")
    Lm += ["", "排序门（★ 集成）：" + "，".join(f"{k} {'✓' if v else '✗'}" for k, v in R.items()), "",
           "## 搜索（24 任务；seed 0 选、seed 1 评）", "",
           f"参照：S3-C A {f(out['refs']['A'])} / D {f(out['refs']['D'])} / 随机 R {f(out['refs']['R'])} / 目录任务盲前 3 {f(out['refs']['cat3'])} / 目录 oracle {f(out['refs']['oracle'])}", "",
           "| 臂 | 最终 acc | vs R | vs 目录前 3 | vs S3-C A | vs S3-C D | vs oracle |", "|---|---:|---|---|---|---|---|"]
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"
    for m, v in search.items():
        Lm.append(f"| E-{m} | {f(v['final'])} | {g(v['vs_R'])} | {g(v['vs_cat3'])} | {g(v['vs_s3cA'])} | {g(v['vs_s3cD'])} | {g(v['vs_oracle'])} |")
    Lm += ["", "搜索门（E-lcb）：" + "，".join(f"{k} {'✓' if v else '✗'}" for k, v in S.items()), "",
           "## 搜索逐源（最终 acc）", "", "| 源 | E-mean | E-lcb | E-lcb-div | S3-C A | S3-C D | 随机 R | 目录前 3 | 目录 oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in g89.FOLDS:
        Lm.append(f"| {s} | " + " | ".join(f(search[m]["per_source"][s]) for m in ("mean", "lcb", "lcbdiv")) + " | " + " | ".join(f(out["refs_per_source"][k][s]) for k in ("A", "D", "R", "cat3", "oracle")) + " |")
    (outdir / "tables.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")
    print("\n".join(Lm), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    import types
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 2, "resume": False, "ks": 1000, "gens": 2})
    g89 = importlib.import_module("scripts.89_acts_s3c")
    for arm in ("diff-on-L4", "field-off-L0"):          # 冒烟只覆盖两种头 / 两种标签 / 辅助开关两态
        for fold in g89.FOLDS[:1]:
            cmd_train(types.SimpleNamespace(**{**base, "arm": arm, "fold": fold, "seed": 0}))
    for sd in (1,):
        cmd_train(types.SimpleNamespace(**{**base, "arm": "diff-on-L4", "fold": g89.FOLDS[0], "seed": sd}))
    cmd_search(types.SimpleNamespace(**{**base, "arm": "diff-on-L4", "fold": g89.FOLDS[0], "seeds": [0, 1]}))
    r = json.loads((Path(args.outdir) / "search" / f"{g89.FOLDS[0]}.json").read_text(encoding="utf-8"))
    assert all(len(v[m]) == 3 for v in r.values() for m in ("mean", "lcb", "lcbdiv"))
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "train", "select", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--s3c", default=str(S3C))
    ap.add_argument("--s3b", default=str(ROOT / "results" / "acts" / "s3b"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("prep", cmd_prep), ("smoke", cmd_smoke), ("train", cmd_train), ("select", cmd_select), ("search", cmd_search),
                     ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
