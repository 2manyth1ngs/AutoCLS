"""WP17-ACTS S3-EA（results/acts/s3ea/PROTOCOL.md）：S3-E 搜索增益的归因对照——任务盲 field 集成（B 臂）+ ★ 模型换源 E（X 臂）。

阶段：
  --train         服务器：--fold <源> --seed s → models/blind_s<seed>_<fold>.pt、train/blind_s<seed>_<fold>.npz（B 臂，L4 标签，同 S3-E 训练制度）
  --search        服务器：--fold <源> → search/<fold>.json（B-mean / B-lcb / B-lcb-div + X-lcb）
  --plan-runs     服务器：提名去重（S2 / S3-C / S3-E 已有格复用）→ runs/plan_X.json（83 phase X）
  --analyze       本地
  --smoke         本地冒烟（合成窗口）
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3ea"
S3C = ROOT / "results" / "acts" / "s3c"
S3E = ROOT / "results" / "acts" / "s3e"
S2_DIR = ROOT / "results" / "acts" / "s2"
SEARCH_ARMS = ("B-mean", "B-lcb", "B-lcb-div", "X-lcb")
CROSS_OFFSET = 1          # 与 86.cross_assign / 88.cross_maps 偏移 1 相同


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def verdict_s3ea(dE: Mapping, dX: Mapping, blind_vs_cat3: float) -> str:
    """主体按 Δ_E = E-lcb − B-lcb；后缀 +CONTENT（Δ_X = E-lcb − X-lcb）/ +BLIND-PAR（B-lcb − 目录前 3 ≥ 0）。"""
    if dE["d"] >= 0.010 and dE["p"] < 0.05 and dE["npos"] >= 4:
        code = "EA-E-CONTRIB"
    elif dE["d"] <= -0.010 and dE["p"] < 0.05:
        code = "EA-E-HURT"
    else:
        code = "EA-E-NULL"
    if dX["d"] >= 0.010 and dX["p"] < 0.05 and dX["npos"] >= 4:
        code += "+CONTENT"
    if blind_vs_cat3 >= 0:
        code += "+BLIND-PAR"
    return code


def cross_index(tasks: Sequence[str], groups: Sequence, order: Sequence[str]) -> list[int]:
    """每任务的换源 E 来源：池顺序下一个源的同 OA 行号任务（88.cross_maps 偏移 1）。"""
    return _g("88_acts_s3bpp").cross_maps(tasks, groups, order, offsets=(CROSS_OFFSET,))[0]


def build_blind(cls_dim: int):
    """B 臂：S3-E ★（field 头）去掉任务支路 → logit = s₀(l_a) − s₀(l_b)（无 SetTF、无 field_w、无辅助损失）。"""
    from autocls.comparator.tclsc_model import TCLSC
    return TCLSC(cls_dim, None, task_branch="none", head="field")


def new_tags(noms: Mapping[str, Mapping[str, Sequence[str]]], done: set) -> list[str]:
    """提名 {任务: {臂: [cls_key]}} → 需新跑的 "task|key|seed"（seed 0 / 1；已跑与重复去掉，保持出现顺序）。"""
    out, seen = [], set(done)
    for t, arms in noms.items():
        for keys in arms.values():
            for k in keys:
                for sd in (0, 1):
                    tag = f"{t}|{k}|{sd}"
                    if tag not in seen:
                        seen.add(tag); out.append(tag)
    return out


# ================================================================ 训练（服务器）
def cmd_train(args) -> None:
    import torch
    g89, g91 = _g("89_acts_s3c"), _g("91_acts_s3e")
    outdir = Path(args.outdir)
    meta, arr = g89.load_data(Path(args.s3c))
    tasks = meta["tasks"]; groups = np.array(meta["groups"])
    fold, seed, dev = args.fold, int(args.seed), args.device
    tag = f"blind_s{seed}_{fold}"
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
    torch.manual_seed(seed)
    model = build_blind(V[0].shape[1]).to(dev)
    sc = g89.Scorer(model, "D", dev)
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
        p = sc.forward(va_, vb_, None).clamp(1e-6, 1 - 1e-6)
        return -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(args.max_epochs or g89.MAX_EPOCHS):
        model.train()
        pairs = [p for i in tr for p in weighted(i, g89.linear_pairs(R[i], rng))]
        perm = rng.permutation(len(pairs))
        tot, nb = 0.0, 0
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
        print(f"[train] {tag} ep {ep} train {tot / max(nb, 1):.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-9:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= g89.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    out = {"hist": json.dumps(hist), "te": np.array(te), "va": np.array(va)}
    for i in te + va:
        out[f"P/{tasks[i]}"] = sc.pmat(V[i], None).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best val {best:.4f} epochs {len(hist)} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 集成搜索（服务器）
class ArmEnsemble:
    """91.Ensemble 的推广：Scorer 的臂可为 "A"（E_pt + SetTF）或 "D"（任务盲）。"""

    def __init__(self, models, arm: str, dev):
        self.sc = [_g("89_acts_s3c").Scorer(m, arm, dev) for m in models]

    wins = _g("91_acts_s3e").Ensemble.wins


def _search_task(ens, Ts, t: str, modes: Sequence[tuple], space, enc, cfg, ks: int, gens: int) -> dict:
    """与 91.cmd_search 同一流程（同采样种子 / 同进化种子）；modes = [(名, λ, 是否另出家族分散版)]。"""
    ac, g89, g91 = _g("acts_common"), _g("89_acts_s3c"), _g("91_acts_s3e")
    from autocls.labeling import cls_key, run_seed_for
    cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(t, "s3c_ks")), ks, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
    rec = {}
    for name, lam, div in modes:
        fin = []
        for s in range(0, len(cands), g89.CHUNK):
            idx = list(range(s, min(len(cands), s + g89.CHUNK)))
            o = g91.ens_order(ens.wins(Vc[idx], Ts), [keys[i] for i in idx], lam)
            fin += [idx[j] for j in o[:g89.PER_CHUNK]]
        o = g91.ens_order(ens.wins(Vc[fin], Ts), [keys[i] for i in fin], lam)
        init = [cands[fin[j]][0] for j in o[:g89.KP]]
        final = g91._evolve(ens, Ts, space, enc, cfg, init, np.random.default_rng(run_seed_for(t, "s3c_evo")), lam, g89.KP, gens, g89.P_CROSS, g89.P_MUT)
        rec[name] = [{"cls_key": cls_key(c), "cls": c.to_dict()} for c in final[:3]]
        if div:
            fk = [cls_key(c) for c in final]; fam = {cls_key(c): g91.family_of(c.to_dict()) for c in final}; byk = {cls_key(c): c for c in final}
            rec[div] = [{"cls_key": k, "cls": byk[k].to_dict()} for k in g91.diversified_top(fk, fam, 3)]
    return rec


def cmd_search(args) -> None:
    import torch
    ac, g89, g91 = _g("acts_common"), _g("89_acts_s3c"), _g("91_acts_s3e")
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.task_embed import TaskSetEncoder
    outdir, s3e = Path(args.outdir), Path(args.s3e)
    meta, _arr = g89.load_data(Path(args.s3c)); tasks = meta["tasks"]; groups = meta["groups"]; fold, dev = args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    star = args.star or json.loads((s3e / "select.json").read_text(encoding="utf-8"))["star"]
    head, _aux, _lab = g91.parse_arm(star)
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    seeds = args.seeds or g91.ENS_SEEDS
    blind = []
    for sd in seeds:
        m = build_blind(enc.dim).to(dev); m.load_state_dict(torch.load(outdir / "models" / f"blind_s{sd}_{fold}.pt", map_location=dev)); m.eval(); blind.append(m)
    starm = []
    for sd in seeds:
        m = TCLSC(enc.dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head=head).to(dev)
        m.load_state_dict(torch.load(s3e / "models" / f"{star}_s{sd}_{fold}.pt", map_location=dev)); m.eval(); starm.append(m)
    ens_b, ens_x = ArmEnsemble(blind, "D", dev), ArmEnsemble(starm, "A", dev)
    st = list(meta["search"][fold]); cx = cross_index(tasks, groups, meta["order"])
    xsrc = [tasks[cx[tasks.index(t)]] for t in st]
    Wx = g89.load_windows("A", fold, xsrc, Path(args.s3b), synthetic=args.synthetic)
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    res, t0 = {}, time.time()
    for t, xs, w in zip(st, xsrc, Wx):
        rec = _search_task(ens_b, [None] * len(blind), t, (("B-mean", 0.0, None), ("B-lcb", g91.LAM, "B-lcb-div")), space, enc, cfg, ks, gens)
        with torch.no_grad():
            Ts = [sc.task_vec(w) for sc in ens_x.sc]
        rec.update(_search_task(ens_x, Ts, t, (("X-lcb", g91.LAM, None),), space, enc, cfg, ks, gens))
        rec["x_task"] = xs; rec["star"] = star
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in SEARCH_ARMS) + f" xE←{xs} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def _done_tags(meta: Mapping, dirs: Sequence[Path]) -> set:
    done = set(meta["s2_done"])
    for d in dirs:
        p = d / "runs" / "runs.jsonl"
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(l)
                if not r.get("error"):
                    done.add(f"{r['task_id']}|{r['cls_key']}|{int(r['seed'])}")
    return done


def cmd_plan_runs(args) -> None:
    ac, g35, g89 = _g("acts_common"), _g("35_protocol_alignment"), _g("89_acts_s3c")
    from autocls.space.cls_space import CLS
    outdir, s3c, s3e = Path(args.outdir), Path(args.s3c), Path(args.s3e)
    meta, _arr = g89.load_data(s3c)
    cfg = ac.load_acts_config(profile="s2"); protos = g35.load_protocols()
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8")); tm = tj["tasks"]
    done = _done_tags(meta, [s3c, s3e, outdir])
    noms, byk = {}, {}
    for f in sorted((outdir / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in SEARCH_ARMS}
            for a in SEARCH_ARMS:
                for e in rec[a]:
                    byk[e["cls_key"]] = e["cls"]
    cells = []
    for tag in new_tags(noms, done):
        t, k, sd = tag.split("|")
        cells.append(ac.make_cell(t, tm[t], ac.ARM_RANDOM, CLS(**byk[k]), k, int(sd), protos, cfg))
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    st = [t for v in meta["search"].values() for t in v]
    plan = {"phase": "X", "profile": cfg.get("profile"), "n_cells": len(cells), "probe_tasks": st, "protocol": cfg["protocol"],
            "tasks_meta": {t: {k: tm[t][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in tm[t]} for t in st},
            "cells": cells}
    (rd / "plan_X.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (rd / "tasks.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    print(f"[plan-runs] {len(cells)} new cells（复用 S2 / S3-C / S3-E 已有格）", flush=True)


# ================================================================ analyze（本地）
def _rank_readout(zs: Mapping[str, list], meta: Mapping, arr: Mapping, held: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """λ = 0 胜场均值集成：shared-30 regret@3（seed 0 选、seed 1 评）与 random-20 Spearman（同 91.readout）。"""
    from scipy.stats import spearmanr
    from autocls.search.rank import symmetrize, wins_from_scores
    g89 = _g("89_acts_s3c"); N = g89.N_SH; tasks = meta["tasks"]; groups = meta["groups"]
    end0, end1 = arr["end0"], arr["end1"]

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    r3, sp = np.full(len(held), np.nan), np.full(len(held), np.nan)
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; zl = zs[groups[i]]
        Wf = np.mean([wins_of(z[f"P/{t}"]) for z in zl], axis=0)
        Ws = np.mean([wins_of(z[f"P/{t}"][:N, :N]) for z in zl], axis=0)
        o = sorted(range(N), key=lambda j: (-Ws[j], keys[j]))[:3]
        r3[jj] = end1[i, int(np.argmax(end0[i]))] - end1[i, o[int(np.argmax(end0[i, o]))]]
        R0 = arr[f"{t}/R0"]
        if len(R0) >= 10:
            sp[jj] = float(spearmanr(Wf[N:N + len(R0)], R0)[0])
    return r3, sp


def cmd_analyze(args) -> None:
    from scipy.stats import wilcoxon
    g83, g84, g89, g91 = _g("83_acts_corpus"), _g("84_acts_s3a"), _g("89_acts_s3c"), _g("91_acts_s3e")
    outdir, s3c, s3e = Path(args.outdir), Path(args.s3c), Path(args.s3e)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"])
    end0 = arr["end0"]
    star = json.loads((s3e / "select.json").read_text(encoding="utf-8"))["star"]
    s3c_rank = json.loads((s3c / "per_task.json").read_text(encoding="utf-8"))
    # ---- 搜索真值
    rows = dict(g83._g35.load_done(S2_DIR / "runs.jsonl"))
    plans = [json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8"))]
    for d in (s3c, s3e, outdir):
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
    srch_old = s3c_rank["search"]; stasks = list(srch_old)
    sg = np.array([srch_old[t]["source"] for t in stasks])
    noms = {}
    for t in stasks:
        s = srch_old[t]["source"]
        e = json.loads((s3e / "search" / f"{s}.json").read_text(encoding="utf-8"))[t]
        b = json.loads((outdir / "search" / f"{s}.json").read_text(encoding="utf-8"))[t]
        noms[t] = {"E-mean": e["mean"], "E-lcb": e["lcb"], **{a: b[a] for a in SEARCH_ARMS}}
    arms = ("E-mean", "E-lcb") + SEARCH_ARMS
    final, miss = {a: [] for a in arms}, {a: 0 for a in arms}
    for t in stasks:
        i = tasks.index(t); fl = float(np.nanmin(end0[i]))
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            v0 = {k: val.get((t, k, 0), np.nan) for k in ks_}; v1 = {k: val.get((t, k, 1), np.nan) for k in ks_}
            if any(not np.isfinite(v) for v in list(v0.values()) + list(v1.values())):
                miss[a] += 1
                v0 = {k: (v if np.isfinite(v) else fl) for k, v in v0.items()}; v1 = {k: (v if np.isfinite(v) else fl) for k, v in v1.items()}
            final[a].append(g89.pick_final(ks_, v0, v1)[1])
    ref = {k: np.array([srch_old[t][k] for t in stasks]) for k in ("A", "D", "R", "cat3", "oracle")}

    def cmp(a, b):
        d = np.array(a) - np.array(b)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    search = {a: {"final": float(np.mean(v)), "vs_R": cmp(v, ref["R"]), "vs_cat3": cmp(v, ref["cat3"]), "vs_s3cA": cmp(v, ref["A"]), "vs_s3cD": cmp(v, ref["D"]),
                  "vs_oracle": cmp(v, ref["oracle"]), "per_source": {s: float(np.mean([v[j] for j in range(len(stasks)) if sg[j] == s])) for s in g89.FOLDS}}
              for a, v in final.items()}
    dE, dX = cmp(final["E-lcb"], final["B-lcb"]), cmp(final["E-lcb"], final["X-lcb"])
    blind_lcb_vs_mean = cmp(final["B-lcb"], final["B-mean"])
    code = verdict_s3ea(dE, dX, search["B-lcb"]["vs_cat3"]["d"])
    overlap = {f"{a}∩{b}": int(sum(len({x["cls_key"] for x in noms[t][a]} & {x["cls_key"] for x in noms[t][b]}) for t in stasks))
               for a, b in (("E-lcb", "B-lcb"), ("E-lcb", "X-lcb"), ("B-lcb", "B-mean"))}
    # ---- 排序（56 留出任务）
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    zb = {s: [np.load(outdir / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    ze = {s: [np.load(s3e / "train" / f"{star}_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    rb, spb = _rank_readout(zb, meta, arr, held); re_, spe = _rank_readout(ze, meta, arr, held)
    rank = {"B-ens": {"regret3": float(np.mean(rb)), "sp_median": float(np.nanmedian(spb))},
            "★-ens": {"regret3": float(np.mean(re_)), "sp_median": float(np.nanmedian(spe))},
            "★−B regret3": {"d": float(np.mean(re_ - rb)), "p_source": g84.source_wilcoxon(re_ - rb, hg), "ci": g84.cluster_bootstrap_ci(re_ - rb, hg)}}
    per_seed = {}
    for sd in g91.ENS_SEEDS:
        r1, _ = _rank_readout({s: [zb[s][sd]] for s in g89.FOLDS}, meta, arr, held); per_seed[sd] = float(np.mean(r1))
    rank["B per-seed regret3"] = per_seed
    be = [int(np.argmin([h["val_loss"] for h in json.loads(str(z["hist"]))])) for s in g89.FOLDS for z in zb[s]]
    ne = [len(json.loads(str(z["hist"]))) for s in g89.FOLDS for z in zb[s]]
    rank["B best_epoch"] = {"n_epoch0": int(sum(b == 0 for b in be)), "median": float(np.median(be)), "all": be, "epochs_run_median": float(np.median(ne))}
    out = {"verdict": code, "star": star, "dE": dE, "dX": dX, "B-lcb_vs_B-mean": blind_lcb_vs_mean, "search": search, "overlap": overlap, "missing": miss,
           "refs": {k: float(v.mean()) for k, v in ref.items()}, "refs_per_source": {k: {s: float(np.mean([ref[k][j] for j in range(len(stasks)) if sg[j] == s])) for s in g89.FOLDS} for k in ref},
           "rank": rank}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-EA 读数（{code}）；★ = {star}", "", "## 搜索（24 任务；seed 0 选、seed 1 评）", "",
         f"参照：S3-C A {f(out['refs']['A'])} / D {f(out['refs']['D'])} / 随机 R {f(out['refs']['R'])} / 目录任务盲前 3 {f(out['refs']['cat3'])} / 目录 oracle {f(out['refs']['oracle'])}", "",
         "| 臂 | 最终 acc | vs R | vs 目录前 3 | vs S3-C A | vs S3-C D | vs oracle | 缺格任务 |", "|---|---:|---|---|---|---|---|---:|"]
    for a in arms:
        v = search[a]
        L.append(f"| {a} | {f(v['final'])} | {g(v['vs_R'])} | {g(v['vs_cat3'])} | {g(v['vs_s3cA'])} | {g(v['vs_s3cD'])} | {g(v['vs_oracle'])} | {miss[a]} |")
    L += ["", f"主对比 Δ_E = E-lcb − B-lcb：{g(dE)}，CI [{f(dE['ci'][0])}, {f(dE['ci'][1])}]",
          f"内容 Δ_X = E-lcb − X-lcb：{g(dX)}，CI [{f(dX['ci'][0])}, {f(dX['ci'][1])}]",
          f"B-lcb − B-mean：{g(blind_lcb_vs_mean)}", f"提名重合（24 任务 × 3 条）：{overlap}", "",
          "## 搜索逐源（最终 acc）", "", "| 源 | " + " | ".join(arms) + " | S3-C A | S3-C D | 随机 R | 目录前 3 | 目录 oracle |", "|---|" + "---:|" * (len(arms) + 5)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(search[a]["per_source"][s]) for a in arms) + " | " + " | ".join(f(out["refs_per_source"][k][s]) for k in ("A", "D", "R", "cat3", "oracle")) + " |")
    L += ["", "## 排序（56 留出任务，λ = 0 胜场均值集成）", "",
          f"- B 集成 regret@3 {f(rank['B-ens']['regret3'])}、random-20 中位 Sp {f(rank['B-ens']['sp_median'])}；★ 集成 {f(rank['★-ens']['regret3'])} / {f(rank['★-ens']['sp_median'])}",
          f"- ★ − B regret@3 {rank['★−B regret3']['d']:+.4f}（源级 p {rank['★−B regret3']['p_source']:.3f}）；B 逐种子 {json.dumps({k: round(v, 4) for k, v in per_seed.items()})}",
          f"- B 最好 epoch：epoch 0 {rank['B best_epoch']['n_epoch0']}/30，中位 {rank['B best_epoch']['median']}，训练 epoch 数中位 {rank['B best_epoch']['epochs_run_median']}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    import types
    g89, g91 = _g("89_acts_s3c"), _g("91_acts_s3e")
    od = Path(args.outdir); se = od / "_s3e"; se.mkdir(parents=True, exist_ok=True)
    shutil.copy(S3E / "axis_targets.npz", se / "axis_targets.npz")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 2, "resume": False, "ks": 1000, "gens": 2, "s3e": str(se), "star": "field-on-L4", "seeds": [0, 1]})
    fold = g89.FOLDS[0]
    for sd in (0, 1):
        g91.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(se), "arm": "field-on-L4", "fold": fold, "seed": sd}))
        cmd_train(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    cmd_search(types.SimpleNamespace(**{**base, "fold": fold}))
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert all(len(v[a]) == 3 for v in r.values() for a in SEARCH_ARMS) and all(v["x_task"].split("_a")[0] != t.split("_a")[0] for t, v in r.items())
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
