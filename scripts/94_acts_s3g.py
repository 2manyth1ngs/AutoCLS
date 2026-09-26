"""WP17-ACTS S3-G（results/acts/s3g/PROTOCOL.md）：L3 源侧对抗扩充一轮——任务盲集成 B 在训练源上跑部署同款搜索 → 提名真跑 → 并入本折标签 → 重训 B′ / F′ → 48 任务考卷。

阶段：
  --aug-search    服务器：--fold <源> → search_aug/<fold>.json（30 个训练源任务 × 前 5 条，含 B 的预测胜率）
  --plan-aug      服务器：→ runs_aug/plan_X.json（seed 0；去重 S2 / S3-C / S3-E / S3-EA / S3-F 已有格）
  --train-b       服务器：--fold --seed → models/blind_s<seed>_<fold>.pt（B′，= 92.cmd_train + 扩充标签）
  --train-f       服务器：--fold --seed → models/F_s<seed>_<fold>.pt（F′，= 93.cmd_train + 扩充标签，起点 B′）
  --search        服务器：--fold <源> → search/<fold>.json（48 任务考卷 B′-lcb / F′-lcb + 预测胜率）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1）
  --analyze       本地
  --smoke         本地冒烟（合成窗口 + 合成真值）
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

OUT = ROOT / "results" / "acts" / "s3g"
S3C = ROOT / "results" / "acts" / "s3c"
S3E = ROOT / "results" / "acts" / "s3e"
S3EA = ROOT / "results" / "acts" / "s3ea"
S3F = ROOT / "results" / "acts" / "s3f"
S2_DIR = ROOT / "results" / "acts" / "s2"
AUG_ROWS = (0, 4)
AUG_TOP = 5
EXAM_ARMS = ("B′-lcb", "F′-lcb")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def aug_tasks(tasks: Sequence[str], groups: Sequence, order: Sequence[str], fold: str, val: str) -> list[str]:
    """折 f 的扩充任务：训练源（去掉 f 与验证源）× OA 行 a00 / a04，按池顺序。"""
    have = set(tasks)
    return [f"{s}_a{r:02d}" for s in order if s not in (fold, val) for r in AUG_ROWS if f"{s}_a{r:02d}" in have]


def merge_aug(V: Sequence, R: Sequence, NS: Sequence, aug: Mapping[int, tuple]) -> tuple[list, list, list]:
    """把 {任务下标: (配置向量, seed-0 终点)} 接在该任务原配置之后（单种子 n = 1）；不改原数组。"""
    V2, R2, NS2 = list(V), list(R), list(NS)
    for i, (vec, r) in aug.items():
        V2[i] = np.vstack([V[i], np.asarray(vec, V[i].dtype)])
        R2[i] = np.concatenate([np.asarray(R[i], float), np.asarray(r, float)])
        NS2[i] = np.concatenate([np.asarray(NS[i], int), np.ones(len(r), int)])
    return V2, R2, NS2


def true_beat(r: float, ref: np.ndarray) -> float:
    return float(np.mean(float(r) > np.asarray(ref, float)))


def pred_beat_from(P: np.ndarray) -> float:
    """P：模型 × shared 条目的 P(提名胜过该条)；预测胜率 = 全体均值。"""
    return float(np.mean(P))


def verdict_s3g(dL_m3: Mapping, dL_p1: Mapping, dE_m3: Mapping, dE_p1: Mapping, cat_ok: bool) -> str:
    if dL_m3["d"] >= 0.010 and dL_m3["p"] < 0.05 and dL_m3["npos"] >= 4 and dL_p1["d"] > 0:
        code = "S3G-L3-WIN"
    elif dL_m3["d"] <= -0.010 and dL_m3["p"] < 0.05:
        code = "S3G-L3-HURT"
    else:
        code = "S3G-L3-NULL"
    if cat_ok:
        code += "+CAT-PAR"
    if dE_m3["d"] >= 0.010 and dE_m3["p"] < 0.05 and dE_m3["npos"] >= 4 and dE_p1["d"] > 0:
        code += "+E"
    return code


# ================================================================ 公用
def _ctx(args):
    ac, g89 = _g("acts_common"), _g("89_acts_s3c")
    from autocls.space.cls_encoder import CLSEncoder
    meta, arr = g89.load_data(Path(args.s3c))
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    return meta, arr, cfg, space, enc


def _pred_beat(ens, Ts, vc: np.ndarray, Vsh: np.ndarray) -> float:
    import torch
    rows = []
    for sc, T in zip(ens.sc, Ts):
        a = torch.as_tensor(np.repeat(np.asarray(vc, np.float32)[None], len(Vsh), 0), device=sc.dev)
        b = torch.as_tensor(np.asarray(Vsh, np.float32), device=sc.dev)
        Tb = None if T is None else T.unsqueeze(0).expand(len(Vsh), -1)
        with torch.no_grad():
            rows.append(sc.forward(a, b, Tb).float().cpu().numpy())
    return pred_beat_from(np.array(rows))


def _ens(kind: str, paths: Sequence[Path], dim: int, dev):
    g92, g93 = _g("92_acts_s3ea"), _g("93_acts_s3f")
    build = (lambda: g92.build_blind(dim)) if kind == "D" else (lambda: g93.build_field_e(dim))
    return g92.ArmEnsemble(g93._load_models(build, paths, dev), kind, dev)


def _val_map(dirs: Sequence[Path]) -> dict:
    """(任务, cls_key, seed) → 终点 acc（S2 + 各目录 runs / runs_aug 的 plan_X）。"""
    ac, g83 = _g("acts_common"), _g("83_acts_corpus")
    rows = dict(g83._g35.load_done(S2_DIR / "runs.jsonl")); plans = [json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8"))]
    for d in dirs:
        for sub in ("runs", "runs_aug"):
            if (d / sub / "runs.jsonl").exists():
                rows.update(g83._g35.load_done(d / sub / "runs.jsonl"))
            if (d / sub / "plan_X.json").exists():
                plans.append(json.loads((d / sub / "plan_X.json").read_text(encoding="utf-8")))
    val = {}
    for pl in plans:
        for c in pl["cells"]:
            v = ac.endpoint_of(rows.get(c["key"]))
            if v is not None:
                val[(c["task_id"], c["cls_key"], int(c["seed"]))] = v
    return val


def _dirs(args) -> list[Path]:
    return [Path(args.s3c), Path(args.s3e), Path(args.s3ea), Path(args.s3f), Path(args.outdir)]


def load_aug(args, fold: str, meta: Mapping, enc) -> dict:
    """折 f 的扩充标签 {任务下标: (向量, seed-0 终点)}（无真值的提名跳过）。"""
    from autocls.space.cls_space import CLS
    p = Path(args.outdir) / "search_aug" / f"{fold}.json"
    noms = json.loads(p.read_text(encoding="utf-8"))
    val = _val_map(_dirs(args)); tasks = meta["tasks"]; out = {}
    for t, lst in noms.items():
        vs, rs = [], []
        for e in lst:
            v = val.get((t, e["cls_key"], 0))
            if v is not None:
                vs.append(enc.encode(CLS(**e["cls"]))); rs.append(v)
        if rs:
            out[tasks.index(t)] = (np.array(vs, np.float32), np.array(rs, float))
    return out


# ================================================================ 扩充（服务器）
def cmd_aug_search(args) -> None:
    import torch  # noqa: F401
    g91, g92 = _g("91_acts_s3e"), _g("92_acts_s3ea")
    meta, arr, cfg, space, enc = _ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search_aug" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[aug-search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS
    ens = _ens("D", [Path(args.s3ea) / "models" / f"blind_s{sd}_{fold}.pt" for sd in seeds], enc.dim, dev)
    ts = aug_tasks(meta["tasks"], meta["groups"], meta["order"], fold, meta["val"][fold])
    ks = args.ks or _g("89_acts_s3c").KS; gens = args.gens if args.gens is not None else _g("89_acts_s3c").GENS
    res, t0 = {}, time.time()
    for t in ts:
        rec = g92._search_task(ens, [None] * len(ens.sc), t, (("aug", g91.LAM, None),), space, enc, cfg, ks, gens, top=AUG_TOP)
        have = set(meta["keys"][t]); Vsh = arr[f"{t}/vec"][:_g("89_acts_s3c").N_SH]
        lst = []
        for e in rec["aug"]:
            if e["cls_key"] in have:
                continue
            from autocls.space.cls_space import CLS
            e["pred_beat"] = _pred_beat(ens, [None] * len(ens.sc), enc.encode(CLS(**e["cls"])), Vsh)
            lst.append(e)
        res[t] = lst
        print(f"[aug-search] {fold} {t} {[x['cls_key'] for x in lst]} pred {[round(x['pred_beat'], 2) for x in lst]} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def _plan(args, noms: Mapping[str, Mapping[str, Sequence[str]]], byk: Mapping, seeds: Sequence[int], sub: str) -> None:
    ac, g35 = _g("acts_common"), _g("35_protocol_alignment")
    from autocls.space.cls_space import CLS
    meta, _arr, cfg, _s, _e = _ctx(args)
    protos = g35.load_protocols()
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8")); tm = tj["tasks"]
    done = set(meta["s2_done"]) | {f"{t}|{k}|{s}" for (t, k, s) in _val_map(_dirs(args))}
    cells, seen = [], set(done)
    for t, arms in noms.items():
        for keys in arms.values():
            for k in keys:
                for sd in seeds:
                    tag = f"{t}|{k}|{sd}"
                    if tag in seen:
                        continue
                    seen.add(tag); cells.append(ac.make_cell(t, tm[t], ac.ARM_RANDOM, CLS(**byk[k]), k, int(sd), protos, cfg))
    rd = Path(args.outdir) / sub; rd.mkdir(parents=True, exist_ok=True)
    st = list(noms)
    plan = {"phase": "X", "profile": cfg.get("profile"), "n_cells": len(cells), "probe_tasks": st, "protocol": cfg["protocol"],
            "tasks_meta": {t: {k: tm[t][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in tm[t]} for t in st},
            "cells": cells}
    (rd / "plan_X.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (rd / "tasks.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    print(f"[plan] {sub}: {len(cells)} new cells（已有格复用）", flush=True)


def cmd_plan_aug(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search_aug").glob("*.json")):
        for t, lst in json.loads(f.read_text(encoding="utf-8")).items():
            noms.setdefault(t, {})[f.stem] = [e["cls_key"] for e in lst]
            byk.update({e["cls_key"]: e["cls"] for e in lst})
    _plan(args, noms, byk, (0,), "runs_aug")


# ================================================================ 重训（服务器）
def _hook(args):
    meta, _arr, _c, _s, enc = _ctx(args)
    aug = load_aug(args, args.fold, meta, enc)
    print(f"[aug] {args.fold}: {len(aug)} 任务、{sum(len(v[1]) for v in aug.values())} 条扩充标签", flush=True)
    return lambda V, R, NS: merge_aug(V, R, NS, aug)


def cmd_train_b(args) -> None:
    _g("92_acts_s3ea").cmd_train(args, extra=_hook(args))


def cmd_train_f(args) -> None:
    a = types.SimpleNamespace(**{**vars(args), "s3ea": args.outdir})     # 残差起点 = 本目录的 B′
    _g("93_acts_s3f").cmd_train(a, extra=_hook(args))


# ================================================================ 考卷搜索（服务器）
def _old_b_noms(args, fold: str, meta: Mapping) -> dict:
    """旧 B-lcb 提名：奇数行取 S3-EA、偶数行取 S3-F（文件缺则空）。"""
    out = {}
    for d in (Path(args.s3ea), Path(args.s3f)):
        p = d / "search" / f"{fold}.json"
        if p.exists():
            for t, rec in json.loads(p.read_text(encoding="utf-8")).items():
                if "B-lcb" in rec:
                    out[t] = rec["B-lcb"]
    return out


def cmd_search(args) -> None:
    import torch
    g89, g91, g92, g93 = _g("89_acts_s3c"), _g("91_acts_s3e"), _g("92_acts_s3ea"), _g("93_acts_s3f")
    from autocls.space.cls_space import CLS
    meta, arr, cfg, space, enc = _ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS
    ens_b2 = _ens("D", [outdir / "models" / f"blind_s{sd}_{fold}.pt" for sd in seeds], enc.dim, dev)
    ens_f2 = _ens("A", [outdir / "models" / f"F_s{sd}_{fold}.pt" for sd in seeds], enc.dim, dev)
    ens_b = _ens("D", [Path(args.s3ea) / "models" / f"blind_s{sd}_{fold}.pt" for sd in seeds], enc.dim, dev)
    tasks = meta["tasks"]; ex = g93.exam_tasks(tasks, meta["groups"], [fold])
    W = g89.load_windows("A", fold, ex, Path(args.s3b), synthetic=args.synthetic)
    old = _old_b_noms(args, fold, meta)
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t, w in zip(ex, W):
        with torch.no_grad():
            Tf = [sc.task_vec(w) for sc in ens_f2.sc]
        nb = [None] * len(seeds); Vsh = arr[f"{t}/vec"][:g89.N_SH]
        rec = g92._search_task(ens_b2, nb, t, lcb("B′-lcb"), space, enc, cfg, ks, gens)
        rec.update(g92._search_task(ens_f2, Tf, t, lcb("F′-lcb"), space, enc, cfg, ks, gens))
        for e in rec["B′-lcb"]:
            e["pred_beat"] = _pred_beat(ens_b2, nb, enc.encode(CLS(**e["cls"])), Vsh)
        for e in rec["F′-lcb"]:
            e["pred_beat"] = _pred_beat(ens_f2, Tf, enc.encode(CLS(**e["cls"])), Vsh)
        rec["B-lcb-old"] = [{"cls_key": e["cls_key"], "pred_B": _pred_beat(ens_b, nb, enc.encode(CLS(**e["cls"])), Vsh),
                             "pred_B2": _pred_beat(ens_b2, nb, enc.encode(CLS(**e["cls"])), Vsh)} for e in old.get(t, [])]
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
    _plan(args, noms, byk, (0, 1), "runs")


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, wilcoxon
    g84, g86, g89, g91, g92, g93 = (_g(m) for m in ("84_acts_s3a", "86_acts_s3b", "89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f"))
    from autocls.search.rank import symmetrize, wins_from_scores
    outdir, s3c, s3e, s3ea, s3f = Path(args.outdir), Path(args.s3c), Path(args.s3e), Path(args.s3ea), Path(args.s3f)
    meta, arr = g89.load_data(s3c); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    val = _val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    # ---- 扩充：预测 vs 真值（seed-0 终点 vs shared seed-0 终点）
    aug_rows = []
    for s in g89.FOLDS:
        for t, lst in js(outdir / "search_aug" / f"{s}.json").items():
            i = tasks.index(t)
            for e in lst:
                v = val.get((t, e["cls_key"], 0))
                if v is not None:
                    aug_rows.append({"fold": s, "task": t, "src": groups[i], "pred": e["pred_beat"], "true": true_beat(v, end0[i]), "acc": v,
                                     "task_best0": float(end0[i].max()), "task_med0": float(np.median(end0[i]))})
    A = {k: np.array([r[k] for r in aug_rows]) for k in ("pred", "true", "acc", "task_best0", "task_med0")}
    aug = {"n": len(aug_rows), "pred": float(A["pred"].mean()), "true": float(A["true"].mean()), "over": float((A["pred"] - A["true"]).mean()),
           "beat_best_frac": float(np.mean(A["acc"] > A["task_best0"])), "above_median_frac": float(np.mean(A["acc"] > A["task_med0"])),
           "per_fold": {s: {"n": int(sum(r["fold"] == s for r in aug_rows)), "over": float(np.mean([r["pred"] - r["true"] for r in aug_rows if r["fold"] == s]))} for s in g89.FOLDS}}
    # ---- 考卷
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    sg_ = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}
    sf = {s: js(s3f / "search" / f"{s}.json") for s in g89.FOLDS}; se = {s: js(s3e / "search" / f"{s}.json") for s in g89.FOLDS}
    sa = {s: js(s3ea / "search" / f"{s}.json") for s in g89.FOLDS}; sc_ = {s: js(s3c / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("B′-lcb", "F′-lcb", "B-lcb", "F-lcb", "E-lcb", "R")
    noms = {}
    for t, s in zip(ex, sg):
        f = sf[s][t]; new = f["new_row"]
        noms[t] = {"B′-lcb": sg_[s][t]["B′-lcb"], "F′-lcb": sg_[s][t]["F′-lcb"], "F-lcb": f["F-lcb"],
                   "B-lcb": f["B-lcb"] if new else sa[s][t]["B-lcb"], "E-lcb": f["E-lcb"] if new else se[s][t]["lcb"], "R": f["R"] if new else sc_[s][t]["random"]}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    cat_p1, cat_m3, orc = [], [], []
    over = {"B′ own": [], "F′ own": [], "B old (by B)": [], "B old (by B′)": []}
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        tv = lambda k: true_beat((val.get((t, k, 0), fl) + val.get((t, k, 1), fl)) / 2, ref)  # noqa: E731
        for e in sg_[s][t]["B′-lcb"]:
            over["B′ own"].append(e["pred_beat"] - tv(e["cls_key"]))
        for e in sg_[s][t]["F′-lcb"]:
            over["F′ own"].append(e["pred_beat"] - tv(e["cls_key"]))
        for e in sg_[s][t]["B-lcb-old"]:
            over["B old (by B)"].append(e["pred_B"] - tv(e["cls_key"])); over["B old (by B′)"].append(e["pred_B2"] - tv(e["cls_key"]))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
        orc.append(float(end1[i, int(np.argmax(end0[i]))]))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"], P1["oracle"] = np.array(cat_p1), np.array(cat_m3), np.array(orc)

    def cmp(x, y, grp=None):
        grp = sg if grp is None else grp
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, grp), "ci": g84.cluster_bootstrap_ci(d, grp)}
    pairs = (("B′-lcb", "B-lcb"), ("F′-lcb", "B′-lcb"), ("F′-lcb", "F-lcb"), ("B′-lcb", "cat3"), ("F′-lcb", "cat3"), ("B-lcb", "cat3"), ("B′-lcb", "R"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    C["B′-lcb−oracle"] = {"p1": cmp(P1["B′-lcb"], P1["oracle"])}
    old_rows = np.array([not sf[s][t]["new_row"] for t, s in zip(ex, sg)])
    C_half = {h: cmp(M3["B′-lcb"][m], M3["B-lcb"][m], sg[m]) for h, m in (("odd", old_rows), ("even", ~old_rows))}
    cat_ok = bool(C["B′-lcb−cat3"]["m3"]["d"] >= 0 and C["B′-lcb−cat3"]["p1"]["d"] >= 0)
    code = verdict_s3g(C["B′-lcb−B-lcb"]["m3"], C["B′-lcb−B-lcb"]["p1"], C["F′-lcb−B′-lcb"]["m3"], C["F′-lcb−B′-lcb"]["p1"], cat_ok)
    ov = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in over.items()}
    # ---- 训练与排序
    zb2 = {s: [np.load(outdir / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    zf2 = {s: [np.load(outdir / "train" / f"F_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    zb = {s: [np.load(s3ea / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    rb2, spb2 = g92._rank_readout(zb2, meta, arr, held); rf2, spf2 = g92._rank_readout(zf2, meta, arr, held); rb, spb = g92._rank_readout(zb, meta, arr, held)

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    rx = np.zeros(len(held))
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; xs = []
        for k in range(16):
            Wx = np.mean([wins_of(z[f"X/{t}"][k]) for z in zf2[groups[i]]], axis=0)
            ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
            xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
        rx[jj] = float(np.mean(xs))
    ba_b2 = [int(np.argmin([h["val_loss"] for h in json.loads(str(z["hist"]))])) for s in g89.FOLDS for z in zb2[s]]
    ba_f2 = [tuple(int(x) for x in z["best_at"]) for s in g89.FOLDS for z in zf2[s]]
    rank = {"B′": {"regret3": float(rb2.mean()), "sp": float(np.nanmedian(spb2))}, "F′": {"regret3": float(rf2.mean()), "sp": float(np.nanmedian(spf2))},
            "B": {"regret3": float(rb.mean()), "sp": float(np.nanmedian(spb))},
            "B′−B": {"d": float((rb2 - rb).mean()), "p_source": g84.source_wilcoxon(rb2 - rb, hg)},
            "F′ cross": {"d": float((rx - rf2).mean()), "p_source": g84.source_wilcoxon(rx - rf2, hg), "npos": g89.n_sources_positive(rx - rf2, hg)},
            "B′ best_epoch_median": float(np.median(ba_b2)), "F′ n_gain": int(sum(b[0] >= 0 for b in ba_f2)),
            "F′ val_gain_mean": float(np.mean([float(z["start_val"]) - float(z["best_val"]) for s in g89.FOLDS for z in zf2[s]]))}
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "n_tasks": len(ex), "aug": aug, "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "cmp_half_B′−B": C_half, "missing": miss, "over_exam": ov, "rank": rank, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-G 读数（{code}）", "", "## 扩充（训练源上的 B 搜索提名）", "",
         f"- {aug['n']} 条提名有真值；B 预测胜率 {f(aug['pred'])}、真值胜率 {f(aug['true'])}、高估 {aug['over']:+.3f}",
         f"- 提名 seed-0 终点高于该任务 shared-30 最好者的比例 {f(aug['beat_best_frac'])}、高于中位数的比例 {f(aug['above_median_frac'])}",
         "- 逐折高估：" + "，".join(f"{s} {v['over']:+.3f}（n {v['n']}）" for s, v in aug["per_fold"].items()), "",
         f"## 考卷（{len(ex)} 任务）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3", "oracle"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'].get(a))} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"]
    for k, v in C.items():
        L.append(f"| {k} | {g(v['m3']) if 'm3' in v else '—'} | {g(v['p1'])} |")
    L += ["", "奇 / 偶行 B′ − B（3 条平均）：" + "；".join(f"{h} {v['d']:+.3f}（p {v['p']:.2f}）" for h, v in C_half.items()), "",
          "考卷提名的高估（预测胜率 − 真值胜率，真值 = 两种子均值对 shared-30 两种子均值）：" + "；".join(f"{k} {v['mean']:+.3f}（n {v['n']}）" for k, v in ov.items()), "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    L += ["", "## 训练与排序（56 留出任务）", "",
          f"- regret@3：B′ {f(rank['B′']['regret3'])} / F′ {f(rank['F′']['regret3'])} / B {f(rank['B']['regret3'])}；B′ − B {rank['B′−B']['d']:+.4f}（源级 p {rank['B′−B']['p_source']:.3f}）",
          f"- random-20 中位 Sp：B′ {f(rank['B′']['sp'])} / F′ {f(rank['F′']['sp'])} / B {f(rank['B']['sp'])}",
          f"- F′ 换源 Δ {rank['F′ cross']['d']:+.4f}（源级 p {rank['F′ cross']['p_source']:.3f}，同号 {rank['F′ cross']['npos']}/6）；F′ 验证损失低于起点 {rank['F′ n_gain']}/30（均值下降 {rank['F′ val_gain_mean']:.4f}）；B′ 最好 epoch 中位 {rank['B′ best_epoch_median']}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89, g92 = _g("89_acts_s3c"), _g("92_acts_s3ea")
    od = Path(args.outdir); sa, se = od / "_s3ea", od / "_s3e"
    for d in (sa, se):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(S3E / "axis_targets.npz", se / "axis_targets.npz")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "s3ea": str(sa), "s3e": str(se), "seeds": [0, 1]})
    fold = g89.FOLDS[0]
    for sd in (0, 1):
        g92.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sa), "fold": fold, "seed": sd}))
    ns = types.SimpleNamespace(**{**base, "fold": fold})
    cmd_aug_search(ns); cmd_plan_aug(ns)
    plan = json.loads((od / "runs_aug" / "plan_X.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    with open(od / "runs_aug" / "runs.jsonl", "w", encoding="utf-8") as fh:            # 合成真值（只为走通并入路径）
        for c in plan["cells"]:
            fh.write(json.dumps({"key": c["key"], "checkpoint_results": [{"acc": float(rng.uniform(0.2, 0.9))}]}) + "\n")
    for sd in (0, 1):
        cmd_train_b(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
        cmd_train_f(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 and all("pred_beat" in e for e in v[a]) for v in r.values() for a in EXAM_ARMS)
    print(f"[smoke] OK（扩充格 {plan['n_cells']}）", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("aug-search", "plan-aug", "train-b", "train-f", "search", "plan-runs", "analyze", "smoke"):
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
    ap.add_argument("--s3c", default=str(S3C))
    ap.add_argument("--s3e", default=str(S3E))
    ap.add_argument("--s3ea", default=str(S3EA))
    ap.add_argument("--s3f", default=str(S3F))
    ap.add_argument("--s3b", default=str(ROOT / "results" / "acts" / "s3b"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("smoke", cmd_smoke), ("aug_search", cmd_aug_search), ("plan_aug", cmd_plan_aug), ("train_b", cmd_train_b), ("train_f", cmd_train_f),
                     ("search", cmd_search), ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
