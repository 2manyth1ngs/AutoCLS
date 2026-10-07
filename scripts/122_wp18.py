"""WP18（results/acts/wp18/PROTOCOL.md）：扩展空间（AutoCL 19 字段 ⊕ 新轴）上的 T-CLSC 全空间搜索 + 任务感知（E 选源专家）部署。

阶段（服务器 GPU 的标 [GPU]）：
  --plan-src             源侧新轴标签计划（configs/wp18.yaml 的 src_labels 段）→ wp18/src_plan.json
  --run-src  [GPU]       跑源标签格（--part / --parts），追加 wp18/src_runs/part{i}.jsonl
  --train    [GPU]       10 个任务盲 field 头成员（全部 21 源）→ wp18/models/m{q}.pt + train.json
  --fit                  源专家 + 内层留一源选 (α_w, τ, γ)（臂 ME / UNI；BLIND = γ 0）→ wp18/experts.npz / fit.json
  --search   [GPU]       每靶 × 臂：Ks 2 万扩展候选 → 11 个搜索者进化 → 并集 U → wp18/search/{臂}_{靶}.json
  --plan-cv / --run-tgt [GPU] / --select / --plan-final / --run-tgt [GPU] / --ensemble / --analyze
臂：ME（完整方法）/ UNI（E 换成均匀权重）/ BLIND（无源专家 = 任务盲）/ LEG（ME 但新轴固定默认 = AutoCL 旧空间）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "wp18"
ACTS = ROOT / "results" / "acts"
UT_TARGETS = ("AtrialFibrillation", "BasicMotions", "DuckDuckGeese", "Epilepsy", "FaceDetection", "FingerMovements",
              "HandMovementDirection", "Heartbeat", "MotorImagery", "SelfRegulationSCP2", "StandWalkJump")
TS2VEC = {"AtrialFibrillation": 0.200, "BasicMotions": 0.975, "DuckDuckGeese": 0.680, "Epilepsy": 0.964, "FaceDetection": 0.501,
          "FingerMovements": 0.480, "HandMovementDirection": 0.338, "Heartbeat": 0.683, "MotorImagery": 0.510,
          "SelfRegulationSCP2": 0.578, "StandWalkJump": 0.467}
INFOTS_S = {"AtrialFibrillation": 0.267, "BasicMotions": 1.000, "DuckDuckGeese": 0.600, "Epilepsy": 0.993, "FaceDetection": 0.525,
            "FingerMovements": 0.620, "HandMovementDirection": 0.514, "Heartbeat": 0.771, "MotorImagery": 0.610,
            "SelfRegulationSCP2": 0.572, "StandWalkJump": 0.467}
DEV_TARGETS = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1")
TS2VEC.update({"Handwriting": 0.515, "PhonemeSpectra": 0.233, "UWaveGestureLibrary": 0.906, "Cricket": 0.972, "EthanolConcentration": 0.308,
               "SelfRegulationSCP1": 0.812})
INFOTS_S.update({"Handwriting": 0.554, "PhonemeSpectra": 0.233, "UWaveGestureLibrary": 0.884, "Cricket": 1.000, "EthanolConcentration": 0.323,
                 "SelfRegulationSCP1": 0.887})


def scope_targets(args) -> tuple:
    if getattr(args, "targets", None):
        return tuple(args.targets)
    return DEV_TARGETS if getattr(args, "scope", "ut") == "dev" else UT_TARGETS


def fold_of(d: str) -> str:
    """dev 靶本身是源 → 留一源（比较器 / 专家都不见该源）；未见靶 → 全部源（'all'）。"""
    return d if d in DEV_TARGETS else "all"


ARMS = {"ME": {"uniform": False, "legacy": False}, "UNI": {"uniform": True, "legacy": False},
        "BLIND": {"uniform": True, "legacy": False, "gamma0": True}, "LEG": {"uniform": False, "legacy": True},
        "FV": {"uniform": True, "legacy": False, "gamma0": True, "fv": True},       # 只有 轴 × E_V 交互项
        "MEFV": {"uniform": False, "legacy": False, "fv": True}}                    # 源专家（E 选源）+ 轴 × E_V
MEMBERS = tuple(range(10))
KS, KP, GENS, P_CROSS, P_MUT, TOP_PER = 20000, 10, 10, 0.8, 0.2, 2
N_MEMBER_SEARCHERS = 5          # 搜索者 = 集成 + 前 5 个成员（S3-Z 11 个搜索者 → 候选上限升但 CV 噪声抵消；成本减半）
K_TCV, TOPK_ENS, FINAL_SEEDS = 3, 5, (0, 1, 2)      # 终点跑 CV 前 5；集成大小 k ∈ ENS_KS 在 dev 上定
ENS_KS = (1, 3, 5)
MIN_FOLD_EVAL = 10


def _g(name):
    return importlib.import_module(f"scripts.{name}")


def _w(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _r(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wp18_cfg() -> dict:
    import yaml
    return yaml.safe_load(open(ROOT / "configs" / "wp18.yaml", encoding="utf-8"))


def xspace(legacy_only: bool = False):
    ac = _g("acts_common")
    from autocls.search.xspace import XSpace, load_axes
    cfg = ac.load_acts_config(profile="s2")
    axes = load_axes()
    fixed = {n: a["default"] for n, a in axes.items()} if legacy_only else None
    return XSpace(ac.acts_space(cfg), cfg, axes=axes, is_pruned=ac.is_pruned, fixed=fixed)


def survey() -> dict:
    import csv
    return {r["name"]: r for r in csv.DictReader(open(ROOT / "results" / "uea_survey" / "uea_survey.csv", encoding="utf-8"))}


# ================================================================ 源侧新轴标签
def build_src_plan(xs, D_old: Mapping, wc: Mapping) -> list[dict]:
    """每任务：前 n_shared 条 shared（按 S2 全池 z 均值排的先验前列，跨任务同一批）× 每个非默认视图 + n_rand 条该任务已标 random × 随机非默认视图；
    每条轨迹 mult 全档截点 × 全部读出；另对同 n_shared 条跑一次 raw（拿 raw 的 mult / readout 变体）。"""
    from autocls.labeling import run_seed_for
    from autocls.space.cls_space import CLS
    sc = wc["src_labels"]
    views = [v for v in xs.axes["view"]["choices"]] if "view" in xs.axes else ["raw"]
    mults = list(xs.axes["mult"]["choices"]) if "mult" in xs.axes else [1]
    reads = list(xs.axes["readout"]["choices"]) if "readout" in xs.axes else ["max"]
    cat = _r(ACTS / "s2" / "catalog.json")
    shared = [s["cls"] for s in cat["shared"]][: int(sc["n_shared"])] if sc.get("shared_order") == "catalog" else None
    if shared is None:                                    # 先验前列：S2 shared 在全部训练任务上的 z 均值
        keys = [s["cls_key"] for s in cat["shared"]]
        zs = []
        for i, t in enumerate(D_old["tasks"]):
            kk = D_old["keys"][i]; r = D_old["r"][i]
            z = (r - r.mean()) / (r.std() + 1e-9)
            zs.append([z[kk.index(k)] if k in kk else np.nan for k in keys])
        mu = np.nanmean(np.array(zs), 0)
        order = np.argsort(-mu)
        shared = [cat["shared"][j]["cls"] for j in order[: int(sc["n_shared"])]]
    cells = []
    meta = D_old["meta"]
    for i, t in enumerate(D_old["tasks"]):
        m = meta[t]
        base = {"task_id": t, "dataset": m["dataset"], "iters": int(m["iters"]), "mults": mults, "readouts": reads, "seed": 0,
                "n": int(m["n_train"]), "T": int(m["T"]), "C": int(m["C"]), "input_mask": "none"}
        for c in shared:
            for v in views:
                if v == "raw" and len(mults) == 1 and len(reads) == 1:
                    continue                              # raw × 默认 = 旧标签已有
                cells.append({**base, "cls": c, "view": v, "ext_base": {"view": v}, "origin": "shared"})
        rng = np.random.default_rng(run_seed_for(t, "wp18_src_rand"))
        rand = [x["cls"] for x in cat["random"].get(t, [])] if t in cat["random"] else []
        if not rand:
            xcat = _r(ACTS / "xs" / "catalog.json")
            rand = [x["cls"] for x in xcat["random"].get(t, [])]
        for j in rng.permutation(len(rand))[: int(sc["n_rand"])]:
            nondef = [v for v in views if v != "raw"] or ["raw"]
            v = nondef[int(rng.integers(len(nondef)))]
            cells.append({**base, "cls": rand[int(j)], "view": v, "ext_base": {"view": v}, "origin": "random"})
    for c in cells:
        from autocls.labeling import cls_key
        c["key"] = f"{c['task_id']}|{cls_key(CLS(**c['cls']))}|{c['view']}|s{c['seed']}"
    return cells


def src_cost(c: Mapping) -> float:
    tl = c["T"] * (0.35 if c["view"] in ("psd", "spec") else 1.0)
    return max(c["mults"]) * c["iters"] * 8 * tl * 1e-5 + c["n"] ** 1.3 * 2e-3 * len(c["mults"]) * len(c["readouts"]) + 5


def cmd_plan_src(args) -> None:
    w = _g("wp18_data")
    xs = xspace(); wc = wp18_cfg()
    D = w.assemble(xs, include_new=False)
    cells = build_src_plan(xs, D, wc)
    _w(OUT / "src_plan.json", {"cells": cells})
    print(f"[plan-src] {len(cells)} 格（{len(D['tasks'])} 任务）；估计成本 {sum(src_cost(c) for c in cells) / 3600:.1f} 单位·h", flush=True)


def cmd_run_src(args) -> None:
    r = _g("wp18_run"); g83 = _g("83_acts_corpus"); ac = _g("acts_common")
    cells = _r(OUT / "src_plan.json")["cells"]
    mine = r.lpt_parts(cells, args.parts, src_cost)[args.part]
    done = r.done_keys(OUT / "src_runs")
    todo = sorted([c for c in mine if c["key"] not in done], key=lambda c: c["task_id"])
    print(f"[run-src] part {args.part}/{args.parts}: {len(todo)}/{len(mine)} 待跑", flush=True)
    tasks_meta = {}
    for d in (ACTS / "s2", ACTS / "xs"):
        tasks_meta.update(_r(d / "tasks.json")["tasks"])
    cfg = ac.load_acts_config(profile="s2")
    loader = g83._loader_for(cfg, {"tasks": tasks_meta})
    out = OUT / "src_runs" / f"part{args.part}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(todo):
        arrays = loader(c["task_id"])
        row = r.run_src_cell(c, arrays, args.device)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[run-src] {i + 1}/{len(todo)} {c['key']} {row['seconds']}s err={row['error']}", flush=True)


# ================================================================ 比较器成员
def cmd_train(args) -> None:
    import torch
    w, cp = _g("wp18_data"), _g("wp18_comp")
    xs = xspace(); D = w.assemble(xs)
    folds = sorted({fold_of(d) for d in scope_targets(args)})
    for fold in folds:
        idx = [i for i, g in enumerate(D["groups"]) if g != fold]
        (OUT / "models" / fold).mkdir(parents=True, exist_ok=True)
        for q in (args.members or MEMBERS):
            p = OUT / "models" / fold / f"m{q}.pt"
            if p.exists() and args.resume:
                continue
            model, h = cp.train_member(D, idx, int(q), args.device, log=lambda m: print(f"[train {fold}] {m}", flush=True))
            torch.save(model.state_dict(), p)
            _w(OUT / "models" / fold / f"m{q}.json", h)
        print(f"[train] {fold} 完成；任务 {len(idx)}，维度 {xs.dim}", flush=True)


def load_members(xs, device: str, fold: str = "all", members=None):
    import torch
    from autocls.comparator.tclsc_model import TCLSC
    ms = []
    for q in (MEMBERS if members is None else members):
        m = TCLSC(xs.dim, None, task_branch="none", head="field").to(device)
        m.load_state_dict(torch.load(OUT / "models" / fold / f"m{q}.pt", map_location=device)); m.eval()
        ms.append(m)
    return ms


# ================================================================ E（只做相似度）
E_BLOCKS = ("pt", "A", "V")     # E = E_pt（每任务 TS2Vec 窗口表示均值，256）⊕ E_A（机制特征，25）⊕ E_V（视图地标，7）


def _ve() -> dict:
    p = OUT / "ve.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else {}


def source_E_blocks(D: Mapping) -> dict:
    ept = {}
    for p in (ACTS / "s3b" / "E_pt" / "_means.npz", ACTS / "xs" / "E_pt" / "_means.npz"):
        with np.load(p, allow_pickle=True) as z:
            ept.update(dict(zip([str(t) for t in z["tasks"]], z["E_mean"])))
    ea = {}
    for p in (ACTS / "s3n" / "ea.npz", ACTS / "xs" / "ea_new.npz"):
        with np.load(p, allow_pickle=True) as z:
            ea.update(dict(zip([str(t) for t in z["tasks"]], z["EA"])))
    ve = _ve()
    return {"pt": np.array([ept[t] for t in D["tasks"]], float), "A": np.array([ea[t] for t in D["tasks"]], float),
            "V": np.array([ve[f"src/{t}"] for t in D["tasks"]], float)}


def target_E_blocks(d: str) -> dict:
    """靶 E 只用官方 TRAIN：未见靶 = UT 的 E_pt 窗口 / E_A；dev 靶 = S3-MD 的 Ept_target / ea_targets（同一口径）。"""
    if d in DEV_TARGETS:
        with np.load(ACTS / "s3md" / "s0src.npz") as z:
            ept = np.asarray(z[f"Ept_target/{d}"], float)
        with np.load(ACTS / "s3md" / "ea_targets.npz") as z:
            ea = np.asarray(z[d], float)
        return {"pt": ept, "A": ea, "V": np.asarray(_ve()[f"tgt/{d}"], float)}
    with np.load(ACTS / "ut" / "E" / f"{d}.npz") as z:
        ept = np.asarray(z["windows_repr"], np.float64).mean(axis=(0, 1))
    with np.load(ACTS / "ut" / "md" / "ea_targets.npz") as z:
        ea = np.asarray(z[d], float)
    return {"pt": ept, "A": ea, "V": np.asarray(_ve()[f"tgt/{d}"], float)}


class EStd:
    """训练任务上逐维标准化；每块再除以 √维数（各块对余弦的贡献同量级）；blocks 选哪几块进 E。"""

    def __init__(self, src: Mapping[str, np.ndarray], blocks: Sequence[str] = E_BLOCKS):
        self.blocks = tuple(blocks)
        self.mu = {b: src[b].mean(0) for b in self.blocks}
        self.sd = {b: src[b].std(0) + 1e-9 for b in self.blocks}

    def __call__(self, e: Mapping[str, np.ndarray]) -> np.ndarray:
        parts = []
        for b in self.blocks:
            z = (np.asarray(e[b], float) - self.mu[b]) / self.sd[b]
            parts.append(z / np.sqrt(z.shape[-1]))
        return np.concatenate(parts, axis=-1)


def cmd_fit(args) -> None:
    """每折：成员分数（训练源，样本内）→ 源专家；内层留一源选 (α_w, τ, γ)；对该折的靶算 π。键 = {折}/{臂}/…。"""
    w, cp, g115 = _g("wp18_data"), _g("wp18_comp"), _g("115_acts_s3md")
    dev = args.device
    xs = xspace(); D = w.assemble(xs)
    Eb_all = source_E_blocks(D)
    targets = scope_targets(args)
    for fold in sorted({fold_of(d) for d in targets}):
        p = OUT / "experts" / f"{fold}.npz"; p.parent.mkdir(parents=True, exist_ok=True)
        res, fit = {}, {}
        g = D["groups"]
        idx = [i for i in range(len(g)) if g[i] != fold]
        sidx = set(idx)
        sources = list(dict.fromkeys(g[idx].tolist()))
        Z = [(r - r.mean()) / (r.std() + 1e-9) for r in D["r"]]
        models = load_members(xs, dev, fold)
        S0m = [[cp.score(m, D["X"][i], dev) if i in sidx else None for i in range(len(g))] for m in models]
        S0e = [np.mean([g115.task_z(S0m[q][i]) for q in range(len(models))], axis=0) if i in sidx else None for i in range(len(g))]
        est = EStd({b: v[idx] for b, v in Eb_all.items()}, args.e_blocks or E_BLOCKS)
        Es = est(Eb_all)
        fit.setdefault(fold, {})
        for arm in ("ME", "UNI"):
            k, means = cp.fit_inner(D, Z, S0e, Es, sources, idx, uniform=ARMS[arm]["uniform"])
            Ws, cs = zip(*[g115.fit_experts(D["X"], Z, S0m[q], g, sources, k[0], idx) for q in range(len(models))])
            res[f"{fold}/{arm}/W"] = np.array(Ws); res[f"{fold}/{arm}/c"] = np.array(cs)
            res[f"{fold}/{arm}/gamma"] = np.array(k[2]); res[f"{fold}/{arm}/tau"] = np.array(k[1])
            fit[fold][arm] = {"alpha_w": k[0], "tau": k[1], "gamma": k[2], "inner": means[k],
                              "inner_gamma0": max(v for kk, v in means.items() if kk[2] == 0)}
            print(f"[fit] {fold} {arm}: α_w {k[0]} τ {k[1]} γ {k[2]} inner {means[k]:.4f}（γ=0 最好 {fit[fold][arm]['inner_gamma0']:.4f}）", flush=True)
        res[f"{fold}/sources"] = np.array(sources)
        # ---- 轴 × E_V 交互项（FV）：E_V 在训练任务上逐维标准化
        sl = cp.axis_slice(xs, "view") if "view" in xs.axes else None
        if sl is not None:
            ev_mu, ev_sd = Eb_all["V"][idx].mean(0), Eb_all["V"][idx].std(0) + 1e-9
            Ev = (Eb_all["V"] - ev_mu) / ev_sd
            kf, mf = cp.fit_fv_inner(D, Z, S0e, Ev, sources, idx, sl)
            res[f"{fold}/FV/B"] = cp.fit_fv(D["X"], Z, S0e, Ev, idx, sl, kf[0]); res[f"{fold}/FV/gamma"] = np.array(kf[1])
            fit[fold]["FV"] = {"alpha": kf[0], "gamma": kf[1], "inner": mf[kf], "inner_gamma0": max(v for kk, v in mf.items() if kk[1] == 0)}
            print(f"[fit] {fold} FV: α {kf[0]} γ {kf[1]} inner {mf[kf]:.4f}（γ=0 {fit[fold]['FV']['inner_gamma0']:.4f}）", flush=True)
        for d in [d for d in targets if fold_of(d) == fold]:
            if sl is not None:
                res[f"{fold}/FV/ev/{d}"] = (np.asarray(target_E_blocks(d)["V"], float) - ev_mu) / ev_sd
            e_t = est(target_E_blocks(d))
            for arm in ("ME", "UNI"):
                pi = g115.target_pi(e_t, Es[idx], g[idx], sources, float(res[f"{fold}/{arm}/tau"]), ARMS[arm]["uniform"])
                res[f"{fold}/{arm}/pi/{d}"] = pi
                top = np.argsort(-pi)[:3]
                fit.setdefault("pi", {}).setdefault(d, {})[arm] = [[sources[j], float(pi[j])] for j in top]
            print(f"[fit] {d} π(ME) 前 3: {fit['pi'][d]['ME']}", flush=True)
        np.savez_compressed(p, **res)
        _w(OUT / "experts" / f"{fold}.json", fit)


def cmd_prep_ve(args) -> None:
    """视图地标 E_V（autocls/data/views.view_profile，7 维）：源任务用整切片、靶用官方 TRAIN（TEST 不读）→ wp18/ve.npz。"""
    from autocls.data.views import VIEW_PROFILE_COLS, view_profile
    ac, g83, w = _g("acts_common"), _g("83_acts_corpus"), _g("wp18_data")
    p = OUT / "ve.npz"
    have = dict(np.load(p, allow_pickle=True)) if p.exists() else {}
    tasks, meta = w.pool_tasks()
    tm = {}
    for d in (ACTS / "s2", ACTS / "xs"):
        tm.update(_r(d / "tasks.json")["tasks"])
    loader = g83._loader_for(ac.load_acts_config(profile="s2"), {"tasks": tm})
    out = {k: v for k, v in have.items()}
    for t in tasks:
        if f"src/{t}" in out:
            continue
        a = loader(t)
        vp = view_profile(a["X"], a["y"])
        out[f"src/{t}"] = np.array([vp[c] for c in VIEW_PROFILE_COLS])
        print(f"[prep-ve] {t} {np.round(out[f'src/{t}'], 3).tolist()}", flush=True)
        np.savez_compressed(p, **out)
    tmeta = targets_meta(scope_targets(args))
    for d in scope_targets(args):
        if f"tgt/{d}" in out:
            continue
        X, y = ac.load_target_full(d)
        wa = ac.whole_arrays_official(X, y, n_train=tmeta[d]["n_train"])
        vp = view_profile(wa["Xtr"], wa["ytr"])
        out[f"tgt/{d}"] = np.array([vp[c] for c in VIEW_PROFILE_COLS])
        print(f"[prep-ve] {d} {np.round(out[f'tgt/{d}'], 3).tolist()}", flush=True)
        np.savez_compressed(p, **out)


# ================================================================ 搜索
def cmd_search(args) -> None:
    import torch
    cp, g115 = _g("wp18_comp"), _g("115_acts_s3md")
    from autocls.labeling import run_seed_for
    dev = args.device
    xs_full = xspace()
    for d in scope_targets(args):
        fold = fold_of(d)
        ex = np.load(OUT / "experts" / f"{fold}.npz", allow_pickle=True)
        models = load_members(xs_full, dev, fold)
        for arm in args.arms or list(ARMS):
            op = OUT / "search" / f"{arm}_{d}.json"
            if op.exists() and args.resume:
                continue
            t0 = time.time()
            A = ARMS[arm]
            xs = xspace(legacy_only=A["legacy"])
            src = "ME" if arm in ("ME", "LEG", "MEFV") else "UNI"
            W, cw, pi = ex[f"{fold}/{src}/W"], ex[f"{fold}/{src}/c"], ex[f"{fold}/{src}/pi/{d}"]
            gam = 0.0 if A.get("gamma0") else float(ex[f"{fold}/{src}/gamma"])
            cands = xs.sample_unique(np.random.default_rng(run_seed_for(d, "wp18_ks")), KS)
            Vc = xs_full.encode_many(cands)          # 编码统一用全空间维度（旧空间臂的新轴取默认档）
            raw = lambda V: np.stack([cp.score(m, V, dev) for m in models])   # noqa: E731
            S = raw(Vc); st = [(float(s.mean()), float(s.std() + 1e-12)) for s in S]

            fv = None
            if A.get("fv") and f"{fold}/FV/B" in ex.files:
                sl = cp.axis_slice(xs_full, "view")
                fv = (float(ex[f"{fold}/FV/gamma"]), ex[f"{fold}/FV/B"], ex[f"{fold}/FV/ev/{d}"], sl)

            def total(V):
                R = raw(V)
                T = np.stack([g115.member_score((R[q] - st[q][0]) / st[q][1], V, W[q], cw[q], pi, gam) for q in range(len(models))])
                if fv is not None and fv[0] > 0:
                    T = T + fv[0] * cp.fv_term(V, fv[2], fv[1], fv[3])[None, :]
                return T
            Tc = total(Vc)
            searchers = [("ens", lambda V: total(V).mean(0), Tc.mean(0))] + \
                        [(f"m{q}", (lambda q: (lambda V: total(V)[q]))(q), Tc[q]) for q in range(min(len(models), N_MEMBER_SEARCHERS))]
            picks, byk = {}, {}
            for name, fn, f0 in searchers:
                init = [cands[j] for j in np.argsort(-f0, kind="stable")[:KP]]
                fin = xs.evolve(lambda V, fn=fn: fn(V), init, np.random.default_rng(run_seed_for(d, f"wp18_evo_{name}")), KP, GENS, P_CROSS, P_MUT)
                picks[name] = [xs.key(c) for c in fin[:TOP_PER]]
                for c in fin[:TOP_PER]:
                    byk[xs.key(c)] = c
            U = list(dict.fromkeys(k for n, _f, _z in searchers for k in picks[n]))
            Tu = total(xs_full.encode_many([byk[k] for k in U]))
            res = {"union": [{"key": k, **byk[k].to_dict(), "origins": [n for n in picks if k in picks[n]], "ens": float(Tu.mean(0)[j])}
                             for j, k in enumerate(U)],
                   "comp_only": picks["ens"][0], "gamma": gam, "pi_top": sorted(zip(ex[f"{fold}/sources"].tolist(), pi.tolist()), key=lambda x: -x[1])[:5],
                   "seconds": time.time() - t0}
            _w(op, res)
            vc = {}
            for k in U:
                v = byk[k].get("view") if "view" in xs.axes else "raw"
                vc[v] = vc.get(v, 0) + 1
            print(f"[search] {arm}_{d}: |U| {len(U)} γ {gam:g} 视图 {vc} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 部署格
def targets_meta(targets=None) -> dict:
    from autocls.labeling import run_seed_for
    sv = survey()
    out = {}
    for d in (targets or (UT_TARGETS + DEV_TARGETS)):
        r = sv[d]; n, T, C = int(r["n_train"]), int(r["length_max"]), int(r["n_channels"])
        out[d] = {"n_train": n, "T": T, "C": C, "base_iters": 200 if n * T * C <= 100_000 else 600, "tcv_seed": int(run_seed_for(d, "s3y_tcv"))}
    return out


def tgt_cell(d: str, mode: str, e: Mapping, seed: int, tm: Mapping, save_reps: bool = False) -> dict:
    ext = dict(e.get("ext") or {})
    key = f"{d}|{mode}|{e['key']}|s{seed}"
    return {"key": key, "dataset": d, "mode": mode, "cls": e["cls"], "ext": ext, "xkey": e["key"], "seed": int(seed),
            "base_iters": tm[d]["base_iters"], "input_mask": "none", "save_reps": bool(save_reps),
            "n": tm[d]["n_train"], "T": tm[d]["T"], "C": tm[d]["C"]}


def tgt_cost(c: Mapping) -> float:
    ext = c.get("ext") or {}
    tl = c["T"] * (0.35 if ext.get("view") in ("psd", "spec") else 1.0)
    return int(ext.get("mult", 1)) * c["base_iters"] * 8 * tl * 1e-5 + c["n"] ** 1.3 * 2e-3 + 5


def cmd_plan_cv(args) -> None:
    tm = targets_meta(); cells, seen = [], set()
    for arm in args.arms or list(ARMS):
        for d in scope_targets(args):
            p = OUT / "search" / f"{arm}_{d}.json"
            if not p.exists() or tm[d]["n_train"] < 3 * MIN_FOLD_EVAL:     # 小 TRAIN：不跑 CV（选择按比较器分数）
                continue
            for e in _r(p)["union"]:
                for i in range(K_TCV):
                    c = tgt_cell(d, f"tcv{i}", e, 0, tm)
                    if c["key"] not in seen:
                        seen.add(c["key"]); cells.append(c)
    _w(OUT / f"plan_cv_{args.scope}.json", {"cells": cells})
    print(f"[plan-cv] {len(cells)} 格", flush=True)


def cv_scores(runs: Mapping, d: str, keys: Sequence[str]) -> dict:
    out = {}
    for k in keys:
        a = [runs.get(f"{d}|tcv{i}|{k}|s0", {}).get("acc") for i in range(K_TCV)]
        a = [x for x in a if x is not None]
        if a:
            out[k] = float(np.mean(a))
    return out


def cmd_select(args) -> None:
    r = _g("wp18_run")
    runs = r.load_runs(OUT / "tgt_runs")
    sel = _r(OUT / "select.json") if (OUT / "select.json").exists() else {}
    for arm in args.arms or list(ARMS):
        for d in scope_targets(args):
            p = OUT / "search" / f"{arm}_{d}.json"
            if not p.exists():
                continue
            S = _r(p); U = [e["key"] for e in S["union"]]; ens = {e["key"]: e["ens"] for e in S["union"]}
            cv = cv_scores(runs, d, U)
            small = targets_meta([d])[d]["n_train"] < 3 * MIN_FOLD_EVAL      # 小 TRAIN 规则：验证折 < 10 个样本 → CV 无分辨率，按比较器分数
            if small:
                ranked = sorted(U, key=lambda k: (-ens.get(k, -np.inf), U.index(k)))
            else:
                ranked = sorted(cv, key=lambda k: (-round(cv[k], 12), -ens.get(k, -np.inf), U.index(k)))
            sel.setdefault(arm, {})[d] = {"top": ranked[:TOPK_ENS], "cv": cv, "n_union": len(U), "n_cv": len(cv), "rule": "comp" if small else "cv"}
    _w(OUT / "select.json", sel)
    print(f"[select] {sum(len(v) for v in sel.values())} 个（臂, 靶）", flush=True)


def cmd_plan_final(args) -> None:
    tm = targets_meta(); sel = _r(OUT / "select.json"); cells, seen = [], set()
    tg = set(scope_targets(args))
    for arm, per in sel.items():
        for d, s in per.items():
            if d not in tg:
                continue
            S = {e["key"]: e for e in _r(OUT / "search" / f"{arm}_{d}.json")["union"]}
            for k in s["top"]:
                for sd in FINAL_SEEDS:
                    c = tgt_cell(d, "official", S[k], sd, tm, save_reps=True)
                    if c["key"] not in seen:
                        seen.add(c["key"]); cells.append(c)
    _w(OUT / f"plan_final_{args.scope}.json", {"cells": cells})
    print(f"[plan-final] {len(cells)} 格", flush=True)


def cmd_run_tgt(args) -> None:
    r = _g("wp18_run")
    cells = _r(OUT / f"plan_{args.phase}_{args.scope}.json")["cells"]
    mine = r.lpt_parts(cells, args.parts, tgt_cost)[args.part]
    done = r.done_keys(OUT / "tgt_runs")
    todo = sorted([c for c in mine if c["key"] not in done], key=lambda c: (c["dataset"], c["mode"]))
    print(f"[run-tgt] {args.phase} part {args.part}/{args.parts}: {len(todo)}/{len(mine)} 待跑", flush=True)
    out = OUT / "tgt_runs" / f"{args.phase}_{args.scope}_part{args.part}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    slot: dict = {}
    for i, c in enumerate(todo):
        tag = (c["dataset"], c["mode"])
        if slot.get("tag") != tag:
            slot = {"tag": tag, "arr": r.target_arrays(c["dataset"], c["mode"], c["n"], targets_meta()[c["dataset"]]["tcv_seed"])}
        row = r.run_tgt_cell(c, slot["arr"], args.device, rep_dir=OUT / "reps")
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[run-tgt] {i + 1}/{len(todo)} {c['key']} acc={row.get('acc')} {row['seconds']}s err={row['error']}", flush=True)


def ensemble_acc(reps: Sequence[Mapping]) -> float:
    """多个编码器终点读出逐样本 L2 归一后拼接 → 官方 SVM（= UT-ENS 口径）。"""
    from autocls.downstream.linear_eval import fit_svm
    from sklearn.metrics import accuracy_score
    def l2(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    ftr = np.hstack([l2(z["ftr"]) for z in reps]); fev = np.hstack([l2(z["fev"]) for z in reps])
    clf = fit_svm(ftr, reps[0]["ytr"], grid="ts2vec")
    return float(accuracy_score(reps[0]["yev"], clf.predict(fev)))


def cmd_ensemble(args) -> None:
    r = _g("wp18_run")
    runs = r.load_runs(OUT / "tgt_runs"); sel = _r(OUT / "select.json"); out = {}
    for arm, per in sel.items():
        for d, s in per.items():
            res = {}
            for k in ENS_KS:
                accs = []
                for sd in FINAL_SEEDS:
                    keys = [f"{d}|official|{x}|s{sd}" for x in s["top"][:k]]
                    if len(keys) < k or not all(x in runs for x in keys):
                        continue
                    if k == 1:
                        accs.append(runs[keys[0]]["acc"]); continue
                    reps = []
                    for x in keys:
                        with np.load(OUT / "reps" / f"{x}.npz") as z:
                            reps.append({kk: z[kk] for kk in z.files})
                    accs.append(ensemble_acc(reps))
                if len(accs) == len(FINAL_SEEDS):
                    res[f"k{k}"] = float(np.mean(accs)); res[f"k{k}_seeds"] = accs
            if res:
                out.setdefault(arm, {})[d] = res
                print(f"[ensemble] {arm} {d}: " + " ".join(f"k{k} {res.get(f'k{k}', float('nan')):.3f}" for k in ENS_KS), flush=True)
    _w(OUT / "final.json", out)


def cmd_analyze(args) -> None:
    F = _r(OUT / "final.json")
    arms = [a for a in ARMS if a in F]
    ds = [d for d in scope_targets(args) if all(d in F[a] for a in arms)]
    lines = [f"# WP18 部署读数（{args.scope}；官方 TEST，3 种子均值；k = CV 前 k 拼接集成，k1 = CV 第 1 单模型）", ""]
    for k in ENS_KS:
        lines += [f"## k = {k}", "", "| 数据集 | " + " | ".join(arms) + " | TS2Vec | InfoTS_s |", "|---|" + "---:|" * (len(arms) + 2)]
        for d in ds:
            lines.append(f"| {d} | " + " | ".join(f"{F[a][d].get(f'k{k}', float('nan')):.3f}" for a in arms) + f" | {TS2VEC[d]:.3f} | {INFOTS_S[d]:.3f} |")
        lines.append("| **均值** | " + " | ".join(f"{np.mean([F[a][d].get(f'k{k}', np.nan) for d in ds]):.4f}" for a in arms)
                     + f" | {np.mean([TS2VEC[d] for d in ds]):.4f} | {np.mean([INFOTS_S[d] for d in ds]):.4f} |")
        lines.append("")
    text = chr(10).join(lines) + chr(10)
    (OUT / f"tables_{args.scope}.md").write_text(text, encoding="utf-8")
    print(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for f in ("plan-src", "run-src", "prep-ve", "train", "fit", "search", "plan-cv", "select", "plan-final", "run-tgt", "ensemble", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--phase", default="cv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--members", type=int, nargs="*")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--e-blocks", nargs="*")
    ap.add_argument("--scope", choices=("dev", "ut"), default="ut")
    a = ap.parse_args(argv)
    for f, fn in (("plan_src", cmd_plan_src), ("run_src", cmd_run_src), ("prep_ve", cmd_prep_ve), ("train", cmd_train), ("fit", cmd_fit), ("search", cmd_search),
                  ("plan_cv", cmd_plan_cv), ("select", cmd_select), ("plan_final", cmd_plan_final), ("run_tgt", cmd_run_tgt),
                  ("ensemble", cmd_ensemble), ("analyze", cmd_analyze)):
        if getattr(a, f):
            fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
