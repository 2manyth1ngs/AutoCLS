"""WP17-ACTS S3-C（results/acts/s3c/PROTOCOL.md）：黑盒比较器 T-CLSC（AutoCTS++ 原样标签）+ 源侧搜索验证（6 个 UEA 折）。

阶段：
  --prep          本地：S2 缓存 → best-of-3 单种子标签、候选向量、折 / 验证源、臂 C 的 PCA ctx、搜索任务、S2 已有格清单 → data.json / data.npz
  --train         服务器：--arm A|B|C|D --fold <UEA 源> → models/{arm}_{fold}.pt + train/{arm}_{fold}.npz（留出任务 P 矩阵、16 偏移 cross、E_all）
  --search        服务器：--arm --fold → search/{arm}_{fold}.json（24 任务里本折 4 个：Ks 采样 → 分块 RR → 进化 → RR 前 3 + 随机对照）
  --plan-runs     服务器：提名去重（S2 已有格复用）→ runs/plan_X.json（83 --run --phase X 真跑）
  --analyze       本地：排序门 C1–C5 + 搜索门 S1–S3 + 参照 → analysis.json / tables.md
  --smoke         本地冒烟（合成窗口、2 折 × 2 epoch、Ks 2,000 × 2 代）
T-CLSC 结构 ← autocls/comparator/tclsc_model.py（对齐 ahc_engine.py::AHC）；配对 ← AutoCTS++ AHC_DataLoader_linear.generate_pairs（平局记 0）。
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3c"
S2_DIR = ROOT / "results" / "acts" / "s2"
S3B = ROOT / "results" / "acts" / "s3b"
FOLDS = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1")
SEARCH_ROWS = (1, 3, 5, 7)
ARMS = ("A", "B", "C", "D")
KS, CHUNK, PER_CHUNK, KP, GENS, P_CROSS, P_MUT = 20000, 500, 10, 10, 10, 0.8, 0.2
SUB_WIN, BATCH, LR, MAX_EPOCHS, PATIENCE, R_C = 32, 64, 1e-4, 100, 5, 4
N_SH = 30


# ================================================================ 纯函数
def best_of3(vals: Sequence) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(float(x))]
    return max(v) if v else float("nan")


def linear_pairs(r: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """← AHC_DataLoader_linear.generate_pairs：记录与其乱序副本逐位配对，y = 1 若 r_a > r_b 否则 0（平局记 0）；跳自配对。"""
    r = np.asarray(r, float)
    perm = rng.permutation(len(r))
    return [(i, int(perm[i]), int(r[i] > r[perm[i]])) for i in range(len(r)) if int(perm[i]) != i]


def all_ordered_pairs(r: np.ndarray) -> list[tuple[int, int, int]]:
    r = np.asarray(r, float)
    return [(i, j, int(r[i] > r[j])) for i in range(len(r)) for j in range(len(r)) if i != j]


def rr_order(P: np.ndarray, keys: Sequence[str]) -> list[int]:
    """Round-Robin：S = 对称化 P，胜场 = #{S > 0.5}，按 (−胜场, key) 排序（= evolution 的稳定 tie-break）。"""
    from autocls.search.rank import symmetrize, wins_from_scores
    P = np.array(P, float)
    np.fill_diagonal(P, 0.5)
    w = wins_from_scores(symmetrize(P))
    return sorted(range(len(keys)), key=lambda i: (-int(w[i]), keys[i]))


def chunk_select(n: int, pmat: Callable[[Sequence[int]], np.ndarray], keys: Sequence[str], *, chunk: int = CHUNK,
                 per_chunk: int = PER_CHUNK, final: int = KP) -> list[int]:
    """← 源码"分块 nlargest + 块赢家两两计分"：连续分块，块内 RR 取前 per_chunk → 块赢家 RR 取前 final。"""
    fin: list[int] = []
    for s in range(0, n, chunk):
        idx = list(range(s, min(n, s + chunk)))
        o = rr_order(pmat(idx), [keys[i] for i in idx])
        fin += [idx[j] for j in o[:per_chunk]]
    o = rr_order(pmat(fin), [keys[i] for i in fin])
    return [fin[j] for j in o[:final]]


def pick_final(keys: Sequence[str], v0: Mapping[str, float], v1: Mapping[str, float]) -> tuple[str, float]:
    """seed 0 终点选 1 条（并列取先出现），报 seed 1 终点。"""
    best = max(keys, key=lambda k: v0[k])
    return best, float(v1[best])


def n_sources_positive(d: np.ndarray, groups: Sequence) -> int:
    d = np.asarray(d, float); groups = np.asarray(groups)
    return int(sum(1 for g in dict.fromkeys(groups.tolist()) if np.mean(d[groups == g]) > 0))


def verdict_rank(g: Mapping[str, bool]) -> str:
    base = ("S3-E-WIN" if g["C3"] else "S3-LOWRANK-SUFFICES") if g["C2"] else "S3-E-INERT"
    return base + ("+ID-OK" if g["C1"] else "+IDENTITY") + ("+SPACE" if g["C5"] else "")


def verdict_search(g: Mapping[str, bool]) -> str:
    base = ("SEARCH-WIN" if g["S2"] else "SEARCH-BLIND") if g["S1"] else "SEARCH-DEAD"
    return base + ("+CATALOG-OK" if g["S3"] else "+CATALOG-LOSS")


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    ac = importlib.import_module("scripts.acts_common"); g83 = importlib.import_module("scripts.83_acts_corpus")
    g85 = importlib.import_module("scripts.85_acts_s3ap"); g86 = importlib.import_module("scripts.86_acts_s3b"); g87 = importlib.import_module("scripts.87_acts_s3bp")
    cfg = ac.load_acts_config(profile="s2")
    data = g85.load_all(); D = data["pool"]
    tasks = [str(t) for t in D["tasks"]]; groups = D["groups"]; order = list(dict.fromkeys(groups.tolist()))
    plan = json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8")); cat = json.loads((S2_DIR / "catalog.json").read_text(encoding="utf-8"))
    rows = g83._g35.load_done(S2_DIR / "runs.jsonl")
    frames = g83._frames_c(plan, rows, cat, cfg)
    by = {(c["task_id"], c["cls_key"], int(c["seed"]), c["arm"]): c["key"] for c in plan["cells"]}
    labs = {(r["task_id"], r["cls_key"]): r for r in g83.read_jsonl(S2_DIR / "labels.jsonl")}
    sh_keys = g83.cat_keys(cat)

    def ckpts(t, k, seed, arm):
        r = rows.get(by.get((t, k, seed, arm)))
        if not r or r.get("error") or not r.get("checkpoint_results"):
            return None
        return [c.get("acc") for c in r["checkpoint_results"]]

    npz, meta, c0_bad, n_err = {}, {"tasks": tasks, "groups": groups.tolist(), "order": order, "folds": list(FOLDS)}, 0, 0
    keys_of = {}
    for i, t in enumerate(tasks):
        rk = frames[t]["random_keys"]
        keys = list(sh_keys) + list(rk)
        arms = [ac.ARM_SHARED] * N_SH + [ac.ARM_RANDOM] * len(rk)
        r = []
        for k, arm in zip(keys, arms):
            ck = ckpts(t, k, 0, arm)
            if ck is None:
                n_err += 1; r.append(np.nan); continue
            r.append(best_of3(ck))
            lab = labs.get((t, k))
            if lab is not None and abs(float(ck[-1]) - float(lab["seed_vals"][0])) > 1e-12:
                c0_bad += 1
        r = np.array(r, float)
        r[np.isnan(r)] = np.nanmin(r) if np.isfinite(r).any() else 0.0
        keys_of[t] = keys
        npz[f"{t}/vec"] = np.vstack([data["phi_sh"], D["rand"][i]["phi"]]).astype(np.float32)
        npz[f"{t}/r"] = r
        npz[f"{t}/R0"] = np.asarray(D["rand"][i]["R0"], float)
    npz["end0"] = D["end0"]; npz["end1"] = D["end1"]
    em_sh = g86._e_means(S3B, "sh", g86._load_targets(S3B)["tasks"].astype(str).tolist(), order + ["ALL"])
    n = len(tasks)
    val = {}
    for s in FOLDS:
        tr = np.where(groups != s)[0]
        f = g87.pca_fit(em_sh[s][:n][tr], R_C)
        npz[f"ctx/{s}"] = g87.pca_apply(f, em_sh[s][:n]).astype(np.float32)
        val[s] = importlib.import_module("scripts.86_acts_s3b").val_source(order, s)
    search = {s: [f"{s}_a{r:02d}" for r in SEARCH_ROWS if f"{s}_a{r:02d}" in tasks] for s in FOLDS}
    s2_done = sorted({f"{c['task_id']}|{c['cls_key']}|{int(c['seed'])}" for c in plan["cells"]
                      if c["task_id"] in {t for v in search.values() for t in v} and rows.get(c["key"]) and not rows[c["key"]].get("error")})
    meta.update({"keys": keys_of, "val": val, "search": search, "s2_done": s2_done, "n_sh": N_SH,
                 "c0": {"endpoint_mismatch": c0_bad, "n_error_cells": n_err}})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "data.npz", **npz)
    (OUT / "data.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[prep] {n} tasks; endpoint mismatch {c0_bad}; error cells {n_err}; search tasks {sum(len(v) for v in search.values())}; s2 done {len(s2_done)}", flush=True)


def load_data(outdir: Path) -> tuple[dict, dict]:
    meta = json.loads((outdir / "data.json").read_text(encoding="utf-8"))
    with np.load(outdir / "data.npz") as z:
        arr = {k: z[k] for k in z.files}
    return meta, arr


# ================================================================ 模型
def build_model(arm: str, cls_dim: int, device: str):
    import torch
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.task_embed import TaskSetEncoder
    torch.manual_seed(0)
    if arm in ("A", "B"):
        m = TCLSC(cls_dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head="concat")
    elif arm == "C":
        m = TCLSC(cls_dim, None, ctx_dim=R_C, task_branch="ctx", head="field")
    else:
        m = TCLSC(cls_dim, None, task_branch="none", head="concat")
    return m.to(device)


def load_windows(arm: str, fold: str, tasks: Sequence[str], s3b: Path, synthetic: bool = False) -> Optional[list]:
    if arm not in ("A", "B"):
        return None
    if synthetic:
        rng = np.random.default_rng(0)
        return [rng.normal(size=(20, 16, 256)).astype(np.float16) for _ in tasks]
    d = s3b / ("E_pt" if arm == "A" else f"E_sh/{fold}")
    out = []
    for t in tasks:
        with np.load(d / f"{t}.npz") as z:
            out.append(z["windows_repr"])
    return out


class Scorer:
    """把 (配置向量对, 任务条件) 送进 T-CLSC；任务条件 = E（A/B，SetTF 输出）/ ctx（C）/ 无（D）。"""

    def __init__(self, model, arm: str, device: str):
        self.m, self.arm, self.dev = model, arm, device

    def task_vec(self, windows=None, ctx=None, rng=None, sub=False):
        import torch
        if self.arm in ("A", "B"):
            from importlib import import_module
            w = import_module("scripts.86_acts_s3b").subsample(windows, SUB_WIN, rng) if sub else windows
            return self.m.encode_tasks([torch.as_tensor(np.asarray(w), dtype=torch.float32, device=self.dev)])[0]
        if self.arm == "C":
            return torch.as_tensor(np.asarray(ctx), dtype=torch.float32, device=self.dev)
        return None

    def forward(self, va, vb, T):
        """va / vb (B, d) 张量；T (B, nz|ctx) 张量或 None。"""
        if self.arm in ("A", "B"):
            return self.m(va, vb, T)
        if self.arm == "C":
            return self.m(va, vb, None, T)
        return self.m(va, vb, None)

    def pmat(self, V: np.ndarray, T, batch: int = 65536) -> np.ndarray:
        import torch
        n = len(V)
        P = np.full((n, n), 0.5)
        ii, jj = np.where(~np.eye(n, dtype=bool))
        Vt = torch.as_tensor(np.asarray(V, np.float32), device=self.dev)
        with torch.no_grad():
            for s in range(0, len(ii), batch):
                a, b = ii[s:s + batch], jj[s:s + batch]
                Tb = None if T is None else T.unsqueeze(0).expand(len(a), -1)
                P[a, b] = self.forward(Vt[a], Vt[b], Tb).float().cpu().numpy()
        return P


# ================================================================ train（服务器）
def cmd_train(args) -> None:
    import torch
    outdir = Path(args.outdir); meta, arr = load_data(outdir)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]; fold, arm = args.fold, args.arm
    dev = args.device
    tag = f"{arm}_{fold}"
    (outdir / "train").mkdir(parents=True, exist_ok=True); (outdir / "models").mkdir(parents=True, exist_ok=True)
    op = outdir / "train" / f"{tag}.npz"
    if op.exists() and args.resume:
        print(f"[train] skip {tag}"); return
    vs = meta["val"][fold]
    tr = [i for i, g in enumerate(groups) if g not in (fold, vs)]
    va = [i for i, g in enumerate(groups) if g == vs]
    te = [i for i, g in enumerate(groups) if g == fold]
    V = [arr[f"{t}/vec"] for t in tasks]; R = [arr[f"{t}/r"] for t in tasks]
    ctx = arr.get(f"ctx/{fold}")
    W = load_windows(arm, fold, tasks, S3B, synthetic=args.synthetic)
    model = build_model(arm, V[0].shape[1], dev); sc = Scorer(model, arm, dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = torch.nn.BCELoss()
    rng = np.random.default_rng(0)
    val_pairs = [(i, a, b, y) for i in va for a, b, y in all_ordered_pairs(R[i])]
    max_ep = args.max_epochs or MAX_EPOCHS

    def tvecs(idx, sub):
        if arm == "D":
            return None
        return torch.stack([sc.task_vec(W[i] if W else None, ctx[i] if ctx is not None else None, rng, sub) for i in idx])

    def batch_loss(batch, sub):
        tids = sorted({b[0] for b in batch}); pos = {t: j for j, t in enumerate(tids)}
        T = tvecs(tids, sub)
        va_ = torch.as_tensor(np.stack([V[b[0]][b[1]] for b in batch]), dtype=torch.float32, device=dev)
        vb_ = torch.as_tensor(np.stack([V[b[0]][b[2]] for b in batch]), dtype=torch.float32, device=dev)
        y = torch.as_tensor([float(b[3]) for b in batch], dtype=torch.float32, device=dev)
        Tb = None if T is None else T[[pos[b[0]] for b in batch]]
        return bce(sc.forward(va_, vb_, Tb), y)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        pairs = [(i, a, b, y) for i in tr for a, b, y in linear_pairs(R[i], rng)]
        perm = rng.permutation(len(pairs))
        tot, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            batch = [pairs[j] for j in perm[s:s + BATCH]]
            loss = batch_loss(batch, True)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]
                vl += float(batch_loss(b, False)) * len(b); vn += len(b)
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl, "sec": round(time.time() - t0, 1)})
        print(f"[train] {tag} ep {ep} train {tot / max(nb, 1):.4f} val {vl:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl < best - 1e-9:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    g88 = importlib.import_module("scripts.88_acts_s3bpp")
    cmaps = g88.cross_maps(tasks, groups, order)
    out = {"hist": json.dumps(hist), "te": np.array(te)}
    with torch.no_grad():
        Tall = None if arm == "D" else torch.stack([sc.task_vec(W[i] if W else None, ctx[i] if ctx is not None else None) for i in range(len(tasks))])
        if Tall is not None:
            out["T_all"] = Tall.float().cpu().numpy()
        for i in te:
            Ti = None if Tall is None else Tall[i]
            out[f"P/{tasks[i]}"] = sc.pmat(V[i], Ti).astype(np.float32)
            if Tall is not None:
                out[f"X/{tasks[i]}"] = np.stack([sc.pmat(V[i][:N_SH], Tall[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best val {best:.4f} epochs {len(hist)} {time.time() - t0:.0f}s", flush=True)


# ================================================================ search（服务器）
def cmd_search(args) -> None:
    import torch
    ac = importlib.import_module("scripts.acts_common")
    from autocls.labeling import run_seed_for
    from autocls.search.evolution import evolutionary_search
    from autocls.space.cls_encoder import CLSEncoder
    outdir = Path(args.outdir); meta, arr = load_data(outdir)
    tasks = meta["tasks"]; fold, arm = args.fold, args.arm; dev = args.device
    tag = f"{arm}_{fold}"
    (outdir / "search").mkdir(parents=True, exist_ok=True)
    op = outdir / "search" / f"{tag}.json"
    if op.exists() and args.resume:
        print(f"[search] skip {tag}"); return
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    model = build_model(arm, arr[f"{tasks[0]}/vec"].shape[1], dev)
    model.load_state_dict(torch.load(outdir / "models" / f"{tag}.pt", map_location=dev)); model.eval()
    sc = Scorer(model, arm, dev)
    with np.load(outdir / "train" / f"{tag}.npz") as z:
        T_all = z["T_all"] if "T_all" in z.files else None
    ks = args.ks or KS; gens = args.gens if args.gens is not None else GENS
    res = {}
    t0 = time.time()
    for t in meta["search"][fold]:
        i = tasks.index(t)
        T = None if T_all is None else torch.as_tensor(T_all[i], dtype=torch.float32, device=dev)
        rng = np.random.default_rng(run_seed_for(t, "s3c_ks"))
        cands = ac._sample_catalog(space, cfg, rng, ks, set())
        Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
        top = chunk_select(len(cands), lambda idx: sc.pmat(Vc[np.asarray(idx)], T), keys)

        def score_pairs(a, b):
            with torch.no_grad():
                at = torch.as_tensor(np.asarray(a, np.float32), device=dev); bt = torch.as_tensor(np.asarray(b, np.float32), device=dev)
                Tb = None if T is None else T.unsqueeze(0).expand(len(at), -1)
                return sc.forward(at, bt, Tb).float().cpu().numpy()

        evo = evolutionary_search(score_pairs, space, enc, ks=0, kp=KP, generations=gens, p_cross=P_CROSS, p_mut=P_MUT, top_k=3, batch_size=65536,
                                  rng=np.random.default_rng(run_seed_for(t, "s3c_evo")), seed_population=[cands[j][0] for j in top],
                                  accept=lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg))
        res[t] = {"nominated": [{"cls_key": e["cls_key"], "cls": e["cls"], "wins": e["wins"], "generation_found": e["generation_found"]} for e in evo],
                  "init_top": [keys[j] for j in top],
                  "random": [{"cls_key": k, "cls": c.to_dict()} for c, k in cands[:3]], "ks": len(cands), "gens": gens}
        print(f"[search] {tag} {t} → {[e['cls_key'] for e in evo]} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


# ================================================================ plan-runs（服务器）
def cmd_plan_runs(args) -> None:
    ac = importlib.import_module("scripts.acts_common")
    from autocls.space.cls_space import CLS
    g35 = importlib.import_module("scripts.35_protocol_alignment")
    outdir = Path(args.outdir); meta, _arr = load_data(outdir)
    cfg = ac.load_acts_config(profile="s2"); protos = g35.load_protocols()
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8"))
    tm = tj["tasks"]
    done = set(meta["s2_done"])
    cells, seen = [], set()
    for f in sorted((outdir / "search").glob("*.json")):
        res = json.loads(f.read_text(encoding="utf-8"))
        for t, r in res.items():
            for e in r["nominated"] + r["random"]:
                for sd in (0, 1):
                    tag = f"{t}|{e['cls_key']}|{sd}"
                    if tag in done or tag in seen:
                        continue
                    seen.add(tag)
                    cells.append(ac.make_cell(t, tm[t], ac.ARM_RANDOM, CLS(**e["cls"]), e["cls_key"], sd, protos, cfg))
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    search_tasks = [t for v in meta["search"].values() for t in v]
    plan = {"phase": "X", "profile": cfg.get("profile"), "n_cells": len(cells), "probe_tasks": search_tasks, "protocol": cfg["protocol"],
            "tasks_meta": {t: {k: tm[t][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in tm[t]} for t in search_tasks},
            "cells": cells}
    (rd / "plan_X.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    (rd / "tasks.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    print(f"[plan-runs] {len(cells)} new cells over {len(search_tasks)} tasks", flush=True)


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    g83 = importlib.import_module("scripts.83_acts_corpus"); g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    g86 = importlib.import_module("scripts.86_acts_s3b"); g88 = importlib.import_module("scripts.88_acts_s3bpp")
    outdir = Path(args.outdir); meta, arr = load_data(outdir)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    data = g85.load_all(); D = data["pool"]; b_ref = g85.orient_bref(D)
    assert [str(t) for t in D["tasks"]] == tasks
    end0, end1 = D["end0"], D["end1"]
    held = [i for i, g in enumerate(groups) if g in FOLDS]; hg = groups[held]
    # 参照：任务盲 / ④ / ③（84 同路径，C0 = 全 160 均值与 S3-A 逐位）+ field-E ridge（88，E_sh，只算 6 折）
    ref = g84.loso_evaluate(D["Z"], groups, end0, end1, {"probe": g84.probe_feature(D["acc25"], D["pfloor"], D["maj"]), "land25": g84.land25_feature(D["Z25"])}, b_ref=b_ref)
    s3a = json.loads((ROOT / "results/acts/s3a/analysis.json").read_text(encoding="utf-8"))
    c0 = {"blind3": (float(ref["_blind"]["regret3"].mean()), s3a["blind"]["regret3"]), "probe3": (float(ref["probe"]["regret3"].mean()), s3a["summary"]["probe"]["regret3"]),
          "land3": (float(ref["land25"]["regret3"].mean()), s3a["summary"]["land25"]["regret3"]), "prep_endpoint_mismatch": (meta["c0"]["endpoint_mismatch"], 0)}
    T = g86._load_targets(S3B); all_t = [str(t) for t in T["tasks"]]
    em_sh = {f: v[:len(tasks)] for f, v in g86._e_means(S3B, "sh", all_t, order).items()}
    fe = g88.run_arm_multi(D, data["phi_sh"], (lambda s: em_sh[s]), None, list(FOLDS), g88.cross_maps(tasks, groups, order))
    refs = {"blind": ref["_blind"]["regret3"][held], "probe": ref["probe"]["regret3"][held], "land25": ref["land25"]["regret3"][held],
            "fieldE": g84.regret_at_k(fe["sel"]["pred_sh"][held], end0[held], end1[held], 3)}
    # 排序
    rank = {}
    for arm in ARMS:
        r3, r1, sp, rx = np.full(len(held), np.nan), np.full(len(held), np.nan), np.full(len(held), np.nan), np.full(len(held), np.nan)
        Tall = {}
        ok = True
        for s in FOLDS:
            p = outdir / "train" / f"{arm}_{s}.npz"
            if not p.exists():
                ok = False; break
            with np.load(p) as z:
                for jj, i in enumerate(held):
                    t = tasks[i]
                    if f"P/{t}" not in z.files:
                        continue
                    P = z[f"P/{t}"]; keys = meta["keys"][t]
                    o = rr_order(P[:N_SH, :N_SH], keys[:N_SH])
                    S = np.full((1, N_SH), 0.0); S[0, o] = -np.arange(N_SH)
                    r3[jj] = g84.regret_at_k(S, end0[[i]], end1[[i]], 3)[0]; r1[jj] = g84.regret_at_k(S, end0[[i]], end1[[i]], 1)[0]
                    oa = rr_order(P, keys); rank_of = np.empty(len(keys)); rank_of[oa] = np.arange(len(keys))
                    R0 = arr[f"{t}/R0"]
                    if len(R0) >= 10:
                        sp[jj] = float(spearmanr(-rank_of[N_SH:N_SH + len(R0)], R0)[0])
                    if f"X/{t}" in z.files:
                        xs = []
                        for Px in z[f"X/{t}"]:
                            ox = rr_order(Px, keys[:N_SH]); Sx = np.zeros((1, N_SH)); Sx[0, ox] = -np.arange(N_SH)
                            xs.append(g84.regret_at_k(Sx, end0[[i]], end1[[i]], 3)[0])
                        rx[jj] = float(np.mean(xs))
                if "T_all" in z.files:
                    Tall[s] = z["T_all"]
        if not ok:
            continue
        row = {"regret3": float(np.nanmean(r3)), "regret1": float(np.nanmean(r1)), "sp_median": float(np.nanmedian(sp)), "_r3": r3}
        for nm, rv in refs.items():
            d = rv - r3
            row[f"vs_{nm}"] = float(d.mean()); row[f"vs_{nm}_ci"] = g84.cluster_bootstrap_ci(d, hg); row[f"p_vs_{nm}"] = g84.source_wilcoxon(d, hg)
        if np.isfinite(rx).all():
            d = rx - r3
            row.update({"cross": float(d.mean()), "cross_ci": g84.cluster_bootstrap_ci(d, hg), "p_cross": g84.source_wilcoxon(d, hg), "cross_npos": n_sources_positive(d, hg)})
        if arm in ("A", "B") and Tall:
            E = np.vstack([Tall[s][[i]] for s in FOLDS for i in held if groups[i] == s])
            row["cos_within"], row["cos_between"] = g86.cosine_within_between(E, np.array([groups[i] for s in FOLDS for i in held if groups[i] == s]))
        rank[arm] = row
    gates_rank = None
    if "A" in rank:
        a = rank["A"]
        gates_rank = {"C1": bool(a.get("cos_within", 1.0) < 0.9), "C2": bool(a.get("cross", 0) >= 0.010 and a.get("p_cross", 1) < 0.05),
                      "C3": bool(a["vs_probe"] >= 0.005 and a["p_vs_probe"] < 0.05), "C5": bool(a["sp_median"] >= 0.50)}
    # 搜索
    search = None
    if list((outdir / "search").glob("*.json")) and (outdir / "runs" / "runs.jsonl").exists():
        rows = dict(g83._g35.load_done(S2_DIR / "runs.jsonl")); rows.update(g83._g35.load_done(outdir / "runs" / "runs.jsonl"))
        plan_s2 = json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8")); plan_x = json.loads((outdir / "runs" / "plan_X.json").read_text(encoding="utf-8"))
        val_of = {}
        for c in plan_s2["cells"] + plan_x["cells"]:
            r = rows.get(c["key"])
            if r and not r.get("error") and r.get("checkpoint_results"):
                val_of[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
        stasks = [t for s in FOLDS for t in meta["search"][s]]
        sidx = [tasks.index(t) for t in stasks]; sg = groups[sidx]
        # 目录任务盲前 3（训练源平均秩，S2 H_cat）与目录 oracle
        from scipy.stats import rankdata
        cat3, oracle = [], []
        for i in sidx:
            trn = np.where(groups != groups[i])[0]
            mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
            cat3.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); oracle.append(float(end1[i, int(np.argmax(end0[i]))]))
        cat3, oracle = np.array(cat3), np.array(oracle)
        final, miss = {}, 0
        for arm in list(ARMS) + ["R"]:
            vals = []
            for t, i in zip(stasks, sidx):
                src = "A" if arm == "R" else arm
                p = outdir / "search" / f"{src}_{groups[i]}.json"
                if not p.exists():
                    vals.append(np.nan); continue
                e = json.loads(p.read_text(encoding="utf-8"))[t]
                ent = e["random"] if arm == "R" else e["nominated"]
                ks_ = [x["cls_key"] for x in ent]
                v0 = {k: val_of.get((t, k, 0), np.nan) for k in ks_}; v1 = {k: val_of.get((t, k, 1), np.nan) for k in ks_}
                if any(not np.isfinite(v) for v in list(v0.values()) + list(v1.values())):
                    miss += 1
                    fl = float(np.nanmin(end0[i]))
                    v0 = {k: (v if np.isfinite(v) else fl) for k, v in v0.items()}; v1 = {k: (v if np.isfinite(v) else fl) for k, v in v1.items()}
                vals.append(pick_final(ks_, v0, v1)[1])
            final[arm] = np.array(vals)
        from scipy.stats import wilcoxon
        def cmp(a, b):
            d = final[a] - (final[b] if isinstance(b, str) else b)
            ok_ = np.isfinite(d)
            p = float(wilcoxon(d[ok_]).pvalue) if np.any(np.abs(d[ok_]) > 1e-12) else 1.0
            return {"d": float(np.nanmean(d)), "ci": g84.cluster_bootstrap_ci(d[ok_], sg[ok_]), "p": p, "npos": n_sources_positive(np.nan_to_num(d), sg)}
        search = {"final": {k: float(np.nanmean(v)) for k, v in final.items()}, "cat3": float(cat3.mean()), "oracle": float(oracle.mean()), "missing_tasks": miss,
                  "cmp": {a: {"vs_R": cmp(a, "R"), "vs_D": cmp(a, "D") if a != "D" else None, "vs_cat3": cmp(a, cat3), "vs_oracle": cmp(a, oracle)} for a in ARMS if np.isfinite(final[a]).any()},
                  "beat_oracle_frac": {a: float(np.nanmean(final[a] > oracle)) for a in ARMS}}
        A = search["cmp"].get("A")
        if A:
            search["gates"] = {"S1": bool(A["vs_R"]["d"] >= 0.010 and A["vs_R"]["p"] < 0.05 and A["vs_R"]["npos"] >= 5),
                               "S2": bool(A["vs_D"]["d"] > 0 and A["vs_D"]["p"] < 0.05 and A["vs_D"]["npos"] >= 5), "S3": bool(A["vs_cat3"]["d"] >= 0)}
            search["verdict"] = verdict_search(search["gates"])
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    out = {"rank_verdict": verdict_rank(gates_rank) if gates_rank else "INCOMPLETE", "gates_rank": gates_rank, "c0": {k: list(v) for k, v in c0.items()},
           "refs": {k: float(np.mean(v)) for k, v in refs.items()}, "rank": {k: strip(v) for k, v in rank.items()}, "search": search}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


def _f(x, nd=3):
    try:
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-C 读数：排序 {o['rank_verdict']} ｜ 搜索 {(o.get('search') or {}).get('verdict', '—')}", "",
         "C0：" + "；".join(f"{k} {_f(a, 5)} vs {_f(b, 5)}" for k, (a, b) in o["c0"].items()), "",
         "参照（56 留出任务 regret@3）：" + "；".join(f"{k} {_f(v)}" for k, v in o["refs"].items()), "",
         "| 臂 | regret@1 | regret@3 | vs 任务盲 [CI] p | vs ④ [CI] p | vs field-E [CI] p | 换源（16 次）Δ [CI] p / 同号源 | random-20 中位 Sp | 余弦 集内 / 跨集 |",
         "|---|---:|---:|---|---|---|---|---:|---|"]
    for k, r in o["rank"].items():
        cr = f"{_f(r['cross'])} [{_f(r['cross_ci'][0])}, {_f(r['cross_ci'][1])}] {_f(r['p_cross'], 3)} / {r['cross_npos']}" if "cross" in r else "—"
        cs = f"{_f(r['cos_within'], 2)} / {_f(r['cos_between'], 2)}" if "cos_within" in r else "—"
        L.append(f"| {k} | {_f(r['regret1'])} | {_f(r['regret3'])} | {_f(r['vs_blind'])} [{_f(r['vs_blind_ci'][0])}, {_f(r['vs_blind_ci'][1])}] {_f(r['p_vs_blind'], 3)} | "
                 f"{_f(r['vs_probe'])} [{_f(r['vs_probe_ci'][0])}, {_f(r['vs_probe_ci'][1])}] {_f(r['p_vs_probe'], 3)} | "
                 f"{_f(r['vs_fieldE'])} [{_f(r['vs_fieldE_ci'][0])}, {_f(r['vs_fieldE_ci'][1])}] {_f(r['p_vs_fieldE'], 3)} | {cr} | {_f(r['sp_median'], 2)} | {cs} |")
    L += ["", "排序门（主臂 A）：" + (", ".join(f"{g} {'✓' if b else '✗'}" for g, b in o["gates_rank"].items()) if o["gates_rank"] else "—")]
    s = o.get("search")
    if s:
        L += ["", "## 源侧搜索验证（24 任务；seed 0 选、seed 1 评）", "",
              f"目录任务盲前 3 {_f(s['cat3'])}；目录 oracle {_f(s['oracle'])}；缺值任务 {s['missing_tasks']}", "",
              "| 臂 | 最终 acc | vs 随机 R [CI] p / 同号源 | vs 任务盲比较器 D | vs 目录任务盲前 3 | vs 目录 oracle | 超 oracle 份额 |", "|---|---:|---|---|---|---|---:|"]
        for a, c in s["cmp"].items():
            fmt = lambda x: "—" if not x else f"{_f(x['d'])} [{_f(x['ci'][0])}, {_f(x['ci'][1])}] {_f(x['p'], 3)} / {x['npos']}"
            L.append(f"| {a} | {_f(s['final'][a])} | {fmt(c['vs_R'])} | {fmt(c['vs_D'])} | {fmt(c['vs_cat3'])} | {fmt(c['vs_oracle'])} | {_f(s['beat_oracle_frac'][a], 2)} |")
        L += ["", f"随机对照 R 最终 acc {_f(s['final']['R'])}", "", "搜索门（主臂 A）：" + ", ".join(f"{g} {'✓' if b else '✗'}" for g, b in s.get("gates", {}).items())]
    return "\n".join(L) + "\n"


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    import types
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 2, "resume": False, "ks": 2000, "gens": 2})
    for arm in ARMS:
        for fold in FOLDS[:2]:
            cmd_train(types.SimpleNamespace(**{**base, "arm": arm, "fold": fold}))
    cmd_search(types.SimpleNamespace(**{**base, "arm": "A", "fold": FOLDS[0]}))
    cmd_search(types.SimpleNamespace(**{**base, "arm": "C", "fold": FOLDS[0]}))
    r = json.loads((Path(args.outdir) / "search" / f"A_{FOLDS[0]}.json").read_text(encoding="utf-8"))
    ac = importlib.import_module("scripts.acts_common")
    from autocls.space.cls_space import CLS
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg)
    for t, e in r.items():
        assert len(e["nominated"]) == 3
        for x in e["nominated"]:
            c = CLS(**x["cls"]); assert space.validate(c) and not ac.is_pruned(c, cfg)
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "train", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm", choices=ARMS, default="A")
    ap.add_argument("--fold", default=FOLDS[0])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="冒烟：合成窗口")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.prep:
        cmd_prep(args)
    if args.smoke:
        cmd_smoke(args)
    if args.train:
        cmd_train(args)
    if args.search:
        cmd_search(args)
    if args.plan_runs:
        cmd_plan_runs(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
