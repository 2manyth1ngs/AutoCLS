"""WP17-ACTS S3-V（results/acts/s3v/PROTOCOL.md）：部署口径对表——AutoCTS++ 式消融 + 手工模型门。

6 个 UEA 集官方 train / test。各臂 = 该数据集留出那一折的 3 种子比较器（概率取均值）→ 全空间搜索前 3
→ 官方 TRAIN 内分层 20% D_val 选 1（= AutoCTS++ Algorithm 2）→ 完整 TRAIN 种子 0 / 1 / 2 的 TEST 均值。
阶段：
  --prep          本地：尺寸 / D_val 规模 / tasks_meta → meta.json
  --prep-raw      服务器（CPU）：源任务与 6 个靶的原始窗口（与 E_pt 同一批 100 窗，时间池化 ≤ 152）→ raw/<task>.npz
  --search        服务器：--arm <FULL|WO_SETTF|WO_TS2VEC|WO_SHARED|BLIND> --dataset <D> → search/<arm>_<D>.json
  --plan val      服务器：各臂前 3 × D_val（种子 0）→ runs/plan_W.json
  --select        服务器 / 本地：D_val 准确率选 1 → select.json
  --plan final    服务器：并入完整 TRAIN 的 TEST 格（前 3 种子 0 + 选中者种子 1 / 2）
  --analyze       本地：门、消融表、读数码
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

from autocls.comparator.raw_mlp_encoder import RawMLPSetEncoder  # noqa: E402,F401  （单测经本模块取用）

OUT = ROOT / "results" / "acts" / "s3v"
S3CD = ROOT / "results" / "acts" / "s3cd"
DATASETS = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1")
ARMS = {
    "FULL": {"dir": "results/acts/s3rp", "arm": "E", "e_mode": "settf", "input": "E"},
    "WO_SETTF": {"dir": "results/acts/s3v/abl_mean", "arm": "E", "e_mode": "mean", "input": "E"},
    "WO_TS2VEC": {"dir": "results/acts/s3v/abl_raw", "arm": "E", "e_mode": "raw", "input": "raw"},
    "WO_SHARED": {"dir": "results/acts/s3v/abl_noshared", "arm": "E", "e_mode": "settf", "input": "E"},
    "BLIND": {"dir": "results/acts/s3rp", "arm": "B", "e_mode": "settf", "input": None},
}
ABL = ("WO_SETTF", "WO_TS2VEC", "WO_SHARED")
SEEDS = (0, 1, 2)
VAL_FRAC = 0.2
TS2VEC = {"Handwriting": 0.515, "PhonemeSpectra": 0.233, "UWaveGestureLibrary": 0.906, "Cricket": 0.972,
          "EthanolConcentration": 0.308, "SelfRegulationSCP1": 0.812}                     # TS2Vec 原文 UEA 表
INFOTS_S = {"Handwriting": 0.554, "PhonemeSpectra": 0.233, "UWaveGestureLibrary": 0.884, "Cricket": 1.000,
            "EthanolConcentration": 0.323, "SelfRegulationSCP1": 0.887}                    # InfoTS 原文 UEA 表（InfoTS_s 列）


def off_tid(d: str) -> str:
    return f"{d}_whole_official"                     # = S3-C-D task_id（E 窗口、Ks / 进化种子同名）


def val_tid(d: str) -> str:
    return f"{d}_whole_val"


# ================================================================ 纯函数
def select_by_val(keys_ranked: Sequence[str], val: Mapping[str, float]) -> str:
    """D_val 准确率最高者；平局取比较器排名靠前者；缺格跳过。"""
    cand = [k for k in keys_ranked if k in val]
    best = max(val[k] for k in cand)
    return next(k for k in cand if val[k] >= best - 1e-12)


def verdict_s3v(full_mean: float, abl_means: Mapping[str, float]) -> str:
    ts, inf = float(np.mean(list(TS2VEC.values()))), float(np.mean(list(INFOTS_S.values())))
    if full_mean > inf and full_mean > ts:
        code = "S3V-BEATS-BOTH"
    elif full_mean > ts:
        code = "S3V-BEATS-TS2VEC"
    else:
        code = "S3V-BELOW"
    return code + ("+ABL" if all(full_mean > v for v in abl_means.values()) else "")


# ================================================================ prep
def cmd_prep(args) -> None:
    ac = importlib.import_module("scripts.acts_common"); g83 = importlib.import_module("scripts.83_acts_corpus")
    from autocls.labeling import run_seed_for
    sizes = g83._uea_sizes(list(DATASETS))
    tm = {}
    for d in DATASETS:
        sz = sizes[d]; X, y = ac.load_dataset_full(d)
        vs = run_seed_for(d, "s3v_val")
        a = ac.whole_arrays_val(X, y, n_train=int(sz["n_train"]), val_frac=VAL_FRAC, seed=vs)
        base = {"dataset": d, "T": int(sz["L"]), "C": int(sz["C"]), "n_train_official": int(sz["n_train"]), "n_test_official": int(sz["n_test"])}
        tm[off_tid(d)] = {**base, "n_train": int(sz["n_train"]), "eval_mode": "official", "tag": "whole_official"}
        tm[val_tid(d)] = {**base, "n_train": int(len(a["train_idx"])), "n_val": int(len(a["eval_idx"])), "eval_mode": "val",
                          "val_frac": VAL_FRAC, "val_seed": int(vs), "tag": "whole_val"}
        print(f"[prep] {d}: TRAIN {sz['n_train']} → D_train {len(a['train_idx'])} / D_val {len(a['eval_idx'])}", flush=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "meta.json").write_text(json.dumps({"datasets": list(DATASETS), "tasks_meta": tm}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_meta(outdir: Path) -> dict:
    return json.loads((Path(outdir) / "meta.json").read_text(encoding="utf-8"))


def cmd_prep_raw(args) -> None:
    """源任务：86.task_X 物化切片（按通道 z-score）取 E_pt 同一批 100 窗；靶：官方 TRAIN（TRAIN 统计量标准化）取 S3-C-D 同一批窗。"""
    ac = importlib.import_module("scripts.acts_common"); g86 = importlib.import_module("scripts.86_acts_s3b")
    from autocls.task_embed import pool_time_axis
    rd = Path(args.outdir) / "raw"; rd.mkdir(parents=True, exist_ok=True)
    tasks, tmeta, _s, _e = g86.task_meta()
    for tid in sorted(tasks, key=lambda t: tmeta[t]["dataset"]):
        p = rd / f"{tid}.npz"
        if p.exists() and args.resume:
            continue
        X = g86.task_X(tid, tmeta)
        r = pool_time_axis(X[g86.window_idx(X.shape[0], tid)].astype(np.float32), g86.T_MAX)
        np.savez_compressed(p, raw=r.astype(np.float16))
    meta = load_meta(Path(args.outdir))
    for d in DATASETS:
        p = rd / f"{off_tid(d)}.npz"
        if p.exists() and args.resume:
            continue
        m = meta["tasks_meta"][off_tid(d)]
        X, y = ac.load_dataset_full(d)
        Xtr = ac.whole_arrays_official(X, y, n_train=m["n_train_official"])["Xtr"]
        r = pool_time_axis(Xtr[g86.window_idx(Xtr.shape[0], off_tid(d))], g86.T_MAX)
        np.savez_compressed(p, raw=r.astype(np.float16))
    print(f"[prep-raw] {len(list(rd.glob('*.npz')))} 个窗口文件", flush=True)


# ================================================================ search（服务器）
def _ensemble(arm: str, d: str, dev: str, cls_dim: int, outdir: Path):
    import torch
    g89 = importlib.import_module("scripts.89_acts_s3c"); r105 = importlib.import_module("scripts.105_acts_s3r")
    A = ARMS[arm]; scs, Ts = [], []
    W = None
    if A["input"] == "E":
        with np.load(S3CD / "E" / f"{d}.npz") as z:
            W = z["windows_repr"]
    elif A["input"] == "raw":
        with np.load(outdir / "raw" / f"{off_tid(d)}.npz") as z:
            W = z["raw"]
    for s in SEEDS:
        m = r105._build(A["arm"], cls_dim, s, dev, A["e_mode"])
        m.load_state_dict(torch.load(ROOT / A["dir"] / "models" / f"{A['arm']}_s{s}_{d}.pt", map_location=dev)); m.eval()
        sc = g89.Scorer(m, "A" if A["arm"] == "E" else "D", dev)
        with torch.no_grad():
            Ts.append(None if W is None else sc.task_vec(W))
        scs.append(sc)
    return scs, Ts


def cmd_search(args) -> None:
    import torch
    ac = importlib.import_module("scripts.acts_common"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.labeling import run_seed_for
    from autocls.search.evolution import evolutionary_search
    from autocls.space.cls_encoder import CLSEncoder
    outdir = Path(args.outdir); d, arm, dev = args.dataset, args.arm, args.device
    op = outdir / "search" / f"{arm}_{d}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {arm}_{d}"); return
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    scs, Ts = _ensemble(arm, d, dev, enc.dim, outdir)
    tid = off_tid(d); t0 = time.time()
    rng = np.random.default_rng(run_seed_for(tid, "s3c_ks"))                                  # 与 S3-C / S3-C-D 同一批 Ks 候选
    cands = ac._sample_catalog(space, cfg, rng, args.ks or g89.KS, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
    top = g89.chunk_select(len(cands), lambda idx: np.mean([sc.pmat(Vc[np.asarray(idx)], T) for sc, T in zip(scs, Ts)], 0), keys)

    def score_pairs(a, b):
        with torch.no_grad():
            at = torch.as_tensor(np.asarray(a, np.float32), device=dev); bt = torch.as_tensor(np.asarray(b, np.float32), device=dev)
            ps = []
            for sc, T in zip(scs, Ts):
                Tb = None if T is None else T.unsqueeze(0).expand(len(at), -1)
                ps.append(sc.forward(at, bt, Tb).float().cpu().numpy())
            return np.mean(ps, 0)

    gens = args.gens if args.gens is not None else g89.GENS
    evo = evolutionary_search(score_pairs, space, enc, ks=0, kp=g89.KP, generations=gens, p_cross=g89.P_CROSS, p_mut=g89.P_MUT, top_k=3, batch_size=65536,
                              rng=np.random.default_rng(run_seed_for(tid, "s3c_evo")), seed_population=[cands[j][0] for j in top],
                              accept=lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg))
    res = {"nominated": [{"cls_key": e["cls_key"], "cls": e["cls"], "wins": e["wins"], "generation_found": e["generation_found"]} for e in evo],
           "ks": len(cands), "gens": gens, "seconds": time.time() - t0}
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[search] {arm}_{d} → {[e['cls_key'] for e in evo]} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 计划 / 选择
def _noms(outdir: Path, arm: str, d: str) -> list:
    return json.loads((outdir / "search" / f"{arm}_{d}.json").read_text(encoding="utf-8"))["nominated"]


def _rows(outdir: Path) -> dict:
    out = {}
    p = outdir / "runs" / "runs.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if not r.get("error") and r.get("checkpoint_results"):
                out[(r["task_id"], r["cls_key"], int(r.get("seed", 0)))] = float(r["checkpoint_results"][-1]["acc"])
    return out


def cmd_plan(args) -> None:
    ac = importlib.import_module("scripts.acts_common"); g35 = importlib.import_module("scripts.35_protocol_alignment")
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); meta = load_meta(outdir); tm = meta["tasks_meta"]
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols()
    ent = []                                            # (tid, key, cls, seed)
    arms = [a for a in ARMS if all((outdir / "search" / f"{a}_{d}.json").exists() for d in DATASETS)]
    for a in arms:
        for d in DATASETS:
            for e in _noms(outdir, a, d):
                ent.append((val_tid(d), e["cls_key"], e["cls"], 0))
    if args.plan == "final":
        sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
        for a in arms:
            for d in DATASETS:
                noms = {e["cls_key"]: e["cls"] for e in _noms(outdir, a, d)}
                ent += [(off_tid(d), k, c, 0) for k, c in noms.items()]
                k = sel[a][d]["pick"]
                ent += [(off_tid(d), k, noms[k], s) for s in SEEDS[1:]]
    cells, seen = [], set()
    for tid, k, c, s in ent:
        mode = tm[tid]["eval_mode"]
        cell = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**c), k, s, protos, cfg, extra={"eval": mode, "whole_mode": mode})
        if cell["key"] not in seen:
            seen.add(cell["key"]); cells.append(cell)
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    plan = {"phase": "W", "profile": "s1pp", "n_cells": len(cells), "n_tasks": len(tm), "probe_tasks": list(tm), "protocol": "ts2vec_official",
            "tasks_meta": tm, "cells": cells}
    (rd / "plan_W.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print(f"[plan {args.plan}] 臂 {arms}：{len(cells)} 格", flush=True)


def cmd_select(args) -> None:
    outdir = Path(args.outdir); rows = _rows(outdir); sel = {}
    for a in ARMS:
        if not all((outdir / "search" / f"{a}_{d}.json").exists() for d in DATASETS):
            continue
        sel[a] = {}
        for d in DATASETS:
            ks = [e["cls_key"] for e in _noms(outdir, a, d)]
            val = {k: rows[(val_tid(d), k, 0)] for k in ks if (val_tid(d), k, 0) in rows}
            sel[a][d] = {"pick": select_by_val(ks, val), "val": val, "ranked": ks}
    (outdir / "select.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[select] {json.dumps({a: {d[:6]: v['pick'][:8] for d, v in s.items()} for a, s in sel.items()}, ensure_ascii=False)}", flush=True)


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    outdir = Path(args.outdir); rows = _rows(outdir)
    sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
    res = {}
    for a, s in sel.items():
        res[a] = {}
        for d in DATASETS:
            k = s[d]["pick"]
            tests = [rows.get((off_tid(d), k, sd)) for sd in SEEDS]
            seed0 = {kk: rows.get((off_tid(d), kk, 0)) for kk in s[d]["ranked"]}
            best0 = max(v for v in seed0.values() if v is not None) if any(v is not None for v in seed0.values()) else None
            res[a][d] = {"pick": k, "test_seeds": tests, "test": float(np.mean([t for t in tests if t is not None])) if any(t is not None for t in tests) else None,
                         "val": s[d]["val"], "test_seed0_top3": seed0, "sel_loss_seed0": (best0 - tests[0]) if (best0 is not None and tests[0] is not None) else None}
    means = {a: float(np.mean([res[a][d]["test"] for d in DATASETS])) for a in res}
    ts, inf = float(np.mean(list(TS2VEC.values()))), float(np.mean(list(INFOTS_S.values())))
    full = means.get("FULL")
    o = {"per_arm": res, "means": means, "baselines": {"TS2Vec": ts, "InfoTS_s": inf},
         "wins_vs": {a: {"TS2Vec": int(sum(res[a][d]["test"] > TS2VEC[d] for d in DATASETS)),
                         "InfoTS_s": int(sum(res[a][d]["test"] > INFOTS_S[d] for d in DATASETS))} for a in res}}
    o["verdict"] = verdict_s3v(full, {a: means[a] for a in ABL if a in means})
    o["predictions"] = {"P1": full > ts, "P2": full > inf, "P3": o["verdict"].endswith("+ABL"),
                        "P4": full > means.get("BLIND", float("inf")),
                        "P5": min(ABL, key=lambda a: means.get(a, 9)) == "WO_SHARED", "P6": o["wins_vs"]["FULL"]["TS2Vec"] >= 4}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    order = [a for a in ("FULL", *ABL, "BLIND") if a in res]
    L = ["# S3-V 部署口径读数（官方 TEST，D_val 选 1，3 种子均值）", "", "| 数据集 | " + " | ".join(order) + " | TS2Vec | InfoTS_s |", "|---|" + "---:|" * (len(order) + 2)]
    for d in DATASETS:
        L.append(f"| {d} | " + " | ".join(f(res[a][d]['test']) for a in order) + f" | {TS2VEC[d]:.3f} | {INFOTS_S[d]:.3f} |")
    L.append("| **均值** | " + " | ".join(f"**{means[a]:.4f}**" for a in order) + f" | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；FULL 逐数据集胜 TS2Vec {o['wins_vs']['FULL']['TS2Vec']}/6、胜 InfoTS_s {o['wins_vs']['FULL']['InfoTS_s']}/6", "",
          "选择损失（种子 0：前 3 里 TEST 最好 − 选中）：" + "；".join(f"{a} {np.mean([res[a][d]['sel_loss_seed0'] for d in DATASETS if res[a][d]['sel_loss_seed0'] is not None]):+.3f}" for a in order)]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "prep_raw", "search", "select", "analyze", "resume", "smoke"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--plan", choices=("val", "final"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prep:
        cmd_prep(args)
    if args.prep_raw:
        cmd_prep_raw(args)
    if args.search:
        cmd_search(args)
    if args.plan:
        cmd_plan(args)
    if args.select:
        cmd_select(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
