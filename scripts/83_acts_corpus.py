"""WP17-ACTS S1 / S2 语料脚本：AutoCTS++ 式切片 → 2L 目录 → ts2vec_official 轨迹（39.execute）→ iters=0 探针地板 → 校验 → 读数。
协议 results/acts/s1/PROTOCOL.md；真源 configs/acts.yaml；公共层 scripts/acts_common.py。

子命令（按执行顺序）：
  --slice [--stage pilot|full]   切片（AutoCTSTierSlicer）→ tasks.json（零 GPU；含 16 任务前缀自检）
  --sanity                       试点 + 全设计切片 vs 29 官方靶的 (C, L, n_train, 类别数) 散点 + 覆盖表（零 GPU）
  --precheck                     §1.7 零 GPU 前置对照（冠军保留 / 剪掉与改写份额 / 保真度参照 / 旧构造对半分）→ precheck.{json,md}
  --plan                         shared-20 + random-20 + 锚 → cells → plan.json / catalog.json（含成本估计）
  --run [--parts N --part i]     39.execute（--resume；loader = 任务 id → 切片数组）
  --probe0                       每任务 3 个 iters=0 探针 → probe0.jsonl（--resume）
  --verify                       plan vs runs / probe0 对账
  --smoke                        本地冒烟（HMD 2 任务 × 3 条 × 8 iters + 探针 1 个）→ results/acts/s1/smoke/
  --analyze                      标签 / G-A 读数 / 保真度 / 相似度 / 读数码 → analysis.json / tables.md / labels.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from scipy.stats import spearmanr  # noqa: E402

from autocls.data.acts_slicer import AutoCTSTierSlicer  # noqa: E402
from autocls.data.task_slicer import SlicedTask  # noqa: E402
from autocls.labeling import run_seed_for  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import acts_common as ac  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g39 = importlib.import_module("scripts.39_sh_deploy")
_g45 = importlib.import_module("scripts.45_uea30_deploy")

OUT = ac.OUT_ROOT / "s1"
OFFICIAL_RUNS = (ROOT / "results" / "mx4_deploy" / "runs.jsonl", ROOT / "results" / "mx4_mask" / "runs.jsonl",
                 ROOT / "results" / "uea30" / "runs.jsonl")
UCR_CACHE = ROOT / "results" / "ucr_rprime" / "rprime_cache.jsonl"
MX3_CACHE = ROOT / "results" / "mx3_pq" / "rprime_cache.jsonl"
SMOKE_DATASET = "HandMovementDirection"
SMOKE_ITERS = 8
TASK_FIELDS = [f.name for f in dataclasses.fields(SlicedTask)]


def _install_win_timeout() -> None:
    if sys.platform.startswith("win"):
        importlib.import_module("scripts.43_bo_nominate").install_windows_eval_timeout()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"不可序列化: {type(o)}")


def _f(x, nd: int = 3) -> str:
    try:
        return "nan" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------- 切片
def slice_dataset(cfg: Mapping, name: str, n_tasks: int, *, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> tuple[list[SlicedTask], dict[str, dict]]:
    """切 n_tasks 个任务并算每任务元数据（n_train / n_eval / T / C / majority / iters）。"""
    if X is None or y is None:
        X, y = ac.load_dataset_full(name)
    slicer = AutoCTSTierSlicer.from_config(cfg["slicing"])
    tasks = slicer.slice(y, n_channels=int(X.shape[2]), seq_len=int(X.shape[1]), n_tasks=int(n_tasks), pool_name=name)
    proto = _g35.load_protocols()[cfg["protocol"]]
    meta: dict[str, dict] = {}
    for t in tasks:
        arr = ac.task_arrays(t, X, y, eval_ratio=float(cfg["slicing"]["eval_ratio"]), seed=ac.split_seed_for(t.task_id))
        n_train, T, C = (int(v) for v in arr["Xtr"].shape)
        meta[t.task_id] = {**dataclasses.asdict(t), "dataset": name, "n_train": n_train, "n_eval": int(len(arr["yte"])),
                           "T": T, "C": C, "k": int(len(t.classes)), "majority": float(arr["majority"]),
                           "iters": int(_g35.resolve_iters_from_size(proto, n_train * T * C)),
                           "L_full": int(X.shape[1]), "C_full": int(X.shape[2]), "N_full": int(len(y))}
    return tasks, meta


def cmd_slice(cfg: Mapping, outdir: Path, stage: str) -> dict:
    block = cfg[stage]
    n_tasks = int(block["n_tasks"])
    all_meta: dict[str, dict] = {}
    prefix_ok = True
    for name in block["datasets"]:
        X, y = ac.load_dataset_full(name)
        tasks, meta = slice_dataset(cfg, name, n_tasks, X=X, y=y)
        more, _m2 = slice_dataset(cfg, name, max(16, n_tasks), X=X, y=y)   # 前缀自检：S2 扩到 16 时试点任务逐字段不变
        prefix_ok &= more[: len(tasks)] == tasks
        all_meta.update(meta)
        for t in tasks:
            m = meta[t.task_id]
            print(f"[slice] {t.task_id} {t.tag} C={m['C']}/{m['C_full']} win={t.time_window} T={m['T']}/{m['L_full']} "
                  f"n={len(t.sample_indices)}/{m['N_full']} n_train={m['n_train']} k={m['k']} maj={m['majority']:.3f} iters={m['iters']}", flush=True)
    payload = {"stage": stage, "datasets": list(block["datasets"]), "n_tasks": n_tasks, "protocol": cfg["protocol"],
               "slicing": dict(cfg["slicing"]), "prefix_stable_16": bool(prefix_ok), "tasks": all_meta}
    _write_json(outdir / "tasks.json", payload)
    print(f"[slice] {len(all_meta)} 任务 → {outdir / 'tasks.json'}；16 任务前缀自检 {'OK' if prefix_ok else 'FAILED'}", flush=True)
    return payload


def load_tasks(outdir: Path) -> dict:
    return json.loads((outdir / "tasks.json").read_text(encoding="utf-8"))


def tasks_from_meta(meta: Mapping[str, Mapping]) -> dict[str, SlicedTask]:
    out = {}
    for tid, m in meta.items():
        kw = {k: m[k] for k in TASK_FIELDS if k in m}
        for k in ("classes", "channels", "sample_indices"):
            kw[k] = tuple(int(v) for v in kw[k])
        if kw.get("time_window") is not None:
            kw["time_window"] = tuple(int(v) for v in kw["time_window"])
        out[tid] = SlicedTask(**kw)
    return out


# ---------------------------------------------------------------- sanity（PROTOCOL §1.7-5）
def _subset_points(meta: Mapping[str, Mapping]) -> list[dict]:
    return [{"id": tid, "C": m["C"], "L": m["T"], "n": m["n_train"], "k": m["k"], "dataset": m["dataset"]} for tid, m in meta.items()]


def cmd_sanity(cfg: Mapping, outdir: Path) -> dict:
    pilot = load_tasks(outdir)["tasks"] if (outdir / "tasks.json").exists() else None
    if pilot is None:
        pilot = cmd_slice(cfg, outdir, "pilot")["tasks"]
    full_meta: dict[str, dict] = {}
    skipped = []
    for name in cfg["full"]["datasets"]:
        try:
            _t, m = slice_dataset(cfg, name, int(cfg["full"]["n_tasks"]))
            full_meta.update(m)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{name}: {type(e).__name__}: {e}")
    targets = ac.target_sizes_29()
    pts_pilot, pts_full = _subset_points(pilot), _subset_points(full_meta)
    table = ac.coverage_table(pts_pilot + pts_full, targets)
    table_pilot = ac.coverage_table(pts_pilot, targets)
    payload = {"pilot": pts_pilot, "full": pts_full, "targets": targets, "coverage_all": table, "coverage_pilot": table_pilot, "skipped": skipped}
    _write_json(outdir / "slice_sanity.json", payload)
    md = sanity_markdown(payload)
    (outdir / "slice_sanity.md").write_text(md, encoding="utf-8")
    png = plot_sanity(payload, outdir / "slice_sanity.png")
    print(md, flush=True)
    print(f"[sanity] → {png}", flush=True)
    return payload


def sanity_markdown(p: Mapping) -> str:
    t, tp = p["coverage_all"], p["coverage_pilot"]
    lines = ["# 切片分布 sanity（PROTOCOL §1.7-5）：子集云 vs 29 官方靶的 (C, L, n_train, 类别数)", "",
             f"子集 = 试点 {len(p['pilot'])} + 全设计 {len(p['full'])}（8 集 × 16，零 GPU）；靶 = {t['summary']['n_targets']}；"
             f"云外靶（任一轴）{t['summary']['outside_any']}，逐轴 {t['summary']['outside_by_axis']}；只看试点：云外 {tp['summary']['outside_any']}，逐轴 {tp['summary']['outside_by_axis']}。",
             "", "| 轴 | 子集云 [min, max]（试点+全设计） | 试点 [min, max] |", "|---|---|---|"]
    for a in ac.SANITY_AXES:
        lines.append(f"| {a} | {t['cloud_range'][a][0]:g} – {t['cloud_range'][a][1]:g} | {tp['cloud_range'][a][0]:g} – {tp['cloud_range'][a][1]:g} |")
    lines += ["", "| 靶 | 类 | C | L | n_train | k | 云内(C/L/n/k) | 最近子集 log 距 |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for name, r in sorted(t["per_target"].items(), key=lambda kv: -kv[1]["nearest_log_dist"]):
        v, ins = r["values"], r["inside"]
        flag = "".join("✓" if ins[a] else "✗" for a in ac.SANITY_AXES)
        lines.append(f"| {name} | {p['targets'][name]['class']} | {v['C']:g} | {v['L']:g} | {v['n']:g} | {v['k']:g} | {flag} | {r['nearest_log_dist']:.2f} |")
    if p.get("skipped"):
        lines += ["", "跳过的全设计数据集：" + "; ".join(p["skipped"])]
    return "\n".join(lines) + "\n"


def plot_sanity(p: Mapping, path: Path) -> Path:
    """2 × 2 小多图（对数轴）：靶（橙，菱形）/ 试点子集（蓝，实心）/ 全设计子集（青，空心）；云外靶直接标名。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3"
    c_target, c_pilot, c_full = "#eb6834", "#2a78d6", "#1baf7a"      # dataviz 参考调色板前三槽（all-pairs 校验通过）
    panels = (("C", "L"), ("n", "k"), ("L", "n"), ("C", "k"))
    names = {"C": "channels C", "L": "length L (window)", "n": "n_train", "k": "classes k"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor=surface)
    per = p["coverage_all"]["per_target"]
    handles = None
    for ax, (xa, ya) in zip(axes.ravel(), panels):
        outside = {n for n, r in per.items() if not (r["inside"][xa] and r["inside"][ya])}   # 只标在本面板两轴上云外的靶
        ax.set_facecolor(surface)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        fx = [d[xa] for d in p["full"]]
        fy = [d[ya] for d in p["full"]]
        ax.scatter(fx, fy, s=36, facecolors="none", edgecolors=c_full, linewidths=1.2, label=f"full-design subsets (n={len(p['full'])})", zorder=2)
        px = [d[xa] for d in p["pilot"]]
        py = [d[ya] for d in p["pilot"]]
        ax.scatter(px, py, s=36, color=c_pilot, edgecolors=surface, linewidths=1.0, label=f"pilot subsets (n={len(p['pilot'])})", zorder=3)
        tx = [t[xa] for t in p["targets"].values()]
        ty = [t[ya] for t in p["targets"].values()]
        ax.scatter(tx, ty, s=48, marker="D", color=c_target, edgecolors=surface, linewidths=1.2, label=f"official targets (n={len(p['targets'])})", zorder=4)
        for j, (name, t) in enumerate(sorted(p["targets"].items(), key=lambda kv: (kv[1][xa], kv[1][ya]))):
            if name in outside:
                dy = 3 if j % 2 == 0 else -9          # 交替上下偏移，减少相邻标签重叠
                ax.annotate(name, (t[xa], t[ya]), xytext=(4, dy), textcoords="offset points", fontsize=7, color=ink2, zorder=5)
        handles = handles or ax.get_legend_handles_labels()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(names[xa], color=ink)
        ax.set_ylabel(names[ya], color=ink)
        ax.tick_params(colors=ink2, labelsize=8)
    s = p["coverage_all"]["summary"]
    fig.suptitle("WP17-ACTS S1 slice sanity: do the subsets cover the targets?", color=ink, fontsize=13, x=0.02, y=0.985, ha="left")
    fig.text(0.02, 0.955, f"Subsets = pilot + full design (8 sources × 16 tiers); targets = 29 official UEA sets (n = official n_train). "
                          f"Targets outside the subset cloud on any axis: {s['outside_any']}/{s['n_targets']}",
             color=ink2, fontsize=8.5)
    fig.text(0.02, 0.937, f"(C {s['outside_by_axis']['C']}, L {s['outside_by_axis']['L']}, n {s['outside_by_axis']['n']}, k {s['outside_by_axis']['k']}). "
                          f"A target is labeled in a panel when it lies outside the cloud on that panel's axes.", color=ink2, fontsize=8.5)
    fig.legend(*handles, loc="upper left", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.015, 0.93))
    fig.tight_layout(rect=(0, 0, 1, 0.895))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- precheck（PROTOCOL §1.7）
def official_rows() -> list[dict]:
    rows: list[dict] = []
    for p in OFFICIAL_RUNS:
        if p.exists():
            rows += read_jsonl(p)
    return rows


def fidelity_from_official(rows: Sequence[Mapping]) -> dict:
    traj: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    augmax: dict[str, float] = {}
    for r in rows:
        if r.get("stage") != "s1_catalog" or int(r.get("seed", 0)) != 0 or r.get("error") or not r.get("checkpoint_results"):
            continue
        ck = r["checkpoint_results"]
        if len(ck) < 3:
            continue
        traj[r["dataset"]][r["cls_key"]] = (float(ck[0]["acc"]), float(ck[-1]["acc"]))
        augmax[r["cls_key"]] = ac.aug_max(CLS(**r["cls"]))
    return ac.fidelity_metrics(traj, augmax)


def fidelity_from_ucr(path: Path = UCR_CACHE) -> Optional[dict]:
    if not path.exists():
        return None
    by: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    augmax: dict[str, float] = {}
    for r in read_jsonl(path):
        if r.get("error") or r.get("r_prime") is None or int(r.get("seed", 0)) != 0:
            continue
        by[(r["task_id"], r["cls_key"])][int(r["iters"])] = float(r["r_prime"])
        augmax[r["cls_key"]] = ac.aug_max(CLS(**r["cls"]))
    traj: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (tid, k), d in by.items():
        if d:
            lo, hi = min(d), max(d)
            if lo != hi:
                traj[tid][k] = (d[lo], d[hi])
    return ac.fidelity_metrics(traj, augmax)


def old_split_half(cfg: Mapping, path: Path = MX3_CACHE) -> Optional[dict]:
    """旧构造（mx3_pq：每集 6 切片、shared-10 + random-10、60/120/290）按 §1.5 同一定义算对半分（half 3）。"""
    if not path.exists():
        return None
    g = cfg["gates"]["G-A1"]
    best: dict[tuple[str, str], float] = {}
    origin: dict[str, str] = {}
    for r in read_jsonl(path):
        if r.get("error") or r.get("r_prime") is None or int(r.get("seed", 0)) != 0 or r.get("origin") not in ("shared", "random"):
            continue
        key = (r["task_id"], r["cls_key"])
        best[key] = max(best.get(key, -np.inf), float(r["r_prime"]))
        origin[r["cls_key"]] = r["origin"]
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for (tid, k), v in best.items():
        by_task[tid][k] = v
    shared_keys = sorted(k for k, o in origin.items() if o == "shared")
    by_ds: dict[str, list[str]] = defaultdict(list)
    for tid in by_task:
        by_ds[tid.rsplit("_", 1)[0]].append(tid)
    out = {}
    for d, tids in sorted(by_ds.items()):
        rows = []
        for tid in sorted(tids):
            vals = by_task[tid]
            if not all(k in vals for k in shared_keys):
                continue
            allv = np.array(list(vals.values()), dtype=float)
            mu, sd = allv.mean(), allv.std() or 1.0
            rows.append([(vals[k] - mu) / sd for k in shared_keys])
        Z = np.array(rows)
        if Z.shape[0] >= 2 * int(g["half_size"]):
            out[d] = {"n_tasks": int(Z.shape[0]), "split_half_3": ac.split_half_rho(Z, half_size=int(g["half_size"]), n_draws=int(g["n_draws"]), seed=0)}
    return out


def cmd_precheck(cfg: Mapping, outdir: Path) -> dict:
    rows = official_rows()
    ret = ac.champion_retention(rows, cfg)
    dspace = CLSSpace.from_yaml()
    share = ac.pruned_share(dspace, cfg, n=20000, seed=0)
    norm = ac.normalization_stats(cfg, n=20000, seed=0)
    fid_off = fidelity_from_official(rows)
    fid_ucr = fidelity_from_ucr()
    old = old_split_half(cfg)
    ga0 = bool(ret["rate"] >= float(cfg["prune"]["champion_retention_min"]))
    payload = {"champion_retention": ret, "G-A0": ga0, "pruned_share_uniform": share, "normalization": norm,
               "fidelity_official29": fid_off, "fidelity_ucr84": fid_ucr, "old_split_half_mx3": old}
    _write_json(outdir / "precheck.json", payload)
    md = precheck_markdown(payload, cfg)
    (outdir / "precheck.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return payload


def _fid_row(name: str, m: Optional[Mapping]) -> str:
    if not m:
        return f"| {name} | — | — | — | — | — | — |"
    rb = m["residual_by_augmax_mean"]
    return (f"| {name} | {m['n_tasks']} | {_f(m['rho_median'])} [{_f(m['rho_q25'])}, {_f(m['rho_q75'])}] | {_f(m['recall3_mean'])} | "
            f"{_f(rb.get('le0.5'))} | {_f(rb.get('0.5-0.8'))} | {_f(rb.get('gt0.8'))} |")


def precheck_markdown(p: Mapping, cfg: Mapping) -> str:
    ret = p["champion_retention"]
    lines = ["# 零 GPU 前置对照（PROTOCOL §1.7）", "",
             f"- G-A0 冠军保留率：{ret['kept']}/{ret['n']} = {_f(ret['rate'])}（门 ≥ {cfg['prune']['champion_retention_min']}）→ {'✓' if p['G-A0'] else '✗'}；"
             f"丢失：{', '.join(f'{d} {k}' for d, k in ret['lost']) or '无'}",
             f"- 均匀采样剪掉份额：{_f(p['pruned_share_uniform'])}；kernel=0 规范化改写份额：{_f(p['normalization']['rewritten_share'])}，"
             f"空间规范形计数比 acts / 默认：{_f(p['normalization']['space_size_ratio'])}（= 39 / 49）",
             "", "## 保真度参照带（§3.6-5）", "",
             "| 语料 | 任务数 | ρ(早, 终) 中位 [q25, q75] | recall@3 | 残差 aug_max ≤ 0.5 | (0.5, 0.8] | > 0.8 |", "|---|---:|---|---:|---:|---:|---:|",
             _fid_row("官方 29 × 目录（25% → 100%）", p["fidelity_official29"]),
             _fid_row("UCR 84 × 20（60 → 290）", p["fidelity_ucr84"]), ""]
    if p.get("old_split_half_mx3"):
        lines += ["## 旧构造对半分（mx3_pq，6 切片 / shared-10 / best-of-3，half 3，§1.5 同一定义）", "",
                  "| 数据集 | 切片数 | 对半分 ρ |", "|---|---:|---:|"]
        for d, r in p["old_split_half_mx3"].items():
            lines.append(f"| {d} | {r['n_tasks']} | {_f(r['split_half_3'])} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- plan（PROTOCOL §1.2 / §1.4）
def build_plan(cfg: Mapping, tasks_payload: Mapping, *, n_shared: Optional[int] = None, n_random: Optional[int] = None,
               iters_override: Optional[int] = None, anchor_aug_max_le: Optional[float] = None) -> tuple[dict, dict]:
    space = ac.acts_space(cfg)
    protos = _g35.load_protocols()
    meta = tasks_payload["tasks"]
    shared = ac.shared_catalog(space, cfg, n=n_shared)
    excl = {k for _c, k in shared}
    randoms = {tid: ac.random_catalog(space, cfg, tid, exclude_keys=excl, n=n_random) for tid in meta}
    anc = cfg["labels"]["anchors"]
    thr = float(str(anc["rule"]).rsplit("_", 1)[-1]) if str(anc["rule"]).startswith("first_shared_aug_max_le_") else 0.8
    if anchor_aug_max_le is not None:
        thr = float(anchor_aug_max_le)          # 冒烟：shared-2 里未必有 aug_max ≤ 0.8 的，放宽只为走通锚格路径
    anchors = ac.pick_anchors(shared, n=int(anc["n"]), aug_max_le=thr)
    cells = ac.build_cells(meta, shared, randoms, anchors, protos, cfg, iters_override=iters_override)
    est = float(sum(ac.estimate_cell_seconds(c, cfg) for c in cells))
    catalog = {"shared": [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c)} for c, k in shared],
               "random": {tid: [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c)} for c, k in r] for tid, r in randoms.items()},
               "anchors": [k for _c, k in anchors], "anchor_seeds": list(anc["seeds"])}
    plan = {"n_cells": len(cells), "n_tasks": len(meta), "protocol": cfg["protocol"], "stage": tasks_payload.get("stage"),
            "datasets": list(tasks_payload.get("datasets", [])), "est_seconds_4090": est,
            "est_hours_5090": est * float(cfg["cost_model"]["gpu_factor_5090"]) / 3600.0,
            "tasks_meta": {tid: {k: m[k] for k in ("dataset", "n_train", "n_eval", "T", "C", "k", "majority", "iters", "tag", "time_window")} for tid, m in meta.items()},
            "cells": cells}
    return plan, catalog


def cmd_plan(cfg: Mapping, outdir: Path) -> dict:
    plan, catalog = build_plan(cfg, load_tasks(outdir))
    _write_json(outdir / "plan.json", plan)
    _write_json(outdir / "catalog.json", catalog)
    from collections import Counter
    by_iters = Counter(c["iters"] for c in plan["cells"])
    by_arm = Counter(c["arm"] for c in plan["cells"])
    print(f"[plan] {plan['n_cells']} 格 = {plan['n_tasks']} 任务 × ({by_arm})；iters {dict(by_iters)}；"
          f"估时 4090 {plan['est_seconds_4090'] / 3600:.1f} h / 5090 ≈ {plan['est_hours_5090']:.1f} h；锚 {catalog['anchors']}", flush=True)
    return plan


def load_plan(outdir: Path) -> dict:
    return json.loads((outdir / "plan.json").read_text(encoding="utf-8"))


def split_parts(cells: Sequence[Mapping], n_parts: int, cfg: Mapping) -> list[list[str]]:
    """按任务估时贪心分 n_parts 份（同一任务的格不拆，loader 单槽缓存友好）。返回每份的任务 id 列表。"""
    est: dict[str, float] = defaultdict(float)
    for c in cells:
        est[c["task_id"]] += ac.estimate_cell_seconds(c, cfg)
    parts: list[list[str]] = [[] for _ in range(int(n_parts))]
    load = [0.0] * int(n_parts)
    for tid in sorted(est, key=lambda t: -est[t]):
        i = int(np.argmin(load))
        parts[i].append(tid)
        load[i] += est[tid]
    return parts


# ---------------------------------------------------------------- run / probe0
def _loader_for(cfg: Mapping, tasks_payload: Mapping):
    meta = tasks_payload["tasks"]
    tasks = tasks_from_meta(meta)
    dataset_of = {tid: m["dataset"] for tid, m in meta.items()}
    return ac.make_task_loader(tasks, dataset_of, ac.load_dataset_full, eval_ratio=float(cfg["slicing"]["eval_ratio"]))


def cmd_run(args, cfg: Mapping, outdir: Path) -> None:
    plan = load_plan(outdir)
    cells = plan["cells"]
    if getattr(args, "parts", 0):
        sel = set(split_parts(cells, int(args.parts), cfg)[int(args.part)])
        cells = [c for c in cells if c["task_id"] in sel]
    if args.tasks:
        sel = set(args.tasks)
        cells = [c for c in cells if c["task_id"] in sel]
    cells = _g45.drop_repeated_oom(cells, outdir / "oom_failures.jsonl")
    _install_win_timeout()
    ns = argparse.Namespace(outdir=str(outdir), resume=args.resume, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    _g39.execute(cells, ns, "ACTS-S1", loader=_loader_for(cfg, load_tasks(outdir)))


def run_probe0(task_id: str, arrays: Mapping, s: int, device: str) -> dict:
    """iters=0 探针：同口径随机初始化 encoder（seed = run_seed_for(task_id, "probe0:{s}")）编码 + 官方 SVM 网格。"""
    import torch
    from autocls.downstream.linear_eval import encode_features
    from autocls.encoders.registry import build_encoder

    seed = run_seed_for(task_id, f"probe0:{s}")
    torch.manual_seed(seed)
    enc = build_encoder(input_dims=int(arrays["Xtr"].shape[-1]))
    t0 = time.perf_counter()
    ftr = encode_features(enc, arrays["Xtr"], device=device)
    fte = encode_features(enc, arrays["Xte_train"], device=device)
    v = _g39.eval_with_timeout(
        lambda: _g35._svm_variants(ftr, arrays["ytr"], {"train": fte}, arrays["yte"], ("ts2vec",), seed)["train/ts2vec"], _g39.EVAL_TIMEOUT_S)
    return {"task_id": task_id, "probe": int(s), "weight_seed": int(seed), "iters": 0, **v,
            "majority": float(arrays["majority"]), "n_train": int(arrays["Xtr"].shape[0]), "n_test": int(len(arrays["yte"])),
            "seconds": round(time.perf_counter() - t0, 1)}


def load_probe0(outdir: Path) -> dict[tuple[str, int], dict]:
    p = outdir / "probe0.jsonl"
    return {(r["task_id"], int(r["probe"])): r for r in read_jsonl(p)} if p.exists() else {}


def cmd_probe0(args, cfg: Mapping, outdir: Path, *, n_seeds: Optional[int] = None, task_ids: Optional[Sequence[str]] = None) -> None:
    from autocls.labeling import append_jsonl

    payload = load_tasks(outdir)
    loader = _loader_for(cfg, payload)
    done = load_probe0(outdir) if args.resume else {}
    n = int(cfg["labels"]["probe0_seeds"] if n_seeds is None else n_seeds)
    tids = list(task_ids) if task_ids else list(payload["tasks"])
    _install_win_timeout()
    for tid in tids:
        for s in range(n):
            if (tid, s) in done:
                continue
            arrays = loader(tid)
            row = run_probe0(tid, arrays, s, args.device)
            append_jsonl(outdir / "probe0.jsonl", row)
            print(f"[probe0] {tid} s{s} acc={row['acc']:.3f} maj={row['majority']:.3f} ({row['seconds']}s)", flush=True)


# ---------------------------------------------------------------- verify
def cmd_verify(cfg: Mapping, outdir: Path) -> dict:
    plan = load_plan(outdir)
    rows = _g35.load_done(outdir / "runs.jsonl") if (outdir / "runs.jsonl").exists() else {}
    want = {c["key"]: c for c in plan["cells"]}
    missing = [k for k in want if k not in rows]
    err = [k for k, r in rows.items() if k in want and r.get("error")]
    bad = [k for k, r in rows.items() if k in want and not r.get("error") and len(r.get("checkpoint_results") or []) != len(want[k]["checkpoints"])]
    probe = load_probe0(outdir)
    n_probe = int(cfg["labels"]["probe0_seeds"])
    probe_missing = [(t, s) for t in plan["tasks_meta"] for s in range(n_probe) if (t, s) not in probe]
    v = {"planned": len(want), "have": len(want) - len(missing), "missing": len(missing), "error_rows": len(err), "bad": len(bad),
         "probe_planned": len(plan["tasks_meta"]) * n_probe, "probe_missing": len(probe_missing),
         "ok": not missing and not bad and not probe_missing, "missing_examples": missing[:5], "error_examples": err[:5]}
    print(f"[verify] planned {v['planned']} have {v['have']} missing {v['missing']} error {v['error_rows']} bad {v['bad']}; "
          f"probe0 {v['probe_planned'] - v['probe_missing']}/{v['probe_planned']} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- smoke
def cmd_smoke(args, cfg: Mapping, outdir: Path) -> bool:
    sm = outdir / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    X, y = ac.load_dataset_full(SMOKE_DATASET)
    _tasks, meta = slice_dataset(cfg, SMOKE_DATASET, 2, X=X, y=y)
    payload = {"stage": "smoke", "datasets": [SMOKE_DATASET], "n_tasks": 2, "protocol": cfg["protocol"], "slicing": dict(cfg["slicing"]), "tasks": meta}
    _write_json(sm / "tasks.json", payload)
    plan, catalog = build_plan(cfg, payload, n_shared=2, n_random=1, iters_override=SMOKE_ITERS, anchor_aug_max_le=1.0)
    _write_json(sm / "plan.json", plan)
    _write_json(sm / "catalog.json", catalog)
    _install_win_timeout()
    ns = argparse.Namespace(outdir=str(sm), resume=False, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    _g39.execute(plan["cells"], ns, "smoke", loader=_loader_for(cfg, payload))
    ns2 = argparse.Namespace(resume=False, device=args.device)
    cmd_probe0(ns2, cfg, sm, n_seeds=1)
    rows = list(_g35.load_done(sm / "runs.jsonl").values())
    probe = load_probe0(sm)
    ok = (len(rows) == plan["n_cells"] and all(r.get("error") is None and len(r.get("checkpoint_results") or []) == 3 for r in rows)
          and len(probe) == 2)
    print(f"smoke {'OK' if ok else 'FAILED'}: {len(rows)}/{plan['n_cells']} 格、probe0 {len(probe)}/2；"
          f"{[(r['task_id'], r['arm'], r['seed'], round(r['acc'], 3)) for r in rows]}", flush=True)
    return ok


# ---------------------------------------------------------------- analyze（PROTOCOL §1.5 / §2 / §3）
def floors_from_probe(probe: Mapping[tuple[str, int], Mapping], tasks_meta: Mapping[str, Mapping]) -> dict[str, dict]:
    out = {}
    for tid, m in tasks_meta.items():
        accs = [float(r["acc"]) for (t, _s), r in probe.items() if t == tid and r.get("acc") is not None]
        out[tid] = {"majority": float(m["majority"]), "probe_accs": accs, "floor": ac.floor_of(float(m["majority"]), accs)}
    return out


def build_labels(plan: Mapping, rows: Mapping[str, Mapping], floors: Mapping[str, Mapping]) -> list[dict]:
    labels = []
    for cell in plan["cells"]:
        r = rows.get(cell["key"])
        best = ac.best_of_checkpoints(r) if r else None
        rp, failed = ac.label_of(best, floors[cell["task_id"]]["floor"])
        labels.append({"task_id": cell["task_id"], "dataset": plan["tasks_meta"][cell["task_id"]]["dataset"], "cls_key": cell["cls_key"],
                       "arm": cell["arm"], "seed": int(cell["seed"]), "key": cell["key"], "best3": best, "r_prime": rp, "failed": bool(failed),
                       "missing": r is None, "error": (r.get("error") if r else "missing"),
                       "ckpt": ([float(c["acc"]) for c in r["checkpoint_results"]] if r and r.get("checkpoint_results") else None),
                       "pretrain_seconds": (r or {}).get("pretrain_seconds"), "eval_seconds": (r or {}).get("eval_seconds"),
                       "device_name": (r or {}).get("device_name")})
    return labels


def cmd_analyze(cfg: Mapping, outdir: Path) -> dict:
    plan = load_plan(outdir)
    catalog = json.loads((outdir / "catalog.json").read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl")
    probe = load_probe0(outdir)
    floors = floors_from_probe(probe, plan["tasks_meta"])
    labels = build_labels(plan, rows, floors)
    from autocls.labeling import append_jsonl_many
    (outdir / "labels.jsonl").unlink(missing_ok=True)
    append_jsonl_many(outdir / "labels.jsonl", labels)

    shared_keys = [s["cls_key"] for s in catalog["shared"]]
    anchor_keys = list(catalog["anchors"])
    g1, g2 = cfg["gates"]["G-A1"], cfg["gates"]["G-A2"]
    tasks_by_ds: dict[str, list[str]] = defaultdict(list)
    for tid, m in plan["tasks_meta"].items():
        tasks_by_ds[m["dataset"]].append(tid)
    # within-task z（seed 0 的 shared + random）
    l0 = {(l["task_id"], l["cls_key"]): l for l in labels if l["seed"] == 0 and l["arm"] in (ac.ARM_SHARED, ac.ARM_RANDOM)}
    l1 = {(l["task_id"], l["cls_key"]): l for l in labels if l["arm"] == ac.ARM_ANCHOR}
    stats, z0 = {}, {}
    for tid in plan["tasks_meta"]:
        vals = np.array([l["r_prime"] for (t, _k), l in l0.items() if t == tid], dtype=float)
        mu, sd = (float(vals.mean()), float(vals.std())) if vals.size else (float("nan"), 0.0)
        stats[tid] = (mu, sd if sd > 0 else 1.0)
        for (t, k), l in l0.items():
            if t == tid:
                z0[(t, k)] = (l["r_prime"] - stats[tid][0]) / stats[tid][1]
    per_ds: dict[str, dict] = {}
    pooled_ss, pooled_df, noise_all = 0.0, 0, []
    for d, tids in sorted(tasks_by_ds.items()):
        Z = np.array([[z0.get((t, k), np.nan) for k in shared_keys] for t in tids], dtype=float)
        Z = np.where(np.isfinite(Z), Z, 0.0)
        a0 = np.array([[z0.get((t, k), np.nan) for k in anchor_keys] for t in tids], dtype=float)
        a1 = np.array([[((l1[(t, k)]["r_prime"] - stats[t][0]) / stats[t][1]) if (t, k) in l1 else np.nan for k in anchor_keys] for t in tids], dtype=float)
        noise = ac.anchor_noise_ms(a0, a1)
        noise_all += ((a0 - a1).ravel()[np.isfinite((a0 - a1).ravel())] ** 2 / 2.0).tolist()
        inter = ac.interaction_share(Z, noise_ms=noise)
        pooled_ss += inter["ss_int"]
        pooled_df += inter["df"]
        sh3 = ac.split_half_rho(Z, half_size=int(g1["half_size"]), n_draws=int(g1["n_draws"]), seed=0) if Z.shape[0] >= 2 * int(g1["half_size"]) else float("nan")
        sh4 = ac.split_half_rho(Z, half_size=4, n_draws=int(g1["n_draws"]), seed=0) if Z.shape[0] >= 8 else float("nan")
        fl = [l for l in labels if l["dataset"] == d and l["seed"] == 0]
        per_ds[d] = {"n_tasks": len(tids), "split_half_3": sh3, "split_half_4": sh4, **inter,
                     "failed_share_shared": float(np.mean([l["failed"] for l in fl if l["arm"] == ac.ARM_SHARED])) if fl else float("nan"),
                     "failed_share_random": float(np.mean([l["failed"] for l in fl if l["arm"] == ac.ARM_RANDOM])) if fl else float("nan"),
                     "missing": int(sum(l["missing"] for l in fl)), "errors": int(sum(1 for l in fl if l["error"] and not l["missing"]))}
    pooled_ms_int = pooled_ss / max(1, pooled_df)
    pooled_noise = float(np.mean(noise_all)) if noise_all else float("nan")
    pooled_share = (pooled_ms_int - pooled_noise) / pooled_ms_int if pooled_ms_int > 0 else float("nan")
    # 前置对照（旧构造 before 数、G-A0）
    pre = json.loads((outdir / "precheck.json").read_text(encoding="utf-8")) if (outdir / "precheck.json").exists() else {}
    before = {d: r["split_half_3"] for d, r in (pre.get("old_split_half_mx3") or {}).items()}
    ga0 = bool(pre.get("G-A0")) if pre else bool(ac.champion_retention(official_rows(), cfg)["rate"] >= float(cfg["prune"]["champion_retention_min"]))
    sh3_vals = [r["split_half_3"] for r in per_ds.values() if np.isfinite(r["split_half_3"])]
    mean_sh3 = float(np.mean(sh3_vals)) if sh3_vals else float("nan")
    below_before = all(per_ds[d]["split_half_3"] < before[d] for d in ("PhonemeSpectra", "Cricket") if d in per_ds and d in before)
    ga1 = bool(np.isfinite(mean_sh3) and mean_sh3 <= float(g1["split_half_rho_max"]) and below_before)
    n_pass = sum(1 for r in per_ds.values() if np.isfinite(r["real_share"]) and r["real_share"] >= float(g2["real_interaction_share_min"]))
    ga2 = bool(np.isfinite(pooled_share) and pooled_share >= float(g2["real_interaction_share_min"]) and n_pass >= int(g2["min_datasets_pass"]))
    gates = {"G-A0": ga0, "G-A1": ga1, "G-A2": ga2}
    verdict = ac.verdict_s1(gates)
    # 保真度（试点）
    traj: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    augmax = {s["cls_key"]: s["aug_max"] for s in catalog["shared"]}
    for r in catalog["random"].values():
        augmax.update({x["cls_key"]: x["aug_max"] for x in r})
    for l in labels:
        if l["seed"] == 0 and l["ckpt"] and len(l["ckpt"]) == 3:
            traj[l["task_id"]][l["cls_key"]] = (l["ckpt"][0], l["ckpt"][-1])
    fid = ac.fidelity_metrics(traj, augmax)
    # 任务相似度（shared-20 raw best-of-3 / 地板后 r_prime）
    tids_all = list(plan["tasks_meta"])
    M = np.array([[l0[(t, k)]["r_prime"] if (t, k) in l0 else np.nan for k in shared_keys] for t in tids_all], dtype=float)
    S = np.full((len(tids_all), len(tids_all)), np.nan)
    for i in range(len(tids_all)):
        for j in range(len(tids_all)):
            ok = np.isfinite(M[i]) & np.isfinite(M[j])
            S[i, j] = spearmanr(M[i][ok], M[j][ok]).correlation if ok.sum() >= 3 else np.nan
    ds_of = [plan["tasks_meta"][t]["dataset"] for t in tids_all]
    same = [S[i, j] for i in range(len(tids_all)) for j in range(i + 1, len(tids_all)) if ds_of[i] == ds_of[j] and np.isfinite(S[i, j])]
    cross = [S[i, j] for i in range(len(tids_all)) for j in range(i + 1, len(tids_all)) if ds_of[i] != ds_of[j] and np.isfinite(S[i, j])]
    sim = {"same_dataset_mean": float(np.mean(same)) if same else float("nan"), "cross_dataset_mean": float(np.mean(cross)) if cross else float("nan"),
           "n_same_pairs": len(same), "n_cross_pairs": len(cross)}
    # 成本
    sec = [float(l["pretrain_seconds"] or 0) + float(l["eval_seconds"] or 0) for l in labels if not l["missing"]]
    cost = {"n_rows": len(sec), "gpu_seconds": float(sum(sec)), "gpu_hours": float(sum(sec)) / 3600.0,
            "devices": sorted({str(l["device_name"]) for l in labels if l["device_name"]})}
    failed_random = float(np.mean([l["failed"] for l in labels if l["seed"] == 0 and l["arm"] == ac.ARM_RANDOM])) if labels else float("nan")
    order = sorted(per_ds, key=lambda d: per_ds[d]["split_half_3"] if np.isfinite(per_ds[d]["split_half_3"]) else np.inf)
    sanity = json.loads((outdir / "slice_sanity.json").read_text(encoding="utf-8")) if (outdir / "slice_sanity.json").exists() else None
    preds = {"P1": ga1, "P2": ga2,
             "P3": bool(order and order[0] == "HandMovementDirection" and order[-1] == "Cricket"),
             "P4": bool(np.isfinite(failed_random) and failed_random >= 0.25),
             "P5": bool(np.isfinite(fid["rho_median"]) and 0.57 <= fid["rho_median"] <= 0.85),
             "P6": None,
             "P7": (bool(sanity["coverage_all"]["summary"]["outside_by_axis"]["C"] >= 8) if sanity else None)}
    a = {"gates": gates, "verdict": verdict, "per_dataset": per_ds, "split_half_3_mean": mean_sh3, "split_half_before": before,
         "pooled": {"ms_int": pooled_ms_int, "ms_noise": pooled_noise, "real_share": pooled_share, "n_datasets_pass": n_pass},
         "floors": floors, "fidelity_pilot": fid, "fidelity_reference": {"official29": (pre.get("fidelity_official29") or {}).get("rho_median"),
                                                                        "ucr84": (pre.get("fidelity_ucr84") or {}).get("rho_median")},
         "similarity": sim, "failed_share_random_all": failed_random, "cost": cost, "predictions": preds,
         "n_labels": len(labels), "n_missing": int(sum(l["missing"] for l in labels)), "n_error": int(sum(1 for l in labels if l["error"] and not l["missing"]))}
    _write_json(outdir / "analysis.json", a)
    md = analysis_markdown(a, cfg)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return a


def analysis_markdown(a: Mapping, cfg: Mapping) -> str:
    g1, g2 = cfg["gates"]["G-A1"], cfg["gates"]["G-A2"]
    lines = [f"# WP17-ACTS S1 读数（{a['verdict']}）", "",
             f"标签 {a['n_labels']}（缺 {a['n_missing']}、error {a['n_error']}）；GPU {a['cost']['gpu_hours']:.2f} h（{', '.join(a['cost']['devices'])}）", "",
             "| 数据集 | 任务 | 对半分 ρ（half 3） | 旧构造 before | 对半分 ρ（half 4） | MS_int | MS_noise | 真实交互份额 | failed shared / random | 缺 / error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for d, r in a["per_dataset"].items():
        lines.append(f"| {d} | {r['n_tasks']} | {_f(r['split_half_3'])} | {_f(a['split_half_before'].get(d))} | {_f(r['split_half_4'])} | "
                     f"{_f(r['ms_int'])} | {_f(r['ms_noise'])} | {_f(r['real_share'])} | {_f(r['failed_share_shared'], 2)} / {_f(r['failed_share_random'], 2)} | {r['missing']} / {r['errors']} |")
    p = a["pooled"]
    lines += ["", f"合并：对半分 ρ（half 3）均值 {_f(a['split_half_3_mean'])}（门 ≤ {g1['split_half_rho_max']}）；MS_int {_f(p['ms_int'])} / MS_noise {_f(p['ms_noise'])} → "
                  f"真实交互份额 {_f(p['real_share'])}（门 ≥ {g2['real_interaction_share_min']}，逐集过门 {p['n_datasets_pass']}）", "",
              "| 门 | 结果 |", "|---|---|"]
    for k, v in a["gates"].items():
        lines.append(f"| {k} | {'✓' if v else '✗'} |")
    f = a["fidelity_pilot"]
    rb = f["residual_by_augmax_mean"]
    lines += ["", "## 保真度（试点，25% → 100%；参照带见 precheck.md）", "",
              "| 语料 | 任务数 | ρ 中位 [q25, q75] | recall@3 | 残差 ≤ 0.5 | (0.5, 0.8] | > 0.8 |", "|---|---:|---|---:|---:|---:|---:|",
              f"| 试点 | {f['n_tasks']} | {_f(f['rho_median'])} [{_f(f['rho_q25'])}, {_f(f['rho_q75'])}] | {_f(f['recall3_mean'])} | {_f(rb.get('le0.5'))} | {_f(rb.get('0.5-0.8'))} | {_f(rb.get('gt0.8'))} |",
              f"| 参照 官方 29 / UCR 84 ρ 中位 | | {_f(a['fidelity_reference']['official29'])} / {_f(a['fidelity_reference']['ucr84'])} | | | | |",
              "", f"任务相似度（shared-20 Spearman）：同集 {_f(a['similarity']['same_dataset_mean'])}（{a['similarity']['n_same_pairs']} 对）/ 异集 {_f(a['similarity']['cross_dataset_mean'])}（{a['similarity']['n_cross_pairs']} 对）；"
                  f"random-20 failed 份额 {_f(a['failed_share_random_all'])}", "",
              "| 预测 | 结果 |", "|---|---|"]
    for k, v in a["predictions"].items():
        lines.append(f"| {k} | {'—' if v is None else ('✓' if v else '✗')} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("slice", "sanity", "precheck", "plan", "run", "probe0", "verify", "smoke", "analyze"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--stage", default="pilot", choices=("pilot", "full"))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--config", default=str(ac.ACTS_YAML))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--joblib-backend", default="threading", choices=("threading", "loky"))
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None, help="只跑这些任务 id（--run）")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ac.load_acts_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.slice:
        cmd_slice(cfg, outdir, args.stage)
    if args.sanity:
        cmd_sanity(cfg, outdir)
    if args.precheck:
        cmd_precheck(cfg, outdir)
    if args.plan:
        cmd_plan(cfg, outdir)
    if args.smoke:
        return 0 if cmd_smoke(args, cfg, outdir) else 1
    if args.run:
        cmd_run(args, cfg, outdir)
    if args.probe0:
        cmd_probe0(args, cfg, outdir)
    if args.verify:
        v = cmd_verify(cfg, outdir)
        if not v["ok"]:
            return 1
    if args.analyze:
        cmd_analyze(cfg, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
