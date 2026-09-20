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
  --check                        部署自检：本机数据（npz）重建的切片划分（n_train / n_eval / T / C / majority / split_md5）与 tasks.json
                                 逐任务一致 + 本机重建 plan 的 key 序与 plan.json 一致（实验室服务器 setup 调用）
  --smoke                        本地冒烟（HMD 2 任务 × 3 条 × 8 iters + 探针 1 个）→ results/acts/s1/smoke/
  --analyze                      标签 / G-A 读数 / 保真度 / 相似度 / 读数码 → analysis.json / tables.md / labels.jsonl

S1′（--profile s1p，PROTOCOL results/acts/s1p/PROTOCOL.md；评估 = 整切片预训练 + 分层 5 折 CV、R′ = 终点、shared 两种子均值）：
  --slice-all                    切 admission.candidates 全部候选源（8 任务 / 源）→ tasks.json（含 5 折划分 md5）
  --plan-a                       阶段 A：准入块（每候选 2 个 n 档 3 任务 × shared × 2 种子）+ Cricket 3× 消融 → plan_A.json
  --run / --probe0 / --verify --phase A|B   CV 执行体 / CV 探针（探针任务 = 该阶段计划涉及的任务）
  --analyze-a                    ADM（准入）+ BUD（预算）判定 → analysis_A.json / tables_A.md
  --plan-b                       阶段 B：试点集 × 8 任务 × (shared × 2 种子 + random) − 阶段 A 已跑 → plan_B.json
  --analyze                      （profile s1p 时）终点 / 两种子均值标签、G-A1 / G-A2（无偏噪声）、读数码 → analysis.json / tables.md
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

import torch  # noqa: E402

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.acts_slicer import AutoCTSTierSlicer  # noqa: E402
from autocls.data.task_slicer import SlicedTask  # noqa: E402
from autocls.downstream.linear_eval import encode_features  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import device_label, run_seed_for  # noqa: E402
from autocls.oom_guard import RprimeOOM, is_oom_error  # noqa: E402
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
    cv = _is_cv(cfg)
    meta: dict[str, dict] = {}
    for t in tasks:
        if cv:
            arr = ac.task_arrays_cv(t, X, y, n_folds=int(cfg["eval"]["n_folds"]), seed=ac.cv_fold_seed_for(t.task_id, cfg["eval"].get("fold_seed_key", "cv_folds")))
            n_train, T, C = (int(v) for v in arr["X"].shape)          # 预训练集 = 整个切片
            extra = {"n_eval": 0, "fold_md5": arr["fold_md5"], "n_folds": int(cfg["eval"]["n_folds"])}
        else:
            arr = ac.task_arrays(t, X, y, eval_ratio=float(cfg["slicing"]["eval_ratio"]), seed=ac.split_seed_for(t.task_id))
            n_train, T, C = (int(v) for v in arr["Xtr"].shape)
            extra = {"n_eval": int(len(arr["yte"])), "split_md5": ac.split_md5_of(arr)}
        meta[t.task_id] = {**dataclasses.asdict(t), "dataset": name, "n_train": n_train, **extra,
                           "T": T, "C": C, "k": int(len(t.classes)), "majority": float(arr["majority"]),
                           "iters": int(_g35.resolve_iters_from_size(proto, n_train * T * C)),
                           "L_full": int(X.shape[1]), "C_full": int(X.shape[2]), "N_full": int(len(y))}
    return tasks, meta


def _is_cv(cfg: Mapping) -> bool:
    return str((cfg.get("eval") or {}).get("mode", "holdout")).startswith("cv")


def cmd_slice_all(cfg: Mapping, outdir: Path) -> dict:
    """S1′：切 admission.candidates 全部候选源，每源 pilot.n_tasks 个任务（前缀 = OA 前 8 行）。"""
    n_tasks = int(cfg["pilot"]["n_tasks"])
    all_meta: dict[str, dict] = {}
    prefix_ok = True
    for name in cfg["admission"]["candidates"]:
        X, y = ac.load_dataset_full(name)
        tasks, meta = slice_dataset(cfg, name, n_tasks, X=X, y=y)
        more, _m2 = slice_dataset(cfg, name, max(16, n_tasks), X=X, y=y)
        prefix_ok &= more[: len(tasks)] == tasks
        all_meta.update(meta)
        for t_ in tasks:
            m = meta[t_.task_id]
            print(f"[slice] {t_.task_id} {t_.tag} C={m['C']}/{m['C_full']} win={t_.time_window} T={m['T']}/{m['L_full']} "
                  f"n={len(t_.sample_indices)}/{m['N_full']} k={m['k']} maj={m['majority']:.3f} iters={m['iters']}", flush=True)
    payload = {"stage": "s1p", "profile": cfg.get("profile"), "datasets": list(cfg["admission"]["candidates"]), "n_tasks": n_tasks,
               "protocol": cfg["protocol"], "slicing": dict(cfg["slicing"]), "eval": dict(cfg.get("eval") or {}), "prefix_stable_16": bool(prefix_ok), "tasks": all_meta}
    _write_json(outdir / "tasks.json", payload)
    print(f"[slice] {len(all_meta)} 任务 → {outdir / 'tasks.json'}；16 任务前缀自检 {'OK' if prefix_ok else 'FAILED'}", flush=True)
    return payload


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
    if _is_cv(cfg):
        return make_task_loader_cv(tasks, dataset_of, ac.load_dataset_full, n_folds=int(cfg["eval"]["n_folds"]),
                                   fold_seed_key=str(cfg["eval"].get("fold_seed_key", "cv_folds")))
    return ac.make_task_loader(tasks, dataset_of, ac.load_dataset_full, eval_ratio=float(cfg["slicing"]["eval_ratio"]))


def make_task_loader_cv(tasks: Mapping[str, SlicedTask], dataset_of: Mapping[str, str], load_dataset, *, n_folds: int, fold_seed_key: str):
    """CV 版 loader：loader(task_id, har_npz) → {X, y, folds, ...}（单槽缓存，同 ac.make_task_loader）。"""
    slot: dict = {}

    def loader(task_id: str, har_npz=None) -> dict:
        if slot.get("task") == task_id:
            return slot["arrays"]
        name = dataset_of[task_id]
        if slot.get("dataset") != name:
            X, y = load_dataset(name)
            slot.clear()
            slot.update({"dataset": name, "X": X, "y": y})
            print(f"[data] {name}: X={X.shape} classes={len(np.unique(y))}", flush=True)
        t = tasks[task_id]
        arrays = ac.task_arrays_cv(t, slot["X"], slot["y"], n_folds=n_folds, seed=ac.cv_fold_seed_for(task_id, fold_seed_key))
        slot.update({"task": task_id, "arrays": arrays})
        print(f"[data] {task_id}: X={arrays['X'].shape} classes={len(np.unique(arrays['y']))} folds={n_folds} window={t.time_window}", flush=True)
        return arrays

    return loader


def plan_path(outdir: Path, phase: Optional[str]) -> Path:
    return outdir / (f"plan_{phase}.json" if phase else "plan.json")


def part_tasks(cells: Sequence[Mapping], done_keys: set, n_parts: int, cfg: Mapping) -> list[list[str]]:
    """分片只按**未完成**的格均衡（--resume 重启时已完成的任务不再占负载；无完成记录时与 split_parts 全量分法相同）。"""
    pending = [c for c in cells if c["key"] not in done_keys]
    return split_parts(pending, n_parts, cfg) if pending else [[] for _ in range(int(n_parts))]


def cmd_run(args, cfg: Mapping, outdir: Path) -> None:
    plan = json.loads(plan_path(outdir, getattr(args, "phase", None)).read_text(encoding="utf-8"))
    cells = plan["cells"]
    if getattr(args, "parts", 0):
        done_keys = set(_g35.load_done(outdir / "runs.jsonl")) if args.resume and (outdir / "runs.jsonl").exists() else set()
        sel = set(part_tasks(cells, done_keys, int(args.parts), cfg)[int(args.part)])
        cells = [c for c in cells if c["task_id"] in sel]
    if args.tasks:
        sel = set(args.tasks)
        cells = [c for c in cells if c["task_id"] in sel]
    cells = _g45.drop_repeated_oom(cells, outdir / "oom_failures.jsonl")
    _install_win_timeout()
    ns = argparse.Namespace(outdir=str(outdir), resume=args.resume, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    phase = getattr(args, "phase", None)
    if phase == "W":                                  # S1″ 第 2 步：整集 official / cv5 混排，按格分发
        loader, runner = make_whole_loader(cfg, plan["tasks_meta"]), dispatch_runner
    elif phase == "C":                                # S1″ 主批：切片 loader 按 cfg eval 模式，执行体按格分发
        loader, runner = _loader_for(cfg, load_tasks(outdir)), dispatch_runner
    else:
        loader, runner = _loader_for(cfg, load_tasks(outdir)), (run_trajectory_cell_cv if _is_cv(cfg) else None)
    _g39.execute(cells, ns, f"ACTS-{cfg.get('profile') or 'S1'}", loader=loader, runner=runner)


def run_probe0_cv(task_id: str, arrays: Mapping, s: int, device: str) -> dict:
    """S1′ 探针：随机初始化 encoder 编码整个切片 + 同一 5 折网格 → 折均值 acc。"""
    seed = run_seed_for(task_id, f"probe0:{s}")
    torch.manual_seed(seed)
    enc = build_encoder(input_dims=int(arrays["X"].shape[-1]))
    t0 = time.perf_counter()
    feats = encode_features(enc, arrays["X"], device=device)
    v = _g39.eval_with_timeout(lambda: ac.cv_eval(feats, arrays["y"], arrays["folds"], "ts2vec", seed), _g39.EVAL_TIMEOUT_S)
    return {"task_id": task_id, "probe": int(s), "weight_seed": int(seed), "iters": 0, "eval": "cv5", **v,
            "majority": float(arrays["majority"]), "n": int(arrays["X"].shape[0]), "seconds": round(time.perf_counter() - t0, 1)}


def run_probe0(task_id: str, arrays: Mapping, s: int, device: str) -> dict:
    """iters=0 探针：同口径随机初始化 encoder（seed = run_seed_for(task_id, "probe0:{s}")）编码 + 官方 SVM 网格。"""
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

    done = load_probe0(outdir) if args.resume else {}
    n = int(cfg["labels"]["probe0_seeds"] if n_seeds is None else n_seeds)
    phase = getattr(args, "phase", None)
    if phase == "W":                                  # S1″ 第 2 步：整集任务，探针按该任务的 eval 模式分发
        plan = json.loads(plan_path(outdir, phase).read_text(encoding="utf-8"))
        loader = make_whole_loader(cfg, plan["tasks_meta"])
        tids = list(task_ids) if task_ids else list(plan["probe_tasks"])
        n = int(cfg["whole_eval"].get("probe0_seeds", n)) if n_seeds is None else n
        runner_of = lambda tid: probe_runner_for(plan["tasks_meta"][tid]["eval_mode"])  # noqa: E731
    else:
        payload = load_tasks(outdir)
        loader = _loader_for(cfg, payload)
        if task_ids:
            tids = list(task_ids)
        elif phase:
            tids = list(json.loads(plan_path(outdir, phase).read_text(encoding="utf-8"))["probe_tasks"])
        else:
            tids = list(payload["tasks"])
        runner_of = lambda tid: (run_probe0_cv if _is_cv(cfg) else run_probe0)  # noqa: E731
    _install_win_timeout()
    for tid in tids:
        for s in range(n):
            if (tid, s) in done:
                continue
            arrays = loader(tid)
            row = runner_of(tid)(tid, arrays, s, args.device)
            append_jsonl(outdir / "probe0.jsonl", row)
            print(f"[probe0] {tid} s{s} acc={row['acc']:.3f} maj={row['majority']:.3f} ({row['seconds']}s)", flush=True)


# ---------------------------------------------------------------- verify
def cmd_verify(cfg: Mapping, outdir: Path, phase: Optional[str] = None) -> dict:
    plan = json.loads(plan_path(outdir, phase).read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl") if (outdir / "runs.jsonl").exists() else {}
    want = {c["key"]: c for c in plan["cells"]}
    missing = [k for k in want if k not in rows]
    err = [k for k, r in rows.items() if k in want and r.get("error")]
    bad = [k for k, r in rows.items() if k in want and not r.get("error")
           and (len(r.get("checkpoint_results") or []) != len(want[k]["checkpoints"])
                or ("eval" in want[k] and "eval" in r and r["eval"] != want[k]["eval"]))]     # S1″：行的 eval 模式须与计划格一致（holdout / cv5 / official）
    probe = load_probe0(outdir)
    n_probe = int(cfg["labels"]["probe0_seeds"])
    if phase == "W":
        n_probe = int(cfg["whole_eval"].get("probe0_seeds", n_probe))
    probe_tasks = list(plan.get("probe_tasks") or plan["tasks_meta"])
    probe_missing = [(t, s) for t in probe_tasks for s in range(n_probe) if (t, s) not in probe]
    v = {"planned": len(want), "have": len(want) - len(missing), "missing": len(missing), "error_rows": len(err), "bad": len(bad),
         "probe_planned": len(probe_tasks) * n_probe, "probe_missing": len(probe_missing),
         "ok": not missing and not bad and not probe_missing, "missing_examples": missing[:5], "error_examples": err[:5]}
    print(f"[verify] planned {v['planned']} have {v['have']} missing {v['missing']} error {v['error_rows']} bad {v['bad']}; "
          f"probe0 {v['probe_planned'] - v['probe_missing']}/{v['probe_planned']} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- 部署自检（跨机器 / 跨 numpy 版本）
def check_deployment(cfg: Mapping, outdir: Path, *, load_dataset=None, plan_kwargs: Optional[Mapping] = None) -> dict:
    """从本机数据重建每任务切片数组，n_train / n_eval / T / C / majority / split_md5 须与 tasks.json 逐任务一致；
    plan.json 存在时本机重建的 cell key 序须逐位一致（目录采样与 cache_key 的跨机器一致性）。"""
    payload = load_tasks(outdir)
    meta = payload["tasks"]
    tasks = tasks_from_meta(meta)
    dataset_of = {tid: m["dataset"] for tid, m in meta.items()}
    if _is_cv(cfg):
        loader = make_task_loader_cv(tasks, dataset_of, load_dataset or ac.load_dataset_full, n_folds=int(cfg["eval"]["n_folds"]),
                                     fold_seed_key=str(cfg["eval"].get("fold_seed_key", "cv_folds")))
    else:
        loader = ac.make_task_loader(tasks, dataset_of, load_dataset or ac.load_dataset_full, eval_ratio=float(cfg["slicing"]["eval_ratio"]))
    mismatches = []
    for tid, m in meta.items():
        arr = loader(tid)
        if _is_cv(cfg):
            got = {"n_train": int(arr["X"].shape[0]), "T": int(arr["X"].shape[1]), "C": int(arr["X"].shape[2]),
                   "majority": round(float(arr["majority"]), 6), "fold_md5": arr["fold_md5"]}
            exp = {"n_train": int(m["n_train"]), "T": int(m["T"]), "C": int(m["C"]), "majority": round(float(m["majority"]), 6), "fold_md5": m.get("fold_md5")}
        else:
            got = {"n_train": int(arr["Xtr"].shape[0]), "n_eval": int(len(arr["yte"])), "T": int(arr["Xtr"].shape[1]), "C": int(arr["Xtr"].shape[2]),
                   "majority": round(float(arr["majority"]), 6), "split_md5": ac.split_md5_of(arr)}
            exp = {"n_train": int(m["n_train"]), "n_eval": int(m["n_eval"]), "T": int(m["T"]), "C": int(m["C"]),
                   "majority": round(float(m["majority"]), 6), "split_md5": m.get("split_md5")}
        if got != exp:
            mismatches.append((tid, exp, got))
    plan_ok = None
    if _is_cv(cfg) and (outdir / "plan_A.json").exists():
        plan = json.loads((outdir / "plan_A.json").read_text(encoding="utf-8"))
        rebuilt, _c = build_plan_s1p(cfg, payload["tasks"], phase="A", **dict(plan_kwargs or {}))
        plan_ok = bool([c["key"] for c in rebuilt["cells"]] == [c["key"] for c in plan["cells"]])
    elif (outdir / "plan.json").exists():
        plan = load_plan(outdir)
        rebuilt, _c = build_plan(cfg, payload, **dict(plan_kwargs or {}))
        plan_ok = bool([c["key"] for c in rebuilt["cells"]] == [c["key"] for c in plan["cells"]])
    ok = not mismatches and plan_ok is not False
    print(f"[check] tasks {len(meta)} mismatches {len(mismatches)} plan_keys_match {plan_ok} → {'OK' if ok else 'NOT OK'}", flush=True)
    for tid, exp, got in mismatches[:5]:
        print(f"  {tid}: expected {exp} got {got}", flush=True)
    return {"ok": bool(ok), "n_tasks": len(meta), "mismatches": mismatches, "plan_keys_match": plan_ok}


# ================================================================ S1′（PROTOCOL results/acts/s1p/PROTOCOL.md）
ARM_ABLATION = "ablation3x"


def run_trajectory_cell_cv(cell: dict, arrays: Mapping, device: str) -> dict:
    """S1′ 执行体：整切片预训练（39.run_trajectory_cell 同一 pretrain_encoder 调用）+ 每截点 5 折 CV 评估（折内官方网格）。OOM 抛 RprimeOOM。"""
    _g39.check_extension_consistency(cell)
    opts, seed, iters = cell["options"], int(cell["seed"]), int(cell["iters"])
    X, y, folds = arrays["X"], arrays["y"], arrays["folds"]
    grid = opts["svm_grid"]
    results: list[dict] = []
    t0 = time.perf_counter()

    def on_ckpt(step: int, module) -> None:
        te = time.perf_counter()
        feats = encode_features(module, X, device=device)
        v = _g39.eval_with_timeout(lambda: ac.cv_eval(feats, y, folds, grid, seed), _g39.EVAL_TIMEOUT_S)
        results.append({"iters": int(step), **v, "elapsed_pretrain_s": round(te - t0, 1), "eval_seconds": round(time.perf_counter() - te, 1)})
        module.train()

    try:
        cls = CLS(**cell["cls"])
        torch.manual_seed(seed)
        ov = {}
        if opts.get("input_mask", "none") != "none":
            ov["input_mask"] = opts["input_mask"]
        if float(opts.get("repr_dropout", 0.0)) > 0:
            ov["repr_dropout"] = float(opts["repr_dropout"])
        enc = build_encoder(input_dims=int(X.shape[-1]), overrides=ov or None)
        losses = pretrain_encoder(
            enc, cls, X, max_iters=iters, batch_size=opts["batch_size"], lr=opts["lr"], device=device, seed=seed, use_swa=opts["swa"],
            drop_last=opts["drop_last"], crop_mode=opts["crop_mode"], checkpoints=tuple(cell["checkpoints"]), on_checkpoint=on_ckpt)
    except Exception as e:  # noqa: BLE001
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        return {**cell, "eval": "cv5", "acc": None, "f1": None, "checkpoint_results": results or None,
                "pretrain_seconds": round(time.perf_counter() - t0, 1), "device_name": device_label(device), "error": f"{type(e).__name__}: {e}"}
    final = results[-1]
    return {**cell, "eval": "cv5", "acc": final["acc"], "f1": final["f1"], "checkpoint_results": results,
            "pretrain_seconds": results[-1]["elapsed_pretrain_s"], "eval_seconds": round(sum(float(c["eval_seconds"]) for c in results), 1),
            "device_name": device_label(device), "first_pass_loss": (float(losses[0]) if losses else None),
            "last_pass_loss": (float(losses[-1]) if losses else None), "n_train": int(X.shape[0]), "n_test": int(X.shape[0]), "error": None}


def _catalogs_s1p(cfg: Mapping, meta: Mapping[str, Mapping], *, n_shared=None, n_random=None):
    space = ac.acts_space(cfg)
    shared = ac.shared_catalog(space, cfg, n=n_shared)
    excl = {k for _c, k in shared}
    randoms = {tid: ac.random_catalog(space, cfg, tid, exclude_keys=excl, n=n_random) for tid in meta}
    catalog = {"shared": [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c)} for c, k in shared],
               "random": {tid: [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c)} for c, k in r] for tid, r in randoms.items()},
               "anchors": [], "anchor_seeds": []}
    return shared, randoms, catalog


def admission_tasks(cfg: Mapping, meta: Mapping[str, Mapping]) -> dict[str, list[str]]:
    """{候选源: [准入任务 id]}（OA 行 task_rows；只取 meta 里存在的）。"""
    rows = [int(r) for r in cfg["admission"]["task_rows"]]
    out: dict[str, list[str]] = {}
    for src in cfg["admission"]["candidates"]:
        tids = [f"{src}_a{r:02d}" for r in rows if f"{src}_a{r:02d}" in meta]
        if tids:
            out[src] = tids
    return out


def build_plan_s1p(cfg: Mapping, meta: Mapping[str, Mapping], *, phase: str, n_shared=None, n_random=None,
                   admitted: Optional[Sequence[str]] = None, skip_keys: Optional[set] = None) -> tuple[dict, dict]:
    """阶段 A = 准入块（候选 × 2 任务 × shared × shared_seeds）+ 消融（ablation.dataset 全部任务 × shared × seed 0 × 3× 轨迹）；
    阶段 B = admitted 源 × 全部任务 × (shared × shared_seeds + random × random_seeds) − skip_keys。"""
    protos = _g35.load_protocols()
    shared, randoms, catalog = _catalogs_s1p(cfg, meta, n_shared=n_shared, n_random=n_random)
    s_seeds = [int(s) for s in cfg["labels"]["shared_seeds"]]
    r_seeds = [int(s) for s in cfg["labels"].get("random_seeds", [0])]
    cells: list[dict] = []
    tasks_used: list[str] = []
    if phase == "A":
        for src, tids in admission_tasks(cfg, meta).items():
            for tid in tids:
                tasks_used.append(tid)
                for c, k in shared:
                    for s in s_seeds:
                        cells.append(ac.make_cell(tid, meta[tid], ac.ARM_SHARED, c, k, s, protos, cfg))
        ab = cfg.get("ablation") or {}
        if ab:
            for tid, m in meta.items():
                if m["dataset"] != ab["dataset"]:
                    continue
                tasks_used.append(tid)
                base_iters = int(_g35.resolve_iters_from_size(protos[cfg["protocol"]], int(m["n_train"]) * int(m["T"]) * int(m["C"])))
                for c, k in shared:
                    cells.append(ac.make_cell(tid, m, ab["arm"], c, k, 0, protos, cfg, iters=base_iters * int(ab["budget_mult"]),
                                              fractions=[float(f) for f in ab["fractions"]], extra={"base_iters": base_iters, "budget_mult": int(ab["budget_mult"])}))
    elif phase == "B":
        srcs = list(admitted) if admitted is not None else list(cfg["pilot"]["datasets"])
        for tid, m in meta.items():
            if m["dataset"] not in srcs:
                continue
            tasks_used.append(tid)
            for c, k in shared:
                for s in s_seeds:
                    cells.append(ac.make_cell(tid, m, ac.ARM_SHARED, c, k, s, protos, cfg))
            for c, k in randoms[tid]:
                for s in r_seeds:
                    cells.append(ac.make_cell(tid, m, ac.ARM_RANDOM, c, k, s, protos, cfg))
    else:
        raise ValueError(f"未知 phase={phase!r}")
    keys = [c["key"] for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("cache_key 撞键")
    if skip_keys:
        cells = [c for c in cells if c["key"] not in skip_keys]
    tasks_used = list(dict.fromkeys(tasks_used))
    est = float(sum(ac.estimate_cell_seconds(c, cfg) for c in cells))
    plan = {"phase": phase, "profile": cfg.get("profile"), "n_cells": len(cells), "n_tasks": len(tasks_used), "probe_tasks": tasks_used,
            "n_probe": len(tasks_used) * int(cfg["labels"]["probe0_seeds"]), "protocol": cfg["protocol"], "est_seconds_4090": est,
            "est_hours_5090": est * float(cfg["cost_model"]["gpu_factor_5090"]) / 3600.0,
            "tasks_meta": {tid: {k: meta[tid][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window") if k in meta[tid]} for tid in tasks_used},
            "cells": cells}
    return plan, catalog


def pilot_sets(cfg: Mapping, admitted) -> list[str]:
    adm = set(admitted)
    return list(cfg["pilot"]["datasets"]) + [d for d in cfg["pilot"].get("add_if_admitted", []) if d in adm]


def admission_decision(metrics: Mapping[str, float], cfg: Mapping) -> dict:
    """ADM：候选按序，ρ ≥ rho_min 准入（fallback 源不计入常规名单）；不足 pool_floor 时按 fallback 顺序补过门者。"""
    a = cfg["admission"]
    fallback = list(a.get("fallback", []))
    rho_min = float(a["rho_min"])

    def ok(src):
        v = metrics.get(src, float("nan"))
        return v is not None and np.isfinite(v) and v >= rho_min

    regular = [s for s in a["candidates"] if s not in fallback]
    admitted = [s for s in regular if ok(s)]
    rejected = [s for s in regular if not ok(s)]
    used = []
    for f in fallback:
        if len(admitted) + len(used) >= int(a["pool_floor"]):
            break
        if ok(f):
            used.append(f)
    pool = admitted + used
    return {"admitted": admitted, "rejected": rejected, "fallback_used": used, "pool": pool, "below_floor": len(pool) < int(a["pool_floor"]),
            "metrics": {k: (None if v is None or not np.isfinite(v) else float(v)) for k, v in metrics.items()}}


def budget_decision(readout: Mapping, cfg: Mapping) -> str:
    d = cfg["ablation"]["decision"]
    fd, rho = readout.get("failed_drop"), readout.get("surv_rho_13_median")
    if fd is not None and rho is not None and np.isfinite(fd) and np.isfinite(rho) and fd >= float(d["failed_drop_min"]) and rho < float(d["surv_rho_max"]):
        return "3x"
    return "1x"


def ablation_readout(rows: Sequence[Mapping], floors: Mapping[str, float]) -> dict:
    """3× 轨迹：截点 [1/12, 1/6, 1/3, 2/3, 1]；1× 终点 = 第 3 截点。逐任务：failed 1× / 3×、3× 存活配置上 ρ(1/3, 1) 与 ρ(2/3, 1)。"""
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("error") or not r.get("checkpoint_results") or len(r["checkpoint_results"]) < 5:
            continue
        by[r["task_id"]].append([float(c["acc"]) for c in r["checkpoint_results"]])
    per: dict[str, dict] = {}
    for tid, accs in by.items():
        A = np.array(accs)
        fl = float(floors.get(tid, float("nan")))
        f1x = float(np.mean(A[:, 2] <= fl)) if np.isfinite(fl) else float("nan")
        f3x = float(np.mean(A[:, 4] <= fl)) if np.isfinite(fl) else float("nan")
        surv = A[:, 4] > fl if np.isfinite(fl) else np.ones(len(A), bool)
        r13 = float(spearmanr(A[surv, 2], A[surv, 4]).correlation) if surv.sum() >= 5 and np.std(A[surv, 2]) > 0 and np.std(A[surv, 4]) > 0 else float("nan")
        r23 = float(spearmanr(A[surv, 3], A[surv, 4]).correlation) if surv.sum() >= 5 and np.std(A[surv, 3]) > 0 and np.std(A[surv, 4]) > 0 else float("nan")
        per[tid] = {"n": int(len(A)), "failed_1x": f1x, "failed_3x": f3x, "n_surv_3x": int(surv.sum()), "surv_rho_13": r13, "surv_rho_23": r23,
                    "mean_1x": float(A[:, 2].mean()), "mean_3x": float(A[:, 4].mean())}
    f1 = [v["failed_1x"] for v in per.values() if np.isfinite(v["failed_1x"])]
    f3 = [v["failed_3x"] for v in per.values() if np.isfinite(v["failed_3x"])]
    r13 = [v["surv_rho_13"] for v in per.values() if np.isfinite(v["surv_rho_13"])]
    r23 = [v["surv_rho_23"] for v in per.values() if np.isfinite(v["surv_rho_23"])]
    return {"per_task": per, "failed_1x_mean": float(np.mean(f1)) if f1 else float("nan"), "failed_3x_mean": float(np.mean(f3)) if f3 else float("nan"),
            "failed_drop": (float(np.mean(f1) - np.mean(f3)) if f1 and f3 else float("nan")),
            "surv_rho_13_median": float(np.median(r13)) if r13 else float("nan"), "surv_rho_23_median": float(np.median(r23)) if r23 else float("nan")}


def verdict_s1p(gates: Mapping[str, bool]) -> str:
    if not gates.get("G-A1", False):
        return "ACTS-S1P-SAME"
    return "ACTS-S1P-GO" if gates.get("G-A2", False) else "ACTS-S1P-NOISE"


def cmd_plan_a(cfg: Mapping, outdir: Path) -> dict:
    payload = load_tasks(outdir)
    plan, catalog = build_plan_s1p(cfg, payload["tasks"], phase="A")
    _write_json(outdir / "plan_A.json", plan)
    _write_json(outdir / "catalog.json", catalog)
    from collections import Counter
    print(f"[plan A] {plan['n_cells']} 格（{dict(Counter(c['arm'] for c in plan['cells']))}），{plan['n_tasks']} 任务、探针 {plan['n_probe']}；"
          f"iters {dict(Counter(c['iters'] for c in plan['cells']))}；估时 4090 {plan['est_seconds_4090'] / 3600:.1f} h", flush=True)
    return plan


def floors_cv(probe: Mapping[tuple[str, int], Mapping], tasks_meta: Mapping[str, Mapping]) -> dict[str, dict]:
    out = {}
    for tid, m in tasks_meta.items():
        accs = [float(r["acc"]) for (t, _s), r in probe.items() if t == tid and r.get("acc") is not None]
        out[tid] = {"majority": float(m["majority"]), "probe_accs": accs, "floor": ac.floor_of(float(m["majority"]), accs)}
    return out


def cmd_analyze_a(cfg: Mapping, outdir: Path) -> dict:
    """阶段 A 读数：ADM（每候选两任务 shared 两种子存活 ρ）+ BUD（Cricket 3× 消融）。"""
    plan = json.loads((outdir / "plan_A.json").read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl")
    probe = load_probe0(outdir)
    floors = floors_cv(probe, plan["tasks_meta"])
    meta = load_tasks(outdir)["tasks"]
    cat = json.loads((outdir / "catalog.json").read_text(encoding="utf-8"))
    shared_keys = [s["cls_key"] for s in cat["shared"]]
    by = {(c["task_id"], c["cls_key"], int(c["seed"]), c["arm"]): rows.get(c["key"]) for c in plan["cells"]}
    adm_metrics: dict[str, float] = {}
    adm_detail: dict[str, dict] = {}
    for src, tids in admission_tasks(cfg, meta).items():
        per = {}
        for tid in tids:
            fl = floors[tid]["floor"]
            s0 = np.array([ac.label_of(ac.endpoint_of(by.get((tid, k, 0, ac.ARM_SHARED))), fl)[0] for k in shared_keys])
            s1 = np.array([ac.label_of(ac.endpoint_of(by.get((tid, k, 1, ac.ARM_SHARED))), fl)[0] for k in shared_keys])
            f0 = np.array([ac.label_of(ac.endpoint_of(by.get((tid, k, 0, ac.ARM_SHARED))), fl)[1] for k in shared_keys])
            f1 = np.array([ac.label_of(ac.endpoint_of(by.get((tid, k, 1, ac.ARM_SHARED))), fl)[1] for k in shared_keys])
            m = ac.admission_rho(s0, s1, f0, f1, min_survivors=int(cfg["admission"]["min_survivors"]))
            per[tid] = {**m, "floor": fl, "failed0": float(f0.mean()), "failed1": float(f1.mean()),
                        "noise_ms_single": float(np.mean(((s0 - s1) / (np.std(s0) or 1.0)) ** 2) / 2)}
        rhos = [v["rho"] for v in per.values()]
        adm_metrics[src] = float(np.mean(rhos)) if rhos and all(np.isfinite(rhos)) else float("nan")
        adm_detail[src] = per
    adm = admission_decision(adm_metrics, cfg)
    abl_rows = [r for r in rows.values() if r.get("arm") == cfg["ablation"]["arm"]]
    abl = ablation_readout(abl_rows, {t: v["floor"] for t, v in floors.items()})
    bud = budget_decision(abl, cfg)
    a = {"admission": {**adm, "detail": adm_detail, "published_gap": cfg["admission"].get("published_gap", {})}, "ablation": abl, "budget": bud,
         "pilot_sets": pilot_sets(cfg, adm["pool"]), "floors": floors}
    _write_json(outdir / "analysis_A.json", a)
    md = ["# S1′ 阶段 A 读数（ADM / BUD）", "", "| 候选源 | 任务 | 存活 | ρ(seed0, seed1) | failed s0 / s1 | 地板 | 发表差 |", "|---|---|---:|---:|---|---:|---:|"]
    for src, per in adm_detail.items():
        for tid, v in per.items():
            md.append(f"| {src} | {tid[-3:]} | {v['n_survivors']} | {_f(v['rho'])} | {_f(v['failed0'], 2)} / {_f(v['failed1'], 2)} | {_f(v['floor'])} | {_f(cfg['admission'].get('published_gap', {}).get(src))} |")
    md += ["", f"ADM：准入 {adm['admitted']}；拒 {adm['rejected']}；fallback {adm['fallback_used']}；源池 {adm['pool']}（底线 {cfg['admission']['pool_floor']}，{'不足' if adm['below_floor'] else '达标'}）；试点集 {a['pilot_sets']}", "",
           "| Cricket 任务 | n | failed 1× | failed 3× | 3× 存活 | ρ(1/3, 1) | ρ(2/3, 1) | mean 1× / 3× |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for tid, v in abl["per_task"].items():
        md.append(f"| {tid} | {v['n']} | {_f(v['failed_1x'], 2)} | {_f(v['failed_3x'], 2)} | {v['n_surv_3x']} | {_f(v['surv_rho_13'])} | {_f(v['surv_rho_23'])} | {_f(v['mean_1x'])} / {_f(v['mean_3x'])} |")
    md += ["", f"BUD：failed 降 {_f(abl['failed_drop'])}（门 ≥ {cfg['ablation']['decision']['failed_drop_min']}）、存活区 ρ(1/3, 1) 中位 {_f(abl['surv_rho_13_median'])}（门 < {cfg['ablation']['decision']['surv_rho_max']}）→ **{bud}**"]
    (outdir / "tables_A.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return a


def cmd_plan_b(cfg: Mapping, outdir: Path) -> dict:
    a = json.loads((outdir / "analysis_A.json").read_text(encoding="utf-8"))
    if a["budget"] != "1x":
        raise SystemExit("BUD = 3x：须先修订协议（train.yaml 协议块 + PROTOCOL）再建阶段 B 计划")
    plan_a = json.loads((outdir / "plan_A.json").read_text(encoding="utf-8"))
    payload = load_tasks(outdir)
    plan, catalog = build_plan_s1p(cfg, payload["tasks"], phase="B", admitted=a["pilot_sets"], skip_keys={c["key"] for c in plan_a["cells"]})
    _write_json(outdir / "plan_B.json", plan)
    _write_json(outdir / "catalog.json", catalog)
    from collections import Counter
    print(f"[plan B] 试点 {a['pilot_sets']}：{plan['n_cells']} 格（{dict(Counter(c['arm'] for c in plan['cells']))}，已扣除阶段 A）+ 探针 {plan['n_probe']}；"
          f"估时 4090 {plan['est_seconds_4090'] / 3600:.1f} h", flush=True)
    return plan


def cmd_analyze_s1p(cfg: Mapping, outdir: Path) -> dict:
    """S1′ 终读：试点集上终点 / 两种子均值标签 → G-A1 / G-A2（无偏噪声）、失败份额、保真度、相似度、读数码。"""
    plan_a = json.loads((outdir / "plan_A.json").read_text(encoding="utf-8"))
    plan_b = json.loads((outdir / "plan_B.json").read_text(encoding="utf-8"))
    a_res = json.loads((outdir / "analysis_A.json").read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl")
    probe = load_probe0(outdir)
    meta = load_tasks(outdir)["tasks"]
    cat = json.loads((outdir / "catalog.json").read_text(encoding="utf-8"))
    shared_keys = [s["cls_key"] for s in cat["shared"]]
    pilot = a_res["pilot_sets"]
    tids = [t for t, m in meta.items() if m["dataset"] in pilot]
    floors = floors_cv(probe, {t: meta[t] for t in tids})
    cells = [c for c in plan_a["cells"] + plan_b["cells"] if c["task_id"] in set(tids) and c["arm"] in (ac.ARM_SHARED, ac.ARM_RANDOM)]
    by: dict[tuple, dict] = {}
    for c in cells:
        by[(c["task_id"], c["cls_key"], int(c["seed"]))] = rows.get(c["key"])
    s_seeds = [int(s) for s in cfg["labels"]["shared_seeds"]]
    labels = []
    for t in tids:
        fl = floors[t]["floor"]
        for k in shared_keys:
            r, failed = ac.label_multi_seed([by.get((t, k, s)) for s in s_seeds], fl)
            labels.append({"task_id": t, "dataset": meta[t]["dataset"], "cls_key": k, "arm": ac.ARM_SHARED, "r_prime": r, "failed": failed,
                           "seed_vals": [ac.endpoint_of(by.get((t, k, s))) for s in s_seeds], "missing": any(by.get((t, k, s)) is None for s in s_seeds)})
        for x in cat["random"][t]:
            k = x["cls_key"]
            r, failed = ac.label_multi_seed([by.get((t, k, 0))], fl)
            labels.append({"task_id": t, "dataset": meta[t]["dataset"], "cls_key": k, "arm": ac.ARM_RANDOM, "r_prime": r, "failed": failed,
                           "seed_vals": [ac.endpoint_of(by.get((t, k, 0)))], "missing": by.get((t, k, 0)) is None})
    from autocls.labeling import append_jsonl_many
    (outdir / "labels.jsonl").unlink(missing_ok=True)
    append_jsonl_many(outdir / "labels.jsonl", labels)
    stats = {}
    for t in tids:
        v = np.array([l["r_prime"] for l in labels if l["task_id"] == t]); stats[t] = (float(v.mean()), float(v.std()) or 1.0)
    g1, g2 = cfg["gates"]["G-A1"], cfg["gates"]["G-A2"]
    per_ds, pooled_ss, pooled_df, noise_all = {}, 0.0, 0, []
    for d in pilot:
        dt = [t for t in tids if meta[t]["dataset"] == d]
        Z, Z0, Z1, B = [], [], [], []
        for t in dt:
            mu, sd = stats[t]
            fl = floors[t]["floor"]
            lab = {l["cls_key"]: l for l in labels if l["task_id"] == t and l["arm"] == ac.ARM_SHARED}
            Z.append([(lab[k]["r_prime"] - mu) / sd for k in shared_keys])
            z0, z1, both = [], [], []
            for k in shared_keys:
                sv = lab[k]["seed_vals"]
                v0 = sv[0] if len(sv) > 0 else None
                v1 = sv[1] if len(sv) > 1 else None
                l0 = ac.label_of(v0, fl) if v0 is not None else (fl, True)
                l1 = ac.label_of(v1, fl) if v1 is not None else (fl, True)
                z0.append((l0[0] - mu) / sd if v0 is not None else np.nan)
                z1.append((l1[0] - mu) / sd if v1 is not None else np.nan)
                both.append(bool(v0 is not None and v1 is not None and not l0[1] and not l1[1]))
            Z0.append(z0); Z1.append(z1); B.append(both)
        Z, Z0, Z1 = np.array(Z), np.array(Z0, dtype=float), np.array(Z1, dtype=float)
        B = np.array(B, dtype=bool) if B else np.zeros((0, len(shared_keys)), bool)
        noise_mean = ac.noise_ms_of_mean(Z0, Z1)
        inter = ac.interaction_share(Z, noise_ms=noise_mean)
        pooled_ss += inter["ss_int"]; pooled_df += inter["df"]
        dd = (Z0 - Z1).ravel(); noise_all += (dd[np.isfinite(dd)] ** 2 / 4).tolist()
        cond = ((Z0 - Z1)[B] ** 2 / 2) if B.size else np.array([])
        sh3 = ac.split_half_rho(Z, half_size=int(g1["half_size"]), n_draws=int(g1["n_draws"]), seed=0) if Z.shape[0] >= 2 * int(g1["half_size"]) else float("nan")
        fl_ = [l for l in labels if l["dataset"] == d]
        per_ds[d] = {"n_tasks": len(dt), "split_half_3": sh3, **inter, "noise_ms_single": float(np.mean((dd[np.isfinite(dd)] ** 2) / 2)) if dd.size else float("nan"),
                     "noise_ms_cond_surv": float(np.mean(cond)) if cond.size else float("nan"), "n_cond_pairs": int(cond.size),
                     "failed_share_shared": float(np.mean([l["failed"] for l in fl_ if l["arm"] == ac.ARM_SHARED])), "failed_share_random": float(np.mean([l["failed"] for l in fl_ if l["arm"] == ac.ARM_RANDOM])),
                     "missing": int(sum(l["missing"] for l in fl_))}
    pooled_ms_int = pooled_ss / max(1, pooled_df)
    pooled_noise = float(np.mean(noise_all)) if noise_all else float("nan")
    pooled_share = (pooled_ms_int - pooled_noise) / pooled_ms_int if pooled_ms_int > 0 else float("nan")
    sh_vals = [v["split_half_3"] for v in per_ds.values() if np.isfinite(v["split_half_3"])]
    mean_sh3 = float(np.mean(sh_vals)) if sh_vals else float("nan")
    n_pass = sum(1 for v in per_ds.values() if np.isfinite(v["real_share"]) and v["real_share"] >= float(g2["real_interaction_share_min"]))
    ga1 = bool(np.isfinite(mean_sh3) and mean_sh3 <= float(g1["split_half_rho_max"]))
    ga2 = bool(np.isfinite(pooled_share) and pooled_share >= float(g2["real_interaction_share_min"]) and n_pass >= int(np.ceil(float(g2.get("min_datasets_pass_frac", 0.5)) * len(per_ds))))
    gates = {"G-A0": True, "G-A1": ga1, "G-A2": ga2}
    # 保真度（存活区，25% → 终点）、相似度
    fid = []
    for t in tids:
        e, f = [], []
        for k in shared_keys + [x["cls_key"] for x in cat["random"][t]]:
            r = by.get((t, k, 0))
            if not r or r.get("error") or not r.get("checkpoint_results"):
                continue
            lab = {l["cls_key"]: l for l in labels if l["task_id"] == t}[k]
            if lab["failed"]:
                continue
            e.append(float(r["checkpoint_results"][0]["acc"])); f.append(float(r["checkpoint_results"][-1]["acc"]))
        if len(e) >= 5 and np.std(e) > 0 and np.std(f) > 0:
            fid.append(float(spearmanr(e, f).correlation))
    M = np.array([[{l["cls_key"]: l for l in labels if l["task_id"] == t and l["arm"] == ac.ARM_SHARED}[k]["r_prime"] for k in shared_keys] for t in tids])
    same, cross = [], []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            if np.std(M[i]) == 0 or np.std(M[j]) == 0:
                continue
            r = float(spearmanr(M[i], M[j]).correlation)
            (same if meta[tids[i]]["dataset"] == meta[tids[j]]["dataset"] else cross).append(r)
    a = {"gates": gates, "verdict": verdict_s1p(gates), "pilot_sets": pilot, "per_dataset": per_ds, "split_half_3_mean": mean_sh3,
         "pooled": {"ms_int": pooled_ms_int, "ms_noise_mean": pooled_noise, "real_share": pooled_share, "n_datasets_pass": n_pass},
         "fidelity_surv_median": float(np.median(fid)) if fid else float("nan"), "fidelity_surv_n": len(fid),
         "similarity": {"same_dataset_mean": float(np.mean(same)) if same else float("nan"), "cross_dataset_mean": float(np.mean(cross)) if cross else float("nan")},
         "failed_share_all": float(np.mean([l["failed"] for l in labels])), "n_labels": len(labels), "n_missing": int(sum(l["missing"] for l in labels)),
         "admission": a_res["admission"], "budget": a_res["budget"]}
    _write_json(outdir / "analysis.json", a)
    md = [f"# WP17-ACTS S1′ 读数（{a['verdict']}）", "", f"试点 {pilot}；标签 {a['n_labels']}（缺 {a['n_missing']}）；BUD {a['budget']}", "",
          "| 数据集 | 任务 | 对半分 ρ（half 3） | MS_int | MS_noise(均值) | 真实份额 | 单种子噪声 | 存活条件噪声 (n) | failed shared / random |", "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for d, v in per_ds.items():
        md.append(f"| {d} | {v['n_tasks']} | {_f(v['split_half_3'])} | {_f(v['ms_int'])} | {_f(v['ms_noise'])} | {_f(v['real_share'])} | {_f(v['noise_ms_single'])} | {_f(v['noise_ms_cond_surv'])} ({v['n_cond_pairs']}) | {_f(v['failed_share_shared'], 2)} / {_f(v['failed_share_random'], 2)} |")
    md += ["", f"合并：对半分均值 {_f(mean_sh3)}（门 ≤ {g1['split_half_rho_max']}）；真实份额 {_f(pooled_share)}（门 ≥ {g2['real_interaction_share_min']}，逐集过门 {n_pass}/{len(per_ds)}）；"
               f"存活区保真度中位 {_f(a['fidelity_surv_median'])}（{a['fidelity_surv_n']} 任务）；相似度 同集 {_f(a['similarity']['same_dataset_mean'])} / 异集 {_f(a['similarity']['cross_dataset_mean'])}；failed 总 {_f(a['failed_share_all'], 2)}", "",
           "| 门 | 结果 |", "|---|---|"] + [f"| {k} | {'✓' if v else '✗'} |" for k, v in gates.items()]
    (outdir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return a


# ================================================================ S1″（PROTOCOL results/acts/s1pp/PROTOCOL.md）
WHOLE_SUFFIX = {"official": "_whole_official", "cv5": "_whole_cv5"}
S1P_OUT = ac.OUT_ROOT / "s1p"


def dispatch_runner(cell: dict, arrays: Mapping, device: str) -> dict:
    """按格的 eval 模式分发执行体：official / holdout → 39.run_trajectory_cell（TRAIN 预训练、eval 出 acc）；cv5 → 整切片预训练 + 5 折 CV。"""
    mode = str(cell.get("eval", "holdout"))
    if mode in ("official", "holdout"):
        return _g39.run_trajectory_cell(cell, arrays, device)
    if mode == "cv5":
        return run_trajectory_cell_cv(cell, arrays, device)
    raise ValueError(f"未知 eval 模式 {mode!r}")


def probe_runner_for(mode: str):
    if mode in ("official", "holdout"):
        return run_probe0
    if mode == "cv5":
        return run_probe0_cv
    raise ValueError(f"未知 eval 模式 {mode!r}")


def make_whole_loader(cfg: Mapping, tasks_meta: Mapping[str, Mapping]):
    """第 2 步 loader：整集 npz → official（前 n_train 行 = 官方 TRAIN，TRAIN 拟合标准化、TEST 出 acc）或 cv5（整集标准化 + 分层 5 折）。单槽缓存。"""
    slot: dict = {}
    n_folds = int(cfg["eval"]["n_folds"])
    fold_key = str(cfg["eval"].get("fold_seed_key", "cv_folds"))

    def loader(task_id: str, har_npz=None) -> dict:
        if slot.get("task") == task_id:
            return slot["arrays"]
        m = tasks_meta[task_id]
        name = m["dataset"]
        if slot.get("dataset") != name:
            X, y = ac.load_dataset_full(name)
            slot.clear()
            slot.update({"dataset": name, "X": X, "y": y})
            print(f"[data] {name}: X={X.shape} classes={len(np.unique(y))}", flush=True)
        if m["eval_mode"] == "official":
            arrays = ac.whole_arrays_official(slot["X"], slot["y"], n_train=int(m["n_train_official"]))
            print(f"[data] {task_id}: official train={arrays['Xtr'].shape} test={arrays['Xte_train'].shape}", flush=True)
        else:
            arrays = ac.whole_arrays_cv(slot["X"], slot["y"], n_folds=n_folds, seed=ac.cv_fold_seed_for(task_id, fold_key))
            print(f"[data] {task_id}: cv5 X={arrays['X'].shape} folds={n_folds}", flush=True)
        slot.update({"task": task_id, "arrays": arrays})
        return arrays

    return loader


def build_plan_w(cfg: Mapping, sizes: Mapping[str, Mapping], *, shared_keys: Sequence[str], shared_cls: Mapping[str, CLS]) -> dict:
    """第 2 步计划：whole_eval.datasets × modes × shared（S1′ 目录）× seeds；task_id = <D>_whole_<mode>；official 格尺寸 = 官方 TRAIN，cv5 = 整集。"""
    protos = _g35.load_protocols()
    we = cfg["whole_eval"]
    seeds = [int(s) for s in we["seeds"]]
    cells: list[dict] = []
    tasks_meta: dict[str, dict] = {}
    order: list[str] = []
    for d in we["datasets"]:
        sz = sizes[d]
        for mode in we["modes"]:
            tid = f"{d}{WHOLE_SUFFIX[mode]}"
            n = int(sz["n_train"]) if mode == "official" else int(sz["n_train"]) + int(sz["n_test"])
            m = {"dataset": d, "n_train": n, "T": int(sz["L"]), "C": int(sz["C"]), "eval_mode": mode, "n_train_official": int(sz["n_train"]),
                 "n_test_official": int(sz["n_test"]), "tag": f"whole_{mode}"}
            tasks_meta[tid] = m
            order.append(tid)
            for k in shared_keys:
                for s in seeds:
                    cells.append(ac.make_cell(tid, m, ac.ARM_SHARED, shared_cls[k], k, s, protos, cfg, extra={"eval": mode, "whole_mode": mode}))
    keys = [c["key"] for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("cache_key 撞键")
    est = float(sum(ac.estimate_cell_seconds(c, cfg) for c in cells))
    return {"phase": "W", "profile": cfg.get("profile"), "n_cells": len(cells), "n_tasks": len(order), "probe_tasks": order,
            "n_probe": len(order) * int(we.get("probe0_seeds", cfg["labels"]["probe0_seeds"])), "protocol": cfg["protocol"],
            "est_seconds_4090": est, "shared_keys": list(shared_keys), "tasks_meta": tasks_meta, "cells": cells}


def _uea_sizes(datasets: Sequence[str]) -> dict[str, dict]:
    import yaml
    u = yaml.safe_load(Path(ac.UEA30_YAML).read_text(encoding="utf-8"))["datasets"]
    return {d: {"n_train": int(u[d]["n_train"]), "n_test": int(u[d]["n_test"]), "L": int(u[d]["L"]), "C": int(u[d]["C"])} for d in datasets}


def cmd_plan_w(cfg: Mapping, outdir: Path) -> dict:
    we = cfg["whole_eval"]
    sizes = _uea_sizes(we["datasets"])
    for d, n in (we.get("n_train_official") or {}).items():
        if sizes[d]["n_train"] != int(n):
            raise SystemExit(f"{d}: acts.yaml n_train_official {n} ≠ uea30.yaml n_train {sizes[d]['n_train']}")
    cat_p = json.loads((ac.OUT_ROOT / str(we.get("shared_from", "s1p")) / "catalog.json").read_text(encoding="utf-8"))
    shared = cat_p["shared"]
    plan = build_plan_w(cfg, sizes, shared_keys=[s["cls_key"] for s in shared], shared_cls={s["cls_key"]: CLS(**s["cls"]) for s in shared})
    _write_json(outdir / "plan_W.json", plan)
    _write_json(outdir / "catalog_W.json", {"shared": shared, "sizes": sizes})
    from collections import Counter
    print(f"[plan W] {plan['n_cells']} 格（{dict(Counter(c['eval'] for c in plan['cells']))}；iters {dict(Counter(c['iters'] for c in plan['cells']))}）"
          f"+ 探针 {plan['n_probe']}；估时 4090 {plan['est_seconds_4090'] / 3600:.1f} h", flush=True)
    return plan


def _slice_profiles_s1p(dataset: str, shared_keys: Sequence[str], q: float) -> Optional[np.ndarray]:
    """S1′ 切片（8 任务）40 候选钉底 z 的 shared 列均值（原始两种子均值；error 格取任务内最小值）。"""
    lp = S1P_OUT / "labels.jsonl"
    if not lp.exists():
        return None
    labels = read_jsonl(lp)
    by_task: dict[str, dict] = defaultdict(dict)
    for l in labels:
        if l["dataset"] == dataset:
            by_task[l["task_id"]][l["cls_key"]] = l
    prof = []
    for tid, lab in by_task.items():
        keys = list(shared_keys) + [k for k in lab if k not in shared_keys]
        v = np.array([np.mean([x for x in lab[k]["seed_vals"] if x is not None]) if any(x is not None for x in lab[k]["seed_vals"]) else np.nan for k in keys], float)
        v = np.where(np.isfinite(v), v, np.nanmin(v))
        prof.append(ac.clip_z(v, q)[0][: len(shared_keys)])
    return np.mean(prof, 0) if prof else None


def cmd_analyze_w(cfg: Mapping, outdir: Path) -> dict:
    """第 2 步读数：每集 ρ_eval（official vs cv5 两种子均值）、ρ_seed（同口径两种子）、ρ_slice（cv5 整集 vs S1′ 切片画像）、赢家 τ、探针地板 → 决策 W。"""
    plan = json.loads((outdir / "plan_W.json").read_text(encoding="utf-8"))
    cat = json.loads((outdir / "catalog_W.json").read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl")
    probe = load_probe0(outdir)
    keys = list(plan["shared_keys"])
    cls_of = {s["cls_key"]: s["cls"] for s in cat["shared"]}
    we = cfg["whole_eval"]
    seeds = [int(s) for s in we["seeds"]]
    q = float(cfg["scale"]["clip_quantile"])
    per: dict[str, dict] = {}
    for d in we["datasets"]:
        M: dict[str, np.ndarray] = {}
        for mode in we["modes"]:
            tid = f"{d}{WHOLE_SUFFIX[mode]}"
            A = np.full((len(seeds), len(keys)), np.nan)
            for c in plan["cells"]:
                if c["task_id"] != tid:
                    continue
                v = ac.endpoint_of(rows.get(c["key"]))
                A[seeds.index(int(c["seed"])), keys.index(c["cls_key"])] = np.nan if v is None else v
            M[mode] = A
        mean = {mode: np.nanmean(A, 0) for mode, A in M.items()}
        rho_seed = {mode: (float(spearmanr(A[0], A[1], nan_policy="omit").correlation) if A.shape[0] >= 2 else float("nan")) for mode, A in M.items()}
        rho_eval = float(spearmanr(mean["official"], mean["cv5"], nan_policy="omit").correlation) if {"official", "cv5"} <= set(mean) else float("nan")
        prof = _slice_profiles_s1p(d, keys, q)
        rho_slice = {mode: (float(spearmanr(mean[mode], prof, nan_policy="omit").correlation) if prof is not None else float("nan")) for mode in mean}
        winners, tau_mean, floors, failed = {}, {}, {}, {}
        for mode in mean:
            tid = f"{d}{WHOLE_SUFFIX[mode]}"
            order = np.argsort(-np.where(np.isfinite(mean[mode]), mean[mode], -np.inf))
            winners[mode] = [(keys[i][:8], str(cls_of[keys[i]]["temperature"]), round(float(mean[mode][i]), 4)) for i in order[:3]]
            g = defaultdict(list)
            for i, k in enumerate(keys):
                if np.isfinite(mean[mode][i]):
                    g[str(cls_of[k]["temperature"])].append(float(mean[mode][i]))
            tau_mean[mode] = {t: float(np.mean(v)) for t, v in sorted(g.items(), key=lambda kv: float(kv[0]))}
            pr = [float(r["acc"]) for (t, _s), r in probe.items() if t == tid and r.get("acc") is not None]
            floors[mode] = float(np.mean(pr)) if pr else float("nan")
            failed[mode] = float(np.mean(mean[mode][np.isfinite(mean[mode])] <= floors[mode])) if pr else float("nan")
        per[d] = {"rho_eval": rho_eval, "rho_seed": rho_seed, "rho_slice": rho_slice, "winners_top3": winners, "mean_by_tau": tau_mean,
                  "probe_floor": floors, "failed_share_diag": failed, "n_missing": int(sum(int(np.isnan(A).sum()) for A in M.values()))}
    rho_eval_mean = float(np.nanmean([v["rho_eval"] for v in per.values()]))
    rho_slice_mean = float(np.nanmean([v["rho_slice"].get("cv5", float("nan")) for v in per.values()]))
    decision = ac.step2_decision(rho_eval_mean, rho_slice_mean, cfg)
    a = {"per_dataset": per, "rho_eval_mean": rho_eval_mean, "rho_slice_cv5_mean": rho_slice_mean, "decision": decision}
    _write_json(outdir / "analysis_W.json", a)
    md = [f"# S1″ 第 2 步读数（{decision['code']}：切片口径 → {decision['eval_mode']}）", "",
          "| 数据集 | ρ_eval(official, cv5) | ρ_seed official / cv5 | ρ_slice official / cv5 | official 前 3（key, τ, acc） | cv5 前 3 | 各 τ 均值 official | 各 τ 均值 cv5 | 探针地板 off / cv5 | failed(诊断) off / cv5 |",
          "|---|---:|---|---|---|---|---|---|---|---|"]
    for d, v in per.items():
        md.append(f"| {d} | {_f(v['rho_eval'])} | {_f(v['rho_seed'].get('official'))} / {_f(v['rho_seed'].get('cv5'))} | {_f(v['rho_slice'].get('official'))} / {_f(v['rho_slice'].get('cv5'))} | "
                  f"{v['winners_top3'].get('official')} | {v['winners_top3'].get('cv5')} | {v['mean_by_tau'].get('official')} | {v['mean_by_tau'].get('cv5')} | "
                  f"{_f(v['probe_floor'].get('official'))} / {_f(v['probe_floor'].get('cv5'))} | {_f(v['failed_share_diag'].get('official'), 2)} / {_f(v['failed_share_diag'].get('cv5'), 2)} |")
    md += ["", f"ρ_eval 均值 {_f(rho_eval_mean)}（门 {cfg['whole_eval']['decision']['rho_eval_min']}）；ρ_slice(cv5) 均值 {_f(rho_slice_mean)}（门 {cfg['whole_eval']['decision']['rho_slice_min']}）→ **{decision['code']}**，"
           f"切片口径 {decision['eval_mode']}，n/L 效应 {decision['nl_effect']}"]
    (outdir / "tables_W.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return a


def build_plan_c(cfg: Mapping, meta: Mapping[str, Mapping], *, shared: Sequence[tuple[CLS, str]], existing_keys: set,
                 randoms: Optional[Mapping[str, Sequence[tuple[CLS, str]]]] = None) -> dict:
    """第 3 步主批：准入候选各 4 任务（OA 行 task_rows）× shared × shared_seeds；试点集全部 8 任务（+ 复用的 random × random_seeds）。
    已在缓存里的格照列（--resume 跳过），n_existing / n_new 分开报。"""
    protos = _g35.load_protocols()
    pilot = list(cfg["pilot"]["datasets"])
    cands = list(cfg["admission"]["candidates"])
    rows = [int(r) for r in cfg["admission"]["task_rows"]]
    s_seeds = [int(s) for s in cfg["labels"]["shared_seeds"]]
    r_seeds = [int(s) for s in cfg["labels"].get("random_seeds", [0])]
    tids: list[str] = []
    for src in cands + [p for p in pilot if p not in cands]:
        if src in pilot:
            tids += sorted(t for t, m in meta.items() if m["dataset"] == src)
        else:
            tids += [f"{src}_a{r:02d}" for r in rows if f"{src}_a{r:02d}" in meta]
    tids = list(dict.fromkeys(tids))
    cells: list[dict] = []
    for tid in tids:
        for c, k in shared:
            for s in s_seeds:
                cells.append(ac.make_cell(tid, meta[tid], ac.ARM_SHARED, c, k, s, protos, cfg))
        for c, k in (randoms or {}).get(tid, []):
            for s in r_seeds:
                cells.append(ac.make_cell(tid, meta[tid], ac.ARM_RANDOM, c, k, s, protos, cfg))
    keys = [c["key"] for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("cache_key 撞键")
    n_exist = sum(1 for k in keys if k in existing_keys)
    est = float(sum(ac.estimate_cell_seconds(c, cfg) for c in cells if c["key"] not in existing_keys))
    return {"phase": "C", "profile": cfg.get("profile"), "n_cells": len(cells), "n_existing": n_exist, "n_new": len(cells) - n_exist,
            "n_tasks": len(tids), "probe_tasks": tids, "n_probe": len(tids) * int(cfg["labels"]["probe0_seeds"]), "protocol": cfg["protocol"],
            "est_seconds_4090_new": est, "eval_mode": str((cfg.get("eval") or {}).get("mode", "holdout")),
            "tasks_meta": {tid: {k: meta[tid][k] for k in ("dataset", "n_train", "T", "C", "k", "majority", "iters", "tag", "time_window", "fold_md5") if k in meta[tid]} for tid in tids},
            "cells": cells}


def cmd_plan_c(cfg: Mapping, outdir: Path) -> dict:
    payload = load_tasks(outdir)
    meta = payload["tasks"]
    meta_p = load_tasks(S1P_OUT)["tasks"]
    mism = [t for t in meta_p if t not in meta or meta[t].get("fold_md5") != meta_p[t].get("fold_md5") or meta[t].get("n_train") != meta_p[t].get("n_train")]
    if mism:
        raise SystemExit(f"C0：S1″ tasks.json 与 S1′ 旧任务不一致（{len(mism)}，如 {mism[:3]}）——切片器 seed / 设计被改动？")
    space = ac.acts_space(cfg)
    shared = ac.shared_catalog(space, cfg)
    cat_p = json.loads((S1P_OUT / "catalog.json").read_text(encoding="utf-8"))
    pilot = set(cfg["pilot"]["datasets"])
    randoms = {tid: [(CLS(**x["cls"]), x["cls_key"]) for x in xs if not ac.is_pruned(CLS(**x["cls"]), cfg)]
               for tid, xs in cat_p["random"].items() if tid in meta and meta[tid]["dataset"] in pilot}
    existing = set(_g35.load_done(outdir / "runs.jsonl")) if (outdir / "runs.jsonl").exists() else set()
    plan = build_plan_c(cfg, meta, shared=shared, existing_keys=existing, randoms=randoms)
    old_keys = [s["cls_key"] for s in cat_p["shared"] if not ac.is_pruned(CLS(**s["cls"]), cfg)]
    if [k for _c, k in shared[: len(old_keys)]] != old_keys:
        raise SystemExit("C0：shared-30 前缀 ≠ S1′ 未剪 shared（rng 流被改动？）")
    catalog = {"shared": [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c), "retained_from_s1p": k in old_keys} for c, k in shared],
               "random": {tid: [{"cls_key": k, "cls": c.to_dict(), "aug_max": ac.aug_max(c)} for c, k in r] for tid, r in randoms.items()},
               "anchors": [], "anchor_seeds": []}
    _write_json(outdir / "plan_C.json", plan)
    _write_json(outdir / "catalog.json", catalog)
    from collections import Counter
    print(f"[plan C] {plan['n_cells']} 格（{dict(Counter(c['arm'] for c in plan['cells']))}；已有 {plan['n_existing']}、新 {plan['n_new']}；eval {plan['eval_mode']}）"
          f"，{plan['n_tasks']} 任务、探针 {plan['n_probe']}；新格估时 4090 {plan['est_seconds_4090_new'] / 3600:.1f} h；shared 复用 {len(old_keys)} + 新 {len(shared) - len(old_keys)}", flush=True)
    return plan


def _frames_c(plan: Mapping, rows: Mapping, cat: Mapping, cfg: Mapping) -> dict[str, dict]:
    """每任务候选帧：shared 两种子原始终点（单缺 → 另一种子；双缺 → 该任务 shared 最小值）+ random 单种子；n_seeds 向量。"""
    shared_keys = [s["cls_key"] for s in cat["shared"]]
    s_seeds = [int(s) for s in cfg["labels"]["shared_seeds"]]
    by: dict[tuple, Optional[float]] = {}
    for c in plan["cells"]:
        by[(c["task_id"], c["cls_key"], int(c["seed"]), c["arm"])] = ac.endpoint_of(rows.get(c["key"]))
    frames: dict[str, dict] = {}
    for tid in plan["probe_tasks"]:
        a0 = np.array([by.get((tid, k, s_seeds[0], ac.ARM_SHARED)) if by.get((tid, k, s_seeds[0], ac.ARM_SHARED)) is not None else np.nan for k in shared_keys], float)
        a1 = np.array([by.get((tid, k, s_seeds[1], ac.ARM_SHARED)) if len(s_seeds) > 1 and by.get((tid, k, s_seeds[1], ac.ARM_SHARED)) is not None else np.nan for k in shared_keys], float)
        both_nan = np.isnan(a0) & np.isnan(a1)
        a0 = np.where(np.isnan(a0), a1, a0); a1 = np.where(np.isnan(a1), a0, a1)
        fill = np.nanmin(np.concatenate([a0, a1])) if np.isfinite(np.concatenate([a0, a1])).any() else np.nan
        a0 = np.where(both_nan, fill, a0); a1 = np.where(both_nan, fill, a1)
        rk = [x["cls_key"] for x in cat["random"].get(tid, [])]
        rv = np.array([by.get((tid, k, 0, ac.ARM_RANDOM)) if by.get((tid, k, 0, ac.ARM_RANDOM)) is not None else np.nan for k in rk], float)
        ok = np.isfinite(rv)
        frames[tid] = {"a0": a0, "a1": a1, "m": (a0 + a1) / 2.0, "shared_keys": shared_keys, "random_keys": [k for k, o in zip(rk, ok) if o],
                       "frame_vals": np.concatenate([(a0 + a1) / 2.0, rv[ok]]), "frame_keys": shared_keys + [k for k, o in zip(rk, ok) if o],
                       "n_seeds": [len(s_seeds)] * len(shared_keys) + [1] * int(ok.sum()), "n_both_missing": int(both_nan.sum())}
    return frames


def verdict_s1pp(gates: Mapping[str, bool]) -> str:
    if not gates.get("G-A1n", False):
        return "ACTS-S1PP-SAME"
    return "ACTS-S1PP-GO" if gates.get("G-A2", False) else "ACTS-S1PP-NOISE"


def cmd_analyze_c(cfg: Mapping, outdir: Path) -> dict:
    """第 3 步终读：(a′) 标签 → G-A0 regret 门、准入 v2、G-A1′ / G-A2（钉底尺度）、余量 H（同域 / 跨域 LODO）、诊断（探针地板 failed、存活数、配对产量）。"""
    plan = json.loads((outdir / "plan_C.json").read_text(encoding="utf-8"))
    cat = json.loads((outdir / "catalog.json").read_text(encoding="utf-8"))
    rows = _g35.load_done(outdir / "runs.jsonl")
    probe = load_probe0(outdir)
    meta = plan["tasks_meta"]
    q, tf = float(cfg["scale"]["clip_quantile"]), float(cfg["scale"]["noise_top_frac"])
    frames = _frames_c(plan, rows, cat, cfg)
    cls_of = {s["cls_key"]: s["cls"] for s in cat["shared"]}
    for xs in cat["random"].values():
        for x in xs:
            cls_of.setdefault(x["cls_key"], x["cls"])
    n_sh = len(cat["shared"])
    floors = floors_cv(probe, meta)
    # (a′) 标签落盘
    from autocls.labeling import append_jsonl_many
    labels = []
    for tid, f in frames.items():
        for i, k in enumerate(f["frame_keys"]):
            labels.append({"task_id": tid, "dataset": meta[tid]["dataset"], "cls_key": k, "arm": (ac.ARM_SHARED if i < n_sh else ac.ARM_RANDOM),
                           "r_prime": float(f["frame_vals"][i]), "seed_vals": ([float(f["a0"][i]), float(f["a1"][i])] if i < n_sh else [float(f["frame_vals"][i])]),
                           "floor_diag": floors[tid]["floor"], "failed_diag": bool(f["frame_vals"][i] <= floors[tid]["floor"])})
    (outdir / "labels.jsonl").unlink(missing_ok=True)
    append_jsonl_many(outdir / "labels.jsonl", labels)
    # G-A0 regret 门
    ga0 = ac.pruning_regret(official_rows(), cfg)
    # 准入 v2
    adm_metrics = {src: ac.admission_v2_metrics({tid: (frames[tid]["a0"], frames[tid]["a1"]) for tid in tids if tid in frames}, clip_quantile=q, noise_top_frac=tf)
                   for src, tids in admission_tasks(cfg, meta).items()}
    adm = ac.admission_v2_decision(adm_metrics, cfg)
    # 门（试点集，帧 = shared + random）
    pilot = list(cfg["pilot"]["datasets"])
    g1, g2 = cfg["gates"]["G-A1n"], cfg["gates"]["G-A2"]
    per_ds: dict[str, dict] = {}
    pooled_ss = pooled_df = 0.0
    noise_all: list[float] = []
    for d in pilot:
        dt = [tid for tid in plan["probe_tasks"] if meta[tid]["dataset"] == d]
        Z, Z0, Z1 = [], [], []
        for tid in dt:
            f = frames[tid]
            z, mu, sd, cut = ac.clip_z(f["frame_vals"], q)
            Z.append(z[:n_sh]); Z0.append(ac.clip_z_apply(f["a0"], mu, sd, cut)); Z1.append(ac.clip_z_apply(f["a1"], mu, sd, cut))
        Z, Z0, Z1 = np.array(Z), np.array(Z0), np.array(Z1)
        sh = ac.net_interaction_share(Z, Z0, Z1)
        pooled_ss += sh["ss_int"]; pooled_df += sh["df"]
        dd = (Z0 - Z1).ravel(); noise_all += (dd[np.isfinite(dd)] ** 2 / 4).tolist()
        per_ds[d] = {**sh, "n_tasks": len(dt),
                     "failed_diag_shared": float(np.mean([frames[t]["m"] <= floors[t]["floor"] for t in dt])),
                     "surv_both_median": float(np.median([int(np.sum((frames[t]["a0"] > floors[t]["floor"]) & (frames[t]["a1"] > floors[t]["floor"]))) for t in dt]))}
    pooled_ms_int = pooled_ss / max(1.0, pooled_df)
    pooled_noise = float(np.mean(noise_all)) if noise_all else float("nan")
    pooled_real = (pooled_ms_int - pooled_noise) / pooled_ms_int if pooled_ms_int > 0 else float("nan")
    need = int(np.ceil(float(g1.get("min_datasets_pass_frac", 0.5)) * len(per_ds)))
    n1 = sum(1 for v in per_ds.values() if np.isfinite(v["net_share"]) and v["net_share"] >= float(g1["net_share_min"]))
    n2 = sum(1 for v in per_ds.values() if np.isfinite(v["real_share"]) and v["real_share"] >= float(g2["real_interaction_share_min"]))
    gates = {"G-A0": bool(ga0["ok"]), "G-A1n": bool(n1 >= need), "G-A2": bool(np.isfinite(pooled_real) and pooled_real >= float(g2["real_interaction_share_min"]) and n2 >= need)}
    # 余量 H（准入源的全部任务；同域 LODO / 跨域）
    hs = [tid for tid in plan["probe_tasks"] if meta[tid]["dataset"] in set(adm["pool"])]
    H_same = H_cross = None
    if len(hs) >= 2 and len({meta[t]["dataset"] for t in hs}) >= 2:
        S0 = np.array([frames[t]["a0"] for t in hs]); S1 = np.array([frames[t]["a1"] for t in hs])
        H_same = ac.honest_headroom(S0, S1, [meta[t]["dataset"] for t in hs])
        ext = set((cfg["admission"].get("domains") or {}).get("external", []))
        dom = ["external" if meta[t]["dataset"] in ext else "uea" for t in hs]
        H_cross = ac.honest_headroom(S0, S1, dom) if len(set(dom)) == 2 else None
    h_dec = ac.headroom_decision(H_same["regret1_mean"] if H_same else float("nan"), cfg)
    # 诊断：failed 按 τ（探针地板口径，全部任务的 shared 均值）、配对产量
    by_tau: dict[str, list] = defaultdict(list)
    yields: dict[str, int] = {}
    for tid, f in frames.items():
        fl = floors[tid]["floor"]
        for i, k in enumerate(f["shared_keys"]):
            by_tau[str(cls_of[k]["temperature"])].append(bool(f["m"][i] <= fl))
        sig, _sm, _n = ac.top_half_sigma(f["a0"], f["a1"], top_frac=tf)
        if np.isfinite(sig) and sig > 0:
            yields[tid] = ac.pair_yield(ac.pair_weights(f["frame_vals"], f["n_seeds"], sig, **cfg["labels"]["pairs"]))
    a = {"gates": gates, "verdict": verdict_s1pp(gates), "headroom_decision": h_dec, "pruning_regret": ga0, "admission": adm, "admission_metrics": adm_metrics,
         "per_dataset": per_ds, "pooled": {"ms_int": pooled_ms_int, "ms_noise": pooled_noise, "real_share": pooled_real, "n_pass_net": n1, "n_pass_real": n2},
         "headroom_same": H_same, "headroom_cross": H_cross, "n_headroom_tasks": len(hs),
         "failed_diag_by_tau": {t: float(np.mean(v)) for t, v in sorted(by_tau.items(), key=lambda kv: float(kv[0]))},
         "pair_yield_median": float(np.median(list(yields.values()))) if yields else float("nan"), "pair_yield": yields,
         "n_labels": len(labels), "n_both_missing": int(sum(f["n_both_missing"] for f in frames.values()))}
    _write_json(outdir / "analysis_C.json", a)
    md = [f"# S1″ 第 3 步读数（{a['verdict']}；H：{h_dec}）", "",
          f"G-A0 regret 门：29 靶均值 {_f(ga0['mean_gap'])}（门 ≤ {ga0['gate_mean_max']}）、最大 {_f(ga0['max_gap'])}、名义丢冠军 {ga0['n_lost']} → {'✓' if ga0['ok'] else '✗'}", "",
          "| 源 | 任务 | (i) 前一半 seed ρ | (ii) 净占比 | 真实份额 | (iii) IQR/σ_mean | 判定 |", "|---|---:|---:|---:|---:|---:|---|"]
    for src, m in adm_metrics.items():
        md.append(f"| {src} | {m['n_tasks']} | {_f(m['rho_top'], 2)} | {_f(m['net_share'], 2)} | {_f(m['real_share'], 2)} | {_f(m['iqr_sigma'], 1)} | {'准入' if src in adm['admitted'] else '拒 ' + ','.join(adm['reasons'].get(src, []))} |")
    md += ["", f"源池 {adm['pool']}（底线 {cfg['admission']['pool_floor']}，{'不足' if adm['below_floor'] else '达标'}）", "",
           "| 试点集 | 任务 | 净占比 | 毛占比 | 真实份额 | MS_noise | failed(诊断) shared | 两种子都超地板 中位 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d, v in per_ds.items():
        md.append(f"| {d} | {v['n_tasks']} | {_f(v['net_share'], 2)} | {_f(v['gross_share'], 2)} | {_f(v['real_share'], 2)} | {_f(v['ms_noise'])} | {_f(v['failed_diag_shared'], 2)} | {v['surv_both_median']:.0f} |")
    md += ["", f"合并真实份额 {_f(pooled_real)}（门 ≥ {g2['real_interaction_share_min']}，逐集过 {n2}/{len(per_ds)}）；净占比过门 {n1}/{len(per_ds)}（门 ≥ {g1['net_share_min']}）",
           f"余量 H_same regret@1 {_f(H_same['regret1_mean']) if H_same else 'nan'}（中位 {_f(H_same['regret1_median']) if H_same else 'nan'}，regret@3 {_f(H_same['regret3_mean']) if H_same else 'nan'}，{len(hs)} 任务）；"
           f"H_cross {_f(H_cross['regret1_mean']) if H_cross else 'nan'} → **{h_dec}**",
           f"failed（探针地板口径）按 τ：{a['failed_diag_by_tau']}；配对产量中位 {_f(a['pair_yield_median'], 0)} / 任务；标签 {len(labels)}，双缺 {a['n_both_missing']}", "",
           "| 门 | 结果 |", "|---|---|"] + [f"| {k} | {'✓' if v else '✗'} |" for k, v in gates.items()]
    (outdir / "tables_C.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return a


# ---------------------------------------------------------------- smoke
def cmd_smoke(args, cfg: Mapping, outdir: Path) -> bool:
    sm = outdir / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    X, y = ac.load_dataset_full(SMOKE_DATASET)
    _tasks, meta = slice_dataset(cfg, SMOKE_DATASET, 2, X=X, y=y)
    payload = {"stage": "smoke", "datasets": [SMOKE_DATASET], "n_tasks": 2, "protocol": cfg["protocol"], "slicing": dict(cfg["slicing"]), "eval": dict(cfg.get("eval") or {}), "tasks": meta}
    _write_json(sm / "tasks.json", payload)
    if _is_cv(cfg):
        c2 = {**cfg, "admission": {**cfg["admission"], "candidates": [SMOKE_DATASET], "task_rows": [0, 1]}, "ablation": {**cfg["ablation"], "dataset": SMOKE_DATASET}}
        plan, catalog = build_plan_s1p(c2, meta, phase="A", n_shared=2, n_random=1)
        for c in plan["cells"]:                     # 冒烟只验管线：把 iters 压到 8（消融格 24）并重算截点
            c["iters"] = SMOKE_ITERS * (3 if c["arm"] == ARM_ABLATION else 1)
            c["checkpoints"] = list(_g39.checkpoints_for(c["iters"], [float(f) for f in cfg["ablation"]["fractions"]]) if c["arm"] == ARM_ABLATION else _g39.checkpoints_for(c["iters"]))
        n_ck = {c["key"]: len(c["checkpoints"]) for c in plan["cells"]}
        runner = run_trajectory_cell_cv
    else:
        plan, catalog = build_plan(cfg, payload, n_shared=2, n_random=1, iters_override=SMOKE_ITERS, anchor_aug_max_le=1.0)
        n_ck = {c["key"]: 3 for c in plan["cells"]}
        runner = None
    _write_json(sm / "plan.json", plan)
    _write_json(sm / "plan_A.json", plan)
    _write_json(sm / "catalog.json", catalog)
    _install_win_timeout()
    ns = argparse.Namespace(outdir=str(sm), resume=False, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    _g39.execute(plan["cells"], ns, "smoke", loader=_loader_for(cfg, payload), runner=runner)
    ns2 = argparse.Namespace(resume=False, device=args.device, phase=None)
    cmd_probe0(ns2, cfg, sm, n_seeds=1)
    rows = list(_g35.load_done(sm / "runs.jsonl").values())
    probe = load_probe0(sm)
    ok = (len(rows) == plan["n_cells"] and all(r.get("error") is None and len(r.get("checkpoint_results") or []) == n_ck[r["key"]] for r in rows)
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
    for flag in ("slice", "sanity", "precheck", "plan", "run", "probe0", "verify", "check", "smoke", "analyze",
                 "slice-all", "plan-a", "analyze-a", "plan-b", "plan-w", "analyze-w", "plan-c", "analyze-c"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--stage", default="pilot", choices=("pilot", "full"))
    ap.add_argument("--profile", default=None, help="acts.yaml profiles.<name>（S1′ = s1p）")
    ap.add_argument("--phase", default=None, choices=(None, "A", "B", "W", "C"), help="S1′ 阶段 A / B；S1″ 阶段 W（整集口径对照）/ C（主批）（run / probe0 / verify 读 plan_<phase>.json）")
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
    cfg = ac.load_acts_config(args.config, profile=args.profile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.slice_all:
        cmd_slice_all(cfg, outdir)
    if args.plan_a:
        cmd_plan_a(cfg, outdir)
    if args.plan_b:
        cmd_plan_b(cfg, outdir)
    if args.plan_w:
        cmd_plan_w(cfg, outdir)
    if args.plan_c:
        cmd_plan_c(cfg, outdir)
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
        v = cmd_verify(cfg, outdir, phase=args.phase)
        if not v["ok"]:
            return 1
    if args.analyze_a:
        cmd_analyze_a(cfg, outdir)
    if args.analyze_w:
        cmd_analyze_w(cfg, outdir)
    if args.analyze_c:
        cmd_analyze_c(cfg, outdir)
    if args.check:
        if not check_deployment(cfg, outdir)["ok"]:
            return 1
    if args.analyze:
        if _is_cv(cfg):
            cmd_analyze_s1p(cfg, outdir)
        else:
            cmd_analyze(cfg, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
