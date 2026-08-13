"""SWA 噪声探针（协议 results/swa_probe/PROTOCOL.md，先于数据写死）。

评估"向 TS2Vec 官方修正的 SWA 权重平均"能否压 R' 跨 weight_seed 噪声：
与 labels_workpoint 噪声探针逐字同口径（2 数据集 × 前 2 任务 × 10 shared CLS ×
3 weight_seed），唯一变量 SWA off/on。off 臂兼作实现正确性检查——120 条须逐位
复现 labels_workpoint 既有值（同机 4060，否决条件 2）。

用法（幂等、--resume 贯穿；off 臂先跑）：
    python scripts/7_swa_noise_probe.py --run --resume
    python scripts/7_swa_noise_probe.py --analyze
"""
from __future__ import annotations

import os

# 造 label CPU-bound 教训：不设 OMP=1 慢 4.2×（须在 numpy/torch 导入前生效）
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402

from autocls.labeling import (  # noqa: E402
    evaluate_rprime, load_probe_cls, run_seed_for,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402

# 池加载/常量/canonical 种子/读 jsonl 原样复用 2c（模块名数字开头走 importlib）
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

DEFAULT_OUTDIR = ROOT / "results" / "swa_probe"
WORKPOINT_DIR = ROOT / "results" / "labels_workpoint"
# 信号侧不变（outlet_y §5 实测 |Δ| 中位）：required-k = (P95/容p50)²
SIGNAL_P50 = {"Handwriting": 0.033, "UniMiB-SHAR": 0.023}


def build_cells(per_ds: list, iters_map: dict, shared_cls: list) -> list[tuple]:
    """(name, X, y, task, cls, cls_key, rep, iters) × 前2任务 × 10 shared × rep 0-2。"""
    return [(name, X, y, t, c, k, rep, iters_map[name])
            for name, X, y, tasks in per_ds
            for t in tasks[:_c2.NOISE_TASKS]
            for c, k in shared_cls
            for rep in (0,) + _c2.NOISE_REPS]


def seeds_for(task_id: str, cls_key_: str, rep: int) -> tuple[int, int]:
    """rep0=canonical；rep1/2=噪声种子（split 仍 canonical）——与 2c 逐字一致。"""
    if rep == 0:
        return _c2.canonical_seeds(task_id, cls_key_)
    return (run_seed_for(task_id, f"{cls_key_}:noise{rep}"),
            run_seed_for(task_id, "train_eval_split"))


def run(outdir: Path, per_ds: list, iters_map: dict, shared_cls: list,
        train_cfg: dict, labels_cfg: dict, device: str, resume: bool) -> None:
    cache_path = outdir / "probe_cache.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"], r["swa"])
            for r in _c2.read_jsonl(cache_path) if r.get("r_prime") is not None}
    cells = build_cells(per_ds, iters_map, shared_cls)
    # off 臂先跑：逐位复现检查尽早出结果（否决条件 2 命中即全盘作废）
    work = [(swa, *cell) for swa in (False, True) for cell in cells]
    todo = [w for w in work
            if not (resume and (w[4].task_id, w[6], w[8], w[7], w[0]) in done)]
    print(f"SWA 探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：fit_svm 的 GridSearchCV 逐条 spawn loky 占 99% 耗时（长期教训）
    with parallel_backend("threading", n_jobs=1):
        for i, (swa, name, X, y, t, c, k, rep, iters) in enumerate(todo):
            w_seed, sp_seed = seeds_for(t.task_id, k, rep)

            def make_row(rec, _n=name, _t=t, _k=k, _c=c, _rep=rep, _it=iters):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(
                    t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                    iters=iters, eval_ratio=labels_cfg["eval_ratio"],
                    batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                    device=device, use_swa=swa),
                make_row, cache_path=cache_path, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=k, iters=iters,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": name, "noise_rep": rep, "swa": swa},
                log=lambda m: print(f"[swa {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[swa {i + 1}/{len(todo)}] swa={'on' if swa else 'off'} "
                  f"{name}/{t.task_id} {k[:8]} rep{rep} R'={full['r_prime']} "
                  f"({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 分析
def _p95(diffs: list[float]) -> float:
    """与 2c calibrate 同式：sorted 后取 round(0.95*(n-1)) 索引。"""
    ds = sorted(diffs)
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


def _pctl(diffs: list[float], q: float) -> float:
    ds = sorted(diffs)
    return ds[min(len(ds) - 1, int(round(q * (len(ds) - 1))))]


def load_cells(rows: list[dict]) -> dict:
    """(swa, dataset, task_id, cls_key) → {rep: r_prime}（同键后行覆盖前行）。"""
    cells: dict = defaultdict(dict)
    for r in rows:
        if r.get("r_prime") is None:
            continue
        cells[(r["swa"], r["dataset"], r["task_id"], r["cls_key"])][r["seed"]] = (
            float(r["r_prime"]))
    return cells


def bitwise_check(rows: list[dict], iters_map: dict) -> dict:
    """off 臂 120 条 ↔ labels_workpoint 既有值逐位比对（否决条件 2）。"""
    ref: dict = {}
    for r in _c2.read_jsonl(WORKPOINT_DIR / "rprime_cache.jsonl"):
        if r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], r["iters"], 0)] = float(r["r_prime"])
    for r in _c2.read_jsonl(WORKPOINT_DIR / "noise_probe.jsonl"):
        if r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], r["iters"], r["seed"])] = float(r["r_prime"])
    n_match = n_total = n_missing_ref = 0
    max_diff, mismatches = 0.0, []
    for r in rows:
        if r["swa"] or r.get("r_prime") is None:
            continue
        n_total += 1
        key = (r["task_id"], r["cls_key"], r["iters"], r["seed"])
        if key not in ref:
            n_missing_ref += 1
            continue
        d = abs(float(r["r_prime"]) - ref[key])
        max_diff = max(max_diff, d)
        if d == 0.0:
            n_match += 1
        else:
            mismatches.append({"key": list(key), "new": r["r_prime"], "ref": ref[key]})
    return {"n_off": n_total, "n_match": n_match, "n_missing_ref": n_missing_ref,
            "max_abs_diff": max_diff, "mismatches": mismatches[:20],
            "passed": n_total == 120 and n_match == 120}


def noise_stats(cells: dict, name: str, swa: bool) -> dict:
    diffs = []
    for (s, ds, _tid, _k), reps in cells.items():
        if s != swa or ds != name:
            continue
        vals = list(reps.values())
        diffs += [abs(vals[i] - vals[j])
                  for i in range(len(vals)) for j in range(i + 1, len(vals))]
    if not diffs:
        return {}
    import statistics
    return {"n_pairs": len(diffs), "median": statistics.median(diffs),
            "p90": _pctl(diffs, 0.90), "p95": _p95(diffs), "max": max(diffs)}


def calibrate_delta(rows: list[dict], cells: dict, name: str, swa: bool) -> dict:
    """2c calibrate 同式：δ = max(P95(重复对|ΔR'|), 2×量子) 上取整 0.005。"""
    st = noise_stats(cells, name, swa)
    quanta = [1.0 / r["n_eval"] for r in rows
              if r["swa"] == swa and r["dataset"] == name and r.get("n_eval")
              and r.get("r_prime") is not None]
    floor = 2.0 * max(quanta) if quanta else 0.0
    raw = max(st["p95"], floor)
    return {**st, "quantum_floor": floor,
            "delta": float(np.ceil(raw / 0.005) * 0.005),
            "required_k": float((st["p95"] / SIGNAL_P50[name]) ** 2)}


def gains(cells: dict, name: str, swa: bool, baseline: dict) -> dict:
    """rep0 口径的 gain（R'_canonical − 基线均值）：best_gain 与 ↑/↓（现行工作点 δ）。"""
    delta_wp = {"Handwriting": 0.110, "UniMiB-SHAR": 0.075}[name]
    out = []
    for (s, ds, tid, k), reps in cells.items():
        if s != swa or ds != name or 0 not in reps or tid not in baseline:
            continue
        out.append({"task_id": tid, "cls_key": k,
                    "gain": reps[0] - baseline[tid][0]})
    gs = [c["gain"] for c in out]
    return {"n_cells": len(out), "best_gain": max(gs) if gs else None,
            "up": sum(g > delta_wp for g in gs),
            "down": sum(g < -delta_wp for g in gs),
            "best_cell": max(out, key=lambda c: c["gain"]) if out else None}


def blind_ceiling(cells: dict, swa: bool, delta_map: dict) -> dict:
    """任务盲天花板（outlet_y 探针 cell 口径）：rep-mean、任务内 shared-shared 配对、
    |Δ|>δ 过滤、任务盲多数方向可达 acc。"""
    mean_r: dict = {}
    for (s, ds, tid, k), reps in cells.items():
        if s == swa:
            mean_r[(ds, tid, k)] = (sum(reps.values()) / len(reps), ds)
    by_task: dict = defaultdict(dict)
    for (ds, tid, k), (v, _) in mean_r.items():
        by_task[(ds, tid)][k] = v
    votes: dict = defaultdict(lambda: [0, 0])  # (ka,kb) 有序对 → [a胜, b胜]
    n_pairs = 0
    for (ds, _tid), ks in by_task.items():
        keys = sorted(ks)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = ks[keys[i]] - ks[keys[j]]
                if abs(d) <= delta_map.get(ds, 0.0):
                    continue
                votes[(keys[i], keys[j])][0 if d > 0 else 1] += 1
                n_pairs += 1
    correct = sum(max(v) for v in votes.values())
    return {"n_pairs": n_pairs, "n_cls_pairs": len(votes),
            "ceiling": correct / n_pairs if n_pairs else None}


def analyze(outdir: Path, iters_map: dict) -> dict:
    rows = _c2.read_jsonl(outdir / "probe_cache.jsonl")
    cells = load_cells(rows)
    res: dict = {"bitwise": bitwise_check(rows, iters_map)}
    for name in _c2.SPEC:
        res[name] = {
            "off": calibrate_delta(rows, cells, name, False),
            "on": calibrate_delta(rows, cells, name, True),
        }
        base = _c2.build_baseline_bands(name)
        res[name]["gain_off"] = gains(cells, name, False, base)
        res[name]["gain_on"] = gains(cells, name, True, base)
    d_new = {n: res[n]["on"]["delta"] for n in _c2.SPEC}
    res["ceiling"] = {
        "off_delta0": blind_ceiling(cells, False, {}),          # 校验：应≈0.8427
        "on_delta0": blind_ceiling(cells, True, {}),
        "on_delta_new": blind_ceiling(cells, True, d_new),
        "delta_new": d_new,
    }
    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    # ---- 判据打印（PROTOCOL.md 写死的阈值）----
    bw = res["bitwise"]
    print(f"\n[否决2·逐位复现] off 臂 {bw['n_match']}/{bw['n_off']} 逐位相同"
          f"（缺参考 {bw['n_missing_ref']}，max|Δ|={bw['max_abs_diff']:.6f}）"
          f" → {'PASS' if bw['passed'] else 'FAIL'}", flush=True)
    for name in _c2.SPEC:
        o, n = res[name]["off"], res[name]["on"]
        print(f"[{name}] 噪声 off: med={o['median']:.3f} P95={o['p95']:.3f} "
              f"max={o['max']:.3f} | on: med={n['median']:.3f} P95={n['p95']:.3f} "
              f"max={n['max']:.3f} | δ_on={n['delta']:.3f} "
              f"required-k={n['required_k']:.1f}", flush=True)
        go, gn = res[name]["gain_off"], res[name]["gain_on"]
        print(f"[{name}] best_gain off={go['best_gain']:+.3f} on={gn['best_gain']:+.3f}"
              f" | ↑/↓ off={go['up']}/{go['down']} on={gn['up']}/{gn['down']}", flush=True)
    hw_on = res["Handwriting"]["on"]["p95"]
    verdict = ("task-aware 有救 (<0.05)" if hw_on < 0.05
               else "部分改善，用户拍板 (0.05–0.08)" if hw_on <= 0.08
               else "SWA 不是出路 (>0.08)")
    go, gn = res["Handwriting"]["gain_off"], res["Handwriting"]["gain_on"]
    veto1 = (go["best_gain"] and gn["best_gain"] is not None
             and gn["best_gain"] < 0.7 * go["best_gain"])
    print(f"\n[主判据] HW P95_on={hw_on:.3f} → {verdict}", flush=True)
    print(f"[否决1·信号] on/off best_gain 比 = "
          f"{gn['best_gain'] / go['best_gain']:.2f}"
          f"（阈 0.70）→ {'FAIL(信号被压)' if veto1 else 'PASS'}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.analyze):
        ap.error("至少指定 --run 或 --analyze")

    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run:
        per_ds = _c2.load_pool(labels_cfg)
        shared_cls = load_probe_cls(
            _c2.PROBE_CATALOG, labels_cfg["route_a"]["n_shared_cls"])[0]
        run(args.outdir, per_ds, iters_map, shared_cls, train_cfg, labels_cfg,
            args.device, args.resume)
    if args.analyze:
        analyze(args.outdir, iters_map)


if __name__ == "__main__":
    main()
