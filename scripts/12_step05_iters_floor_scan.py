"""Step 0.5：轨迹噪声地板 × iters 扫描（协议 results/step05_iters_floor/PROTOCOL.md）。

排除"地板 = 欠训练"混淆因素：UM 前 6 任务 × 10 shared × 3 种子，臂 {60,120,240}
新跑（30 臂零成本读 step0 缓存），双轨读数（δ_nd 地板 + 少数票可及率），
2×2 判读（GO-iters / 噪声降信号同缩 / 地板刚性）先于数据写死。

用法（幂等、--resume 贯穿）：
    python scripts/12_step05_iters_floor_scan.py --run --resume
    python scripts/12_step05_iters_floor_scan.py --analyze
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # loky/OMP 教训，勿删

import argparse
import dataclasses
import importlib
import json
import statistics
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

from autocls.labeling import load_probe_cls  # noqa: E402
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402

probe11 = importlib.import_module("scripts.11_step0_um_expansion")
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

DEFAULT_OUTDIR = ROOT / "results" / "step05_iters_floor"
STEP0_DIR = probe11.DEFAULT_OUTDIR
DATASET = probe11.DATASET
REPS = probe11.REPS
ARMS_NEW = (60, 120, 240)
ARM_BASE = 30
N_TASKS = 6
# 判读常数（协议 §4，先于数据写死）
ND_SHARE_MIN = 0.60
N_PLUS_DELTA = 0.025
S_PLUS_REACH = 0.10


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def minority_gaps(votes) -> list:
    """每个 shared 无序对内取多数方向，返回少数方向票的 gap 列表。

    平票（pos×2 == n）按 sign=1 记多数（确定性规则，与 step0 诊断口径一致）。
    """
    per: dict = defaultdict(list)
    for v in votes:
        per[(v["a"], v["b"])].append(v)
    out = []
    for vs in per.values():
        pos = sum(v["sign"] for v in vs)
        maj = 1 if pos * 2 >= len(vs) else 0
        out += [v["gap"] for v in vs if v["sign"] != maj]
    return out


def reach_fraction(gaps, delta) -> tuple:
    """少数票中 gap ≥ δ 的比例（可及率）。空票返回 (None, 0)。"""
    if not gaps or delta is None:
        return None, len(gaps or [])
    n_ok = sum(1 for g in gaps if g >= delta)
    return n_ok / len(gaps), len(gaps)


def scan_verdict(per_arm: dict) -> dict:
    """2×2 判读（协议 §4）。per_arm={arm:{delta_nd, nd_share, reach}}。

    有效臂 = delta_nd 非空且 nd_share ≥ 0.60；全崩 → 以 60 臂为准（forced_60）。
    best_arm = 有效臂中 δ_nd 最低者；N+ = δ_nd ≤ 0.025；S+ = reach ≥ 0.10。
    """
    valid = {a: v for a, v in per_arm.items()
             if v.get("delta_nd") is not None
             and (v.get("nd_share") or 0.0) >= ND_SHARE_MIN}
    forced = False
    if not valid:
        if per_arm.get(60, {}).get("delta_nd") is None:
            return {"cell": "undecidable", "best_arm": None, "forced_60": False}
        valid, forced = {60: per_arm[60]}, True
    best = min(valid, key=lambda a: valid[a]["delta_nd"])
    n_plus = valid[best]["delta_nd"] <= N_PLUS_DELTA
    s_plus = (valid[best].get("reach") or 0.0) >= S_PLUS_REACH
    cell = ("GO_iters" if n_plus and s_plus
            else "noise_down_signal_shrinks" if n_plus
            else "floor_rigid")
    return {"cell": cell, "best_arm": best, "n_plus": n_plus, "s_plus": s_plus,
            "best_delta_nd": valid[best]["delta_nd"],
            "best_reach": valid[best].get("reach"), "forced_60": forced}


# ---------------------------------------------------------------- 采集
def run(outdir: Path, device: str, resume: bool) -> None:
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    n_shared = labels_cfg["route_a"]["n_shared_cls"]
    per_ds = [p for p in _c2.load_pool(labels_cfg) if p[0] == DATASET]
    _name, X, y, tasks_all = per_ds[0]
    tasks = tasks_all[:N_TASKS]
    shared_cls = load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]
    extras = probe11.build_extras(tasks, y, labels_cfg["eval_ratio"])
    # extras 与 step0 逐位同集断言（同种子派生；协议 §5.2）
    meta = json.loads((STEP0_DIR / "extras_meta.json").read_text(encoding="utf-8"))
    for t in tasks:
        assert len(extras[t.task_id]) == meta[t.task_id]["n_extra"], \
            f"{t.task_id} extras 与 step0 不一致"

    cache = outdir / "scan_cache.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in _c2.read_jsonl(cache) if r.get("r_prime") is not None}
    work = [(t, c, k, rep, arm) for arm in ARMS_NEW
            for t in tasks for c, k in shared_cls for rep in REPS]
    todo = [w for w in work
            if not (resume and (w[0].task_id, w[2], w[4], w[3]) in done)]
    print(f"step0.5 iters 扫描: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：loky spawn 99% 耗时教训，勿删
    with parallel_backend("threading", n_jobs=1):
        for i, (t, c, key, rep, arm) in enumerate(todo):
            w_seed, sp_seed = probe11.seeds_for(t.task_id, key, rep)

            def make_row(rec, _t=t, _k=key, _c=c, _rep=rep, _it=arm):
                rp = rec["r_prime"]
                return {"dataset": DATASET, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: probe11.eval_cell_two_readouts(
                    t, extras[t.task_id], c, X, y, weight_seed=w_seed,
                    split_seed=sp_seed, iters=arm,
                    eval_ratio=labels_cfg["eval_ratio"],
                    batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                    device=device),
                make_row, cache_path=cache, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=key, iters=arm,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": DATASET, "rep": rep, "arm": arm},
                log=lambda m: print(f"[scan {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[scan {i + 1}/{len(todo)}] {t.task_id} {key[:8]} rep{rep} "
                  f"@{arm} can={full.get('r_prime_can')} "
                  f"exp={full.get('r_prime_exp')} ({full['seconds']}s) "
                  f"ETA {eta / 60:.1f}min", flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 分析
def analyze(outdir: Path) -> dict:
    task_ids = [f"{DATASET}_{i:02d}" for i in range(N_TASKS)]
    rows30 = [r for r in _c2.read_jsonl(STEP0_DIR / "probe_cache.jsonl")
              if r["task_id"] in task_ids]
    scan_rows = _c2.read_jsonl(outdir / "scan_cache.jsonl")
    rows_by_arm = {ARM_BASE: rows30}
    for arm in ARMS_NEW:
        rows_by_arm[arm] = [r for r in scan_rows if r["iters"] == arm]
    base_rows = [r for r in _c2.read_jsonl(STEP0_DIR / "baseline_cache.jsonl")
                 if r["task_id"] in task_ids]
    bcells = probe11._cells(base_rows)
    base_stats = {}
    for (tid, _k), sv in bcells.items():
        vals = [float(sv[s]["r_prime_exp"]) for s in sorted(sv)]
        n_ev = int(next(iter(sv.values()))["n_eval_exp"])
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        base_stats[tid] = {"mean": statistics.fmean(vals),
                           "band": 2.0 * max(sd, 1.0 / n_ev)}

    res: dict = {"protocol": "results/step05_iters_floor/PROTOCOL.md",
                 "per_arm": {}}
    per_arm_verdict_in: dict = {}
    tt = "classification"
    for arm in (ARM_BASE, *ARMS_NEW):
        rows = rows_by_arm[arm]
        cells = probe11._cells(rows)
        if not cells:
            res["per_arm"][arm] = {"n_rows": len(rows), "empty": True}
            per_arm_verdict_in[arm] = {"delta_nd": None, "nd_share": 0.0,
                                       "reach": None}
            continue
        tt = next(iter(next(iter(cells.values())).values()))["task_type"]
        r_exp = {rep: defaultdict(dict) for rep in REPS}
        r_k3: dict = defaultdict(dict)
        for (tid, k), sv in cells.items():
            for rep, row in sv.items():
                r_exp[rep][tid][k] = float(row["r_prime_exp"])
            if len(sv) == 3:
                r_k3[tid][k] = float(np.mean(
                    [sv[s]["r_prime_exp"] for s in sorted(sv)]))
        strict = {(tid, k) for (tid, k) in cells
                  if (m := r_k3.get(tid, {}).get(k)) is not None
                  and tid in base_stats
                  and m < base_stats[tid]["mean"] - base_stats[tid]["band"]}

        def diffs(only_nd: bool) -> list:
            out = []
            for (tid, k), sv in cells.items():
                if only_nd and (tid, k) in strict:
                    continue
                v = [float(sv[s]["r_prime_exp"]) for s in sorted(sv)]
                out += [abs(v[i] - v[j]) for i in range(len(v))
                        for j in range(i + 1, len(v))]
            return out

        q = [1.0 / probe11.N_TARGET_EVAL]
        d_all = probe11.delta_from_diffs(diffs(False), q)
        d_nd = probe11.delta_from_diffs(diffs(True), q)
        votes, _nd, _nt = probe11.collect_votes(
            {t: dict(v) for t, v in r_k3.items()},
            sorted({k for _, k in cells}), lambda *_: 0.0, tt)
        ceil_val, n_votes = probe11.ceiling_from_votes(votes)
        gaps = minority_gaps(votes)
        reach, n_min = reach_fraction(gaps, d_nd["delta"])
        nd_share = 1.0 - len(strict) / len(cells)
        arm_out = {
            "n_rows": len(rows),
            "n_cells": len(cells),
            "n_complete_cells": sum(1 for sv in cells.values() if len(sv) == 3),
            "n_strict": len(strict), "nd_share": nd_share,
            "delta_all": d_all, "delta_nd": d_nd,
            "ceiling_k3_delta0": ceil_val, "n_votes": n_votes,
            "minority": {
                "n": n_min,
                "p50": float(np.percentile(gaps, 50)) if gaps else None,
                "p90": float(np.percentile(gaps, 90)) if gaps else None,
            },
            "reach": reach,
        }
        res["per_arm"][arm] = arm_out
        per_arm_verdict_in[arm] = {"delta_nd": d_nd["delta"],
                                   "nd_share": nd_share, "reach": reach}

    verdict = scan_verdict(per_arm_verdict_in)
    res["verdict"] = verdict

    # ---- 有效性闸（协议 §5）----
    keys = [(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in scan_rows]
    n_finite = sum(1 for r in scan_rows if r.get("r_prime") is not None)
    validity = {
        "n_scan_rows": len(scan_rows), "n_finite": n_finite,
        "n_dup_keys": len(keys) - len(set(keys)),
        "n_rows30": len(rows30),
        "rows30_expected": N_TASKS * 10 * 3,
        "n_eval_exp_ok": all(r.get("n_eval_exp") == probe11.N_TARGET_EVAL
                             for r in scan_rows if r.get("n_eval_exp")),
        "devices": sorted({r.get("device_name", "?") for r in scan_rows}),
    }
    validity["passed"] = (
        len(scan_rows) == 540 and n_finite >= 0.95 * 540
        and validity["n_dup_keys"] == 0
        and len(rows30) == N_TASKS * 10 * 3 and validity["n_eval_exp_ok"])
    res["validity"] = validity

    # ---- 预测对账（协议 §6）----
    dn = {a: res["per_arm"][a].get("delta_nd", {}).get("delta")
          for a in (ARM_BASE, *ARMS_NEW)}
    strict_share = {a: (res["per_arm"][a]["n_strict"] / res["per_arm"][a]["n_cells"])
                    for a in (ARM_BASE, *ARMS_NEW)
                    if res["per_arm"][a].get("n_cells")}
    mono_down = all(dn[a] is not None for a in (60, 120, 240)) and \
        dn[60] > dn[120] > dn[240]
    mono_strict = all(a in strict_share for a in (30, 60, 120, 240)) and \
        strict_share[30] <= strict_share[60] <= strict_share[120] <= strict_share[240]
    res["predictions"] = {
        "P1_delta_nd_monotone_down": {"values": dn, "hit": bool(mono_down)},
        "P2_delta_nd_240_band": {"value": dn.get(240),
                                 "hit": dn.get(240) is not None
                                 and 0.020 <= dn[240] <= 0.040},
        "P3_go_iters": {"hit": verdict["cell"] == "GO_iters"},
        "P4_strict_monotone_up": {"values": strict_share,
                                  "hit": bool(mono_strict)},
        "P5_validity": {"hit": bool(validity["passed"])},
    }

    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 控制台 ----
    print(f"\n[有效性] scan {n_finite}/{len(scan_rows)} finite "
          f"dup={validity['n_dup_keys']} 30臂读入 {len(rows30)}/"
          f"{validity['rows30_expected']} → "
          f"{'PASS' if validity['passed'] else 'FAIL'}", flush=True)
    for arm in (ARM_BASE, *ARMS_NEW):
        a = res["per_arm"][arm]
        if a.get("empty"):
            print(f"[臂 {arm}] 空", flush=True)
            continue
        print(f"[臂 {arm}] δ_nd={a['delta_nd']['delta']} "
              f"(med={a['delta_nd']['median']:.4f} P95={a['delta_nd']['p95']:.4f}) "
              f"δ_all={a['delta_all']['delta']} nd_share={a['nd_share']:.2f} "
              f"| E2 天花板={a['ceiling_k3_delta0']} 少数票 n={a['minority']['n']} "
              f"p50={a['minority']['p50']} 可及率={a['reach']}", flush=True)
    print(f"\n[判读] → {verdict}", flush=True)
    for name, p in res["predictions"].items():
        print(f"[预测 {name}] {'✓' if p['hit'] else '✗'} {p}", flush=True)
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
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.run:
        run(args.outdir, args.device, args.resume)
    if args.analyze:
        analyze(args.outdir)


if __name__ == "__main__":
    main()
