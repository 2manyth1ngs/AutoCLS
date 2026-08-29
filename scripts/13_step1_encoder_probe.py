"""Step 1：encoder 换族探针（协议 results/step1_encoder_probe/PROTOCOL.md）。

transformer 四问（贴顶开了吗 / 学得动了吗 / 对照没坏吧 / 地板多低），单变量
实验：只换 encoder（经 build_encoder(config_path=局部 yaml) 注入，全局配置不动），
切片/评估栈/CLS 空间/种子机器全部冻结。判读表（transformer_closed / GO_full /
GO_narrow）先于数据写死。

用法（幂等、--resume 贯穿）：
    python scripts/13_step1_encoder_probe.py --run --resume
    python scripts/13_step1_encoder_probe.py --analyze
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

from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.labeling import (  # noqa: E402
    device_label, load_probe_cls, run_seed_for, split_task_train_eval,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run, RprimeOOM, is_oom_error  # noqa: E402

probe11 = importlib.import_module("scripts.11_step0_um_expansion")
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

DEFAULT_OUTDIR = ROOT / "results" / "step1_encoder_probe"
ENC_CFG = DEFAULT_OUTDIR / "encoder_transformer.yaml"
CNN_BASELINE_CACHE = ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl"
ENCODER_TAG = "transformer"

SIGNAL_DATASETS = ("SpokenArabicDigits", "MotorImagery", "Handwriting")
FLOOR_DATASET = "UniMiB-SHAR"
N_TASKS = 2
ARMS = (60, 240)
BASELINE_KEY = "__baseline__"
SIGNAL_BASE_REPS = (0, 1, 2, 3, 4)   # 5 种子零假设带
FLOOR_REPS = (0, 1, 2)               # k=3
# 判据常数（协议 §3，先于数据写死）
Q1_BASE_DROP = 0.05
Q1_N_ABOVE = 6
Q2_N_ABOVE, Q2_BEST_GAIN = 6, 0.05
Q3_N_ABOVE, Q3_BEST_GAIN = 12, 0.15
Q4_DELTA_ND, Q4_ND_SHARE = 0.015, 0.60


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def null_bands(base_vals: dict) -> dict:
    """{tid: [基线 R' 列表]} → {tid: (mean, band)}，band=2·max(σ, 量子)。"""
    out = {}
    for tid, entry in base_vals.items():
        vals, quantum = entry["values"], entry["quantum"]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[tid] = (statistics.fmean(vals), 2.0 * max(sd, quantum))
    return out


def count_above_band(cell_r: dict, bands: dict) -> dict:
    """cell_r={(tid,key):R'} → 越上带计数与 best gain（R'−基线均值 的最大值）。"""
    n_above, n_total, best = 0, 0, None
    for (tid, _k), r in cell_r.items():
        if tid not in bands or r is None:
            continue
        mean, band = bands[tid]
        n_total += 1
        gain = r - mean
        best = gain if best is None else max(best, gain)
        if r > mean + band:
            n_above += 1
    return {"n_above": n_above, "n_total": n_total, "best_gain": best}


def q_signal(per_arm: dict, *, n_above_min: int, best_gain_min: float) -> dict:
    """信号问共用门：某臂 越上带≥n 且 best_gain≥g。per_arm={arm:count_above_band 输出}。"""
    ok_arms = [a for a, s in per_arm.items()
               if s["n_above"] >= n_above_min
               and s["best_gain"] is not None and s["best_gain"] >= best_gain_min]
    return {"passed": bool(ok_arms), "ok_arms": sorted(ok_arms)}


def verdict_cell(q1: bool, q2: bool, q3: bool, q4: bool) -> str:
    """判读表（协议 §3）。"""
    if not (q3 and q4):
        return "transformer_closed"
    return "GO_full" if (q1 or q2) else "GO_narrow"


# ---------------------------------------------------------------- 采集
def probe_cell(task, cls_cfg, X, y, *, iters: int, weight_seed: int,
               split_seed: int, extras, eval_ratio: float, batch_size: int,
               lr: float, device) -> dict:
    """transformer 版 cell：canonical 读数恒有；extras 非 None 时加 expanded 读数。"""
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.data.task_slicer import materialize_task
    from autocls.downstream.linear_eval import encode_features, fit_svm
    from autocls.encoders.registry import build_encoder

    Xt, yt = materialize_task(task, X, y)
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=eval_ratio, seed=split_seed)
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)
        encoder = build_encoder(input_dims=Xt.shape[-1], config_path=ENC_CFG)
        pretrain_encoder(encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
                         batch_size=batch_size, lr=lr, device=device,
                         seed=weight_seed)
        train_feats = encode_features(encoder, Xt[tr_idx], device=device)
        can_feats = encode_features(encoder, Xt[ev_idx], device=device)
        clf = fit_svm(train_feats, yt[tr_idx], seed=weight_seed)
        c_can = (clf.predict(can_feats) == yt[ev_idx]).astype(int)
        rec.update({"r_prime": float(c_can.mean()),
                    "r_prime_can": float(c_can.mean()),
                    "svm_c": float(clf.C)})
        if extras is not None:
            Xe, ye = probe11.materialize_indices(X, y, task, extras)
            c_ext = (clf.predict(encode_features(encoder, Xe, device=device))
                     == ye).astype(int)
            rec["r_prime_exp"] = float(
                (c_can.sum() + c_ext.sum()) / (len(c_can) + len(c_ext)))
            rec["n_eval_exp"] = int(len(c_can) + len(c_ext))
    except Exception as e:  # noqa: BLE001
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec.update({"n_train": int(len(tr_idx)), "n_eval_can": int(len(ev_idx)),
                "weight_seed": int(weight_seed), "split_seed": int(split_seed),
                "device_name": device_label(device), "encoder": ENCODER_TAG})
    return rec


def load_probe_datasets(labels_cfg: dict):
    """(signal: {name:(X,y,tasks2)}, floor: (X,y,tasks2,extras))。"""
    reg = load_registry()
    signal = {}
    for name in SIGNAL_DATASETS:
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _, tasks = slice_source_tasks(entry, reg, y)
        signal[name] = (X, y, tasks[:N_TASKS])
        print(f"[{name}] X={X.shape} tasks={[t.task_id for t in tasks[:N_TASKS]]}",
              flush=True)
    per_um = [p for p in _c2.load_pool(labels_cfg) if p[0] == FLOOR_DATASET]
    _n, Xu, yu, tasks_u = per_um[0]
    tasks_u = tasks_u[:N_TASKS]
    extras = probe11.build_extras(tasks_u, yu, labels_cfg["eval_ratio"])
    meta = json.loads((probe11.DEFAULT_OUTDIR / "extras_meta.json")
                      .read_text(encoding="utf-8"))
    for t in tasks_u:
        assert len(extras[t.task_id]) == meta[t.task_id]["n_extra"], \
            f"{t.task_id} extras 与 step0 不一致"
    return signal, (Xu, yu, tasks_u, extras)


def run(outdir: Path, device: str, resume: bool) -> None:
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    signal, (Xu, yu, tasks_u, extras) = load_probe_datasets(labels_cfg)
    shared_cls = load_probe_cls(_c2.PROBE_CATALOG,
                                labels_cfg["route_a"]["n_shared_cls"])[0]
    dummy_cls = shared_cls[0][0]

    base_cache = outdir / "baseline_cache.jsonl"
    cache = outdir / "probe_cache.jsonl"
    done = {(r["dataset"], r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for p in (cache, base_cache) for r in _c2.read_jsonl(p)
            if r.get("r_prime") is not None}

    # (dataset, task, cls_cfg, cls_key, rep, iters, extras, role, cache_path)
    work = []
    for name, (X, y, tasks) in signal.items():
        for t in tasks:
            for rep in SIGNAL_BASE_REPS:
                work.append((name, X, y, t, dummy_cls, BASELINE_KEY, rep, 0,
                             None, "baseline", base_cache))
    for t in tasks_u:
        for rep in FLOOR_REPS:
            work.append((FLOOR_DATASET, Xu, yu, t, dummy_cls, BASELINE_KEY,
                         rep, 0, extras[t.task_id], "baseline", base_cache))
    for arm in ARMS:                       # 便宜臂先跑，中断也有全臂覆盖
        for name, (X, y, tasks) in signal.items():
            for t in tasks:
                for c, k in shared_cls:
                    work.append((name, X, y, t, c, k, 0, arm, None,
                                 "signal", cache))
        for t in tasks_u:
            for c, k in shared_cls:
                for rep in FLOOR_REPS:
                    work.append((FLOOR_DATASET, Xu, yu, t, c, k, rep, arm,
                                 extras[t.task_id], "floor", cache))

    todo = [w for w in work
            if not (resume and (w[0], w[3].task_id, w[5], w[7], w[6]) in done)]
    print(f"step1 探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    with parallel_backend("threading", n_jobs=1):   # loky spawn 教训，勿删
        for i, (name, X, y, t, c, key, rep, iters, ext, role, path) in enumerate(todo):
            if key == BASELINE_KEY:
                w_seed = run_seed_for(t.task_id, f"{BASELINE_KEY}:rep{rep}")
                sp_seed = _c2.canonical_seeds(t.task_id, key)[1]
            else:
                w_seed, sp_seed = probe11.seeds_for(t.task_id, key, rep)

            def make_row(rec, _n=name, _t=t, _k=key, _c=c, _rep=rep, _it=iters,
                         _role=role):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it,
                        "role": _role, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": None if _k == BASELINE_KEY
                        else dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: probe_cell(t, c, X, y, iters=iters, weight_seed=w_seed,
                                   split_seed=sp_seed, extras=ext,
                                   eval_ratio=labels_cfg["eval_ratio"],
                                   batch_size=train_cfg["batch_size"],
                                   lr=train_cfg["lr"], device=device),
                make_row, cache_path=path, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=key, iters=iters,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": name, "rep": rep, "role": role},
                log=lambda m: print(f"[s1 {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[s1 {i + 1}/{len(todo)}] {name}/{t.task_id} {key[:10]} "
                  f"rep{rep} @{iters} R'={full['r_prime']} "
                  f"({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 分析
def cnn_baseline_reference(task_map: dict) -> dict:
    """{tid: CNN 基线均值}（baseline_probe 缓存，classes 逐字段校验）。"""
    by_task = defaultdict(list)
    for r in _c2.read_jsonl(CNN_BASELINE_CACHE):
        tid = r.get("task_id")
        if tid in task_map and r.get("r_prime") is not None:
            if list(r.get("classes", [])) != list(task_map[tid]):
                continue
            by_task[tid].append(float(r["r_prime"]))
    return {tid: statistics.fmean(v) for tid, v in by_task.items() if v}


def analyze(outdir: Path) -> dict:
    base_rows = _c2.read_jsonl(outdir / "baseline_cache.jsonl")
    rows = _c2.read_jsonl(outdir / "probe_cache.jsonl")
    res: dict = {"protocol": "results/step1_encoder_probe/PROTOCOL.md"}

    # 零假设带（transformer 基线）
    base_vals: dict = defaultdict(lambda: {"values": [], "quantum": 0.0})
    task_classes: dict = {}
    for r in base_rows:
        if r.get("r_prime") is None:
            continue
        e = base_vals[r["task_id"]]
        e["values"].append(float(r["r_prime"]))
        e["quantum"] = max(e["quantum"], 1.0 / r["n_eval_can"])
        task_classes[r["task_id"]] = r["classes"]
    bands = null_bands(base_vals)
    cnn_ref = cnn_baseline_reference(
        {tid: cls for tid, cls in task_classes.items()
         if not tid.startswith(FLOOR_DATASET)})

    # ---- Q1–Q3（信号问）----
    sig = [r for r in rows if r.get("role") == "signal"
           and r.get("r_prime") is not None]
    per_ds_arm: dict = defaultdict(dict)
    for name in SIGNAL_DATASETS:
        for arm in ARMS:
            cell_r = {(r["task_id"], r["cls_key"]): float(r["r_prime"])
                      for r in sig if r["dataset"] == name and r["iters"] == arm}
            per_ds_arm[name][arm] = count_above_band(cell_r, bands)
    tf_base = {name: statistics.fmean(
        [v for tid, e in base_vals.items() if tid.startswith(name)
         for v in e["values"]]) for name in SIGNAL_DATASETS}
    cnn_base = {name: (statistics.fmean(
        [m for tid, m in cnn_ref.items() if tid.startswith(name)])
        if any(tid.startswith(name) for tid in cnn_ref) else None)
        for name in SIGNAL_DATASETS}

    sad, mi, hw = SIGNAL_DATASETS
    q1a = (cnn_base[sad] is not None
           and tf_base[sad] <= cnn_base[sad] - Q1_BASE_DROP)
    q1b = q_signal(per_ds_arm[sad], n_above_min=Q1_N_ABOVE, best_gain_min=0.0)
    q1 = q1a and q1b["passed"]
    q2g = q_signal(per_ds_arm[mi], n_above_min=Q2_N_ABOVE,
                   best_gain_min=Q2_BEST_GAIN)
    q3g = q_signal(per_ds_arm[hw], n_above_min=Q3_N_ABOVE,
                   best_gain_min=Q3_BEST_GAIN)
    res["Q1"] = {"tf_base": tf_base[sad], "cnn_base": cnn_base[sad],
                 "q1a": q1a, "q1b": q1b, "per_arm": per_ds_arm[sad],
                 "passed": q1}
    res["Q2"] = {"tf_base": tf_base[mi], "cnn_base": cnn_base[mi],
                 "per_arm": per_ds_arm[mi], **q2g}
    res["Q3"] = {"tf_base": tf_base[hw], "cnn_base": cnn_base[hw],
                 "per_arm": per_ds_arm[hw], **q3g}

    # ---- Q4（地板问，step0 同式）----
    floor_rows = [r for r in rows if r.get("role") == "floor"]
    ub = {tid: e for tid, e in base_vals.items()
          if tid.startswith(FLOOR_DATASET)}
    base_exp: dict = defaultdict(list)
    for r in base_rows:
        if r["task_id"].startswith(FLOOR_DATASET) and r.get("r_prime_exp") is not None:
            base_exp[r["task_id"]].append(float(r["r_prime_exp"]))
    ub_stats = {tid: (statistics.fmean(v),
                      2.0 * max(statistics.stdev(v) if len(v) > 1 else 0.0,
                                1.0 / probe11.N_TARGET_EVAL))
                for tid, v in base_exp.items()}
    q4_arms: dict = {}
    for arm in ARMS:
        cells = probe11._cells([r for r in floor_rows if r["iters"] == arm])
        if not cells:
            q4_arms[arm] = {"delta_nd": None, "nd_share": 0.0}
            continue
        k3 = {ck: float(np.mean([sv[s]["r_prime_exp"] for s in sorted(sv)]))
              for ck, sv in cells.items() if len(sv) == 3}
        strict = {ck for ck, m in k3.items()
                  if ck[0] in ub_stats
                  and m < ub_stats[ck[0]][0] - ub_stats[ck[0]][1]}
        diffs = []
        for ck, sv in cells.items():
            if ck in strict:
                continue
            v = [float(sv[s]["r_prime_exp"]) for s in sorted(sv)]
            diffs += [abs(v[i] - v[j]) for i in range(len(v))
                      for j in range(i + 1, len(v))]
        d = probe11.delta_from_diffs(diffs, [1.0 / probe11.N_TARGET_EVAL])
        q4_arms[arm] = {"delta_nd": d["delta"], "p95": d["p95"],
                        "median": d["median"],
                        "nd_share": 1.0 - len(strict) / len(cells),
                        "n_cells": len(cells), "n_strict": len(strict)}
    q4 = any(a.get("delta_nd") is not None and a["delta_nd"] <= Q4_DELTA_ND
             and a.get("nd_share", 0) >= Q4_ND_SHARE for a in q4_arms.values())
    res["Q4"] = {"per_arm": q4_arms, "passed": q4}

    cell = verdict_cell(q1, q2g["passed"], q3g["passed"], q4)
    res["verdict"] = {"cell": cell, "Q1": q1, "Q2": q2g["passed"],
                      "Q3": q3g["passed"], "Q4": q4}

    # ---- 有效性闸（协议 §4）----
    all_rows = rows + base_rows
    keys = [(r["dataset"], r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in all_rows]
    n_finite = sum(1 for r in all_rows if r.get("r_prime") is not None)
    validity = {
        "n_rows": len(all_rows), "n_baseline": len(base_rows),
        "n_signal": len(sig),
        "n_floor": sum(1 for r in floor_rows if r.get("r_prime") is not None),
        "n_finite": n_finite,
        "n_dup_keys": len(keys) - len(set(keys)),
        "cnn_ref_matched": sorted(cnn_ref),
        "encoder_uniform": all(r.get("encoder") == ENCODER_TAG for r in all_rows),
        "devices": sorted({r.get("device_name", "?") for r in all_rows}),
    }
    validity["passed"] = (
        len(all_rows) == 276 and n_finite >= 0.95 * 276
        and validity["n_dup_keys"] == 0 and validity["encoder_uniform"]
        and len(cnn_ref) == 6)
    res["validity"] = validity

    res["predictions"] = {
        "P1_q1": {"hit": q1}, "P2_q2": {"hit": q2g["passed"]},
        "P3_q3": {"hit": q3g["passed"]}, "P4_q4": {"hit": q4},
        "P5_validity": {"hit": bool(validity["passed"])},
    }
    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 控制台 ----
    print(f"\n[有效性] {n_finite}/{len(all_rows)} finite (期望 276) "
          f"dup={validity['n_dup_keys']} CNN参照 {len(cnn_ref)}/6 → "
          f"{'PASS' if validity['passed'] else 'FAIL'}", flush=True)
    for name in SIGNAL_DATASETS:
        r = res[{sad: 'Q1', mi: 'Q2', hw: 'Q3'}[name]]
        arms_txt = {a: f"{s['n_above']}/{s['n_total']}↑ best={s['best_gain'] if s['best_gain'] is None else round(s['best_gain'], 3)}"
                    for a, s in per_ds_arm[name].items()}
        print(f"[{name}] tf基线={tf_base[name]:.3f} cnn基线="
              f"{cnn_base[name] if cnn_base[name] is None else round(cnn_base[name], 3)} "
              f"| {arms_txt} → {'PASS' if r['passed'] else 'FAIL'}", flush=True)
    for arm, a in q4_arms.items():
        print(f"[Q4 臂{arm}] δ_nd={a.get('delta_nd')} nd_share="
              f"{a.get('nd_share')} (门 ≤{Q4_DELTA_ND} 且 ≥{Q4_ND_SHARE})",
              flush=True)
    print(f"\n[判读] → {res['verdict']}", flush=True)
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
