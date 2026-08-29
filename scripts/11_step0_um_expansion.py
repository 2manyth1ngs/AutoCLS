"""Step 0 探针：UniMiB 评估侧扩容 + 两制式 δ（协议 results/step0_um_eval_expansion/PROTOCOL.md）。

不换池不换 encoder，实测"n_eval 307→2000 + 破坏制式隔离"能否把任务盲天花板
压下来。判读表四格（A 尺子未兑现 / B UM 同质 / C GO / D 真 kill）先于数据写死，
范围声明：本探针只裁决 UM 内可及性，不得外推 task-aware 全局死刑。

用法（幂等、--resume 贯穿）：
    python scripts/11_step0_um_expansion.py --run --resume
    python scripts/11_step0_um_expansion.py --analyze
"""
from __future__ import annotations

import os

# 造 label CPU-bound 教训：不设 OMP=1 慢 4.2×（须在 numpy/torch 导入前生效）
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import importlib
import json
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
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
    device_label, load_probe_cls, run_seed_for, split_task_train_eval,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run, RprimeOOM, is_oom_error  # noqa: E402
from autocls.task_types import is_better  # noqa: E402

_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

DEFAULT_OUTDIR = ROOT / "results" / "step0_um_eval_expansion"
WORKPOINT_DIR = ROOT / "results" / "labels_workpoint"
DATASET = "UniMiB-SHAR"
REPS = (0, 1, 2)                 # rep0=canonical，rep1/2=2c noise 种子（k=3）
BASELINE_KEY = "__baseline__"    # iters=0 基线行的 cls_key（CLS 不参与，见协议 §2）
N_TARGET_EVAL = 2000             # expanded 总量（16/16 任务可达，协议 §1 余量实测）
DELTA_CAN_PROD = 0.075           # 今日生产 δ（labels.yaml workpoint，守卫的对照口径）
# 判读表阈值（协议 §4，先于数据写死）
E1_RATIO_BAND = (0.30, 0.55)
E1_DELTA_ND_MAX = 0.045
E2_HOMOGENEOUS = 0.98
E3_CEILING_MAX = 0.95
E3_GUARD_MIN = 0.95


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def allocate_largest_remainder(counts: dict, n: int) -> dict:
    """按 counts 占比最大余数法分配 n 个名额（封顶 counts；n≥总量则全给）。"""
    total = sum(counts.values())
    if total <= 0 or n <= 0:
        return {c: 0 for c in counts}
    if n >= total:
        return dict(counts)
    quotas = {c: n * cnt / total for c, cnt in counts.items()}
    alloc = {c: min(int(quotas[c]), counts[c]) for c in counts}
    rem = n - sum(alloc.values())
    order = sorted(counts, key=lambda c: (-(quotas[c] - int(quotas[c])), c))
    for c in order:
        if rem <= 0:
            break
        if alloc[c] < counts[c]:
            alloc[c] += 1
            rem -= 1
    while rem > 0:  # 余数轮被封顶吃掉时向余量最大的类溢出
        spill = [c for c in sorted(counts) if alloc[c] < counts[c]]
        if not spill:
            break
        c = max(spill, key=lambda c: counts[c] - alloc[c])
        alloc[c] += 1
        rem -= 1
    return alloc


def expand_eval_indices(
    y: np.ndarray, classes, sample_indices, n_extra: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """extras：eligible（y∈classes）− 任务样本，按 leftover 类占比分层确定性抽取。

    返回升序全局索引；与 sample_indices 零交集（协议 §2 断言在调用方复验）。
    """
    used = set(int(i) for i in sample_indices)
    leftover = {int(c): [int(i) for i in np.where(y == c)[0] if int(i) not in used]
                for c in classes}
    alloc = allocate_largest_remainder({c: len(v) for c, v in leftover.items()},
                                       n_extra)
    picked: list[int] = []
    for c in sorted(leftover):
        k = alloc.get(c, 0)
        if k > 0:
            picked += [int(v) for v in
                       rng.choice(leftover[c], size=k, replace=False)]
    return np.sort(np.array(picked, dtype=np.int64))


def delta_from_diffs(diffs, quanta) -> dict:
    """δ 公式（2c calibrate 逐字同式）：max(P95, 2×max 量子)，上取整到 0.005。"""
    ds = sorted(float(d) for d in diffs)
    if not ds:
        return {"n": 0, "p95": None, "floor": None, "delta": None, "median": None}
    p95 = ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]
    floor = 2.0 * max(quanta) if quanta else 0.0
    delta = float(np.ceil(max(p95, floor) / 0.005) * 0.005)
    return {"n": len(ds), "p95": p95, "floor": floor, "delta": delta,
            "median": statistics.median(ds)}


def collect_votes(rvals: dict, keys, delta_of, task_type: str):
    """任务内 shared 无序对投票。rvals={tid:{key:R'}}；delta_of(tid,a,b)→δ。

    返回 (votes, n_delta_dropped, n_tie_dropped)；vote={task_id,a,b,sign,gap}，
    sign=1 表示 a 优（is_better 语义，铁律 8）。
    """
    votes, n_delta, n_tie = [], 0, 0
    for tid in sorted(rvals):
        vals = rvals[tid]
        for a, b in combinations(sorted(keys), 2):
            ra, rb = vals.get(a), vals.get(b)
            if ra is None or rb is None:
                continue
            gap = abs(float(ra) - float(rb))
            if gap < delta_of(tid, a, b):
                n_delta += 1
                continue
            ab = is_better(ra, rb, task_type)
            ba = is_better(rb, ra, task_type)
            if not (ab or ba):
                n_tie += 1
                continue
            votes.append({"task_id": tid, "a": a, "b": b,
                          "sign": 1 if ab else 0, "gap": gap})
    return votes, n_delta, n_tie


def ceiling_from_votes(votes) -> tuple:
    """任务盲天花板（noise_structure_analysis.ceiling 同语义）：Σmax方向票/Σ票。"""
    per: dict = defaultdict(list)
    for v in votes:
        per[(v["a"], v["b"])].append(v["sign"])
    maj = tot = 0
    for signs in per.values():
        pos = sum(signs)
        maj += max(pos, len(signs) - pos)
        tot += len(signs)
    return (maj / tot if tot else None), tot


def all_same_sign_rate(triples) -> tuple:
    """守卫：每票三种子方向全同的比率。triple 含 None（缺值/平局）计不一致。"""
    if not triples:
        return None, 0
    ok = sum(1 for t in triples
             if None not in t and len(set(t)) == 1)
    return ok / len(triples), len(triples)


def verdict_cell(e1_pass: bool, e2_ceiling, e3_pass: bool) -> str:
    """判读表四格（协议 §4）。E2 为 None（票不足）并入 B（无可测分歧）。"""
    if not e1_pass:
        return "A_ruler_not_delivered"
    if e2_ceiling is None or e2_ceiling >= E2_HOMOGENEOUS:
        return "B_um_homogeneous"
    return "C_go" if e3_pass else "D_kill_within_reach"


# ---------------------------------------------------------------- 采集
def seeds_for(task_id: str, key: str, rep: int) -> tuple[int, int]:
    """2c 逐字同口径：rep0=canonical；rep1/2=噪声种子（split 恒 canonical）。"""
    if rep == 0:
        return _c2.canonical_seeds(task_id, key)
    return (run_seed_for(task_id, f"{key}:noise{rep}"),
            run_seed_for(task_id, "train_eval_split"))


def materialize_indices(X, y, task, idx: np.ndarray):
    """materialize_task 同语义的索引版（同 channels、同 classes 保序重映射）。"""
    Xe = X[idx][:, :, list(task.channels)].astype(np.float32, copy=False)
    remap = {orig: local for local, orig in enumerate(task.classes)}
    ye = np.array([remap[v] for v in y[idx]], dtype=np.int64)
    return Xe, ye


def eval_cell_two_readouts(task, extra_idx, cls_cfg, X, y, *, weight_seed: int,
                           split_seed: int, iters: int, eval_ratio: float,
                           batch_size: int, lr: float, device) -> dict:
    """训练一次、canonical + expanded 两个读数（协议 §2）。iters=0 即基线臂。"""
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.data.task_slicer import materialize_task
    from autocls.downstream.linear_eval import encode_features, fit_svm
    from autocls.encoders.registry import build_encoder

    assert not (set(int(i) for i in extra_idx)
                & set(int(i) for i in task.sample_indices)), "extras 与任务样本相交"
    Xt, yt = materialize_task(task, X, y)
    Xe, ye = materialize_indices(X, y, task, extra_idx)
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=eval_ratio, seed=split_seed)
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)
        encoder = build_encoder(input_dims=Xt.shape[-1])
        pretrain_encoder(encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
                         batch_size=batch_size, lr=lr, device=device,
                         seed=weight_seed)
        train_feats = encode_features(encoder, Xt[tr_idx], device=device)
        can_feats = encode_features(encoder, Xt[ev_idx], device=device)
        ext_feats = encode_features(encoder, Xe, device=device)
        clf = fit_svm(train_feats, yt[tr_idx], seed=weight_seed)
        c_can = (clf.predict(can_feats) == yt[ev_idx]).astype(int)
        c_ext = (clf.predict(ext_feats) == ye).astype(int)
        r_can = float(c_can.mean())
        r_exp = float((c_can.sum() + c_ext.sum()) / (len(c_can) + len(c_ext)))
        rec.update({
            "r_prime": r_can,          # 主键口径与既有缓存同（canonical），锚定用
            "r_prime_can": r_can, "r_prime_exp": r_exp,
            "svm_c": float(clf.C),
            "correct_can": "".join(str(int(v)) for v in c_can),
            "correct_ext": "".join(str(int(v)) for v in c_ext),
        })
    except Exception as e:  # noqa: BLE001 —— 训练发散记 NaN；OOM 抛出不落结论
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec.update({"n_train": int(len(tr_idx)), "n_eval_can": int(len(ev_idx)),
                "n_eval_exp": int(len(ev_idx) + len(extra_idx)),
                "weight_seed": int(weight_seed), "split_seed": int(split_seed),
                "device_name": device_label(device)})
    return rec


def build_extras(tasks, y, eval_ratio: float) -> dict:
    """{task_id: extras 索引}；确定性（run_seed_for(tid,"eval_expand")）。

    n_extra = N_TARGET_EVAL − 真实 canonical eval 大小（分层切分逐类取整，非
    round(n×0.3) 估值——保证 n_eval_exp 恰为 2000，有效性闸 §5.2 依赖）。
    """
    out = {}
    for t in tasks:
        remap = {orig: local for local, orig in enumerate(t.classes)}
        yt = np.array([remap[v] for v in y[np.array(t.sample_indices)]],
                      dtype=np.int64)
        sp_seed = _c2.canonical_seeds(t.task_id, "x")[1]
        _tr, ev = split_task_train_eval(yt, t.task_type,
                                        eval_ratio=eval_ratio, seed=sp_seed)
        rng = np.random.default_rng(run_seed_for(t.task_id, "eval_expand"))
        extras = expand_eval_indices(y, t.classes, t.sample_indices,
                                     N_TARGET_EVAL - int(len(ev)), rng)
        out[t.task_id] = extras
    return out


def run(outdir: Path, device: str, resume: bool) -> None:
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iters = labels_cfg["workpoint"]["iters_per_dataset"][DATASET]
    n_shared = labels_cfg["route_a"]["n_shared_cls"]
    per_ds = [p for p in _c2.load_pool(labels_cfg) if p[0] == DATASET]
    assert len(per_ds) == 1, "load_pool 未产出 UniMiB-SHAR"
    _name, X, y, tasks = per_ds[0]
    shared_cls = load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]
    extras = build_extras(tasks, y, labels_cfg["eval_ratio"])
    (outdir / "extras_meta.json").write_text(json.dumps({
        t.task_id: {"n_extra": int(len(extras[t.task_id])),
                    "classes": list(t.classes)} for t in tasks},
        indent=2, ensure_ascii=False), encoding="utf-8")

    cache = outdir / "probe_cache.jsonl"
    base_cache = outdir / "baseline_cache.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for p in (cache, base_cache) for r in _c2.read_jsonl(p)
            if r.get("r_prime") is not None}
    # 基线臂在前（快、且 regime 判定要用）；训练臂 rep 交错保证中断也有全任务覆盖
    work = [(t, None, BASELINE_KEY, rep, 0, base_cache) for t in tasks
            for rep in REPS]
    work += [(t, c, k, rep, iters, cache) for rep in REPS
             for t in tasks for c, k in shared_cls]
    todo = [w for w in work
            if not (resume and (w[0].task_id, w[2], w[4], w[3]) in done)]
    print(f"step0 扩容探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    dummy_cls = shared_cls[0][0]   # iters=0 时 CLS 不参与（既有单测断言）
    t0 = time.time()
    # threading 后端：fit_svm 的 GridSearchCV 逐条 spawn loky 占 99% 耗时（长期教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (t, c, key, rep, it, path) in enumerate(todo):
            if key == BASELINE_KEY:
                w_seed = run_seed_for(t.task_id, f"{BASELINE_KEY}:rep{rep}")
                sp_seed = _c2.canonical_seeds(t.task_id, key)[1]
                cfg = dummy_cls
            else:
                w_seed, sp_seed = seeds_for(t.task_id, key, rep)
                cfg = c

            def make_row(rec, _t=t, _k=key, _c=cfg, _rep=rep, _it=it):
                rp = rec["r_prime"]
                return {"dataset": DATASET, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": None if _k == BASELINE_KEY
                        else dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: eval_cell_two_readouts(
                    t, extras[t.task_id], cfg, X, y, weight_seed=w_seed,
                    split_seed=sp_seed, iters=it,
                    eval_ratio=labels_cfg["eval_ratio"],
                    batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                    device=device),
                make_row, cache_path=path, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=key, iters=it,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": DATASET, "rep": rep},
                log=lambda m: print(f"[step0 {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[step0 {i + 1}/{len(todo)}] {t.task_id} {key[:10]} rep{rep} "
                  f"can={full.get('r_prime_can')} exp={full.get('r_prime_exp')} "
                  f"({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 分析
def _cells(rows) -> dict:
    out: dict = defaultdict(dict)
    for r in rows:
        if r.get("r_prime_can") is not None and r.get("r_prime_exp") is not None \
                and r.get("error") is None:
            out[(r["task_id"], r["cls_key"])][r["seed"]] = r
    return out


def bitwise_anchor(rows) -> dict:
    """rep0 canonical vs rprime_cache（160 参考）+ rep1/2 vs noise_probe（40 参考）。"""
    ref: dict = {}
    for r in _c2.read_jsonl(WORKPOINT_DIR / "rprime_cache.jsonl"):
        if r.get("dataset") == DATASET and r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], 0)] = float(r["r_prime"])
    for r in _c2.read_jsonl(WORKPOINT_DIR / "noise_probe.jsonl"):
        if r.get("dataset") == DATASET and r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], r["seed"])] = float(r["r_prime"])
    n_match = n_total = 0
    max_diff, mism = 0.0, []
    for r in rows:
        if r.get("r_prime_can") is None or r["cls_key"] == BASELINE_KEY:
            continue
        key = (r["task_id"], r["cls_key"], r["seed"])
        if key not in ref:
            continue
        n_total += 1
        d = abs(float(r["r_prime_can"]) - ref[key])
        max_diff = max(max_diff, d)
        if d == 0.0:
            n_match += 1
        else:
            mism.append({"key": list(key), "new": r["r_prime_can"], "ref": ref[key]})
    return {"n_matched_refs": n_total, "n_bitwise": n_match,
            "max_abs_diff": max_diff, "mismatches": mism[:10],
            "passed": n_total > 0 and n_match >= 0.8 * n_total
            and max_diff <= 0.02}


def analyze(outdir: Path) -> dict:
    rows = _c2.read_jsonl(outdir / "probe_cache.jsonl")
    base_rows = _c2.read_jsonl(outdir / "baseline_cache.jsonl")
    cells = _cells(rows)
    bcells = _cells(base_rows)
    task_ids = sorted({tid for tid, _ in cells})
    keys = sorted({k for _, k in cells})
    res: dict = {"protocol": "results/step0_um_eval_expansion/PROTOCOL.md"}

    # ---- 有效性闸（协议 §5）----
    all_keys = [(r["task_id"], r["cls_key"], r["iters"], r["seed"])
                for r in rows + base_rows]
    n_eval_exp = {r["task_id"]: r.get("n_eval_exp") for r in rows
                  if r.get("n_eval_exp")}
    anchor = bitwise_anchor(rows)
    validity = {
        "n_trained_rows": len(rows), "n_baseline_rows": len(base_rows),
        "n_finite": sum(1 for r in rows if r.get("r_prime") is not None),
        "n_dup_keys": len(all_keys) - len(set(all_keys)),
        "n_tasks": len(task_ids),
        "cells_complete": all(len(sv) == 3 for sv in cells.values()),
        "bcells_complete": all(len(sv) == 3 for sv in bcells.values()),
        "n_eval_exp_all_target": all(v == N_TARGET_EVAL
                                     for v in n_eval_exp.values()),
        "anchor": anchor,
        "devices": sorted({r.get("device_name", "?") for r in rows}),
    }
    validity["passed"] = (
        validity["n_finite"] == 480 and validity["n_tasks"] == 16
        and validity["n_dup_keys"] == 0 and validity["cells_complete"]
        and validity["bcells_complete"] and validity["n_eval_exp_all_target"]
        and anchor["passed"])
    res["validity"] = validity

    # ---- 值表 ----
    def rv(readout: str, rep=None, k3=False) -> dict:
        out: dict = {tid: {} for tid in task_ids}
        for (tid, k), sv in cells.items():
            if k3:
                vals = [sv[s][readout] for s in sorted(sv)]
                if len(vals) == 3:
                    out[tid][k] = float(np.mean(vals))
            elif rep in sv:
                out[tid][k] = float(sv[rep][readout])
        return out

    r_can = {rep: rv("r_prime_can", rep=rep) for rep in REPS}
    r_exp = {rep: rv("r_prime_exp", rep=rep) for rep in REPS}
    r_exp_k3 = rv("r_prime_exp", k3=True)

    # ---- 噪声对与 δ 表（协议 §3）----
    def diffs(vals_by_rep) -> list:
        out = []
        for (tid, k), sv in cells.items():
            v = [vals_by_rep[rep][tid][k] for rep in REPS
                 if k in vals_by_rep[rep].get(tid, {})]
            out += [abs(v[i] - v[j]) for i in range(len(v))
                    for j in range(i + 1, len(v))]
        return out

    d_can_all, d_exp_all = diffs(r_can), diffs(r_exp)

    # regime：strict 破坏（k3 均值 exp < 基线均值 − 带；协议 §3）
    base_stats = {}
    for (tid, k), sv in bcells.items():
        vals = [float(sv[s]["r_prime_exp"]) for s in sorted(sv)]
        n_ev = int(next(iter(sv.values()))["n_eval_exp"])
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        base_stats[tid] = {"mean": statistics.fmean(vals),
                           "band": 2.0 * max(sd, 1.0 / n_ev), "n_eval": n_ev}
    strict: set = set()
    for (tid, k) in cells:
        m = r_exp_k3.get(tid, {}).get(k)
        b = base_stats.get(tid)
        if m is not None and b is not None and m < b["mean"] - b["band"]:
            strict.add((tid, k))

    def nd_diffs(vals_by_rep) -> list:
        out = []
        for (tid, k), sv in cells.items():
            if (tid, k) in strict:
                continue
            v = [vals_by_rep[rep][tid][k] for rep in REPS
                 if k in vals_by_rep[rep].get(tid, {})]
            out += [abs(v[i] - v[j]) for i in range(len(v))
                    for j in range(i + 1, len(v))]
        return out

    q_can = [1.0 / r["n_eval_can"] for r in rows if r.get("n_eval_can")]
    q_exp = [1.0 / r["n_eval_exp"] for r in rows if r.get("n_eval_exp")]
    delta_table = {
        "can_all": delta_from_diffs(d_can_all, q_can),
        "exp_all": delta_from_diffs(d_exp_all, q_exp),
        "can_nd": delta_from_diffs(nd_diffs(r_can), q_can),
        "exp_nd": delta_from_diffs(nd_diffs(r_exp), q_exp),
    }
    res["delta_table"] = delta_table
    res["regime"] = {"n_strict_cells": len(strict), "n_cells": len(cells),
                     "per_task_strict": {tid: sum(1 for (t, _k) in strict
                                                  if t == tid)
                                         for tid in task_ids}}

    # 标定锚（§5.4）：任务 00/01 子集 canonical δ vs 发布 0.075
    sub = [f"{DATASET}_00", f"{DATASET}_01"]
    d_sub = []
    for (tid, k), sv in cells.items():
        if tid in sub:
            v = [r_can[rep][tid][k] for rep in REPS
                 if k in r_can[rep].get(tid, {})]
            d_sub += [abs(v[i] - v[j]) for i in range(len(v))
                      for j in range(i + 1, len(v))]
    d_sub_stats = delta_from_diffs(d_sub, q_can)
    res["calib_anchor"] = {
        "delta_can_tasks01": d_sub_stats,
        "published": DELTA_CAN_PROD,
        "passed": (d_sub_stats["delta"] is not None
                   and abs(d_sub_stats["delta"] - DELTA_CAN_PROD) <= 0.02)}

    # ---- E1 尺子兑现 ----
    med_ratio = (statistics.median(d_exp_all) / statistics.median(d_can_all)
                 if d_can_all and statistics.median(d_can_all) > 0 else None)
    delta_nd_exp = delta_table["exp_nd"]["delta"]
    e1 = {"med_ratio": med_ratio, "ratio_band": E1_RATIO_BAND,
          "delta_nd_exp": delta_nd_exp, "delta_nd_max": E1_DELTA_ND_MAX}
    e1["passed"] = (med_ratio is not None and delta_nd_exp is not None
                    and E1_RATIO_BAND[0] <= med_ratio <= E1_RATIO_BAND[1]
                    and delta_nd_exp <= E1_DELTA_ND_MAX)
    res["E1"] = e1

    # ---- E2 信号存在性上界（k3 均值、expanded、δ=0）----
    tt = (next(iter(next(iter(cells.values())).values()))["task_type"]
          if cells else "classification")
    v2, nd2, nt2 = collect_votes(r_exp_k3, keys, lambda *_: 0.0, tt)
    c2_val, n2 = ceiling_from_votes(v2)
    res["E2"] = {"ceiling_k3_exp_delta0": c2_val, "n_votes": n2,
                 "n_tie_dropped": nt2, "homogeneous_line": E2_HOMOGENEOUS}

    # ---- E3 可及性（rep0 expanded、两制式 δ；守卫）----
    def two_regime_delta(tid, a, b) -> float:
        if (tid, a) in strict or (tid, b) in strict:
            return delta_table["exp_all"]["delta"] or 0.0
        return delta_nd_exp or 0.0

    v3, nd3, nt3 = collect_votes(r_exp[0], keys, two_regime_delta, tt)
    c3_val, n3 = ceiling_from_votes(v3)
    nd_votes = [v for v in v3 if (v["task_id"], v["a"]) not in strict
                and (v["task_id"], v["b"]) not in strict]
    c3_nd, n3_nd = ceiling_from_votes(nd_votes)
    # 守卫：新增存活 ND 票（today 口径会弃）三种子方向全同率
    triples = []
    n_new = 0
    for v in nd_votes:
        ca = r_can[0].get(v["task_id"], {}).get(v["a"])
        cb = r_can[0].get(v["task_id"], {}).get(v["b"])
        if ca is None or cb is None or abs(ca - cb) >= DELTA_CAN_PROD:
            continue
        n_new += 1
        tri = []
        for rep in REPS:
            ra = r_exp[rep].get(v["task_id"], {}).get(v["a"])
            rb = r_exp[rep].get(v["task_id"], {}).get(v["b"])
            if ra is None or rb is None or ra == rb:
                tri.append(None)
            else:
                tri.append(1 if is_better(ra, rb, tt) else 0)
        triples.append(tuple(tri))
    guard_rate, guard_n = all_same_sign_rate(triples)
    guard_pass = guard_rate is None or guard_rate >= E3_GUARD_MIN  # 空守卫=过（协议 §3）
    e3 = {"ceiling_two_regime": c3_val, "n_votes": n3,
          "ceiling_nd_only": c3_nd, "n_nd_votes": n3_nd,
          "n_delta_dropped": nd3, "n_tie_dropped": nt3,
          "guard_rate": guard_rate, "guard_n_new_votes": guard_n,
          "guard_passed": guard_pass}
    e3["passed"] = (c3_val is not None and c3_val < E3_CEILING_MAX and guard_pass)
    res["E3"] = e3

    # ---- 派生诊断（协议 §6，不判定）----
    v_old, _, _ = collect_votes(r_can[0], keys, lambda *_: DELTA_CAN_PROD, tt)
    c_old, n_old = ceiling_from_votes(v_old)
    v_k1, _, _ = collect_votes(r_exp[0], keys, lambda *_: 0.0, tt)
    c_k1, n_k1 = ceiling_from_votes(v_k1)
    res["diagnostics"] = {
        "continuity_ceiling_can_prod": {"ceiling": c_old, "n_votes": n_old,
                                        "published_um_internal": 0.9890},
        "ceiling_k1_exp_delta0": {"ceiling": c_k1, "n_votes": n_k1},
        "survival": {"two_regime_votes": n3, "today_votes": n_old,
                     "new_nd_votes": n_new},
    }

    # ---- 判读与预测对账 ----
    cell = verdict_cell(e1["passed"], res["E2"]["ceiling_k3_exp_delta0"],
                        e3["passed"])
    res["verdict"] = {"cell": cell}
    res["predictions"] = {
        "P1_ratio": {"value": med_ratio,
                     "hit": med_ratio is not None
                     and E1_RATIO_BAND[0] <= med_ratio <= E1_RATIO_BAND[1]},
        "P2_delta_nd": {"value": delta_nd_exp,
                        "hit": delta_nd_exp is not None
                        and delta_nd_exp <= E1_DELTA_ND_MAX},
        "P3_e2_band": {"value": c2_val,
                       "hit": c2_val is not None and 0.80 <= c2_val <= 0.97},
        "P4_e3_pass": {"hit": bool(e3["passed"])},
        "P5_validity": {"hit": bool(validity["passed"])},
    }

    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 控制台判读 ----
    a = anchor
    print(f"\n[有效性] 训练 {validity['n_finite']}/480 基线 {len(base_rows)}/48 "
          f"dup={validity['n_dup_keys']} 锚 {a['n_bitwise']}/{a['n_matched_refs']} "
          f"(max|Δ|={a['max_abs_diff']:.4f}) n_eval_exp 全 {N_TARGET_EVAL}: "
          f"{validity['n_eval_exp_all_target']} → "
          f"{'PASS' if validity['passed'] else 'FAIL'}", flush=True)
    for name, d in delta_table.items():
        if d["delta"] is not None:
            print(f"[δ {name}] n={d['n']} med={d['median']:.4f} "
                  f"P95={d['p95']:.4f} → δ={d['delta']:.3f}", flush=True)
    print(f"[regime] strict 破坏 cell {len(strict)}/{len(cells)}", flush=True)
    print(f"[E1] med 比={med_ratio if med_ratio is None else round(med_ratio, 3)} "
          f"(带 {E1_RATIO_BAND}) δ_nd(exp)={delta_nd_exp} (≤{E1_DELTA_ND_MAX}) "
          f"→ {'PASS' if e1['passed'] else 'FAIL'}", flush=True)
    print(f"[E2] k3/exp/δ=0 天花板={c2_val} (n={n2}, 同质线 {E2_HOMOGENEOUS})",
          flush=True)
    print(f"[E3] 两制式天花板={c3_val} (n={n3}) ND-only={c3_nd} (n={n3_nd}) "
          f"守卫={guard_rate} (新增票 {guard_n}) → "
          f"{'PASS' if e3['passed'] else 'FAIL'}", flush=True)
    dg = res["diagnostics"]
    print(f"[连续性] 今日口径天花板={dg['continuity_ceiling_can_prod']['ceiling']} "
          f"(发布 0.9890) | k1/exp/δ=0={c_k1}", flush=True)
    print(f"\n[判读] → {cell}", flush=True)
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
