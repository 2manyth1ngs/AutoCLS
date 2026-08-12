"""目标池两尺度学习空间预检：Epilepsy / FaceDetection（协议
results/target_preflight/PROTOCOL.md，判据先于数据 commit cbb8c94）。

UCI-HAR 终判不可评（且排查一在 test 侧直接验证坐实）后，目标池剩余两成员先过
本预检、预检过了才谈完整 zero-shot 评估投入：
- 尺度1（--prep + --calibrate）：通用切片 8 任务 + 40 随机基线 + 工作点标定
  {1,2,5,10,20}ep × 10 shared CLS × 前 2 任务 = 100 条 R'（判据 T1/T2/T3 与
  UCI-HAR 两轮逐字同源）。
- 尺度2（--fullscale）：全量口径 GGS 增益诊断（随机初始化 3 种子 vs GGS@10ep /
  @2320 iters），绕过切片可能的伪影——UCI-HAR 就死在这一步（+0.011/−0.004）。
- --verdict：三分判定（可评 / 边缘 / 不可评，过闸线 gain ≥ +0.03）。

切分协议（结构探针实证，PROTOCOL §2-3）：
- FaceDetection：train 按 10 个连续 589-trial 受试者块排列（8/9 边界 z 4.7-28.9）
  → 任务内组切分 + 全量 val 预注册留出块 {8,9}。
- Epilepsy：参与者 ID 已删且不可恢复 → 分层随机切分 + 局限声明；剔除 #104
  （#102 逐位重复，档案伪影，剔前 fail-fast 校验）；全量 val 5 切分种子取均值
  （n_val≈27 的 acc 量子 0.037 > 判据 0.03，必须平均压分辨率）。

铁律 6：仅读 *_TRAIN.ts，官方 TEST 全程 0 触碰；预检不需要任务嵌入 E → 共享
TS2Vec / T-CLSC 完全不碰。铁律 2/3：iters 显式、R' 逐条落盘、--resume 幂等。

用法（每数据集独立）：
    python scripts/6_target_preflight.py --dataset Epilepsy --all --resume
    python scripts/6_target_preflight.py --dataset FaceDetection --all --resume
    python scripts/6_target_preflight.py --dataset Epilepsy --verify
"""
from __future__ import annotations

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

from autocls.baseline_probe import (  # noqa: E402
    BEST_GAIN_FLOOR, CEILING_BASELINE, T3_ABOVE_FRACTION, T3_ABOVE_OVER_BELOW,
    baseline_seed_specs, inert_cls, summarize_dataset, summarize_task,
)
from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, TargetPoolLeakageError, load_registry, load_source_dataset,
    slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.data.uea import load_uea_cached  # noqa: E402
from autocls.downstream.linear_eval import encode_features, fit_svm  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    cls_key, device_label, evaluate_rprime, load_probe_cls, load_rprime_cache,
    run_seed_for, split_task_train_eval, split_train_eval_by_group,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402
from autocls.workpoint import (  # noqa: E402
    no_workpoint_verdict, select_workpoint, steps_per_epoch,
)

_m1 = importlib.import_module("scripts.m1_ggs_har")

# --- 预注册常数（PROTOCOL §4-5；与 5_target_zero_shot 逐字同源处注明）---
EP_GRID = (1, 2, 5, 10, 20)          # 标定网格（5_target 同款）
N_PROBE_TASKS = 2
N_SHARED_CLS = 10
N_BASELINE_SEEDS = 5
N_FULLBASE_SEEDS = 3                 # 尺度2 随机初始化 weight 种子数
VAL_RATIO = 0.2                      # 尺度2 val 比例（5_target VAL_RATIO 同值）
WORKPOINT_MARGIN = 0.02
FULLSCALE_GAIN_FLOOR = 0.03          # 尺度2 过闸线（PROTOCOL §4）
FULLSCALE_EP = 10                    # GGS 轻量档（UCI-HAR 诊断同款 10ep）
M1_FULL_ITERS = 2320                 # M1 anchor 同训练量档
EPI_DUP_PAIR = (102, 104)            # 结构探针实证的逐位重复对（保留前者）
EPI_SPLIT_SEEDS = (0, 1, 2, 3, 4)    # 尺度2 多切分平均（Epilepsy 专用）
FD_BLOCK = 589                       # FaceDetection 受试者块大小（10 × 589 = 5890）
FD_N_BLOCKS = 10
FD_VAL_BLOCKS = (8, 9)               # 尺度2 预注册留出块
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
OUT_BASE = ROOT / "results" / "target_preflight"
PREFLIGHT_DATASETS = ("Epilepsy", "FaceDetection")


# ---------------------------------------------------------------------------
# 可测助手（tests/test_target_preflight.py）
# ---------------------------------------------------------------------------


def dedup_epilepsy(X: np.ndarray, y: np.ndarray,
                   pair: tuple[int, int] = EPI_DUP_PAIR
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    """剔除预注册的逐位重复样本（保留 pair[0]、剔 pair[1]）。

    fail-fast：剔前校验该对余弦 >0.999——文件顺序若与结构探针时不同，拒绝剔除
    而非误伤正常样本。"""
    i, j = pair
    a, b = X[i].ravel().astype(np.float64), X[j].ravel().astype(np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    if cos <= 0.999:
        raise ValueError(
            f"Epilepsy 重复对校验失败：#{i}/#{j} 余弦 {cos:.4f} ≤ 0.999——"
            "文件顺序与结构探针时不一致，拒绝剔除（重跑 structure_probe.py 定位）")
    keep = np.ones(len(y), dtype=bool)
    keep[j] = False
    return X[keep], y[keep], {"dropped_index": j, "kept_index": i, "cosine": cos}


def fd_subject_groups(n: int) -> np.ndarray:
    """FaceDetection train 的受试者组向量：10 个连续 589 块（结构探针实证）。"""
    if n != FD_BLOCK * FD_N_BLOCKS:
        raise ValueError(f"FaceDetection train 应为 {FD_BLOCK * FD_N_BLOCKS} 例，"
                         f"得到 {n}——受试者块前提破坏，不得组切分")
    return np.arange(n) // FD_BLOCK


def fd_val_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """尺度2 的跨受试者 val：预注册留出块 FD_VAL_BLOCKS（PROTOCOL §3）。"""
    groups = fd_subject_groups(n)
    mask = np.isin(groups, FD_VAL_BLOCKS)
    return np.where(~mask)[0], np.where(mask)[0]


def mean_split_gain(rand_by_split: list[float], arm_by_split: list[float]) -> float:
    """逐切分配对相减再平均（多切分平均把 acc 量子压细，PROTOCOL §3）。"""
    if len(rand_by_split) != len(arm_by_split):
        raise ValueError(f"切分数不齐：{len(rand_by_split)} vs {len(arm_by_split)}")
    return float(np.mean(np.asarray(arm_by_split) - np.asarray(rand_by_split)))


def preflight_verdict(workpoint_found: bool, fullscale_gain: float) -> str:
    """三分判定（PROTOCOL §4，先于数据写死）。"""
    scale2 = fullscale_gain >= FULLSCALE_GAIN_FLOOR
    if workpoint_found and scale2:
        return "evaluable"
    if workpoint_found or scale2:
        return "borderline"
    return "unevaluable"


def grid_iters(spe: int) -> list[int]:
    return [ep * spe for ep in EP_GRID]


# ---------------------------------------------------------------------------
# 数据与切分（协议分发）
# ---------------------------------------------------------------------------


def load_target_train(name: str, reg) -> tuple[np.ndarray, np.ndarray, dict]:
    """目标专用加载路径：仅 *_TRAIN.ts（铁律 6：TEST 0 触碰）。"""
    entry = reg.get(name)
    assert entry.role == ROLE_TARGET, f"{name} 应为目标池成员"
    X, y = load_uea_cached(name, "TRAIN", crop_len=entry.crop_len)
    meta = {"n_loaded": int(len(y))}
    if name == "Epilepsy":
        X, y, info = dedup_epilepsy(X, y)
        meta["dedup"] = info
    return X, y, meta


def target_groups(name: str, n: int) -> np.ndarray | None:
    """组切分协议分发：FD 受试者块；Epilepsy 无组（分层随机 + 局限声明）。"""
    return fd_subject_groups(n) if name == "FaceDetection" else None


def task_split(name: str, task, yt: np.ndarray, groups: np.ndarray | None,
               eval_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    """任务内切分（canonical：seed 由 task_id 派生，与执行顺序无关）。"""
    if groups is not None:
        gidx = np.asarray(task.sample_indices)
        seed = run_seed_for(task.task_id, "subject_eval_split")
        tr, ev = split_train_eval_by_group(
            yt, groups[gidx], eval_ratio=eval_ratio, seed=seed)
    else:
        seed = run_seed_for(task.task_id, "train_eval_split")
        tr, ev = split_task_train_eval(
            yt, task.task_type, eval_ratio=eval_ratio, seed=seed)
    return tr, ev, seed


def target_tasks(reg, name: str, y: np.ndarray):
    entry = reg.get(name)
    _, tasks = slice_source_tasks(entry, reg, y)
    return tasks


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 被杀留下的半行
    return out


def ddir_of(name: str) -> Path:
    d = OUT_BASE / name.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_protocol_of(name: str) -> str:
    return "subject_block_disjoint" if name == "FaceDetection" else "stratified"


# ---------------------------------------------------------------------------
# 阶段 1：--prep 切任务 + 随机初始化基线（40 条）
# ---------------------------------------------------------------------------


def stage_prep(name: str, reg, labels_cfg: dict, device: str, resume: bool) -> None:
    ddir = ddir_of(name)
    # 泄漏自检①（铁律 6）：目标走源加载路径必须 fail-fast
    try:
        load_source_dataset(name, reg)
        sys.exit(f"泄漏自检失败：load_source_dataset({name}) 未 fail-fast！")
    except TargetPoolLeakageError:
        print(f"泄漏自检①通过：load_source_dataset({name}) fail-fast", flush=True)

    X, y, meta = load_target_train(name, reg)
    groups = target_groups(name, len(y))
    tasks = target_tasks(reg, name, y)
    eval_ratio = labels_cfg["eval_ratio"]
    print(f"{name}: X={X.shape} → {len(tasks)} 任务"
          f"（{split_protocol_of(name)}）", flush=True)

    split_cache: dict[str, tuple] = {}

    def split_of(t):
        if t.task_id not in split_cache:
            _Xt, yt = materialize_task(t, X, y)
            split_cache[t.task_id] = task_split(name, t, yt, groups, eval_ratio)
        return split_cache[t.task_id]

    task_meta = []
    for t in tasks:
        tr_idx, ev_idx, sp_seed = split_of(t)
        row = {**dataclasses.asdict(t), "n_train": len(tr_idx),
               "n_eval": len(ev_idx), "spe": steps_per_epoch(len(tr_idx)),
               "split_seed": sp_seed}
        if groups is not None:
            gidx = np.asarray(t.sample_indices)
            row["eval_blocks"] = sorted(set(groups[gidx][ev_idx].tolist()))
        task_meta.append(row)
    (ddir / "tasks.json").write_text(json.dumps({
        "dataset": name, "n_tasks": len(tasks), "load_meta": meta,
        "slice": "RandomClassSubsetSlicer",
        "split_protocol": split_protocol_of(name),
        "tasks": task_meta}, indent=2, ensure_ascii=False), encoding="utf-8")

    cache_path = ddir / "baseline_cache.jsonl"
    done = {(r["task_id"], r["seed"]) for r in read_jsonl(cache_path)}
    work = [(t, s) for t in tasks for s in range(N_BASELINE_SEEDS)]
    todo = [w for w in work if not (resume and (w[0].task_id, w[1]) in done)]
    print(f"基线: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    base_cls = inert_cls()

    # threading 后端：fit_svm 的 GridSearchCV loky spawn 占 99% 耗时教训（勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (t, s) in enumerate(todo):
            w_seed, _canon = baseline_seed_specs(t.task_id, s)
            tr_idx, ev_idx, sp_seed = split_of(t)

            def make_row(rec, _t=t, _s=s):
                rp = rec["r_prime"]
                return {"dataset": name, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": "baseline", "seed": _s, "iters": 0, **rec,
                        "split_protocol": split_protocol_of(name),
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            full = guarded_run(
                lambda: evaluate_rprime(t, base_cls, X, y,
                                        weight_seed=w_seed, split_seed=sp_seed,
                                        iters=0, eval_ratio=eval_ratio,
                                        split_indices=(tr_idx, ev_idx),
                                        device=device),
                make_row, cache_path=cache_path, inflight=ddir / "inflight.txt",
                task_id=t.task_id, cls_key="baseline", iters=0,
                on_oom=ON_OOM_SKIP, oom_log=ddir / "oom_failures.jsonl",
                context={"dataset": name, "baseline_seed": s},
                log=lambda m: print(f"[base {i + 1}/{len(todo)}] {m}", flush=True))
            if full is not None:
                print(f"[base {i + 1}/{len(todo)}] {t.task_id} s{s} "
                      f"R'={full['r_prime']}", flush=True)

    rows_by_task = defaultdict(list)
    for r in read_jsonl(cache_path):
        rows_by_task[r["task_id"]].append(r)
    n_classes_of = {t.task_id: len(t.classes) for t in tasks}
    per_task = {tid: summarize_task(rows, n_classes=n_classes_of[tid])
                for tid, rows in rows_by_task.items()}
    summary = {"dataset_name": name, "split_protocol": split_protocol_of(name),
               "per_task": per_task, "dataset": summarize_dataset(per_task)}
    (ddir / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    d = summary["dataset"]
    print(f"\n{name} 基线汇总: mean={d['baseline_mean']:.4f} "
          f"[{d['task_min']:.4f}, {d['task_max']:.4f}] tier={d['tier']}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 2：--calibrate 工作点标定（{1,2,5,10,20}ep × 10 shared × 前 2 任务）
# ---------------------------------------------------------------------------


def stage_calibrate(name: str, reg, labels_cfg: dict, train_cfg: dict,
                    device: str, resume: bool) -> None:
    ddir = ddir_of(name)
    X, y, _meta = load_target_train(name, reg)
    groups = target_groups(name, len(y))
    tasks = target_tasks(reg, name, y)
    probe = tasks[:N_PROBE_TASKS]
    eval_ratio = labels_cfg["eval_ratio"]

    split_cache: dict[str, tuple] = {}

    def split_of(t):
        if t.task_id not in split_cache:
            _Xt, yt = materialize_task(t, X, y)
            split_cache[t.task_id] = task_split(name, t, yt, groups, eval_ratio)
        return split_cache[t.task_id]

    spes = {steps_per_epoch(len(split_of(t)[0])) for t in probe}
    if len(spes) != 1:
        sys.exit(f"探针任务 steps/epoch 不唯一 {spes}，epoch 网格不可换算")
    spe = spes.pop()
    iters_grid = grid_iters(spe)
    shared_cls = load_probe_cls(PROBE_CATALOG, N_SHARED_CLS)[0]
    print(f"{name} 标定: spe={spe} → iters 网格 {iters_grid}"
          f"（= {list(EP_GRID)} ep）", flush=True)

    fill_path = ddir / "trained_fill.jsonl"
    cache = load_rprime_cache(fill_path)
    work = [(t, c, k, it) for it in iters_grid for t in probe for c, k in shared_cls]
    todo = [w for w in work if not (resume and (w[0].task_id, w[2], w[3]) in cache)]
    print(f"标定 fill: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    with parallel_backend("threading", n_jobs=1):   # loky 教训（勿删）
        for i, (t, c, k, it) in enumerate(todo):
            w_seed = run_seed_for(t.task_id, k)
            tr_idx, ev_idx, sp_seed = split_of(t)

            def make_row(rec, _t=t, _k=k, _c=c, _it=it):
                rp = rec["r_prime"]
                return {"dataset": name, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "origin": "shared", "seed": 0, "iters": _it,
                        **rec, "r_prime": None if not np.isfinite(rp) else float(rp),
                        "split_protocol": split_protocol_of(name),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed,
                                        split_seed=sp_seed, iters=it,
                                        eval_ratio=eval_ratio,
                                        batch_size=train_cfg["batch_size"],
                                        lr=train_cfg["lr"], device=device,
                                        split_indices=(tr_idx, ev_idx)),
                make_row, cache_path=fill_path, inflight=ddir / "inflight.txt",
                task_id=t.task_id, cls_key=k, iters=it,
                on_oom=ON_OOM_SKIP, oom_log=ddir / "oom_failures.jsonl",
                context={"dataset": name},
                log=lambda m: print(f"[cal {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            cache[(t.task_id, k, it)] = full
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[cal {i + 1}/{len(todo)}] {t.task_id}@{it} {k[:8]} "
                  f"R'={full['r_prime']:.4f} ({full['seconds']}s) "
                  f"ETA {eta / 60:.1f}min", flush=True)

    # 臂汇总 → 选点（5_target stage_calibrate 同式；判据常数同源）
    base_rows = defaultdict(list)
    for r in read_jsonl(ddir / "baseline_cache.jsonl"):
        if r["task_id"] in {t.task_id for t in probe}:
            base_rows[r["task_id"]].append(r)
    by_id = {t.task_id: t for t in probe}
    arm_rows = N_SHARED_CLS * N_PROBE_TASKS
    arms = {}
    for it in iters_grid:
        per_task = {}
        for tid, rows in base_rows.items():
            trained = [cache[(tid, k, it)]["r_prime"] for _c, k in shared_cls
                       if (tid, k, it) in cache]
            per_task[tid] = summarize_task(rows, n_classes=len(by_id[tid].classes),
                                           trained=trained)
        d = summarize_dataset(per_task)
        n_tr, n_up, n_dn = (d["n_trained_total"], d["n_trained_above_null"],
                            d["n_trained_below_null"])
        bg = d["best_gain_max"]
        arm = {
            "epoch": it / spe, "n_trained": n_tr, "complete": n_tr >= arm_rows,
            "baseline": d["baseline_mean"],
            "realized_gain": d["realized_gain_mean"], "best_gain": bg,
            "n_above": n_up, "n_below": n_dn, "n_band": n_tr - n_up - n_dn,
            "T1": d["baseline_mean"] <= CEILING_BASELINE,
            "T2": bool(bg is not None and bg >= BEST_GAIN_FLOOR),
            "T3": bool(n_tr > 0 and n_up >= T3_ABOVE_FRACTION * n_tr
                       and n_up >= T3_ABOVE_OVER_BELOW * n_dn),
        }
        arm["usable"] = bool(arm["T1"] and arm["T2"] and arm["T3"] and arm["complete"])
        arms[it] = arm

    complete = {it: a for it, a in arms.items() if a["complete"]}
    selection = select_workpoint(complete, margin=WORKPOINT_MARGIN)
    quantum = max((1.0 / r["n_eval"] for rows in base_rows.values() for r in rows
                   if r.get("n_eval")), default=None)
    nw = (no_workpoint_verdict(complete, quantum=quantum)
          if quantum and complete else None)
    wp = {"dataset": name, "spe": spe, "ep_grid": list(EP_GRID),
          "split_protocol": split_protocol_of(name),
          "arms": {str(k): v for k, v in arms.items()},
          "selection": selection, "no_workpoint": nw,
          "quantum": quantum, "workpoint_margin": WORKPOINT_MARGIN}
    (ddir / "workpoint.json").write_text(
        json.dumps(wp, indent=2, ensure_ascii=False), encoding="utf-8")
    for it, a in arms.items():
        print(f"  {it:>4} iters ({a['epoch']:.0f}ep): base={a['baseline']:.3f} "
              f"realized={a['realized_gain']:+.3f} best={a['best_gain']:+.3f} "
              f"↑{a['n_above']} ↓{a['n_below']} "
              f"T1{'✓' if a['T1'] else '✗'}T2{'✓' if a['T2'] else '✗'}"
              f"T3{'✓' if a['T3'] else '✗'}", flush=True)
    print(f"{name} 工作点: selection={selection} "
          f"no_workpoint={nw.get('verdict') if nw else None}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 3：--fullscale 全量 GGS 增益诊断（尺度2）
# ---------------------------------------------------------------------------


def _fullscale_splits(name: str, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """协议分发：FD 预注册留出块单切分；Epilepsy 5 切分种子（PROTOCOL §3）。"""
    if name == "FaceDetection":
        return [fd_val_split(len(y))]
    return [split_task_train_eval(y, "classification", eval_ratio=VAL_RATIO, seed=s)
            for s in EPI_SPLIT_SEEDS]


def ensure_pretrained_full(ddir: Path, cls_cfg, key: str, seed: int, iters: int,
                           X: np.ndarray, train_cfg: dict, device: str):
    """全量预训练 encoder 落盘复用（铁律 3：最贵一步只训一次）。"""
    p = ddir / "encoders" / f"{key}_it{iters}_s{seed}.pt"
    encoder = build_encoder(input_dims=X.shape[-1])
    if p.exists():
        encoder.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        return encoder, True
    torch.manual_seed(seed)
    encoder = build_encoder(input_dims=X.shape[-1])
    pretrain_encoder(encoder, cls_cfg, X, max_iters=iters,
                     batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                     device=device, seed=seed)
    p.parent.mkdir(exist_ok=True)
    torch.save(encoder.state_dict(), p)
    return encoder, False


def _accs_over_splits(encoder, X, y, splits, device, seed: int) -> list[float]:
    """encoder 编码一次，逐切分拟合 SVM 评 val acc。"""
    feats = encode_features(encoder, X, device=device)
    accs = []
    for fit_idx, val_idx in splits:
        clf = fit_svm(feats[fit_idx], y[fit_idx], seed=seed)
        accs.append(float(np.mean(clf.predict(feats[val_idx]) == y[val_idx])))
    return accs


def stage_fullscale(name: str, reg, train_cfg: dict, device: str) -> None:
    ddir = ddir_of(name)
    X, y, _meta = load_target_train(name, reg)
    splits = _fullscale_splits(name, y)
    spe_full = steps_per_epoch(len(y))
    ggs_k = cls_key(_m1.GGS)
    arms_iters = (FULLSCALE_EP * spe_full, M1_FULL_ITERS)
    out: dict = {"dataset": name, "n": int(len(y)), "spe_full": spe_full,
                 "split_protocol": split_protocol_of(name),
                 "n_splits": len(splits),
                 "split_sizes": [[int(len(a)), int(len(b))] for a, b in splits],
                 "val_blocks": list(FD_VAL_BLOCKS) if name == "FaceDetection" else None,
                 "split_seeds": list(EPI_SPLIT_SEEDS) if name == "Epilepsy" else None,
                 "device_name": device_label(device)}

    with parallel_backend("threading", n_jobs=1):   # loky 教训（勿删）
        rand_rows = []
        for s in range(N_FULLBASE_SEEDS):
            torch.manual_seed(s)
            enc = build_encoder(input_dims=X.shape[-1])
            accs = _accs_over_splits(enc, X, y, splits, device, seed=s)
            rand_rows.append({"seed": s, "accs_by_split": accs,
                              "mean": float(np.mean(accs))})
            print(f"[rand s{s}] {['%.4f' % a for a in accs]}", flush=True)
        # 逐切分对 weight 种子取均值 → 随机基线的 per-split 参考
        rand_by_split = np.mean([r["accs_by_split"] for r in rand_rows],
                                axis=0).tolist()

        arm_rows = []
        for iters in arms_iters:
            t0 = time.time()
            enc, cached = ensure_pretrained_full(
                ddir, _m1.GGS, ggs_k, 0, iters, X, train_cfg, device)
            accs = _accs_over_splits(enc, X, y, splits, device, seed=0)
            gain = mean_split_gain(rand_by_split, accs)
            arm_rows.append({"cls_key": ggs_k, "iters": iters,
                             "epoch_equiv": iters / spe_full,
                             "accs_by_split": accs, "mean": float(np.mean(accs)),
                             "gain_vs_random": gain, "encoder_cached": cached,
                             "seconds": round(time.time() - t0, 1)})
            print(f"[GGS@{iters}] mean={np.mean(accs):.4f} gain={gain:+.4f} "
                  f"({arm_rows[-1]['seconds']}s)", flush=True)

    out.update({
        "random_init": {"rows": rand_rows, "by_split": rand_by_split,
                        "mean": float(np.mean(rand_by_split))},
        "ggs_arms": arm_rows,
        "gain_max": max(a["gain_vs_random"] for a in arm_rows),
        "gain_floor": FULLSCALE_GAIN_FLOOR,
        "scale2_pass": bool(max(a["gain_vs_random"] for a in arm_rows)
                            >= FULLSCALE_GAIN_FLOOR),
    })
    (ddir / "fullscale_gain.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{name} 尺度2: random={out['random_init']['mean']:.4f} "
          f"gain_max={out['gain_max']:+.4f} pass={out['scale2_pass']}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 4：--verdict 三分判定
# ---------------------------------------------------------------------------


def stage_verdict(name: str) -> dict:
    ddir = ddir_of(name)
    wp = json.loads((ddir / "workpoint.json").read_text(encoding="utf-8"))
    fs = json.loads((ddir / "fullscale_gain.json").read_text(encoding="utf-8"))
    workpoint_found = wp["selection"] is not None
    verdict = preflight_verdict(workpoint_found, fs["gain_max"])
    out = {
        "dataset": name,
        "scale1_workpoint": wp["selection"],
        "scale1_no_workpoint": (wp["no_workpoint"] or {}).get("verdict"),
        "scale2_gain_max": fs["gain_max"],
        "scale2_pass": fs["scale2_pass"],
        "verdict": verdict,
    }
    (ddir / "verdict.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n===== {name} 预检判定: {verdict} =====\n"
          f"  尺度1 workpoint: {wp['selection']} "
          f"(no_workpoint={out['scale1_no_workpoint']})\n"
          f"  尺度2 gain_max: {fs['gain_max']:+.4f} "
          f"(≥{FULLSCALE_GAIN_FLOOR} → {fs['scale2_pass']})", flush=True)
    return out


# ---------------------------------------------------------------------------
# --verify 收官校验（种子幂等教训：重验行数/重复键/null）
# ---------------------------------------------------------------------------


def stage_verify(name: str, reg) -> None:
    ddir = ddir_of(name)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("✅ " if cond else "❌ ") + msg, flush=True)
        ok = ok and cond

    base = read_jsonl(ddir / "baseline_cache.jsonl")
    keys = [(r["task_id"], r["seed"]) for r in base]
    check(len(base) == 8 * N_BASELINE_SEEDS and len(set(keys)) == len(base),
          f"基线 {len(base)}/40 条、键唯一")
    check(all(r["r_prime"] is not None for r in base), "基线无 null R'")
    fill = read_jsonl(ddir / "trained_fill.jsonl")
    fkeys = [(r["task_id"], r["cls_key"], r["iters"]) for r in fill]
    check(len(fill) == len(EP_GRID) * N_SHARED_CLS * N_PROBE_TASKS
          and len(set(fkeys)) == len(fill),
          f"标定 fill {len(fill)}/100 条、键唯一")
    check(all(r.get("device_name") for r in base + fill), "device_name 100% 覆盖")
    try:
        load_source_dataset(name, reg)
        check(False, "load_source_dataset fail-fast")
    except TargetPoolLeakageError:
        check(True, "load_source_dataset fail-fast（铁律 6）")
    test_ts = ROOT / "data" / "uea_cache" / name / f"{name}_TEST.ts"
    check(test_ts.exists(), "TEST 文件在缓存中未被移动（加载路径只认 *_TRAIN.ts，0 触碰）")
    for f in ("workpoint.json", "fullscale_gain.json", "verdict.json"):
        check((ddir / f).exists(), f"{f} 落盘")
    print("verify:", "全绿" if ok else "有挂项", flush=True)
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=PREFLIGHT_DATASETS)
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--fullscale", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    reg = load_registry()
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    if args.all or args.prep:
        stage_prep(args.dataset, reg, labels_cfg, args.device, args.resume)
    if args.all or args.calibrate:
        stage_calibrate(args.dataset, reg, labels_cfg, train_cfg,
                        args.device, args.resume)
    if args.all or args.fullscale:
        stage_fullscale(args.dataset, reg, train_cfg, args.device)
    if args.all or args.verdict:
        stage_verdict(args.dataset)
    if args.verify:
        stage_verify(args.dataset, reg)


if __name__ == "__main__":
    main()
