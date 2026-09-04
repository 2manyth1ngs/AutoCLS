"""WP7-UCR：在 UCR 单变量语料上生成 (数据集 × 20 CLS × 截点 {60,120,290}) 的 R' 缓存 + iters=0 基线。

口径与 results/mx3_pq/rprime_cache.jsonl 逐位同（bs128 / lr 1e-3 / eval_ratio 0.3 / canonical seeds / swa 关 /
crop_mode fixed / SVM-RBF 评估），只换数据集：**整数据集 = 一个任务**（task_id = `<name>_00`），绕开切片器——
切片器的 n_samples 上限会静默截断且随机取通道；这里显式：classes = 全部类、channels = (0,)、
sample_indices = N ≤ n_cap 全取 / N > n_cap 按类分层 canonical 子采样（run_seed_for(name, "ucr_subsample:0")）。
自建两道闸（切片层原本代劳）：每类 ≥ 2 样本、n_eval ≥ MIN_N_EVAL。
注册表 configs/ucr_pool.yaml（由 results/ucr_survey/scripts/ucr_survey.py --aggregate 生成）；缓存目录 data/raw/ucr_cache。
复用 32 号的行合成 / resume 四态 / verify 计数（importlib），执行体 autocls.oom_guard.guarded_run_multi。
不跑噪声探针 / δ / 复现校验；不动 pq_common.REG_OF / PQ_DATASETS；产物 results/ucr_rprime/。

用法：
  python scripts/41_ucr_rprime.py --prefetch                    # 下载 .ts（Zenodo，可续）
  python scripts/41_ucr_rprime.py --tasks                       # 装载 + 构造整数据集任务 + 两道闸 → tasks.json（零 GPU）
  python scripts/41_ucr_rprime.py --fill --resume [--datasets A B] [--shard k/n] [--min-len 0 --max-len 600]
  python scripts/41_ucr_rprime.py --baseline --resume           # iters=0 × 5 种子
  python scripts/41_ucr_rprime.py --verify                      # 满格 / 零重复 / 零 error 判定
  python scripts/41_ucr_rprime.py --fill --smoke --datasets A   # 截点 4/8/12 → results/ucr_rprime_smoke
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("AUTOCLS_TRIPLET_CHUNK_ELEMS", "67108864")
os.environ.setdefault("AUTOCLS_INFONCE_CHUNK_ELEMS", "67108864")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import NamedTuple, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.baseline_probe import BASELINE_ITERS, baseline_seed_specs, inert_cls  # noqa: E402
from autocls.data.source_pool import load_registry, load_source_dataset  # noqa: E402
from autocls.data.task_slicer import SlicedTask  # noqa: E402
from autocls.labeling import cls_key, evaluate_rprime, evaluate_rprime_trajectory, run_seed_for  # noqa: E402
from autocls.oom_guard import guarded_run, guarded_run_multi  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
cache_index = _g32.cache_index
trajectory_state = _g32.trajectory_state
pending_cells = _g32.pending_cells
cell_states = _g32.cell_states
read_rows = _g32.read_rows

# ---------------------------------------------------------------- 常量
OUT = ROOT / "results" / "ucr_rprime"
POOL_YAML = ROOT / "configs" / "ucr_pool.yaml"
BUDGETS = tuple(pq.BUDGETS)                  # (60, 120, 290)
SMOKE_BUDGETS = (4, 8, 12)
MIN_N_EVAL = 50                              # 两道闸之二（PROTOCOL §3）
N_BASELINE_SEEDS = 5
EVAL_RATIO = 0.3
SUBSAMPLE_SEED = 0


class Cell(NamedTuple):
    slice_id: str
    origin: str
    cls: CLS
    cls_key: str


class Job(NamedTuple):
    dataset: str
    task: SlicedTask
    X: np.ndarray
    y: np.ndarray
    origin: str
    cls: CLS
    cls_key: str
    weight_seed: int
    split_seed: int
    seed: int = 0


# ---------------------------------------------------------------- 整数据集任务
def full_dataset_task(name: str, y: np.ndarray, *, n_cap: int, seed: int = SUBSAMPLE_SEED) -> SlicedTask:
    """整数据集 = 一个任务：全部类、单通道、N ≤ n_cap 全取 / 否则按类分层 canonical 子采样（最大余数法配额，索引升序）。"""
    y = np.asarray(y)
    N = len(y)
    classes = tuple(sorted(int(v) for v in np.unique(y)))
    if N <= n_cap:
        idx = tuple(range(N))
    else:
        rng = np.random.default_rng(run_seed_for(name, f"ucr_subsample:{int(seed)}"))
        counts = {c: int(np.sum(y == c)) for c in classes}
        raw = {c: n_cap * counts[c] / N for c in classes}
        quota = {c: int(np.floor(raw[c])) for c in classes}
        rem = n_cap - sum(quota.values())
        for c in sorted(classes, key=lambda c: -(raw[c] - quota[c]))[:rem]:
            quota[c] += 1
        chosen = []
        for c in classes:
            pool = np.flatnonzero(y == c)
            k = min(quota[c], len(pool))
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
        idx = tuple(sorted(int(i) for i in chosen))
    return SlicedTask(task_id=f"{name}_00", pool=name, classes=classes, channels=(0,),
                      sample_indices=idx, tag="full", task_type=TASK_CLASSIFICATION)


def task_gates(task: SlicedTask, y: np.ndarray, *, min_n_eval: int = MIN_N_EVAL,
               eval_ratio: float = EVAL_RATIO) -> int:
    """两道闸：每类 ≥ 2 样本（split_task_train_eval 前提）、n_eval ≥ min_n_eval。返回 n_eval（同分层公式）。"""
    yt = np.asarray(y)[list(task.sample_indices)]
    n_eval = 0
    for c in task.classes:
        n_c = int(np.sum(yt == c))
        if n_c < 2:
            raise ValueError(f"{task.task_id}: class {c} 只有 {n_c} 个样本（每类须 ≥ 2）")
        n_eval += min(max(int(round(n_c * eval_ratio)), 1), n_c - 1)
    if n_eval < min_n_eval:
        raise ValueError(f"{task.task_id}: n_eval={n_eval} < {min_n_eval}")
    return n_eval


def task_to_json(task: SlicedTask, *, dataset: str, n_total: int, n_eval: int) -> dict:
    d = dataclasses.asdict(task)
    for k in ("classes", "channels", "sample_indices"):
        d[k] = [int(v) for v in d[k]]
    d.update({"dataset": dataset, "n_total": int(n_total), "n_eval": int(n_eval),
              "n_task": len(task.sample_indices)})
    return d


def task_from_json(d: dict) -> SlicedTask:
    return SlicedTask(task_id=d["task_id"], pool=d["pool"], classes=tuple(int(c) for c in d["classes"]),
                      channels=tuple(int(c) for c in d["channels"]),
                      sample_indices=tuple(int(i) for i in d["sample_indices"]),
                      tag=d.get("tag", ""), task_type=d.get("task_type", TASK_CLASSIFICATION))


# ---------------------------------------------------------------- 注册表 / 计划
def registry(path: Path = POOL_YAML):
    return load_registry(path)


def dataset_names(reg) -> list[str]:
    return [e.name for e in reg.source]


def build_plan(names: Sequence[str], *, space: Optional[CLSSpace] = None) -> list[Cell]:
    """每数据集 20 条：pq_common.catalog_for_slice（10 shared 目录前 10 + 10 random 由 task_id 派生）。零数据加载。"""
    space = space or CLSSpace.from_yaml()
    shared = None
    cells: list[Cell] = []
    for name in names:
        sid = f"{name}_00"
        cat = pq.catalog_for_slice(space, sid, shared)
        if shared is None:
            shared = [(c, k) for o, c, k in cat if o == "shared"]
        cells.extend(Cell(sid, o, c, k) for o, c, k in cat)
    keys = [(c.slice_id, c.cls_key) for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError("计划里 (slice_id, cls_key) 撞键")
    return cells


def sharded(names: Sequence[str], shard: Optional[str]) -> list[str]:
    if not shard:
        return list(names)
    k, n = (int(x) for x in shard.split("/"))
    return [nm for i, nm in enumerate(names) if i % n == k]


# ---------------------------------------------------------------- 装载
def load_dataset(name: str, reg) -> tuple[np.ndarray, np.ndarray]:
    from scripts.candidates._ucr import UCR_CACHE
    return load_source_dataset(name, reg, cache_dir=UCR_CACHE)


def cmd_prefetch(names: Sequence[str]) -> None:
    from scripts.candidates._ucr import UCR_CACHE, _ensure_local
    for i, nm in enumerate(names):
        _ensure_local(nm, UCR_CACHE)
        print(f"[prefetch {i + 1}/{len(names)}] {nm} ok", flush=True)


def cmd_tasks(outdir: Path, names: Sequence[str], reg) -> dict:
    """装载 + 构造整数据集任务 + 两道闸 → tasks.json；闸不过的数据集记 rejected（不入网格）。"""
    tasks, rejected = {}, {}
    for nm in names:
        entry = reg.get(nm)
        X, y = load_dataset(nm, reg)
        t = full_dataset_task(nm, y, n_cap=reg.n_samples)
        try:
            n_eval = task_gates(t, y)
        except ValueError as e:
            rejected[nm] = str(e)
            print(f"[tasks] {nm}: REJECT {e}", flush=True)
            continue
        tasks[t.task_id] = task_to_json(t, dataset=nm, n_total=len(y), n_eval=n_eval)
        tasks[t.task_id].update({"T": int(X.shape[1]), "C": int(X.shape[2]), "crop_len": entry.crop_len})
        print(f"[tasks] {nm}: N={len(y)}→{len(t.sample_indices)} T={X.shape[1]} K={len(t.classes)} n_eval={n_eval}", flush=True)
    out = {"pool": list(tasks and [tasks[k]["dataset"] for k in tasks]), "budgets": list(BUDGETS),
           "n_cap": reg.n_samples, "eval_ratio": EVAL_RATIO, "min_n_eval": MIN_N_EVAL,
           "subsample_seed": SUBSAMPLE_SEED, "tasks": tasks, "rejected": rejected}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tasks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[tasks] {len(tasks)} 任务，{len(rejected)} 拒收 → {outdir / 'tasks.json'}", flush=True)
    return out


def load_tasks(outdir: Path) -> dict:
    return json.loads((outdir / "tasks.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- fill
def make_jobs(cells: Sequence[Cell], name: str, task: SlicedTask, X: np.ndarray, y: np.ndarray) -> list[Job]:
    jobs = []
    for c in cells:
        w_seed, sp_seed = pq.canonical_seeds(c.slice_id, c.cls_key)
        jobs.append(Job(name, task, X, y, c.origin, c.cls, c.cls_key, w_seed, sp_seed, 0))
    return jobs


def run_jobs(jobs: Sequence[Job], *, budgets: Sequence[int], cache_path: Path, inflight: Path, on_oom: str,
             oom_log: Path, device: str, label: str, batch_size: int = 128, lr: float = 1e-3) -> int:
    from joblib import parallel_backend
    if not jobs:
        return 0
    t0, n_oom = time.time(), 0
    budgets = tuple(int(b) for b in budgets)
    with parallel_backend("threading", n_jobs=1):        # Windows loky spawn 教训（36/32 同）
        for i, jb in enumerate(jobs):
            make_row = _g32.row_builder(dataset=jb.dataset, task=jb.task, cls_key=jb.cls_key, origin=jb.origin,
                                        cls_cfg=jb.cls, weight_seed=jb.weight_seed, seed=jb.seed)
            rows = guarded_run_multi(
                evaluate_rprime_trajectory, make_row, cache_path=cache_path, inflight_path=inflight,
                task_id=jb.task.task_id, cls_key=jb.cls_key, iters_list=budgets, on_oom=on_oom,
                oom_log_path=oom_log, context={"dataset": jb.dataset, "origin": jb.origin, "seed": jb.seed},
                log=lambda m, _i=i: print(f"[{label} {_i + 1}/{len(jobs)}] {m}", flush=True),
                task=jb.task, cls_cfg=jb.cls, X=jb.X, y=jb.y, weight_seed=jb.weight_seed,
                split_seed=jb.split_seed, checkpoints=budgets, eval_ratio=EVAL_RATIO,
                batch_size=batch_size, lr=lr, device=device)
            if rows is None:
                n_oom += 1
                continue
            vals = ["null" if r["r_prime"] is None else f"{r['r_prime']:.4f}" for r in rows]
            eta = (time.time() - t0) / (i + 1) * (len(jobs) - i - 1)
            print(f"[{label} {i + 1}/{len(jobs)}] {jb.dataset} {jb.origin}:{jb.cls_key[:8]} R'={'/'.join(vals)} "
                  f"({rows[-1].get('trajectory_seconds')}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"[{label}] 完成 {len(jobs)} 条轨迹，墙钟 {(time.time() - t0) / 60:.1f} min"
          + (f"，OOM {n_oom}（--resume 重试）" if n_oom else ""), flush=True)
    return n_oom


def cmd_fill(outdir: Path, names: Sequence[str], reg, *, budgets: Sequence[int], resume: bool, redo_partial: bool,
             on_oom: str, device: str, inflight: Optional[Path]) -> None:
    tasks = load_tasks(outdir)["tasks"]
    cache_path = outdir / "rprime_cache.jsonl"
    inflight = inflight or (outdir / "inflight.txt")
    oom_log = outdir / "oom_failures.jsonl"
    space = CLSSpace.from_yaml()
    index = cache_index(read_rows(cache_path))
    total_oom = 0
    for nm in names:
        sid = f"{nm}_00"
        if sid not in tasks:
            print(f"[fill] {nm} 不在 tasks.json（被闸拒收或未装载），跳过", flush=True)
            continue
        cells = build_plan([nm], space=space)
        todo = pending_cells(cells, index, budgets, resume=resume, redo_partial=redo_partial)
        if not todo:
            print(f"[fill] {nm}: 20/20 已完成", flush=True)
            continue
        X, y = load_dataset(nm, reg)
        task = task_from_json(tasks[sid])
        jobs = make_jobs(todo, nm, task, X, y)
        print(f"[fill] {nm}: 待跑 {len(jobs)}/20（N_task={len(task.sample_indices)} T={X.shape[1]}）", flush=True)
        total_oom += run_jobs(jobs, budgets=budgets, cache_path=cache_path, inflight=inflight, on_oom=on_oom,
                              oom_log=oom_log, device=device, label=f"fill {nm}")
    print(f"[fill] 全部结束，OOM 合计 {total_oom}", flush=True)


# ---------------------------------------------------------------- baseline（iters=0 × 5 种子）
def cmd_baseline(outdir: Path, names: Sequence[str], reg, *, resume: bool, device: str, on_oom: str,
                 n_seeds: int = N_BASELINE_SEEDS) -> None:
    from joblib import parallel_backend
    tasks = load_tasks(outdir)["tasks"]
    cache_path = outdir / "baseline_cache.jsonl"
    inflight = outdir / "inflight_baseline.txt"
    done = {(r["task_id"], int(r["seed"])) for r in read_rows(cache_path)}
    cls_cfg, key = inert_cls(), cls_key(inert_cls())
    with parallel_backend("threading", n_jobs=1):
        for nm in names:
            sid = f"{nm}_00"
            if sid not in tasks:
                continue
            todo = [s for s in range(n_seeds) if not (resume and (sid, s) in done)]
            if not todo:
                continue
            X, y = load_dataset(nm, reg)
            t = task_from_json(tasks[sid])
            for s in todo:
                w_seed, sp_seed = baseline_seed_specs(t.task_id, s)

                def make_row(rec, _t=t, _s=s, _n=nm):
                    rp = rec["r_prime"]
                    return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag, "task_type": _t.task_type,
                            "classes": list(_t.classes), "channels": list(_t.channels), "n_classes": len(_t.classes),
                            "cls_key": key, "seed": _s, "iters": BASELINE_ITERS, **rec,
                            "r_prime": None if not np.isfinite(rp) else float(rp),
                            "cls": dataclasses.asdict(cls_cfg)}

                full = guarded_run(
                    lambda _t=t, _w=w_seed, _sp=sp_seed: evaluate_rprime(
                        _t, cls_cfg, X, y, weight_seed=_w, split_seed=_sp, iters=BASELINE_ITERS, device=device),
                    make_row, cache_path=cache_path, inflight=inflight, task_id=t.task_id, cls_key=key,
                    iters=BASELINE_ITERS, on_oom=on_oom, oom_log=outdir / "oom_failures.jsonl",
                    context={"dataset": nm, "seed": s}, log=lambda m: print(f"[baseline] {m}", flush=True))
                if full is not None:
                    print(f"[baseline] {nm} s{s} R'={full['r_prime']} ({full['seconds']}s n_eval={full.get('n_eval')})",
                          flush=True)


# ---------------------------------------------------------------- verify
def cmd_verify(outdir: Path, names: Sequence[str], *, budgets: Sequence[int]) -> int:
    tasks = load_tasks(outdir)["tasks"]
    rows = read_rows(outdir / "rprime_cache.jsonl")
    keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
    dup = len(keys) - len(set(keys))
    errors = [r for r in rows if r.get("error")]
    index = cache_index(rows)
    space = CLSSpace.from_yaml()
    names = [nm for nm in names if f"{nm}_00" in tasks]
    cells = build_plan(names, space=space)
    st = cell_states(cells, index, budgets)
    base = read_rows(outdir / "baseline_cache.jsonl")
    base_n = defaultdict(int)
    for r in base:
        if r.get("r_prime") is not None:
            base_n[r["task_id"]] += 1
    base_short = [nm for nm in names if base_n[f"{nm}_00"] < N_BASELINE_SEEDS]
    per_ds = defaultdict(lambda: defaultdict(int))
    for k, cs in st.items():
        for c in cs:
            per_ds[c.slice_id][k] += 1
    print(f"[verify] 数据集 {len(names)}，格 {len(cells)}：done {len(st['done'])} partial {len(st['partial'])} "
          f"terminal {len(st['terminal'])} missing {len(st['missing'])}；行 {len(rows)} 重复键 {dup} error 行 {len(errors)}；"
          f"基线不足 5 种子：{len(base_short)}", flush=True)
    for sid, d in sorted(per_ds.items()):
        if d["done"] != 20:
            print(f"   {sid}: {dict(d)}", flush=True)
    ok = (len(st["done"]) == len(cells) and dup == 0 and not base_short)
    rep = {"n_datasets": len(names), "n_cells": len(cells), "states": {k: len(v) for k, v in st.items()},
           "n_rows": len(rows), "dup_keys": dup, "n_error_rows": len(errors), "baseline_short": base_short,
           "terminal": [f"{c.slice_id}:{c.cls_key}" for c in st["terminal"]], "pass": bool(ok)}
    (outdir / "verify.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="截点 4/8/12，写 results/ucr_rprime_smoke")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--redo-partial", action="store_true")
    ap.add_argument("--on-oom", choices=("skip", "record"), default="skip")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--pool", default=str(POOL_YAML))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--shard", default=None, help="k/n：按注册表顺序取第 k 个分片（pod 多进程）")
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--inflight", default=None)
    args = ap.parse_args(argv)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    budgets = BUDGETS
    if args.smoke:
        budgets = SMOKE_BUDGETS
        if outdir == OUT:
            outdir = ROOT / "results" / "ucr_rprime_smoke"
    reg = registry(Path(args.pool))
    names = args.datasets or dataset_names(reg)
    known = set(dataset_names(reg))
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"[warn] 不在 {args.pool} 的数据集被跳过（未过 usable 判据或未调查）：{unknown}", flush=True)
        names = [n for n in names if n in known]
    if args.min_len is not None or args.max_len is not None:
        names = [n for n in names if (args.min_len is None or (reg.get(n).l_enc or 0) >= args.min_len)
                 and (args.max_len is None or (reg.get(n).l_enc or 0) <= args.max_len)]
    names = sharded(names, args.shard)
    ran = False
    if args.prefetch:
        cmd_prefetch(names); ran = True
    if args.tasks:
        cmd_tasks(outdir, names, reg); ran = True
    if args.fill:
        cmd_fill(outdir, names, reg, budgets=budgets, resume=args.resume, redo_partial=args.redo_partial,
                 on_oom=args.on_oom, device=device, inflight=Path(args.inflight) if args.inflight else None); ran = True
    if args.baseline:
        cmd_baseline(outdir, names, reg, resume=args.resume, device=device, on_oom=args.on_oom); ran = True
    if args.verify:
        rc = cmd_verify(outdir, names, budgets=budgets); ran = True
        if rc:
            sys.exit(rc)
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
