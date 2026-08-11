"""决定性实验：工作点口径造 T-CLSC 训练数据（协议 results/decisive_e_ablation/PROTOCOL.md）。

与 2b（路线 A，全局 early_iters=290）的差异——除此之外骨架同款、能复用尽复用：
1. **per-dataset iters**（labels.yaml: workpoint.iters_per_dataset，用户拍板
   "full"=工作点训练量）：Handwriting 60、UniMiB-SHAR 30。iters 入缓存键（铁律 2/3）。
2. **双注册表**：Handwriting 在 configs/source_pool.yaml、UniMiB-SHAR 在
   configs/candidate_pool.yaml——池组成继承 workpoint confirm 的源池判定
   （28 任务），不合并注册表（两文件各是各池的唯一真源）。
3. **既有记录幂等种入**：confirm/followup 已跑的工作点 R'（HW@60、UniMiB@30 各
   60 条）按键增补 append-only，种入前逐条校验 classes + split_seed = canonical
   （命令重放事故教训：绝不截断，收官 --verify 重验行数/重复键）。
4. **δ 工作点重标定**（协议 §2 公式先于数据）：k=3 噪声探针（每数据集任务 00–01 ×
   10 shared × 2 额外 weight seed）→ δ = max(P95(同 CLS 重复对 |ΔR'|), 2×量子)
   上取整到 0.005。标定值人工回填 labels.yaml workpoint.delta_per_dataset 后
   --pairs 才放行（fail-fast，防拿 290 口径 δ 配工作点 label）。

用法（各阶段幂等、--resume 贯穿；顺序：seed → noise-probe → calibrate →
（回填 δ）→ fill → features → pairs）：
    python scripts/2c_generate_labels_workpoint.py --seed-existing
    python scripts/2c_generate_labels_workpoint.py --noise-probe --resume
    python scripts/2c_generate_labels_workpoint.py --calibrate
    python scripts/2c_generate_labels_workpoint.py --fill --resume
    python scripts/2c_generate_labels_workpoint.py --features
    python scripts/2c_generate_labels_workpoint.py --pairs
    python scripts/2c_generate_labels_workpoint.py --verify
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

from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.labeling import (  # noqa: E402
    ORIGIN_RANDOM, ORIGIN_SHARED, append_jsonl, cls_key, evaluate_rprime,
    load_probe_cls, load_rprime_cache, make_pairs, run_seed_for,
    sample_random_cls,
)
from joblib import parallel_backend  # noqa: E402

from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.workpoint import destruction_participation  # noqa: E402

# 2b 的 TS2Vec 池构建/加载、SetTF init、任务特征、origin 补丁原样复用（模块名以
# 数字开头，走 importlib）。
_r2b = importlib.import_module("scripts.2b_generate_labels_route_a")

PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
DEFAULT_OUTDIR = ROOT / "results" / "labels_workpoint"

# 数据集 → (注册表, 既有工作点记录来源, 基线缓存)。注册表 None = source_pool.yaml。
SPEC = {
    "Handwriting": {
        "registry": None,
        "seed_sources": ["results/workpoint_confirm/uea/trained_fill.jsonl"],
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
    },
    "UniMiB-SHAR": {
        "registry": "configs/candidate_pool.yaml",
        "seed_sources": [
            "results/workpoint_followup/candidates/trained_fill.jsonl",
            "results/iters_workpoint/candidates/trained_fill.jsonl",
        ],
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
    },
}
NOISE_TASKS = 2          # 协议 §2：每数据集前 2 任务
NOISE_REPS = (1, 2)      # canonical(=0) 之外的 2 个额外 weight seed → k=3


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


def canonical_seeds(task_id: str, key: str) -> tuple[int, int]:
    return run_seed_for(task_id, key), run_seed_for(task_id, "train_eval_split")


def load_pool(labels_cfg: dict):
    """两注册表各自加载 + 切片 + 泄漏自检 → per_ds = [(name, X, y, tasks)]。"""
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    per_ds = []
    for name, sp in SPEC.items():
        reg = load_registry(ROOT / sp["registry"]) if sp["registry"] else load_registry()
        targets = set(reg.names(ROLE_TARGET))
        assert name not in targets, f"{name} 是目标池成员（铁律 6）"
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _, tasks = slice_source_tasks(entry, reg, y)
        assert name in iters_map, f"{name} 缺 workpoint.iters_per_dataset（labels.yaml）"
        print(f"[{name}] X={X.shape} → {len(tasks)} tasks @ iters={iters_map[name]}",
              flush=True)
        per_ds.append((name, X, y, tasks))
    n_total = sum(len(t) for _, _, _, t in per_ds)
    assert n_total == 28, f"源池应为 28 任务，得到 {n_total}"
    return per_ds


def build_candidates(per_ds: list, ra: dict, space: CLSSpace):
    """每任务候选 [(origin, cls, key)]：shared 10（探针目录）+ random 10（task_id 派生）。"""
    shared_cls = load_probe_cls(PROBE_CATALOG, ra["n_shared_cls"])[0]
    shared_keys = {k for _, k in shared_cls}
    out = []
    for name, X, y, tasks in per_ds:
        for t in tasks:
            rnd = [(c, cls_key(c)) for c in sample_random_cls(
                space, t.task_id, n=ra["n_random_cls"], seed=ra["cls_seed"],
                exclude_keys=shared_keys)]
            cand = ([(ORIGIN_SHARED, c, k) for c, k in shared_cls]
                    + [(ORIGIN_RANDOM, c, k) for c, k in rnd])
            out.append((name, X, y, t, cand))
    return out, shared_cls


def seed_existing(outdir: Path, per_ds: list, iters_map: dict,
                  shared_keys: set[str]) -> None:
    """confirm/followup 的工作点记录幂等种入（键 (task_id, cls_key, iters) 增补，
    classes/split_seed 校验，append-only）。"""
    cache_path = outdir / "rprime_cache.jsonl"
    existing = {(r["task_id"], r["cls_key"], r["iters"]) for r in read_jsonl(cache_path)}
    stats = defaultdict(int)
    for name, _X, _y, tasks in per_ds:
        by_id = {t.task_id: t for t in tasks}
        want_iters = iters_map[name]
        for src in SPEC[name]["seed_sources"]:
            for r in read_jsonl(ROOT / src):
                if r.get("dataset") != name or r.get("iters") != want_iters:
                    continue
                t = by_id.get(r["task_id"])
                if t is None:
                    stats["unknown_task"] += 1
                    continue
                if list(r.get("classes", [])) != list(t.classes):
                    stats["rejected_classes"] += 1
                    continue
                if r.get("split_seed") != canonical_seeds(r["task_id"], "x")[1]:
                    stats["rejected_split"] += 1
                    continue
                if r["cls_key"] not in shared_keys:
                    stats["not_probe_cls"] += 1
                    continue
                key = (r["task_id"], r["cls_key"], r["iters"])
                if key in existing:
                    stats["already"] += 1
                    continue
                append_jsonl(cache_path, {**r, "origin": ORIGIN_SHARED})
                existing.add(key)
                stats[f"seeded_{name}"] += 1
    print(f"种入: {dict(stats)}", flush=True)


def noise_probe(outdir: Path, per_ds: list, iters_map: dict, shared_cls: list,
                train_cfg: dict, labels_cfg: dict, device: str, resume: bool) -> None:
    """k=3 噪声探针的 2 个额外 weight seed（协议 §2）。split 仍 canonical。"""
    probe_path = outdir / "noise_probe.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in read_jsonl(probe_path)}
    work = [(name, X, y, t, c, k, rep)
            for name, X, y, tasks in per_ds
            for t in tasks[:NOISE_TASKS]
            for c, k in shared_cls
            for rep in NOISE_REPS]
    todo = [w for w in work
            if not (resume and (w[3].task_id, w[5], iters_map[w[0]], w[6]) in done)]
    print(f"噪声探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：fit_svm 的 GridSearchCV(n_jobs=-1) 在 Windows 上逐条 spawn
    # loky 进程池（8–15s/条，占 99% 耗时；R' 逐位同值）——长期记忆教训，勿删
    with parallel_backend("threading", n_jobs=1):
        _noise_loop(todo, outdir, iters_map, train_cfg, labels_cfg, device, t0)


def _noise_loop(todo, outdir, iters_map, train_cfg, labels_cfg, device, t0):
    probe_path = outdir / "noise_probe.jsonl"
    for i, (name, X, y, t, c, k, rep) in enumerate(todo):
        iters = iters_map[name]
        w_seed = run_seed_for(t.task_id, f"{k}:noise{rep}")
        sp_seed = canonical_seeds(t.task_id, k)[1]

        def make_row(rec, _n=name, _t=t, _k=k, _c=c, _rep=rep, _it=iters):
            rp = rec["r_prime"]
            return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                    "task_type": _t.task_type, "classes": list(_t.classes),
                    "cls_key": _k, "seed": _rep, "iters": _it, **rec,
                    "r_prime": None if not np.isfinite(rp) else float(rp),
                    "cls": dataclasses.asdict(_c)}

        full = guarded_run(
            lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                                    iters=iters, eval_ratio=labels_cfg["eval_ratio"],
                                    batch_size=train_cfg["batch_size"],
                                    lr=train_cfg["lr"], device=device),
            make_row, cache_path=probe_path, inflight=outdir / "inflight.txt",
            task_id=t.task_id, cls_key=k, iters=iters,
            on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
            context={"dataset": name, "noise_rep": rep},
            log=lambda m: print(f"[probe {i + 1}/{len(todo)}] {m}", flush=True))
        if full is None:
            continue
        eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
        print(f"[probe {i + 1}/{len(todo)}] {name}/{t.task_id} {k[:8]} rep{rep} "
              f"R'={full['r_prime']} ({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)


def calibrate(outdir: Path, per_ds: list, iters_map: dict, shared_cls: list) -> None:
    """δ 标定（协议 §2 公式，先于数据写死）：
    δ = max(P95(同 CLS 重复对 |ΔR'|), 2×max_task 量子)，上取整到 0.005。"""
    cache = load_rprime_cache(outdir / "rprime_cache.jsonl")
    probe = read_jsonl(outdir / "noise_probe.jsonl")
    out = {}
    for name, _X, _y, tasks in per_ds:
        iters = iters_map[name]
        tids = [t.task_id for t in tasks[:NOISE_TASKS]]
        diffs, quanta = [], []
        for tid in tids:
            for _c, k in shared_cls:
                reps = {}
                rec = cache.get((tid, k, iters))
                if rec and rec.get("r_prime") is not None:
                    reps[0] = float(rec["r_prime"])
                    if rec.get("n_eval"):
                        quanta.append(1.0 / rec["n_eval"])
                for r in probe:
                    if (r["task_id"] == tid and r["cls_key"] == k
                            and r["iters"] == iters and r.get("r_prime") is not None):
                        reps[r["seed"]] = float(r["r_prime"])
                vals = list(reps.values())
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        diffs.append(abs(vals[i] - vals[j]))
        if not diffs:
            print(f"[{name}] 无可用重复对，标定失败", flush=True)
            continue
        diffs.sort()
        p95 = diffs[min(len(diffs) - 1, int(round(0.95 * (len(diffs) - 1))))]
        floor = 2.0 * max(quanta) if quanta else 0.0
        raw = max(p95, floor)
        delta = float(np.ceil(raw / 0.005) * 0.005)
        out[name] = {"iters": iters, "n_diff_pairs": len(diffs),
                     "p95_replicate_diff": p95, "quantum_floor": floor,
                     "delta": delta,
                     "diff_median": statistics.median(diffs), "diff_max": diffs[-1]}
        print(f"[{name}] 重复对 {len(diffs)} 个 | P95={p95:.4f} 量子下限={floor:.4f} "
              f"→ δ={delta:.3f}（中位 {out[name]['diff_median']:.4f}，"
              f"最大 {diffs[-1]:.4f}）", flush=True)
    (outdir / "delta_calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {outdir / 'delta_calibration.json'}\n"
          f"请把 delta 回填 configs/labels.yaml: workpoint.delta_per_dataset 后再 --pairs",
          flush=True)


def fill(outdir: Path, candidates: list, iters_map: dict, train_cfg: dict,
         labels_cfg: dict, device: str, resume: bool) -> None:
    """R' 主循环（先全 shared 后全 random；per-dataset iters；逐条落盘）。"""
    cache_path = outdir / "rprime_cache.jsonl"
    cache = load_rprime_cache(cache_path)
    for name, _X, _y, t, cand in candidates:   # step0/种入记录补 origin（幂等）
        for org, _c, key in cand:
            _r2b.patch_missing_origin(cache, cache_path, t.task_id, key,
                                      iters_map[name], org)
    worklist = [(name, X, y, t, org, c, key)
                for org_want in (ORIGIN_SHARED, ORIGIN_RANDOM)
                for name, X, y, t, cand in candidates
                for org, c, key in cand if org == org_want]
    todo = [w for w in worklist
            if not (resume and (w[3].task_id, w[6], iters_map[w[0]]) in cache)]
    print(f"R' fill: 计划 {len(worklist)} 条，已缓存 {len(worklist) - len(todo)}，"
          f"待跑 {len(todo)}", flush=True)
    # threading 后端：同 noise_probe 处注释（loky spawn 99% 耗时教训）
    with parallel_backend("threading", n_jobs=1):
        _fill_loop(todo, cache, cache_path, outdir, iters_map, train_cfg,
                   labels_cfg, device)


def _fill_loop(todo, cache, cache_path, outdir, iters_map, train_cfg,
               labels_cfg, device):
    t0, n_oom = time.time(), 0
    for i, (name, X, y, t, org, c, key) in enumerate(todo):
        iters = iters_map[name]
        w_seed, sp_seed = canonical_seeds(t.task_id, key)

        def make_row(rec, _n=name, _t=t, _org=org, _k=key, _c=c, _it=iters):
            rp = rec["r_prime"]
            return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                    "task_type": _t.task_type, "classes": list(_t.classes),
                    "cls_key": _k, "origin": _org, "seed": 0, "iters": _it, **rec,
                    "r_prime": None if not np.isfinite(rp) else float(rp),
                    "cls": dataclasses.asdict(_c)}

        full = guarded_run(
            lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                                    iters=iters, eval_ratio=labels_cfg["eval_ratio"],
                                    batch_size=train_cfg["batch_size"],
                                    lr=train_cfg["lr"], device=device),
            make_row, cache_path=cache_path, inflight=outdir / "inflight.txt",
            task_id=t.task_id, cls_key=key, iters=iters,
            on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
            context={"dataset": name, "origin": org},
            log=lambda m: print(f"[fill {i + 1}/{len(todo)}] {m}", flush=True))
        if full is None:
            n_oom += 1
            continue
        cache[(t.task_id, key, iters)] = full
        st = (f"R'={full['r_prime']:.4f}" if full["r_prime"] is not None
              else f"失败: {full['error']}")
        eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
        print(f"[fill {i + 1}/{len(todo)}] {name}/{t.task_id} {org}:{key[:8]} {st} "
              f"({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"fill 完成，墙钟 {(time.time() - t0) / 60:.1f} min"
          + (f"，OOM {n_oom}（--resume 重试）" if n_oom else ""), flush=True)


def features(outdir: Path, per_ds: list, ra: dict, device: str) -> None:
    """共享 TS2Vec（只在 {HW, UniMiB} 池 fit，铁律 6）+ SetTF init + 28 任务特征。"""
    pool = [name for name, _X, _y, _t in per_ds]
    per_ds_2b = [(name, X, y, tasks) for name, X, y, tasks in per_ds]
    shared = _r2b.get_shared_ts2vec(outdir, per_ds_2b, pool, ra, device)
    set_enc = _r2b.get_set_encoder(outdir / "set_encoder_init.pt", shared, ra)
    feat_dir = outdir / "task_features"
    feat_dir.mkdir(exist_ok=True)
    for name, X, y, tasks in per_ds:
        for t in tasks:
            _r2b.ensure_task_features(t, X, y, shared, set_enc, ra, feat_dir)
    print("任务特征就绪", flush=True)


def build_baseline_bands(name: str) -> dict[str, tuple[float, float]]:
    """task_id → (基线均值, null_band)（summarize_task 同式：2·max(σ, 量子)）。"""
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(ROOT / SPEC[name]["baseline"]):
        if r.get("r_prime") is not None and r["task_id"].startswith(name):
            rows_by_task[r["task_id"]].append(r)
    out = {}
    for tid, rows in rows_by_task.items():
        vals = [float(r["r_prime"]) for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        quantum = max((1.0 / r["n_eval"] for r in rows if r.get("n_eval")), default=0.0)
        out[tid] = (statistics.fmean(vals), 2.0 * max(std, quantum))
    return out


def pairs_stage(outdir: Path, candidates: list, iters_map: dict,
                labels_cfg: dict) -> None:
    """配对 + summary（δ 从 labels.yaml workpoint 段读，空则 fail-fast）+
    破坏参与率（destruction_participation，δ=标定值）。"""
    delta_map = labels_cfg["workpoint"].get("delta_per_dataset") or {}
    missing = [n for n in SPEC if n not in delta_map]
    if missing:
        sys.exit(f"workpoint.delta_per_dataset 缺 {missing}：先 --calibrate 并回填 labels.yaml")

    cache = load_rprime_cache(outdir / "rprime_cache.jsonl")
    pairs_path = outdir / "pairs.jsonl"
    per_task, per_dataset = {}, {}
    rprime_by_task: dict[str, dict[str, float]] = {}
    ds_of_task: dict[str, str] = {}
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for name, _X, _y, t, cand in candidates:
            iters, delta = iters_map[name], delta_map[name]
            rprime, origin = {}, {}
            for org, _c, key in cand:
                rec = cache.get((t.task_id, key, iters))
                if rec is not None:
                    rprime[key] = (rec["r_prime"] if rec["r_prime"] is not None
                                   else float("nan"))
                    origin[key] = org
            rprime_by_task[t.task_id] = rprime
            ds_of_task[t.task_id] = name
            ps, stats = make_pairs(rprime, t.task_type, delta=delta, origin=origin)
            for p in ps:
                fh.write(json.dumps({"dataset": name, "task_id": t.task_id,
                                     "tag": t.tag, "task_type": t.task_type,
                                     "iters": iters, **p}, ensure_ascii=False) + "\n")
            per_task[t.task_id] = {"dataset": name, "delta": delta, "iters": iters,
                                   **stats}
            agg = per_dataset.setdefault(name, {"delta": delta, "iters": iters,
                                                "n_tasks": 0, **{k: 0 for k in stats}})
            agg["n_tasks"] += 1
            for k, v in stats.items():
                agg[k] += v

    destruction = {}
    for name in per_dataset:
        bands = build_baseline_bands(name)
        pt = {tid: rp for tid, rp in rprime_by_task.items()
              if ds_of_task[tid] == name and tid in bands}
        destruction[name] = destruction_participation(
            pt, bands, delta=delta_map[name])

    total = {k: sum(d[k] for d in per_dataset.values())
             for k in ("n_cls", "n_failed_cls", "n_candidates", "n_delta_dropped",
                       "n_tie_dropped", "n_kept")}
    summary = {"protocol": "results/decisive_e_ablation/PROTOCOL.md",
               "iters_per_dataset": iters_map, "delta_per_dataset": delta_map,
               "n_tasks": len(per_task), "per_task": per_task,
               "per_dataset": per_dataset, "destruction": destruction,
               "total": total}
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'数据集':<14}{'δ':>6}{'iters':>6}{'任务':>4}{'候选对':>7}{'δ弃':>6}"
          f"{'平局':>5}{'败CLS':>6}{'产出':>7}{'存活率':>7}{'strict':>7}{'loose':>7}")
    for name, s in sorted(per_dataset.items()):
        d = destruction[name]
        sr = s["n_kept"] / s["n_candidates"] if s["n_candidates"] else float("nan")
        pc = lambda v: f"{v:.0%}" if v is not None else "—"
        print(f"{name:<14}{s['delta']:>6}{s['iters']:>6}{s['n_tasks']:>4}"
              f"{s['n_candidates']:>7}{s['n_delta_dropped']:>6}{s['n_tie_dropped']:>5}"
              f"{s['n_failed_cls']:>6}{s['n_kept']:>7}{sr:>7.0%}"
              f"{pc(d['strict_share']):>7}{pc(d['loose_share']):>7}")
    print(f"合计产出 {total['n_kept']} / 候选 {total['n_candidates']} 对 → {pairs_path}",
          flush=True)


def verify(outdir: Path, iters_map: dict) -> None:
    """收官校验：行数、重复键、null、device_name 覆盖（幂等事故教训）。"""
    rows = read_jsonl(outdir / "rprime_cache.jsonl")
    keys = [(r["task_id"], r["cls_key"], r["iters"]) for r in rows]
    by_ds = defaultdict(int)
    for r in rows:
        if r.get("r_prime") is not None:
            by_ds[(r["dataset"], r["iters"])] += 1
    dup = len(keys) - len(set(keys))
    n_null = sum(1 for r in rows if r.get("r_prime") is None)
    n_dev = sum(1 for r in rows if r.get("device_name"))
    print(f"rprime_cache: {len(rows)} 行 | 唯一键 {len(set(keys))}（同键后行覆盖前行"
          f"{'，重复 ' + str(dup) + ' 行' if dup else '，零重复'}）| null {n_null} | "
          f"device_name {n_dev}/{len(rows)}")
    for (ds, it), n in sorted(by_ds.items()):
        expect = {"Handwriting": 240, "UniMiB-SHAR": 320}[ds]
        ok = "✅" if (n >= expect and it == iters_map[ds]) else "⚠"
        print(f"  {ds}@{it}: {n}/{expect} {ok}")
    probe = read_jsonl(outdir / "noise_probe.jsonl")
    print(f"noise_probe: {len(probe)} 行（期望 80）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed-existing", action="store_true")
    ap.add_argument("--noise-probe", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not any((args.seed_existing, args.noise_probe, args.calibrate, args.fill,
                args.features, args.pairs, args.verify)):
        ap.error("至少指定一个阶段")

    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    ra = labels_cfg["route_a"]
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_ds = load_pool(labels_cfg)
    space = CLSSpace.from_yaml()
    candidates, shared_cls = build_candidates(per_ds, ra, space)
    (args.outdir / "tasks.json").write_text(json.dumps({
        "pool": [n for n, _, _, _ in per_ds],
        "iters_per_dataset": iters_map,
        "source_tasks": [dataclasses.asdict(t) for _n, _X, _y, tasks in per_ds
                         for t in tasks],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.outdir / "cls_catalog.json").write_text(json.dumps({
        "shared": [{"cls_key": k, "cls": dataclasses.asdict(c)} for c, k in shared_cls],
        "random": {t.task_id: [{"cls_key": k, "cls": dataclasses.asdict(c)}
                               for _o, c, k in cand if _o == ORIGIN_RANDOM]
                   for _n, _X, _y, t, cand in candidates},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.seed_existing:
        seed_existing(args.outdir, per_ds, iters_map, {k for _, k in shared_cls})
    if args.noise_probe:
        noise_probe(args.outdir, per_ds, iters_map, shared_cls, train_cfg,
                    labels_cfg, args.device, args.resume)
    if args.calibrate:
        calibrate(args.outdir, per_ds, iters_map, shared_cls)
    if args.fill:
        fill(args.outdir, candidates, iters_map, train_cfg, labels_cfg,
             args.device, args.resume)
    if args.features:
        features(args.outdir, per_ds, ra, args.device)
    if args.pairs:
        pairs_stage(args.outdir, candidates, iters_map, labels_cfg)
    if args.verify:
        verify(args.outdir, iters_map)


if __name__ == "__main__":
    main()
