"""随机初始化基线探针：测每个候选数据集"不训练能拿多少分"（协议见
`results/baseline_probe/PROTOCOL.md`，判据先于数据写死）。

动机（解耦实验 results/decouple_random_cls/REPORT.md）：AWR/CT/ERing/NATOPS 的随机
初始化基线贴顶 0.90–0.99，常规 CLS 训练增益只有 ±0.013 → R' 测不出学习质量、只能
测破坏程度。本探针把"哪些数据集能用"换成有机制解释的客观量：

    headroom = 1 − baseline = 学习质量的可测空间**上界**
    realized/best gain = 阶段1 canonical R'(290 iters) − baseline = **实际用上**多少

口径与阶段1 严格一致（PROTOCOL.md §1）：同一个 `evaluate_rprime`、同一个
`slice_source_tasks`、同一个 canonical split；唯一差别是 `iters=0`（`pretrain_encoder`
的 while 循环不进入 → encoder 停在随机初始化，**CLS 完全不参与**，见 inert_cls）。

铁律 6：目标池（UCI-HAR/Epilepsy/FaceDetection）绝不入探针范围，本地有数据也不碰。
铁律 3：逐条落盘 baseline_cache.jsonl + --resume。

用法：
    python scripts/baseline_probe.py --run --resume            # 16 数据集 × 全部任务 × 5 种子
    python scripts/baseline_probe.py --run --datasets NATOPS --max-tasks 1 --seeds 2   # 试跑
    python scripts/baseline_probe.py --analyze                 # → analysis.json + 报表
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 本机控制台默认 GBK，报表里的 ∅/±/≥ 等会让 print 抛 UnicodeEncodeError 中断长跑
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # 已重定向/不支持时保持原样
        pass

from autocls.baseline_probe import (  # noqa: E402
    BASELINE_ITERS,
    BEST_GAIN_FLOOR,
    CEILING_BASELINE,
    OPEN_BASELINE,
    baseline_seed_specs,
    inert_cls,
    probe_scope,
    summarize_dataset,
    summarize_task,
)
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET,
    TargetPoolLeakageError,
    load_registry,
    load_source_dataset,
    slice_source_tasks,
)
from autocls.labeling import append_jsonl, cls_key, evaluate_rprime  # noqa: E402
from autocls.oom_guard import ON_OOM_RECORD, ON_OOM_SKIP, guarded_run  # noqa: E402

OUTDIR = ROOT / "results" / "baseline_probe"
CACHE = OUTDIR / "baseline_cache.jsonl"
INFLIGHT = OUTDIR / "inflight.txt"
OOM_LOG = OUTDIR / "oom_failures.jsonl"
N_SEEDS = 5

# 阶段1 canonical R'（290 iters，同 canonical split）——realized/best gain 的配对来源。
# 两份历史缓存都带 classes + split_seed，配对前逐字段校验任务定义一致（见 _trained_by_task）。
TRAINED_CACHES = (
    ROOT / "results" / "stage1_rerun_20260804" / "results" / "labels_route_a" / "rprime_cache.jsonl",
    ROOT / "results" / "route_a_probe" / "rprime_cache.jsonl",   # 早期探针（任务定义已变，多半被校验拒收）
    OUTDIR / "trained_fill.jsonl",                               # --fill-trained 补跑（见下）
)
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
TRAINED_FILL = OUTDIR / "trained_fill.jsonl"
REGISTRY_PATH: Path | None = None      # None = configs/source_pool.yaml（load_registry 默认）


def configure_paths(outdir: str | Path) -> None:
    """把全部落盘路径整体重定向到 outdir（换池调研步骤 2 用）。

    候选数据集要用**同一个**探针测（口径一致性是本轮结论的前提，见
    results/candidate_survey/PROTOCOL.md §3），但绝不能写进 results/baseline_probe/
    ——那 685 条 UEA 基线是已定稿数据（铁律 3）。TRAINED_CACHES 里的
    `OUTDIR/trained_fill.jsonl` 也要跟着走，否则候选会去读 UEA 的补跑结果做配对。
    """
    global OUTDIR, CACHE, INFLIGHT, OOM_LOG, TRAINED_FILL, TRAINED_CACHES
    OUTDIR = Path(outdir)
    CACHE = OUTDIR / "baseline_cache.jsonl"
    INFLIGHT = OUTDIR / "inflight.txt"
    OOM_LOG = OUTDIR / "oom_failures.jsonl"
    TRAINED_FILL = OUTDIR / "trained_fill.jsonl"
    TRAINED_CACHES = tuple(p for p in TRAINED_CACHES if p.name != "trained_fill.jsonl") + \
        (TRAINED_FILL,)


FILL_ITERS = 290          # = configs/train.yaml: early_iters（阶段1 造 label 口径）
FILL_TASKS, FILL_CLS = 2, 10   # 与阶段1 rprime_cache 逐字段同口径：前 2 任务 × 前 10 shared CLS

# 配对用的训练量（--pair-iters）。默认 290 = 既有 analysis.json 的口径，改默认会让
# 已发布结果不可复算。full-iters 判别实验（results/full_iters_probe/PROTOCOL.md）用
# 2320 重跑同一批 (任务, CLS, 初始化, 划分)，两份 R' 必须按 iters 分开配对——否则
# realized_gain 会变成 early 与 full 的混合均值（铁律 2："early 与 final 禁止混用"）。
PAIR_ITERS = FILL_ITERS


def load_registry_cfg():
    """按 --registry 解析注册表（None = configs/source_pool.yaml 默认）。"""
    return load_registry(REGISTRY_PATH) if REGISTRY_PATH else load_registry()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 被杀留下的半行：跳过
    return out


def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    """→ {(task_id, seed): 记录}。append-only，同键后行覆盖前行。"""
    return {(r["task_id"], r["seed"]): r for r in _read_jsonl(path)}


def fill_done_keys(path: Path) -> set[tuple[str, str, int]]:
    """--fill-trained 的 resume 去重键：**(task_id, cls_key, iters)**。

    iters 必须入键。同一 (任务, CLS) 在 290 与 2320 下是两条不同的测量，漏掉 iters
    会让 early 的记录把 full 的重跑整批挡掉——`--resume` 下表现为"秒完成"，实则一条
    没跑。口径与 `labeling.load_rprime_cache` 一致（它早就注明"iters 入键防 early/full
    混用"），本处此前是唯一的例外。
    """
    return {(r["task_id"], r["cls_key"], r["iters"]) for r in _read_jsonl(path)}


def leak_check(registry) -> None:
    """铁律 6：目标池绝不进探针范围，且走源加载路径必须 fail-fast。"""
    targets = sorted(registry.names(ROLE_TARGET))
    scope = set(probe_scope(registry))
    assert not (scope & set(targets)), f"目标池混入探针范围 {scope & set(targets)}（铁律 6）"
    for name in targets:
        try:
            load_source_dataset(name, registry)
        except (TargetPoolLeakageError, ValueError):
            continue
        raise AssertionError(f"目标池 {name} 未被源加载路径拒绝（铁律 6 防护失效）")
    print(f"泄漏自检：探针范围 {len(scope)} 个 ∩ 目标池 {targets} = ∅，"
          f"且目标池走源加载路径全部 fail-fast（铁律 6）", flush=True)


def sliced_tasks(registry, names: list[str], max_tasks: int | None,
                 n_classes_range: tuple[int, int] | None = None):
    """逐数据集加载 + 切**全部 planned_tasks**（与阶段1 同一入口、注册表默认档）。

    n_classes_range 覆盖档（--n-classes-range，"加难口径"子实验）：task_id 由
    pool_name_for 打 +c<lo>-<hi> 标，与默认档任务不同 id → 同一份缓存里共存不互撞。
    """
    out = []
    for name in names:
        entry = registry.get(name)
        X, y = load_source_dataset(name, registry)
        rng = None
        if n_classes_range is not None:   # 超出该数据集类别数的档自动收敛到"全类别"
            lo, hi = n_classes_range
            rng = (min(lo, entry.n_classes), min(hi, entry.n_classes))
        _, tasks = slice_source_tasks(entry, registry, y, n_classes_range=rng)
        acts = tasks[:max_tasks] if max_tasks else tasks
        print(f"[{name}] X={X.shape} n_classes={entry.n_classes} "
              f"planned={entry.planned_tasks} → {len(acts)} tasks", flush=True)
        out.append((name, entry, X, y, acts))
    return out


def run(names: list[str], *, device: str, resume: bool, n_seeds: int,
        max_tasks: int | None, on_oom: str, backend: str,
        n_classes_range: tuple[int, int] | None = None) -> None:
    if torch.get_num_threads() > 1:
        print(f"⚠️ torch.get_num_threads()={torch.get_num_threads()}>1：本负载 CPU-bound，"
              f"请用 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 启动（configs/runtime.yaml）。",
              flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry_cfg()
    leak_check(registry)

    per_ds = sliced_tasks(registry, names, max_tasks, n_classes_range)
    cls_cfg, key = inert_cls(), cls_key(inert_cls())
    cache = load_cache(CACHE)

    work = [(name, X, y, t, s) for name, _e, X, y, tasks in per_ds
            for t in tasks for s in range(n_seeds)]
    todo = [w for w in work if not (resume and (w[3].task_id, w[4]) in cache)]
    print(f"\n设备 {device} | iters={BASELINE_ITERS}（不训练，CLS 不参与）| "
          f"{len(per_ds)} 数据集 × 任务 × {n_seeds} 种子 = 计划 {len(work)} 条，"
          f"已缓存 {len(work) - len(todo)}，待跑 {len(todo)}\n", flush=True)

    # joblib 后端：fit_svm 的 GridSearchCV(n_jobs=-1) 在 Windows 上每条都要 spawn 一个
    # loky 进程池（本机实测 8–15 s/条，占单条耗时 ~99%；Linux fork 只要 0.1 s，故 5090
    # 上的 0.13 s/条在这里不成立）。threading 后端逐位同值——linear_eval 的既有注释已
    # 记明"n_jobs 不影响取值，只慢不改值"，本机 NATOPS/ERing/MotorImagery 三例实测
    # R'/f1/选中 C 全部逐位相同（tests/test_baseline_probe.py 有断言）。
    with parallel_backend(backend, n_jobs=1 if backend == "threading" else -1):
        _run_loop(todo, cache, cls_cfg, key, device, on_oom)


def _run_loop(todo, cache, cls_cfg, key, device, on_oom) -> None:
    t_start, n_oom = time.time(), 0
    for i, (name, X, y, t, s) in enumerate(todo):
        w_seed, sp_seed = baseline_seed_specs(t.task_id, s)

        def make_row(rec, _n=name, _t=t, _s=s):
            rp = rec["r_prime"]
            return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                    "task_type": _t.task_type, "classes": list(_t.classes),
                    "channels": list(_t.channels), "n_classes": len(_t.classes),
                    "cls_key": key, "seed": _s, "iters": BASELINE_ITERS, **rec,
                    "r_prime": None if not np.isfinite(rp) else float(rp),
                    "cls": dataclasses.asdict(cls_cfg)}

        full = guarded_run(
            lambda: evaluate_rprime(t, cls_cfg, X, y, weight_seed=w_seed,
                                    split_seed=sp_seed, iters=BASELINE_ITERS,
                                    device=device),
            make_row, cache_path=CACHE, inflight=INFLIGHT,
            task_id=t.task_id, cls_key=key, iters=BASELINE_ITERS,
            on_oom=on_oom, oom_log=OOM_LOG,
            context={"dataset": name, "seed": s, "on_oom": on_oom},
            log=lambda m: print(f"[{i + 1}/{len(todo)}] {m}", flush=True),
        )
        if full is None:
            n_oom += 1
            continue
        cache[(t.task_id, s)] = full
        st = (f"baseline={full['r_prime']:.4f}" if full["r_prime"] is not None
              else f"失败: {full['error']}")
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
        print(f"[{i + 1}/{len(todo)}] {name}/{t.task_id} s{s} {st} "
              f"({full['seconds']}s, n_ev={full.get('n_eval')}) ETA {eta / 60:.1f}min",
              flush=True)
    print(f"\n跑完 {len(todo) - n_oom}/{len(todo)} 条，"
          f"墙钟 {(time.time() - t_start) / 60:.1f} min" + (f"，OOM {n_oom}" if n_oom else ""),
          flush=True)


def _trained_by_task(tasks_by_id: dict) -> tuple[dict[str, list[float]], dict]:
    """canonical R'（PAIR_ITERS 口径）按 task_id 归集，**逐字段校验任务定义一致**。

    只接受 iters == PAIR_ITERS、classes 与当前切片逐字段相同、且 split_seed 等于
    canonical 划分的记录——历史缓存跨过若干次注册表调整，静默错配会让 realized gain
    变成两个不同任务相减；而 iters 不过滤则会让它变成 early 与 full 的混合均值。
    """
    by_task: dict[str, list[float]] = {}
    audit = {"matched": 0, "rejected_classes": 0, "rejected_split": 0,
             "rejected_iters": 0, "unknown_task": 0}
    for path in TRAINED_CACHES:
        for r in _read_jsonl(path):
            tid = r["task_id"]
            t = tasks_by_id.get(tid)
            if t is None:
                audit["unknown_task"] += 1
                continue
            if r.get("iters") != PAIR_ITERS:
                audit["rejected_iters"] += 1
                continue
            if list(r.get("classes", [])) != list(t.classes):
                audit["rejected_classes"] += 1
                continue
            if r.get("split_seed") != baseline_seed_specs(tid, 0)[1]:
                audit["rejected_split"] += 1
                continue
            if r.get("r_prime") is None:
                continue
            by_task.setdefault(tid, []).append(float(r["r_prime"]))
            audit["matched"] += 1
    return by_task, audit


def fill_trained(names: list[str], *, device: str, resume: bool, backend: str,
                 iters: int = FILL_ITERS, n_tasks: int = FILL_TASKS) -> None:
    """补跑 canonical R'（默认 290 iters = 阶段1 口径），给缺配对参照的数据集。

    `iters` 显式传参（铁律 2）：full-iters 判别实验用 2320 重跑**完全相同**的
    (任务, CLS, weight_seed, split_seed)，唯一变量是训练量。weight_seed 的派生
    `run_seed_for(task_id, cls_key)` 不含 iters，故两轮的随机初始化逐位相同
    ——这正是严格配对比较所要求的。

    为什么需要：realized/best gain 要拿 R'(290) 与基线相减，而探针骨干
    （Cricket/PhonemeSpectra/SpokenArabicDigits）只有 results/route_a_probe 那份**早于
    source_pool.yaml 定稿**的缓存——任务定义（类别档/通道档/样本上限）已变，被
    _trained_by_task 的逐字段校验如实拒收，于是它们判不了象限。这里按当前注册表切片
    重跑：前 2 任务 × 前 10 shared CLS × canonical 种子 —— 与阶段1 rprime_cache 逐字段
    同口径（同 evaluate_rprime、同 iters、同 weight/split 种子派生），故可直接并表。
    """
    from autocls.labeling import load_probe_cls, run_seed_for

    OUTDIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry_cfg()
    per_ds = sliced_tasks(registry, names, n_tasks)
    cls_list, _ = load_probe_cls(PROBE_CATALOG, FILL_CLS)
    done = fill_done_keys(TRAINED_FILL)

    work = [(n, X, y, t, c, k) for n, _e, X, y, tasks in per_ds
            for t in tasks for c, k in cls_list]
    todo = [w for w in work if not (resume and (w[3].task_id, w[5], iters) in done)]
    print(f"\n补跑 trained 参照：iters={iters} | 计划 {len(work)} 条，待跑 {len(todo)}\n",
          flush=True)

    t_start = time.time()
    with parallel_backend(backend, n_jobs=1 if backend == "threading" else -1):
        for i, (name, X, y, t, c, k) in enumerate(todo):
            w_seed = run_seed_for(t.task_id, k)
            sp_seed = run_seed_for(t.task_id, "train_eval_split")   # canonical，与基线同划分

            def make_row(rec, _n=name, _t=t, _k=k, _c=c):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": 0, "iters": iters, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                                        iters=iters, device=device),
                make_row, cache_path=TRAINED_FILL, inflight=INFLIGHT,
                task_id=t.task_id, cls_key=k, iters=iters,
                on_oom=ON_OOM_SKIP, oom_log=OOM_LOG, context={"dataset": name},
                log=lambda m: print(f"[fill {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
            st = (f"R'={full['r_prime']:.4f}" if full["r_prime"] is not None
                  else f"失败: {full['error']}")
            print(f"[fill {i + 1}/{len(todo)}] {name}/{t.task_id} {k[:8]} {st} "
                  f"({full['seconds']}s) ETA {eta / 60:.1f}min", flush=True)
    print(f"\n补跑完成，墙钟 {(time.time() - t_start) / 60:.1f} min → {TRAINED_FILL}", flush=True)


GATE_V2 = ROOT / "results" / "gate_v2" / "gate_v2.json"
STAGE1_ANALYSIS = (ROOT / "results" / "stage1_rerun_20260804" / "results" /
                   "step0_snr_gate" / "analysis.json")


def crosstab(ds_out: dict) -> dict:
    """本探针象限 × gate v2 判定 的交叉表（用户问的关键洞察在这里）。

    两个判定测的是**不同的东西**：gate v2 测"δ 过滤后配对方向正确率与产量"（对探针
    CLS 构成敏感，Pearson 0.719），本探针测"离顶距离 + 实际可用增益"（机制量）。
    交叉表能回答：判 fail 的是"噪声大"还是"根本没信号"，判 pass 的是不是靠破坏排序。
    """
    v2 = json.loads(GATE_V2.read_text(encoding="utf-8"))["datasets"] if GATE_V2.exists() else {}
    v1 = (json.loads(STAGE1_ANALYSIS.read_text(encoding="utf-8"))["datasets"]
          if STAGE1_ANALYSIS.exists() else {})
    rows, cells = {}, {}
    for name, d in ds_out.items():
        gv = v2.get(name, {})
        verdict = gv.get("verdict", "not_gated")
        rows[name] = {
            "baseline": d["baseline_mean"], "headroom": d["headroom"],
            "tier": d["tier"], "quadrant": d["quadrant"],
            "best_gain_max": d["best_gain_max"], "realized_gain_mean": d["realized_gain_mean"],
            "gate_v2_verdict": verdict, "gate_v2_passed": gv.get("passed"),
            "gate_v2_delta_star": gv.get("delta_star"),
            "gate_v2_accuracy": gv.get("accuracy"), "gate_v2_yield": gv.get("yield"),
            "gate_v1_verdict": v1.get(name, {}).get("verdict"),
            "stage1_trained_mean": v1.get(name, {}).get("rprime_grand_mean"),
            "n_above": d.get("n_trained_above_null"), "n_below": d.get("n_trained_below_null"),
            "n_trained_total": d.get("n_trained_total"),
        }
        cells.setdefault(d["quadrant"], {}).setdefault(verdict, []).append(name)
    return {"per_dataset": rows, "cells": cells}


def print_crosstab(ct: dict) -> None:
    rows = ct["per_dataset"]
    print(f"\n{'—'*118}\n交叉对比：本探针（机制量）× gate v2（统计判定）")
    print(f"{'数据集':<27}{'基线':>7}{'离顶':>7}{'最好增益':>9}{'象限':<10}"
          f"{'gate v2':<17}{'v2 A':>7}{'v2 Y':>7}{'v1':<16}{'阶段1 R̄':>9}")
    for n, r in sorted(rows.items(), key=lambda kv: kv[1]["baseline"]):
        f = lambda v, w=7, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
        print(f"{n:<27}{f(r['baseline'])}{f(r['headroom'])}{f(r['best_gain_max'],9)}"
              f"{r['quadrant']:<10}{r['gate_v2_verdict']:<17}{f(r['gate_v2_accuracy'])}"
              f"{f(r['gate_v2_yield'])}{str(r['gate_v1_verdict'] or '—'):<16}"
              f"{f(r['stage1_trained_mean'],9)}")
    print(f"\n零假设诊断（不参与判定）：训练后的 CLS 有多少条真的跑出随机初始化的 ±2σ 带")
    print(f"{'数据集':<27}{'超出↑':>7}{'跌破↓':>7}{'带内':>7}{'合计':>6}  说明")
    for n, r in sorted(rows.items(), key=lambda kv: -(kv[1].get("n_above") or 0)):
        tot = r.get("n_trained_total") or 0
        if not tot:
            continue
        up, dn = r.get("n_above") or 0, r.get("n_below") or 0
        note = ("真有 CLS 学出东西" if up else
                "没有一条 CLS 跑赢随机初始化" + ("（且多数在破坏）" if dn > tot / 2 else ""))
        print(f"{n:<27}{up:>7}{dn:>7}{tot - up - dn:>7}{tot:>6}  {note}")

    print("\n交叉格（象限 × gate v2 判定）")
    for q, per in sorted(ct["cells"].items()):
        for v, names in sorted(per.items()):
            print(f"  {q:<9} × {v:<17} {len(names):>2}: {', '.join(sorted(names))}")


def analyze(names: list[str], max_tasks: int | None) -> None:
    registry = load_registry_cfg()
    per_ds = sliced_tasks(registry, names, max_tasks)
    tasks_by_id = {t.task_id: t for _n, _e, _X, _y, tasks in per_ds for t in tasks}
    ds_of_task = {t.task_id: n for n, _e, _X, _y, tasks in per_ds for t in tasks}

    rows = _read_jsonl(CACHE)
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    trained, audit = _trained_by_task(tasks_by_id)
    print(f"\n配对 R'（iters={PAIR_ITERS}）：命中 {audit['matched']} 条 / 覆盖 "
          f"{len(trained)} 个任务；拒收 iters 不符 {audit['rejected_iters']}、"
          f"classes 不符 {audit['rejected_classes']}、split 不符 {audit['rejected_split']}、"
          f"任务不在当前切片 {audit['unknown_task']}", flush=True)

    task_out: dict[str, dict] = {}
    for tid, recs in by_task.items():
        t = tasks_by_id.get(tid)
        if t is None:
            continue
        st = summarize_task(recs, n_classes=len(t.classes), trained=trained.get(tid, []))
        st["dataset"] = ds_of_task[tid]
        task_out[tid] = st

    ds_out: dict[str, dict] = {}
    for name, entry, _X, _y, tasks in per_ds:
        mine = {tid: s for tid, s in task_out.items() if s["dataset"] == name}
        if not mine:
            continue
        ds_out[name] = {
            **summarize_dataset(mine),
            "role": entry.role, "n_classes_total": entry.n_classes,
            "l_enc": entry.l_enc, "domain": entry.domain,
            "n_classes_range": list(entry.n_classes_range),
            "planned_tasks": entry.planned_tasks,
        }

    ct = crosstab(ds_out)
    out = {
        "criterion": {"ceiling_baseline": CEILING_BASELINE, "open_baseline": OPEN_BASELINE,
                      "best_gain_floor": BEST_GAIN_FLOOR, "iters": BASELINE_ITERS,
                      "n_seeds_planned": N_SEEDS, "doc": "results/baseline_probe/PROTOCOL.md"},
        "trained_pairing_audit": audit,
        "datasets": ds_out, "tasks": task_out, "crosstab": ct,
    }
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*118}\n随机初始化基线（iters=0，{N_SEEDS} 个初始化种子，canonical split）\n{'='*118}")
    print(f"{'数据集':<27}{'角色':<9}{'任务':>4}{'类':>4}{'基线':>7}{'±':>7}"
          f"{'任务min':>8}{'中位':>7}{'max':>7}{'离顶':>7}{'猜测':>7}{'超猜测':>8}"
          f"{'实得':>7}{'最好':>7}  {'档':<8}{'象限':<9}{'判据v2'}")
    for name, d in sorted(ds_out.items(), key=lambda kv: kv[1]["baseline_mean"]):
        f = lambda v, w=7, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
        print(f"{name:<27}{d['role']:<9}{d['n_tasks']:>4}{d['n_classes_total']:>4}"
              f"{f(d['baseline_mean'])}{f(d['baseline_spread'])}{f(d['task_min'],8)}"
              f"{f(d['task_median'])}{f(d['task_max'])}{f(d['headroom'])}"
              f"{f(d['chance_mean'])}{f(d['lift_over_chance'],8)}"
              f"{f(d['realized_gain_mean'])}{f(d['best_gain_max'])}  "
              f"{d['tier']:<8}{d['quadrant']:<9}{d.get('verdict', '—')}")

    print(f"\n{'—'*118}\n分档汇总（数据集级）")
    for tier in ("ceiling", "mid", "open"):
        hit = [n for n, d in ds_out.items() if d["tier"] == tier]
        print(f"  {tier:<9} {len(hit):>2} 个: {', '.join(sorted(hit)) or '—'}")
    print("象限汇总（旧判据：headroom × best_gain）")
    for q in ("ceiling", "usable", "inert", "unknown"):
        hit = [n for n, d in ds_out.items() if d["quadrant"] == q]
        print(f"  {q:<9} {len(hit):>2} 个: {', '.join(sorted(hit)) or '—'}")
    print("判据 v2 汇总（candidate_survey/PROTOCOL.md §4.2：T1 离顶 + T2 幅度 + T3 显著性）")
    for q in ("ceiling", "usable", "inert", "unknown"):
        hit = [n for n, d in ds_out.items() if d.get("verdict") == q]
        print(f"  {q:<9} {len(hit):>2} 个: {', '.join(sorted(hit)) or '—'}")

    n_task_tier = {}
    for s in task_out.values():
        n_task_tier[s["tier"]] = n_task_tier.get(s["tier"], 0) + 1
    print(f"\n任务级分档（共 {len(task_out)} 任务）: "
          + "  ".join(f"{k}={v}" for k, v in sorted(n_task_tier.items())))
    vals = [s["baseline_mean"] for s in task_out.values()]
    print(f"任务级基线：均值 {statistics.fmean(vals):.3f} 中位 {statistics.median(vals):.3f} "
          f"范围 [{min(vals):.3f}, {max(vals):.3f}]")
    print_crosstab(ct)
    print(f"\n分析 → {OUTDIR / 'analysis.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fill-trained", action="store_true",
                    help=f"补跑 {FILL_ITERS} iters 的 canonical R' 参照（前 {FILL_TASKS} 任务 × "
                         f"前 {FILL_CLS} shared CLS），给缺阶段1 配对缓存的数据集")
    ap.add_argument("--fill-iters", type=int, default=FILL_ITERS,
                    help=f"--fill-trained 的训练量（默认 {FILL_ITERS} = early_iters）。"
                         f"full-iters 判别实验用 2320（= train.yaml full_iters）；"
                         f"iters 入缓存键与配对口径，两者绝不混算（铁律 2）")
    ap.add_argument("--fill-tasks", type=int, default=FILL_TASKS,
                    help=f"--fill-trained 每数据集取前 N 个任务（默认 {FILL_TASKS} = 历史"
                         f"口径，勿改默认）。workpoint followup 加测用 6（60 条确认口径，"
                         f"任务 00–01 已有记录由 --resume 逐键跳过）")
    ap.add_argument("--pair-iters", type=int, default=None,
                    help="--analyze 配对 R' 的训练量（默认跟随 --fill-iters）。"
                         "既有 analysis.json 全部按 290 生成")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="只跑指定数据集（默认源池 13 + reserve 3；目标池绝不可选，铁律 6）")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="随机初始化种子数")
    ap.add_argument("--max-tasks", type=int, default=None, help="试跑：每数据集只前 N 个任务")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--on-oom", choices=[ON_OOM_SKIP, ON_OOM_RECORD], default=ON_OOM_SKIP)
    ap.add_argument("--n-classes-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="覆盖注册表类别子集档（'加难口径'子实验：测贴顶是不是切片太易造成的）。"
                         "任务 id 会打 +c<lo>-<hi> 标，与默认档缓存共存不互撞")
    ap.add_argument("--backend", choices=["threading", "loky"], default="threading",
                    help="fit_svm 的 joblib 后端。Windows 上 loky 每条 spawn 进程池要 "
                         # %% 转义：argparse 对 help 串做 % 展开，裸 % 后接中文会
                         # 抛 ValueError（本行此前就让 --help 整个崩掉）
                         "8–15s（占 99%% 耗时），threading 逐位同值（默认）")
    ap.add_argument("--registry", default=None,
                    help="池注册表 yaml（默认 configs/source_pool.yaml）。换池调研用 "
                         "configs/candidate_pool.yaml")
    ap.add_argument("--outdir", default=None,
                    help="落盘目录（默认 results/baseline_probe）。测候选务必改到别处——"
                         "那 685 条 UEA 基线是已定稿数据，勿混（铁律 3）")
    args = ap.parse_args()

    if args.outdir:
        configure_paths(args.outdir)
    global REGISTRY_PATH, PAIR_ITERS
    REGISTRY_PATH = Path(args.registry) if args.registry else None
    PAIR_ITERS = args.pair_iters if args.pair_iters is not None else args.fill_iters

    registry = load_registry_cfg()
    names = args.datasets or probe_scope(registry)
    bad = set(names) & set(registry.names(ROLE_TARGET))
    if bad:
        sys.exit(f"{sorted(bad)} 是目标池数据集，绝不可测（铁律 6：zero-shot 边界）")

    if args.run:
        run(names, device=args.device, resume=args.resume, n_seeds=args.seeds,
            max_tasks=args.max_tasks, on_oom=args.on_oom, backend=args.backend,
            n_classes_range=tuple(args.n_classes_range) if args.n_classes_range else None)
    if args.fill_trained:
        fill_trained(names, device=args.device, resume=args.resume, backend=args.backend,
                     iters=args.fill_iters, n_tasks=args.fill_tasks)
    if args.analyze:
        analyze(names, args.max_tasks)
    if not (args.run or args.analyze or args.fill_trained):
        ap.error("至少指定 --run / --fill-trained / --analyze 之一")


if __name__ == "__main__":
    main()
