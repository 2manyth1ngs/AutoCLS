"""step0 SNR 预探针（上云全量造 label 的第一步；含天花板自动加难重探）。

路线 A 源池 13 个里只有 3 个骨干（SpokenArabicDigits/PhonemeSpectra/Cricket）经探针
实测 SNR>1；其余 10 个"推测可靠、未实测"，可能藏"PenDigits 第二"（短序列失配或天
花板效应导致 CLS 零区分度，只产噪声 label）。烧全量前先逐一过 SNR 闸：

  对每个未实测源数据集：切 n_tasks(2-3) × n_cls(8-10 探针 CLS，跨 HAR 好坏) ×
  n_seeds(3-5) → 测 R' 种子噪声 σ 与 CLS 区分 SNR（autocls/snr_gate，探针同款判据）。

天花板/失配区分（Part A-1）：SNR<1 不再一刀切踢——R̄ 高低区分不了天花板型
（NATOPS R̄0.94 贴顶，任务过易该救）与失配型（PenDigits R̄0.98 贴顶，架构盲该踢），
唯一判据是"加难重探 SNR 是否恢复"：
  SNR<1 且贴顶且 range 未触顶 → 自动上调 n_classes_range（escalate_range，lo/hi
  各 +2 封顶 n_classes）重切重探，≤2 轮（snr_gate.ceiling_escalation）：
  · 某轮 SNR 恢复 → 天花板型，判可用，恢复档记入 analysis.json 的
    recommended_n_classes_range（阶段2 造 label 沿用该更难档）；
  · 轮数用尽/触顶仍 <1 → 失配型，踢入 reserve 候选。
  重探任务的 pool_name 打 +c<lo>-<hi> 档标（source_pool.pool_name_for）——任务定义
  随档变，task_id 必须区分；重探的 canonical R' 因此并入全量缓存不浪费，阶段2 用
  恢复档时逐键复用。

缓存复用（铁律 3，⚠️ R' 并入全量缓存不浪费）：
- 切片走 slice_source_tasks（注册表唯一真源），任务定义与全量造 label 逐字段一致；
- 每 (task, cls) 的 **canonical 种子**（s=0：weight=run_seed_for(tid,key)、
  split=run_seed_for(tid,'train_eval_split')、iters=early_iters）即全量口径 → 写
  results/labels_route_a/rprime_cache.jsonl（全量直接命中复用；含重探轮）；
- 噪声种子（s≥1：weight/split 各自派生）只为测 σ/SNR → 写独立 noise_cache.jsonl。
- --smoke 低 iters，canonical+噪声都写独立 smoke_cache.jsonl，绝不污染全量缓存。

看门狗配合（Part A-3）：每条 R' 开跑前把 (task_id, cls_key, iters) 写
results/step0_snr_gate/inflight.txt、落盘成功后清空——进程被杀时看门狗据此统计
毒条目（scripts/cloud_run.sh）。

任务数下限闸：--analyze 末尾统计"探针实测骨干 + 通过闸的未实测源"的计划任务总数，
<100 则停并报告（提示 reserve 回填），不硬着头皮往下。

用法：
  # smoke：2 个未实测源各 1task×5cls×3seed 低 iters，验链路（含重探机制）
  python scripts/step0_snr_gate.py --run --smoke --datasets NATOPS ERing
  python scripts/step0_snr_gate.py --analyze --smoke --datasets NATOPS ERing
  # 全量（上云，脱离会话跑）：所有未实测源逐一过闸，canonical 并入全量缓存
  python scripts/step0_snr_gate.py --run --resume
  python scripts/step0_snr_gate.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import (  # noqa: E402
    delta_for,
    load_registry,
    load_source_dataset,
    slice_source_tasks,
    task_count_gate,
)
from autocls.labeling import (  # noqa: E402
    evaluate_rprime,
    load_probe_cls as _load_probe_cls,
    load_rprime_cache,
    run_seed_for,
)
from autocls.oom_guard import ON_OOM_RECORD, ON_OOM_SKIP, guarded_run  # noqa: E402
from autocls.snr_gate import (  # noqa: E402
    MAX_CEILING_ROUNDS,
    ceiling_escalation,
    compute_snr_stats,
    decide,
    is_range_maxed,
)
from autocls.space.cls_space import CLS  # noqa: E402

OUTDIR = ROOT / "results" / "step0_snr_gate"
SHARED_CACHE = ROOT / "results" / "labels_route_a" / "rprime_cache.jsonl"  # 全量共用
NOISE_CACHE = OUTDIR / "noise_cache.jsonl"
SMOKE_CACHE = OUTDIR / "smoke_cache.jsonl"
INFLIGHT = OUTDIR / "inflight.txt"
OOM_LOG = OUTDIR / "oom_failures.jsonl"
META_DIR = "tasks_meta.d"      # 每数据集一份，见 write_tasks_meta
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"


@dataclass(frozen=True)
class ProbeCfg:
    n_tasks: int = 2       # 每数据集切几个"难任务"测 SNR
    n_cls: int = 10        # 用前 n_cls 个探针 CLS（跨 HAR 好坏排名）
    n_seeds: int = 4       # canonical(s=0) + 噪声(s≥1)


SMOKE = ProbeCfg(n_tasks=1, n_cls=5, n_seeds=3)


def resolve_paths(*, smoke: bool, baseline: bool, cache_dir: Path | None
                  ) -> tuple[Path, Path, Path]:
    """→ (canonical 缓存, 噪声缓存, tasks_meta 根目录)。

    解耦实验（results/decouple_random_cls/PREREGISTRATION.md §4）的隔离规则：
    - --cache-dir 给定 → 全部产物进该目录，不碰生产缓存（这批 CLS 与阶段2 的
      shared/random 采样流无交集，混写只污染 preflight 计数）；
    - --baseline（iters=0 配对基线）→ 所有种子进同一 baseline_cache.jsonl，且
      **必须**给 cache-dir：往生产 canonical 缓存写 iters=0 记录是不可接受的污染。
    - 两者都不给 → 生产路径逐字不变（smoke 全进 SMOKE_CACHE，不污染全量）。
    """
    if baseline:
        if cache_dir is None:
            raise ValueError("--baseline 必须搭配 --cache-dir（防 iters=0 记录混进生产缓存）")
        p = Path(cache_dir) / "baseline_cache.jsonl"
        return p, p, Path(cache_dir)
    if cache_dir is not None:
        d = Path(cache_dir)
        return d / "rprime_cache.jsonl", d / "noise_cache.jsonl", d
    if smoke:
        return SMOKE_CACHE, SMOKE_CACHE, OUTDIR
    return SHARED_CACHE, NOISE_CACHE, OUTDIR


def load_probe_cls(n_cls: int) -> tuple[list[tuple[CLS, str]], dict[str, int]]:
    """读探针 CLS 目录 → 前 n_cls 个 (CLS, cls_key) + {cls_key: har_rank}（单调性用）。

    与 2b 造 label 共用 autocls.labeling.load_probe_cls（shared CLS 唯一真源）。
    """
    return _load_probe_cls(PROBE_CATALOG, n_cls)


def slice_dataset(name: str, registry, n_tasks_run: int):
    """加载源数据集（TRAIN+TEST 合并 + 注册表 crop）+ 切**全部 planned_tasks**。

    切全量 planned_tasks（而非只切 n_tasks_run）保证任务定义与全量造 label 逐字段一致
    → canonical R' 缓存可复用；只取前 n_tasks_run 个真正跑 R'。
    """
    entry = registry.get(name)
    X, y = load_source_dataset(name, registry)
    _, tasks = slice_source_tasks(entry, registry, y)
    return entry, X, y, tasks[:n_tasks_run]


def _seed_specs(task_id: str, key: str, s: int, early_iters: int) -> tuple[int, int]:
    """(weight_seed, split_seed)：s=0 是全量 canonical 口径；s≥1 派生噪声种子。"""
    if s == 0:
        return run_seed_for(task_id, key), run_seed_for(task_id, "train_eval_split")
    return run_seed_for(task_id, f"{key}:{s}"), run_seed_for(task_id, f"split:{s}")


def _task_meta(t) -> dict:
    return {"task_id": t.task_id, "classes": list(t.classes),
            "channels": list(t.channels), "tag": t.tag,
            "n_samples": len(t.sample_indices)}


def _stats_for(recs: list[dict], dataset: str, task_ids: list[str], har_rank):
    """缓存记录 → 指定 (数据集, 任务子集) 的 SnrStats（None = 尚无记录）。"""
    ids = set(task_ids)
    rprime: dict = defaultdict(lambda: defaultdict(dict))
    for r in recs:
        if r.get("dataset") != dataset or r["task_id"] not in ids:
            continue
        rp = r["r_prime"]
        rprime[r["task_id"]][r["cls_key"]][r["seed"]] = (
            float(rp) if rp is not None else float("nan"))
    if not rprime:
        return None
    return compute_snr_stats({tid: {k: dict(sm) for k, sm in cm.items()}
                              for tid, cm in rprime.items()}, har_rank=har_rank)


def run(datasets: list[str], smoke: bool, resume: bool, device: str, *,
        inflight: Path = INFLIGHT, on_oom: str = ON_OOM_SKIP,
        oom_log: Path = OOM_LOG, catalog: Path = PROBE_CATALOG,
        n_cls: int | None = None, cache_dir: Path | None = None,
        baseline: bool = False, no_escalate: bool = False) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if torch.get_num_threads() > 1:
        # GATE §3.1：54 核容器里默认 192 线程 → 53 s/条（0.63×，比 4060 笔记本还慢）。
        # 正常启动路径（cloud_run.sh / run_stage1.py）会注入 OMP_NUM_THREADS=1。
        print(f"⚠️ torch.get_num_threads()={torch.get_num_threads()}>1："
              f"本负载 CPU-bound，多线程 OpenMP 同步开销会让它慢 ~4.2×。"
              f"请用 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 启动（configs/runtime.yaml）。",
              flush=True)
    registry = load_registry()
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    # baseline（解耦实验 §3 配对基线）：iters=0 = pretrain_encoder 零步，encoder
    # 停在随机初始化；weight/split 种子派生与正式跑逐字相同 → 严格配对
    early_iters = 0 if baseline else (30 if smoke else train_cfg["early_iters"])
    pcfg = SMOKE if smoke else ProbeCfg()
    if n_cls is not None:
        pcfg = replace(pcfg, n_cls=n_cls)
    cls_list, har_rank = _load_probe_cls(catalog, pcfg.n_cls)

    # canonical(s=0) 与噪声(s≥1) 的缓存路由（smoke/实验目录/基线隔离见 resolve_paths）
    shared_path, noise_path, meta_root = resolve_paths(
        smoke=smoke, baseline=baseline, cache_dir=cache_dir)
    meta_root.mkdir(parents=True, exist_ok=True)
    shared_cache = load_rprime_cache(shared_path)
    noise_done = _load_noise_keys(noise_path)

    tasks_meta: dict = {}
    per_ds = []
    for name in datasets:
        entry, X, y, tasks = slice_dataset(name, registry, pcfg.n_tasks)
        print(f"[{name}] X={X.shape} y_cls={len(np.unique(y))} planned={entry.planned_tasks} "
              f"→ 跑 {len(tasks)} tasks", flush=True)
        tasks_meta[name] = {
            "L": int(X.shape[1]), "n_ch": int(X.shape[2]),
            "n_classes": entry.n_classes,
            "n_classes_range": list(entry.n_classes_range),
            "tasks": [_task_meta(t) for t in tasks],
            "retry_rounds": [],
        }
        per_ds.append((name, X, y, tasks))
        write_tasks_meta(meta_root, name, tasks_meta[name])

    bs, lr, evr = train_cfg["batch_size"], train_cfg["lr"], labels_cfg["eval_ratio"]

    def is_done(name, tid, key, s):
        if baseline:  # 基线所有种子同文件，按 (ds, task, cls, seed) 记账
            return (name, tid, key, s) in noise_done
        if s == 0 and not smoke:
            return (tid, key, early_iters) in shared_cache
        return (name, tid, key, s) in noise_done

    def exec_worklist(worklist, resume_flag: bool, phase: str) -> None:
        """跑一批 (name, X, y, task, cls, key, s)：逐条落盘 + inflight 打点（Part A-3）。"""
        todo = [w for w in worklist
                if not (resume_flag and is_done(w[0], w[3].task_id, w[5], w[6]))]
        print(f"[{phase}] 设备 {device} | iters={early_iters} | "
              f"计划 {len(worklist)} 条，待跑 {len(todo)}", flush=True)
        n_oom = 0
        for i, (name, X, y, t, c, key, s) in enumerate(todo):
            w_seed, sp_seed = _seed_specs(t.task_id, key, s, early_iters)

            def make_row(rec, _name=name, _t=t, _key=key, _s=s, _c=c):
                rp = rec["r_prime"]
                return {"dataset": _name, "task_id": _t.task_id, "tag": _t.tag,
                        "classes": list(_t.classes), "cls_key": _key, "seed": _s,
                        "iters": early_iters, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": _asdict(_c)}

            # canonical(s=0) → 全量缓存（全量口径，键 task_id+cls_key+iters）；噪声 → 独立缓存
            # （smoke 两路都是 SMOKE_CACHE；baseline 两路都是 baseline_cache，见 resolve_paths）
            target = shared_path if s == 0 else noise_path
            # OOM 走 oom_guard：不写缓存、清 inflight（不计毒条目）、按 on_oom 记账
            full = guarded_run(
                lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                                        iters=early_iters, eval_ratio=evr, batch_size=bs,
                                        lr=lr, device=device),
                make_row, cache_path=target, inflight=inflight,
                task_id=t.task_id, cls_key=key, iters=early_iters,
                on_oom=on_oom, oom_log=oom_log,
                context={"dataset": name, "seed": s, "phase": phase, "on_oom": on_oom},
                log=lambda m: print(f"[{phase} {i+1}/{len(todo)}] {m}", flush=True),
            )
            if full is None:      # OOM：该条保持"未跑"，下一轮 --resume 会重试
                n_oom += 1
                continue
            if baseline or s != 0 or smoke:
                noise_done.add((name, t.task_id, key, s))
            else:
                shared_cache[(t.task_id, key, early_iters)] = full
            rp_json = full["r_prime"]
            st = f"R'={rp_json:.4f}" if rp_json is not None else f"FAIL {full['error']}"
            print(f"[{phase} {i+1}/{len(todo)}] {name}/{t.task_id} {key[:8]} s{s} {st} "
                  f"({full['seconds']}s, n_ev={full.get('n_eval')})", flush=True)
        if n_oom:
            print(f"[{phase}] ⚠️ {n_oom}/{len(todo)} 条因显存不足未完成"
                  + ("（已记入 oom_failures.jsonl）" if on_oom == ON_OOM_RECORD
                     else "（未写缓存，留待 worker=1 串行收尾重试）"), flush=True)

    # ---- base 轮：seed-major（先跑全部 s=0 填全量缓存，再加深噪声）----
    worklist = []
    for s in range(pcfg.n_seeds):
        for name, X, y, tasks in per_ds:
            for t in tasks:
                for c, key in cls_list:
                    worklist.append((name, X, y, t, c, key, s))
    exec_worklist(worklist, resume, "base")

    if baseline or no_escalate:
        # 解耦实验：任务集写死（PREREGISTRATION.md §4），不做加难重探；
        # baseline（iters=0）算 SNR 也毫无意义
        print("step0 运行完成（--baseline/--no-escalate：跳过天花板加难重探）。", flush=True)
        return

    # ---- 天花板加难重探（Part A-1）：SNR<1 且贴顶 → 自动加难 ≤2 轮 ----
    def all_recs() -> list[dict]:
        return (_read_jsonl(shared_path) if shared_path == noise_path
                else _read_jsonl(shared_path) + _read_jsonl(noise_path))

    for name, X, y, tasks in per_ds:
        entry = registry.get(name)
        base_stats = _stats_for(all_recs(), name, [t.task_id for t in tasks], har_rank)
        if base_stats is None:
            continue

        def run_round(r: int, new_range: tuple[int, int],
                      _name=name, _X=X, _y=y, _entry=entry):
            _, rall = slice_source_tasks(_entry, registry, _y, n_classes_range=new_range)
            rtasks = rall[:pcfg.n_tasks]
            # 幂等登记本轮任务（resume 重放同轮同档同任务，覆盖写不累积）
            rounds = tasks_meta[_name]["retry_rounds"]
            rounds[:] = [rm for rm in rounds if rm["round"] != r] + [{
                "round": r, "n_classes_range": list(new_range),
                "tasks": [_task_meta(t) for t in rtasks],
            }]
            rounds.sort(key=lambda rm: rm["round"])
            write_tasks_meta(meta_root, _name, tasks_meta[_name])
            wl = [(_name, _X, _y, t, c, key, s)
                  for s in range(pcfg.n_seeds) for t in rtasks for c, key in cls_list]
            exec_worklist(wl, True, f"{_name} 加难重探 r{r} c{new_range[0]}-{new_range[1]}")
            return _stats_for(all_recs(), _name,
                              [t.task_id for t in rtasks], har_rank)

        outcome = ceiling_escalation(
            base_stats, entry.n_classes_range, entry.n_classes, run_round)
        if outcome.rounds:
            lo, hi = outcome.n_classes_range
            print(f"[{name}] 重探 {len(outcome.rounds)} 轮 → {outcome.verdict.label} "
                  f"(最终档 c{lo}-{hi}): {outcome.verdict.reason}", flush=True)
    print("step0 运行完成。", flush=True)


def write_tasks_meta(outdir: Path, name: str, meta: dict) -> None:
    """任务定义/重探轮记录落盘：**每数据集一个文件** tasks_meta.d/<数据集>.json。

    原先是整份 tasks_meta.json 覆盖写，而 run() 每次从空字典重建 —— 换一批
    --datasets 再跑一次，上一批的记录就没了，--analyze 对它们报"无 tasks_meta 记录，
    跳过"，天花板重探记录全丢、判读残废（手册 §0.6 记的是 4 卡共享目录版本的同一
    问题）。阶段1 单卡按数据集并发跑，这是必然场景：分文件后并发写互不相干。
    """
    d = Path(outdir) / META_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def read_tasks_meta(outdir: Path) -> dict:
    """合并读：先旧的整份 tasks_meta.json（本机 smoke 与 5090 闸门产物都是旧格式，
    必须还能判读），再用 tasks_meta.d/ 里的逐数据集文件覆盖。"""
    outdir = Path(outdir)
    out: dict = {}
    legacy = outdir / "tasks_meta.json"
    if legacy.exists():
        try:
            out.update(json.loads(legacy.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    d = outdir / META_DIR
    if d.exists():
        for p in sorted(d.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


def _asdict(c: CLS) -> dict:
    import dataclasses
    return dataclasses.asdict(c)


def _load_noise_keys(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["dataset"], r["task_id"], r["cls_key"], r["seed"]))
    return done


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _stats_row(stats) -> dict:
    return {
        "n_tasks": stats.n_tasks, "n_fail": stats.n_fail,
        "rprime_std_mean": round(stats.rprime_std_mean, 4),
        "cls_spread_std_mean": round(stats.cls_spread_std_mean, 4),
        "rprime_grand_mean": round(stats.rprime_grand_mean, 4)
        if stats.rprime_grand_mean == stats.rprime_grand_mean else None,
        "snr": round(stats.snr, 2) if stats.snr == stats.snr else None,
        "pair_sign_stability": round(stats.pair_sign_stability, 3)
        if stats.pair_sign_stability == stats.pair_sign_stability else None,
        "monotonicity": round(stats.monotonicity, 3) if stats.monotonicity is not None else None,
        "suggested_delta": stats.suggested_delta,
    }


def analyze(datasets: list[str], smoke: bool) -> None:
    """重放判定（与 run 的 ceiling_escalation 同判据）：base → 逐重探轮 → 终局。

    重探记录缺失（--run 未跑/中断）时如实报 retry_ceiling + pending_retry=True，
    不臆断通过/踢除。
    """
    registry = load_registry()
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    _, har_rank = load_probe_cls((SMOKE if smoke else ProbeCfg()).n_cls)

    recs = _read_jsonl(SMOKE_CACHE) if smoke else (_read_jsonl(SHARED_CACHE) + _read_jsonl(NOISE_CACHE))
    tasks_meta = read_tasks_meta(OUTDIR)

    out: dict = {"datasets": {}}
    passed: set[str] = set()
    print(f"\n{'='*88}\nstep0 SNR 预探针分析{'（smoke）' if smoke else ''}\n{'='*88}")
    for ds in datasets:
        meta = tasks_meta.get(ds)
        if meta is None:
            print(f"[{ds}] 无 tasks_meta 记录，跳过（先 --run）")
            continue
        entry = registry.get(ds)
        base_range = tuple(meta.get("n_classes_range") or entry.n_classes_range)
        base_stats = _stats_for(recs, ds, [t["task_id"] for t in meta["tasks"]], har_rank)
        if base_stats is None:
            print(f"[{ds}] 无 R' 记录，跳过（先 --run）")
            continue

        verdict = decide(base_stats, range_maxed=is_range_maxed(base_range, entry.n_classes))
        final_range, final_stats = base_range, base_stats
        rounds_out, pending = [], False
        for i, rm in enumerate(meta.get("retry_rounds", []), start=1):
            if verdict.action != "retry_ceiling":
                break
            rng = tuple(rm["n_classes_range"])
            stats_r = _stats_for(recs, ds, [t["task_id"] for t in rm["tasks"]], har_rank)
            if stats_r is None:
                pending = True
                break
            maxed = i >= MAX_CEILING_ROUNDS or is_range_maxed(rng, entry.n_classes)
            verdict = decide(stats_r, range_maxed=maxed)
            final_range, final_stats = rng, stats_r
            rounds_out.append({"round": i, "n_classes_range": list(rng),
                               "verdict": verdict.label, **_stats_row(stats_r)})
        if verdict.action == "retry_ceiling":
            pending = True  # 重探未跑完：结论悬置，--run 会自动补跑

        if verdict.passed:
            passed.add(ds)
        out["datasets"][ds] = {
            **_stats_row(final_stats),
            "base": {"n_classes_range": list(base_range), **_stats_row(base_stats)},
            "rounds": rounds_out,
            "registry_delta": delta_for(ds, labels_cfg),
            "verdict": verdict.label, "action": verdict.action,
            "passed": verdict.passed, "pending_retry": pending,
            "recommended_n_classes_range": list(final_range) if verdict.passed else None,
            "reason": verdict.reason,
        }
        d = out["datasets"][ds]
        print(f"\n[{ds}] SNR={d['snr']} R̄={d['rprime_grand_mean']} σ={d['rprime_std_mean']} "
              f"spread={d['cls_spread_std_mean']} 符号稳定={d['pair_sign_stability']} "
              f"单调={d['monotonicity']}")
        if rounds_out:
            for r in rounds_out:
                print(f"  重探 r{r['round']} c{r['n_classes_range'][0]}-{r['n_classes_range'][1]}: "
                      f"SNR={r['snr']} R̄={r['rprime_grand_mean']} → {r['verdict']}")
        rec_rng = d["recommended_n_classes_range"]
        print(f"  建议 δ≈{d['suggested_delta']}（注册表 {d['registry_delta']}）"
              + (f" | 阶段2 沿用档 c{rec_rng[0]}-{rec_rng[1]}" if rec_rng else "")
              + (" | ⚠️ 重探未跑完（pending）" if d["pending_retry"] else ""))
        print(f"  判定: {verdict.reason}")

    # 任务数下限闸：探针骨干 + 通过闸的未实测源
    gate = task_count_gate(registry, passed_untested=passed, floor=100)
    out["gate"] = {"ok": gate.ok, "count": gate.count, "floor": gate.floor,
                   "passed_untested": sorted(passed), "message": gate.message}
    print(f"\n{'—'*88}\n任务数下限闸：{gate.message}")
    (OUTDIR / f"analysis{'_smoke' if smoke else ''}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"分析 → {OUTDIR / ('analysis_smoke.json' if smoke else 'analysis.json')}")
    if not gate.ok and not smoke:
        sys.exit(2)  # 任务数不足：非零退出，提醒回填（smoke 不触发，机制验证用）


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1task×5cls×3seed 低 iters 验链路，写 smoke 缓存")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="只跑指定源数据集（默认全部未实测源 tested=false）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inflight", type=Path, default=INFLIGHT,
                    help="看门狗打点文件；单卡多数据集并发时每进程一份，否则毒条目归因错乱")
    ap.add_argument("--on-oom", choices=[ON_OOM_SKIP, ON_OOM_RECORD], default=ON_OOM_SKIP,
                    help="显存不足时：skip=不写缓存留待串行收尾重试（并发档默认）；"
                         "record=已是 worker=1，仍 OOM 即终局失败，记 oom_failures.jsonl")
    ap.add_argument("--oom-log", type=Path, default=OOM_LOG)
    # --- 解耦实验注入点（results/decouple_random_cls/PREREGISTRATION.md §4）---
    ap.add_argument("--catalog", type=Path, default=PROBE_CATALOG,
                    help="探针 CLS 目录（默认生产目录；解耦实验换随机目录）")
    ap.add_argument("--n-cls", type=int, default=None,
                    help="覆盖 ProbeCfg.n_cls（解耦实验 20；默认档不变）")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="全部产物改进该目录（rprime/noise/tasks_meta），不碰生产缓存")
    ap.add_argument("--baseline", action="store_true",
                    help="iters=0 配对基线（同 weight/split 种子，encoder 停在随机"
                         "初始化）；须搭配 --cache-dir，全部种子进 baseline_cache.jsonl")
    ap.add_argument("--no-escalate", action="store_true",
                    help="禁天花板加难重探（解耦实验任务集写死）")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    registry = load_registry()
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [e.name for e in registry.source if not e.tested]  # 未实测源

    if args.run:
        run(datasets, smoke=args.smoke, resume=args.resume, device=args.device,
            inflight=args.inflight, on_oom=args.on_oom, oom_log=args.oom_log,
            catalog=args.catalog, n_cls=args.n_cls, cache_dir=args.cache_dir,
            baseline=args.baseline, no_escalate=args.no_escalate)
    if args.analyze:
        analyze(datasets, smoke=args.smoke)
    if not (args.run or args.analyze):
        ap.error("至少指定 --run / --analyze 之一")


if __name__ == "__main__":
    main()
