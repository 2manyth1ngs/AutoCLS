"""step0 SNR 预探针（上云全量造 label 的第一步；本会话先小规模验证机制）。

路线 A 源池 13 个里只有 3 个骨干（SpokenArabicDigits/PhonemeSpectra/Cricket）经探针
实测 SNR>1；其余 10 个"推测可靠、未实测"，可能藏"PenDigits 第二"（短序列失配或天
花板效应导致 CLS 零区分度，只产噪声 label）。烧全量前先逐一过 SNR 闸：

  对每个未实测源数据集：切 n_tasks(2-3) × n_cls(8-10 探针 CLS，跨 HAR 好坏) ×
  n_seeds(3-5) → 测 R' 种子噪声 σ 与 CLS 区分 SNR（autocls/snr_gate，探针同款判据）。
  SNR>1 且符号稳定 → 通过入池；SNR<1（PenDigits 惰性模式）→ 踢入 reserve。

缓存复用（铁律 3，⚠️ R' 并入全量缓存不浪费）：
- 切片走 build_slice_kwargs（注册表唯一真源），任务定义与全量造 label 逐字段一致；
- 每 (task, cls) 的 **canonical 种子**（s=0：weight=run_seed_for(tid,key)、
  split=run_seed_for(tid,'train_eval_split')、iters=early_iters）即全量口径 → 写
  results/labels_route_a/rprime_cache.jsonl（全量直接命中复用）；
- 噪声种子（s≥1：weight/split 各自派生）只为测 σ/SNR → 写独立 noise_cache.jsonl。
- --smoke 低 iters，canonical+噪声都写独立 smoke_cache.jsonl，绝不污染全量缓存。

任务数下限闸（本会话新增）：--analyze 末尾统计"探针实测骨干 + 通过闸的未实测源"的
计划任务总数，<100 则停并报告（提示 reserve 回填），不硬着头皮往下。

用法：
  # smoke：2 个未实测源各 1task×5cls×3seed 低 iters，验链路（本会话验收）
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import (  # noqa: E402
    build_slice_kwargs,
    delta_for,
    load_registry,
    load_source_dataset,
    task_count_gate,
)
from autocls.data.task_slicer import slice_pool_tasks  # noqa: E402
from autocls.labeling import (  # noqa: E402
    append_jsonl,
    cls_key,
    evaluate_rprime,
    load_rprime_cache,
    run_seed_for,
)
from autocls.snr_gate import compute_snr_stats, decide  # noqa: E402
from autocls.space.cls_space import CLS  # noqa: E402

OUTDIR = ROOT / "results" / "step0_snr_gate"
SHARED_CACHE = ROOT / "results" / "labels_route_a" / "rprime_cache.jsonl"  # 全量共用
NOISE_CACHE = OUTDIR / "noise_cache.jsonl"
SMOKE_CACHE = OUTDIR / "smoke_cache.jsonl"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"


@dataclass(frozen=True)
class ProbeCfg:
    n_tasks: int = 2       # 每数据集切几个"难任务"测 SNR
    n_cls: int = 10        # 用前 n_cls 个探针 CLS（跨 HAR 好坏排名）
    n_seeds: int = 4       # canonical(s=0) + 噪声(s≥1)


SMOKE = ProbeCfg(n_tasks=1, n_cls=5, n_seeds=3)


def load_probe_cls(n_cls: int) -> tuple[list[tuple[CLS, str]], dict[str, int]]:
    """读探针 CLS 目录 → 前 n_cls 个 (CLS, cls_key) + {cls_key: har_rank}（单调性用）。"""
    cat = json.loads(PROBE_CATALOG.read_text(encoding="utf-8"))["cls"][:n_cls]
    cls_list = [(CLS(**cd["cls"]), cd["cls_key"]) for cd in cat]
    har_rank = {cd["cls_key"]: cd["har_rank"] for cd in cat}
    return cls_list, har_rank


def slice_dataset(name: str, registry, n_tasks_run: int):
    """加载源数据集（TRAIN+TEST 合并 + 注册表 crop）+ 切**全部 planned_tasks**。

    切全量 planned_tasks（而非只切 n_tasks_run）保证任务定义与全量造 label 逐字段一致
    → canonical R' 缓存可复用；只取前 n_tasks_run 个真正跑 R'。
    """
    entry = registry.get(name)
    X, y = load_source_dataset(name, registry)
    kw = build_slice_kwargs(entry, registry)
    tasks = slice_pool_tasks(y, np.arange(len(y)), pool_name=name,
                             n_tasks=entry.planned_tasks, **kw)
    return entry, X, y, tasks[:n_tasks_run]


def _seed_specs(task_id: str, key: str, s: int, early_iters: int) -> tuple[int, int]:
    """(weight_seed, split_seed)：s=0 是全量 canonical 口径；s≥1 派生噪声种子。"""
    if s == 0:
        return run_seed_for(task_id, key), run_seed_for(task_id, "train_eval_split")
    return run_seed_for(task_id, f"{key}:{s}"), run_seed_for(task_id, f"split:{s}")


def run(datasets: list[str], smoke: bool, resume: bool, device: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    early_iters = 30 if smoke else train_cfg["early_iters"]
    pcfg = SMOKE if smoke else ProbeCfg()
    cls_list, _ = load_probe_cls(pcfg.n_cls)

    # canonical(s=0) 与噪声(s≥1) 的缓存：smoke 全进 smoke_cache 不污染全量
    shared_path = SMOKE_CACHE if smoke else SHARED_CACHE
    noise_path = SMOKE_CACHE if smoke else NOISE_CACHE
    shared_cache = load_rprime_cache(shared_path)
    noise_done = _load_noise_keys(noise_path) if not smoke else _load_noise_keys(SMOKE_CACHE)

    tasks_meta: dict = {}
    per_ds = []
    for name in datasets:
        entry, X, y, tasks = slice_dataset(name, registry, pcfg.n_tasks)
        print(f"[{name}] X={X.shape} y_cls={len(np.unique(y))} planned={entry.planned_tasks} "
              f"→ 跑 {len(tasks)} tasks", flush=True)
        tasks_meta[name] = {
            "L": int(X.shape[1]), "n_ch": int(X.shape[2]),
            "tasks": [{"task_id": t.task_id, "classes": list(t.classes),
                       "channels": list(t.channels), "tag": t.tag,
                       "n_samples": len(t.sample_indices)} for t in tasks],
        }
        per_ds.append((name, X, y, tasks))
    (OUTDIR / "tasks_meta.json").write_text(
        json.dumps(tasks_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # seed-major：先跑全部 s=0（填全量缓存 + 每 (task,cls) 首个 R'），再加深噪声
    worklist = []
    for s in range(pcfg.n_seeds):
        for name, X, y, tasks in per_ds:
            for t in tasks:
                for c, key in cls_list:
                    worklist.append((name, X, y, t, c, key, s))

    def is_done(name, tid, key, s):
        if s == 0 and not smoke:
            return (tid, key, early_iters) in shared_cache
        return (name, tid, key, s) in noise_done

    todo = [w for w in worklist if not (resume and is_done(w[0], w[3].task_id, w[5], w[6]))]
    print(f"设备 {device} | iters={early_iters} | 计划 {len(worklist)} 条，待跑 {len(todo)}", flush=True)

    bs, lr, evr = train_cfg["batch_size"], train_cfg["lr"], labels_cfg["eval_ratio"]
    for i, (name, X, y, t, c, key, s) in enumerate(todo):
        w_seed, sp_seed = _seed_specs(t.task_id, key, s, early_iters)
        rec = evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                              iters=early_iters, eval_ratio=evr, batch_size=bs, lr=lr,
                              device=device)
        rp = rec["r_prime"]
        rp_json = None if not np.isfinite(rp) else float(rp)
        full = {"dataset": name, "task_id": t.task_id, "tag": t.tag,
                "classes": list(t.classes), "cls_key": key, "seed": s,
                "iters": early_iters, **rec, "r_prime": rp_json,
                "cls": _asdict(c)}
        # canonical(s=0) → 全量缓存（全量口径，键 task_id+cls_key+iters）；噪声 → 独立缓存
        target = SMOKE_CACHE if smoke else (shared_path if s == 0 else noise_path)
        append_jsonl(target, full)
        if s == 0 and not smoke:
            shared_cache[(t.task_id, key, early_iters)] = full
        else:
            noise_done.add((name, t.task_id, key, s))
        st = f"R'={rp:.4f}" if rp_json is not None else f"FAIL {rec['error']}"
        print(f"[{i+1}/{len(todo)}] {name}/{t.task_id} {key[:8]} s{s} {st} "
              f"({rec['seconds']}s, n_ev={rec.get('n_eval')})", flush=True)
    print("step0 运行完成。", flush=True)


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


def analyze(datasets: list[str], smoke: bool) -> None:
    registry = load_registry()
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    _, har_rank = load_probe_cls((SMOKE if smoke else ProbeCfg()).n_cls)

    # 汇总所有相关记录（canonical + 噪声；smoke 全在 smoke_cache）
    recs = _read_jsonl(SMOKE_CACHE) if smoke else (_read_jsonl(SHARED_CACHE) + _read_jsonl(NOISE_CACHE))
    tasks_meta = json.loads((OUTDIR / "tasks_meta.json").read_text(encoding="utf-8"))
    ds_of_task = {t["task_id"]: ds for ds, m in tasks_meta.items() for t in m["tasks"]}

    # rprime[dataset][task_id][cls_key][seed]
    by_ds = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in recs:
        ds = r.get("dataset") or ds_of_task.get(r["task_id"])
        if ds is None or ds not in datasets:
            continue
        rp = r["r_prime"]
        by_ds[ds][r["task_id"]][r["cls_key"]][r["seed"]] = float(rp) if rp is not None else float("nan")

    out: dict = {"datasets": {}}
    passed: set[str] = set()
    print(f"\n{'='*88}\nstep0 SNR 预探针分析{'（smoke）' if smoke else ''}\n{'='*88}")
    for ds in datasets:
        rprime = {tid: dict(cm) for tid, cm in by_ds.get(ds, {}).items()}
        rprime = {tid: {k: dict(sm) for k, sm in cm.items()} for tid, cm in rprime.items()}
        if not rprime:
            print(f"[{ds}] 无记录，跳过（先 --run）")
            continue
        stats = compute_snr_stats(rprime, har_rank=har_rank,
                                  delta_ref=delta_for(ds, labels_cfg))
        verdict = decide(stats)
        if verdict.passed:
            passed.add(ds)
        out["datasets"][ds] = {
            "n_tasks": stats.n_tasks, "n_fail": stats.n_fail,
            "rprime_std_mean": round(stats.rprime_std_mean, 4),
            "cls_spread_std_mean": round(stats.cls_spread_std_mean, 4),
            "snr": round(stats.snr, 2) if stats.snr == stats.snr else None,
            "pair_sign_stability": round(stats.pair_sign_stability, 3)
            if stats.pair_sign_stability == stats.pair_sign_stability else None,
            "monotonicity": round(stats.monotonicity, 3) if stats.monotonicity is not None else None,
            "suggested_delta": stats.suggested_delta,
            "registry_delta": delta_for(ds, labels_cfg),
            "verdict": verdict.label, "passed": verdict.passed, "reason": verdict.reason,
        }
        d = out["datasets"][ds]
        print(f"\n[{ds}] SNR={d['snr']} σ={d['rprime_std_mean']} spread={d['cls_spread_std_mean']} "
              f"符号稳定={d['pair_sign_stability']} 单调={d['monotonicity']}")
        print(f"  建议 δ≈{d['suggested_delta']}（注册表 {d['registry_delta']}）| 判定: {verdict.reason}")

    # 任务数下限闸（本会话新增）：探针骨干 + 通过闸的未实测源
    gate = task_count_gate(registry, passed_untested=passed, floor=100)
    out["gate"] = {"ok": gate.ok, "count": gate.count, "floor": gate.floor,
                   "passed_untested": sorted(passed), "message": gate.message}
    print(f"\n{'—'*88}\n任务数下限闸：{gate.message}")
    (OUTDIR / f"analysis{'_smoke' if smoke else ''}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"分析 → {OUTDIR / ('analysis_smoke.json' if smoke else 'analysis.json')}")
    if not gate.ok and not smoke:
        sys.exit(2)  # 任务数不足：非零退出，提醒回填（smoke 不触发，机制验证用）


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1task×5cls×3seed 低 iters 验链路，写 smoke 缓存")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="只跑指定源数据集（默认全部未实测源 tested=false）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    registry = load_registry()
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = [e.name for e in registry.source if not e.tested]  # 未实测源

    if args.run:
        run(datasets, smoke=args.smoke, resume=args.resume, device=args.device)
    if args.analyze:
        analyze(datasets, smoke=args.smoke)
    if not (args.run or args.analyze):
        ap.error("至少指定 --run / --analyze 之一")


if __name__ == "__main__":
    main()
