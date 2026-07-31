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
from dataclasses import dataclass
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
    append_jsonl,
    clear_inflight,
    evaluate_rprime,
    load_probe_cls as _load_probe_cls,
    load_rprime_cache,
    mark_inflight,
    run_seed_for,
)
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
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"


@dataclass(frozen=True)
class ProbeCfg:
    n_tasks: int = 2       # 每数据集切几个"难任务"测 SNR
    n_cls: int = 10        # 用前 n_cls 个探针 CLS（跨 HAR 好坏排名）
    n_seeds: int = 4       # canonical(s=0) + 噪声(s≥1)


SMOKE = ProbeCfg(n_tasks=1, n_cls=5, n_seeds=3)


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


def run(datasets: list[str], smoke: bool, resume: bool, device: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    early_iters = 30 if smoke else train_cfg["early_iters"]
    pcfg = SMOKE if smoke else ProbeCfg()
    cls_list, har_rank = load_probe_cls(pcfg.n_cls)

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
            "n_classes": entry.n_classes,
            "n_classes_range": list(entry.n_classes_range),
            "tasks": [_task_meta(t) for t in tasks],
            "retry_rounds": [],
        }
        per_ds.append((name, X, y, tasks))
    _write_tasks_meta(tasks_meta)

    bs, lr, evr = train_cfg["batch_size"], train_cfg["lr"], labels_cfg["eval_ratio"]

    def is_done(name, tid, key, s):
        if s == 0 and not smoke:
            return (tid, key, early_iters) in shared_cache
        return (name, tid, key, s) in noise_done

    def exec_worklist(worklist, resume_flag: bool, phase: str) -> None:
        """跑一批 (name, X, y, task, cls, key, s)：逐条落盘 + inflight 打点（Part A-3）。"""
        todo = [w for w in worklist
                if not (resume_flag and is_done(w[0], w[3].task_id, w[5], w[6]))]
        print(f"[{phase}] 设备 {device} | iters={early_iters} | "
              f"计划 {len(worklist)} 条，待跑 {len(todo)}", flush=True)
        for i, (name, X, y, t, c, key, s) in enumerate(todo):
            w_seed, sp_seed = _seed_specs(t.task_id, key, s, early_iters)
            mark_inflight(INFLIGHT, t.task_id, key, early_iters)
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
            clear_inflight(INFLIGHT)
            if s == 0 and not smoke:
                shared_cache[(t.task_id, key, early_iters)] = full
            else:
                noise_done.add((name, t.task_id, key, s))
            st = f"R'={rp:.4f}" if rp_json is not None else f"FAIL {rec['error']}"
            print(f"[{phase} {i+1}/{len(todo)}] {name}/{t.task_id} {key[:8]} s{s} {st} "
                  f"({rec['seconds']}s, n_ev={rec.get('n_eval')})", flush=True)

    # ---- base 轮：seed-major（先跑全部 s=0 填全量缓存，再加深噪声）----
    worklist = []
    for s in range(pcfg.n_seeds):
        for name, X, y, tasks in per_ds:
            for t in tasks:
                for c, key in cls_list:
                    worklist.append((name, X, y, t, c, key, s))
    exec_worklist(worklist, resume, "base")

    # ---- 天花板加难重探（Part A-1）：SNR<1 且贴顶 → 自动加难 ≤2 轮 ----
    def all_recs() -> list[dict]:
        return (_read_jsonl(SMOKE_CACHE) if smoke
                else _read_jsonl(SHARED_CACHE) + _read_jsonl(NOISE_CACHE))

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
            _write_tasks_meta(tasks_meta)
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


def _write_tasks_meta(tasks_meta: dict) -> None:
    (OUTDIR / "tasks_meta.json").write_text(
        json.dumps(tasks_meta, indent=2, ensure_ascii=False), encoding="utf-8")


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
    tasks_meta = json.loads((OUTDIR / "tasks_meta.json").read_text(encoding="utf-8"))

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
