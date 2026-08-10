"""workpoint followup 分析（协议 results/workpoint_followup/PROTOCOL.md）。

三件事，判据代码全部复用（T1/T2/T3 与扫描逐字不动；确认层级/Wilson/破坏参与
是 autocls.workpoint 的纯函数，先测后写）：
1. 重扫判定：ChestAccel / USC-HAD / ElectricDevices 在 {6,12,30,60,120} 新臂 +
   290 历史臂上的逐臂判定与翻盘（口径 = iters_workpoint_scan.collect 原样，
   n_tasks=2 / arm_rows=20）。
2. 加测确认：Insect @6/@12、UniMiB @12/@30/@60 扩到 60 条（n_tasks=6 /
   arm_rows=60 / allow_mixed_spe——Insect 任务02 n_train=560→5 steps/ep，
   epoch 对齐偏差 ≤1.2×，判定按 iters 臂）。确认层级 A/B/C 见 confirm_verdict
   （用户裁决：B 不算入池依据）。
3. 破坏参与率逐臂报告（如实报告，不设硬门槛）：δ=0.05 主口径 + 0.03 敏感性；
   参照臂（@290 与 Handwriting 工作点）用既有缓存并列。

用法：
    python scripts/workpoint_followup.py --out results/workpoint_followup/followup.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import summarize_task  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.workpoint import (  # noqa: E402
    confirm_verdict, destruction_participation, wilson_interval,
)
from scripts.iters_workpoint_scan import collect, read_jsonl  # noqa: E402

OUTDIR = "results/workpoint_followup/candidates"
CAND_BASE = f"{OUTDIR}/baseline_cache.jsonl"
CAND_FILL = f"{OUTDIR}/trained_fill.jsonl"

# --- 任务1 重扫（口径 = 扫描协议 §1，20 条/臂；290 臂从第3步缓存只读并入）---
RESCAN = {
    name: {
        "registry": "configs/candidate_pool.yaml",
        "baseline": CAND_BASE,
        "trained": ["results/candidate_survey/baseline/trained_fill.jsonl", CAND_FILL],
        "expected": [6, 12, 30, 60, 120, 290],
    }
    for name in ("ChestAccel", "USC-HAD", "ElectricDevices")
}

# --- 任务2 加测确认（60 条/臂；只读 followup 文件——既有 20 条已复制入内，
# 不能再并旧缓存，否则同一条测量会被计两次）---
CONFIRM = {
    "InsectWingbeatSound": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": CAND_BASE, "trained": [CAND_FILL], "expected": [6, 12],
    },
    "UniMiB-SHAR": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": CAND_BASE, "trained": [CAND_FILL], "expected": [12, 30, 60],
    },
}
CONFIRM_WORKPOINTS = {"InsectWingbeatSound": [6, 12], "UniMiB-SHAR": [30]}

# --- 破坏参与的参照臂（既有缓存，n=20；与 followup 臂并列看趋势）---
REFERENCE_ARMS = {
    ("InsectWingbeatSound", 290): {
        "registry": "configs/candidate_pool.yaml", "n_tasks": 2,
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": ["results/candidate_survey/baseline/trained_fill.jsonl"],
    },
    ("UniMiB-SHAR", 290): {
        "registry": "configs/candidate_pool.yaml", "n_tasks": 2,
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": ["results/candidate_survey/baseline/trained_fill.jsonl"],
    },
    ("Handwriting", 60): {
        "registry": None, "n_tasks": 2,
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
        "trained": ["results/iters_workpoint/uea/trained_fill.jsonl"],
    },
    ("Handwriting", 120): {
        "registry": None, "n_tasks": 2,
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
        "trained": ["results/iters_workpoint/uea/trained_fill.jsonl"],
    },
    ("Handwriting", 290): {
        "registry": None, "n_tasks": 2,
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
        "trained": ["results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl"],
    },
}

DELTA_MAIN, DELTA_SENS = 0.05, 0.03   # 主口径 = T2 floor；敏感性 = 阶梯下限


def _tasks_and_baselines(name: str, registry: str | None, n_tasks: int,
                         baseline_path: str):
    """→ (by_id, {task_id: (基线均值, null_band)})——带子与 ↑/↓ 判定同源
    （summarize_task 的 null_band = 2·max(σ, 1/n_eval)）。"""
    reg = load_registry(ROOT / registry) if registry else load_registry()
    entry = reg.get(name)
    _X, y = load_source_dataset(name, reg)
    _, tasks = slice_source_tasks(entry, reg, y)
    tasks = tasks[:n_tasks]
    by_id = {t.task_id: t for t in tasks}
    rows_by_task: dict[str, list[dict]] = {}
    for r in read_jsonl(ROOT / baseline_path):
        if r["task_id"] in by_id:
            rows_by_task.setdefault(r["task_id"], []).append(r)
    bands = {}
    for tid, rows in rows_by_task.items():
        st = summarize_task(rows, n_classes=len(by_id[tid].classes))
        bands[tid] = (st["baseline_mean"], st["null_band"])
    return by_id, bands


def _arm_cls_rprime(name: str, iters: int, by_id: dict, trained_paths: list[str],
                    bands: dict) -> dict[str, dict[str, float]]:
    """→ {task_id: {cls_key: R'}}，配对校验与 collect 相同（classes/split_seed）。"""
    from autocls.baseline_probe import baseline_seed_specs

    per_task: dict[str, dict[str, float]] = {}
    for p in trained_paths:
        for r in read_jsonl(ROOT / p):
            t = by_id.get(r["task_id"])
            if t is None or r.get("dataset") not in (None, name):
                continue
            if int(r["iters"]) != iters or r.get("r_prime") is None:
                continue
            if list(r.get("classes", [])) != list(t.classes):
                continue
            if r.get("split_seed") != baseline_seed_specs(r["task_id"], 0)[1]:
                continue
            per_task.setdefault(r["task_id"], {})[r["cls_key"]] = float(r["r_prime"])
    return {tid: m for tid, m in per_task.items() if tid in bands}


def destruction_table(specs: dict) -> list[dict]:
    """specs: {(name, iters): {registry, n_tasks, baseline, trained}} → 逐臂行。"""
    out = []
    cache: dict[tuple, tuple] = {}
    for (name, iters), sp in specs.items():
        key = (name, sp["registry"], sp["n_tasks"], sp["baseline"])
        if key not in cache:
            cache[key] = _tasks_and_baselines(name, sp["registry"], sp["n_tasks"],
                                              sp["baseline"])
        by_id, bands = cache[key]
        per_task = _arm_cls_rprime(name, iters, by_id, sp["trained"], bands)
        row = {"dataset": name, "iters": iters,
               "n_rows": sum(len(m) for m in per_task.values())}
        for tag, delta in (("main", DELTA_MAIN), ("sens", DELTA_SENS)):
            row[tag] = {"delta": delta,
                        **destruction_participation(per_task, bands, delta=delta)}
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/workpoint_followup/followup.json")
    args = ap.parse_args()

    # ---- 任务1：重扫 --------------------------------------------------------
    rescan = {name: collect(name, spec) for name, spec in RESCAN.items()}
    print(f"\n{'='*104}\n任务1 重扫（判据 T1/T2/T3 逐字不动；20 条/臂；290 = 第3步历史臂）\n{'='*104}")
    print(f"{'数据集':<17}{'iters':>6}{'epoch':>7}{'条数':>5}{'基线':>7}{'实得':>8}"
          f"{'最好':>8}{'↑':>4}{'↓':>4}{'带内':>5}  {'T1':<3}{'T2':<3}{'T3':<3} 判定")
    fmt = lambda v, w=8: (f"{v:>+{w}.3f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
    tick = lambda b: "✅" if b else "❌"
    for name, r in rescan.items():
        for iters in sorted(r["arms"]):
            a = r["arms"][iters]
            if not a["n_trained"]:
                continue
            print(f"{name:<17}{iters:>6}{a['epoch']:>7.1f}{a['n_trained']:>5}"
                  f"{a['baseline']:>7.3f}{fmt(a['realized_gain'])}{fmt(a['best_gain'])}"
                  f"{a['n_above']:>4}{a['n_below']:>4}{a['n_band']:>5}  "
                  f"{tick(a['T1']):<3}{tick(a['T2']):<3}{tick(a['T3']):<3} {a['verdict']}")
        fl = r["flip"]
        nw = r["no_workpoint"]
        print(f"{'':<17}翻盘: {fl['flip']}@{fl['at']}"
              f" | 无工作点判定: {nw['verdict'] if nw else '—'}"
              f"{' | ⚠ 缺臂: ' + str(r['missing_arms']) if r['missing_arms'] else ''}\n")

    # ---- 任务2：加测确认 ----------------------------------------------------
    confirm = {}
    print(f"{'='*104}\n任务2 加测确认（60 条 = 10 CLS × 前 6 任务；层级 A/B/C，B 不算入池——用户裁决）\n{'='*104}")
    print(f"{'数据集':<21}{'iters':>6}{'条数':>5}{'基线':>7}{'最好':>8}{'↑':>4}{'↓':>4}"
          f"{'带内':>5}{'Wilson90':>16}  层级")
    for name, spec in CONFIRM.items():
        r = collect(name, spec, n_tasks=6, arm_rows=60, allow_mixed_spe=True)
        confirm[name] = r
        for iters in sorted(r["arms"]):
            a = r["arms"][iters]
            cv = confirm_verdict(n_above=a["n_above"], n_below=a["n_below"],
                                 n_trained=a["n_trained"], t1=a["T1"], t2=a["T2"])
            a["confirm"] = cv
            a["is_workpoint"] = iters in CONFIRM_WORKPOINTS.get(name, [])
            wl = (f"[{cv['wilson_low']:.3f},{cv['wilson_high']:.3f}]"
                  if cv["wilson_low"] is not None else "—")
            role = "工作点" if a["is_workpoint"] else "邻点"
            print(f"{name:<21}{iters:>6}{a['n_trained']:>5}{a['baseline']:>7.3f}"
                  f"{fmt(a['best_gain'])}{a['n_above']:>4}{a['n_below']:>4}"
                  f"{a['n_band']:>5}{wl:>16}  {cv['tier']}（{role}）")
        if r["missing_arms"]:
            print(f"{'':<21}⚠ 缺臂: {r['missing_arms']}")
        print()

    # ---- 破坏参与率（逐臂，如实报告，不设硬门槛）----------------------------
    specs = {}
    for name, spec in CONFIRM.items():
        for iters in spec["expected"]:
            specs[(name, iters)] = {"registry": spec["registry"], "n_tasks": 6,
                                    "baseline": spec["baseline"],
                                    "trained": spec["trained"]}
    for name, spec in RESCAN.items():
        for iters in spec["expected"]:
            specs[(name, iters)] = {"registry": spec["registry"], "n_tasks": 2,
                                    "baseline": spec["baseline"],
                                    "trained": spec["trained"]}
    specs.update(REFERENCE_ARMS)
    dtable = destruction_table(specs)
    print(f"{'='*104}\n破坏参与率逐臂报告（δ=0.05 主 / 0.03 敏感；strict = 至少一方 < 基线−null_band）\n{'='*104}")
    print(f"{'数据集':<21}{'iters':>6}{'条数':>5}{'<带':>5}{'<均值':>6}"
          f"{'存活对':>8}{'存活率':>8}{'strict参与':>11}{'loose参与':>10}")
    pct = lambda v: f"{v:>7.0%}" if v is not None else f"{'—':>7}"
    for row in sorted(dtable, key=lambda r: (r["dataset"], r["iters"])):
        m = row["main"]
        print(f"{row['dataset']:<21}{row['iters']:>6}{m['n_cls']:>5}"
              f"{m['n_below_band']:>5}{m['n_below_mean']:>6}"
              f"{m['n_survive']:>5}/{m['n_pairs']:<4}{pct(m['survive_rate'])}"
              f" {m['n_strict']:>3}/{m['n_survive']:<3}{pct(m['strict_share'])}"
              f" {m['n_loose']:>3}/{m['n_survive']:<3}{pct(m['loose_share'])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"rescan": rescan, "confirm": confirm, "destruction": dtable,
               "wilson_threshold_60": {"escape_low": 37,
                                       "low_37": wilson_interval(37, 60)[0],
                                       "low_36": wilson_interval(36, 60)[0]}}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
