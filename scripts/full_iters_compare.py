"""full-iters 判别实验的跨 iters 并列分析（协议 results/full_iters_probe/PROTOCOL.md §4）。

只做一件事：把同一批 (任务, CLS, 初始化, 划分) 在 **iters ∈ {30, 290, 2320}** 下的
R' 各自与**同一条**随机初始化基线相减，按**未改动的**判据代码出判定。

判据一行不改：T1/T2/T3 全部由 `autocls.baseline_probe.classify_verdict` 计算，
基线统计由 `summarize_task`/`summarize_dataset` 计算——与第3步逐字段同一实现。
本脚本只负责按 iters 分组喂数据，以及打并列表。

配对校验沿用 `_trained_by_task` 的三条（iters / classes / split_seed），逐条审计，
不静默丢弃。

用法：
    python scripts/full_iters_compare.py --out results/full_iters_probe/comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import (  # noqa: E402
    BEST_GAIN_FLOOR, CEILING_BASELINE, T3_ABOVE_FRACTION, T3_ABOVE_OVER_BELOW,
    baseline_seed_specs, summarize_dataset, summarize_task,
)
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)

FILL_TASKS = 2          # 与第3步 §4.1 同：每数据集前 2 个任务
ITERS_ARMS = (30, 290, 2320)

# 每个数据集：注册表、基线缓存、以及**全部**可能含其 trained R' 的缓存。
# 290 那批在第3步/阶段1 的历史目录里，30/2320 在本实验的独立 outdir（协议 §2）。
SPEC = {
    "WISDM2019": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/full_iters_probe/candidates/baseline_cache.jsonl",
        "trained": ["results/candidate_survey/baseline/trained_fill.jsonl",
                    "results/full_iters_probe/candidates/trained_fill.jsonl"],
    },
    "InsectWingbeatSound": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/full_iters_probe/candidates/baseline_cache.jsonl",
        "trained": ["results/candidate_survey/baseline/trained_fill.jsonl",
                    "results/full_iters_probe/candidates/trained_fill.jsonl"],
    },
    "Handwriting": {   # 对照：UEA 唯一无争议 usable，用源池注册表
        "registry": None,
        "baseline": "results/full_iters_probe/uea/baseline_cache.jsonl",
        "trained": ["results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl",
                    "results/full_iters_probe/uea/trained_fill.jsonl"],
    },
}


def read_jsonl(path: Path) -> list[dict]:
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


def collect(name: str, spec: dict) -> dict:
    reg = load_registry(ROOT / spec["registry"]) if spec["registry"] else load_registry()
    entry = reg.get(name)
    _X, y = load_source_dataset(name, reg)
    _, tasks = slice_source_tasks(entry, reg, y)
    tasks = tasks[:FILL_TASKS]
    by_id = {t.task_id: t for t in tasks}

    base_rows: dict[str, list[dict]] = {}
    for r in read_jsonl(ROOT / spec["baseline"]):
        if r["task_id"] in by_id:
            base_rows.setdefault(r["task_id"], []).append(r)

    # (iters, task_id) → [R']，逐条走与 _trained_by_task 相同的三条校验
    trained: dict[int, dict[str, list[float]]] = {}
    audit = {"matched": 0, "rejected_classes": 0, "rejected_split": 0, "unknown_task": 0}
    for p in spec["trained"]:
        for r in read_jsonl(ROOT / p):
            tid = r["task_id"]
            t = by_id.get(tid)
            if t is None or r.get("dataset") not in (None, name):
                audit["unknown_task"] += 1
                continue
            if list(r.get("classes", [])) != list(t.classes):
                audit["rejected_classes"] += 1
                continue
            if r.get("split_seed") != baseline_seed_specs(tid, 0)[1]:
                audit["rejected_split"] += 1
                continue
            if r.get("r_prime") is None:
                continue
            trained.setdefault(int(r["iters"]), {}).setdefault(tid, []).append(float(r["r_prime"]))
            audit["matched"] += 1

    out = {"dataset": name, "l_enc": entry.l_enc, "n_tasks": len(tasks), "audit": audit,
           "arms": {}}
    for iters in sorted(set(ITERS_ARMS) | set(trained)):
        per_task = {}
        for tid, rows in base_rows.items():
            st = summarize_task(rows, n_classes=len(by_id[tid].classes),
                                trained=trained.get(iters, {}).get(tid, []))
            st["dataset"] = name
            per_task[tid] = st
        d = summarize_dataset(per_task)
        n_tr, n_up, n_dn = d["n_trained_total"], d["n_trained_above_null"], d["n_trained_below_null"]
        bg = d["best_gain_max"]
        out["arms"][iters] = {
            "n_trained": n_tr, "baseline": d["baseline_mean"],
            "realized_gain": d["realized_gain_mean"], "best_gain": bg,
            "n_above": n_up, "n_below": n_dn, "n_band": n_tr - n_up - n_dn,
            "T1": d["baseline_mean"] <= CEILING_BASELINE,
            "T2": (bg is not None and bg >= BEST_GAIN_FLOOR),
            "T3": (n_tr > 0 and n_up >= T3_ABOVE_FRACTION * n_tr
                   and n_up >= T3_ABOVE_OVER_BELOW * n_dn),
            "verdict": d["verdict"],
            "null_band": {tid: s["null_band"] for tid, s in per_task.items()},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/full_iters_probe/comparison.json")
    args = ap.parse_args()

    res = {name: collect(name, spec) for name, spec in SPEC.items()}

    print(f"\n{'='*104}")
    print("full-iters 判别实验：同一 (任务, CLS, 初始化, 划分)，唯一变量 = iters")
    print(f"{'='*104}")
    hdr = (f"{'数据集':<22}{'iters':>6}{'条数':>5}{'基线':>7}{'实得增益':>10}{'最好增益':>10}"
           f"{'↑':>4}{'↓':>4}{'带内':>5}  {'T1':<3}{'T2':<3}{'T3':<3} {'判定'}")
    print(hdr)
    for name, r in res.items():
        for iters in sorted(r["arms"]):
            a = r["arms"][iters]
            if not a["n_trained"]:
                continue
            f = lambda v, w=10: (f"{v:>+{w}.3f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
            tick = lambda b: ("✅" if b else "❌")
            print(f"{name:<22}{iters:>6}{a['n_trained']:>5}{a['baseline']:>7.3f}"
                  f"{f(a['realized_gain'])}{f(a['best_gain'])}"
                  f"{a['n_above']:>4}{a['n_below']:>4}{a['n_band']:>5}  "
                  f"{tick(a['T1']):<3}{tick(a['T2']):<3}{tick(a['T3']):<3} {a['verdict']}")
        print()

    print(f"{'—'*104}\n配对审计（三条校验：iters/classes/split_seed）")
    for name, r in res.items():
        print(f"  {name:<22} {r['audit']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
