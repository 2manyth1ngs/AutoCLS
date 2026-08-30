"""MX-1 三态可行性 label（协议 results/mx1_feasibility/PROTOCOL.md §2-§3；零 GPU）。

全部来自既有缓存：R' = k3 均值优先（stage1 rerun 噪声缓存）+ k1 canonical 补齐
3 个骨干（trained_fill）；基线带与 scripts/14_destruction_matrix.py 逐字同式
（直接复用其 collect_* 与 destruction_flag，配对校验一并继承）。

并表规则（协议 §2 写死）：k1 只用于 k3 零覆盖的数据集；同数据集混源一律拒收。

用法：python scripts/15_feasibility_labels.py [--force]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.labeling import load_probe_cls  # noqa: E402

OUTDIR = ROOT / "results" / "mx1_feasibility"
LABELS_PATH = OUTDIR / "labels.jsonl"
AUDIT_PATH = OUTDIR / "label_audit.json"
K3_CACHE = (ROOT / "results" / "stage1_rerun_20260804" / "results"
            / "step0_snr_gate" / "noise_cache.jsonl")
FILL_CACHE = ROOT / "results" / "baseline_probe" / "trained_fill.jsonl"
BASELINE_CACHE = ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
ITERS = 290                      # label 绑 @290 口径（协议 §1）
LABELS = ("improve", "inert", "destroy")   # 三态顺序唯一真源（16 脚本引用）


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def dataset_of(task_id: str) -> str:
    """task_id → 数据集名（与 scripts/14 内联口径同式，含 +c 档标剥离）。"""
    return task_id.rsplit("_", 1)[0].split("+", 1)[0]


def tri_state(r_prime: float, base_mean: float, band: float) -> str:
    """协议 §3：破坏腿逐字复用 destruction_flag；提升 = 严格大于 mean+band。"""
    if dm.destruction_flag(r_prime, base_mean, band):
        return "destroy"
    if r_prime > base_mean + band:
        return "improve"
    return "inert"


def merge_trained(k3: dict, k1: dict) -> tuple[dict, dict]:
    """并表（协议 §2 写死）：k1 只用于 k3 零覆盖的数据集，拒收计数如实。

    k3/k1: {(task_id, cls_key): r_prime} → {(tid, key): {r_prime, r_source}}。
    """
    k3_datasets = {dataset_of(tid) for tid, _ in k3}
    merged = {k: {"r_prime": v, "r_source": "k3"} for k, v in k3.items()}
    rejects: dict = defaultdict(int)
    for (tid, key), v in k1.items():
        if dataset_of(tid) in k3_datasets:
            rejects["k1_dataset_has_k3"] += 1
            continue
        if (tid, key) in merged:   # 数据集不相交时不可达，防御性
            rejects["k1_duplicate"] += 1
            continue
        merged[(tid, key)] = {"r_prime": v, "r_source": "k1"}
    return merged, dict(rejects)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 labels.jsonl")
    args = ap.parse_args()
    if LABELS_PATH.exists() and not args.force:
        sys.exit(f"{LABELS_PATH} 已存在：确认要重造加 --force（确定性重建，缓存不受影响）")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    shared_keys = {k for _c, k in load_probe_cls(PROBE_CATALOG, 10)[0]}
    bands, classes, rej_base = dm.collect_baselines(
        [BASELINE_CACHE], exclude_plus_c=True)
    k3, rej_k3 = dm.collect_trained_k3(K3_CACHE, shared_keys, bands, classes)
    k1, rej_k1 = dm.collect_trained(
        [FILL_CACHE], shared_keys, bands, classes, lambda ds: ITERS)
    merged, rej_merge = merge_trained(k3, k1)

    rows = []
    for (tid, key), info in sorted(merged.items()):
        mean, band = bands[tid]
        rows.append({
            "dataset": dataset_of(tid), "task_id": tid, "cls_key": key,
            "classes": list(classes[tid]), "iters": ITERS,
            "r_source": info["r_source"], "r_prime": info["r_prime"],
            "base_mean": mean, "band": band,
            "label": tri_state(info["r_prime"], mean, band),
        })
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 审计：面板满格性 + 分布 + 拒收 ----
    per_ds: dict = defaultdict(Counter)
    tids_per_ds: dict = defaultdict(set)
    for r in rows:
        per_ds[r["dataset"]][r["label"]] += 1
        per_ds[r["dataset"]]["_cells"] += 1
        tids_per_ds[r["dataset"]].add(r["task_id"])
    incomplete = {ds: c["_cells"] for ds, c in per_ds.items() if c["_cells"] != 20}
    audit = {
        "protocol": "results/mx1_feasibility/PROTOCOL.md",
        "iters": ITERS,
        "n_cells": len(rows),
        "n_datasets": len(per_ds),
        "expected_cells_per_dataset": 20,
        "incomplete_datasets": incomplete,
        "k1_datasets": sorted({r["dataset"] for r in rows if r["r_source"] == "k1"}),
        "per_dataset": {ds: {"n_cells": c["_cells"],
                             **{s: c.get(s, 0) for s in LABELS},
                             "destroy_rate": c.get("destroy", 0) / c["_cells"]}
                        for ds, c in sorted(per_ds.items())},
        "global_rates": {s: sum(1 for r in rows if r["label"] == s) / len(rows)
                         for s in LABELS},
        "rejects": {"baseline": rej_base, "k3": rej_k3, "k1": rej_k1,
                    "merge": rej_merge},
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print(f"labels: {len(rows)} cells / {len(per_ds)} 数据集 → {LABELS_PATH}")
    print(f"{'数据集':<28}{'cells':>6}{'提升':>6}{'惰性':>6}{'破坏':>6}{'破坏率':>8}  源")
    for ds, c in sorted(per_ds.items(), key=lambda kv: -kv[1].get("destroy", 0)):
        src = "k1" if ds in audit["k1_datasets"] else "k3"
        print(f"{ds:<28}{c['_cells']:>6}{c.get('improve', 0):>6}"
              f"{c.get('inert', 0):>6}{c.get('destroy', 0):>6}"
              f"{c.get('destroy', 0) / c['_cells']:>8.2f}  {src}")
    g = audit["global_rates"]
    print(f"全局率: 提升 {g['improve']:.3f} | 惰性 {g['inert']:.3f} | "
          f"破坏 {g['destroy']:.3f}")
    if incomplete:
        print(f"⚠️ 缺格数据集: {incomplete}（协议 §2：<10 cell 的折剔除）")
    print(f"审计 → {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
