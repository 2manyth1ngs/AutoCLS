"""iters 工作点扫描的跨臂并列分析（协议 results/iters_workpoint/PROTOCOL.md §4）。

把同一批 (任务, CLS, 初始化, 划分) 在全部 iters 臂下的 R' 各自与**同一条**随机
初始化基线相减，按未改动的判据代码出判定，再套 autocls.workpoint 的纯函数判定
（峰/一致性/翻盘/无工作点）。缓存合并、三条配对校验与打表口径沿用
scripts/full_iters_compare.py，历史臂（第3步 290、full-iters 30/2320）只读并入。

用法：
    python scripts/iters_workpoint_scan.py --out results/iters_workpoint/comparison.json
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
from autocls.workpoint import (  # noqa: E402
    arm_epoch, classify_flip, find_peak, no_workpoint_verdict, peak_consistency,
    steps_per_epoch,
)

FILL_TASKS = 2          # 与第3步/full-iters 同：每数据集前 2 个任务
ARM_ROWS = 20           # 每臂满格条数（10 CLS × 2 任务）；测不满的臂不进判定

CAND_TRAINED = [
    "results/candidate_survey/baseline/trained_fill.jsonl",       # 290（第3步）
    "results/full_iters_probe/candidates/trained_fill.jsonl",     # 30 / 2320
    "results/iters_workpoint/candidates/trained_fill.jsonl",      # 本扫描新臂
]

# 期望臂 = 协议 §1 的网格（已有 + 新跑）。缺臂如实报告，不静默。
SPEC = {
    "WISDM2019": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": CAND_TRAINED,
        "expected": [6, 12, 30, 60, 120, 290, 2320],
    },
    "InsectWingbeatSound": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": CAND_TRAINED,
        "expected": [6, 12, 30, 60, 120, 290],
    },
    "UniMiB-SHAR": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": CAND_TRAINED,
        "expected": [6, 12, 30, 60, 120, 290],
    },
    "PAMAP2": {
        "registry": "configs/candidate_pool.yaml",
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "trained": CAND_TRAINED,
        "expected": [6, 12, 30, 60, 120, 290],
    },
    "Handwriting": {   # 正对照；290 的 R' 来自 stage1_rerun（5090，协议 §3 已注明）
        "registry": None,
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
        "trained": [
            "results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl",
            "results/full_iters_probe/uea/trained_fill.jsonl",
            "results/iters_workpoint/uea/trained_fill.jsonl",
        ],
        "expected": [3, 6, 15, 30, 60, 120, 290, 2320],
    },
}
CANDIDATES = ("WISDM2019", "InsectWingbeatSound", "UniMiB-SHAR", "PAMAP2")


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

    # (iters, task_id) → [R']，逐条走与 _trained_by_task 相同的三条校验；
    # 同时收 n_train（epoch 换算）与 device_name（硬件审计，判据 7 附注用）
    trained: dict[int, dict[str, list[float]]] = {}
    spe_seen: dict[str, set[int]] = {}
    devices: set[str] = set()
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
            if r.get("n_train"):
                spe_seen.setdefault(tid, set()).add(steps_per_epoch(int(r["n_train"])))
            if r.get("device_name"):
                devices.add(r["device_name"])
            if r.get("r_prime") is None:
                continue
            trained.setdefault(int(r["iters"]), {}).setdefault(tid, []).append(float(r["r_prime"]))
            audit["matched"] += 1

    # 协议 §0.1：数据集内 steps/epoch 恒定，是 epoch 对齐退化为 per-dataset iters
    # 的前提——不成立就必须报错，而不是悄悄取均值
    spe_all = set().union(*spe_seen.values()) if spe_seen else set()
    if len(spe_all) != 1:
        raise RuntimeError(f"{name}: steps/epoch 不唯一 {spe_seen}，epoch 对齐前提破坏")
    spe = spe_all.pop()

    found = sorted(trained)
    missing = [it for it in spec["expected"] if it not in found]
    extra = [it for it in found if it not in spec["expected"]]

    out = {"dataset": name, "l_enc": entry.l_enc, "n_tasks": len(tasks),
           "steps_per_epoch": spe, "audit": audit, "devices": sorted(devices),
           "missing_arms": missing, "extra_arms": extra, "arms": {}}
    for iters in found:
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
            "epoch": round(arm_epoch(iters, spe), 3),
            "n_trained": n_tr, "complete": n_tr >= ARM_ROWS,
            "baseline": d["baseline_mean"],
            "realized_gain": d["realized_gain_mean"], "best_gain": bg,
            "n_above": n_up, "n_below": n_dn, "n_band": n_tr - n_up - n_dn,
            "T1": d["baseline_mean"] <= CEILING_BASELINE,
            "T2": (bg is not None and bg >= BEST_GAIN_FLOOR),
            "T3": (n_tr > 0 and n_up >= T3_ABOVE_FRACTION * n_tr
                   and n_up >= T3_ABOVE_OVER_BELOW * n_dn),
            "verdict": d["verdict"],
        }
        out["arms"][iters]["usable"] = bool(
            out["arms"][iters]["T1"] and out["arms"][iters]["T2"]
            and out["arms"][iters]["T3"] and n_tr >= ARM_ROWS)

    # 判据 2：峰 = best_gain 的 argmax（测满的臂才参与）；realized 峰作辅助诊断
    complete = {it: a for it, a in out["arms"].items() if a["complete"]}
    out["peak_best_gain"] = find_peak({it: a["best_gain"] for it, a in complete.items()})
    out["peak_realized"] = find_peak({it: a["realized_gain"] for it, a in complete.items()})
    for key in ("peak_best_gain", "peak_realized"):
        if out[key] is not None:
            out[key]["epoch"] = round(arm_epoch(out[key]["iters"], spe), 3)

    # 判据 4/5（只对候选有意义，对照 Handwriting 也算但只作参考）
    out["flip"] = classify_flip(complete)
    n_eval_min = min((r.get("n_eval") for rows in base_rows.values() for r in rows
                      if r.get("n_eval")), default=None)
    out["quantum"] = (1.0 / n_eval_min) if n_eval_min else None
    out["no_workpoint"] = (
        no_workpoint_verdict(complete, quantum=out["quantum"])
        if out["quantum"] and complete else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/iters_workpoint/comparison.json")
    args = ap.parse_args()

    res = {name: collect(name, spec) for name, spec in SPEC.items()}

    print(f"\n{'='*112}")
    print("iters 工作点扫描：同一 (任务, CLS, 初始化, 划分)，唯一变量 = iters；epoch = iters / ⌈n_train/128⌉")
    print(f"{'='*112}")
    print(f"{'数据集':<20}{'iters':>6}{'epoch':>8}{'条数':>5}{'基线':>7}{'实得增益':>10}"
          f"{'最好增益':>10}{'↑':>4}{'↓':>4}{'带内':>5}  {'T1':<3}{'T2':<3}{'T3':<3} {'判定'}")
    f = lambda v, w=10: (f"{v:>+{w}.3f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
    tick = lambda b: ("✅" if b else "❌")
    for name, r in res.items():
        for iters in sorted(r["arms"]):
            a = r["arms"][iters]
            if not a["n_trained"]:
                continue
            mark = "" if a["complete"] else " ⚠测不满"
            print(f"{name:<20}{iters:>6}{a['epoch']:>8.1f}{a['n_trained']:>5}"
                  f"{a['baseline']:>7.3f}{f(a['realized_gain'])}{f(a['best_gain'])}"
                  f"{a['n_above']:>4}{a['n_below']:>4}{a['n_band']:>5}  "
                  f"{tick(a['T1']):<3}{tick(a['T2']):<3}{tick(a['T3']):<3} "
                  f"{a['verdict']}{mark}")
        if r["missing_arms"]:
            print(f"{'':<20}⚠ 缺臂: {r['missing_arms']}")
        print()

    print(f"{'—'*112}\n判定（协议 §4；峰按 best_gain 定义，realized 作辅助）")
    for name, r in res.items():
        pk, pr = r["peak_best_gain"], r["peak_realized"]
        pks = (f"best_gain 峰 iters={pk['iters']}({pk['epoch']}ep, {pk['value']:+.3f}"
               f"{', 截尾:' + pk['censored'] if pk['censored'] else ''})") if pk else "峰不可判"
        prs = (f"realized 峰 iters={pr['iters']}({pr['epoch']}ep, {pr['value']:+.3f}"
               f"{', 截尾:' + pr['censored'] if pr['censored'] else ''})") if pr else "—"
        nw = r["no_workpoint"]
        nws = f" | 无工作点判定: {nw['verdict']}" if (nw and name in CANDIDATES) else ""
        fl = r["flip"]
        fls = (f" | 翻盘: {fl['flip']}@{fl['at']}"
               if (name in CANDIDATES and fl["flip"] != "none") else "")
        print(f"  {name:<20} {pks} | {prs}{nws}{fls}")

    # 判据 3：峰一致性——只取"至少一个测满臂 best_gain ≥ δ"的数据集
    eligible = {}
    for name, r in res.items():
        if r["peak_best_gain"] and any(
                a["complete"] and a["best_gain"] is not None
                and a["best_gain"] >= BEST_GAIN_FLOOR for a in r["arms"].values()):
            eligible[name] = (r["peak_best_gain"]["epoch"], r["peak_best_gain"]["censored"])
    pc = peak_consistency(eligible)
    ratio_s = f"，max/min={pc['ratio']:.2f}" if pc["ratio"] else ""
    print(f"\n  峰一致性（{sorted(eligible)}）: {pc['verdict']}{ratio_s}")

    # 判据 6：Handwriting 5ep（iters=15）vs 当前口径 290
    hw = res["Handwriting"]["arms"]
    if 15 in hw and 290 in hw and hw[15]["complete"]:
        a5, a290 = hw[15], hw[290]
        print(f"\n  Handwriting 5ep vs 290: best_gain {f(a5['best_gain'],7)} vs {f(a290['best_gain'],7)}"
              f" | realized {f(a5['realized_gain'],7)} vs {f(a290['realized_gain'],7)}"
              f" | ↑/↓ {a5['n_above']}/{a5['n_below']} vs {a290['n_above']}/{a290['n_below']}")

    print(f"\n{'—'*112}\n配对审计（classes/split_seed 校验）与硬件来源")
    for name, r in res.items():
        print(f"  {name:<20} {r['audit']} devices={r['devices'] or ['(历史记录无字段)']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"datasets": res, "peak_consistency": {**pc, "datasets": sorted(eligible)}}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
