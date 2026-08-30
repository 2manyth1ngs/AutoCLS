"""金丝雀探针有效性验证（协议 results/canary_validation/PROTOCOL.md；零 GPU）。

320 格面板模拟部署协议：毒哨选择与线性校准只用 15 个训练数据集（task-blind，
无 E/TS2Vec）。主设计 sent_cal（毒哨 j=3 + 线性校准）过 V1 排序门 + V2
分档门 = 有效 → 看门层定案；无效 → 决策树进任务级条件化预检。

用法：python scripts/20_canary_validation.py
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")
ps = importlib.import_module("scripts.18_prior_shrinkage")
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

OUTDIR = ROOT / "results" / "canary_validation"
BUDGETS = (2, 3, 5)
J_PRIMARY, M_SHRINK = 3, 4           # 协议 §3 写死
N_DRAWS = 500
DANGER_TRUE, DANGER_FLAG = 0.40, 0.30   # 协议 §4 V2 阈写死
V1_RHO, V2_BACC = 0.5, 0.75


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def per_cls_destroy_rates(train_cells: list[dict]) -> dict[str, float]:
    """训练折 per-CLS 全局破坏率（task-blind，毒哨排名依据）。"""
    by_cls: dict = defaultdict(list)
    for c in train_cells:
        by_cls[c["cls_key"]].append(int(c["label"] == "destroy"))
    return {k: float(np.mean(v)) for k, v in by_cls.items()}


def sentinel_cells(rates: dict, tids: list, j: int) -> list[tuple]:
    """毒哨设计（协议 §3）：破坏率降序前 j 条（并列按名字），轮转指派任务。"""
    ranked = sorted(rates, key=lambda k: (-rates[k], k))[:j]
    return [(tids[i % len(tids)], k) for i, k in enumerate(ranked)]


def linear_calibrate(s_train, p_train) -> tuple[float, float]:
    """最小二乘 p≈a+b·s（毒哨有偏样本的偏差校正）；s 全同退化为均值。"""
    s = np.asarray(s_train, dtype=float)
    p = np.asarray(p_train, dtype=float)
    if np.allclose(s, s[0]):
        return float(p.mean()), 0.0
    b, a = np.polyfit(s, p, 1)
    return float(a), float(b)


def calibrated_predict(s: float, a: float, b: float) -> float:
    return float(np.clip(a + b * s, 0.0, 1.0))


def balanced_accuracy(y_true, y_pred) -> float:
    y, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tpr = (y & p).sum() / max(y.sum(), 1)
    tnr = (~y & ~p).sum() / max((~y).sum(), 1)
    return float((tpr + tnr) / 2)


def v1_pass(rho: float, pval: float) -> bool:
    return bool(rho >= V1_RHO and pval < 0.05)


def v2_pass(bacc: float) -> bool:
    return bool(bacc >= V2_BACC)


def verdict_canary(v1: bool, v2: bool) -> str:
    if v1 and v2:
        return "valid"
    if v1 or v2:
        return "edge"
    return "invalid"


# ---------------------------------------------------------------- R1 数据装配
def collect_k3_per_seed(path, shared_keys, bands, classes) -> dict:
    """噪声缓存逐种子行 → {(tid, key): [r' 按 weight_seed 升序]}（R1 单种子臂）。"""
    vals: dict = defaultdict(list)
    for r in dm._c2.read_jsonl(path):
        tid, key = r.get("task_id"), r.get("cls_key")
        if (tid is None or r.get("r_prime") is None or r.get("iters") != 290
                or key not in shared_keys or tid not in bands
                or list(r.get("classes", [])) != list(classes.get(tid, []))):
            continue
        vals[(tid, key)].append((r.get("weight_seed", 0), float(r["r_prime"])))
    return {k: [x for _, x in sorted(v)] for k, v in vals.items() if len(v) == 3}


# ---------------------------------------------------------------- 主流程
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = lo.load_panel()
    datasets = sorted({r["dataset"] for r in rows})
    tids_by_ds: dict = defaultdict(list)
    flag: dict = {}
    band_of: dict = {}
    for r in rows:
        if r["task_id"] not in tids_by_ds[r["dataset"]]:
            tids_by_ds[r["dataset"]].append(r["task_id"])
        flag[(r["task_id"], r["cls_key"])] = int(r["label"] == "destroy")
        band_of[r["task_id"]] = (r["base_mean"], r["band"])
    for d in tids_by_ds:
        tids_by_ds[d].sort()
    p_true = {d: float(np.mean([flag[(t, r["cls_key"])] for r in rows
                                if r["dataset"] == d
                                for t in [r["task_id"]]]))
              for d in datasets}

    per_fold: dict = {}
    for heldout, train_ds in lo.fold_split(datasets):
        train_cells = [r for r in rows if r["dataset"] != heldout]
        rates = per_cls_destroy_rates(train_cells)
        entry = {"p_true": round(p_true[heldout], 4), "sent": {}, "rand": {}}
        for j in BUDGETS:
            # sent_cal：确定性单值
            s_train = [float(np.mean([flag[c] for c in
                                      sentinel_cells(rates, tids_by_ds[d], j)]))
                       for d in train_ds]
            a, b = linear_calibrate(s_train, [p_true[d] for d in train_ds])
            sent = sentinel_cells(rates, tids_by_ds[heldout], j)
            s_held = float(np.mean([flag[c] for c in sent]))
            entry["sent"][f"j{j}"] = {
                "cells": [f"{t}:{k[:8]}" for t, k in sent],
                "s": s_held, "a": round(a, 4), "b": round(b, 4),
                "est": calibrated_predict(s_held, a, b)}
            # shrink_rand 对照：逐 draw
            prior_g = float(np.mean([flag[(r["task_id"], r["cls_key"])]
                                     for r in train_cells]))
            cells = [flag[(r["task_id"], r["cls_key"])]
                     for r in rows if r["dataset"] == heldout]
            rng = np.random.default_rng(
                [run_seed_for(f"{heldout}:j{j}", "canary_rand"), 20260901])
            ests = [ps.shrink(float(np.mean(
                np.asarray(cells)[rng.choice(len(cells), size=j, replace=False)])),
                prior_g, j=j, m=M_SHRINK) for _ in range(N_DRAWS)]
            entry["rand"][f"j{j}"] = ests
        per_fold[heldout] = entry

    truth = np.array([p_true[d] for d in datasets])
    danger = truth >= DANGER_TRUE
    tests: dict = {"budgets": {}}
    for j in BUDGETS:
        est = np.array([per_fold[d]["sent"][f"j{j}"]["est"] for d in datasets])
        rho, pv = dm.spearman(list(est), list(truth))
        bacc = balanced_accuracy(danger, est >= DANGER_FLAG)
        # 对照：逐 draw Spearman 分布
        rand = np.array([per_fold[d]["rand"][f"j{j}"] for d in datasets])  # (16, R)
        rhos = [dm.spearman(list(rand[:, r]), list(truth))[0]
                for r in range(N_DRAWS)]
        baccs = [balanced_accuracy(danger, rand[:, r] >= DANGER_FLAG)
                 for r in range(N_DRAWS)]
        tests["budgets"][f"j{j}"] = {
            "sent_rho": rho, "sent_rho_p": pv, "sent_bacc": bacc,
            "rand_rho_median": float(np.median(rhos)),
            "rand_rho_p05": float(np.percentile(rhos, 5)),
            "rand_bacc_median": float(np.median(baccs))}
    prim = tests["budgets"][f"j{J_PRIMARY}"]
    v1 = v1_pass(prim["sent_rho"], prim["sent_rho_p"])
    v2 = v2_pass(prim["sent_bacc"])
    tests["V1_pass"], tests["V2_pass"] = v1, v2
    tests["P3e_sent_gt_rand"] = bool(prim["sent_rho"] > prim["rand_rho_median"])

    # R1 单种子现实性（13 个 k3 数据集；校准表保持 k3——部署时离线数据即 k3）
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    bands, classes, _rej = dm.collect_baselines([fl.BASELINE_CACHE],
                                                exclude_plus_c=True)
    per_seed = collect_k3_per_seed(fl.K3_CACHE, shared_keys, bands, classes)
    k3_ds = sorted({fl.dataset_of(t) for t, _ in per_seed})
    r1 = {}
    for rep in range(3):
        ests, tr = [], []
        for d in k3_ds:
            e = per_fold[d]["sent"][f"j{J_PRIMARY}"]
            rates = per_cls_destroy_rates([r for r in rows if r["dataset"] != d])
            sent = sentinel_cells(rates, tids_by_ds[d], J_PRIMARY)
            s = float(np.mean([dm.destruction_flag(per_seed[(t, k)][rep],
                                                   *band_of[t])
                               for t, k in sent]))
            ests.append(calibrated_predict(s, e["a"], e["b"]))
            tr.append(p_true[d])
        rho, pv = dm.spearman(ests, tr)
        bacc = balanced_accuracy(np.asarray(tr) >= DANGER_TRUE,
                                 np.asarray(ests) >= DANGER_FLAG)
        r1[f"rep{rep}"] = {"rho": rho, "p": pv, "bacc": bacc}
    tests["R1_single_seed"] = r1
    tests["P4e_r1_rho_positive_all"] = all(v["rho"] > 0 for v in r1.values())

    verdict = verdict_canary(v1, v2)
    out = {"protocol": "results/canary_validation/PROTOCOL.md",
           "per_fold": {d: {k: v for k, v in e.items() if k != "rand"}
                        for d, e in per_fold.items()},
           "tests": tests, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print(f"===== 金丝雀有效性（16 折 LOO，主判 j={J_PRIMARY}）=====")
    print(f"{'数据集':<28}{'真值':>6}{'哨兵s':>7}{'估计':>7}  哨兵格")
    for d in datasets:
        e = per_fold[d]["sent"][f"j{J_PRIMARY}"]
        print(f"{d:<28}{p_true[d]:>6.2f}{e['s']:>7.2f}{e['est']:>7.3f}  "
              f"{e['cells']}")
    for j in BUDGETS:
        t = tests["budgets"][f"j{j}"]
        print(f"j={j}: sent ρ={t['sent_rho']:.3f}(p={t['sent_rho_p']:.4f}) "
              f"bacc={t['sent_bacc']:.3f} | rand ρ中位={t['rand_rho_median']:.3f}"
              f"(p05={t['rand_rho_p05']:.3f}) bacc中位={t['rand_bacc_median']:.3f}")
    print(f"V1(ρ≥{V1_RHO} 且 p<0.05)={'过' if v1 else '不过'} | "
          f"V2(bacc≥{V2_BACC})={'过' if v2 else '不过'}")
    print("R1 单种子: " + "  ".join(
        f"rep{r}:ρ={v['rho']:.3f} bacc={v['bacc']:.3f}"
        for r, v in ((k[-1], v) for k, v in r1.items())))
    print(f"\n[判读] → {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
