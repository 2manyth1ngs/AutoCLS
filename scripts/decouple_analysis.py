"""解耦实验主分析（PREREGISTRATION.md §5 清单的执行件；跑完只填数，不加不减）。

判定核心逐字复用已提交代码，不重写：
- gate v2 判定/产量/正确率/留出 δ*：scripts/gate_v2_eval.py（commit 1136cfb 定义）
- 退化标注（v2 破坏性检验）：scripts/validate_degenerate_criterion.py
输入：results/decouple_random_cls/main/{rprime_cache,noise_cache,baseline_cache}.jsonl
输出：results/decouple_random_cls/analysis.json + 控制台摘要（REPORT.md 手写引用）。

用法：python scripts/decouple_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import gate_v2_eval as gv  # noqa: E402  判据唯一真源（A_FLOOR/Y_FLOOR/网格在其内）
import validate_degenerate_criterion as vd  # noqa: E402  退化判据 v2

FOUR = ["ArticularyWordRecognition", "CharacterTrajectories", "ERing", "NATOPS"]
PLANNED = {"ArticularyWordRecognition": 8, "CharacterTrajectories": 16,
           "ERing": 6, "NATOPS": 10}
REGISTRY_DELTA = {"ArticularyWordRecognition": 0.08, "CharacterTrajectories": 0.05,
                  "ERing": 0.06, "NATOPS": 0.06}   # configs/labels.yaml（对照用途）

MAIN = ROOT / "results" / "decouple_random_cls" / "main"
CATALOG = ROOT / "results" / "decouple_random_cls" / "random_cls_catalog.json"
OUT = ROOT / "results" / "decouple_random_cls" / "analysis.json"

SNAPSHOT = ROOT / "results" / "stage1_rerun_20260804" / "results"
N_SUBSETS = 2000
SUBSET_K = 10
SUBSET_SEED = 20260804
FINAL_BASELINE_PEARSON = 0.584   # VALIDATION.md §6（v2 标注、旧缓存、同 4 数据集）


# ---------------------------------------------------------------------------
# 纯函数（tests/test_decouple_analysis.py 覆盖）
# ---------------------------------------------------------------------------


def prediction_verdict(passed: dict[str, bool]) -> dict:
    """§1 写死规则。passed = R-reg 口径各数据集 gate v2 是否 pass。

    NATOPS 阳性对照优先：它不过 → design_failure（不得把其余 fail 当预测应验）。
    """
    others = ["ArticularyWordRecognition", "CharacterTrajectories", "ERing"]
    if not passed["NATOPS"]:
        return {"outcome": "design_failure",
                "deviations": ["NATOPS"],
                "note": "阳性对照未通过——按 §1 主结论判实验设计存疑，"
                        "其余三个的 fail 不作为预测应验报告"}
    deviations = [d for d in others if passed[d]]
    return {"outcome": "hit" if not deviations else "miss",
            "deviations": deviations,
            "note": "预测=只有 NATOPS 过" + ("，逐字应验" if not deviations else
                                             f"，但 {deviations} 也过了")}


def sample_subsets(keys: list, *, k: int, n: int, seed: int) -> list[tuple]:
    """固定种子抽 n 个大小 k 的无重复子集（有放回于子集层面，可有重复子集——
    均匀采样口径，同一子集重复出现权重自然）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.choice(len(keys), size=k, replace=False)
        out.append(tuple(keys[i] for i in sorted(idx)))
    return out


def wilson_interval(k: int, n: int, *, conf: float = 0.90) -> tuple[float, float]:
    z = {0.90: 1.6449, 0.95: 1.9600}[conf]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_idx(paths: list[Path]) -> dict:
    """缓存 → {ds: {task: {cls: {seed: R'}}}}（非有限记 NaN，口径同 gate_v2_eval）。"""
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for p in paths:
        for r in vd._read_jsonl(p):
            if r.get("dataset") not in FOUR:
                continue
            rp = r["r_prime"]
            idx[r["dataset"]][r["task_id"]][r["cls_key"]][r["seed"]] = (
                float(rp) if rp is not None else float("nan"))
    return {ds: {t: {k: dict(sm) for k, sm in cm.items()} for t, cm in tm.items()}
            for ds, tm in idx.items()}


def verdict_row(obs) -> dict:
    v = gv.gate_v2_decide(obs)
    return {"verdict": v.label, "passed": v.passed, "delta_star": v.delta_star,
            "accuracy": None if v.accuracy is None else round(v.accuracy, 4),
            "yield": None if v.yield_frac is None else round(v.yield_frac, 4),
            "n_kept": v.n_kept, "n_comparable": v.n_comparable,
            "n_candidates": len(obs), "reason": v.reason}


def main() -> None:
    keys = [c["cls_key"] for c in
            json.loads(CATALOG.read_text(encoding="utf-8"))["cls"]]
    assert len(keys) == 20

    idx = load_idx([MAIN / "rprime_cache.jsonl", MAIN / "noise_cache.jsonl"])
    rprime_flat = vd.load_indexed([MAIN / "rprime_cache.jsonl",
                                   MAIN / "noise_cache.jsonl"], set(FOUR))
    baseline = vd.load_indexed([MAIN / "baseline_cache.jsonl"], set(FOUR))
    n_cells = len(rprime_flat)
    print(f"[载入] R' {n_cells} 格（期望 640）| 基线 {len(baseline)} 格（期望 640）"
          f"| 数据集 {sorted(idx)}")

    # ---- §5.5 退化标注（判据 v2，逐字复用）----
    per_cls = {}
    for k in keys:
        diffs = vd.paired_diffs(rprime_flat, baseline, cls_key=k)
        per_cls[k] = vd.degenerate_verdict(diffs)
    tails = {k for k, v in per_cls.items() if v["degenerate"]}
    unknown = sorted(k for k, v in per_cls.items() if v["degenerate"] is None)
    reg_keys = [k for k in keys if k not in tails and k not in unknown]
    lo, hi = wilson_interval(len(tails), len(keys))
    print(f"[退化标注] {len(tails)}/20 判退化（Wilson90 [{lo:.2f},{hi:.2f}]）"
          f"，unknown={unknown}")

    # ---- §5.1 判定：R-reg 主口径 + R-full 对照 ----
    per_ds = {}
    for ds in FOUR:
        obs_reg = gv.pair_observations(idx[ds], reg_keys)
        obs_full = gv.pair_observations(idx[ds], keys)
        reg_d = REGISTRY_DELTA[ds]
        per_ds[ds] = {
            "r_reg": verdict_row(obs_reg),
            "r_full": verdict_row(obs_full),
            # §5.4 注册表 δ 兑现（R-full=生产口径；R-reg 附报）
            "registry_delta": {
                "delta": reg_d,
                "r_full": {k2: (round(v2, 4) if isinstance(v2, float) else v2)
                           for k2, v2 in gv.yield_accuracy(obs_full, reg_d).items()},
                "r_reg": {k2: (round(v2, 4) if isinstance(v2, float) else v2)
                          for k2, v2 in gv.yield_accuracy(obs_reg, reg_d).items()},
            },
            # §5.6 诊断：任务级留出 δ*（R-full）与平凡 label 占比
            "holdout_delta_bias_full": gv.holdout_delta_bias(idx[ds], keys),
            "trivial_share_at_delta_star_full": (
                None if verdict_row(obs_full)["delta_star"] is None else
                round(gv.trivial_share(obs_full, verdict_row(obs_full)["delta_star"],
                                       tails) or 0.0, 4)),
        }

    # ---- §5.2 预测判定 ----
    passed_reg = {ds: per_ds[ds]["r_reg"]["passed"] for ds in FOUR}
    pred = prediction_verdict(passed_reg)

    # ---- §5.3 剂量-反应（k=10 × 2000 子集，固定种子）----
    key_index = {k: i for i, k in enumerate(keys)}
    tables = {ds: gv.build_table(gv.pair_observations(idx[ds], keys), key_index)
              for ds in FOUR}
    dose_stats = None
    if len(tails) >= 2:
        subsets = sample_subsets(keys, k=SUBSET_K, n=N_SUBSETS, seed=SUBSET_SEED)
        dose, resp = [], []
        for sub in subsets:
            sel = {key_index[x] for x in sub}
            tasks = 0
            for ds in FOUR:
                tbl = tables[ds]
                rows = np.isin(tbl.a, list(sel)) & np.isin(tbl.b, list(sel))
                ok, _lab, _d = gv.decide_table(tbl, rows)
                if ok:
                    tasks += PLANNED[ds]
            dose.append(sum(x in tails for x in sub) / SUBSET_K)
            resp.append(tasks)
        d, r = np.array(dose), np.array(resp)
        by = defaultdict(list)
        for dd, rr in zip((d * SUBSET_K).round().astype(int), resp):
            by[int(dd)].append(rr)
        dose_stats = {
            "pearson": (round(float(np.corrcoef(d, r)[0, 1]), 3)
                        if r.std() > 0 and d.std() > 0 else None),
            "n_subsets": N_SUBSETS, "k": SUBSET_K, "seed": SUBSET_SEED,
            "final_baseline_pearson_old_cache": FINAL_BASELINE_PEARSON,
            "by_n_tail": {str(n): {"n": len(v), "mean": round(float(np.mean(v)), 1),
                                   "range": [int(min(v)), int(max(v))]}
                          for n, v in sorted(by.items())},
        }
    else:
        dose_stats = {"pearson": None, "n_subsets": 0,
                      "note": f"退化条数 {len(tails)}<2：剂量-反应不可估"
                              "（预注册 §5.3 预案：这本身就是结论）"}

    # ---- §5.6 诊断：随机 CLS 的 R̄ spread vs 旧探针 ----
    def spread(idx_ds, ks):
        rbar = [np.nanmean([v for t in idx_ds.values() if k in t
                            for v in t[k].values()]) for k in ks]
        rbar = [x for x in rbar if np.isfinite(x)]
        return round(float(np.std(rbar)), 4) if rbar else None

    old_idx = load_idx([SNAPSHOT / "labels_route_a" / "rprime_cache.jsonl",
                        SNAPSHOT / "step0_snr_gate" / "noise_cache.jsonl"])
    old_keys = sorted({k for tm in old_idx.values() for cm in tm.values()
                       for k in cm})
    spread_cmp = {ds: {"random20": spread(idx[ds], keys),
                       "old_probe10": spread(old_idx[ds], old_keys)}
                  for ds in FOUR}

    out = {
        "inputs": {"n_rprime_cells": n_cells, "n_baseline_cells": len(baseline),
                   "catalog": str(CATALOG.relative_to(ROOT)),
                   "criterion_doc": "PREREGISTRATION.md §1/§3(v2)/§5 + VALIDATION.md"},
        "degenerate_annotation": {
            "per_cls": per_cls, "flagged": sorted(tails), "unknown": unknown,
            "share": len(tails) / len(keys),
            "wilson90": [round(lo, 3), round(hi, 3)]},
        "datasets": per_ds,
        "prediction": {"passed_r_reg": passed_reg, **pred},
        "dose_response": dose_stats,
        "spread_comparison": spread_cmp,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'数据集':<28}{'R-reg 判定':<18}{'R-full 判定':<18}"
          f"{'reg δ*/Y/A':<22}{'full δ*/Y/A':<22}")
    for ds in FOUR:
        rr, rf = per_ds[ds]["r_reg"], per_ds[ds]["r_full"]

        def fmt(r):
            ds_ = "—" if r["delta_star"] is None else f"{r['delta_star']:.3f}"
            y = "—" if r["yield"] is None else f"{r['yield']:.2f}"
            a = "—" if r["accuracy"] is None else f"{r['accuracy']:.2f}"
            return f"{ds_}/{y}/{a}"

        print(f"{ds:<28}{rr['verdict']:<18}{rf['verdict']:<18}"
              f"{fmt(rr):<22}{fmt(rf):<22}")
    print(f"\n预测判定：{pred['outcome']}（{pred['note']}）")
    print(f"退化占比：{len(tails)}/20 = {len(tails)/20:.2f}，"
          f"Wilson90 [{lo:.2f},{hi:.2f}]，判退化：{sorted(tails)}")
    if dose_stats.get("pearson") is not None:
        print(f"剂量-反应 Pearson = {dose_stats['pearson']}"
              f"（旧缓存 v2 标注对照 {FINAL_BASELINE_PEARSON}）")
    else:
        print(f"剂量-反应：{dose_stats.get('note', '不可估')}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
