"""退化 CLS 判据的先行验证（PREREGISTRATION.md §3.1；先于任何新 GPU 数据）。

只用旧数据：R'_c 取阶段1 快照（results/stage1_rerun_20260804，5090 实测），
配对基线取 results/decouple_random_cls/validation_baseline/baseline_cache.jsonl
（同 weight_seed/split_seed、iters=0，migration 后 pod 补跑）。

判据 **v2**（§3 经 §3.1 修订路径修订，全程只用旧数据，见 VALIDATION.md）：
对每条 CLS，d_i = R'_c(t,s) − R'_base(t,s)，池化 4 数据集 × 2 任务 × 4 种子 =
32 格；**退化 ⟺ 单侧配对 t 检验 α=0.05 下拒绝 H0: mean(d) ≥ 0**（训练产生
统计显著的**破坏性**效应）。Wilcoxon 符号秩同报作诊断，不参与判定。

v1（"不能拒绝无正向效应"）在先行验证 round 1 被证伪（validation_round1.json）：
4 数据集随机初始化基线全部贴顶（0.90–0.99），正向增益原则上不可测 → 9/10 全
判退化，无区分度。平凡配对的制造者是"远低于群体的 R'"，而群体就坐在基线上
——故把检验方向翻成"是否显著破坏"。α、min_cells、池化口径均不变。

通过条件（随 v2 修订，理由见 VALIDATION.md）：20be4f74 判退化（机制案例）；
旧 |z|>2 尾部 {20be4f74, 8b87d66f} ⊆ 判退化集（向后一致）；pooled 增益为正的
CLS 全部判常规；任务定义与阶段1 逐字相同。超出旧 |z|>2 集的新增判退化条目
（预期 65627517/e490b88e）单独列报——65627517 是留一法第三面承重墙（去掉它
闸门 74→56），旧组内 |z| 定义漏掉了它，绝对判据抓到是修订的预期收益而非失败。

附带产出：用新退化标注重算旧缓存 4 数据集剂量-反应 Pearson（637 子集、响应 =
Σ planned_tasks(过闸者)，AWR 8/CT 16/ERing 6/NATOPS 10）——预注册 §5.3 的最终
对照数（|z|>2 口径的临时值 0.744）。

用法：python scripts/validate_degenerate_criterion.py  → results/decouple_random_cls/validation.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ALPHA = 0.05
MIN_CELLS = 24
N_EXPECTED = 32
FOUR = ["ArticularyWordRecognition", "CharacterTrajectories", "ERing", "NATOPS"]
PLANNED = {"ArticularyWordRecognition": 8, "CharacterTrajectories": 16,
           "ERing": 6, "NATOPS": 10}

SNAPSHOT = ROOT / "results" / "stage1_rerun_20260804" / "results"
CANON_CACHE = SNAPSHOT / "labels_route_a" / "rprime_cache.jsonl"
NOISE_CACHE = SNAPSHOT / "step0_snr_gate" / "noise_cache.jsonl"
SNAP_META = SNAPSHOT / "step0_snr_gate" / "tasks_meta.d"
VAL_DIR = ROOT / "results" / "decouple_random_cls" / "validation_baseline"
BASELINE_CACHE = VAL_DIR / "baseline_cache.jsonl"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
OUT = ROOT / "results" / "decouple_random_cls" / "validation.json"

KNOWN_DEGENERATE = "20be4f7443e8"      # 机制案例：表征停在随机初始化
UNCONSTRAINED = "8b87d66faad3"         # HAR 最差但 UEA 方差近常规：归类本身是测量


# ---------------------------------------------------------------------------
# 纯函数（tests/test_degenerate_criterion.py 覆盖）
# ---------------------------------------------------------------------------


def degenerate_verdict(diffs, *, alpha: float = ALPHA, min_cells: int = MIN_CELLS) -> dict:
    """池化配对差值 → 退化判定（v2）。

    d>0 = 训练有益。单侧 t 检验 H0: mean(d) ≥ 0；p < alpha（训练效应**显著为负**，
    表征被训到比随机初始化更差）⟺ 退化。以 0 为中心的噪声（贴顶数据集上"不动"）
    与群体同位、不制造平凡配对 → 常规。
    """
    finite = [float(d) for d in diffs if np.isfinite(d)]
    if len(finite) < min_cells:
        return {"status": "unknown", "degenerate": None, "n_cells": len(finite),
                "mean_gain": None, "p_t": None, "p_wilcoxon": None}
    t, p = stats.ttest_1samp(finite, 0.0, alternative="less")
    try:
        pw = float(stats.wilcoxon(finite, alternative="less").pvalue)
    except ValueError:  # 全零差值等退化情形
        pw = 1.0
    return {"status": "ok", "degenerate": bool(p < alpha),
            "n_cells": len(finite), "mean_gain": round(float(np.mean(finite)), 4),
            "t": round(float(t), 3), "p_t": float(p), "p_wilcoxon": pw}


def paired_diffs(rprime: dict, baseline: dict, *, cls_key: str) -> list[float]:
    """按 (dataset, task_id, cls_key, seed) 四元组严格对齐的 R'_c − R'_base。

    基线缺格 → 该格丢弃（缺格数最终反映在 n_cells，触发 unknown 而非臆断）。
    """
    out = []
    for (ds, tid, key, seed), rc in rprime.items():
        if key != cls_key:
            continue
        rb = baseline.get((ds, tid, key, seed))
        if rb is None:
            continue
        out.append(float(rc) - float(rb))
    return out


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def load_indexed(paths: list[Path], datasets: set[str]) -> dict:
    """缓存行 → {(ds, task_id, cls_key, seed): r_prime}（非有限记 NaN）。"""
    idx: dict = {}
    for p in paths:
        for r in _read_jsonl(p):
            if r.get("dataset") not in datasets:
                continue
            rp = r.get("r_prime")
            idx[(r["dataset"], r["task_id"], r["cls_key"], r["seed"])] = (
                float(rp) if rp is not None else float("nan"))
    return idx


def check_task_identity() -> dict:
    """验证基线跑的任务定义与阶段1 快照逐字相同（预注册 §4 的实验控制前提）。"""
    out = {}
    for ds in FOUR:
        snap = json.loads((SNAP_META / f"{ds}.json").read_text(encoding="utf-8"))
        val = json.loads((VAL_DIR / "tasks_meta.d" / f"{ds}.json").read_text(encoding="utf-8"))
        same = [t_s["task_id"] for t_s in snap["tasks"]] == \
               [t_v["task_id"] for t_v in val["tasks"]]
        out[ds] = same
    return out


# ---------------------------------------------------------------------------
# 附带产出：新退化标注下重算旧缓存 4 数据集剂量-反应（预注册 §5.3 最终对照数）
# ---------------------------------------------------------------------------


def dose_response_old_cache(tails: set[str]) -> dict:
    import gate_v2_eval as gv
    idx, _meta, keys, _rank = gv.load_index()
    key_index = {k: i for i, k in enumerate(keys)}
    tables = {ds: gv.build_table(gv.pair_observations(idx[ds], keys), key_index)
              for ds in FOUR}
    dose, resp = [], []
    for k in range(5, 10):
        for sub in combinations(keys, k):
            sel = {key_index[x] for x in sub}
            tasks = 0
            for ds in FOUR:
                tbl = tables[ds]
                rows = np.isin(tbl.a, list(sel)) & np.isin(tbl.b, list(sel))
                ok, _lab, _d = gv.decide_table(tbl, rows)
                if ok:
                    tasks += PLANNED[ds]
            dose.append(sum(x in tails for x in sub) / k)
            resp.append(tasks)
    d, r = np.array(dose), np.array(resp)
    by = defaultdict(list)
    i = 0
    for k in range(5, 10):
        for sub in combinations(keys, k):
            by[sum(x in tails for x in sub)].append(resp[i])
            i += 1
    return {
        "pearson": round(float(np.corrcoef(d, r)[0, 1]), 3) if r.std() > 0 else 0.0,
        "n_subsets": len(resp),
        "by_n_tail": {str(n): {"n": len(v), "mean": round(float(np.mean(v)), 1),
                               "range": [int(min(v)), int(max(v))]}
                      for n, v in sorted(by.items())},
        "tails": sorted(tails),
    }


def main() -> None:
    cat = json.loads(PROBE_CATALOG.read_text(encoding="utf-8"))["cls"][:10]
    keys = [c["cls_key"] for c in cat]

    rprime = load_indexed([CANON_CACHE, NOISE_CACHE], set(FOUR))
    baseline = load_indexed([BASELINE_CACHE], set(FOUR))
    n_base = len(baseline)
    print(f"[载入] R' {len(rprime)} 格（快照）| 基线 {n_base} 格（期望 320）")
    task_ok = check_task_identity()
    print(f"[任务一致性] {task_ok}")

    per_cls = {}
    for k in keys:
        diffs = paired_diffs(rprime, baseline, cls_key=k)
        v = degenerate_verdict(diffs)
        # 逐数据集平均训练增益（诊断）
        by_ds = {}
        for ds in FOUR:
            ds_diffs = [rprime[(d, t, c, s)] - baseline[(d, t, c, s)]
                        for (d, t, c, s) in rprime
                        if d == ds and c == k and (d, t, c, s) in baseline
                        and np.isfinite(rprime[(d, t, c, s)])]
            by_ds[ds] = round(float(np.mean(ds_diffs)), 4) if ds_diffs else None
        per_cls[k] = {**v, "gain_by_dataset": by_ds}

    tails_new = {k for k, v in per_cls.items() if v["degenerate"]}
    old_z2_tails = {KNOWN_DEGENERATE, UNCONSTRAINED}
    positives = [k for k in keys
                 if per_cls[k]["mean_gain"] is not None and per_cls[k]["mean_gain"] > 0]
    passed = (KNOWN_DEGENERATE in tails_new
              and old_z2_tails <= tails_new
              and all(per_cls[k]["degenerate"] is False for k in positives)
              and all(task_ok.values()))

    dr = dose_response_old_cache(tails_new) if tails_new else None

    out = {
        "criterion": {"version": 2, "alternative": "less", "alpha": ALPHA,
                      "min_cells": MIN_CELLS,
                      "doc": "PREREGISTRATION.md §3 + VALIDATION.md 修订记录"},
        "task_identity_ok": task_ok,
        "n_baseline_cells": n_base,
        "per_cls": per_cls,
        "flagged_degenerate": sorted(tails_new),
        "pass_conditions": {
            "known_degenerate_flagged": KNOWN_DEGENERATE in tails_new,
            "old_z2_tails_subset": old_z2_tails <= tails_new,
            "positive_gain_all_clear": all(per_cls[k]["degenerate"] is False
                                           for k in positives),
            "passed": passed,
        },
        "newly_flagged_beyond_z2": sorted(tails_new - old_z2_tails),
        "dose_response_old_cache_new_tails": dr,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'cls_key':<16}{'退化?':<8}{'均值增益':>9}{'p_t':>12}{'格数':>5}   逐数据集增益")
    for k in keys:
        v = per_cls[k]
        tag = {True: "退化", False: "常规", None: "unknown"}[v["degenerate"]]
        p_t = "—" if v["p_t"] is None else f"{v['p_t']:.2e}"
        mg = "—" if v["mean_gain"] is None else f"{v['mean_gain']:+.4f}"
        print(f"{k:<16}{tag:<8}{mg:>9}{p_t:>12}{v['n_cells']:>5}   "
              + " ".join(f"{ds[:4]}:{v['gain_by_dataset'][ds]:+.3f}"
                         if v["gain_by_dataset"][ds] is not None else f"{ds[:4]}:—"
                         for ds in FOUR))
    print(f"\n判退化：{sorted(tails_new)}")
    print(f"通过条件：{out['pass_conditions']}")
    if dr:
        print(f"旧缓存 4 数据集剂量-反应（新标注）：Pearson={dr['pearson']}"
              f"（|z|>2 口径临时值 0.744）")
    print(f"→ {OUT}")
    sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
