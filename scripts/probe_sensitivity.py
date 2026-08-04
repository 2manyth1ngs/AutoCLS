"""探针 CLS 敏感性分析（纯离线，只读 stage1 补跑的 1040 条 R' 缓存，不跑 GPU）。

起因（stage1 补跑判读后的方差分析发现）：step0 SNR 闸的 13 个判定、以及由此得出
的"源池任务 74"这个数字，全都建立在**一组特定的 10 条探针 CLS**（
results/route_a_probe/probe_cls_catalog.json 前 10 条）之上。已观察到改动 1 条探针
CLS（20be4f74）就翻转 3/13 个判定 —— 即"用哪 10 条探针"本身是闸门结果的一个自由度。
写论文时这是可攻击点，先量化它。

本脚本**不改判据、不改任何配置**，只是把 step0 的判读（autocls.snr_gate.compute_snr_stats
+ decide + task_count_gate，逐字复用同一份代码）在探针 CLS 的**子集**上重放：

  1. 留一法：10 轮，每轮去掉 1 条探针 CLS，用剩 9 条重算 13 个数据集的判定与任务数闸；
  2. 子集穷举：k=5..9 全部 C(10,k) 子集（252+210+120+45+10=637 个，无需随机采样），
     统计每数据集的 pass 率随 k 的变化 + 任务数闸的分布；
  3. 按 pass 率给数据集分类（稳健 pass / 摇摆 / 稳健 fail）；
  4. 判据阈值敏感性：在 snr_floor × sign_floor 网格上重放 decide（**只报告不改判据**）；
  5. 20be4f74 角色澄清：把"R̄ 离群"与"跨种子高方差"两个属性拆开量化，看是否存在
     离群但方差正常的替代 CLS。

判定口径与 step0_snr_gate.analyze 完全一致：base 轮 stats → decide(range_maxed=
is_range_maxed(base_range, n_classes))。补跑产物里 13 个数据集的 retry_rounds 全为空
（无天花板重探轮），故子集重放若判出 retry_ceiling，本脚本如实记为 retry_ceiling 且
passed=False（不臆断重探能否救回），并单列一类统计。

任务数闸口径同 task_count_gate：只有 role=source 的数据集计入 74，reserve
（Ethanol/Heartbeat/FingerMovements）即使判 pass 也不自动进池，故另报一个"若允许
reserve 回填"的补充口径。

用法：
  python scripts/probe_sensitivity.py                 # 全部分析 → results/probe_sensitivity/
  python scripts/probe_sensitivity.py --verify-only   # 只校验能复现 analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import load_registry, task_count_gate  # noqa: E402
from autocls.labeling import load_probe_cls  # noqa: E402
from autocls.snr_gate import (  # noqa: E402
    compute_snr_stats,
    decide,
    is_range_maxed,
)

# stage1 补跑的产物快照（1040 条 = 260 canonical + 780 噪声）——唯一数据源
SNAPSHOT = ROOT / "results" / "stage1_rerun_20260804" / "results"
CANON_CACHE = SNAPSHOT / "labels_route_a" / "rprime_cache.jsonl"
NOISE_CACHE = SNAPSHOT / "step0_snr_gate" / "noise_cache.jsonl"
META_DIR = SNAPSHOT / "step0_snr_gate" / "tasks_meta.d"
BASE_ANALYSIS = SNAPSHOT / "step0_snr_gate" / "analysis.json"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"

OUTDIR = ROOT / "results" / "probe_sensitivity"
N_PROBE_CLS = 10          # step0 ProbeCfg.n_cls


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_tasks_meta() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(META_DIR.glob("*.json"))}


def build_rprime_index(recs: list[dict], meta: dict[str, dict]) -> dict:
    """记录 → {dataset: {task_id: {cls_key: {seed: R'}}}}，只保留 base 轮任务。"""
    keep = {ds: {t["task_id"] for t in m["tasks"]} for ds, m in meta.items()}
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in recs:
        ds = r.get("dataset")
        if ds not in keep or r["task_id"] not in keep[ds]:
            continue
        rp = r["r_prime"]
        idx[ds][r["task_id"]][r["cls_key"]][r["seed"]] = (
            float(rp) if rp is not None else float("nan"))
    return {ds: {tid: {k: dict(sm) for k, sm in cm.items()}
                 for tid, cm in tm.items()} for ds, tm in idx.items()}


def subset_stats(ds_idx: dict, cls_subset: tuple[str, ...], har_rank: dict):
    """在指定 CLS 子集上重算该数据集的 SnrStats（判据代码逐字复用 step0）。"""
    sub = {tid: {k: sm for k, sm in cm.items() if k in cls_subset}
           for tid, cm in ds_idx.items()}
    return compute_snr_stats(sub, har_rank=har_rank)


def verdict_for(ds: str, stats, registry) -> object:
    """单轮静态判定，口径同 step0_snr_gate.analyze 的 base 轮。"""
    entry = registry.get(ds)
    return decide(stats, range_maxed=is_range_maxed(
        tuple(entry.n_classes_range), entry.n_classes))


def gate_count(passed: set[str], registry) -> int:
    return task_count_gate(registry, passed_untested=passed).count


def reserve_bonus(passed: set[str], registry) -> int:
    """补充口径：若允许把判 pass 的 reserve 回填进源池，能再加多少任务。"""
    return sum(e.planned_tasks for e in registry.reserve if e.name in passed)


# ---------------------------------------------------------------------------
# 0. 复现校验：全 10 条 CLS 必须逐字复现 analysis.json 的 base 行
# ---------------------------------------------------------------------------


def verify(idx, all_keys, har_rank, registry) -> list[str]:
    base = json.loads(BASE_ANALYSIS.read_text(encoding="utf-8"))
    problems = []
    for ds, row in base["datasets"].items():
        st = subset_stats(idx[ds], all_keys, har_rank)
        v = verdict_for(ds, st, registry)
        checks = [
            ("snr", round(st.snr, 2), row["snr"]),
            ("rprime_std_mean", round(st.rprime_std_mean, 4), row["rprime_std_mean"]),
            ("cls_spread_std_mean", round(st.cls_spread_std_mean, 4),
             row["cls_spread_std_mean"]),
            ("rprime_grand_mean", round(st.rprime_grand_mean, 4),
             row["rprime_grand_mean"]),
            ("pair_sign_stability", round(st.pair_sign_stability, 3),
             row["pair_sign_stability"]),
            ("monotonicity", round(st.monotonicity, 3), row["monotonicity"]),
            ("verdict", v.label, row["verdict"]),
        ]
        for field, got, want in checks:
            if got != want:
                problems.append(f"{ds}.{field}: 重算={got} vs analysis.json={want}")
    passed = {ds for ds in base["datasets"]
              if verdict_for(ds, subset_stats(idx[ds], all_keys, har_rank),
                             registry).passed}
    if gate_count(passed, registry) != base["gate"]["count"]:
        problems.append(f"gate: 重算={gate_count(passed, registry)} "
                        f"vs analysis.json={base['gate']['count']}")
    return problems


# ---------------------------------------------------------------------------
# 1/2. 子集重放（留一法是 k=9 的 10 个子集；此处统一按子集枚举）
# ---------------------------------------------------------------------------


def replay_subsets(idx, datasets, all_keys, har_rank, registry, ks) -> dict:
    """对每个 k 穷举全部 C(10,k) 子集，重放 13 个数据集的判定 + 任务数闸。"""
    out: dict = {}
    for k in ks:
        rows = []
        for sub in combinations(all_keys, k):
            per_ds, passed = {}, set()
            for ds in datasets:
                st = subset_stats(idx[ds], sub, har_rank)
                v = verdict_for(ds, st, registry)
                per_ds[ds] = {
                    "verdict": v.label, "passed": v.passed,
                    "snr": round(st.snr, 3), "sign": round(st.pair_sign_stability, 3),
                    "rbar": round(st.rprime_grand_mean, 4),
                    "sigma": round(st.rprime_std_mean, 4),
                    "spread": round(st.cls_spread_std_mean, 4),
                }
                if v.passed:
                    passed.add(ds)
            rows.append({
                "dropped": sorted(set(all_keys) - set(sub)),
                "subset": list(sub),
                "gate": gate_count(passed, registry),
                "reserve_bonus": reserve_bonus(passed, registry),
                "datasets": per_ds,
            })
        out[k] = rows
    return out


def summarize(rows: list[dict], datasets: list[str]) -> dict:
    """一批子集 → 每数据集 pass 率/判定分布 + 任务数闸分布。"""
    n = len(rows)
    per_ds = {}
    for ds in datasets:
        labels = [r["datasets"][ds]["verdict"] for r in rows]
        snrs = [r["datasets"][ds]["snr"] for r in rows]
        signs = [r["datasets"][ds]["sign"] for r in rows]
        cnt: dict[str, int] = defaultdict(int)
        for lab in labels:
            cnt[lab] += 1
        per_ds[ds] = {
            "n_subsets": n,
            "pass_rate": round(sum(r["datasets"][ds]["passed"] for r in rows) / n, 4),
            "verdict_counts": dict(sorted(cnt.items(), key=lambda kv: -kv[1])),
            "snr_min": round(min(snrs), 3), "snr_med": round(float(np.median(snrs)), 3),
            "snr_max": round(max(snrs), 3),
            "sign_min": round(min(signs), 3),
            "sign_med": round(float(np.median(signs)), 3),
            "sign_max": round(max(signs), 3),
        }
    gates = np.array([r["gate"] for r in rows], dtype=float)
    return {
        "n_subsets": n,
        "datasets": per_ds,
        "gate": {
            "mean": round(float(gates.mean()), 2),
            "median": float(np.median(gates)),
            "min": float(gates.min()), "max": float(gates.max()),
            "p10": float(np.percentile(gates, 10)),
            "p25": float(np.percentile(gates, 25)),
            "p75": float(np.percentile(gates, 75)),
            "p90": float(np.percentile(gates, 90)),
            "frac_ge_100": round(float((gates >= 100).mean()), 4),
            "frac_ge_74": round(float((gates >= 74).mean()), 4),
            "frac_eq_74": round(float((gates == 74).mean()), 4),
            "hist": {str(int(v)): int(c) for v, c in
                     zip(*np.unique(gates, return_counts=True))},
        },
    }


# ---------------------------------------------------------------------------
# 4. 判据阈值敏感性（只报告，不改判据）
# ---------------------------------------------------------------------------


def threshold_grid(idx, datasets, all_keys, har_rank, registry,
                   snr_floors, sign_floors) -> list[dict]:
    stats = {ds: subset_stats(idx[ds], all_keys, har_rank) for ds in datasets}
    out = []
    for sf in snr_floors:
        for gf in sign_floors:
            per_ds, passed = {}, set()
            for ds in datasets:
                e = registry.get(ds)
                v = decide(stats[ds], snr_floor=sf, sign_floor=gf,
                           range_maxed=is_range_maxed(
                               tuple(e.n_classes_range), e.n_classes))
                per_ds[ds] = v.label
                if v.passed:
                    passed.add(ds)
            out.append({
                "snr_floor": sf, "sign_floor": gf,
                "n_pass": len(passed), "passed": sorted(passed),
                "gate": gate_count(passed, registry),
                "reserve_bonus": reserve_bonus(passed, registry),
                "verdicts": per_ds,
            })
    return out


# ---------------------------------------------------------------------------
# 5. 每条探针 CLS 的角色：R̄ 离群 vs 跨种子高方差
# ---------------------------------------------------------------------------


def cls_role_analysis(idx, datasets, all_keys, har_rank, registry) -> dict:
    """把两个属性拆开量化：

    - **R̄ 离群度**：数据集内该 CLS 的 R̄（跨任务的 seed 均值）相对其余 CLS 的
      稳健 z 分数 z = (R̄_c - median) / (1.4826·MAD)（MAD 退化为 0 时用 std 兜底）。
    - **跨种子方差**：该 CLS 在该数据集上 (task,cls) 跨 seed 的 std / 极差，
      再除以该数据集的 σ（rprime_std_mean）得相对倍率——">1 = 比平均条目更抖"。
    - **撑 spread 的程度**：留一后该数据集 cls_spread_std_mean 的相对跌幅。
    """
    full = {ds: subset_stats(idx[ds], all_keys, har_rank) for ds in datasets}
    per_cls: dict[str, dict] = {k: {"har_rank": har_rank[k], "datasets": {}}
                                for k in all_keys}
    for ds in datasets:
        # 每 CLS 的 R̄（跨任务平均的 seed 均值） / 跨种子 std / 跨种子极差
        rbar, sstd, srange = {}, {}, {}
        for k in all_keys:
            means, stds, rngs = [], [], []
            for tid, cm in idx[ds].items():
                vals = np.array([v for v in cm.get(k, {}).values()
                                 if np.isfinite(v)], dtype=float)
                if len(vals) >= 1:
                    means.append(float(vals.mean()))
                if len(vals) >= 2:
                    stds.append(float(vals.std(ddof=1)))
                    rngs.append(float(vals.max() - vals.min()))
            rbar[k] = float(np.mean(means)) if means else float("nan")
            sstd[k] = float(np.mean(stds)) if stds else float("nan")
            srange[k] = float(np.mean(rngs)) if rngs else float("nan")
        arr = np.array([rbar[k] for k in all_keys])
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad if mad > 1e-9 else float(arr.std(ddof=1))
        sigma_ds = full[ds].rprime_std_mean
        for k in all_keys:
            loo = subset_stats(idx[ds], tuple(x for x in all_keys if x != k), har_rank)
            spread_drop = (1 - loo.cls_spread_std_mean / full[ds].cls_spread_std_mean
                           if full[ds].cls_spread_std_mean else float("nan"))
            per_cls[k]["datasets"][ds] = {
                "rbar": round(rbar[k], 4),
                "robust_z": round((rbar[k] - med) / scale, 2) if scale > 1e-9 else None,
                "seed_std": round(sstd[k], 4),
                "seed_std_rel": round(sstd[k] / sigma_ds, 2) if sigma_ds else None,
                "seed_range": round(srange[k], 4),
                "loo_snr": round(loo.snr, 3),
                "loo_snr_delta": round(loo.snr - full[ds].snr, 3),
                "loo_sign": round(loo.pair_sign_stability, 3),
                "loo_sign_delta": round(
                    loo.pair_sign_stability - full[ds].pair_sign_stability, 3),
                "spread_drop_frac": round(spread_drop, 3),
            }
    for k in all_keys:
        d = per_cls[k]["datasets"]
        zs = [abs(v["robust_z"]) for v in d.values() if v["robust_z"] is not None]
        rels = [v["seed_std_rel"] for v in d.values() if v["seed_std_rel"] is not None]
        drops = [v["spread_drop_frac"] for v in d.values()]
        per_cls[k]["summary"] = {
            "mean_abs_z": round(float(np.mean(zs)), 2),
            "max_abs_z": round(float(np.max(zs)), 2),
            "n_ds_outlier_z2": int(sum(z > 2 for z in zs)),
            "n_ds_outlier_z3": int(sum(z > 3 for z in zs)),
            "mean_seed_std_rel": round(float(np.mean(rels)), 2),
            "max_seed_std_rel": round(float(np.max(rels)), 2),
            "n_ds_highvar": int(sum(r > 1.5 for r in rels)),
            "mean_spread_drop": round(float(np.mean(drops)), 3),
            "max_spread_drop": round(float(np.max(drops)), 3),
        }
    return per_cls


# ---------------------------------------------------------------------------
# 6/7. 机制拆解：符号稳定率究竟在测什么 + δ 过滤后还剩多少 label
# ---------------------------------------------------------------------------


def _pair_sign_const(cm: dict, a: str, b: str) -> float | None:
    """一对 CLS 在某任务上跨 seed 的差值符号是否恒定（口径同 compute_snr_stats）。"""
    seeds = sorted(set(cm.get(a, {})) & set(cm.get(b, {})))
    diffs = [cm[a][s] - cm[b][s] for s in seeds
             if np.isfinite(cm[a][s]) and np.isfinite(cm[b][s])]
    signs = np.sign([d for d in diffs if abs(d) > 1e-12])
    return float(np.all(signs == signs[0])) if len(signs) >= 2 else None


def sign_by_strata(idx, datasets, all_keys, outliers) -> dict:
    """把 45 个成对拆成三层看符号稳定率：

    outlier×outlier / normal×normal / 跨层。若"通过"全靠跨层（正常 CLS vs 离群
    CLS 这种平凡对比），则该指标测的是**探针集合内部的好坏落差**，而不是数据集
    本身能否区分 CLS —— 而真正要造的 label 主要落在 normal×normal 层。
    """
    norm = tuple(k for k in all_keys if k not in outliers)
    out = {}
    for ds in datasets:
        buckets: dict[str, list[float]] = {"all": [], "normal": [], "outlier": [],
                                           "cross": []}
        for cm in idx[ds].values():
            for a, b in combinations(all_keys, 2):
                v = _pair_sign_const(cm, a, b)
                if v is None:
                    continue
                buckets["all"].append(v)
                in_a, in_b = a in outliers, b in outliers
                buckets["outlier" if in_a and in_b
                        else "normal" if not (in_a or in_b) else "cross"].append(v)
        out[ds] = {k: (round(float(np.mean(v)), 4) if v else None)
                   for k, v in buckets.items()}
        out[ds]["n_normal_pairs"] = len(buckets["normal"])
        out[ds]["n_cross_pairs"] = len(buckets["cross"])
    return {"outliers": sorted(outliers), "normal": sorted(norm), "datasets": out}


def delta_yield(idx, datasets, all_keys, outliers, labels_cfg, *,
                canonical_seed: int = 0, delta_mode: str = "registry",
                har_rank: dict | None = None) -> dict:
    """δ 过滤后每数据集还剩多少可用 label，以及这些 label 方向稳不稳。

    口径对齐真实造 label（autocls.labeling.make_pairs）：路线 A k=1，配对用
    **canonical 种子**的 R'，|R'_a-R'_b| < δ 弃样。对留下的配对再问一句"这个方向
    跨 seed 稳吗"——即该 label 的正确率代理（探针 §2.2 的 ≥95% 判据同源）。

    delta_mode="registry" 用 labels.yaml 的 per-dataset δ（阶段2 实际会用的值）；
    "suggested" 用本批数据实测的 3σ（SnrStats.suggested_delta）——两者在
    Handwriting/NATOPS 上差 ~2×（注册表 δ 是 N/C 阶梯推测值，未经实测校准），
    结论若只在其中一种口径下成立就不算稳。

    ⚠️ 只覆盖 10 条 shared 探针 CLS；阶段2 每任务另有 10 条 random CLS 未在本缓存
    内，其配对产出不在此统计中（见报告 limitation）。
    """
    from autocls.data.source_pool import delta_for
    norm = tuple(k for k in all_keys if k not in outliers)
    out = {}
    for ds in datasets:
        d = (delta_for(ds, labels_cfg) if delta_mode == "registry"
             else subset_stats(idx[ds], all_keys, har_rank).suggested_delta)
        rows = {}
        for label, keys in (("all", all_keys), ("normal", norm)):
            n_pair = n_kept = 0
            stable = []
            for cm in idx[ds].values():
                for a, b in combinations(keys, 2):
                    ra, rb = cm.get(a, {}).get(canonical_seed), cm.get(b, {}).get(canonical_seed)
                    if ra is None or rb is None or not (np.isfinite(ra) and np.isfinite(rb)):
                        continue
                    n_pair += 1
                    if abs(ra - rb) < d:
                        continue
                    n_kept += 1
                    v = _pair_sign_const(cm, a, b)
                    if v is not None:
                        stable.append(v)
            rows[label] = {
                "n_pairs": n_pair, "n_kept": n_kept,
                "keep_rate": round(n_kept / n_pair, 4) if n_pair else None,
                "sign_stable_among_kept": round(float(np.mean(stable)), 4) if stable else None,
            }
        out[ds] = {"delta": d, **rows}
    return out


def outlier_dose_response(replay: dict, outliers: set) -> dict:
    """剂量-反应：子集里"退化 CLS"的条数 vs 任务数闸。

    符号稳定率是"成对差值符号跨 seed 恒定"的比例——探针集合里放进一条 R' 远低于
    其余的退化 CLS，就凭空造出一批"永远稳"的平凡成对，把该比例整体抬高。若闸门
    结果随退化 CLS 条数单调走高，说明 74 测的是**探针集合的构成**而非数据集质量。
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    pairs: list[tuple[float, int]] = []
    for rows in replay.values():
        for r in rows:
            n_out = sum(k in outliers for k in r["subset"])
            buckets[n_out].append(r["gate"])
            pairs.append((n_out / len(r["subset"]), r["gate"]))
    frac = np.array([p[0] for p in pairs])
    gates = np.array([p[1] for p in pairs], dtype=float)
    return {
        "outliers": sorted(outliers),
        "by_n_outliers": {
            str(n): {"n_subsets": len(v), "gate_mean": round(float(np.mean(v)), 2),
                     "gate_median": float(np.median(v)),
                     "gate_min": int(min(v)), "gate_max": int(max(v)),
                     "frac_ge_100": round(float(np.mean(np.array(v) >= 100)), 4)}
            for n, v in sorted(buckets.items())},
        "pearson_frac_vs_gate": round(float(np.corrcoef(frac, gates)[0, 1]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()

    registry = load_registry()
    cls_list, har_rank = load_probe_cls(PROBE_CATALOG, N_PROBE_CLS)
    all_keys = tuple(k for _, k in cls_list)
    cls_cfgs = {k: c for c, k in cls_list}

    meta = load_tasks_meta()
    recs = read_jsonl(CANON_CACHE) + read_jsonl(NOISE_CACHE)
    idx = build_rprime_index(recs, meta)
    datasets = sorted(idx)
    print(f"缓存 {len(recs)} 条 → {len(datasets)} 数据集 × "
          f"{sum(len(v) for v in idx.values())} 任务 × {len(all_keys)} 探针 CLS")

    problems = verify(idx, all_keys, har_rank, registry)
    if problems:
        print("❌ 未能复现 analysis.json：")
        for p in problems:
            print("   " + p)
        sys.exit(1)
    print("✅ 全 10 条 CLS 重放逐字复现 analysis.json（13 个判定 + 任务数闸 74）")
    if args.verify_only:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    roles = cls_role_analysis(idx, datasets, all_keys, har_rank, registry)
    # 离群 CLS 两种定义：数据驱动（跨数据集平均 |z|>2）与 HAR 质量尾三条
    out_z = {k for k, v in roles.items() if v["summary"]["mean_abs_z"] > 2.0}
    out_tail3 = {k for k in all_keys if har_rank[k] >= 15}

    ks = [9, 8, 7, 6, 5]
    replay = replay_subsets(idx, datasets, all_keys, har_rank, registry, ks)
    summaries = {k: summarize(rows, datasets) for k, rows in replay.items()}
    pooled = summarize([r for k in ks for r in replay[k]], datasets)

    result = {
        "meta": {
            "snapshot": str(SNAPSHOT.relative_to(ROOT)).replace("\\", "/"),
            "n_records": len(recs),
            "datasets": {ds: {"role": registry.get(ds).role,
                              "planned_tasks": registry.get(ds).planned_tasks,
                              "n_tasks_probed": len(idx[ds]),
                              "n_classes_range": list(registry.get(ds).n_classes_range),
                              "n_classes": registry.get(ds).n_classes,
                              "l_enc": registry.get(ds).l_enc,
                              "domain": registry.get(ds).domain}
                         for ds in datasets},
            "probe_cls": [{"cls_key": k, "har_rank": har_rank[k]} for k in all_keys],
            "baseline_gate": gate_count(
                {ds for ds in datasets
                 if verdict_for(ds, subset_stats(idx[ds], all_keys, har_rank),
                                registry).passed}, registry),
        },
        "loo": {
            "rounds": [{"dropped": r["dropped"][0], "gate": r["gate"],
                        "reserve_bonus": r["reserve_bonus"],
                        "verdicts": {ds: r["datasets"][ds]["verdict"] for ds in datasets},
                        "snr": {ds: r["datasets"][ds]["snr"] for ds in datasets},
                        "sign": {ds: r["datasets"][ds]["sign"] for ds in datasets}}
                       for r in replay[9]],
            "summary": summaries[9],
        },
        "subsets": {str(k): summaries[k] for k in ks},
        "subsets_pooled": pooled,
        "thresholds": threshold_grid(idx, datasets, all_keys, har_rank, registry,
                                     snr_floors=[1.0, 0.9, 0.8],
                                     sign_floors=[0.7, 0.65, 0.6]),
        "cls_roles": roles,
        "cls_configs": {k: {f: getattr(cls_cfgs[k], f)
                            for f in cls_cfgs[k].__dataclass_fields__}
                        for k in all_keys},
        "outlier_dose_response": outlier_dose_response(replay, out_z),
        "sign_by_strata": {
            "data_driven": sign_by_strata(idx, datasets, all_keys, out_z),
            "har_tail3": sign_by_strata(idx, datasets, all_keys, out_tail3),
        },
        "delta_yield": {
            "data_driven": delta_yield(idx, datasets, all_keys, out_z, labels_cfg),
            "har_tail3": delta_yield(idx, datasets, all_keys, out_tail3, labels_cfg),
            "data_driven_suggested_delta": delta_yield(
                idx, datasets, all_keys, out_z, labels_cfg,
                delta_mode="suggested", har_rank=har_rank),
        },
    }
    (outdir / "sensitivity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {outdir / 'sensitivity.json'}")

    # 控制台速览（详表见 REPORT.md）
    print("\n== 留一法：每数据集 10 轮中判 pass 的次数 ==")
    for ds in sorted(datasets, key=lambda d: -summaries[9]["datasets"][d]["pass_rate"]):
        s = summaries[9]["datasets"][ds]
        print(f"  {ds:28s} pass {int(s['pass_rate']*10):2d}/10  {s['verdict_counts']}")
    print("  任务数闸：" + ", ".join(f"{r['gate']}" for r in replay[9]))
    print("\n== 子集 pass 率（k=5..9 穷举）==")
    hdr = "  " + "数据集".ljust(26) + "".join(f"k={k}".rjust(8) for k in [5, 6, 7, 8, 9])
    print(hdr)
    for ds in sorted(datasets, key=lambda d: -pooled["datasets"][d]["pass_rate"]):
        print("  " + ds.ljust(26)
              + "".join(f"{summaries[k]['datasets'][ds]['pass_rate']:8.2f}"
                        for k in [5, 6, 7, 8, 9]))
    print("\n== 任务数闸分布 ==")
    for k in [5, 6, 7, 8, 9]:
        g = summaries[k]["gate"]
        print(f"  k={k} (n={summaries[k]['n_subsets']:3d}): mean={g['mean']:6.2f} "
              f"med={g['median']:5.1f} [{g['min']:.0f},{g['max']:.0f}] "
              f"P(=74)={g['frac_eq_74']:.2f} P(≥100)={g['frac_ge_100']:.2f}")

    dr = result["outlier_dose_response"]
    print(f"\n== 剂量-反应：探针集含退化 CLS 条数 vs 任务数闸（637 子集）==")
    for n, v in dr["by_n_outliers"].items():
        print(f"  含 {n} 条退化 CLS (n={v['n_subsets']:3d}): gate mean={v['gate_mean']:6.2f} "
              f"med={v['gate_median']:5.1f} [{v['gate_min']},{v['gate_max']}] "
              f"P(≥100)={v['frac_ge_100']:.2f}")
    print(f"  Pearson(退化 CLS 占比, 任务数闸) = {dr['pearson_frac_vs_gate']}")

    st = result["sign_by_strata"]["data_driven"]
    dy = result["delta_yield"]["data_driven"]
    print(f"\n== 符号稳定率分层（离群 CLS = {st['outliers']}）+ δ 产出 ==")
    print("  " + "数据集".ljust(26) + "全45  正常×正常  跨层   δ  正常对留存 留存者符号稳")
    dys = result["delta_yield"]["data_driven_suggested_delta"]
    for ds in sorted(datasets, key=lambda d: -(st["datasets"][d]["all"] or 0)):
        r, y = st["datasets"][ds], dy[ds]["normal"]
        y2 = dys[ds]["normal"]
        f = lambda v: f"{v:.3f}" if v is not None else " n/a "  # noqa: E731
        print(f"  {ds.ljust(26)}{r['all']:.3f}     {r['normal']:.3f} {r['cross']:.3f} "
              f"{dy[ds]['delta']:.2f} {y['n_kept']:3d}/{y['n_pairs']:<3d} "
              f"{f(y['sign_stable_among_kept'])}   |3σ={dys[ds]['delta']:.3f} "
              f"{y2['n_kept']:3d} {f(y2['sign_stable_among_kept'])}")


if __name__ == "__main__":
    main()
