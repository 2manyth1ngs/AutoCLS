"""破坏矩阵：可行性边界的任务依赖性存在性检验（协议 results/destruction_matrix/PROTOCOL.md）。

零 GPU——全部来自既有缓存。Panel A（@290，16 UEA × 2 任务）主判，
Panel B（工作点，HW/UM）佐证。判读带（成立/边缘/不成立）先于计算写死。

用法：python scripts/14_destruction_matrix.py
"""
from __future__ import annotations

import importlib
import json
import statistics
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

from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

OUTDIR = ROOT / "results" / "destruction_matrix"
PANEL_A_TRAINED = (
    ROOT / "results" / "stage1_rerun_20260804" / "results" / "labels_route_a"
    / "rprime_cache.jsonl",
    ROOT / "results" / "baseline_probe" / "trained_fill.jsonl",
)
PANEL_A_BASELINE = ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl"
PANEL_A_K3 = (ROOT / "results" / "stage1_rerun_20260804" / "results"
              / "step0_snr_gate" / "noise_cache.jsonl")
PANEL_B_TRAINED = ROOT / "results" / "labels_workpoint" / "rprime_cache.jsonl"
PANEL_B_BASELINES = (
    ROOT / "results" / "iters_workpoint" / "uea" / "baseline_cache.jsonl",
    ROOT / "results" / "iters_workpoint" / "candidates" / "baseline_cache.jsonl",
)
PANEL_B_ITERS = {"Handwriting": 60, "UniMiB-SHAR": 30}
# 判读带常数（协议 §4，先于计算写死）
TDS_ESTABLISH, TDS_EDGE = 0.25, 0.10
MIXED_ESTABLISH, MIXED_EDGE = 3, 2
MIXED_DESTR_DS, MIXED_SAFE_DS = 3, 3
# 诊断用数据集 L（baseline_probe REPORT §1 表）
DATASET_L = {
    "HandMovementDirection": 400, "EthanolConcentration": 1024,
    "MotorImagery": 512, "PhonemeSpectra": 217, "SelfRegulationSCP2": 1152,
    "Handwriting": 152, "FingerMovements": 50, "Heartbeat": 405,
    "SelfRegulationSCP1": 896, "Cricket": 256, "UWaveGestureLibrary": 315,
    "SpokenArabicDigits": 93, "ERing": 65, "NATOPS": 51,
    "CharacterTrajectories": 182, "ArticularyWordRecognition": 144,
    "UniMiB-SHAR": 151,
}


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def destruction_flag(r_trained: float, base_mean: float, band: float) -> int:
    """协议 §3：D=1 ⇔ trained < mean − band（summarize_task 同带）。"""
    return int(r_trained < base_mean - band)


def rate_table(D: dict) -> dict:
    """D={(cls, dataset): [0/1 per task]} → p[cls][ds] = 均值。"""
    out: dict = defaultdict(dict)
    for (cls, ds), flags in D.items():
        out[cls][ds] = statistics.fmean(flags)
    return dict(out)


def ss_decomposition(p: dict) -> dict:
    """速率表两因素 SS 分解（完备格；缺格数据集整列剔除并报告）。

    返回 shares: cls / dataset / interaction 与 TDS=(ds+inter)/total。
    """
    cls_keys = sorted(p)
    ds_all = sorted({ds for row in p.values() for ds in row})
    ds_keys = [ds for ds in ds_all if all(ds in p[c] for c in cls_keys)]
    dropped = [ds for ds in ds_all if ds not in ds_keys]
    M = np.array([[p[c][ds] for ds in ds_keys] for c in cls_keys], dtype=float)
    grand = M.mean()
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    ss_cls = float(((row - grand) ** 2).sum() * M.shape[1])
    ss_ds = float(((col - grand) ** 2).sum() * M.shape[0])
    ss_inter = float(((M - row - col + grand) ** 2).sum())
    total = ss_cls + ss_ds + ss_inter
    return {
        "n_cls": len(cls_keys), "n_datasets": len(ds_keys),
        "dropped_datasets": dropped, "grand_rate": grand,
        "ss_cls": ss_cls, "ss_ds": ss_ds, "ss_inter": ss_inter,
        "share_cls": ss_cls / total if total else None,
        "share_ds": ss_ds / total if total else None,
        "share_inter": ss_inter / total if total else None,
        "tds": (ss_ds + ss_inter) / total if total else None,
    }


def mixed_cls(p: dict, *, destr_min: int = MIXED_DESTR_DS,
              safe_min: int = MIXED_SAFE_DS) -> list:
    """在 ≥destr_min 个数据集上破坏（p≥0.5）且 ≥safe_min 个上安全（p=0）的 CLS。"""
    out = []
    for cls, row in sorted(p.items()):
        n_destr = sum(1 for v in row.values() if v >= 0.5)
        n_safe = sum(1 for v in row.values() if v == 0.0)
        if n_destr >= destr_min and n_safe >= safe_min:
            out.append({"cls_key": cls, "n_destructive_ds": n_destr,
                        "n_safe_ds": n_safe})
    return out


def verdict(tds, n_mixed: int) -> str:
    """判读带（协议 §4）。"""
    if tds is None:
        return "undecidable"
    if tds >= TDS_ESTABLISH and n_mixed >= MIXED_ESTABLISH:
        return "established"
    if tds >= TDS_EDGE or n_mixed >= MIXED_EDGE:
        return "edge"
    return "not_established"


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def null_calibration(p: dict, *, n_tasks: int, n_sim: int, seed: int,
                     mode: str) -> dict:
    """零模型校准（补充稳健性，报告随判读带一并出）。

    mode='additive'：格子概率 = row+col−grand（无真交互）→ share_inter 与
    mixed 计数的抽样分布，检验观测值是否超出"每格 n_tasks 个二值标志"的噪声；
    mode='cls_only'：格子概率 = row（无数据集效应）→ share_ds 的抽样分布，
    检验数据集主效应显著性。任务间真实异质会被算进"噪声"，检验偏保守。
    """
    cls_keys = sorted(p)
    ds_keys = sorted({d for row in p.values() for d in row
                      if all(d in p[c] for c in cls_keys)})
    M = np.array([[p[c][d] for d in ds_keys] for c in cls_keys])
    grand = M.mean()
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    P0 = np.clip(row + col - grand, 0, 1) if mode == "additive" \
        else np.broadcast_to(row, M.shape)
    rng = np.random.default_rng(seed)
    stat_inter, stat_ds, stat_mixed = [], [], []
    for _ in range(n_sim):
        flags = rng.binomial(n_tasks, P0) / n_tasks
        pt = {c: {d: float(flags[i, j]) for j, d in enumerate(ds_keys)}
              for i, c in enumerate(cls_keys)}
        dec = ss_decomposition(pt)
        stat_inter.append(dec["share_inter"])
        stat_ds.append(dec["share_ds"])
        stat_mixed.append(len(mixed_cls(pt)))
    obs = ss_decomposition(p)
    n_mix_obs = len(mixed_cls(p))
    si, sd, sm = map(np.array, (stat_inter, stat_ds, stat_mixed))
    return {
        "mode": mode, "n_sim": n_sim,
        "inter": {"observed": obs["share_inter"],
                  "null_median": float(np.median(si)),
                  "null_p95": float(np.percentile(si, 95)),
                  "p_value": float((si >= obs["share_inter"]).mean())},
        "ds": {"observed": obs["share_ds"],
               "null_median": float(np.median(sd)),
               "null_p95": float(np.percentile(sd, 95)),
               "p_value": float((sd >= obs["share_ds"]).mean())},
        "mixed": {"observed": n_mix_obs, "null_median": float(np.median(sm)),
                  "p_value": float((sm >= n_mix_obs).mean())},
    }


# ---------------------------------------------------------------- 配对
def canonical_split(tid: str) -> int:
    return run_seed_for(tid, "train_eval_split")


def collect_baselines(paths, *, exclude_plus_c: bool) -> tuple[dict, dict]:
    """{tid: (mean, band)}（≥3 种子）与 {tid: classes}。band=2·max(σ, max 量子)。"""
    vals: dict = defaultdict(list)
    quanta: dict = defaultdict(float)
    classes: dict = {}
    rejects = defaultdict(int)
    for path in paths:
        for r in _c2.read_jsonl(path):
            tid = r.get("task_id")
            if tid is None or r.get("r_prime") is None or r.get("iters") != 0:
                rejects["null_or_iters"] += 1
                continue
            if exclude_plus_c and "+c" in tid:
                rejects["plus_c"] += 1
                continue
            if r.get("split_seed") != canonical_split(tid):
                rejects["split"] += 1
                continue
            if tid in classes and list(classes[tid]) != list(r.get("classes", [])):
                rejects["classes_conflict"] += 1
                continue
            classes[tid] = r.get("classes", [])
            vals[tid].append(float(r["r_prime"]))
            if r.get("n_eval"):
                quanta[tid] = max(quanta[tid], 1.0 / r["n_eval"])
    bands = {}
    for tid, v in vals.items():
        if len(v) < 3:
            rejects["lt3_seeds"] += 1
            continue
        sd = statistics.stdev(v)
        bands[tid] = (statistics.fmean(v), 2.0 * max(sd, quanta[tid]))
    return bands, classes, dict(rejects)


def collect_trained(paths, shared_keys, bands, classes, iters_of) -> tuple[dict, dict]:
    """配对校验后 {(tid, cls_key): R'}；iters_of(dataset)->期望 iters。"""
    out: dict = {}
    rejects = defaultdict(int)
    for path in paths:
        for r in _c2.read_jsonl(path):
            tid, key = r.get("task_id"), r.get("cls_key")
            if tid is None or r.get("r_prime") is None:
                rejects["null"] += 1
                continue
            if key not in shared_keys:
                rejects["not_shared"] += 1
                continue
            ds = tid.rsplit("_", 1)[0].split("+", 1)[0]
            if r.get("iters") != iters_of(ds):
                rejects["iters"] += 1
                continue
            if tid not in bands:
                rejects["no_baseline"] += 1
                continue
            if list(r.get("classes", [])) != list(classes.get(tid, [])):
                rejects["classes"] += 1
                continue
            if r.get("split_seed") != canonical_split(tid):
                rejects["split"] += 1
                continue
            if (tid, key) not in out:   # 先到先得（stage1 缓存优先于补跑）
                out[(tid, key)] = float(r["r_prime"])
    return out, dict(rejects)


def collect_trained_k3(path, shared_keys, bands, classes) -> tuple[dict, dict]:
    """k=3 噪声缓存 → 同 (tid, cls) 跨种子均值（classes 校验；@290）。"""
    vals: dict = defaultdict(list)
    rejects = defaultdict(int)
    for r in _c2.read_jsonl(path):
        tid, key = r.get("task_id"), r.get("cls_key")
        if tid is None or r.get("r_prime") is None or r.get("iters") != 290:
            rejects["null_or_iters"] += 1
            continue
        if key not in shared_keys:
            rejects["not_shared"] += 1
            continue
        if tid not in bands:
            rejects["no_baseline"] += 1
            continue
        if list(r.get("classes", [])) != list(classes.get(tid, [])):
            rejects["classes"] += 1
            continue
        vals[(tid, key)].append(float(r["r_prime"]))
    out = {k: statistics.fmean(v) for k, v in vals.items() if len(v) == 3}
    rejects["lt3_reps"] = sum(1 for v in vals.values() if len(v) != 3)
    return out, dict(rejects)


def build_panel(trained: dict, bands: dict) -> dict:
    """(tid,cls)→R' + 基线带 → 矩阵/速率表/分解/mixed/判读。"""
    D: dict = defaultdict(list)
    matrix_rows = []
    for (tid, key), rp in sorted(trained.items()):
        ds = tid.rsplit("_", 1)[0].split("+", 1)[0]
        mean, band = bands[tid]
        d = destruction_flag(rp, mean, band)
        D[(key, ds)].append(d)
        matrix_rows.append({"task_id": tid, "dataset": ds, "cls_key": key,
                            "r_prime": rp, "base_mean": mean, "band": band,
                            "destroyed": d})
    p = rate_table(D)
    dec = ss_decomposition(p)
    mix = mixed_cls(p)
    per_ds_rate = {ds: statistics.fmean(
        [v for row in p.values() for d2, v in row.items() if d2 == ds])
        for ds in {d2 for row in p.values() for d2 in row}}
    ls = [(DATASET_L[ds], r) for ds, r in per_ds_rate.items() if ds in DATASET_L]
    rho, pval = spearman([x for x, _ in ls], [y for _, y in ls]) \
        if len(ls) >= 5 else (None, None)
    return {"n_pairs": len(trained), "rate_table": p,
            "per_dataset_rate": per_ds_rate,
            "decomposition": dec, "mixed_cls": mix,
            "verdict": verdict(dec["tds"], len(mix)),
            "L_alignment": {"spearman": rho, "p": pval, "n": len(ls)},
            "matrix_rows": matrix_rows}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shared_keys = {k for _c, k in load_probe_cls(_c2.PROBE_CATALOG, 10)[0]}

    bands_a, classes_a, rej_ba = collect_baselines(
        [PANEL_A_BASELINE], exclude_plus_c=True)
    trained_a, rej_ta = collect_trained(
        PANEL_A_TRAINED, shared_keys, bands_a, classes_a, lambda ds: 290)
    panel_a = build_panel(trained_a, bands_a)

    trained_a3, rej_ta3 = collect_trained_k3(
        PANEL_A_K3, shared_keys, bands_a, classes_a)
    panel_a3 = build_panel(trained_a3, bands_a)

    bands_b, classes_b, rej_bb = collect_baselines(
        list(PANEL_B_BASELINES), exclude_plus_c=True)
    trained_b, rej_tb = collect_trained(
        [PANEL_B_TRAINED], shared_keys, bands_b, classes_b,
        lambda ds: PANEL_B_ITERS.get(ds, -1))
    panel_b = build_panel(trained_b, bands_b)

    # 零模型校准（稳健性；additive 检交互与 mixed、cls_only 检数据集主效应）
    for pn in (panel_a, panel_a3):
        pn["null_additive"] = null_calibration(
            pn["rate_table"], n_tasks=2, n_sim=1000, seed=20260830,
            mode="additive")
        pn["null_cls_only"] = null_calibration(
            pn["rate_table"], n_tasks=2, n_sim=1000, seed=20260831,
            mode="cls_only")

    res = {"protocol": "results/destruction_matrix/PROTOCOL.md",
           "panel_A": {**panel_a, "rejects_baseline": rej_ba,
                       "rejects_trained": rej_ta},
           "panel_A_k3": {**panel_a3, "rejects_trained": rej_ta3},
           "panel_B": {**panel_b, "rejects_baseline": rej_bb,
                       "rejects_trained": rej_tb}}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    for name, pn in (("A(@290 k1, 16 UEA)", panel_a),
                     ("A'(@290 k3, 13 UEA)", panel_a3),
                     ("B(工作点, HW/UM)", panel_b)):
        dec = pn["decomposition"]
        print(f"\n===== Panel {name} =====")
        print(f"配对 {pn['n_pairs']} 条 | 数据集 {dec['n_datasets']}"
              f"（剔除 {dec['dropped_datasets']}）| 总破坏率 {dec['grand_rate']:.2f}")
        print(f"SS 份额: cls={dec['share_cls']:.2f} ds={dec['share_ds']:.2f} "
              f"inter={dec['share_inter']:.2f} → TDS={dec['tds']:.2f}")
        print(f"mixed CLS: {len(pn['mixed_cls'])}/10 "
              f"{[m['cls_key'][:8] for m in pn['mixed_cls']]}")
        la = pn["L_alignment"]
        if la["spearman"] is not None:
            print(f"L 对齐: ρ={la['spearman']:.2f} (p={la['p']:.3f}, n={la['n']})")
        print(f"per-dataset 破坏率: "
              f"{ {d: round(v, 2) for d, v in sorted(pn['per_dataset_rate'].items(), key=lambda kv: -kv[1])} }")
        for tag in ("null_additive", "null_cls_only"):
            if tag in pn:
                n = pn[tag]
                print(f"{tag}: inter p={n['inter']['p_value']:.3f} "
                      f"(观测 {n['inter']['observed']:.2f} vs 零中位 "
                      f"{n['inter']['null_median']:.2f}) | ds p="
                      f"{n['ds']['p_value']:.3f} | mixed p="
                      f"{n['mixed']['p_value']:.3f}")
        print(f"[判读] → {pn['verdict']}")
    print(f"\n→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
