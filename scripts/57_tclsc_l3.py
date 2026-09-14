"""WP16-TCLSC-L3 阶段 0：融合头 × 标签形式，oracle 数据集身份 token，S2 划分（协议 results/tclsc_l3/PROTOCOL.md）。

子命令（零新 R'，`results/mx3_pq/` 只读；比较器训练在本地 GPU/CPU 上分钟级）：
  --c0        本地重训 `onehot_oracle`（scalar，pod 同臂）→ normal acc 与 runs/S2/onehot_oracle_s0.json 差 ≤ 0.005
  --prepare   变体目录 results/tclsc_l3/binned/ 只复制输入（56 号 prepare_variant_dir）
  --train     `33 --train --arms <L3_ARMS> --encoder-mode binned`（S2，3 种子）
  --rank      每臂 3 种子多数票排名：normal（各切片自己的数据集 one-hot）+ cross_s（数据集 one-hot 错位，3 种子）
  --analyze   M1 好半区配对 acc / M2 |Δz| 加权全对 acc / M0 δ 对 acc + 消融 Δ + 门 G0/GA/GB/GC + 读数码 + 预测对账

任务无关（铁律 7）：数据集 one-hot 只在本脚本与 33 号构造；模型 / 引擎 / zdiff 标签模块不见数据集概念。
阶段 0 是 S2 机制上界（8 数据集全参与、留出切片考试），**不是 zero-shot**；zero-shot 只在阶段 1（真 E + LODO）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pq_common as pq  # noqa: E402

_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
zscore_within = _g56.zscore_within
rank_by = _g56.rank_by
paired_wilcoxon_less = _g56.paired_wilcoxon_less
pair_accuracy = _g56.pair_accuracy
scores_from_ranking = _g56.scores_from_ranking
expected_max_of_k = _g56.expected_max_of_k
instance_metrics = _g56.instance_metrics

SOURCE = pq.OUT
OUT = ROOT / "results" / "tclsc_l3"
BUDGETS = pq.BUDGETS
DATASETS = tuple(pq.PQ_DATASETS)
S2_FOLD = "S2"
L3_ARMS = ("l3_none_hard", "l3_none_zdiff", "l3_concat_hard", "l3_concat_zdiff",
           "l3_diff_hard", "l3_diff_zdiff", "l3_film_hard", "l3_film_zdiff")
CTX_ARMS = tuple(a for a in L3_ARMS if not a.startswith("l3_none"))
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
# 门（协议 §4，先于数据）
C0_TOL = 0.005
M1_MIN, M2_MIN, ALPHA = 0.03, 0.01, 0.05
PRIMARY = "acc_top10"


# ---------------------------------------------------------------- ctx
def ctx_row(dataset: str, budget: int, datasets: Sequence[str] = DATASETS,
            budgets: Sequence[int] = BUDGETS) -> np.ndarray:
    """数据集 one-hot ⊕ 预算 one-hot（与 33 号 context_for(ctx_kind='dataset_budget') 逐位相同）。"""
    order, bs = list(datasets), tuple(budgets)
    v = np.zeros(len(order) + len(bs), dtype=np.float32)
    v[order.index(dataset)] = 1.0
    v[len(order) + bs.index(int(budget))] = 1.0
    return v


def cross_dataset_map(datasets: Sequence[str], seed: int) -> dict[str, str]:
    """数据集间错位（无不动点排列），按 seed 逐位确定。"""
    ds = list(datasets)
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(len(ds))
        if not np.any(p == np.arange(len(ds))):
            return {ds[i]: ds[int(p[i])] for i in range(len(ds))}


def ctx_rows_for(slices: Sequence[str], *, cross: Optional[Mapping[str, str]] = None,
                 budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    out = {}
    for s in slices:
        d = pq.dataset_of(s)
        d_use = cross[d] if cross is not None else d
        for b in budgets:
            out[(s, int(b))] = ctx_row(d_use, int(b))
    return out


def s2_test_slices(slices: Sequence[str]) -> list[str]:
    return list(pq.s2_split(list(slices))[1])


# ---------------------------------------------------------------- 指标
def weighted_pair_acc(score: Mapping[str, float], truth: Mapping[str, float],
                      z: Mapping[str, float]) -> tuple[Optional[float], float]:
    """M2：全部无序对（不过 δ），权重 |z_i − z_j|，得分方向 = 真值方向 的加权比例。"""
    keys = list(truth)
    hit, wsum = 0.0, 0.0
    for a, b in combinations(keys, 2):
        w = abs(float(z[a]) - float(z[b]))
        if w == 0:
            continue
        d = float(truth[a]) - float(truth[b])
        s = float(score[a]) - float(score[b])
        wsum += w
        if s != 0 and np.sign(s) == np.sign(d):
            hit += w
    return (hit / wsum if wsum > 0 else None), wsum


def eval_rankings_l3(rankings: Mapping, cat_z: Sequence[dict], delta: Mapping, *,
                     arms: Sequence[str]) -> list[dict]:
    """{预算: {切片: {臂: [key…]}}}（单靶 S2 结构）→ 实例行：56 号指标 + wacc。只评 rankings 里出现的切片。"""
    out = []
    for (ds, s, b), rs in sorted(_g56._instances(cat_z).items()):
        per = rankings.get(str(b), {}).get(s, {})
        scores = {a: scores_from_ranking(per[a]) for a in arms if a in per}
        if not scores:
            continue
        truth = {r["cls_key"]: float(r["r_prime"]) for r in rs}
        z = {r["cls_key"]: float(r["z"]) for r in rs}
        dl = float(delta[(ds, int(b))])
        m, reg = instance_metrics(scores, truth, dl)
        for a, sc in scores.items():
            m[a]["wacc"], m[a]["wsum"] = weighted_pair_acc(sc, truth, z)
        out.append({"fold": S2_FOLD, "dataset": ds, "slice": s, "budget": int(b), "delta": dl,
                    "metrics": m, "regret": reg})
    return out


def _metric_map(rows: Sequence[dict], arm: str, metric: str) -> dict[tuple[str, int], float]:
    return {(r["slice"], int(r["budget"])): r["metrics"][arm][metric]
            for r in rows if arm in r["metrics"] and r["metrics"][arm].get(metric) is not None}


def ablation_delta(rows_normal: Sequence[dict], rows_cross: Sequence[Sequence[dict]],
                   arm: str, metric: str) -> dict:
    """Δ = normal − mean_s cross_s，按实例配对；Wilcoxon H1：normal > cross。"""
    nm = _metric_map(rows_normal, arm, metric)
    cms = [_metric_map(rs, arm, metric) for rs in rows_cross]
    keys = sorted(k for k in nm if all(k in cm for cm in cms))
    if not keys:
        return {"arm": arm, "metric": metric, "n": 0, "mean_delta": None, "p": None}
    x = np.array([nm[k] for k in keys])
    y = np.array([np.mean([cm[k] for cm in cms]) for k in keys])
    return {"arm": arm, "metric": metric, "n": len(keys), "normal": float(x.mean()), "cross": float(y.mean()),
            "mean_delta": float(np.mean(x - y)), "p": float(paired_wilcoxon_less(y, x)),
            "wins": int(np.sum(x > y)), "losses": int(np.sum(x < y)), "ties": int(np.sum(x == y))}


# ---------------------------------------------------------------- 门 / 读数码
def _positive(d: Mapping, *, min_delta: float, alpha: float = ALPHA) -> bool:
    md, p = d.get("mean_delta"), d.get("p")
    return bool(md is not None and p is not None and md >= min_delta and p < alpha)


def gates_l3(deltas: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """deltas[臂][指标] = {mean_delta, p}。G0：concat_hard 零；GA：diff/film hard 任一正；GB：diff/film zdiff；GC：concat_zdiff。"""
    def pos(arm):
        return _positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)

    g = {"G0": {"arm": "l3_concat_hard", "pass": not pos("l3_concat_hard"),
                "delta": deltas.get("l3_concat_hard", {}).get(PRIMARY)},
         "GA": {"arms": ["l3_diff_hard", "l3_film_hard"], "pass": pos("l3_diff_hard") or pos("l3_film_hard")},
         "GB": {"arms": ["l3_diff_zdiff", "l3_film_zdiff"], "pass": pos("l3_diff_zdiff") or pos("l3_film_zdiff")},
         "GC": {"arm": "l3_concat_zdiff", "pass": pos("l3_concat_zdiff")}}
    for k in ("GA", "GB"):
        g[k]["deltas"] = {a: deltas.get(a, {}).get(PRIMARY) for a in g[k]["arms"]}
    g["GC"]["delta"] = deltas.get("l3_concat_zdiff", {}).get(PRIMARY)
    g["thresholds"] = {"metric": PRIMARY, "min_delta": M1_MIN, "alpha": ALPHA}
    return g


def verdict_l3(g: Mapping) -> str:
    if not g["G0"]["pass"]:
        return "L3-ENC-SUFFICES"
    if g["GA"]["pass"]:
        return "L3-HEAD-ONLY"
    if g["GC"]["pass"]:
        return "L3-LABEL-ONLY"
    if g["GB"]["pass"]:
        return "L3-HEAD+LABEL"
    return "L3-FLAT"


# ---------------------------------------------------------------- 子命令
def load_source(source: Path = SOURCE):
    rows = [r for r in _g56.read_jsonl(source / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] != "ggs"]
    delta = pq.load_delta_calibration(source / "delta_calibration.json")
    return rows, delta


def cmd_c0(source: Path, outdir: Path, *, device: str) -> dict:
    m33 = importlib.import_module("scripts.33_pq_comparator_loo")
    d = outdir / "c0"
    _g56.prepare_variant_dir(source, d)
    m33.cmd_train(d, arms=["onehot_oracle"], folds=None, seeds=[0], resume=True, device=device,
                  epochs=None, settf_windows=None, encoder_mode="scalar")
    local = json.loads((d / "runs" / S2_FOLD / "onehot_oracle_s0.json").read_text(encoding="utf-8"))
    pod = json.loads((source / "runs" / S2_FOLD / "onehot_oracle_s0.json").read_text(encoding="utf-8"))
    la, pa = float(local["normal"]["acc"]), float(pod["normal"]["acc"])
    res = {"local_acc": la, "pod_acc": pa, "delta": la - pa, "tol": C0_TOL, "pass": bool(abs(la - pa) <= C0_TOL),
           "local_cross": local.get("cross"), "pod_cross": pod.get("cross")}
    (outdir / "c0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"C0 onehot_oracle normal acc 本地 {la:.4f} vs pod {pa:.4f} → {'PASS' if res['pass'] else 'FAIL'}", flush=True)
    return res


def cmd_train(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str,
              epochs: Optional[int], resume: bool) -> None:
    m33 = importlib.import_module("scripts.33_pq_comparator_loo")
    d = outdir / "binned"
    _g56.prepare_variant_dir(source, d)
    m33.cmd_train(d, arms=list(arms), folds=None, seeds=list(seeds), resume=resume, device=device,
                  epochs=epochs, settf_windows=None, encoder_mode="binned")


def _load_arm_models(d: Path, arm: str, seeds: Sequence[int], device: str) -> Optional[list]:
    paths = [d / "models" / S2_FOLD / f"{arm}_s{s}.pt" for s in seeds]
    have = [p for p in paths if p.exists()]
    if len(have) != len(paths):
        print(f"⚠️ [{arm}] 模型 {len(have)}/{len(paths)}，整臂跳过", flush=True)
        return None
    return [_g34.load_pq_model(p, map_location=device, expect_fold=S2_FOLD, expect_budgets=BUDGETS) for p in have]


def cmd_rank(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str = "cpu") -> None:
    d = outdir / "binned"
    rows, _delta = load_source(source)
    all_slices = sorted({r["task_id"] for r in rows})
    test_slices = s2_test_slices(all_slices)
    catalog, _g, _k = _g34.load_catalog(source, _g56.space(), slices=test_slices)
    cat = {s: catalog[s] for s in test_slices}
    models = {a: m for a in arms if (m := _load_arm_models(d, a, seeds, device)) is not None}
    if not models:
        sys.exit("没有可排名的臂：先 --train")
    normal_ctx = ctx_rows_for(test_slices)
    out = {"slices": test_slices, "normal": {}, "cross": {}}
    # 任务盲臂与 ctx 臂分开调：前者不吃 ctx_rows，后者 normal / cross 各一份
    blind = {a: m for a, m in models.items() if a.startswith("l3_none")}
    ctxa = {a: m for a, m in models.items() if a not in blind}
    merged: dict[str, dict] = {}
    if blind:
        merged = _g56.rank_local_models(blind, cat, budgets=BUDGETS, device=device)
    if ctxa:
        rk = _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device, ctx_rows=normal_ctx)
        for b, per in rk.items():
            for s, pa in per.items():
                merged.setdefault(b, {}).setdefault(s, {}).update(pa)
        for cs in CROSS_SEEDS:
            cmap = cross_dataset_map(DATASETS, cs)
            out["cross"][str(cs)] = {"map": cmap,
                                     "rankings": _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device,
                                                                        ctx_rows=ctx_rows_for(test_slices, cross=cmap))}
    out["normal"] = merged
    (d / "rankings_l3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {d / 'rankings_l3.json'}：{len(test_slices)} 切片 × {len(BUDGETS)} 预算，臂 {sorted(models)}，"
          f"cross 种子 {list(CROSS_SEEDS) if ctxa else '—'}", flush=True)


def _pooled(rows: Sequence[dict], arm: str, metric: str) -> Optional[float]:
    v = [r["metrics"][arm][metric] for r in rows if arm in r["metrics"] and r["metrics"][arm].get(metric) is not None]
    return float(np.mean(v)) if v else None


def predictions(pooled: Mapping, deltas: Mapping, g: Mapping, c0: Optional[Mapping], m0_pairs: Mapping) -> dict:
    def d(arm):
        return (deltas.get(arm, {}).get(PRIMARY) or {}).get("mean_delta")

    out = {"P1": {"desc": "C0 过", "hit": None if c0 is None else bool(c0["pass"])},
           "P2": {"desc": "G0 过（A8 在 binned 下仍零）", "hit": g["G0"]["pass"]},
           "P3": {"desc": "G-A 过", "hit": g["GA"]["pass"]},
           "P4": {"desc": "G-B 过", "hit": g["GB"]["pass"]},
           "P5": {"desc": "G-C 过", "hit": g["GC"]["pass"]}}
    fz, dz = d("l3_film_zdiff"), d("l3_diff_zdiff")
    out["P6"] = {"desc": "film Δ_M1 ≥ diff Δ_M1（zdiff）", "value": [fz, dz],
                 "hit": None if fz is None or dz is None else bool(fz >= dz)}
    worst = min((m0_pairs[k] for k in m0_pairs), default=None)
    out["P7"] = {"desc": "zdiff 臂 M0 不比同头 hard 臂低 0.02 以上", "value": m0_pairs,
                 "hit": None if worst is None else bool(worst >= -0.02)}
    nz, nh = pooled.get("l3_none_zdiff", {}).get(PRIMARY), pooled.get("l3_none_hard", {}).get(PRIMARY)
    out["P8"] = {"desc": "任务盲 zdiff M1 ≥ hard + 0.02", "value": None if nz is None or nh is None else nz - nh,
                 "hit": None if nz is None or nh is None else bool(nz - nh >= 0.02)}
    return out


def markdown_tables(a: dict) -> str:
    arms = a["arms"]
    p = a["pooled"]
    L = [f"# WP16-TCLSC-L3 阶段 0 tables（{a['n_instances']} 实例，S2 考侧）", "", f"读数码 **{a['verdict']}**", "",
         "## normal（各切片自己的数据集 token）", "",
         "| 臂 | M1 好半区 acc | M2 |Δz| 加权 acc | M0 δ 对 acc | Spearman | regret@1 | regret@3 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        b = p[arm]
        f = lambda k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
        L.append(f"| {arm} | {f('acc_top10')} | {f('wacc')} | {f('pair_acc')} | {f('rho')} | {f('regret1')} | {f('regret3')} |")
    L += ["", "## 消融 Δ = normal − cross（数据集 token 错位，3 种子均值；Wilcoxon H1 normal > cross）", "",
          "| 臂 | 指标 | normal | cross | Δ | p | 胜/负/平 |", "|---|---|---:|---:|---:|---:|---|"]
    for arm in arms:
        for metric in ("acc_top10", "wacc", "pair_acc"):
            d = a["deltas"].get(arm, {}).get(metric)
            if not d or d.get("mean_delta") is None:
                continue
            L.append(f"| {arm} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | "
                     f"{d['wins']}/{d['losses']}/{d['ties']} |")
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k == "thresholds":
            continue
        L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    L += ["", "## 33 号 run json（δ 对，模型输出口径）：normal acc / cross 档 Δ̄", ""]
    for arm, v in a["run_json"].items():
        L.append(f"- {arm}: normal {v['normal_acc']}, cross Δ̄ {v['cross_delta']} tier {v['cross_tier']}, zero_ctx Δ̄ {v['zero_ctx_delta']}")
    L += ["", "## 预测对账", ""]
    for k, v in a["predictions"].items():
        L.append(f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else ""))
    return "\n".join(L) + "\n"


def _run_json_summary(d: Path, arms: Sequence[str], seeds: Sequence[int]) -> dict:
    out = {}
    for arm in arms:
        accs, cross, zero, tiers = [], [], [], []
        for s in seeds:
            pth = d / "runs" / S2_FOLD / f"{arm}_s{s}.json"
            if not pth.exists():
                continue
            r = json.loads(pth.read_text(encoding="utf-8"))
            accs.append(float(r["normal"]["acc"]))
            if r.get("cross"):
                cross.append(float(r["cross"]["mean_delta"]))
                tiers.append(r["cross"].get("tier"))
            if r.get("zero_ctx"):
                zero.append(float(r["zero_ctx"]["mean_delta"]))
        if accs:
            out[arm] = {"normal_acc": round(float(np.mean(accs)), 4),
                        "cross_delta": round(float(np.mean(cross)), 4) if cross else None,
                        "cross_tier": sorted(set(t for t in tiers if t)) if tiers else None,
                        "zero_ctx_delta": round(float(np.mean(zero)), 4) if zero else None}
    return out


def cmd_analyze(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int]) -> dict:
    d = outdir / "binned"
    rk = json.loads((d / "rankings_l3.json").read_text(encoding="utf-8"))
    rows, delta = load_source(source)
    cat_z = zscore_within(rows)
    present = [a for a in arms if any(a in per for b in rk["normal"].values() for per in b.values())]
    normal = eval_rankings_l3(rk["normal"], cat_z, delta, arms=present)
    cross = [eval_rankings_l3(v["rankings"], cat_z, delta, arms=[a for a in present if a in CTX_ARMS])
             for v in rk["cross"].values()]
    pooled = {a: {m: _pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")} for a in present}
    for a in present:
        pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
        pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
    deltas = {a: {m: ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")}
              for a in present if a in CTX_ARMS}
    g = gates_l3(deltas)
    m0_pairs = {}
    for head in ("none", "concat", "diff", "film"):
        h, z = f"l3_{head}_hard", f"l3_{head}_zdiff"
        if h in pooled and z in pooled and pooled[h]["pair_acc"] is not None and pooled[z]["pair_acc"] is not None:
            m0_pairs[head] = round(pooled[z]["pair_acc"] - pooled[h]["pair_acc"], 4)
    blind_vs = {a: {"acc_top10": (pooled[a]["acc_top10"] - pooled["l3_none_hard"]["acc_top10"])
                    if "l3_none_hard" in pooled and pooled[a]["acc_top10"] is not None else None} for a in present}
    c0_path = outdir / "c0.json"
    c0 = json.loads(c0_path.read_text(encoding="utf-8")) if c0_path.exists() else None
    a = {"protocol": "results/tclsc_l3/PROTOCOL.md", "stage": 0, "split": "S2（非 zero-shot）",
         "n_instances": len(normal), "arms": present, "pooled": pooled, "deltas": deltas, "gates": g,
         "verdict": verdict_l3(g), "m0_zdiff_minus_hard": m0_pairs, "vs_blind_hard": blind_vs,
         "run_json": _run_json_summary(d, present, seeds), "c0": c0, "cross_maps": {k: v["map"] for k, v in rk["cross"].items()},
         "per_instance_normal": normal}
    a["predictions"] = predictions(pooled, deltas, g, c0, m0_pairs)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']} | " + " | ".join(
        f"{arm} M1 {pooled[arm]['acc_top10']:.3f}" + (f" Δ {deltas[arm]['acc_top10']['mean_delta']:+.3f}" if arm in deltas else "")
        for arm in present), flush=True)
    return a


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(L3_ARMS), choices=list(L3_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 33 号训练 epoch（只用于冒烟）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.c0, args.prepare, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --c0 / --prepare / --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.c0:
        res = cmd_c0(args.source, args.outdir, device=args.device)
        if not res["pass"]:
            sys.exit("C0 不过：本地与 pod 的 onehot_oracle 不在带内，不往下跑")
    if args.prepare:
        print(f"[prepare] {_g56.prepare_variant_dir(args.source, args.outdir / 'binned') or '已就位'}", flush=True)
    if args.train:
        cmd_train(args.source, args.outdir, arms=args.arms, seeds=args.seeds, device=args.device,
                  epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.source, args.outdir, arms=args.arms, seeds=args.seeds, device="cpu")
    if args.analyze:
        cmd_analyze(args.source, args.outdir, arms=args.arms, seeds=args.seeds)


if __name__ == "__main__":
    main()
