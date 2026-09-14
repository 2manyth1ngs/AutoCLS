"""WP16-TCLSC 阶段 1 转导指纹臂：diff 头 + 目标侧 6 探针 @60 实测指纹，7 折 LODO，零 GPU（协议 results/tclsc_s1fp/PROTOCOL.md）。

子命令：
  --prepare / --train   变体目录 results/tclsc_s1fp/binned/；`33 --train --arms <臂> --encoder-mode binned`（S1 7 折 × 3 种子）
  --rank                每折：normal（自己的指纹）/ cross_s（他数据集训练切片的指纹，3 种子）/ zero（指纹置 0）
  --analyze             两个视图（nonprobe = 剔掉 6 探针 / all20）× M1/M2/M0/regret + 消融 Δ + probe6_top3 参照 + 门 + 读数码

指纹只在本脚本与 33 号装配（铁律 7）；探针与标准化统计取模型 meta（33 号按该折训练切片算，铁律 6）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pq_common as pq  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
zscore_within = _g56.zscore_within
paired_wilcoxon_less = _g56.paired_wilcoxon_less
FP_DIM, FP_K, FP_BUDGET = _m33.FP_DIM, _m33.FP_K, _m33.FP_BUDGET

SOURCE = pq.OUT
OUT = ROOT / "results" / "tclsc_s1fp"
BUDGETS = pq.BUDGETS
FOLDS = tuple(pq.FOLD_TARGETS)
ARMS_ALL = ("blind_diff_hard",) + tuple(_m33.FP_ARMS)
HARD_ARMS = ("blind_diff_hard", "fp_concat_hard", "fp_diff_hard")
CTX_ARMS = tuple(_m33.FP_ARMS)
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
PHI_ANALYSIS = ROOT / "results" / "tclsc_s1phi" / "analysis.json"
# 门（协议 §4）
C0_BAND, C0_REF = 0.01, 0.756
M1_MIN, ALPHA = 0.03, 0.05
PRIMARY = "acc_top10"
TOP_NONPROBE = 7                       # nonprobe 视图 14 条候选的"好半区" = 前 7


# ---------------------------------------------------------------- ctx 行 / 视图 / 参照
def fp_ctx_rows(raw: Mapping[str, np.ndarray], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                cross_map: Optional[Mapping[str, str]] = None, zero: bool = False,
                budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    out = {}
    for s in slices:
        src = cross_map[s] if cross_map is not None else s
        x = np.zeros(FP_DIM) if zero else (np.asarray(raw[src], np.float64) - mu) / sd
        for b in budgets:
            out[(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


cross_slice_map = _g58.cross_slice_map


def nonprobe_view(rankings: Mapping, rows: Sequence[dict], probes: Sequence[str]) -> tuple[dict, list[dict]]:
    """把探针从排名列表与真值行里剔掉（干净的"外推到未测候选"视图）。"""
    pset = set(probes)
    rk = {b: {s: {a: [k for k in lst if k not in pset] for a, lst in pa.items()} for s, pa in per.items()}
          for b, per in rankings.items()}
    return rk, [r for r in rows if r["cls_key"] not in pset]


def probe_top3_regret(cache: Mapping[tuple, Mapping], slice_id: str, probes: Sequence[str], keys: Sequence[str], *,
                      budget: int, fp_budget: int = FP_BUDGET) -> float:
    """同成本参照：探针按 @fp_budget 实测取前 3，regret@3 = 该预算下目录 oracle − 前 3 里的最好真值。"""
    m = {p: float(cache[(slice_id, p, int(fp_budget))]["r_prime"]) for p in probes}
    top3 = sorted(probes, key=lambda p: (-m[p], p))[:3]
    oracle = max(float(cache[(slice_id, k, int(budget))]["r_prime"]) for k in keys)
    return oracle - max(float(cache[(slice_id, p, int(budget))]["r_prime"]) for p in top3)


# ---------------------------------------------------------------- 门 / 读数码
def gates_s1fp(deltas: Mapping, v2: Mapping, v3: Mapping) -> dict:
    """deltas = nonprobe 视图的 cross 消融；v2 = nonprobe regret@3（a=blind, b=fp）；v3 = all20 @120/@290 regret@3（a=probe6_top3, b=fp）。"""
    def pos(arm):
        return _g57._positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)

    def lower(v):
        return bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])

    return {"GF1": {"arm": "fp_diff_hard", "view": "nonprobe", "delta": deltas.get("fp_diff_hard", {}).get(PRIMARY), "pass": pos("fp_diff_hard")},
            "GF2": {"view": "nonprobe", "metric": "regret3", "blind": v2.get("mean_a"), "fp": v2.get("mean_b"), "p": v2.get("p"), "pass": lower(v2)},
            "GF3": {"view": "all20@120/290", "metric": "regret3", "probe6_top3": v3.get("mean_a"), "fp": v3.get("mean_b"), "p": v3.get("p"),
                    "pass": lower(v3)},
            "GF4": {"arm": "fp_concat_hard", "delta": deltas.get("fp_concat_hard", {}).get(PRIMARY), "pass": not pos("fp_concat_hard")},
            "thresholds": {"metric": PRIMARY, "min_delta": M1_MIN, "alpha": ALPHA}}


def verdict_s1fp(g: Mapping) -> str:
    if not g["GF1"]["pass"]:
        return "FP-DEAD"
    if not g["GF2"]["pass"]:
        return "FP-USED-NO-GAIN"
    return "FP-ALIVE" if g["GF3"]["pass"] else "FP-EXTRAPOLATES"


# ---------------------------------------------------------------- 子命令
def cmd_train(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str,
              epochs: Optional[int], resume: bool) -> None:
    d = outdir / "binned"
    _g56.prepare_variant_dir(source, d)
    _m33.cmd_train(d, arms=list(arms), folds=None, seeds=list(seeds), resume=resume, device=device,
                   epochs=epochs, settf_windows=None, encoder_mode="binned")


def cmd_rank(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], folds: Sequence[str],
             device: str = "cpu") -> None:
    d = outdir / "binned"
    cache = _m33.load_rprime_cache(source / "rprime_cache.jsonl")
    rows, _delta = _g57.load_source(source)
    all_slices = sorted({r["task_id"] for r in rows})
    out: dict[str, dict] = {}
    for target in folds:
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        train_slices = [s for s in all_slices if pq.dataset_of(s) != target]
        catalog, _g, _k = _g34.load_catalog(source, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        models = {a: m for a in arms if (m := _g58._load_arm_models(d, target, a, seeds, device)) is not None}
        if not models:
            continue
        blind = {a: m for a, m in models.items() if a.startswith("blind_")}
        ctxa = {a: m for a, m in models.items() if a not in blind}
        per: dict = {"normal": {}, "cross": {}, "zero": {}, "probes": None}
        if blind:
            per["normal"] = _g56.rank_local_models(blind, cat, budgets=BUDGETS, device=device)
        if ctxa:
            meta0 = next(iter(ctxa.values()))[0][1]
            probes, mu, sd = list(meta0["fp_probes"]), np.array(meta0["fp_mu"]), np.array(meta0["fp_sd"])
            for a, mm in ctxa.items():
                for _m, meta in mm:
                    if list(meta["fp_probes"]) != probes or not (np.allclose(meta["fp_mu"], mu) and np.allclose(meta["fp_sd"], sd)):
                        raise ValueError(f"[{target}/{a}] 各臂/种子的探针或标准化统计不一致")
            raw = _m33.fp_raw(cache, sorted(set(test_slices) | set(train_slices)), probes, budget=int(meta0["fp_budget"]))
            per["probes"] = probes

            def merge(dst: dict, rk: dict) -> None:
                for b, pb in rk.items():
                    for s, pa in pb.items():
                        dst.setdefault(b, {}).setdefault(s, {}).update(pa)

            merge(per["normal"], _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device,
                                                        ctx_rows=fp_ctx_rows(raw, test_slices, mu, sd)))
            per["zero"] = _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device,
                                                 ctx_rows=fp_ctx_rows(raw, test_slices, mu, sd, zero=True))
            for cs in CROSS_SEEDS:
                cmap = cross_slice_map(test_slices, train_slices, cs)
                per["cross"][str(cs)] = {"map": cmap, "rankings": _g56.rank_local_models(
                    ctxa, cat, budgets=BUDGETS, device=device, ctx_rows=fp_ctx_rows(raw, test_slices, mu, sd, cross_map=cmap))}
        out[target] = per
        print(f"[{target}] {len(test_slices)} 切片，臂 {sorted(models)}，探针 {per['probes']}", flush=True)
    (d / "rankings_s1fp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {d / 'rankings_s1fp.json'}（{len(out)} 折）", flush=True)


def _eval(rk_by_fold: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], which: str, *,
          view: str, cross_seed: Optional[str] = None, top_k: int) -> list[dict]:
    rows = []
    for target, per in rk_by_fold.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        cat_t = [r for r in cat_z if r["dataset"] == target]
        if view == "nonprobe" and per.get("probes"):
            rk, cat_t = nonprobe_view(rk, cat_t, per["probes"])
        fr = _eval_rows(rk, cat_t, delta, arms, top_k)
        for r in fr:
            r["fold"] = target
        rows += fr
    return rows


def _eval_rows(rankings: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], top_k: int) -> list[dict]:
    """57 号 eval_rankings_l3 的同式，只是 top-k 可变（nonprobe 视图 14 条候选取前 7）。"""
    out = []
    for (ds, s, b), rs in sorted(_g56._instances(cat_z).items()):
        per = rankings.get(str(b), {}).get(s, {})
        scores = {a: _g56.scores_from_ranking(per[a]) for a in arms if a in per}
        if not scores:
            continue
        truth = {r["cls_key"]: float(r["r_prime"]) for r in rs}
        z = {r["cls_key"]: float(r["z"]) for r in rs}
        dl = float(delta[(ds, int(b))])
        keys = sorted(truth)
        vals = list(truth.values())
        oracle = max(vals)
        metrics, regret = {}, {"oracle": 0.0, "random1": oracle - float(np.mean(vals)),
                               "random3": oracle - float(_g56.expected_max_of_k(vals, _g56.K_TOP))}
        for a, sc in scores.items():
            acc, n = _g56.pair_accuracy(sc, truth, dl)
            acc_top, n_top = _g56.pair_accuracy(sc, truth, dl, top_k=top_k)
            ranked = _g56.rank_by(sc, keys)
            wacc, wsum = _g57.weighted_pair_acc(sc, truth, z)
            from scipy.stats import spearmanr
            metrics[a] = {"pair_acc": acc, "n_pairs": n, "acc_top10": acc_top, "n_top10_pairs": n_top, "wacc": wacc, "wsum": wsum,
                          "rho": float(spearmanr([sc[k] for k in keys], [truth[k] for k in keys]).correlation)}
            regret[a + "1"] = oracle - truth[ranked[0]]
            regret[a + "3"] = oracle - max(truth[k] for k in ranked[:_g56.K_TOP])
        out.append({"fold": ds, "dataset": ds, "slice": s, "budget": int(b), "delta": dl, "metrics": metrics, "regret": regret})
    return out


def cmd_analyze(source: Path, outdir: Path, *, arms: Sequence[str]) -> dict:
    d = outdir / "binned"
    rk = json.loads((d / "rankings_s1fp.json").read_text(encoding="utf-8"))
    cache = _m33.load_rprime_cache(source / "rprime_cache.jsonl")
    rows, delta = _g57.load_source(source)
    cat_z = zscore_within(rows)
    present = [a for a in arms if any(a in pa for per in rk.values() for pb in per["normal"].values() for pa in pb.values())]
    ctx_present = [a for a in present if a in CTX_ARMS]
    cross_seeds = sorted({cs for per in rk.values() for cs in per["cross"]})
    views: dict[str, dict] = {}
    for view, top_k in (("nonprobe", TOP_NONPROBE), ("all20", _g56.TOP_GOOD)):
        normal = _eval(rk, cat_z, delta, present, "normal", view=view, top_k=top_k)
        cross = [_eval(rk, cat_z, delta, ctx_present, "cross", view=view, cross_seed=cs, top_k=top_k) for cs in cross_seeds]
        zero = _eval(rk, cat_z, delta, ctx_present, "zero", view=view, top_k=top_k)
        pooled = {}
        for a in present:
            pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
            pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
            pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
        deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_present},
                  "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_present}}
        by_fold = {a: {fo: _g57.ablation_delta([r for r in normal if r["fold"] == fo], [[r for r in c if r["fold"] == fo] for c in cross],
                                               a, PRIMARY).get("mean_delta") for fo in sorted({r["fold"] for r in normal})}
                   for a in ctx_present}
        value = [_g58.paired_regret(normal, "blind_diff_hard", b) for b in ctx_present if "blind_diff_hard" in present]
        views[view] = {"n_instances": len(normal), "pooled": pooled, "deltas": deltas, "by_fold_delta": by_fold, "value_tests": value,
                       "per_instance_normal": normal}
    # 同成本参照（all20，@120/@290）：probe6_top3 vs fp_diff_hard 的 regret@3
    ref_rows = []
    for r in views["all20"]["per_instance_normal"]:
        if int(r["budget"]) == FP_BUDGET or "fp_diff_hard3" not in r["regret"]:
            continue
        probes = rk[r["fold"]]["probes"]
        keys = [x["cls_key"] for x in cat_z if x["task_id"] == r["slice"] and int(x["iters"]) == int(r["budget"])]
        ref_rows.append({"slice": r["slice"], "budget": r["budget"],
                         "probe6_top3": probe_top3_regret(cache, r["slice"], probes, keys, budget=int(r["budget"])),
                         "fp": float(r["regret"]["fp_diff_hard3"]), "blind": float(r["regret"].get("blind_diff_hard3", np.nan))})
    if ref_rows:
        x = np.array([r["probe6_top3"] for r in ref_rows]); y = np.array([r["fp"] for r in ref_rows])
        v3 = {"a": "probe6_top3", "b": "fp_diff_hard", "n": int(x.size), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
              "p": float(paired_wilcoxon_less(y, x)), "wins_b": int(np.sum(y < x)), "losses_b": int(np.sum(y > x)), "ties": int(np.sum(y == x)),
              "blind_mean": float(np.nanmean([r["blind"] for r in ref_rows]))}
    else:
        v3 = {"a": "probe6_top3", "b": "fp_diff_hard", "n": 0, "mean_a": None, "mean_b": None, "p": None}
    v2 = next((t for t in views["nonprobe"]["value_tests"] if t["b"] == "fp_diff_hard"), {"p": None, "mean_a": None, "mean_b": None})
    g = gates_s1fp(views["nonprobe"]["deltas"]["cross"], v2, v3)
    c0_val = views["all20"]["pooled"].get("blind_diff_hard", {}).get("pair_acc")
    c0 = {"blind_diff_hard": c0_val, "ref": C0_REF, "band": C0_BAND, "pass": None if c0_val is None else bool(abs(c0_val - C0_REF) <= C0_BAND)}
    phi_ref = None
    if PHI_ANALYSIS.exists():
        phi_ref = json.loads(PHI_ANALYSIS.read_text(encoding="utf-8"))["pooled"]
    a = {"protocol": "results/tclsc_s1fp/PROTOCOL.md", "split": "S1 LODO", "arms": present, "views": views, "ref_same_cost": v3,
         "gates": g, "verdict": verdict_s1fp(g), "c0": c0, "phi_ref": phi_ref, "probes_by_fold": {t: per["probes"] for t, per in rk.items()}}
    a["predictions"] = predictions(a)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    pn = views["nonprobe"]["pooled"]
    print(f"verdict {a['verdict']} | " + " | ".join(
        f"{arm} M1n {pn[arm]['acc_top10']:.3f} r3n {pn[arm]['regret3']:.3f}"
        + (f" Δ {views['nonprobe']['deltas']['cross'][arm]['acc_top10']['mean_delta']:+.3f}" if arm in ctx_present else "")
        for arm in present) + f" | probe6_top3 {v3.get('mean_a')} vs fp {v3.get('mean_b')}", flush=True)
    return a


def predictions(a: dict) -> dict:
    g, vn, va = a["gates"], a["views"]["nonprobe"], a["views"]["all20"]
    d_n = (vn["deltas"]["cross"].get("fp_diff_hard", {}).get(PRIMARY) or {}).get("mean_delta")
    d_a = (va["deltas"]["cross"].get("fp_diff_hard", {}).get(PRIMARY) or {}).get("mean_delta")
    m0 = vn["pooled"].get("fp_diff_hard", {}).get("pair_acc")
    return {"P1": {"desc": "C0 带内", "hit": a["c0"]["pass"]},
            "P2": {"desc": "G-F1 过", "hit": g["GF1"]["pass"]},
            "P3": {"desc": "G-F2 过", "hit": g["GF2"]["pass"]},
            "P4": {"desc": "G-F3 过", "hit": g["GF3"]["pass"]},
            "P5": {"desc": "all20 Δ_M1 > nonprobe Δ_M1", "value": [d_a, d_n], "hit": None if d_a is None or d_n is None else bool(d_a > d_n)},
            "P6": {"desc": "fp 臂 M0（nonprobe）≥ φ 臂 M0 0.780", "value": m0, "hit": None if m0 is None else bool(m0 >= 0.780)},
            "P7": {"desc": "concat 对照 nonprobe Δ_M1 零", "hit": g["GF4"]["pass"]}}


def markdown_tables(a: dict) -> str:
    f = lambda b, k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
    L = [f"# WP16-TCLSC 阶段 1 转导指纹臂 tables（7 折 LODO）", "", f"读数码 **{a['verdict']}**", ""]
    for view in ("nonprobe", "all20"):
        v = a["views"][view]
        L += [f"## 视图 {view}（{v['n_instances']} 实例）— normal", "",
              "| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in a["arms"]:
            b = v["pooled"][arm]
            L.append(f"| {arm} | {f(b,'acc_top10')} | {f(b,'wacc')} | {f(b,'pair_acc')} | {f(b,'rho')} | {f(b,'regret1')} | {f(b,'regret3')} |")
        L += ["", f"### {view} 消融 Δ", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
        for arm in a["arms"]:
            for kind in ("cross", "zero"):
                for metric in ("acc_top10", "wacc", "pair_acc"):
                    d = v["deltas"].get(kind, {}).get(arm, {}).get(metric)
                    if not d or d.get("mean_delta") is None:
                        continue
                    L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | "
                             f"{d['wins']}/{d['losses']}/{d['ties']} |")
        L += ["", f"### {view} 逐折 Δ_M1（cross）", ""]
        for arm, bf in v["by_fold_delta"].items():
            L.append(f"- {arm}: " + ", ".join(f"{fo[:6]} {('—' if x is None else f'{x:+.3f}')}" for fo, x in bf.items()))
        L += ["", f"### {view} 价值（regret@3 配对，H1 fp 臂更低）", ""]
        for t in v["value_tests"]:
            L.append(f"- {t['a']} → {t['b']}: n={t['n']} {t['mean_a']} → {t['mean_b']} p={t['p']} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
        L.append("")
    r = a["ref_same_cost"]
    L += ["## 同成本参照（all20，@120/@290）", "",
          f"- probe6_top3 regret@3 {r.get('mean_a')} vs fp_diff_hard {r.get('mean_b')}（blind {r.get('blind_mean')}）n={r.get('n')} p={r.get('p')} "
          f"胜/负/平 {r.get('wins_b')}/{r.get('losses_b')}/{r.get('ties')}", "", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    L += ["", "## 探针（逐折）", ""] + [f"- {t}: {p}" for t, p in a["probes_by_fold"].items()]
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(HARD_ARMS), choices=list(ARMS_ALL))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--folds", nargs="+", default=list(FOLDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.prepare, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --prepare / --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        print(f"[prepare] {_g56.prepare_variant_dir(args.source, args.outdir / 'binned') or '已就位'}", flush=True)
    if args.train:
        cmd_train(args.source, args.outdir, arms=args.arms, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.source, args.outdir, arms=list(ARMS_ALL), seeds=args.seeds, folds=args.folds)
    if args.analyze:
        cmd_analyze(args.source, args.outdir, arms=list(ARMS_ALL))


if __name__ == "__main__":
    main()
