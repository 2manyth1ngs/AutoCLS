"""MX-4 WP2：字段结构化代理 vs T-CLSC 的 LODO 评估 + 字段主效应 + 目录外高估诊断——零 GPU。

协议：results/mx4_surrogate/PROTOCOL.md（臂 / 口径 / 门 S1–S3 / 预测先写死）。
数据：results/mx3_pq/{rprime_cache.jsonl, evo_cache.jsonl, delta_calibration.json, rankings.json}（只读）。

三件事：
1. 7 折 LODO：ridge / hgb × scalar / binned 四个代理 + T-CLSC 对照（rankings.json 名次 → 得分），
   同一 δ 口径的配对 acc、top-10 内 acc、regret@1/@3、Spearman；
2. 字段主效应：ridge_binned 逐档系数（字段内去均值）+ hgb_binned 按字段分组置换重要性；
3. 目录外：进化 CLS（evo_cache）用 LODO 模型预测，量高估量 / 合并 Spearman / 到训练集的字段汉明距离，
   并看 LCB（bag10）与信任域能否修正。

用法：python scripts/37_surrogate_loo.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.search.surrogate import FieldSurrogate, field_hamming, in_hull  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")
zscore_within = _g36.zscore_within
features = _g36.features
rank_by = _g36.rank_by
expected_max_of_k = _g36.expected_max_of_k
paired_wilcoxon_less = _g36.paired_wilcoxon_less

OUT = ROOT / "results" / "mx4_surrogate"
MX3 = pq.OUT
BUDGETS = pq.BUDGETS
ARMS = ("ridge_scalar", "ridge_binned", "hgb_scalar", "hgb_binned")
MAIN_ARM = "hgb_binned"
BAG = 10
KAPPA = 1.0
HULL_RADIUS = 3            # 字段汉明半径（PROTOCOL §2：信任域臂）
K_TOP = 3
TOP_GOOD = 10
ALPHA = 0.05
S1_MIN_FOLDS = 5
S3_BIAS_CUT = 0.5          # 高估量须下降 ≥ 50%


# ---------------------------------------------------------------- 纯函数
def pair_accuracy(score: Mapping[str, float], truth: Mapping[str, float], delta: float,
                  top_k: Optional[int] = None) -> tuple[Optional[float], int]:
    """δ 存活无序对上"得分方向 = 真值方向"的比例；top_k 限定两端都在真值前 k。无存活对 → (None, 0)。"""
    keys = list(truth)
    if top_k is not None:
        good = set(sorted(keys, key=lambda k: (-float(truth[k]), k))[:top_k])
        keys = [k for k in keys if k in good]
    hits, n = 0, 0
    for a, b in combinations(keys, 2):
        d = float(truth[a]) - float(truth[b])
        if abs(d) < delta:
            continue
        n += 1
        s = float(score[a]) - float(score[b])
        if s != 0 and np.sign(s) == np.sign(d):
            hits += 1
    return (hits / n if n else None), n


def scores_from_ranking(ranked: Sequence[str]) -> dict[str, float]:
    """名次 → 得分（第 0 名最高），给 T-CLSC 对照臂用同一套配对口径。"""
    return {k: -float(i) for i, k in enumerate(ranked)}


def grouped_permutation_importance(model: FieldSurrogate, X: np.ndarray, y: np.ndarray,
                                   field_slices: Mapping[str, slice], *, seed: int = 0,
                                   n_repeats: int = 3) -> dict[str, float]:
    """按字段整段置换（one-hot 段一起打乱）后的 ΔMSE 均值——字段级重要性。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    base = float(np.mean((model.predict(X)[0] - y) ** 2))
    rng = np.random.default_rng(seed)
    out = {}
    for name, sl in field_slices.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, sl] = X[rng.permutation(X.shape[0])][:, sl]
            deltas.append(float(np.mean((model.predict(Xp)[0] - y) ** 2)) - base)
        out[name] = float(np.mean(deltas))
    return out


def instance_z(r: float, cat_values) -> float:
    v = np.asarray(cat_values, float)
    sd = float(v.std())
    return (float(r) - float(v.mean())) / sd if sd > 0 else 0.0


def lcb(mean, std, kappa: float = KAPPA) -> np.ndarray:
    return np.asarray(mean, float) - float(kappa) * np.asarray(std, float)


def run_gates(fold_acc: Mapping[str, Mapping[str, float]], rows: Sequence[dict],
              evo: Mapping[str, float]) -> dict:
    """S1：主臂配对 acc 赢 T-CLSC 的折数 ≥5/7；S2：regret@1 配对 Wilcoxon；S3：LCB 修正。"""
    better = [f for f, d in fold_acc.items() if d[MAIN_ARM] > d["tclsc"]]
    s1 = {"n_folds": len(fold_acc), "n_folds_better": len(better), "folds_better": sorted(better),
          "pass": bool(len(better) >= S1_MIN_FOLDS)}
    x = np.array([float(r["regret"][MAIN_ARM + "1"]) for r in rows])
    yv = np.array([float(r["regret"]["tclsc1"]) for r in rows])
    p = float(paired_wilcoxon_less(x, yv))
    s2 = {"n": len(rows), "mean_a": float(x.mean()), "mean_b": float(yv.mean()), "p": p,
          "n_better": int(np.sum(x < yv)), "n_worse": int(np.sum(x > yv)), "pass": bool(p < ALPHA)}
    bias_ok = (evo["bias_plain"] > 0 and evo["bias_lcb"] <= (1.0 - S3_BIAS_CUT) * evo["bias_plain"])
    s3 = {**{k: float(v) for k, v in evo.items()},
          "pass": bool(evo["spearman_lcb"] > evo["spearman_plain"] and bias_ok)}
    return {"S1": s1, "S2": s2, "S3": s3}


def verdict_code(g: Mapping[str, dict]) -> str:
    if not g["S1"]["pass"]:
        return "SUR-DEAD"
    return "SUR-WIN" if g["S2"]["pass"] else "SUR-RANK"


# ---------------------------------------------------------------- 装载
def load_inputs(mx3: Path):
    rows = [r for r in read_jsonl(mx3 / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] != "ggs"]
    evo = [r for r in read_jsonl(mx3 / "evo_cache.jsonl") if r.get("error") is None]
    delta = pq.load_delta_calibration(mx3 / "delta_calibration.json")
    rankings = json.loads((mx3 / "rankings.json").read_text(encoding="utf-8"))
    return rows, evo, delta, rankings


def fit_arms(train: Sequence[dict], enc: Mapping[str, CLSEncoder], *, seed: int = 0,
             bag: bool = True) -> dict[str, FieldSurrogate]:
    y = np.array([r["z"] for r in train])
    Xs = features(train, enc["scalar"], max(BUDGETS))
    Xb = features(train, enc["binned"], max(BUDGETS))
    models = {"ridge_scalar": FieldSurrogate("ridge", seed=seed).fit(Xs, y),
              "ridge_binned": FieldSurrogate("ridge", seed=seed, alpha=3.0).fit(Xb, y),
              "hgb_scalar": FieldSurrogate("hgb", seed=seed).fit(Xs, y),
              "hgb_binned": FieldSurrogate("hgb", seed=seed).fit(Xb, y)}
    if bag:
        models["hgb_binned_bag"] = FieldSurrogate("hgb", seed=seed, bagging=BAG).fit(Xb, y)
    return models


def arm_features(rows: Sequence[dict], arm: str, enc: Mapping[str, CLSEncoder]) -> np.ndarray:
    return features(rows, enc["binned" if "binned" in arm else "scalar"], max(BUDGETS))


# ---------------------------------------------------------------- LODO
def fold_eval(target: str, cat_z: Sequence[dict], delta: Mapping, rankings: Mapping,
              enc: Mapping[str, CLSEncoder], *, seed: int = 0) -> list[dict]:
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    models = fit_arms(train, enc, seed=seed, bag=False)
    out = []
    for s in sorted({r["task_id"] for r in test}):
        for b in BUDGETS:
            rows_sb = sorted((r for r in test if r["task_id"] == s and int(r["iters"]) == b),
                             key=lambda r: r["cls_key"])
            keys = [r["cls_key"] for r in rows_sb]
            truth = {r["cls_key"]: float(r["r_prime"]) for r in rows_sb}
            dl = float(delta[(target, b)])
            scores = {arm: dict(zip(keys, models[arm].predict(arm_features(rows_sb, arm, enc))[0]))
                      for arm in ARMS}
            scores["tclsc"] = scores_from_ranking(rankings[target][str(b)][s]["main"])
            vals = list(truth.values())
            oracle = max(vals)
            metrics, regret = {}, {"oracle": 0.0, "random1": oracle - float(np.mean(vals)),
                                   "random3": oracle - float(expected_max_of_k(vals, K_TOP))}
            for arm, sc in scores.items():
                acc, n = pair_accuracy(sc, truth, dl)
                acc_top, n_top = pair_accuracy(sc, truth, dl, top_k=TOP_GOOD)
                ranked = rank_by(sc, keys)
                metrics[arm] = {"pair_acc": acc, "n_pairs": n, "acc_top10": acc_top, "n_top10_pairs": n_top,
                                "rho": float(spearmanr([sc[k] for k in keys], [truth[k] for k in keys]).correlation)}
                regret[arm + "1"] = oracle - truth[ranked[0]]
                regret[arm + "3"] = oracle - max(truth[k] for k in ranked[:K_TOP])
            out.append({"fold": target, "dataset": target, "slice": s, "budget": b, "delta": dl,
                        "metrics": metrics, "regret": regret})
    return out


def summarize(rows: Sequence[dict]) -> dict:
    arms = list(ARMS) + ["tclsc"]
    pooled = {}
    for arm in arms:
        accs = [r["metrics"][arm]["pair_acc"] for r in rows if r["metrics"][arm]["pair_acc"] is not None]
        tops = [r["metrics"][arm]["acc_top10"] for r in rows if r["metrics"][arm]["acc_top10"] is not None]
        pooled[arm] = {"pair_acc": float(np.mean(accs)), "acc_top10": float(np.mean(tops)),
                       "rho": float(np.mean([r["metrics"][arm]["rho"] for r in rows])),
                       "regret1": float(np.mean([r["regret"][arm + "1"] for r in rows])),
                       "regret3": float(np.mean([r["regret"][arm + "3"] for r in rows]))}
    pooled["random"] = {"regret1": float(np.mean([r["regret"]["random1"] for r in rows])),
                        "regret3": float(np.mean([r["regret"]["random3"] for r in rows]))}
    by_fold: dict[str, dict[str, float]] = {}
    for f in sorted({r["fold"] for r in rows}):
        rs = [r for r in rows if r["fold"] == f]
        by_fold[f] = {arm: float(np.mean([r["metrics"][arm]["pair_acc"] for r in rs
                                          if r["metrics"][arm]["pair_acc"] is not None]))
                      for arm in arms}
    by_budget = {}
    for b in BUDGETS:
        rs = [r for r in rows if int(r["budget"]) == b]
        by_budget[str(b)] = {arm: {"pair_acc": float(np.mean([r["metrics"][arm]["pair_acc"] for r in rs
                                                             if r["metrics"][arm]["pair_acc"] is not None])),
                                   "regret1": float(np.mean([r["regret"][arm + "1"] for r in rs]))}
                             for arm in arms}
    return {"pooled": pooled, "by_fold_pair_acc": by_fold, "by_budget": by_budget}


# ---------------------------------------------------------------- 字段主效应
def field_effects(cat_z: Sequence[dict], enc: Mapping[str, CLSEncoder], space: CLSSpace,
                  *, seed: int = 0) -> dict:
    Xb = features(cat_z, enc["binned"], max(BUDGETS))
    y = np.array([r["z"] for r in cat_z])
    ridge = FieldSurrogate("ridge", seed=seed, alpha=3.0).fit(Xb, y)
    coef = ridge._models[0].coef_
    sl = enc["binned"].field_slices
    levels = {}
    for name, s in sl.items():
        spec = space.field_spec(name)
        if spec["type"] == "bool":
            levels[name] = {"True": float(coef[s.start])}       # 相对 False 的增量
            continue
        c = coef[s]
        c = c - c.mean()
        levels[name] = {str(v): float(x) for v, x in zip(spec["choices"], c)}
    ranges = {name: (max(v.values()) - min(v.values()) if len(v) > 1 else abs(list(v.values())[0]))
              for name, v in levels.items()}
    hgb = FieldSurrogate("hgb", seed=seed).fit(Xb, y)
    imp = grouped_permutation_importance(hgb, Xb, y, sl, seed=seed)
    return {"ridge_binned_levels": levels,
            "ridge_binned_range": dict(sorted(ranges.items(), key=lambda kv: -kv[1])),
            "hgb_binned_grouped_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1]))}


# ---------------------------------------------------------------- 目录外诊断
def evo_diagnosis(cat_z: Sequence[dict], evo: Sequence[dict], enc: Mapping[str, CLSEncoder],
                  *, seed: int = 0) -> dict:
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in evo:
        by_ds[r["dataset"]].append(r)
    per_instance = []
    for ds, ev_rows in sorted(by_ds.items()):
        train = [r for r in cat_z if r["dataset"] != ds]
        Xtr = features(train, enc["binned"], max(BUDGETS))
        model = FieldSurrogate("hgb", seed=seed, bagging=BAG).fit(Xtr, np.array([r["z"] for r in train]))
        sl = enc["binned"].field_slices
        inst: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for r in ev_rows:
            inst[(r["task_id"], int(r["iters"]))].append(r)
        for (t, b), rows_e in sorted(inst.items()):
            cat_rows = [r for r in cat_z if r["task_id"] == t and int(r["iters"]) == b]
            cat_vals = np.array([r["r_prime"] for r in cat_rows])
            Xe = features(rows_e, enc["binned"], max(BUDGETS))
            Xc = features(cat_rows, enc["binned"], max(BUDGETS))
            m_e, s_e = model.predict(Xe)
            m_c, s_c = model.predict(Xc)
            true_e = np.array([instance_z(r["r_prime"], cat_vals) for r in rows_e])
            true_c = np.array([instance_z(r["r_prime"], cat_vals) for r in cat_rows])
            ham_e = field_hamming(Xe, Xtr, sl)
            ham_c = field_hamming(Xc, Xtr, sl)
            allt = np.concatenate([true_e, true_c])
            plain = np.concatenate([m_e, m_c])
            lcb_all = np.concatenate([lcb(m_e, s_e), lcb(m_c, s_c)])
            hull = np.where(np.concatenate([in_hull(Xe, Xtr, sl, HULL_RADIUS), in_hull(Xc, Xtr, sl, HULL_RADIUS)]),
                            plain, -np.inf)
            per_instance.append({
                "dataset": ds, "slice": t, "budget": b, "n_evo": len(rows_e), "n_cat": len(cat_rows),
                "bias_plain": float(m_e.mean() - true_e.mean()),
                "bias_lcb": float(lcb(m_e, s_e).mean() - true_e.mean()),
                "bias_cat_plain": float(m_c.mean() - true_c.mean()),
                "spearman_plain": float(spearmanr(plain, allt).correlation),
                "spearman_lcb": float(spearmanr(lcb_all, allt).correlation),
                "spearman_hull": float(spearmanr(hull, allt).correlation),
                "spearman_cat_only": float(spearmanr(m_c, true_c).correlation),
                "hamming_evo_median": float(np.median(ham_e)), "hamming_cat_median": float(np.median(ham_c)),
                "std_evo_mean": float(s_e.mean()), "std_cat_mean": float(s_c.mean()),
                "frac_evo_in_hull": float(np.mean(ham_e <= HULL_RADIUS)),
                "frac_cat_in_hull": float(np.mean(ham_c <= HULL_RADIUS)),
                "evo_true_z_mean": float(true_e.mean()), "evo_pred_z_mean": float(m_e.mean()),
            })
    keys = [k for k in per_instance[0] if isinstance(per_instance[0][k], float)]
    summary = {k: float(np.mean([p[k] for p in per_instance])) for k in keys}
    summary["n_instances"] = len(per_instance)
    return {"summary": summary, "per_instance": per_instance}


# ---------------------------------------------------------------- 预测对账 / 表
INERT = ("temporal", "cross_scale", "pool", "adj_neighbor", "loss_type")
ACTIVE = ("temperature", "freq_mask", "jitter", "resize", "crop", "sim_func")


def predictions(summary: dict, fe: dict, evo_sum: dict) -> dict:
    acc = summary["pooled"][MAIN_ARM]["pair_acc"]
    top3 = list(fe["ridge_binned_range"])[:3]
    imp_order = list(fe["hgb_binned_grouped_importance"])
    inert_pos = [imp_order.index(f) for f in INERT]
    gain = evo_sum["spearman_lcb"] - evo_sum["spearman_plain"]
    return {
        "P1": {"claim": "hgb_binned pooled pair acc ∈ [0.76, 0.81]", "value": acc, "hit": bool(0.76 <= acc <= 0.81)},
        "P2": {"claim": "主效应 range top-3 ⊂ ACTIVE；惰性 5 字段分组重要性都在末 8 位",
               "top3": top3, "inert_positions": dict(zip(INERT, inert_pos)),
               "hit": bool(set(top3) <= set(ACTIVE) and all(p >= len(imp_order) - 8 for p in inert_pos))},
        "P3": {"claim": "进化行高估量 ≥ +0.5 且到训练集字段汉明距离中位 ≥ 4",
               "bias": evo_sum["bias_plain"], "hamming_median": evo_sum["hamming_evo_median"],
               "hit": bool(evo_sum["bias_plain"] >= 0.5 and evo_sum["hamming_evo_median"] >= 4)},
        "P4": {"claim": "LCB 修正后合并 Spearman 提升 < 0.10", "gain": gain, "hit": bool(gain < 0.10)},
    }


def markdown_tables(a: dict) -> str:
    s = a["summary"]
    arms = list(ARMS) + ["tclsc"]
    L = ["## LODO pooled（126 实例）", "", "| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        p = s["pooled"][arm]
        L.append(f"| {arm} | {p['pair_acc']:.3f} | {p['acc_top10']:.3f} | {p['rho']:.3f} | {p['regret1']:.3f} | {p['regret3']:.3f} |")
    L.append(f"| random | — | — | — | {s['pooled']['random']['regret1']:.3f} | {s['pooled']['random']['regret3']:.3f} |")
    L += ["", "## 逐折配对 acc", "", "| 折 | " + " | ".join(arms) + " |", "|---|" + "|".join(["---:"] * len(arms)) + "|"]
    for f, d in s["by_fold_pair_acc"].items():
        L.append(f"| {f} | " + " | ".join(f"{d[a]:.3f}" for a in arms) + " |")
    L += ["", "## 逐预算（配对 acc / regret@1）", "", "| 预算 | " + " | ".join(arms) + " |", "|---|" + "|".join(["---:"] * len(arms)) + "|"]
    for b, d in s["by_budget"].items():
        L.append(f"| @{b} | " + " | ".join(f"{d[a]['pair_acc']:.3f} / {d[a]['regret1']:.3f}" for a in arms) + " |")
    fe = a["field_effects"]
    L += ["", "## 字段主效应（ridge_binned 档内去均值系数的极差；hgb_binned 分组置换 ΔMSE）", "",
          "| 字段 | ridge range | hgb ΔMSE | 最好档 → 最差档 |", "|---|---:|---:|---|"]
    imp = fe["hgb_binned_grouped_importance"]
    for name, rg in fe["ridge_binned_range"].items():
        lv = fe["ridge_binned_levels"][name]
        best = max(lv, key=lv.get)
        worst = min(lv, key=lv.get)
        L.append(f"| {name} | {rg:.3f} | {imp[name]:.4f} | {best} ({lv[best]:+.2f}) → {worst} ({lv[worst]:+.2f}) |")
    ev = a["evo"]["summary"]
    L += ["", "## 目录外（进化 CLS，21 实例均值）", "",
          "| 量 | 值 |", "|---|---:|"]
    for k in ("bias_plain", "bias_lcb", "bias_cat_plain", "spearman_plain", "spearman_lcb", "spearman_hull",
              "spearman_cat_only", "hamming_evo_median", "hamming_cat_median", "std_evo_mean", "std_cat_mean",
              "frac_evo_in_hull", "frac_cat_in_hull", "evo_true_z_mean", "evo_pred_z_mean"):
        L.append(f"| {k} | {ev[k]:.3f} |")
    g = a["gates"]
    L += ["", "## 门", "",
          f"- S1：{MAIN_ARM} 配对 acc 赢 T-CLSC {g['S1']['n_folds_better']}/{g['S1']['n_folds']} 折 → {'✓' if g['S1']['pass'] else '✗'}",
          f"- S2：regret@1 {g['S2']['mean_a']:.3f} vs T-CLSC {g['S2']['mean_b']:.3f}，p={g['S2']['p']:.2e}（{g['S2']['n_better']} 胜 / {g['S2']['n_worse']} 负）→ {'✓' if g['S2']['pass'] else '✗'}",
          f"- S3：Spearman {g['S3']['spearman_plain']:.3f} → LCB {g['S3']['spearman_lcb']:.3f}；高估 {g['S3']['bias_plain']:.3f} → {g['S3']['bias_lcb']:.3f} → {'✓' if g['S3']['pass'] else '✗'}",
          "", f"读数码：**{a['verdict']}**", "", "## 预测对账", "", "| 预测 | 内容 | 读数 | 命中 |", "|---|---|---|---|"]
    for k, p in a["predictions"].items():
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in p.items() if kk not in ("claim", "hit")}
        L.append(f"| {k} | {p['claim']} | {detail} | {'✓' if p['hit'] else '✗'} |")
    return "\n".join(L) + "\n"


def cmd_analyze(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0) -> dict:
    rows, evo, delta, rankings = load_inputs(mx3)
    cat_z = zscore_within(rows)
    space = CLSSpace.from_yaml()
    enc = {"scalar": CLSEncoder(space), "binned": CLSEncoder(space, mode="binned")}
    inst: list[dict] = []
    for target in folds:
        fr = fold_eval(target, cat_z, delta, rankings, enc, seed=seed)
        inst += fr
        def _acc(arm: str) -> float:
            v = [r["metrics"][arm]["pair_acc"] for r in fr if r["metrics"][arm]["pair_acc"] is not None]
            return float(np.mean(v)) if v else float("nan")
        n_empty = sum(r["metrics"][MAIN_ARM]["pair_acc"] is None for r in fr)
        print(f"[fold {target}] {len(fr)} 实例（{n_empty} 个无 δ 存活对）；{MAIN_ARM} pair acc "
              f"{_acc(MAIN_ARM):.3f} vs tclsc {_acc('tclsc'):.3f}", flush=True)
    summary = summarize(inst)
    fe = field_effects(cat_z, enc, space, seed=seed)
    ev = evo_diagnosis(cat_z, evo, enc, seed=seed)
    gates = run_gates(summary["by_fold_pair_acc"], inst, ev["summary"])
    a = {"protocol": "results/mx4_surrogate/PROTOCOL.md", "folds": list(folds), "n_instances": len(inst),
         "summary": summary, "field_effects": fe, "evo": ev, "gates": gates, "verdict": verdict_code(gates),
         "per_instance": inst}
    a["predictions"] = predictions(summary, fe, ev["summary"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']}；{MAIN_ARM} pooled pair acc {summary['pooled'][MAIN_ARM]['pair_acc']:.4f}；"
          f"regret@1 {summary['pooled'][MAIN_ARM]['regret1']:.4f} vs tclsc {summary['pooled']['tclsc']['regret1']:.4f}")
    return a


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--mx3", default=str(MX3))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.analyze:
        cmd_analyze(Path(args.mx3), Path(args.outdir), args.folds, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
