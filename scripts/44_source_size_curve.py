"""源池规模曲线：先验（shared-10 的 within-task z 均值）随训练源数量 k 的 regret 曲线（零 GPU）。

数据 = results/mx3_pq/rprime_cache.jsonl；折 = 7 个靶（SCP2 只训练）；每折训练源 = 其余 7 个，k=1..7 枚举全部子集。
指标（靶上每个 切片×预算 实例）：regret@1 / regret@3 / oracle 是否在先验前 6 / @290 的 SH6-3-3 regret。
协议 results/mx4_source_size/PROTOCOL.md。用法：python scripts/44_source_size_curve.py [--max-k 7] [--outdir …]
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")

OUT = ROOT / "results" / "mx4_source_size"
MX3 = _g36.MX3
BUDGETS = tuple(int(b) for b in _g36.BUDGETS)
TRAIN_ONLY = ("SelfRegulationSCP2",)
SH_KS = (6, 3, 3)
K_TOP = 3
TOP_IN = 6


def subsets(sources: Sequence[str], k: int) -> list[tuple[str, ...]]:
    return sorted(itertools.combinations(sorted(sources), int(k)))


def prior_from_sources(cat_z: Sequence[dict], sources: Sequence[str]) -> dict[str, float]:
    src = set(sources)
    return _g36.prior_scores([r for r in cat_z if r["dataset"] in src and r["origin"] == ORIGIN_SHARED])


def eval_prior_on_target(prior: Mapping[str, float], target_rows: Sequence[dict], *,
                         budgets: Sequence[int] = BUDGETS, ks: Sequence[int] = SH_KS,
                         k_top: int = K_TOP, top_in: int = TOP_IN) -> list[dict]:
    """靶上每个 切片×预算 实例的先验读数。SH 只在最后一个预算（同一轨迹三截点）上算。"""
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    out = []
    for s in sorted({r["task_id"] for r in target_rows}):
        rows_s = [r for r in target_rows if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        for b in stages:
            keys = sorted(r["cls_key"] for r in rows_s if int(r["iters"]) == b)
            if not keys:
                continue
            truth = {k: float(traj[k][b]) for k in keys}
            oracle_key = sorted(keys, key=lambda k: (-truth[k], k))[0]
            oracle = truth[oracle_key]
            ranked = _g36.rank_by(prior, keys)
            r1 = truth[ranked[0]]
            r3 = max(truth[k] for k in ranked[:int(k_top)])
            row = {"slice": s, "budget": b, "n_candidates": len(keys), "oracle": oracle, "oracle_key": oracle_key,
                   "regret1": oracle - r1, "regret3": oracle - r3,
                   "in_top6": oracle_key in ranked[:int(top_in)], "regret_sh": None, "sh_pick": None}
            if b == bf and all(all(bb in traj[k] for bb in stages) for k in ranked[:int(ks[0])]):
                path = _g36.sh_path(ranked, traj, ks, stages)
                pick = _g36.sh_pick(path[-1], traj, bf)
                row["regret_sh"] = oracle - truth[pick]
                row["sh_pick"] = pick
            out.append(row)
    return out


def curve(cat_z: Sequence[dict], *, folds: Sequence[str], max_k: Optional[int] = None,
          train_only: Sequence[str] = TRAIN_ONLY, budgets: Sequence[int] = BUDGETS) -> list[dict]:
    """每 (折, k, 子集) 一行：靶上实例均值 + 与全量先验的一致性。"""
    datasets = sorted({r["dataset"] for r in cat_z})
    out = []
    bf = int(budgets[-1])
    for fold in folds:
        train_src = [d for d in datasets if d != fold]
        test = [r for r in cat_z if r["dataset"] == fold]
        full_prior = prior_from_sources(cat_z, train_src)
        full_keys = sorted(full_prior)
        full_top1 = _g36.rank_by(full_prior, full_keys)[0]
        kmax = len(train_src) if max_k is None else min(int(max_k), len(train_src))
        for k in range(1, kmax + 1):
            for sub in subsets(train_src, k):
                prior = prior_from_sources(cat_z, sub)
                inst = eval_prior_on_target(prior, test, budgets=budgets)
                keys = [key for key in full_keys if key in prior]
                rho = (float(spearmanr([prior[x] for x in keys], [full_prior[x] for x in keys]).correlation)
                       if len(keys) >= 3 else None)
                top1 = _g36.rank_by(prior, keys)[0] if keys else None
                row = {"fold": fold, "k": k, "sources": list(sub), "n_instances": len(inst),
                       "top1": top1, "top1_same_as_full": (top1 == full_top1),
                       "spearman_vs_full": (1.0 if k == kmax else rho), "full_top1": full_top1}
                for b in budgets:
                    ib = [r for r in inst if r["budget"] == int(b)]
                    row[f"regret1_{b}"] = float(np.mean([r["regret1"] for r in ib])) if ib else None
                    row[f"regret3_{b}"] = float(np.mean([r["regret3"] for r in ib])) if ib else None
                    row[f"in_top6_{b}"] = float(np.mean([r["in_top6"] for r in ib])) if ib else None
                sh = [r["regret_sh"] for r in inst if r["budget"] == bf and r["regret_sh"] is not None]
                row[f"regret_sh_{bf}"] = float(np.mean(sh)) if sh else None
                out.append(row)
    return out


def summarize(rows: Sequence[dict], *, budgets: Sequence[int] = BUDGETS) -> dict[int, dict]:
    """按 k 汇总：折内对子集取均值再对折取均值；sd_across_subsets = 折内子集间 sd 的折均值。"""
    bf = int(budgets[-1])
    by_k: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_k[int(r["k"])][r["fold"]].append(r)
    out = {}
    for k, folds in sorted(by_k.items()):
        d: dict = {"n_subsets": sum(len(v) for v in folds.values()), "n_folds": len(folds)}
        for name in [f"regret1_{b}" for b in budgets] + [f"regret3_{b}" for b in budgets] + \
                    [f"in_top6_{b}" for b in budgets] + [f"regret_sh_{bf}"]:
            fold_means, fold_sds = [], []
            for v in folds.values():
                vals = [x[name] for x in v if x.get(name) is not None]
                if vals:
                    fold_means.append(float(np.mean(vals)))
                    fold_sds.append(float(np.std(vals)))
            d[f"{name}_mean"] = float(np.mean(fold_means)) if fold_means else None
            d[f"{name}_sd_across_subsets"] = float(np.mean(fold_sds)) if fold_sds else None
            d[f"{name}_per_fold"] = {f: (float(np.mean([x[name] for x in v if x.get(name) is not None]))
                                         if any(x.get(name) is not None for x in v) else None) for f, v in folds.items()}
        d["top1_same_frac"] = float(np.mean([float(np.mean([x["top1_same_as_full"] for x in v])) for v in folds.values()]))
        rhos = [x["spearman_vs_full"] for v in folds.values() for x in v if x["spearman_vs_full"] is not None]
        d["spearman_vs_full_mean"] = float(np.mean(rhos)) if rhos else None
        out[k] = d
    return out


def markdown(summ: Mapping[int, dict], *, budgets: Sequence[int] = BUDGETS) -> str:
    bf = int(budgets[-1])
    lines = ["| k（训练源数） | 子集数 | regret@1 @290 | regret@3 @60 / @120 / @290 | SH6-3-3 @290 | oracle 入前 6 @290 | 子集间 sd(regret@3 @290) | top-1 同全量 | ρ vs 全量 |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---:|"]

    def f(x, nd=4):
        return "—" if x is None else f"{x:.{nd}f}"

    for k, d in sorted(summ.items()):
        lines.append(f"| {k} | {d['n_subsets']} | {f(d[f'regret1_{bf}_mean'])} | "
                     + " / ".join(f(d[f'regret3_{b}_mean']) for b in budgets)
                     + f" | {f(d[f'regret_sh_{bf}_mean'])} | {f(d[f'in_top6_{bf}_mean'], 3)} | {f(d[f'regret3_{bf}_sd_across_subsets'])} | "
                     f"{f(d['top1_same_frac'], 2)} | {f(d['spearman_vs_full_mean'], 2)} |")
    lines.append("")
    lines.append("逐折 regret@3 @290（行 = k）：")
    folds = sorted(next(iter(summ.values()))[f"regret3_{bf}_per_fold"])
    lines.append("| k | " + " | ".join(folds) + " |")
    lines.append("|---:|" + "---:|" * len(folds))
    for k, d in sorted(summ.items()):
        lines.append(f"| {k} | " + " | ".join(f(d[f'regret3_{bf}_per_fold'].get(x)) for x in folds) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args(argv)
    rows = [r for r in read_jsonl(MX3 / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] != "ggs"]
    cat_z = _g36.zscore_within(rows)
    datasets = sorted({r["dataset"] for r in cat_z})
    folds = [d for d in datasets if d not in TRAIN_ONLY]
    print(f"[source-size] 数据集 {datasets}；折 {folds}；行 {len(cat_z)}", flush=True)
    out = curve(cat_z, folds=folds, max_k=args.max_k)
    summ = summarize(out)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "curve_rows.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown(summ)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
