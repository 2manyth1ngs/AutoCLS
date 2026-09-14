"""WP16-TCLSC-L2：CLS 流水线图 + GIN，L1 的 7 折 LODO cls_only 考场（协议 results/tclsc_l2/PROTOCOL.md）。

  --prepare   每臂一个变体目录 results/tclsc_l2/<arm>/（56 号 prepare_variant_dir）
  --train     `33 --train --arms cls_only --encoder-mode graph --graph-*`（3 种子）
  --rank      每臂 rankings.json（56 号 rank_local_models，打包器按 meta.graph_opts 重建）
  --analyze   与 L1 同口径（126 实例 rank→score）：主表 + 逐折 + 门 G-L2a/b + 读数码 + 预测对账；ref = L1 cls_only_binned
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
_g56 = importlib.import_module("scripts.56_tclsc_l1")

SOURCE = pq.OUT
OUT = ROOT / "results" / "tclsc_l2"
L1_DIR = ROOT / "results" / "tclsc_l1"
BUDGETS = pq.BUDGETS
FOLDS = tuple(pq.FOLD_TARGETS)
SEEDS = (0, 1, 2)
ARMS: dict[str, dict] = {
    "gin_pipe_hub": {"topology": "pipe", "readout": "hub", "scalar": True, "gin_layers": 2},
    "gin_pipe_mean": {"topology": "pipe", "readout": "mean", "scalar": True, "gin_layers": 2},
    "gin_full_mean": {"topology": "full", "readout": "mean", "scalar": True, "gin_layers": 2},
    "gin_none_mean": {"topology": "none", "readout": "mean", "scalar": True, "gin_layers": 2},
    "gin_pipe_hub_noscalar": {"topology": "pipe", "readout": "hub", "scalar": False, "gin_layers": 2},
}
REF = "mlp_binned"
# 门（协议 §4）
C0_REF, C0_BAND = 0.758, 0.03
GA_MIN, GA_FOLDS = 0.020, 5
GB_MIN, GB_FOLDS = 0.010, 5


def gates_l2(pooled: Mapping[str, Mapping[str, float]], by_fold: Mapping[str, Mapping[str, float]]) -> dict:
    def acc(a):
        v = pooled.get(a, {}).get("pair_acc")
        return None if v is None or not np.isfinite(v) else float(v)

    def n_better(a, b):
        return sum(1 for d in by_fold.values() if a in d and b in d and d[a] > d[b])

    pa, mb, fu = acc("gin_pipe_hub"), acc(REF), acc("gin_full_mean")
    da = None if pa is None or mb is None else pa - mb
    db = None if pa is None or fu is None else pa - fu
    na, nb = n_better("gin_pipe_hub", REF), n_better("gin_pipe_hub", "gin_full_mean")
    return {"G_L2a": {"delta": da, "n_folds_better": na, "min": GA_MIN, "min_folds": GA_FOLDS,
                      "pass": bool(da is not None and da >= GA_MIN and na >= GA_FOLDS)},
            "G_L2b": {"delta": db, "n_folds_better": nb, "min": GB_MIN, "min_folds": GB_FOLDS,
                      "pass": bool(db is not None and db >= GB_MIN and nb >= GB_FOLDS)}}


def verdict_l2(g: Mapping) -> str:
    if not g["G_L2a"]["pass"]:
        return "L2-FLAT"
    return "L2-GRAPH-WIN" if g["G_L2b"]["pass"] else "L2-EMB-ONLY"


def cmd_train(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str,
              epochs: Optional[int], resume: bool) -> None:
    for arm in arms:
        d = outdir / arm
        _g56.prepare_variant_dir(source, d)
        print(f"===== [{arm}] 33 --train --arms cls_only --encoder-mode graph {ARMS[arm]} =====", flush=True)
        _m33.cmd_train(d, arms=["cls_only"], folds=None, seeds=list(seeds), resume=resume, device=device,
                       epochs=epochs, settf_windows=None, encoder_mode="graph", graph_opts=ARMS[arm])


def cmd_rank(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str = "cpu") -> None:
    for arm in arms:
        d = outdir / arm
        rankings: dict[str, dict] = {}
        for target in FOLDS:
            paths = [d / "models" / target / f"cls_only_s{s}.pt" for s in seeds]
            if not all(p.exists() for p in paths):
                print(f"⚠️ [{arm}/{target}] 模型不全，跳过", flush=True)
                continue
            g34 = importlib.import_module("scripts.34_pq_search")
            models = {"cls_only": [g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=BUDGETS) for p in paths]}
            catalog, _g, _k = g34.load_catalog(source, _g56.space(), slices=pq.pq_slice_ids(target))
            cat = {s: catalog[s] for s in pq.pq_slice_ids(target) if s in catalog}
            rankings[target] = _g56.rank_local_models(models, cat, budgets=BUDGETS, device=device)
        (d / "rankings.json").write_text(json.dumps(rankings, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{arm}] → {d / 'rankings.json'}（{len(rankings)} 靶）", flush=True)


def cmd_analyze(source: Path, outdir: Path, *, arms: Sequence[str]) -> dict:
    rows, delta, _pod = _g56.load_source(source)
    cat_z = _g56.zscore_within(rows)
    groups = []
    l1 = L1_DIR / "binned" / "rankings.json"
    if l1.exists():
        groups.append(_g56.eval_rankings(json.loads(l1.read_text(encoding="utf-8")), cat_z, delta, arms=["cls_only"],
                                         rename={"cls_only": REF}))
    for arm in arms:
        rp = outdir / arm / "rankings.json"
        if rp.exists():
            groups.append(_g56.eval_rankings(json.loads(rp.read_text(encoding="utf-8")), cat_z, delta, arms=["cls_only"],
                                             rename={"cls_only": arm}))
        else:
            print(f"⚠️ 缺 {rp}", flush=True)
    inst = _g56.merge_rows(*groups)
    summary = _g56.summarize(inst)
    g = gates_l2(summary["pooled"], summary["by_fold_pair_acc"])
    tests = [_g56.paired_test(inst, REF, a) for a in arms if a in summary["arms"]]
    tests += [_g56.paired_test(inst, "gin_full_mean", "gin_pipe_hub"), _g56.paired_test(inst, "gin_pipe_mean", "gin_pipe_hub"),
              _g56.paired_test(inst, "gin_pipe_hub_noscalar", "gin_pipe_hub"), _g56.paired_test(inst, "gin_none_mean", "gin_pipe_hub")]
    c0_ref = summary["pooled"].get(REF, {}).get("pair_acc")
    c0_none = summary["pooled"].get("gin_none_mean", {}).get("pair_acc")
    c0 = {"mlp_binned": c0_ref, "expected": C0_REF, "gin_none_mean": c0_none,
          "pass": bool(c0_ref is not None and abs(c0_ref - C0_REF) <= 1e-3 and c0_none is not None and abs(c0_none - C0_REF) <= C0_BAND)}
    a = {"protocol": "results/tclsc_l2/PROTOCOL.md", "n_instances": len(inst), "summary": summary, "gates": g, "verdict": verdict_l2(g),
         "tests": tests, "c0": c0, "per_instance": inst}
    a["predictions"] = predictions(summary, g, c0)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']} | " + " | ".join(f"{arm} {summary['pooled'][arm]['pair_acc']:.3f}/{summary['pooled'][arm]['acc_top10']:.3f}"
                                                   for arm in summary["arms"]), flush=True)
    return a


def predictions(summary: dict, g: dict, c0: dict) -> dict:
    p = summary["pooled"]
    acc = lambda a, k="pair_acc": p.get(a, {}).get(k)
    hub, mean = acc("gin_pipe_hub"), acc("gin_pipe_mean")
    sc, nosc = acc("gin_pipe_hub"), acc("gin_pipe_hub_noscalar")
    top = max((p[a]["acc_top10"] for a in summary["arms"] if a != REF and p[a].get("acc_top10") is not None), default=None)
    return {"P1": {"desc": "C0 过", "hit": c0["pass"]},
            "P2": {"desc": "G-L2a 过", "hit": g["G_L2a"]["pass"]},
            "P3": {"desc": "G-L2b 过（条件于 P2）", "hit": None if not g["G_L2a"]["pass"] else g["G_L2b"]["pass"]},
            "P4": {"desc": "hub ≥ mean（pipe）", "value": [hub, mean], "hit": None if hub is None or mean is None else bool(hub >= mean)},
            "P5": {"desc": "标量序数边际 ≥ +0.005", "value": None if sc is None or nosc is None else sc - nosc,
                   "hit": None if sc is None or nosc is None else bool(sc - nosc >= 0.005)},
            "P6": {"desc": "任一 GIN 臂 top-10 acc ≥ 0.55", "value": top, "hit": None if top is None else bool(top >= 0.55)}}


def markdown_tables(a: dict) -> str:
    p = a["summary"]["pooled"]
    arms = a["summary"]["arms"]
    L = [f"# WP16-TCLSC-L2 tables（{a['n_instances']} 实例，7 折 LODO cls_only）", "", f"读数码 **{a['verdict']}**", "",
         "| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        b = p[arm]
        L.append(f"| {arm} | {b['pair_acc']:.3f} | {b['acc_top10']:.3f} | {b['rho']:.3f} | {b['regret1']:.3f} | {b['regret3']:.3f} |")
    L += ["", "## 逐折配对 acc", "", "| 折 | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for f, d in a["summary"]["by_fold_pair_acc"].items():
        L.append(f"| {f} | " + " | ".join(f"{d[x]:.3f}" if x in d else "—" for x in arms) + " |")
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    L += ["", f"C0: {json.dumps(a['c0'], ensure_ascii=False)}", "", "## 配对检验（Wilcoxon，H1：后者更高）", ""]
    for t in a["tests"]:
        L.append(f"- {t['a']} → {t['b']}: n={t['n']} p={t.get('p')} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')} Δ̄={t.get('mean_delta')}")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
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
        for arm in args.arms:
            print(f"[prepare/{arm}] {_g56.prepare_variant_dir(args.source, args.outdir / arm) or '已就位'}", flush=True)
    if args.train:
        cmd_train(args.source, args.outdir, arms=args.arms, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.source, args.outdir, arms=args.arms, seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.source, args.outdir, arms=list(ARMS))


if __name__ == "__main__":
    main()
