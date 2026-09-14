"""WP16-TCLSC 阶段 1 φ 臂：diff 头 + 12 维 φ ⊕ 预算 one-hot，7 折 LODO，零 GPU（协议 results/tclsc_s1phi/PROTOCOL.md）。

子命令：
  --prepare   变体目录 results/tclsc_s1phi/binned/（56 号 prepare_variant_dir）
  --train     `33 --train --arms <PHI_ARMS> --encoder-mode binned --phi-file <φ>`（S1 7 折 × 3 种子）
  --rank      每折：3 种子多数票排名 normal（自己的 φ）/ cross_s（φ 换成他数据集训练切片的，3 种子）/ zero（φ 置 0）
  --analyze   126 实例 M1/M2/M0/regret + 消融 Δ + 门 G-P1/G-P2/G-P3 + 读数码 + 预测对账 → analysis.json / tables.md

φ 只在本脚本与 33 号装配成浮点向量（铁律 7）；标准化统计取模型 meta（33 号按该折训练切片算，铁律 6）。
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
PHI_COLS = _m33.PHI_COLS
PHI_FILE = _m33.PHI_FILE
zscore_within = _g56.zscore_within
paired_wilcoxon_less = _g56.paired_wilcoxon_less

SOURCE = pq.OUT
OUT = ROOT / "results" / "tclsc_s1phi"
BUDGETS = pq.BUDGETS
FOLDS = tuple(pq.FOLD_TARGETS)
PHI_ARMS = tuple(_m33.PHI_ARMS)
HARD_ARMS = ("blind_diff_hard", "phi_concat_hard", "phi_diff_hard")
CTX_ARMS = tuple(a for a in PHI_ARMS if a.startswith("phi_"))
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
L1_ANALYSIS = ROOT / "results" / "tclsc_l1" / "analysis.json"
# 门（协议 §4）
C0_BAND, C0_REF = 0.02, 0.758
M1_MIN, ALPHA = 0.03, 0.05
PRIMARY = "acc_top10"


# ---------------------------------------------------------------- ctx 行
def phi_ctx_rows(phi: Mapping[str, Mapping], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                 cross_map: Optional[Mapping[str, str]] = None, zero: bool = False,
                 budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    """{(切片, 预算): 标准化 φ(供体或自己) ⊕ 预算 one-hot}；zero=True 时 φ 部分全 0。"""
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    out = {}
    for s in slices:
        src = cross_map[s] if cross_map is not None else s
        if zero:
            x = np.zeros(len(PHI_COLS), dtype=np.float64)
        else:
            x = (np.array([float(phi[src][c]) for c in PHI_COLS], dtype=np.float64) - mu) / sd
        for b in budgets:
            out[(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


def cross_slice_map(test_slices: Sequence[str], train_slices: Sequence[str], seed: int) -> dict[str, str]:
    """每个考侧切片 → 一个**别的数据集**的训练切片（供体），按 seed 逐位确定（同 pq.cross_assignment_slices 的语义）。"""
    by_ds: dict[str, list[str]] = {}
    for s in sorted(set(train_slices)):
        by_ds.setdefault(pq.dataset_of(s), []).append(s)
    rng = np.random.default_rng([seed, 202609])
    out = {}
    for s in sorted(test_slices):
        others = [x for d, sl in sorted(by_ds.items()) if d != pq.dataset_of(s) for x in sl]
        if not others:
            raise ValueError(f"{s} 没有跨数据集供体")
        out[s] = others[int(rng.integers(len(others)))]
    return out


# ---------------------------------------------------------------- 门 / 读数码
def gates_s1phi(deltas: Mapping[str, Mapping[str, Mapping]], value: Mapping) -> dict:
    """deltas[臂][指标] = {mean_delta, p}；value = {a: blind, b: phi, mean_a, mean_b, p}（regret@3 配对，H1 b<a）。"""
    def pos(arm):
        return _g57._positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)

    gp2 = bool(value.get("p") is not None and value["p"] < ALPHA and value["mean_b"] < value["mean_a"])
    return {"GP1": {"arm": "phi_diff_hard", "delta": deltas.get("phi_diff_hard", {}).get(PRIMARY), "pass": pos("phi_diff_hard")},
            "GP2": {"metric": "regret3", "blind": value.get("mean_a"), "phi": value.get("mean_b"), "p": value.get("p"),
                    "pass": gp2},
            "GP3": {"arm": "phi_concat_hard", "delta": deltas.get("phi_concat_hard", {}).get(PRIMARY),
                    "pass": not pos("phi_concat_hard")},
            "thresholds": {"metric": PRIMARY, "min_delta": M1_MIN, "alpha": ALPHA}}


def verdict_s1phi(g: Mapping) -> str:
    if not g["GP1"]["pass"]:
        return "PHI-DEAD"
    return "PHI-ALIVE" if g["GP2"]["pass"] else "PHI-USED-NO-GAIN"


# ---------------------------------------------------------------- 子命令
def cmd_train(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], device: str,
              epochs: Optional[int], resume: bool, phi_file: Path) -> None:
    d = outdir / "binned"
    _g56.prepare_variant_dir(source, d)
    _m33.cmd_train(d, arms=list(arms), folds=None, seeds=list(seeds), resume=resume, device=device,
                   epochs=epochs, settf_windows=None, encoder_mode="binned", phi_file=phi_file)


def _load_arm_models(d: Path, target: str, arm: str, seeds: Sequence[int], device: str) -> Optional[list]:
    paths = [d / "models" / target / f"{arm}_s{s}.pt" for s in seeds]
    have = [p for p in paths if p.exists()]
    if len(have) != len(paths):
        print(f"⚠️ [{target}/{arm}] 模型 {len(have)}/{len(paths)}，整臂跳过", flush=True)
        return None
    return [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=BUDGETS) for p in have]


def cmd_rank(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], folds: Sequence[str],
             phi_file: Path, device: str = "cpu") -> None:
    d = outdir / "binned"
    phi = {r["task_id"]: r for r in _g56.read_jsonl(phi_file)}
    rows, _delta = _g57.load_source(source)
    all_slices = sorted({r["task_id"] for r in rows})
    out: dict[str, dict] = {}
    for target in folds:
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        train_slices = [s for s in all_slices if pq.dataset_of(s) != target]
        catalog, _g, _k = _g34.load_catalog(source, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        models = {a: m for a in arms if (m := _load_arm_models(d, target, a, seeds, device)) is not None}
        if not models:
            continue
        blind = {a: m for a, m in models.items() if a.startswith("blind_")}
        ctxa = {a: m for a, m in models.items() if a not in blind}
        per: dict[str, dict] = {"normal": {}, "cross": {}, "zero": {}}
        if blind:
            per["normal"] = _g56.rank_local_models(blind, cat, budgets=BUDGETS, device=device)
        if ctxa:
            # 标准化统计取 meta（33 号按该折训练切片算）；同臂各种子相同，跨臂也相同（同折同训练切片）
            meta0 = next(iter(ctxa.values()))[0][1]
            mu, sd = np.array(meta0["phi_mu"]), np.array(meta0["phi_sd"])
            for a, mm in ctxa.items():
                for _m, meta in mm:
                    if not (np.allclose(meta["phi_mu"], mu) and np.allclose(meta["phi_sd"], sd)):
                        raise ValueError(f"[{target}/{a}] 各臂/种子的 φ 标准化统计不一致")

            def merge(dst: dict, rk: dict) -> None:
                for b, pb in rk.items():
                    for s, pa in pb.items():
                        dst.setdefault(b, {}).setdefault(s, {}).update(pa)

            merge(per["normal"], _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device,
                                                        ctx_rows=phi_ctx_rows(phi, test_slices, mu, sd)))
            per["zero"] = _g56.rank_local_models(ctxa, cat, budgets=BUDGETS, device=device,
                                                 ctx_rows=phi_ctx_rows(phi, test_slices, mu, sd, zero=True))
            for cs in CROSS_SEEDS:
                cmap = cross_slice_map(test_slices, train_slices, cs)
                per["cross"][str(cs)] = {"map": cmap, "rankings": _g56.rank_local_models(
                    ctxa, cat, budgets=BUDGETS, device=device,
                    ctx_rows=phi_ctx_rows(phi, test_slices, mu, sd, cross_map=cmap))}
        out[target] = per
        print(f"[{target}] {len(test_slices)} 切片 × {len(BUDGETS)} 预算，臂 {sorted(models)}", flush=True)
    (d / "rankings_s1phi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {d / 'rankings_s1phi.json'}（{len(out)} 折）", flush=True)


def _eval_all(rk_by_fold: Mapping[str, Mapping], cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str],
              which: str, cross_seed: Optional[str] = None) -> list[dict]:
    rows = []
    for target, per in rk_by_fold.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        fr = _g57.eval_rankings_l3(rk, [r for r in cat_z if r["dataset"] == target], delta, arms=arms)
        for r in fr:
            r["fold"] = target
        rows += fr
    return rows


def paired_regret(rows: Sequence[dict], a: str, b: str, key: str = "3") -> dict:
    """regret@k 配对 Wilcoxon，H1：b 的 regret 更低。"""
    xs = [(r["regret"][a + key], r["regret"][b + key]) for r in rows if a + key in r["regret"] and b + key in r["regret"]]
    if not xs:
        return {"a": a, "b": b, "metric": "regret" + key, "n": 0, "p": None, "mean_a": None, "mean_b": None}
    x, y = np.array([v[0] for v in xs]), np.array([v[1] for v in xs])
    return {"a": a, "b": b, "metric": "regret" + key, "n": int(x.size), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "p": float(paired_wilcoxon_less(y, x)), "wins_b": int(np.sum(y < x)), "losses_b": int(np.sum(y > x)),
            "ties": int(np.sum(y == x))}


def predictions(g: Mapping, c0: Mapping, by_fold: Mapping, pooled: Mapping) -> dict:
    d_fold = by_fold.get("phi_diff_hard", {})
    n_pos = sum(1 for v in d_fold.values() if v is not None and v > 0)
    out = {"P1": {"desc": "C0 带内", "hit": c0.get("pass")},
           "P2": {"desc": "G-P1 过", "hit": g["GP1"]["pass"]},
           "P3": {"desc": "G-P2 过", "hit": g["GP2"]["pass"]},
           "P4": {"desc": "G-P3 零（concat 仍零）", "hit": g["GP3"]["pass"]},
           "P5": {"desc": "逐折 Δ_M1 > 0 的折 ≥ 5/7（若 G-P1 过）", "value": n_pos,
                  "hit": None if not g["GP1"]["pass"] else bool(n_pos >= 5)}}
    z, h = pooled.get("phi_diff_zdiff", {}).get("regret3"), pooled.get("phi_diff_hard", {}).get("regret3")
    out["P6"] = {"desc": "phi_diff_zdiff regret@3 < phi_diff_hard（若跑）", "value": [z, h],
                 "hit": None if z is None or h is None else bool(z < h)}
    return out


def markdown_tables(a: dict) -> str:
    arms = a["arms"]
    p = a["pooled"]
    f = lambda b, k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
    L = [f"# WP16-TCLSC 阶段 1 φ 臂 tables（{a['n_instances']} 实例，7 折 LODO）", "", f"读数码 **{a['verdict']}**", "",
         "## normal", "", "| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        b = p[arm]
        L.append(f"| {arm} | {f(b,'acc_top10')} | {f(b,'wacc')} | {f(b,'pair_acc')} | {f(b,'rho')} | {f(b,'regret1')} | {f(b,'regret3')} |")
    for name, ref in a["refs"].items():
        L.append(f"| ref {name} | {f(ref,'acc_top10')} | — | {f(ref,'pair_acc')} | — | {f(ref,'regret1')} | {f(ref,'regret3')} |")
    L += ["", "## 消融 Δ（normal − cross 3 种子均值 / normal − zero；Wilcoxon H1 normal 更高）", "",
          "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
    for arm in arms:
        for kind in ("cross", "zero"):
            for metric in ("acc_top10", "wacc", "pair_acc"):
                d = a["deltas"].get(kind, {}).get(arm, {}).get(metric)
                if not d or d.get("mean_delta") is None:
                    continue
                L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | "
                         f"{d['wins']}/{d['losses']}/{d['ties']} |")
    L += ["", "## 逐折 Δ_M1（cross）", "", "| 折 | " + " | ".join(a["by_fold_delta"]) + " |", "|---|" + "---:|" * len(a["by_fold_delta"])]
    folds = sorted({f for v in a["by_fold_delta"].values() for f in v})
    for fo in folds:
        L.append(f"| {fo} | " + " | ".join(("—" if a["by_fold_delta"][arm].get(fo) is None else f"{a['by_fold_delta'][arm][fo]:+.3f}")
                                          for arm in a["by_fold_delta"]) + " |")
    L += ["", "## 价值门（regret@3 配对，H1 φ 臂更低）", ""]
    for t in a["value_tests"]:
        L.append(f"- {t['a']} → {t['b']}: n={t['n']} {t['mean_a']} → {t['mean_b']} p={t['p']} 胜/负/平 "
                 f"{t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    L += ["", "## 预测对账", ""]
    for k, v in a["predictions"].items():
        L.append(f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else ""))
    return "\n".join(L) + "\n"


def cmd_analyze(source: Path, outdir: Path, *, arms: Sequence[str]) -> dict:
    d = outdir / "binned"
    rk = json.loads((d / "rankings_s1phi.json").read_text(encoding="utf-8"))
    rows, delta = _g57.load_source(source)
    cat_z = zscore_within(rows)
    present = [a for a in arms if any(a in pa for per in rk.values() for pb in per["normal"].values() for pa in pb.values())]
    normal = _eval_all(rk, cat_z, delta, present, "normal")
    ctx_present = [a for a in present if a in CTX_ARMS]
    cross = [_eval_all(rk, cat_z, delta, ctx_present, "cross", cs) for cs in sorted({cs for per in rk.values() for cs in per["cross"]})]
    zero = _eval_all(rk, cat_z, delta, ctx_present, "zero")
    pooled = {}
    for a in present:
        pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
        pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
        pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
    deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")}
                        for a in ctx_present},
              "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")}
                       for a in ctx_present}}
    by_fold_delta = {}
    for a in ctx_present:
        by_fold_delta[a] = {}
        for fo in sorted({r["fold"] for r in normal}):
            dd = _g57.ablation_delta([r for r in normal if r["fold"] == fo], [[r for r in c if r["fold"] == fo] for c in cross],
                                     a, PRIMARY)
            by_fold_delta[a][fo] = dd.get("mean_delta")
    value_tests = [paired_regret(normal, "blind_diff_hard", b) for b in ctx_present if "blind_diff_hard" in present]
    value_main = next((t for t in value_tests if t["b"] == "phi_diff_hard"), {"p": None, "mean_a": None, "mean_b": None})
    g = gates_s1phi(deltas["cross"], value_main)
    refs = {}
    if L1_ANALYSIS.exists():
        l1 = json.loads(L1_ANALYSIS.read_text(encoding="utf-8"))["summary"]["pooled"]
        refs["L1 cls_only_binned"] = l1.get("cls_only_binned", {})
    c0_val = pooled.get("blind_diff_hard", {}).get("pair_acc")
    c0 = {"blind_diff_hard": c0_val, "ref": C0_REF, "band": C0_BAND,
          "pass": None if c0_val is None else bool(abs(c0_val - C0_REF) <= C0_BAND)}
    a = {"protocol": "results/tclsc_s1phi/PROTOCOL.md", "split": "S1 LODO", "n_instances": len(normal), "arms": present,
         "pooled": pooled, "refs": refs, "deltas": deltas, "by_fold_delta": by_fold_delta, "value_tests": value_tests,
         "gates": g, "verdict": verdict_s1phi(g), "c0": c0, "per_instance_normal": normal}
    a["predictions"] = predictions(g, c0, by_fold_delta, pooled)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']} | " + " | ".join(
        f"{arm} M1 {pooled[arm]['acc_top10']:.3f} r3 {pooled[arm]['regret3']:.3f}"
        + (f" Δ {deltas['cross'][arm]['acc_top10']['mean_delta']:+.3f}" if arm in deltas["cross"] else "") for arm in present),
        flush=True)
    return a


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(HARD_ARMS), choices=list(PHI_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--folds", nargs="+", default=list(FOLDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--phi-file", type=Path, default=PHI_FILE)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.prepare, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --prepare / --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        print(f"[prepare] {_g56.prepare_variant_dir(args.source, args.outdir / 'binned') or '已就位'}", flush=True)
    if args.train:
        cmd_train(args.source, args.outdir, arms=args.arms, seeds=args.seeds, device=args.device, epochs=args.epochs,
                  resume=args.resume, phi_file=args.phi_file)
    if args.rank:
        cmd_rank(args.source, args.outdir, arms=list(PHI_ARMS), seeds=args.seeds, folds=args.folds, phi_file=args.phi_file)
    if args.analyze:
        cmd_analyze(args.source, args.outdir, arms=list(PHI_ARMS))


if __name__ == "__main__":
    main()
