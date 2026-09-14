"""WP16-TCLSC 阶段 1 真 E 臂：diff 头 + SetTF 任务嵌入，7 折 LODO（协议 results/tclsc_s1e/PROTOCOL.md）。

  --rank      （pod 上，需 folds/<target>/task_features/*.npz）每折：3 种子各自的 E → 多数票排名 normal / cross_s（他数据集训练切片窗口
              经同一 SetTF）/ zero（E 置 0）+ seed 0 的 E 诊断（跨集 / 集内余弦、范数）→ rankings_s1e.json
  --analyze   （本地）126 实例 M1/M2/M0/regret + 消融 Δ + φ 阶段 blind_diff_hard 逐实例基线 + 33 号 run json 的 δ 对消融 + 门 + 读数码

E 只是浮点向量（铁律 7）；训练侧折内 TS2Vec 只 fit 源池（33 号 --fit-folds，铁律 6）。
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

SOURCE = pq.OUT
OUT = ROOT / "results" / "tclsc_s1e"
BUDGETS = pq.BUDGETS
FOLDS = tuple(pq.FOLD_TARGETS)
E_ARMS = tuple(_m33.E_ARMS)
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
PHI_ANALYSIS = ROOT / "results" / "tclsc_s1phi" / "analysis.json"
BLIND = "blind_diff_hard"
# 门（协议 §4）
C0_BAND = (0.70, 0.80)
COS_CROSS_MAX, COS_GAP_MIN = 0.99, 0.01
M1_MIN, ALPHA = 0.03, 0.05
PRIMARY = "acc_top10"


# ---------------------------------------------------------------- E 行 / 诊断
def e_rows(model: torch.nn.Module, windows: Mapping[str, torch.Tensor], slices: Sequence[str], *,
           cross_map: Optional[Mapping[str, str]] = None, zero: bool = False) -> dict[str, torch.Tensor]:
    """{切片: E}——按模型自己的 SetTF 编码；cross 取供体切片的窗口；zero 置 0。"""
    out = {}
    with torch.no_grad():
        for s in slices:
            src = cross_map[s] if cross_map is not None else s
            e = model.encode_tasks([windows[src]])[0]
            out[s] = torch.zeros_like(e) if zero else e
    return out


def e_diagnostics(E: Mapping[str, np.ndarray]) -> dict:
    """实验 1 F1 同式：跨数据集 / 数据集内平均余弦、范数中位。"""
    keys = sorted(E)
    X = np.stack([np.asarray(E[k], np.float64) for k in keys])
    norms = np.linalg.norm(X, axis=1)
    Xn = X / np.maximum(norms[:, None], 1e-12)
    C = Xn @ Xn.T
    ds = [pq.dataset_of(k) for k in keys]
    within, cross = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (within if ds[i] == ds[j] else cross).append(C[i, j])
    return {"n_slices": len(keys), "cos_within": float(np.mean(within)) if within else None,
            "cos_cross": float(np.mean(cross)) if cross else None, "norm_median": float(np.median(norms)),
            "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def rank_target(models: Sequence[tuple], catalog: Mapping[str, Sequence[tuple]], windows: Mapping[str, torch.Tensor], *,
                train_slices: Sequence[str], budgets: Sequence[int] = BUDGETS, cross_seeds: Sequence[int] = CROSS_SEEDS,
                device: str = "cpu", arm: str = "arm") -> dict:
    """一个折靶：{normal, cross{s: {map, rankings}}, zero, diag}，排名结构 {预算: {切片: {arm: [key…]}}}（34 号 rank_candidates 多数票）。"""
    test = sorted(catalog)
    enc = _g56.encoder_for_meta(models[0][1])
    vec_cache = {s: torch.as_tensor(np.stack([enc.encode(c) for _o, c, _k in catalog[s]]), dtype=torch.float32, device=device) for s in test}
    mm = [m for m, _ in models]
    for m in mm:
        m.eval()

    def rank_with(e_by_model: Sequence[Mapping[str, torch.Tensor]]) -> dict:
        out: dict[str, dict] = {}
        for b in budgets:
            ctx_row = torch.as_tensor(np.asarray(pq.ctx_vector(int(b), budgets)), dtype=torch.float32, device=device)
            per_b = {}
            for s in test:
                keys = [k for _o, _c, k in catalog[s]]
                ranked, _S = _g34.rank_candidates(mm, [e[s] for e in e_by_model], vec_cache[s], keys, ctx_row=ctx_row)
                per_b[s] = {arm: ranked}
            out[str(int(b))] = per_b
        return out

    normal_E = [e_rows(m, windows, test) for m in mm]
    zero_E = [e_rows(m, windows, test, zero=True) for m in mm]
    res = {"normal": rank_with(normal_E), "zero": rank_with(zero_E), "cross": {}}
    for cs in cross_seeds:
        cmap = _g58.cross_slice_map(test, train_slices, cs)
        res["cross"][str(cs)] = {"map": cmap, "rankings": rank_with([e_rows(m, windows, test, cross_map=cmap) for m in mm])}
    all_slices = sorted(set(test) | set(train_slices))
    E0 = e_rows(mm[0], windows, [s for s in all_slices if s in windows])
    res["diag"] = e_diagnostics({k: v.cpu().numpy() for k, v in E0.items()})
    return res


# ---------------------------------------------------------------- 门 / 读数码
def gates_e(diag: Mapping, deltas: Mapping, value: Mapping) -> dict:
    cc, cw = diag.get("cos_cross"), diag.get("cos_within")
    ge0 = bool(cc is not None and cw is not None and cc < COS_CROSS_MAX and (cw - cc) >= COS_GAP_MIN)
    pos = lambda arm: _g57._positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)
    ge2 = bool(value.get("p") is not None and value["p"] < ALPHA and value["mean_b"] < value["mean_a"])
    return {"GE0": {"cos_cross": cc, "cos_within": cw, "pass": ge0},
            "GE1": {"arm": "e_diff_hard", "delta": deltas.get("e_diff_hard", {}).get(PRIMARY), "pass": pos("e_diff_hard")},
            "GE2": {"metric": "regret3", "blind": value.get("mean_a"), "e": value.get("mean_b"), "p": value.get("p"), "pass": ge2},
            "GE3": {"arm": "e_concat_hard", "delta": deltas.get("e_concat_hard", {}).get(PRIMARY), "pass": not pos("e_concat_hard")}}


def verdict_e(g: Mapping) -> str:
    if not g["GE0"]["pass"]:
        return "E-COLLAPSED"
    if not g["GE1"]["pass"]:
        return "E-INERT"
    return "E-ALIVE" if g["GE2"]["pass"] else "E-USED-NO-GAIN"


# ---------------------------------------------------------------- 子命令
def cmd_rank(outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], folds: Sequence[str], device: str) -> None:
    """在含折特征的目录（pod 上的变体目录 = outdir）里排名。"""
    space = _g56.space()
    out: dict[str, dict] = {}
    for target in folds:
        catalog, _g, _k = _g34.load_catalog(outdir, space, slices=pq.pq_slice_ids(target))
        test = [s for s in pq.pq_slice_ids(target) if s in catalog]
        feat_dir = outdir / "folds" / target / "task_features"
        all_slices = sorted(p.stem for p in feat_dir.glob("*.npz"))
        train_slices = [s for s in all_slices if pq.dataset_of(s) != target]
        windows = {}
        for s in all_slices:
            w = _g34.load_slice_windows(outdir, target, s, device=device)
            if w is not None:
                windows[s] = w
        per: dict[str, dict] = {"test": test, "normal": {}, "zero": {}, "cross": {}, "diag": {}}
        for arm in arms:
            paths = [outdir / "models" / target / f"{arm}_s{s}.pt" for s in seeds]
            if not all(p.exists() for p in paths):
                print(f"⚠️ [{target}/{arm}] 模型不全，跳过", flush=True)
                continue
            models = [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=BUDGETS) for p in paths]
            for m, _meta in models:
                m.to(device)
            cat = {s: catalog[s] for s in test if s in windows}
            r = rank_target(models, cat, windows, train_slices=[s for s in train_slices if s in windows], device=device, arm=arm)

            def merge(dst, rk):
                for b, pb in rk.items():
                    for s, pa in pb.items():
                        dst.setdefault(b, {}).setdefault(s, {}).update(pa)

            merge(per["normal"], r["normal"])
            merge(per["zero"], r["zero"])
            for cs, v in r["cross"].items():
                per["cross"].setdefault(cs, {"map": v["map"], "rankings": {}})
                merge(per["cross"][cs]["rankings"], v["rankings"])
            per["diag"][arm] = r["diag"]
            print(f"[{target}/{arm}] 切片 {len(cat)} | E cos cross {r['diag']['cos_cross']:.4f} within {r['diag']['cos_within']:.4f} "
                  f"norm {r['diag']['norm_median']:.3f}", flush=True)
        out[target] = per
    (outdir / "rankings_s1e.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {outdir / 'rankings_s1e.json'}（{len(out)} 折）", flush=True)


def _run_json_cross(d: Path, arms: Sequence[str], seeds: Sequence[int]) -> dict:
    out = {}
    for arm in arms:
        per = {}
        for t in FOLDS:
            vals = []
            for s in seeds:
                p = d / "runs" / t / f"{arm}_s{s}.json"
                if p.exists():
                    r = json.loads(p.read_text(encoding="utf-8"))
                    if r.get("cross"):
                        vals.append((float(r["cross"]["mean_delta"]), r["cross"].get("tier"), float(r["normal"]["acc"])))
            if vals:
                per[t] = {"cross_delta": float(np.mean([v[0] for v in vals])), "tiers": sorted({v[1] for v in vals if v[1]}),
                          "normal_acc": float(np.mean([v[2] for v in vals]))}
        out[arm] = per
    return out


def cmd_analyze(source: Path, outdir: Path, *, arms: Sequence[str], seeds: Sequence[int]) -> dict:
    rk = json.loads((outdir / "rankings_s1e.json").read_text(encoding="utf-8"))
    rows, delta = _g57.load_source(source)
    cat_z = _g56.zscore_within(rows)
    present = [a for a in arms if any(a in pa for per in rk.values() for pb in per["normal"].values() for pa in pb.values())]
    cross_seeds = sorted({cs for per in rk.values() for cs in per["cross"]})
    normal = _g58._eval_all(rk, cat_z, delta, present, "normal")
    cross = [_g58._eval_all(rk, cat_z, delta, present, "cross", cs) for cs in cross_seeds]
    zero = _g58._eval_all(rk, cat_z, delta, present, "zero")
    # φ 阶段的任务盲 diff 头逐实例行并进来（同口径 126 实例）
    if PHI_ANALYSIS.exists():
        phi = json.loads(PHI_ANALYSIS.read_text(encoding="utf-8"))
        blind_rows = [{**r, "metrics": {BLIND: r["metrics"][BLIND]}, "regret": {k: v for k, v in r["regret"].items() if k.startswith(BLIND) or k in ("oracle", "random1", "random3")}}
                      for r in phi["per_instance_normal"] if BLIND in r["metrics"]]
        normal = _g56.merge_rows(normal, blind_rows)
    arms_all = present + ([BLIND] if PHI_ANALYSIS.exists() else [])
    pooled = {}
    for a in arms_all:
        pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
        pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
        pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
    deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in present},
              "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in present}}
    by_fold = {a: {fo: _g57.ablation_delta([r for r in normal if r["fold"] == fo], [[r for r in c if r["fold"] == fo] for c in cross], a, PRIMARY).get("mean_delta")
                   for fo in sorted({r["fold"] for r in normal})} for a in present}
    value = [_g58.paired_regret(normal, BLIND, b) for b in present if BLIND in arms_all]
    v_main = next((t for t in value if t["b"] == "e_diff_hard"), {"p": None, "mean_a": None, "mean_b": None})
    diag_folds = {t: per.get("diag", {}).get("e_diff_hard") for t, per in rk.items()}
    diag_vals = [d for d in diag_folds.values() if d]
    diag = {"cos_cross": float(np.median([d["cos_cross"] for d in diag_vals])) if diag_vals else None,
            "cos_within": float(np.median([d["cos_within"] for d in diag_vals])) if diag_vals else None,
            "norm_median": float(np.median([d["norm_median"] for d in diag_vals])) if diag_vals else None, "by_fold": diag_folds,
            "concat_by_fold": {t: per.get("diag", {}).get("e_concat_hard") for t, per in rk.items()}}
    g = gates_e(diag, deltas["cross"], v_main)
    run_json = _run_json_cross(outdir, present, seeds)
    c0_m0 = pooled.get("e_concat_hard", {}).get("pair_acc")
    c0_tiers = [v["tiers"] for v in run_json.get("e_concat_hard", {}).values()]
    c0 = {"concat_M0": c0_m0, "concat_cross_tiers": c0_tiers,
          "pass": bool(c0_m0 is not None and C0_BAND[0] <= c0_m0 <= C0_BAND[1] and all(t == ["zero"] for t in c0_tiers if t))}
    a = {"protocol": "results/tclsc_s1e/PROTOCOL.md", "n_instances": len(normal), "arms": arms_all, "pooled": pooled, "deltas": deltas,
         "by_fold_delta": by_fold, "value_tests": value, "diag": diag, "run_json_cross": run_json, "gates": g, "verdict": verdict_e(g), "c0": c0,
         "per_instance_normal": normal}
    a["predictions"] = predictions(a)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']} | " + " | ".join(f"{arm} M1 {pooled[arm]['acc_top10']:.3f} M0 {pooled[arm]['pair_acc']:.3f} r3 {pooled[arm]['regret3']:.3f}"
                                                   + (f" Δ {deltas['cross'][arm]['acc_top10']['mean_delta']:+.3f}" if arm in present else "")
                                                   for arm in arms_all) + f" | E cos cross {diag['cos_cross']} within {diag['cos_within']}", flush=True)
    return a


def predictions(a: dict) -> dict:
    g, p, d = a["gates"], a["pooled"], a["deltas"]["cross"]
    m0p = (d.get("e_diff_hard", {}).get("pair_acc") or {}).get("p")
    rj = a["run_json_cross"].get("e_diff_hard", {})
    rj_sig = [v["tiers"] for v in rj.values()]
    e_m0, c_m0 = p.get("e_diff_hard", {}).get("pair_acc"), p.get("e_concat_hard", {}).get("pair_acc")
    return {"P1": {"desc": "C0 过", "hit": a["c0"]["pass"]},
            "P2": {"desc": "G-E0 不塌缩", "hit": g["GE0"]["pass"]},
            "P3": {"desc": "G-E1 过", "hit": g["GE1"]["pass"]},
            "P4": {"desc": "G-E2 过", "hit": g["GE2"]["pass"]},
            "P5": {"desc": "e_diff_hard M0 cross Δ 显著（p<0.01，排名口径或 δ 对档位 pass）",
                   "value": [m0p, rj_sig], "hit": None if m0p is None else bool(m0p < 0.01 or any("pass" in t for t in rj_sig))},
            "P6": {"desc": "e_diff_hard M0 ≥ e_concat_hard + 0.01", "value": None if e_m0 is None or c_m0 is None else e_m0 - c_m0,
                   "hit": None if e_m0 is None or c_m0 is None else bool(e_m0 - c_m0 >= 0.01)}}


def markdown_tables(a: dict) -> str:
    f = lambda b, k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
    L = [f"# WP16-TCLSC 阶段 1 真 E 臂 tables（{a['n_instances']} 实例，7 折 LODO）", "", f"读数码 **{a['verdict']}**", "",
         "| 臂 | M1 好半区 acc | M2 | M0 δ 对 acc | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in a["arms"]:
        b = a["pooled"][arm]
        L.append(f"| {arm} | {f(b,'acc_top10')} | {f(b,'wacc')} | {f(b,'pair_acc')} | {f(b,'rho')} | {f(b,'regret1')} | {f(b,'regret3')} |")
    L += ["", "## 排名口径消融 Δ（normal − cross 3 种子均值 / normal − zero）", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
    for arm in a["arms"]:
        for kind in ("cross", "zero"):
            for metric in ("acc_top10", "wacc", "pair_acc"):
                dd = a["deltas"].get(kind, {}).get(arm, {}).get(metric)
                if not dd or dd.get("mean_delta") is None:
                    continue
                L.append(f"| {arm} | {kind} | {metric} | {dd['normal']:.3f} | {dd['cross']:.3f} | {dd['mean_delta']:+.3f} | {dd['p']:.3g} | {dd['wins']}/{dd['losses']}/{dd['ties']} |")
    L += ["", "## 逐折 Δ_M1（cross）", ""] + [f"- {arm}: " + ", ".join(f"{fo[:6]} {('—' if x is None else f'{x:+.3f}')}" for fo, x in bf.items())
                                           for arm, bf in a["by_fold_delta"].items()]
    L += ["", "## E 诊断（seed 0，逐折 / 中位）", "", f"- e_diff_hard 中位：cos cross {a['diag']['cos_cross']} / within {a['diag']['cos_within']} / norm {a['diag']['norm_median']}"]
    for t, d in a["diag"]["by_fold"].items():
        if d:
            L.append(f"  - {t}: cross {d['cos_cross']:.4f} within {d['cos_within']:.4f} norm {d['norm_median']:.3f}")
    cc_parts = []
    for t, d in a["diag"]["concat_by_fold"].items():
        cc_parts.append(f"{t[:6]} " + ("—" if not d else f"{d['cos_cross']:.4f}"))
    L += ["- e_concat_hard 逐折：" + ", ".join(cc_parts)]
    L += ["", "## 33 号 run json（δ 对口径）cross 档", ""]
    for arm, per in a["run_json_cross"].items():
        L.append(f"- {arm}: " + ", ".join(f"{t[:6]} {v['cross_delta']:+.4f} {'/'.join(v['tiers'])}" for t, v in per.items()))
    L += ["", "## 价值（regret@3 配对，H1 E 臂更低）", ""] + [
        f"- {t['a']} → {t['b']}: n={t['n']} {t['mean_a']} → {t['mean_b']} p={t['p']} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}" for t in a["value_tests"]]
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"] + [
        f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |" for k, v in a["gates"].items()]
    L += ["", f"C0: {json.dumps(a['c0'], ensure_ascii=False)}", "", "## 预测对账", ""] + [
        f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(E_ARMS), choices=list(E_ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--folds", nargs="+", default=list(FOLDS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=OUT / "binned")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.rank or args.analyze):
        build_parser().error("至少指定 --rank / --analyze 之一")
    if args.rank:
        cmd_rank(args.outdir, arms=args.arms, seeds=args.seeds, folds=args.folds, device=args.device)
    if args.analyze:
        cmd_analyze(args.source, args.outdir, arms=args.arms, seeds=args.seeds)


if __name__ == "__main__":
    main()
