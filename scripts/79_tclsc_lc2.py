"""WP16 LC2 检验：未见配置学习曲线——训练侧每切片的好区域（S⁺）随机配置数 → held-out 数据集上对从未见过的配置的排序质量。协议 results/tclsc_lc2/PROTOCOL.md。零新 R'。

子命令：
  --labels   标签池（shared k3 + random k1 + NB1 邻居 k1，三截点）+ 真值（k1）+ S⁺ 过滤（SP1 模型）→ labels_pool.jsonl / labels_truth.jsonl / splus_keys.json
  --train    五个曲线点 R0 / R3 / R6 / RA / RN × ft60_hard / blind_diff_hard × LODO 7 折 × 3 种子（73 号 train_fold_uea，tag = 点名）
  --rank     每点每折：12 考侧切片 × 30 条固定考卷（shared 10 + 固定邻居 4 = 已见；random 10 + 切片邻居 6 = 未见）→ rankings.json
  --analyze  未见配置指标曲线、配对检验、门、读数码、表
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED, run_seed_for  # noqa: E402
from autocls.space.cls_space import CLS  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g73 = importlib.import_module("scripts.73_tclsc_lc")
_g74 = importlib.import_module("scripts.74_tclsc_slices")
_g75 = importlib.import_module("scripts.75_tclsc_k3")
_g76 = importlib.import_module("scripts.76_tclsc_nb")
_g78 = importlib.import_module("scripts.78_tclsc_sp")
OUT = ROOT / "results" / "tclsc_lc2"
NB, K3, SP, SOURCE, SX = _g76.OUT, _g75.OUT, _g78.OUT, pq.OUT, _g74.OUT
POINTS = ("R0", "R3", "R6", "RA", "RN")
POINT_K = {"R0": 0, "R3": 3, "R6": 6}          # RA = 全部 S⁺ 随机；RN = RA + NB 邻居
ARMS = ("ft60_hard", "blind_diff_hard")
E_ARM, B_ARM = ARMS
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
SUB_SEED = 0
ZBAD = -1.0
G1_STEP1, G1_STEP2, G2_MIN, G3_MIN, ALPHA = 0.02, 0.01, 0.01, 0.10, 0.05
SRC_RANDOM, SRC_NB, SRC_NB_FIXED = "random", "nb", "nb_fixed"


# ---------------------------------------------------------------- 标签池
def load_pool() -> tuple[list[dict], list[dict]]:
    """训练池（shared k3 + random k1 + NB k1，带 src）与真值（全部 k1）。"""
    k3 = [dict(r, src="shared") for r in _g32.read_rows(K3 / "labels_k3.jsonl")]
    k1_shared = [dict(r, src="shared") for r in _g32.read_rows(NB / "labels_truth_k1.jsonl") if "nb_h" not in r]
    rand = [dict(r, src=SRC_RANDOM) for r in _g32.read_rows(SOURCE / "rprime_cache.jsonl") + _g32.read_rows(SX / "rprime_cache.jsonl")
            if r.get("origin") == ORIGIN_RANDOM and r.get("r_prime") is not None and r.get("error") is None]
    cat = _g76.load_catalog(NB)
    nb = []
    for r in _g76.annotate(_g32.read_rows(NB / "rprime_nb.jsonl"), cat):
        nb.append(dict(r, src=(SRC_NB_FIXED if r.get("nb_fixed") else SRC_NB), origin=(ORIGIN_SHARED if r.get("nb_fixed") else ORIGIN_RANDOM)))
    return k3 + rand + nb, k1_shared + rand + nb


def splus_filter(rows: Sequence[dict]) -> set[str]:
    """SP1 任务盲过滤器：random 配置里 ẑ ≥ τ 的 key。"""
    import joblib
    model = joblib.load(SP / "sp_model.joblib")
    tau = float(json.loads((SP / "tau.json").read_text(encoding="utf-8"))["tau"])
    space = _g56.space()
    keys, cfgs = [], []
    seen = set()
    for r in rows:
        if _src(r) == SRC_RANDOM and r["cls_key"] not in seen:
            seen.add(r["cls_key"]); keys.append(r["cls_key"]); cfgs.append(CLS(**r["cls"]))
    zh = _g78.predict_quality(model, _g78.encode(space, cfgs))
    return {k for k, v in zip(keys, zh) if v >= tau}


def _src(r: Mapping) -> str:
    """行来源：显式 src；缺省按 origin（shared → shared，其余 → random）。"""
    return r.get("src") or ("shared" if r.get("origin") == ORIGIN_SHARED else SRC_RANDOM)


def curve_rows(rows: Sequence[dict], *, splus_keys: set[str], seed: int = SUB_SEED) -> dict[str, list[dict]]:
    """五个曲线点的训练行：shared 全留；random 只取 S⁺ 内、按切片固定种子排列取前 k（三截点同进同出，嵌套）；RN 再加全部 NB 行。"""
    by_slice: dict[str, list[str]] = {}
    for r in rows:
        if _src(r) == SRC_RANDOM and r["cls_key"] in splus_keys:
            lst = by_slice.setdefault(r["task_id"], [])
            if r["cls_key"] not in lst:
                lst.append(r["cls_key"])
    order = {s: [ks[i] for i in np.random.default_rng([seed, run_seed_for(s, "lc2")]).permutation(len(ks))] for s, ks in by_slice.items()}
    out: dict[str, list[dict]] = {}
    for name in POINTS:
        keep_rand = {}
        for s, ks in order.items():
            keep_rand[s] = set(ks if name in ("RA", "RN") else ks[:POINT_K[name]])
        sub = []
        for r in rows:
            src = _src(r)
            if src == "shared":
                sub.append(r)
            elif src == SRC_RANDOM and r["cls_key"] in keep_rand.get(r["task_id"], set()):
                sub.append(r)
            elif src in (SRC_NB, SRC_NB_FIXED) and name == "RN":
                sub.append(r)
        out[name] = sub
    return out


def exam_catalog(rows: Sequence[dict], slice_id: str) -> list[tuple[str, dict, str]]:
    """30 条考卷：shared 10 + 固定邻居 4（已见，origin shared）+ random 10 + 切片邻居 6（未见，origin random）；(origin, cls dict, key)。"""
    seen_keys, out = set(), []
    for src in ("shared", SRC_NB_FIXED, SRC_RANDOM, SRC_NB):
        for r in rows:
            if r["task_id"] == slice_id and _src(r) == src and r["cls_key"] not in seen_keys:
                seen_keys.add(r["cls_key"])
                out.append((ORIGIN_SHARED if src in ("shared", SRC_NB_FIXED) else ORIGIN_RANDOM, r["cls"], r["cls_key"]))
    return out


def unseen_keys(cat: Sequence[tuple[str, dict, str]]) -> set[str]:
    return {k for o, _c, k in cat if o == ORIGIN_RANDOM}


def cmd_labels(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    pool, truth = load_pool()
    splus = splus_filter(pool)
    for name, rows in (("labels_pool.jsonl", pool), ("labels_truth.jsonl", truth)):
        with open(outdir / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (outdir / "splus_keys.json").write_text(json.dumps(sorted(splus)), encoding="utf-8")
    pts = curve_rows(pool, splus_keys=splus)
    n_rand = {r["cls_key"] for r in pool if _src(r) == SRC_RANDOM}
    summ = {"pool_rows": len(pool), "truth_rows": len(truth), "random_configs": len(n_rand), "splus_random_configs": len(splus), "splus_share": len(splus) / len(n_rand),
            "points_rows": {k: len(v) for k, v in pts.items()},
            "points_nonshared_per_slice": {k: float(np.mean([len({r["cls_key"] for r in v if r["task_id"] == s and _src(r) != "shared"}) for s in {r["task_id"] for r in v}])) for k, v in pts.items()}}
    (outdir / "labels_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    return summ


def load_pool_from(outdir: Path) -> tuple[list[dict], list[dict], set[str]]:
    return (_g32.read_rows(outdir / "labels_pool.jsonl"), _g32.read_rows(outdir / "labels_truth.jsonl"),
            set(json.loads((outdir / "splus_keys.json").read_text(encoding="utf-8"))))


# ---------------------------------------------------------------- 训练 / 排名
def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, points: Sequence[str] = POINTS,
              folds_sel: Optional[Sequence[str]] = None) -> None:
    pool, _truth, splus = load_pool_from(outdir)
    pts = curve_rows(pool, splus_keys=splus)
    dk = pq.load_delta_calibration(SOURCE / "delta_calibration.json")
    table = _g74.ctx_table(_g74.OUT)
    for name in points:
        for target in (folds_sel or FOLDS):
            for arm in ARMS:
                for seed in seeds:
                    if resume and (outdir / "models" / name / target / f"{arm}_s{seed}.pt").exists():
                        continue
                    run = _g73.train_fold_uea(pts[name], dk, table, target=target, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs, tag=name)
                    _g73._log(outdir, f"[{name}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")


def rank_point(outdir: Path, name: str, *, pool: Sequence[dict], seeds: Sequence[int], device: str = "cpu") -> dict:
    table = _g74.ctx_table(_g74.OUT)
    out = {}
    for target in FOLDS:
        if not (outdir / "models" / name / target).exists():
            continue
        test_slices = _g75.exam_slices(target)
        catm = {s: [(o, CLS(**c), k) for o, c, k in exam_catalog(pool, s)] for s in test_slices}
        per: dict = {"normal": {}, "test": test_slices}
        for arm in ARMS:
            paths = [outdir / "models" / name / target / f"{arm}_s{sd}.pt" for sd in seeds]
            if not all(p.exists() for p in paths):
                continue
            mm = [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in paths]
            if _m33.arm_spec(arm).ctx_kind == "table_budget":
                meta0 = mm[0][1]
                ctx_rows = _g64.slice_ctx_rows(table, tuple(meta0["ctx_cols"]), test_slices, np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"]))
                rk = _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device, ctx_rows=ctx_rows)
            else:
                rk = _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device)
            for b, pb in rk.items():
                for s, pa in pb.items():
                    per["normal"].setdefault(str(b), {}).setdefault(s, {}).update(pa)
        out[target] = per
    return out


def cmd_rank(outdir: Path, *, seeds: Sequence[int] = SEEDS, points: Sequence[str] = POINTS) -> dict:
    pool, _truth, _splus = load_pool_from(outdir)
    rk = {name: rank_point(outdir, name, pool=pool, seeds=seeds) for name in points if (outdir / "models" / name).exists()}
    (outdir / "rankings.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    return rk


# ---------------------------------------------------------------- 指标
def unseen_metrics(ranking: Sequence[str], truth: Mapping[str, float], z: Mapping[str, float], unseen: set[str], *, delta: float) -> dict:
    order = [k for k in ranking if k in truth]
    pos = {k: i for i, k in enumerate(order)}
    un = [k for k in order if k in unseen]
    pairs = [(x, y) for x, y in itertools.combinations(un, 2) if abs(truth[x] - truth[y]) >= delta]
    acc = float(np.mean([(pos[x] < pos[y]) == (truth[x] > truth[y]) for x, y in pairs])) if pairs else None
    good = [(x, y) for x, y in pairs if z[x] >= ZBAD and z[y] >= ZBAD]
    acc_good = float(np.mean([(pos[x] < pos[y]) == (truth[x] > truth[y]) for x, y in good])) if good else None
    oracle_un = max(truth[k] for k in un) if un else None
    regret3 = float(oracle_un - max(truth[k] for k in un[:3])) if un else None
    top3_pred = order[:3]
    top3_true = sorted(order, key=lambda k: -truth[k])[:3]
    over = float(np.mean([k in unseen for k in top3_pred]) - np.mean([k in unseen for k in top3_true]))
    shared = [k for k in order if k not in unseen]
    best_shared_pos = pos[shared[0]] if shared else len(order)
    best_shared_truth = max(truth[k] for k in shared) if shared else -np.inf
    return {"unseen_pair_acc": acc, "n_pairs": len(pairs), "unseen_good_pair_acc": acc_good, "n_good_pairs": len(good), "unseen_regret3": regret3, "over": over,
            "above_trust_pred": int(sum(1 for k in un if pos[k] < best_shared_pos)), "above_trust_true": int(sum(1 for k in un if truth[k] > best_shared_truth))}


def instance_rows(rk: Mapping, truth_rows: Sequence[dict], pool: Sequence[dict], dk: Mapping, arms: Sequence[str]) -> list[dict]:
    val: dict[tuple, float] = {}
    for r in truth_rows:
        val[(r["task_id"], int(r["iters"]), r["cls_key"])] = float(r["r_prime"])
    out = []
    for target, per in rk.items():
        for b, pb in per["normal"].items():
            for s, pa in pb.items():
                cat = exam_catalog(pool, s)
                keys = [k for _o, _c, k in cat if (s, int(b), k) in val]
                truth = {k: val[(s, int(b), k)] for k in keys}
                v = np.array(list(truth.values())); mu, sd = v.mean(), v.std()
                z = {k: (truth[k] - mu) / sd if sd > 0 else 0.0 for k in keys}
                unseen = unseen_keys(cat) & set(keys)
                dl = float(dk[(pq.dataset_of(s), int(b))])
                row = {"fold": target, "slice": s, "budget": int(b), "n_keys": len(keys), "metrics": {}}
                for a in arms:
                    if a in pa:
                        row["metrics"][a] = unseen_metrics(pa[a], truth, z, unseen, delta=dl)
                out.append(row)
    return out


def _pool(rows: Sequence[dict], arm: str, key: str) -> Optional[float]:
    v = [r["metrics"][arm][key] for r in rows if arm in r["metrics"] and r["metrics"][arm].get(key) is not None]
    return float(np.mean(v)) if v else None


def _paired(rows_a: Sequence[dict], rows_b: Sequence[dict], arm: str, key: str) -> dict:
    """同实例配对（a 点 → b 点），H1：b 更高。"""
    ia = {(r["fold"], r["slice"], r["budget"]): r["metrics"].get(arm, {}).get(key) for r in rows_a}
    ib = {(r["fold"], r["slice"], r["budget"]): r["metrics"].get(arm, {}).get(key) for r in rows_b}
    ks = [k for k in ia if k in ib and ia[k] is not None and ib[k] is not None]
    if not ks:
        return {"n": 0, "delta": None, "p": None}
    x, y = np.array([ia[k] for k in ks]), np.array([ib[k] for k in ks])
    return {"n": len(ks), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "delta": float((y - x).mean()), "p": float(_g56.paired_wilcoxon_less(x, y)),
            "wins": int(np.sum(y > x)), "losses": int(np.sum(y < x))}


def gate_l1(d: Mapping) -> bool:
    a, b = d.get("R3_RA", {}), d.get("RA_RN", {})
    return bool(a.get("delta") is not None and a["delta"] >= G1_STEP1 and a.get("p") is not None and a["p"] < ALPHA and b.get("delta") is not None and b["delta"] >= G1_STEP2)


def gate_l2(gap: Mapping) -> bool:
    return bool(gap.get("RN") is not None and gap.get("R3") is not None and gap["RN"] >= gap["R3"] + G2_MIN)


def gate_l3(over: Mapping) -> bool:
    return bool(over.get("RN") is not None and over.get("R3") is not None and over["R3"] - over["RN"] >= G3_MIN)


def verdict_lc2(g: Mapping[str, bool]) -> str:
    if g["G-L1"]:
        return "LC2-RISING"
    return "LC2-CALIBRATE-ONLY" if g["G-L3"] else "LC2-FLAT"


def cmd_analyze(outdir: Path) -> dict:
    rk = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    pool, truth, _splus = load_pool_from(outdir)
    dk = pq.load_delta_calibration(SOURCE / "delta_calibration.json")
    rows = {name: instance_rows(rk[name], truth, pool, dk, ARMS) for name in rk}
    keys = ("unseen_pair_acc", "unseen_good_pair_acc", "unseen_regret3", "over", "above_trust_pred", "above_trust_true")
    curve = {name: {arm: {k: _pool(rs, arm, k) for k in keys} | {"n": len(rs)} for arm in ARMS} for name, rs in rows.items()}
    steps = {}
    for a, b in (("R0", "R3"), ("R3", "R6"), ("R6", "RA"), ("R3", "RA"), ("RA", "RN"), ("R3", "RN")):
        if a in rows and b in rows:
            steps[f"{a}_{b}"] = {arm: _paired(rows[a], rows[b], arm, "unseen_good_pair_acc") for arm in ARMS}
    gap = {name: (curve[name][E_ARM]["unseen_good_pair_acc"] or 0) - (curve[name][B_ARM]["unseen_good_pair_acc"] or 0) for name in curve}
    e_vs_b = {name: _paired([{**r, "metrics": {E_ARM: r["metrics"].get(B_ARM, {})}} for r in rs], rs, E_ARM, "unseen_good_pair_acc") for name, rs in rows.items()}
    over = {name: curve[name][E_ARM]["over"] for name in curve}
    g = {"G-L1": gate_l1({k: v[E_ARM] for k, v in steps.items()}), "G-L2": gate_l2(gap), "G-L3": gate_l3(over)}
    by_fold = {name: {arm: {f: float(np.mean([r["metrics"][arm]["unseen_good_pair_acc"] for r in rs if r["fold"] == f and arm in r["metrics"] and r["metrics"][arm]["unseen_good_pair_acc"] is not None]))
                            for f in sorted({r["fold"] for r in rs})} for arm in ARMS} for name, rs in rows.items()}
    a = {"protocol": "results/tclsc_lc2/PROTOCOL.md", "curve": curve, "steps": steps, "e_vs_blind": e_vs_b, "gap": gap, "over": over, "by_fold": by_fold, "gates": g, "verdict": verdict_lc2(g),
         "labels": json.loads((outdir / "labels_summary.json").read_text(encoding="utf-8"))}
    best = max(curve, key=lambda n: curve[n][E_ARM]["unseen_good_pair_acc"] or -1)
    a["predictions"] = {"P1": {"desc": "G-L1", "hit": g["G-L1"]},
                        "P2": {"desc": "RN 点 E_T > 盲（p<0.05）", "value": e_vs_b.get("RN"), "hit": bool("RN" in e_vs_b and e_vs_b["RN"].get("p") is not None and e_vs_b["RN"]["p"] < ALPHA and e_vs_b["RN"]["delta"] > 0)},
                        "P3": {"desc": "G-L3", "hit": g["G-L3"]}, "P4": {"desc": "RN 是 E_T 最好点", "value": best, "hit": best == "RN"},
                        "P5": {"desc": "R0 的 unseen_good_pair_acc ≤ 0.55", "value": curve.get("R0", {}).get(E_ARM, {}).get("unseen_good_pair_acc"),
                               "hit": bool("R0" in curve and curve["R0"][E_ARM]["unseen_good_pair_acc"] is not None and curve["R0"][E_ARM]["unseen_good_pair_acc"] <= 0.55)}}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], g, flush=True)
    return a


def markdown_tables(a: dict) -> str:
    f = lambda v, d=3: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    L = ["# LC2 检验 tables：未见配置学习曲线", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", "", f"标签：{a['labels']}", "",
         "| 点 | 臂 | n | unseen pair acc | unseen good pair acc | unseen regret@3 | over | above_trust pred / true |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, per in a["curve"].items():
        for arm, c in per.items():
            L.append(f"| {name} | {arm} | {c['n']} | {f(c['unseen_pair_acc'])} | {f(c['unseen_good_pair_acc'])} | {f(c['unseen_regret3'], 4)} | {f(c['over'])} | {f(c['above_trust_pred'], 2)} / {f(c['above_trust_true'], 2)} |")
    L += ["", "## 台阶（unseen good pair acc，同实例配对，H1 = 后点更高）", "", "| 台阶 | 臂 | n | 前 | 后 | Δ | p | 胜:负 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for st, per in a["steps"].items():
        for arm, d in per.items():
            if d.get("delta") is not None:
                L.append(f"| {st} | {arm} | {d['n']} | {f(d['mean_a'])} | {f(d['mean_b'])} | {d['delta']:+.4f} | {d['p']:.3g} | {d['wins']}:{d['losses']} |")
    L += ["", f"E_T − 盲（unseen good pair acc）：{ {k: round(v, 4) for k, v in a['gap'].items()} }；E vs 盲配对：{ {k: (round(v['delta'], 4), round(v['p'], 3)) for k, v in a['e_vs_blind'].items() if v.get('delta') is not None} }",
          f"E_T over：{ {k: round(v, 3) for k, v in a['over'].items() if v is not None} }", "", "逐折 unseen good pair acc：", ""]
    for name, per in a["by_fold"].items():
        L.append(f"- {name}: " + "; ".join(f"{arm} {{{', '.join(f'{k[:6]} {v:.3f}' for k, v in d.items())}}}" for arm, d in per.items()))
    L += ["", f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for fl in ("labels", "train", "rank", "analyze", "resume"):
        ap.add_argument(f"--{fl}", action="store_true")
    ap.add_argument("--points", nargs="+", default=list(POINTS), choices=list(POINTS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.labels:
        cmd_labels(args.outdir)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, points=args.points, folds_sel=args.folds)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds, points=args.points)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
