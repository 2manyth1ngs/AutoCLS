"""WP16-TCLSC 阶段 1 n 放大：UCR 84 数据集分组 7 折，diff 头 + φ / 转导指纹（协议 results/tclsc_s1ucr/PROTOCOL.md）。

零新 R'：语料 = WP7 的 `results/ucr_rprime/rprime_cache.jsonl`（84 × 20 × 3），φ = `results/ucr_module/phi_ucr.jsonl`（取 PHI_COLS 12 列）。
δ = 0.060 全局（协议 §0 标定）。臂表自带（ArmSpec 复用 33 号），训练直接走 TCLSCEngine（不经 33 号 PQContext——它绑定 UEA 切片 / labels.yaml）。

子命令：--train（脱离启动 + --resume）/ --rank / --analyze。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import ClsRecord, TCLSCEngine  # noqa: E402
from autocls.labeling import cls_key  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g59 = importlib.import_module("scripts.59_tclsc_s1fp")
ArmSpec = _m33.ArmSpec
PHI_COLS = _m33.PHI_COLS
BUDGETS = pq.BUDGETS
DELTA_UCR = 0.06
UCR_CACHE = ROOT / "results" / "ucr_rprime" / "rprime_cache.jsonl"
UCR_PHI = ROOT / "results" / "ucr_module" / "phi_ucr.jsonl"
OUT = ROOT / "results" / "tclsc_s1ucr"
N_FOLDS = 7
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
ARMS: dict[str, ArmSpec] = {
    s.name: s for s in (
        ArmSpec("u_blind_diff_hard", "S1", "none", None, None, "none", head="diff", seeds=(0, 1, 2)),
        ArmSpec("u_phi_concat_hard", "S1", "ctx", None, None, "phi_budget", head="concat", seeds=(0, 1, 2)),
        ArmSpec("u_phi_diff_hard", "S1", "ctx", None, None, "phi_budget", head="diff", seeds=(0, 1, 2)),
        ArmSpec("u_fp_diff_hard", "S1", "ctx", None, None, "fp_budget", head="diff", seeds=(0, 1, 2)),
        # R 检验 §9 消融：field 头 + 只有预算 one-hot（无数据集信息）
        ArmSpec("u_fwb_hard", "S1", "ctx", None, None, "budget", head="field", seeds=(0, 1, 2)),
    )
}
PHI_ARMS = ("u_phi_concat_hard", "u_phi_diff_hard")
FP_ARMS = ("u_fp_diff_hard",)
# 门（协议 §4）
C0_BAND = (0.65, 0.85)
M1_MIN, ALPHA = 0.03, 0.05
PRIMARY = "acc_top10"
TOP_NONPROBE = 7

_ENC: Optional[CLSEncoder] = None


def enc() -> CLSEncoder:
    global _ENC
    if _ENC is None:
        _ENC = CLSEncoder(CLSSpace.from_yaml(), mode="binned")
    return _ENC


def cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 语料
def grouped_folds(names: Sequence[str], k: int = N_FOLDS) -> list[list[str]]:
    """数据集名排序后 i % k 分组（逐位确定、互斥）。"""
    ordered = sorted(set(names))
    return [[n for i, n in enumerate(ordered) if i % k == j] for j in range(k)]


def load_rows(path: Path = UCR_CACHE) -> list[dict]:
    return [r for r in read_jsonl(path) if r.get("error") is None and r.get("r_prime") is not None]


def load_phi(path: Path = UCR_PHI) -> dict[str, dict]:
    return {r["dataset"]: r for r in read_jsonl(path)}


def rows_to_cache(rows: Sequence[dict]) -> dict[tuple[str, str, int], dict]:
    return {(r["task_id"], r["cls_key"], int(r["iters"])): r for r in rows}


def build_records(rows: Sequence[dict], datasets: Sequence[str]) -> list[ClsRecord]:
    keep = set(datasets)
    e = enc()
    return [ClsRecord(task_id=pq.engine_task_id(r["task_id"], int(r["iters"])), key=r["cls_key"], origin=r["origin"],
                      vec=e.encode(CLS(**r["cls"])), r=float(r["r_prime"]))
            for r in rows if r["dataset"] in keep]


def catalog_for(rows: Sequence[dict], dataset: str) -> dict[str, list[tuple[str, CLS, str]]]:
    """{任务: [(origin, CLS, key)]}——该数据集 20 条候选（任一预算的行去重）。"""
    seen: dict[str, tuple[str, CLS, str]] = {}
    tid = None
    for r in rows:
        if r["dataset"] != dataset:
            continue
        tid = r["task_id"]
        if r["cls_key"] not in seen:
            c = CLS(**r["cls"])
            seen[r["cls_key"]] = (r["origin"], c, r["cls_key"])
    if tid is None:
        raise KeyError(f"语料里没有数据集 {dataset}")
    return {tid: [seen[k] for k in sorted(seen)]}


def _tids(datasets: Sequence[str]) -> list[str]:
    return [pq.engine_task_id(f"{d}_00", int(b)) for d in sorted(datasets) for b in BUDGETS]


# ---------------------------------------------------------------- ctx
def phi_context(phi: Mapping[str, Mapping], train: Sequence[str], test: Sequence[str]) -> tuple[dict, dict]:
    """φ ⊕ 预算 one-hot；μ/σ 只用训练集（铁律 6）。返回 ({engine tid: ctx}, meta)。"""
    table = {f"{d}_00": phi[d] for d in phi}
    mu, sd = _m33.phi_standardizer(table, [f"{d}_00" for d in train], PHI_COLS)
    tids = _tids(list(train) + list(test))
    ctx = _m33.context_for(ARMS["u_phi_diff_hard"], tids, BUDGETS, (), phi=table, phi_mu=mu, phi_sd=sd)
    return ctx, {"phi_cols": list(PHI_COLS), "phi_mu": [float(v) for v in mu], "phi_sd": [float(v) for v in sd]}


def fp_context(cache: Mapping, train: Sequence[str], test: Sequence[str]) -> tuple[dict, dict]:
    """探针 = 训练集先验前 K 条 shared；指纹 @60；μ/σ 只用训练集。"""
    train_t = [f"{d}_00" for d in train]
    probes = _m33.fp_probes(cache, train_t, k=_m33.FP_K)
    raw = _m33.fp_raw(cache, train_t + [f"{d}_00" for d in test], probes, budget=_m33.FP_BUDGET)
    mu, sd = _m33.fp_standardizer(raw, train_t)
    ctx = _m33.context_for(ARMS["u_fp_diff_hard"], _tids(list(train) + list(test)), BUDGETS, (), fp=raw, fp_mu=mu, fp_sd=sd)
    return ctx, {"fp_probes": list(probes), "fp_budget": _m33.FP_BUDGET, "fp_mu": [float(v) for v in mu], "fp_sd": [float(v) for v in sd]}


# ---------------------------------------------------------------- 训练 / 排名
def train_fold(rows: Sequence[dict], phi: Mapping, folds: Sequence[Sequence[str]], idx: int, *, arm: str, seed: int,
               outdir: Path, device: str, epochs: Optional[int] = None) -> dict:
    spec = ARMS[arm]
    test = list(folds[idx])
    train = sorted({r["dataset"] for r in rows} - set(test))
    c = dict(cfg())
    if epochs:
        c["epochs"], c["curriculum_epochs"] = int(epochs), min(int(c["curriculum_epochs"]), max(1, int(epochs) // 2))
    recs = build_records(rows, train)
    ctx_map, meta_extra = None, None
    if spec.ctx_kind == "phi_budget":
        ctx_map, meta_extra = phi_context(phi, train, test)
    elif spec.ctx_kind == "budget":
        ctx_map = {t: pq.ctx_vector(pq.split_engine_task_id(t)[1], BUDGETS) for t in _tids(list(train) + list(test))}
    elif spec.ctx_kind == "fp_budget":
        ctx_map, meta_extra = fp_context(rows_to_cache(rows), train, test)
    cdim = 0 if ctx_map is None else int(next(iter(ctx_map.values())).shape[0])
    cls_dim = int(recs[0].vec.shape[0])
    model = _m33.build_model(spec, cls_dim=cls_dim, ctx_dim=cdim, nz=256, dim_input=256, cfg=c, seed=seed)
    train_ctx = None if ctx_map is None else {t: ctx_map[t] for t in sorted({r.task_id for r in recs})}
    engine = TCLSCEngine(model, {}, device=device, lr=c["lr"], batch_size=c["batch_size"], seed=seed, context_of=train_ctx)
    t0 = time.time()
    hist = engine.train(recs, epochs=c["epochs"], curriculum_epochs=c["curriculum_epochs"],
                        better_fn=lambda a, b: is_better(a, b, TASK_CLASSIFICATION), delta=DELTA_UCR)
    secs = round(time.time() - t0, 1)
    fold = f"fold{idx}"
    _m33.save_model(outdir / "models" / fold / f"{arm}_s{seed}.pt", model, spec, fold=fold, seed=seed, cls_dim=cls_dim, ctx_dim=cdim,
                    cfg=c, nz=256, settf_windows=0, settf_t_max=0, budgets=BUDGETS, ctx_encoding="onehot", encoder_mode="binned",
                    head=spec.head, ctx_kind=spec.ctx_kind, phi_stats=meta_extra)
    run = {"fold": fold, "arm": arm, "seed": seed, "train_datasets": train, "test_datasets": test, "n_train_records": len(recs),
           "delta": DELTA_UCR, "train_seconds": secs, "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"],
           "train_pairs_last": hist[-1]["n_pairs"], "settings": {"head": spec.head, "ctx_kind": spec.ctx_kind, "ctx_dim": cdim,
                                                                 "epochs": c["epochs"], "encoder_mode": "binned"},
           "ctx_meta": meta_extra}
    p = outdir / "runs" / fold / f"{arm}_s{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


def rank_fold(rows: Sequence[dict], phi: Mapping, folds: Sequence[Sequence[str]], idx: int, *, arms: Sequence[str],
              seeds: Sequence[int], outdir: Path, device: str = "cpu") -> dict:
    fold = f"fold{idx}"
    test = list(folds[idx])
    train = sorted({r["dataset"] for r in rows} - set(test))
    test_t = [f"{d}_00" for d in sorted(test)]
    train_t = [f"{d}_00" for d in train]
    cat: dict = {}
    for d in test:
        cat.update(catalog_for(rows, d))
    models = {}
    for a in arms:
        paths = [outdir / "models" / fold / f"{a}_s{s}.pt" for s in seeds]
        if all(p.exists() for p in paths):
            models[a] = [_g34.load_pq_model(p, map_location=device, expect_fold=fold, expect_budgets=BUDGETS) for p in paths]
        else:
            print(f"⚠️ [{fold}/{a}] 模型不全，跳过", flush=True)
    per: dict = {"normal": {}, "cross": {}, "zero": {}, "probes": None, "test": test}

    def merge(dst: dict, rk: dict) -> None:
        for b, pb in rk.items():
            for s, pa in pb.items():
                dst.setdefault(b, {}).setdefault(s, {}).update(pa)

    blind = {a: m for a, m in models.items() if ARMS[a].ctx_kind in ("none", "budget")}   # budget 臂用 rank_local_models 的默认预算 one-hot
    if blind:
        merge(per["normal"], _g56.rank_local_models(blind, cat, budgets=BUDGETS, device=device))
    phi_models = {a: m for a, m in models.items() if ARMS[a].ctx_kind == "phi_budget"}
    fp_models = {a: m for a, m in models.items() if ARMS[a].ctx_kind == "fp_budget"}
    table = {f"{d}_00": phi[d] for d in phi}
    cache = rows_to_cache(rows) if fp_models else None
    groups = []
    if phi_models:
        meta0 = next(iter(phi_models.values()))[0][1]
        mu, sd = np.array(meta0["phi_mu"]), np.array(meta0["phi_sd"])
        # 闭包按默认参数绑定 mu/sd：下面 fp 块会重绑同名变量（晚绑定会把 φ 的 12 维统计换成指纹的 8 维）
        groups.append((phi_models, lambda cm=None, zero=False, mu=mu, sd=sd: _g58.phi_ctx_rows(table, test_t, mu, sd, cross_map=cm, zero=zero)))
    if fp_models:
        meta0 = next(iter(fp_models.values()))[0][1]
        probes, mu, sd = list(meta0["fp_probes"]), np.array(meta0["fp_mu"]), np.array(meta0["fp_sd"])
        raw = _m33.fp_raw(cache, train_t + test_t, probes, budget=int(meta0["fp_budget"]))
        per["probes"] = probes
        groups.append((fp_models, lambda cm=None, zero=False, raw=raw, mu=mu, sd=sd: _g59.fp_ctx_rows(raw, test_t, mu, sd, cross_map=cm, zero=zero)))
    for mm, rows_fn in groups:
        merge(per["normal"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn()))
        merge(per["zero"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
        for cs in CROSS_SEEDS:
            cmap = _g58.cross_slice_map(test_t, train_t, cs)
            per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
            merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
    return per


# ---------------------------------------------------------------- 门 / 读数码
def gates_phi(deltas: Mapping, value: Mapping) -> dict:
    pos = lambda arm: _g57._positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)
    gp2 = bool(value.get("p") is not None and value["p"] < ALPHA and value["mean_b"] < value["mean_a"])
    return {"GU1": {"arm": "u_phi_diff_hard", "delta": deltas.get("u_phi_diff_hard", {}).get(PRIMARY), "pass": pos("u_phi_diff_hard")},
            "GU2": {"metric": "regret3", "blind": value.get("mean_a"), "phi": value.get("mean_b"), "p": value.get("p"), "pass": gp2},
            "GU6": {"arm": "u_phi_concat_hard", "delta": deltas.get("u_phi_concat_hard", {}).get(PRIMARY), "pass": not pos("u_phi_concat_hard")}}


def verdict_phi(g: Mapping) -> str:
    if not g["GU1"]["pass"]:
        return "UCR-PHI-DEAD"
    return "UCR-PHI-ALIVE" if g["GU2"]["pass"] else "UCR-PHI-USED-NO-GAIN"


def gates_fp(deltas: Mapping, v2: Mapping, v3: Mapping) -> dict:
    pos = _g57._positive(deltas.get("u_fp_diff_hard", {}).get(PRIMARY, {}), min_delta=M1_MIN)
    lower = lambda v: bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])
    return {"GU3": {"arm": "u_fp_diff_hard", "view": "nonprobe", "delta": deltas.get("u_fp_diff_hard", {}).get(PRIMARY), "pass": pos},
            "GU4": {"view": "nonprobe", "metric": "regret3", "blind": v2.get("mean_a"), "fp": v2.get("mean_b"), "p": v2.get("p"), "pass": lower(v2)},
            "GU5": {"view": "all20@120/290", "probe6_top3": v3.get("mean_a"), "fp": v3.get("mean_b"), "p": v3.get("p"), "pass": lower(v3)}}


def verdict_fp(g: Mapping) -> str:
    if not g["GU3"]["pass"]:
        return "UCR-FP-DEAD"
    if not g["GU4"]["pass"]:
        return "UCR-FP-USED-NO-GAIN"
    return "UCR-FP-ALIVE" if g["GU5"]["pass"] else "UCR-FP-EXTRAPOLATES"


# ---------------------------------------------------------------- 子命令
def cmd_train(outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], folds_sel: Optional[Sequence[int]], device: str,
              epochs: Optional[int], resume: bool) -> None:
    rows, phi = load_rows(), load_phi()
    folds = grouped_folds(sorted({r["dataset"] for r in rows}))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=1), encoding="utf-8")
    log = outdir / "train.log"
    for idx in (folds_sel if folds_sel else range(len(folds))):
        for arm in arms:
            for seed in seeds:
                p = outdir / "models" / f"fold{idx}" / f"{arm}_s{seed}.pt"
                if resume and p.exists():
                    continue
                run = train_fold(rows, phi, folds, idx, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
                line = f"[fold{idx}/{arm}_s{seed}] {run['train_seconds']}s | bce {run['bce_first']:.4f}→{run['bce_last']:.4f} | pairs {run['train_pairs_last']}"
                print(line, flush=True)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")


def cmd_rank(outdir: Path, *, arms: Sequence[str], seeds: Sequence[int]) -> None:
    rows, phi = load_rows(), load_phi()
    folds = json.loads((outdir / "folds.json").read_text(encoding="utf-8"))
    out = {}
    for idx in range(len(folds)):
        out[f"fold{idx}"] = rank_fold(rows, phi, folds, idx, arms=arms, seeds=seeds, outdir=outdir)
        print(f"[fold{idx}] 靶 {folds[idx]}", flush=True)
    (outdir / "rankings_ucr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def _eval(rk_by_fold: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], which: str, *, view: str,
          cross_seed: Optional[str] = None, top_k: int) -> list[dict]:
    rows = []
    for fold, per in rk_by_fold.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        cat_f = [r for r in cat_z if r["dataset"] in set(per["test"])]
        if view == "nonprobe" and per.get("probes"):
            rk, cat_f = _g59.nonprobe_view(rk, cat_f, per["probes"])
        fr = _g59._eval_rows(rk, cat_f, delta, arms, top_k)
        for r in fr:
            r["fold"] = fold
        rows += fr
    return rows


def cmd_analyze(outdir: Path) -> dict:
    rows, phi = load_rows(), load_phi()
    cache = rows_to_cache(rows)
    rk = json.loads((outdir / "rankings_ucr.json").read_text(encoding="utf-8"))
    cat_z = _g56.zscore_within(rows)
    delta = {(d, int(b)): DELTA_UCR for d in {r["dataset"] for r in rows} for b in BUDGETS}
    present = [a for a in ARMS if any(a in pa for per in rk.values() for pb in per["normal"].values() for pa in pb.values())]
    ctx_present = [a for a in present if ARMS[a].ctx_kind != "none"]
    cross_seeds = sorted({cs for per in rk.values() for cs in per["cross"]})

    def block(view: str, top_k: int, arms: Sequence[str]) -> dict:
        normal = _eval(rk, cat_z, delta, arms, "normal", view=view, top_k=top_k)
        cross = [_eval(rk, cat_z, delta, [a for a in arms if a in ctx_present], "cross", view=view, cross_seed=cs, top_k=top_k) for cs in cross_seeds]
        zero = _eval(rk, cat_z, delta, [a for a in arms if a in ctx_present], "zero", view=view, top_k=top_k)
        pooled = {}
        for a in arms:
            pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
            pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
            pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
        deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in arms if a in ctx_present},
                  "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in arms if a in ctx_present}}
        by_fold = {a: {fo: _g57.ablation_delta([r for r in normal if r["fold"] == fo], [[r for r in c if r["fold"] == fo] for c in cross], a, PRIMARY).get("mean_delta")
                       for fo in sorted({r["fold"] for r in normal})} for a in arms if a in ctx_present}
        value = [_g58.paired_regret(normal, "u_blind_diff_hard", b) for b in arms if b in ctx_present and "u_blind_diff_hard" in arms]
        by_budget = {str(b): {a: _g57._pooled([r for r in normal if r["budget"] == b], a, "pair_acc") for a in arms} for b in BUDGETS}
        return {"n_instances": len(normal), "pooled": pooled, "deltas": deltas, "by_fold_delta": by_fold, "value_tests": value,
                "by_budget_pair_acc": by_budget, "per_instance_normal": normal}

    views = {"all20": block("all20", _g56.TOP_GOOD, present),
             "nonprobe": block("nonprobe", TOP_NONPROBE, [a for a in present if a in ("u_blind_diff_hard",) + FP_ARMS])}
    # 同成本参照（all20 @120/@290）
    ref_rows = []
    for r in views["all20"]["per_instance_normal"]:
        if int(r["budget"]) == _m33.FP_BUDGET or "u_fp_diff_hard3" not in r["regret"]:
            continue
        probes = rk[r["fold"]]["probes"]
        keys = [x["cls_key"] for x in cat_z if x["task_id"] == r["slice"] and int(x["iters"]) == int(r["budget"])]
        ref_rows.append({"probe6_top3": _g59.probe_top3_regret(cache, r["slice"], probes, keys, budget=int(r["budget"])),
                         "fp": float(r["regret"]["u_fp_diff_hard3"]), "blind": float(r["regret"].get("u_blind_diff_hard3", np.nan))})
    if ref_rows:
        x = np.array([r["probe6_top3"] for r in ref_rows]); y = np.array([r["fp"] for r in ref_rows])
        v3 = {"a": "probe6_top3", "b": "u_fp_diff_hard", "n": int(x.size), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
              "p": float(_g56.paired_wilcoxon_less(y, x)), "wins_b": int(np.sum(y < x)), "losses_b": int(np.sum(y > x)),
              "blind_mean": float(np.nanmean([r["blind"] for r in ref_rows]))}
    else:
        v3 = {"n": 0, "mean_a": None, "mean_b": None, "p": None}
    v_phi = next((t for t in views["all20"]["value_tests"] if t["b"] == "u_phi_diff_hard"), {"p": None, "mean_a": None, "mean_b": None})
    v_fp = next((t for t in views["nonprobe"]["value_tests"] if t["b"] == "u_fp_diff_hard"), {"p": None, "mean_a": None, "mean_b": None})
    g_phi = gates_phi(views["all20"]["deltas"]["cross"], v_phi)
    g_fp = gates_fp(views["nonprobe"]["deltas"]["cross"], v_fp, v3)
    c0_val = views["all20"]["pooled"].get("u_blind_diff_hard", {}).get("pair_acc")
    c0 = {"blind_M0": c0_val, "band": C0_BAND, "pass": None if c0_val is None else bool(C0_BAND[0] <= c0_val <= C0_BAND[1])}
    a = {"protocol": "results/tclsc_s1ucr/PROTOCOL.md", "n_datasets": len({r["dataset"] for r in rows}), "delta": DELTA_UCR, "arms": present,
         "views": views, "ref_same_cost": v3, "gates_phi": g_phi, "gates_fp": g_fp, "verdict_phi": verdict_phi(g_phi),
         "verdict_fp": verdict_fp(g_fp), "c0": c0, "probes_by_fold": {f: per["probes"] for f, per in rk.items()}}
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    pa = views["all20"]["pooled"]
    print(f"φ {a['verdict_phi']} | 指纹 {a['verdict_fp']} | " + " | ".join(
        f"{arm} M1 {pa[arm]['acc_top10']:.3f} M0 {pa[arm]['pair_acc']:.3f} r3 {pa[arm]['regret3']:.3f}" for arm in present)
        + f" | probe6_top3 {v3.get('mean_a')} vs fp {v3.get('mean_b')}", flush=True)
    return a


def predictions(a: dict) -> dict:
    gp, gf = a["gates_phi"], a["gates_fp"]
    d_fp = (a["views"]["nonprobe"]["deltas"]["cross"].get("u_fp_diff_hard", {}).get(PRIMARY) or {}).get("mean_delta")
    m0 = (a["views"]["all20"]["deltas"]["cross"].get("u_phi_diff_hard", {}).get("pair_acc") or {})
    return {"P1": {"desc": "C0 带内", "hit": a["c0"]["pass"]},
            "P2": {"desc": "G-U1 过", "hit": gp["GU1"]["pass"]}, "P3": {"desc": "G-U2 过", "hit": gp["GU2"]["pass"]},
            "P4": {"desc": "G-U3 过", "hit": gf["GU3"]["pass"]}, "P5": {"desc": "G-U4 过", "hit": gf["GU4"]["pass"]},
            "P6": {"desc": "G-U5 过", "hit": gf["GU5"]["pass"]},
            "P7": {"desc": "指纹 nonprobe cross Δ_M1 由负转正", "value": d_fp, "hit": None if d_fp is None else bool(d_fp > 0)},
            "P8": {"desc": "φ 臂 M0 cross Δ 仍显著（p<0.01）", "value": m0.get("p"), "hit": None if m0.get("p") is None else bool(m0["p"] < 0.01)}}


def markdown_tables(a: dict) -> str:
    f = lambda b, k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
    L = [f"# WP16-TCLSC 阶段 1 n 放大 tables（UCR {a['n_datasets']} 集，7 折，δ={a['delta']}）", "",
         f"φ **{a['verdict_phi']}** | 指纹 **{a['verdict_fp']}**", ""]
    for view in ("all20", "nonprobe"):
        v = a["views"][view]
        arms = list(v["pooled"])
        L += [f"## 视图 {view}（{v['n_instances']} 实例）— normal", "",
              "| 臂 | M1 | M2 | M0 | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in arms:
            b = v["pooled"][arm]
            L.append(f"| {arm} | {f(b,'acc_top10')} | {f(b,'wacc')} | {f(b,'pair_acc')} | {f(b,'rho')} | {f(b,'regret1')} | {f(b,'regret3')} |")
        L += ["", f"### {view} 消融 Δ", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
        for arm in arms:
            for kind in ("cross", "zero"):
                for metric in ("acc_top10", "wacc", "pair_acc"):
                    d = v["deltas"].get(kind, {}).get(arm, {}).get(metric)
                    if not d or d.get("mean_delta") is None:
                        continue
                    L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | {d['wins']}/{d['losses']}/{d['ties']} |")
        L += ["", f"### {view} 逐折 Δ_M1（cross）", ""]
        for arm, bf in v["by_fold_delta"].items():
            L.append(f"- {arm}: " + ", ".join(f"{fo} {('—' if x is None else f'{x:+.3f}')}" for fo, x in bf.items()))
        L += ["", f"### {view} 价值（regret@3 配对，H1 条件臂更低）", ""]
        for t in v["value_tests"]:
            L.append(f"- {t['a']} → {t['b']}: n={t['n']} {t['mean_a']} → {t['mean_b']} p={t['p']} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
        L.append("")
    r = a["ref_same_cost"]
    L += ["## 同成本参照（all20，@120/@290）", "", f"- probe6_top3 {r.get('mean_a')} vs u_fp_diff_hard {r.get('mean_b')}（blind {r.get('blind_mean')}）n={r.get('n')} p={r.get('p')}",
          "", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for gs in (a["gates_phi"], a["gates_fp"]):
        for k, v in gs.items():
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.train:
        cmd_train(args.outdir, arms=args.arms, seeds=args.seeds, folds_sel=args.folds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.outdir, arms=list(ARMS), seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
