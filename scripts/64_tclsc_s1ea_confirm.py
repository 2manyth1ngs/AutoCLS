"""WP16-TCLSC 阶段 1 A′ 确认版：E_A ⊕ φ12 的选择价值与内容依赖，UEA 8 集 7 折 LODO（主）+ UCR 84 新种子（副）。协议 results/tclsc_s1ea_confirm/PROTOCOL.md。

子命令：
  --features   UEA 45 切片的 E_A（训练侧，与 40/42 号 φ 同划分）并入 φ12 → uea/ea_uea.jsonl
  --train      变体目录 uea/（56 号 prepare_variant_dir）→ `33 --train --arms ea_diff_hard eaphi_diff_hard --encoder-mode binned --ctx-file`
  --train-ucr  UCR 84 分组 7 折种子 3/4/5：60 号训练器（u_blind / u_phi）+ 63 号训练器（u_eaphi）同一目录 ucr_s345/
  --rank       两考场排名（normal / cross×3 / zero 只置零 E_A 列），参照臂从 s1phi（UEA）合并
  --analyze    门 C0 / G-C1 / G-C2 / G-C3 + 只报项 + 读数码 + 预测 → analysis.json / tables.md

E_A / φ 只在本脚本与 33 号装配成浮点向量（铁律 7）；标准化统计取模型 meta（33 号按该折训练切片算，铁律 6）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.mech_features import RESPONSE_PROFILE_COLS, RESPONSE_SEEDS, response_profile  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g59 = importlib.import_module("scripts.59_tclsc_s1fp")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(_m33.PHI_COLS)
BUDGETS = pq.BUDGETS
OUT = ROOT / "results" / "tclsc_s1ea_confirm"
SOURCE = pq.OUT                                            # results/mx3_pq
S1PHI_RANK = ROOT / "results" / "tclsc_s1phi" / "binned" / "rankings_s1phi.json"
S1PHI_ANALYSIS = ROOT / "results" / "tclsc_s1phi" / "analysis.json"
S1UCR_FOLDS = ROOT / "results" / "tclsc_s1ucr" / "folds.json"
UEA_ARMS = ("eaphi_diff_hard", "ea_diff_hard")
REF_ARMS = ("blind_diff_hard", "phi_diff_hard")
UCR_ARMS = ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")
UCR_SEEDS = (3, 4, 5)
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
ALPHA = 0.05
C0_TOL = 0.001


# ---------------------------------------------------------------- ctx 行（切片键）
def slice_ctx_rows(table: Mapping[str, Mapping], cols: Sequence[str], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                   cross_map: Optional[Mapping[str, str]] = None, zero_cols: Sequence[str] = (),
                   budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    """{(切片, 预算): 标准化列(供体或自己) ⊕ 预算 one-hot}；zero_cols 里的列置 0（其余列不动）。"""
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    zero_idx = [i for i, c in enumerate(cols) if c in set(zero_cols)]
    out = {}
    for s in slices:
        src = cross_map[s] if cross_map is not None else s
        x, _n = _m33.table_row_vector(table[src], cols, mu, sd)
        if zero_idx:
            x = x.copy()
            x[zero_idx] = 0.0
        for b in budgets:
            out[(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


# ---------------------------------------------------------------- 特征（UEA 切片）
def ea_rows_uea(datasets: Mapping[str, object], phi: Mapping[str, Mapping], *, done: set[str],
                seeds: Sequence[int] = RESPONSE_SEEDS) -> Iterable[dict]:
    """每切片一行：response_profile（训练侧，与 40/42 号 φ 同划分）⊕ φ12 列。"""
    from autocls.data.task_slicer import materialize_task
    from autocls.labeling import run_seed_for, split_task_train_eval
    for name, d in datasets.items():
        for t in d.slices:
            if t.task_id in done:
                continue
            Xt, yt = materialize_task(t, d.X, d.y)
            tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(t.task_id, "train_eval_split"))
            t0 = time.time()
            prof = response_profile(Xt[tr], yt[tr], task_type=t.task_type, seeds=seeds)
            row = {"task_id": t.task_id, "dataset": name, "n_train": int(len(tr)), "seconds": round(time.time() - t0, 1), **prof}
            if t.task_id in phi:
                row.update({c: phi[t.task_id][c] for c in PHI_COLS})
            yield row


def cmd_features(outdir: Path, *, resume: bool) -> None:
    d = outdir / "uea"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "ea_uea.jsonl"
    done = {r["task_id"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
    if not resume and path.exists():
        path.unlink()
    phi = {r["task_id"]: r for r in read_jsonl(_m33.PHI_FILE)}
    ds = pq.load_pq_datasets(pq.PQ_DATASETS)
    with open(path, "a", encoding="utf-8") as fh:
        for row in ea_rows_uea(ds, phi, done=done):
            if any(c not in row for c in PHI_COLS):
                raise KeyError(f"φ 表缺切片 {row['task_id']}")
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[ea uea] {row['task_id']}: n={row['n_train']} {row['seconds']}s " + " ".join(
                f"{c}={row[c]:+.2f}" for c in ("ret_jitter", "ret_highpass05", "ret_inst_norm", "hardness_1nn")), flush=True)


# ---------------------------------------------------------------- 训练
def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool) -> None:
    d = outdir / "uea"
    _g56.prepare_variant_dir(SOURCE, d)
    _m33.cmd_train(d, arms=list(UEA_ARMS), folds=None, seeds=list(seeds), resume=resume, device=device, epochs=epochs,
                   settf_windows=None, encoder_mode="binned", ctx_file=d / "ea_uea.jsonl")


def train_ucr(rows: Sequence[dict], ea: Mapping, phi: Mapping, folds: Sequence[Sequence[str]], outdir: Path, *, seeds: Sequence[int],
              folds_sel: Optional[Sequence[int]], device: str, epochs: Optional[int], resume: bool) -> None:
    """UCR 副考场：60 号训练器（u_blind / u_phi）+ 63 号训练器（u_eaphi）同一目录。"""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "folds.json").write_text(json.dumps([list(f) for f in folds], ensure_ascii=False, indent=1), encoding="utf-8")
    log = outdir / "train.log"
    for idx in (folds_sel if folds_sel else range(len(folds))):
        for arm in UCR_ARMS:
            for seed in seeds:
                p = outdir / "models" / f"fold{idx}" / f"{arm}_s{seed}.pt"
                if resume and p.exists():
                    continue
                if arm in _g63.ARMS:
                    run = _g63.train_fold(rows, ea, phi, folds, idx, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
                else:
                    run = _g60.train_fold(rows, phi, folds, idx, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
                line = f"[fold{idx}/{arm}_s{seed}] {run['train_seconds']}s | bce {run['bce_first']:.4f}→{run['bce_last']:.4f} | pairs {run['train_pairs_last']}"
                print(line, flush=True)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")


def cmd_train_ucr(outdir: Path, *, seeds: Sequence[int], folds_sel: Optional[Sequence[int]], device: str, epochs: Optional[int], resume: bool) -> None:
    rows, ea, phi = _g63._load_all()
    folds = json.loads(S1UCR_FOLDS.read_text(encoding="utf-8"))
    train_ucr(rows, ea, phi, folds, outdir / "ucr_s345", seeds=seeds, folds_sel=folds_sel, device=device, epochs=epochs, resume=resume)


# ---------------------------------------------------------------- 排名
def _load_arm_models(d: Path, target: str, arm: str, seeds: Sequence[int], device: str) -> Optional[list]:
    paths = [d / "models" / target / f"{arm}_s{s}.pt" for s in seeds]
    if not all(p.exists() for p in paths):
        print(f"⚠️ [{target}/{arm}] 模型不全，整臂跳过", flush=True)
        return None
    return [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=BUDGETS) for p in paths]


def rank_uea(outdir: Path, *, seeds: Sequence[int], device: str = "cpu", arms: Optional[Sequence[str]] = None) -> dict:
    arms = tuple(arms) if arms is not None else UEA_ARMS
    d = outdir / "uea"
    table = {r["task_id"]: r for r in read_jsonl(d / "ea_uea.jsonl")}
    rows, _delta = _g57.load_source(SOURCE)
    all_slices = sorted({r["task_id"] for r in rows})
    out: dict[str, dict] = {}
    for target in FOLDS:
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        train_slices = [s for s in all_slices if pq.dataset_of(s) != target]
        catalog, _g, _k = _g34.load_catalog(SOURCE, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        models = {a: m for a in arms if (m := _load_arm_models(d, target, a, seeds, device)) is not None}
        if not models:
            continue
        per: dict = {"normal": {}, "cross": {}, "zero": {}}

        def merge(dst: dict, rk: dict) -> None:
            for b, pb in rk.items():
                for s, pa in pb.items():
                    dst.setdefault(b, {}).setdefault(s, {}).update(pa)

        for a, mm in models.items():
            meta0 = mm[0][1]
            cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
            if cols != tuple(_m33.arm_spec(a).ctx_cols):
                raise ValueError(f"[{target}/{a}] 存档列表与臂表不符")
            rows_fn = lambda cm=None, zero=False, cols=cols, mu=mu, sd=sd: slice_ctx_rows(
                table, cols, test_slices, mu, sd, cross_map=cm, zero_cols=(EA_COLS if zero else ()))
            one = {a: mm}
            merge(per["normal"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn()))
            merge(per["zero"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
            for cs in CROSS_SEEDS:
                cmap = _g58.cross_slice_map(test_slices, train_slices, cs)
                per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
                merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
        out[target] = per
        print(f"[uea/{target}] 臂 {sorted(models)}", flush=True)
    return out


def merge_ref(own: Mapping[str, Mapping], ref: Mapping[str, Mapping], *, arms: Sequence[str] = REF_ARMS) -> dict:
    """把 s1phi 参照臂的 normal 排名（同折同种子已训模型）并进本实验的排名表（按靶名对折）。"""
    out = {f: {"normal": {b: {s: dict(pa) for s, pa in pb.items()} for b, pb in per["normal"].items()},
               "cross": per["cross"], "zero": per["zero"], **({"test": per["test"]} if "test" in per else {})}
           for f, per in own.items()}
    for f, per in out.items():
        if f not in ref:
            raise ValueError(f"参照排名缺折 {f}")
        for b, pb in ref[f]["normal"].items():
            for s, pa in pb.items():
                keep = {a: pa[a] for a in arms if a in pa}
                if keep:
                    per["normal"].setdefault(b, {}).setdefault(s, {}).update(keep)
    return out


def rank_ucr(rows: Sequence[dict], ea: Mapping, phi: Mapping, folds: Sequence[Sequence[str]], outdir: Path, *, seeds: Sequence[int],
             folds_sel: Optional[Sequence[int]] = None, device: str = "cpu") -> dict:
    out = {}
    for idx in (folds_sel if folds_sel else range(len(folds))):
        a = _g60.rank_fold(rows, phi, folds, idx, arms=[x for x in UCR_ARMS if x in _g60.ARMS], seeds=seeds, outdir=outdir, device=device)
        b = _g63.rank_fold(rows, ea, phi, folds, idx, arms=[x for x in UCR_ARMS if x in _g63.ARMS], seeds=seeds, outdir=outdir, device=device)
        per = {"normal": a["normal"], "cross": b["cross"], "zero": b["zero"], "test": a["test"], "probes": a.get("probes")}
        for bud, pb in b["normal"].items():
            for s, pa in pb.items():
                per["normal"].setdefault(bud, {}).setdefault(s, {}).update(pa)
        out[f"fold{idx}"] = per
        print(f"[ucr/fold{idx}] 靶 {folds[idx]}", flush=True)
    return out


def cmd_rank(outdir: Path, *, seeds: Sequence[int], ucr_seeds: Sequence[int], device: str) -> None:
    rk_uea = rank_uea(outdir, seeds=seeds, device=device)
    (outdir / "uea" / "rankings_uea.json").write_text(json.dumps(rk_uea, ensure_ascii=False, indent=1), encoding="utf-8")
    ucr_dir = outdir / "ucr_s345"
    if (ucr_dir / "folds.json").exists():
        rows, ea, phi = _g63._load_all()
        folds = json.loads((ucr_dir / "folds.json").read_text(encoding="utf-8"))
        rk_ucr = rank_ucr(rows, ea, phi, folds, ucr_dir, seeds=ucr_seeds, device=device)
        (ucr_dir / "rankings_ucr_s345.json").write_text(json.dumps(rk_ucr, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 门 / 读数码
def _lower(v: Mapping) -> bool:
    return bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])


def gates_confirm(v_phi: Mapping, v_cross: Mapping, v_ucr: Mapping) -> dict:
    """v_phi = φ → eaphi regret@3 配对（H1 b<a）；v_cross = eaphi normal → cross（H1 cross 更差：mean_b > mean_a）；v_ucr = UCR s345 φ → eaphi。"""
    worse = bool(v_cross.get("p") is not None and v_cross["p"] < ALPHA and v_cross["mean_b"] > v_cross["mean_a"])
    return {"GC1": {"metric": "regret3", "phi": v_phi.get("mean_a"), "eaphi": v_phi.get("mean_b"), "p": v_phi.get("p"), "pass": _lower(v_phi)},
            "GC2": {"metric": "regret3", "normal": v_cross.get("mean_a"), "cross": v_cross.get("mean_b"), "p": v_cross.get("p"), "pass": worse},
            "GC3": {"metric": "regret3", "phi": v_ucr.get("mean_a"), "eaphi": v_ucr.get("mean_b"), "p": v_ucr.get("p"),
                    "pass": (None if v_ucr.get("p") is None else _lower(v_ucr))},
            "thresholds": {"alpha": ALPHA}}


def verdict_confirm(g: Mapping) -> str:
    c1, c2, c3 = g["GC1"]["pass"], g["GC2"]["pass"], g["GC3"]["pass"]
    code = {(True, True): "EA-CONFIRMED", (True, False): "EA-VALUE-NO-CONTENT",
            (False, True): "EA-CONTENT-NO-VALUE", (False, False): "EA-NOT-CONFIRMED"}[(bool(c1), bool(c2))]
    return code + ("?UCR-REP" if c3 is None else ("+UCR-REP" if c3 else "-UCR-REP"))


# ---------------------------------------------------------------- 分析
def _paired_cross(normal: Sequence[dict], cross: Sequence[Sequence[dict]], arm: str, key: str = "3") -> dict:
    """regret@k：normal vs cross（3 种子均值），H1 cross 更差（normal < cross）。"""
    key_of = lambda r: (r["slice"], r["budget"])
    cms = [{key_of(r): r for r in c} for c in cross]
    xs = [(r["regret"][arm + key], np.mean([cm[key_of(r)]["regret"][arm + key] for cm in cms]))
          for r in normal if arm + key in r["regret"] and all(key_of(r) in cm for cm in cms)]
    if not xs:
        return {"arm": arm, "metric": "regret" + key, "n": 0, "mean_a": None, "mean_b": None, "p": None}
    x, y = np.array([v[0] for v in xs]), np.array([v[1] for v in xs])
    return {"arm": arm, "metric": "regret" + key, "n": int(x.size), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "p": float(_g56.paired_wilcoxon_less(x, y)), "worse": int(np.sum(y > x)), "better": int(np.sum(y < x)), "ties": int(np.sum(y == x))}


def _block(rk: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], ctx_arms: Sequence[str], *, eval_fn) -> dict:
    cross_seeds = sorted({cs for per in rk.values() for cs in per["cross"]})
    normal = eval_fn(rk, cat_z, delta, arms, "normal")
    cross = [eval_fn(rk, cat_z, delta, ctx_arms, "cross", cs) for cs in cross_seeds]
    zero = eval_fn(rk, cat_z, delta, ctx_arms, "zero")
    present = [a for a in arms if any(a + "3" in r["regret"] for r in normal)]
    pooled = {}
    for a in present:
        pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
        pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
        pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
    ctx_present = [a for a in ctx_arms if a in present]
    deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_present},
              "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_present}}
    regret_abl = {a: {k: {"cross": _paired_cross(normal, cross, a, k), "zero": _paired_cross(normal, [zero], a, k)} for k in ("1", "3")} for a in ctx_present}
    by_fold = {a: {fo: float(np.mean([r["regret"][a + "3"] for r in normal if r["fold"] == fo])) for fo in sorted({r["fold"] for r in normal})} for a in present}
    return {"n_instances": len(normal), "arms": present, "pooled": pooled, "deltas": deltas, "regret_ablation": regret_abl,
            "by_fold_regret3": by_fold, "per_instance_normal": normal}


def _eval_ucr(rk, cat_z, delta, arms, which, cross_seed=None):
    return _g63._eval(rk, cat_z, delta, arms, which, cross_seed=cross_seed, top_k=_g56.TOP_GOOD)


def cmd_analyze(outdir: Path) -> dict:
    # --- UEA（主）
    rk_own = json.loads((outdir / "uea" / "rankings_uea.json").read_text(encoding="utf-8"))
    rk = merge_ref(rk_own, json.loads(S1PHI_RANK.read_text(encoding="utf-8")), arms=REF_ARMS)
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    uea = _block(rk, cat_z, delta, list(REF_ARMS) + list(UEA_ARMS), list(UEA_ARMS), eval_fn=_g58._eval_all)
    normal = uea["per_instance_normal"]
    pairs = [("phi_diff_hard", "eaphi_diff_hard"), ("blind_diff_hard", "eaphi_diff_hard"), ("phi_diff_hard", "ea_diff_hard"), ("blind_diff_hard", "ea_diff_hard")]
    uea["value_tests"] = {k: [_g58.paired_regret(normal, a, b, key=k) for a, b in pairs if a in uea["arms"] and b in uea["arms"]] for k in ("1", "3")}
    find = lambda tests, a, b: next((t for t in tests if t["a"] == a and t["b"] == b), {"p": None, "mean_a": None, "mean_b": None})
    v_phi = find(uea["value_tests"]["3"], "phi_diff_hard", "eaphi_diff_hard")
    v_cross = uea["regret_ablation"].get("eaphi_diff_hard", {}).get("3", {}).get("cross", {"p": None, "mean_a": None, "mean_b": None})
    # C0
    ref_an = json.loads(S1PHI_ANALYSIS.read_text(encoding="utf-8")) if S1PHI_ANALYSIS.exists() else None
    c0 = {}
    for a in REF_ARMS:
        ref_v = None if ref_an is None else ref_an["pooled"].get(a, {}).get("pair_acc")
        got = uea["pooled"].get(a, {}).get("pair_acc")
        c0[a] = {"ref": ref_v, "got": got, "pass": None if (ref_v is None or got is None) else bool(abs(ref_v - got) <= C0_TOL)}
    c0["pass"] = all(v["pass"] for k, v in c0.items() if k != "pass") if all(v["pass"] is not None for k, v in c0.items() if k != "pass") else None
    # --- UCR s345（副）
    ucr = None
    v_ucr = {"p": None, "mean_a": None, "mean_b": None}
    ucr_rk_path = outdir / "ucr_s345" / "rankings_ucr_s345.json"
    if ucr_rk_path.exists():
        rows_u, _ea, _phi = _g63._load_all()
        rk_u = json.loads(ucr_rk_path.read_text(encoding="utf-8"))
        cat_u = _g56.zscore_within(rows_u)
        delta_u = {(d, int(b)): _g60.DELTA_UCR for d in {r["dataset"] for r in rows_u} for b in BUDGETS}
        ucr = _block(rk_u, cat_u, delta_u, list(UCR_ARMS), ["u_eaphi_diff_hard"], eval_fn=_eval_ucr)
        nu = ucr["per_instance_normal"]
        ucr["value_tests"] = {k: [_g58.paired_regret(nu, a, b, key=k) for a, b in (("u_phi_diff_hard", "u_eaphi_diff_hard"), ("u_blind_diff_hard", "u_eaphi_diff_hard"),
                                                                                  ("u_blind_diff_hard", "u_phi_diff_hard"))] for k in ("1", "3")}
        v_ucr = find(ucr["value_tests"]["3"], "u_phi_diff_hard", "u_eaphi_diff_hard")
    g = gates_confirm(v_phi, v_cross, v_ucr)
    a = {"protocol": "results/tclsc_s1ea_confirm/PROTOCOL.md", "uea": uea, "ucr_s345": ucr, "c0": c0, "gates": g, "verdict": verdict_confirm(g)}
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    pu = uea["pooled"]
    print(f"{a['verdict']} | UEA " + " | ".join(f"{arm} M0 {pu[arm]['pair_acc']:.3f} r1 {pu[arm]['regret1']:.4f} r3 {pu[arm]['regret3']:.4f}" for arm in uea["arms"])
          + (f" | UCR s345 " + " | ".join(f"{arm} r3 {ucr['pooled'][arm]['regret3']:.4f}" for arm in ucr["arms"]) if ucr else ""), flush=True)
    return a


def predictions(a: dict) -> dict:
    g, pu = a["gates"], a["uea"]["pooled"]
    find = lambda tests, x, y: next((t for t in tests if t["a"] == x and t["b"] == y), {"p": None, "mean_a": None, "mean_b": None})
    r1 = find(a["uea"]["value_tests"]["1"], "phi_diff_hard", "eaphi_diff_hard")
    r3b = find(a["uea"]["value_tests"]["3"], "blind_diff_hard", "eaphi_diff_hard")
    m0e, m0p = pu.get("eaphi_diff_hard", {}).get("pair_acc"), pu.get("phi_diff_hard", {}).get("pair_acc")
    return {"P1": {"desc": "C0 逐位", "hit": a["c0"].get("pass")},
            "P2": {"desc": "G-C1 过", "hit": g["GC1"]["pass"]}, "P3": {"desc": "G-C2 过", "hit": g["GC2"]["pass"]},
            "P4": {"desc": "G-C3 过", "hit": g["GC3"]["pass"]},
            "P5": {"desc": "UEA eaphi M0 ≥ φ", "value": [m0e, m0p], "hit": None if m0e is None or m0p is None else bool(m0e >= m0p)},
            "P6": {"desc": "UEA eaphi regret@1 < φ（p<0.05）", "value": [r1.get("mean_a"), r1.get("mean_b"), r1.get("p")], "hit": _lower(r1)},
            "P7": {"desc": "UEA eaphi regret@3 < blind（p<0.05）", "value": [r3b.get("mean_a"), r3b.get("mean_b"), r3b.get("p")], "hit": _lower(r3b)}}


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def markdown_tables(a: dict) -> str:
    L = [f"# WP16-TCLSC 阶段 1 A′ 确认版 tables", "", f"读数码 **{a['verdict']}**", ""]
    for title, blk in (("UEA 8 集 7 折 LODO（主）", a["uea"]), ("UCR 84 种子 3/4/5（副）", a.get("ucr_s345"))):
        if not blk:
            continue
        L += [f"## {title}（{blk['n_instances']} 实例）", "", "| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|"]
        for arm, b in blk["pooled"].items():
            L.append(f"| {arm} | {_fmt(b.get('acc_top10'), 3)} | {_fmt(b.get('wacc'), 3)} | {_fmt(b.get('pair_acc'), 3)} | {_fmt(b['regret1'])} | {_fmt(b['regret3'])} |")
        L += ["", "### 价值（配对 Wilcoxon，H1 b < a）", ""]
        for k in ("3", "1"):
            for t in blk["value_tests"][k]:
                L.append(f"- regret@{k} {t['a']} → {t['b']}: n={t['n']} {_fmt(t['mean_a'])} → {_fmt(t['mean_b'])} p={t['p']:.3g} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
        L += ["", "### 消融下的 regret（H1 消融更差）", ""]
        for arm, kk in blk["regret_ablation"].items():
            for k, d in kk.items():
                for kind, t in d.items():
                    if t.get("p") is None:
                        continue
                    L.append(f"- {arm} regret@{k} {kind}: {_fmt(t['mean_a'])} → {_fmt(t['mean_b'])} p={t['p']:.3g} 更差/更好/平 {t['worse']}/{t['better']}/{t['ties']}")
        L += ["", "### 消融 Δ（配对精度）", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
        for kind in ("cross", "zero"):
            for arm, dd in blk["deltas"][kind].items():
                for metric, d in dd.items():
                    if d.get("mean_delta") is None:
                        continue
                    L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | {d['wins']}/{d['losses']}/{d['ties']} |")
        L += ["", "### 逐折 regret@3", ""]
        for arm, bf in blk["by_fold_regret3"].items():
            L.append(f"- {arm}: " + ", ".join(f"{fo} {v:.4f}" for fo, v in bf.items()))
        L.append("")
    L += ["## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    for k, v in a["c0"].items():
        if k != "pass":
            L.append(f"| C0 {k} | ref {v['ref']} got {v['got']} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-ucr", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--ucr-seeds", nargs="+", type=int, default=list(UCR_SEEDS))
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="--train-ucr 的折序号子集")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.features, args.train, args.train_ucr, args.rank, args.analyze)):
        build_parser().error("至少指定 --features / --train / --train-ucr / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.features:
        cmd_features(args.outdir, resume=args.resume)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.train_ucr:
        cmd_train_ucr(args.outdir, seeds=args.ucr_seeds, folds_sel=args.folds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds, ucr_seeds=args.ucr_seeds, device="cpu")
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
