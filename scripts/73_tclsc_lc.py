"""WP16 LC 检验：比较器的标签学习曲线（配置轴 / 切片轴 / 数据集轴）。协议 results/tclsc_lc/PROTOCOL.md。零新 R'。

子命令：--train [--axes config slice dataset] / --rank / --analyze。满量点复用已有排名（T 检验 UEA、s1phi、s1et ucr、s1ucr）。
臂：E_T = ft60_hard / u_ft60_hard（E_T@60 ⊕ φ12 ⊕ b，field 头）；blind = blind_diff_hard / u_blind_diff_hard。
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import TCLSCEngine  # noqa: E402
from autocls.data.mech_features import rep_cols_at  # noqa: E402
from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
PHI_COLS = tuple(_m33.PHI_COLS)
ET_COLS = rep_cols_at(60) + PHI_COLS
OUT = ROOT / "results" / "tclsc_lc"
SOURCE = pq.OUT
S1ET = ROOT / "results" / "tclsc_s1et"
S1PHI = ROOT / "results" / "tclsc_s1phi"
S1UCR = ROOT / "results" / "tclsc_s1ucr"
CONFIG_POINTS = (6, 10, 14, 20)
SLICE_POINTS = (2, 4, 6)
DATASET_POINTS = (7, 14, 28, 56, 72)
UEA_ARMS = ("ft60_hard", "blind_diff_hard")
UCR_ARMS = ("u_ft60_hard", "u_blind_diff_hard")
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
RISE_MIN = 0.001
SUB_SEED = 0


# ---------------------------------------------------------------- 子采样
def _perm(keys: Sequence[str], seed) -> list[str]:
    keys = sorted(keys)
    rng = np.random.default_rng(seed if isinstance(seed, int) else [abs(hash(x)) % (2 ** 31) for x in seed])
    return [keys[i] for i in rng.permutation(len(keys))]


def subsample_configs(rows: Sequence[dict], *, k: int, seed: int = SUB_SEED) -> list[dict]:
    """每 (切片, 预算) 保留 k 条：k//2 条 shared（全局固定子集）+ k − k//2 条该切片的 random（切片种子）。k ≥ 20 原样。"""
    shared_all = sorted({r["cls_key"] for r in rows if r["origin"] == ORIGIN_SHARED})
    n_s = k // 2
    keep_shared = set(_perm(shared_all, seed)[:n_s]) if n_s < len(shared_all) else set(shared_all)
    per_slice: dict[str, set] = {}
    for s in sorted({r["task_id"] for r in rows}):
        rnd = sorted({r["cls_key"] for r in rows if r["task_id"] == s and r["origin"] != ORIGIN_SHARED})
        n_r = k - n_s
        per_slice[s] = set(_perm(rnd, (str(seed), s))[:n_r]) if n_r < len(rnd) else set(rnd)
    return [r for r in rows if (r["cls_key"] in keep_shared if r["origin"] == ORIGIN_SHARED else r["cls_key"] in per_slice[r["task_id"]])]


def subsample_slices(rows: Sequence[dict], *, s: int, seed: int = SUB_SEED) -> list[dict]:
    keep: set = set()
    for d in sorted({r["dataset"] for r in rows}):
        sl = sorted({r["task_id"] for r in rows if r["dataset"] == d})
        keep |= set(_perm(sl, (str(seed), d))[:s])
    return [r for r in rows if r["task_id"] in keep]


def subsample_datasets(train: Sequence[str], *, n: int, seed: int = SUB_SEED) -> list[str]:
    train = sorted(train)
    if n >= len(train):
        return train
    return sorted(_perm(train, seed)[:n])


# ---------------------------------------------------------------- 判读
def axis_verdict(points: Sequence[tuple]) -> str:
    """points = [(size, E regret@3, blind regret@3)]；RISING = 满量点比次大点低 ≥ RISE_MIN 且 E 增益（blind − E）仍在涨（严格大于；
    增益持平 = 涨的是任务盲部分 → FLAT，单测的 both 例）。"""
    pts = sorted(points)
    (_sb, eb, bb), (_sa, ea, ba) = pts[-2], pts[-1]
    return "RISING" if (ea <= eb - RISE_MIN and (ba - ea) > (bb - eb) + 1e-9) else "FLAT"


def verdict_lc(v: Mapping[str, str]) -> str:
    on = [k.upper() for k in ("config", "slice", "dataset") if v.get(k) == "RISING"]
    return "LC-" + "+".join(on) if on else "LC-SATURATED"


# ---------------------------------------------------------------- UEA 折内训练（子采样）
def train_fold_uea(rows: Sequence[dict], delta: Mapping, table: Mapping[str, Mapping], *, target: str, arm: str, seed: int, outdir: Path,
                   device: str, epochs: Optional[int], tag: str) -> dict:
    spec = _m33.arm_spec(arm)
    train_rows = [r for r in rows if r["dataset"] != target]
    slices = sorted({r["task_id"] for r in train_rows})
    # 与 33 号 run_arm 同一构造式（预算主序、剔 GGS），只是缓存换成子采样后的行
    recs = pq.build_records(_g60.rows_to_cache(train_rows), _g60.enc(), budgets=pq.BUDGETS, slices=slices)
    c = dict(_g60.cfg())
    if epochs:
        c["epochs"], c["curriculum_epochs"] = int(epochs), min(int(c["curriculum_epochs"]), max(1, int(epochs) // 2))
    cls_dim = int(recs[0].vec.shape[0])
    ctx_map, meta_extra, cdim = None, None, 0
    if spec.ctx_kind == "table_budget":
        cols = tuple(spec.ctx_cols)
        mu, sd = _m33.table_standardizer(table, slices, cols)
        ctx_map = {}
        for s in slices:
            x, _n = _m33.table_row_vector(table[s], cols, mu, sd)
            for b in pq.BUDGETS:
                ctx_map[pq.engine_task_id(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), pq.BUDGETS)]).astype(np.float32)
        meta_extra = {"ctx_cols": list(cols), "ctx_mu": [float(v) for v in mu], "ctx_sd": [float(v) for v in sd], "n_nan": {}}
        cdim = len(cols) + len(pq.BUDGETS)
    model = _m33.build_model(spec, cls_dim=cls_dim, ctx_dim=cdim, nz=256, dim_input=256, cfg=c, seed=seed)
    engine = TCLSCEngine(model, {}, device=device, lr=c["lr"], batch_size=c["batch_size"], seed=seed, context_of=ctx_map)
    t0 = time.time()
    hist = engine.train(recs, epochs=c["epochs"], curriculum_epochs=c["curriculum_epochs"],
                        better_fn=lambda a, b: is_better(a, b, TASK_CLASSIFICATION), delta=pq.delta_map_for_fold(delta, slices, pq.BUDGETS))
    p = outdir / "models" / tag / target / f"{arm}_s{seed}.pt"
    _m33.save_model(p, model, spec, fold=target, seed=seed, cls_dim=cls_dim, ctx_dim=cdim, cfg=c, nz=256, settf_windows=0, settf_t_max=0,
                    budgets=pq.BUDGETS, ctx_encoding="onehot", encoder_mode="binned", head=spec.head, ctx_kind=spec.ctx_kind, phi_stats=meta_extra)
    run = {"tag": tag, "test": target, "arm": arm, "seed": seed, "n_train_slices": len(slices), "n_train_records": len(recs),
           "train_seconds": round(time.time() - t0, 1), "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"], "train_pairs_last": hist[-1]["n_pairs"]}
    rp = outdir / "runs" / tag / target / f"{arm}_s{seed}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


def _uea_table() -> dict:
    return {r["task_id"]: r for r in read_jsonl(S1ET / "uea" / "ctx_uea.jsonl")}


def _log(outdir: Path, line: str) -> None:
    print(line, flush=True)
    with open(outdir / "train.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cmd_train(outdir: Path, *, axes: Sequence[str], seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool,
              folds_sel: Optional[Sequence[str]] = None, points_sel: Optional[Sequence[int]] = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    keep_pt = (lambda x: True) if not points_sel else (lambda x: x in set(points_sel))
    keep_fold = (lambda f: True) if not folds_sel else (lambda f: f in set(folds_sel))
    if "config" in axes or "slice" in axes:
        rows, delta = _g57.load_source(SOURCE)
        table = _uea_table()
        points = ([("config", k, subsample_configs(rows, k=k)) for k in CONFIG_POINTS if k < 20 and keep_pt(k)] if "config" in axes else []) + \
                 ([("slice", s, subsample_slices(rows, s=s)) for s in SLICE_POINTS if s < 6 and keep_pt(s)] if "slice" in axes else [])
        for axis, size, sub in points:
            tag = f"{'K' if axis == 'config' else 'S'}{size}"
            for target in [t for t in FOLDS if keep_fold(t)]:
                for arm in UEA_ARMS:
                    for seed in seeds:
                        if resume and (outdir / "models" / tag / target / f"{arm}_s{seed}.pt").exists():
                            continue
                        run = train_fold_uea(sub, delta, table, target=target, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs, tag=tag)
                        _log(outdir, f"[{tag}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")
    if "dataset" in axes:
        rows_u, ea, phi = _g63._load_all()
        folds = json.loads((S1UCR / "folds.json").read_text(encoding="utf-8"))
        for n in DATASET_POINTS:
            if n >= 72 or not keep_pt(n):
                continue
            d = outdir / f"ucr_n{n}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "folds.json").write_text(json.dumps(folds, ensure_ascii=False), encoding="utf-8")
            for idx, test in enumerate(folds):
                if not keep_fold(f"fold{idx}"):
                    continue
                train_all = sorted({r["dataset"] for r in rows_u} - set(test))
                keep = set(subsample_datasets(train_all, n=n)) | set(test)
                sub = [r for r in rows_u if r["dataset"] in keep]
                for arm in UCR_ARMS:
                    for seed in seeds:
                        if resume and (d / "models" / f"fold{idx}" / f"{arm}_s{seed}.pt").exists():
                            continue
                        if arm in _g63.ARMS:
                            run = _g63.train_fold(sub, ea, phi, folds, idx, arm=arm, seed=seed, outdir=d, device=device, epochs=epochs)
                        else:
                            run = _g60.train_fold(sub, phi, folds, idx, arm=arm, seed=seed, outdir=d, device=device, epochs=epochs)
                        _log(outdir, f"[n{n}/fold{idx}/{arm}_s{seed}] {run['train_seconds']}s n_train {len(run['train_datasets'])} pairs {run['train_pairs_last']}")


# ---------------------------------------------------------------- 排名
def rank_uea_tag(outdir: Path, tag: str, *, seeds: Sequence[int], device: str = "cpu") -> dict:
    table = _uea_table()
    rows, _delta = _g57.load_source(SOURCE)
    all_slices = sorted({r["task_id"] for r in rows})
    out = {}
    for target in FOLDS:
        if not (outdir / "models" / tag / target).exists():
            continue
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        catalog, _g, _k = _g34.load_catalog(SOURCE, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        per: dict = {"normal": {}}
        for arm in UEA_ARMS:
            paths = [outdir / "models" / tag / target / f"{arm}_s{s}.pt" for s in seeds]
            if not all(p.exists() for p in paths):
                print(f"⚠️ [{tag}/{target}/{arm}] 模型不全", flush=True)
                continue
            mm = [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in paths]
            if _m33.arm_spec(arm).ctx_kind == "table_budget":
                meta0 = mm[0][1]
                rows_ctx = _g64.slice_ctx_rows(table, tuple(meta0["ctx_cols"]), test_slices, np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"]))
                rk = _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_ctx)
            else:
                rk = _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device)
            for b, pb in rk.items():
                for s, pa in pb.items():
                    per["normal"].setdefault(b, {}).setdefault(s, {}).update(pa)
        out[target] = per
    return out


def cmd_rank(outdir: Path, *, seeds: Sequence[int]) -> None:
    rk: dict = {}
    for tag in [f"K{k}" for k in CONFIG_POINTS if k < 20] + [f"S{s}" for s in SLICE_POINTS if s < 6]:
        if (outdir / "models" / tag).exists():
            rk[tag] = rank_uea_tag(outdir, tag, seeds=seeds)
            print(f"[rank] {tag}", flush=True)
    (outdir / "rankings_uea.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    rows_u, ea, phi = _g63._load_all()
    for n in DATASET_POINTS:
        d = outdir / f"ucr_n{n}"
        if not (d / "folds.json").exists():
            continue
        folds = json.loads((d / "folds.json").read_text(encoding="utf-8"))
        out = {}
        for idx in range(len(folds)):
            if not (d / "models" / f"fold{idx}").exists():
                continue
            a = _g60.rank_fold(rows_u, phi, folds, idx, arms=["u_blind_diff_hard"], seeds=seeds, outdir=d)
            b = _g63.rank_fold(rows_u, ea, phi, folds, idx, arms=["u_ft60_hard"], seeds=seeds, outdir=d)
            per = {"normal": a["normal"], "test": a["test"]}
            for bud, pb in b["normal"].items():
                for s, pa in pb.items():
                    per["normal"].setdefault(bud, {}).setdefault(s, {}).update(pa)
            out[f"fold{idx}"] = per
        (d / "rankings.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[rank] ucr n{n}", flush=True)


# ---------------------------------------------------------------- 分析
def _pooled(rows_eval: Sequence[dict], arm: str) -> dict:
    by_fold = {}
    for f in sorted({r["fold"] for r in rows_eval}):
        v = [r["regret"][arm + "3"] for r in rows_eval if r["fold"] == f and arm + "3" in r["regret"]]
        if v:
            by_fold[f] = round(float(np.mean(v)), 4)
    return {"regret3": float(np.mean([r["regret"][arm + "3"] for r in rows_eval if arm + "3" in r["regret"]])),
            "regret1": float(np.mean([r["regret"][arm + "1"] for r in rows_eval if arm + "1" in r["regret"]])),
            "M1": _g57._pooled(rows_eval, arm, "acc_top10"), "M0": _g57._pooled(rows_eval, arm, "pair_acc"), "n": len(rows_eval),
            "regret3_by_fold": by_fold}


def cmd_analyze(outdir: Path) -> dict:
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    rk = json.loads((outdir / "rankings_uea.json").read_text(encoding="utf-8"))
    full = _g64.merge_ref(json.loads((S1ET / "uea" / "rankings_uea.json").read_text(encoding="utf-8")),
                          json.loads((S1PHI / "binned" / "rankings_s1phi.json").read_text(encoding="utf-8")), arms=("blind_diff_hard",))
    curves: dict = {"config": [], "slice": []}
    for axis, pts, pref in (("config", CONFIG_POINTS, "K"), ("slice", SLICE_POINTS, "S")):
        for size in pts:
            tag = f"{pref}{size}"
            r = full if (axis == "config" and size == 20) or (axis == "slice" and size == 6) else rk.get(tag)
            if r is None:
                continue
            ev = _g58._eval_all({t: {"normal": per["normal"], "cross": {}, "zero": {}} for t, per in r.items()}, cat_z, delta, list(UEA_ARMS), "normal")
            curves[axis].append({"size": size, **{a: _pooled(ev, a) for a in UEA_ARMS}})
    rows_u, _ea, _phi = _g63._load_all()
    cat_u = _g56.zscore_within(rows_u)
    delta_u = {(d, int(b)): _g60.DELTA_UCR for d in {r["dataset"] for r in rows_u} for b in pq.BUDGETS}
    curves["dataset"] = []
    for n in DATASET_POINTS:
        if n == 72:
            r = _g63.merge_ref_rankings(json.loads((S1ET / "ucr" / "rankings_ea.json").read_text(encoding="utf-8")),
                                        json.loads((S1UCR / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_blind_diff_hard",))
        else:
            p = outdir / f"ucr_n{n}" / "rankings.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
        ev = _g63._eval({f: {"normal": per["normal"], "cross": {}, "zero": {}, "test": per["test"]} for f, per in r.items()}, cat_u, delta_u, list(UCR_ARMS), "normal", top_k=_g56.TOP_GOOD)
        curves["dataset"].append({"size": n, **{a: _pooled(ev, a) for a in UCR_ARMS}})
    ref = {"config": (0.0207, 0.0294), "slice": (0.0207, 0.0294), "dataset": (0.0197, 0.0220)}    # PROTOCOL §0
    c0 = {}
    for axis, (e_arm, b_arm) in (("config", UEA_ARMS), ("slice", UEA_ARMS), ("dataset", UCR_ARMS)):
        full = [c for c in curves[axis] if c["size"] == max({"config": CONFIG_POINTS, "slice": SLICE_POINTS, "dataset": DATASET_POINTS}[axis])]
        if full:
            e, b = full[0][e_arm]["regret3"], full[0][b_arm]["regret3"]
            c0[axis] = {"E_T": round(e, 4), "blind": round(b, 4), "ref": ref[axis], "pass": abs(e - ref[axis][0]) < 0.001 and abs(b - ref[axis][1]) < 0.001}
    verd = {}
    for axis, (e_arm, b_arm) in (("config", UEA_ARMS), ("slice", UEA_ARMS), ("dataset", UCR_ARMS)):
        pts = [(c["size"], c[e_arm]["regret3"], c[b_arm]["regret3"]) for c in curves[axis]]
        verd[axis] = axis_verdict(pts) if len(pts) >= 2 else None
    a = {"protocol": "results/tclsc_lc/PROTOCOL.md", "curves": curves, "c0": c0, "axis_verdict": verd,
         "verdict": verdict_lc({k: (v or "FLAT") for k, v in verd.items()})}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], verd, flush=True)
    return a


# ---------------------------------------------------------------- 事后：换种子复验（协议外，REPORT 单列）
REP_TAGS = ("S2", "S4", "S6", "K14")            # S6 = K20 = 满量


def cmd_replicate(outdir: Path, *, seeds: Sequence[int], device: str, resume: bool) -> dict:
    """切片轴三点 + 配置轴 K14（满量点即 S6）用新种子重训 → 排名 → 同口径读数，写 replicate_s<seeds>.json / .md。"""
    tagdir = outdir / ("rep_s" + "".join(str(x) for x in seeds))
    tagdir.mkdir(parents=True, exist_ok=True)
    rows, delta = _g57.load_source(SOURCE)
    table = _uea_table()
    subs = {"S2": subsample_slices(rows, s=2), "S4": subsample_slices(rows, s=4), "S6": list(rows), "K14": subsample_configs(rows, k=14)}
    for tag in REP_TAGS:
        for target in FOLDS:
            for arm in UEA_ARMS:
                for seed in seeds:
                    if resume and (tagdir / "models" / tag / target / f"{arm}_s{seed}.pt").exists():
                        continue
                    run = train_fold_uea(subs[tag], delta, table, target=target, arm=arm, seed=seed, outdir=tagdir, device=device, epochs=None, tag=tag)
                    _log(tagdir, f"[{tag}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")
    cat_z = _g56.zscore_within(rows)
    out = {"seeds": list(seeds), "points": {}}
    for tag in REP_TAGS:
        rk = rank_uea_tag(tagdir, tag, seeds=seeds)
        ev = _g58._eval_all({t: {"normal": per["normal"], "cross": {}, "zero": {}} for t, per in rk.items()}, cat_z, delta, list(UEA_ARMS), "normal")
        out["points"][tag] = {a: _pooled(ev, a) for a in UEA_ARMS}
    e, b = UEA_ARMS
    pts_s = [(int(t[1:]), out["points"][t][e]["regret3"], out["points"][t][b]["regret3"]) for t in ("S2", "S4", "S6")]
    pts_k = [(14, out["points"]["K14"][e]["regret3"], out["points"]["K14"][b]["regret3"]), (20, out["points"]["S6"][e]["regret3"], out["points"]["S6"][b]["regret3"])]
    out["axis_verdict"] = {"slice": axis_verdict(pts_s), "config_14_20": axis_verdict(pts_k)}
    (tagdir / "replicate.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    L = [f"# 事后换种子复验（seeds {list(seeds)}）：切片轴 S2/S4/S6 + 配置轴 K14（S6 = K20 满量）", "",
         f"判读 {out['axis_verdict']}", "", "| 点 | n | E_T regret@3 | blind regret@3 | E 增益 | E_T regret@1 | blind regret@1 | E_T M1 | blind M1 | E_T M0 | blind M0 |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in REP_TAGS:
        pe, pb = out["points"][t][e], out["points"][t][b]
        L.append(f"| {t} | {pe['n']} | {pe['regret3']:.4f} | {pb['regret3']:.4f} | {pb['regret3'] - pe['regret3']:+.4f} | {pe['regret1']:.4f} | {pb['regret1']:.4f} | "
                 f"{pe['M1']:.3f} | {pb['M1']:.3f} | {pe['M0']:.3f} | {pb['M0']:.3f} |")
    (tagdir / "replicate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(out["axis_verdict"], flush=True)
    return out


def markdown_tables(a: dict) -> str:
    L = ["# LC 检验 tables：比较器标签学习曲线", "", f"读数码 **{a['verdict']}**（{a['axis_verdict']}）", ""]
    for axis, title, (e_arm, b_arm) in (("config", "配置轴（UEA，每任务 K 条）", UEA_ARMS), ("slice", "切片轴（UEA，每集 S 个切片）", UEA_ARMS), ("dataset", "数据集轴（UCR，训练侧 n 集）", UCR_ARMS)):
        L += [f"## {title}", "", "| 点 | n | E_T regret@3 | blind regret@3 | E 增益 | E_T regret@1 | blind regret@1 | E_T M1 | blind M1 | E_T M0 | blind M0 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in a["curves"][axis]:
            e, b = c[e_arm], c[b_arm]
            L.append(f"| {c['size']} | {e['n']} | {e['regret3']:.4f} | {b['regret3']:.4f} | {b['regret3'] - e['regret3']:+.4f} | {e['regret1']:.4f} | {b['regret1']:.4f} | "
                     f"{e['M1']:.3f} | {b['M1']:.3f} | {e['M0']:.3f} | {b['M0']:.3f} |")
        L.append("")
        folds = sorted({f for c in a["curves"][axis] for f in c[e_arm]["regret3_by_fold"]})
        if folds:
            L += ["逐折 regret@3（E_T / blind）：", "", "| 点 | " + " | ".join(folds) + " |", "|---:|" + "---:|" * len(folds)]
            for c in a["curves"][axis]:
                L.append(f"| {c['size']} | " + " | ".join(f"{c[e_arm]['regret3_by_fold'].get(f, float('nan')):.3f} / {c[b_arm]['regret3_by_fold'].get(f, float('nan')):.3f}" for f in folds) + " |")
            L.append("")
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--replicate", action="store_true", help="事后：切片轴 + K14 换种子复验（--seeds 给新种子）")
    ap.add_argument("--axes", nargs="+", default=["config", "slice", "dataset"], choices=["config", "slice", "dataset"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--folds", nargs="+", default=None, help="冒烟限折：UEA 靶名 / UCR fold<idx>")
    ap.add_argument("--points", nargs="+", type=int, default=None, help="冒烟限点：K / S / n 的取值")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.train, args.rank, args.analyze, args.replicate)):
        build_parser().error("至少指定 --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.train:
        cmd_train(args.outdir, axes=args.axes, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume,
                  folds_sel=args.folds, points_sel=args.points)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.outdir)
    if args.replicate:
        cmd_replicate(args.outdir, seeds=args.seeds, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
