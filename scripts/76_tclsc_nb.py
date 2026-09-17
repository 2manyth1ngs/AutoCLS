"""WP16 NB1 检验：源侧邻域标签 + 开放空间排序。协议 results/tclsc_nb/PROTOCOL.md。

子命令（按执行顺序）：
  --sample    邻域目录：中心 = k3 先验前 5 + GGS；固定邻居 4（前 2 中心 × h∈{1,2}，全切片同）+ 切片邻居 6（该切片 k3 前 2 为中心，h=1/1/2/2/3/3）→ nb_catalog.json
  --rprime    邻居轨迹（32 号执行体，canonical 种子，三截点）→ rprime_nb.jsonl；--verify
  --labels    训练行（shared k3 均值 + 邻居 k1）/ 真值行（全部 k1）+ 净空统计
  --train     ft60_hard / blind_diff_hard，7 折 × 6 种子（73 号 train_fold_uea）
  --rank      考侧 12 切片 × 20 条：normal / cross(3) / zero + prior；--analyze 门 / 读数码 / 表
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED, cls_key, run_seed_for  # noqa: E402
from autocls.oom_guard import ON_OOM_SKIP  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g59 = importlib.import_module("scripts.59_tclsc_s1fp")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g73 = importlib.import_module("scripts.73_tclsc_lc")
_g74 = importlib.import_module("scripts.74_tclsc_slices")
_g75 = importlib.import_module("scripts.75_tclsc_k3")
OUT = ROOT / "results" / "tclsc_nb"
K3 = _g75.OUT
SOURCE = pq.OUT
FIXED_SEED = 7
N_CENTERS = 5                       # k3 先验前 5（+ GGS）
H_FIXED = (1, 2)                    # 前 2 中心各一条
H_SLICE = (1, 2, 3)                 # 切片前 2 中心各三条
RATIO_STEP_MAX = 2
TOP_GOOD_NB = 10                    # 20 条候选的好半区
ARMS = ("ft60_hard", "blind_diff_hard")
E_ARM, B_ARM = ARMS
PRIOR = "prior"
TAG = "NB"
SEED_GROUPS = ((0, 1, 2), (3, 4, 5))
CROSS_SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
ALPHA = 0.05
G2_MIN, G3_OVER_MAX, G1_POS, G1_DELTA = 0.003, 0.15, 0.40, 0.20


# ---------------------------------------------------------------- 变异
def is_ratio_field(space: CLSSpace, name: str) -> bool:
    ch = space.choices(name)
    return len(ch) >= 5 and all(isinstance(v, float) for v in ch)


def _hamming(a: CLS, b: CLS, fields: Sequence[str]) -> int:
    return sum(1 for f in fields if getattr(a, f) != getattr(b, f))


def mutate(base: CLS, h: int, rng: np.random.Generator, space: CLSSpace, max_tries: int = 500) -> CLS:
    """改 h 个基础字段：比例档只走 ±1 / ±2 档，类别档任取另一档；套约束后实际汉明必须等于 h。"""
    fields = list(space.field_names)
    base_vals = {f: getattr(base, f) for f in space.all_field_names}
    for _ in range(max_tries):
        picks = [fields[i] for i in rng.choice(len(fields), size=h, replace=False)]
        vals = dict(base_vals)
        for f in picks:
            ch = space.choices(f)
            cur = vals[f]
            if is_ratio_field(space, f):
                i = ch.index(cur) if cur in ch else int(rng.integers(len(ch)))
                cands = [j for j in range(len(ch)) if 0 < abs(j - i) <= RATIO_STEP_MAX]
                vals[f] = ch[cands[int(rng.integers(len(cands)))]]
            else:
                others = [v for v in ch if v != cur]
                if not others:
                    break
                vals[f] = others[int(rng.integers(len(others)))]
        else:
            vals.update(space.inactive_overrides(vals))
            m = CLS(**vals)
            if space.validate(m) and _hamming(base, m, fields) == h and cls_key(m) != cls_key(base):
                return m
    raise RuntimeError(f"汉明 {h} 变异 {max_tries} 次都不合法")


def _exclusion_keys(space: CLSSpace) -> set[str]:
    shared = pq.load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    return {k for _c, k in shared} | {pq.cls_key(pq.ggs_cls())}


def fixed_neighbors(space: CLSSpace, centers: Sequence[CLS], *, seed: int = FIXED_SEED) -> list[tuple[CLS, CLS, int]]:
    """前 2 个中心 × h ∈ H_FIXED 各一条，固定种子，全切片共用。"""
    taken = _exclusion_keys(space)
    out = []
    for i, ctr in enumerate(list(centers)[:2]):
        for h in H_FIXED:
            rng = np.random.default_rng([seed, i, h])
            while True:
                m = mutate(ctr, h, rng, space)
                if cls_key(m) not in taken:
                    break
            taken.add(cls_key(m))
            out.append((ctr, m, h))
    return out


def slice_neighbors(space: CLSSpace, slice_id: str, centers: Sequence[CLS], *, taken: set[str], seed: int = FIXED_SEED) -> list[tuple[CLS, CLS, int]]:
    """该切片前 2 中心 × h ∈ H_SLICE 各一条，task_id 派生种子；与 taken 去重（就地扩充）。"""
    rng = np.random.default_rng([seed, run_seed_for(slice_id, "nb_cls")])
    out = []
    for ctr in list(centers)[:2]:
        for h in H_SLICE:
            while True:
                m = mutate(ctr, h, rng, space)
                if cls_key(m) not in taken:
                    break
            taken.add(cls_key(m))
            out.append((ctr, m, h))
    return out


def slice_centers(rows: Sequence[dict], slice_id: str, *, n: int = 2) -> list[str]:
    """该切片 shared 行三截点 within-(切片, 截点) z 均值前 n。"""
    cat_z = _g56.zscore_within([r for r in rows if r["task_id"] == slice_id and r["origin"] == ORIGIN_SHARED])
    acc: dict[str, list] = {}
    for r in cat_z:
        acc.setdefault(r["cls_key"], []).append(float(r["z"]))
    return sorted(acc, key=lambda k: -float(np.mean(acc[k])))[:n]


# ---------------------------------------------------------------- 目录
def build_catalog(space: CLSSpace, shared: Sequence[tuple[CLS, str]], slice_ids: Sequence[str], *, slice_center_keys: Mapping[str, Sequence[str]],
                  seed: int = FIXED_SEED, global_center_keys: Optional[Sequence[str]] = None) -> dict:
    by_key = {k: c for c, k in shared}
    ggs = pq.ggs_cls()
    by_key[pq.cls_key(ggs)] = ggs
    gk = list(global_center_keys) if global_center_keys else [k for _c, k in shared[:2]]
    fixed = fixed_neighbors(space, [by_key[k] for k in gk[:2]], seed=seed)
    cat = {"shared": [{"cls_key": k, "cls": c.to_dict()} for c, k in shared],
           "fixed": [{"cls_key": cls_key(m), "cls": m.to_dict(), "center": cls_key(ctr), "h": h} for ctr, m, h in fixed],
           "random": {}, "centers": {"fixed": gk[:2], "slice": {}}}
    taken0 = _exclusion_keys(space) | {e["cls_key"] for e in cat["fixed"]}
    for sid in slice_ids:
        ck = list(slice_center_keys[sid])[:2]
        sl = slice_neighbors(space, sid, [by_key[k] for k in ck], taken=set(taken0), seed=seed)
        cat["random"][sid] = [{"cls_key": cls_key(m), "cls": m.to_dict(), "center": cls_key(ctr), "h": h} for ctr, m, h in sl]
        cat["centers"]["slice"][sid] = ck
    return cat


def catalog_entries(cat: Mapping, sid: str) -> list[tuple[str, CLS, str]]:
    """{切片: [(origin, CLS, key)]}：shared 10 + 固定 4（origin shared）+ 切片 6（origin random）。"""
    ent = [(ORIGIN_SHARED, CLS(**e["cls"]), e["cls_key"]) for e in cat["shared"]]
    ent += [(ORIGIN_SHARED, CLS(**e["cls"]), e["cls_key"]) for e in cat["fixed"]]
    ent += [(ORIGIN_RANDOM, CLS(**e["cls"]), e["cls_key"]) for e in cat["random"][sid]]
    return ent


def nb_cells(cat: Mapping, slice_ids: Sequence[str]) -> list:
    cells = []
    for sid in slice_ids:
        cells += [_g32.Cell(sid, ORIGIN_SHARED, CLS(**e["cls"]), e["cls_key"]) for e in cat["fixed"]]
        cells += [_g32.Cell(sid, ORIGIN_RANDOM, CLS(**e["cls"]), e["cls_key"]) for e in cat["random"][sid]]
    keys = [(c.slice_id, c.cls_key) for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("邻域网格撞键")
    return cells


def nb_meta(cat: Mapping) -> dict[tuple[str, str], dict]:
    """(切片, key) → {center, h, fixed}。"""
    out = {}
    for sid in cat["random"]:
        for e in cat["fixed"]:
            out[(sid, e["cls_key"])] = {"nb_center": e["center"], "nb_h": e["h"], "nb_fixed": True}
        for e in cat["random"][sid]:
            out[(sid, e["cls_key"])] = {"nb_center": e["center"], "nb_h": e["h"], "nb_fixed": False}
    return out


def cmd_sample(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    space = _g56.space()
    shared = pq.load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    rows_k3 = _g32.read_rows(K3 / "labels_k3.jsonl")
    slice_ids = _g75.all_slices()
    order = _g75.prior_order(rows_k3, exclude_dataset="__none__")
    global_centers = order[:N_CENTERS]
    centers = {sid: slice_centers(rows_k3, sid, n=2) for sid in slice_ids}
    cat = build_catalog(space, shared, slice_ids, slice_center_keys=centers, seed=FIXED_SEED, global_center_keys=global_centers)
    cat["centers"]["global_top5"] = global_centers
    (outdir / "nb_catalog.json").write_text(json.dumps(cat, ensure_ascii=False, indent=1), encoding="utf-8")
    n_rand = sum(len(v) for v in cat["random"].values())
    uniq = len({e["cls_key"] for v in cat["random"].values() for e in v})
    ctr_use = {}
    for v in cat["centers"]["slice"].values():
        for k in v:
            ctr_use[k] = ctr_use.get(k, 0) + 1
    L = ["# 邻域目录核对（--sample）", "", f"中心（k3 先验前 5）：{[k[:8] for k in global_centers]}；固定邻居中心：{[k[:8] for k in cat['centers']['fixed']]}", "",
         "| 固定邻居 | 中心 | h |", "|---|---|---:|"] + [f"| {e['cls_key'][:8]} | {e['center'][:8]} | {e['h']} |" for e in cat["fixed"]]
    L += ["", f"切片邻居：{len(cat['random'])} 切片 × 6 = {n_rand} 条，不同配置 {uniq} 条", f"切片中心使用次数：{ {k[:8]: v for k, v in sorted(ctr_use.items(), key=lambda x: -x[1])} }",
          f"网格：{len(nb_cells(cat, slice_ids))} 格（固定 4 + 切片 6 每切片）", ""]
    (outdir / "sample_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return cat


def load_catalog(outdir: Path) -> dict:
    return json.loads((outdir / "nb_catalog.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- R'
def cmd_rprime(outdir: Path, *, device: str, resume: bool, smoke: bool, names: Sequence[str]) -> int:
    if smoke and outdir.resolve() == OUT.resolve():
        raise SystemExit("--smoke 不得写进真值目录 results/tclsc_nb")
    cat = load_catalog(OUT if not (outdir / "nb_catalog.json").exists() else outdir)
    ds = _g75.load_k3_datasets(names)
    if smoke:
        ds = {k: v for k, v in list(ds.items())[:1]}
        for d in ds.values():
            d.slices = d.slices[-1:]
    cells = nb_cells(cat, [t.task_id for d in ds.values() for t in d.slices])
    if smoke:
        cells = cells[:2]
    idx = _g32.cache_index(_g32.read_rows(outdir / "rprime_nb.jsonl"))
    cells.sort(key=lambda c: (c.slice_id, c.origin, c.cls_key))
    todo = _g32.pending_cells(cells, idx, pq.BUDGETS, resume=resume, redo_partial=False)
    print(f"NB 网格: 计划 {len(cells)} 格 → 待跑 {len(todo)}（已有 {len(cells) - len(todo)}）", flush=True)
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    return _g32.run_jobs(_g32.make_jobs(todo, _g32.slice_index(ds)), budgets=pq.BUDGETS, cache_path=outdir / "rprime_nb.jsonl",
                         inflight=outdir / "inflight.txt", on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                         eval_ratio=labels_cfg["eval_ratio"], batch_size=train_cfg["batch_size"], lr=train_cfg["lr"], device=device, label="nb")


def cmd_verify(outdir: Path) -> dict:
    cat = load_catalog(outdir)
    cells = nb_cells(cat, list(cat["random"]))
    rows = _g32.read_rows(outdir / "rprime_nb.jsonl")
    idx = _g32.cache_index(rows)
    keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
    missing = [(c.slice_id, c.cls_key) for c in cells if any((c.slice_id, c.cls_key, int(b)) not in idx for b in pq.BUDGETS)]
    errs = [k for k, r in idx.items() if r.get("r_prime") is None]
    v = {"cells": len(cells), "rows": len(rows), "missing_cells": len(missing), "dup_keys": len(keys) - len(set(keys)), "errors": len(errs), "missing_examples": missing[:5]}
    v["ok"] = bool(not missing and v["dup_keys"] == 0 and not errs)
    print(f"[verify] cells {len(cells)} rows {len(rows)} missing {len(missing)} dup {v['dup_keys']} errors {len(errs)} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- 标签
def annotate(rows: Sequence[dict], cat: Mapping) -> list[dict]:
    meta = nb_meta(cat)
    out = []
    for r in rows:
        m = meta.get((r["task_id"], r["cls_key"]))
        if m is None or r.get("error") is not None or r.get("r_prime") is None:
            continue
        out.append({**r, **m})
    return out


def merge_labels(shared_rows: Sequence[dict], nb_rows: Sequence[dict]) -> list[dict]:
    """shared 行（k3 或 k1）+ 邻居行（k1，带 nb_center / nb_h）。"""
    return [dict(r) for r in shared_rows] + [dict(r) for r in nb_rows if r.get("error") is None and r.get("r_prime") is not None]


def headroom(rows: Sequence[dict], *, delta: Mapping, budget: int = 290) -> dict:
    """每 (切片, 截点)：最好邻居 − 最好 shared-10（同口径真值）；占比与按数据集中位。"""
    by: dict[str, dict] = {}
    for r in rows:
        if int(r["iters"]) != budget:
            continue
        d = by.setdefault(r["task_id"], {"ds": r["dataset"], "sh": [], "nb": []})
        (d["nb"] if "nb_h" in r else d["sh"]).append(float(r["r_prime"]))
    gaps, by_ds = [], {}
    for s, d in by.items():
        if not d["sh"] or not d["nb"]:
            continue
        g = max(d["nb"]) - max(d["sh"])
        dl = float(delta[(d["ds"], budget)])
        gaps.append((g, dl))
        by_ds.setdefault(d["ds"], []).append(g)
    return {"budget": budget, "n": len(gaps), "frac_positive": float(np.mean([g > 0 for g, _ in gaps])) if gaps else None,
            "frac_over_delta": float(np.mean([g > dl for g, dl in gaps])) if gaps else None, "median_gap": float(np.median([g for g, _ in gaps])) if gaps else None,
            "by_dataset": {d: {"median_gap": float(np.median(v)), "frac_positive": float(np.mean([x > 0 for x in v])), "n": len(v)} for d, v in by_ds.items()}}


def beat_center_by_h(rows: Sequence[dict], *, budget: int = 290) -> dict:
    """邻居比自己中心更好的比例，按汉明分层（P2）。"""
    val = {(r["task_id"], r["cls_key"]): float(r["r_prime"]) for r in rows if int(r["iters"]) == budget}
    out: dict[int, list] = {}
    for r in rows:
        if "nb_h" in r and int(r["iters"]) == budget and (r["task_id"], r["nb_center"]) in val:
            out.setdefault(int(r["nb_h"]), []).append(float(r["r_prime"]) > val[(r["task_id"], r["nb_center"])])
    return {h: {"frac_better": float(np.mean(v)), "n": len(v)} for h, v in sorted(out.items())}


def cmd_labels(outdir: Path) -> dict:
    cat = load_catalog(outdir)
    nb = annotate(_g32.read_rows(outdir / "rprime_nb.jsonl"), cat)
    k3 = _g32.read_rows(K3 / "labels_k3.jsonl")
    k1 = _g32.read_rows(K3 / "labels_k1.jsonl")
    train = merge_labels(k3, nb)
    truth = merge_labels(k1, nb)
    for name, rows in (("labels_train.jsonl", train), ("labels_truth_k1.jsonl", truth)):
        with open(outdir / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    d1 = pq.load_delta_calibration(SOURCE / "delta_calibration.json")
    hd = {str(b): headroom(truth, delta=d1, budget=b) for b in pq.BUDGETS}
    res = {"train_rows": len(train), "truth_rows": len(truth), "nb_rows": len(nb), "headroom": hd, "beat_center_by_h": beat_center_by_h(truth)}
    (outdir / "labels_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "headroom"}, ensure_ascii=False), flush=True)
    print(json.dumps(hd["290"], ensure_ascii=False, indent=1), flush=True)
    return res


def load_rows(outdir: Path, which: str) -> tuple[list[dict], dict]:
    rows = _g32.read_rows(outdir / ("labels_train.jsonl" if which == "train" else "labels_truth_k1.jsonl"))
    return rows, pq.load_delta_calibration(SOURCE / "delta_calibration.json")


# ---------------------------------------------------------------- 训练 / 排名
def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, folds_sel: Optional[Sequence[str]] = None) -> None:
    rows, dk = load_rows(outdir, "train")
    table = _g74.ctx_table(_g74.OUT)
    for target in (folds_sel or FOLDS):
        for arm in ARMS:
            for seed in seeds:
                if resume and (outdir / "models" / TAG / target / f"{arm}_s{seed}.pt").exists():
                    continue
                run = _g73.train_fold_uea(rows, dk, table, target=target, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs, tag=TAG)
                _g73._log(outdir, f"[{TAG}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")


def prior_order_nb(rows: Sequence[dict], cat: Mapping, sid: str, *, exclude_dataset: str) -> list[str]:
    """训练侧 shared + 固定邻居的 z 均值降序；该切片自己的 6 条邻居无先验 → 按目录顺序排最后。"""
    fixed_keys = {e["cls_key"] for e in cat["fixed"]}
    shared_keys = {e["cls_key"] for e in cat["shared"]} | fixed_keys
    cat_z = _g56.zscore_within([r for r in rows if r["dataset"] != exclude_dataset and r["cls_key"] in shared_keys])
    acc: dict[str, list] = {}
    for r in cat_z:
        acc.setdefault(r["cls_key"], []).append(float(r["z"]))
    head = sorted(acc, key=lambda k: -float(np.mean(acc[k])))
    tail = [k for _o, _c, k in catalog_entries(cat, sid) if k not in acc]      # 训练侧无标签的（切片邻居；冒烟时的固定邻居）按目录顺序排最后
    return head + tail


def rank_point_nb(outdir: Path, *, seeds: Sequence[int], device: str = "cpu") -> dict:
    cat = load_catalog(outdir)
    rows, _dk = load_rows(outdir, "train")
    table = _g74.ctx_table(_g74.OUT)
    all_sl = _g75.all_slices()
    out = {}
    for target in FOLDS:
        if not (outdir / "models" / TAG / target).exists():
            continue
        test_slices = _g75.exam_slices(target)
        train_slices = [s for s in all_sl if pq.dataset_of(s) != target]
        catm = {s: catalog_entries(cat, s) for s in test_slices}
        per: dict = {"normal": {}, "cross": {}, "zero": {}, "test": test_slices}
        for s in test_slices:
            order = prior_order_nb(rows, cat, s, exclude_dataset=target)
            for b in pq.BUDGETS:
                per["normal"].setdefault(str(b), {}).setdefault(s, {})[PRIOR] = list(order)

        def merge(dst: dict, rk: dict) -> None:
            for b, pb in rk.items():
                for s, pa in pb.items():
                    dst.setdefault(str(b), {}).setdefault(s, {}).update(pa)

        for arm in ARMS:
            paths = [outdir / "models" / TAG / target / f"{arm}_s{sd}.pt" for sd in seeds]
            if not all(p.exists() for p in paths):
                continue
            mm = [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in paths]
            if _m33.arm_spec(arm).ctx_kind != "table_budget":
                merge(per["normal"], _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device))
                continue
            meta0 = mm[0][1]
            cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
            rows_fn = lambda cm=None, zero=False: _g64.slice_ctx_rows(table, cols, test_slices, mu, sd, cross_map=cm, zero_cols=(cols if zero else ()))
            merge(per["normal"], _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn()))
            merge(per["zero"], _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
            for cs in CROSS_SEEDS:
                cmap = _g58.cross_slice_map(test_slices, train_slices, cs)
                per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
                merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models({arm: mm}, catm, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
        out[target] = per
    return out


def cmd_rank(outdir: Path) -> dict:
    rk = {"".join(map(str, g)): rank_point_nb(outdir, seeds=g) for g in SEED_GROUPS}
    (outdir / "rankings.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    return rk


# ---------------------------------------------------------------- 分析
def _eval(rk_by_target: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], which: str, *, cross_seed: Optional[str] = None, group: str = "") -> list[dict]:
    rows = []
    for target, per in rk_by_target.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        fr = _g59._eval_rows(rk, [r for r in cat_z if r["dataset"] == target], delta, arms, TOP_GOOD_NB)
        for r in fr:
            r["fold"], r["group"] = target, group
        rows += fr
    return rows


def neighbor_metrics(rk_by_target: Mapping, truth_rows: Sequence[dict], arms: Sequence[str]) -> dict:
    """邻居专项：预测前 3 邻居占比 − 真前 3 邻居占比（高估签名）、邻居内配对 acc、邻居 vs 中心判对率（按 h）。"""
    val, nb_of = {}, {}
    for r in truth_rows:
        val[(r["task_id"], int(r["iters"]), r["cls_key"])] = float(r["r_prime"])
        if "nb_h" in r:
            nb_of[(r["task_id"], r["cls_key"])] = (r["nb_center"], int(r["nb_h"]))
    res = {a: {"pred_top3_nb": [], "true_top3_nb": [], "nb_pair_acc": [], "vs_center": {}} for a in arms}
    for target, per in rk_by_target.items():
        for b, pb in per["normal"].items():
            for s, pa in pb.items():
                keys = [k for (t, bb, k) in val if t == s and bb == int(b)]
                if not keys:
                    continue
                truth = {k: val[(s, int(b), k)] for k in keys}
                nbs = [k for k in keys if (s, k) in nb_of]
                true_top3 = sorted(keys, key=lambda k: -truth[k])[:3]
                for a in arms:
                    if a not in pa:
                        continue
                    order = [k for k in pa[a] if k in truth]
                    pos = {k: i for i, k in enumerate(order)}
                    res[a]["pred_top3_nb"].append(np.mean([k in nb_of and (s, k) in nb_of for k in order[:3]]))
                    res[a]["true_top3_nb"].append(np.mean([(s, k) in nb_of for k in true_top3]))
                    pairs = [(x, y) for i, x in enumerate(nbs) for y in nbs[i + 1:] if truth[x] != truth[y]]
                    if pairs:
                        res[a]["nb_pair_acc"].append(np.mean([(pos[x] < pos[y]) == (truth[x] > truth[y]) for x, y in pairs]))
                    for k in nbs:
                        ctr, h = nb_of[(s, k)]
                        if ctr in truth and truth[ctr] != truth[k]:
                            res[a]["vs_center"].setdefault(h, []).append((pos[k] < pos[ctr]) == (truth[k] > truth[ctr]))
    out = {}
    for a, d in res.items():
        out[a] = {"pred_top3_nb_share": float(np.mean(d["pred_top3_nb"])) if d["pred_top3_nb"] else None,
                  "true_top3_nb_share": float(np.mean(d["true_top3_nb"])) if d["true_top3_nb"] else None,
                  "nb_pair_acc": float(np.mean(d["nb_pair_acc"])) if d["nb_pair_acc"] else None,
                  "vs_center_acc_by_h": {h: float(np.mean(v)) for h, v in sorted(d["vs_center"].items())},
                  "vs_center_acc": float(np.mean([x for v in d["vs_center"].values() for x in v])) if d["vs_center"] else None}
        out[a]["over"] = (out[a]["pred_top3_nb_share"] - out[a]["true_top3_nb_share"]) if out[a]["pred_top3_nb_share"] is not None else None
    return out


def gate_n1(hd: Mapping) -> bool:
    return bool(hd.get("frac_positive") is not None and hd["frac_positive"] >= G1_POS and hd["frac_over_delta"] >= G1_DELTA)


def gate_n2(pr: Mapping) -> bool:
    return bool(pr.get("p") is not None and pr["p"] < ALPHA and (pr["mean_b"] - pr["mean_a"]) <= -G2_MIN)


def gate_n3(x: Mapping) -> bool:
    return bool(x.get("over") is not None and x["over"] <= G3_OVER_MAX and x.get("cross_p") is not None and x["cross_p"] < 1e-3)


def verdict_nb(g: Mapping[str, bool]) -> str:
    if not g["G-N1"]:
        return "NB-NO-HEADROOM"
    if g["G-N2"] and g["G-N3"]:
        return "NB-OPEN-WIN"
    if g["G-N2"]:
        return "NB-OPEN-OVER"
    return "NB-OPEN-FLAT"


def cmd_analyze(outdir: Path) -> dict:
    rk = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    truth, dk = load_rows(outdir, "truth")
    train_rows, _ = load_rows(outdir, "train")
    arms_all = list(ARMS) + [PRIOR]
    a: dict = {"protocol": "results/tclsc_nb/PROTOCOL.md", "tables": {}, "paired": {}, "ablation": {}, "neighbor": {}}
    for tname, rows_t in (("k1", truth), ("k3shared", train_rows)):
        cat_z = _g56.zscore_within(rows_t)
        rows_n, rows_z, rows_c = [], [], {cs: [] for cs in map(str, CROSS_SEEDS)}
        for g, per in rk.items():
            rows_n += _eval(per, cat_z, dk, arms_all, "normal", group=g)
            rows_z += _eval(per, cat_z, dk, [E_ARM], "zero", group=g)
            for cs in map(str, CROSS_SEEDS):
                rows_c[cs] += _eval(per, cat_z, dk, [E_ARM], "cross", cross_seed=cs, group=g)
        tab = {arm: _g73._pooled(rows_n, arm) for arm in arms_all}
        for arm in arms_all:
            tab[arm]["hit3"] = float(np.mean([r["regret"][arm + "3"] == 0 for r in rows_n if arm + "3" in r["regret"]]))
        a["tables"][tname] = tab
        a["paired"][tname] = {"e_vs_blind": _g58.paired_regret(rows_n, B_ARM, E_ARM, "3"), "e_vs_prior": _g58.paired_regret(rows_n, PRIOR, E_ARM, "3"),
                              "blind_vs_prior": _g58.paired_regret(rows_n, PRIOR, B_ARM, "3")}
        a["ablation"][tname] = {"cross": _g75._cross_ablation(rows_n, list(rows_c.values()), E_ARM), "zero": _g75._cross_ablation(rows_n, [rows_z], E_ARM)}
        nm = {}
        for g, per in rk.items():
            m = neighbor_metrics(per, rows_t, arms_all)
            for arm, d in m.items():
                nm.setdefault(arm, []).append(d)
        a["neighbor"][tname] = {arm: {k: (float(np.mean([d[k] for d in ds if d[k] is not None])) if isinstance(ds[0][k], (int, float)) or ds[0][k] is None else
                                          {h: float(np.mean([d[k][h] for d in ds if h in d[k]])) for h in ds[0][k]}) for k in ds[0]} for arm, ds in nm.items()}
    ls = json.loads((outdir / "labels_summary.json").read_text(encoding="utf-8"))
    a["headroom"], a["beat_center_by_h"] = ls["headroom"], ls["beat_center_by_h"]
    hd = ls["headroom"]["290"]
    e_nb = a["neighbor"]["k1"][E_ARM]
    g = {"G-N1": gate_n1(hd), "G-N2": gate_n2(a["paired"]["k1"]["e_vs_blind"]), "G-N3": gate_n3({"over": e_nb["over"], "cross_p": a["ablation"]["k1"]["cross"]["p"]})}
    a["gates"], a["verdict"] = g, verdict_nb(g)
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], g, flush=True)
    return a


def predictions(a: dict) -> dict:
    hd = a["headroom"]["290"]; bh = a["beat_center_by_h"]; g = a["gates"]; nb = a["neighbor"]["k1"]
    hs = sorted(int(h) for h in bh)
    mono = all(bh[str(hs[i])]["frac_better"] >= bh[str(hs[i + 1])]["frac_better"] for i in range(len(hs) - 1)) if len(hs) >= 2 else None
    bd = hd["by_dataset"]
    p6 = None
    if all(d in bd for d in ("EthanolConcentration", "Handwriting", "Cricket", "UWaveGestureLibrary")):
        top = sorted(bd, key=lambda d: -bd[d]["median_gap"])
        p6 = bool(set(top[:2]) & {"EthanolConcentration", "Handwriting"} and set(top[-2:]) & {"Cricket", "UWaveGestureLibrary"})
    return {"P1": {"desc": "G-N1", "hit": g["G-N1"]}, "P2": {"desc": "邻居优于中心的比例随 h 单调降", "value": {h: round(bh[h]["frac_better"], 3) for h in bh}, "hit": mono},
            "P3": {"desc": "G-N2", "hit": g["G-N2"]}, "P4": {"desc": "G-N3", "hit": g["G-N3"]},
            "P5": {"desc": "邻居 vs 中心判对率 E_T − 盲 ≥ 0.05", "value": (nb[E_ARM]["vs_center_acc"], nb[B_ARM]["vs_center_acc"]),
                   "hit": None if nb[E_ARM]["vs_center_acc"] is None else bool(nb[E_ARM]["vs_center_acc"] - nb[B_ARM]["vs_center_acc"] >= 0.05)},
            "P6": {"desc": "净空 Ethanol/HW 最大、Cricket/UWave 最小", "hit": p6}}


def markdown_tables(a: dict) -> str:
    L = ["# NB1 检验 tables：邻域标签 + 开放空间排序（考侧 12 切片 × 20 条）", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", ""]
    hd = a["headroom"]["290"]
    L += ["## 净空（k1 真值 @290：最好邻居 − 最好 shared-10）", "", f"n {hd['n']}，正占比 {hd['frac_positive']:.2f}，超 δ 占比 {hd['frac_over_delta']:.2f}，中位差 {hd['median_gap']:+.4f}", "",
          "| 数据集 | 中位差 | 正占比 | n |", "|---|---:|---:|---:|"] + [f"| {d} | {v['median_gap']:+.4f} | {v['frac_positive']:.2f} | {v['n']} |" for d, v in hd["by_dataset"].items()]
    L += ["", f"邻居优于自己中心的比例（按 h）：{a['beat_center_by_h']}", ""]
    for tname, title in (("k1", "真值 k1（主）"), ("k3shared", "shared 用 k3 均值（副）")):
        tab = a["tables"][tname]
        L += [f"## {title}", "", "| 臂 | n | regret@3 | regret@1 | hit@3 | M1(top-10) | M0 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, c in tab.items():
            L.append(f"| {arm} | {c['n']} | {c['regret3']:.4f} | {c['regret1']:.4f} | {c['hit3']:.3f} | {c['M1']:.3f} | {c['M0']:.3f} |")
        L += ["", "| 对比 | n | a | b | p | 胜:负 |", "|---|---:|---:|---:|---:|---|"]
        for name, pr in a["paired"][tname].items():
            L.append(f"| {name} | {pr['n']} | {pr['mean_a']:.4f} | {pr['mean_b']:.4f} | {pr['p']:.2g} | {pr['wins_b']}:{pr['losses_b']} |")
        for name, c in a["ablation"][tname].items():
            L.append(f"| E_T {name} 消融 | {c['n']} | {c['normal']:.4f} | {c['cross']:.4f} | {c['p']:.2g} | {c['wins']}:{c['losses']} |")
        L += ["", "| 臂 | 预测前 3 邻居占比 | 真前 3 邻居占比 | 高估 | 邻居内配对 acc | 邻居 vs 中心判对率 | 按 h |", "|---|---:|---:|---:|---:|---:|---|"]
        for arm, d in a["neighbor"][tname].items():
            f = lambda v: "—" if v is None else f"{v:.3f}"
            L.append(f"| {arm} | {f(d['pred_top3_nb_share'])} | {f(d['true_top3_nb_share'])} | {f(d['over'])} | {f(d['nb_pair_acc'])} | {f(d['vs_center_acc'])} | {d['vs_center_acc_by_h']} |")
        L.append("")
    L += [f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("sample", "rprime", "verify", "labels", "train", "rank", "analyze", "smoke", "resume"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(pq.PQ_DATASETS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[s for g in SEED_GROUPS for s in g])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.sample:
        cmd_sample(args.outdir)
    if args.rprime:
        cmd_rprime(args.outdir, device=args.device, resume=args.resume, smoke=args.smoke, names=args.datasets)
    if args.verify:
        cmd_verify(args.outdir)
    if args.labels:
        cmd_labels(args.outdir)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, folds_sel=args.folds)
    if args.rank:
        cmd_rank(args.outdir)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
