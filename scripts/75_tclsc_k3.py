"""WP16 K3 检验：标签多种子化——shared-10 × 87 切片补 rep {1,2} 权重种子，信任域内比较器 2 × 2（标签 K1/K3 × 臂 E_T/任务盲）。
协议 results/tclsc_k3/PROTOCOL.md。

子命令（按执行顺序）：
  --rprime    补种子轨迹（32 号 NoiseCell 执行体：noise_seed，划分 canonical，三截点）→ rprime_k3.jsonl；已有 16 切片探针 --resume 复用；--verify 三点齐全 / 撞键
  --labels    k=1 / k=3 标签行（只留 shared，三点不齐剔除）+ δ_k3 + 噪声统计（P1）
  --train     K1 / K3 × ft60_hard / blind_diff_hard，7 折 × 6 种子（73 号 train_fold_uea，rows 只含 shared）
  --rank      考侧 12 切片 × shared-10，normal / cross(3) / zero + prior 臂；--analyze 门 / 读数码 / 表；--simulate-official 副读数（71 号 D′ 口径）
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import load_source_dataset, slice_extension_tasks, slice_source_tasks  # noqa: E402
from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.oom_guard import ON_OOM_SKIP  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g59 = importlib.import_module("scripts.59_tclsc_s1fp")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g73 = importlib.import_module("scripts.73_tclsc_lc")
_g74 = importlib.import_module("scripts.74_tclsc_slices")
OUT = ROOT / "results" / "tclsc_k3"
SOURCE = pq.OUT
SX = _g74.OUT
NOISE_PROBE = SOURCE / "noise_probe.jsonl"
REPS = (1, 2)
K = 3
TOP_GOOD_K3 = 5                                  # shared-10 的好半区
ARMS = ("ft60_hard", "blind_diff_hard")
E_ARM, B_ARM = ARMS
PRIOR = "prior"
TAGS = ("K1", "K3")
SEED_GROUPS = ((0, 1, 2), (3, 4, 5))
CROSS_SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
EXT_DATASETS = _g74.EXT_DATASETS
DELTA_STEP = pq.DELTA_QUANTUM_STEP
ALPHA = 0.05
G1_MIN, G3_MIN = 0.003, 0.002


# ---------------------------------------------------------------- 切片 / 网格
def exam_slices(target: str) -> list[str]:
    """考侧 = 原 6 + 扩展 6（SCP2 不作靶）。"""
    return pq.pq_slice_ids(target) + (_g74.ext_slice_ids(target) if target in EXT_DATASETS else [])


def all_slices() -> list[str]:
    return [s for d in pq.PQ_DATASETS for s in (pq.pq_slice_ids(d) + (_g74.ext_slice_ids(d) if d in EXT_DATASETS else []))]


def k3_cells(slice_ids: Sequence[str], *, space: Optional[CLSSpace] = None) -> list:
    """每切片 shared-10 × rep ∈ REPS 的 NoiseCell（32 号 make_jobs 按 rep 取 noise_seed）。"""
    space = space or CLSSpace.from_yaml()
    shared = None
    cells = []
    for sid in slice_ids:
        entries = pq.catalog_for_slice(space, sid, shared)
        if shared is None:
            shared = [(c, k) for org, c, k in entries if org == ORIGIN_SHARED]
        for org, c, k in entries:
            if org == ORIGIN_SHARED:
                cells += [_g32.NoiseCell(sid, org, c, k, int(rep)) for rep in REPS]
    keys = [(c.slice_id, c.cls_key, c.rep) for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("噪声格撞键")
    return cells


def load_k3_datasets(names: Sequence[str] = pq.PQ_DATASETS) -> dict[str, pq.PQDataset]:
    """每数据集装载一次：原切片（注册表流前 n_slices_for）+ 扩展切片（SX 同一构造，逐字段与 tclsc_slices/tasks.json 一致）。"""
    out = {}
    for name in names:
        reg = pq.registry_for(name)
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _eff, planned = slice_source_tasks(entry, reg, y)
        slices = list(planned[: pq.n_slices_for(name)])
        if name in EXT_DATASETS:
            slices += slice_extension_tasks(entry, reg, y, n_new=_g74.EXT_N, exclude_first=pq.n_slices_for(name))
        out[name] = pq.PQDataset(name=name, X=X, y=y, entry=entry, registry=reg, slices=slices)
    return out


def rep_rows(outdir: Path) -> list[dict]:
    return _g32.read_rows(NOISE_PROBE) + _g32.read_rows(outdir / "rprime_k3.jsonl")


def cmd_rprime(outdir: Path, *, device: str, resume: bool, smoke: bool, names: Sequence[str]) -> int:
    if smoke and outdir.resolve() == OUT.resolve():
        raise SystemExit("--smoke 不得写进真值目录 results/tclsc_k3")
    outdir.mkdir(parents=True, exist_ok=True)
    ds = load_k3_datasets(names)
    if smoke:
        ds = {k: v for k, v in list(ds.items())[:1]}
        for d in ds.values():
            d.slices = d.slices[-1:]
    cells = k3_cells([t.task_id for d in ds.values() for t in d.slices])
    if smoke:
        cells = cells[:2]
    nidx = _g32.index_by_seed(rep_rows(outdir))
    cells.sort(key=lambda c: (c.slice_id, c.rep))
    todo = _g32.pending_cells(cells, nidx, pq.BUDGETS, resume=resume, redo_partial=False)
    print(f"K3 网格: 计划 {len(cells)} 格 → 待跑 {len(todo)}（已有 {len(cells) - len(todo)}）", flush=True)
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    return _g32.run_jobs(_g32.make_jobs(todo, _g32.slice_index(ds)), budgets=pq.BUDGETS, cache_path=outdir / "rprime_k3.jsonl",
                         inflight=outdir / "inflight.txt", on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                         eval_ratio=labels_cfg["eval_ratio"], batch_size=train_cfg["batch_size"], lr=train_cfg["lr"], device=device, label="k3")


def cmd_verify(outdir: Path) -> dict:
    cells = k3_cells(all_slices())
    rows = rep_rows(outdir)
    idx = {(r["task_id"], r["cls_key"], int(r["iters"]), int(r["seed"])): r for r in rows}
    keys = [(r["task_id"], r["cls_key"], int(r["iters"]), int(r["seed"])) for r in rows]
    missing = [(c.slice_id, c.cls_key, c.rep) for c in cells if any((c.slice_id, c.cls_key, int(b), c.rep) not in idx for b in pq.BUDGETS)]
    errs = [k for k, r in idx.items() if r.get("r_prime") is None]
    v = {"cells": len(cells), "rows": len(rows), "missing_cells": len(missing), "dup_keys": len(keys) - len(set(keys)), "errors": len(errs),
         "missing_examples": missing[:5]}
    v["ok"] = bool(not missing and v["dup_keys"] == 0 and not errs)
    print(f"[verify] cells {len(cells)} rows {len(rows)} missing {len(missing)} dup {v['dup_keys']} errors {len(errs)} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- 标签
def k_mean_rows(main_rows: Sequence[dict], rep_rows_: Sequence[dict], *, k: int) -> list[dict]:
    """shared 行 → r_prime 换成 k 个种子的均值（k=1 = seed 0 逐位；k=3 须 rep 1、2 齐全且有限，否则剔除）。"""
    reps = {}
    for r in rep_rows_:
        if r.get("r_prime") is not None and np.isfinite(r["r_prime"]):
            reps[(r["task_id"], r["cls_key"], int(r["iters"]), int(r["seed"]))] = float(r["r_prime"])
    out = []
    for r in main_rows:
        if r.get("origin") != ORIGIN_SHARED or r.get("error") is not None or r.get("r_prime") is None:
            continue
        vals = [float(r["r_prime"])]
        if k > 1:
            extra = [reps.get((r["task_id"], r["cls_key"], int(r["iters"]), rep)) for rep in REPS[: k - 1]]
            if any(v is None for v in extra):
                continue
            vals += extra
        out.append({**r, "r_prime": float(np.mean(vals)), "seed": 0, "k": int(k)})
    return out


def delta_k(delta_k1: Mapping, *, k: int) -> dict:
    if k == 1:
        return dict(delta_k1)
    return {key: round(math.ceil(v / math.sqrt(k) / DELTA_STEP - 1e-9) * DELTA_STEP, 3) for key, v in delta_k1.items()}


def main_rows() -> list[dict]:
    """seed-0 主缓存：mx3_pq（原切片）+ tclsc_slices（扩展切片），只留 shared。"""
    rows = [r for r in read_jsonl(SOURCE / "rprime_cache.jsonl")] + _g32.read_rows(SX / "rprime_cache.jsonl")
    return [r for r in rows if r.get("origin") == ORIGIN_SHARED and r.get("error") is None and r.get("r_prime") is not None]


def noise_stats(main: Sequence[dict], reps: Sequence[dict]) -> dict:
    """P1：新切片（非探针）@ 各截点的种子间 Pearson r₁（10 条 shared 上）与单格 sd。"""
    by = {}
    for r in main:
        by.setdefault((r["task_id"], int(r["iters"])), {}).setdefault(r["cls_key"], {})[0] = float(r["r_prime"])
    for r in reps:
        if r.get("r_prime") is not None:
            by.setdefault((r["task_id"], int(r["iters"])), {}).setdefault(r["cls_key"], {})[int(r["seed"])] = float(r["r_prime"])
    probe = {r["task_id"] for r in _g32.read_rows(NOISE_PROBE)}
    out = {}
    for group, pick in (("probe16", lambda s: s in probe), ("new71", lambda s: s not in probe)):
        for b in pq.BUDGETS:
            r1, sds = [], []
            for (s, bb), d in by.items():
                if bb != b or not pick(s):
                    continue
                full = {kk: v for kk, v in d.items() if len(v) >= 3}
                if len(full) < 8:
                    continue
                ks = sorted(full)
                for a, c in ((0, 1), (0, 2), (1, 2)):
                    r1.append(pearsonr([full[kk][a] for kk in ks], [full[kk][c] for kk in ks])[0])
                sds += [float(np.std(list(v.values()), ddof=1)) for v in full.values()]
            if sds:
                out[f"{group}@{b}"] = {"r1_mean": float(np.nanmean(r1)), "sd_median": float(np.median(sds)), "sd_mean": float(np.mean(sds)),
                                       "sb_k3": float(3 * np.nanmean(r1) / (1 + 2 * np.nanmean(r1))), "n_cells": len(sds)}
    return out


def cmd_labels(outdir: Path) -> dict:
    main, reps = main_rows(), rep_rows(outdir)
    d1 = pq.load_delta_calibration(SOURCE / "delta_calibration.json")
    res = {}
    for k in (1, K):
        rows = k_mean_rows(main, reps, k=k)
        with open(outdir / f"labels_k{k}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        dk = delta_k(d1, k=k)
        (outdir / f"delta_k{k}.json").write_text(json.dumps({f"{d}|{b}": v for (d, b), v in dk.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
        res[f"k{k}"] = {"rows": len(rows), "slices": len({r["task_id"] for r in rows})}
    # C0′：k1 逐位 = 主缓存；k3 的探针切片均值 = 探针行均值（同源自证，记录）
    res["noise"] = noise_stats(main, reps)
    (outdir / "labels_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return res


def load_labels(outdir: Path, tag: str) -> tuple[list[dict], dict]:
    k = 1 if tag == "K1" else K
    rows = _g32.read_rows(outdir / f"labels_k{k}.jsonl")
    dk = {tuple([kk.split("|")[0], int(kk.split("|")[1])]): v for kk, v in json.loads((outdir / f"delta_k{k}.json").read_text(encoding="utf-8")).items()}
    return rows, dk


def prior_order(rows: Sequence[dict], *, exclude_dataset: str) -> list[str]:
    """源池先验：训练侧（≠ 目标数据集）切片上 shared 的 within-(切片, 截点) z 均值降序。"""
    cat_z = _g56.zscore_within([r for r in rows if r["dataset"] != exclude_dataset])
    acc: dict[str, list] = {}
    for r in cat_z:
        acc.setdefault(r["cls_key"], []).append(float(r["z"]))
    return sorted(acc, key=lambda k: -float(np.mean(acc[k])))


# ---------------------------------------------------------------- 训练 / 排名
def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, tags: Sequence[str] = TAGS,
              folds_sel: Optional[Sequence[str]] = None) -> None:
    table = _g74.ctx_table(SX)
    for tag in tags:
        rows, dk = load_labels(outdir, tag)
        for target in (folds_sel or FOLDS):
            for arm in ARMS:
                for seed in seeds:
                    if resume and (outdir / "models" / tag / target / f"{arm}_s{seed}.pt").exists():
                        continue
                    run = _g73.train_fold_uea(rows, dk, table, target=target, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs, tag=tag)
                    _g73._log(outdir, f"[{tag}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")


def shared_catalog(cat: Mapping[str, Sequence]) -> dict:
    return {s: [e for e in entries if e[0] == ORIGIN_SHARED] for s, entries in cat.items()}


def _catalog_for(test_slices: Sequence[str]) -> dict:
    space = _g56.space()
    orig = [s for s in test_slices if not _g74.is_ext(s)]
    ext = [s for s in test_slices if _g74.is_ext(s)]
    cat = {}
    if orig:
        c, _g, _k = _g34.load_catalog(SOURCE, space, slices=orig)
        cat.update({s: c[s] for s in orig})
    if ext:
        c, _g, _k = _g34.load_catalog(SX, space, slices=ext)
        cat.update({s: c[s] for s in ext})
    return shared_catalog(cat)


def rank_point_k3(outdir: Path, tag: str, *, seeds: Sequence[int], rows: Sequence[dict], device: str = "cpu") -> dict:
    table = _g74.ctx_table(SX)
    all_sl = all_slices()
    out = {}
    for target in FOLDS:
        if not (outdir / "models" / tag / target).exists():
            continue
        test_slices = exam_slices(target)
        train_slices = [s for s in all_sl if pq.dataset_of(s) != target]
        cat = _catalog_for(test_slices)
        per: dict = {"normal": {}, "cross": {}, "zero": {}, "test": test_slices}
        order = prior_order(rows, exclude_dataset=target)
        for b in pq.BUDGETS:
            for s in test_slices:
                per["normal"].setdefault(str(b), {}).setdefault(s, {})[PRIOR] = list(order)

        def merge(dst: dict, rk: dict) -> None:
            for b, pb in rk.items():
                for s, pa in pb.items():
                    dst.setdefault(str(b), {}).setdefault(s, {}).update(pa)

        for arm in ARMS:
            paths = [outdir / "models" / tag / target / f"{arm}_s{sd}.pt" for sd in seeds]
            if not all(p.exists() for p in paths):
                continue
            mm = [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in paths]
            if _m33.arm_spec(arm).ctx_kind != "table_budget":
                merge(per["normal"], _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device))
                continue
            meta0 = mm[0][1]
            cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
            rows_fn = lambda cm=None, zero=False: _g64.slice_ctx_rows(table, cols, test_slices, mu, sd, cross_map=cm, zero_cols=(cols if zero else ()))
            merge(per["normal"], _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn()))
            merge(per["zero"], _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
            for cs in CROSS_SEEDS:
                cmap = _g58.cross_slice_map(test_slices, train_slices, cs)
                per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
                merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models({arm: mm}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
        out[target] = per
    return out


def cmd_rank(outdir: Path, *, tags: Sequence[str] = TAGS) -> dict:
    rk: dict = {}
    for tag in tags:
        if not (outdir / "models" / tag).exists():
            continue
        rows, _dk = load_labels(outdir, tag)
        rk[tag] = {}
        for g in SEED_GROUPS:
            rk[tag]["".join(map(str, g))] = rank_point_k3(outdir, tag, seeds=g, rows=rows)
            print(f"[rank] {tag} seeds {g}", flush=True)
    (outdir / "rankings.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    return rk


# ---------------------------------------------------------------- 分析
def _eval(rk_by_target: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], which: str, *, cross_seed: Optional[str] = None,
          group: str = "") -> list[dict]:
    rows = []
    for target, per in rk_by_target.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        rows_t = [r for r in cat_z if r["dataset"] == target]
        # 三点不齐被剔除的格会让某些 (切片, 截点) 少于 3 条候选（冒烟 / 部分缓存）：跳过，不评
        cnt = {}
        for r in rows_t:
            cnt[(r["task_id"], int(r["iters"]))] = cnt.get((r["task_id"], int(r["iters"])), 0) + 1
        rows_t = [r for r in rows_t if cnt[(r["task_id"], int(r["iters"]))] >= 3]
        fr = _g59._eval_rows(rk, rows_t, delta, arms, TOP_GOOD_K3)
        for r in fr:
            r["fold"], r["group"] = target, group
        rows += fr
    return rows


def gate_k1(pr: Mapping) -> bool:
    """G-K1：K3 标签下 E_T（b）vs 任务盲（a）regret@3，p < α 且 Δ ≤ −G1_MIN。"""
    return bool(pr.get("p") is not None and pr["p"] < ALPHA and (pr["mean_b"] - pr["mean_a"]) <= -G1_MIN)


def gate_k2(ab: Mapping) -> bool:
    """G-K2：cross 消融 regret@3 升高（mean_delta = cross − normal > 0）且 p < 0.001。"""
    return bool(ab.get("p") is not None and ab["p"] < 1e-3 and ab["mean_delta"] > 0)


def gate_k3(pr: Mapping) -> bool:
    """G-K3：E_T 的 K3（b）vs K1（a）标签 regret@3，p < α 且 Δ ≤ −G3_MIN。"""
    return bool(pr.get("p") is not None and pr["p"] < ALPHA and (pr["mean_b"] - pr["mean_a"]) <= -G3_MIN)


def verdict_k3(g: Mapping[str, bool]) -> str:
    if g["G-K1"] and g["G-K2"]:
        return "K3-E-WIN"
    if g["G-K3"]:
        return "K3-LABEL-WIN"
    return "K3-FLAT"


def _cross_ablation(rows_normal: Sequence[dict], rows_cross: Sequence[Sequence[dict]], arm: str) -> dict:
    """regret@3：normal vs 各错位种子均值（实例 = (组, 折, 切片, 预算)），H1 normal 更低。"""
    key = lambda r: (r["group"], r["fold"], r["slice"], int(r["budget"]))
    nm = {key(r): r["regret"][arm + "3"] for r in rows_normal if arm + "3" in r["regret"]}
    cms = [{key(r): r["regret"][arm + "3"] for r in rs if arm + "3" in r["regret"]} for rs in rows_cross]
    ks = sorted(k for k in nm if all(k in cm for cm in cms))
    x = np.array([nm[k] for k in ks]); y = np.array([np.mean([cm[k] for cm in cms]) for k in ks])
    return {"n": int(x.size), "normal": float(x.mean()), "cross": float(y.mean()), "mean_delta": float((y - x).mean()),
            "p": float(_g56.paired_wilcoxon_less(x, y)) if x.size else None, "wins": int(np.sum(y > x)), "losses": int(np.sum(y < x))}


def _top3_agreement(rk_tag: Mapping, arm: str) -> float:
    g1, g2 = (rk_tag[k] for k in sorted(rk_tag))
    agree = []
    for t in g1:
        for b, pb in g1[t]["normal"].items():
            for s, pa in pb.items():
                if arm in pa and arm in g2[t]["normal"].get(b, {}).get(s, {}):
                    agree.append(len(set(pa[arm][:3]) & set(g2[t]["normal"][b][s][arm][:3])) / 3)
    return float(np.mean(agree)) if agree else float("nan")


def cmd_analyze(outdir: Path) -> dict:
    rk = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    truths = {}
    for k in (K, 1):
        rows, dk = load_labels(outdir, f"K{k}" if k == 1 else "K3")
        truths[f"k{k}"] = (_g56.zscore_within(rows), dk)
    arms_all = list(ARMS) + [PRIOR]
    a: dict = {"protocol": "results/tclsc_k3/PROTOCOL.md", "tables": {}, "ablation": {}, "paired": {}, "gates": {}, "top3_agreement": {}}
    ev: dict = {}
    for truth, (cat_z, dk) in truths.items():
        a["tables"][truth] = {}
        for tag in TAGS:
            if tag not in rk:
                continue
            rows_n, rows_c, rows_z = [], {cs: [] for cs in map(str, CROSS_SEEDS)}, []
            for g, per in rk[tag].items():
                rows_n += _eval(per, cat_z, dk, arms_all, "normal", group=g)
                rows_z += _eval(per, cat_z, dk, [E_ARM], "zero", group=g)
                for cs in map(str, CROSS_SEEDS):
                    rows_c[cs] += _eval(per, cat_z, dk, [E_ARM], "cross", cross_seed=cs, group=g)
            ev[(truth, tag)] = rows_n
            tab = {arm: _g73._pooled(rows_n, arm) for arm in arms_all}
            for arm in arms_all:
                tab[arm]["hit3"] = float(np.mean([r["regret"][arm + "3"] == 0 for r in rows_n if arm + "3" in r["regret"]]))
            a["tables"][truth][tag] = tab
            a["ablation"][f"{truth}:{tag}"] = {"cross": _cross_ablation(rows_n, list(rows_c.values()), E_ARM), "zero": _cross_ablation(rows_n, [rows_z], E_ARM),
                                              "e_vs_blind": _g58.paired_regret(rows_n, B_ARM, E_ARM, "3"), "e_vs_prior": _g58.paired_regret(rows_n, PRIOR, E_ARM, "3"),
                                              "blind_vs_prior": _g58.paired_regret(rows_n, PRIOR, B_ARM, "3")}
    for truth in truths:
        for arm in ARMS:
            if (truth, "K1") in ev and (truth, "K3") in ev:
                a["paired"][f"{truth}:{arm}:K1->K3"] = _g74._paired_same_arm(ev[(truth, "K1")], ev[(truth, "K3")], arm)
    for tag in TAGS:
        if tag in rk:
            a["top3_agreement"][tag] = {arm: _top3_agreement(rk[tag], arm) for arm in ARMS}
    if ("k3", "K3") in ev:
        ab = a["ablation"]["k3:K3"]
        g = {"G-K1": gate_k1(ab["e_vs_blind"]), "G-K2": gate_k2(ab["cross"]),
             "G-K3": gate_k3(a["paired"].get(f"k3:{E_ARM}:K1->K3", {}))}
        a["gates"], a["verdict"] = g, verdict_k3(g)
    ls = json.loads((outdir / "labels_summary.json").read_text(encoding="utf-8")) if (outdir / "labels_summary.json").exists() else {}
    a["noise"] = ls.get("noise", {})
    off = outdir / "official_K3.json"
    if off.exists():
        a["official"] = json.loads(off.read_text(encoding="utf-8"))
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a.get("verdict"), a.get("gates"), flush=True)
    return a


def predictions(a: dict) -> dict:
    p: dict = {}
    n = a.get("noise", {}).get("new71@290")
    p["P1"] = {"desc": "新切片 r₁@290 ∈ 0.74 ± 0.10", "value": None if n is None else round(n["r1_mean"], 3), "hit": None if n is None else bool(0.64 <= n["r1_mean"] <= 0.84)}
    ag = a.get("top3_agreement", {})
    if "K1" in ag and "K3" in ag:
        p["P2"] = {"desc": "K3 臂两组种子 top-3 一致率 > K1 臂", "value": (round(ag["K3"][E_ARM], 3), round(ag["K1"][E_ARM], 3)), "hit": bool(ag["K3"][E_ARM] > ag["K1"][E_ARM])}
    g = a.get("gates", {})
    p["P3"] = {"desc": "G-K3", "hit": g.get("G-K3")}
    p["P4"] = {"desc": "G-K1", "hit": g.get("G-K1")}
    p["P5"] = {"desc": "G-K2", "hit": g.get("G-K2")}
    pb = a.get("paired", {}).get(f"k3:{B_ARM}:K1->K3")
    p["P6"] = {"desc": "任务盲 K3 vs K1 Δ ≤ −0.002", "value": None if pb is None else round(pb["mean_b"] - pb["mean_a"], 4), "hit": None if pb is None else bool(pb["mean_b"] - pb["mean_a"] <= -0.002)}
    if a.get("official"):
        m = a["official"]["summary"]["et_tclsc"]["mean_seed0"]
        p["P7"] = {"desc": "副读数 K3 E_T 官方六靶 seed-0 均值 ≥ 0.564", "value": round(m, 3), "hit": bool(m >= 0.564)}
    return p


def markdown_tables(a: dict) -> str:
    L = ["# K3 检验 tables：标签多种子化（信任域内比较器，考侧 12 切片 × shared-10）", "", f"读数码 **{a.get('verdict')}**（门 {a.get('gates')}）", ""]
    for truth in ("k3", "k1"):
        if truth not in a["tables"]:
            continue
        L += [f"## 真值 {truth}", "", "| 标签 | 臂 | n | regret@3 | regret@1 | hit@3 | M1(top-5) | M0 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for tag, tab in a["tables"][truth].items():
            for arm, c in tab.items():
                L.append(f"| {tag} | {arm} | {c['n']} | {c['regret3']:.4f} | {c['regret1']:.4f} | {c['hit3']:.3f} | {c['M1']:.3f} | {c['M0']:.3f} |")
        L += ["", "| 标签 | 对比 | n | a | b | p | 胜:负 |", "|---|---|---:|---:|---:|---:|---|"]
        for tag in a["tables"][truth]:
            ab = a["ablation"][f"{truth}:{tag}"]
            for name in ("e_vs_blind", "e_vs_prior", "blind_vs_prior"):
                pr = ab[name]
                L.append(f"| {tag} | {name} | {pr['n']} | {pr['mean_a']:.4f} | {pr['mean_b']:.4f} | {pr['p']:.2g} | {pr['wins_b']}:{pr['losses_b']} |")
            for name in ("cross", "zero"):
                c = ab[name]
                L.append(f"| {tag} | E_T {name} 消融 | {c['n']} | {c['normal']:.4f} | {c['cross']:.4f} | {c['p']:.2g} | {c['wins']}:{c['losses']} |")
        L.append("")
    L += ["## 标签 K1 → K3（同臂配对，真值 k3 / k1）", "", "| 真值 | 臂 | n | K1 | K3 | p | 胜:负:平 |", "|---|---|---:|---:|---:|---:|---|"]
    for k, pr in a["paired"].items():
        truth, arm, _ = k.split(":")
        L.append(f"| {truth} | {arm} | {pr['n']} | {pr['mean_a']:.4f} | {pr['mean_b']:.4f} | {pr['p']:.2g} | {pr['wins_b']}:{pr['losses_b']}:{pr['ties']} |")
    L += ["", f"两组种子 top-3 一致率：{a.get('top3_agreement')}", "", f"噪声统计：{a.get('noise')}", "", f"预测：{a.get('predictions')}", ""]
    if a.get("official"):
        o = a["official"]
        L += ["## 副读数：K3 折模型代入 D′ 口径（官方六靶信任域 SH6-3-3，seed 0）", "", "| 靶 | D′ S6 E_T | K3 E_T | 冠军变 | D′ 盲 | K3 盲 |", "|---|---:|---:|---|---:|---:|"]
        for d, r in o["targets"].items():
            L.append(f"| {d} | {r['ref']['et_tclsc']['acc']:.3f} | {r['et_tclsc']['acc']:.3f} | {'是' if r['et_tclsc']['champion'] != r['ref']['et_tclsc']['champion'] else '否'} | "
                     f"{r['ref']['blind_tclsc']['acc']:.3f} | {r['blind_tclsc']['acc']:.3f} |")
        L.append(f"| **均值** | {o['summary']['ref_et_mean']:.3f} | {o['summary']['et_tclsc']['mean_seed0']:.3f} | | {o['summary']['ref_blind_mean']:.3f} | {o['summary']['blind_tclsc']['mean_seed0']:.3f} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("rprime", "verify", "labels", "train", "rank", "analyze", "simulate-official", "smoke", "resume"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(pq.PQ_DATASETS))
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[s for g in SEED_GROUPS for s in g])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.rprime:
        cmd_rprime(args.outdir, device=args.device, resume=args.resume, smoke=args.smoke, names=args.datasets)
    if args.verify:
        cmd_verify(args.outdir)
    if args.labels:
        cmd_labels(args.outdir)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, tags=args.tags, folds_sel=args.folds)
    if args.rank:
        cmd_rank(args.outdir, tags=args.tags)
    if args.simulate_official:
        _g74.cmd_simulate_official(args.outdir, tag="K3")
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
