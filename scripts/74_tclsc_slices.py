"""WP16 SX 检验：每集加切片——切片轴 S ∈ {6, 9, 12} 嵌套读数。协议 results/tclsc_slices/PROTOCOL.md。

子命令（按执行顺序）：
  --slice     扩展切片定义（7 个折数据集各 6 个，id `{name}_x00…x05`）→ tasks.json + slice_check.{json,md}（零 GPU）
  --rprime    新切片 R'：每切片 shared-10 + random-10 + GGS，一条轨迹三截点（32 号执行体；GPU，pod）；--verify 行数 / 撞键
  --baseline  iters=0 随机初始化 R' 5 种子（φ 的 base_mean / base_band）
  --features  φ12 ⊕ E_A ⊕ E_T → ctx_ext.jsonl（列集同 results/tclsc_s1et/uea/ctx_uea.jsonl）
  --train     S9 / S12 折内训练（73 号 train_fold_uea；rows = mx3_pq ∪ 扩展，ctx 表 = ctx_uea ∪ ctx_ext）
  --rank      S6（复用 T / s1phi / LC rep_s345 模型）/ S9 / S12 排名，两组 3 种子多数票（0–2、3–5）
  --analyze   C0 / G-S1 / G-S2 / 曲线 / 读数码；--simulate-official 副读数（71 号 D′ 口径，S12 折模型）
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.baseline_probe import BASELINE_ITERS, baseline_seed_specs, inert_cls  # noqa: E402
from autocls.data.mech_features import RESPONSE_PROFILE_COLS, RESPONSE_SEEDS, rep_cols_at, response_profile  # noqa: E402
from autocls.data.source_pool import load_source_dataset, slice_extension_tasks, slice_source_tasks  # noqa: E402
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import evaluate_rprime, run_seed_for, split_task_train_eval  # noqa: E402
from autocls.oom_guard import ON_OOM_SKIP  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g40 = importlib.import_module("scripts.40_task_feature_scorer")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g69 = importlib.import_module("scripts.69_tclsc_s1et")
_g73 = importlib.import_module("scripts.73_tclsc_lc")
PHI_COLS = tuple(_m33.PHI_COLS)
OUT = ROOT / "results" / "tclsc_slices"
SOURCE = pq.OUT
S1ET = ROOT / "results" / "tclsc_s1et"
S1PHI = ROOT / "results" / "tclsc_s1phi"
LC = ROOT / "results" / "tclsc_lc"
EXT_DATASETS = tuple(pq.FOLD_TARGETS)            # SCP2 不扩（PROTOCOL §1.3）
EXT_N = 6
S_POINTS = (6, 9, 12)
SEED_GROUPS = ((0, 1, 2), (3, 4, 5))             # 排名单位 = 3 种子多数票；两组 pooled
ARMS = ("ft60_hard", "blind_diff_hard")
E_ARM, B_ARM = ARMS
FOLDS = tuple(pq.FOLD_TARGETS)
N_BASELINE_SEEDS = 5
C0_REF = {E_ARM: 0.0242, B_ARM: 0.0311}          # PROTOCOL §0：LC 六种子均值
C0_TOL = 0.0005
G1_MIN_GAIN = 0.002
ALPHA = 0.05


# ---------------------------------------------------------------- 切片
def ext_slice_ids(name: str) -> list[str]:
    return [f"{name}_x{i:02d}" for i in range(EXT_N)]


def is_ext(slice_id: str) -> bool:
    return pq.slice_of(slice_id).rsplit("_", 1)[1].startswith("x")


def ext_index(slice_id: str) -> int:
    return int(pq.slice_of(slice_id).rsplit("_", 1)[1][1:])


def load_ext_datasets(names: Sequence[str] = EXT_DATASETS) -> dict[str, pq.PQDataset]:
    """装载数据集 + 扩展切片（类别子集避开缓存里实际存在的原切片 = 前 n_slices_for 个）。"""
    out = {}
    for name in names:
        reg = pq.registry_for(name)
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        ext = slice_extension_tasks(entry, reg, y, n_new=EXT_N, exclude_first=pq.n_slices_for(name))
        assert [t.task_id for t in ext] == ext_slice_ids(name)
        out[name] = pq.PQDataset(name=name, X=X, y=y, entry=entry, registry=reg, slices=ext)
    return out


def _jac(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def cmd_slice(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    ds = load_ext_datasets()
    pq.write_tasks_json(ds, outdir / "tasks.json")
    orig = pq.load_tasks_json(SOURCE / "tasks.json")
    check: dict = {"datasets": {}, "originals_unchanged": True}
    L = ["# 切法核对（--slice 实测，零 GPU）", "", "| 切片 | k | 通道 | 样本 | n_train | 类别 Jaccard vs 原(max) | 样本 Jaccard vs 原(max) | 通道 Jaccard vs 原(max) | 类别子集重复原切片 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    new_tasks = pq.load_tasks_json(outdir / "tasks.json")
    for name, d in ds.items():
        # 原切片逐字段不变：注册表流重切 vs mx3_pq/tasks.json
        _eff, planned = slice_source_tasks(d.entry, d.registry, d.y)
        for t in planned[: pq.n_slices_for(name)]:
            o = orig[t.task_id]
            same = (list(t.classes) == list(o["classes"]) and list(t.channels) == list(o["channels"]) and list(t.sample_indices) == list(o["sample_indices"]))
            check["originals_unchanged"] &= same
        os_ = [orig[s] for s in pq.pq_slice_ids(name)]
        per = []
        for t in d.slices:
            nt = new_tasks[t.task_id]
            cj = max(_jac(t.classes, o["classes"]) for o in os_)
            sj = max(_jac(t.sample_indices, o["sample_indices"]) for o in os_)
            chj = max(_jac(t.channels, o["channels"]) for o in os_)
            dup = any(set(t.classes) == set(o["classes"]) for o in os_)
            per.append({"task_id": t.task_id, "k": len(t.classes), "n_channels": len(t.channels), "n_samples": len(t.sample_indices), "n_train": nt["n_train"],
                        "class_jaccard_max": round(cj, 3), "sample_jaccard_max": round(sj, 3), "channel_jaccard_max": round(chj, 3), "class_subset_repeats_original": dup})
            L.append(f"| {t.task_id} | {len(t.classes)} | {len(t.channels)} | {len(t.sample_indices)} | {nt['n_train']} | {cj:.2f} | {sj:.2f} | {chj:.2f} | {'是' if dup else '否'} |")
        ext_pairs = [(_jac(a.classes, b.classes), _jac(a.sample_indices, b.sample_indices)) for i, a in enumerate(d.slices) for b in d.slices[i + 1:]]
        check["datasets"][name] = {"slices": per, "ext_vs_ext_class_jaccard_mean": round(float(np.mean([p[0] for p in ext_pairs])), 3),
                                   "ext_vs_ext_sample_jaccard_mean": round(float(np.mean([p[1] for p in ext_pairs])), 3),
                                   "n_dup_class_subset": int(sum(p["class_subset_repeats_original"] for p in per))}
    L += ["", f"原切片逐字段不变：{check['originals_unchanged']}", "",
          "| 数据集 | 新 vs 新 类别 Jaccard 均值 | 新 vs 新 样本 Jaccard 均值 | 类别子集重复原切片数 |", "|---|---:|---:|---:|"]
    for name, c in check["datasets"].items():
        L.append(f"| {name} | {c['ext_vs_ext_class_jaccard_mean']:.2f} | {c['ext_vs_ext_sample_jaccard_mean']:.2f} | {c['n_dup_class_subset']} |")
    (outdir / "slice_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "slice_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    if not check["originals_unchanged"]:
        raise RuntimeError("原切片与 mx3_pq/tasks.json 不一致——注册表流被改动（铁律 3）")
    return check


# ---------------------------------------------------------------- R'
def ext_plan(slice_ids: Sequence[str], *, space: Optional[CLSSpace] = None) -> list:
    """每切片 20 条（10 shared + 10 random 按 id 派生）+ 1 条 GGS，与 32 号 build_plan 同构。"""
    space = space or CLSSpace.from_yaml()
    ggs = pq.ggs_cls()
    ggs_key = pq.cls_key(ggs)
    shared = None
    plan = []
    for sid in slice_ids:
        entries = pq.catalog_for_slice(space, sid, shared)
        if shared is None:
            shared = [(c, k) for org, c, k in entries if org == pq.ORIGIN_SHARED]
        plan += [_g32.Cell(sid, org, c, k) for org, c, k in entries]
        plan.append(_g32.Cell(sid, _g32.ORIGIN_GGS, ggs, ggs_key))
    keys = [(c.slice_id, c.cls_key) for c in plan]
    if len(keys) != len(set(keys)):
        raise ValueError("网格内 (切片, cls_key) 撞键")
    return plan


def write_catalog(outdir: Path, plan: Sequence) -> None:
    shared, per_slice, ggs, seen = [], {}, None, set()
    for c in plan:
        if c.origin == pq.ORIGIN_SHARED and c.cls_key not in seen:
            seen.add(c.cls_key)
            shared.append({"cls_key": c.cls_key, "cls": c.cls.to_dict()})
        elif c.origin == pq.ORIGIN_RANDOM:
            per_slice.setdefault(c.slice_id, []).append({"cls_key": c.cls_key, "cls": c.cls.to_dict()})
        elif c.origin == _g32.ORIGIN_GGS and ggs is None:
            ggs = {"cls_key": c.cls_key, "cls": c.cls.to_dict()}
    (outdir / "cls_catalog.json").write_text(json.dumps({"shared": shared, "random": per_slice, "ggs": ggs}, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_rprime(outdir: Path, *, device: str, resume: bool, smoke: bool, names: Sequence[str]) -> int:
    if smoke and outdir.resolve() == OUT.resolve():
        raise SystemExit("--smoke 不得写进真值目录 results/tclsc_slices")
    outdir.mkdir(parents=True, exist_ok=True)
    ds = load_ext_datasets(names)
    if smoke:
        ds = {k: v for k, v in list(ds.items())[:1]}
        for d in ds.values():
            d.slices = d.slices[:1]
    ids = [t.task_id for d in ds.values() for t in d.slices]
    plan = ext_plan(ids)
    if smoke:
        plan = [c for c in plan if c.origin == pq.ORIGIN_SHARED][:1] + [c for c in plan if c.origin == pq.ORIGIN_RANDOM][:1]
    if not (outdir / "tasks.json").exists() or smoke:
        pq.write_tasks_json(ds, outdir / "tasks.json")
    write_catalog(outdir, ext_plan([t.task_id for d in ds.values() for t in d.slices]))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    cache = outdir / "rprime_cache.jsonl"
    idx = _g32.cache_index(_g32.read_rows(cache))
    order = {pq.ORIGIN_SHARED: 0, pq.ORIGIN_RANDOM: 1, _g32.ORIGIN_GGS: 2}
    plan.sort(key=lambda c: (order[c.origin], c.slice_id))
    todo = _g32.pending_cells(plan, idx, pq.BUDGETS, resume=resume, redo_partial=False)
    st = _g32.cell_states(plan, idx, pq.BUDGETS)
    print(f"R' 网格: 计划 {len(plan)} 格 | done {len(st[_g32.DONE])} / partial {len(st[_g32.PARTIAL])} / terminal {len(st[_g32.TERMINAL])} → 待跑 {len(todo)}", flush=True)
    return _g32.run_jobs(_g32.make_jobs(todo, _g32.slice_index(ds)), budgets=pq.BUDGETS, cache_path=cache, inflight=outdir / "inflight.txt",
                         on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl", eval_ratio=labels_cfg["eval_ratio"],
                         batch_size=train_cfg["batch_size"], lr=train_cfg["lr"], device=device, label="sx")


def cmd_verify(outdir: Path) -> dict:
    rows = _g32.read_rows(outdir / "rprime_cache.jsonl")
    tasks = pq.load_tasks_json(outdir / "tasks.json")
    keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
    want = len(tasks) * 21 * len(pq.BUDGETS)
    errs = [r for r in rows if r.get("error") is not None or r.get("r_prime") is None]
    per_slice = {s: sum(1 for k in keys if k[0] == s) for s in tasks}
    v = {"rows": len(rows), "want": want, "dup_keys": len(keys) - len(set(keys)), "errors": len(errs),
         "incomplete_slices": {s: n for s, n in per_slice.items() if n != 21 * len(pq.BUDGETS)}, "ok": None}
    v["ok"] = bool(len(rows) == want and v["dup_keys"] == 0 and not errs)
    print(f"[verify] rows {len(rows)}/{want} dup {v['dup_keys']} errors {len(errs)} incomplete {len(v['incomplete_slices'])} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- 基线探针 + 特征
def cmd_baseline(outdir: Path, *, device: str, resume: bool, names: Sequence[str], smoke: bool = False) -> None:
    path = outdir / "baseline_cache.jsonl"
    done = {(r["task_id"], int(r["seed"])) for r in _g32.read_rows(path) if r.get("r_prime") is not None} if resume else set()
    ds = load_ext_datasets(names)
    if smoke:
        ds = {k: v for k, v in list(ds.items())[:1]}
        for d in ds.values():
            d.slices = d.slices[:1]
    cls_cfg = inert_cls()
    with parallel_backend("threading", n_jobs=1), open(path, "a", encoding="utf-8") as fh:   # Windows loky spawn 教训（baseline_probe 同）
        for name, d in ds.items():
            for t in d.slices:
                for s in range(2 if smoke else N_BASELINE_SEEDS):
                    if (t.task_id, s) in done:
                        continue
                    w_seed, sp_seed = baseline_seed_specs(t.task_id, s)
                    rec = evaluate_rprime(t, cls_cfg, d.X, d.y, weight_seed=w_seed, split_seed=sp_seed, iters=BASELINE_ITERS, device=device)
                    rp = rec.get("r_prime")
                    row = {"dataset": name, "task_id": t.task_id, "seed": s, "iters": BASELINE_ITERS, **rec,
                           "r_prime": None if rp is None or not np.isfinite(rp) else float(rp)}
                    fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n"); fh.flush()
                    print(f"[baseline] {t.task_id} s{s} R'={row['r_prime']} ({rec.get('seconds')}s)", flush=True)


def baseline_bands_ext(rows: Sequence[Mapping]) -> dict[str, tuple[float, float]]:
    """{切片: (基线均值, 2·max(σ, 量子))}——pq.baseline_bands 同式，缓存换成扩展切片自己的。"""
    import statistics
    by: dict[str, list] = {}
    for r in rows:
        if r.get("r_prime") is not None:
            by.setdefault(r["task_id"], []).append(r)
    out = {}
    for tid, rs in by.items():
        vals = [float(r["r_prime"]) for r in rs]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        quantum = max((1.0 / r["n_eval"] for r in rs if r.get("n_eval")), default=0.0)
        out[tid] = (statistics.fmean(vals), 2.0 * max(std, quantum))
    return out


def phi_row(task_id: str, dataset: str, Xtr: np.ndarray, *, k: int, n_train: int, band: tuple[float, float]) -> dict:
    """φ12（40 号公式：meta ⊕ base ⊕ signal）——只用训练划分。"""
    return {"task_id": task_id, "dataset": dataset, **_g40.phi_meta(Xtr.shape[1], Xtr.shape[2], k, n_train), **_g40.phi_base(*band), **_g40.signal_stats(Xtr)}


def cmd_features(outdir: Path, *, device: str, resume: bool, names: Sequence[str], smoke: bool = False) -> None:
    path = outdir / "ctx_ext.jsonl"
    done = {r["task_id"] for r in _g32.read_rows(path)} if resume else set()
    bands = baseline_bands_ext(_g32.read_rows(outdir / "baseline_cache.jsonl"))
    ds = load_ext_datasets(names)
    if smoke:
        ds = {k: v for k, v in list(ds.items())[:1]}
        for d in ds.values():
            d.slices = d.slices[:1]
    need = tuple(rep_cols_at(60)) + tuple(rep_cols_at(290)) + PHI_COLS + tuple(RESPONSE_PROFILE_COLS)
    with open(path, "a", encoding="utf-8") as fh:
        for name, d in ds.items():
            for t in d.slices:
                if t.task_id in done:
                    continue
                if t.task_id not in bands:
                    raise KeyError(f"{t.task_id} 缺基线探针（先跑 --baseline）")
                Xt, yt = materialize_task(t, d.X, d.y)
                tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(t.task_id, "train_eval_split"))
                t0 = time.time()
                ea = response_profile(Xt[tr], yt[tr], task_type=t.task_type, seeds=RESPONSE_SEEDS)
                t1 = time.time()
                prof, _acc = _g69.probe_profiles(Xt[tr], yt[tr], t.task_type, device=device)
                et = _g69.feature_row(t.task_id, name, prof, seconds=time.time() - t1, acc290=None)
                phi = phi_row(t.task_id, name, Xt[tr], k=len(t.classes), n_train=int(len(tr)), band=bands[t.task_id])
                row = {"task_id": t.task_id, "dataset": name, "n_train": int(len(tr)), "seconds": round(time.time() - t0, 1), "seconds_ea": round(t1 - t0, 1),
                       **ea, **{c: phi[c] for c in PHI_COLS}, **{k: v for k, v in et.items() if k not in ("task_id", "dataset", "seconds")}, "seconds_et": et["seconds"]}
                missing = [c for c in need if c not in row]
                if missing:
                    raise KeyError(f"{t.task_id} 特征行缺列 {missing[:5]}…")
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                print(f"[features] {t.task_id}: n={row['n_train']} {row['seconds']}s rep_highpass05@60 {row['rep_highpass05@60']:+.2f} ret_highpass05 {row['ret_highpass05']:+.2f} base {row['base_mean']:.3f}", flush=True)


# ---------------------------------------------------------------- 训练 / 排名
def load_ext_rows(outdir: Path) -> list[dict]:
    return [r for r in _g32.read_rows(outdir / "rprime_cache.jsonl") if r.get("error") is None and r.get("r_prime") is not None and r["origin"] != "ggs"]


def rows_for_point(orig: Sequence[dict], ext: Sequence[dict], s: int) -> list[dict]:
    """S6 = 原；S9 = 原 + x00–x02；S12 = 原 + x00–x05（嵌套）。"""
    if s <= 6:
        return list(orig)
    return list(orig) + [r for r in ext if ext_index(r["task_id"]) < s - 6]


def test_slices_for(rows: Sequence[dict], target: str) -> list[str]:
    """考侧永远只用原切片（126 实例，与 T / LC 逐位可比）。"""
    return sorted({r["task_id"] for r in rows if r["dataset"] == target and not is_ext(r["task_id"])})


def ctx_table(outdir: Path) -> dict[str, dict]:
    t = {r["task_id"]: r for r in read_jsonl(S1ET / "uea" / "ctx_uea.jsonl")}
    t.update({r["task_id"]: r for r in _g32.read_rows(outdir / "ctx_ext.jsonl")})
    return t


def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, points: Sequence[int],
              folds_sel: Optional[Sequence[str]] = None) -> None:
    orig, delta = _g57.load_source(SOURCE)
    ext = load_ext_rows(outdir)
    table = ctx_table(outdir)
    for s in points:
        if s <= 6:
            continue
        rows = rows_for_point(orig, ext, s)
        n_ext = len({r["task_id"] for r in rows if is_ext(r["task_id"])})
        for tid in {r["task_id"] for r in rows}:
            if tid not in table:
                raise KeyError(f"ctx 表缺切片 {tid}（先跑 --features）")
        tag = f"S{s}"
        for target in (folds_sel or FOLDS):
            for arm in ARMS:
                for seed in seeds:
                    if resume and (outdir / "models" / tag / target / f"{arm}_s{seed}.pt").exists():
                        continue
                    run = _g73.train_fold_uea(rows, delta, table, target=target, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs, tag=tag)
                    _g73._log(outdir, f"[{tag}/{target}/{arm}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} (ext {n_ext}) recs {run['n_train_records']} pairs {run['train_pairs_last']}")


def stage_s6_models(outdir: Path) -> None:
    """S6 满量模型复用：seeds 0–2 = T 检验 ft60 / s1phi blind；seeds 3–5 = LC rep_s345/S6（与 33 号路径逐位同一）。"""
    for target in FOLDS:
        dst = outdir / "models" / "S6" / target
        dst.mkdir(parents=True, exist_ok=True)
        srcs = {f"ft60_hard_s{s}.pt": S1ET / "uea" / "models" / target / f"ft60_hard_s{s}.pt" for s in (0, 1, 2)}
        srcs.update({f"blind_diff_hard_s{s}.pt": S1PHI / "binned" / "models" / target / f"blind_diff_hard_s{s}.pt" for s in (0, 1, 2)})
        srcs.update({f"{a}_s{s}.pt": LC / "rep_s345" / "models" / "S6" / target / f"{a}_s{s}.pt" for a in ARMS for s in (3, 4, 5)})
        for fn, src in srcs.items():
            if not (dst / fn).exists():
                if not src.exists():
                    raise FileNotFoundError(f"S6 复用模型缺 {src}")
                shutil.copy2(src, dst / fn)


def cmd_rank(outdir: Path, *, points: Sequence[int] = S_POINTS) -> dict:
    stage_s6_models(outdir)
    rk: dict = {}
    for s in points:
        tag = f"S{s}"
        if not (outdir / "models" / tag).exists():
            continue
        rk[tag] = {}
        for g in SEED_GROUPS:
            rk[tag]["".join(map(str, g))] = _g73.rank_uea_tag(outdir, tag, seeds=g)
            print(f"[rank] {tag} seeds {g}", flush=True)
    (outdir / "rankings.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    return rk


# ---------------------------------------------------------------- 分析
def _eval_group(rk_group: Mapping, cat_z, delta, group: str) -> list[dict]:
    ev = _g58._eval_all({t: {"normal": per["normal"], "cross": {}, "zero": {}} for t, per in rk_group.items()}, cat_z, delta, list(ARMS), "normal")
    for r in ev:
        r["group"] = group
    return ev


def _paired_same_arm(rows_a: Sequence[dict], rows_b: Sequence[dict], arm: str, key: str = "3") -> dict:
    """同一臂在两个点上的 regret@k 配对（实例 = (组, 折, 切片, 预算)），H1：b 更低。"""
    ka = {(r["group"], r["fold"], r["slice"], int(r["budget"])): r["regret"][arm + key] for r in rows_a if arm + key in r["regret"]}
    kb = {(r["group"], r["fold"], r["slice"], int(r["budget"])): r["regret"][arm + key] for r in rows_b if arm + key in r["regret"]}
    ks = sorted(set(ka) & set(kb))
    x, y = np.array([ka[k] for k in ks]), np.array([kb[k] for k in ks])
    return {"n": int(x.size), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "p": float(_g58.paired_wilcoxon_less(y, x)) if x.size else None,
            "wins_b": int(np.sum(y < x)), "losses_b": int(np.sum(y > x)), "ties": int(np.sum(y == x))}


def gate_s1(pr: Mapping) -> bool:
    """G-S1：S12（b）vs S6（a）配对 p < α 且均值差 ≤ −G1_MIN_GAIN。"""
    return bool(pr.get("p") is not None and pr["p"] < ALPHA and (pr["mean_b"] - pr["mean_a"]) <= -G1_MIN_GAIN)


def verdict_sx(gates: Mapping[str, bool], curve: str) -> str:
    if gates["G-S1"] and gates["G-S2"]:
        return "SLICE-WIN"
    if gates["G-S1"]:
        return "SLICE-BLIND"
    if curve == "RISING":
        return "SLICE-RISING"
    return "SLICE-FLAT"


def cmd_analyze(outdir: Path) -> dict:
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    rk = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    ev: dict[str, list] = {}
    for tag, groups in rk.items():
        ev[tag] = []
        for g, r in groups.items():
            ev[tag] += _eval_group(r, cat_z, delta, g)
    curves = []
    for s in S_POINTS:
        tag = f"S{s}"
        if tag not in ev or not ev[tag]:
            continue
        c = {"size": s, **{a: _g73._pooled(ev[tag], a) for a in ARMS}}
        c["gain"] = c[B_ARM]["regret3"] - c[E_ARM]["regret3"]
        curves.append(c)
    by = {c["size"]: c for c in curves}
    c0 = {a: {"value": round(by[6][a]["regret3"], 4), "ref": C0_REF[a], "pass": abs(by[6][a]["regret3"] - C0_REF[a]) < C0_TOL} for a in ARMS} if 6 in by else None
    paired = {}
    for a_s, b_s in ((6, 12), (6, 9), (9, 12)):
        if a_s in by and b_s in by:
            paired[f"S{a_s}->S{b_s}"] = {arm: _paired_same_arm(ev[f"S{a_s}"], ev[f"S{b_s}"], arm) for arm in ARMS}
    gates = {}
    if "S6->S12" in paired:
        gates["G-S1"] = gate_s1(paired["S6->S12"][E_ARM])
        gates["G-S2"] = bool(by[12]["gain"] > by[6]["gain"])
    pts = [(c["size"], c[E_ARM]["regret3"], c[B_ARM]["regret3"]) for c in curves]
    curve = _g73.axis_verdict(pts) if len(pts) >= 2 else None
    verdict = verdict_sx(gates, curve or "FLAT") if gates else None
    a = {"protocol": "results/tclsc_slices/PROTOCOL.md", "curves": curves, "c0": c0, "paired": paired, "gates": gates, "curve": curve, "verdict": verdict}
    off = outdir / "official_S12.json"
    if off.exists():
        a["official"] = json.loads(off.read_text(encoding="utf-8"))
    a["predictions"] = predictions(a, by)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(verdict, gates, curve, flush=True)
    return a


def predictions(a: dict, by: Mapping[int, dict]) -> dict:
    p = {"P1": {"desc": "C0 过", "hit": bool(all(v["pass"] for v in a["c0"].values())) if a.get("c0") else None}}
    if 6 in by and 12 in by:
        d = by[12][E_ARM]["regret3"] - by[6][E_ARM]["regret3"]
        p["P2"] = {"desc": "S12 E_T 比 S6 低 ≥ 0.002", "value": round(d, 4), "hit": bool(d <= -0.002)}
        p["P3"] = {"desc": "G-S1 显著", "hit": a["gates"].get("G-S1")}
        p["P4"] = {"desc": "E 增益(S12) > 0.0069", "value": round(by[12]["gain"], 4), "hit": bool(by[12]["gain"] > 0.0069)}
        p["P5"] = {"desc": "任务盲 S12 < S6", "hit": bool(by[12][B_ARM]["regret3"] < by[6][B_ARM]["regret3"])}
    if a.get("official"):
        m = a["official"]["summary"]["et_tclsc"]["mean_seed0"]
        p["P6"] = {"desc": "副读数 S12 官方六靶 seed-0 均值 ≥ 0.564", "value": m, "hit": bool(m >= 0.564)}
    if all(s in by for s in (6, 9, 12)):
        e6, e9, e12 = (by[s][E_ARM]["regret3"] for s in (6, 9, 12))
        p["P7"] = {"desc": "S9 在 S6 与 S12 之间", "hit": bool(min(e6, e12) <= e9 <= max(e6, e12))}
    return p


def markdown_tables(a: dict) -> str:
    L = ["# SX 检验 tables：每集加切片 S ∈ {6, 9, 12}", "", f"读数码 **{a['verdict']}**（门 {a['gates']}，曲线 {a['curve']}）", "",
         "| S | n | E_T regret@3 | blind regret@3 | E 增益 | E_T regret@1 | blind regret@1 | E_T M1 | blind M1 | E_T M0 | blind M0 |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in a["curves"]:
        e, b = c[E_ARM], c[B_ARM]
        L.append(f"| {c['size']} | {e['n']} | {e['regret3']:.4f} | {b['regret3']:.4f} | {c['gain']:+.4f} | {e['regret1']:.4f} | {b['regret1']:.4f} | "
                 f"{e['M1']:.3f} | {b['M1']:.3f} | {e['M0']:.3f} | {b['M0']:.3f} |")
    L += ["", "配对（同臂，两点；H1 后者更低）：", "", "| 对 | 臂 | n | 前 | 后 | p | 胜:负:平 |", "|---|---|---:|---:|---:|---:|---|"]
    for k, per in a["paired"].items():
        for arm, pr in per.items():
            L.append(f"| {k} | {arm} | {pr['n']} | {pr['mean_a']:.4f} | {pr['mean_b']:.4f} | {pr['p']:.2g} | {pr['wins_b']}:{pr['losses_b']}:{pr['ties']} |")
    folds = sorted({f for c in a["curves"] for f in c[E_ARM]["regret3_by_fold"]})
    if folds:
        L += ["", "逐折 regret@3（E_T / blind）：", "", "| S | " + " | ".join(folds) + " |", "|---:|" + "---:|" * len(folds)]
        for c in a["curves"]:
            L.append(f"| {c['size']} | " + " | ".join(f"{c[E_ARM]['regret3_by_fold'].get(f, float('nan')):.3f} / {c[B_ARM]['regret3_by_fold'].get(f, float('nan')):.3f}" for f in folds) + " |")
    if a.get("c0"):
        L += ["", f"C0：{a['c0']}"]
    if a.get("official"):
        o = a["official"]
        L += ["", "## 副读数：S12 折模型代入 D′ 口径（官方六靶信任域 SH6-3-3，seed 0）", "", "| 靶 | D′ S6 E_T | S12 E_T | 冠军变 | D′ S6 盲 | S12 盲 |", "|---|---:|---:|---|---:|---:|"]
        for d, r in o["targets"].items():
            L.append(f"| {d} | {r['ref']['et_tclsc']['acc']:.3f} | {r['et_tclsc']['acc']:.3f} | {'是' if r['et_tclsc']['champion'] != r['ref']['et_tclsc']['champion'] else '否'} | "
                     f"{r['ref']['blind_tclsc']['acc']:.3f} | {r['blind_tclsc']['acc']:.3f} |")
        L.append(f"| **均值** | {o['summary']['ref_et_mean']:.3f} | {o['summary']['et_tclsc']['mean_seed0']:.3f} | | {o['summary']['ref_blind_mean']:.3f} | {o['summary']['blind_tclsc']['mean_seed0']:.3f} |")
    L += ["", f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


def cmd_simulate_official(outdir: Path, *, tag: str = "S12") -> dict:
    """副读数：S{tag} 折模型（seeds 0–2）代入 71 号 D′（信任域 + 裁 ±4 + SH6-3-3，seed-0 缓存），对照 D′ S6。零 GPU。"""
    _g71 = importlib.import_module("scripts.71_tclsc_s1et_deploy")
    _g71.ET_MODELS = outdir / "models" / tag
    _g71.BLIND_MODELS = outdir / "models" / tag
    ref = json.loads((_g71.OUT / "analysis_trust.json").read_text(encoding="utf-8"))["targets"]
    rows = list(read_jsonl(_g71.DEPLOY / "runs.jsonl"))
    cands_all = json.loads((_g71.DEPLOY / "candidates.json").read_text(encoding="utf-8"))
    et_rows = {r["dataset"]: r for r in read_jsonl(_g71.OUT / "et_official.jsonl")}
    phi_all = json.loads(_g71.PHI_OFFICIAL.read_text(encoding="utf-8"))
    per = {}
    for d in _g71.TARGETS:
        traj, _f, budgets = _g71.catalog_traj(rows, d)
        allowed = set(cands_all[d]["prior_order"])
        lists = _g71.rank_official(d, cands_all[d], et_rows[d], phi_all[d], device="cpu", clip=_g71.CLIP)
        res = {"ref": {n: {"acc": ref[d]["lists"][n]["acc"], "champion": ref[d]["lists"][n]["champion"]} for n in _g71.LISTS}}
        for n in _g71.LISTS:
            r = _g71.sh_on_list(_g71.restrict_list(lists[n], allowed), traj, ks=_g71.SH_ARMS["SH6-3-3"], budgets=budgets)
            res[n] = {"acc": r["acc"], "champion": r["champion"], "match_seed0": bool(r["acc"] >= _g71.TS2VEC_PAPER[d] - _g71.MATCH_MARGIN)}
        per[d] = res
    summary = {n: {"mean_seed0": float(np.mean([per[d][n]["acc"] for d in _g71.TARGETS])), "n_match_seed0": int(sum(per[d][n]["match_seed0"] for d in _g71.TARGETS)),
                   "n_champion_changed": int(sum(per[d][n]["champion"] != per[d]["ref"][n]["champion"] for d in _g71.TARGETS))} for n in _g71.LISTS}
    summary["ref_et_mean"] = float(np.mean([per[d]["ref"]["et_tclsc"]["acc"] for d in _g71.TARGETS]))
    summary["ref_blind_mean"] = float(np.mean([per[d]["ref"]["blind_tclsc"]["acc"] for d in _g71.TARGETS]))
    o = {"tag": tag, "targets": per, "summary": summary}
    (outdir / f"official_{tag}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[official {tag}] E_T 均值 {summary['et_tclsc']['mean_seed0']:.3f}（D′ S6 {summary['ref_et_mean']:.3f}）冠军变 {summary['et_tclsc']['n_champion_changed']}/6；"
          f"盲 {summary['blind_tclsc']['mean_seed0']:.3f}（{summary['ref_blind_mean']:.3f}）", flush=True)
    return o


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("slice", "rprime", "verify", "baseline", "features", "train", "rank", "analyze", "simulate-official", "smoke", "resume"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(EXT_DATASETS))
    ap.add_argument("--points", nargs="+", type=int, default=list(S_POINTS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[s for g in SEED_GROUPS for s in g])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.slice:
        cmd_slice(args.outdir)
    if args.rprime:
        cmd_rprime(args.outdir, device=args.device, resume=args.resume, smoke=args.smoke, names=args.datasets)
    if args.verify:
        cmd_verify(args.outdir)
    if args.baseline:
        cmd_baseline(args.outdir, device=args.device, resume=args.resume, names=args.datasets, smoke=args.smoke)
    if args.features:
        cmd_features(args.outdir, device=args.device, resume=args.resume, names=args.datasets, smoke=args.smoke)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, points=args.points, folds_sel=args.folds)
    if args.rank:
        cmd_rank(args.outdir, points=args.points)
    if args.simulate_official:
        cmd_simulate_official(args.outdir)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
