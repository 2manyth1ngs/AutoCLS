"""共享种子交叉探针（协议 results/shared_seed_probe/PROTOCOL.md，先于数据写死）。

诊断"配对比较（common random numbers）能否压 R' 噪声"：现有设计 weight_seed =
run_seed_for(task_id, cls_key) 让每个 (task, cls) 用各自初始化，比较 cls_a/cls_b 时
ΔR' 同时含 CLS 效果与初始化效果（R' = f(cls) + g(seed) + ε(cls,seed)，独立配对
噪声方差 2Var(g)+2Var(ε)）；共享种子（weight_seed 只依赖 task+rep）则 g 完全
抵消，只剩 2Var(ε)。Var(g):Var(ε) 的比例从未测过——既有数据种子与 CLS 非交叉，
无法分解。本探针补完整 10×3 交叉：Handwriting @60 前 2 任务 × 10 shared CLS ×
3 共享 weight_seed = 60 条全新 R'（共享种子流与 canonical/噪声流不相交，零复用）。

用法（幂等、--resume 贯穿）：
    python scripts/9_shared_seed_probe.py --run --resume
    python scripts/9_shared_seed_probe.py --analyze
"""
from __future__ import annotations

import os

# 造 label CPU-bound 教训：不设 OMP=1 慢 4.2×（须在 numpy/torch 导入前生效）
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import importlib
import json
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402

from autocls.labeling import (  # noqa: E402
    evaluate_rprime, load_probe_cls, run_seed_for, shared_seed_for,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402

# 池加载/常量/读 jsonl 复用 2c；任务盲天花板复用 outlet_y 的 ceiling（同一语义，
# published 0.9525/0.7473 即出自该实现）。模块名数字开头走 importlib。
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")
_nsa = importlib.import_module("scripts.noise_structure_analysis")

DEFAULT_OUTDIR = ROOT / "results" / "shared_seed_probe"
DATASET = "Handwriting"          # 铁律 6：源池骨干，噪声基准 δ=0.110 在此标定
REPS = (0, 1, 2)                 # 3 个共享 weight_seed（全 CLS 共用，完整交叉）
SIGNAL_P50_HW = 0.033            # outlet_y §5：HW 任务特异票容 |Δ| p50 → required-k
SQRT2 = float(np.sqrt(2.0))
INDEP_P95_SANITY = (0.055, 0.22)  # 有效性闸：独立臂 P95 对 0.110 的漂移带（±2×/½）
DELTA_HW_PUBLISHED = 0.110


# ---------------------------------------------------------------- 采集
def seeds_for(task_id: str, rep: int) -> tuple[int, int]:
    """weight = 共享种子（不含 cls_key，本探针核心）；split 仍 canonical。"""
    return shared_seed_for(task_id, rep), run_seed_for(task_id, "train_eval_split")


def build_cells(per_ds: list, iters_map: dict, shared_cls: list) -> list[tuple]:
    """(name, X, y, task, cls, cls_key, rep, iters) × HW 前 2 任务 × 10 shared × 3 rep。"""
    return [(name, X, y, t, c, k, rep, iters_map[name])
            for name, X, y, tasks in per_ds if name == DATASET
            for t in tasks[:_c2.NOISE_TASKS]
            for c, k in shared_cls
            for rep in REPS]


def run(outdir: Path, per_ds: list, iters_map: dict, shared_cls: list,
        train_cfg: dict, labels_cfg: dict, device: str, resume: bool) -> None:
    cache_path = outdir / "probe_cache.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in _c2.read_jsonl(cache_path) if r.get("r_prime") is not None}
    work = build_cells(per_ds, iters_map, shared_cls)
    todo = [w for w in work
            if not (resume and (w[3].task_id, w[5], w[7], w[6]) in done)]
    print(f"共享种子探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：fit_svm 的 GridSearchCV 逐条 spawn loky 占 99% 耗时（长期教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (name, X, y, t, c, k, rep, iters) in enumerate(todo):
            w_seed, sp_seed = seeds_for(t.task_id, rep)

            def make_row(rec, _n=name, _t=t, _k=k, _c=c, _rep=rep, _it=iters):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it,
                        "shared_seed": True, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(
                    t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                    iters=iters, eval_ratio=labels_cfg["eval_ratio"],
                    batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                    device=device),
                make_row, cache_path=cache_path, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=k, iters=iters,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": name, "shared_rep": rep},
                log=lambda m: print(f"[shseed {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[shseed {i + 1}/{len(todo)}] {t.task_id} {k[:8]} rep{rep} "
                  f"R'={full['r_prime']} ({full['seconds']}s) ETA {eta / 60:.1f}min",
                  flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 纯分析函数
def _p95(vals: list[float]) -> float:
    """与 2c calibrate 同式：sorted 后取 round(0.95*(n-1)) 索引（最近秩不插值）。"""
    ds = sorted(vals)
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


def load_cells(rows: list[dict]) -> dict:
    """(task_id, cls_key) → {rep: r_prime}（null 跳过；同键后行覆盖前行）。"""
    cells: dict = defaultdict(dict)
    for r in rows:
        if r.get("r_prime") is None:
            continue
        cells[(r["task_id"], r["cls_key"])][r["seed"]] = float(r["r_prime"])
    return cells


def validate_rows(rows: list[dict]) -> list[tuple]:
    """行级自检（隔离闸）：shared_seed 标记 + weight/split 种子逐行核对。
    抓"实现回退到生产口径（种子含 cls_key）"这类致命错误。"""
    bad = []
    for r in rows:
        if (not r.get("shared_seed")
                or r.get("weight_seed") != shared_seed_for(r["task_id"], r["seed"])
                or r.get("split_seed") != run_seed_for(r["task_id"], "train_eval_split")):
            bad.append((r["task_id"], r["cls_key"], r.get("seed")))
    return bad


def build_matrix(cells: dict, task_id: str, shared_keys: list[str]):
    """→ (R'[n_cls × 3] 矩阵, 收录 keys, 剔除 keys)：只收 3 rep 齐全的 CLS，
    行序 = 探针目录序（跨任务可比）。"""
    used, dropped, rows = [], [], []
    for k in shared_keys:
        sv = cells.get((task_id, k), {})
        if all(r in sv for r in REPS):
            used.append(k)
            rows.append([sv[r] for r in REPS])
        else:
            dropped.append(k)
    m = np.array(rows, dtype=float) if rows else np.zeros((0, len(REPS)))
    return m, used, dropped


def anova2(m: np.ndarray) -> dict:
    """双因素（cls × seed）SS 分解。单格无重复 → 交互与残差混同（resid）。
    Var(g)̂=(MS_seed−MS_resid)/n_cls 可为负（seed 主效应≈0 时），如实报告不截断。"""
    gm = float(m.mean())
    cls_eff = m.mean(axis=1) - gm
    seed_eff = m.mean(axis=0) - gm
    resid = m - gm - cls_eff[:, None] - seed_eff[None, :]
    n_cls, n_seed = m.shape
    ss = {"cls": float((cls_eff ** 2).sum() * n_seed),
          "seed": float((seed_eff ** 2).sum() * n_cls),
          "resid": float((resid ** 2).sum())}
    sst = sum(ss.values())
    ms_seed = ss["seed"] / (n_seed - 1)
    ms_resid = ss["resid"] / ((n_cls - 1) * (n_seed - 1))
    return {"ss": ss,
            "ss_share": {k: (v / sst if sst else float("nan")) for k, v in ss.items()},
            "ms_seed": ms_seed, "ms_resid": ms_resid,
            "var_g_hat": (ms_seed - ms_resid) / n_cls,
            "var_eps_hat": ms_resid,
            "grand_mean": gm,
            "seed_effects": [float(v) for v in seed_eff]}


def indep_diffs(m: np.ndarray) -> list[float]:
    """独立臂噪声：同 CLS 跨种子重复差 |ΔR'|（δ 标定同口径，方差=2Var(g)+2Var(ε)）。"""
    return [abs(float(m[c, i] - m[c, j])) for c in range(m.shape[0])
            for i, j in combinations(range(m.shape[1]), 2)]


def shared_rep_diffs(m: np.ndarray) -> list[float]:
    """共享臂：配对统计量重复差 |D|，D = Δ_shared(sᵢ) − Δ_shared(sⱼ)。
    方差 = 4Var(ε) = 2×共享臂单次比较噪声 → 单次口径除 √2（协议 §3）。"""
    out = []
    for a, b in combinations(range(m.shape[0]), 2):
        d = m[a] - m[b]
        out += [abs(float(d[i] - d[j])) for i, j in combinations(range(m.shape[1]), 2)]
    return out


def centered_devs(m: np.ndarray) -> tuple[list[float], list[float]]:
    """佐证口径（用户构造）：逐 CLS 对以跨种子均值 Δ 为中心的单次偏差。
    共享偏差=|Δ_shared(s)−center|；独立偏差=|R'(a,sᵢ)−R'(b,sⱼ)−center| (i≠j)。
    ε 项两臂同等缩水(×2/3)、g 项只进独立臂不缩水 → 对降幅略偏乐观，只作佐证。"""
    sh, ind = [], []
    for a, b in combinations(range(m.shape[0]), 2):
        center = float((m[a] - m[b]).mean())
        sh += [abs(float(m[a, s] - m[b, s]) - center) for s in range(m.shape[1])]
        ind += [abs(float(m[a, i] - m[b, j]) - center)
                for i in range(m.shape[1]) for j in range(m.shape[1]) if i != j]
    return sh, ind


def paired_direction_stability(m: np.ndarray, delta: float) -> dict:
    """|mean Δ_shared| ≥ δ 的 CLS 对里，3 种子 Δ 全同号（无零）的比例
    ——配对设计下存活 label 的跨种子方向稳定性（outlet_y §4 对应量）。"""
    n = ok = 0
    for a, b in combinations(range(m.shape[0]), 2):
        d = m[a] - m[b]
        if abs(float(d.mean())) < delta:
            continue
        n += 1
        signs = set(np.sign(d).tolist())
        ok += int(signs in ({1.0}, {-1.0}))
    return {"n_pairs": n, "stable_frac": ok / n if n else None}


BAND_ORDER = ("red", "mid", "green")


def band_of(reduction: float) -> str:
    """主判据档位（协议 §4）：≥50% 绿 / 20–50% 中 / <20% 红。"""
    return "green" if reduction >= 0.50 else "mid" if reduction >= 0.20 else "red"


def decide(red_corr: float, red_raw: float) -> str:
    """预注册：√2 校正与原始（保守上界）两口径若跨档，取更保守（更低）档。"""
    bands = {band_of(red_corr), band_of(red_raw)}
    for b in BAND_ORDER:
        if b in bands:
            return b
    raise AssertionError("unreachable")


def required_k(p95_single: float) -> float:
    """outlet_y §5 同式：容 HW 任务特异票 |Δ| p50=0.033 所需 k=(P95/0.033)²。"""
    return float((p95_single / SIGNAL_P50_HW) ** 2)


def delta_from_p95(p95_val: float, quanta: list[float]) -> float:
    """2c calibrate 同式：δ = max(P95, 2×最大量子) 上取整 0.005。"""
    floor = 2.0 * max(quanta) if quanta else 0.0
    return float(np.ceil(max(p95_val, floor) / 0.005) * 0.005)


# ---------------------------------------------------------------- 分析主流程
def _stats(vals: list[float]) -> dict:
    return {"n": len(vals), "median": statistics.median(vals),
            "p95": _p95(vals), "max": max(vals)}


def analyze(outdir: Path, n_shared: int) -> dict:
    rows = _c2.read_jsonl(outdir / "probe_cache.jsonl")
    cells = load_cells(rows)
    shared_keys = [k for _, k in load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]]
    task_ids = sorted({tid for tid, _ in cells})
    res: dict = {"protocol": "results/shared_seed_probe/PROTOCOL.md"}

    # ---- 有效性闸（协议 §5）----
    keys = [(r["task_id"], r["cls_key"], r["iters"], r["seed"]) for r in rows]
    seed_check = validate_rows(rows)
    mats = {tid: build_matrix(cells, tid, shared_keys) for tid in task_ids}
    validity = {
        "n_rows": len(rows),
        "n_finite": sum(1 for r in rows if r.get("r_prime") is not None),
        "n_dup_keys": len(keys) - len(set(keys)),
        "n_seed_violations": len(seed_check), "seed_violations": seed_check[:10],
        "n_tasks": len(task_ids),
        "usable_cls_per_task": {tid: len(mats[tid][1]) for tid in task_ids},
        "dropped_cls": {tid: mats[tid][2] for tid in task_ids if mats[tid][2]},
        "devices": sorted({r.get("device_name", "?") for r in rows}),
    }
    validity["passed"] = (
        validity["n_finite"] == 60 and validity["n_seed_violations"] == 0
        and validity["n_tasks"] == 2
        and all(v >= 8 for v in validity["usable_cls_per_task"].values()))
    res["validity"] = validity

    # ---- 噪声对照三口径（协议 §3）----
    per_task, pool_ind, pool_sh, pool_dev_sh, pool_dev_ind = {}, [], [], [], []
    for tid in task_ids:
        m = mats[tid][0]
        ind, sh = indep_diffs(m), shared_rep_diffs(m)
        dsh, dind = centered_devs(m)
        per_task[tid] = {"indep": _stats(ind), "shared_rep_diff": _stats(sh),
                         "anova": anova2(m)}
        pool_ind += ind
        pool_sh += sh
        pool_dev_sh += dsh
        pool_dev_ind += dind
    res["per_task"] = per_task

    p95_ind = _p95(pool_ind)
    p95_sh_raw = _p95(pool_sh)
    p95_sh_corr = p95_sh_raw / SQRT2
    med_ind = statistics.median(pool_ind)
    med_sh_raw = statistics.median(pool_sh)
    red_corr = 1.0 - p95_sh_corr / p95_ind
    red_raw = 1.0 - p95_sh_raw / p95_ind
    p95_dev_sh, p95_dev_ind = _p95(pool_dev_sh), _p95(pool_dev_ind)
    # pooled ANOVA：两任务 SS 相加后的份额
    ss_pool = {k: sum(per_task[t]["anova"]["ss"][k] for t in task_ids)
               for k in ("cls", "seed", "resid")}
    sst = sum(ss_pool.values())
    res["pooled"] = {
        "indep": _stats(pool_ind),
        "shared_rep_diff": _stats(pool_sh),
        "p95_shared_corrected_sqrt2": p95_sh_corr,
        "median_shared_corrected_sqrt2": med_sh_raw / SQRT2,
        "reduction_corr": red_corr, "reduction_raw": red_raw,
        "reduction_median_corr": 1.0 - (med_sh_raw / SQRT2) / med_ind,
        "dev_witness": {"p95_shared": p95_dev_sh, "p95_indep": p95_dev_ind,
                        "reduction": 1.0 - p95_dev_sh / p95_dev_ind},
        "anova_ss_share": {k: v / sst for k, v in ss_pool.items()},
    }

    sanity_ok = INDEP_P95_SANITY[0] <= p95_ind <= INDEP_P95_SANITY[1]
    band_final = decide(red_corr, red_raw)
    res["verdict"] = {
        "p95_indep": p95_ind, "p95_shared_corr": p95_sh_corr,
        "p95_shared_raw": p95_sh_raw,
        "reduction_corr": red_corr, "reduction_raw": red_raw,
        "band_corr": band_of(red_corr), "band_raw": band_of(red_raw),
        "band_final": band_final,
        "indep_p95_sanity_ok": sanity_ok,
        "note": None if sanity_ok else
        f"独立臂 P95={p95_ind:.3f} 超出 {INDEP_P95_SANITY} 漂移带：标定漂移，主判据暂缓解读（用户拍板）",
    }

    # ---- 派生诊断（协议 §6，不参与判定）----
    quanta = [1.0 / r["n_eval"] for r in rows
              if r.get("n_eval") and r.get("r_prime") is not None]
    delta_paired = delta_from_p95(p95_sh_corr, quanta)
    derived: dict = {
        "delta_paired": delta_paired,
        "delta_paired_raw": delta_from_p95(p95_sh_raw, quanta),
        "required_k_corr": required_k(p95_sh_corr),
        "required_k_raw": required_k(p95_sh_raw),
        "required_k_published": {"k1": 11.0, "swa": 5.5},
    }
    # 探针内真配对天花板：逐共享种子把 R'(·,s) 当作该任务的一份 k=1 配对测量
    probe_ceils, stab = [], {}
    for s in REPS:
        rv = {tid: {k: float(mats[tid][0][i, s])
                    for i, k in enumerate(mats[tid][1])} for tid in task_ids}
        c = _nsa.ceiling(rv, task_ids, shared_keys,
                         _nsa.const_delta({DATASET: delta_paired}))
        c.pop("votes", None)
        probe_ceils.append(c)
    for tid in task_ids:
        stab[tid] = paired_direction_stability(mats[tid][0], delta_paired)
    derived["probe_paired_ceiling_per_seed"] = probe_ceils
    derived["paired_direction_stability"] = stab

    # labels_workpoint 反事实（k=1 独立种子值 + δ_paired 混合口径，outlet_y 变体 B
    # 近似同款局限：survival 带内的 k=1 噪声翻转仍在，非真配对数据）
    D = _nsa.load()
    hw_tasks = [t for t in D["tasks"] if t.startswith("Handwriting_")]
    dmap_pub = {n: D["calib"][n]["delta"] for n in D["calib"]}

    def _ceil(rvals, tasks, dmap):
        c = _nsa.ceiling(rvals, tasks, D["shared"], _nsa.const_delta(dmap))
        c.pop("votes", None)
        return c

    ratio = p95_sh_corr / p95_ind
    delta_um_scaled = delta_from_p95(
        D["calib"]["UniMiB-SHAR"]["p95_replicate_diff"] * ratio, [])
    derived["ceiling_hw12"] = {
        "paired_delta": _ceil(D["r0"], hw_tasks, {DATASET: delta_paired}),
        "published_delta_ref": _ceil(D["r0"], hw_tasks, {DATASET: dmap_pub[DATASET]}),
        "delta0_ref": _ceil(D["r0"], hw_tasks, {DATASET: 0.0}),
        "note": "k=1 独立种子 R' + δ_paired 混合口径（非真配对数据）",
    }
    derived["ceiling_pooled28_scaled"] = {
        "delta_um_scaled": delta_um_scaled, "hw_um_ratio_applied": ratio,
        "paired_delta": _ceil(D["r0"], D["tasks"],
                              {DATASET: delta_paired, "UniMiB-SHAR": delta_um_scaled}),
        "published_ref": _ceil(D["r0"], D["tasks"], dmap_pub),
        "assumption": "UM 配对噪声按 HW 降幅同比例缩放（未实测，仅参考）",
    }
    res["derived"] = derived

    # ---- 预测对账（协议 §7）----
    resid_share = res["pooled"]["anova_ss_share"]["resid"]
    seed_share = res["pooled"]["anova_ss_share"]["seed"]
    res["predictions"] = {
        "P1_p95_indep_in_band": {"band": [0.08, 0.15], "value": p95_ind,
                                 "hit": 0.08 <= p95_ind <= 0.15},
        "P2_resid_share_gt_seed_share": {"resid": resid_share, "seed": seed_share,
                                         "hit": resid_share > seed_share},
        "P3_band_mid": {"band_final": band_final, "hit": band_final == "mid"},
    }

    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 控制台判读 ----
    v = validity
    print(f"\n[有效性] 行 {v['n_rows']}/finite {v['n_finite']}/60 dup={v['n_dup_keys']} "
          f"种子违规={v['n_seed_violations']} 每任务可用 CLS={v['usable_cls_per_task']} "
          f"devices={v['devices']} → {'PASS' if v['passed'] else 'FAIL'}", flush=True)
    print(f"[独立臂] P95={p95_ind:.4f} med={med_ind:.4f}（δ 标定同口径，n={len(pool_ind)}；"
          f"对照 0.110）sanity={'OK' if sanity_ok else 'DRIFT'}", flush=True)
    print(f"[共享臂] |D| P95={p95_sh_raw:.4f} med={med_sh_raw:.4f} (n={len(pool_sh)}) "
          f"→ /√2 单次口径 P95={p95_sh_corr:.4f}", flush=True)
    print(f"[降幅] √2 校正={red_corr:.1%} ({band_of(red_corr)}) | "
          f"原始保守={red_raw:.1%} ({band_of(red_raw)}) | "
          f"佐证 dev 口径={res['pooled']['dev_witness']['reduction']:.1%}", flush=True)
    for tid in task_ids:
        a = per_task[tid]["anova"]["ss_share"]
        print(f"[ANOVA {tid}] SS 份额 cls={a['cls']:.3f} seed={a['seed']:.3f} "
              f"交互/残差={a['resid']:.3f}", flush=True)
    ap_ = res["pooled"]["anova_ss_share"]
    print(f"[ANOVA pooled] cls={ap_['cls']:.3f} seed={ap_['seed']:.3f} "
          f"交互/残差={ap_['resid']:.3f}", flush=True)
    print(f"[派生] δ_paired={delta_paired:.3f}（现行 0.110）| required-k：校正 "
          f"{derived['required_k_corr']:.1f} / 保守 {derived['required_k_raw']:.1f}"
          f"（对照 k=1:11.0, SWA:5.5）", flush=True)
    hw12 = derived["ceiling_hw12"]
    print(f"[天花板 HW12 反事实] δ_paired={hw12['paired_delta']['Handwriting']['ceiling']}"
          f" (n={hw12['paired_delta']['Handwriting']['n_survive']}) | "
          f"δ=0.110 参照={hw12['published_delta_ref']['Handwriting']['ceiling']} | "
          f"δ=0 参照={hw12['delta0_ref']['Handwriting']['ceiling']:.4f}", flush=True)
    p28 = derived["ceiling_pooled28_scaled"]
    print(f"[天花板 pooled28 反事实] δ_paired+UM同比例={p28['paired_delta']['pooled']['ceiling']:.4f} "
          f"(n={p28['paired_delta']['pooled']['n_survive']}；对照发布 0.9525)", flush=True)
    for i, c in enumerate(probe_ceils):
        hwp = c["Handwriting"]
        print(f"[探针内真配对天花板 seed{i}] {hwp['ceiling']} (n={hwp['n_survive']})",
              flush=True)
    print(f"[配对方向稳定性] {stab}", flush=True)
    for name, p in res["predictions"].items():
        print(f"[预测 {name}] {'✓' if p['hit'] else '✗'} {p}", flush=True)
    print(f"\n[主判据] 降幅 校正={red_corr:.1%}/保守={red_raw:.1%} → "
          f"band_final={band_final.upper()}"
          + ("" if sanity_ok else "（⚠ 独立臂漂移，暂缓解读）"), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.analyze):
        ap.error("至少指定 --run 或 --analyze")

    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    n_shared = labels_cfg["route_a"]["n_shared_cls"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run:
        per_ds = _c2.load_pool(labels_cfg)
        shared_cls = load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]
        run(args.outdir, per_ds, iters_map, shared_cls, train_cfg, labels_cfg,
            args.device, args.resume)
    if args.analyze:
        analyze(args.outdir, n_shared)


if __name__ == "__main__":
    main()
