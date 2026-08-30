"""闸 2 试点 B：特征×标的选拔赛（协议 results/gate2_pilot_b/PROTOCOL.md；零 GPU）。

设计阶段多重比较（明确声明）：特征 {E, P, G, PG, PGE} × 标的 {三态 Brier,
连续增益 MAE}，估计器/抽样与试点 A 逐字同款；赢家按写死判据选出，交宽面板
@290 作闸 2 唯一确认性检验。任何单臂 p 值不作证据。

用法：python scripts/23_gate2_pilot_b.py
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")
fb = importlib.import_module("scripts.17_feasibility_loo_b")
gp = importlib.import_module("scripts.22_gate2_pilot")
dm = importlib.import_module("scripts.14_destruction_matrix")

import yaml  # noqa: E402

from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

OUTDIR = ROOT / "results" / "gate2_pilot_b"
DATASET = "UniMiB-SHAR"
CANDIDATE_REGISTRY = ROOT / "configs" / "candidate_pool.yaml"
K_ANCHOR, N_REPS, N_PERM = 3, 200, 10_000
FEATS = ("E", "P", "G", "PG", "PGE")
TARGETS = ("T1_state", "T2_gain")
STATES = fl.LABELS


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def gain_predict(gains: dict, anchors: list, weights: dict) -> np.ndarray:
    """连续增益画像的锚加权平均。"""
    total = sum(weights[a] for a in anchors)
    return sum(weights[a] * np.asarray(gains[a], dtype=float)
               for a in anchors) / total


def gain_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(np.asarray(pred) - np.asarray(actual)).mean())


def smoothness_generic(features: dict, profiles: dict, *, dist_fn,
                       n_perm: int, seed: int) -> tuple[float, float]:
    """通用平滑性：特征距离 vs 画像距离（dist_fn）Spearman + 任务置换 p。"""
    from scipy.stats import spearmanr
    names = sorted(features)
    M = np.stack([np.asarray(features[t], dtype=float) for t in names])
    mu, sd = lo.standardize_fit(M)
    Z = (M - mu) / sd
    n = len(names)
    dF = np.array([float(np.linalg.norm(Z[i] - Z[j]))
                   for i in range(n) for j in range(i + 1, n)])
    dP = [dist_fn(profiles[names[i]], profiles[names[j]])
          for i in range(n) for j in range(i + 1, n)]
    rho = float(spearmanr(dF, dP).statistic)
    rng = np.random.default_rng([seed, 20260906])
    n_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        dPp = [dist_fn(profiles[names[perm[i]]], profiles[names[perm[j]]])
               for i in range(n) for j in range(i + 1, n)]
        if spearmanr(dF, dPp).statistic >= rho:
            n_ge += 1
    return rho, (1 + n_ge) / (n_perm + 1)


def select_winner(combos: dict):
    """方向正的组合中 D1 p 最小者；并列取平滑性 ρ 大者；无正方向 → None。"""
    pos = {k: v for k, v in combos.items() if v["direction"] > 0}
    if not pos:
        return None
    return min(pos, key=lambda k: (pos[k]["d1_p"], -pos[k]["smooth_rho"]))


def verdict_pilot_b(d1_p, smooth_p) -> str:
    """判读带（协议 §3）。赢家为 None → null。"""
    if d1_p is None:
        return "null"
    if d1_p < 0.05 or smooth_p < 0.01:
        return "strong"
    if d1_p < 0.30:
        return "weak"
    return "null"


# ---------------------------------------------------------------- 数据装配
def build_um_geometry(tasks_needed: list, ra: dict) -> dict:
    """G 组：candidate 注册表确定性重切 → 窗口标签同源重采样 → 类别几何。"""
    reg = load_registry(CANDIDATE_REGISTRY)
    entry = reg.get(DATASET)
    X, y = load_source_dataset(DATASET, reg)
    _, tasks = slice_source_tasks(entry, reg, y)
    tmap = {t.task_id: t for t in tasks}
    G = {}
    for tid in tasks_needed:
        t = tmap[tid]
        Xt, yt = materialize_task(t, X, y)
        with np.load(gp.FEAT_DIR / f"{tid}.npz") as z:
            W = z["windows_repr"]
        idx = fb.sample_indices(len(Xt), ra["sample_num"], ra["embed_seed"])
        assert len(idx) == W.shape[0], \
            f"{tid} 窗口计数漂移: 采样 {len(idx)} vs 存储 {W.shape[0]}"
        V = fb.l2_normalize_rows(W.mean(axis=1))
        G[tid] = fb.class_geometry(V, np.asarray(yt)[idx])
    return G


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ra = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                        .read_text(encoding="utf-8"))["route_a"]
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    bands, classes, _ = dm.collect_baselines(list(dm.PANEL_B_BASELINES),
                                             exclude_plus_c=True)
    trained, _ = dm.collect_trained([dm.PANEL_B_TRAINED], shared_keys, bands,
                                    classes, lambda ds: dm.PANEL_B_ITERS.get(ds, -1))
    keys = sorted(shared_keys)
    flags: dict = defaultdict(dict)
    gains: dict = defaultdict(dict)
    for (tid, key), rp in trained.items():
        if fl.dataset_of(tid) == DATASET:
            mean, band = bands[tid]
            flags[tid][key] = STATES.index(fl.tri_state(rp, mean, band))
            gains[tid][key] = rp - mean
    tasks = sorted(flags)
    prof_state = {t: np.array([flags[t][k] for k in keys]) for t in tasks}
    prof_gain = {t: np.array([gains[t][k] for k in keys]) for t in tasks}

    feats: dict = {f: {} for f in FEATS}
    G = build_um_geometry(tasks, ra)
    for t in tasks:
        with np.load(gp.FEAT_DIR / f"{t}.npz") as z:
            E = z["E"].astype(float)
        mean, band = bands[t]
        P = np.array([mean, 1.0 / len(classes[t]), band])
        feats["E"][t] = E
        feats["P"][t] = P
        feats["G"][t] = G[t]
        feats["PG"][t] = np.concatenate([P, G[t]])
        feats["PGE"][t] = np.concatenate([P, G[t], E])
    print(f"[{DATASET}] {len(tasks)} 任务特征齐：E(256)/P(3)/G(5)", flush=True)

    combos: dict = {}
    per_task_out: dict = defaultdict(dict)
    for fname in FEATS:
        F = feats[fname]
        for tname in TARGETS:
            errs_k, errs_u = [], []
            for held in tasks:
                rest = [t for t in tasks if t != held]
                M = np.stack([F[t] for t in rest])
                mu, sd = lo.standardize_fit(M)
                z = {t: (F[t] - mu) / sd for t in tasks}
                dists = {t: float(np.linalg.norm(z[t] - z[held])) for t in rest}
                sigma = float(np.median([np.linalg.norm(z[a] - z[b])
                                         for i, a in enumerate(rest)
                                         for b in rest[i + 1:]]))
                ek, eu = [], []
                for rep in range(N_REPS):
                    anchors = gp.sample_anchors(
                        tasks, held=held, k=K_ANCHOR,
                        seed=run_seed_for(f"{held}:k{K_ANCHOR}", f"rep{rep}"))
                    wk = gp.kernel_w({a: dists[a] for a in anchors}, sigma)
                    wu = {a: 1.0 for a in anchors}
                    if tname == "T1_state":
                        ek.append(gp.profile_brier(
                            gp.anchor_predict(prof_state, anchors, wk),
                            prof_state[held]))
                        eu.append(gp.profile_brier(
                            gp.anchor_predict(prof_state, anchors, wu),
                            prof_state[held]))
                    else:
                        ek.append(gain_mae(gain_predict(prof_gain, anchors, wk),
                                           prof_gain[held]))
                        eu.append(gain_mae(gain_predict(prof_gain, anchors, wu),
                                           prof_gain[held]))
                errs_k.append(float(np.mean(ek)))
                errs_u.append(float(np.mean(eu)))
                per_task_out[held][f"{fname}|{tname}"] = (errs_k[-1], errs_u[-1])
            d1 = lo.paired_wilcoxon_less(np.array(errs_k), np.array(errs_u))
            direction = 1 if np.median(errs_k) < np.median(errs_u) else -1
            prof = prof_state if tname == "T1_state" else prof_gain
            dist_fn = ((lambda a, b: float((np.asarray(a) != np.asarray(b)).mean()))
                       if tname == "T1_state"
                       else (lambda a, b: gain_mae(a, b)))
            rho, sp = smoothness_generic(F, prof, dist_fn=dist_fn,
                                         n_perm=N_PERM, seed=20260907)
            combos[f"{fname}|{tname}"] = {
                "d1_p": d1, "direction": direction,
                "median_kernel": float(np.median(errs_k)),
                "median_uniform": float(np.median(errs_u)),
                "smooth_rho": rho, "smooth_p": sp,
            }
            print(f"{fname:>4}|{tname}: D1 p={d1:.4f} 方向={'+' if direction>0 else '-'} "
                  f"中位 {np.median(errs_k):.4f}/{np.median(errs_u):.4f} | "
                  f"平滑 ρ={rho:+.3f} p={sp:.4f}", flush=True)

    winner = select_winner(combos)
    if winner is None:
        verdict = verdict_pilot_b(None, None)
    else:
        verdict = verdict_pilot_b(combos[winner]["d1_p"],
                                  combos[winner]["smooth_p"])
    out = {"protocol": "results/gate2_pilot_b/PROTOCOL.md",
           "combos": combos, "winner": winner, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\n[赢家] {winner} | [判读] → {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
