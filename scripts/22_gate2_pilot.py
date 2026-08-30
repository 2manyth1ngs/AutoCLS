"""闸 2 试点：E 锚点外推零 GPU 定价（协议 results/gate2_pilot/PROTOCOL.md）。

UM@工作点 16 任务 × shared 10：留一任务 × 随机锚（k=3, R=200），
E 核插值 vs 均匀锚（任务盲对照），三态画像 Brier 记分；平滑性直测
（E 距离 vs 画像距离，置换 p）。定价工具，非论文证据（协议 §5 五重限定）。

用法：python scripts/22_gate2_pilot.py
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
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

OUTDIR = ROOT / "results" / "gate2_pilot"
FEAT_DIR = ROOT / "results" / "labels_workpoint" / "task_features"
DATASET = "UniMiB-SHAR"
K_PRIMARY, K_SENS = 3, 2
N_REPS = 200
N_PERM = 10_000
STATES = fl.LABELS


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def kernel_w(dists: dict, sigma: float) -> dict:
    return {k: float(np.exp(-d * d / (2 * sigma * sigma)))
            for k, d in dists.items()}


def anchor_predict(profiles: dict, anchors: list, weights: dict) -> np.ndarray:
    """锚画像加权平均 → (n_cls, 3) 概率；profiles[t] = 状态码向量 (n_cls,)。"""
    n_cls = len(profiles[anchors[0]])
    pred = np.zeros((n_cls, len(STATES)))
    total = 0.0
    for a in anchors:
        w = weights[a]
        onehot = np.eye(len(STATES))[profiles[a]]
        pred += w * onehot
        total += w
    return pred / total


def profile_brier(pred: np.ndarray, actual: np.ndarray) -> float:
    """(n_cls,3) 预测 vs 状态码真值 → 三态 Brier 均值。"""
    onehot = np.eye(len(STATES))[actual]
    return float(((pred - onehot) ** 2).sum(axis=1).mean())


def sample_anchors(tasks: list, *, held: str, k: int, seed: int) -> list:
    rng = np.random.default_rng([seed, 20260903])
    pool = sorted(t for t in tasks if t != held)
    return sorted(rng.choice(pool, size=k, replace=False).tolist())


def smoothness_test(E: dict, profiles: dict, *, n_perm: int,
                    seed: int) -> tuple[float, float]:
    """S1p：任务对 E 距离 vs 画像不一致率的 Spearman；任务标签置换 p。"""
    from scipy.stats import spearmanr
    names = sorted(E)
    M = np.stack([np.asarray(E[t], dtype=float) for t in names])
    mu, sd = lo.standardize_fit(M)
    Z = (M - mu) / sd
    n = len(names)
    dE, dP = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dE.append(float(np.linalg.norm(Z[i] - Z[j])))
            dP.append(float((np.asarray(profiles[names[i]])
                             != np.asarray(profiles[names[j]])).mean()))
    rho = float(spearmanr(dE, dP).statistic)
    rng = np.random.default_rng([seed, 20260904])
    n_ge = 0
    dE_arr = np.array(dE)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        dPp = [float((np.asarray(profiles[names[perm[i]]])
                      != np.asarray(profiles[names[perm[j]]])).mean())
               for i in range(n) for j in range(i + 1, n)]
        if spearmanr(dE_arr, dPp).statistic >= rho:
            n_ge += 1
    return rho, (1 + n_ge) / (n_perm + 1)


def verdict_pilot(p_d1: float, direction: int) -> str:
    """判读带（协议 §4）：direction=+1 表 E 核中位更好。"""
    if p_d1 < 0.05 and direction > 0:
        return "positive"
    if direction > 0:
        return "lean_positive"
    return "negative"


# ---------------------------------------------------------------- 主流程
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    bands, classes, _ = dm.collect_baselines(list(dm.PANEL_B_BASELINES),
                                             exclude_plus_c=True)
    trained, _ = dm.collect_trained([dm.PANEL_B_TRAINED], shared_keys, bands,
                                    classes, lambda ds: dm.PANEL_B_ITERS.get(ds, -1))
    keys = sorted(shared_keys)
    flags: dict = defaultdict(dict)
    for (tid, key), rp in trained.items():
        if fl.dataset_of(tid) == DATASET:
            mean, band = bands[tid]
            flags[tid][key] = STATES.index(fl.tri_state(rp, mean, band))
    tasks = sorted(flags)
    profiles = {t: np.array([flags[t][k] for k in keys]) for t in tasks}
    E = {}
    for t in tasks:
        with np.load(FEAT_DIR / f"{t}.npz") as z:
            E[t] = z["E"].astype(float)
    print(f"[{DATASET}] {len(tasks)} 任务 × {len(keys)} CLS，特征齐", flush=True)

    per_task: dict = {}
    for k_anchor in (K_PRIMARY, K_SENS):
        for held in tasks:
            rest = [t for t in tasks if t != held]
            M = np.stack([E[t] for t in rest])
            mu, sd = lo.standardize_fit(M)          # 按折标准化（只用非留出）
            z = {t: (E[t] - mu) / sd for t in tasks}
            dists_all = {t: float(np.linalg.norm(z[t] - z[held])) for t in rest}
            pair_d = [float(np.linalg.norm(z[a] - z[b]))
                      for i, a in enumerate(rest) for b in rest[i + 1:]]
            sigma = float(np.median(pair_d))
            errs_e, errs_u = [], []
            for rep in range(N_REPS):
                anchors = sample_anchors(
                    tasks, held=held, k=k_anchor,
                    seed=run_seed_for(f"{held}:k{k_anchor}", f"rep{rep}"))
                wk = kernel_w({a: dists_all[a] for a in anchors}, sigma)
                errs_e.append(profile_brier(
                    anchor_predict(profiles, anchors, wk), profiles[held]))
                errs_u.append(profile_brier(
                    anchor_predict(profiles, anchors, {a: 1.0 for a in anchors}),
                    profiles[held]))
            glob = np.mean([np.eye(len(STATES))[profiles[t]] for t in rest], axis=0)
            per_task.setdefault(held, {})[f"k{k_anchor}"] = {
                "err_kernel": float(np.mean(errs_e)),
                "err_uniform": float(np.mean(errs_u)),
                "err_global": profile_brier(glob, profiles[held]),
            }

    def arr(kk, arm):
        return np.array([per_task[t][kk][arm] for t in tasks])

    d1p = lo.paired_wilcoxon_less(arr(f"k{K_PRIMARY}", "err_kernel"),
                                  arr(f"k{K_PRIMARY}", "err_uniform"))
    med_e = float(np.median(arr(f"k{K_PRIMARY}", "err_kernel")))
    med_u = float(np.median(arr(f"k{K_PRIMARY}", "err_uniform")))
    direction = 1 if med_e < med_u else -1
    rho, p_s = smoothness_test(E, profiles, n_perm=N_PERM, seed=20260905)
    d1p_k2 = lo.paired_wilcoxon_less(arr(f"k{K_SENS}", "err_kernel"),
                                     arr(f"k{K_SENS}", "err_uniform"))
    # 分解诊断：均匀锚→E 核的逐状态误差变化（对 200 组锚的均值差按状态无法
    # 直接拆 Brier，用画像级近似：预测概率对真值 one-hot 的逐状态平方差）
    verdict = verdict_pilot(d1p, direction)

    tests = {"D1p_p": d1p, "D1p_p_k2": d1p_k2,
             "median": {"kernel": med_e, "uniform": med_u,
                        "global": float(np.median(arr(f"k{K_PRIMARY}", "err_global")))},
             "win_tasks": int((arr(f"k{K_PRIMARY}", "err_kernel")
                               < arr(f"k{K_PRIMARY}", "err_uniform")).sum()),
             "S1p_smoothness": {"rho": rho, "perm_p": p_s}}
    out = {"protocol": "results/gate2_pilot/PROTOCOL.md",
           "per_task": per_task, "tests": tests, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print(f"\n===== 闸 2 试点（{DATASET}，留一任务 × k={K_PRIMARY} 锚 × {N_REPS} 组）=====")
    print(f"{'任务':<20}{'E核':>8}{'均匀锚':>8}{'全局盲':>8}")
    for t in tasks:
        e = per_task[t][f"k{K_PRIMARY}"]
        print(f"{t:<20}{e['err_kernel']:>8.4f}{e['err_uniform']:>8.4f}"
              f"{e['err_global']:>8.4f}")
    print(f"\nD1p(E核<均匀锚) p={d1p:.4f}（{tests['win_tasks']}/16 任务胜；"
          f"中位 {med_e:.4f} vs {med_u:.4f}）| k=2 敏感性 p={d1p_k2:.4f}")
    print(f"S1p 平滑性: ρ={rho:.3f} 置换 p={p_s:.4f}")
    print(f"\n[判读] → {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
