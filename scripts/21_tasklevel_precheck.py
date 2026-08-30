"""任务级脆弱度存在性预检（协议 results/tasklevel_precheck/PROTOCOL.md；零 GPU）。

工作点缓存（HW@60 12 任务 + UM@30 16 任务 × shared 10）→ 三态矩阵 →
行内置换检验：任务间脆弱度差异是否超出可交换噪声（CLS 主效应保留在零假设内）。
阳性 → 宽任务面板 GPU 拍板先验升；阴性 → 先验降（口径折扣见协议 §3）。

用法：python scripts/21_tasklevel_precheck.py
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
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.labeling import load_probe_cls  # noqa: E402

OUTDIR = ROOT / "results" / "tasklevel_precheck"
N_PERM = 10_000
SEED = 20260902
ALPHA_BONF = 0.025          # 协议 §3：Bonferroni ×2
STATES = fl.LABELS          # ("improve", "inert", "destroy")


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def t2_statistic(mat: np.ndarray) -> tuple[float, dict]:
    """T2 = Σ_状态 Σ_任务 (p_{t,s} − p_s)²；mat: (n_cls, n_tasks) 状态码 0/1/2。"""
    comp = {}
    for si, s in enumerate(STATES):
        p_ts = (mat == si).mean(axis=0)
        p_s = float((mat == si).mean())
        comp[s] = float(((p_ts - p_s) ** 2).sum())
    return float(sum(comp.values())), comp


def permute_within_cls(mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """零假设：每条 CLS 行独立置换任务归属（保 per-CLS 状态计数）。"""
    return rng.permuted(mat, axis=1)


def perm_test(mat: np.ndarray, *, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, 20260902])
    t2_obs, comp = t2_statistic(mat)
    n_ge = sum(1 for _ in range(n_perm)
               if t2_statistic(permute_within_cls(mat, rng))[0] >= t2_obs)
    return {"t2_obs": t2_obs, "components": comp,
            "p": (1 + n_ge) / (n_perm + 1), "n_perm": n_perm}


def verdict_tasklevel(pvals: dict) -> str:
    """判读带（协议 §3）：任一数据集 p < 0.025 → 阳性。"""
    return "positive" if any(p < ALPHA_BONF for p in pvals.values()) \
        else "negative"


# ---------------------------------------------------------------- 主流程
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    bands, classes, rej_b = dm.collect_baselines(
        list(dm.PANEL_B_BASELINES), exclude_plus_c=True)
    trained, rej_t = dm.collect_trained(
        [dm.PANEL_B_TRAINED], shared_keys, bands, classes,
        lambda ds: dm.PANEL_B_ITERS.get(ds, -1))

    cells: dict = defaultdict(dict)   # ds -> {(tid, key): state_idx}
    for (tid, key), rp in trained.items():
        ds = fl.dataset_of(tid)
        mean, band = bands[tid]
        cells[ds][(tid, key)] = STATES.index(fl.tri_state(rp, mean, band))

    results: dict = {}
    pvals: dict = {}
    for ds, cmap in sorted(cells.items()):
        tids = sorted({t for t, _ in cmap})
        keys = sorted({k for _, k in cmap})
        mat = np.array([[cmap[(t, k)] for t in tids] for k in keys])
        res = perm_test(mat, n_perm=N_PERM, seed=SEED)
        per_task = {t: {s: int(sum(1 for k in keys
                                   if cmap[(t, k)] == si))
                        for si, s in enumerate(STATES)} for t in tids}
        results[ds] = {**res, "n_tasks": len(tids), "n_cls": len(keys),
                       "iters": dm.PANEL_B_ITERS.get(ds),
                       "per_task_counts": per_task,
                       "global_rates": {s: float((mat == si).mean())
                                        for si, s in enumerate(STATES)}}
        pvals[ds] = res["p"]

    verdict = verdict_tasklevel(pvals)
    out = {"protocol": "results/tasklevel_precheck/PROTOCOL.md",
           "rejects": {"baseline": rej_b, "trained": rej_t},
           "datasets": results, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print("===== 任务级脆弱度存在性预检（工作点口径）=====")
    for ds, r in results.items():
        g = r["global_rates"]
        print(f"\n[{ds}] @iters={r['iters']} | {r['n_tasks']} 任务 × "
              f"{r['n_cls']} CLS | 全局 提/惰/破 = "
              f"{g['improve']:.2f}/{g['inert']:.2f}/{g['destroy']:.2f}")
        print(f"  T2={r['t2_obs']:.4f} p={r['p']:.4f} | 分量 "
              + " ".join(f"{s}={v:.4f}" for s, v in r["components"].items()))
        print(f"  {'任务':<22}{'提':>4}{'惰':>4}{'破':>4}")
        for t, c in r["per_task_counts"].items():
            print(f"  {t:<22}{c['improve']:>4}{c['inert']:>4}{c['destroy']:>4}")
    print(f"\n判读（任一 p<{ALPHA_BONF}）→ {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
