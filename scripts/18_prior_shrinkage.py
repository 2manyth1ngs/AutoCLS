"""MX-1c 先验收缩：金丝雀预算下的毒性估计（协议 results/mx1c_prior_shrinkage/PROTOCOL.md）。

零 GPU 纯模拟。真值 = 320 格 label 的数据集破坏率；prior_f = MX-1b bm 的
LOO 预测经**秩校准**（分位数映射到训练折观测破坏率），只用序不信标定
——这是 ρ=0.555 能支撑的用法。deliverable = 测量效率，非预测主张复活
（勿复活条款语境见协议 §1）。

用法：python scripts/18_prior_shrinkage.py
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

from autocls.labeling import run_seed_for  # noqa: E402

OUTDIR = ROOT / "results" / "mx1c_prior_shrinkage"
MX1B_ANALYSIS = ROOT / "results" / "mx1b_feasibility" / "analysis.json"
BUDGETS = (2, 3, 5, 8)
J_PRIMARY, M_PRIMARY = 3, 4          # 协议 §3 主判读点写死
M_SENS = (2, 8)
N_DRAWS = 500
DESTROY = fl.LABELS.index("destroy") if hasattr(fl.LABELS, "index") else 2


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def quantile_map(q_d: float, q_train, o_train) -> float:
    """秩校准：(LOO 预测值 → 观测破坏率) 映射表按预测值排序线性插值，端点截断。"""
    q = np.asarray(q_train, dtype=float)
    o = np.asarray(o_train, dtype=float)
    order = np.argsort(q)
    return float(np.interp(q_d, q[order], o[order]))


def shrink(p_bar: float, prior: float, *, j: int, m: int) -> float:
    """Dirichlet 伪计数收缩（协议 §3）。"""
    return (j * p_bar + m * prior) / (j + m)


def simulate_fold(flags, *, prior_g: float, prior_f: float, j: int, m: int,
                  n_draws: int, seed: int) -> dict:
    """单折模拟：j 格无放回抽样 → 三个估计器的 |误差| 对 draws 取均值。"""
    flags = np.asarray(flags, dtype=float)
    p_true = flags.mean()
    rng = np.random.default_rng([seed, 20260831])
    errs: dict = {"raw": [], "shrink_g": [], "shrink_f": []}
    for _ in range(n_draws):
        pb = flags[rng.choice(len(flags), size=j, replace=False)].mean()
        errs["raw"].append(abs(pb - p_true))
        errs["shrink_g"].append(abs(shrink(pb, prior_g, j=j, m=m) - p_true))
        errs["shrink_f"].append(abs(shrink(pb, prior_f, j=j, m=m) - p_true))
    return {k: float(np.mean(v)) for k, v in errs.items()}


def verdict_mx1c(p_d1c: float, directions: list) -> str:
    """判读带（协议 §4）。directions: 各 j 档 shrink_f 是否更好（+1/-1）。"""
    if p_d1c < 0.05:
        return "established"
    if p_d1c < 0.15 and sum(1 for d in directions if d > 0) >= 3:
        return "edge"
    return "not_established"


# ---------------------------------------------------------------- 主流程
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = lo.load_panel()
    datasets = sorted({r["dataset"] for r in rows})
    flags_by_ds: dict = defaultdict(list)
    for r in sorted(rows, key=lambda x: (x["task_id"], x["cls_key"])):
        flags_by_ds[r["dataset"]].append(int(r["label"] == "destroy"))
    p_obs = {d: float(np.mean(flags_by_ds[d])) for d in datasets}

    mx1b = json.loads(MX1B_ANALYSIS.read_text(encoding="utf-8"))
    q_loo = {d: mx1b["per_fold"][d]["pred_rates"]["bm"][DESTROY]
             for d in datasets}          # 全部 out-of-sample（各自折产出）

    per_fold: dict = {}
    for heldout, train_ds in lo.fold_split(datasets):
        prior_g = float(np.mean([f for d in train_ds for f in flags_by_ds[d]]))
        prior_f = quantile_map(q_loo[heldout],
                               [q_loo[d] for d in train_ds],
                               [p_obs[d] for d in train_ds])
        entry = {"p_obs": p_obs[heldout], "prior_g": round(prior_g, 4),
                 "prior_f": round(prior_f, 4),
                 "prior_g_err": abs(prior_g - p_obs[heldout]),
                 "prior_f_err": abs(prior_f - p_obs[heldout]), "sims": {}}
        for j in BUDGETS:
            for m in (M_PRIMARY, *M_SENS):
                entry["sims"][f"j{j}_m{m}"] = simulate_fold(
                    flags_by_ds[heldout], prior_g=prior_g, prior_f=prior_f,
                    j=j, m=m, n_draws=N_DRAWS,
                    seed=run_seed_for(f"{heldout}:j{j}:m{m}", "mx1c_draws"))
        per_fold[heldout] = entry

    def earr(j, m, arm):
        return np.array([per_fold[d]["sims"][f"j{j}_m{m}"][arm]
                         for d in datasets])

    key = f"j{J_PRIMARY}_m{M_PRIMARY}"
    tests = {
        "D1c_p": lo.paired_wilcoxon_less(earr(J_PRIMARY, M_PRIMARY, "shrink_f"),
                                         earr(J_PRIMARY, M_PRIMARY, "shrink_g")),
        "D2c_p": lo.paired_wilcoxon_less(
            np.array([per_fold[d]["prior_f_err"] for d in datasets]),
            np.array([per_fold[d]["prior_g_err"] for d in datasets])),
        "D3c_p": lo.paired_wilcoxon_less(earr(J_PRIMARY, M_PRIMARY, "shrink_g"),
                                         earr(J_PRIMARY, M_PRIMARY, "raw")),
        "primary_point": key,
        "median": {arm: float(np.median(earr(J_PRIMARY, M_PRIMARY, arm)))
                   for arm in ("raw", "shrink_g", "shrink_f")},
        "prior_err_median": {
            "prior_g": float(np.median([per_fold[d]["prior_g_err"] for d in datasets])),
            "prior_f": float(np.median([per_fold[d]["prior_f_err"] for d in datasets]))},
    }
    directions, sens = [], {}
    for j in BUDGETS:
        med_f = float(np.median(earr(j, M_PRIMARY, "shrink_f")))
        med_g = float(np.median(earr(j, M_PRIMARY, "shrink_g")))
        directions.append(1 if med_f < med_g else -1)
        sens[f"j{j}_m{M_PRIMARY}"] = {
            "shrink_f_med": med_f, "shrink_g_med": med_g,
            "p": lo.paired_wilcoxon_less(earr(j, M_PRIMARY, "shrink_f"),
                                         earr(j, M_PRIMARY, "shrink_g"))}
    for m in M_SENS:
        sens[f"j{J_PRIMARY}_m{m}"] = {
            "p": lo.paired_wilcoxon_less(earr(J_PRIMARY, m, "shrink_f"),
                                         earr(J_PRIMARY, m, "shrink_g"))}
    tests["directions_by_j"] = directions
    tests["sensitivity"] = sens
    verdict = verdict_mx1c(tests["D1c_p"], directions)

    out = {"protocol": "results/mx1c_prior_shrinkage/PROTOCOL.md",
           "budgets": list(BUDGETS), "n_draws": N_DRAWS,
           "per_fold": per_fold, "tests": tests, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print(f"===== MX-1c 先验收缩（16 折，j={J_PRIMARY} m={M_PRIMARY} 主判）=====")
    print(f"{'数据集':<28}{'真值':>6}{'g先验':>7}{'f先验':>7}{'raw':>7}"
          f"{'收缩g':>7}{'收缩f':>7}")
    for d in datasets:
        e = per_fold[d]
        s = e["sims"][key]
        print(f"{d:<28}{e['p_obs']:>6.2f}{e['prior_g']:>7.2f}{e['prior_f']:>7.2f}"
              f"{s['raw']:>7.3f}{s['shrink_g']:>7.3f}{s['shrink_f']:>7.3f}")
    t = tests
    print(f"\nD1c(收缩f<收缩g @j3m4) p={t['D1c_p']:.4f} | "
          f"D2c(先验f<先验g 单独) p={t['D2c_p']:.4f} | "
          f"D3c(收缩g<raw) p={t['D3c_p']:.4f}")
    print(f"中位: raw {t['median']['raw']:.3f} | 收缩g {t['median']['shrink_g']:.3f}"
          f" | 收缩f {t['median']['shrink_f']:.3f} | 先验单独 g "
          f"{t['prior_err_median']['prior_g']:.3f} / f "
          f"{t['prior_err_median']['prior_f']:.3f}")
    print("各 j 档方向(f 更好=+1): " + str(directions)
          + " | 敏感性 p: " + "  ".join(f"{k}={v['p']:.3f}"
                                       for k, v in sens.items()))
    print(f"\n[判读] → {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
