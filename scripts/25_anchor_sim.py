"""锚子采样模拟：金丝雀直测 + 适用性闸门的部署功效标定（协议 results/anchor_sim/PROTOCOL.md；零 GPU）。

宽面板 400 格（5 数据集 × 8 任务 × shared 10 CLS @290）自助抽 a×j 子格，
标定三旋钮读数（破坏率/基线带/锚间异质性）对全面板真值的功效，按预注册
选择规则给出部署工作点 (a*, j*)。只标定功效，不宣布新发现（协议 §0/§1）。

用法：python scripts/25_anchor_sim.py [--r-draws 400] [--n-perm 4000] [--outdir ...]
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
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
tp = importlib.import_module("scripts.21_tasklevel_precheck")
dm = importlib.import_module("scripts.14_destruction_matrix")
wp = importlib.import_module("scripts.24_wide_panel")

from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

OUTDIR_DEFAULT = ROOT / "results" / "anchor_sim"
TRUTH_ANALYSIS = ROOT / "results" / "wide_panel" / "analysis.json"
# 协议 §3/§4 写死的扫描与判读常数
ANCHOR_COUNTS = (2, 3, 4)
J_VALUES = (5, 6, 7, 8, 9, 10)
R_DRAWS = 400
N_PERM_SUB = 4000
ALPHA_DEPLOY = 0.01                 # 蓝图逐字：p<0.01 才启用任务级差异
ALPHA_SENS = 0.05                   # 仅敏感性读数，不参与判定
TPR_MIN, FPR_MAX, MAE_MAX = 0.80, 0.15, 0.10   # 协议 §4(c) 可行带
DESTROY_IDX = fl.LABELS.index("destroy")
RECON_TOL = 1e-9
SEED_SALT = 20260831


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def t2_statistic_batch(mats: np.ndarray) -> np.ndarray:
    """tp.t2_statistic 的批量版（同式：T2 = Σ_s Σ_t (p_{t,s} − p_s)²）。"""
    mats = np.asarray(mats)
    t2 = np.zeros(mats.shape[0])
    for si in range(len(tp.STATES)):
        ind = mats == si
        p_ts = ind.mean(axis=1)                    # (B, n_tasks)
        p_s = ind.mean(axis=(1, 2))                # (B,)
        t2 += ((p_ts - p_s[:, None]) ** 2).sum(axis=1)
    return t2


def perm_test_vec(mat: np.ndarray, *, n_perm: int, rng) -> dict:
    """行内置换 T2 检验的向量化版（scripts/21 同统计量、同零假设：每 CLS 行
    独立置换任务归属）。等价性由 tests/test_anchor_sim.py 对 tp.perm_test 钉死。"""
    mat = np.asarray(mat)
    t2_obs = float(t2_statistic_batch(mat[None])[0])
    keys = rng.random((n_perm,) + mat.shape)
    idx = np.argsort(keys, axis=2)
    perms = np.take_along_axis(np.broadcast_to(mat, keys.shape), idx, axis=2)
    t2_null = t2_statistic_batch(perms)
    n_ge = int((t2_null >= t2_obs).sum())
    return {"t2_obs": t2_obs, "p": float((1 + n_ge) / (n_perm + 1))}


def min_attainable_p(j: int) -> float:
    """a=2 的解析地板（协议 PA2）：恒等排列与全局列交换必复现 T2_obs → p≥2/2^j。"""
    return 2.0 / 2 ** j


def draw_subpanel(rng, *, n_tasks: int, n_cls: int, a: int, j: int):
    """锚任务 a 个 + 共用 CLS j 条，均无放回均匀抽（协议 §3；毒哨勿复活）。"""
    t_idx = np.sort(rng.choice(n_tasks, size=a, replace=False))
    c_idx = np.sort(rng.choice(n_cls, size=j, replace=False))
    return t_idx, c_idx


def destroy_share(sub: np.ndarray) -> float:
    """旋钮 1（破坏率）：子格中 destroy 状态占比。"""
    return float((np.asarray(sub) == DESTROY_IDX).mean())


def gate_metrics(fire: dict, truth_pass: set) -> dict:
    """闸门操作特性（协议 §4b）：TPR = 真值过闸数据集触发率均值，FPR = 其余。"""
    tpr = [v for ds, v in fire.items() if ds in truth_pass]
    fpr = [v for ds, v in fire.items() if ds not in truth_pass]
    return {"tpr": float(np.mean(tpr)), "fpr": float(np.mean(fpr))}


def select_workpoint(points: dict):
    """协议 §4(c) 写死：可行集内 a×j 最小、并列取 a 小；空 → None。"""
    feas = [(a, j) for (a, j), m in points.items()
            if m["tpr"] >= TPR_MIN and m["fpr"] <= FPR_MAX
            and m["destroy_mae_pooled"] <= MAE_MAX]
    return min(feas, key=lambda aj: (aj[0] * aj[1], aj[0])) if feas else None


def band_error_stats(band_vals: np.ndarray, a: int) -> dict:
    """旋钮 2（基线带）误差：|锚均值−全任务均值|/真值，全 C(n,a) 精确枚举。"""
    vals = np.asarray(band_vals, dtype=float)
    truth = float(vals.mean())
    errs = [abs(float(vals[list(s)].mean()) - truth) / truth
            for s in itertools.combinations(range(len(vals)), a)]
    return {"n_subsets": len(errs), "median": float(np.median(errs)),
            "p90": float(np.percentile(errs, 90))}


def reconcile_truth(recomputed: dict, reference: dict,
                    tol: float = RECON_TOL) -> None:
    """真值对账硬闸（协议 §2）：闸 1 p/t2 逐数据集一致，缺失或超容差即抛。"""
    missing = set(reference) - set(recomputed)
    if missing:
        raise ValueError(f"对账缺数据集: {sorted(missing)}")
    for ds, ref in reference.items():
        got = recomputed[ds]
        for f in ("p", "t2_obs"):
            if abs(got[f] - ref[f]) > tol:
                raise ValueError(f"[{ds}] {f} 不合: 重算 {got[f]} vs 参考 {ref[f]}"
                                 "（先修装配再谈模拟）")


# ---------------------------------------------------------------- 数据装配与真值
def assemble_panel() -> tuple[dict, list, dict]:
    """宽面板装配（scripts/24 --analyze 逐字同式）→ {ds: mat/tids/bands}。"""
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    keys = sorted(shared_keys)
    wp_out = wp.OUTDIR_DEFAULT
    bands, classes, rej_b = dm.collect_baselines(
        list(wp.EXISTING_BASELINES) + [wp_out / "baseline_cache.jsonl"],
        exclude_plus_c=True)
    trained, rej_t = dm.collect_trained(
        list(wp.EXISTING_TRAINED) + [wp_out / "rprime_cache.jsonl"],
        shared_keys, bands, classes, lambda ds: wp.ITERS)

    flags: dict = defaultdict(dict)
    for (tid, key), rp in trained.items():
        ds = fl.dataset_of(tid)
        if ds in wp.PANEL and int(tid.rsplit("_", 1)[1]) < wp.N_TASKS_PANEL:
            mean, band = bands[tid]
            flags[(ds, tid)][key] = fl.LABELS.index(fl.tri_state(rp, mean, band))

    panel: dict = {}
    for ds in wp.PANEL:
        tids = sorted(t for (d, t) in flags
                      if d == ds and len(flags[(d, t)]) == len(keys))
        if len(tids) != wp.N_TASKS_PANEL:
            raise ValueError(f"[{ds}] 满格任务 {len(tids)} != "
                             f"{wp.N_TASKS_PANEL}（协议 §2 硬闸）")
        mat = np.array([[flags[(ds, t)][k] for t in tids] for k in keys])
        panel[ds] = {"mat": mat, "tids": tids,
                     "bands": np.array([bands[t][1] for t in tids]),
                     "base_means": np.array([bands[t][0] for t in tids])}
    return panel, keys, {"baseline": rej_b, "trained": rej_t}


def compute_truth(panel: dict) -> dict:
    """全面板真值（协议 §2）：闸 1（scripts/21 机器原样）+ 三态率 + 带均值。"""
    truth = {}
    for ds, d in panel.items():
        g1 = tp.perm_test(d["mat"], n_perm=tp.N_PERM, seed=tp.SEED)
        truth[ds] = {
            "gate1_p": g1["p"], "t2_obs": g1["t2_obs"],
            "gate_pass": g1["p"] < ALPHA_DEPLOY,
            "rates": {s: float((d["mat"] == si).mean())
                      for si, s in enumerate(fl.LABELS)},
            "band_mean": float(d["bands"].mean()),
            "base_mean_mean": float(d["base_means"].mean()),
        }
    return truth


# ---------------------------------------------------------------- 扫描
def run_sweep(panel: dict, truth: dict, *, r_draws: int, n_perm: int) -> dict:
    """逐 (ds, a, j) 抽 r_draws 个 a×j 子格：破坏率误差 + 闸门触发率。"""
    curves: dict = {}
    t0 = time.time()
    for ds, d in panel.items():
        mat = d["mat"]
        n_cls, n_tasks = mat.shape
        p_true = truth[ds]["rates"]["destroy"]
        for a in ANCHOR_COUNTS:
            for j in J_VALUES:
                errs, f01, f05 = [], 0, 0
                for r in range(r_draws):
                    seed = run_seed_for(f"anchor_sim:{ds}:a{a}:j{j}",
                                        f"draw{r}")
                    rng = np.random.default_rng([seed, SEED_SALT])
                    t_idx, c_idx = draw_subpanel(rng, n_tasks=n_tasks,
                                                 n_cls=n_cls, a=a, j=j)
                    sub = mat[np.ix_(c_idx, t_idx)]
                    errs.append(destroy_share(sub) - p_true)
                    p_sub = perm_test_vec(sub, n_perm=n_perm, rng=rng)["p"]
                    f01 += p_sub < ALPHA_DEPLOY
                    f05 += p_sub < ALPHA_SENS
                curves[(ds, a, j)] = {
                    "destroy_mae": float(np.mean(np.abs(errs))),
                    "destroy_bias": float(np.mean(errs)),
                    "fire_alpha01": f01 / r_draws,
                    "fire_alpha05": f05 / r_draws,
                }
            print(f"[{ds}] a={a} 完成（累计 {time.time() - t0:.0f}s）",
                  flush=True)
    return curves


def aggregate_points(curves: dict, truth: dict, ds_names: list) -> dict:
    """逐 (a, j) 汇总：TPR/FPR（α=0.01）+ pooled 破坏率 MAE + 逐数据集触发率。"""
    pass_ds = {ds for ds in ds_names if truth[ds]["gate_pass"]}
    points = {}
    for a in ANCHOR_COUNTS:
        for j in J_VALUES:
            fire01 = {ds: curves[(ds, a, j)]["fire_alpha01"] for ds in ds_names}
            points[(a, j)] = {
                **gate_metrics(fire01, pass_ds),
                "destroy_mae_pooled": float(np.mean(
                    [curves[(ds, a, j)]["destroy_mae"] for ds in ds_names])),
                "fire_alpha01": fire01,
                "fire_alpha05": {ds: curves[(ds, a, j)]["fire_alpha05"]
                                 for ds in ds_names},
            }
    return points


def check_predictions(rec, points: dict, curves: dict, truth: dict,
                      ds_names: list) -> dict:
    """协议 §5 预测对账（PA2 判 ≈0 = 每点触发率 ≤0.01，MC 容差内）。"""
    pa2_fires = {f"{ds}:j{j}": curves[(ds, 2, j)]["fire_alpha01"]
                 for ds in ds_names for j in (5, 6, 7)}
    pa2_max = max(pa2_fires.values())
    fail_ds = [ds for ds in ds_names if not truth[ds]["gate_pass"]]
    mean_fire_fail = {ds: float(np.mean(
        [curves[(ds, a, j)]["fire_alpha01"]
         for a in ANCHOR_COUNTS for j in J_VALUES])) for ds in fail_ds}
    ct = "CharacterTrajectories"
    return {
        "PA1": {"claim": "可行集非空", "prior": 0.55,
                "hit": rec is not None},
        "PA2": {"claim": "a=2 在 j≤7 结构性零功率（p≥2/2^j>0.01）",
                "prior": 0.90, "max_fire": pa2_max, "hit": pa2_max <= 0.01},
        "PA3": {"claim": "推荐锚数 a*=3（条件于 PA1）", "prior": 0.60,
                "hit": None if rec is None else rec[0] == 3},
        "PA4": {"claim": "CT 触发率为真值不过闸三者中最高（一致性核对）",
                "prior": 0.75, "mean_fire_fail": mean_fire_fail,
                "hit": mean_fire_fail.get(ct) == max(mean_fire_fail.values())},
    }


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT),
                    help="冒烟务必改到别处")
    ap.add_argument("--r-draws", type=int, default=R_DRAWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM_SUB)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    panel, keys, rejects = assemble_panel()
    ds_names = list(panel)
    truth = compute_truth(panel)

    # ---- 真值对账硬闸（协议 §2）----
    ref = json.loads(TRUTH_ANALYSIS.read_text(encoding="utf-8"))
    reconcile_truth(
        {ds: {"p": truth[ds]["gate1_p"], "t2_obs": truth[ds]["t2_obs"]}
         for ds in ds_names},
        {ds: {"p": e["gate1"]["p"], "t2_obs": e["gate1"]["t2_obs"]}
         for ds, e in ref["per_dataset"].items()})
    print("真值对账 ✓：闸 1 p/t2 与 wide_panel/analysis.json 逐数据集一致",
          flush=True)

    curves = run_sweep(panel, truth, r_draws=args.r_draws, n_perm=args.n_perm)
    points = aggregate_points(curves, truth, ds_names)
    band_err = {ds: {a: band_error_stats(panel[ds]["bands"], a)
                     for a in ANCHOR_COUNTS} for ds in ds_names}
    rec = select_workpoint(points)
    preds = check_predictions(rec, points, curves, truth, ds_names)

    out = {
        "protocol": "results/anchor_sim/PROTOCOL.md",
        "constants": {"anchor_counts": list(ANCHOR_COUNTS),
                      "j_values": list(J_VALUES), "r_draws": args.r_draws,
                      "n_perm_sub": args.n_perm, "alpha_deploy": ALPHA_DEPLOY,
                      "alpha_sens": ALPHA_SENS,
                      "bands": {"tpr_min": TPR_MIN, "fpr_max": FPR_MAX,
                                "mae_max": MAE_MAX},
                      "seed_salt": SEED_SALT},
        "reconciliation": {"status": "ok", "reference": str(
            TRUTH_ANALYSIS.relative_to(ROOT)).replace("\\", "/")},
        "truth": truth,
        "rejects": rejects,
        "band_error": {ds: {f"a{a}": v for a, v in per.items()}
                       for ds, per in band_err.items()},
        "curves": {ds: {f"a{a}": {f"j{j}": curves[(ds, a, j)]
                                  for j in J_VALUES}
                        for a in ANCHOR_COUNTS} for ds in ds_names},
        "points": {f"a{a}_j{j}": points[(a, j)]
                   for a in ANCHOR_COUNTS for j in J_VALUES},
        "recommendation": {
            "workpoint": list(rec) if rec else None,
            "feasible": [[a, j] for (a, j), m in points.items()
                         if m["tpr"] >= TPR_MIN and m["fpr"] <= FPR_MAX
                         and m["destroy_mae_pooled"] <= MAE_MAX]},
        "predictions": preds,
    }
    (outdir / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 终端汇总 ----
    print("\n===== 锚子采样模拟（协议判读带执行）=====")
    print("真值：" + " | ".join(
        f"{ds} p={truth[ds]['gate1_p']:.4f}"
        f"{'✓' if truth[ds]['gate_pass'] else '✗'}"
        f" 破坏率={truth[ds]['rates']['destroy']:.3f}" for ds in ds_names))
    print(f"\n{'a':>2}{'j':>4}{'格数':>6}{'TPR':>8}{'FPR':>8}{'MAE':>8}  可行")
    for a in ANCHOR_COUNTS:
        for j in J_VALUES:
            m = points[(a, j)]
            ok = (m["tpr"] >= TPR_MIN and m["fpr"] <= FPR_MAX
                  and m["destroy_mae_pooled"] <= MAE_MAX)
            print(f"{a:>2}{j:>4}{a * j:>6}{m['tpr']:>8.3f}{m['fpr']:>8.3f}"
                  f"{m['destroy_mae_pooled']:>8.3f}  {'✓' if ok else ''}")
    print(f"\n推荐工作点：{rec if rec else '无（可行集空，如实报 Pareto）'}")
    for k, p in preds.items():
        print(f"{k}: {p['claim']} → {'✓' if p['hit'] else '✗' if p['hit'] is not None else 'n/a'}")
    print(f"→ {outdir / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
