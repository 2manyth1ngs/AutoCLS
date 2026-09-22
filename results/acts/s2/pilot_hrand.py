"""S2 修订 1 零 GPU 复算（读数见 results/acts/s2/pilot_hrand.md）：S1″ 试点 24 任务（有 random-20）上用 83 的新函数算 H_cat / H_rand。
口径与 cmd_analyze_s2 一致：frames = _frames_c、M = _fill_pairs(两种子终点) + q25/q50（_ckpt）、groups = 数据集、LODO 先验只在其他数据集上。
用法（仓库根目录）：python results/acts/s2/pilot_hrand.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
ac = importlib.import_module("scripts.acts_common")
g = importlib.import_module("scripts.83_acts_corpus")
_g35 = g._g35

cfg = ac.load_acts_config(profile="s1pp")
outdir = Path("results/acts/s1pp")
plan = json.loads((outdir / "plan_C.json").read_text(encoding="utf-8"))
cat = json.loads((outdir / "catalog.json").read_text(encoding="utf-8"))
rows = _g35.load_done(outdir / "runs.jsonl")
meta = plan["tasks_meta"]
frames = g._frames_c(plan, rows, cat, cfg)
raw = g._raw_seed_frames(plan, rows, cat, cfg)
n_sh = len(cat["shared"])
keys = g.cat_keys(cat)


def run(tasks, label):
    M = {"end_0": np.array([raw[t][0] for t in tasks]), "end_1": np.array([raw[t][1] for t in tasks])}
    M = g._fill_pairs(M)
    q25 = np.array([[g._ckpt(rows, plan, t, k, 0, 0) for k in keys] for t in tasks])
    q50 = np.array([[g._ckpt(rows, plan, t, k, 0, 1) for k in keys] for t in tasks])
    M.update({"q25_0": g._fill_single(q25, M["end_0"]), "q50_0": g._fill_single(q50, M["end_0"])})
    groups = np.array([meta[t]["dataset"] for t in tasks])
    R = g._random_end0_matrix(frames, tasks, n_sh)
    H = g._headroom_four(M, groups, rand_end0=R)
    print(f"== {label}: {len(tasks)} 任务, {len(set(groups))} 源, random 有效条数中位 {np.median(np.isfinite(R).sum(1)):.0f}")
    print(f"{'底':10s} {'H_cat':>8s} {'H_rand_raw':>11s} {'H_rand_adj':>11s}")
    for b in ("prior1", "top3", "sh633", "sh1043"):
        print(f"{b:10s} {H[b]:8.4f} {H['rand_raw'][b]:11.4f} {H['rand_adj'][b]:11.4f}")
    print(f"{'top3_med':10s} {H['top3_median']:8.4f} {H['rand_raw']['top3_median']:11.4f} {H['rand_adj']['top3_median']:11.4f}")
    e = H["oracle_ext"]
    print(f"oracle 扩展: raw 份额 {e['raw_frac']:.2f} / 均值 {e['raw_mean']:.4f}; adj 份额 {e['adj_frac']:.2f} / 均值 {e['adj_mean']:.4f}; "
          f"κ 均值 {e['kappa_mean']:.2f}; σ 中位 {e['sigma_median']:.4f}")
    per = H["per_task"]
    d = np.array(per["rand_adj"]["top3"]) - np.array(per["top3"])
    print(f"逐任务 H_rand_adj − H_cat (regret@3): >0 的任务 {int((d > 0).sum())}/{len(d)}, 最大 {d.max():.4f}")
    return H


pilot = [t for t in plan["probe_tasks"] if len(frames[t]["random_keys"]) > 0]
H_p = run(pilot, "试点（有 random-20）")
# 试点任务的 LODO 先验只用其它两个试点源 → 与 S1″ REPORT §4.2 "试点 24 任务"列同口径；另给全 64 任务版（random 缺的任务 H_rand = H_cat）
H_all = run([t for t in plan["probe_tasks"]], "全部任务（无 random 的任务 H_rand = H_cat）")
