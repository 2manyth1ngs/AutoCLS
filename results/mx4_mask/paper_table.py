"""论文结果表：UEA 官方划分 test acc，3 种子 mean ± std（seed 0 = 目录格，seeds 1/2 = finals）。
列：文献基线（TS2Vec Table 10）| GGS（protocol_alignment 3 种子）| 本文各臂。"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
g40 = importlib.import_module("scripts.40_mask_field_sim"); g36 = g40._g36; g39 = g40._g39
OUT = ROOT / "results" / "mx4_mask"
T = list(g39.TARGETS)
SHORT = {"Handwriting": "Handwriting", "UWaveGestureLibrary": "UWaveGestureLibrary", "PhonemeSpectra": "PhonemeSpectra",
         "EthanolConcentration": "EthanolConcentration", "Cricket": "Cricket", "HandMovementDirection": "HandMovementDirection"}
off, on = g40.load_wp3_rows(), g40.load_mask_rows()
s2 = [r for r in off + on if r.get("stage") == "s2_finals" and r.get("error") is None]
cands = g39.load_candidates(OUT)
# GGS 3 种子（protocol_alignment ts2vec_official 主口径，REPORT §4）
GGS3 = {"Handwriting": (0.532, 0.015), "UWaveGestureLibrary": (0.869, 0.003), "PhonemeSpectra": (0.203, 0.002),
        "EthanolConcentration": (0.241, 0.007), "Cricket": (0.986, 0.011), "HandMovementDirection": (0.270, 0.019)}
TS2VEC_LOCAL3 = {"Handwriting": (0.529, 0.006), "UWaveGestureLibrary": (0.903, 0.009), "PhonemeSpectra": (0.235, 0.001),
                 "EthanolConcentration": (0.288, 0.016), "Cricket": (0.977, 0.007), "HandMovementDirection": (0.266, 0.039)}

ARMS = [("SH6-3-3 掩码关（名单 6）", "prior_off", (6, 3, 3), False, False),
        ("SH10-4-3 掩码关（名单 10）", "prior_off", (10, 4, 3), False, False),
        ("SH20-4-3 两版都量（名单 10 + 掩码孪生）", "both10", (20, 4, 3), True, False),
        ("SH20-3 两版都量，从 50% 截点起", "both10", (20, 3), True, True)]

def seeds_of(d, k, truth):
    ex = [float(r["acc"]) for r in s2 if r["dataset"] == d and r["cls_key"] == k]
    return [truth[k]] + ex

rows = {}
for d in T:
    ro, rn = g40._s1_seed0(off, d), g40._s1_seed0(on, d)
    to, _ = g39.trajectory_from_rows(ro); tn, _ = g39.trajectory_from_rows(rn)
    traj = {**to, **tn}; budgets = sorted({b for v in to.values() for b in v}); bf = budgets[-1]
    keys_off = sorted(k for k in to if all(b in to[k] for b in budgets))
    twin_map = {r["twin_of"]: r["cls_key"] for r in rn if r.get("twin_of") in to and all(b in tn.get(r["cls_key"], {}) for b in budgets)}
    keys = keys_off + sorted(twin_map.values())
    truth = {k: traj[k][bf] for k in keys}
    base = g40.wp3_rankings(cands[d], keys_off)["prior"]
    rows[d] = {}
    for label, rk, ks, use_twins, two_level in ARMS:
        ranked = g40.measured_ranking(base, twin_map, 10) if use_twins else base
        ranked = [k for k in ranked if k in truth]
        bud = budgets[1:] if two_level else budgets
        p = g36.sh_path(ranked, traj, ks, bud); pick = g36.sh_pick(p[-1], traj, bf)
        s = seeds_of(d, pick, truth)
        rows[d][label] = (float(np.mean(s)), float(np.std(s)), len(s), pick, pick in set(twin_map.values()))

ref = {d: g39.paper_reference(d) for d in T}
lit_cols = [c for c in ["DTW", "TNC", "TS-TCC", "T-Loss", "TS2Vec"] if all(c in ref[d] for d in T)]
hdr = ["数据集"] + lit_cols + ["TS2Vec 官方代码本机（3 种子）", "GGS（3 种子）"] + [a[0] for a in ARMS]
L = ["| " + " | ".join(hdr) + " |", "|---|" + "---:|" * (len(hdr) - 1)]
avg = {c: [] for c in hdr[1:]}
for d in T:
    cells = [SHORT[d]]
    for c in lit_cols:
        v = float(ref[d][c]); cells.append(f"{v:.3f}"); avg[c].append(v)
    m, s = TS2VEC_LOCAL3[d]; cells.append(f"{m:.3f} ± {s:.3f}"); avg["TS2Vec 官方代码本机（3 种子）"].append(m)
    m, s = GGS3[d]; cells.append(f"{m:.3f} ± {s:.3f}"); avg["GGS（3 种子）"].append(m)
    for label, *_ in ARMS:
        m, s, n, pick, on_ = rows[d][label]
        cells.append(f"{m:.3f} ± {s:.3f}" + ("" if n == 3 else f"（{n} 种子）") + ("†" if on_ else ""))
        avg[label].append(m)
    L.append("| " + " | ".join(cells) + " |")
L.append("| **平均** | " + " | ".join(f"{np.mean(avg[c]):.3f}" for c in hdr[1:]) + " |")
paper_ts2vec = [float(ref[d]["TS2Vec"]) for d in T]
L.append("| 追平 TS2Vec（≥ 论文 − 0.03） | " + " | ".join(
    ("—" if c in lit_cols else f"{sum(1 for i, d in enumerate(T) if avg[c][i] >= paper_ts2vec[i] - 0.03)}/6") for c in hdr[1:]) + " |")
L.append("| 超过 TS2Vec 论文数 | " + " | ".join(
    ("—" if c in lit_cols else f"{sum(1 for i, d in enumerate(T) if avg[c][i] > paper_ts2vec[i])}/6") for c in hdr[1:]) + " |")
L += ["", "† 冠军为掩码开的孪生。口径：UEA 官方 train/test 划分，ts2vec_official 训练协议（HW 200 步 / 其余 600 步，bs 8，SWA），",
      "官方 SVM 网格；3 种子 = seed 0（目录格）+ seeds 1/2（finals），mean ± std（总体标准差）。文献列取自 TS2Vec Table 10。"]
# 选中的配置
L += ["", "| 数据集 | " + " | ".join(a[0] for a in ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
for d in T:
    L.append(f"| {SHORT[d]} | " + " | ".join(f"{rows[d][a[0]][3][:8]}{'†' if rows[d][a[0]][4] else ''}" for a in ARMS) + " |")
(OUT / "paper_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
