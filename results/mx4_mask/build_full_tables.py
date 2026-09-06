"""把 WP6 全部读数拼成一份完整报表 results/mx4_mask/FULL_TABLES.md（只读既有产物）。"""
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
SHORT = {"Handwriting": "HW", "UWaveGestureLibrary": "UWave", "PhonemeSpectra": "PS",
         "EthanolConcentration": "Ethanol", "Cricket": "Cricket", "HandMovementDirection": "HMD"}
T = list(g39.TARGETS)
paper = {d: g39.paper_ts2vec(d) for d in T}
off, on = g40.load_wp3_rows(), g40.load_mask_rows()
s2 = [r for r in off + on if r.get("stage") == "s2_finals" and r.get("error") is None]
cands = g39.load_candidates(OUT)
L = []

# ---------------------------------------------------------------- 1. 全臂表（seed 0 + 3 种子）
arm_rows = {}
per = {}
for d in T:
    ro, rn = g40._s1_seed0(off, d), g40._s1_seed0(on, d)
    to, so = g39.trajectory_from_rows(ro); tn, sn = g39.trajectory_from_rows(rn)
    traj = {**to, **tn}; secs = {**so, **sn}
    budgets = sorted({b for v in to.values() for b in v}); bf = budgets[-1]
    keys_off = sorted(k for k in to if all(b in to[k] for b in budgets))
    twin_map = {r["twin_of"]: r["cls_key"] for r in rn if r.get("twin_of") in to and all(b in tn.get(r["cls_key"], {}) for b in budgets)}
    keys = keys_off + sorted(twin_map.values())
    base = g40.wp3_rankings(cands[d], keys_off)["prior"]
    rk = {"prior_off": base, "interleave": g40.interleave(base, twin_map), "allon": g40.all_on(base, twin_map)}
    row = g40.mask_arms(keys, traj, rk, twin_map, budgets=budgets, seconds=secs)
    truth = {k: traj[k][bf] for k in keys}
    on_keys = set(twin_map.values())

    def mean3(k):
        ex = [float(r["acc"]) for r in s2 if r["dataset"] == d and r["cls_key"] == k]
        return float(np.mean([truth[k]] + ex)) if len(ex) == 2 else None
    per[d] = {"row": row, "truth": truth, "on_keys": on_keys, "twin_map": twin_map, "mean3": mean3,
              "ro": ro, "rn": rn, "budgets": budgets, "cls_of": {r["cls_key"]: r["cls"] for r in ro}}
    for a, v in row["arms"].items():
        k = row["picks"].get(a)
        arm_rows.setdefault(a, {})[d] = (v, (mean3(k) if k else None), (k in on_keys) if k else False,
                                         row["cost_iters"].get(a))
L += ["# MX-4 WP6 完整报表：encoder 侧二项时间步掩码作 CLS 第 20 字段（官方口径，UEA 官方划分）", "",
      "口径：ts2vec_official（HW 200 步 / 其余 600 步，bs 8，SWA，train-fit scaler，官方 SVM 网格），截点 {25,50,100}%；"
      "目录 = WP3 的 shared-10 + GGS + 10 提名（21，PS 20）各配一条只把 input_mask 改为 binomial 的孪生。"
      "seed 0 = S1 目录；3 种子 = seed 0 + finals seeds {1,2}（仅各臂冠军有）。追平线 = 论文 TS2Vec − 0.03。“开”= 冠军是掩码开的孪生。", "",
      "## 1. 全臂表（每格：seed 0 / 3 种子；成本 = HW 的 iters）", "",
      "| 臂 | " + " | ".join(f"{SHORT[d]} ({paper[d]:.3f})" for d in T) + " | 追平 seed0 | 追平 3种子(可算) | 成本 |",
      "|---|" + "---:|" * len(T) + "---:|---:|---:|"]
for a in sorted(arm_rows):
    cells, n0, n3, av = [], 0, 0, 0
    for d in T:
        v, m, on_, c = arm_rows[a].get(d, (None, None, False, None))
        if v is None:
            cells.append("—"); continue
        cells.append(f"{v:.3f}" + (f" / {m:.3f}" if m is not None else "") + ("开" if on_ else ""))
        if v >= paper[d] - 0.03: n0 += 1
        if m is not None:
            av += 1
            if m >= paper[d] - 0.03: n3 += 1
    cost = arm_rows[a].get("Handwriting", (None, None, None, None))[3]
    L.append(f"| {a} | " + " | ".join(cells) + f" | {n0}/6 | {n3}/{av} | {cost if cost is not None else '—'} |")

# ---------------------------------------------------------------- 2. 事后扫臂
L += ["", "## 2. 事后扫臂（名单 × 掩码用法 × 级数；非预注册）", ""]
L += (OUT / "s1_sweep.md").read_text(encoding="utf-8").splitlines()

# ---------------------------------------------------------------- 3. 每对孪生 Δ（125 对）
L += ["", "## 3. 每对孪生的掩码收益（S1 seed 0；Δ = 开 − 关，100% 截点；括号内为 25% / 50% / 100% 三截点 acc）", ""]
for d in T:
    P = per[d]
    acc_off = {r["cls_key"]: r for r in P["ro"]}
    rows = []
    for r in P["rn"]:
        k = r.get("twin_of")
        if k not in acc_off:
            continue
        a, b = acc_off[k], r
        ca = [c["acc"] for c in sorted(a["checkpoint_results"], key=lambda c: c["iters"])]
        cb = [c["acc"] for c in sorted(b["checkpoint_results"], key=lambda c: c["iters"])]
        c = P["cls_of"][k]
        rows.append((float(b["acc"]) - float(a["acc"]), k, a["arm"], ca, cb, c))
    rows.sort(key=lambda x: -x[0])
    L += [f"### {d}（论文 {paper[d]:.3f}；{len(rows)} 对，正向 {sum(1 for x in rows if x[0] > 0)}，Δ 均值 {np.mean([x[0] for x in rows]):+.4f}）", "",
          "| 配置 | 来源 | 关 25/50/100 | 开 25/50/100 | Δ@100 | point_mask | emb_mask | crop | freq_mask | jitter |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for dl, k, arm, ca, cb, c in rows:
        L.append(f"| {k[:8]} | {arm} | {'/'.join(f'{x:.3f}' for x in ca)} | {'/'.join(f'{x:.3f}' for x in cb)} | {dl:+.3f} | "
                 f"{c['point_mask']} | {c['emb_mask']} | {c['crop']} | {c['freq_mask']} | {c['jitter']} |")
    L.append("")

# ---------------------------------------------------------------- 4. S0
L += ["## 4. S0：protocol_alignment 既有配对（官方口径，3 种子）", ""]
L += (OUT / "s0_alignment.md").read_text(encoding="utf-8").splitlines()

# ---------------------------------------------------------------- 5. 探针
L += ["", "## 5. 描述性探针（机制性数据集描述量；非预注册）", ""]
L += (OUT / "mech_probe.md").read_text(encoding="utf-8").splitlines()

# ---------------------------------------------------------------- 6. 门
a = json.loads((OUT / "analysis_deploy.json").read_text(encoding="utf-8"))
g = a.get("gates") or {}
L += ["", "## 6. 门与读数码（PROTOCOL §4）", "",
      "| 门 | 内容 | 状态 |", "|---|---|---|",
      "| C0 | WP1 / WP2 / WP3 三锚逐位 | 过 |",
      "| K1 | 切片 Δ 与官方 Δ 符号一致 ≥ 0.7（n ≥ 15） | 未测（S1b 未跑） |",
      "| K2 | gbdt20 孪生方向 acc ≥ 0.70 且 regret@1 < prior20 | 未测（S2 未跑）；探针 config_only LODO 0.70 |",
      f"| K3 | 主臂 3 种子追平 ≥ 5/6 | 临时主臂 {g.get('K3', {}).get('n_parity')}/6；事后“名单 10 两版三级”6/6 |",
      f"| K4 | 学习臂追平测量臂 | 不可算（临时主臂 {g.get('K4', {}).get('n_within')}/6） |",
      f"| 读数码 | 预注册 | **{a.get('verdict')}**（占位，临时主臂 interleave） |"]
(OUT / "FULL_TABLES.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:12]))
print(f"... 共 {len(L)} 行 → {OUT / 'FULL_TABLES.md'}")
