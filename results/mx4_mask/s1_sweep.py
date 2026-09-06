"""S1 零 GPU 扫臂：42 目录上把名单长度 × 起点截点 × 掩码用法组合算尽（seed 0；已有 finals 的键顺带报 3 种子）。
只读 results/mx4_deploy/runs.jsonl + results/mx4_mask/runs.jsonl，不写产物。"""
import sys, importlib
sys.path.insert(0, r"D:\DevelopDoc\CLS\.claude\worktrees\wp6-mask")
import numpy as np
g40 = importlib.import_module("scripts.40_mask_field_sim"); g36 = g40._g36; g39 = g40._g39
off, on = g40.load_wp3_rows(), g40.load_mask_rows()
s2 = [r for r in off + on if r.get("stage") == "s2_finals" and r.get("error") is None]
cands = g39.load_candidates(g40.OUT)
paper = {d: g39.paper_ts2vec(d) for d in g39.TARGETS}
short = {"Handwriting": "HW", "UWaveGestureLibrary": "UWave", "PhonemeSpectra": "PS",
         "EthanolConcentration": "Ethanol", "Cricket": "Cricket", "HandMovementDirection": "HMD"}
per = {}
for d in g39.TARGETS:
    ro, rn = g40._s1_seed0(off, d), g40._s1_seed0(on, d)
    to, _ = g39.trajectory_from_rows(ro); tn, _ = g39.trajectory_from_rows(rn)
    traj = {**to, **tn}; budgets = sorted({b for v in to.values() for b in v}); bf = budgets[-1]
    keys_off = sorted(k for k in to if all(b in to[k] for b in budgets))
    twin_map = {r["twin_of"]: r["cls_key"] for r in rn
                if r.get("twin_of") in to and all(b in tn.get(r["cls_key"], {}) for b in budgets)}
    on_keys = set(twin_map.values())
    keys = keys_off + sorted(on_keys)
    truth = {k: traj[k][bf] for k in keys}
    base = g40.wp3_rankings(cands[d], keys_off)["prior"]
    arms = {}

    def sh(name, ranked, ks, bud):
        ranked = [k for k in ranked if k in truth]
        p = g36.sh_path(ranked, traj, ks, bud); pick = g36.sh_pick(p[-1], traj, bf)
        arms[name] = (truth[pick], pick, g36.path_cost_iters(p, bud))

    for top in (6, 10):
        off_r, on_r, mix_r = base[:top], g40.all_on(base[:top], twin_map), g40.interleave(base[:top], twin_map)
        n_mix = len(mix_r)
        # 三级（从 25% 起）
        sh(f"top{top}_off_3lvl", off_r, (top, 4 if top == 10 else 3, 3), budgets)
        sh(f"top{top}_on_3lvl", on_r, (len(on_r), 4 if top == 10 else 3, 3), budgets)
        sh(f"top{top}_both_3lvl", mix_r, (n_mix, 4 if top == 10 else 3, 3), budgets)
        # 两级（从 50% 起）
        sh(f"top{top}_off_2lvl", off_r, (top, 3), budgets[1:])
        sh(f"top{top}_on_2lvl", on_r, (len(on_r), 3), budgets[1:])
        sh(f"top{top}_both_2lvl", mix_r, (n_mix, 3), budgets[1:])
        sh(f"top{top}_both_2lvl_keep4", mix_r, (n_mix, 4), budgets[1:])
    arms["oracle_off"] = (max(truth[k] for k in keys_off), None, 0)
    arms["oracle_on"] = (max(truth[k] for k in on_keys), None, 0)

    def mean3(k):
        ex = [float(r["acc"]) for r in s2 if r["dataset"] == d and r["cls_key"] == k]
        return (float(np.mean([truth[k]] + ex)) if len(ex) == 2 else None)
    per[d] = {a: (v, k, c, (mean3(k) if k else None), (k in on_keys) if k else False) for a, (v, k, c) in arms.items()}

names = list(next(iter(per.values())).keys())
print("| 臂（名单_掩码_级数） | " + " | ".join(f"{short[d]} ({paper[d]:.3f})" for d in g39.TARGETS) + " | 追平 seed0 | 追平 3种子(可算) | HW 成本 iters |")
print("|---|" + "---:|" * len(g39.TARGETS) + "---:|---:|---:|")
for a in names:
    cells, n0, n3, avail = [], 0, 0, 0
    for d in g39.TARGETS:
        v, k, c, m, on_ = per[d][a]
        s = f"{v:.3f}" + (f"/{m:.3f}" if m is not None else "") + ("开" if on_ else "")
        cells.append(s)
        if v >= paper[d] - 0.03: n0 += 1
        if m is not None:
            avail += 1
            if m >= paper[d] - 0.03: n3 += 1
    print(f"| {a} | " + " | ".join(cells) + f" | {n0}/6 | {n3}/{avail} | {per['Handwriting'][a][2]} |")
