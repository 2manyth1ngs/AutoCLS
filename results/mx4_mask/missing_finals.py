"""汇总部署臂 + 事后扫臂里所有冠军配置，找出还没有 seeds {1,2} 的 (dataset, cls_key)，写成 picks.json 供 39 --finals。"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
g40 = importlib.import_module("scripts.40_mask_field_sim"); g36 = g40._g36; g39 = g40._g39
OUT = ROOT / "results" / "mx4_mask"
T = list(g39.TARGETS)
off, on = g40.load_wp3_rows(), g40.load_mask_rows()
s2 = [r for r in off + on if r.get("stage") == "s2_finals" and r.get("error") is None]
cands = g39.load_candidates(OUT)
picks: dict[str, dict[str, str]] = {}
summary = []
for d in T:
    ro, rn = g40._s1_seed0(off, d), g40._s1_seed0(on, d)
    to, so = g39.trajectory_from_rows(ro); tn, sn = g39.trajectory_from_rows(rn)
    traj = {**to, **tn}; secs = {**so, **sn}
    budgets = sorted({b for v in to.values() for b in v}); bf = budgets[-1]
    keys_off = sorted(k for k in to if all(b in to[k] for b in budgets))
    twin_map = {r["twin_of"]: r["cls_key"] for r in rn if r.get("twin_of") in to and all(b in tn.get(r["cls_key"], {}) for b in budgets)}
    keys = keys_off + sorted(twin_map.values())
    truth = {k: traj[k][bf] for k in keys}
    base = g40.wp3_rankings(cands[d], keys_off)["prior"]
    rk = {"prior_off": base, "interleave": g40.interleave(base, twin_map), "allon": g40.all_on(base, twin_map)}
    row = g40.mask_arms(keys, traj, rk, twin_map, budgets=budgets, seconds=secs)
    champs = dict(row["picks"])
    # 扫臂
    for top in (6, 10):
        off_r, on_r, mix_r = base[:top], g40.all_on(base[:top], twin_map), g40.interleave(base[:top], twin_map)
        for name, ranked, ks, bud in [
            (f"top{top}_off_3lvl", off_r, (top, 4 if top == 10 else 3, 3), budgets),
            (f"top{top}_on_3lvl", on_r, (len(on_r), 4 if top == 10 else 3, 3), budgets),
            (f"top{top}_both_3lvl", mix_r, (len(mix_r), 4 if top == 10 else 3, 3), budgets),
            (f"top{top}_off_2lvl", off_r, (top, 3), budgets[1:]),
            (f"top{top}_on_2lvl", on_r, (len(on_r), 3), budgets[1:]),
            (f"top{top}_both_2lvl", mix_r, (len(mix_r), 3), budgets[1:]),
            (f"top{top}_both_2lvl_keep4", mix_r, (len(mix_r), 4), budgets[1:]),
        ]:
            ranked = [k for k in ranked if k in truth]
            p = g36.sh_path(ranked, traj, ks, bud)
            champs[name] = g36.sh_pick(p[-1], traj, bf)
    have = {}
    for r in s2:
        if r["dataset"] == d:
            have.setdefault(r["cls_key"], set()).add(int(r["seed"]))
    need = {}
    for arm, k in champs.items():
        if not {1, 2} <= have.get(k, set()):
            need[arm] = k
    picks[d] = need
    uniq = sorted(set(need.values()))
    summary.append((d, len(champs), uniq, [truth[k] for k in uniq], [k in set(twin_map.values()) for k in uniq]))
(OUT / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
tot = 0
for d, n_arms, uniq, accs, ons in summary:
    tot += len(uniq)
    print(f"{d}: {n_arms} 臂冠军，缺 3 种子的配置 {len(uniq)} 条 → " +
          ", ".join(f"{k[:8]}{'(开)' if o else ''} {a:.3f}" for k, a, o in zip(uniq, accs, ons)))
print(f"合计 {tot} 条配置 × 2 种子 = {2 * tot} 格")
