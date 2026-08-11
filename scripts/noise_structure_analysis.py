# -*- coding: utf-8 -*-
"""出路 Y 第一部分（零 GPU）：R' 噪声结构 + per-CLS / trimmed δ 反事实天花板。

背景（results/decisive_e_ablation/REPORT.md §5/§9-L1）：δ=P95 被少数高噪声
(task,cls) 撑大（HW 中位 0.029 vs P95 0.110）。本脚本用已有 80 条噪声探针 +
560 条 R' 回答：
  1) 重复差的分布结构（重尾 or 连续谱）；
  2) 噪声是 CLS 属性 / 任务属性 / 交互（决定 per-CLS δ 是否可事先标定）；
  3) 反事实天花板：per-CLS δ / trimmed δ 下，存活 shared-shared label 的
     任务盲天花板，对照标定 δ 的 0.9525 与 δ=0 的 0.7473；
  4) 新增存活 label 的跨种子方向稳定性（探针 4 任务上可直接验证）；
  5) 被 δ 剃掉的任务特异比较的 |ΔR'| 幅度分布；
  6) k=3（均值 R'）主判据的 shared-only 预演：探针 4 任务上已有 3 种子，
     均值 R' + δ/√3 的天花板零 GPU 可算。

⚠️ 反事实局限（如实标注，非新判据）：全部估计基于 k=1 数据 + 每数据集仅
2 个探针任务 × 10 shared CLS 的噪声测量；per-CLS δ 用 6 个重复差的 max 估计
（样本极小，低估风险）；"若噪声更小会怎样"的推断本身有偏。

任务盲天花板口径（与 decisive REPORT §5 一致）：shared 10 CLS 的 45 个无序对
× 28 任务，seed-0 R'，|Δ|<δ 弃、严格平局弃（同 make_pairs/铁律 8 via is_better）；
每对取跨任务多数方向，天花板 = Σ多数票 / Σ总票。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autocls.task_types import is_better  # noqa: E402  铁律 8

WP = ROOT / "results" / "labels_workpoint"
OUT = ROOT / "results" / "outlet_y" / "part1"
DS_OF = {"Handwriting_": "Handwriting", "UniMiB-SHAR_": "UniMiB-SHAR"}
TASK_TYPE = "classification"
ROUND = 0.005  # δ 上取整粒度（与 2c calibrate 一致）


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ds_of(task_id: str) -> str:
    for pre, name in DS_OF.items():
        if task_id.startswith(pre):
            return name
    raise KeyError(task_id)


def ceil_delta(x: float) -> float:
    return float(np.ceil(x / ROUND) * ROUND)


def p95(sorted_vals: list[float]) -> float:
    """与 2c calibrate 同款经验分位（最近秩，非插值）。"""
    return sorted_vals[min(len(sorted_vals) - 1, int(round(0.95 * (len(sorted_vals) - 1))))]


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

def load() -> dict:
    cache = read_jsonl(WP / "rprime_cache.jsonl")
    probe = read_jsonl(WP / "noise_probe.jsonl")
    calib = json.loads((WP / "delta_calibration.json").read_text(encoding="utf-8"))

    shared = sorted({r["cls_key"] for r in cache if r.get("origin") == "shared"})
    assert len(shared) == 10, shared

    # seed-0 R'（造 label 用的那份）
    r0: dict[str, dict[str, float]] = defaultdict(dict)  # task_id -> cls_key -> R'
    for r in cache:
        if r.get("r_prime") is not None:
            r0[r["task_id"]][r["cls_key"]] = float(r["r_prime"])
    tasks = sorted(r0)
    assert len(tasks) == 28

    # 探针 cell：(task_id, cls_key) -> {seed: R'}（seed0 来自 cache）
    reps: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in probe:
        assert r["error"] is None
        reps[(r["task_id"], r["cls_key"])][r["seed"]] = float(r["r_prime"])
    for (tid, k) in list(reps):
        reps[(tid, k)][0] = r0[tid][k]
    assert all(len(v) == 3 for v in reps.values()) and len(reps) == 40

    return {"r0": r0, "tasks": tasks, "shared": shared, "reps": reps, "calib": calib,
            "probe_tasks": sorted({t for t, _ in reps})}


# ---------------------------------------------------------------------------
# 任务盲天花板（δ 可为标量 map，也可为逐对函数）
# ---------------------------------------------------------------------------

def ceiling(rvals: dict[str, dict[str, float]], tasks: list[str], shared: list[str],
            delta_of) -> dict:
    """delta_of(dataset, cls_a, cls_b) -> δ。返回 pooled/within 天花板与存活计数。"""
    votes: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)  # pair -> [(task, sign)]
    n_delta = n_tie = 0
    for t in tasks:
        name = ds_of(t)
        for a, b in combinations(shared, 2):
            ra, rb = rvals[t].get(a), rvals[t].get(b)
            if ra is None or rb is None:
                continue
            if abs(ra - rb) < delta_of(name, a, b):
                n_delta += 1
                continue
            ab, ba = is_better(ra, rb, TASK_TYPE), is_better(rb, ra, TASK_TYPE)
            if not (ab or ba):
                n_tie += 1
                continue
            votes[(a, b)].append((t, 1 if ab else 0))

    def tally(filter_ds: str | None) -> tuple[int, int]:
        maj = tot = 0
        for vs in votes.values():
            sel = [s for t, s in vs if filter_ds is None or ds_of(t) == filter_ds]
            if not sel:
                continue
            pos = sum(sel)
            maj += max(pos, len(sel) - pos)
            tot += len(sel)
        return maj, tot

    out = {"n_delta_dropped": n_delta, "n_tie_dropped": n_tie}
    maj, tot = tally(None)
    out["pooled"] = {"ceiling": maj / tot if tot else None, "n_survive": tot}
    for name in ("Handwriting", "UniMiB-SHAR"):
        maj, tot = tally(name)
        out[name] = {"ceiling": maj / tot if tot else None, "n_survive": tot}
    out["votes"] = votes
    return out


def const_delta(dmap: dict[str, float]):
    return lambda name, a, b: dmap[name]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    D = load()
    r0, tasks, shared, reps = D["r0"], D["tasks"], D["shared"], D["reps"]
    calib = D["calib"]
    OUT.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    dmap = {n: calib[n]["delta"] for n in calib}
    floor = {n: calib[n]["quantum_floor"] for n in calib}

    # ---- §0 基线复现 ---------------------------------------------------
    base = ceiling(r0, tasks, shared, const_delta(dmap))
    zero = ceiling(r0, tasks, shared, const_delta({n: 0.0 for n in dmap}))
    R["baseline"] = {
        "delta_map": dmap,
        "calibrated": {k: v for k, v in base.items() if k != "votes"},
        "delta0": {k: v for k, v in zero.items() if k != "votes"},
    }
    print("== §0 基线复现（目标 0.9525 / HW 1.0000 / UM 0.9890；δ=0 → 0.7473）==")
    print(f"  δ标定  pooled={base['pooled']['ceiling']:.4f} (n={base['pooled']['n_survive']}) "
          f"HW={base['Handwriting']['ceiling']:.4f} UM={base['UniMiB-SHAR']['ceiling']:.4f}")
    print(f"  δ=0    pooled={zero['pooled']['ceiling']:.4f} (n={zero['pooled']['n_survive']})")

    # ---- §1 噪声分布结构 -----------------------------------------------
    # cell = (task, cls)：3 个重复 → 3 个两两差
    cells: dict[tuple[str, str], dict] = {}
    for (tid, k), sv in reps.items():
        vals = [sv[s] for s in sorted(sv)]
        diffs = [abs(x - y) for x, y in combinations(vals, 2)]
        cells[(tid, k)] = {
            "dataset": ds_of(tid), "task": tid, "cls": k, "reps": vals,
            "diffs": diffs, "mean_diff": float(np.mean(diffs)),
            "max_diff": float(max(diffs)), "std": float(np.std(vals, ddof=1)),
            "mean_r": float(np.mean(vals)),
        }
    dist = {}
    for name in dmap:
        ds_diffs = sorted(d for c in cells.values() if c["dataset"] == name for d in c["diffs"])
        cell_means = sorted((c["mean_diff"] for c in cells.values() if c["dataset"] == name),
                            reverse=True)
        qs = {f"p{q}": float(np.percentile(ds_diffs, q)) for q in (10, 25, 50, 75, 90, 95, 100)}
        top4 = sum(cell_means[:4]) / sum(cell_means)  # 最吵 20% cell 占总噪声比
        hist, edges = np.histogram(ds_diffs, bins=np.arange(0, 0.185, 0.02))
        dist[name] = {"n_diffs": len(ds_diffs), "quantiles": qs,
                      "top20pct_cells_share_of_noise": float(top4),
                      "hist_0.02bins": hist.tolist(),
                      "cell_mean_diffs_sorted": [round(v, 4) for v in cell_means]}
    R["noise_distribution"] = dist
    print("\n== §1 重复差分布（每数据集 20 cell × 3 diff）==")
    for name, d in dist.items():
        q = d["quantiles"]
        print(f"  {name}: med={q['p50']:.3f} p90={q['p90']:.3f} p95={q['p95']:.3f} "
              f"max={q['p100']:.3f} | 最吵20%cell占噪声 {d['top20pct_cells_share_of_noise']:.0%}")
        print(f"    hist(0.02 bins): {d['hist_0.02bins']}")
        print(f"    cell mean_diff 降序: {d['cell_mean_diffs_sorted']}")

    # ---- §2 噪声归因：CLS / 任务 / 交互 --------------------------------
    attr = {}
    for name in dmap:
        tids = sorted({t for (t, _k) in cells if ds_of(t) == name})
        grid = np.array([[cells[(t, k)]["mean_diff"] for k in shared] for t in tids])  # 2×10
        gm = grid.mean()
        cls_eff = grid.mean(axis=0) - gm
        task_eff = grid.mean(axis=1) - gm
        resid = grid - gm - cls_eff[None, :] - task_eff[:, None]
        sst = float(((grid - gm) ** 2).sum())
        attr[name] = {
            "ss_share": {"cls": float((cls_eff ** 2).sum() * len(tids) / sst),
                         "task": float((task_eff ** 2).sum() * len(shared) / sst),
                         "interaction_resid": float((resid ** 2).sum() / sst)},
            "spearman_task0_vs_task1_cell_noise": float(_spearman(grid[0], grid[1])),
            "per_cls_noise": {k: float(grid[:, i].mean()) for i, k in enumerate(shared)},
        }
    hw = np.array([attr["Handwriting"]["per_cls_noise"][k] for k in shared])
    um = np.array([attr["UniMiB-SHAR"]["per_cls_noise"][k] for k in shared])
    attr["spearman_perCLS_HW_vs_UM"] = float(_spearman(hw, um))
    # 噪声 vs 配置质量（mean R'）
    for name in dmap:
        mr = np.array([np.mean([cells[(t, k)]["mean_r"] for (t, kk) in cells
                                if kk == k and ds_of(t) == name and (t, k) in cells])
                       for k in shared])
        nz = np.array([attr[name]["per_cls_noise"][k] for k in shared])
        attr[name]["spearman_noise_vs_meanR"] = float(_spearman(nz, mr))
    R["attribution"] = attr
    print("\n== §2 噪声归因（cell mean_diff 的二因素分解；每数据集仅 2 任务，粗）==")
    for name in dmap:
        a = attr[name]
        s = a["ss_share"]
        print(f"  {name}: SS份额 cls={s['cls']:.2f} task={s['task']:.2f} "
              f"交互/残差={s['interaction_resid']:.2f} | "
              f"任务间 per-CLS 噪声 Spearman={a['spearman_task0_vs_task1_cell_noise']:.3f} | "
              f"噪声~meanR' Spearman={a['spearman_noise_vs_meanR']:.3f}")
    print(f"  跨数据集 per-CLS 噪声 Spearman(HW,UM) = {attr['spearman_perCLS_HW_vs_UM']:.3f}")

    # ---- §3 反事实天花板：per-CLS δ / trimmed δ ------------------------
    # per-CLS δ：δ_c = ceil(max(该 CLS 6 个重复差的 max, 量子下限))；δ_pair = max(δ_a, δ_b)
    per_cls_delta: dict[str, dict[str, float]] = {}
    per_cls_delta_2nd: dict[str, dict[str, float]] = {}
    for name in dmap:
        per_cls_delta[name] = {}
        per_cls_delta_2nd[name] = {}
        for k in shared:
            ds6 = sorted(d for (t, kk), c in cells.items()
                         if kk == k and c["dataset"] == name for d in c["diffs"])
            per_cls_delta[name][k] = ceil_delta(max(ds6[-1], floor[name]))
            per_cls_delta_2nd[name][k] = ceil_delta(max(ds6[-2], floor[name]))

    variants: dict[str, dict] = {}

    def run_variant(tag: str, delta_of, note: str, deltas_repr) -> None:
        c = ceiling(r0, tasks, shared, delta_of)
        variants[tag] = {"note": note, "deltas": deltas_repr,
                         **{k: v for k, v in c.items() if k != "votes"}}
        variants[tag]["_votes"] = c["votes"]

    run_variant("perCLS_max", lambda n, a, b: max(per_cls_delta[n][a], per_cls_delta[n][b]),
                "δ_pair=max(δ_a,δ_b)，δ_c=cell 6 diff 的 max（保守）", per_cls_delta)
    run_variant("perCLS_2nd", lambda n, a, b: max(per_cls_delta_2nd[n][a], per_cls_delta_2nd[n][b]),
                "δ_c=6 diff 的次大值（宽松，P83≈级）", per_cls_delta_2nd)
    for trim in (10, 20, 30):
        tdmap = {}
        for name in dmap:
            keep_cells = sorted((c for c in cells.values() if c["dataset"] == name),
                                key=lambda c: c["mean_diff"])
            n_drop = int(round(len(keep_cells) * trim / 100))
            kept = keep_cells[:len(keep_cells) - n_drop]
            ds_diffs = sorted(d for c in kept for d in c["diffs"])
            tdmap[name] = ceil_delta(max(p95(ds_diffs), floor[name]))
        run_variant(f"trimmed{trim}", const_delta(tdmap),
                    f"去掉最吵 {trim}% cell 后重算 P95 全局 δ", tdmap)

    print("\n== §3 反事实天花板（shared-shared，28 任务；对照 δ标定 0.9525 / δ=0 0.7473）==")
    print(f"  per-CLS δ (max)   HW={sorted(per_cls_delta['Handwriting'].values())} ")
    print(f"                    UM={sorted(per_cls_delta['UniMiB-SHAR'].values())}")
    for tag, v in variants.items():
        p = v["pooled"]
        print(f"  {tag:<12} pooled={p['ceiling']:.4f} (n={p['n_survive']}, "
              f"存活率 {p['n_survive'] / zero['pooled']['n_survive']:.0%} of δ=0) "
              f"HW={v['Handwriting']['ceiling']:.4f}(n={v['Handwriting']['n_survive']}) "
              f"UM={v['UniMiB-SHAR']['ceiling']:.4f}(n={v['UniMiB-SHAR']['n_survive']})")

    # trimmed δ 的“带病存活”率：存活对里 |Δ| < 该对自身实测噪声上限（per-CLS max）的占比
    for trim in (10, 20, 30):
        tag = f"trimmed{trim}"
        at_risk = tot = 0
        tdmap = variants[tag]["deltas"]
        for t in tasks:
            name = ds_of(t)
            for a, b in combinations(shared, 2):
                ra, rb = r0[t].get(a), r0[t].get(b)
                if ra is None or rb is None or abs(ra - rb) < tdmap[name] or ra == rb:
                    continue
                tot += 1
                if abs(ra - rb) < max(per_cls_delta[name][a], per_cls_delta[name][b]):
                    at_risk += 1
        variants[tag]["at_risk_fraction"] = at_risk / tot if tot else None
        print(f"  {tag}: 存活对中 |Δ| 低于该对实测噪声上限的比例 = {at_risk / tot:.1%}")

    # ---- §4 新增存活 label 的跨种子方向稳定性（探针 4 任务直接验证） ----
    stab = {}
    for tag in ("perCLS_max", "perCLS_2nd"):
        dvar = variants[tag]["deltas"]
        cat = {"new": [], "old": []}
        for t in D["probe_tasks"]:
            name = ds_of(t)
            for a, b in combinations(shared, 2):
                ra, rb = r0[t][a], r0[t][b]
                if ra == rb:
                    continue
                d = abs(ra - rb)
                in_base = d >= dmap[name]
                in_var = d >= max(dvar[name][a], dvar[name][b])
                if not in_var:
                    continue
                signs = []
                for s in (0, 1, 2):
                    va, vb = reps[(t, a)][s], reps[(t, b)][s]
                    signs.append(0 if va == vb else (1 if is_better(va, vb, TASK_TYPE) else -1))
                stable = len({x for x in signs if x != 0}) == 1 and 0 not in signs
                mean_sign = np.sign(np.mean([reps[(t, a)][s] for s in (0, 1, 2)])
                                    - np.mean([reps[(t, b)][s] for s in (0, 1, 2)]))
                seed0_matches_mean = (np.sign(ra - rb) == mean_sign)
                cat["old" if in_base else "new"].append((stable, bool(seed0_matches_mean)))
        stab[tag] = {c: {"n": len(v),
                         "sign_stable_3seed": float(np.mean([x[0] for x in v])) if v else None,
                         "seed0_matches_mean3": float(np.mean([x[1] for x in v])) if v else None}
                     for c, v in cat.items()}
    R["stability_probe_tasks"] = stab
    print("\n== §4 探针 4 任务上的方向稳定性（new=perCLS 新增存活、old=标定 δ 也存活）==")
    for tag, sc in stab.items():
        for c, v in sc.items():
            if v["n"]:
                print(f"  {tag}/{c}: n={v['n']} 三种子同向率={v['sign_stable_3seed']:.1%} "
                      f"seed0 与均值同向率={v['seed0_matches_mean3']:.1%}")

    # ---- §5 被剃掉的任务特异比较：|Δ| 幅度分布 -------------------------
    maj_dir = {}
    for pair, vs in zero["votes"].items():
        pos = sum(s for _t, s in vs)
        maj_dir[pair] = 1 if pos * 2 >= len(vs) else 0
    shaved = {"task_specific": [], "majority": []}
    for t in tasks:
        name = ds_of(t)
        for a, b in combinations(shared, 2):
            ra, rb = r0[t].get(a), r0[t].get(b)
            if ra is None or rb is None or ra == rb:
                continue
            d = abs(ra - rb)
            if d >= dmap[name]:
                continue  # 存活，不是被剃的
            s = 1 if is_better(ra, rb, TASK_TYPE) else 0
            key = "majority" if s == maj_dir[(a, b)] else "task_specific"
            shaved[key].append(d)
    R["shaved_magnitudes"] = {
        k: {"n": len(v),
            "quantiles": {f"p{q}": float(np.percentile(v, q)) for q in (25, 50, 75, 90)} if v else {}}
        for k, v in shaved.items()}
    ts = shaved["task_specific"]
    print("\n== §5 被标定 δ 剃掉的比较（按 δ=0 跨任务多数方向分型）==")
    print(f"  少数方向（任务特异票）: n={len(ts)} |Δ| 分位 "
          f"p25={np.percentile(ts, 25):.3f} p50={np.percentile(ts, 50):.3f} "
          f"p75={np.percentile(ts, 75):.3f} p90={np.percentile(ts, 90):.3f}")
    print(f"  多数方向（同向小差距）: n={len(shaved['majority'])}")

    # ---- §6 k=3 预演（shared-only，探针 4 任务，零 GPU） ----------------
    rbar: dict[str, dict[str, float]] = defaultdict(dict)
    for (t, k), sv in reps.items():
        rbar[t][k] = float(np.mean(list(sv.values())))
    dmap_k3 = {}
    for name in dmap:
        ds_diffs = sorted(d for c in cells.values() if c["dataset"] == name for d in c["diffs"])
        # 均值的重复差分布 ≈ 单种子重复差 / sqrt(3)（iid 假设；floor 也除 3：量子=1/(3n)）
        dmap_k3[name] = ceil_delta(max(p95(ds_diffs) / np.sqrt(3), floor[name] / 3))
    pt = D["probe_tasks"]
    k3 = ceiling(rbar, pt, shared, const_delta(dmap_k3))
    k1_probe = ceiling(r0, pt, shared, const_delta(dmap))
    k1_probe0 = ceiling(r0, pt, shared, const_delta({n: 0.0 for n in dmap}))
    k3_0 = ceiling(rbar, pt, shared, const_delta({n: 0.0 for n in dmap}))
    k3_percls = ceiling(rbar, pt, shared,
                        lambda n, a, b: max(per_cls_delta[n][a], per_cls_delta[n][b]) / np.sqrt(3))
    # 28 任务近似：k=1 值 + δ_k3（非真 k=3——survival 带边缘的 k=1 噪声翻转仍在，
    # 方向不严格单边；仅作 28 任务 k=3 结果的先行估计）
    k3_approx28 = ceiling(r0, tasks, shared, const_delta(dmap_k3))
    R["k3_preview_probe_tasks"] = {
        "k1values_28tasks_delta_k3_approx": {k: v for k, v in k3_approx28.items()
                                             if k != "votes"},
        "delta_k3": dmap_k3,
        "k1_calibrated": {k: v for k, v in k1_probe.items() if k != "votes"},
        "k1_delta0": {k: v for k, v in k1_probe0.items() if k != "votes"},
        "k3_calibrated_sqrt3": {k: v for k, v in k3.items() if k != "votes"},
        "k3_delta0": {k: v for k, v in k3_0.items() if k != "votes"},
        "k3_perCLS_sqrt3": {k: v for k, v in k3_percls.items() if k != "votes"},
    }
    print("\n== §6 k=3 预演（仅探针 4 任务、shared 45 对；同任务集 k=1 作对照）==")
    print(f"  δ_k3(P95/√3): {dmap_k3}")
    for tag, c in (("k1 δ标定", k1_probe), ("k1 δ=0", k1_probe0),
                   ("k3 δ/√3", k3), ("k3 δ=0", k3_0), ("k3 perCLSδ/√3", k3_percls)):
        p = c["pooled"]
        print(f"  {tag:<14} pooled={p['ceiling']:.4f} (n={p['n_survive']}) "
              f"HW={c['Handwriting']['ceiling']:.4f} UM={c['UniMiB-SHAR']['ceiling']:.4f}")
    p = k3_approx28["pooled"]
    print(f"  28任务近似（k=1 值+δ_k3，非真 k=3）: pooled={p['ceiling']:.4f} "
          f"(n={p['n_survive']}) HW={k3_approx28['Handwriting']['ceiling']:.4f} "
          f"UM={k3_approx28['UniMiB-SHAR']['ceiling']:.4f}")

    # ---- §7 需要把噪声压到多低：required-k 表 --------------------------
    # 任务特异票（δ=0 跨任务少数方向）的 |Δ| 分位 → 要让它们过 δ，P95 需压到该值
    # → iid 假设下 k = (P95_k1 / δ_target)²。⚠️ k=1 的 |Δ| 本身含噪声膨胀，
    # 真实信号幅度只会更小 → k 只会更大（下界）。
    ts_by_ds = {n: [] for n in dmap}
    for t in tasks:
        name = ds_of(t)
        for a, b in combinations(shared, 2):
            ra, rb = r0[t].get(a), r0[t].get(b)
            if ra is None or rb is None or ra == rb:
                continue
            s = 1 if is_better(ra, rb, TASK_TYPE) else 0
            if s != maj_dir[(a, b)]:
                ts_by_ds[name].append(abs(ra - rb))
    req = {}
    for name in dmap:
        p95_k1 = calib[name]["p95_replicate_diff"]
        v = np.array(ts_by_ds[name])
        rows = {}
        for q in (25, 50, 75):
            dt = float(np.percentile(v, q))
            rows[f"p{q}"] = {"delta_target": dt, "k_needed": float((p95_k1 / dt) ** 2)}
        rows["frac_ts_above_delta_k3"] = float(np.mean(v >= dmap_k3[name]))
        rows["n_task_specific_votes"] = int(len(v))
        req[name] = rows
    R["required_k"] = req
    print("\n== §7 要压到多低才够（任务特异票 |Δ| 分位 → 所需 k，iid 下界）==")
    for name, rows in req.items():
        print(f"  {name}: n_ts={rows['n_task_specific_votes']} "
              f"δ_k3 之上占比={rows['frac_ts_above_delta_k3']:.1%}")
        for q in ("p25", "p50", "p75"):
            r_ = rows[q]
            print(f"    容 {q}: δ 需≤{r_['delta_target']:.3f} → k≈{r_['k_needed']:.0f}")

    for v in variants.values():
        v.pop("_votes", None)
    R["counterfactual_variants"] = variants
    (OUT / "analysis.json").write_text(
        json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n→ {OUT / 'analysis.json'}")


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return spearmanr(x, y).statistic


if __name__ == "__main__":
    main()
