"""MX-4 WP1：源池先验 + 目标侧逐次减半（SH）的缓存模拟——零 GPU。

协议：results/mx4_prior_sh/PROTOCOL.md（臂表 §3、判读 §4、预测 §5 先于正式跑写死；§0 如实
声明本轮不是盲预注册——探索读数已看过，本脚本是把它固化成可复现、可审计的正式口径）。

数据：results/mx3_pq/{rprime_cache.jsonl, noise_probe.jsonl, rankings.json}（只读）。
MX-3-PQ 的 60/120/290 是**同一条轨迹的三个截点**，因此"先在目标切片上短训 k0 条、按截点
真值逐级留下、最后只把幸存者跑到 b_final"这套多保真搜索可以在缓存上精确模拟——部署时
选中的就是那条轨迹（继续训练即可），不存在额外近似。

臂：零样本（prior / gbdt / hybrid / tclsc / clsb / clsonly / ggs / random）+ 多保真
（SH{k0}-{k1}-{k2}_{warm}、all20@60→top{1,3}）；regret = 目录 oracle@b_final − 臂。
成本：iters 累加 + 缓存 `seconds`（该截点为止的累计墙钟）换算 GPU 秒。

用法：
  python scripts/36_prior_sh_sim.py --analyze            # 全量 126 实例 → analysis.json + tables.md
  python scripts/36_prior_sh_sim.py --analyze --folds Handwriting UniMiB-SHAR
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_mx2 = importlib.import_module("scripts.25_mx2_endtoend")
_g16 = importlib.import_module("scripts.16_feasibility_loo")
expected_max_of_k = _mx2.expected_max_of_k
bottom_quartile_hit = _mx2.bottom_quartile_hit
paired_wilcoxon_less = _g16.paired_wilcoxon_less

# ---------------------------------------------------------------- 协议常量（PROTOCOL §3–§4）
OUT = ROOT / "results" / "mx4_prior_sh"
MX3 = pq.OUT
BUDGETS = pq.BUDGETS                       # (60, 120, 290)：同一轨迹的截点
SH_CONFIGS = ((6, 3, 1), (6, 3, 3), (10, 4, 1), (10, 4, 3))
SH_WARMS = ("prior", "hybrid", "gbdt")     # + random（抽样期望）
N_RANDOM_DRAWS = 20
HYBRID_BIAS = 10.0                         # hybrid：shared 按先验置顶，random 按 GBDT
ALPHA = 0.05
K_TOP = 3
MAIN_SH = "SH6-3-3_prior"                  # 主门臂
FRESH_SLICES = ("00", "01")                # 噪声探针覆盖的切片（PROTOCOL §2）
FRESH_REPS = (1, 2)


# ---------------------------------------------------------------- 纯函数
def zscore_within(rows: Iterable[dict]) -> list[dict]:
    """每条记录加 within-(task_id, iters) z 分数（副本，不改入参）。"""
    rows = [dict(r) for r in rows]
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task_id"], int(r["iters"]))].append(r)
    for g in groups.values():
        v = np.array([float(r["r_prime"]) for r in g])
        mu, sd = float(v.mean()), float(v.std())
        for r in g:
            r["z"] = (float(r["r_prime"]) - mu) / sd if sd > 0 else 0.0
    return rows


def prior_scores(rows: Iterable[dict]) -> dict[str, float]:
    """先验 = 每个 cls_key 的 z 均值（跨训练侧全部切片与预算）。"""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["cls_key"]].append(float(r["z"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def rank_by(score: Mapping[str, float], keys: Sequence[str],
            unseen: float = float("-inf")) -> list[str]:
    """得分降序，同分按 key 升序，未见键垫底（逐位确定）。"""
    return sorted(keys, key=lambda k: (-float(score.get(k, unseen)), k))


def sh_path(ranked: Sequence[str], traj: Mapping[str, Mapping[int, float]],
            ks: Sequence[int], budgets: Sequence[int]) -> list[list[str]]:
    """逐级留存路径：path[i] = 第 i 级（预算 budgets[i]）在训练的候选（按上一级截点真值降序）。"""
    if len(ks) != len(budgets) or not ks:
        raise ValueError(f"ks {ks} 与 budgets {budgets} 必须等长且非空")
    if list(budgets) != sorted(budgets):
        raise ValueError("budgets 必须升序")
    cur = list(ranked)[:int(ks[0])]
    path = [list(cur)]
    for i in range(1, len(budgets)):
        prev_b = int(budgets[i - 1])
        cur = sorted(cur, key=lambda k: (-float(traj[k][prev_b]), k))[:int(ks[i])]
        path.append(list(cur))
    return path


def path_cost_iters(path: Sequence[Sequence[str]], budgets: Sequence[int]) -> int:
    prev = 0
    cost = 0
    for stage, b in zip(path, budgets):
        cost += len(stage) * (int(b) - prev)
        prev = int(b)
    return int(cost)


def path_cost_seconds(path: Sequence[Sequence[str]], budgets: Sequence[int],
                      seconds: Mapping[str, Mapping[int, float]]) -> float:
    """seconds[k][b] 是累计墙钟：每级只计增量 seconds[k][b_i] − seconds[k][b_{i−1}]。"""
    total = 0.0
    prev: Optional[int] = None
    for stage, b in zip(path, budgets):
        for k in stage:
            total += float(seconds[k][int(b)]) - (float(seconds[k][prev]) if prev is not None else 0.0)
        prev = int(b)
    return float(total)


def successive_halving(ranked: Sequence[str], traj, ks, budgets) -> tuple[list[str], int]:
    """(最后一级幸存者, 成本 iters)。幸存者顺序 = 最后一次筛选截点的 R' 降序。"""
    path = sh_path(ranked, traj, ks, budgets)
    return path[-1], path_cost_iters(path, budgets)


def sh_pick(final: Sequence[str], traj, final_budget: int) -> str:
    """幸存者里 b_final 真值最高者，同分按 key 升序。"""
    return sorted(final, key=lambda k: (-float(traj[k][int(final_budget)]), k))[0]


def expected_random_sh(keys: Sequence[str], traj, ks, budgets, n_draws: int, seed: int) -> float:
    """随机起点的 SH 期望 R'@b_final（n_draws 次随机排列的均值，seed 定死）。"""
    rng = np.random.default_rng(seed)
    keys = list(keys)
    acc = []
    for _ in range(int(n_draws)):
        order = [keys[i] for i in rng.permutation(len(keys))]
        final, _ = successive_halving(order, traj, ks, budgets)
        acc.append(float(traj[sh_pick(final, traj, budgets[-1])][int(budgets[-1])]))
    return float(np.mean(acc))


def sh_label(ks: Sequence[int]) -> str:
    return "SH" + "-".join(str(int(k)) for k in ks)


def instance_row(keys: Sequence[str], traj: Mapping[str, Mapping[int, float]],
                 rank: Mapping[str, Sequence[str]], *, ggs_r: Optional[float],
                 final_budget: int, budgets: Sequence[int],
                 seconds: Optional[Mapping[str, Mapping[int, float]]],
                 random_seed: int, k_top: int = K_TOP) -> dict:
    """一个搜索实例的全部臂 / regret / 成本（PROTOCOL §3）。

    rank 必须含 prior / gbdt / hybrid / tclsc / clsb / clsonly（tclsc 缺席 fail-fast：
    它是 G3 的分母，不能拿别的臂顶包）。budgets = 截点升序且以 final_budget 结尾。
    """
    for need in ("prior", "gbdt", "hybrid", "tclsc", "clsb", "clsonly"):
        if need not in rank:
            raise KeyError(f"instance_row 缺 {need!r} 排名")
    b = int(final_budget)
    stages = [int(x) for x in budgets]
    if stages[-1] != b:
        raise ValueError(f"budgets {budgets} 必须以 final_budget {b} 结尾")
    keys = list(keys)
    truth = {k: float(traj[k][b]) for k in keys}
    vals = list(truth.values())
    oracle = max(vals)
    arms: dict[str, float] = {"oracle": oracle}
    cost: dict[str, int] = {}
    secs: Optional[dict[str, float]] = {} if seconds is not None else None
    picks: dict[str, str] = {}

    def add_zero_shot(name: str, ranked: Sequence[str], top: int) -> None:
        chosen = list(ranked)[:top]
        best = sorted(chosen, key=lambda k: (-truth[k], k))[0]
        arms[name] = truth[best]
        cost[name] = top * b
        picks[name] = best
        if secs is not None:
            secs[name] = float(sum(float(seconds[k][b]) for k in chosen))

    for arm in ("prior", "gbdt", "hybrid", "tclsc"):
        add_zero_shot(arm + "1", rank[arm], 1)
        add_zero_shot(arm + "3", rank[arm], k_top)
    add_zero_shot("clsb1", rank["clsb"], 1)
    add_zero_shot("clsonly1", rank["clsonly"], 1)
    if ggs_r is not None:
        arms["ggs"] = float(ggs_r)
        cost["ggs"] = b
    arms["random1"] = float(np.mean(vals))
    arms["random3"] = float(expected_max_of_k(vals, min(k_top, len(vals))))
    arms["bottom"] = float(min(vals))
    cost["random1"] = b
    cost["random3"] = k_top * b
    cost["catalog_direct"] = len(keys) * b
    if secs is not None:
        mean_s = float(np.mean([float(seconds[k][b]) for k in keys]))
        secs["random1"] = mean_s
        secs["random3"] = k_top * mean_s
        secs["catalog_direct"] = float(sum(float(seconds[k][b]) for k in keys))

    if len(stages) >= 2:
        for cfg in SH_CONFIGS:
            ks = tuple(cfg[:len(stages)])
            label = sh_label(ks)
            for warm in SH_WARMS:
                path = sh_path(rank[warm], traj, ks, stages)
                pick = sh_pick(path[-1], traj, b)
                name = f"{label}_{warm}"
                arms[name] = truth[pick]
                picks[name] = pick
                cost[name] = path_cost_iters(path, stages)
                if secs is not None:
                    secs[name] = path_cost_seconds(path, stages, seconds)
            name = f"{label}_random"
            arms[name] = expected_random_sh(keys, traj, ks, stages, N_RANDOM_DRAWS, random_seed)
            ref_path = sh_path(keys, traj, ks, stages)          # 同 k 截断的成本（与起点无关）
            cost[name] = path_cost_iters(ref_path, stages)
            if secs is not None:
                secs[name] = float(np.mean([
                    path_cost_seconds(sh_path([keys[i] for i in np.random.default_rng(random_seed + d)
                                               .permutation(len(keys))], traj, ks, stages),
                                      stages, seconds)
                    for d in range(N_RANDOM_DRAWS)]))
        if len(stages) == 3:
            n = len(keys)
            for top in (1, k_top):
                path = sh_path(keys, traj, (n, top, top), stages)
                pick = sh_pick(path[-1], traj, b)
                name = f"all20@{stages[0]}->top{top}"
                arms[name] = truth[pick]
                picks[name] = pick
                cost[name] = path_cost_iters(path, stages)
                if secs is not None:
                    secs[name] = path_cost_seconds(path, stages, seconds)

    regret = {k: oracle - v for k, v in arms.items()}
    hits = {k: bool(bottom_quartile_hit(truth, p)) for k, p in picks.items()}
    return {"arms": arms, "regret": regret, "cost_iters": cost, "cost_seconds": secs,
            "picks": picks, "bottom_quartile_hit": hits}


def gate(rows: Sequence[dict], arm_a: str, arm_b: str, *, alpha: float = ALPHA) -> dict:
    """配对单侧 Wilcoxon（34 号同式）：H1 = regret[arm_a] < regret[arm_b]；缺臂 fail-fast。"""
    missing = [r for r in rows if arm_a not in r["regret"] or arm_b not in r["regret"]]
    if not rows or missing:
        raise KeyError(f"{len(missing)}/{len(rows)} 个实例缺 {arm_a}/{arm_b} 臂" if rows else "无实例")
    x = np.array([float(r["regret"][arm_a]) for r in rows])
    y = np.array([float(r["regret"][arm_b]) for r in rows])
    p = float(paired_wilcoxon_less(x, y))
    return {"arm_a": arm_a, "arm_b": arm_b, "n": len(rows), "p": p, "pass": bool(p < alpha),
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "median_delta": float(np.median(x - y)),
            "n_better": int(np.sum(x < y)), "n_worse": int(np.sum(x > y)),
            "n_tie": int(np.sum(x == y))}


def run_gates(rows: Sequence[dict], fresh_rows: Sequence[dict]) -> dict:
    """PROTOCOL §4：G1–G3 在 @290 实例上，G4 在新鲜种子行上（描述性），G5 在 pooled 上。"""
    r290 = [r for r in rows if int(r["budget"]) == 290]
    g = {"G1": gate(r290, MAIN_SH, "prior3"),
         "G2": gate(r290, MAIN_SH, "SH6-3-3_random"),
         "G3": gate(r290, MAIN_SH, "tclsc3"),
         "G5": gate(rows, "prior1", "ggs")}
    a = np.array([float(r["regret"][MAIN_SH]) for r in fresh_rows])
    b = np.array([float(r["regret"]["prior3"]) for r in fresh_rows])
    g["G4"] = {"arm_a": MAIN_SH, "arm_b": "prior3", "n": len(fresh_rows), "descriptive": True,
               "mean_a": float(a.mean()) if a.size else None,
               "mean_b": float(b.mean()) if b.size else None,
               "p": float(paired_wilcoxon_less(a, b)) if a.size else None,
               "pass": bool(a.size and a.mean() <= b.mean())}
    return g


def verdict_code(g: Mapping[str, dict]) -> str:
    ok = {k: bool(g[k].get("pass")) for k in ("G1", "G2", "G3", "G4")}
    if all(ok.values()):
        return "SH-WIN"
    if ok["G1"] and (ok["G2"] or ok["G3"]):
        return "SH-PARTIAL"
    return "SH-DEAD"


# ---------------------------------------------------------------- 装载与折
def load_inputs(mx3: Path):
    rows = [r for r in read_jsonl(mx3 / "rprime_cache.jsonl") if r.get("error") is None]
    cat = [r for r in rows if r["origin"] != "ggs"]
    ggs = {(r["task_id"], int(r["iters"])): float(r["r_prime"]) for r in rows if r["origin"] == "ggs"}
    probe_path = mx3 / "noise_probe.jsonl"
    probe = ([r for r in read_jsonl(probe_path) if r.get("error") is None]
             if probe_path.exists() else [])
    rankings = json.loads((mx3 / "rankings.json").read_text(encoding="utf-8"))
    return cat, ggs, probe, rankings


def features(rows: Sequence[dict], encoder: CLSEncoder, max_budget: int) -> np.ndarray:
    X = np.stack([encoder.encode(CLS(**r["cls"])) for r in rows])
    b = np.array([[int(r["iters"]) / float(max_budget)] for r in rows])
    return np.hstack([X, b])


def trajectories(rows: Sequence[dict]):
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    secs: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        traj[r["cls_key"]][int(r["iters"])] = float(r["r_prime"])
        secs[r["cls_key"]][int(r["iters"])] = float(r["seconds"])
    return traj, secs


def fold_instances(target: str, cat_z: Sequence[dict], ggs: Mapping, rankings: Mapping,
                   encoder: CLSEncoder, *, seed: int = 0) -> tuple[list[dict], dict]:
    """一折：先验 / GBDT 只见训练侧（其余 6 + SCP2），目标切片 × 3 预算各出一个实例。"""
    _train_names, _ = pq.fold_split(target)
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior = prior_scores([r for r in train if r["origin"] == ORIGIN_SHARED])
    model = FieldSurrogate(kind="hgb", seed=seed).fit(
        features(train, encoder, max(BUDGETS)), np.array([r["z"] for r in train]))
    slices = sorted({r["task_id"] for r in test})
    out = []
    for i, s in enumerate(slices):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, secs = trajectories(rows_s)
        for b in BUDGETS:
            rows_sb = [r for r in rows_s if int(r["iters"]) == b]
            keys = sorted(r["cls_key"] for r in rows_sb)
            by_key = {r["cls_key"]: r for r in rows_sb}
            g_mean, _ = model.predict(features([by_key[k] for k in keys], encoder, max(BUDGETS)))
            gbdt = {k: float(v) for k, v in zip(keys, g_mean)}
            hybrid = {k: (prior[k] + HYBRID_BIAS if k in prior else gbdt[k]) for k in keys}
            rk = rankings[target][str(b)][s]
            rank = {"prior": rank_by(prior, keys), "gbdt": rank_by(gbdt, keys),
                    "hybrid": rank_by(hybrid, keys),
                    "tclsc": list(rk["main"]), "clsb": list(rk["cls_b"]),
                    "clsonly": list(rk["cls_only"])}
            stages = tuple(x for x in BUDGETS if x <= b)
            row = instance_row(keys, traj, rank, ggs_r=ggs.get((s, b)), final_budget=b,
                               budgets=stages, seconds=secs,
                               random_seed=seed * 100003 + i * 7 + int(b))
            row.update({"fold": target, "dataset": target, "slice": s, "budget": b,
                        "n_candidates": len(keys)})
            out.append(row)
    return out, {"prior": prior, "prior_top": rank_by(prior, list(prior))[:5]}


def fresh_instances(target: str, cat_z: Sequence[dict], probe: Sequence[dict],
                    prior: Mapping[str, float]) -> list[dict]:
    """G4：切片 00/01 × shared-10，选择用 canonical 轨迹，真值用探针 rep 均值 @290。"""
    b = max(BUDGETS)
    stages = tuple(BUDGETS)
    by_cell: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in probe:
        if r["dataset"] == target and int(r["iters"]) == b:
            by_cell[(r["task_id"], r["cls_key"])][int(r["seed"])] = float(r["r_prime"])
    out = []
    for suffix in FRESH_SLICES:
        s = f"{target}_{suffix}"
        rows_s = [r for r in cat_z if r["task_id"] == s and r["origin"] == ORIGIN_SHARED]
        traj, _ = trajectories(rows_s)
        keys = sorted(k for k in traj if all(rep in by_cell.get((s, k), {}) for rep in FRESH_REPS))
        if len(keys) < 2:
            continue
        fresh = {k: float(np.mean([by_cell[(s, k)][rep] for rep in FRESH_REPS])) for k in keys}
        oracle = max(fresh.values())
        ranked = rank_by(prior, keys)
        path = sh_path(ranked, traj, (6, 3, 3), stages)
        # 两臂同规则：选择只能用 canonical 轨迹（部署时可得的信息），新鲜真值只用来打分。
        # prior3 = 先验 top-3 里按 canonical@b 最好的那条；*_fresh_max = 用新鲜真值挑 = 乐观上界。
        pick_sh = sh_pick(path[-1], traj, b)
        pick_p3 = sh_pick(ranked[:K_TOP], traj, b)
        arms = {"oracle": oracle, "prior1": fresh[ranked[0]],
                "prior3": fresh[pick_p3],
                "prior3_fresh_max": max(fresh[k] for k in ranked[:K_TOP]),
                MAIN_SH: fresh[pick_sh],
                MAIN_SH + "_fresh_max": max(fresh[k] for k in path[-1]),
                "random1": float(np.mean(list(fresh.values())))}
        same = {k: float(traj[k][b]) for k in keys}
        arms_same = {"oracle": max(same.values()), "prior1": same[ranked[0]],
                     "prior3": same[pick_p3], MAIN_SH: same[pick_sh]}
        out.append({"fold": target, "slice": s, "n_keys": len(keys), "arms": arms,
                    "regret": {k: oracle - v for k, v in arms.items()},
                    "regret_same_trajectory": {k: arms_same["oracle"] - v for k, v in arms_same.items()}})
    return out


# ---------------------------------------------------------------- 汇总
def mean_regret(rows: Sequence[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for a, v in r["regret"].items():
            acc[a].append(float(v))
    return {a: float(np.mean(v)) for a, v in sorted(acc.items())}


def median_regret(rows: Sequence[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for a, v in r["regret"].items():
            acc[a].append(float(v))
    return {a: float(np.median(v)) for a, v in sorted(acc.items())}


def strata(rows: Sequence[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    key = (lambda kv: (0, float(kv[0])) if kv[0].isdigit() else (1, kv[0]))
    return {k: {"n": len(v), "mean_regret": mean_regret(v), "median_regret": median_regret(v)}
            for k, v in sorted(groups.items(), key=key)}


def cost_summary(rows: Sequence[dict]) -> dict[str, dict]:
    """逐预算：各臂 iters / GPU 秒总和，及相对目录直搜的倍率。"""
    out = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["budget"])].append(r)
    groups["pooled"] = list(rows)
    for k, rs in groups.items():
        iters: dict[str, float] = defaultdict(float)
        secs: dict[str, float] = defaultdict(float)
        for r in rs:
            for a, v in r["cost_iters"].items():
                iters[a] += float(v)
            for a, v in (r.get("cost_seconds") or {}).items():
                secs[a] += float(v)
        direct_s = secs.get("catalog_direct", 0.0)
        direct_i = iters.get("catalog_direct", 0.0)
        out[k] = {"n": len(rs),
                  "iters": dict(sorted(iters.items())),
                  "seconds": dict(sorted(secs.items())),
                  "speedup_vs_direct_iters": {a: (direct_i / v if v else None) for a, v in sorted(iters.items())},
                  "speedup_vs_direct_seconds": {a: (direct_s / v if v else None) for a, v in sorted(secs.items())}}
    return out


def avoid_worst(rows: Sequence[dict]) -> dict[str, dict]:
    acc: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for a, h in r["bottom_quartile_hit"].items():
            acc[a].append(bool(h))
    return {a: {"hits": int(sum(v)), "n": len(v)} for a, v in sorted(acc.items())}


def predictions(rows: Sequence[dict], by_budget: Mapping[str, dict]) -> dict:
    """PROTOCOL §5 P1–P5 对账。"""
    m290 = by_budget["290"]["mean_regret"]
    pooled = mean_regret(rows)
    p = {}
    v = m290[MAIN_SH]
    p["P1"] = {"claim": "SH6-3-3_prior @290 ∈ [0.010, 0.030]", "value": v, "hit": bool(0.010 <= v <= 0.030)}
    d = m290["SH6-3-3_random"] - m290[MAIN_SH]
    p["P2"] = {"claim": "SH6-3-3_random − SH6-3-3_prior ≥ 0.010", "value": d, "hit": bool(d >= 0.010)}
    cost = cost_summary(rows)["290"]["iters"]
    ratio = cost["all20@60->top3"] / cost[MAIN_SH]
    dd = abs(m290["all20@60->top3"] - m290[MAIN_SH])
    p["P3"] = {"claim": "all20@60->top3 与 SH6-3-3_prior 同带 (|Δ|<0.010) 且成本 ≥1.8×",
               "delta": dd, "cost_ratio": ratio, "hit": bool(dd < 0.010 and ratio >= 1.8)}
    d290 = m290["prior1"] - m290["ggs"]
    d60 = by_budget["60"]["mean_regret"]["prior1"] - by_budget["60"]["mean_regret"]["ggs"]
    d120 = by_budget["120"]["mean_regret"]["prior1"] - by_budget["120"]["mean_regret"]["ggs"]
    p["P4"] = {"claim": "prior1 @290 与 GGS 持平 (|Δ|<0.01)，@60/@120 优于 GGS",
               "delta_290": d290, "delta_60": d60, "delta_120": d120,
               "hit": bool(abs(d290) < 0.01 and d60 < 0 and d120 < 0)}
    r = m290["SH6-3-1_prior"] / m290[MAIN_SH] if m290[MAIN_SH] > 0 else None
    p["P5"] = {"claim": "SH6-3-1_prior regret ≥ 2× SH6-3-3_prior", "ratio": r,
               "hit": bool(r is not None and r >= 2.0)}
    p["_pooled_prior1"] = pooled["prior1"]
    return p


def markdown_tables(analysis: dict) -> str:
    arms_zero = ["oracle", "prior1", "prior3", "gbdt1", "gbdt3", "hybrid1", "hybrid3", "tclsc1", "tclsc3",
                 "clsb1", "clsonly1", "ggs", "random1", "random3", "bottom"]
    arms_sh = [f"{sh_label(c)}_{w}" for c in SH_CONFIGS for w in (*SH_WARMS, "random")] + \
              ["all20@60->top1", "all20@60->top3"]
    lines = ["## 零样本臂：mean regret（行 = 分层，列 = 臂）", ""]
    cols = arms_zero
    lines.append("| 分层 | n | " + " | ".join(cols) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(cols)) + "|")
    for name, strat in (("pooled", {"n": analysis["n_instances"], "mean_regret": analysis["mean_regret"]}),
                        *analysis["by_budget"].items(), *analysis["by_dataset"].items()):
        mr = strat["mean_regret"]
        lines.append(f"| {name} | {strat['n']} | " + " | ".join(
            f"{mr[a]:.3f}" if a in mr else "—" for a in cols) + " |")
    lines += ["", "## 多保真臂 @290：mean / median regret、成本（iters；GPU 秒）、垫底四分位命中", ""]
    m = analysis["by_budget"]["290"]["mean_regret"]
    md = analysis["by_budget"]["290"]["median_regret"]
    c = analysis["cost"]["290"]
    aw = analysis["avoid_worst"]
    lines.append("| 臂 | mean | median | iters/实例 | GPU 秒/实例 | ×直搜(秒) | 垫底命中 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    n290 = analysis["by_budget"]["290"]["n"]
    for a in ["prior1", "prior3", "tclsc3", "ggs", "random1", "random3"] + arms_sh:
        if a not in m:
            continue
        it = c["iters"].get(a)
        sc = c["seconds"].get(a)
        sp = c["speedup_vs_direct_seconds"].get(a)
        hit = aw.get(a)
        lines.append(f"| {a} | {m[a]:.3f} | {md[a]:.3f} | "
                     f"{(it / n290) if it else float('nan'):.0f} | {(sc / n290) if sc else float('nan'):.0f} | "
                     f"{sp if sp else float('nan'):.1f} | {(str(hit['hits']) + '/' + str(hit['n'])) if hit else '—'} |")
    lines += ["", "## 门（PROTOCOL §4）", "", "| 门 | 臂 | n | mean_a | mean_b | p | 过 |", "|---|---|---:|---:|---:|---:|---|"]
    for k in ("G1", "G2", "G3", "G4", "G5"):
        g = analysis["gates"][k]
        p_str = "—" if g["p"] is None else f"{g['p']:.2e}"
        tag = "（描述性）" if g.get("descriptive") else ""
        lines.append(f"| {k} | {g['arm_a']} vs {g['arm_b']} | {g['n']} | {g['mean_a']:.3f} | {g['mean_b']:.3f} | "
                     f"{p_str} | {'✓' if g['pass'] else '✗'}{tag} |")
    lines += ["", f"读数码：**{analysis['verdict']}**", ""]
    lines += ["## 预测对账（PROTOCOL §5）", "", "| 预测 | 内容 | 读数 | 命中 |", "|---|---|---|---|"]
    for k in ("P1", "P2", "P3", "P4", "P5"):
        p = analysis["predictions"][k]
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in p.items() if kk not in ("claim", "hit")}
        lines.append(f"| {k} | {p['claim']} | {detail} | {'✓' if p['hit'] else '✗'} |")
    return "\n".join(lines) + "\n"


def cmd_analyze(mx3: Path, outdir: Path, folds: Sequence[str], *, seed: int = 0) -> dict:
    cat, ggs, probe, rankings = load_inputs(mx3)
    cat_z = zscore_within(cat)
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)                      # scalar 32 维（PROTOCOL §2）
    rows: list[dict] = []
    fresh: list[dict] = []
    fold_meta = {}
    for target in folds:
        inst, meta = fold_instances(target, cat_z, ggs, rankings, encoder, seed=seed)
        rows += inst
        fresh += fresh_instances(target, cat_z, probe, meta["prior"])
        fold_meta[target] = {"prior_top5": meta["prior_top"],
                             "prior_scores": {k: round(v, 4) for k, v in meta["prior"].items()}}
        print(f"[fold {target}] {len(inst)} 实例；先验 top-5 {meta['prior_top']}", flush=True)
    by_budget = strata(rows, "budget")
    by_dataset = strata(rows, "dataset")
    gates = run_gates(rows, fresh)
    analysis = {
        "protocol": "results/mx4_prior_sh/PROTOCOL.md",
        "source": str(mx3.relative_to(ROOT)) if mx3.is_relative_to(ROOT) else str(mx3),
        "folds": list(folds), "n_instances": len(rows), "n_fresh": len(fresh),
        "mean_regret": mean_regret(rows), "median_regret": median_regret(rows),
        "by_budget": by_budget, "by_dataset": by_dataset,
        "gates": gates, "verdict": verdict_code(gates),
        "fresh": {"rows": fresh,
                  "mean_regret": mean_regret(fresh),
                  "mean_regret_same_trajectory": mean_regret(
                      [{"regret": r["regret_same_trajectory"]} for r in fresh])},
        "cost": cost_summary(rows), "avoid_worst": avoid_worst(rows),
        "fold_meta": fold_meta, "per_instance": rows,
    }
    analysis["predictions"] = predictions(rows, by_budget)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(analysis), encoding="utf-8")
    print(f"verdict {analysis['verdict']}；pooled prior1 {analysis['mean_regret']['prior1']:.4f}；"
          f"@290 {MAIN_SH} {by_budget['290']['mean_regret'][MAIN_SH]:.4f}")
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--mx3", default=str(MX3))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.analyze:
        cmd_analyze(Path(args.mx3), Path(args.outdir), args.folds, seed=args.seed)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
