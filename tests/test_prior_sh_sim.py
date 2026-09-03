"""MX-4 WP1：先验 + 逐次减半模拟（scripts/36_prior_sh_sim.py）测试——先于实现（铁律 5）。

协议：results/mx4_prior_sh/PROTOCOL.md。覆盖纯函数：
- within-(task, iters) z 分数；先验 = 训练侧 z 均值；排名并列按 key 升序、未见 CLS 垫底；
- 逐次减半：每级按该截点真 R' 留前 k、成本 = Σ k_i·(b_i − b_{i−1})、k 超过候选数时截断；
- 随机起点期望：同 seed 逐位复现、介于 oracle 与 random@1 之间；
- 实例装配：臂/regret/成本字段齐全，tclsc 臂缺席 fail-fast；
- 门：复用 34 号的配对 Wilcoxon。
全部零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

g36 = importlib.import_module("scripts.36_prior_sh_sim")


def _rows():
    """两个切片 × 两个预算 × 3 个 CLS 的玩具缓存行。"""
    out = []
    for t in ("A_00", "A_01"):
        for b, vals in ((60, (0.5, 0.7, 0.9)), (120, (0.6, 0.8, 0.7))):
            for k, r in zip(("x", "y", "z"), vals):
                out.append({"task_id": t, "dataset": "A", "iters": b, "cls_key": k,
                            "origin": "shared", "r_prime": r, "seconds": b / 10})
    return out


# ---------------------------------------------------------------- z / 先验 / 排名
def test_zscore_within_task_and_budget():
    rows = g36.zscore_within(_rows())
    for t in ("A_00", "A_01"):
        for b in (60, 120):
            z = np.array([r["z"] for r in rows if r["task_id"] == t and r["iters"] == b])
            assert z.mean() == pytest.approx(0.0, abs=1e-9)
            assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_prior_scores_is_mean_z_per_key():
    rows = g36.zscore_within(_rows())
    prior = g36.prior_scores(rows)
    assert set(prior) == {"x", "y", "z"}
    zx = [r["z"] for r in rows if r["cls_key"] == "x"]
    assert prior["x"] == pytest.approx(float(np.mean(zx)))
    assert prior["x"] < prior["y"]          # x 在两预算都垫底


def test_rank_by_desc_tiebreak_key_unseen_last():
    ranked = g36.rank_by({"b": 1.0, "a": 1.0, "c": 2.0}, keys=["a", "b", "c", "d"])
    assert ranked == ["c", "a", "b", "d"]   # 同分按 key 升序；d 未见 → 垫底


# ---------------------------------------------------------------- 逐次减半
TRAJ = {
    "p": {60: 0.90, 120: 0.60, 290: 0.50},   # 早强晚崩
    "q": {60: 0.80, 120: 0.85, 290: 0.95},   # 晚熟冠军
    "r": {60: 0.70, 120: 0.80, 290: 0.90},
    "s": {60: 0.10, 120: 0.20, 290: 0.30},
    "t": {60: 0.75, 120: 0.70, 290: 0.60},
}


def test_successive_halving_keeps_by_checkpoint_truth():
    ranked = ["p", "q", "r", "s", "t"]
    final, cost = g36.successive_halving(ranked, TRAJ, ks=(4, 2, 1), budgets=(60, 120, 290))
    # 级 1：前 4 = p,q,r,s；@60 留前 2 = p(0.90), q(0.80)；@120 留前 1 = q(0.85 > 0.60)
    assert final == ["q"]
    assert cost == 4 * 60 + 2 * (120 - 60) + 1 * (290 - 120)


def test_successive_halving_returns_k_last_candidates():
    final, _ = g36.successive_halving(["p", "q", "r", "s", "t"], TRAJ, ks=(5, 3, 3),
                                      budgets=(60, 120, 290))
    # @60 前 3 = p(.90) q(.80) t(.75)；@120 前 3 = q(.85) t(.70) p(.60) → 三条都到 290
    assert final == ["q", "t", "p"]           # 按最后一次筛选截点的 R' 降序


def test_successive_halving_clips_k_to_candidates():
    final, cost = g36.successive_halving(["p", "q"], TRAJ, ks=(6, 3, 1), budgets=(60, 120, 290))
    assert final == ["q"]                      # @60 两条都留；@120 q(.85) > p(.60)
    assert cost == 2 * 60 + 2 * 60 + 1 * 170   # k 截断到 2/2/1


def test_successive_halving_two_stage_for_final_120():
    final, cost = g36.successive_halving(["p", "q", "r"], TRAJ, ks=(3, 1), budgets=(60, 120))
    assert final == ["p"]                      # @60 p 最强 → 只剩它到 120
    assert cost == 3 * 60 + 1 * 60


def test_sh_pick_takes_best_at_final_budget_tiebreak_key():
    traj = {"a": {290: 0.9}, "b": {290: 0.9}, "c": {290: 0.1}}
    assert g36.sh_pick(["c", "b", "a"], traj, final_budget=290) == "a"


def test_expected_random_sh_is_deterministic_and_bounded():
    keys = list(TRAJ)
    args = dict(ks=(3, 2, 1), budgets=(60, 120, 290), n_draws=10)
    a = g36.expected_random_sh(keys, TRAJ, seed=5, **args)
    b = g36.expected_random_sh(keys, TRAJ, seed=5, **args)
    assert a == b
    oracle = max(v[290] for v in TRAJ.values())
    random1 = float(np.mean([v[290] for v in TRAJ.values()]))
    assert random1 <= a <= oracle


# ---------------------------------------------------------------- 实例装配
def _instance_inputs():
    traj = dict(TRAJ)
    keys = list(traj)
    rank = {"prior": ["q", "r", "p", "t", "s"], "gbdt": ["r", "q", "p", "s", "t"],
            "hybrid": ["q", "p", "r", "t", "s"], "tclsc": ["p", "q", "r", "s", "t"],
            "clsb": ["t", "q", "p", "r", "s"], "clsonly": ["s", "q", "p", "r", "t"]}
    return keys, traj, rank


def test_instance_row_has_all_arms_and_costs():
    keys, traj, rank = _instance_inputs()
    row = g36.instance_row(keys, traj, rank, ggs_r=0.85, final_budget=290,
                           budgets=(60, 120, 290), seconds=None, random_seed=0)
    arms = row["arms"]
    for name in ("oracle", "prior1", "prior3", "gbdt1", "gbdt3", "hybrid1", "hybrid3",
                 "tclsc1", "tclsc3", "clsb1", "clsonly1", "ggs", "random1", "random3",
                 "bottom", "SH6-3-3_prior", "SH6-3-3_random", "SH6-3-1_prior",
                 "SH10-4-3_gbdt", "all20@60->top3", "all20@60->top1"):
        assert name in arms, name
        assert name in row["regret"], name
    assert row["arms"]["oracle"] == pytest.approx(0.95)
    assert row["arms"]["prior1"] == pytest.approx(0.95)    # q 是先验第一，@290 = 0.95
    assert row["arms"]["tclsc1"] == pytest.approx(0.50)    # p 早强晚崩
    assert row["arms"]["SH6-3-3_prior"] == pytest.approx(0.95)
    assert row["regret"]["ggs"] == pytest.approx(0.10)
    assert row["cost_iters"]["prior1"] == 290
    assert row["cost_iters"]["prior3"] == 3 * 290
    assert row["cost_iters"]["SH6-3-3_prior"] == 5 * 60 + 3 * 60 + 3 * 170   # k0 截断到 5 候选
    assert row["cost_iters"]["catalog_direct"] == 5 * 290


def test_instance_row_final_120_has_two_stage_sh_only():
    keys, traj, rank = _instance_inputs()
    row = g36.instance_row(keys, traj, rank, ggs_r=0.85, final_budget=120,
                           budgets=(60, 120), seconds=None, random_seed=0)
    assert "SH6-3_prior" in row["arms"] and "SH6-3-3_prior" not in row["arms"]
    assert "all20@60->top3" not in row["arms"]
    assert row["cost_iters"]["SH6-3_prior"] == 5 * 60 + 3 * 60


def test_instance_row_final_60_has_no_sh():
    keys, traj, rank = _instance_inputs()
    row = g36.instance_row(keys, traj, rank, ggs_r=0.85, final_budget=60,
                           budgets=(60,), seconds=None, random_seed=0)
    assert not any(a.startswith("SH") for a in row["arms"])
    assert row["arms"]["prior1"] == pytest.approx(0.80)


def test_instance_row_requires_tclsc_ranking():
    keys, traj, rank = _instance_inputs()
    rank = {k: v for k, v in rank.items() if k != "tclsc"}
    with pytest.raises(KeyError):
        g36.instance_row(keys, traj, rank, ggs_r=0.85, final_budget=290,
                         budgets=(60, 120, 290), seconds=None, random_seed=0)


def test_instance_row_gpu_seconds_from_cumulative_checkpoints():
    keys, traj, rank = _instance_inputs()
    seconds = {k: {60: 1.0, 120: 2.5, 290: 6.0} for k in keys}   # 累计墙钟
    row = g36.instance_row(keys, traj, rank, ggs_r=0.85, final_budget=290,
                           budgets=(60, 120, 290), seconds=seconds, random_seed=0)
    # SH 5-3-3：5 条到 60（5×1.0）+ 3 条 60→120（3×1.5）+ 3 条 120→290（3×3.5）
    assert row["cost_seconds"]["SH6-3-3_prior"] == pytest.approx(5 * 1.0 + 3 * 1.5 + 3 * 3.5)
    assert row["cost_seconds"]["prior1"] == pytest.approx(6.0)
    assert row["cost_seconds"]["catalog_direct"] == pytest.approx(5 * 6.0)


# ---------------------------------------------------------------- 门 / 读数码
def test_gates_and_verdict_codes():
    rows = []
    for i in range(12):
        good, bad = 0.01 * i, 0.03 + 0.01 * i
        rows.append({"budget": 290, "regret": {
            "SH6-3-3_prior": good, "prior3": bad, "SH6-3-3_random": bad, "tclsc3": bad,
            "prior1": bad, "ggs": bad + 0.01}})
    g = g36.run_gates(rows, fresh_rows=[{"regret": {"SH6-3-3_prior": 0.0, "prior3": 0.01}}])
    assert g["G1"]["pass"] and g["G2"]["pass"] and g["G3"]["pass"] and g["G4"]["pass"]
    assert g["G5"]["pass"]
    assert g36.verdict_code(g) == "SH-WIN"
    g["G2"]["pass"] = False
    assert g36.verdict_code(g) == "SH-PARTIAL"
    g["G1"]["pass"] = False
    assert g36.verdict_code(g) == "SH-DEAD"


# ---------------------------------------------------------------- G4：新鲜种子真值
def test_fresh_instances_both_arms_commit_with_canonical_then_score_fresh():
    """选择只能用 canonical 轨迹（部署可得），真值换成探针 rep 均值——两臂同规则。"""
    target = "A"
    cat_rows = []
    probe = []
    traj = {"a": {60: 0.9, 120: 0.9, 290: 0.90}, "b": {60: 0.5, 120: 0.8, 290: 0.95},
            "c": {60: 0.1, 120: 0.1, 290: 0.10}}
    fresh = {"a": (0.7, 0.7), "b": (0.6, 0.6), "c": (0.1, 0.1)}     # b 的 canonical 0.95 是运气
    for k, tr in traj.items():
        for b, r in tr.items():
            cat_rows.append({"task_id": "A_00", "dataset": target, "iters": b, "cls_key": k,
                             "origin": "shared", "r_prime": r, "seconds": 1.0, "z": 0.0})
        for rep, r in zip((1, 2), fresh[k]):
            probe.append({"task_id": "A_00", "dataset": target, "iters": 290, "cls_key": k,
                          "seed": rep, "r_prime": r})
    prior = {"a": 1.0, "b": 0.5, "c": 0.0}
    rows = g36.fresh_instances(target, cat_rows, probe, prior)
    assert len(rows) == 1
    arms = rows[0]["arms"]
    # canonical 下 prior top-3 里最好的是 b(0.95)；换新鲜真值 b 只有 0.6，a 才是 0.7
    assert arms["prior3"] == pytest.approx(0.6)             # 按 canonical 选 b，新鲜真值 0.6
    assert arms["prior3_fresh_max"] == pytest.approx(0.7)   # 用新鲜真值挑 = 乐观上界
    assert arms["SH6-3-3_prior"] == pytest.approx(0.6)      # SH 幸存 a,b,(c) → canonical 选 b
    assert rows[0]["regret"]["prior3"] == pytest.approx(0.1)
