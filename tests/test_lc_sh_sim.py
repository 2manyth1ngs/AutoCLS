"""MX-4 WP5：外推器驱动 SH 淘汰模拟（scripts/38_lc_sh_sim.py）测试——先于实现（铁律 5）。

协议：results/mx4_lc_extrapolator/PROTOCOL.md。覆盖纯函数：
- 可换评分器的逐次减半 sh_path_scored：评分器 = 当前分时与 36 号 sh_path 逐位相同；
- 六种淘汰规则评分器 cur / slope / blend / lc（lc 用桩模型，按 (观测截点, b_final) 选模型）；
- 实例装配：参考臂 oracle / prior3 / prior6_oracle，SH 档位 × 规则臂，成本口径同 WP1；
- λ 训练侧选择：先验有信息 → 大 λ，当前分有信息 → 小 λ，并列取小；
- 机制指标：top-k 内 Spearman 与 δ 存活对配对准确率；
- 门 / 读数码：C0 一致性、L1–L4、LC-WIN / LC-CHEAP / LC-DEAD。
全部零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

g36 = importlib.import_module("scripts.36_prior_sh_sim")
g38 = importlib.import_module("scripts.38_lc_sh_sim")

TRAJ = {
    "p": {60: 0.90, 120: 0.60, 290: 0.50},   # 早强晚崩
    "q": {60: 0.80, 120: 0.85, 290: 0.95},   # 晚熟冠军
    "r": {60: 0.70, 120: 0.80, 290: 0.90},
    "s": {60: 0.10, 120: 0.20, 290: 0.30},
    "t": {60: 0.75, 120: 0.70, 290: 0.60},
}
B3 = (60, 120, 290)


class _StubModel:
    """predict = 最后观测分 + cls_X[:, 0]（偏置由测试给定，模拟"学到的增益"）。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def predict(self, observed, *, cls_X=None, extra=None):
        observed = np.asarray(observed, dtype=float)
        self.calls.append((observed.shape[1], None if extra is None else np.asarray(extra).shape))
        return observed[:, -1] + np.asarray(cls_X, dtype=float)[:, 0]


# ---------------------------------------------------------------- 可换评分器的 SH
@pytest.mark.parametrize("ks", [(4, 2, 1), (5, 3, 3), (6, 3, 1), (3, 1)])
def test_sh_path_scored_with_current_score_equals_g36_sh_path(ks):
    budgets = B3[:len(ks)]
    ranked = ["p", "q", "r", "s", "t"]
    scorer = g38.scorer_current(TRAJ)
    assert g38.sh_path_scored(ranked, ks, budgets, scorer) == g36.sh_path(ranked, TRAJ, ks, budgets)


def test_sh_path_scored_passes_observed_budgets_and_final_to_scorer():
    seen = []

    def scorer(survivors, observed_budgets, b_final):
        seen.append((tuple(survivors), tuple(observed_budgets), b_final))
        return {k: -i for i, k in enumerate(survivors)}        # 保持顺序

    path = g38.sh_path_scored(["p", "q", "r", "s"], (4, 2, 1), B3, scorer)
    assert path == [["p", "q", "r", "s"], ["p", "q"], ["p"]]
    assert seen == [(("p", "q", "r", "s"), (60,), 290), (("p", "q"), (60, 120), 290)]


def test_sh_path_scored_validates_lengths():
    with pytest.raises(ValueError):
        g38.sh_path_scored(["p"], (2, 1), (60,), g38.scorer_current(TRAJ))


# ---------------------------------------------------------------- 评分器
def test_scorer_current_is_last_observed_truth():
    sc = g38.scorer_current(TRAJ)(["p", "q"], (60, 120), 290)
    assert sc == {"p": 0.60, "q": 0.85}


def test_scorer_slope_one_point_equals_current_two_points_log_linear():
    sl = g38.scorer_slope(TRAJ)
    assert sl(["p", "q"], (60,), 290) == {"p": 0.90, "q": 0.80}
    k = np.log(290 / 120) / np.log(120 / 60)
    two = sl(["p", "q"], (60, 120), 290)
    assert two["q"] == pytest.approx(0.85 + 0.05 * k)
    assert two["p"] == pytest.approx(0.60 - 0.30 * k)


def test_scorer_blend_lambda_zero_follows_current_large_lambda_follows_prior():
    prior = {"p": 0.0, "q": 2.0, "r": 1.0}
    surv = ["p", "q", "r"]
    s0 = g38.scorer_blend(TRAJ, prior, 0.0)(surv, (60,), 290)
    assert sorted(surv, key=lambda k: -s0[k]) == ["p", "q", "r"]
    s4 = g38.scorer_blend(TRAJ, prior, 4.0)(surv, (60,), 290)
    assert sorted(surv, key=lambda k: -s4[k]) == ["q", "r", "p"]
    # z 在幸存者内算：均值 0
    assert sum(s0.values()) == pytest.approx(0.0)


def test_scorer_blend_missing_prior_counts_as_zero_and_constant_scores_give_zero_z():
    flat = {"a": {60: 0.5}, "b": {60: 0.5}}
    sc = g38.scorer_blend(flat, {"a": 1.0}, 1.0)(["a", "b"], (60,), 290)
    assert sc == {"a": 1.0, "b": 0.0}


def test_scorer_lc_selects_model_by_observed_budgets_and_final():
    m1, m2 = _StubModel(), _StubModel()
    models = {((60,), 290): m1, ((60, 120), 290): m2}
    cls_vec = {"p": np.array([0.5]), "q": np.array([0.2]), "r": np.array([0.0])}
    extra = np.array([2.0, 1.5, 3.0])
    sc = g38.scorer_lc(models, TRAJ, cls_vec, extra)
    s1 = sc(["p", "q", "r"], (60,), 290)
    assert s1 == {"p": pytest.approx(1.4), "q": pytest.approx(1.0), "r": pytest.approx(0.7)}
    assert m1.calls == [(1, (3, 3))] and m2.calls == []
    s2 = sc(["p", "q"], (60, 120), 290)
    assert s2 == {"p": pytest.approx(1.1), "q": pytest.approx(1.05)}
    assert m2.calls == [(2, (2, 3))]


def test_scorer_lc_missing_model_fails_fast():
    sc = g38.scorer_lc({}, TRAJ, {"p": np.zeros(1)}, np.zeros(3))
    with pytest.raises(KeyError):
        sc(["p"], (60,), 290)


# ---------------------------------------------------------------- 实例装配
def _scorers():
    m1, m2 = _StubModel(), _StubModel()
    models = {((60,), 290): m1, ((60, 120), 290): m2, ((60,), 120): m1}
    cls_vec = {"p": np.array([0.5]), "q": np.array([0.2]), "r": np.array([0.0]),
               "s": np.array([0.0]), "t": np.array([-0.1])}
    return {"cur": g38.scorer_current(TRAJ),
            "lc_hgb": g38.scorer_lc(models, TRAJ, cls_vec, np.zeros(3))}


def test_instance_row_lc_reference_arms_and_rule_arms():
    prior_rank = ["q", "r", "p", "t", "s"]
    row = g38.instance_row_lc(list(TRAJ), TRAJ, prior_rank, _scorers(),
                              final_budget=290, budgets=B3, seconds=None)
    arms = row["arms"]
    assert arms["oracle"] == pytest.approx(0.95)
    assert arms["prior1"] == pytest.approx(0.95)
    assert arms["prior3"] == pytest.approx(0.95)
    assert arms["prior6_oracle"] == pytest.approx(0.95)       # top-6 截断到 5 条
    for cfg in g38.SH_CONFIGS:
        for rule in ("cur", "lc_hgb"):
            name = f"{g36.sh_label(cfg)}_{rule}"
            assert name in arms and name in row["regret"] and name in row["cost_iters"], name
    assert "all5@60->top3_cur" in arms and "all5@60->top1_lc_hgb" in arms
    # cur 与 36 号逐位同：SH6-3-3_prior 在 WP1 里选 q
    assert arms["SH6-3-3_cur"] == pytest.approx(0.95)
    # SH4-2-1：cur 留 p,q → @120 q 胜 → 0.95；lc 桩偏爱 p（+0.5）→ 留 p,q → @120 p 1.1 > q 1.05 → p 0.50
    assert arms["SH4-2-1_cur"] == pytest.approx(0.95)
    assert arms["SH4-2-1_lc_hgb"] == pytest.approx(0.50)
    assert row["regret"]["SH4-2-1_lc_hgb"] == pytest.approx(0.45)
    assert row["picks"]["SH4-2-1_lc_hgb"] == "p"


def test_instance_row_lc_costs_match_wp1_formula_and_do_not_depend_on_rule():
    row = g38.instance_row_lc(list(TRAJ), TRAJ, ["q", "r", "p", "t", "s"], _scorers(),
                              final_budget=290, budgets=B3,
                              seconds={k: {60: 1.0, 120: 2.5, 290: 6.0} for k in TRAJ})
    assert row["cost_iters"]["SH6-3-3_cur"] == 5 * 60 + 3 * 60 + 3 * 170      # k0 截断到 5
    assert row["cost_iters"]["SH6-3-3_lc_hgb"] == row["cost_iters"]["SH6-3-3_cur"]
    assert row["cost_iters"]["SH4-2-2_cur"] == 4 * 60 + 2 * 60 + 2 * 170
    assert row["cost_seconds"]["SH6-3-3_cur"] == pytest.approx(5 * 1.0 + 3 * 1.5 + 3 * 3.5)
    assert row["cost_iters"]["prior3"] == 3 * 290 and row["cost_iters"]["catalog_direct"] == 5 * 290


def test_instance_row_lc_records_stage1_scores_of_prior_top6_per_rule():
    row = g38.instance_row_lc(list(TRAJ), TRAJ, ["q", "r", "p", "t", "s"], _scorers(),
                              final_budget=290, budgets=B3, seconds=None)
    s1 = row["stage1_scores"]
    assert s1["cur"] == {k: TRAJ[k][60] for k in TRAJ}
    assert s1["lc_hgb"]["p"] == pytest.approx(1.4)
    assert row["top6"] == ["q", "r", "p", "t", "s"]


def test_instance_row_lc_two_stage_final_120():
    row = g38.instance_row_lc(list(TRAJ), TRAJ, ["q", "r", "p", "t", "s"], _scorers(),
                              final_budget=120, budgets=(60, 120), seconds=None)
    assert "SH6-3_cur" in row["arms"] and "SH6-3-3_cur" not in row["arms"]
    assert row["cost_iters"]["SH6-3_cur"] == 5 * 60 + 3 * 60
    assert "stage1_scores" in row


# ---------------------------------------------------------------- λ 训练侧选择
def _lambda_slices(prior_informative: bool):
    keys = ["a", "b", "c", "d", "e", "f"]
    final = dict(zip(keys, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    early = dict(zip(keys, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])) if prior_informative else final
    traj = {k: {60: early[k], 120: early[k], 290: final[k]} for k in keys}
    prior = (dict(zip(keys, [3, 2, 1, 0, -1, -2])) if prior_informative
             else dict(zip(keys, [-2, -1, 0, 1, 2, 3])))
    return [traj, traj], prior


def test_tune_lambda_prefers_prior_when_prior_informative_ties_take_smallest():
    trajs, prior = _lambda_slices(prior_informative=True)
    lam = g38.tune_lambda(trajs, prior, grid=(0.25, 0.5, 1.0, 2.0, 4.0), ks=(6, 3, 3), budgets=B3)
    assert lam == 1.0            # λ ∈ {1,2,4} 都 regret 0，并列取小


def test_tune_lambda_prefers_current_when_early_score_informative():
    trajs, prior = _lambda_slices(prior_informative=False)
    lam = g38.tune_lambda(trajs, prior, grid=(0.25, 0.5, 1.0, 2.0, 4.0), ks=(6, 3, 3), budgets=B3)
    assert lam == 0.25


# ---------------------------------------------------------------- 机制指标
def test_mechanism_metrics_spearman_and_delta_surviving_pairs():
    keys = ["a", "b", "c", "d", "e", "f"]
    truth = dict(zip(keys, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]))
    scores = {"perfect": dict(truth), "reversed": {k: -v for k, v in truth.items()}}
    m = g38.mechanism_metrics(keys, truth, scores, delta=0.15)
    assert m["perfect"]["spearman"] == pytest.approx(1.0)
    assert m["reversed"]["spearman"] == pytest.approx(-1.0)
    assert m["perfect"]["pair_total"] == 10           # |Δ| ≥ 0.15 的无序对
    assert m["perfect"]["pair_correct"] == 10
    assert m["reversed"]["pair_correct"] == 0


# ---------------------------------------------------------------- 门 / 读数码
def _rows(n=12, lc_better=True, cheap_ok=True):
    rows = []
    for i in range(n):
        cur = 0.02 + 0.01 * (i % 3)
        lc = cur - 0.01 if lc_better else cur + 0.01
        cheap = cur if cheap_ok else cur + 0.03
        rows.append({"budget": 290, "slice": f"A_{i:02d}", "regret": {
            "SH6-3-3_cur": cur, "SH6-3-3_lc_hgb": lc,
            "SH4-2-2_lc_hgb": cheap, "SH6-3-1_lc_hgb": cheap + 0.05}})
    return rows


MECH = {"cur": {"spearman_median": 0.3, "pair_acc": 0.6},
        "lc_hgb": {"spearman_median": 0.5, "pair_acc": 0.7}}


def test_gates_win_when_l1_and_l3():
    g = g38.run_gates(_rows(lc_better=True),
                      fresh_rows=[{"regret": {"SH6-3-3_cur": 0.02, "SH6-3-3_lc_hgb": 0.01}}],
                      mech=MECH, c0_pass=True)
    assert g["C0"]["pass"] and g["L1"]["pass"] and g["L3"]["pass"] and g["L4"]["pass"]
    assert g38.verdict_code(g) == "LC-WIN"


def test_gates_cheap_when_not_l1_but_l2_and_l3():
    g = g38.run_gates(_rows(lc_better=False, cheap_ok=True), fresh_rows=[], mech=MECH, c0_pass=True)
    assert not g["L1"]["pass"] and g["L2"]["pass"]
    assert g["L2"]["winner"] == "SH4-2-2_lc_hgb"
    assert g38.verdict_code(g) == "LC-CHEAP"


def test_gates_dead_when_neither_and_l3_alone_is_not_enough():
    g = g38.run_gates(_rows(lc_better=False, cheap_ok=False), fresh_rows=[], mech=MECH, c0_pass=True)
    assert not g["L1"]["pass"] and not g["L2"]["pass"] and g["L3"]["pass"]
    assert g38.verdict_code(g) == "LC-DEAD"


def test_verdict_invalid_when_c0_fails():
    g = g38.run_gates(_rows(), fresh_rows=[], mech=MECH, c0_pass=False)
    assert g38.verdict_code(g) == "INVALID"


def test_consistency_check_matches_wp1_per_instance():
    rows = [{"budget": 290, "slice": "A_00", "regret": {"SH6-3-3_cur": 0.01}},
            {"budget": 290, "slice": "A_01", "regret": {"SH6-3-3_cur": 0.00}}]
    wp1 = [{"budget": 290, "slice": "A_00", "regret": {"SH6-3-3_prior": 0.01}},
           {"budget": 290, "slice": "A_01", "regret": {"SH6-3-3_prior": 0.00}},
           {"budget": 120, "slice": "A_00", "regret": {"SH6-3-3_prior": 0.5}}]     # 其他预算忽略
    c0 = g38.consistency_check(rows, wp1)
    assert c0["pass"] and c0["n"] == 2 and c0["max_abs_diff"] == 0.0
    wp1[1]["regret"]["SH6-3-3_prior"] = 0.02
    assert not g38.consistency_check(rows, wp1)["pass"]
