"""Gate v2 判据核心的单测（铁律 5 先测后写；判据定义见 results/gate_v2/CRITERION.md）。

这里守的是三条一旦错掉就会**静默给出漂亮但错误结论**的性质：

1. **δ 过滤口径必须与真实造 label 逐字一致**——判据的全部意义就是"直接测下游
   拿到的东西"，口径一偏（比如把 `<δ` 写成 `<=δ`、或漏了非有限剔除）就又变成
   在测一个下游不存在的量。故直接与 `autocls.labeling.make_pairs` 对拍。
2. **共识方向必须只由 seed 1/2/3 给出**——canonical(seed 0) 一旦混进共识，label
   就在给自己投票，正确率被系统性抬高，而这个偏置从结果上完全看不出来。
3. **方向一律经 is_better**（铁律 8），不得出现 `r_a > r_b` 式硬编码。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.labeling import make_pairs  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402

CLF = TASK_CLASSIFICATION


def _load():
    spec = importlib.util.spec_from_file_location(
        "gate_v2_mod", ROOT / "scripts" / "gate_v2_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_v2_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load()


def _tm(**cls_seed_vals):
    """便捷构造 {task: {cls_key: {seed: R'}}}（单任务 't'）。"""
    return {"t": {k: dict(v) for k, v in cls_seed_vals.items()}}


# ---------------------------------------------------------------------------
# 1. 口径对齐：δ 过滤/候选集必须与 make_pairs 一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.0, 0.01, 0.02, 0.05, 0.2])
def test_delta_filter_matches_make_pairs(delta):
    """存活对数逐 δ 等于 make_pairs 的 n_kept（含非有限剔除与严格平局弃样）。"""
    canon = {"a": 0.90, "b": 0.88, "c": 0.60, "d": float("nan"), "e": 0.90}
    tm = {"t": {k: {0: v, 1: v, 2: v, 3: v} for k, v in canon.items()}}
    obs = G.pair_observations(tm, tuple(canon), task_type=CLF)
    stats_ours = G.yield_accuracy(obs, delta)
    _, stats_ref = make_pairs(canon, CLF, delta=delta)

    assert stats_ours["n_candidates"] == stats_ref["n_candidates"]
    assert stats_ours["n_kept"] == stats_ref["n_kept"]


def test_nonfinite_canonical_cls_is_dropped_entirely():
    tm = _tm(a={0: 0.9, 1: 0.9}, bad={0: float("nan"), 1: 0.5}, c={0: 0.5, 1: 0.5})
    obs = G.pair_observations(tm, ("a", "bad", "c"), task_type=CLF)
    assert {frozenset((o.a, o.b)) for o in obs} == {frozenset(("a", "c"))}


def test_label_direction_goes_through_is_better():
    """classification 大者优：R'_a>R'_b → y=1（a better）。"""
    tm = _tm(a={0: 0.9}, b={0: 0.5})
    (o,) = G.pair_observations(tm, ("a", "b"), task_type=CLF)
    assert (o.a, o.b) == ("a", "b") and o.label_a_better is True
    assert G.pair_observations(_tm(a={0: 0.5}, b={0: 0.9}), ("a", "b"),
                               task_type=CLF)[0].label_a_better is False


def test_forecasting_direction_inverted_via_is_better():
    """铁律 7/8：forecasting（MSE 小者优）方向必须与分类相反——证明经 is_better
    分发而非按分类语义蒙混（2026-08-13 起 forecasting 分支已实现，原 stub 断言更新）。"""
    (o,) = G.pair_observations(_tm(a={0: 0.9}, b={0: 0.5}), ("a", "b"),
                               task_type="forecasting")
    assert o.label_a_better is False  # MSE 0.9 > 0.5 → a 更差


def test_unknown_task_type_not_silently_assumed():
    """未登记的任务类型必须炸，不得按任何既有语义蒙混。"""
    with pytest.raises((NotImplementedError, ValueError)):
        G.pair_observations(_tm(a={0: 0.9}, b={0: 0.5}), ("a", "b"),
                            task_type="segmentation")


# ---------------------------------------------------------------------------
# 2. 共识方向：只由 seed 1/2/3 决定，canonical 不参与
# ---------------------------------------------------------------------------


def test_consensus_excludes_canonical_seed():
    """seed0 说 a 好，1/2/3 一致说 b 好 → 该 label 判错（正确率 0）。

    若实现把 seed0 混进共识（4 个里 1 个反对），均值仍可能翻向 a，正确率变 1.0 ——
    这正是本测试要挡住的静默偏置。
    """
    tm = _tm(a={0: 0.90, 1: 0.50, 2: 0.50, 3: 0.50},
             b={0: 0.50, 1: 0.90, 2: 0.90, 3: 0.90})
    (o,) = G.pair_observations(tm, ("a", "b"), task_type=CLF)
    assert o.label_a_better is True and o.cons_a_better is False
    st = G.yield_accuracy([o], 0.01)
    assert st["n_kept"] == 1 and st["n_agree"] == 0 and st["accuracy"] == 0.0


def test_consensus_uses_mean_over_seeds_not_majority():
    """共识 = 三种子 R' 均值经 is_better（口径同探针 §2.2 的"多种子共识方向"）。"""
    tm = _tm(a={0: 0.9, 1: 0.90, 2: 0.10, 3: 0.10},   # 均值 0.3667
             b={0: 0.1, 1: 0.20, 2: 0.20, 3: 0.20})   # 均值 0.20 → 共识 a 更优
    (o,) = G.pair_observations(tm, ("a", "b"), task_type=CLF)
    assert o.cons_a_better is True          # 逐种子多数票会给 b（2:1），均值给 a


def test_consensus_tie_is_undefined_and_left_out_of_denominator():
    """三种子均值严格相等 → 方向无定义，不计入正确率分母（同 delta_decision_curve）。"""
    tm = _tm(a={0: 0.9, 1: 0.5, 2: 0.5, 3: 0.5},
             b={0: 0.1, 1: 0.5, 2: 0.5, 3: 0.5})
    (o,) = G.pair_observations(tm, ("a", "b"), task_type=CLF)
    assert o.cons_a_better is None
    st = G.yield_accuracy([o], 0.01)
    assert st["n_kept"] == 1 and st["n_comparable"] == 0 and st["accuracy"] is None


def test_missing_consensus_seeds_leave_direction_undefined():
    tm = _tm(a={0: 0.9}, b={0: 0.1})       # 只有 canonical，无 1/2/3
    (o,) = G.pair_observations(tm, ("a", "b"), task_type=CLF)
    assert o.cons_a_better is None


# ---------------------------------------------------------------------------
# 3. 判定：pass ⟺ ∃δ (A≥0.95 ∧ Y≥1/3)，δ* 取最小
# ---------------------------------------------------------------------------


def _obs(spec):
    """spec: [(|Δ|, 方向是否与共识一致)] → PairObs 列表（canonical 恒 a 更优）。"""
    out = []
    for i, (gap, agree) in enumerate(spec):
        out.append(G.PairObs(task_id="t", a=f"a{i}", b=f"b{i}",
                             d_canon=gap, label_a_better=True,
                             cons_a_better=bool(agree)))
    return out


GRID = G.DELTA_GRID


def test_pass_takes_minimal_delta_that_is_clean_and_productive():
    # 20 对大间隔且方向全对 + 10 对小间隔方向全错 → δ 抬过 0.05 即干净，产量 20/30>1/3
    obs = _obs([(0.10, True)] * 20 + [(0.02, False)] * 10)
    v = G.gate_v2_decide(obs)
    assert v.passed and v.label == "pass"
    assert v.delta_star == pytest.approx(0.025, abs=1e-9)   # 首个 >0.02 的网格点
    assert v.accuracy == pytest.approx(1.0)
    assert v.yield_frac == pytest.approx(20 / 30)


def test_fail_noisy_when_clean_delta_costs_too_much_yield():
    """存在干净 δ，但那时产量 <1/3 → fail_noisy（干净得起，产量不够）。"""
    obs = _obs([(0.10, True)] * 5 + [(0.02, False)] * 25)
    v = G.gate_v2_decide(obs)
    assert not v.passed and v.label == "fail_noisy"
    assert v.best_clean_yield == pytest.approx(5 / 30)


def test_fail_unreliable_when_no_delta_reaches_accuracy_floor():
    """方向在任何间隔上都是掷硬币（PenDigits 模式）→ fail_unreliable。"""
    obs = _obs([(0.02 + 0.01 * i, i % 2 == 0) for i in range(30)])
    v = G.gate_v2_decide(obs)
    assert not v.passed and v.label == "fail_unreliable"
    assert v.delta_star is None


def test_zero_yield_is_fail_not_crash():
    obs = _obs([(0.001, True)] * 30)
    v = G.gate_v2_decide(obs, grid=(0.05, 0.1))
    assert not v.passed and v.delta_star is None


def test_thresholds_are_explicit_parameters_not_hardcoded():
    """阈值可注入（敏感性检验要扫网格），默认值即 CRITERION.md 写定的 0.95 / 1/3。"""
    assert G.A_FLOOR == 0.95 and G.Y_FLOOR == pytest.approx(1 / 3)
    obs = _obs([(0.10, True)] * 5 + [(0.02, False)] * 25)
    assert G.gate_v2_decide(obs, y_floor=0.15).passed        # 放宽产量 → 过
    assert not G.gate_v2_decide(obs, y_floor=0.15, a_floor=1.01).passed


def test_wilson_lower_bound_is_conservative():
    lo = G.wilson_lower(29, 30, conf=0.90)
    assert 0.80 < lo < 29 / 30
    assert G.wilson_lower(0, 0, conf=0.90) is None
