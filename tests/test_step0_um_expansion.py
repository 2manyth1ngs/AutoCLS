"""Step 0 扩容探针（scripts/11_step0_um_expansion.py）纯逻辑测试。

期望值手工推导（不以被测代码为 oracle）：
- extras 构造：零交集、确定性、分层配额、封顶；
- δ 公式与 2c calibrate 同式（P95 最近秩、量子下限、上取整 0.005）；
- 投票/天花板在合成 R' 表上手算；
- 守卫（三种子方向全同率）与判读表四格边界。
"""
import importlib

import numpy as np

from autocls.data.task_slicer import SlicedTask, materialize_task
from autocls.labeling import run_seed_for
from autocls.task_types import TASK_CLASSIFICATION

probe = importlib.import_module("scripts.11_step0_um_expansion")
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")


class TestSeeds:
    def test_rep0_canonical_rep12_noise(self):
        assert probe.seeds_for("t0", "k0", 0) == _c2.canonical_seeds("t0", "k0")
        for rep in (1, 2):
            w, sp = probe.seeds_for("t0", "k0", rep)
            assert w == run_seed_for("t0", f"k0:noise{rep}")
            assert sp == run_seed_for("t0", "train_eval_split")


class TestAllocate:
    def test_proportional_hand(self):
        # 100/50/50 分 4 → 配额 2/1/1
        out = probe.allocate_largest_remainder({0: 100, 1: 50, 2: 50}, 4)
        assert out == {0: 2, 1: 1, 2: 1}

    def test_capped_spills(self):
        # 类 0 只有 1 个可用，其配额溢给余量最大的类
        out = probe.allocate_largest_remainder({0: 1, 1: 10, 2: 10}, 9)
        assert out[0] <= 1 and sum(out.values()) == 9

    def test_n_ge_total_takes_all(self):
        assert probe.allocate_largest_remainder({0: 3, 1: 2}, 99) == {0: 3, 1: 2}


class TestExpandEvalIndices:
    def _setup(self):
        # 类 0: 索引 0..9；类 1: 10..19；类 2: 20..29（各 10 个）
        y = np.repeat([0, 1, 2], 10)
        sample = (0, 1, 10, 11, 20, 21)   # 每类占用 2 个
        return y, (0, 1, 2), sample

    def test_disjoint_and_eligible(self):
        y, classes, sample = self._setup()
        rng = np.random.default_rng(7)
        idx = probe.expand_eval_indices(y, classes, sample, 12, rng)
        assert len(idx) == 12
        assert not (set(idx.tolist()) & set(sample))
        assert set(y[idx].tolist()) <= set(classes)

    def test_deterministic(self):
        y, classes, sample = self._setup()
        a = probe.expand_eval_indices(y, classes, sample, 12,
                                      np.random.default_rng(7))
        b = probe.expand_eval_indices(y, classes, sample, 12,
                                      np.random.default_rng(7))
        assert a.tolist() == b.tolist()

    def test_stratified_equal_leftover(self):
        # 每类 leftover 均为 8 → 12 个 extras 每类恰 4
        y, classes, sample = self._setup()
        idx = probe.expand_eval_indices(y, classes, sample, 12,
                                        np.random.default_rng(0))
        counts = {c: int((y[idx] == c).sum()) for c in classes}
        assert counts == {0: 4, 1: 4, 2: 4}

    def test_capped_at_leftover(self):
        y, classes, sample = self._setup()
        idx = probe.expand_eval_indices(y, classes, sample, 999,
                                        np.random.default_rng(0))
        assert len(idx) == 30 - len(sample)


class TestMaterializeIndices:
    def test_matches_materialize_task_semantics(self):
        # 8 样本 × T=4 × C=3；类 {5, 9} 保序重映射 5→0、9→1
        X = np.arange(8 * 4 * 3, dtype=np.float32).reshape(8, 4, 3)
        y = np.array([5, 9, 5, 9, 5, 9, 5, 9])
        task = SlicedTask(task_id="t", pool="p", classes=(5, 9),
                          channels=(0, 2), sample_indices=(0, 1, 2, 3),
                          tag="x", task_type=TASK_CLASSIFICATION)
        Xt, yt = materialize_task(task, X, y)
        Xe, ye = probe.materialize_indices(X, y, task, np.array([0, 1, 2, 3]))
        assert np.array_equal(Xt, Xe) and np.array_equal(yt, ye)
        # extras 索引（4..7）：同 remap
        _Xe2, ye2 = probe.materialize_indices(X, y, task, np.array([4, 5]))
        assert ye2.tolist() == [0, 1]


class TestDeltaFromDiffs:
    def test_p95_nearest_rank_and_ceil(self):
        # 20 个差值 0.001..0.020：P95 最近秩 = 第 round(0.95*19)=18 位（0 起）= 0.019
        diffs = [i / 1000 for i in range(1, 21)]
        out = probe.delta_from_diffs(diffs, quanta=[0.001])
        assert abs(out["p95"] - 0.019) < 1e-12
        assert out["delta"] == 0.02   # ceil(0.019/0.005)*0.005

    def test_quantum_floor_dominates(self):
        out = probe.delta_from_diffs([0.001, 0.001], quanta=[0.02])
        assert out["delta"] == 0.04   # 2×0.02 下限

    def test_empty(self):
        assert probe.delta_from_diffs([], [])["delta"] is None


class TestVotesAndCeiling:
    def test_hand_computed_ceiling(self):
        # 2 任务 × 3 CLS。t1: a=0.9 b=0.5 c=0.4；t2: a=0.5 b=0.9 c=0.4
        rvals = {"t1": {"a": 0.9, "b": 0.5, "c": 0.4},
                 "t2": {"a": 0.5, "b": 0.9, "c": 0.4}}
        votes, nd, nt = probe.collect_votes(
            rvals, ["a", "b", "c"], lambda *_: 0.0, TASK_CLASSIFICATION)
        # (a,b)：t1 a优 / t2 b优 → 多数 1/2；(a,c)：2 票同向；(b,c)：t1 b优 t2 b优 → 2 同向
        ceil_val, n = probe.ceiling_from_votes(votes)
        assert n == 6 and abs(ceil_val - 5 / 6) < 1e-12
        assert nd == 0 and nt == 0

    def test_delta_drops_small_gaps(self):
        rvals = {"t1": {"a": 0.90, "b": 0.89}}
        votes, nd, _ = probe.collect_votes(
            rvals, ["a", "b"], lambda *_: 0.05, TASK_CLASSIFICATION)
        assert votes == [] and nd == 1

    def test_tie_dropped(self):
        rvals = {"t1": {"a": 0.9, "b": 0.9}}
        votes, _nd, nt = probe.collect_votes(
            rvals, ["a", "b"], lambda *_: 0.0, TASK_CLASSIFICATION)
        assert votes == [] and nt == 1

    def test_per_task_delta_callback(self):
        rvals = {"t1": {"a": 0.9, "b": 0.8}, "t2": {"a": 0.9, "b": 0.8}}
        delta_of = lambda tid, a, b: 0.2 if tid == "t1" else 0.05
        votes, nd, _ = probe.collect_votes(
            rvals, ["a", "b"], delta_of, TASK_CLASSIFICATION)
        assert len(votes) == 1 and votes[0]["task_id"] == "t2" and nd == 1


class TestGuard:
    def test_all_same_sign_rate(self):
        rate, n = probe.all_same_sign_rate([(1, 1, 1), (1, 0, 1), (0, 0, 0)])
        assert n == 3 and abs(rate - 2 / 3) < 1e-12

    def test_none_counts_inconsistent(self):
        rate, _ = probe.all_same_sign_rate([(1, None, 1)])
        assert rate == 0.0

    def test_empty_is_none(self):
        assert probe.all_same_sign_rate([]) == (None, 0)


class TestVerdictCell:
    def test_four_cells(self):
        assert probe.verdict_cell(False, 0.5, True) == "A_ruler_not_delivered"
        assert probe.verdict_cell(True, 0.99, True) == "B_um_homogeneous"
        assert probe.verdict_cell(True, None, True) == "B_um_homogeneous"
        assert probe.verdict_cell(True, 0.90, True) == "C_go"
        assert probe.verdict_cell(True, 0.90, False) == "D_kill_within_reach"

    def test_homogeneous_boundary(self):
        # 0.98 恰在同质线上 → B（≥ 判定，协议 §4）
        assert probe.verdict_cell(True, 0.98, True) == "B_um_homogeneous"
        assert probe.verdict_cell(True, 0.9799, False) == "D_kill_within_reach"
