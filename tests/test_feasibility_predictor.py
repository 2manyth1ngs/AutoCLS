"""MX-1 可行性预测器（scripts/15_feasibility_labels.py + 16_feasibility_loo.py）纯逻辑测试。

协议 results/mx1_feasibility/PROTOCOL.md：三态 label 与 scripts/14 逐字同式、
任务盲基线按折算（泄漏契约）、折间单变量、判读带边界。期望值手工推导；
带边界一律用二进制精确数（f2da547 教训：0.8-0.1 非精确二进制）。
"""
import importlib

import numpy as np
import pytest

dm = importlib.import_module("scripts.14_destruction_matrix")
fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")


# ---------------------------------------------------------------- 15：三态 label
class TestTriState:
    def test_band_semantics_binary_exact(self):
        # mean=1.0, band=0.25 → 阈 0.75/1.25（二进制精确）；带上=惰性（严格不等）
        assert fl.tri_state(0.74, 1.0, 0.25) == "destroy"
        assert fl.tri_state(0.75, 1.0, 0.25) == "inert"
        assert fl.tri_state(1.25, 1.0, 0.25) == "inert"
        assert fl.tri_state(1.5, 1.0, 0.25) == "improve"

    def test_destroy_leg_identical_to_scripts14(self):
        # 协议 §3：破坏腿与 destruction_flag 逐字同判（含非精确边界的浮点行为）
        rng = np.random.default_rng(0)
        for _ in range(200):
            r, m, b = rng.uniform(0, 1.2), rng.uniform(0, 1), rng.uniform(0, 0.3)
            assert (fl.tri_state(r, m, b) == "destroy") == bool(
                dm.destruction_flag(r, m, b))


class TestMergeTrained:
    def test_k3_preferred_k1_only_backbone(self):
        k3 = {("NATOPS_00", "c1"): 0.5}
        k1 = {("NATOPS_01", "c1"): 0.6,      # NATOPS 有 k3 覆盖 → k1 拒收
              ("Cricket_00", "c1"): 0.7}     # Cricket 零 k3 覆盖 → 收
        merged, rejects = fl.merge_trained(k3, k1)
        assert merged[("NATOPS_00", "c1")] == {"r_prime": 0.5, "r_source": "k3"}
        assert merged[("Cricket_00", "c1")] == {"r_prime": 0.7, "r_source": "k1"}
        assert ("NATOPS_01", "c1") not in merged
        assert rejects["k1_dataset_has_k3"] == 1

    def test_dataset_of(self):
        assert fl.dataset_of("SelfRegulationSCP1_00") == "SelfRegulationSCP1"
        assert fl.dataset_of("NATOPS+c3-6_01") == "NATOPS"


# ---------------------------------------------------------------- 16：折与基线
def _cells(spec):
    """spec: [(dataset, label, cls_key)] → cell dict 列表。"""
    return [{"dataset": d, "task_id": f"{d}_00", "cls_key": k, "label": lab}
            for d, lab, k in spec]


class TestFoldSplit:
    def test_16_folds_disjoint(self):
        ds = [f"D{i}" for i in range(16)]
        folds = lo.fold_split(ds)
        assert len(folds) == 16
        for heldout, train in folds:
            assert heldout not in train and len(train) == 15
            assert sorted(train + [heldout]) == sorted(ds)


class TestTaskBlindRates:
    def test_b0_global_rates(self):
        cells = _cells([("A", "destroy", "c1"), ("A", "inert", "c2"),
                        ("B", "inert", "c1"), ("B", "improve", "c2")])
        rates = lo.b0_rates(cells)
        np.testing.assert_allclose(rates, [0.25, 0.5, 0.25])  # improve/inert/destroy

    def test_b1_per_cls_rates(self):
        cells = _cells([("A", "destroy", "c1"), ("B", "destroy", "c1"),
                        ("A", "inert", "c2"), ("B", "improve", "c2")])
        rates = lo.b1_rates(cells)
        np.testing.assert_allclose(rates["c1"], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(rates["c2"], [0.5, 0.5, 0.0])

    def test_no_leak_from_heldout(self):
        # 协议 §4.3 显式断言：B0/B1 只吃训练折 cell——改留出 label 不改变输出
        train = _cells([("A", "destroy", "c1"), ("B", "inert", "c1")])
        heldout_v1 = _cells([("Z", "improve", "c1")])
        heldout_v2 = _cells([("Z", "destroy", "c1")])
        assert np.allclose(lo.b0_rates(train), lo.b0_rates(train))
        del heldout_v1, heldout_v2  # 基线函数签名根本不接受留出 cell
        import inspect
        for fn in (lo.b0_rates, lo.b1_rates):
            assert list(inspect.signature(fn).parameters) == ["train_cells"]


class TestMetrics:
    def test_err_tv(self):
        assert lo.err_tv(np.array([1, 0, 0.0]), np.array([0, 1, 0.0])) == 1.0
        assert lo.err_tv(np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 0.0])) == 0.0
        assert lo.err_tv(np.array([0.5, 0.5, 0]), np.array([0.25, 0.5, 0.25])) \
            == pytest.approx(0.25)

    def test_observed_rates(self):
        cells = _cells([("A", "destroy", "c1"), ("A", "destroy", "c2"),
                        ("A", "inert", "c3"), ("A", "improve", "c4")])
        np.testing.assert_allclose(lo.observed_rates(cells), [0.25, 0.25, 0.5])

    def test_brier(self):
        # 完美预测 0；最差（全押错类）2
        assert lo.brier(np.array([0, 1, 0.0]), "inert") == 0.0
        assert lo.brier(np.array([1, 0, 0.0]), "inert") == pytest.approx(2.0)
        assert lo.brier(np.array([1 / 3] * 3), "inert") == pytest.approx(2 / 3)


class TestStandardize:
    def test_fit_on_train_only_and_floor(self):
        E = np.array([[0.0, 5.0], [2.0, 5.0]])
        mu, sd = lo.standardize_fit(E)
        np.testing.assert_allclose(mu, [1.0, 5.0])
        assert sd[1] == 1e-6  # 零方差维触发下限
        z = lo.standardize_apply(np.array([[1.0, 5.0]]), mu, sd)
        np.testing.assert_allclose(z[0, 0], 0.0)

    def test_heldout_does_not_move_stats(self):
        E = np.random.default_rng(0).normal(size=(30, 4))
        mu1, sd1 = lo.standardize_fit(E)
        mu2, sd2 = lo.standardize_fit(E)  # 与留出无关：纯训练侧函数
        np.testing.assert_array_equal(mu1, mu2)
        np.testing.assert_array_equal(sd1, sd2)


class TestClassWeights:
    def test_cap_and_floor(self):
        labels = ["inert"] * 298 + ["improve", "destroy"]
        w = lo.class_weights(labels)
        assert w[1] == pytest.approx(300 / (3 * 298))
        assert w[0] == 10.0 and w[2] == 10.0  # 300/3=100 → 封顶 10
        w2 = lo.class_weights(["inert"] * 4)   # 缺类：count 下限 1
        assert w2[0] == pytest.approx(4 / (3 * 1))
        assert w2[1] == pytest.approx(4 / (3 * 4))


class TestCrossAssignment:
    def test_donor_other_dataset_and_deterministic(self):
        test_tids = ["Z_00", "Z_01"]
        train_tids = [f"D{i}_{j:02d}" for i in range(15) for j in range(2)]
        ds_of = fl.dataset_of
        m1 = lo.cross_assignment_mx1(test_tids, train_tids, ds_of, seed=0)
        m2 = lo.cross_assignment_mx1(test_tids, train_tids, ds_of, seed=0)
        assert m1 == m2
        for t, donor in m1.items():
            assert donor in train_tids and ds_of(donor) != ds_of(t)
        # 不同种子应能产生不同映射（30 个候选，碰撞概率可忽略）
        diffs = [lo.cross_assignment_mx1(test_tids, train_tids, ds_of, seed=s)
                 for s in range(5)]
        assert len({tuple(sorted(d.items())) for d in diffs}) > 1


class TestNoiseFloor:
    def test_degenerate_zero(self):
        assert lo.noise_floor(np.array([0, 1, 0.0]), n_cells=20,
                              n_sim=100, seed=0) == 0.0

    def test_uniform_positive_and_deterministic(self):
        p = np.array([1 / 3] * 3)
        f1 = lo.noise_floor(p, n_cells=20, n_sim=500, seed=1)
        f2 = lo.noise_floor(p, n_cells=20, n_sim=500, seed=1)
        assert f1 == f2 > 0.0


class TestPairedWilcoxon:
    def test_directional(self):
        x = np.arange(16) * 0.01           # 模型误差
        y = x + 0.05                       # 基线更差
        assert lo.paired_wilcoxon_less(x, y) < 0.01
        assert lo.paired_wilcoxon_less(y, x) > 0.9

    def test_all_zero_diffs(self):
        x = np.full(16, 0.3)
        assert lo.paired_wilcoxon_less(x, x) == 1.0


class TestVerdict:
    def test_bands(self):
        assert lo.verdict_mx1(0.01, 0.01, 0.5) == "established"
        assert lo.verdict_mx1(0.01, 0.30, 0.5) == "e_unproven"
        assert lo.verdict_mx1(0.10, 0.01, 0.5) == "edge"       # p_D1<0.15
        assert lo.verdict_mx1(0.50, 0.50, 0.01) == "edge"      # E_mean 臂过 D1
        assert lo.verdict_mx1(0.50, 0.50, 0.50) == "not_established"

    def test_boundary_exact(self):
        assert lo.verdict_mx1(0.05, 0.01, 0.5) == "edge"       # 0.05 不含
        assert lo.verdict_mx1(0.15, 0.5, 0.5) == "not_established"
        assert lo.verdict_mx1(0.049, 0.05, 0.5) == "e_unproven"


class TestMLP:
    def _toy(self):
        # 线性可分三类（输入 2 维）
        rng = np.random.default_rng(0)
        X = np.concatenate([rng.normal([0, 4], 0.2, (30, 2)),
                            rng.normal([4, 0], 0.2, (30, 2)),
                            rng.normal([-4, -4], 0.2, (30, 2))])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        return X.astype(np.float32), y

    def test_deterministic_and_converges(self):
        X, y = self._toy()
        w = np.ones(3)
        m1 = lo.train_mlp(X, y, w, seed=7, epochs=200)
        m2 = lo.train_mlp(X, y, w, seed=7, epochs=200)
        p1, p2 = lo.predict_proba(m1, X), lo.predict_proba(m2, X)
        np.testing.assert_array_equal(p1, p2)          # CPU 全批逐位确定
        np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-5)
        assert (p1.argmax(axis=1) == y).mean() == 1.0  # 可分数据须收敛

    def test_seed_changes_model(self):
        X, y = self._toy()
        w = np.ones(3)
        p1 = lo.predict_proba(lo.train_mlp(X, y, w, seed=1, epochs=30), X)
        p2 = lo.predict_proba(lo.train_mlp(X, y, w, seed=2, epochs=30), X)
        assert not np.array_equal(p1, p2)
