"""MX-1b 探针条件化可行性预测（scripts/17_feasibility_loo_b.py）纯逻辑测试。

协议 results/mx1b_feasibility/PROTOCOL.md：G 组类别几何公式、采样索引与
encode_task_windows 逐位同源、判读带（含勿复活分支）。期望值手工推导。
"""
import importlib

import numpy as np
import pytest

fb = importlib.import_module("scripts.17_feasibility_loo_b")


class _StubTS2Vec:
    """encode_task_windows 的最小替身：D=1 恒等编码，用于验证采样索引同源。"""
    input_dims = 1
    frozen = True
    n_iters = 1

    def encode(self, data, encoding_window=None):
        return data  # (n, t, 1) 恒等 → 调用方 reshape 后跨通道取均值


class TestSampleIndices:
    def test_mirrors_encode_task_windows(self):
        # 协议 §3：G 组窗口标签的采样必须与 encode_task_windows 完全同源。
        # 恒等替身编码器下，encode_task_windows 的输出行 == windows[idx]（单通道）。
        from autocls.task_embed import encode_task_windows
        X = np.arange(40, dtype=np.float32).reshape(20, 2, 1)  # 每行取值唯一
        idx = fb.sample_indices(20, sample_num=7, seed=0)
        reprs = encode_task_windows(X, _StubTS2Vec(), sample_num=7, seed=0)
        np.testing.assert_array_equal(reprs[:, :, 0], X[idx, :, 0])

    def test_deterministic_and_capped(self):
        assert list(fb.sample_indices(10, sample_num=100, seed=3)) == list(range(10))
        i1 = fb.sample_indices(500, sample_num=100, seed=3)
        i2 = fb.sample_indices(500, sample_num=100, seed=3)
        np.testing.assert_array_equal(i1, i2)
        assert len(i1) == 100 and np.all(np.diff(i1) > 0)  # 有序无重复


class TestL2Normalize:
    def test_unit_rows_and_zero_guard(self):
        V = np.array([[3.0, 4.0], [0.0, 0.0]])
        Z = fb.l2_normalize_rows(V)
        np.testing.assert_allclose(np.linalg.norm(Z[0]), 1.0)
        assert np.all(np.isfinite(Z))


class TestClassGeometry:
    def _separated(self):
        rng = np.random.default_rng(0)
        V = np.concatenate([rng.normal([10, 0], 0.1, (20, 2)),
                            rng.normal([0, 10], 0.1, (20, 2)),
                            rng.normal([-10, -10], 0.1, (20, 2))])
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        return V, y

    def test_separated_classes(self):
        V, y = self._separated()
        g = fb.class_geometry(V, y)
        assert g.shape == (5,)
        assert g[0] == 1.0                       # g1 NCC acc 满分
        assert g[4] == pytest.approx(1.0 - 1 / 3)  # g5 = g1 − 1/C
        assert g[2] > 0.5                        # g3 silhouette 高
        assert g[1] > 1.0 and g[3] > 1.0         # Fisher/裕度 大

    def test_shuffled_labels_low(self):
        V, y = self._separated()
        rng = np.random.default_rng(1)
        y_sh = rng.permutation(y)               # 同分布乱标签 → 几何应塌
        g = fb.class_geometry(V, y_sh)
        assert g[2] < 0.1                        # silhouette 掉到 ≈0 以下
        assert g[1] < 0.5                        # Fisher 比塌缩

    def test_single_class_returns_zeros(self):
        V = np.random.default_rng(0).normal(size=(10, 4))
        g = fb.class_geometry(V, np.zeros(10, dtype=int))
        np.testing.assert_array_equal(g, np.zeros(5))

    def test_deterministic(self):
        V, y = self._separated()
        np.testing.assert_array_equal(fb.class_geometry(V, y),
                                      fb.class_geometry(V, y))


class TestProbeVec:
    def test_fields(self):
        row = {"base_mean": 0.75, "classes": [1, 2, 3, 4], "band": 0.125}
        np.testing.assert_allclose(fb.probe_vec(row), [0.75, 0.25, 0.125])


class TestVerdictB:
    def test_bands(self):
        assert fb.verdict_mx1b(0.01, 0.01) == "established"
        assert fb.verdict_mx1b(0.01, 0.30) == "feature_signal_unproven"
        assert fb.verdict_mx1b(0.10, 0.01) == "edge"
        assert fb.verdict_mx1b(0.50, 0.01) == "not_established_norevive"

    def test_boundary_exact(self):
        assert fb.verdict_mx1b(0.05, 0.01) == "edge"          # 0.05 不含
        assert fb.verdict_mx1b(0.15, 0.01) == "not_established_norevive"
        assert fb.verdict_mx1b(0.049, 0.05) == "feature_signal_unproven"


class TestFeatureAssembly:
    def test_concat_no_nan(self):
        P, G, E = np.ones(3), np.zeros(5), np.full(4, 2.0)
        for arm, vec in fb.assemble_features(P, G, E).items():
            assert np.all(np.isfinite(vec)), arm
        d = fb.assemble_features(P, G, E)
        assert d["bm"].shape == (8,) and d["be"].shape == (12,)
        assert d["bp"].shape == (3,) and d["bg"].shape == (5,)
        np.testing.assert_array_equal(d["bm"], np.concatenate([P, G]))
