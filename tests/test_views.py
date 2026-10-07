"""autocls/data/views.py：输入视图（时域 / 差分 / 频域 / 时频）的纯函数性质。"""
import numpy as np
import pytest

from autocls.data import views as V


def _x(n=4, t=64, c=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, t, c)).astype(np.float32)


def test_raw_is_identity():
    x = _x()
    np.testing.assert_array_equal(V.apply_view(x, "raw"), x)


def test_unknown_view_raises():
    with pytest.raises(ValueError):
        V.apply_view(_x(), "nope")


def test_diff_shape_and_values():
    x = _x()
    y = V.apply_view(x, "diff")
    assert y.shape == (4, 64, 6) and y.dtype == np.float32
    np.testing.assert_array_equal(y[..., :3], x)
    np.testing.assert_allclose(y[:, 1:, 3:], x[:, 1:] - x[:, :-1], rtol=1e-6)
    assert np.all(y[:, 0, 3:] == 0)


def test_psd_shape_and_peak():
    t = 128
    tt = np.arange(t)
    x = np.zeros((2, t, 1), np.float32)
    x[0, :, 0] = np.sin(2 * np.pi * 10 * tt / t)      # 第 10 个频点
    x[1, :, 0] = np.sin(2 * np.pi * 30 * tt / t)
    y = V.apply_view(x, "psd")
    assert y.shape == (2, t // 2, 1)                  # 去掉直流后 T//2 个频点（未超上限不池化）
    assert int(np.argmax(y[0, :, 0])) == 9 and int(np.argmax(y[1, :, 0])) == 29
    assert np.all(np.isfinite(y))


def test_psd_pools_long_series():
    y = V.apply_view(_x(n=2, t=3000, c=2), "psd")
    assert y.shape[1] <= V.PSD_MAX_BINS and y.shape[2] == 2 and np.all(np.isfinite(y))


def test_spec_shape_bounds():
    for t, c in [(50, 28), (3000, 64), (270, 1345), (62, 144)]:
        x = _x(n=2, t=t, c=c)
        y = V.apply_view(x, "spec")
        n_win = V.spec_window(t)
        n_frames = 1 + (t - n_win) // (n_win // 2)
        assert y.shape[0] == 2 and y.shape[1] == n_frames
        assert y.shape[2] % c == 0 and y.shape[2] <= max(c, V.SPEC_MAX_CHANNELS)
        assert np.all(np.isfinite(y))


def test_spec_band_energy_moves_with_frequency():
    t = 256
    tt = np.arange(t)
    x = np.zeros((2, t, 1), np.float32)
    x[0, :, 0] = np.sin(2 * np.pi * 4 * tt / 32)      # 低频
    x[1, :, 0] = np.sin(2 * np.pi * 14 * tt / 32)     # 高频
    y = V.apply_view(x, "spec")
    low = int(np.argmax(y[0].mean(0))); high = int(np.argmax(y[1].mean(0)))
    assert low < high


def test_views_are_per_sample():
    """每个样本只依赖自身：批内改动别的样本不影响结果（训练 / 测试可分开变换）。"""
    x = _x(n=3, t=100, c=2)
    for v in V.VIEWS:
        a = V.apply_view(x, v)
        x2 = x.copy(); x2[1:] = 0
        b = V.apply_view(x2, v)
        np.testing.assert_allclose(a[0], b[0], rtol=1e-5, atol=1e-6)


def test_standardize_with_train_stats():
    tr, te = _x(n=5, seed=1) * 3 + 2, _x(n=2, seed=2)
    a, b = V.standardize_train(tr, te)
    flat = a.reshape(-1, a.shape[-1])
    np.testing.assert_allclose(flat.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(flat.std(0), 1, atol=1e-4)
    assert b.shape == te.shape
