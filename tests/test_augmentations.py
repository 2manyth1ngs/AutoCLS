"""数据增强测试（AutoCL 附录 A + Table 8；TECHNICAL_DESIGN.md 模块 3）。

每个增强验证：p=0 恒等（0.0 = 不启用）、输出形状正确、p 边界（0.95）不崩。
张量约定 (B, T, C)。
"""
import pytest
import torch

from autocls.contrastive.augmentations import (
    ORDERS,
    augment_two_views,
    crop_views,
    freq_mask,
    jitter,
    point_mask,
    rescale,
    resize,
)


def _x(B=4, T=128, C=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, C, generator=g)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------- jitter: x + N(0, p)（附录 A） ----------

class TestJitter:
    def test_p0_identity(self):
        x = _x()
        assert torch.equal(jitter(x, 0.0, generator=_gen()), x)

    def test_shape_and_change(self):
        x = _x()
        out = jitter(x, 0.5, generator=_gen())
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_noise_scale(self):
        # 大样本下噪声 std ≈ p
        x = torch.zeros(64, 128, 9)
        out = jitter(x, 0.5, generator=_gen())
        assert abs(out.std().item() - 0.5) < 0.02

    def test_boundary_p(self):
        out = jitter(_x(), 0.95, generator=_gen())
        assert torch.isfinite(out).all()


# ---------- crop: 两段带共享子序列的裁剪（附录 A：p = T'/T） ----------

class TestCrop:
    def test_p0_identity(self):
        x = _x()
        v1, v2, sl1, sl2 = crop_views(x, 0.0, generator=_gen())
        assert torch.equal(v1, x) and torch.equal(v2, x)
        assert torch.equal(v1[:, sl1], x) and torch.equal(v2[:, sl2], x)

    def test_shared_segment_alignment(self):
        # 共享段在两个视图中的切片内容一致，长度 = round(p*T)
        x = _x()
        T = x.shape[1]
        for p in (0.3, 0.5, 0.95):
            v1, v2, sl1, sl2 = crop_views(x, p, generator=_gen(1))
            assert torch.equal(v1[:, sl1], v2[:, sl2])
            assert sl1.stop - sl1.start == round(p * T)

    def test_views_are_subsequences(self):
        # t1 <= t2 < t1' <= t2'：v1 是 x 的前段裁剪、v2 是后段裁剪
        x = _x()
        v1, v2, sl1, sl2 = crop_views(x, 0.4, generator=_gen(2))
        B, T, C = x.shape
        assert v1.shape[0] == B and v1.shape[2] == C
        assert v1.shape[1] <= T and v2.shape[1] <= T
        # 共享段确实来自原序列的同一段
        shared = v1[:, sl1]
        found = any(
            torch.equal(shared, x[:, s : s + shared.shape[1]])
            for s in range(T - shared.shape[1] + 1)
        )
        assert found

    def test_tiny_T(self):
        x = _x(T=2)
        v1, v2, sl1, sl2 = crop_views(x, 0.5, generator=_gen())
        assert torch.equal(v1[:, sl1], v2[:, sl2])
        assert v1.shape[1] >= 1 and v2.shape[1] >= 1


# ---------- point_mask: x ⊙ m, m[t]~Bernoulli(1-p)（附录 A） ----------

class TestPointMask:
    def test_p0_identity(self):
        x = _x()
        assert torch.equal(point_mask(x, 0.0, generator=_gen()), x)

    def test_masks_whole_timesteps(self):
        # m 按时间步采样，掩掉的时间步整条置零（跨通道广播）
        x = _x() + 10.0  # 远离 0，避免误判
        out = point_mask(x, 0.5, generator=_gen())
        zero_t = (out == 0).all(dim=-1)   # (B, T)：整个时间步为 0
        elem_zero = out == 0
        assert torch.equal(elem_zero, zero_t.unsqueeze(-1).expand_as(elem_zero))

    def test_mask_ratio_approx_p(self):
        x = torch.ones(64, 128, 4)
        for p in (0.3, 0.7):
            out = point_mask(x, p, generator=_gen())
            ratio = (out == 0).float().mean().item()
            assert abs(ratio - p) < 0.05

    def test_boundary_p(self):
        out = point_mask(_x(), 0.95, generator=_gen())
        assert torch.isfinite(out).all()


# ---------- freq_mask: DFT → 掩频 → IDFT（附录 A） ----------

class TestFreqMask:
    def test_p0_identity(self):
        x = _x()
        assert torch.equal(freq_mask(x, 0.0, generator=_gen()), x)

    def test_output_real_and_shape(self):
        x = _x()
        out = freq_mask(x, 0.5, generator=_gen())
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert torch.isfinite(out).all()

    def test_changes_values(self):
        x = _x()
        assert not torch.equal(freq_mask(x, 0.5, generator=_gen()), x)

    def test_mask_all_frequencies_gives_zero(self):
        # p→1 掩掉全部频率 → 信号趋于 0；p=0.95 时能量应大幅下降
        x = _x()
        out = freq_mask(x, 0.95, generator=_gen())
        assert out.pow(2).mean() < x.pow(2).mean()


# ---------- rescale: 幅度乘 (1+N(0,p))（附录 A 公式截断，按语义补全） ----------

class TestRescale:
    def test_p0_identity(self):
        x = _x()
        assert torch.equal(rescale(x, 0.0, generator=_gen()), x)

    def test_per_sample_scalar_scaling(self):
        # 每个样本一个缩放因子：out/x 在样本内为常数
        x = _x().abs() + 1.0
        out = rescale(x, 0.3, generator=_gen())
        ratio = out / x
        per_sample = ratio.reshape(ratio.shape[0], -1)
        assert torch.allclose(per_sample, per_sample[:, :1].expand_as(per_sample), atol=1e-5)

    def test_scale_distribution(self):
        # 大样本下缩放因子 std ≈ p
        x = torch.ones(4096, 4, 1)
        out = rescale(x, 0.3, generator=_gen())
        factors = out.reshape(4096, -1)[:, 0]
        assert abs(factors.std().item() - 0.3) < 0.02
        assert abs(factors.mean().item() - 1.0) < 0.02

    def test_boundary_p(self):
        assert torch.isfinite(rescale(_x(), 0.95, generator=_gen())).all()


# ---------- resize: 长度乘 (1+N(0,p)) 后插值（附录 A 公式截断，按语义补全） ----------

class TestResize:
    def test_p0_identity(self):
        x = _x()
        assert torch.equal(resize(x, 0.0, generator=_gen()), x)

    def test_length_changes_channels_kept(self):
        x = _x(T=128)
        out = resize(x, 0.5, generator=_gen(3))
        assert out.shape[0] == x.shape[0] and out.shape[2] == x.shape[2]
        assert out.shape[1] != x.shape[1]  # 该种子下长度确实变化

    def test_length_bounds(self):
        # 极端 p 下长度仍被夹在 [2, 2T]，不崩
        x = _x(T=16)
        for seed in range(8):
            out = resize(x, 0.95, generator=_gen(seed))
            assert 2 <= out.shape[1] <= 32
            assert torch.isfinite(out).all()

    def test_content_preserved_coarsely(self):
        # 插值应保持信号大体形态：常数序列 resize 后仍为常数
        x = torch.ones(2, 64, 3) * 5.0
        out = resize(x, 0.3, generator=_gen(1))
        assert torch.allclose(out, torch.full_like(out, 5.0), atol=1e-5)


# ---------- 顺序表（AutoCL Table 8） ----------

class TestOrders:
    def test_five_orders_cover_six_augs(self):
        assert len(ORDERS) == 5
        for order in ORDERS:
            assert sorted(order) == sorted(
                ["resize", "rescale", "jitter", "point_mask", "freq_mask", "crop"]
            )

    def test_resize_always_first(self):
        # Table 8：resize 在全部 5 种顺序中都排第一
        assert all(order[0] == "resize" for order in ORDERS)

    def test_crop_positions(self):
        # Table 8：crop 位置随 order 依次前移 6,5,4,3,2
        assert [order.index("crop") for order in ORDERS] == [5, 4, 3, 2, 1]


# ---------- 组合管线：两视图生成 ----------

class TestAugmentTwoViews:
    def test_all_zero_p_views_identical_to_input(self):
        # 全部 p=0：两视图都等于原输入，共享段为全长
        x = _x()
        params = dict(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.0)
        v1, v2, sl1, sl2 = augment_two_views(x, params, order=0, generator=_gen())
        assert torch.equal(v1, x) and torch.equal(v2, x)
        assert torch.equal(v1[:, sl1], v2[:, sl2])

    def test_jitter_crop_views_differ(self):
        # 最小路径 CLS：仅 jitter+crop 开启，两视图不同但共享段对齐（长度一致）
        x = _x()
        params = dict(resize=0.0, rescale=0.0, jitter=0.3, point_mask=0.0, freq_mask=0.0, crop=0.5)
        v1, v2, sl1, sl2 = augment_two_views(x, params, order=0, generator=_gen())
        assert not torch.equal(v1[:, sl1], v2[:, sl2])  # jitter 独立随机
        assert sl1.stop - sl1.start == sl2.stop - sl2.start == round(0.5 * x.shape[1])

    def test_boundary_all_max(self):
        x = _x()
        params = dict(resize=0.0, rescale=0.0, jitter=0.95, point_mask=0.0, freq_mask=0.0, crop=0.95)
        for order in range(5):
            v1, v2, sl1, sl2 = augment_two_views(x, params, order=order, generator=_gen())
            assert torch.isfinite(v1).all() and torch.isfinite(v2).all()

    def test_all_augs_enabled_all_orders(self):
        # 6 种增强全开 × 5 种顺序：形状一致、共享段长度对齐、数值有限
        x = _x()
        params = dict(resize=0.3, rescale=0.3, jitter=0.3, point_mask=0.3, freq_mask=0.3, crop=0.5)
        for order in range(5):
            v1, v2, sl1, sl2 = augment_two_views(x, params, order=order, generator=_gen(order))
            assert v1.shape[0] == v2.shape[0] == x.shape[0]
            assert v1.shape[2] == v2.shape[2] == x.shape[2]
            assert sl1.stop - sl1.start == sl2.stop - sl2.start >= 1
            assert sl1.stop <= v1.shape[1] and sl2.stop <= v2.shape[1]
            assert torch.isfinite(v1).all() and torch.isfinite(v2).all()

    def test_resize_shares_target_length(self):
        # resize 两视图共享目标长度（否则 crop 坐标失配）：仅开 resize 时两视图同长
        x = _x()
        params = dict(resize=0.5, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.0)
        v1, v2, sl1, sl2 = augment_two_views(x, params, order=0, generator=_gen(3))
        assert v1.shape == v2.shape

    def test_extreme_boundary_all_095(self):
        # 全部 p=0.95 的极端边界 × 全部顺序：不崩、共享段至少 1 步
        x = _x(T=32)
        params = {k: 0.95 for k in ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop")}
        for order in range(5):
            v1, v2, sl1, sl2 = augment_two_views(x, params, order=order, generator=_gen(order))
            assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
            assert sl1.stop - sl1.start >= 1


# ---------- crop_mode="ts2vec"：官方逐迭代 crop_l~U[2,T] + 逐行偏移 ----------
# 镜像 reference/AutoCTS_plusplus/NAS_Net/ts2vec.py:126-132（全量协议对齐用，
# 默认 "fixed" 路径逐位不变）。

from autocls.contrastive.augmentations import (  # noqa: E402
    _crop_per_row,
    sample_crop_coords_ts2vec,
)


class TestCropTs2vec:
    _params = dict(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5)

    def test_fixed_mode_is_default_and_bitwise_equal(self):
        x = _x()
        a = augment_two_views(x, self._params, order=0, generator=_gen(5))
        b = augment_two_views(x, self._params, order=0, generator=_gen(5), crop_mode="fixed")
        for u, v in zip(a[:2], b[:2]):
            assert torch.equal(u, v)
        assert a[2] == b[2] and a[3] == b[3]

    def test_shared_segment_aligned_per_row(self):
        x = _x(B=8, T=64)
        for seed in range(6):
            v1, v2, sl1, sl2 = augment_two_views(
                x, self._params, order=0, generator=_gen(seed), crop_mode="ts2vec")
            assert torch.equal(v1[:, sl1], v2[:, sl2])
            crop_l = sl1.stop - sl1.start
            assert 2 <= crop_l <= x.shape[1]
            assert sl1.stop <= v1.shape[1] and sl2.stop <= v2.shape[1]
            # 官方 out1[:, -crop_l:] / out2[:, :crop_l]
            assert sl2.start == 0 and sl1.stop == v1.shape[1]

    def test_shared_segment_is_row_subsequence_of_input(self):
        # 每行的共享段必须来自该行原序列的同一段（逐行偏移后仍对齐）
        x = _x(B=6, T=48)
        coords = sample_crop_coords_ts2vec(6, 48, generator=_gen(3))
        v1, v2, sl1, sl2 = _crop_per_row(x, x, coords)
        crop_l, left, _right, _eleft, _eright, offset = coords
        for b in range(6):
            s = int(offset[b]) + left
            assert torch.equal(v1[b, sl1], x[b, s:s + crop_l])
            assert torch.equal(v2[b, sl2], x[b, s:s + crop_l])

    def test_per_row_offsets_differ(self):
        # 逐行偏移是本模式与 fixed 的本质区别：B=32 下不可能全同
        coords = sample_crop_coords_ts2vec(32, 128, generator=_gen(0))
        offset = coords[-1]
        assert tuple(offset.shape) == (32,)
        assert len(set(int(o) for o in offset)) > 1

    def test_indices_within_bounds(self):
        T = 40
        for seed in range(50):
            crop_l, left, right, eleft, eright, offset = sample_crop_coords_ts2vec(
                16, T, generator=_gen(seed))
            assert 2 <= crop_l <= T and 0 <= left and right == left + crop_l <= T
            assert 0 <= eleft <= left and right <= eright <= T
            assert int(offset.min()) + eleft >= 0
            assert int(offset.max()) + eright <= T

    def test_p0_identity_in_both_modes(self):
        x = _x()
        params = dict(self._params, crop=0.0)
        for mode in ("fixed", "ts2vec"):
            v1, v2, sl1, sl2 = augment_two_views(x, params, order=0, generator=_gen(), crop_mode=mode)
            assert torch.equal(v1, x) and torch.equal(v2, x)
            assert torch.equal(v1[:, sl1], x)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            augment_two_views(_x(), self._params, order=0, generator=_gen(), crop_mode="random")

    def test_ts2vec_mode_with_prior_augs_all_orders(self):
        # 与其他增强组合 × 5 种顺序：共享段仍对齐、数值有限
        x = _x(B=4, T=64)
        params = dict(resize=0.3, rescale=0.3, jitter=0.3, point_mask=0.3, freq_mask=0.3, crop=0.5)
        for order in range(5):
            v1, v2, sl1, sl2 = augment_two_views(
                x, params, order=order, generator=_gen(order), crop_mode="ts2vec")
            assert sl1.stop - sl1.start == sl2.stop - sl2.start >= 2
            assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
