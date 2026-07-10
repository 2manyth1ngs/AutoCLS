"""M0 模块 2 测试：CLS 向量编码（TECHNICAL_DESIGN.md 第 5/6 节，方案 A）。

先于实现编写（铁律 5）。验收：
- encode 维度固定
- encode→decode 能还原类别字段
- 归一化范围正确
"""
import numpy as np
import pytest

from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLS, CLSSpace


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def encoder(space):
    return CLSEncoder(space)


def _active_cls():
    """手工构造的确定样本（temporal 开启，门控字段有效）。"""
    return CLS(
        resize=0.0, rescale=0.95, jitter=0.5, point_mask=0.1, freq_mask=0.9,
        crop=0.2, order=3, emb_jitter=0.0, emb_mask=0.95, norm="l2",
        instance=True, temporal=True, cross_scale=False,
        kernel=5, pool="max", adj_neighbor=False,
        loss_type="triplet", sim_func="cosine", temperature=1.0,
    )


def _inactive_cls():
    """temporal/cross_scale 均关闭，门控字段为无效档 None。"""
    return CLS(
        resize=0.1, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0,
        crop=0.0, order=0, emb_jitter=0.5, emb_mask=0.0, norm="none",
        instance=True, temporal=False, cross_scale=False,
        kernel=None, pool=None, adj_neighbor=None,
        loss_type="info_nce", sim_func="dot", temperature=0.01,
    )


# ---------------------------------------------------------------- 维度

def test_dim_fixed(space, encoder):
    # 第 5 节：数值/对数字段各 1 维，bool 1 维，类别 one-hot len(choices) 维
    # 8 比率 + temperature + instance/temporal/cross_scale/adj_neighbor
    # + one-hot: order 5 + norm 3 + kernel 4 + pool 2 + loss 2 + sim 3 = 32
    assert encoder.dim == 32


def test_encode_shape_dtype_range(space, encoder):
    rng = np.random.default_rng(0)
    for _ in range(300):
        v = encoder.encode(space.sample_random(rng))
        assert v.shape == (encoder.dim,)
        assert v.dtype == np.float32
        assert np.all(np.isfinite(v))
        assert np.all(v >= 0.0) and np.all(v <= 1.0)


# ---------------------------------------------------------------- 归一化

def test_numeric_normalization(encoder):
    v = encoder.encode(_active_cls())
    sl = encoder.field_slices
    assert v[sl["resize"]][0] == 0.0            # 最小档 → 0
    assert v[sl["rescale"]][0] == 1.0           # 最大档 0.95 → 1
    assert v[sl["jitter"]][0] == pytest.approx(0.5 / 0.95)
    assert v[sl["emb_mask"]][0] == 1.0
    # temperature 跨 5 个数量级，必须 log10 归一化：1e0 在 [1e-2,1e2] 的对数中点
    assert v[sl["temperature"]][0] == pytest.approx(0.5)


def test_temperature_log_endpoints(encoder):
    lo = encoder.encode(_inactive_cls())  # temperature=0.01
    assert lo[encoder.field_slices["temperature"]][0] == pytest.approx(0.0)
    import dataclasses
    hi_cls = dataclasses.replace(_active_cls(), temperature=100.0)
    hi = encoder.encode(hi_cls)
    assert hi[encoder.field_slices["temperature"]][0] == pytest.approx(1.0)


def test_onehot_and_bool_slots(space, encoder):
    v = encoder.encode(_active_cls())
    sl = encoder.field_slices

    def onehot(name, value):
        seg = v[sl[name]]
        assert seg.sum() == 1.0
        assert int(seg.argmax()) == space.choices(name).index(value)

    onehot("order", 3)
    onehot("norm", "l2")
    onehot("kernel", 5)
    onehot("pool", "max")
    onehot("loss_type", "triplet")
    onehot("sim_func", "cosine")
    assert v[sl["instance"]][0] == 1.0
    assert v[sl["temporal"]][0] == 1.0
    assert v[sl["cross_scale"]][0] == 0.0
    assert v[sl["adj_neighbor"]][0] == 0.0


def test_inactive_gated_fields_encode_to_zeros(encoder):
    v = encoder.encode(_inactive_cls())
    sl = encoder.field_slices
    assert np.all(v[sl["kernel"]] == 0.0)   # 无效档 → 全零 one-hot
    assert np.all(v[sl["pool"]] == 0.0)
    assert v[sl["adj_neighbor"]][0] == 0.0


# ---------------------------------------------------------------- decode

def test_roundtrip_restores_categorical_fields(space, encoder):
    rng = np.random.default_rng(1)
    for _ in range(1000):
        c = space.sample_random(rng)
        d = encoder.decode(encoder.encode(c))
        assert d.order == c.order
        assert d.norm == c.norm
        assert d.kernel == c.kernel
        assert d.pool == c.pool
        assert d.loss_type == c.loss_type
        assert d.sim_func == c.sim_func
        assert d.instance is c.instance
        assert d.temporal is c.temporal
        assert d.cross_scale is c.cross_scale
        assert d.adj_neighbor == c.adj_neighbor


def test_roundtrip_exact_all_fields(space, encoder):
    # 所有字段均为离散档位：数值字段按最近档吸附后应精确还原
    rng = np.random.default_rng(2)
    for _ in range(1000):
        c = space.sample_random(rng)
        assert encoder.decode(encoder.encode(c)) == c


def test_decode_restores_inactive_none(encoder):
    d = encoder.decode(encoder.encode(_inactive_cls()))
    assert d.kernel is None and d.pool is None and d.adj_neighbor is None


def test_decode_rejects_wrong_shape(encoder):
    with pytest.raises(ValueError):
        encoder.decode(np.zeros(encoder.dim + 1, dtype=np.float32))
