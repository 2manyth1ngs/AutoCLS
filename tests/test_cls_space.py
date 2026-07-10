"""M0 模块 1 测试：CLS 策略空间（TECHNICAL_DESIGN.md 第 4/6 节，AutoCL Table 1）。

先于实现编写（铁律 5）。验收：
- 空间规模数量级 ≈ 3×10¹²
- 随机采样 10000 个全部通过 validate
"""
import dataclasses

import numpy as np
import pytest

from autocls.space.cls_space import CLS, CLSSpace


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


# ---------------------------------------------------------------- 结构

def test_nineteen_fields_match_dataclass(space):
    assert len(space.field_names) == 19
    assert space.field_names == [f.name for f in dataclasses.fields(CLS)]


def test_choices_come_from_yaml(space):
    # AutoCL Table 1：6 增强比率 11 档、order 5 种、temperature 5 档
    assert len(space.choices("resize")) == 11
    assert 0.0 in space.choices("resize") and 0.95 in space.choices("resize")
    assert len(space.choices("order")) == 5
    assert len(space.choices("temperature")) == 5
    assert space.choices("instance") == [True]


# ---------------------------------------------------------------- 空间规模

def test_space_size_order_of_magnitude(space):
    # 设计文档第 4 节：约 3×10¹²（sanity check 按数量级断言）
    size = space.space_size()
    assert isinstance(size, int)
    assert 1e12 <= size < 1e13


def _small_config():
    """手算可验的迷你空间：非门控自由字段积 1024，门控组合 3 → 3072。"""
    ratio = [0.0, 0.5]
    fields = [
        {"name": n, "type": "numeric", "choices": ratio}
        for n in ["resize", "rescale", "jitter", "point_mask", "freq_mask", "crop"]
    ] + [
        {"name": "order", "type": "categorical", "choices": [0, 1]},
        {"name": "emb_jitter", "type": "numeric", "choices": ratio},
        {"name": "emb_mask", "type": "numeric", "choices": ratio},
        {"name": "norm", "type": "categorical", "choices": ["none"]},
        {"name": "instance", "type": "bool", "choices": [True]},
        {"name": "temporal", "type": "bool", "choices": [True, False]},
        {"name": "cross_scale", "type": "bool", "choices": [False]},
        {"name": "kernel", "type": "categorical", "choices": [2]},
        {"name": "pool", "type": "categorical", "choices": ["avg"]},
        {"name": "adj_neighbor", "type": "bool", "choices": [True, False]},
        {"name": "loss_type", "type": "categorical", "choices": ["info_nce"]},
        {"name": "sim_func", "type": "categorical", "choices": ["dot"]},
        {"name": "temperature", "type": "log_numeric", "choices": [0.1, 1.0]},
    ]
    constraints = [{
        "when_any": ["temporal", "cross_scale"],
        "gated_fields": ["kernel", "pool", "adj_neighbor"],
        "inactive_value": None,
    }]
    return {"fields": fields, "constraints": constraints}


def test_space_size_exact_on_small_config():
    # 自由字段：2^6 × 2 × 2^2 × 1 × 1 × 1 × 1 × 2 = 1024
    # 门控：temporal∈{T,F} × cross_scale∈{F} → T 时 1×1×2=2，F 时无效档记 1 → 共 3
    assert CLSSpace(_small_config()).space_size() == 1024 * 3


# ---------------------------------------------------------------- 采样

def test_sample_10000_all_valid(space):
    rng = np.random.default_rng(0)
    for _ in range(10000):
        assert space.validate(space.sample_random(rng))


def test_sample_values_within_yaml_choices(space):
    rng = np.random.default_rng(1)
    for _ in range(200):
        c = space.sample_random(rng)
        assert c.instance is True
        assert c.resize in space.choices("resize")
        assert c.order in space.choices("order")
        assert c.norm in space.choices("norm")
        assert c.loss_type in space.choices("loss_type")
        assert c.sim_func in space.choices("sim_func")
        assert c.temperature in space.choices("temperature")


def test_gated_fields_none_iff_pair_dims_off(space):
    rng = np.random.default_rng(2)
    saw_inactive = saw_active = False
    for _ in range(500):
        c = space.sample_random(rng)
        if not (c.temporal or c.cross_scale):
            saw_inactive = True
            assert c.kernel is None and c.pool is None and c.adj_neighbor is None
        else:
            saw_active = True
            assert c.kernel in space.choices("kernel")
            assert c.pool in space.choices("pool")
            assert c.adj_neighbor in space.choices("adj_neighbor")
    assert saw_inactive and saw_active  # 两种情形都要被覆盖到


# ---------------------------------------------------------------- 合法性

def _sample_where(space, rng, pred, tries=1000):
    for _ in range(tries):
        c = space.sample_random(rng)
        if pred(c):
            return c
    raise AssertionError("未采到满足条件的样本")


def test_validate_rejects_invalid(space):
    rng = np.random.default_rng(3)
    active = _sample_where(space, rng, lambda c: c.temporal or c.cross_scale)
    inactive = _sample_where(space, rng, lambda c: not (c.temporal or c.cross_scale))

    # 取值不在 yaml 档位内
    assert not space.validate(dataclasses.replace(active, kernel=7))
    assert not space.validate(dataclasses.replace(active, norm="batch_norm"))
    assert not space.validate(dataclasses.replace(active, resize=0.33))
    assert not space.validate(dataclasses.replace(active, temperature=3.0))
    # instance 恒 True
    assert not space.validate(dataclasses.replace(active, instance=False))
    # 门控开启时不得为无效档
    assert not space.validate(dataclasses.replace(active, kernel=None))
    # 门控关闭时必须为无效档
    assert not space.validate(dataclasses.replace(inactive, kernel=2))
    assert not space.validate(dataclasses.replace(inactive, pool="avg"))
