"""autocls/search/xspace.py：扩展空间的键 / 编码 / 采样 / 进化算子。"""
import numpy as np
import pytest

from autocls.labeling import cls_key
from autocls.search.xspace import XSpace
from autocls.space.cls_encoder import CLSEncoder

AXES = {"view": {"choices": ["raw", "diff", "psd", "spec"], "default": "raw"},
        "mult": {"choices": [1, 2, 4], "default": 1}}


@pytest.fixture(scope="module")
def xs():
    import importlib
    ac = importlib.import_module("scripts.acts_common")
    cfg = ac.load_acts_config(profile="s2")
    return XSpace(ac.acts_space(cfg), cfg, axes=AXES, is_pruned=ac.is_pruned)


def test_key_back_compat(xs):
    x = xs.sample(np.random.default_rng(0))
    x0 = xs.make(x.cls)
    assert xs.key(x0) == cls_key(x.cls)                         # 默认轴 → 旧键
    assert xs.key(xs.make(x.cls, view="psd")) == cls_key(x.cls) + "|view=psd"
    assert xs.key(xs.make(x.cls, view="psd", mult=4)) == cls_key(x.cls) + "|view=psd,mult=4"


def test_encode_prefix_is_legacy(xs):
    x = xs.make(xs.sample(np.random.default_rng(1)).cls, view="spec", mult=2)
    v = xs.encode(x)
    legacy = CLSEncoder(xs.space, mode="binned").encode(x.cls)
    assert v.shape == (xs.dim,) and xs.dim == legacy.shape[0] + 7
    np.testing.assert_array_equal(v[: legacy.shape[0]], legacy)
    np.testing.assert_array_equal(v[legacy.shape[0]:], [0, 0, 0, 1, 0, 1, 0])


def test_make_rejects_bad(xs):
    c = xs.sample(np.random.default_rng(2)).cls
    with pytest.raises(ValueError):
        xs.make(c, view="nope")
    with pytest.raises(ValueError):
        xs.make(c, colour="red")


def test_sample_covers_axes_and_accepts(xs):
    rng = np.random.default_rng(3)
    got = xs.sample_unique(rng, 200)
    assert len({xs.key(x) for x in got}) == 200
    assert {x.get("view") for x in got} == set(AXES["view"]["choices"])
    assert all(xs.accept(x) for x in got)


def test_evolve_finds_preferred_view(xs):
    rng = np.random.default_rng(4)
    off = xs.enc.dim
    psd = off + 2

    def fit(V):                       # 只奖励 view = psd
        return V[:, psd]
    init = xs.sample_unique(rng, 40)                              # 初始各视图混合（部署时 = Ks 样本按适应度取前 kp）
    pop = xs.evolve(fit, init, rng, kp=10, gens=6)
    assert all(xs.accept(x) for x in pop)
    assert all(x.get("view") == "psd" for x in pop)


def test_dict_roundtrip(xs):
    x = xs.make(xs.sample(np.random.default_rng(5)).cls, view="diff", mult=4)
    assert xs.key(xs.from_dict(x.to_dict())) == xs.key(x)


def test_fixed_axes_restrict_sampling_not_encoding():
    import importlib
    ac = importlib.import_module("scripts.acts_common")
    cfg = ac.load_acts_config(profile="s2")
    leg = XSpace(ac.acts_space(cfg), cfg, axes=AXES, is_pruned=ac.is_pruned, fixed={"view": "raw", "mult": 1})
    rng = np.random.default_rng(6)
    got = leg.sample_unique(rng, 50)
    assert all(x.get("view") == "raw" and x.get("mult") == 1 for x in got)
    assert leg.dim == XSpace(ac.acts_space(cfg), cfg, axes=AXES).dim
    pop = leg.evolve(lambda V: V[:, leg.enc.dim + 2], got, rng, kp=10, gens=4)
    assert all(x.get("view") == "raw" for x in pop)
