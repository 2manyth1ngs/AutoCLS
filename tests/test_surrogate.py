"""MX-4 WP2：字段结构化 CLS 代理（autocls/search/surrogate.py）测试——先于实现（铁律 5）。

代理 = 在 CLS 字段编码（CLSEncoder，scalar 或 binned）上拟合"调用方给定的目标 y"
（脚本层已做 within-(task, iters) 归一化；模块本身不知道任务，铁律 7）。
验收：
- fit/predict 能从 one-hot 字段上恢复可加的档位效应；
- bagging=1 时 std 全零，bagging>1 时给出非负、有限的不确定性；
- pairwise_matrix 产出 rank.py 可直接消费的 P 矩阵（对角 NaN、P+Pᵀ=1）；
- field_hamming / in_hull：训练点距离 0；差 k 个字段的点距离 k；
- 模块不引入任务类型 / 数据集概念（铁律 7）。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from autocls.search.rank import symmetrize, wins_from_scores
from autocls.search.surrogate import (
    FieldSurrogate, field_hamming, in_hull, pairwise_matrix,
)
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def binned(space):
    return CLSEncoder(space, mode="binned")


def _synthetic(space, binned, n: int, seed: int, noise: float = 0.05):
    """随机 CLS 的 one-hot 编码 + 可加档位效应目标（真效应固定种子 123）。"""
    rng = np.random.default_rng(seed)
    X = np.stack([binned.encode(space.sample_random(rng)) for _ in range(n)])
    w = np.random.default_rng(123).normal(size=X.shape[1])
    y_true = X @ w
    return X, y_true + rng.normal(scale=noise, size=n)


# ---------------------------------------------------------------- fit / predict
@pytest.mark.parametrize("kind", ["ridge", "hgb"])
def test_recovers_additive_field_effects_on_heldout(space, binned, kind):
    Xtr, ytr = _synthetic(space, binned, 600, seed=0)
    Xte, yte = _synthetic(space, binned, 200, seed=1, noise=0.0)
    m = FieldSurrogate(kind=kind, seed=0).fit(Xtr, ytr)
    mean, std = m.predict(Xte)
    assert mean.shape == (200,) and std.shape == (200,)
    assert np.corrcoef(mean, yte)[0, 1] > 0.9


def test_std_is_zero_without_bagging(space, binned):
    X, y = _synthetic(space, binned, 100, seed=2)
    _mean, std = FieldSurrogate(kind="ridge", bagging=1).fit(X, y).predict(X)
    assert np.all(std == 0.0)


def test_bagging_gives_finite_nonnegative_uncertainty(space, binned):
    X, y = _synthetic(space, binned, 200, seed=3)
    m = FieldSurrogate(kind="ridge", bagging=5, seed=0).fit(X, y)
    mean, std = m.predict(X)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    assert np.all(std >= 0.0) and std.max() > 0.0


def test_bagging_is_deterministic_under_seed(space, binned):
    X, y = _synthetic(space, binned, 150, seed=4)
    a = FieldSurrogate(kind="hgb", bagging=3, seed=7).fit(X, y).predict(X)
    b = FieldSurrogate(kind="hgb", bagging=3, seed=7).fit(X, y).predict(X)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        FieldSurrogate(kind="gp")


def test_predict_before_fit_rejected(space, binned):
    X, _y = _synthetic(space, binned, 5, seed=5)
    with pytest.raises(RuntimeError):
        FieldSurrogate(kind="ridge").predict(X)


# ---------------------------------------------------------------- pairwise matrix
def test_pairwise_matrix_shape_diag_and_antisymmetry():
    s = np.array([0.3, -1.0, 2.0, 0.3])
    P = pairwise_matrix(s)
    assert P.shape == (4, 4)
    assert np.all(np.isnan(np.diag(P)))
    off = ~np.eye(4, dtype=bool)
    assert np.allclose((P + P.T)[off], 1.0)
    assert P[2, 1] > 0.5 > P[1, 2]           # 高分优于低分
    assert P[0, 3] == pytest.approx(0.5)     # 同分无倾向


def test_pairwise_matrix_tau_sharpens():
    s = np.array([0.0, 1.0])
    assert pairwise_matrix(s, tau=0.1)[1, 0] > pairwise_matrix(s, tau=10.0)[1, 0] > 0.5


def test_pairwise_matrix_is_drop_in_for_rank_pipeline():
    s = np.array([0.1, 0.9, -0.4, 0.5])
    wins = wins_from_scores(symmetrize(pairwise_matrix(s)))
    assert list(np.argsort(-wins)) == list(np.argsort(-s))


# ---------------------------------------------------------------- 信任域
def test_field_hamming_zero_for_training_rows(space, binned):
    X, _y = _synthetic(space, binned, 50, seed=6)
    d = field_hamming(X, X, binned.field_slices)
    assert d.shape == (50,) and np.all(d == 0)


def test_field_hamming_counts_differing_fields(space, binned):
    rng = np.random.default_rng(8)
    base = space.sample_random(rng)
    import dataclasses
    other = dataclasses.replace(
        base, jitter=0.95 if base.jitter != 0.95 else 0.0,
        norm="l2" if base.norm != "l2" else "none",
        temperature=100.0 if base.temperature != 100.0 else 0.01,
    )
    Xtr = binned.encode(base)[None, :]
    d = field_hamming(binned.encode(other)[None, :], Xtr, binned.field_slices)
    assert d[0] == 3


def test_in_hull_radius(space, binned):
    rng = np.random.default_rng(9)
    base = space.sample_random(rng)
    import dataclasses
    far = dataclasses.replace(base, jitter=0.95 if base.jitter != 0.95 else 0.0,
                              crop=0.95 if base.crop != 0.95 else 0.0)
    Xtr = binned.encode(base)[None, :]
    Xq = np.stack([binned.encode(base), binned.encode(far)])
    assert list(in_hull(Xq, Xtr, binned.field_slices, radius=0)) == [True, False]
    assert list(in_hull(Xq, Xtr, binned.field_slices, radius=2)) == [True, True]


# ---------------------------------------------------------------- 铁律 7
def test_module_has_no_task_or_dataset_concepts():
    src = (ROOT / "autocls" / "search" / "surrogate.py").read_text(encoding="utf-8")
    assert "task_types" not in src
    assert not re.search(r"\b(Handwriting|UniMiB|classification|forecasting)\b", src)
