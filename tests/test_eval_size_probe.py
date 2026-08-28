"""评估集规模探针（scripts/10_eval_size_probe.py）纯逻辑测试。

期望值全部手工推导（不以被测代码为 oracle）：
- 种子派生必须与 2c 噪声探针逐字同口径（δ=0.110 的标定协议；逐位对账的前提）；
- 翻转统计 q/net/方向不平衡在小向量上手算；
- 子集缩放机制在合成判对向量上的解析行为：对称独立翻转 → β≈0.5；
  单向系统性翻转 → β≈0（这是主判据两端的生成模型）；
- β 拟合在精确幂律点上无偏；判据档位边界。
"""
import importlib

import numpy as np
import pytest

from autocls.labeling import run_seed_for

probe = importlib.import_module("scripts.10_eval_size_probe")
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")


class TestSeedsFor:
    def test_rep0_is_canonical(self):
        assert probe.seeds_for("t0", "k0", 0) == _c2.canonical_seeds("t0", "k0")

    def test_rep12_are_2c_noise_seeds(self):
        for rep in (1, 2):
            w, sp = probe.seeds_for("t0", "k0", rep)
            assert w == run_seed_for("t0", f"k0:noise{rep}")
            assert sp == run_seed_for("t0", "train_eval_split")


class TestParseCorrect:
    def test_roundtrip(self):
        v = probe.parse_correct("1101")
        assert v.tolist() == [1, 1, 0, 1]
        assert v.dtype == np.int8


class TestFlipStats:
    def test_hand_computed(self):
        # a=[1,1,0,0], b=[1,0,1,0]：2 处不同 → q=0.5；acc 均 0.5 → net=0；
        # d=a−b=[0,+1,−1,0] → 正负各 1，方向不平衡 0
        out = probe.flip_stats(np.array([1, 1, 0, 0], dtype=np.int8),
                               np.array([1, 0, 1, 0], dtype=np.int8))
        assert out["q"] == pytest.approx(0.5)
        assert out["net"] == pytest.approx(0.0)
        assert (out["flips_pos"], out["flips_neg"]) == (1, 1)
        assert out["imbalance"] == pytest.approx(0.0)

    def test_one_directional(self):
        # b 把 a 的两个 1 变 0 → q=0.5、net=0.5、不平衡 1.0
        out = probe.flip_stats(np.array([1, 1, 1, 1], dtype=np.int8),
                               np.array([0, 0, 1, 1], dtype=np.int8))
        assert out["net"] == pytest.approx(0.5)
        assert out["imbalance"] == pytest.approx(1.0)

    def test_no_flips(self):
        out = probe.flip_stats(np.ones(4, dtype=np.int8), np.ones(4, dtype=np.int8))
        assert out["q"] == 0.0 and out["imbalance"] is None


class TestFitBeta:
    def test_exact_power_law(self):
        # med ∝ n^{-0.5}：(100, 0.1) 与 (25, 0.2) → β=0.5 精确
        assert probe.fit_beta([(100, 0.1), (25, 0.2)]) == pytest.approx(0.5)

    def test_flat_is_zero(self):
        assert probe.fit_beta([(100, 0.1), (25, 0.1)]) == pytest.approx(0.0)


class TestScalingVerdict:
    def test_bands(self):
        assert probe.scaling_verdict(0.50) == "confirmed"
        assert probe.scaling_verdict(0.35) == "confirmed"
        assert probe.scaling_verdict(0.30) == "mixed"
        assert probe.scaling_verdict(0.19) == "refuted"
        assert probe.scaling_verdict(0.70) == "anomalous"


def _pairs(n, n_pairs, q, systematic, seed=0):
    """合成判对向量对。对称：p=0.5 基线、每位独立以 q 翻转（E[d]=0，净差
    ~√(q/n)）；系统：前 ceil(q·n) 个 a=1 的位单向置 0（净差≈常数）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_pairs):
        a = (rng.random(n) < 0.5).astype(np.int8)
        b = a.copy()
        if systematic:
            ones = np.where(a == 1)[0][: int(np.ceil(q * n))]
            b[ones] = 0
        else:
            flip = rng.random(n) < q
            b[flip] = 1 - b[flip]
        out[f"c{i}"] = [a, b]
    return out


class TestSubsetScalingSynthetic:
    FRACS = (1.0, 0.5, 0.25)

    def _beta(self, systematic):
        n = 128
        vecs = _pairs(n, n_pairs=150, q=0.25, systematic=systematic)
        vals = probe.subset_noise_values(vecs, n, self.FRACS, n_draws=60,
                                         rng=np.random.default_rng(1))
        pts = [(int(round(f * n)), float(np.median(np.concatenate(
            list(vals[f].values()))))) for f in self.FRACS]
        return probe.fit_beta(pts)

    def test_symmetric_flips_give_half(self):
        assert self._beta(systematic=False) == pytest.approx(0.5, abs=0.15)

    def test_systematic_flips_give_flat(self):
        assert self._beta(systematic=True) == pytest.approx(0.0, abs=0.12)

    def test_full_fraction_single_draw(self):
        vecs = _pairs(32, n_pairs=3, q=0.25, systematic=False)
        vals = probe.subset_noise_values(vecs, 32, (1.0,), n_draws=50,
                                         rng=np.random.default_rng(2))
        # f=1 只有全集一份（无重复抽样）：每 cls 1 对 × 1 draw = 1 值
        assert all(len(v) == 1 for v in vals[1.0].values())
