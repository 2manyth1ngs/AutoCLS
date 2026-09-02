"""软 label 模块单测（实验 2；results/soft_label_tclsc/PROTOCOL.md §1–§3）。

覆盖：噪声模型拟合与 CDF 性质、配对概率对称性、软配对生成（δ=0 不弃样）、
翻转注入的确定性与对称性、加权任务盲天花板、SoftTCLSCEngine 训练/预测。
模块任务无关（铁律 7）：方向经注入的 margin_fn（测试绑 r_a − r_b）。
"""
import numpy as np
import pytest
import torch
from scipy.stats import t as student_t

from autocls.comparator.soft_labels import (
    NoiseModel,
    SoftPair,
    SoftTCLSCEngine,
    apply_flips,
    fit_noise_model,
    fit_student_t,
    hard_dynamic_pairs_soft,
    flip_decision,
    hard_to_soft,
    info_weight,
    pair_probability,
    soft_dynamic_pairs,
    soft_pairs_all,
    symmetrize,
    task_scale_factor,
    weighted_task_blind_ceiling,
    with_reversed_soft,
)
from autocls.comparator.tclsc_engine import ClsRecord, Pair, dynamic_pairs
from autocls.comparator.tclsc_model import TCLSC
from autocls.task_embed.set_transformer import TaskSetEncoder

CLS_DIM = 4


def margin(ra, rb):
    return ra - rb


def _rec(task_i, rec_i, r, origin="shared"):
    vec = np.zeros(CLS_DIM, dtype=np.float32)
    vec[0], vec[1], vec[2] = r, rec_i, task_i
    return ClsRecord(task_id=f"t{task_i}", key=f"k{task_i}_{rec_i}", origin=origin, vec=vec, r=r)


# ---------------------------------------------------------------- 噪声模型


class TestNoiseModel:
    def test_symmetrize(self):
        s = symmetrize([0.1, -0.2, 0.3])
        assert sorted(s.tolist()) == sorted([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])

    def test_fit_student_t_recovers_scale(self):
        rng = np.random.default_rng(0)
        x = student_t.rvs(df=4, scale=0.03, size=4000, random_state=rng)
        df, scale = fit_student_t(x)
        assert 0.025 < scale < 0.035
        assert 2 <= df <= 100

    def test_fit_student_t_df_clipped(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.02, size=2000)  # 高斯 → df 应撞上界
        df, _ = fit_student_t(x, df_bounds=(2, 10))
        assert 2 <= df <= 10

    def test_fit_noise_model_regimes(self):
        rng = np.random.default_rng(2)
        nd = rng.normal(0, 0.02, 40).tolist()
        dest = rng.normal(0, 0.08, 6).tolist()
        m = fit_noise_model({"nd": nd, "dest": dest}, kind="student_t", min_regime_n=5)
        assert set(m.scale) == {"nd", "dest"}
        assert m.scale["dest"] > m.scale["nd"]
        # 破坏制式样本不足 → 合并为单制式
        m2 = fit_noise_model({"nd": nd, "dest": dest[:3]}, kind="student_t", min_regime_n=5)
        assert set(m2.scale) == {"nd"}

    def test_cdf_properties_student_t(self):
        m = NoiseModel(kind="student_t", df=5.0, scale={"nd": 0.03}, samples=None)
        s = m.scale["nd"]
        assert m.cdf(np.array([0.0]), s)[0] == pytest.approx(0.5)
        big = m.cdf(np.array([1.0]), s)[0]
        assert big > 0.999
        d = np.linspace(-0.2, 0.2, 41)
        c = m.cdf(d, s)
        assert np.all(np.diff(c) > 0)
        assert np.allclose(c + c[::-1], 1.0, atol=1e-9)  # 对称

    def test_cdf_properties_empirical(self):
        m = NoiseModel(kind="empirical", df=None, scale={"nd": 1.0},
                       samples=symmetrize([0.01, 0.02, 0.05, 0.1]))
        assert m.cdf(np.array([0.0]), 1.0)[0] == pytest.approx(0.5)
        assert m.cdf(np.array([0.5]), 1.0)[0] == pytest.approx(1.0)
        assert m.cdf(np.array([-0.5]), 1.0)[0] == pytest.approx(0.0)
        d = np.linspace(-0.2, 0.2, 41)
        c = m.cdf(d, 1.0)
        assert np.all(np.diff(c) >= 0)
        assert np.allclose(c + c[::-1], 1.0, atol=1e-9)
        # 尺度因子缩放样本
        assert m.cdf(np.array([0.05]), 2.0)[0] < m.cdf(np.array([0.05]), 1.0)[0]

    def test_task_scale_factor(self):
        assert task_scale_factor(n_eval=300, n_ref=300, kappa=0.6) == pytest.approx(1.0)
        assert task_scale_factor(n_eval=75, n_ref=300, kappa=1.0) == pytest.approx(2.0)
        assert task_scale_factor(n_eval=75, n_ref=300, kappa=0.0) == pytest.approx(1.0)
        f = task_scale_factor(n_eval=150, n_ref=300, kappa=0.6)
        assert 1.0 < f < np.sqrt(2)

    def test_pair_probability_symmetry_and_mixed_scale(self):
        m = NoiseModel(kind="student_t", df=4.0, scale={"nd": 0.02, "dest": 0.06}, samples=None)
        p = pair_probability(0.03, m, scale_a=0.02, scale_b=0.02)
        q = pair_probability(-0.03, m, scale_a=0.02, scale_b=0.02)
        assert p == pytest.approx(1 - q)
        assert 0.5 < p < 1
        # 混合制式 s_pair = sqrt((sa²+sb²)/2) 介于两者之间 → 概率介于两者之间
        p_nd = pair_probability(0.03, m, 0.02, 0.02)
        p_dd = pair_probability(0.03, m, 0.06, 0.06)
        p_mix = pair_probability(0.03, m, 0.02, 0.06)
        assert p_dd < p_mix < p_nd

    def test_info_weight(self):
        assert info_weight(0.5) == 0
        assert info_weight(1.0) == pytest.approx(1.0)
        assert info_weight(0.25) == pytest.approx(0.5)
        assert np.allclose(info_weight(np.array([0.0, 0.5, 0.75])), [1.0, 0.0, 0.5])


# ---------------------------------------------------------------- 配对


def const_prob(p_pos: float):
    """prob_fn(task_id, key_a, key_b, delta) → 依 delta 符号给固定概率。"""
    def f(task_id, key_a, key_b, delta):
        if delta > 0:
            return p_pos
        if delta < 0:
            return 1 - p_pos
        return 0.5
    return f


class TestSoftPairs:
    @pytest.fixture
    def records(self):
        return [_rec(0, 0, 0.9), _rec(0, 1, 0.5), _rec(0, 2, 0.88, "random"),
                _rec(1, 3, 0.3), _rec(1, 4, 0.3), _rec(1, 5, 0.75, "random")]

    def test_soft_pairs_all_exhaustive_no_delta_drop(self, records):
        pairs = soft_pairs_all(records, margin, const_prob(0.8))
        # 每任务 C(3,2)=3 个无序对，含 |Δ|=0.02 的近平局与 Δ=0 的严格平局
        assert len(pairs) == 6
        for p in pairs:
            assert p.vec_a[2] == p.vec_b[2]
            assert p.w == pytest.approx(abs(2 * p.p - 1))
        tie = [p for p in pairs if p.task_id == "t1" and {p.key_a, p.key_b} == {"k1_3", "k1_4"}][0]
        assert tie.p == 0.5 and tie.w == 0 and tie.y_hard in (0, 1)
        pos = [p for p in pairs if p.key_a == "k0_0" and p.key_b == "k0_1"][0]
        assert pos.p == 0.8 and pos.y_hard == 1 and pos.kind == "shared-shared"

    def test_soft_dynamic_pairs_one_partner_each_and_origin_filter(self, records):
        for seed in range(6):
            pairs = soft_dynamic_pairs(records, margin, const_prob(0.9),
                                       rng=np.random.default_rng(seed))
            assert len(pairs) <= len(records)
            for p in pairs:
                assert p.vec_a[2] == p.vec_b[2] and p.key_a != p.key_b
                assert p.y_hard == int(margin(float(p.vec_a[0]), float(p.vec_b[0])) > 0) or p.p == 0.5
        only_shared = [p for s in range(6)
                       for p in soft_dynamic_pairs(records, margin, const_prob(0.9),
                                                   rng=np.random.default_rng(s),
                                                   origins=("shared",))]
        assert only_shared and all(p.kind == "shared-shared" for p in only_shared)

    def test_hard_dynamic_pairs_soft_matches_dynamic_pairs(self, records):
        for seed in range(6):
            hard = dynamic_pairs(records, lambda a, b: a > b, delta={"t0": 0.05, "t1": 0.1},
                                 rng=np.random.default_rng(seed))
            soft = hard_dynamic_pairs_soft(records, margin, delta={"t0": 0.05, "t1": 0.1},
                                           rng=np.random.default_rng(seed))
            assert len(hard) == len(soft)
            for h, s in zip(hard, soft):
                assert h.task_id == s.task_id and h.y == s.y_hard == int(s.p) and s.w == 1.0
                assert np.array_equal(h.vec_a, s.vec_a) and s.key_a != s.key_b
                assert s.key_a and s.key_b

    def test_hard_to_soft_and_reverse(self, records):
        hard = dynamic_pairs(records, lambda a, b: a > b, delta=0.05,
                             rng=np.random.default_rng(0))
        soft = hard_to_soft(hard)
        assert len(soft) == len(hard)
        for h, s in zip(hard, soft):
            assert s.p == float(h.y) and s.w == 1.0 and s.y_hard == h.y and s.task_id == h.task_id
        rev = with_reversed_soft(soft)
        assert len(rev) == 2 * len(soft)
        for s, r in zip(soft, rev[len(soft):]):
            assert r.p == pytest.approx(1 - s.p) and r.y_hard == 1 - s.y_hard
            assert np.array_equal(r.vec_a, s.vec_b) and r.key_a == s.key_b

    def test_flip_decision_deterministic_symmetric_rate(self):
        assert flip_decision(0, "t0", "a", "b", 0.3) == flip_decision(0, "t0", "b", "a", 0.3)
        assert flip_decision(0, "t0", "a", "b", 0.3) == flip_decision(0, "t0", "a", "b", 0.3)
        assert not flip_decision(0, "t0", "a", "b", 0.0)
        assert flip_decision(0, "t0", "a", "b", 1.0)
        hits = sum(flip_decision(1, f"t{i % 7}", f"a{i}", f"b{i}", 0.3) for i in range(4000))
        assert 0.26 < hits / 4000 < 0.34

    def test_apply_flips(self, records):
        soft = soft_pairs_all(records, margin, const_prob(0.8))
        flipped = apply_flips(soft, seed=0, rate=1.0)
        for s, f in zip(soft, flipped):
            assert f.y_hard == 1 - s.y_hard and f.p == pytest.approx(1 - s.p) and f.w == s.w
        assert apply_flips(soft, seed=0, rate=0.0) == soft


# ---------------------------------------------------------------- 加权天花板


class TestWeightedCeiling:
    def test_all_agree(self):
        votes = {("a", "b"): [("t0", 1, 1.0), ("t1", 1, 0.5), ("t2", 1, 0.2)]}
        r = weighted_task_blind_ceiling(votes)
        assert r["ceiling"] == pytest.approx(1.0) and r["mass"] == pytest.approx(1.7)

    def test_balanced_disagreement(self):
        votes = {("a", "b"): [("t0", 1, 1.0), ("t1", 0, 1.0)],
                 ("a", "c"): [("t0", 0, 0.3), ("t1", 1, 0.3)]}
        assert weighted_task_blind_ceiling(votes)["ceiling"] == pytest.approx(0.5)

    def test_weights_matter_and_filter(self):
        votes = {("a", "b"): [("HW_0", 1, 1.0), ("UM_0", 0, 0.1), ("UM_1", 0, 0.1)]}
        r = weighted_task_blind_ceiling(votes)
        assert r["ceiling"] == pytest.approx(1.0 / 1.2)
        r_um = weighted_task_blind_ceiling(votes, task_filter=lambda t: t.startswith("UM"))
        assert r_um["ceiling"] == pytest.approx(1.0) and r_um["mass"] == pytest.approx(0.2)
        assert weighted_task_blind_ceiling({})["ceiling"] is None


# ---------------------------------------------------------------- 引擎


DIN, NZ = 5, 6


def _windows(seed, n=4, t=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, t, DIN, generator=g).numpy()


class TestSoftEngine:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        model = TCLSC(cls_dim=CLS_DIM, set_encoder=TaskSetEncoder(dim_input=DIN, nz=NZ),
                      embedding_dim=8, fz=8)
        windows = {"t0": _windows(0), "t1": _windows(1)}
        engine = SoftTCLSCEngine(model, windows, lr=1e-2, batch_size=8, seed=0)
        rng = np.random.default_rng(0)
        recs = [_rec(ti, ti * 10 + i, float(r)) for ti in range(2)
                for i, r in enumerate(rng.uniform(0.3, 0.9, 6))]
        return engine, recs

    def test_train_soft_reduces_weighted_loss(self, setup):
        engine, recs = setup

        def sampler(epoch, rng):
            return soft_dynamic_pairs(recs, margin, const_prob(0.95), rng=rng)

        hist = engine.train_soft(sampler, epochs=30)
        assert len(hist) == 30
        assert hist[-1]["loss"] < hist[0]["loss"]
        assert all(np.isfinite(h["loss"]) for h in hist)
        assert "n_pairs" in hist[0] and "w_mean" in hist[0]

    def test_zero_weight_pairs_do_not_train(self, setup):
        engine, recs = setup
        before = [p.detach().clone() for p in engine.model.parameters()]

        def sampler(epoch, rng):
            ps = soft_dynamic_pairs(recs, margin, const_prob(0.9), rng=rng)
            return [SoftPair(**{**p.__dict__, "w": 0.0}) for p in ps]

        engine.train_soft(sampler, epochs=2)
        after = list(engine.model.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after))

    def test_predict_matches_evaluate_and_supports_modes(self, setup):
        engine, recs = setup
        soft = with_reversed_soft(soft_pairs_all(recs, margin, const_prob(0.9)))
        probs = engine.predict(soft)
        assert probs.shape == (len(soft),) and np.all((probs >= 0) & (probs <= 1))
        hard = [Pair(task_id=s.task_id, vec_a=s.vec_a, vec_b=s.vec_b, y=s.y_hard, kind=s.kind)
                for s in soft]
        ev = engine.evaluate(hard, return_correct=True)
        assert np.array_equal(ev["correct"], (np.round(probs) == np.array([s.y_hard for s in soft])))
        p_shuf = engine.predict(soft, e_mode="shuffle", seed=0)
        p_zero = engine.predict(soft, e_mode="zero")
        assert p_shuf.shape == probs.shape and p_zero.shape == probs.shape
        assert not np.allclose(p_zero, probs)
        p_assign = engine.predict(soft, e_mode="assign", e_assignment={"t0": "t1", "t1": "t0"})
        assert np.allclose(p_assign, p_shuf)  # 2 任务下 derangement 唯一
