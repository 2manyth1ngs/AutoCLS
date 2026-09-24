"""WP17-ACTS S3-A（results/acts/s3a/PROTOCOL.md）——先于实现。覆盖：训练池任务选择、低秩交互模型（拟合 / 投影 / 行去均值不变性 / 符号对齐）、
ridge + 内层 LOSO 选 α、regret@k、LOSO 不泄漏、种下信号端到端、源聚类 bootstrap、门分支、元特征解析、命令行入口。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _g():
    return importlib.import_module("scripts.84_acts_s3a")


def _planted(n_src=12, per=8, n_cfg=30, noise=0.2, seed=0, within=0.3):
    """源级轴：a(t) = 源坐标 + 源内扰动；Z = μ + a·b + 噪声；end0/end1 = Z 的两份带噪 acc。"""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=n_cfg)
    b = rng.normal(size=n_cfg); b /= np.linalg.norm(b)
    src_pos = rng.normal(scale=2.0, size=n_src)
    groups = np.repeat([f"S{i}" for i in range(n_src)], per)
    a = np.repeat(src_pos, per) + rng.normal(scale=within, size=n_src * per)
    Z = mu[None, :] + a[:, None] * b[None, :] * 3.0 + rng.normal(scale=noise, size=(n_src * per, n_cfg))
    end0 = 0.5 + 0.05 * Z + rng.normal(scale=0.005, size=Z.shape)
    end1 = 0.5 + 0.05 * Z + rng.normal(scale=0.005, size=Z.shape)
    return Z, groups, a, b, end0, end1


class TestPool:
    def test_pool_tasks(self):
        g = _g()
        meta = {f"{d}_a{i:02d}": {"dataset": d} for d in ("A", "E", "H") for i in range(16)}
        tasks = g.pool_tasks(list(meta), meta, pool=["A", "E"], demoted_to_8=["E"])
        assert [t for t in tasks if t.startswith("E_")] == [f"E_a{i:02d}" for i in range(8)]
        assert len([t for t in tasks if t.startswith("A_")]) == 16 and not any(t.startswith("H_") for t in tasks)

    def test_meta_features(self):
        g = _g()
        m = {"n_train": 100, "T": 50, "C": 3, "k": 4, "majority": 0.3, "tag": "c2w1n3"}
        x = g.meta_features(m)
        assert x.shape == (8,)
        assert np.allclose(x[:5], [np.log(100), np.log(50), np.log(3), np.log(4), 0.3]) and list(x[5:]) == [2, 1, 3]


class TestLowRank:
    def test_recovers_planted_axis(self):
        g = _g()
        Z, groups, a, b, _e0, _e1 = _planted(noise=0.05)
        mu, B = g.lowrank_fit(Z, K=1)
        assert B.shape == (30, 1)
        assert abs(np.corrcoef(B[:, 0], b)[0, 1]) > 0.99
        assert np.allclose(B.T @ B, np.eye(1), atol=1e-8)

    def test_projection_is_lstsq_and_row_shift_invariant(self):
        g = _g()
        Z, *_ = _planted()
        mu, B = g.lowrank_fit(Z, K=3)
        A = g.project(Z, mu, B)
        R = g.residual(Z, mu)
        lst = np.linalg.lstsq(B, R.T, rcond=None)[0].T
        assert np.allclose(A, lst, atol=1e-8)
        A2 = g.project(Z + 7.0, mu, B)          # 任务主效应不改轴坐标
        assert np.allclose(A, A2, atol=1e-8)

    def test_scores_rank_equivalent_to_mu_when_a_zero(self):
        g = _g()
        Z, *_ = _planted()
        mu, B = g.lowrank_fit(Z, K=2)
        S = g.scores_from(mu, B, np.zeros((4, 2)))
        assert np.allclose(S, np.tile(mu, (4, 1)))

    def test_sign_alignment(self):
        g = _g()
        Z, *_ = _planted()
        mu, B = g.lowrank_fit(Z, K=2)
        A = g.project(Z, mu, B)
        B2, A2 = g.align_sign(-B, -A, B[:, 0])
        assert np.allclose(B2[:, 0], B[:, 0]) and np.allclose(A2[:, 0], A[:, 0])
        assert np.allclose(B2[:, 1], -B[:, 1])          # 只对齐第一分量


class TestRegret:
    def test_regret_at_k(self):
        g = _g()
        end0 = np.array([[0.1, 0.9, 0.5, 0.2]])
        end1 = np.array([[0.2, 0.8, 0.6, 0.1]])
        S = np.array([[4.0, 1.0, 3.0, 2.0]])        # 排序 0 > 2 > 3 > 1
        assert np.isclose(g.regret_at_k(S, end0, end1, 1)[0], 0.8 - 0.2)   # oracle 按 end0 = 列 1，end1 评 0.8；pick 列 0
        assert np.isclose(g.regret_at_k(S, end0, end1, 3)[0], 0.8 - 0.6)   # 前 3 = {0,2,3}，end0 最好 = 列 2
        assert np.isclose(g.regret_at_k(S, end0, end1, 4)[0], 0.0)


class TestRidge:
    def test_alpha_selection_signal_vs_noise(self):
        g = _g()
        rng = np.random.default_rng(1)
        groups = np.repeat(np.arange(10), 6)
        X = rng.normal(size=(60, 3))
        y_sig = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=60)
        y_noise = rng.normal(size=60)
        alphas = [0.1, 1, 10, 100, 1000]
        assert g.choose_alpha(X, y_sig, groups, alphas) <= 1
        assert g.choose_alpha(X, y_noise, groups, alphas) >= 100

    def test_ridge_predict_standardized(self):
        g = _g()
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 2)) * np.array([1000.0, 0.001])
        y = X[:, 0] / 1000.0 + X[:, 1] * 1000.0
        m = g.ridge_fit(X, y, alpha=1e-6)
        assert np.allclose(g.ridge_predict(m, X), y, atol=1e-3)

    def test_constant_feature_is_safe(self):
        g = _g()
        X = np.ones((20, 2)); X[:, 1] = np.arange(20)
        m = g.ridge_fit(X, np.arange(20.0), alpha=1.0)
        assert np.all(np.isfinite(g.ridge_predict(m, X)))


class TestLOSO:
    def _run(self, Z, groups, end0, end1, feats, **kw):
        g = _g()
        return g.loso_evaluate(Z, groups, end0, end1, {"f": feats}, Ks=(1, 2, 3), alphas=(0.1, 1, 10, 100, 1000), **kw)

    def test_no_leak_heldout_labels(self):
        """留出源的 Z / end 改动不影响该源的 â（真 a 与 regret 会变）。b_ref 是读数的符号约定（固定传入），不是拟合量。"""
        Z, groups, a, b, e0, e1 = _planted(seed=3)
        X = a[:, None] + np.random.default_rng(0).normal(scale=0.3, size=(len(a), 1))
        r1 = self._run(Z, groups, e0, e1, X, b_ref=b)
        Z2 = Z.copy(); m = groups == "S0"
        Z2[m] = np.random.default_rng(9).normal(size=Z2[m].shape)
        r2 = self._run(Z2, groups, e0, e1, X, b_ref=b)
        assert np.allclose(r1["f"]["a_hat"][m, 0], r2["f"]["a_hat"][m, 0])
        assert not np.allclose(r1["f"]["a_true"][m, 0], r2["f"]["a_true"][m, 0])

    def test_planted_signal_end_to_end(self):
        Z, groups, a, b, e0, e1 = _planted(seed=4)
        good = a[:, None] + np.random.default_rng(0).normal(scale=0.2, size=(len(a), 1))
        bad = np.random.default_rng(1).normal(size=(len(a), 2))
        g = _g()
        r = g.loso_evaluate(Z, groups, e0, e1, {"good": good, "bad": bad}, Ks=(1, 2, 3), alphas=(0.1, 1, 10, 100, 1000))
        assert r["good"]["rho_a1"] > 0.9
        assert abs(r["bad"]["rho_a1"]) < 0.4
        assert r["good"]["delta_regret3"] > 0.0
        assert r["good"]["delta_regret3"] > r["bad"]["delta_regret3"]
        assert r["_oracle"]["delta_regret3"] >= r["good"]["delta_regret3"] - 1e-9
        assert set(r["good"]["k_chosen"]) <= {1, 2, 3} and len(r["good"]["k_chosen"]) == 12

    def test_callable_features_receive_fold_context(self):
        """③ / ④ 式特征依赖折内 μ / B：callable(ctx) 只能看到训练侧拟合量。"""
        Z, groups, a, b, e0, e1 = _planted(seed=5)
        seen = []

        def feat(ctx):
            seen.append(set(ctx["train_idx"].tolist()))
            assert ctx["B"].shape[1] == 3 and ctx["mu"].shape == (30,)
            return g.project(Z, ctx["mu"], ctx["B"])[:, :1]          # 用真 Z 投影（测试用；真实特征来自 @25%）

        g = _g()
        r = g.loso_evaluate(Z, groups, e0, e1, {"c": feat}, Ks=(1,), alphas=(1.0,), inner=False)
        assert r["c"]["rho_a1"] > 0.9
        assert all(len(s) < len(a) for s in seen)


class TestStats:
    def test_cluster_bootstrap(self):
        g = _g()
        x = np.arange(40, dtype=float); groups = np.repeat(np.arange(8), 5)
        lo, hi = g.cluster_bootstrap_ci(x, groups, n_boot=2000, seed=0)
        assert lo < x.mean() < hi
        assert (lo, hi) == g.cluster_bootstrap_ci(x, groups, n_boot=2000, seed=0)

    def test_source_wilcoxon(self):
        g = _g()
        groups = np.repeat(np.arange(10), 4)
        assert g.source_wilcoxon(np.ones(40), groups) < 0.01
        assert g.source_wilcoxon(np.zeros(40), groups) == 1.0


class TestVerdict:
    def test_branches(self):
        g = _g()
        el = ["m", "s", "l", "p"]
        base = {k: {"rho_a1": 0.1, "delta_regret3": 0.0} for k in el}
        assert g.verdict_s3a(base, el) == "S3A-STOP"
        assert g.verdict_s3a({**base, "l": {"rho_a1": 0.75, "delta_regret3": 0.012}}, el) == "S3A-GO"
        assert g.verdict_s3a({**base, "l": {"rho_a1": 0.75, "delta_regret3": 0.005}}, el) == "S3A-FIELD"
        assert g.verdict_s3a({**base, "l": {"rho_a1": 0.55, "delta_regret3": 0.02}}, el) == "S3A-FIELD"
        # GO 要求同一组同时满足
        assert g.verdict_s3a({**base, "l": {"rho_a1": 0.75, "delta_regret3": 0.0}, "p": {"rho_a1": 0.2, "delta_regret3": 0.02}}, el) == "S3A-FIELD"
        # 不进门的组（诊断）不改变读数码
        assert g.verdict_s3a({**base, "diag": {"rho_a1": 0.95, "delta_regret3": 0.05}}, el) == "S3A-STOP"


def test_cli_has_analyze():
    g = _g()
    ap = g.build_parser()
    assert ap.parse_args(["--analyze"]).analyze is True
