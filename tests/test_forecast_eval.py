"""forecast_eval 单测（TS2Vec 官方预测协议逐行对照适配的行为钉死）。

对照源：yuezhihan/ts2vec
- tasks/forecasting.py L5-12 generate_pred_samples：feature r_t ↔ 预测 data[t+1..t+Q]
- tasks/forecasting.py L20-30：表征提取 = encode(causal=True, sliding_length=1,
  sliding_padding=200) —— 每个时间步 t 的表征由"以 t 结尾的 padding+1 长因果窗"
  编码取**末位**得到；本项目 encoder 无 mask 语义，左侧缺失上下文用零填充
  （路线 A 中性零填充口径，官方是 NaN pad + encoder 内置零，等效性见实现注释）
- tasks/_eval_protocols.py L81-109 fit_ridge：Ridge + 13 档 alpha 网格、
  验证集 RMSE+MAE 选 alpha、best alpha 在 train 上重拟合
"""
import numpy as np
import pytest
import torch
from torch import nn

from autocls.downstream.forecast_eval import (
    encode_causal_sliding,
    evaluate_forecasting,
    fit_ridge,
    generate_pred_samples,
)
from autocls.downstream import resolve_evaluator
from autocls.task_types import TASK_FORECASTING


class AnticausalToy(nn.Module):
    """故意非因果的玩具 encoder：每步表征混入全窗均值——用来考察滑窗因果性
    是否由**窗口构造**保证（而非依赖 encoder 自身因果）。(B,T,C)→(B,T,C)。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x.mean(dim=1, keepdim=True)


class TestGeneratePredSamples:
    def _toy(self):
        # data[0, t] = t；features[0, t] = [t, 10t]
        data = np.arange(10, dtype=float).reshape(1, 10, 1)
        feats = np.stack([np.arange(10.0), 10 * np.arange(10.0)], axis=1)[None]
        return feats, data

    def test_alignment_rt_predicts_next_q(self):
        feats, data = self._toy()
        f, l = generate_pred_samples(feats, data, pred_len=3)
        # 官方 L6-12：features[:, :-Q] 对齐 labels[j]=data[j+1..j+Q]
        assert f.shape == (7, 2) and l.shape == (7, 3)
        np.testing.assert_allclose(f[0], [0.0, 0.0])
        np.testing.assert_allclose(l[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(l[6], [7.0, 8.0, 9.0])

    def test_drop_removes_warmup_prefix(self):
        feats, data = self._toy()
        f, l = generate_pred_samples(feats, data, pred_len=3, drop=2)
        # drop=2（官方 L9-10）：前 2 个训练样本（零填充窗污染的表征）被丢弃
        assert f.shape == (5, 2)
        np.testing.assert_allclose(f[0], [2.0, 20.0])
        np.testing.assert_allclose(l[0], [3.0, 4.0, 5.0])

    def test_multichannel_labels_flattened(self):
        data = np.arange(20, dtype=float).reshape(1, 10, 2)  # C=2
        feats = np.arange(10.0).reshape(1, 10, 1)
        f, l = generate_pred_samples(feats, data, pred_len=2)
        # labels 展平成 Q*C 维（官方 L12 reshape）
        assert l.shape == (8, 4)
        np.testing.assert_allclose(l[0], [2.0, 3.0, 4.0, 5.0])  # data[1..2] 两通道


class TestFitRidge:
    def test_clean_linear_system_low_alpha_high_fit(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 8))
        w = rng.normal(size=(8, 3))
        y = X @ w
        Xv, yv = rng.normal(size=(100, 8)), None
        yv = Xv @ w
        lr = fit_ridge(X, y, Xv, yv)
        assert lr.alpha == 0.1  # 无噪线性系统：最小 alpha 胜出（官方网格首项）
        pred = lr.predict(Xv)
        assert pred.shape == (100, 3)
        np.testing.assert_allclose(pred, yv, atol=1e-2)

    def test_alpha_selected_on_validation(self):
        # train 带强噪、维度≈样本数 → 小 alpha 过拟合；干净 valid 应把 alpha 推大
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 50))
        w = np.zeros((50, 1)); w[0] = 1.0
        y = X @ w + rng.normal(scale=3.0, size=(60, 1))
        Xv = rng.normal(size=(200, 50))
        yv = Xv @ w
        lr = fit_ridge(X, y, Xv, yv)
        assert lr.alpha > 0.1  # 具体档位不钉死，但绝不能选到最小档


class TestEncodeCausalSliding:
    def test_shape_and_window_semantics(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(1, 30, 3))
        enc = AnticausalToy()
        out = encode_causal_sliding(enc, data, padding=8, batch_size=7, device="cpu")
        assert out.shape == (1, 30, 3)
        # t≥padding：repr_t = enc(data[t-8..t])[:, -1]（官方 slicing 取末位语义）
        t = 15
        win = torch.from_numpy(data[:, t - 8 : t + 1].astype(np.float32))
        expect = enc(win)[0, -1].numpy()
        np.testing.assert_allclose(out[0, t], expect, rtol=1e-5)
        # t=0：左侧零填充到 9 长
        win0 = torch.cat([torch.zeros(1, 8, 3), torch.from_numpy(data[:, :1].astype(np.float32))], dim=1)
        np.testing.assert_allclose(out[0, 0], enc(win0)[0, -1].numpy(), rtol=1e-5)

    def test_causality_future_perturbation_invariant(self):
        # 非因果 encoder + 滑窗构造：t 之后的数据改动不得影响 t 及之前的表征
        rng = np.random.default_rng(3)
        data = rng.normal(size=(1, 40, 2))
        enc = AnticausalToy()
        base = encode_causal_sliding(enc, data, padding=6, batch_size=64, device="cpu")
        data2 = data.copy()
        data2[:, 25:] += 100.0
        pert = encode_causal_sliding(enc, data2, padding=6, batch_size=64, device="cpu")
        np.testing.assert_allclose(pert[0, :25], base[0, :25], rtol=1e-5)
        assert not np.allclose(pert[0, 25:], base[0, 25:])

    def test_batching_does_not_change_values(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(1, 23, 2))
        enc = AnticausalToy()
        a = encode_causal_sliding(enc, data, padding=5, batch_size=4, device="cpu")
        b = encode_causal_sliding(enc, data, padding=5, batch_size=64, device="cpu")
        np.testing.assert_allclose(a, b, rtol=1e-6)


class TestEvaluateForecasting:
    def _run(self, scaler=None):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(1, 120, 3)).astype(np.float64)  # 前 2 列当协变量
        enc = AnticausalToy()
        return evaluate_forecasting(
            enc, data,
            train_slice=slice(None, 60), valid_slice=slice(60, 80),
            test_slice=slice(80, 120),
            pred_len=5, n_covariate_cols=2, padding=10,
            scaler=scaler, batch_size=64, device="cpu",
        )

    def test_returns_finite_norm_metrics_and_alpha(self):
        m = self._run()
        for k in ("mse", "mae", "alpha"):
            assert np.isfinite(m[k]), k
        assert m["mse"] > 0 and m["mae"] > 0
        assert m["alpha"] in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    def test_sample_counts(self):
        m = self._run()
        # train: 60-5=55 再 drop=10 → 45；test: 40-5=35（官方 valid/test 不 drop）
        assert m["n_train_samples"] == 45
        assert m["n_eval"] == 35

    def test_raw_metrics_identity_scaler_equal_norm(self):
        from sklearn.preprocessing import StandardScaler

        ident = StandardScaler()
        ident.fit(np.random.default_rng(6).normal(size=(50, 1)))
        ident.mean_ = np.zeros(1)
        ident.scale_ = np.ones(1)
        m = self._run(scaler=ident)
        np.testing.assert_allclose(m["mse_raw"], m["mse"], rtol=1e-9)
        np.testing.assert_allclose(m["mae_raw"], m["mae"], rtol=1e-9)

    def test_no_scaler_no_raw_keys(self):
        m = self._run()
        assert "mse_raw" not in m


class TestResolveEvaluatorForecasting:
    def test_dispatch_returns_forecast_evaluator_with_rprime(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(1, 100, 2))
        evaluator = resolve_evaluator(TASK_FORECASTING)
        m = evaluator(
            AnticausalToy(), data,
            train_slice=slice(None, 50), valid_slice=slice(50, 70),
            test_slice=slice(70, 100),
            pred_len=4, n_covariate_cols=1, padding=8,
            batch_size=64, device="cpu",
        )
        assert m["r_prime"] == m["mse"]  # R'=norm MSE，方向由 is_better 判（小者优）
