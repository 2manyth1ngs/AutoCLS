"""task_embed 单测（TECHNICAL_DESIGN.md 模块 5：E = SetTF(TS2Vec(样本序列集))）。

全部用小模型 + 合成数据，CPU 秒级跑完。真实 HAR 上的区分力验证见
scripts/sanity_task_embed.py（sanity check A/B），不放单测。
"""
import inspect

import numpy as np
import pytest
import torch

from autocls.task_embed import embed_task, encode_task_windows, fit_shared_ts2vec
from autocls.task_embed.set_transformer import TaskSetEncoder
from autocls.task_embed.ts2vec_wrapper import TS2VecWrapper

# 单测用小模型（正式默认 output_dims=256/hidden=64/depth=10，照 AutoCTS++）
TINY = dict(output_dims=16, hidden_dims=8, depth=2)


def _toy(n=12, t=16, c=3, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, t, c)).astype(np.float32)


# ---------------------------------------------------------------- TS2VecWrapper
class TestTS2VecWrapper:
    def test_encode_per_timestep_shape(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=2)
        out = w.encode(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (12, 16, TINY["output_dims"])  # (N, T, D) 逐时间步

    def test_encode_full_series_shape(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=2)
        out = w.encode(X, encoding_window="full_series")
        assert out.shape == (12, TINY["output_dims"])  # (N, D) 全序列池化

    def test_fit_counts_param_updates(self):
        # 铁律 2 同款口径：训练量按参数更新次数计，n_iters 必须精确生效
        # （N=12 < batch，1 step/遍 → 5 iters 需跨 5 遍数据）
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(X, n_iters=5)
        assert w.n_iters == 5

    def test_fit_is_unsupervised(self):
        # 任务嵌入必须是"数据分布特征"而非标签信息（zero-shot 迁移前提）：
        # fit 的签名不得出现标签参数
        params = inspect.signature(TS2VecWrapper.fit).parameters
        assert not {"y", "labels", "targets"} & set(params)

    def test_same_seed_reproducible(self):
        X = _toy()
        outs = []
        for _ in range(2):
            w = TS2VecWrapper(input_dims=3, device="cpu", seed=7, **TINY)
            w.fit(X, n_iters=3)
            outs.append(w.encode(X))
        np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)

    def test_fit_changes_encoding(self):
        X = _toy()
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        before = w.encode(X)
        w.fit(X, n_iters=5)
        after = w.encode(X)
        assert not np.allclose(before, after)


# ---------------------------------------------------------------- TaskSetEncoder
class TestTaskSetEncoder:
    def _enc(self, seed=0):
        torch.manual_seed(seed)
        return TaskSetEncoder(dim_input=16, nz=8)

    def test_single_task_output_shape(self):
        enc = self._enc()
        x = torch.randn(5, 7, 16)  # (n_windows, seq_len, D)
        e = enc(x)
        assert e.shape == (8,)

    def test_set_encode_batch_matches_single(self):
        enc = self._enc()
        x1, x2 = torch.randn(5, 7, 16), torch.randn(4, 7, 16)
        batch = enc.set_encode([x1, x2])
        assert batch.shape == (2, 8)
        torch.testing.assert_close(batch[0], enc(x1), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(batch[1], enc(x2), atol=1e-5, rtol=1e-4)

    def test_window_permutation_invariant(self):
        # Set-Transformer 语义：任务是样本"集合"，窗口顺序不得影响 E
        enc = self._enc()
        x = torch.randn(6, 7, 16)
        perm = torch.randperm(6)
        torch.testing.assert_close(enc(x), enc(x[perm]), atol=1e-5, rtol=1e-4)

    def test_same_seed_same_weights(self):
        a, b = self._enc(seed=3), self._enc(seed=3)
        for (ka, va), (kb, vb) in zip(
            a.state_dict().items(), b.state_dict().items()
        ):
            assert ka == kb
            torch.testing.assert_close(va, vb, atol=0, rtol=0)


# ---------------------------------------------------------------- fit_shared_ts2vec
def _shared(n_iters=2, seed=0):
    """测试用共享冻结 TS2Vec：3 通道合成源池，展开成单变量训练。"""
    return fit_shared_ts2vec(
        _toy(c=3, seed=seed), n_iters=n_iters, device="cpu",
        batch_size=8, seed=seed, **TINY,
    )


class TestFitSharedTS2Vec:
    def test_univariate_and_frozen(self):
        # 定稿语义：源池训练单变量（input_dims=1）TS2Vec，训完即冻结——
        # 冻结后 fit 必须报错（防误再训练/防 zero-shot 数据泄漏）
        w = _shared()
        assert w.input_dims == 1
        assert w.frozen
        assert w.n_iters == 2
        with pytest.raises(RuntimeError, match="冻结"):
            w.fit(_toy(c=1), n_iters=4)

    def test_unrolls_channels_as_instances(self):
        # (N, T, C) 源池必须展开为 (N*C, T, 1) 单变量实例池训练：
        # 若未展开，多通道数据进 input_dims=1 的模型会直接维度报错
        w = fit_shared_ts2vec(
            _toy(c=5), n_iters=2, device="cpu", batch_size=8, seed=0, **TINY
        )
        assert w.input_dims == 1

    def test_signature_is_unsupervised_and_explicit(self):
        # 泄漏防护 API：训练数据必须经 source_data 显式传入；无标签参数
        params = inspect.signature(fit_shared_ts2vec).parameters
        assert "source_data" in params
        assert not {"y", "labels", "targets"} & set(params)

    def test_same_seed_reproducible(self):
        X = _toy(c=2)
        outs = [_shared(seed=7).encode(_toy(c=1)) for _ in range(2)]
        np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)


# ---------------------------------------------------------------- encode_task_windows
class TestEncodeTaskWindows:
    def test_per_timestep_channel_mean_shape(self):
        # 通道当独立实例编码后跨通道取均值（AutoCTS++ generate_task_feature.py
        # L137-140 语义）：(m, T, C) → (m, T, D)
        w = _shared()
        out = encode_task_windows(_toy(c=3), w, sample_num=4, seed=0)
        assert out.shape == (4, 16, TINY["output_dims"])

    def test_full_series_shape(self):
        w = _shared()
        out = encode_task_windows(
            _toy(c=3), w, sample_num=4, seed=0, encoding_window="full_series"
        )
        assert out.shape == (4, TINY["output_dims"])

    def test_channel_count_agnostic(self):
        # 单变量共享编码器天然支持任意通道数，无需零填充
        w = _shared()
        o3 = encode_task_windows(_toy(c=3), w, sample_num=4, seed=0)
        o5 = encode_task_windows(_toy(c=5), w, sample_num=4, seed=0)
        assert o3.shape == o5.shape == (4, 16, TINY["output_dims"])

    def test_deterministic(self):
        w = _shared()
        o1 = encode_task_windows(_toy(), w, sample_num=4, seed=0)
        o2 = encode_task_windows(_toy(), w, sample_num=4, seed=0)
        np.testing.assert_allclose(o1, o2, atol=1e-6)


# ---------------------------------------------------------------- embed_task 流水线
PIPE = dict(sample_num=4, nz=8, seed=0)


class TestEmbedTask:
    def test_shape_and_reproducible(self):
        X = _toy()
        w = _shared()
        e1 = embed_task(X, w, **PIPE)
        e2 = embed_task(X, w, **PIPE)
        assert e1.shape == (8,)
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_channel_count_agnostic(self):
        # 不同通道数的任务（HAR 9ch vs Epilepsy 1ch vs 通道子集）用同一份
        # 共享单变量 TS2Vec 映射到同维 E
        w = _shared()
        e3 = embed_task(_toy(c=3), w, **PIPE)
        e5 = embed_task(_toy(c=5), w, **PIPE)
        assert e3.shape == e5.shape == (8,)

    def test_signature_has_no_labels(self):
        params = inspect.signature(embed_task).parameters
        assert not {"y", "labels", "targets"} & set(params)

    def test_per_task_retrain_path_removed(self):
        # 旧默认"每任务重训 TS2Vec"已证实无区分力（E 被训练随机性主导，
        # results/task_embed_sanity/），路径必须移除：共享编码器是必传参数
        params = inspect.signature(embed_task).parameters
        assert "ts2vec_iters" not in params
        with pytest.raises(TypeError):
            embed_task(_toy())  # 不传 shared_ts2vec 必须直接报错

    def test_sample_num_larger_than_n_ok(self):
        # 任务样本数 < sample_num 时用全部样本，不得报错
        w = _shared()
        e = embed_task(_toy(n=3), w, **{**PIPE, "sample_num": 100})
        assert e.shape == (8,)

    def test_does_not_refit_shared(self):
        w = _shared(n_iters=2)
        embed_task(_toy(), w, **PIPE)
        assert w.n_iters == 2  # 冻结编码器不得被再训练

    def test_rejects_multivariate_ts2vec(self):
        # 泄漏防护 API 只认单变量共享编码器（多变量+零填充方案已被诊断否决）
        w = TS2VecWrapper(input_dims=3, device="cpu", seed=0, **TINY)
        w.fit(_toy(c=3), n_iters=2)
        w.freeze()
        with pytest.raises(AssertionError):
            embed_task(_toy(c=3), w, **PIPE)

    def test_rejects_unfrozen_ts2vec(self):
        # 未经 fit_shared_ts2vec/freeze 的编码器不得使用：强制走显式源池训练入口
        w = TS2VecWrapper(input_dims=1, device="cpu", seed=0, **TINY)
        w.fit(_toy(c=1), n_iters=2)
        with pytest.raises(AssertionError):
            embed_task(_toy(), w, **PIPE)
