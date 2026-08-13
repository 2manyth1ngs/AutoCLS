"""SWA（等权权重平均）测试——向 TS2Vec 官方实现修正的训练期差异（排查二 §4）。

官方语义（reference/AutoCTS_plusplus/NAS_Net/ts2vec.py，与 yuezhihan/ts2vec 一致）：
- `__init__` 创建 `AveragedModel(self._net)` 后立即 `update_parameters`（ts2vec.py:53-54）
  → **初始随机权重是第 1 个快照**；
- fit 循环内每次 `optimizer.step()` 后 `update_parameters`（ts2vec.py:152）→ 逐 iter；
- 平均方式 = PyTorch AveragedModel 默认等权 running mean `avg ← avg + (w−avg)/(n+1)`，
  故 K 次更新后 = {w_init, w_1, …, w_K} 共 K+1 个快照的等权平均；
- `encode` 用平均权重（ts2vec.py:177），梯度只更新原始权重。

加成开关 `use_swa`（默认关）：关闭时必须与既有实现逐位相同（否决条件——
否则历史缓存/结果的可复现性被破坏）；开启时训练轨迹不变、仅评估权重换成平均。
"""
import hashlib
import inspect

import numpy as np
import pytest
import torch

from autocls.contrastive.cl_trainer import pretrain_encoder
from autocls.data.task_slicer import SlicedTask
from autocls.encoders.dilated_cnn import DilatedCNNEncoder
from autocls.labeling import evaluate_rprime
from autocls.task_types import TASK_CLASSIFICATION


def _toy_data(n=32, T=32, C=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
    X = rng.normal(0, 0.3, size=(n, T, C)).astype(np.float32)
    X[: n // 2] += np.sin(t)[None, :, None]
    return X


def _small_encoder():
    return DilatedCNNEncoder(input_dims=3, output_dims=32, hidden_dims=16, depth=3)


def _params_md5(encoder) -> str:
    return hashlib.md5(
        b"".join(p.detach().numpy().tobytes() for p in encoder.parameters())
    ).hexdigest()


def _run(make_cls, *, max_iters, seed_model=123, seed_train=7, **kw):
    """定死双种子的小训练：返回 (encoder, losses)。"""
    torch.manual_seed(seed_model)
    enc = _small_encoder()
    losses = pretrain_encoder(
        enc, make_cls(), _toy_data(), max_iters=max_iters,
        batch_size=8, seed=seed_train, device="cpu", **kw,
    )
    return enc, losses


class TestSwaSwitch:
    def test_use_swa_keyword_only_default_off(self):
        # 加成开关：keyword-only 且默认 False——历史调用方零改动、缓存零污染
        p = inspect.signature(pretrain_encoder).parameters["use_swa"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is False

    def test_swa_off_bitwise_matches_pre_swa_golden(self, make_cls):
        # 金标特征化：2026-08-13 加 SWA 前的 HEAD 行为（本机 CPU、双种子定死，
        # scratchpad capture_golden.py 捕获）。use_swa=False 与不传参都必须逐位复现。
        golden_losses = ["0x1.0d3a434000000p+1", "0x1.02187e4000000p+1"]
        golden_md5 = "73fc381ee7d702aa37bb148a7fbf4164"
        for kw in ({}, {"use_swa": False}):
            enc, losses = _run(make_cls, max_iters=8, **kw)
            assert [float.hex(float(l)) for l in losses] == golden_losses
            assert _params_md5(enc) == golden_md5

    def test_swa_weights_equal_snapshot_running_mean(self, make_cls):
        # 核心语义：K 次更新后权重 = {w_init, w_1..w_K} 等权平均（含 init 快照）。
        # 快照经前缀跑截取：同双种子下 max_iters=k 的终点 = 全程轨迹第 k 步
        # （cl_trainer 保证"恰好 max_iters 次 step 后停止"，RNG 消费逐 batch 一致）。
        K = 4
        snapshots = []
        for k in range(K + 1):
            enc_k, _ = _run(make_cls, max_iters=k)
            snapshots.append([p.detach().numpy().astype(np.float64)
                              for p in enc_k.parameters()])
        enc_swa, _ = _run(make_cls, max_iters=K, use_swa=True)
        for i, p in enumerate(enc_swa.parameters()):
            expect = np.mean([s[i] for s in snapshots], axis=0)
            np.testing.assert_allclose(p.detach().numpy(), expect, atol=1e-6, rtol=0)

    def test_swa_does_not_perturb_training_trajectory(self, make_cls):
        # AveragedModel 的 deepcopy/update 不消耗 RNG → 逐 pass 损失必须逐位相同
        _, losses_off = _run(make_cls, max_iters=8, use_swa=False)
        _, losses_on = _run(make_cls, max_iters=8, use_swa=True)
        assert losses_off == losses_on

    def test_swa_zero_iters_keeps_init_weights(self, make_cls):
        # iters=0：平均 = 仅 init 快照 = init 本身 → 基线缓存不受 SWA 影响
        # （排查二 §4"不影响 iters=0 基线"的单元化）
        torch.manual_seed(123)
        init = [p.detach().clone() for p in _small_encoder().parameters()]
        enc, losses = _run(make_cls, max_iters=0, use_swa=True)
        assert losses == []
        assert all(torch.equal(a, b) for a, b in zip(init, enc.parameters()))


class TestEvaluateRprimeSwa:
    def _tiny_task_data(self, n=80, T=32, C=2, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, T, C)).astype(np.float32)
        y = np.array([0, 1] * (n // 2), dtype=np.int64)
        X[y == 1] += 2.5
        task = SlicedTask(
            task_id="src_00", pool="source", classes=(0, 1),
            channels=tuple(range(C)), sample_indices=tuple(range(n)),
            tag="", task_type=TASK_CLASSIFICATION,
        )
        return task, X, y

    def test_records_swa_flag_default_off(self, make_cls):
        # swa 状态必须进缓存记录（与历史记录隔离的键成分）
        task, X, y = self._tiny_task_data()
        rec = evaluate_rprime(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            iters=5, batch_size=32, device="cpu",
        )
        assert rec["swa"] is False

    def test_use_swa_threads_through_and_records(self, make_cls):
        task, X, y = self._tiny_task_data()
        rec = evaluate_rprime(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            iters=30, batch_size=32, device="cpu", use_swa=True,
        )
        assert rec["swa"] is True
        assert rec["error"] is None
        assert np.isfinite(rec["r_prime"])
