"""cl_trainer 测试（TECHNICAL_DESIGN.md 模块 3；铁律 2：max_iters 必须显式传参）。

训练量计量单位为 iters（optimizer.step() 参数更新次数），而非 epoch：
路线 B 的多切片任务大小不同，按 epoch 截断会让各任务的实际参数更新次数不一致，
污染跨任务 early-val label。max_iters 保证任何任务/CLS 的训练量精确相同。
"""
import inspect

import numpy as np
import pytest
import torch

from autocls.contrastive.cl_trainer import pretrain_encoder
from autocls.encoders.dilated_cnn import DilatedCNNEncoder


def _toy_data(n=32, T=32, C=3, seed=0):
    """两簇可分的玩具时序：类0为正弦+噪声，类1为噪声。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, T, dtype=np.float32)
    X = rng.normal(0, 0.3, size=(n, T, C)).astype(np.float32)
    X[: n // 2] += np.sin(t)[None, :, None]
    return X


def _small_encoder():
    return DilatedCNNEncoder(input_dims=3, output_dims=32, hidden_dims=16, depth=3)


def _patch_step_counter(monkeypatch):
    """替换 torch.optim.AdamW 为计数子类，返回已创建实例列表以读取 step_count。"""
    created = []
    real_adamw = torch.optim.AdamW

    class CountingAdamW(real_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0
            created.append(self)

        def step(self, *args, **kwargs):
            self.step_count += 1
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", CountingAdamW)
    return created


class TestPretrainEncoder:
    def test_max_iters_is_required_keyword(self):
        # 铁律 2：early-val 与完整评估禁止混用 → max_iters 无默认值、必须显式传
        sig = inspect.signature(pretrain_encoder)
        p = sig.parameters["max_iters"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is inspect.Parameter.empty
        assert "max_epochs" not in sig.parameters  # epoch 计量已废弃

    def test_exact_step_count_across_pass_boundary(self, make_cls, monkeypatch):
        # n=32, batch=8 → 每遍数据 4 次更新；max_iters=10 跨 pass 边界（4+4+2）
        created = _patch_step_counter(monkeypatch)
        pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
        )
        assert len(created) == 1
        assert created[0].step_count == 10

    def test_skipped_tiny_batch_not_counted(self, make_cls, monkeypatch):
        # n=17, batch=8 → 每遍 batch 为 8/8/1，末尾 1 样本无负例被跳过、不计 step
        # → 每遍只有 2 次更新；max_iters=5 需要第 3 遍数据（2+2+1）
        created = _patch_step_counter(monkeypatch)
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=17),
            max_iters=5, batch_size=8, seed=0, device="cpu",
        )
        assert created[0].step_count == 5
        assert len(losses) == 3  # 3 遍数据 → 3 个逐 pass 平均损失

    def test_returns_per_pass_losses(self, make_cls):
        # n=32, batch=8 → 4 步/遍；max_iters=10 → 2 个完整 pass + 1 个截断 pass
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
        )
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)

    def test_raises_when_no_batch_can_step(self, make_cls):
        # N=1：所有 batch 都 <2 样本被跳过，step 永不推进 → 必须抛错而非死循环
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(n=1),
                max_iters=4, batch_size=8, seed=0, device="cpu",
            )

    def test_loss_decreases(self, make_cls):
        # n=64, batch=16 → 4 步/遍；32 iters = 原 8 epoch 的等量训练
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=64),
            max_iters=32, batch_size=16, seed=0, device="cpu",
        )
        assert losses[-1] < losses[0]

    def test_parameters_updated(self, make_cls):
        enc = _small_encoder()
        before = [p.detach().clone() for p in enc.parameters()]
        pretrain_encoder(enc, make_cls(), _toy_data(), max_iters=4, batch_size=8, device="cpu")
        changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, enc.parameters()))
        assert changed

    def test_deterministic_given_seed(self, make_cls):
        results = []
        for _ in range(2):
            torch.manual_seed(123)
            enc = _small_encoder()
            losses = pretrain_encoder(
                enc, make_cls(), _toy_data(), max_iters=4, batch_size=8, seed=7, device="cpu"
            )
            results.append(losses[0])
        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# 截点回调（MX-3-PQ T1）：一条轨迹在 60/120/290 三个 budget 各取一次 R'。
# 单跑三次 = 三倍 GPU 账单且三条不同轨迹，只有"同一条轨迹上打点"才让
# (slice, cls) 的 budget 间差异是纯训练量差异（预算维度 ctx 的前提）。
# 铁律 2 不变：max_iters 仍显式传参，checkpoints 只是在途取样，不改训练量语义。
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_callback_fires_at_exact_steps(self, make_cls):
        # n=32, batch=8 → 4 步/遍；截点跨 pass 边界（3 在第一遍内，7/10 在后续遍）
        seen = []
        pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
            checkpoints=(3, 7, 10), on_checkpoint=lambda step, _enc: seen.append(step),
        )
        assert seen == [3, 7, 10]

    def test_trainer_restores_train_mode_after_callback(self, make_cls):
        # 回调里跑的是下游评估器（内部 encoder.eval()）——训练循环必须自己恢复
        # train()，否则截点之后的 iters 全在 eval 模式下训练（静默变成另一条轨迹）
        modes = []

        def cb(_step, enc):
            modes.append(enc.training)
            enc.eval()

        pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
            checkpoints=(3, 7, 10), on_checkpoint=cb,
        )
        assert modes == [True, True, True]

    def test_checkpoint_eval_leaves_trajectory_bit_identical(self, make_cls):
        # 决定性质：截点处跑真实下游特征提取，最终权重必须与"不打截点"的同种子跑
        # 逐位相同——否则 budget=290 的 R' 就不是原口径的 R'，缓存不可比。
        from autocls.downstream.linear_eval import encode_features

        X = _toy_data(n=32)
        torch.manual_seed(123)
        ref = _small_encoder()
        pretrain_encoder(ref, make_cls(), X, max_iters=8, batch_size=8, seed=7, device="cpu")

        torch.manual_seed(123)
        ckpt = _small_encoder()
        pretrain_encoder(
            ckpt, make_cls(), X, max_iters=8, batch_size=8, seed=7, device="cpu",
            checkpoints=(4, 8),
            on_checkpoint=lambda _s, enc: encode_features(enc, X[:8], device="cpu"),
        )

        a, b = ref.state_dict(), ckpt.state_dict()
        assert list(a) == list(b)
        for k in a:
            assert torch.equal(a[k], b[k]), f"截点评估扰动了训练轨迹：{k}"

    def test_checkpoint_at_skipped_batch_boundary_still_fires(self, make_cls):
        # n=5, batch=2 → 每遍 batch 为 2/2/1，末尾单样本被跳过（不计 step）
        # → 2 步/遍。截点 3/6 都落在"跨遍"的位置上，必须照样触发。
        seen = []
        pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=5),
            max_iters=6, batch_size=2, seed=0, device="cpu",
            checkpoints=(3, 6), on_checkpoint=lambda step, _enc: seen.append(step),
        )
        assert seen == [3, 6]

    @pytest.mark.parametrize("checkpoints", [
        (7, 3),        # 未排序
        (3, 3, 7),     # 重复
        (3, 12),       # 超过 max_iters → 永不触发，静默少一条 R'
        (0, 3),        # 非正 → 永不触发
    ])
    def test_invalid_checkpoints_rejected(self, make_cls, checkpoints):
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(n=32),
                max_iters=10, batch_size=8, seed=0, device="cpu",
                checkpoints=checkpoints, on_checkpoint=lambda *_a: None,
            )

    def test_swa_with_checkpoints_rejected(self, make_cls):
        # SWA 把平均权重在训练**结束时**拷回 encoder，途中截点看到的是原始权重
        # ——两种口径混在一条轨迹里无法解释，直接禁止组合
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(n=32),
                max_iters=10, batch_size=8, seed=0, device="cpu",
                use_swa=True, checkpoints=(5, 10), on_checkpoint=lambda *_a: None,
            )

    def test_no_checkpoints_keeps_old_behavior(self, make_cls, monkeypatch):
        # 既有调用方（2b/step0/24/25/27）一律不传 checkpoints，行为零改动
        created = _patch_step_counter(monkeypatch)
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=32),
            max_iters=10, batch_size=8, seed=0, device="cpu",
        )
        assert created[0].step_count == 10 and len(losses) == 3


# ---------------------------------------------------------------------------
# 全量协议对齐（results/protocol_alignment）：两个 opt-in 开关。
# drop_last：TS2Vec 官方 DataLoader(drop_last=True)（reference ts2vec.py:93-94），
#            bs 8 时残批被丢；crop_mode="ts2vec"：官方逐迭代 crop_l~U[2,T] + 逐行偏移
#            （ts2vec.py:126-132）。两者默认关，默认路径必须与 test_swa 的金标逐位相同
#            （历史缓存复现红线，铁律 5 先测后写）。
# ---------------------------------------------------------------------------


def _params_md5(encoder) -> str:
    import hashlib
    return hashlib.md5(
        b"".join(p.detach().numpy().tobytes() for p in encoder.parameters())
    ).hexdigest()


def _golden_run(make_cls, **kw):
    """与 tests/test_swa.py::_run 同口径（seed_model=123, seed_train=7, bs 8, 8 iters）。"""
    torch.manual_seed(123)
    enc = _small_encoder()
    losses = pretrain_encoder(
        enc, make_cls(), _toy_data(), max_iters=8, batch_size=8, seed=7,
        device="cpu", **kw,
    )
    return enc, losses


GOLDEN_LOSSES = ["0x1.0d3a434000000p+1", "0x1.02187e4000000p+1"]   # test_swa 金标
GOLDEN_MD5 = "73fc381ee7d702aa37bb148a7fbf4164"


class TestDropLastAndCropMode:
    def test_switches_keyword_only_and_default_off(self):
        sig = inspect.signature(pretrain_encoder)
        p = sig.parameters["drop_last"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
        q = sig.parameters["crop_mode"]
        assert q.kind is inspect.Parameter.KEYWORD_ONLY and q.default == "fixed"

    def test_default_path_bitwise_matches_golden(self, make_cls):
        # 不传 / 显式传默认值，都必须逐位复现加开关前的金标（loss hex + 参数 md5）
        for kw in ({}, {"drop_last": False, "crop_mode": "fixed"}):
            enc, losses = _golden_run(make_cls, **kw)
            assert [float.hex(float(l)) for l in losses] == GOLDEN_LOSSES
            assert _params_md5(enc) == GOLDEN_MD5

    def test_drop_last_steps_per_pass(self, make_cls, monkeypatch):
        # n=20, bs=8 → 残批 4 样本：默认 3 步/遍（8/8/4）；drop_last 只 2 步/遍（8/8）
        created = _patch_step_counter(monkeypatch)
        losses = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=20),
            max_iters=5, batch_size=8, seed=0, device="cpu", drop_last=True,
        )
        assert created[0].step_count == 5
        assert len(losses) == 3            # 2+2+1 → 3 遍
        losses_off = pretrain_encoder(
            _small_encoder(), make_cls(), _toy_data(n=20),
            max_iters=5, batch_size=8, seed=0, device="cpu",
        )
        assert len(losses_off) == 2        # 3+2 → 2 遍

    def test_drop_last_n_below_batch_raises(self, make_cls):
        # N<bs 且 drop_last：整遍无可训 batch → 必须 ValueError 而非死循环
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(n=5),
                max_iters=4, batch_size=8, seed=0, device="cpu", drop_last=True,
            )

    def test_invalid_crop_mode_raises(self, make_cls):
        with pytest.raises(ValueError):
            pretrain_encoder(
                _small_encoder(), make_cls(), _toy_data(),
                max_iters=2, batch_size=8, seed=0, device="cpu", crop_mode="random",
            )

    def test_crop_mode_ts2vec_runs_finite_and_differs(self, make_cls):
        _, fixed = _golden_run(make_cls)
        _, ts2v = _golden_run(make_cls, crop_mode="ts2vec")
        assert all(np.isfinite(l) for l in ts2v)
        assert fixed != ts2v               # 裁剪坐标分布不同 → 轨迹不同

    def test_crop_mode_ts2vec_with_crop_zero_matches_fixed(self, make_cls):
        # crop=0 → 增强跳过，两种模式必须逐位相同（模式只在 crop 启用时生效）
        _, a = _golden_run(lambda: make_cls(crop=0.0))
        _, b = _golden_run(lambda: make_cls(crop=0.0), crop_mode="ts2vec")
        assert a == b
