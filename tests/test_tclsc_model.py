"""T-CLSC 模型单测（TECHNICAL_DESIGN.md 模块 6 验收：前向输出 ∈ [0,1]；BCE 能下降）。

模型是任务无关模块（铁律 7）：测试全部用合成 CLS 向量与合成窗口表征，
不 import task_types、不假设任何下游任务语义。
"""
import numpy as np
import pytest
import torch
from torch import nn

from autocls.comparator.tclsc_model import TCLSC
from autocls.task_embed.set_transformer import TaskSetEncoder

CLS_DIM = 7   # 合成 CLS 向量维度（模型对编码维度不敏感）
DIN = 8       # 窗口表征维（真实为 TS2Vec output_dims=256，测试取小）
NZ = 16       # 任务嵌入维（真实 256）


@pytest.fixture
def model() -> TCLSC:
    torch.manual_seed(0)
    set_enc = TaskSetEncoder(dim_input=DIN, nz=NZ)
    return TCLSC(cls_dim=CLS_DIM, set_encoder=set_enc, embedding_dim=12, fz=10)


def _cls_batch(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    a = torch.tensor(rng.random((n, CLS_DIM)), dtype=torch.float32)
    b = torch.tensor(rng.random((n, CLS_DIM)), dtype=torch.float32)
    return a, b


def _windows(n_windows: int, seq_len: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_windows, seq_len, DIN, generator=g)


class TestForward:
    def test_prob_shape_and_range(self, model):
        a, b = _cls_batch(5, seed=1)
        E = model.encode_tasks([_windows(4)]).expand(5, -1)
        out = model(a, b, E)
        assert out.shape == (5,)
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()

    def test_output_depends_on_E(self, model):
        # E 参与前向是消融检验（zero-E）有意义的前提
        a, b = _cls_batch(5, seed=2)
        E = model.encode_tasks([_windows(4)]).expand(5, -1)
        out_real = model(a, b, E)
        out_zero = model(a, b, torch.zeros_like(E))
        assert not torch.allclose(out_real, out_zero)

    def test_output_depends_on_cls_order(self, model):
        # concat(l_a, l_b) 顺序编码"谁是 a"——swap 应改变输出（Eq.16）
        a, b = _cls_batch(5, seed=3)
        E = model.encode_tasks([_windows(4)]).expand(5, -1)
        assert not torch.allclose(model(a, b, E), model(b, a, E))


class TestEncodeTasks:
    def test_variable_window_counts(self, model):
        E = model.encode_tasks([_windows(4, seed=1), _windows(3, seed=2)])
        assert E.shape == (2, NZ)
        assert torch.isfinite(E).all()


class TestJointTraining:
    def test_gradients_flow_to_set_encoder_and_cls_mlp(self, model):
        # SetTF 与 T-CLSC 联合训练：BCE 反传必须同时更新两条支路
        a, b = _cls_batch(6, seed=4)
        E = model.encode_tasks([_windows(4)]).expand(6, -1)
        y = torch.tensor([1., 0., 1., 0., 1., 0.])
        loss = nn.BCELoss()(model(a, b, E), y)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.set_encoder.parameters()), "SetTF 未参与联合训练"
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.cls_mlp.parameters()), "CLS MLP 无梯度"

    def test_bce_decreases_on_tiny_separable_data(self, model):
        # 模块 6 验收：BCE 能下降。y 仅由 cls 向量首维决定（可学习的合成规则）
        torch.manual_seed(0)
        a, b = _cls_batch(32, seed=5)
        y = (a[:, 0] > b[:, 0]).float()
        w = _windows(4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.BCELoss()
        losses = []
        for _ in range(60):
            E = model.encode_tasks([w]).expand(32, -1)
            loss = criterion(model(a, b, E), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 0.8, f"BCE 未下降: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------- 任务设定向量 ctx

CTX_DIM = 3  # 合成任务设定维（真实为预算 one-hot，见 MX-3-PQ interfaces.md）


def _ctx_batch(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, CTX_DIM, generator=g)


def _mk(*, ctx_dim: int = 0, task_branch: str = "E+ctx", with_set_encoder: bool = True) -> TCLSC:
    torch.manual_seed(0)
    enc = TaskSetEncoder(dim_input=DIN, nz=NZ) if with_set_encoder else None
    return TCLSC(cls_dim=CLS_DIM, set_encoder=enc, embedding_dim=12, fz=10,
                 ctx_dim=ctx_dim, task_branch=task_branch)


class TestDefaultsUnchanged:
    """ctx_dim=0 + task_branch='E+ctx'（默认）必须与旧构造逐位一致——
    已训模型的 state_dict 可原样加载（消融臂共用同一份权重文件）。"""

    def test_state_dict_keys_and_values_match_legacy(self):
        torch.manual_seed(0)
        legacy = TCLSC(cls_dim=CLS_DIM, set_encoder=TaskSetEncoder(dim_input=DIN, nz=NZ),
                       embedding_dim=12, fz=10)
        new = _mk()
        assert list(legacy.state_dict()) == list(new.state_dict())
        for k, v in legacy.state_dict().items():
            assert torch.equal(v, new.state_dict()[k]), f"{k} 初始化不一致"
        assert new.set_fc[0].in_features == NZ      # ctx_dim=0 不拼接
        assert new.pred_fc[0].in_features == 2 * 10
        assert new.uses_E


class TestContextVector:
    @pytest.mark.parametrize("branch", ["E+ctx", "ctx", "none"])
    def test_forward_shape_and_range(self, branch):
        m = _mk(ctx_dim=CTX_DIM, task_branch=branch)
        a, b = _cls_batch(5, seed=11)
        E = m.encode_tasks([_windows(4)]).expand(5, -1)
        out = m(a, b, E, _ctx_batch(5, 12))
        assert out.shape == (5,)
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()

    def test_output_depends_on_ctx(self):
        m = _mk(ctx_dim=CTX_DIM)
        a, b = _cls_batch(5, seed=13)
        E = m.encode_tasks([_windows(4)]).expand(5, -1)
        assert not torch.allclose(m(a, b, E, _ctx_batch(5, 1)), m(a, b, E, _ctx_batch(5, 2)))

    def test_missing_ctx_raises(self):
        m = _mk(ctx_dim=CTX_DIM)
        a, b = _cls_batch(3, seed=14)
        E = m.encode_tasks([_windows(4)]).expand(3, -1)
        with pytest.raises(ValueError):
            m(a, b, E)

    def test_unexpected_ctx_raises(self):
        m = _mk()  # ctx_dim=0
        a, b = _cls_batch(3, seed=15)
        E = m.encode_tasks([_windows(4)]).expand(3, -1)
        with pytest.raises(ValueError):
            m(a, b, E, _ctx_batch(3, 16))

    def test_wrong_ctx_shape_raises(self):
        m = _mk(ctx_dim=CTX_DIM)
        a, b = _cls_batch(3, seed=17)
        E = m.encode_tasks([_windows(4)]).expand(3, -1)
        with pytest.raises(ValueError):
            m(a, b, E, _ctx_batch(3, 18)[:, :2])   # 维度不符
        with pytest.raises(ValueError):
            m(a, b, E, _ctx_batch(2, 19))          # 批大小不符

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            _mk(task_branch="nope")
        with pytest.raises(ValueError):
            _mk(ctx_dim=0, task_branch="ctx")            # ctx 支路无输入
        with pytest.raises(ValueError):
            _mk(ctx_dim=-1)
        with pytest.raises(ValueError):
            _mk(with_set_encoder=False)                  # E+ctx 支路需要 SetTF


class TestTaskBranchAblations:
    """task_branch 消融：'ctx' 只看任务设定、'none' 两者都不看（纯 CLS 排序器）。"""

    def _two_Es(self, m, n):
        E1 = m.encode_tasks([_windows(4, seed=1)]).expand(n, -1)
        E2 = m.encode_tasks([_windows(6, seed=2)]).expand(n, -1)
        assert not torch.allclose(E1, E2), "两份 E 必须不同，否则消融断言无意义"
        return E1, E2

    def test_none_branch_ignores_E_and_ctx(self):
        m = _mk(ctx_dim=CTX_DIM, task_branch="none")
        a, b = _cls_batch(5, seed=20)
        ctx = _ctx_batch(5, 21)
        E1, E2 = self._two_Es(m, 5)
        assert torch.equal(m(a, b, E1, ctx), m(a, b, E2, ctx))
        assert torch.equal(m(a, b, None, ctx), m(a, b, E1, ctx))
        assert torch.equal(m(a, b, E1, _ctx_batch(5, 22)), m(a, b, E1, ctx))
        assert not m.uses_E
        assert not hasattr(m, "set_fc")
        assert m.pred_fc[0].in_features == 10   # 只剩 CLS 支路

    def test_ctx_branch_ignores_E_but_uses_ctx(self):
        m = _mk(ctx_dim=CTX_DIM, task_branch="ctx")
        a, b = _cls_batch(5, seed=23)
        ctx = _ctx_batch(5, 24)
        E1, E2 = self._two_Es(m, 5)
        assert torch.equal(m(a, b, E1, ctx), m(a, b, E2, ctx))
        assert not torch.allclose(m(a, b, E1, ctx), m(a, b, E1, _ctx_batch(5, 25)))
        assert not m.uses_E
        assert m.set_fc[0].in_features == CTX_DIM

    def test_ctx_branch_works_without_set_encoder(self):
        m = _mk(ctx_dim=CTX_DIM, task_branch="ctx", with_set_encoder=False)
        a, b = _cls_batch(4, seed=26)
        out = m(a, b, None, _ctx_batch(4, 27))
        assert out.shape == (4,) and torch.isfinite(out).all()
        with pytest.raises(ValueError):
            m.encode_tasks([_windows(3)])
