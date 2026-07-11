"""TS2Vec 封装（TECHNICAL_DESIGN.md 模块 5；铁律 4：照 AutoCTS++ 真实源码适配）。

来源（reference/AutoCTS_plusplus，逐段标注）：
- TSEncoder / 掩码生成      ← NAS_Net/ts2vec_model/encoder.py（原样搬运）
- DilatedConvEncoder 等     ← NAS_Net/ts2vec_model/dilated_conv.py（原样搬运）
- hierarchical_contrastive_loss ← NAS_Net/ts2vec_model/losses.py（原样搬运）
- take_per_row              ← NAS_Net/ts2vec_model/utils.py（原样搬运）
- TS2VecWrapper             ← NAS_Net/ts2vec.py::TS2Vec（适配，差异见类 docstring）

论文对应：AutoCTS++（VLDBJ 2024）任务嵌入第一级——对每个窗口 w 用 TS2Vec 得
逐时间步表征 H_w（论文 Eq.9；Eq.10-12 的两级 Set-Transformer 聚合见
set_transformer.py）。

用途约束：本封装只服务任务嵌入 E（数据分布特征），**无监督**（fit 不接触任何
标签）——E 携带标签信息会破坏 zero-shot 迁移到未见数据集的前提。它与
autocls/encoders/dilated_cnn.py（模块 4 的固定 encoder，被 CLS 预训练的对象）
职责不同，勿混用。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 以下到 TS2VecWrapper 之前均原样搬运自 reference/AutoCTS_plusplus/NAS_Net/
# ts2vec_model/{dilated_conv,encoder,losses,utils}.py（TS2Vec, AAAI 2022 官方实现）
# ============================================================================


class SamePadConv(nn.Module):
    # ← ts2vec_model/dilated_conv.py::SamePadConv
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    # ← ts2vec_model/dilated_conv.py::ConvBlock
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    # ← ts2vec_model/dilated_conv.py::DilatedConvEncoder
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


def generate_continuous_mask(B, T, n=5, l=0.1):
    # ← ts2vec_model/encoder.py::generate_continuous_mask
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    # ← ts2vec_model/encoder.py::generate_binomial_mask
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    # ← ts2vec_model/encoder.py::TSEncoder（原样）
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        if mask is None:
            mask = self.mask_mode if self.training else 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        return x


def instance_contrastive_loss(z1, z2):
    # ← ts2vec_model/losses.py::instance_contrastive_loss
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    # ← ts2vec_model/losses.py::temporal_contrastive_loss
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    # ← ts2vec_model/losses.py::hierarchical_contrastive_loss
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def take_per_row(A, indx, num_elem):
    # ← ts2vec_model/utils.py::take_per_row
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


# ============================================================================
# 适配层
# ============================================================================


class TS2VecWrapper:
    """适配 NAS_Net/ts2vec.py::TS2Vec，训练/编码逻辑照原码，差异如下。

    与原码的差异（分类任务语义，TECHNICAL_DESIGN.md 模块 5）：
    1. 删除 max_train_length 切分与 NaN 居中/剔除（原码服务变长预测长序列；
       分类样本是定长短序列且无缺失，如 HAR (N,128,9)）。
    2. 删除 encode 的 sliding_length 滑窗推断（同上：原文"窗口"是预测滑窗
       S=P+Q；分类里一个"窗口"=一个样本序列本身，直接整段编码）。
    3. 可复现性：原码靠外部 init_dl_program 设全局种子；此处收进构造参数 seed
       （__init__ 设权重初始化种子；fit 设全局 np/torch 种子——搬运代码内部用
       全局 np.random 生成掩码/裁剪坐标，无法用局部 generator 替换）。
    4. batch_size 默认 128 = STD_BATCH_SIZE（configs/train.yaml；原码默认 16）。
    5. fit 只接受数据不接受标签（无监督），训练量按 n_iters（参数更新次数）
       精确截断——与铁律 2 的 iters 口径一致。
    6. 新增 freeze()：共享任务嵌入编码器语义（铁律 6）——在源池上训完后冻结，
       冻结后 fit 报错，防误再训练/防 zero-shot 数据泄漏。

    原码保留项：SWA AveragedModel 权重平均、随机双段裁剪 + 层级对比损失、
    binomial 掩码、AdamW。
    """

    def __init__(
        self,
        input_dims: int,
        *,
        output_dims: int = 256,   # 与 AHC set_encoder dim_input=256 对齐（ahc_engine.py）
        hidden_dims: int = 64,
        depth: int = 10,
        device: Optional[str] = None,
        lr: float = 1e-3,
        batch_size: int = 128,    # STD_BATCH_SIZE（configs/train.yaml）
        seed: int = 0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.temporal_unit = 0  # 原码默认：短序列无需减层

        torch.manual_seed(seed)  # 权重初始化可复现（差异 3）
        self._net = TSEncoder(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, depth=depth,
        ).to(self.device)
        # ← ts2vec.py L53-54：SWA 平均权重，encode 用 self.net、训练更新 self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.n_iters = 0  # 已执行的参数更新次数（跨多次 fit 累计，同原码）
        self.frozen = False  # freeze() 后为 True，fit 拒绝执行（差异 6）

    @property
    def input_dims(self) -> int:
        return self._net.input_dims

    @property
    def output_dims(self) -> int:
        return self._net.output_dims

    def freeze(self) -> None:
        """冻结（差异 6）：此后 fit 报错。共享任务嵌入编码器在源池训完后必须
        冻结，目标池/未见数据集绝不参与训练（铁律 6，防 zero-shot 数据泄漏）。"""
        self.frozen = True

    def fit(self, train_data: np.ndarray, *, n_iters: int, verbose: bool = False) -> list[float]:
        """无监督对比训练（← ts2vec.py::TS2Vec.fit，删减见类 docstring）。

        train_data: (N, T, C) float32，一个任务的全部样本序列（不含标签）。
        n_iters: 目标参数更新次数（含历史累计），必须显式传参。
        返回每完整遍历一遍数据的平均损失（末尾不足一遍的不计入，同原码）。
        """
        if self.frozen:
            raise RuntimeError(
                "TS2Vec 已冻结（共享任务嵌入编码器，铁律 6）：禁止再训练，"
                "防 zero-shot 数据泄漏。新任务请直接 encode。"
            )
        assert train_data.ndim == 3
        # 差异 3：搬运代码（binomial 掩码、裁剪坐标）依赖全局 np.random，
        # 这里统一设全局种子保证可复现（原码由 init_dl_program 承担）
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True, drop_last=True,
        )
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        loss_log: list[float] = []
        self._net.train()
        while True:  # ← ts2vec.py L100-173 双层循环，按 n_iters 截断
            cum_loss = 0.0
            n_epoch_iters = 0
            interrupted = False
            for batch in train_loader:
                if self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0].to(self.device)

                # ← ts2vec.py L123-142：随机双段重叠裁剪，两视图在重叠段对比
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
                )

                optimizer.zero_grad()
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(out1, out2, temporal_unit=self.temporal_unit)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1

            if interrupted:
                break

            cum_loss /= max(n_epoch_iters, 1)
            loss_log.append(cum_loss)
            if verbose:
                print(f"pass loss={cum_loss}")

        return loss_log

    def _eval_with_pooling(self, x, mask=None, encoding_window=None):
        # ← ts2vec.py::TS2Vec._eval_with_pooling（删 slicing 参数：无滑窗）
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            out = F.max_pool1d(
                out.transpose(1, 2), kernel_size=out.size(1)
            ).transpose(1, 2)
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2), kernel_size=encoding_window,
                stride=1, padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
        return out.cpu()

    def encode(
        self,
        data: np.ndarray,
        *,
        mask=None,
        encoding_window=None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """← ts2vec.py::TS2Vec.encode（删 sliding_length 分支，差异 2）。

        data: (N, T, C)。返回逐时间步表征 (N, T, D)；
        encoding_window='full_series' 时返回全序列最大池化 (N, D)。
        """
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                out = self._eval_with_pooling(batch[0], mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)
                output.append(out)
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
