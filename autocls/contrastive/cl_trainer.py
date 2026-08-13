"""对比预训练循环（TECHNICAL_DESIGN.md 模块 3：给定 (cls, encoder, D_train) 训练 encoder）。

铁律 2：max_iters（optimizer.step() 参数更新次数）为 keyword-only 且无默认值——
early-validation（290 iters）与最终完整评估（2320 iters，见 configs/train.yaml，
基准 bs=128）必须由调用方显式区分，
禁止混用。计量单位用 iters 而非 epoch：不同大小的切片任务若按 epoch 截断，
实际参数更新次数不一致，会污染跨任务 early-val label（与 AutoCL 附录 B.3 的
iteration 计量一致）。

单个 batch 的流程（AutoCL Fig.1 的 CLS 四维度依次生效）：
    x ─augment_two_views(数据增强×order)→ (v1, v2, 共享段切片)
      ─encoder→ 逐时间步嵌入 ─取共享段─ (z1, z2)
      ─apply_emb_transform(嵌入变换)→ (z1', z2')
      ─total_contrastive_loss(对比对构造×损失)→ loss
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from ..space.cls_space import CLS
from .augmentations import augment_two_views
from .emb_transform import apply_emb_transform
from .pair_construction import total_contrastive_loss

_AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop")


def pretrain_encoder(
    encoder: nn.Module,
    cls_cfg: CLS,
    X_train: np.ndarray,
    *,
    max_iters: int,  # 参数更新次数，必须显式传参（铁律 2）
    batch_size: int = 128,  # STD_BATCH_SIZE（configs/train.yaml），M1 验证配置
    lr: float = 1e-3,
    device: Optional[str] = None,
    seed: int = 0,
    use_swa: bool = False,
    verbose: bool = False,
) -> list[float]:
    """按 cls_cfg 对 encoder 做对比预训练（就地更新权重），返回每遍数据的平均损失。

    恰好执行 max_iters 次 optimizer.step() 后停止；因 <2 样本被跳过的 batch 不计入。
    每完整遍历一遍数据后用同一 generator 重新 randperm 洗牌，故当 max_iters 为
    整遍步数的整数倍时，与旧的按 epoch 训练逐位等价。

    X_train: (N, T, C) float32。数据增强在 CPU 上执行（逐 batch 采样坐标/噪声），
    两个视图再搬运到 device 前向。

    use_swa（默认关，关闭时代码路径零改动）：按 TS2Vec 官方语义做等权权重平均
    （reference/AutoCTS_plusplus/NAS_Net/ts2vec.py，与 yuezhihan/ts2vec 逐行一致）——
    创建 AveragedModel 后立即快照 init 权重（ts2vec.py:53-54），此后每次
    optimizer.step() 后 update 一次（ts2vec.py:152），K 次更新 = {w_init, w_1..w_K}
    共 K+1 个快照的等权 running mean；官方 encode 用平均权重（ts2vec.py:177）、
    梯度只更新原始权重。此处训练结束把平均参数拷回 encoder（后续只有评估，与官方
    eval 走平均权重等价）。AveragedModel 不消耗 RNG，训练轨迹与关闭时逐位相同。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    swa_model = None
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(encoder)
        swa_model.update_parameters(encoder)  # init 快照（官方 ts2vec.py:54）

    aug_params = {name: getattr(cls_cfg, name) for name in _AUG_FIELDS}
    X = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
    N = X.shape[0]
    gen = torch.Generator().manual_seed(seed)  # 增强/打乱共用，保证可复现

    pass_losses: list[float] = []
    step = 0
    while step < max_iters:
        perm = torch.randperm(N, generator=gen)
        batch_losses: list[float] = []
        for start in range(0, N, batch_size):
            if step >= max_iters:
                break
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:  # B=1 无负样本实例，损失恒 0，跳过（不计入 step）
                continue
            x = X[idx]
            v1, v2, sl1, sl2 = augment_two_views(x, aug_params, order=cls_cfg.order, generator=gen)

            z1 = encoder(v1.to(device))[:, sl1]
            z2 = encoder(v2.to(device))[:, sl2]
            z1 = apply_emb_transform(z1, cls_cfg, generator=gen)
            z2 = apply_emb_transform(z2, cls_cfg, generator=gen)

            loss = total_contrastive_loss(z1, z2, cls_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if swa_model is not None:
                swa_model.update_parameters(encoder)  # 逐 iter（官方 ts2vec.py:152）
            step += 1
            batch_losses.append(loss.item())

        if not batch_losses:  # 整遍数据无一 batch 可训练，step 永不推进
            raise ValueError(f"N={N}, batch_size={batch_size}: 所有 batch 均 <2 样本，无法训练")
        mean_loss = float(np.mean(batch_losses))
        pass_losses.append(mean_loss)
        if verbose:
            print(f"[cl_trainer] iter {step}/{max_iters} loss={mean_loss:.4f}")
    if swa_model is not None:
        # swa_model.module 即被 update_parameters 就地平均的 deepcopy，其 state_dict
        # 不含 AveragedModel 自身的 n_averaged buffer，可直接拷回（encoder 无 buffer）。
        encoder.load_state_dict(swa_model.module.state_dict())
    return pass_losses
