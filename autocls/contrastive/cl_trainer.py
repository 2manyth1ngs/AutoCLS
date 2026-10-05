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

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..space.cls_space import CLS
from .augmentations import (CROP_MODES, _crop_per_row, _crop_with_coords, augment_two_views, sample_crop_coords,
                            sample_crop_coords_ts2vec)
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
    checkpoints: Sequence[int] = (),
    on_checkpoint: Optional[Callable[[int, nn.Module], None]] = None,
    drop_last: bool = False,
    crop_mode: str = "fixed",
    aug_learn: str = "none",
    y_train: Optional[np.ndarray] = None,
    aug_info: Optional[dict] = None,
) -> list[float]:
    """按 cls_cfg 对 encoder 做对比预训练（就地更新权重），返回每遍数据的平均损失。

    drop_last / crop_mode（全量协议对齐 results/protocol_alignment 的 opt-in 开关，
    默认关、默认路径逐位不变——tests/test_cl_trainer.py::TestDropLastAndCropMode 金标）：
    - drop_last=True：每遍数据丢掉不足 batch_size 的残批（TS2Vec 官方
      DataLoader(..., drop_last=True)，reference ts2vec.py:93-94），残批不计 step；
      N<batch_size 时整遍无可训 batch → ValueError（与既有 <2 样本判据同一出口）。
    - crop_mode="ts2vec"：crop 增强改走官方逐迭代 crop_l~U[2,T] + 逐行偏移
      （augmentations.sample_crop_coords_ts2vec，ts2vec.py:126-132）；"fixed" 为既有口径。

    aug_learn（WP17-ACTS AL，默认 "none" = 既有路径逐位不变，tests/test_learned_aug.py 钉死）："sup" = InfoTS 式学增强
    （autocls/contrastive/learned_aug.py，忠实于官方代码）：CLS 的 5 个非 crop 增强与 order 不生效，第一视图 = 7 种变换的学习加权混合
    （每 batch 以 0.2 的概率，否则原样）、第二视图 = 原样，之后接 CLS 的 crop（两视图共享坐标）；每第 2、4… 遍开始前做一遍元学习步
    （只用 y_train 的标签交叉熵，不计入 max_iters）。y_train 必须给出；aug_info 若给出则回填 meta_steps / weights / weights_init。

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

    checkpoints / on_checkpoint（MX-3-PQ：一条轨迹上取多个 budget 的 R'）：
    每次 optimizer.step() 后 step 自增，若 step 命中 checkpoints 就以
    `(step, encoder)` 调一次 on_checkpoint——回调跑的是下游评估器（内部
    encoder.eval() + no_grad 前向），故回调返回后本函数立即 encoder.train() 恢复
    训练模式，否则截点之后的 iters 会静默跑在 eval 模式下变成另一条轨迹。
    评估不碰 optimizer 状态、也不消耗本函数的 gen（增强/打乱用独立
    torch.Generator），故打不打截点的最终权重逐位相同（tests/test_cl_trainer.py
    的 bit-identical 用例是这条性质的护栏）——budget=290 的 R' 因此仍是原口径。
    checkpoints 必须是 1..max_iters 内严格递增的整数（越界/重复/乱序都意味着
    某个 budget 永不触发 = 静默少一条 R'，一律 ValueError）。

    use_swa × checkpoints（MX-4 WP3：官方协议 ts2vec_official 开 SWA，淘汰赛又要在同一条
    轨迹上取截点）：截点把 AveragedModel 的**当前平均权重**（swa_model.module）交给回调，
    而不是正在训练的原始权重。TS2Vec 的 SWA 是等权 running mean，任意步 b 的快照就是
    "单独训 b 步的 SWA 终值"（同 seed 逐位相同，tests/test_cl_trainer.py 钉死）；原始权重、
    optimizer 与随机流不受回调影响，终点权重与不打截点逐位相同。
    """
    checkpoints = tuple(int(c) for c in checkpoints)
    if checkpoints:
        if list(checkpoints) != sorted(set(checkpoints)):
            raise ValueError(f"checkpoints={checkpoints} 必须严格递增且不重复")
        if checkpoints[0] < 1 or checkpoints[-1] > max_iters:
            raise ValueError(
                f"checkpoints={checkpoints} 必须落在 1..max_iters={max_iters} 内，"
                "否则该截点永不触发（静默少一条 R'）")
    ckpt_set = set(checkpoints)
    if crop_mode not in CROP_MODES:
        raise ValueError(f"未知 crop_mode={crop_mode!r}（可选 {CROP_MODES}）")
    if aug_learn not in ("none", "sup"):
        raise ValueError(f"未知 aug_learn={aug_learn!r}（可选 none / sup）")
    if aug_learn == "sup" and y_train is None:
        raise ValueError("aug_learn='sup' 需要 y_train（官方 TRAIN 标签）")

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

    la = None
    if aug_learn == "sup":                       # 学增强（默认路径不进这里，不消耗任何随机数）
        from . import learned_aug as la
        _, y_idx = np.unique(np.asarray(y_train), return_inverse=True)
        Y = torch.as_tensor(y_idx, dtype=torch.long)
        aug = la.AutoAug(seed=seed)
        with torch.no_grad():
            was = encoder.training; encoder.eval(); D = int(encoder(X[:2].to(device)).shape[-1]); encoder.train(was)
        head = nn.Linear(D, int(Y.max()) + 1).to(device)
        meta_opt = torch.optim.AdamW(aug.parameters(), lr=la.META_LR)
        cls_opt = torch.optim.AdamW(head.parameters(), lr=la.META_LR)
        w_init = aug.weights(); meta_steps = 0

    pass_losses: list[float] = []
    step = 0
    pass_idx = 0
    while step < max_iters:
        if la is not None and (pass_idx + 1) % la.META_EVERY == 0:      # 官方 (n_epochs + 1) % meta_epoch == 0 → 本遍开始前 meta_fit
            meta_steps += la.meta_pass(aug, head, encoder, X, Y, batch_size=batch_size, device=device, gen=gen,
                                       drop_last=drop_last, meta_opt=meta_opt, cls_opt=cls_opt)
        perm = torch.randperm(N, generator=gen)
        batch_losses: list[float] = []
        for start in range(0, N, batch_size):
            if step >= max_iters:
                break
            idx = perm[start : start + batch_size]
            if drop_last and idx.numel() < batch_size:  # 官方 drop_last：残批不训不计 step
                continue
            if idx.numel() < 2:  # B=1 无负样本实例，损失恒 0，跳过（不计入 step）
                continue
            x = X[idx]
            if la is None:
                v1, v2, sl1, sl2 = augment_two_views(x, aug_params, order=cls_cfg.order, generator=gen,
                                                     crop_mode=crop_mode)
            else:
                with torch.no_grad():
                    v1 = aug.view(x, gen)
                v2 = x
                if cls_cfg.crop > 0.0:
                    if crop_mode == "ts2vec":
                        v1, v2, sl1, sl2 = _crop_per_row(v1, v2, sample_crop_coords_ts2vec(v1.shape[0], v1.shape[1], gen))
                    else:
                        v1, v2, sl1, sl2 = _crop_with_coords(v1, v2, sample_crop_coords(v1.shape[1], cls_cfg.crop, gen))
                else:
                    sl1 = sl2 = slice(0, v1.shape[1])

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
            if step in ckpt_set and on_checkpoint is not None:
                # SWA 开启时交出当前平均权重的模块（= 训到此步的 SWA 终值），否则交原始 encoder
                target = swa_model.module if swa_model is not None else encoder
                on_checkpoint(step, target)
                encoder.train()   # 回调里的评估器会 encoder.eval()，必须恢复
                if swa_model is not None:
                    swa_model.module.train()
            batch_losses.append(loss.item())

        if not batch_losses:  # 整遍数据无一 batch 可训练，step 永不推进
            raise ValueError(
                f"N={N}, batch_size={batch_size}, drop_last={drop_last}: 整遍数据无可训 "
                "batch（全部 <2 样本，或 drop_last 下 N<batch_size），无法训练")
        mean_loss = float(np.mean(batch_losses))
        pass_losses.append(mean_loss)
        pass_idx += 1
        if verbose:
            print(f"[cl_trainer] iter {step}/{max_iters} loss={mean_loss:.4f}")
    if la is not None and aug_info is not None:
        aug_info.update({"meta_steps": int(meta_steps), "weights": aug.weights(), "weights_init": w_init,
                         "transforms": list(la.TRANSFORM_NAMES)})
    if swa_model is not None:
        # swa_model.module 即被 update_parameters 就地平均的 deepcopy，其 state_dict
        # 不含 AveragedModel 自身的 n_averaged buffer，可直接拷回（encoder 无 buffer）。
        encoder.load_state_dict(swa_model.module.state_dict())
    return pass_losses
