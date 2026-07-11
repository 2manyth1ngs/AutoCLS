"""任务嵌入 E = SetTransformer(TS2Vec(样本序列集))（TECHNICAL_DESIGN.md 模块 5）。

流程对应 AutoCTS++ 的 exps/generate_task_feature.py（TS2Vec 训练+编码+采样）与
NAS_Net/AHC/ahc_engine.py::set_encode（两级 Set-Transformer 聚合），分类任务
语义差异见各模块 docstring。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .set_transformer import TaskSetEncoder
from .ts2vec_wrapper import TS2VecWrapper

__all__ = ["TS2VecWrapper", "TaskSetEncoder", "embed_task"]


def embed_task(
    X: np.ndarray,
    *,
    ts2vec_iters: Optional[int] = None,
    ts2vec: Optional[TS2VecWrapper] = None,
    sample_num: int = 100,     # ← generate_task_feature.py --sample_num 默认 100
    batch_size: int = 128,     # STD_BATCH_SIZE（configs/train.yaml）
    output_dims: int = 256,    # TS2Vec 表征维 = intra_setpool dim_input（ahc_engine.py）
    hidden_dims: int = 64,
    depth: int = 10,
    nz: int = 256,             # 任务嵌入维（ahc_engine.py self.nz）
    set_encoder: Optional[TaskSetEncoder] = None,
    device: Optional[str] = None,
    seed: int = 0,
) -> np.ndarray:
    """把一个（切片）任务的无标签样本集 X (N, T, C) 编码为定长任务向量 E (nz,)。

    流程（← generate_task_feature.py 主流程 + ahc_engine.py::set_encode）：
    1. 在该任务全部样本上无监督 fit TS2Vec（ts2vec=None 时；每任务独立训练一份
       权重，与原码逐数据集训练一致）；不接触任何标签——E 必须是"数据分布
       特征"，否则破坏 zero-shot 迁移前提。
    2. 采样 min(sample_num, N) 个样本作为"窗口"集合（原文窗口=预测滑窗，此处
       窗口=分类样本序列本身，无需 P+Q 滑窗）。
    3. TS2Vec.encode 得逐时间步表征 (m, T, output_dims)。原码先全量编码再采样；
       此处先采样再编码，结果等价且省显存（HAR 全量逐时间步表征 ≈1GB）。
    4. TaskSetEncoder 两级聚合（Eq.10-12）→ E (nz,)。

    【已知问题，设计待定】sanity check（results/task_embed_sanity/ +
    scripts/diag_task_embed_shared.py）表明：每任务独立重训 TS2Vec 时，E 被
    训练随机性（对比表征的基任意性）主导，同数据不同种子的 E 距离 ≥ 不同
    任务间距离，区分力≈0。共享冻结 TS2Vec（尤其单变量+通道均值，AutoCTS++
    generate_task_feature.py 原味）时 TS2Vec 级特征区分力清晰。跨任务流程应
    预训练一份共享 TS2Vec 后经 ts2vec= 传入；默认的每任务重训仅保留原语义。

    set_encoder 为 None 时在 torch.manual_seed(seed) 下新建：同 seed 多次调用得到
    同一份初始化权重，保证跨任务 E 可比（sanity check 用法）。正式流程（M3 起）
    应显式传入同一个实例——该编码器最终与 T-CLSC 联合训练。
    """
    assert X.ndim == 3, f"X 应为 (N, T, C)，得到 {X.shape}"
    if ts2vec is not None:
        device = ts2vec.device  # 与共享编码器同设备，避免张量搬运不一致
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if set_encoder is None:
        torch.manual_seed(seed)
        set_encoder = TaskSetEncoder(dim_input=output_dims, nz=nz)
    set_encoder = set_encoder.to(device).eval()

    if ts2vec is not None:
        wrapper = ts2vec  # 共享冻结编码器：不再 fit
    else:
        assert ts2vec_iters is not None, \
            "未提供预训练 ts2vec 时必须显式传 ts2vec_iters（训练量显式化，同铁律 2）"
        wrapper = TS2VecWrapper(
            input_dims=X.shape[-1], output_dims=output_dims, hidden_dims=hidden_dims,
            depth=depth, device=device, batch_size=batch_size, seed=seed,
        )
        wrapper.fit(X, n_iters=ts2vec_iters)

    rng = np.random.default_rng(seed)
    m = min(sample_num, X.shape[0])
    idx = np.sort(rng.choice(X.shape[0], size=m, replace=False))
    reprs = wrapper.encode(X[idx])  # (m, T, output_dims)

    with torch.no_grad():
        e = set_encoder(torch.from_numpy(reprs).to(torch.float32).to(device))
    return e.cpu().numpy()
