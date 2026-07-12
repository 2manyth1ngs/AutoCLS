"""T-CLSC 比较器模型：(任务嵌入 E, cls_a, cls_b) → P(cls_a 优于 cls_b)（TECHNICAL_DESIGN.md 模块 6，核心创新）。

结构对齐 AutoCTS++（VLDBJ 2024）Fig.4 / reference/AutoCTS_plusplus/NAS_Net/AHC/
ahc_engine.py::AHC，**唯一实质改动**是把 arch-hyper 的 GIN 图编码换成 CLS 配置
向量的 MLP 编码（关键决策 1：CLS 编码用向量+MLP，不用图/GIN）：

- CLS 支路（← AHC embedding+GCN → 此处 MLP）：v = cls_encoder.encode(cls) 经共享
  MLP 得 l_a、l_b（Eq.15：两个 CLS 共用同一 MLP，可比性来自权重共享），
  concat 成 L_a（Eq.16 ← ahc_engine.py L163 `torch.cat([embedd1, embedd2])`）。
- 任务支路：E = SetTransformer(windows_repr)（Eq.9-12，task_embed 模块产出）
  经 set_fc 得 E'（Eq.18 ← L119-120 `set_fc`）。SetTF（TaskSetEncoder）由本模型
  持有并联合训练（← 原码 intra/inter_setpool 内嵌于 AHC，L109-118）。
- 融合头：concat(L_a', E') → 两层 FC → Sigmoid（Eq.17/19-21 ← L122-126
  `graph_fc`/`pred_fc` + L169 sigmoid），BCE 训练。

任务无关模块（铁律 7）：本文件不 import task_types，不假设下游任务语义——
"y=1 是什么方向"由造 label 侧（labeling.is_better）定义，模型只拟合二值目标。

【定位（路线 B）】本模型当前在 HAR 同分布切片任务上做机制验证（M3），
留出任务准确率只能说明"泛化到新任务切片"，不构成 zero-shot（跨数据集）证据；
真 zero-shot 结果由路线 A（多 UEA 数据集、源/目标池分离）产出。
"""
from __future__ import annotations

import torch
from torch import nn

from ..task_embed.set_transformer import TaskSetEncoder


class TCLSC(nn.Module):
    """比较器主体。前向输入已编码的 CLS 向量批与任务嵌入批，输出优劣概率。

    参数对齐 ahc_engine.py::AHC.__init__（L97-126）：embedding_dim ≙ 原码
    embedding_dim=128（GIN 输出维，此处为 CLS MLP 输出维）、fz ≙ self.fz=128、
    ratio=2（两个 CLS / 两条支路 concat）固化在层宽里。
    """

    def __init__(
        self,
        cls_dim: int,
        set_encoder: TaskSetEncoder,
        embedding_dim: int = 128,
        fz: int = 128,
    ):
        super().__init__()
        self.set_encoder = set_encoder  # 联合训练（← 原码 setpool 内嵌于 AHC）
        # CLS 支路：共享 MLP（Eq.15；替换原码 embedding+GCN，唯一实质改动）
        self.cls_mlp = nn.Sequential(
            nn.Linear(cls_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # ← ahc_engine.py L122-123 graph_fc：Linear(embedding_dim*ratio, fz)+ReLU
        self.cls_fc = nn.Sequential(nn.Linear(embedding_dim * 2, fz), nn.ReLU())
        # ← ahc_engine.py L119-120 set_fc：Linear(nz, fz)+ReLU（Eq.18）
        self.set_fc = nn.Sequential(nn.Linear(set_encoder.nz, fz), nn.ReLU())
        # ← ahc_engine.py L125-126 pred_fc：两层 FC（Eq.19-21）
        self.pred_fc = nn.Sequential(
            nn.Linear(fz * 2, fz), nn.ReLU(), nn.Linear(fz, 1),
        )

    def encode_tasks(self, windows_list: list[torch.Tensor]) -> torch.Tensor:
        """一批任务的窗口表征 [(m_i, T, D)] → E (len, nz)（Eq.9-12，梯度可回传）。"""
        return self.set_encoder.set_encode(windows_list)

    def forward(
        self, cls_a: torch.Tensor, cls_b: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """(B, cls_dim) × 2 + E (B, nz) → P(cls_a 优于 cls_b) (B,)。

        E 由 encode_tasks 产出（训练时同批去重后展开，梯度经此回传 SetTF）；
        消融检验（E 置零/错位）直接替换 E 输入，无需改模型。
        """
        l_a = self.cls_mlp(cls_a)                       # Eq.15
        l_b = self.cls_mlp(cls_b)                       # Eq.15（共享同一 MLP）
        L = torch.cat([l_a, l_b], dim=1)                # Eq.16 ← L163
        logits = self.pred_fc(                          # Eq.19-21 ← L168
            torch.cat([self.cls_fc(L), self.set_fc(E)], dim=1)
        ).squeeze(1)
        return torch.sigmoid(logits)                    # ← L169
