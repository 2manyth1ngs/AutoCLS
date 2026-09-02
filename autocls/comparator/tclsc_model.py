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

任务设定向量 ctx（MX-3-PQ 新增）：原码的任务 = (数据集, P, Q)——P/Q 是显式的
"任务设定"，与数据集特征一起进任务支路。本项目的对应物是训练预算等设定量，
以一段**通用上下文向量** ctx 拼在 E 后面进 set_fc（论文未给全，按 X 语义补全：
原文只给了 task_feature 一路输入，未写 P/Q 的拼接位置与编码形式）。
任务无关模块（铁律 7）：本文件不 import task_types，不假设下游任务语义——
"y=1 是什么方向"由造 label 侧（labeling.is_better）定义，模型只拟合二值目标；
ctx 只是一段浮点向量，其语义（预算 one-hot / 别的设定量）全留在调用方。

【定位（路线 B）】本模型当前在 HAR 同分布切片任务上做机制验证（M3），
留出任务准确率只能说明"泛化到新任务切片"，不构成 zero-shot（跨数据集）证据；
真 zero-shot 结果由路线 A（多 UEA 数据集、源/目标池分离）产出。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..task_embed.set_transformer import TaskSetEncoder

TASK_BRANCHES = ("E+ctx", "ctx", "none")


class TCLSC(nn.Module):
    """比较器主体。前向输入已编码的 CLS 向量批 + 任务嵌入批 (+ 任务设定 ctx)，输出优劣概率。

    参数对齐 ahc_engine.py::AHC.__init__（L97-126）：embedding_dim ≙ 原码
    embedding_dim=128（GIN 输出维，此处为 CLS MLP 输出维）、fz ≙ self.fz=128、
    ratio=2（两个 CLS / 两条支路 concat）固化在层宽里。

    task_branch 是任务支路的消融开关（不重训无法切换，故做成构造参数）：
    - "E+ctx"（默认）：set_fc 吃 cat([E, ctx])，ctx_dim=0 时退化为只吃 E——
      此档 + ctx_dim=0 与旧构造逐位一致（层序/形状/初始化都不变，
      已训 state_dict 可原样加载）；
    - "ctx"：set_fc 只吃 ctx（E 完全不进前向，set_encoder 可为 None）；
    - "none"：整条任务支路去掉（pred_fc 只吃 CLS 支路），纯 CLS 排序器。
    ctx 当且仅当 ctx_dim>0 时必传，形状 (B, ctx_dim)（铁律 7：只是浮点向量）。
    """

    def __init__(
        self,
        cls_dim: int,
        set_encoder: Optional[TaskSetEncoder],
        embedding_dim: int = 128,
        fz: int = 128,
        *,
        ctx_dim: int = 0,
        task_branch: str = "E+ctx",
    ):
        super().__init__()
        if task_branch not in TASK_BRANCHES:
            raise ValueError(f"未知 task_branch={task_branch!r}（可选 {TASK_BRANCHES}）")
        if ctx_dim < 0:
            raise ValueError(f"ctx_dim 须 >=0，得到 {ctx_dim}")
        if task_branch == "ctx" and ctx_dim == 0:
            raise ValueError("task_branch='ctx' 需 ctx_dim>0（任务支路否则没有输入）")
        if task_branch == "E+ctx" and set_encoder is None:
            raise ValueError("task_branch='E+ctx' 需要 set_encoder")
        self.ctx_dim = int(ctx_dim)
        self.task_branch = task_branch
        self.set_encoder = set_encoder  # 联合训练（← 原码 setpool 内嵌于 AHC）
        # CLS 支路：共享 MLP（Eq.15；替换原码 embedding+GCN，唯一实质改动）
        self.cls_mlp = nn.Sequential(
            nn.Linear(cls_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # ← ahc_engine.py L122-123 graph_fc：Linear(embedding_dim*ratio, fz)+ReLU
        self.cls_fc = nn.Sequential(nn.Linear(embedding_dim * 2, fz), nn.ReLU())
        # ← ahc_engine.py L119-120 set_fc：Linear(nz, fz)+ReLU（Eq.18）；
        #   ctx 拼在 E 后一起进（原码 P/Q 设定的类比，论文未给全，按 X 语义补全）
        if task_branch == "E+ctx":
            self.set_fc = nn.Sequential(nn.Linear(set_encoder.nz + self.ctx_dim, fz), nn.ReLU())
        elif task_branch == "ctx":
            self.set_fc = nn.Sequential(nn.Linear(self.ctx_dim, fz), nn.ReLU())
        # ← ahc_engine.py L125-126 pred_fc：两层 FC（Eq.19-21）
        self.pred_fc = nn.Sequential(
            nn.Linear(fz * (1 if task_branch == "none" else 2), fz), nn.ReLU(),
            nn.Linear(fz, 1),
        )

    @property
    def uses_E(self) -> bool:
        """任务嵌入是否进前向（消融档 ctx/none 为 False——引擎据此跳过 SetTF）。"""
        return self.task_branch == "E+ctx"

    def encode_tasks(self, windows_list: list[torch.Tensor]) -> torch.Tensor:
        """一批任务的窗口表征 [(m_i, T, D)] → E (len, nz)（Eq.9-12，梯度可回传）。"""
        if self.set_encoder is None:
            raise ValueError(f"task_branch={self.task_branch!r} 的模型未持有 set_encoder")
        return self.set_encoder.set_encode(windows_list)

    def _check_ctx(self, ctx: Optional[torch.Tensor], n: int) -> None:
        """ctx 当且仅当 ctx_dim>0 时必传（形状 (B, ctx_dim)），错配一律 fail-fast。"""
        if self.ctx_dim == 0:
            if ctx is not None:
                raise ValueError("ctx_dim=0 的模型不接受 ctx 输入")
            return
        if ctx is None:
            raise ValueError(f"ctx_dim={self.ctx_dim} 的模型必须传 ctx (B, {self.ctx_dim})")
        if tuple(ctx.shape) != (n, self.ctx_dim):
            raise ValueError(f"ctx 形状应为 ({n}, {self.ctx_dim})，得到 {tuple(ctx.shape)}")

    def forward(
        self, cls_a: torch.Tensor, cls_b: torch.Tensor, E: Optional[torch.Tensor],
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(B, cls_dim) × 2 + E (B, nz) + ctx (B, ctx_dim) → P(cls_a 优于 cls_b) (B,)。

        E 由 encode_tasks 产出（训练时同批去重后展开，梯度经此回传 SetTF）；
        消融检验（E/ctx 置零、错位、指派）直接替换输入，无需改模型。
        task_branch 为 "ctx"/"none" 时 E 不进前向，允许传 None。
        """
        self._check_ctx(ctx, cls_a.shape[0])
        l_a = self.cls_mlp(cls_a)                       # Eq.15
        l_b = self.cls_mlp(cls_b)                       # Eq.15（共享同一 MLP）
        L = torch.cat([l_a, l_b], dim=1)                # Eq.16 ← L163
        h = self.cls_fc(L)                              # Eq.17
        if self.task_branch == "E+ctx":
            if E is None:
                raise ValueError("task_branch='E+ctx' 必须传 E")
            task_in = E if self.ctx_dim == 0 else torch.cat([E, ctx], dim=1)
            h = torch.cat([h, self.set_fc(task_in)], dim=1)   # Eq.18
        elif self.task_branch == "ctx":
            h = torch.cat([h, self.set_fc(ctx)], dim=1)
        logits = self.pred_fc(h).squeeze(1)             # Eq.19-21 ← L168
        return torch.sigmoid(logits)                    # ← L169
