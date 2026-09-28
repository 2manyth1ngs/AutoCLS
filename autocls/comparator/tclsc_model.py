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
from .cls_graph import GraphCLSEncoder, GraphSpec

CLS_ENCODERS = ("mlp", "graph")

TASK_BRANCHES = ("E+ctx", "ctx", "none")
# WP16-TCLSC-L3 融合头（协议 results/tclsc_l3/PROTOCOL.md §2）：
#   concat：式 17–21 原样（默认，逐位不变）；
#   diff：logit = s(l_a, t) − s(l_b, t)，s = MLP([l, t])——反对称由构造保证，任务向量 t 的任何加性项相减抵消，
#         任务信息只能经与 CLS 嵌入的交互起作用（堵死实验 1 F2 测到的"E′ 当偏置"退化解）；
#   film：t → (γ, β) 调制 CLS 嵌入 l̃ = (1+γ)⊙l + β（式 15 的位置），再 s(l̃) 打分相减。
#   field（WP16 阶段 1 R 检验，results/tclsc_s1ea_rank/PROTOCOL.md §1）：logit = s₀(l_a) − s₀(l_b) + ⟨A·B·t, x_a − x_b⟩——
#         任务向量只进"字段级加性项"（x = binned 向量，逐字段逐档一个权重，无偏置），数据集维秩 ≤ field_rank；
#         组合 / 交互效应只留给任务盲主项 s₀。只支持 cls_encoder="mlp"（加性项吃的是原 binned 向量）。
HEADS = ("concat", "diff", "film", "field")


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

    HEADS = HEADS   # 融合头取值（类属性，调用方/脚本据此校验）

    def __init__(
        self,
        cls_dim: int,
        set_encoder: Optional[TaskSetEncoder],
        embedding_dim: int = 128,
        fz: int = 128,
        *,
        ctx_dim: int = 0,
        task_branch: str = "E+ctx",
        head: str = "concat",
        cls_encoder: str = "mlp",
        graph_opts: Optional[dict] = None,
        field_rank: Optional[int] = None,
    ):
        super().__init__()
        if task_branch not in TASK_BRANCHES:
            raise ValueError(f"未知 task_branch={task_branch!r}（可选 {TASK_BRANCHES}）")
        if head not in HEADS:
            raise ValueError(f"未知 head={head!r}（可选 {HEADS}）")
        if cls_encoder not in CLS_ENCODERS:
            raise ValueError(f"未知 cls_encoder={cls_encoder!r}（可选 {CLS_ENCODERS}）")
        if ctx_dim < 0:
            raise ValueError(f"ctx_dim 须 >=0，得到 {ctx_dim}")
        if task_branch == "ctx" and ctx_dim == 0:
            raise ValueError("task_branch='ctx' 需 ctx_dim>0（任务支路否则没有输入）")
        if task_branch == "E+ctx" and set_encoder is None:
            raise ValueError("task_branch='E+ctx' 需要 set_encoder")
        if head == "field" and cls_encoder != "mlp":
            raise ValueError("head='field' 的加性项吃 binned 向量，只支持 cls_encoder='mlp'")
        if field_rank is not None and int(field_rank) <= 0:
            raise ValueError(f"field_rank 须 >0 或 None，得到 {field_rank}")
        self.field_rank = None if field_rank is None else int(field_rank)
        self.ctx_dim = int(ctx_dim)
        self.task_branch = task_branch
        self.head = head
        self.cls_encoder = cls_encoder
        self.graph_opts = dict(graph_opts) if graph_opts else None
        self.set_encoder = set_encoder  # 联合训练（← 原码 setpool 内嵌于 AHC）
        if cls_encoder == "graph":
            # WP16-L2：CLS 支路 = 流水线图 + 共享 GIN（式 13–15），吃 GraphPacker 的打包向量；变量名沿用 cls_mlp（前向同一位置）
            if not self.graph_opts:
                raise ValueError("cls_encoder='graph' 需要 graph_opts（topology / readout / scalar / gin_layers）")
            from ..space.cls_space import CLSSpace
            spec = GraphSpec.from_space(CLSSpace.from_yaml())
            if int(cls_dim) != spec.pack_dim:
                raise ValueError(f"graph 编码的 cls_dim 应为 {spec.pack_dim}，得到 {cls_dim}")
            self.cls_mlp = GraphCLSEncoder(spec, embedding_dim, n_layers=int(self.graph_opts.get("gin_layers", 2)),
                                           readout=str(self.graph_opts.get("readout", "hub")),
                                           use_scalar=bool(self.graph_opts.get("scalar", True)))
        else:
            # CLS 支路：共享 MLP（Eq.15；替换原码 embedding+GCN，唯一实质改动）
            self.cls_mlp = nn.Sequential(
                nn.Linear(cls_dim, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        has_task = task_branch != "none"
        # 构造顺序 = RNG 消耗顺序：concat 头必须保持 cls_mlp → cls_fc → set_fc → pred_fc（与改头前逐位相同，
        # 否则同 seed 的初始化会变、旧存档/旧读数不可复现——WP16-L3 C0 曾因此失守 0.008）。
        if head == "concat":
            # ← ahc_engine.py L122-123 graph_fc：Linear(embedding_dim*ratio, fz)+ReLU
            self.cls_fc = nn.Sequential(nn.Linear(embedding_dim * 2, fz), nn.ReLU())
        # ← ahc_engine.py L119-120 set_fc：Linear(nz, fz)+ReLU（Eq.18）；
        #   ctx 拼在 E 后一起进（原码 P/Q 设定的类比，论文未给全，按 X 语义补全）
        if task_branch == "E+ctx":
            self.set_fc = nn.Sequential(nn.Linear(set_encoder.nz + self.ctx_dim, fz), nn.ReLU())
        elif task_branch == "ctx":
            self.set_fc = nn.Sequential(nn.Linear(self.ctx_dim, fz), nn.ReLU())
        if head == "concat":
            # ← ahc_engine.py L125-126 pred_fc：两层 FC（Eq.19-21）
            self.pred_fc = nn.Sequential(
                nn.Linear(fz * (2 if has_task else 1), fz), nn.ReLU(),
                nn.Linear(fz, 1),
            )
        elif head == "diff":
            # s(l, t)：与 pred_fc 同深同宽（Linear+ReLU+Linear），只是逐侧打分
            self.score_mlp = nn.Sequential(
                nn.Linear(embedding_dim + (fz if has_task else 0), fz), nn.ReLU(),
                nn.Linear(fz, 1),
            )
        elif head == "film":
            if has_task:
                self.film = nn.Linear(fz, 2 * embedding_dim)
            self.score_mlp = nn.Sequential(nn.Linear(embedding_dim, fz), nn.ReLU(), nn.Linear(fz, 1))
        else:   # field：任务盲主项 s₀ + 低秩字段加性项 w(t) = A·B·t（新头的模块都在其他头之后构造，不动它们的 RNG 顺序）
            self.score_mlp = nn.Sequential(nn.Linear(embedding_dim, fz), nn.ReLU(), nn.Linear(fz, 1))
            if has_task:
                r = self.field_rank or fz
                self.field_w = nn.Sequential(nn.Linear(fz, r, bias=False), nn.Linear(r, cls_dim, bias=False))

    @property
    def cls_in_dim(self) -> int:
        """CLS 支路吃的向量维度（mlp：cls_mlp[0].in_features；graph：打包维度）。"""
        return int(self.cls_mlp.in_features if self.cls_encoder == "graph" else self.cls_mlp[0].in_features)

    @property
    def antisymmetric(self) -> bool:
        """diff / film 头按构造满足 P(a,b) + P(b,a) = 1；concat 头只靠数据侧反向出样近似。"""
        return self.head in ("diff", "film", "field")

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
        t = self._task_vector(E, ctx)                   # Eq.18：Ẽ′（无任务支路时 None）
        if self.head == "concat":
            L = torch.cat([l_a, l_b], dim=1)            # Eq.16 ← L163
            h = self.cls_fc(L)                          # Eq.17
            if t is not None:
                h = torch.cat([h, t], dim=1)            # Eq.19
            logits = self.pred_fc(h).squeeze(1)         # Eq.20-21 ← L168
        elif self.head == "diff":
            logits = self._score_diff(l_a, l_b, t)
        elif self.head == "field":
            logits = self._score_diff(l_a, l_b, None)
            if t is not None:
                logits = logits + ((cls_a - cls_b) * self.field_w(t)).sum(dim=1)
        else:
            if t is not None:
                gamma, beta = self.film(t).chunk(2, dim=1)
                l_a = (1.0 + gamma) * l_a + beta
                l_b = (1.0 + gamma) * l_b + beta
            logits = self._score_diff(l_a, l_b, None)
        return torch.sigmoid(logits)                    # ← L169

    def score(self, cls: torch.Tensor, E: Optional[torch.Tensor] = None, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """S3-Y：反对称头的单配置分数 s(c, E)，满足 forward(a, b) = σ(score(a) − score(b))。concat 头没有这种分数。"""
        if self.head not in ("diff", "field", "film"):
            raise ValueError(f"head={self.head!r} 不反对称，没有单配置分数")
        self._check_ctx(ctx, cls.shape[0])
        l = self.cls_mlp(cls)
        t = self._task_vector(E, ctx)
        if self.head == "diff":
            s = self.score_mlp(l if t is None else torch.cat([l, t], dim=1))
        elif self.head == "field":
            s = self.score_mlp(l)
            if t is not None:
                s = s + (cls * self.field_w(t)).sum(dim=1, keepdim=True)
        else:
            if t is not None:
                gamma, beta = self.film(t).chunk(2, dim=1)
                l = (1.0 + gamma) * l + beta
            s = self.score_mlp(l)
        return s.squeeze(1)

    def _task_vector(self, E: Optional[torch.Tensor], ctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.task_branch == "E+ctx":
            if E is None:
                raise ValueError("task_branch='E+ctx' 必须传 E")
            task_in = E if self.ctx_dim == 0 else torch.cat([E, ctx], dim=1)
            return self.set_fc(task_in)
        if self.task_branch == "ctx":
            return self.set_fc(ctx)
        return None

    def _score_diff(self, l_a: torch.Tensor, l_b: torch.Tensor,
                    t: Optional[torch.Tensor]) -> torch.Tensor:
        """反对称打分差：s(l_a, t) − s(l_b, t)。"""
        if t is None:
            s_a, s_b = self.score_mlp(l_a), self.score_mlp(l_b)
        else:
            s_a = self.score_mlp(torch.cat([l_a, t], dim=1))
            s_b = self.score_mlp(torch.cat([l_b, t], dim=1))
        return (s_a - s_b).squeeze(1)
