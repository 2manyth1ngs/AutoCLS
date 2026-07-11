"""两层 Set-Transformer 任务聚合（TECHNICAL_DESIGN.md 模块 5；铁律 4）。

来源（reference/AutoCTS_plusplus，逐段标注）：
- MAB / SAB / ISAB / PMA ← NAS_Net/AHC/set_encoder/setenc_modules.py（原样搬运；
  其上游为 Juho Lee SetTransformer ICML 2019 → MetaD2A ICLR 2021）
- SetPool                ← NAS_Net/AHC/set_encoder/setenc_models.py（原样搬运）
- TaskSetEncoder         ← NAS_Net/AHC/ahc_engine.py::AHC 中 intra_setpool /
  inter_setpool 的构造参数与 set_encode() 逻辑（适配，差异见类 docstring）

论文对应：AutoCTS++（VLDBJ 2024）任务嵌入 Eq.(9)-(12)——窗口经 TS2Vec 得逐
时间步表征（Eq.9，见 ts2vec_wrapper.py），intra-set 池化把每个窗口压成单向量
（Eq.10），inter-set 池化跨窗口聚合（Eq.11），得任务嵌入 E（Eq.12）。
（逐式编号按论文任务嵌入小节顺序标注；具体实现细节以 ahc_engine.py 源码为准。）
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ============================================================================
# 以下到 TaskSetEncoder 之前原样搬运自 reference/AutoCTS_plusplus/NAS_Net/AHC/
# set_encoder/{setenc_modules,setenc_models}.py
# ============================================================================


class MAB(nn.Module):
    # ← setenc_modules.py::MAB（Multihead Attention Block）
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), -3)
        K_ = torch.cat(K.split(dim_split, -1), -3)
        V_ = torch.cat(V.split(dim_split, -1), -3)

        A = torch.softmax(Q_.matmul(K_.transpose(-2, -1)) / math.sqrt(self.dim_V), -1)
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(-3), -3), -1)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    # ← setenc_modules.py::SAB（Set Attention Block：自注意力）
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    # ← setenc_modules.py::ISAB（诱导点注意力，本项目未用，保留以对齐源码）
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(-3), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    # ← setenc_modules.py::PMA（Pooling by Multihead Attention：集合→num_seeds 个向量）
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(-3), 1, 1), X)


class SetPool(nn.Module):
    # ← setenc_models.py::SetPool（原样；mode 含 'sab' 用 SAB 编码，含 'P' 只留 PMA
    #    解码、含 'PF' 则 PMA 后接线性层）
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False, mode=None):
        super(SetPool, self).__init__()
        if 'sab' in mode:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        else:
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        if 'PF' in mode:
            self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        elif 'P' in mode:
            self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln))
        else:
            self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        x1 = self.enc(X)
        x2 = self.dec(x1)
        return x2


# ============================================================================
# 适配层
# ============================================================================


class TaskSetEncoder(nn.Module):
    """两级集合聚合：{窗口逐时间步表征} → 任务嵌入 E（Eq.10-12）。

    构造参数与前向逻辑照抄 ahc_engine.py::AHC（L109-118 的 intra/inter_setpool
    与 L173-182 的 set_encode）：
    - intra_setpool（mode='sabPF'）：对单个窗口的 seq_len 个逐时间步嵌入做集合
      池化 → 每窗口一个向量（Eq.10）
    - inter_setpool（mode='sabP'）：跨窗口聚合 → 任务嵌入 E（Eq.11-12）

    与原码差异（分类任务语义）：
    1. 原文"窗口"是预测滑窗（S=P+Q，generate_task_feature.py 采 100 个）；此处
       一个窗口 = 一个分类样本序列（HAR: T=128），集合大小 = 采样的样本数。
    2. 原 set_encode 把各任务的窗口表征 stack 成批（要求各任务窗口数相同，
       原码固定 sample_num=100）；此处逐任务过 inter_setpool——注意力池化在
       batch 维逐行独立，两者数学等价，且允许小任务样本数 < sample_num。
    3. 原码中该编码器嵌在 AHC 内与比较器联合训练；此处独立成模块，M3 起由
       T-CLSC（tclsc_engine.py）持有并联合训练。
    """

    def __init__(self, dim_input: int = 256, nz: int = 256, num_heads: int = 4):
        super().__init__()
        # ← ahc_engine.py L107-118（dim_input=256, nz=256 为原码取值）
        self.nz = nz
        self.intra_setpool = SetPool(
            dim_input=dim_input, num_outputs=1, dim_output=nz,
            dim_hidden=nz, num_heads=num_heads, mode='sabPF',
        )
        self.inter_setpool = SetPool(
            dim_input=nz, num_outputs=1, dim_output=nz,
            dim_hidden=nz, num_heads=num_heads, mode='sabP',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """单个任务：(n_windows, seq_len, dim_input) → E (nz,)。"""
        return self.set_encode([x])[0]

    def set_encode(self, X: list[torch.Tensor]) -> torch.Tensor:
        """一批任务（窗口数可不同）→ (len(X), nz)。← ahc_engine.py::AHC.set_encode。"""
        outs = []
        for x in X:
            cls_protos = self.intra_setpool(x).squeeze(1)      # (n_i, nz)  Eq.10
            v = self.inter_setpool(cls_protos.unsqueeze(0))    # (1, 1, nz) Eq.11
            outs.append(v.squeeze(1).squeeze(0))               # (nz,)      Eq.12
        return torch.stack(outs)
