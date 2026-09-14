"""CLS 作流水线 DAG + 共享 GIN 编码（WP16-TCLSC-L2，协议 results/tclsc_l2/PROTOCOL.md §2）。

AutoCTS++ 式 13–14：GIN(A, F) = H^(L)，H^(k) = MLP^(k)((1+ε_k)·H^(k−1) + A·H^(k−1))；读出取 "Hyper" 枢纽节点（论文）或均值（官方代码）。
CLS 不需要造图——它本来就是一条流水线：
- 节点 = space.yaml 的 19 个旧字段 + 1 个枢纽 `hub`；
- 边 = `order` 字段给出的增强施加链（`augmentations.ORDERS`，Table 8）→ 嵌入变换 → norm → sim_func → temperature → loss_type；
  配对构造 6 字段各 → loss_type；枢纽 ↔ 全部节点。**边随每条 CLS 的 `order` 变化**。
- 节点特征 F = 每字段一张 embedding 表按档位取行（最后一档 = None 无效）⊕ 数值字段的标量序数 × 可学向量。

打包：`GraphPacker.encode(cls)` → [levels(19) | scalars(19) | A(20×20)] 平铺成 438 维浮点向量，走既有 `ClsRecord.vec` 通路
（引擎零改动）；`GraphCLSEncoder.forward(packed)` 解包。铁律 1：档位、字段名、数值口径全部来自 space.yaml（`GraphSpec.from_space`），
拓扑只写字段名的连接关系并在构造时对照 `field_names` 校验；铁律 7：本模块不见任务概念。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..contrastive.augmentations import ORDERS
from ..space.cls_space import CLS, CLSSpace

HUB = "hub"
TOPOLOGIES = ("pipe", "full", "none")
READOUTS = ("hub", "mean")
AUG_FIELDS = tuple(ORDERS[0])                      # 6 个增强节点（顺序由 order 决定）
EMB_CHAIN = ("emb_jitter", "emb_mask", "norm")
LOSS_CHAIN = ("norm", "sim_func", "temperature", "loss_type")
PAIR_FIELDS = ("instance", "temporal", "cross_scale", "kernel", "pool", "adj_neighbor")


def pipeline_edges(order: int) -> list[tuple[str, str]]:
    """流水线边（无向，按字段名）：增强链（ORDERS[order]）→ 嵌入变换 → norm → sim → τ → loss；配对构造 → loss；hub ↔ 全部。"""
    chain = list(ORDERS[int(order)])
    edges: list[tuple[str, str]] = list(zip(chain, chain[1:]))
    edges.append((chain[-1], EMB_CHAIN[0]))
    edges += list(zip(EMB_CHAIN, EMB_CHAIN[1:]))
    edges += list(zip(LOSS_CHAIN, LOSS_CHAIN[1:]))
    edges += [(f, "loss_type") for f in PAIR_FIELDS]
    nodes = sorted({a for a, _ in edges} | {b for _, b in edges} | {"order"})
    edges += [(HUB, f) for f in nodes]
    return edges


@dataclass(frozen=True)
class GraphSpec:
    names: tuple[str, ...]            # 19 个字段（space.yaml 顺序）
    n_levels: tuple[int, ...]         # 每字段档位数（不含 None 档）
    numeric: tuple[bool, ...]         # 是否数值字段（有标量序数）
    kinds: tuple[str, ...]

    @classmethod
    def from_space(cls, space: CLSSpace) -> "GraphSpec":
        names = tuple(space.field_names)
        specs = [space.field_spec(n) for n in names]
        n_levels = tuple(len(s["choices"]) if s["type"] != "bool" else 2 for s in specs)
        numeric = tuple(s["type"] in ("numeric", "log_numeric") for s in specs)
        kinds = tuple(s["type"] for s in specs)
        need = set(AUG_FIELDS) | set(EMB_CHAIN) | set(LOSS_CHAIN) | set(PAIR_FIELDS) | {"order"}
        missing = need - set(names)
        if missing:
            raise ValueError(f"拓扑引用了 space.yaml 里不存在的字段 {sorted(missing)}")
        return cls(names=names, n_levels=n_levels, numeric=numeric, kinds=kinds)

    @property
    def n_fields(self) -> int:
        return len(self.names)

    @property
    def n_nodes(self) -> int:
        return len(self.names) + 1

    @property
    def pack_dim(self) -> int:
        return 2 * self.n_fields + self.n_nodes * self.n_nodes

    def index(self, name: str) -> int:
        return self.n_nodes - 1 if name == HUB else self.names.index(name)


def unpack(vec: np.ndarray, n_fields: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """438 维 → (levels (F,), scalars (F,), A (N,N))，N = F+1。"""
    v = np.asarray(vec, dtype=np.float32)
    n = n_fields + 1
    return v[:n_fields].astype(np.int64), v[n_fields:2 * n_fields], v[2 * n_fields:].reshape(n, n)


class GraphPacker:
    """CLS → 438 维打包向量（duck-type 于 CLSEncoder：.mode / .dim / .encode）。拓扑在打包时决定。"""

    def __init__(self, space: CLSSpace, *, topology: str = "pipe"):
        if topology not in TOPOLOGIES:
            raise ValueError(f"未知拓扑 {topology!r}（可选 {TOPOLOGIES}）")
        self._space = space
        self.spec = GraphSpec.from_space(space)
        self.topology = topology
        self.mode = "graph"
        self.dim = self.spec.pack_dim
        self._adj_cache: dict[int, np.ndarray] = {}

    def adjacency(self, order: int) -> np.ndarray:
        n = self.spec.n_nodes
        if self.topology == "none":
            return np.zeros((n, n), dtype=np.float32)
        if self.topology == "full":
            return (np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32))
        if int(order) not in self._adj_cache:
            A = np.zeros((n, n), dtype=np.float32)
            for a, b in pipeline_edges(int(order)):
                i, j = self.spec.index(a), self.spec.index(b)
                A[i, j] = A[j, i] = 1.0
            np.fill_diagonal(A, 0.0)
            self._adj_cache[int(order)] = A
        return self._adj_cache[int(order)]

    def encode(self, cls_cfg: CLS) -> np.ndarray:
        sp = self.spec
        levels = np.zeros(sp.n_fields, dtype=np.float32)
        scalars = np.zeros(sp.n_fields, dtype=np.float32)
        for i, name in enumerate(sp.names):
            v = getattr(cls_cfg, name)
            kind = sp.kinds[i]
            if v is None:
                levels[i] = float(sp.n_levels[i])                      # 无效档 = 最后一行
                continue
            ch = self._space.choices(name)
            if kind == "bool":
                levels[i] = 1.0 if v else 0.0
            else:
                levels[i] = float(ch.index(v))
            if kind == "numeric":
                lo, hi = min(ch), max(ch)
                scalars[i] = (v - lo) / (hi - lo)
            elif kind == "log_numeric":
                lo, hi = math.log10(min(ch)), math.log10(max(ch))
                scalars[i] = (math.log10(v) - lo) / (hi - lo)
        order = getattr(cls_cfg, "order")
        A = self.adjacency(0 if order is None else int(self._space.choices("order").index(order)))
        return np.concatenate([levels, scalars, A.reshape(-1)]).astype(np.float32)


class GraphCLSEncoder(nn.Module):
    """式 14 的 GIN：输入打包向量，输出 (B, d) 的 l_a。`in_features` = 打包维度（与 cls_mlp[0].in_features 同义）。"""

    def __init__(self, spec: GraphSpec, embedding_dim: int, *, n_layers: int = 2, readout: str = "hub",
                 use_scalar: bool = True):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"未知读出 {readout!r}（可选 {READOUTS}）")
        self.spec = spec
        self.d = int(embedding_dim)
        self.readout = readout
        self.use_scalar = bool(use_scalar)
        self.in_features = spec.pack_dim
        self.embeds = nn.ModuleList([nn.Embedding(n + 1, self.d) for n in spec.n_levels])   # +1 = None 档
        self.scalar_vec = nn.Parameter(torch.zeros(spec.n_fields, self.d))
        nn.init.normal_(self.scalar_vec, std=0.1)
        self.hub = nn.Parameter(torch.zeros(self.d))
        nn.init.normal_(self.hub, std=0.1)
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(int(n_layers))])
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, self.d))
                                   for _ in range(int(n_layers))])
        numeric = torch.tensor(spec.numeric, dtype=torch.float32)
        self.register_buffer("numeric_mask", numeric)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        F, N = self.spec.n_fields, self.spec.n_nodes
        B = packed.shape[0]
        levels = packed[:, :F].long()
        scalars = packed[:, F:2 * F]
        A = packed[:, 2 * F:].reshape(B, N, N)
        h = torch.stack([self.embeds[i](levels[:, i]) for i in range(F)], dim=1)          # (B, F, d)
        if self.use_scalar:
            h = h + (scalars * self.numeric_mask).unsqueeze(-1) * self.scalar_vec.unsqueeze(0)
        hub = self.hub.unsqueeze(0).unsqueeze(0).expand(B, 1, self.d)
        h = torch.cat([h, hub], dim=1)                                                       # (B, N, d)
        for eps, mlp in zip(self.eps, self.mlps):
            h = torch.relu(mlp((1.0 + eps) * h + torch.bmm(A, h)))                            # 式 14
        if self.readout == "hub":
            return h[:, N - 1]
        return h.mean(dim=1)
