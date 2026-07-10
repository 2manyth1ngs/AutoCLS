"""CLS 向量编码（方案 A 核心，TECHNICAL_DESIGN.md 第 5/6 节）。

编码规则（第 5 节表格，宽度与顺序由 space.yaml 唯一决定）：
- numeric（增强/嵌入比率）：线性归一化到 [0,1]，1 维
- log_numeric（temperature，跨 5 个数量级）：log10 后归一化到 [0,1]，1 维
- categorical：one-hot，len(choices) 维；无效档 None → 全零
- bool：0/1，1 维；无效档 None → 0

decode 为近似逆（仅用于调试与可解释性分析）：全部字段均为离散档位，
数值字段按最近档位吸附，因此对合法编码 decode(encode(cls)) == cls。
"""
from __future__ import annotations

import math

import numpy as np

from .cls_space import CLS, CLSSpace


class CLSEncoder:
    def __init__(self, space: CLSSpace):
        self._space = space
        self._slices: dict[str, slice] = {}
        start = 0
        for name in space.field_names:
            spec = space.field_spec(name)
            width = len(spec["choices"]) if spec["type"] == "categorical" else 1
            self._slices[name] = slice(start, start + width)
            start += width
        self._dim = start

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def field_slices(self) -> dict[str, slice]:
        return dict(self._slices)

    def encode(self, cls_cfg: CLS) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for name in self._space.field_names:
            v = getattr(cls_cfg, name)
            if v is None:  # 无效档 → 全零
                continue
            spec = self._space.field_spec(name)
            sl, kind, ch = self._slices[name], spec["type"], spec["choices"]
            if kind == "numeric":
                lo, hi = min(ch), max(ch)
                vec[sl.start] = (v - lo) / (hi - lo)
            elif kind == "log_numeric":
                lo, hi = math.log10(min(ch)), math.log10(max(ch))
                vec[sl.start] = (math.log10(v) - lo) / (hi - lo)
            elif kind == "categorical":
                vec[sl.start + ch.index(v)] = 1.0
            elif kind == "bool":
                vec[sl.start] = 1.0 if v else 0.0
            else:
                raise ValueError(f"未知字段类型: {kind}")
        return vec

    def decode(self, vec: np.ndarray) -> CLS:
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (self._dim,):
            raise ValueError(f"期望形状 ({self._dim},)，得到 {vec.shape}")
        values: dict = {}
        for name in self._space.field_names:
            spec = self._space.field_spec(name)
            seg, kind, ch = vec[self._slices[name]], spec["type"], spec["choices"]
            if kind == "numeric":
                lo, hi = min(ch), max(ch)
                x = lo + float(seg[0]) * (hi - lo)
                values[name] = min(ch, key=lambda c: abs(c - x))
            elif kind == "log_numeric":
                lo, hi = math.log10(min(ch)), math.log10(max(ch))
                x = lo + float(seg[0]) * (hi - lo)
                values[name] = min(ch, key=lambda c: abs(math.log10(c) - x))
            elif kind == "categorical":
                values[name] = ch[int(seg.argmax())] if float(seg.max()) > 0.5 else None
            else:  # bool
                values[name] = bool(seg[0] > 0.5)
        # 门控字段：gate 全关时还原为无效档（与采样/校验共用同一约束逻辑）
        values.update(self._space.inactive_overrides(values))
        return CLS(**values)
