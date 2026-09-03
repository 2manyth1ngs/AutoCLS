"""CLS 向量编码（方案 A 核心，TECHNICAL_DESIGN.md 第 5/6 节）。

编码规则（第 5 节表格，宽度与顺序由 space.yaml 唯一决定）：
- numeric（增强/嵌入比率）：线性归一化到 [0,1]，1 维
- log_numeric（temperature，跨 5 个数量级）：log10 后归一化到 [0,1]，1 维
- categorical：one-hot，len(choices) 维；无效档 None → 全零
- bool：0/1，1 维；无效档 None → 0

decode 为近似逆（仅用于调试与可解释性分析）：全部字段均为离散档位，
数值字段按最近档位吸附，因此对合法编码 decode(encode(cls)) == cls。

两种模式（MX-4 WP2 新增，铁律 1：档位仍只来自 space.yaml）：
- ``mode="scalar"``（默认，逐位不变）：上表口径，32 维；T-CLSC / 旧缓存全部走这里。
- ``mode="binned"``：numeric / log_numeric 也按档位 one-hot（len(choices) 维），
  categorical / bool / None 规则同上。给字段结构化代理用——ratio 字段的效应非单调
  （如 crop 0.2 与 0.5 差一个 z），1 维标量逼模型从一维里学非单调。
"""
from __future__ import annotations

import math

import numpy as np

from .cls_space import CLS, CLSSpace

MODES = ("scalar", "binned")


class CLSEncoder:
    def __init__(self, space: CLSSpace, mode: str = "scalar"):
        if mode not in MODES:
            raise ValueError(f"未知编码模式 {mode!r}（合法：{MODES}）")
        self._space = space
        self._mode = mode
        self._slices: dict[str, slice] = {}
        start = 0
        for name in space.field_names:
            spec = space.field_spec(name)
            width = self._width(spec)
            self._slices[name] = slice(start, start + width)
            start += width
        self._dim = start

    def _width(self, spec: dict) -> int:
        kind = spec["type"]
        if kind == "categorical":
            return len(spec["choices"])
        if kind in ("numeric", "log_numeric") and self._mode == "binned":
            return len(spec["choices"])
        return 1

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def field_slices(self) -> dict[str, slice]:
        return dict(self._slices)

    def encode(self, cls_cfg: CLS) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        binned = self._mode == "binned"
        for name in self._space.field_names:
            v = getattr(cls_cfg, name)
            if v is None:  # 无效档 → 全零
                continue
            spec = self._space.field_spec(name)
            sl, kind, ch = self._slices[name], spec["type"], spec["choices"]
            if kind == "numeric":
                if binned:
                    vec[sl.start + ch.index(v)] = 1.0
                else:
                    lo, hi = min(ch), max(ch)
                    vec[sl.start] = (v - lo) / (hi - lo)
            elif kind == "log_numeric":
                if binned:
                    vec[sl.start + ch.index(v)] = 1.0
                else:
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
        binned = self._mode == "binned"
        values: dict = {}
        for name in self._space.field_names:
            spec = self._space.field_spec(name)
            seg, kind, ch = vec[self._slices[name]], spec["type"], spec["choices"]
            if kind == "numeric":
                if binned:
                    values[name] = ch[int(seg.argmax())]
                else:
                    lo, hi = min(ch), max(ch)
                    x = lo + float(seg[0]) * (hi - lo)
                    values[name] = min(ch, key=lambda c: abs(c - x))
            elif kind == "log_numeric":
                if binned:
                    values[name] = ch[int(seg.argmax())]
                else:
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
