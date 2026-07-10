"""CLS 策略空间：定义、采样、合法性约束（TECHNICAL_DESIGN.md 第 4/6 节）。

configs/space.yaml 是策略空间唯一真源（铁律 1），本文件不硬编码任何取值集合；
CLS dataclass 只固定字段名与顺序（AutoCL Table 1 的结构），加载时与 yaml 互相校验。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

DEFAULT_SPACE_YAML = Path(__file__).resolve().parents[2] / "configs" / "space.yaml"


@dataclass(frozen=True)
class CLS:
    """一个对比学习策略 = AutoCL Table 1 的 19 个字段。取值集合见 configs/space.yaml。"""

    # 数据增强
    resize: float
    rescale: float
    jitter: float
    point_mask: float
    freq_mask: float
    crop: float
    order: int
    # 嵌入变换
    emb_jitter: float
    emb_mask: float
    norm: str
    # 对比对构造（kernel/pool/adj_neighbor 在 temporal 与 cross_scale 均关闭时为无效档 None）
    instance: bool
    temporal: bool
    cross_scale: bool
    kernel: Optional[int]
    pool: Optional[str]
    adj_neighbor: Optional[bool]
    # 对比损失
    loss_type: str
    sim_func: str
    temperature: float


class CLSSpace:
    """由 space.yaml 驱动的策略空间：采样、校验、规模计算。"""

    def __init__(self, config: dict):
        self._fields: list[dict] = list(config["fields"])
        self._constraints: list[dict] = list(config.get("constraints", []))
        self._by_name = {f["name"]: f for f in self._fields}

        names = [f["name"] for f in self._fields]
        expected = [f.name for f in dataclass_fields(CLS)]
        if names != expected:
            raise ValueError(f"space.yaml 字段与 CLS dataclass 不一致: {names} != {expected}")

        seen: set[str] = set()
        for c in self._constraints:
            refs = set(c["when_any"]) | set(c["gated_fields"])
            unknown = refs - set(self._by_name)
            if unknown:
                raise ValueError(f"约束引用了未知字段: {unknown}")
            if refs & seen:
                raise ValueError("暂不支持约束间共享 gate/gated 字段（space_size 计数假设互斥）")
            seen |= refs

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SPACE_YAML) -> "CLSSpace":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    @property
    def field_names(self) -> list[str]:
        return [f["name"] for f in self._fields]

    def field_spec(self, name: str) -> dict:
        return self._by_name[name]

    def choices(self, name: str) -> list:
        return list(self._by_name[name]["choices"])

    def inactive_overrides(self, values: dict) -> dict[str, Any]:
        """给定各字段取值，返回应置为无效档的 {gated_field: inactive_value}。"""
        out: dict[str, Any] = {}
        for c in self._constraints:
            if not any(values[g] for g in c["when_any"]):
                for name in c["gated_fields"]:
                    out[name] = c.get("inactive_value")
        return out

    def sample_random(self, rng: Optional[np.random.Generator] = None) -> CLS:
        rng = np.random.default_rng() if rng is None else rng
        values = {}
        for f in self._fields:
            ch = f["choices"]
            # 按下标取值以保留 yaml 原生 Python 类型（float/int/str/bool）
            values[f["name"]] = ch[int(rng.integers(len(ch)))]
        values.update(self.inactive_overrides(values))
        return CLS(**values)

    def validate(self, cls_cfg: CLS) -> bool:
        values = {name: getattr(cls_cfg, name) for name in self.field_names}
        inactive = self.inactive_overrides(values)
        for f in self._fields:
            name = f["name"]
            v = values[name]
            if name in inactive:
                if v != inactive[name]:
                    return False
            elif v not in f["choices"]:
                return False
        return True

    def space_size(self) -> int:
        """合法配置总数。门控字段在 gate 全关时只有无效档一种取法。"""
        constrained = {n for c in self._constraints for n in (*c["when_any"], *c["gated_fields"])}
        size = math.prod(
            len(f["choices"]) for f in self._fields if f["name"] not in constrained
        )
        for c in self._constraints:
            gate_choices = [self._by_name[g]["choices"] for g in c["when_any"]]
            active = math.prod(len(self._by_name[n]["choices"]) for n in c["gated_fields"])
            size *= sum(
                active if any(combo) else 1
                for combo in itertools.product(*gate_choices)
            )
        return size
