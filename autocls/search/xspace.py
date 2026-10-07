"""WP18：扩展策略空间 = AutoCL 19 字段（space.yaml，经 acts_space 规范化 + 剪枝）⊕ 新轴（configs/wp18.yaml）。

新轴（如 view：对比在哪个域上做；mult：训练量倍数；readout：表示读出）不进 CLS dataclass / space.yaml，
旧缓存键、旧采样流、旧 116 维编码一律不动；扩展配置 = (CLS, ext)，键 = cls_key(CLS) [+ "|轴=值"（非默认轴）]。
编码 = 旧 binned 116 维 ⊕ 每个新轴的 one-hot。全空间搜索的采样 / 交叉 / 变异覆盖旧字段与新轴。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import yaml

from ..labeling import cls_key
from ..space.cls_encoder import CLSEncoder
from ..space.cls_space import CLS

WP18_YAML = Path(__file__).resolve().parents[2] / "configs" / "wp18.yaml"


def load_axes(path: str | Path = WP18_YAML) -> dict[str, dict]:
    """{轴名: {"choices": [...], "default": v}}（顺序 = 编码顺序）。"""
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    out = {}
    for a in cfg["axes"]:
        ch = list(a["choices"])
        d = a.get("default", ch[0])
        if d not in ch:
            raise ValueError(f"轴 {a['name']} 默认值 {d!r} 不在 {ch}")
        out[a["name"]] = {"choices": ch, "default": d}
    return out


@dataclass(frozen=True)
class XCLS:
    cls: CLS
    ext: tuple = field(default=())          # ((轴名, 值), ...)，按轴顺序、含全部轴

    def get(self, name: str):
        return dict(self.ext)[name]

    def to_dict(self) -> dict:
        return {"cls": self.cls.to_dict(), "ext": dict(self.ext)}


class XSpace:
    def __init__(self, space, cfg: Mapping, axes: Optional[Mapping[str, Mapping]] = None,
                 is_pruned: Optional[Callable[[CLS, Mapping], bool]] = None, fixed: Optional[Mapping[str, object]] = None):
        """fixed：{轴: 值}——采样 / 变异只取该值（如“旧空间”臂把新轴全钉在默认档），编码维度与全空间相同。"""
        self.space = space
        self.fixed = dict(fixed or {})
        self.cfg = cfg
        self.axes = dict(load_axes() if axes is None else axes)
        self.enc = CLSEncoder(space, mode="binned")
        self._is_pruned = is_pruned or (lambda c, cfg: False)
        self.legacy_names = list(space.field_names)
        self.axis_names = list(self.axes)
        for n, v in self.fixed.items():
            if n not in self.axes or v not in self.axes[n]["choices"]:
                raise ValueError(f"fixed 轴 {n}={v!r} 非法")
        self.dim = self.enc.dim + sum(len(a["choices"]) for a in self.axes.values())

    # ---------------------------------------------------------------- 构造 / 键 / 编码
    def make(self, cls: CLS, **ext) -> XCLS:
        bad = set(ext) - set(self.axes)
        if bad:
            raise ValueError(f"未知轴 {sorted(bad)}")
        vals = []
        for n, a in self.axes.items():
            v = ext.get(n, a["default"])
            if v not in a["choices"]:
                raise ValueError(f"轴 {n} 取值 {v!r} 不在 {a['choices']}")
            vals.append((n, v))
        return XCLS(cls, tuple(vals))

    def key(self, x: XCLS) -> str:
        k = cls_key(x.cls)
        extra = [f"{n}={v}" for n, v in x.ext if v != self.axes[n]["default"]]
        return k if not extra else k + "|" + ",".join(extra)

    def encode(self, x: XCLS) -> np.ndarray:
        parts = [self.enc.encode(x.cls)]
        d = dict(x.ext)
        for n, a in self.axes.items():
            oh = np.zeros(len(a["choices"]), np.float32)
            oh[a["choices"].index(d[n])] = 1.0
            parts.append(oh)
        return np.concatenate(parts).astype(np.float32)

    def encode_many(self, xs: Sequence[XCLS]) -> np.ndarray:
        return np.stack([self.encode(x) for x in xs]).astype(np.float32)

    def from_dict(self, d: Mapping) -> XCLS:
        return self.make(CLS(**d["cls"]), **dict(d.get("ext") or {}))

    # ---------------------------------------------------------------- 合法性 / 采样
    def _choices(self, n: str) -> list:
        return [self.fixed[n]] if n in self.fixed else list(self.axes[n]["choices"])

    def accept(self, x: XCLS) -> bool:
        if any(x.get(n) != v for n, v in self.fixed.items()):
            return False
        return bool(self.space.validate(x.cls)) and not self._is_pruned(x.cls, self.cfg)

    def sample(self, rng: np.random.Generator) -> XCLS:
        for _ in range(100_000):
            c = self.space.sample_random(rng)
            ext = {n: self._choices(n)[int(rng.integers(len(self._choices(n))))] for n in self.axes}
            x = self.make(c, **ext)
            if self.accept(x):
                return x
        raise RuntimeError("采样 10 万次无合法配置")

    def sample_unique(self, rng: np.random.Generator, n: int, taken: Optional[set] = None) -> list[XCLS]:
        taken = set() if taken is None else taken
        out = []
        guard = 0
        while len(out) < n:
            guard += 1
            if guard > 50 * n + 1000:
                raise RuntimeError("去重采样凑不齐")
            x = self.sample(rng)
            k = self.key(x)
            if k in taken:
                continue
            taken.add(k)
            out.append(x)
        return out

    # ---------------------------------------------------------------- 进化算子（旧字段 + 新轴逐字段同等对待）
    def _values(self, x: XCLS) -> dict:
        v = {n: getattr(x.cls, n) for n in self.legacy_names}
        v.update(dict(x.ext))
        return v

    def _build(self, v: Mapping) -> XCLS:
        return self.make(CLS(**{n: v[n] for n in self.legacy_names}), **{n: v[n] for n in self.axis_names})

    def crossover(self, a: XCLS, b: XCLS, rng: np.random.Generator) -> XCLS:
        va, vb = self._values(a), self._values(b)
        return self._build({n: (va[n] if rng.random() < 0.5 else vb[n]) for n in va})

    def mutate(self, a: XCLS, rng: np.random.Generator) -> XCLS:
        v = self._values(a)
        names = self.legacy_names + self.axis_names
        n = names[int(rng.integers(len(names)))]
        ch = self.space.choices(n) if n in self.legacy_names else self._choices(n)
        v[n] = ch[int(rng.integers(len(ch)))]
        return self._build(v)

    def offspring(self, pop: Sequence[XCLS], rng: np.random.Generator, p_cross: float, p_mut: float) -> Optional[XCLS]:
        n = len(pop)
        for _ in range(100):
            if rng.random() < p_cross:
                child = self.crossover(pop[int(rng.integers(n))], pop[int(rng.integers(n))], rng)
            elif rng.random() < p_mut:
                child = self.mutate(pop[int(rng.integers(n))], rng)
            else:
                child = pop[int(rng.integers(n))]
            if self.accept(child):
                return child
        return None

    def evolve(self, fitness: Callable[[np.ndarray], np.ndarray], init: Sequence[XCLS], rng: np.random.Generator,
               kp: int, gens: int, p_cross: float = 0.8, p_mut: float = 0.2) -> list[XCLS]:
        """按分数的精英保留进化（= 111.score_evolve 语义，扩展到新轴）：种群 kp，每代 kp 个后代，并入后按适应度截断。"""
        def rank(cs):
            uniq, seen = [], set()
            for c in cs:
                k = self.key(c)
                if k not in seen:
                    seen.add(k); uniq.append(c)
            keys = [self.key(c) for c in uniq]
            f = np.asarray(fitness(self.encode_many(uniq)), float)
            return [uniq[i] for i in sorted(range(len(uniq)), key=lambda i: (-f[i], keys[i]))]

        pop = rank(list(init))[:kp]
        for _ in range(int(gens)):
            seen = {self.key(c) for c in pop}
            kids = []
            for _k in range(kp):
                ch = self.offspring(pop, rng, p_cross, p_mut)
                if ch is None or self.key(ch) in seen:
                    continue
                seen.add(self.key(ch)); kids.append(ch)
            pop = rank(pop + kids)[:kp]
        return pop
