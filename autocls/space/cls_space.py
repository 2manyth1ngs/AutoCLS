"""CLS 策略空间：定义、采样、合法性约束（TECHNICAL_DESIGN.md 第 4/6 节）。

configs/space.yaml 是策略空间唯一真源（铁律 1），本文件不硬编码任何取值集合；
CLS dataclass 只固定字段名与顺序（AutoCL Table 1 的结构），加载时与 yaml 互相校验。

**扩展字段（MX-4 WP6）**：AutoCL Table 1 之外、以 ``extension: true`` 标记的字段
（首个是 encoder 侧二项时间步掩码 ``input_mask``）。它们是 **opt-in** 的：

- ``field_names`` / ``sample_random`` / 默认 ``CLSEncoder`` 只看 19 个旧字段，
  随机采样的 RNG 流、``cls_key`` 旧键、32 维编码都逐位不变；
- 只有显式调用 ``all_field_names`` / ``twin`` / ``encoder_overrides`` /
  ``CLSEncoder(fields="all")`` 的新代码才看得到它们；
- dataclass 里"带默认值的字段"恰好就是扩展字段，加载 yaml 时互校（名、默认值、顺序）。

**约束（WP17-ACTS S1 泛化，results/acts/s1/PROTOCOL.md §1.3）**：两种写法——
``when_any: [gates]``（任一 gate 为真则 gated 字段有效，否则置无效档；旧写法）与
``when_equals: {field: value}``（全部相等时触发）；无效档用 ``inactive_value``（各字段同值）或
``inactive_values: {field: value}``。约束按序迭代到不动点（可链式触发，如 kernel=0 → cross_scale=false →
第一条约束再把 kernel/pool/adj 置 null）；只有一条 when_any 约束的旧 space.yaml 一趟即定，采样流 / validate /
编码逐位不变。``space_size`` 按"自由字段积 × 受约束字段规范形计数"枚举，旧空间仍 = 49 × 自由字段积。
第二条约束只在 configs/acts.yaml 里（opt-in，经 scripts/acts_common.acts_space 追加），不进 space.yaml。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import MISSING, asdict, dataclass, fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

DEFAULT_SPACE_YAML = Path(__file__).resolve().parents[2] / "configs" / "space.yaml"


def _gate_fields(c: dict) -> list[str]:
    """一条约束的触发字段（when_any 的成员 / when_equals 的键）。"""
    if "when_any" in c:
        return list(c["when_any"])
    if "when_equals" in c:
        return list(c["when_equals"])
    raise ValueError("约束须有 when_any 或 when_equals 之一")


def _same(a: Any, b: Any) -> bool:
    """取值相等且不把 0/1 与 False/True 混同（when_equals: {kernel: 0} 不得命中布尔字段）。"""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _fires(c: dict, values: dict) -> bool:
    """约束是否触发（触发 = gated 字段置无效档）。"""
    if "when_any" in c:
        return not any(values[g] for g in c["when_any"])
    return all(_same(values[f], v) for f, v in c["when_equals"].items())


def _inactive_map(c: dict) -> dict[str, Any]:
    if "inactive_values" in c:
        return {n: c["inactive_values"].get(n) for n in c["gated_fields"]}
    return {n: c.get("inactive_value") for n in c["gated_fields"]}


@dataclass(frozen=True)
class CLS:
    """一个对比学习策略 = AutoCL Table 1 的 19 个字段 + opt-in 扩展字段。取值集合见 configs/space.yaml。"""

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
    # ---- 扩展字段（MX-4 WP6；opt-in，与 space.yaml 的 extension 标记互校）----
    # encoder 侧二项时间步掩码（TS2Vec 官方 encoder 的训练期 mask，keep p=0.5；
    # autocls/encoders/dilated_cnn.py 的 input_mask 开关）。默认 "none" = 旧行为。
    input_mask: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """落盘序列化 = ``key_payload``：扩展字段取默认值时省略——legacy 配置写出的 dict 与
        历史产物逐键相同（目录 / 缓存行 / 候选表可再生成比对），孪生多出扩展字段。
        读回一律 ``CLS(**d)``（缺席的扩展字段取默认值）。"""
        return key_payload(self)


def extension_defaults() -> dict[str, Any]:
    """dataclass 里带默认值的字段 = 扩展字段 → {名: 默认值}（不硬编码名字）。"""
    return {f.name: f.default for f in dataclass_fields(CLS) if f.default is not MISSING}


def key_payload(cls_cfg: CLS) -> dict[str, Any]:
    """cls_key 的哈希载荷：asdict 去掉"取默认值的扩展字段"——旧配置的键逐字节不变，孪生得新键。"""
    payload = asdict(cls_cfg)
    for name, default in extension_defaults().items():
        if payload.get(name) == default:
            payload.pop(name)
    return payload


def twin(cls_cfg: CLS, **ext: Any) -> CLS:
    """只改扩展字段的副本（"孪生"）；试图改旧字段是错误。"""
    allowed = extension_defaults()
    bad = set(ext) - set(allowed)
    if bad:
        raise ValueError(f"twin 只能改扩展字段 {sorted(allowed)}，收到 {sorted(bad)}")
    return replace(cls_cfg, **ext)


class CLSSpace:
    """由 space.yaml 驱动的策略空间：采样、校验、规模计算。"""

    def __init__(self, config: dict):
        self._all_fields: list[dict] = list(config["fields"])
        self._fields: list[dict] = [f for f in self._all_fields if not f.get("extension")]
        self._ext_fields: list[dict] = [f for f in self._all_fields if f.get("extension")]
        self._constraints: list[dict] = list(config.get("constraints", []))
        self._by_name = {f["name"]: f for f in self._all_fields}

        names = [f["name"] for f in self._all_fields]
        expected = [f.name for f in dataclass_fields(CLS)]
        if names != expected:
            raise ValueError(f"space.yaml 字段与 CLS dataclass 不一致: {names} != {expected}")

        # 扩展字段 ≡ dataclass 带默认值的字段（名、序、默认值三者互校；default ∈ choices；须有 encoder_kwarg）
        defaults = extension_defaults()
        ext_names = [f["name"] for f in self._ext_fields]
        if ext_names != list(defaults):
            raise ValueError(f"yaml 的 extension 字段与 CLS 默认值字段不一致: {ext_names} != {list(defaults)}")
        for f in self._ext_fields:
            name = f["name"]
            if "default" not in f:
                raise ValueError(f"扩展字段 {name} 必须声明 default")
            if f["default"] not in f["choices"]:
                raise ValueError(f"扩展字段 {name} 的 default {f['default']!r} 不在 choices 内")
            if f["default"] != defaults[name]:
                raise ValueError(f"扩展字段 {name} 的 default {f['default']!r} ≠ CLS 默认值 {defaults[name]!r}")
            if "encoder_kwarg" not in f:
                raise ValueError(f"扩展字段 {name} 必须声明 encoder_kwarg（build_encoder overrides 的键）")

        for c in self._constraints:
            gates, gated = _gate_fields(c), list(c["gated_fields"])
            refs = set(gates) | set(gated)
            unknown = refs - set(self._by_name)
            if unknown:
                raise ValueError(f"约束引用了未知字段: {unknown}")
            if refs & set(ext_names):
                raise ValueError(f"约束不得引用扩展字段: {refs & set(ext_names)}")
            for f, v in (c.get("when_equals") or {}).items():
                if not any(_same(v, ch) for ch in self._by_name[f]["choices"]):
                    raise ValueError(f"约束 when_equals {f}={v!r} 不在该字段档位内")
            extra = set(c.get("inactive_values") or {}) - set(gated)
            if extra:
                raise ValueError(f"约束 inactive_values 引用了非 gated 字段: {extra}")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_SPACE_YAML) -> "CLSSpace":
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    @property
    def field_names(self) -> list[str]:
        """19 个旧字段（AutoCL Table 1）。采样 / evolution / 提名 / 默认编码器都只看这些。"""
        return [f["name"] for f in self._fields]

    @property
    def extension_field_names(self) -> list[str]:
        return [f["name"] for f in self._ext_fields]

    @property
    def all_field_names(self) -> list[str]:
        """旧字段 + 扩展字段（= CLS dataclass 顺序）。"""
        return [f["name"] for f in self._all_fields]

    def field_spec(self, name: str) -> dict:
        return self._by_name[name]

    def choices(self, name: str) -> list:
        return list(self._by_name[name]["choices"])

    def inactive_overrides(self, values: dict) -> dict[str, Any]:
        """给定各字段取值，返回应置为无效档的 {gated_field: inactive_value}。

        按序迭代到不动点：某条约束触发后改写的取值可能让另一条触发（acts.yaml 的
        kernel=0 → cross_scale=false → 第一条约束）。已施加的改写一律保留在返回值里
        （即便该约束在不动点处不再触发），调用方 ``values.update(...)`` 后即得规范形；
        只有一条约束时一趟即定，返回值与泛化前逐位相同。
        """
        cur = dict(values)
        out: dict[str, Any] = {}
        for _ in range(len(self._constraints) + 1):
            changed = False
            for c in self._constraints:
                if not _fires(c, cur):
                    continue
                for name, val in _inactive_map(c).items():
                    out[name] = val
                    if not _same(cur[name], val):
                        cur[name] = val
                        changed = True
            if not changed:
                break
        return out

    def sample_random(self, rng: Optional[np.random.Generator] = None) -> CLS:
        """只对旧字段抽样（RNG 流与加扩展字段前逐位相同）；扩展字段取默认值。"""
        rng = np.random.default_rng() if rng is None else rng
        values = {}
        for f in self._fields:
            ch = f["choices"]
            # 按下标取值以保留 yaml 原生 Python 类型（float/int/str/bool）
            values[f["name"]] = ch[int(rng.integers(len(ch)))]
        values.update(self.inactive_overrides(values))
        return CLS(**values)

    def validate(self, cls_cfg: CLS) -> bool:
        values = {name: getattr(cls_cfg, name) for name in self.all_field_names}
        inactive = self.inactive_overrides(values)
        for f in self._all_fields:
            name = f["name"]
            v = values[name]
            if name in inactive:
                if v != inactive[name]:
                    return False
            elif v not in f["choices"]:
                return False
        return True

    def encoder_overrides(self, cls_cfg: CLS) -> dict[str, Any]:
        """扩展字段取非默认值时 → build_encoder(overrides=...) 的键值；旧配置 → {}。"""
        out: dict[str, Any] = {}
        for f in self._ext_fields:
            v = getattr(cls_cfg, f["name"])
            if v != f["default"]:
                out[f["encoder_kwarg"]] = v
        return out

    def space_size(self, include_extensions: bool = False) -> int:
        """合法配置总数 = 自由字段积 × 受约束字段的规范形个数（枚举原始写法经 inactive_overrides
        去重）。旧 space.yaml 的 (temporal, cross_scale, kernel, pool, adj_neighbor) 64 种写法归为 49 种，
        与泛化前的门控公式同值。默认不计扩展字段。"""
        constrained: list[str] = []
        for c in self._constraints:
            for n in (*_gate_fields(c), *c["gated_fields"]):
                if n not in constrained:
                    constrained.append(n)
        size = math.prod(
            len(f["choices"]) for f in self._fields if f["name"] not in constrained
        )
        if constrained:
            forms: set[tuple] = set()
            for combo in itertools.product(*[self._by_name[n]["choices"] for n in constrained]):
                values = dict(zip(constrained, combo))
                values.update(self.inactive_overrides(values))
                forms.add(tuple((values[n], isinstance(values[n], bool)) for n in constrained))
            size *= len(forms)
        if include_extensions:
            size *= math.prod(len(f["choices"]) for f in self._ext_fields)
        return size
