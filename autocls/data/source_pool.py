"""路线 A 源/目标/reserve 池注册表（configs/source_pool.yaml 的可测加载层）。

职责边界：
- 本模块只管**池结构**——哪些数据集、什么角色、切多少任务、裁多长、类别子集档；
  δ 不在此定义（configs/labels.yaml: delta_per_dataset 是 δ 唯一真源，防 split-brain），
  仅提供 `delta_for` 查表助手。
- **铁律 6 泄漏防护落点**：`load_source_dataset` 对目标池数据集 fail-fast——目标池
  样本绝不走源加载路径（进而绝不喂给 fit_shared_ts2vec / 造 label）。TS2Vec 只在
  源池 fit 的最终保证仍在 task_embed 层，本模块是加载入口的第一道闸。
- 通用切片（路线 A 不用 HAR 动/静分层）：`build_slice_strategy` 按注册表档产出
  RandomClassSubsetSlicer；2 类源（类别组合<planned_tasks）自动 allow_repeat，
  多样性转由核心的通道/样本子集承担。
- 源池任务下限闸（本会话新增）：预探针踢除后 `task_count_gate` 检查剩余源任务
  ≥100，不足则停并提示从 reserve 回填（Ethanol/Heartbeat/FingerMovements）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb
from pathlib import Path

import numpy as np
import yaml

from ..task_types import TASK_CLASSIFICATION
from .slice_strategies import RandomClassSubsetSlicer
from .task_slicer import SlicedTask, slice_pool_tasks
from .uea import DEFAULT_CACHE, load_uea_merged

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs" / "source_pool.yaml"

ROLE_SOURCE = "source"
ROLE_RESERVE = "reserve"
ROLE_TARGET = "target"

SOURCE_EXTERNAL = "external"
SOURCE_NPZ = "npz"        # 换池调研的候选：data/candidates/<name>.npz（canonical 已开窗+z-score）

# yaml 顶层键 → 角色
_POOL_KEYS = {"source_pool": ROLE_SOURCE, "reserve": ROLE_RESERVE, "target_pool": ROLE_TARGET}


class TargetPoolLeakageError(RuntimeError):
    """对目标池数据集调用源加载路径——铁律 6 泄漏防护触发。"""


@dataclass(frozen=True)
class DatasetEntry:
    """一个数据集在池中的完整登记项（不含 δ，δ 查 labels.yaml）。"""

    name: str
    role: str
    domain: str
    source: str = "uea"                       # uea | external
    n_channels: int | None = None
    n_classes: int | None = None
    l_enc: int | None = None
    crop_len: int | None = None
    n_classes_range: tuple[int, int] | None = None
    planned_tasks: int = 0
    tested: bool = False

    @property
    def n_class_combos(self) -> int:
        """当前 n_classes_range 档下不重复类别子集总数（判 allow_repeat 用）。"""
        if self.n_classes_range is None or self.n_classes is None:
            return 0
        lo, hi = self.n_classes_range
        return sum(comb(self.n_classes, k) for k in range(lo, min(hi, self.n_classes) + 1))


@dataclass(frozen=True)
class SourcePoolRegistry:
    entries: tuple[DatasetEntry, ...]
    encode_length_floor: int = 50
    slice_seed: int = 0        # 源池切片种子（唯一真源；step0 与全量共用）
    n_samples: int = 1024      # 每任务采样上限
    min_samples: int = 32      # 任务样本数下限

    def get(self, name: str) -> DatasetEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(f"注册表无数据集 {name!r}（configs/source_pool.yaml）")

    def by_role(self, role: str) -> list[DatasetEntry]:
        return [e for e in self.entries if e.role == role]

    def names(self, role: str | None = None) -> set[str]:
        return {e.name for e in self.entries if role is None or e.role == role}

    @property
    def source(self) -> list[DatasetEntry]:
        return self.by_role(ROLE_SOURCE)

    @property
    def reserve(self) -> list[DatasetEntry]:
        return self.by_role(ROLE_RESERVE)

    @property
    def target(self) -> list[DatasetEntry]:
        return self.by_role(ROLE_TARGET)


def _parse_entry(role: str, d: dict) -> DatasetEntry:
    rng = d.get("n_classes_range")
    return DatasetEntry(
        name=d["name"], role=role, domain=d["domain"],
        source=d.get("source", "uea"),
        n_channels=d.get("n_channels"), n_classes=d.get("n_classes"),
        l_enc=d.get("l_enc"), crop_len=d.get("crop_len"),
        n_classes_range=tuple(rng) if rng is not None else None,
        planned_tasks=d.get("planned_tasks", 0), tested=d.get("tested", False),
    )


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> SourcePoolRegistry:
    """解析 source_pool.yaml；校验名字全局唯一（角色互斥，铁律 6 的结构前提）。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries: list[DatasetEntry] = []
    for key, role in _POOL_KEYS.items():
        for d in cfg.get(key, []) or []:
            entries.append(_parse_entry(role, d))
    seen: dict[str, str] = {}
    for e in entries:
        if e.name in seen:
            raise ValueError(
                f"数据集 {e.name!r} 同时出现在 {seen[e.name]} 与 {e.role} 池"
                "——角色必须互斥（铁律 6：目标绝不与源/reserve 重名）"
            )
        seen[e.name] = e.role
    sl = cfg.get("slice", {}) or {}
    return SourcePoolRegistry(
        entries=tuple(entries),
        encode_length_floor=cfg.get("encode_length_floor", 50),
        slice_seed=sl.get("seed", 0),
        n_samples=sl.get("n_samples", 1024),
        min_samples=sl.get("min_samples", 32),
    )


def load_source_dataset(
    name: str,
    registry: SourcePoolRegistry | None = None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    crop_len: int | str | None = "auto",
    candidate_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """加载一个**源/reserve**数据集（TRAIN+TEST 合并 + 注册表 crop）。

    铁律 6 防护：目标池数据集 → TargetPoolLeakageError（绝不走源加载路径）。
    外部数据集（source=external，如 UCI-HAR）→ ValueError（不经 uea_cache）。
    候选数据集（source=npz，换池调研）→ data/candidates/<name>.npz，已开窗+z-score。
    crop_len="auto" 用注册表值；传具体值/None 显式覆盖。

    候选走同一入口而非另开一条加载路径：baseline_probe 的口径一致性靠"同一个
    load_source_dataset + 同一个 slice_source_tasks"保证，换池调研的全部结论都
    建立在"新数字与 UEA 那 16 个可比"之上。
    """
    registry = registry or load_registry()
    entry = registry.get(name)
    if entry.role == ROLE_TARGET:
        raise TargetPoolLeakageError(
            f"{name!r} 是目标池数据集，绝不走源加载路径（铁律 6）——"
            "目标池样本不得参与 TS2Vec 训练/造 label"
        )
    if entry.source == SOURCE_NPZ:
        from .candidates import CANDIDATE_ROOT, load_candidate

        X, y = load_candidate(name, root=candidate_root or CANDIDATE_ROOT)
        eff = entry.crop_len if crop_len == "auto" else crop_len
        return (X[:, :eff] if eff else X), y
    if entry.source == SOURCE_EXTERNAL:
        raise ValueError(
            f"{name!r} 是 external 外部数据集（非 uea_cache），"
            "不由 load_source_dataset 加载"
        )
    eff_crop = entry.crop_len if crop_len == "auto" else crop_len
    return load_uea_merged(name, cache_dir=cache_dir, crop_len=eff_crop)


def delta_for(name: str, labels_cfg: dict) -> float:
    """per-dataset δ 查表（探针定案）；缺项回退 labels_cfg['delta'] 默认值。"""
    return labels_cfg.get("delta_per_dataset", {}).get(name, labels_cfg["delta"])


def build_slice_strategy(entry: DatasetEntry) -> RandomClassSubsetSlicer:
    """按注册表档产出通用切片策略（路线 A 通用"随机类别子集+通道子集"）。

    类别组合数 < planned_tasks（如 2 类源仅 1 个 (2,2) 组合却要多任务）→ allow_repeat：
    先出满不重复子集、耗尽后重复，剩余多样性由核心的通道/样本子集承担。
    """
    if entry.n_classes_range is None:
        raise ValueError(f"{entry.name!r} 未配 n_classes_range，无法建切片策略")
    allow_repeat = entry.n_class_combos < entry.planned_tasks
    return RandomClassSubsetSlicer(
        n_classes_range=entry.n_classes_range, allow_repeat=allow_repeat,
    )


def default_channel_range(n_channels: int) -> tuple[int, int]:
    """通道子集档：约半数到全部（(⌈C/2⌉, C)）。C=3→(2,3)、C=64→(32,64)。

    通道全取会让 2 类源的多任务退化为近重复；子集给它们主要多样性来源。通道在每
    任务内固定（跨 seed 不变），故不混淆 R'/SNR 的种子噪声测量。
    """
    return (max(1, (n_channels + 1) // 2), n_channels)


def build_slice_kwargs(entry: DatasetEntry, registry: SourcePoolRegistry) -> dict:
    """组装 slice_pool_tasks 的关键字参数（除 y/pool_indices/pool_name/n_tasks）。

    切片参数全部取自注册表（唯一真源）——step0 与全量造 label 调同一函数、同一参数，
    保证任务定义逐字段一致，canonical R' 缓存跨两处复用不失效（铁律 3）。
    """
    if entry.n_channels is None:
        raise ValueError(f"{entry.name!r} 未配 n_channels，无法建切片参数")
    return {
        "task_type": TASK_CLASSIFICATION,
        "strategy": build_slice_strategy(entry),
        "n_channels_range": default_channel_range(entry.n_channels),
        "n_total_channels": entry.n_channels,
        "n_samples": registry.n_samples,
        "min_samples": registry.min_samples,
        "seed": registry.slice_seed,
    }


def pool_name_for(entry: DatasetEntry, n_classes_range: tuple[int, int]) -> str:
    """任务 id 前缀：注册表默认档 = 数据集名；覆盖档（天花板重探/恢复档）打 +c<lo>-<hi> 标。

    task_id 只由 pool_name+序号构成（task_slicer），类别档变则任务定义变——不打标
    会与默认档任务同 id，canonical R' 缓存键 (task_id, cls_key, iters) 互撞（铁律 3）。
    step0 重探轮与阶段2 恢复档共用此约定，重探 R' 因此可被阶段2 逐键复用。
    """
    rng = tuple(n_classes_range)
    if rng == entry.n_classes_range:
        return entry.name
    return f"{entry.name}+c{rng[0]}-{rng[1]}"


def dataset_of_task_id(task_id: str) -> str:
    """task_id → 所属数据集名（剥掉 _NN 序号与 +c 档标）。"""
    return task_id.rsplit("_", 1)[0].split("+", 1)[0]


def slice_source_tasks(
    entry: DatasetEntry,
    registry: SourcePoolRegistry,
    y: np.ndarray,
    *,
    n_classes_range: tuple[int, int] | None = None,
) -> tuple[DatasetEntry, list[SlicedTask]]:
    """按注册表（或覆盖档）切全部 planned_tasks——step0 与 2b 唯一共用切片入口。

    切片参数全走 build_slice_kwargs（注册表唯一真源）；n_classes_range 覆盖注册表档
    （天花板重探/analysis.json 的恢复档），此时 pool_name 打档标（见 pool_name_for）。
    两脚本调同一函数、同一参数 → 任务定义逐字段一致，canonical R' 缓存跨两处复用
    不失效（铁律 3）。返回 (生效 entry, 全部 planned_tasks 任务)。
    """
    eff = entry if n_classes_range is None else replace(
        entry, n_classes_range=tuple(n_classes_range))
    kw = build_slice_kwargs(eff, registry)
    tasks = slice_pool_tasks(
        y, np.arange(len(y)), pool_name=pool_name_for(entry, eff.n_classes_range),
        n_tasks=entry.planned_tasks, **kw)
    return eff, tasks


def remaining_task_count(
    registry: SourcePoolRegistry, passed_untested: set[str]
) -> int:
    """预探针后源池计划任务总数：探针实测骨干（自动计入）+ 通过 SNR 闸的未实测源。"""
    total = 0
    for e in registry.source:
        if e.tested or e.name in passed_untested:
            total += e.planned_tasks
    return total


@dataclass(frozen=True)
class GateResult:
    ok: bool
    count: int
    floor: int
    message: str


def task_count_gate(
    registry: SourcePoolRegistry, passed_untested: set[str], *, floor: int = 100
) -> GateResult:
    """源池任务下限闸（本会话新增）：剩余源任务 <floor 则不过，提示回填。

    不足时不硬着头皮往下——message 指出从 reserve（Ethanol/Heartbeat/FingerMovements）
    回填或加码高 N/C 骨干（SpokenArabicDigits N/C880 可吸收更多切片）。
    """
    count = remaining_task_count(registry, passed_untested)
    ok = count >= floor
    if ok:
        msg = f"源池任务 {count} ≥ {floor}，闸通过。"
    else:
        reserve = ", ".join(e.name for e in registry.reserve)
        msg = (
            f"源池任务 {count} < {floor}：停并报告。需从 reserve（{reserve}）回填，"
            "或加码高 N/C 骨干切片预算（SpokenArabicDigits/PhonemeSpectra）。"
            "不要在任务数不足时硬着头皮往下。"
        )
    return GateResult(ok=ok, count=count, floor=floor, message=msg)
