"""路线 A 源/目标/reserve 池注册表测试（铁律 5 先测后写；铁律 6 泄漏防护）。

被测：
- load_registry：解析 configs/source_pool.yaml → 角色分离、池规模、名字唯一。
- load_source_dataset：目标池/外部数据集 fail-fast（铁律 6：目标绝不走源加载路径）。
- delta_for：per-dataset δ 查表 + 回退默认（δ 唯一真源仍是 labels.yaml）。
- build_slice_strategy：按注册表档产出通用切片策略；2 类源自动 allow_repeat。
- remaining_task_count / task_count_gate：预探针后源池任务下限闸（<100 停并报回填）。
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from autocls.data.slice_strategies import RandomClassSubsetSlicer
from autocls.data.source_pool import (
    ROLE_RESERVE,
    ROLE_SOURCE,
    ROLE_TARGET,
    DatasetEntry,
    SourcePoolRegistry,
    TargetPoolLeakageError,
    build_slice_kwargs,
    build_slice_strategy,
    default_channel_range,
    delta_for,
    load_registry,
    load_source_dataset,
    remaining_task_count,
    task_count_gate,
)
from autocls.data.task_slicer import slice_pool_tasks, split_pools
from autocls.task_types import TASK_CLASSIFICATION

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "uea_cache"


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def labels_cfg():
    return yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))


# ---- 注册表结构 ----

def test_pool_sizes(registry):
    assert len(registry.source) == 13    # Ethanol 降 reserve 后源池 13 活跃
    assert len(registry.reserve) == 3
    assert len(registry.target) == 3


def test_roles_disjoint(registry):
    # 铁律 6：任一数据集名只能属于一个池（目标绝不与源/reserve 重名）
    src = registry.names(ROLE_SOURCE)
    res = registry.names(ROLE_RESERVE)
    tgt = registry.names(ROLE_TARGET)
    assert src & tgt == set()
    assert src & res == set()
    assert res & tgt == set()


def test_target_pool_membership(registry):
    assert registry.names(ROLE_TARGET) == {"UCI-HAR", "Epilepsy", "FaceDetection"}


def test_ethanol_demoted_to_reserve(registry):
    # 本会话拍板：EthanolConcentration 从源池 ③ 降为 reserve（长端非 EEG 锚释放）
    assert "EthanolConcentration" in registry.names(ROLE_RESERVE)
    assert "EthanolConcentration" not in registry.names(ROLE_SOURCE)


def test_backbone_tested_flag(registry):
    tested = {e.name for e in registry.source if e.tested}
    assert tested == {"SpokenArabicDigits", "PhonemeSpectra", "Cricket"}


# ---- 铁律 6 泄漏防护：目标/外部 fail-fast ----

def test_load_source_dataset_rejects_target(registry):
    # 目标池数据集绝不走源加载路径（防混入 TS2Vec 训练/造 label）
    with pytest.raises(TargetPoolLeakageError, match="目标池"):
        load_source_dataset("Epilepsy", registry, cache_dir=CACHE)
    with pytest.raises(TargetPoolLeakageError):
        load_source_dataset("FaceDetection", registry, cache_dir=CACHE)


def test_uci_har_is_target_and_leakage_guarded(registry):
    # UCI-HAR 既是目标又是外部：目标闸优先（铁律 6），不泄漏到源加载路径
    with pytest.raises(TargetPoolLeakageError):
        load_source_dataset("UCI-HAR", registry, cache_dir=CACHE)


def test_load_source_dataset_rejects_external_source(registry):
    # 外部（source=external）的**源**数据集：不经 uea_cache 加载（防误当 UEA 源）。
    # 现注册表唯一 external 是目标 UCI-HAR，故用合成注册表单测 external 分支。
    ext = SourcePoolRegistry(entries=(
        DatasetEntry(name="ExtSrc", role=ROLE_SOURCE, domain="X", source="external"),
    ))
    with pytest.raises(ValueError, match="external|外部"):
        load_source_dataset("ExtSrc", ext, cache_dir=CACHE)


def test_load_source_dataset_unknown_name(registry):
    with pytest.raises(KeyError):
        load_source_dataset("NoSuchDataset", registry, cache_dir=CACHE)


# ---- delta_for ----

def test_delta_for_per_dataset(labels_cfg):
    assert delta_for("SpokenArabicDigits", labels_cfg) == 0.03  # labels.yaml 实测填死
    assert delta_for("Cricket", labels_cfg) == 0.08


def test_delta_for_fallback_to_default(labels_cfg):
    # reserve 数据集不在 delta_per_dataset → 回退默认 delta（并入源池时再补）
    assert delta_for("Heartbeat", labels_cfg) == labels_cfg["delta"]


# ---- build_slice_strategy ----

def test_build_slice_strategy_range_from_registry(registry):
    strat = build_slice_strategy(registry.get("NATOPS"))
    assert isinstance(strat, RandomClassSubsetSlicer)
    assert strat.n_classes_range == (3, 6)
    assert strat.allow_repeat is False   # 6 类组合充裕


def test_build_slice_strategy_allow_repeat_for_two_class(registry):
    # 2 类源（组合数=1 < planned_tasks）→ 自动 allow_repeat，多样性转通道/样本子集
    for name in ("MotorImagery", "SelfRegulationSCP1", "SelfRegulationSCP2"):
        strat = build_slice_strategy(registry.get(name))
        assert strat.allow_repeat is True, name


# ---- 切片策略参数（channel 子集 + 采样上限，step0 与全量共用，必须一致以复用缓存）----

def test_default_channel_range():
    assert default_channel_range(3) == (2, 3)
    assert default_channel_range(4) == (2, 4)
    assert default_channel_range(24) == (12, 24)
    assert default_channel_range(64) == (32, 64)


def test_slice_config_loaded(registry):
    # slice 段是源池切片的唯一真源（seed/n_samples/min_samples）——两处复用必须同参
    assert registry.slice_seed == 0
    assert registry.n_samples > 0
    assert registry.min_samples > 0


def test_build_slice_kwargs_matches_registry(registry):
    kw = build_slice_kwargs(registry.get("NATOPS"), registry)
    assert kw["task_type"] == TASK_CLASSIFICATION
    assert kw["n_total_channels"] == 24
    assert kw["n_channels_range"] == (12, 24)
    assert kw["n_samples"] == registry.n_samples
    assert kw["min_samples"] == registry.min_samples
    assert kw["seed"] == registry.slice_seed
    assert isinstance(kw["strategy"], RandomClassSubsetSlicer)


def test_build_slice_kwargs_drives_core_slicing(registry):
    # 端到端：build_slice_kwargs 直接喂 slice_pool_tasks 产出合法任务（通道在档内）
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, size=1200).astype(np.int64)
    pool, _ = split_pools(y, holdout_ratio=0.3, seed=0)
    entry = registry.get("NATOPS")  # 6 类 24 通道
    kw = build_slice_kwargs(entry, registry)
    kw = {**kw, "min_samples": 16, "n_samples": 64}  # 合成数据放宽
    tasks = slice_pool_tasks(y, pool, pool_name="NATOPS", n_tasks=3, **kw)
    assert len(tasks) == 3
    for t in tasks:
        assert 12 <= len(t.channels) <= 24
        assert all(0 <= c < 24 for c in t.channels)


# ---- 源池任务下限闸 ----

def test_remaining_task_count_backbone_only(registry):
    # 无未实测源通过 → 只剩探针实测骨干：16+18+10=44
    assert remaining_task_count(registry, passed_untested=set()) == 44


def test_remaining_task_count_adds_passed_untested(registry):
    got = remaining_task_count(registry, passed_untested={"CharacterTrajectories"})
    assert got == 44 + 16   # CharTraj planned 16


def test_task_count_gate_below_floor_stops(registry):
    # 全部未实测失败 → 只剩 44 <100 → 闸不过，提示 reserve 回填
    gate = task_count_gate(registry, passed_untested=set(), floor=100)
    assert gate.ok is False
    assert gate.count == 44
    assert "reserve" in gate.message or "回填" in gate.message


def test_task_count_gate_above_floor_passes(registry):
    all_untested = {e.name for e in registry.source if not e.tested}
    gate = task_count_gate(registry, passed_untested=all_untested, floor=100)
    assert gate.ok is True
    assert gate.count >= 100    # 44 + 77 = 121
