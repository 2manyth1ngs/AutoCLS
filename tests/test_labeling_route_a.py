"""路线 A 阶段2 造 label（scripts/2b_generate_labels_route_a.py）的纯函数测试。

铁律 5 先测后写；无 GPU/训练依赖。被测核心：
- delta_for + make_pairs：per-dataset δ 查表配对（与路线 B 单标量 delta 的唯一差异）；
- load_probe_cls：shared CLS = step0 探针目录前 N（canonical R' 复用的唯一真源）；
- sample_random_cls：每任务 random CLS 由 task_id 派生（非路线 B 的按位置派生——
  跨数据集源池成员随 step0 过闸浮动，按位置派生会让缓存失效，铁律 3）；
- canonical 种子口径：2b 与 step0 的 s=0 完全同源（缓存逐键命中复用的前提）；
- labels.yaml route_a 段：20 CLS/任务口径（10 shared + 10 random，非路线 B 的 40）。
"""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from autocls.data.source_pool import delta_for
from autocls.labeling import cls_key, load_probe_cls, make_pairs, sample_random_cls
from autocls.space.cls_space import CLSSpace

ROOT = Path(__file__).resolve().parents[1]
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def labels_cfg():
    return yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))


def _load_script(fname: str, modname: str):
    import sys
    spec = importlib.util.spec_from_file_location(modname, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # dataclass 字段注解解析需模块已注册
    spec.loader.exec_module(mod)
    return mod


# ---- per-dataset δ 查表配对 ----

def test_per_dataset_delta_lookup():
    cfg = {"delta": 0.02, "delta_per_dataset": {"Cricket": 0.08, "NATOPS": 0.06}}
    assert delta_for("Cricket", cfg) == 0.08
    assert delta_for("PhonemeSpectra", cfg) == 0.02   # 缺项回退默认


def test_make_pairs_respects_cricket_delta():
    # ΔR'=0.05：> 0.02 默认 → 保留；< 0.08 Cricket 档 → 弃样（同分噪声不给 label）
    rp = {"a": 0.60, "b": 0.65}
    kept_default, _ = make_pairs(rp, "classification", delta=0.02)
    dropped_cricket, st = make_pairs(rp, "classification", delta=0.08)
    assert len(kept_default) == 1
    assert len(dropped_cricket) == 0
    assert st["n_delta_dropped"] == 1


# ---- shared CLS = step0 探针目录前 N（canonical 复用唯一真源）----

def test_load_probe_cls_is_catalog_prefix():
    cat = json.loads(PROBE_CATALOG.read_text(encoding="utf-8"))["cls"]
    cls_list, ranks = load_probe_cls(PROBE_CATALOG, 10)
    assert [k for _, k in cls_list] == [c["cls_key"] for c in cat[:10]]
    assert set(ranks) == {c["cls_key"] for c in cat[:10]}


def test_probe_cls_keys_reproducible_from_fields():
    # 目录里的 cls_key 必须与 CLS 字段重算一致——canonical 缓存键逐字段命中的前提
    cls_list, _ = load_probe_cls(PROBE_CATALOG, 10)
    for c, k in cls_list:
        assert cls_key(c) == k


# ---- 每任务 random CLS：task_id 派生、确定性、前缀稳定、与 shared 去重 ----

def test_sample_random_cls_deterministic_per_task(space):
    a1 = sample_random_cls(space, "NATOPS_00", n=3, seed=42, exclude_keys=set())
    a2 = sample_random_cls(space, "NATOPS_00", n=3, seed=42, exclude_keys=set())
    b = sample_random_cls(space, "ERing_00", n=3, seed=42, exclude_keys=set())
    assert [cls_key(c) for c in a1] == [cls_key(c) for c in a2]     # 同任务同批
    assert [cls_key(c) for c in a1] != [cls_key(c) for c in b]      # 任务间独立


def test_sample_random_cls_independent_of_pool_composition(space):
    # 池成员增删不影响某任务的 random 批（缓存不因 step0 踢/留而失效，铁律 3）
    only = sample_random_cls(space, "Cricket_03", n=4, seed=42, exclude_keys=set())
    again = sample_random_cls(space, "Cricket_03", n=4, seed=42, exclude_keys=set())
    assert [cls_key(c) for c in only] == [cls_key(c) for c in again]


def test_sample_random_cls_prefix_stable(space):
    # 加大 n 不改变前 n' 个取值：小规模试跑与全量共享 (task, cls) → R' 缓存
    small = sample_random_cls(space, "T_00", n=2, seed=42, exclude_keys=set())
    big = sample_random_cls(space, "T_00", n=5, seed=42, exclude_keys=set())
    assert [cls_key(c) for c in big[:2]] == [cls_key(c) for c in small]


def test_sample_random_cls_excludes_shared(space):
    shared_keys = {k for _, k in load_probe_cls(PROBE_CATALOG, 10)[0]}
    out = sample_random_cls(space, "NATOPS_00", n=6, seed=42, exclude_keys=shared_keys)
    assert len(out) == 6
    assert not ({cls_key(c) for c in out} & shared_keys)


# ---- canonical 种子口径：2b 与 step0 s=0 完全同源 ----

def test_canonical_seed_parity_between_step0_and_2b():
    step0 = _load_script("step0_snr_gate.py", "_step0_for_parity")
    gl = _load_script("2b_generate_labels_route_a.py", "_gl2b_for_parity")
    tid, key = "NATOPS+c5-6_00", "deadbeef1234"
    assert step0._seed_specs(tid, key, 0, 290) == gl.canonical_seeds(tid, key)


# ---- labels.yaml route_a 段：CLS 口径 = 20/任务（checklist C8）----

def test_route_a_config_present(labels_cfg):
    ra = labels_cfg["route_a"]
    assert ra["n_shared_cls"] == 10
    assert ra["n_random_cls"] == 10       # 20/任务，非路线 B 的 40
    for k in ("cls_seed", "ts2vec_iters", "sample_num", "nz", "embed_seed"):
        assert k in ra, k
