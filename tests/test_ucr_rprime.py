"""WP7-UCR：语料 R' 生成（scripts/41_ucr_rprime.py）与调查脚本的纯函数测试——先于实现（铁律 5）。

覆盖：
- .ts 文件头解析（不解析数据体）：seriesLength / classLabel / equalLength / missing / 行计数 / 类计数；
- usable 判据：硬排除 9、不等长、缺失、L<50、每类<2、n_eval 下限；
- 整数据集任务构造：N ≤ 上限全取、N > 上限按类分层子采样（确定、覆盖全部类、索引升序）、channels=(0,)、task_id=<name>_00；
- 两道闸：每类 ≥2、n_eval ≥ 下限；
- 计划 / 缓存键：每数据集 20 条（10 shared + 10 random）、resume 键 (task_id, cls_key, iters)、四态。
零 GPU、不读真实缓存、不联网。
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

svy = importlib.import_module("results.ucr_survey.scripts.ucr_survey")
g41 = importlib.import_module("scripts.41_ucr_rprime")


# ---------------------------------------------------------------- 调查：文件头解析与判据
def test_parse_ts_header_reads_meta_and_counts(tmp_path):
    p = tmp_path / "Foo_TRAIN.ts"
    p.write_text("\n".join([
        "#comment", "@problemName Foo", "@timeStamps false", "@missing false", "@univariate true",
        "@equalLength true", "@seriesLength 5", "@classLabel true 1 2 3", "@data",
        "0.1,0.2,0.3,0.4,0.5:1", "1,2,3,4,5:2", "", "5,4,3,2,1:1",
    ]), encoding="utf-8")
    m = svy.parse_ts_header(p)
    assert m["problem"] == "Foo" and m["series_length"] == 5 and m["equal_length"] is True
    assert m["missing"] is False and m["univariate"] is True
    assert m["classes"] == ["1", "2", "3"] and m["n"] == 3 and m["counts"] == {"1": 2, "2": 1}


def _rec(**kw):
    base = {"name": "X", "n_total": 1000, "equal_length": True, "missing": False, "univariate": True,
            "length": 200, "min_class": 10, "n_classes": 3}
    base.update(kw)
    return base


def test_usable_rule_reasons():
    assert svy.usable_rule(_rec()) == (True, "ok")
    assert svy.usable_rule(_rec(name="CricketX"))[1] == "hard_exclude"
    assert svy.usable_rule(_rec(name="SemgHandGenderCh2"))[1] == "semg_duplicate"
    assert svy.usable_rule(_rec(equal_length=False))[1] == "variable_length"
    assert svy.usable_rule(_rec(missing=True))[1] == "missing_values"
    assert svy.usable_rule(_rec(length=24))[1] == "L<50"
    assert svy.usable_rule(_rec(min_class=1))[1] == "class<2"
    assert svy.usable_rule(_rec(n_total=100), min_n_eval=50)[1] == "n_eval<50"
    assert svy.usable_rule(_rec(n_total=5000), n_cap=1024, min_n_eval=300)[0]      # 1024*0.3=307 ≥ 300
    assert len(svy.HARD_EXCLUDE) == 9


def test_pool_entries_fields_and_crop():
    ents = svy.pool_entries([_rec(name="B", length=2000, n_classes=4), _rec(name="A", length=100, n_classes=2)])
    assert [e["name"] for e in ents] == ["A", "B"]
    a, b = ents
    assert a["crop_len"] is None and a["l_enc"] == 100 and a["n_classes_range"] == [2, 2] and a["planned_tasks"] == 1
    assert b["crop_len"] == svy.CROP_LEN and b["l_enc"] == svy.CROP_LEN and b["n_channels"] == 1
    assert svy.CROP_LEN == 512                      # 用户 2026-09-04 拍板（PROTOCOL §3）


# ---------------------------------------------------------------- 整数据集任务
def test_full_dataset_task_small_takes_all():
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    t = g41.full_dataset_task("Foo", y, n_cap=1024, seed=0)
    assert t.task_id == "Foo_00" and t.channels == (0,) and t.classes == (0, 1, 2)
    assert t.sample_indices == tuple(range(7)) and t.pool == "Foo"


def test_full_dataset_task_subsamples_stratified_and_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=3000)
    t1 = g41.full_dataset_task("Big", y, n_cap=1024, seed=0)
    t2 = g41.full_dataset_task("Big", y, n_cap=1024, seed=0)
    assert t1.sample_indices == t2.sample_indices and len(t1.sample_indices) == 1024
    assert list(t1.sample_indices) == sorted(t1.sample_indices)
    sub = y[list(t1.sample_indices)]
    assert set(sub.tolist()) == set(range(5))
    # 分层：子采样类比例与全体类比例之差 < 2%
    for c in range(5):
        assert abs(np.mean(sub == c) - np.mean(y == c)) < 0.02
    t3 = g41.full_dataset_task("Big", y, n_cap=1024, seed=1)
    assert t3.sample_indices != t1.sample_indices


def test_task_gates_reject_singleton_class_and_small_eval():
    y_ok = np.repeat(np.arange(3), 60)                 # 180 样本 → n_eval 54
    assert g41.task_gates(g41.full_dataset_task("A", y_ok, n_cap=1024, seed=0), y_ok, min_n_eval=50) == 54
    y_bad = np.array([0] * 50 + [1])
    with pytest.raises(ValueError, match="class"):
        g41.task_gates(g41.full_dataset_task("B", y_bad, n_cap=1024, seed=0), y_bad, min_n_eval=10)
    with pytest.raises(ValueError, match="n_eval"):
        g41.task_gates(g41.full_dataset_task("C", y_ok, n_cap=1024, seed=0), y_ok, min_n_eval=100)


# ---------------------------------------------------------------- 计划与缓存键
def test_plan_has_20_cls_per_dataset_and_keys_unique():
    cells = g41.build_plan(["A", "B"])
    assert len(cells) == 40
    assert sum(c.origin == "shared" for c in cells) == 20 and sum(c.origin == "random" for c in cells) == 20
    assert len({(c.slice_id, c.cls_key) for c in cells}) == 40
    assert {c.slice_id for c in cells} == {"A_00", "B_00"}
    # random 由 task_id 派生：同名同结果、异名异结果
    ra = [c.cls_key for c in cells if c.slice_id == "A_00" and c.origin == "random"]
    rb = [c.cls_key for c in cells if c.slice_id == "B_00" and c.origin == "random"]
    assert ra == [c.cls_key for c in g41.build_plan(["A"]) if c.origin == "random"] and ra != rb


def test_trajectory_states_and_pending(tmp_path):
    cells = g41.build_plan(["A"])
    c0, c1 = cells[0], cells[1]
    rows = [{"task_id": "A_00", "cls_key": c0.cls_key, "iters": b, "r_prime": 0.5, "error": None} for b in (60, 120, 290)]
    rows += [{"task_id": "A_00", "cls_key": c1.cls_key, "iters": 60, "r_prime": None, "error": "boom"}]
    idx = g41.cache_index(rows)
    st = g41.cell_states(cells, idx, (60, 120, 290))
    assert c0 in st["done"] and c1 in st["terminal"] and len(st["missing"]) == 18
    pend = g41.pending_cells(cells, idx, (60, 120, 290), resume=True, redo_partial=False)
    assert c0 not in pend and c1 not in pend and len(pend) == 18


def test_tasks_json_roundtrip(tmp_path):
    y = np.repeat(np.arange(2), 100)
    t = g41.full_dataset_task("Foo", y, n_cap=1024, seed=0)
    d = g41.task_to_json(t, dataset="Foo", n_total=200, n_eval=60)
    assert d["task_id"] == "Foo_00" and d["n_eval"] == 60 and d["classes"] == [0, 1]
    s = json.dumps({"tasks": {d["task_id"]: d}})
    back = g41.task_from_json(json.loads(s)["tasks"]["Foo_00"])
    assert back == t
