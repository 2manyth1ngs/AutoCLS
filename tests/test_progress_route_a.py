"""进度自查工具（scripts/progress_route_a.py）测试：喂样例落盘文件 → 断言汇总计数。

Part A-4：只读工具，随时可跑。四块信息：
① step0 过闸（每数据集 verdict/SNR/R̄/恢复档/pending）+ 任务数闸 count/floor；
② 存活任务总数（tested 骨干 ∪ analysis 通过者，对照 floor=100）；
③ 阶段2 label 进度（缓存条数/计划总数、per-dataset 完成率、隔离/失败数、配对数）；
④ 看门狗（重启次数=worker 日志数、毒条目计数、inflight 残留）。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def progress_mod():
    spec = importlib.util.spec_from_file_location(
        "_progress_route_a", ROOT / "scripts" / "progress_route_a.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_progress_route_a"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


@pytest.fixture()
def sample_dirs(tmp_path):
    step0 = tmp_path / "step0"
    labels = tmp_path / "labels"
    step0.mkdir()
    labels.mkdir()

    # step0 analysis：NATOPS 重探恢复通过（恢复档 c5-6）、ERing 失配踢除
    (step0 / "analysis.json").write_text(json.dumps({
        "datasets": {
            "NATOPS": {"verdict": "pass", "passed": True, "snr": 1.4,
                       "rprime_grand_mean": 0.88, "pending_retry": False,
                       "recommended_n_classes_range": [5, 6], "suggested_delta": 0.05},
            "ERing": {"verdict": "fail_mismatch", "passed": False, "snr": 0.4,
                      "rprime_grand_mean": 0.91, "pending_retry": False,
                      "recommended_n_classes_range": None, "suggested_delta": 0.1},
        },
        "gate": {"ok": False, "count": 54, "floor": 100, "passed_untested": ["NATOPS"],
                 "message": "源池任务 54 < 100"},
    }), encoding="utf-8")

    # 阶段2 缓存：NATOPS 2 条完成 + 1 条看门狗隔离 + 1 条训练失败；Cricket 1 条完成
    _write_jsonl(labels / "rprime_cache.jsonl", [
        {"task_id": "NATOPS+c5-6_00", "cls_key": "aaa", "iters": 290, "r_prime": 0.7},
        {"task_id": "NATOPS+c5-6_00", "cls_key": "bbb", "iters": 290, "r_prime": 0.6},
        {"task_id": "NATOPS+c5-6_01", "cls_key": "aaa", "iters": 290, "r_prime": None,
         "error": "watchdog: quarantined after 3 crashes"},
        {"task_id": "NATOPS+c5-6_01", "cls_key": "bbb", "iters": 290, "r_prime": None,
         "error": "RuntimeError: boom"},
        {"task_id": "Cricket_00", "cls_key": "aaa", "iters": 290, "r_prime": 0.5},
        {"task_id": "Old_00", "cls_key": "aaa", "iters": 30, "r_prime": 0.9},  # 非 290 口径不计
    ])
    _write_jsonl(labels / "pairs.jsonl", [
        {"task_id": "NATOPS+c5-6_00", "a": "aaa", "b": "bbb", "y": 1},
    ])
    (labels / "summary.json").write_text(json.dumps({
        "total": {"n_candidates": 10, "n_delta_dropped": 4},
        "delta_drop_frac": 0.4,
    }), encoding="utf-8")

    # 看门狗：labels 侧 2 次重启、毒条目 badc0de×2；inflight 残留一条
    (labels / "worker_r0.log").write_text("x", encoding="utf-8")
    (labels / "worker_r1.log").write_text("x", encoding="utf-8")
    (labels / "watchdog_state.tsv").write_text(
        "T_00\tbadc0de\t290\nT_00\tbadc0de\t290\n", encoding="utf-8")
    (labels / "inflight.txt").write_text("T_01\tccc\t290", encoding="utf-8")
    return step0, labels


def test_collect_step0_and_gate(progress_mod, sample_dirs):
    step0, labels = sample_dirs
    p = progress_mod.collect_progress(step0_dir=step0, labels_dir=labels)
    assert p["step0"]["datasets"]["NATOPS"]["passed"] is True
    assert p["step0"]["datasets"]["NATOPS"]["recommended_n_classes_range"] == [5, 6]
    assert p["step0"]["datasets"]["ERing"]["verdict"] == "fail_mismatch"
    assert p["step0"]["gate"]["count"] == 54
    assert p["step0"]["gate"]["floor"] == 100


def test_collect_labels_progress(progress_mod, sample_dirs):
    step0, labels = sample_dirs
    p = progress_mod.collect_progress(step0_dir=step0, labels_dir=labels)
    lab = p["labels"]
    assert lab["n_done"] == 3            # 有限 R'（290 口径）：NATOPS 2 + Cricket 1
    assert lab["n_quarantined"] == 1     # watchdog 隔离
    assert lab["n_failed"] == 1          # 训练失败 NaN（非隔离）
    assert lab["per_dataset"]["NATOPS"]["done"] == 2
    assert lab["per_dataset"]["Cricket"]["done"] == 1
    assert lab["n_pairs"] == 1
    assert lab["delta_drop_frac"] == 0.4
    # 计划总数：tested 骨干 44 任务 + NATOPS(passed) 10 任务 = 54 × 20 CLS = 1080
    assert lab["planned_records"] == 54 * 20


def test_collect_watchdog_state(progress_mod, sample_dirs):
    step0, labels = sample_dirs
    p = progress_mod.collect_progress(step0_dir=step0, labels_dir=labels)
    wd = p["watchdog"]["labels"]
    assert wd["restarts"] == 2
    assert wd["poison"] == {"T_00\tbadc0de\t290": 2}
    assert wd["inflight"] == "T_01\tccc\t290"
    assert p["watchdog"]["step0"]["restarts"] == 0


def test_collect_handles_missing_everything(progress_mod, tmp_path):
    # 全缺省（还没跑任何东西）也不崩：报"未跑"状态
    p = progress_mod.collect_progress(step0_dir=tmp_path / "no1", labels_dir=tmp_path / "no2")
    assert p["step0"] is None
    assert p["labels"]["n_done"] == 0
    assert p["labels"]["n_pairs"] is None
