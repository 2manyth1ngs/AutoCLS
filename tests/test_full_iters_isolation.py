"""`--fill-trained` 的 iters 隔离（full-iters 判别实验配套；铁律 5 先测后写）。

为什么需要（`results/full_iters_probe/PROTOCOL.md` §2）：判别实验要拿 **2320 iters**
重跑与第3步**完全相同**的 (任务, CLS, 初始化, 划分)，唯一变量是训练量。但原实现有
两处把 early 与 full 混在一起，都会静默出错：

1. `fill_trained` 的 resume 去重键是 `(task_id, cls_key)` —— 290 的记录会挡住 2320
   的重跑，`--resume` 下整批直接跳过，看起来"秒完成"实则一条没跑。
   对照 `labeling.load_rprime_cache` 早就把 iters 入键并注明"防 early/full 混用（铁律 2）"
   ——本处是唯一的例外。
2. `_trained_by_task` 不看 iters，把所有历史缓存的 R' 归到同一个 task_id 下 →
   `realized_gain` 会变成 290 与 2320 的混合均值，`best_gain` 取两者最大。

铁律 2 原文："`cl_trainer` 的 `max_iters` 必须显式传参，early-val 与最终评估禁止混用"。
"禁止混用"必须落到缓存键与配对口径上，否则只是口号。
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

bp = importlib.import_module("scripts.baseline_probe")

EARLY, FULL = 290, 2320


def _task(tid: str, classes=(0, 1, 2)):
    """_trained_by_task 只用到 .classes（配对校验）与 dict 的键。"""
    return types.SimpleNamespace(task_id=tid, classes=tuple(classes))


def _rec(tid: str, key: str, iters: int, r_prime: float, classes=(0, 1, 2)) -> dict:
    return {"task_id": tid, "cls_key": key, "iters": iters, "r_prime": r_prime,
            "classes": list(classes), "split_seed": bp.baseline_seed_specs(tid, 0)[1]}


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


# --- 1. resume 去重键必须含 iters -------------------------------------------

def test_fill_done_keys_include_iters(tmp_path):
    """290 已跑过的 (task, cls) 不得让 2320 的同一条被 --resume 跳过。"""
    p = tmp_path / "trained_fill.jsonl"
    _write(p, [_rec("T_00", "cls_a", EARLY, 0.5)])
    done = bp.fill_done_keys(p)
    assert ("T_00", "cls_a", EARLY) in done
    assert ("T_00", "cls_a", FULL) not in done, "iters 未入 resume 键：full 会被 early 挡住"


def test_fill_done_keys_distinguish_two_iters_of_same_pair(tmp_path):
    p = tmp_path / "trained_fill.jsonl"
    _write(p, [_rec("T_00", "cls_a", EARLY, 0.5), _rec("T_00", "cls_a", FULL, 0.7)])
    assert bp.fill_done_keys(p) == {("T_00", "cls_a", EARLY), ("T_00", "cls_a", FULL)}


def test_fill_done_keys_missing_file_is_empty(tmp_path):
    assert bp.fill_done_keys(tmp_path / "nope.jsonl") == set()


# --- 2. 配对时必须按 iters 过滤 ---------------------------------------------

def test_trained_by_task_filters_by_iters(tmp_path, monkeypatch):
    """混在一份缓存里的 290 与 2320，配对时只能取被请求的那个训练量。"""
    p = tmp_path / "mixed.jsonl"
    _write(p, [_rec("T_00", "cls_a", EARLY, 0.50), _rec("T_00", "cls_b", EARLY, 0.52),
               _rec("T_00", "cls_a", FULL, 0.80), _rec("T_00", "cls_b", FULL, 0.82)])
    monkeypatch.setattr(bp, "TRAINED_CACHES", (p,))
    tasks = {"T_00": _task("T_00")}

    monkeypatch.setattr(bp, "PAIR_ITERS", EARLY)
    early, audit_e = bp._trained_by_task(tasks)
    assert sorted(early["T_00"]) == [0.50, 0.52], "290 配对里混进了 full 的 R'"

    monkeypatch.setattr(bp, "PAIR_ITERS", FULL)
    full, audit_f = bp._trained_by_task(tasks)
    assert sorted(full["T_00"]) == [0.80, 0.82], "full 配对里混进了 early 的 R'"
    assert audit_e["matched"] == audit_f["matched"] == 2
    # 被 iters 过滤掉的要能审计出来，不能悄悄消失
    assert audit_e.get("rejected_iters") == 2


def test_trained_by_task_default_pair_iters_is_early():
    """默认口径仍是 290：既有 analysis.json 全部按它生成，改默认会让旧结果不可复算。"""
    assert bp.PAIR_ITERS == EARLY == bp.FILL_ITERS


def test_existing_caches_are_all_early_so_filter_is_noop():
    """实测护栏：5 份历史 trained 缓存若全是 290，则加 iters 过滤对既有分析是 no-op。

    这条断言把"不改任何已发布数字"从口头承诺变成可执行检查——将来若有人往这些
    缓存里追加别的 iters，本测试会立刻失败，提醒他既有报表需要重算。
    """
    for path in bp.TRAINED_CACHES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert json.loads(line).get("iters") == EARLY, f"{path} 混入非 290 记录"


# --- 3. CLI 必须暴露出来 -----------------------------------------------------

def test_cli_exposes_fill_iters_and_pair_iters():
    src = (ROOT / "scripts" / "baseline_probe.py").read_text(encoding="utf-8")
    assert '"--fill-iters"' in src
    assert '"--pair-iters"' in src
