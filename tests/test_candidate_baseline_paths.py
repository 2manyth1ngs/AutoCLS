"""baseline_probe 的路径/注册表可重定向性测试（换池调研步骤 2 配套；铁律 5 先测后写）。

为什么需要：换池调研要拿**同一个** baseline_probe 去测候选数据集（口径一致性是本轮
全部结论的前提，见 `results/candidate_survey/PROTOCOL.md` §3），但它的缓存、
注册表、analysis.json 路径原本硬编码指向 `results/baseline_probe/` 与
`configs/source_pool.yaml`。候选跑进同一份缓存会污染 UEA 那 685 条已定稿数据
（铁律 3：那是最贵的一步，勿删勿扩）。

故要求 `configure_paths(outdir)` 把全部落盘路径整体重定向，且 `--registry` 能换注册表。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

bp = importlib.import_module("scripts.baseline_probe")


def test_default_paths_point_at_uea_probe_outdir():
    assert bp.OUTDIR.name == "baseline_probe"
    assert bp.CACHE.parent == bp.OUTDIR


def test_configure_paths_redirects_every_output(tmp_path):
    orig = bp.OUTDIR
    try:
        bp.configure_paths(tmp_path / "cand")
        for p in (bp.CACHE, bp.INFLIGHT, bp.OOM_LOG, bp.TRAINED_FILL):
            assert p.parent == tmp_path / "cand", p
        # 历史 trained 缓存链里那份 "OUTDIR/trained_fill.jsonl" 也必须跟着走，
        # 否则候选的配对参照会去读 UEA 的补跑结果
        assert bp.TRAINED_FILL in bp.TRAINED_CACHES
        assert all(bp.OUTDIR in p.parents or p.parent == bp.OUTDIR
                   for p in [bp.TRAINED_FILL])
    finally:
        bp.configure_paths(orig)


def test_configure_paths_is_reversible(tmp_path):
    orig_cache = bp.CACHE
    bp.configure_paths(tmp_path / "x")
    bp.configure_paths(orig_cache.parent)
    assert bp.CACHE == orig_cache


def test_cli_exposes_registry_and_outdir():
    """--registry / --outdir 必须真的暴露出来（否则步骤 2 只能改源码跑）。"""
    src = (ROOT / "scripts" / "baseline_probe.py").read_text(encoding="utf-8")
    assert '"--registry"' in src
    assert '"--outdir"' in src


def test_fill_trained_task_count_is_parameterized():
    """workpoint followup 加测要把 fill 扩到前 6 任务（60 条确认口径）。

    FILL_TASKS=2 是模块常数：不暴露参数就只能改源码跑，且默认值必须仍是 2
    ——历史 fill（第3步/full-iters/工作点扫描）全部按前 2 任务口径，默认漂移
    会让 --resume 的"待跑计划"悄悄变大。
    """
    import inspect

    sig = inspect.signature(bp.fill_trained)
    assert "n_tasks" in sig.parameters
    assert sig.parameters["n_tasks"].default == bp.FILL_TASKS == 2
    src = (ROOT / "scripts" / "baseline_probe.py").read_text(encoding="utf-8")
    assert '"--fill-tasks"' in src
