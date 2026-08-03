"""阶段1 单卡调度器（scripts/run_stage1.py）的可测部分。铁律 5 先测后写。

调度决策本身在 autocls/cloud_plan.py（已单测）；这里测"决策怎么变成进程"——
错一个环境变量名就是静默跑错配置（跑 4.2× 慢、或毒条目归因错乱），而这类错误
在 10 小时的无人值守跑里没人会发现。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_mod", ROOT / "scripts" / "run_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_stage1_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


RS = _load()


@pytest.fixture(scope="module")
def cfg():
    from autocls.cloud_plan import load_runtime
    return load_runtime()


def test_spawn_env_carries_single_thread_and_per_dataset_paths(cfg, tmp_path):
    env = RS.spawn_env("SelfRegulationSCP2", cfg, outdir=tmp_path, device="cuda:0")
    assert env["OMP_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "1"
    assert env["STAGE"] == "step0"
    assert env["STEP0_DATASETS"] == "SelfRegulationSCP2"
    # 每数据集独立的看门狗目录 + inflight：共用一份会让毒条目归因到别的数据集
    assert "SelfRegulationSCP2" in env["OUT"]
    assert "SelfRegulationSCP2" in env["EXTRA_ARGS"] or "inflight" in env["EXTRA_ARGS"]
    assert "--inflight" in env["EXTRA_ARGS"]
    # 并发档第一遍：OOM 跳过留待串行收尾，绝不在这里判终局
    assert "--on-oom skip" in env["EXTRA_ARGS"]
    # canonical R' 必须落进全量共享缓存（阶段2 逐键复用的前提）
    assert env["CACHE"].endswith("labels_route_a/rprime_cache.jsonl")
    # 解释器钉死：靠 PATH 解析 `python` 会拉起错的那个（本机实证 rc=49 的 Store 占位符）
    assert env["PY"] == sys.executable


def test_spawn_env_does_not_leak_a_shared_inflight_between_datasets(cfg, tmp_path):
    a = RS.spawn_env("ERing", cfg, outdir=tmp_path, device="cuda:0")
    b = RS.spawn_env("NATOPS", cfg, outdir=tmp_path, device="cuda:0")
    assert a["OUT"] != b["OUT"]
    assert a["EXTRA_ARGS"] != b["EXTRA_ARGS"]


def test_finisher_runs_serially_in_record_mode(tmp_path):
    argv = RS.finisher_argv("SelfRegulationSCP2", outdir=tmp_path, device="cuda:0")
    assert "--on-oom" in argv and argv[argv.index("--on-oom") + 1] == "record"
    assert "--resume" in argv and "--run" in argv
    assert "--datasets" in argv
    # 收尾扫描是"降到 worker=1 单独重跑该条"的落地：一次只喂一个数据集
    assert argv[argv.index("--datasets") + 1] == "SelfRegulationSCP2"
    assert argv.count("--datasets") == 1


def test_remaining_items_discounts_cached_records(tmp_path):
    cache = tmp_path / "rprime_cache.jsonl"
    rows = [{"dataset": "NATOPS", "task_id": "NATOPS_00", "cls_key": f"k{i}", "seed": 0}
            for i in range(20)]
    rows += [{"dataset": "ERing", "task_id": "ERing_00", "cls_key": "k0", "seed": 1}]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert RS.remaining_items("NATOPS", [cache], planned=80) == 60
    assert RS.remaining_items("ERing", [cache], planned=80) == 79
    assert RS.remaining_items("Cricket", [cache], planned=80) == 80


def test_remaining_items_never_negative_when_retry_rounds_inflate_cache(tmp_path):
    # 天花板加难重探会给同一数据集多写一批（+c 档标任务），条数可能超过 planned
    cache = tmp_path / "c.jsonl"
    rows = [{"dataset": "NATOPS", "task_id": f"NATOPS+c5-6_0{i%2}", "cls_key": f"k{i}",
             "seed": i % 4} for i in range(200)]
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert RS.remaining_items("NATOPS", [cache], planned=80) == 0


def test_watchdog_command_uses_posix_path_and_a_real_bash():
    """本机 e2e 实证：`["bash", str(WindowsPath)]` 会被 System32 的 WSL bash 接走并把
    `D:\\...\\cloud_run.sh` 当成 `D:DevelopDocCLS...`（rc=127）。云上 Linux 不会踩，
    但本地跑不了就没法在开跑前验证整条启动链——所以显式解析 bash + 用 posix 路径。"""
    cmd = RS.watchdog_cmd()
    assert cmd[1].endswith("scripts/cloud_run.sh")
    assert "\\" not in cmd[1]
    assert "system32" not in cmd[0].lower(), "System32 的 bash 是 WSL，不吃 Windows 路径"


def test_planned_items_matches_probe_protocol(cfg):
    # step0 协议：每数据集 n_tasks(2) × n_cls(10) × n_seeds(4) = 80 条
    assert RS.PLANNED_ITEMS_PER_DATASET == 80
