"""step0 跑批层：tasks_meta 分数据集落盘 + OOM/inflight 命令行接线。铁律 5 先测后写。

为什么改 tasks_meta 的落盘方式（手册 §0.6 同一问题的根因）：`run()` 从空字典重建
再整份覆盖写 `tasks_meta.json` → **换一批 --datasets 再跑一次，上一批的记录就没了**，
`--analyze` 对丢掉的数据集报"无 tasks_meta 记录，跳过"，天花板重探记录全丢、判读残废。
阶段1 单卡按数据集并发跑（每数据集一个进程），这是必然发生的场景，不是边界情况。
改成**每数据集一个文件**：并发写不同文件天然无竞争，读时合并；旧的整份 json 仍能读
（本机既有 smoke 产物与 5090 闸门产物都是旧格式）。
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_step0():
    spec = importlib.util.spec_from_file_location(
        "step0_snr_gate_mod", ROOT / "scripts" / "step0_snr_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step0_snr_gate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


STEP0 = _load_step0()


def _meta(l_enc: int) -> dict:
    return {"L": l_enc, "n_ch": 3, "n_classes": 6, "n_classes_range": [3, 6],
            "tasks": [{"task_id": "X_00"}], "retry_rounds": []}


def test_per_dataset_meta_files_do_not_clobber_each_other(tmp_path):
    STEP0.write_tasks_meta(tmp_path, "NATOPS", _meta(51))
    STEP0.write_tasks_meta(tmp_path, "SelfRegulationSCP2", _meta(1152))
    got = STEP0.read_tasks_meta(tmp_path)
    assert set(got) == {"NATOPS", "SelfRegulationSCP2"}
    assert got["NATOPS"]["L"] == 51


def test_rewriting_one_dataset_keeps_the_others(tmp_path):
    # 天花板重探会就地更新本数据集的 retry_rounds，不能顺手清掉别的数据集
    STEP0.write_tasks_meta(tmp_path, "ERing", _meta(65))
    STEP0.write_tasks_meta(tmp_path, "NATOPS", _meta(51))
    m = _meta(65)
    m["retry_rounds"] = [{"round": 1, "n_classes_range": [5, 6], "tasks": []}]
    STEP0.write_tasks_meta(tmp_path, "ERing", m)
    got = STEP0.read_tasks_meta(tmp_path)
    assert set(got) == {"ERing", "NATOPS"}
    assert got["ERing"]["retry_rounds"][0]["round"] == 1


def test_read_falls_back_to_legacy_aggregate_file(tmp_path):
    # 既有产物（本机 smoke / 5090 闸门）是旧的整份 tasks_meta.json，必须还能判读
    (tmp_path / "tasks_meta.json").write_text(
        json.dumps({"NATOPS": _meta(51)}), encoding="utf-8")
    got = STEP0.read_tasks_meta(tmp_path)
    assert got["NATOPS"]["L"] == 51


def test_per_dataset_file_wins_over_legacy(tmp_path):
    (tmp_path / "tasks_meta.json").write_text(
        json.dumps({"NATOPS": _meta(51)}), encoding="utf-8")
    fresh = _meta(51)
    fresh["retry_rounds"] = [{"round": 1, "n_classes_range": [5, 6], "tasks": []}]
    STEP0.write_tasks_meta(tmp_path, "NATOPS", fresh)
    assert STEP0.read_tasks_meta(tmp_path)["NATOPS"]["retry_rounds"]


def test_cli_exposes_decouple_experiment_switches():
    # 解耦实验（results/decouple_random_cls/PREREGISTRATION.md §4）：换探针目录、
    # 独立缓存目录、iters=0 配对基线、禁天花板重探——全部走显式参数，默认行为不变
    ap = STEP0.build_parser()
    args = ap.parse_args(["--run", "--catalog", "x/random_cls_catalog.json",
                          "--n-cls", "20", "--cache-dir", "x/exp",
                          "--baseline", "--no-escalate"])
    assert args.catalog == Path("x/random_cls_catalog.json")
    assert args.n_cls == 20
    assert args.cache_dir == Path("x/exp")
    assert args.baseline and args.no_escalate
    d = ap.parse_args(["--run"])
    assert d.catalog == STEP0.PROBE_CATALOG and d.n_cls is None
    assert d.cache_dir is None and not d.baseline and not d.no_escalate


def test_resolve_paths_default_production_unchanged():
    canon, noise, meta_root = STEP0.resolve_paths(smoke=False, baseline=False,
                                                  cache_dir=None)
    assert canon == STEP0.SHARED_CACHE
    assert noise == STEP0.NOISE_CACHE
    assert meta_root == STEP0.OUTDIR


def test_resolve_paths_cache_dir_isolates_experiment(tmp_path):
    canon, noise, meta_root = STEP0.resolve_paths(smoke=False, baseline=False,
                                                  cache_dir=tmp_path)
    assert canon == tmp_path / "rprime_cache.jsonl"
    assert noise == tmp_path / "noise_cache.jsonl"
    assert meta_root == tmp_path


def test_resolve_paths_baseline_goes_to_dedicated_file(tmp_path):
    # iters=0 基线全部种子进同一独立文件——绝不允许混进生产 canonical 缓存
    canon, noise, _ = STEP0.resolve_paths(smoke=False, baseline=True,
                                          cache_dir=tmp_path)
    assert canon == noise == tmp_path / "baseline_cache.jsonl"


def test_resolve_paths_baseline_requires_cache_dir():
    # 防呆：--baseline 不给 --cache-dir 就会往生产缓存写 iters=0 记录，必须硬拦
    import pytest
    with pytest.raises(ValueError):
        STEP0.resolve_paths(smoke=False, baseline=True, cache_dir=None)


def test_cli_exposes_oom_and_inflight_switches():
    # 并发跑时每进程一份 inflight（否则看门狗毒条目归因错乱）；
    # --on-oom record 是"已经降到 worker=1"的串行收尾扫描才用的模式
    ap = STEP0.build_parser()
    args = ap.parse_args(["--run", "--resume", "--datasets", "NATOPS",
                          "--inflight", "x/inflight.txt", "--on-oom", "record",
                          "--oom-log", "x/oom_failures.jsonl"])
    assert args.inflight == Path("x/inflight.txt")
    assert args.on_oom == "record"
    assert args.oom_log == Path("x/oom_failures.jsonl")
    assert ap.parse_args(["--run"]).on_oom == "skip"   # 默认并发档：留给收尾重试
