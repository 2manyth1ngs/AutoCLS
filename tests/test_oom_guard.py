"""OOM 防护（GATE_5090_REPORT §4「必须改的三件事」第 3 条）。铁律 5 先测后写，无 GPU 依赖。

背景（会静默毁数据的配置错误）：`evaluate_rprime` 原先把**一切**异常吞成
`r_prime=null` 写缓存 → `--resume` 按缓存键跳过不重试 → 看门狗还会把连崩条目
主动隔离。于是显存不足 = **永久丢 label 且无任何报错**。中档（L~300-500）
worker=2 是未实测的保守值，本防护是唯一防线，必须做对。

被测四条（对应用户验收项 ①②③④）：
- `is_oom_error`：把"显存/内存不足"与"这条数据/训练本身坏"分开——只有前者
  才有资格"不写缓存"，后者仍须记 NaN 否则实验会被一条坏数据卡死；
- ① `evaluate_rprime` 遇 OOM 抛 `RprimeOOM`，**绝不**返回 r_prime=NaN 的记录；
  普通异常仍记 NaN（不中断实验，保持既有语义）；
- ①② `guarded_run` 遇 OOM 不写任何缓存行 → 该条保持"未跑"，下一轮 --resume
  重试（降 worker 的串行收尾扫描即该重试）；
- ③ OOM 后清空 inflight → `scripts/cloud_run.sh` 的 `[ -s $INFLIGHT ]` 为假，
  不计入 FAIL_LIMIT=3 毒条目计数（OOM 是配置问题，不是"这条数据有毒"）；
- ④ 已降到 worker=1 仍 OOM（on_oom=record）→ 写 oom_failures.jsonl 显式可见，
  而不是悄悄消失。
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autocls.oom_guard import (
    ON_OOM_RECORD,
    ON_OOM_SKIP,
    RprimeOOM,
    append_oom_failure,
    guarded_run,
    is_oom_error,
)

# ---------------------------------------------------------------------------
# is_oom_error：显存/内存不足 vs 普通失败
# ---------------------------------------------------------------------------


def test_is_oom_error_detects_torch_cuda_oom_class():
    torch = pytest.importorskip("torch")
    exc = torch.cuda.OutOfMemoryError(
        "CUDA out of memory. Tried to allocate 15.80 GiB (GPU 0; 31.36 GiB total capacity)")
    assert is_oom_error(exc)


@pytest.mark.parametrize("msg", [
    "CUDA out of memory. Tried to allocate 2.00 GiB",
    "CUDA error: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8589934592 bytes",
])
def test_is_oom_error_detects_by_message(msg):
    # 旧版 torch / 非 torch 层（cuBLAS、cuDNN、CPU 分配器）不抛 OutOfMemoryError，
    # 只能按消息认——漏认一条就等于漏掉一条 label
    assert is_oom_error(RuntimeError(msg))


def test_is_oom_error_detects_host_memory_error():
    # 8 worker 同跑时宿主内存也可能耗尽；同属"配置/资源"问题，处置方式相同（降并发重试）
    assert is_oom_error(MemoryError("std::bad_alloc"))


@pytest.mark.parametrize("exc", [
    ValueError("训练发散：loss 变 NaN"),
    RuntimeError("CUDA error: device-side assert triggered"),
    RuntimeError("BrokenProcessPool: A task has failed to un-serialize"),
    KeyError("cls"),
])
def test_is_oom_error_rejects_ordinary_failures(exc):
    # 误判成 OOM = 这条永远不写缓存 → 每轮 resume 都重跑 → 真死循环。必须严格
    assert not is_oom_error(exc)


# ---------------------------------------------------------------------------
# ① evaluate_rprime：OOM 抛出，普通异常仍记 NaN
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_pipeline(monkeypatch):
    """把 evaluate_rprime 内部的重依赖（切片/建 encoder/预训练）换成桩，无 GPU 无训练。

    返回一个设置"预训练抛什么异常"的函数。
    """
    X = np.random.default_rng(0).normal(size=(40, 16, 2)).astype(np.float32)
    y = np.array([0, 1] * 20)

    def _install(exc: BaseException):
        def boom(*_a, **_kw):
            raise exc

        monkeypatch.setattr("autocls.data.task_slicer.materialize_task",
                            lambda *_a, **_kw: (X, y))
        monkeypatch.setattr("autocls.encoders.registry.build_encoder",
                            lambda *_a, **_kw: object())
        monkeypatch.setattr("autocls.contrastive.cl_trainer.pretrain_encoder", boom)

    return _install


def _eval(make_cls):
    from autocls.labeling import evaluate_rprime
    task = SimpleNamespace(task_type="classification")
    return evaluate_rprime(
        task, make_cls(), np.zeros((1, 1, 1), dtype=np.float32), np.zeros(1),
        weight_seed=1, split_seed=2, iters=3, device="cpu",
    )


def test_evaluate_rprime_raises_on_oom_instead_of_null_record(stub_pipeline, make_cls):
    stub_pipeline(RuntimeError("CUDA out of memory. Tried to allocate 15.80 GiB"))
    with pytest.raises(RprimeOOM) as ei:
        _eval(make_cls)
    assert "out of memory" in str(ei.value).lower()


def test_evaluate_rprime_keeps_nan_record_for_ordinary_error(stub_pipeline, make_cls):
    # 回归护栏：普通异常（训练发散）仍按既有语义记 NaN 落盘，不因本次改动变成抛异常
    stub_pipeline(ValueError("训练发散"))
    rec = _eval(make_cls)
    assert math.isnan(rec["r_prime"])
    assert "ValueError" in rec["error"]


# ---------------------------------------------------------------------------
# ①②③④ guarded_run：缓存 / inflight / oom_failures 三件产物的行为
# ---------------------------------------------------------------------------


def _paths(tmp_path: Path):
    return (tmp_path / "rprime_cache.jsonl", tmp_path / "inflight.txt",
            tmp_path / "oom_failures.jsonl")


def _row(rec):
    return {"task_id": "SCP2_00", "cls_key": "deadbeef12ab", "iters": 290, **rec}


def test_guarded_run_writes_cache_and_clears_inflight_on_success(tmp_path):
    cache, inflight, oom_log = _paths(tmp_path)
    row = guarded_run(lambda: {"r_prime": 0.7}, _row, cache_path=cache, inflight=inflight,
                      task_id="SCP2_00", cls_key="deadbeef12ab", iters=290,
                      on_oom=ON_OOM_SKIP, oom_log=oom_log, log=lambda _m: None)
    assert row["r_prime"] == 0.7
    assert len(cache.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert inflight.read_text(encoding="utf-8").strip() == ""
    assert not oom_log.exists()


def test_guarded_run_oom_writes_no_cache_line(tmp_path):
    # ① OOM 绝不写 r_prime=null：缓存里没有这条 → --resume 视作未跑 → ② 会重试
    cache, inflight, oom_log = _paths(tmp_path)

    def boom():
        raise RprimeOOM("CUDA out of memory. Tried to allocate 15.80 GiB")

    row = guarded_run(boom, _row, cache_path=cache, inflight=inflight,
                      task_id="SCP2_00", cls_key="deadbeef12ab", iters=290,
                      on_oom=ON_OOM_SKIP, oom_log=oom_log, log=lambda _m: None)
    assert row is None
    assert not cache.exists(), "OOM 写了缓存行 = 这条 label 永久丢失"


def test_guarded_run_oom_leaves_inflight_empty(tmp_path):
    # ③ 看门狗只在 [ -s $INFLIGHT ] 为真时累计毒条目；OOM 后必须清空，
    #    否则连续 OOM 3 次会被写 r_prime=null 永久隔离（把配置问题当成毒数据）
    cache, inflight, oom_log = _paths(tmp_path)

    def boom():
        raise RprimeOOM("CUDA out of memory")

    guarded_run(boom, _row, cache_path=cache, inflight=inflight,
                task_id="SCP2_00", cls_key="deadbeef12ab", iters=290,
                on_oom=ON_OOM_SKIP, oom_log=oom_log, log=lambda _m: None)
    assert inflight.exists() and inflight.read_text(encoding="utf-8").strip() == ""


def test_guarded_run_skip_mode_does_not_record_terminal_failure(tmp_path):
    # skip = 并发跑时的第一遍：这条留给串行收尾重试，还不算"跑不了"
    cache, inflight, oom_log = _paths(tmp_path)
    msgs = []

    def boom():
        raise RprimeOOM("CUDA out of memory")

    guarded_run(boom, _row, cache_path=cache, inflight=inflight,
                task_id="SCP2_00", cls_key="deadbeef12ab", iters=290,
                on_oom=ON_OOM_SKIP, oom_log=oom_log, log=msgs.append)
    assert not oom_log.exists()
    assert any("OOM" in m for m in msgs), "skip 模式也必须在日志里显式可见"


def test_guarded_run_record_mode_writes_oom_failures(tmp_path):
    # ④ 已经是 worker=1 仍 OOM → 显式报错并落 oom_failures.jsonl，让人看得见
    #    "哪些条目因显存跑不了"，而不是悄悄消失
    cache, inflight, oom_log = _paths(tmp_path)
    msgs = []

    def boom():
        raise RprimeOOM("CUDA out of memory. Tried to allocate 15.80 GiB")

    row = guarded_run(boom, _row, cache_path=cache, inflight=inflight,
                      task_id="SCP2_00", cls_key="deadbeef12ab", iters=290,
                      on_oom=ON_OOM_RECORD, oom_log=oom_log,
                      context={"dataset": "SelfRegulationSCP2", "workers": 1},
                      log=msgs.append)
    assert row is None
    assert not cache.exists()
    lines = oom_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["task_id"] == "SCP2_00"
    assert rec["cls_key"] == "deadbeef12ab"
    assert rec["iters"] == 290
    assert rec["dataset"] == "SelfRegulationSCP2"
    assert rec["workers"] == 1
    assert "out of memory" in rec["error"].lower()
    assert any("oom_failures" in m for m in msgs), "终局失败必须在日志里指向落盘位置"


def test_guarded_run_lets_ordinary_exception_propagate(tmp_path):
    # 非 OOM 的意外异常不归本防护管（进程级崩溃由看门狗 + 毒条目机制处理）
    cache, inflight, oom_log = _paths(tmp_path)

    def boom():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        guarded_run(boom, _row, cache_path=cache, inflight=inflight,
                    task_id="T_00", cls_key="k", iters=290,
                    on_oom=ON_OOM_SKIP, oom_log=oom_log, log=lambda _m: None)


def test_append_oom_failure_is_append_only(tmp_path):
    p = tmp_path / "sub" / "oom_failures.jsonl"
    append_oom_failure(p, {"task_id": "A", "error": "x"})
    append_oom_failure(p, {"task_id": "B", "error": "y"})
    ids = [json.loads(x)["task_id"] for x in p.read_text(encoding="utf-8").splitlines()]
    assert ids == ["A", "B"]
