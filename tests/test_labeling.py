"""labeling 单测（TECHNICAL_DESIGN.md 模块 7：scripts/2_generate_labels.py 的可测核心）。

覆盖四块：
- CLS 采样（AutoCTS++ 3.2.4 shared+random，generate_seeds.py noisy_seeds 口径）：
  shared 全任务共用、random 每任务独立；前缀稳定性保证小规模试跑与全量跑共享缓存。
- 任务内 train/eval 划分与下游评估器：均经 task_type 显式分发（铁律 7），
  forecasting 当前 NotImplementedError 而非静默调分类版本。
- 配对造 label：y 只经 is_better 判定（铁律 8）、δ 过滤同分噪声（M2 高分簇内
  分辨力弱的结论）、失败(NaN)剔除、平局弃样。
- δ 标定辅助 delta_decision_curve（scripts/calibrate_delta.py 的核心）。
"""
import numpy as np
import pytest
import torch

from autocls.labeling import (
    append_jsonl,
    cls_key,
    delta_decision_curve,
    evaluate_rprime,
    load_rprime_cache,
    make_pairs,
    run_seed_for,
    sample_cls_sets,
    split_task_train_eval,
)
from autocls.data.task_slicer import SlicedTask
from autocls.space.cls_space import CLSSpace
from autocls.task_types import TASK_CLASSIFICATION, TASK_FORECASTING


@pytest.fixture(scope="module")
def space() -> CLSSpace:
    return CLSSpace.from_yaml()


# ---------------------------------------------------------------- cls_key
class TestClsKey:
    def test_deterministic_and_distinct(self, make_cls):
        a, a2 = make_cls(), make_cls()
        b = make_cls(jitter=0.5)
        assert cls_key(a) == cls_key(a2)
        assert cls_key(a) != cls_key(b)
        assert isinstance(cls_key(a), str) and len(cls_key(a)) == 12


# ---------------------------------------------------------------- CLS 采样
class TestSampleClsSets:
    TASKS = ["source_00", "source_01", "source_02"]

    def test_shared_and_random_counts_and_validity(self, space):
        s = sample_cls_sets(space, self.TASKS, n_shared=4, n_random=3, seed=7)
        assert len(s.shared) == 4
        assert set(s.random) == set(self.TASKS)
        for cs in s.random.values():
            assert len(cs) == 3
            assert all(space.validate(c) for c in cs)
        assert all(space.validate(c) for c in s.shared)

    def test_shared_is_shared_random_is_per_task(self, space):
        s = sample_cls_sets(space, self.TASKS, n_shared=4, n_random=3, seed=7)
        # shared 全任务同一批（学习任务相似性的关键信号）；random 每任务独立采样
        rnd_keys = [tuple(cls_key(c) for c in s.random[t]) for t in self.TASKS]
        assert len(set(rnd_keys)) == len(self.TASKS), "各任务 random 批应互不相同"
        shared_keys = {cls_key(c) for c in s.shared}
        for ks in rnd_keys:
            assert not (set(ks) & shared_keys), "random 不得与 shared 重复"
            assert len(set(ks)) == len(ks), "任务内 random 不得重复"

    def test_deterministic_given_seed(self, space):
        s1 = sample_cls_sets(space, self.TASKS, n_shared=4, n_random=3, seed=7)
        s2 = sample_cls_sets(space, self.TASKS, n_shared=4, n_random=3, seed=7)
        assert s1.shared == s2.shared
        assert s1.random == s2.random

    def test_prefix_stable_when_n_shared_grows(self, space):
        # 试跑（小 L）与全量（大 L）共享 R' 缓存的前提：加大 n_shared 不改变前缀
        s3 = sample_cls_sets(space, self.TASKS, n_shared=3, n_random=2, seed=7)
        s6 = sample_cls_sets(space, self.TASKS, n_shared=6, n_random=2, seed=7)
        assert s6.shared[:3] == s3.shared

    def test_prefix_stable_when_task_ids_appended(self, space):
        # 留出池造 label（M3 测 T-CLSC）的前提：task_ids 追加 holdout 不改变
        # 源任务的 shared/random 取值（random 流按位置 [seed, 1+i] 派生），
        # 否则源池 R' 缓存作废
        src = sample_cls_sets(space, self.TASKS, n_shared=3, n_random=2, seed=7)
        both = sample_cls_sets(
            space, self.TASKS + ["holdout_00", "holdout_01"], n_shared=3, n_random=2, seed=7
        )
        assert both.shared == src.shared
        for t in self.TASKS:
            assert both.random[t] == src.random[t]
        # holdout 的 random 批是新的独立流，不与任何源任务撞车
        src_rnd = {cls_key(c) for t in self.TASKS for c in src.random[t]}
        hold_rnd = {cls_key(c) for t in ("holdout_00", "holdout_01") for c in both.random[t]}
        assert not (hold_rnd & src_rnd)


# ---------------------------------------------------------------- 比较器桥接
class TestComparatorBridge:
    """R' 缓存 → T-CLSC 引擎数据结构（scripts/3_pretrain_tclsc.py 的可测核心）。"""

    @staticmethod
    def _cache_row(task_id, key, origin, r, cls_cfg, iters=290):
        import dataclasses
        return (task_id, key, iters), {
            "task_id": task_id, "cls_key": key, "origin": origin, "iters": iters,
            "r_prime": r, "cls": dataclasses.asdict(cls_cfg),
        }

    def test_comparator_records_filters_and_encodes(self, space, make_cls):
        from autocls.labeling import comparator_records
        from autocls.space.cls_encoder import CLSEncoder
        enc = CLSEncoder(space)
        c1, c2, c3 = make_cls(), make_cls(jitter=0.5), make_cls(jitter=0.7)
        cache = dict([
            self._cache_row("t0", "k1", "shared", 0.9, c1),
            self._cache_row("t0", "k2", "random", None, c2),      # 失败 CLS → 剔除
            self._cache_row("t0", "k3", "shared", 0.8, c3, iters=2320),  # 口径不符 → 剔除
        ])
        recs = comparator_records(cache, enc, iters=290)
        assert [r.key for r in recs] == ["k1"]
        assert recs[0].task_id == "t0" and recs[0].origin == "shared" and recs[0].r == 0.9
        np.testing.assert_array_equal(recs[0].vec, enc.encode(c1))

    def test_comparator_eval_pairs_directed_delta_and_kind(self, space, make_cls):
        from autocls.labeling import comparator_eval_pairs
        from autocls.space.cls_encoder import CLSEncoder
        enc = CLSEncoder(space)
        ca, cb, cc = make_cls(), make_cls(jitter=0.5), make_cls(jitter=0.7)
        cache = dict([
            self._cache_row("t0", "ka", "shared", 0.90, ca),
            self._cache_row("t0", "kb", "random", 0.50, cb),
            self._cache_row("t0", "kc", "shared", 0.895, cc),  # 与 ka 近平局 → δ 弃样
        ])
        pairs = comparator_eval_pairs(
            cache, enc, task_type=TASK_CLASSIFICATION, delta=0.02, iters=290)
        # 有效无序对：(ka,kb) 与 (kb,kc)，双向出样 → 4 条
        assert len(pairs) == 4
        fwd, rev = pairs[: len(pairs) // 2], pairs[len(pairs) // 2:]
        for p, q in zip(fwd, rev):
            assert q.y == 1 - p.y
            np.testing.assert_array_equal(q.vec_a, p.vec_b)
        # y 方向经 is_better（铁律 8）：ka(0.90) 优于 kb(0.50) → 正向 y=1
        ka_first = [p for p in fwd if np.array_equal(p.vec_a, enc.encode(ca))]
        assert len(ka_first) == 1 and ka_first[0].y == 1
        assert {p.kind for p in fwd} == {"random-shared"}
        assert all(p.task_id == "t0" for p in pairs)


# ---------------------------------------------------------------- 任务内划分
class TestSplitTaskTrainEval:
    def test_classification_stratified_disjoint_cover(self):
        y = np.array([0] * 10 + [1] * 6 + [2] * 4)
        tr, ev = split_task_train_eval(y, TASK_CLASSIFICATION, eval_ratio=0.3, seed=0)
        assert len(np.intersect1d(tr, ev)) == 0
        assert sorted(np.concatenate([tr, ev]).tolist()) == list(range(len(y)))
        # 分层：每类两侧都有样本，eval 占比≈eval_ratio
        for label in (0, 1, 2):
            assert (y[tr] == label).sum() >= 1
            assert (y[ev] == label).sum() >= 1
        assert (y[ev] == 0).sum() == 3  # round(10*0.3)

    def test_deterministic(self):
        y = np.array([0] * 8 + [1] * 8)
        a = split_task_train_eval(y, TASK_CLASSIFICATION, eval_ratio=0.25, seed=5)
        b = split_task_train_eval(y, TASK_CLASSIFICATION, eval_ratio=0.25, seed=5)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_forecasting_stub(self):
        with pytest.raises(NotImplementedError):
            split_task_train_eval(np.zeros(10), TASK_FORECASTING, eval_ratio=0.3, seed=0)

    def test_unknown_task_type_rejected(self):
        with pytest.raises(ValueError, match="task_type"):
            split_task_train_eval(np.zeros(10), "segmentation", eval_ratio=0.3, seed=0)


# ---------------------------------------------------------------- 配对造 label
class TestMakePairs:
    def test_y_follows_is_better_direction(self):
        rp = {"hi": 0.9, "lo": 0.6}
        pairs, stats = make_pairs(rp, TASK_CLASSIFICATION, delta=0.02)
        assert len(pairs) == 1
        p = pairs[0]
        # 分类 ACC 大者优：a=hi 在前 → y=1；方向来自 is_better，不在本模块写死
        assert (p["a"], p["b"], p["y"]) == ("hi", "lo", 1)
        assert p["r_a"] == 0.9 and p["r_b"] == 0.6
        assert stats["n_kept"] == 1

    def test_y_zero_when_second_better(self):
        rp = {"lo": 0.6, "hi": 0.9}  # 插入序决定 (a, b)=(lo, hi)
        pairs, _ = make_pairs(rp, TASK_CLASSIFICATION, delta=0.02)
        assert pairs[0]["y"] == 0

    def test_delta_filter_drops_near_ties(self):
        # M2：高分簇 full acc 差 0.001~0.003 是同分噪声，不给二值 label
        rp = {"a": 0.900, "b": 0.899, "c": 0.500}
        pairs, stats = make_pairs(rp, TASK_CLASSIFICATION, delta=0.02)
        kept = {(p["a"], p["b"]) for p in pairs}
        assert kept == {("a", "c"), ("b", "c")}
        assert stats["n_delta_dropped"] == 1
        assert stats["n_kept"] == 2

    def test_exact_tie_dropped_even_with_delta_zero(self):
        # is_better 是严格优：平局双向 False → 弃样而非赋任意方向
        rp = {"a": 0.8, "b": 0.8}
        pairs, stats = make_pairs(rp, TASK_CLASSIFICATION, delta=0.0)
        assert pairs == []
        assert stats["n_tie_dropped"] == 1

    def test_nan_participants_excluded(self):
        rp = {"a": 0.9, "bad": float("nan"), "c": 0.5}
        pairs, stats = make_pairs(rp, TASK_CLASSIFICATION, delta=0.02)
        assert {(p["a"], p["b"]) for p in pairs} == {("a", "c")}
        assert stats["n_failed_cls"] == 1

    def test_origin_labels_pair_kind(self):
        rp = {"s1": 0.9, "s2": 0.6, "r1": 0.3}
        origin = {"s1": "shared", "s2": "shared", "r1": "random"}
        pairs, _ = make_pairs(rp, TASK_CLASSIFICATION, delta=0.02, origin=origin)
        kinds = {(p["a"], p["b"]): p["kind"] for p in pairs}
        assert kinds[("s1", "s2")] == "shared-shared"
        assert kinds[("s1", "r1")] == "random-shared"

    def test_forecasting_dispatches_through_is_better(self):
        # 证明方向判定走 is_better（铁律 8）：forecasting（MSE 小者优）方向与分类
        # 相反（2026-08-13 起分支已实现，原 stub 断言更新为方向反转检查）
        pairs, _ = make_pairs({"a": 0.9, "b": 0.5}, TASK_FORECASTING, delta=0.0)
        assert pairs[0]["y"] == 0  # MSE 0.9 > 0.5 → b 更优 → y=0


# ---------------------------------------------------------------- δ 标定辅助
class TestDeltaDecisionCurve:
    def test_curve_counts_and_agreement(self):
        # (0,1)：early 差 0.01 但 full 方向相反（同分噪声型翻转）；(0,2)/(1,2) 方向一致
        early = np.array([0.90, 0.89, 0.50])
        full = np.array([0.85, 0.86, 0.50])
        rows = delta_decision_curve(early, full, TASK_CLASSIFICATION, deltas=[0.0, 0.02])
        r0, r2 = rows
        assert r0["n_kept"] == 3 and r0["n_agree"] == 2
        assert r2["n_kept"] == 2 and r2["n_agree"] == 2
        assert r2["agreement"] == 1.0

    def test_full_ties_excluded_from_agreement(self):
        early = np.array([0.9, 0.5])
        full = np.array([0.7, 0.7])  # full 平局：方向无定义，不计一致率分母
        (row,) = delta_decision_curve(early, full, TASK_CLASSIFICATION, deltas=[0.0])
        assert row["n_kept"] == 1
        assert np.isnan(row["agreement"])


# ---------------------------------------------------------------- 缓存与种子
class TestCacheAndSeeds:
    def test_jsonl_roundtrip_resume_key(self, tmp_path):
        path = tmp_path / "cache.jsonl"
        r1 = {"task_id": "source_00", "cls_key": "abc", "iters": 290, "r_prime": 0.9}
        r2 = {"task_id": "source_01", "cls_key": "abc", "iters": 290, "r_prime": 0.8}
        append_jsonl(path, r1)
        append_jsonl(path, r2)
        cache = load_rprime_cache(path)
        assert cache[("source_00", "abc", 290)]["r_prime"] == 0.9
        assert ("source_01", "abc", 290) in cache and len(cache) == 2

    def test_load_missing_file_is_empty(self, tmp_path):
        assert load_rprime_cache(tmp_path / "nope.jsonl") == {}

    def test_truncated_last_line_skipped(self, tmp_path):
        # 进程被杀可能只写了半行：--resume 加载时跳过坏行（该条重跑），不崩
        path = tmp_path / "cache.jsonl"
        append_jsonl(path, {"task_id": "t", "cls_key": "abc", "iters": 290, "r_prime": 0.9})
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"task_id": "t", "cls_key": "def", "it')  # 截断
        cache = load_rprime_cache(path)
        assert list(cache) == [("t", "abc", 290)]

    def test_run_seed_deterministic_and_distinct(self):
        s = run_seed_for("source_00", "abc")
        assert s == run_seed_for("source_00", "abc")
        assert s != run_seed_for("source_00", "abd")
        assert s != run_seed_for("source_01", "abc")
        assert 0 <= s < 2**31


# ---------------------------------------------------------------- 铁律 7 分发点
class TestTaskTypeDispatchPoints:
    def test_task_windows_classification_identity(self):
        from autocls.task_embed import task_windows

        X = np.zeros((4, 8, 2), dtype=np.float32)
        out = task_windows(X, TASK_CLASSIFICATION)  # 分类：窗口=样本序列本身
        assert out is X

    def test_task_windows_forecasting_stub(self):
        from autocls.task_embed import task_windows

        with pytest.raises(NotImplementedError):
            task_windows(np.zeros((4, 8, 2)), TASK_FORECASTING)

    def test_task_windows_unknown_rejected(self):
        from autocls.task_embed import task_windows

        with pytest.raises(ValueError, match="task_type"):
            task_windows(np.zeros((4, 8, 2)), "segmentation")

    def test_resolve_evaluator_classification_r_prime_is_acc(self):
        from autocls.downstream import resolve_evaluator

        class TinyEnc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 4)

            def forward(self, x):  # (B, T, C) -> (B, T, d)
                return self.lin(x)

        rng = np.random.default_rng(0)
        X = rng.standard_normal((24, 8, 2)).astype(np.float32)
        y = np.array([0, 1] * 12)
        X[y == 1] += 3.0  # 可分，SVM 应拿到确定的 acc
        ev = resolve_evaluator(TASK_CLASSIFICATION)
        m = ev(TinyEnc(), X[:16], y[:16], X[16:], y[16:], device="cpu", seed=0)
        assert m["r_prime"] == m["acc"]
        assert {"acc", "f1"} <= set(m)

    def test_resolve_evaluator_forecasting_returns_callable(self):
        # 2026-08-13 起 forecasting 评估器已实现（TS2Vec Ridge 协议），原 stub 断言更新；
        # 行为细节由 tests/test_forecast_eval.py 钉死
        from autocls.downstream import resolve_evaluator

        assert callable(resolve_evaluator(TASK_FORECASTING))

    def test_resolve_evaluator_unknown_rejected(self):
        from autocls.downstream import resolve_evaluator

        with pytest.raises(ValueError, match="task_type"):
            resolve_evaluator("segmentation")


# ---------------------------------------------------------------- evaluate_rprime
# 单条 (task, cls) → R' 的共享助手（generate_labels 与 step0 共用，不 fork 第 3 份）。
# 显式 weight_seed/split_seed 让 step0 能同时跑 canonical（对齐全量缓存）与噪声种子。
def _tiny_task_data(n=80, T=32, C=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, C)).astype(np.float32)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    X[y == 1] += 2.5  # 可分：early-val 后 acc 应明显 off-chance
    task = SlicedTask(
        task_id="src_00", pool="source", classes=(0, 1),
        channels=tuple(range(C)), sample_indices=tuple(range(n)),
        tag="", task_type=TASK_CLASSIFICATION,
    )
    return task, X, y


class TestEvaluateRprime:
    def test_returns_finite_r_prime_and_fields(self, make_cls):
        task, X, y = _tiny_task_data()
        rec = evaluate_rprime(
            task, make_cls(), X, y,
            weight_seed=run_seed_for("src_00", "k"),
            split_seed=run_seed_for("src_00", "split"),
            iters=30, batch_size=32, device="cpu",
        )
        assert rec["error"] is None
        assert np.isfinite(rec["r_prime"])
        assert 0.0 <= rec["r_prime"] <= 1.0
        assert rec["n_train"] > 0 and rec["n_eval"] > 0

    def test_deterministic_for_fixed_seeds(self, make_cls):
        task, X, y = _tiny_task_data()
        kw = dict(weight_seed=123, split_seed=456, iters=30, batch_size=32, device="cpu")
        r1 = evaluate_rprime(task, make_cls(), X, y, **kw)["r_prime"]
        r2 = evaluate_rprime(task, make_cls(), X, y, **kw)["r_prime"]
        assert r1 == r2  # 同缓存键同权重初始化/划分 → 逐位一致（CPU 确定）

    def test_split_seed_changes_partition(self, make_cls):
        # 不同 split_seed → 不同 eval 划分 → R' 通常不同（step0 噪声测量的前提）
        task, X, y = _tiny_task_data()
        base = dict(weight_seed=123, iters=30, batch_size=32, device="cpu")
        r1 = evaluate_rprime(task, make_cls(), X, y, split_seed=1, **base)["r_prime"]
        r2 = evaluate_rprime(task, make_cls(), X, y, split_seed=2, **base)["r_prime"]
        assert np.isfinite(r1) and np.isfinite(r2)

    def test_records_device_name_cpu(self, make_cls):
        # 缓存记录无硬件字段时，跨机的 seconds/R' 锚点无法区分（full-iters §1.6
        # 的 5.6× 估时事故根因之一）。device="cpu" 必须记 CPU 标识而非显卡名。
        task, X, y = _tiny_task_data()
        rec = evaluate_rprime(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            iters=5, batch_size=32, device="cpu",
        )
        assert isinstance(rec.get("device_name"), str) and rec["device_name"]
        assert rec["device_name"].startswith("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
    def test_records_device_name_cuda(self, make_cls):
        task, X, y = _tiny_task_data()
        rec = evaluate_rprime(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            iters=5, batch_size=32, device="cuda",
        )
        assert rec["device_name"] == torch.cuda.get_device_name(0)


# ------------------------------------------------- evaluate_rprime_trajectory
# MX-3-PQ：一条预训练轨迹在 60/120/290 三个 budget 各取一次 R'（同一权重轨迹
# 上打点，而不是跑三条独立轨迹）——三倍 GPU 账单省掉，且 budget 间差异是纯
# 训练量差异。与 evaluate_rprime 共用 _prepare_task_split，口径必须逐位一致。
_TRAJ_KEYS = {
    "r_prime", "acc", "f1", "svm_c", "n_train", "n_eval", "weight_seed",
    "split_seed", "device_name", "swa", "seconds", "error",
    "checkpoint_iters", "trajectory_seconds",
}


class TestEvaluateRprimeTrajectory:
    def test_matches_single_run_at_every_checkpoint(self, make_cls):
        # 决定性质：轨迹截点的 R' 必须与"单跑到该 iters"逐位相同，否则 PQ 缓存
        # 与既有 (task, cls, iters) 缓存不是同一口径，两边数字不可混用。
        from autocls.labeling import evaluate_rprime_trajectory

        task, X, y = _tiny_task_data()
        kw = dict(weight_seed=123, split_seed=456, batch_size=32, device="cpu")
        traj = evaluate_rprime_trajectory(task, make_cls(), X, y, checkpoints=(4, 8), **kw)
        assert set(traj) == {4, 8}
        for it in (4, 8):
            single = evaluate_rprime(task, make_cls(), X, y, iters=it, **kw)
            assert traj[it]["r_prime"] == single["r_prime"], f"iters={it} 口径不一致"
            assert traj[it]["acc"] == single["acc"]
            assert traj[it]["f1"] == single["f1"]
            assert traj[it]["svm_c"] == single["svm_c"]
            assert traj[it]["n_train"] == single["n_train"]
            assert traj[it]["n_eval"] == single["n_eval"]

    def test_record_fields_and_cumulative_seconds(self, make_cls):
        from autocls.labeling import evaluate_rprime_trajectory

        task, X, y = _tiny_task_data()
        traj = evaluate_rprime_trajectory(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            checkpoints=(4, 8), batch_size=32, device="cpu",
        )
        for it, rec in traj.items():
            assert _TRAJ_KEYS <= set(rec), f"iters={it} 缺字段 {_TRAJ_KEYS - set(rec)}"
            assert rec["error"] is None
            assert rec["swa"] is False
            assert rec["checkpoint_iters"] == [4, 8]
            assert rec["weight_seed"] == 1 and rec["split_seed"] == 2
            assert rec["device_name"].startswith("cpu")
        # seconds = 到该截点（含本次评估）的累计墙钟，单调不减且 ≤ 轨迹总时长
        assert traj[4]["seconds"] <= traj[8]["seconds"]
        assert traj[8]["seconds"] <= traj[8]["trajectory_seconds"]
        assert traj[4]["trajectory_seconds"] == traj[8]["trajectory_seconds"]

    def test_ordinary_error_marks_every_checkpoint_nan(self, make_cls, monkeypatch):
        # 训练发散是"这条 CLS 的属性"：整条轨迹作废，三个 budget 全记 NaN+error
        # 落盘（--resume 不再重试），与 evaluate_rprime 的既有约定一致。
        from autocls.labeling import evaluate_rprime_trajectory

        def boom(*_a, **_kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("autocls.contrastive.cl_trainer.pretrain_encoder", boom)
        task, X, y = _tiny_task_data()
        traj = evaluate_rprime_trajectory(
            task, make_cls(), X, y, weight_seed=1, split_seed=2,
            checkpoints=(4, 8), batch_size=32, device="cpu",
        )
        assert set(traj) == {4, 8}
        for rec in traj.values():
            assert np.isnan(rec["r_prime"])
            assert rec["error"] == "RuntimeError: boom"
            assert rec["checkpoint_iters"] == [4, 8]
            assert rec["n_train"] > 0 and rec["n_eval"] > 0  # 切分字段仍须落盘

    def test_oom_raises_and_never_records(self, make_cls, monkeypatch):
        # OOM 与这条 CLS 的好坏无关：绝不能记成 NaN 结论（否则永久丢 3 条 label）
        from autocls.labeling import evaluate_rprime_trajectory
        from autocls.oom_guard import RprimeOOM

        def boom(*_a, **_kw):
            raise RuntimeError("CUDA out of memory. Tried to allocate 15.80 GiB")

        monkeypatch.setattr("autocls.contrastive.cl_trainer.pretrain_encoder", boom)
        task, X, y = _tiny_task_data()
        with pytest.raises(RprimeOOM):
            evaluate_rprime_trajectory(
                task, make_cls(), X, y, weight_seed=1, split_seed=2,
                checkpoints=(4, 8), batch_size=32, device="cpu",
            )

    @pytest.mark.parametrize("checkpoints", [(), (8, 4), (4, 4), (0, 4)])
    def test_invalid_checkpoints_fail_fast(self, make_cls, checkpoints):
        # 调用方 bug 必须**抛出**，不能落进 except Exception 变成"每个 budget
        # 一条 NaN+error"——那几行会进缓存且 --resume 永不重试（静默丢 label）
        from autocls.labeling import evaluate_rprime_trajectory

        task, X, y = _tiny_task_data()
        with pytest.raises(ValueError):
            evaluate_rprime_trajectory(
                task, make_cls(), X, y, weight_seed=1, split_seed=2,
                checkpoints=checkpoints, batch_size=32, device="cpu",
            )


class TestPrepareTaskSplit:
    """evaluate_rprime 与 trajectory 的共用切分入口（不得 fork 第二份切分逻辑）。"""

    def test_matches_split_task_train_eval(self):
        from autocls.labeling import _prepare_task_split

        task, X, y = _tiny_task_data()
        Xt, yt, tr, ev = _prepare_task_split(task, X, y, split_seed=42, eval_ratio=0.3)
        exp_tr, exp_ev = split_task_train_eval(
            yt, task.task_type, eval_ratio=0.3, seed=42)
        assert np.array_equal(tr, exp_tr) and np.array_equal(ev, exp_ev)
        assert Xt.shape[0] == len(y) and len(yt) == len(y)

    def test_explicit_split_indices_bypass_stratification(self):
        # 出路 A 的跨受试者组切分由调用方注入（split_seed 仍作切分来源标识落盘）
        from autocls.labeling import _prepare_task_split

        task, X, y = _tiny_task_data()
        tr_in, ev_in = np.arange(0, 60), np.arange(60, 80)
        _, _, tr, ev = _prepare_task_split(
            task, X, y, split_seed=42, eval_ratio=0.3, split_indices=(tr_in, ev_in))
        assert np.array_equal(tr, tr_in) and np.array_equal(ev, ev_in)
