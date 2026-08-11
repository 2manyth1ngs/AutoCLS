"""T-CLSC 训练引擎单测（TECHNICAL_DESIGN.md 模块 6；改自 AutoCTS++ 训练机制）。

覆盖：动态配对（AHC_DataLoader_linear 语义 + δ 过滤）、curriculum（shared 先）、
双向评估对、E 消融（zero / shuffle=任务错位 derangement）、toy 数据 BCE 下降。

引擎是任务无关模块（铁律 7）：优劣方向经注入的 better_fn 判定，测试侧自行
绑定 `r1 > r2`；正式脚本绑定 labeling/task_types.is_better（铁律 8）。
为便于校验不变量，测试把 (r, 记录序号, 任务序号) 编进 CLS 向量前三维。
"""
import numpy as np
import pytest
import torch

from autocls.comparator.tclsc_engine import (
    ClsRecord,
    TCLSCEngine,
    derangement,
    dynamic_pairs,
    with_reversed,
)
from autocls.comparator.tclsc_model import TCLSC
from autocls.task_embed.set_transformer import TaskSetEncoder

CLS_DIM = 5
DIN = 6
NZ = 8


def better(r1: float, r2: float) -> bool:
    """测试侧方向绑定（分类语义：大者优）。引擎自身不得假设方向。"""
    return r1 > r2


def _rec(task_i: int, rec_i: int, r: float, origin: str) -> ClsRecord:
    vec = np.zeros(CLS_DIM, dtype=np.float32)
    vec[0], vec[1], vec[2] = r, rec_i, task_i  # 可从 pair 向量反查校验
    return ClsRecord(task_id=f"t{task_i}", key=f"k{task_i}_{rec_i}", origin=origin, vec=vec, r=r)


@pytest.fixture
def records() -> list[ClsRecord]:
    return [
        _rec(0, 0, 0.90, "shared"),
        _rec(0, 1, 0.50, "shared"),
        _rec(0, 2, 0.88, "random"),
        _rec(1, 3, 0.30, "shared"),
        _rec(1, 4, 0.70, "shared"),
        _rec(1, 5, 0.75, "random"),
    ]


class TestDynamicPairs:
    def test_invariants_across_seeds(self, records):
        for seed in range(8):
            pairs = dynamic_pairs(records, better, delta=0.05,
                                  rng=np.random.default_rng(seed))
            assert pairs, "非退化输入应产出配对"
            for p in pairs:
                assert p.vec_a[2] == p.vec_b[2], "配对必须来自同一任务"
                assert p.vec_a[1] != p.vec_b[1], "不得自配对"
                assert abs(p.vec_a[0] - p.vec_b[0]) >= 0.05, "δ 内近平局必须弃样"
                assert p.y == int(better(float(p.vec_a[0]), float(p.vec_b[0])))
                assert p.kind in ("shared-shared", "random-shared", "random-random")

    def test_delta_drops_near_ties(self):
        # 仅两条记录且 |Δr|<δ：任何种子都配不出样本
        two = [_rec(0, 0, 0.90, "shared"), _rec(0, 1, 0.88, "random")]
        for seed in range(8):
            assert dynamic_pairs(two, better, delta=0.05,
                                 rng=np.random.default_rng(seed)) == []

    def test_origin_filter_for_curriculum(self, records):
        # curriculum 第一阶段只用 shared 记录（AutoCTS++ 3.2.4 先易后难）。
        # 每任务仅 2 条 shared 时，某些种子的乱序副本恰好全对位（自配对全部
        # 跳过）→ 空配对是动态配对的固有行为，故跨种子聚合断言
        all_pairs = [p for seed in range(8)
                     for p in dynamic_pairs(records, better, delta=0.0,
                                            rng=np.random.default_rng(seed),
                                            origins=("shared",))]
        assert all_pairs, "8 个种子聚合后应产出配对"
        assert all(p.kind == "shared-shared" for p in all_pairs)

    def test_exact_tie_dropped_when_delta_zero(self):
        two = [_rec(0, 0, 0.8, "shared"), _rec(0, 1, 0.8, "shared")]
        for seed in range(8):
            assert dynamic_pairs(two, better, delta=0.0,
                                 rng=np.random.default_rng(seed)) == []


class TestPerTaskDelta:
    """delta 可以是 {task_id: δ} 映射（决定性实验：per-dataset δ 随任务查表；
    引擎不知道数据集概念，只做 per-task 查表——铁律 7）。"""

    def test_mapping_respected_per_task(self, records):
        # t0 给大 δ（0.3）：|0.90-0.88|、|0.90-0.50| 中仅 0.40 差值存活；
        # t1 给小 δ（0.01）：全部差值存活
        deltas = {"t0": 0.3, "t1": 0.01}
        for seed in range(8):
            for p in dynamic_pairs(records, better, delta=deltas,
                                   rng=np.random.default_rng(seed)):
                tid = p.task_id
                assert abs(p.vec_a[0] - p.vec_b[0]) >= deltas[tid], \
                    f"{tid} 的 δ={deltas[tid]} 未生效"

    def test_mapping_missing_task_raises(self, records):
        with pytest.raises(KeyError):
            dynamic_pairs(records, better, delta={"t0": 0.05},
                          rng=np.random.default_rng(0))

    def test_scalar_still_works(self, records):
        pairs = dynamic_pairs(records, better, delta=0.05,
                              rng=np.random.default_rng(0))
        assert all(abs(p.vec_a[0] - p.vec_b[0]) >= 0.05 for p in pairs)


class TestWithReversed:
    def test_doubles_flips_and_swaps(self, records):
        pairs = dynamic_pairs(records, better, delta=0.0,
                              rng=np.random.default_rng(0))
        both = with_reversed(pairs)
        assert len(both) == 2 * len(pairs)
        for p, q in zip(pairs, both[len(pairs):]):
            assert q.task_id == p.task_id and q.kind == p.kind
            assert q.y == 1 - p.y
            np.testing.assert_array_equal(q.vec_a, p.vec_b)
            np.testing.assert_array_equal(q.vec_b, p.vec_a)


class TestDerangement:
    def test_no_fixed_points(self):
        for n in (2, 3, 5, 8):
            for seed in range(5):
                p = derangement(n, np.random.default_rng(seed))
                assert sorted(p) == list(range(n))
                assert not np.any(p == np.arange(n)), "错位排列不得有不动点"

    def test_rejects_singleton(self):
        with pytest.raises(ValueError):
            derangement(1, np.random.default_rng(0))


@pytest.fixture
def engine_setup():
    torch.manual_seed(0)
    model = TCLSC(cls_dim=CLS_DIM,
                  set_encoder=TaskSetEncoder(dim_input=DIN, nz=NZ),
                  embedding_dim=12, fz=10)
    rng = np.random.default_rng(0)
    task_windows = {
        "t0": rng.normal(size=(5, 4, DIN)).astype(np.float32),
        "t1": rng.normal(size=(4, 4, DIN)).astype(np.float32),
    }
    # 每任务 8 个 CLS，r 仅由向量首维决定（可从 CLS 学到的合成规则）
    records = []
    for ti, tid in enumerate(("t0", "t1")):
        for i in range(8):
            vec = rng.random(CLS_DIM).astype(np.float32)
            records.append(ClsRecord(task_id=tid, key=f"k{ti}_{i}",
                                     origin="shared" if i < 4 else "random",
                                     vec=vec, r=float(vec[0])))
    engine = TCLSCEngine(model, task_windows, device="cpu",
                         lr=5e-3, batch_size=8, seed=0)
    return engine, records


def _all_directed_pairs(records):
    """测试用静态全配对（双向）——独立实现，交叉校验引擎训练效果。"""
    from itertools import combinations
    by_task = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    pairs = []
    for recs in by_task.values():
        for a, b in combinations(recs, 2):
            if a.r == b.r:
                continue
            from autocls.comparator.tclsc_engine import Pair
            kind = "-".join(sorted((a.origin, b.origin)))
            pairs.append(Pair(task_id=a.task_id, vec_a=a.vec, vec_b=b.vec,
                              y=int(better(a.r, b.r)), kind=kind))
    return with_reversed(pairs)


class TestEngineTrainEval:
    def test_bce_decreases_and_learns(self, engine_setup):
        engine, records = engine_setup
        history = engine.train(records, epochs=40, curriculum_epochs=5,
                               better_fn=better, delta=0.0)
        assert len(history) == 40
        assert all(h["phase"] == "shared" for h in history[:5])
        assert all(h["phase"] == "all" for h in history[5:])
        first5 = np.mean([h["loss"] for h in history[:5]])
        last5 = np.mean([h["loss"] for h in history[-5:]])
        assert last5 < first5, f"BCE 未下降: {first5:.4f} → {last5:.4f}"

        res = engine.evaluate(_all_directed_pairs(records))
        assert res["n"] == len(_all_directed_pairs(records))
        assert res["acc"] > 0.7, f"训练集配对准确率过低: {res['acc']:.3f}"

    def test_evaluate_e_modes(self, engine_setup):
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        base = engine.evaluate(pairs)
        zero = engine.evaluate(pairs, e_mode="zero")
        shuf = engine.evaluate(pairs, e_mode="shuffle", seed=1)
        for res in (base, zero, shuf):
            assert set(res) >= {"acc", "loss", "n"}
            assert 0.0 <= res["acc"] <= 1.0 and np.isfinite(res["loss"])
        # E 参与前向 → 置零必须改变输出统计（消融有意义的前提）
        assert zero["loss"] != base["loss"]

    def test_evaluate_rejects_unknown_mode(self, engine_setup):
        engine, records = engine_setup
        with pytest.raises(ValueError):
            engine.evaluate(_all_directed_pairs(records), e_mode="nope")


class TestEvaluateReturnCorrect:
    """return_correct=True：返回逐对正误向量（McNemar 配对检验要用；
    聚合 acc 必须与向量均值逐位一致）。"""

    def test_correct_vector_consistent_with_acc(self, engine_setup):
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        res = engine.evaluate(pairs, return_correct=True)
        assert len(res["correct"]) == len(pairs)
        assert res["acc"] == pytest.approx(float(np.mean(res["correct"])))

    def test_default_omits_vector(self, engine_setup):
        engine, records = engine_setup
        res = engine.evaluate(_all_directed_pairs(records))
        assert "correct" not in res

    def test_works_with_ablation_modes(self, engine_setup):
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        res = engine.evaluate(pairs, e_mode="zero", return_correct=True)
        assert len(res["correct"]) == len(pairs)


class TestEvaluateAssign:
    """e_mode='assign'：调用方显式指定 每任务→取谁的 E（决定性实验的两档 shuffle：
    within=数据集内 derangement / cross=跨数据集对调——数据集知识留在调用方，
    引擎保持任务无关，铁律 7）。"""

    def test_identity_assignment_equals_normal(self, engine_setup):
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        base = engine.evaluate(pairs)
        ident = engine.evaluate(pairs, e_mode="assign",
                                e_assignment={"t0": "t0", "t1": "t1"})
        assert ident["acc"] == base["acc"] and ident["loss"] == pytest.approx(base["loss"])

    def test_swap_assignment_equals_two_task_derangement(self, engine_setup):
        # 2 任务时 derangement 唯一 = 对调 → assign 对调必须与 shuffle 档逐位一致
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        swap = engine.evaluate(pairs, e_mode="assign",
                               e_assignment={"t0": "t1", "t1": "t0"})
        shuf = engine.evaluate(pairs, e_mode="shuffle", seed=0)
        assert swap["acc"] == shuf["acc"] and swap["loss"] == pytest.approx(shuf["loss"])
        base = engine.evaluate(pairs)
        assert swap["loss"] != base["loss"], "E 参与前向 → 对调必须改变输出统计"

    def test_assignment_may_point_outside_test_tasks(self, engine_setup):
        # cross 档会把留出任务的 E 指到训练侧任务——目标只需在 task_windows 里
        engine, records = engine_setup
        pairs = [p for p in _all_directed_pairs(records) if p.task_id == "t0"]
        res = engine.evaluate(pairs, e_mode="assign", e_assignment={"t0": "t1"})
        assert 0.0 <= res["acc"] <= 1.0 and np.isfinite(res["loss"])

    def test_assign_validation(self, engine_setup):
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        with pytest.raises(ValueError):   # 缺映射
            engine.evaluate(pairs, e_mode="assign", e_assignment={"t0": "t0"})
        with pytest.raises(ValueError):   # 目标任务无特征
            engine.evaluate(pairs, e_mode="assign",
                            e_assignment={"t0": "t9", "t1": "t0"})
        with pytest.raises(ValueError):   # assign 档必须给映射
            engine.evaluate(pairs, e_mode="assign")
        with pytest.raises(ValueError):   # 非 assign 档不得带映射
            engine.evaluate(pairs, e_mode="zero", e_assignment={"t0": "t0", "t1": "t1"})
