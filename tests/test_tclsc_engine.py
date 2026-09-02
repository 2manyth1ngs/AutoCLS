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

from autocls.comparator.soft_labels import SoftTCLSCEngine, hard_to_soft
from autocls.comparator.tclsc_engine import (
    ClsRecord,
    Pair,
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

    def test_identity_assignment_rejected(self, engine_setup):
        # 来源解析到与接收方同一个窗口键 = E 实际没换：这种"消融"必须报错而不是
        # 静默等于 normal（否则消融无效应会被读成"模型不看 E"的证据）
        engine, records = engine_setup
        pairs = _all_directed_pairs(records)
        with pytest.raises(ValueError, match="同窗口键"):
            engine.evaluate(pairs, e_mode="assign", e_assignment={"t0": "t0", "t1": "t1"})
        with pytest.raises(ValueError, match="同窗口键"):   # 只要有一个恒等就报错
            engine.evaluate(pairs, e_mode="assign", e_assignment={"t0": "t1", "t1": "t1"})

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


# --------------------------------------------- (切片, 预算) 引擎任务：窗口键去重 + ctx

SLICES = ("s0", "s1")
BUDGETS = (60, 120)
CTX_DIM = len(BUDGETS)


def _tid(slice_id: str, b: int) -> str:
    """引擎任务 id = f"{切片}@{预算}"（组装知识在调用方，引擎只当它是字符串）。"""
    return f"{slice_id}@{b}"


def _ctx_vec(b: int) -> np.ndarray:
    v = np.zeros(CTX_DIM, dtype=np.float32)
    v[BUDGETS.index(b)] = 1.0
    return v


def _budget_of(tid: str) -> int:
    return int(tid.split("@")[1])


def _pq_engine(*, with_ctx: bool = True, cls_engine=TCLSCEngine, task_branch="E+ctx"):
    """2 切片 × 2 预算 = 4 个引擎任务；同切片的两个预算共用一份窗口表征。"""
    torch.manual_seed(0)
    model = TCLSC(cls_dim=CLS_DIM, set_encoder=TaskSetEncoder(dim_input=DIN, nz=NZ),
                  embedding_dim=12, fz=10, ctx_dim=CTX_DIM if with_ctx else 0,
                  task_branch=task_branch)
    rng = np.random.default_rng(0)
    task_windows = {"s0": rng.normal(size=(5, 4, DIN)).astype(np.float32),
                    "s1": rng.normal(size=(4, 4, DIN)).astype(np.float32)}
    tids = [_tid(s, b) for s in SLICES for b in BUDGETS]
    engine = cls_engine(
        model, task_windows, device="cpu", lr=1e-2, batch_size=64, seed=0,
        windows_key_of={t: t.split("@")[0] for t in tids},
        context_of=({t: _ctx_vec(_budget_of(t)) for t in tids} if with_ctx else None),
    )
    return engine, tids


@pytest.fixture
def pq_engine():
    return _pq_engine()


def _same_pairs(tids):
    """4 条完全同 CLS 向量、只差引擎任务的样本——预测差异只可能来自 E/ctx。"""
    rng = np.random.default_rng(7)
    va = rng.random(CLS_DIM).astype(np.float32)
    vb = rng.random(CLS_DIM).astype(np.float32)
    return [Pair(task_id=t, vec_a=va, vec_b=vb, y=1, kind="shared-shared") for t in tids]


def _pq_records(tids, *, flip_budget=None):
    """每引擎任务 8 条记录；flip_budget 给出时该预算优劣方向取反（ctx 依赖 toy）。"""
    rng = np.random.default_rng(3)
    vecs = [rng.random(CLS_DIM).astype(np.float32) for _ in range(8)]
    out = []
    for t in tids:
        sign = -1.0 if (flip_budget is not None and _budget_of(t) == flip_budget) else 1.0
        for i, vec in enumerate(vecs):
            out.append(ClsRecord(task_id=t, key=f"k{i}", origin="shared" if i < 4 else "random",
                                 vec=vec, r=float(sign * vec[0])))
    return out


def _spy_set_encode(model) -> list[int]:
    """记录每次 SetTF 前向的集合数（窗口键去重是否生效的直接证据）。"""
    calls: list[int] = []
    orig = model.set_encoder.set_encode

    def wrapped(X):
        calls.append(len(X))
        return orig(X)

    model.set_encoder.set_encode = wrapped
    return calls


class TestWindowsKeyDedup:
    def test_same_slice_budgets_share_E(self, pq_engine):
        engine, tids = pq_engine
        e_map = engine._build_e_map(sorted(tids), e_mode="normal", seed=0, e_assignment=None)
        for s in SLICES:   # 同切片的各预算共用一份窗口 → 逐位同一行 E
            assert torch.equal(e_map[_tid(s, 60)], e_map[_tid(s, 120)])
        assert not torch.equal(e_map["s0@60"], e_map["s1@60"])
        # 公开口径：ctx 归零后差异只可能来自 E，同切片两预算的预测必须一致
        p = engine.predict(_same_pairs(tids), ctx_mode="zero")
        assert p[0] == pytest.approx(p[1])
        assert p[2] == pytest.approx(p[3])

    def test_settf_runs_once_per_windows_key_per_batch(self, pq_engine):
        engine, tids = pq_engine
        calls = _spy_set_encode(engine.model)
        engine.train(_pq_records(tids), epochs=1, curriculum_epochs=0,
                     better_fn=better, delta=0.0)
        assert calls == [len(SLICES)], f"每 batch 应按窗口键去重（4 个引擎任务→2 次）: {calls}"

    def test_windows_key_missing_raises(self, pq_engine):
        engine, tids = pq_engine
        with pytest.raises(KeyError):
            engine.predict(_same_pairs(list(tids) + ["s9@60"]))


class TestEngineTaskAblations:
    """E 与 ctx 独立消融：E 按切片换、ctx 按预算换，互不牵连。"""

    def test_shuffle_permutes_E_by_slice_across_budgets(self, pq_engine):
        engine, tids = pq_engine
        norm = engine._build_e_map(sorted(tids), e_mode="normal", seed=0, e_assignment=None)
        shuf = engine._build_e_map(sorted(tids), e_mode="shuffle", seed=0, e_assignment=None)
        # 2 个窗口键 → derangement 唯一 = 对调；同切片两预算拿到同一份别家 E
        for a, b in (("s0", "s1"), ("s1", "s0")):
            for budget in BUDGETS:
                assert torch.equal(shuf[_tid(a, budget)], norm[_tid(b, 60)])
        # ctx 不随 E 走：预测仍按各自预算（base 的两个预算不同值，故断言非平凡）
        pairs = _same_pairs(tids)
        base = engine.predict(pairs)
        assert not np.isclose(base[0], base[1]), "两预算的 ctx 必须已经把预测分开"
        assert not np.isclose(base[0], base[2]), "两切片的 E 必须已经把预测分开"
        np.testing.assert_allclose(engine.predict(pairs, e_mode="shuffle", seed=0),
                                   base[[2, 3, 0, 1]], atol=1e-6)

    def test_assign_keeps_own_ctx(self, pq_engine):
        engine, tids = pq_engine
        pairs = _same_pairs(tids)
        base = engine.predict(pairs)
        e_swap = {t: _tid("s1" if t.startswith("s0") else "s0", _budget_of(t)) for t in tids}
        norm = engine._build_e_map(sorted(tids), e_mode="normal", seed=0, e_assignment=None)
        assigned = engine._build_e_map(sorted(tids), e_mode="assign", seed=0,
                                       e_assignment=e_swap)
        for t in tids:
            assert torch.equal(assigned[t], norm[e_swap[t]])
        p_e = engine.predict(pairs, e_mode="assign", e_assignment=e_swap)
        np.testing.assert_allclose(p_e, base[[2, 3, 0, 1]], atol=1e-6)  # 换 E、ctx 不动
        ctx_swap = {t: _tid(t.split("@")[0], BUDGETS[1 - BUDGETS.index(_budget_of(t))])
                    for t in tids}
        p_c = engine.predict(pairs, ctx_mode="assign", ctx_assignment=ctx_swap)
        np.testing.assert_allclose(p_c, base[[1, 0, 3, 2]], atol=1e-6)  # 换 ctx、E 不动
        assert not np.allclose(p_e, p_c)

    def test_ctx_zero_changes_predictions(self, pq_engine):
        engine, tids = pq_engine
        pairs = _same_pairs(tids)
        base = engine.predict(pairs)
        zero = engine.predict(pairs, ctx_mode="zero")
        assert not np.allclose(zero, base)
        assert zero[0] == pytest.approx(zero[1])  # ctx 归零后两预算等价

    def test_evaluate_reports_ctx_mode(self, pq_engine):
        engine, tids = pq_engine
        pairs = _same_pairs(tids)
        res = engine.evaluate(pairs, ctx_mode="zero", return_correct=True)
        assert res["ctx_mode"] == "zero" and res["e_mode"] == "normal"
        assert len(res["correct"]) == len(pairs)

    def test_ctx_mode_validation(self, pq_engine, engine_setup):
        engine, tids = pq_engine
        pairs = _same_pairs(tids)
        with pytest.raises(ValueError):                       # 未知档
            engine.predict(pairs, ctx_mode="nope")
        with pytest.raises(ValueError):                       # assign 档必须给映射
            engine.predict(pairs, ctx_mode="assign")
        with pytest.raises(ValueError):                       # 非 assign 档不得带映射
            engine.predict(pairs, ctx_mode="zero", ctx_assignment={t: t for t in tids})
        with pytest.raises(ValueError):                       # 缺任务映射
            engine.predict(pairs, ctx_mode="assign", ctx_assignment={tids[0]: tids[1]})
        with pytest.raises(ValueError):                       # 指派目标无 ctx
            engine.predict(pairs, ctx_mode="assign",
                           ctx_assignment={t: "s9@60" for t in tids})
        plain, records = engine_setup                          # 无 context_of 的引擎
        with pytest.raises(ValueError):
            plain.evaluate(_all_directed_pairs(records), ctx_mode="zero")

    def test_e_ablation_rejected_when_model_ignores_E(self):
        # task_branch=ctx/none 的模型根本不吃 E：非 normal 档必须报错，否则会返回
        # 一个"贴着 shuffle/zero 标签、值与 normal 逐位相同"的假消融结果
        for branch in ("ctx", "none"):
            engine, tids = _pq_engine(task_branch=branch)
            pairs = _same_pairs(tids)
            assert engine.predict(pairs).shape == (len(tids),)   # normal 档照常可跑
            for mode in ("shuffle", "zero"):
                with pytest.raises(ValueError, match="不吃 E"):
                    engine.predict(pairs, e_mode=mode)
            with pytest.raises(ValueError, match="不吃 E"):
                engine.predict(pairs, e_mode="assign",
                               e_assignment={t: _tid("s1", _budget_of(t)) for t in tids})

    def test_same_slice_e_assignment_rejected(self, pq_engine):
        # pq_common.budget_assignment（同切片、换预算）就是这个形状：误传进
        # e_assignment 会解析到同一个窗口键 → 零效应"消融"，必须报错
        engine, tids = pq_engine
        budget_swap = {t: _tid(t.split("@")[0], BUDGETS[1 - BUDGETS.index(_budget_of(t))])
                       for t in tids}
        with pytest.raises(ValueError, match="同窗口键"):
            engine.predict(_same_pairs(tids), e_mode="assign", e_assignment=budget_swap)
        # 同一张表走 ctx_assignment 才是它的正确用法（换预算、E 不动）
        assert engine.predict(_same_pairs(tids), ctx_mode="assign",
                              ctx_assignment=budget_swap).shape == (len(tids),)

    def test_shuffle_needs_two_windows_keys(self, pq_engine):
        # 同切片的两个引擎任务共用一个窗口键 → 唯一键只有 1 个，无法错位
        engine, tids = pq_engine
        one_slice = [p for p in _same_pairs(tids) if p.task_id.startswith("s0")]
        with pytest.raises(ValueError, match="窗口键"):
            engine.predict(one_slice, e_mode="shuffle", seed=0)

    def test_context_dim_must_match_model(self):
        torch.manual_seed(0)
        model = TCLSC(cls_dim=CLS_DIM, set_encoder=TaskSetEncoder(dim_input=DIN, nz=NZ),
                      embedding_dim=12, fz=10, ctx_dim=CTX_DIM)
        windows = {"s0": np.zeros((3, 4, DIN), dtype=np.float32)}
        with pytest.raises(ValueError):   # 模型要 ctx 却没给 context_of
            TCLSCEngine(model, windows)
        with pytest.raises(ValueError):   # 维度不符
            TCLSCEngine(model, windows,
                        context_of={"s0@60": np.zeros(CTX_DIM + 1, dtype=np.float32)})


class TestSoftEngineWithContext:
    """软 label 引擎共用 _encode_batch：ctx 必须一路带到前向（否则两预算互相打架）。"""

    def _toy(self, with_ctx):
        engine, tids = _pq_engine(with_ctx=with_ctx, cls_engine=SoftTCLSCEngine)
        # 预算 120 的优劣方向取反：ctx 盲模型无法同时拟合两个预算
        return engine, _pq_records(tids, flip_budget=120)

    @staticmethod
    def _sampler(records):
        def sampler(epoch, rng):
            return hard_to_soft(dynamic_pairs(records, better, delta=0.0, rng=rng))
        return sampler

    def test_train_soft_uses_ctx(self):
        engine, records = self._toy(with_ctx=True)
        hist = engine.train_soft(self._sampler(records), epochs=250)
        assert len(hist) == 250 and all(np.isfinite(h["loss"]) for h in hist)
        assert hist[-1]["loss"] < hist[0]["loss"]
        assert hist[-1]["loss"] < 0.35, f"看得见 ctx 才能压到 ln2 以下: {hist[-1]['loss']:.3f}"

    def test_ctx_blind_engine_stalls_at_chance(self):
        engine, records = self._toy(with_ctx=False)
        hist = engine.train_soft(self._sampler(records), epochs=250)
        assert hist[-1]["loss"] > 0.6, f"矛盾 toy 上 ctx 盲模型不该学出来: {hist[-1]['loss']:.3f}"
