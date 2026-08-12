"""目标池 zero-shot 评估的单测（铁律 5：先测后写）。

覆盖三块：
1. autocls/search/rank.py 的纯聚合数学（对称化 / Round-Robin 胜场 / 多数票 /
   Borda 聚合与并列破序确定性）——比较器排名的可测核心；
2. scripts/5_target_zero_shot.py 的可测助手：候选池构造（10 shared + 20 random
   + GGS = 31 唯一键、random 前缀稳定）、epoch↔iters 换算（任务级网格与全量换算）；
3. 目标池口径：UCI-HAR 注册项带切片参数、源加载路径 fail-fast（铁律 6）。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, TargetPoolLeakageError, build_slice_kwargs, load_registry,
    load_source_dataset,
)
from autocls.search.rank import (  # noqa: E402
    borda_ranking, pair_vote_matrix, symmetrize, wins_from_scores,
)

_tz = importlib.import_module("scripts.5_target_zero_shot")


# ---------------------------------------------------------------------------
# rank.py 纯数学
# ---------------------------------------------------------------------------


def test_symmetrize_combines_both_directions():
    """S[i,j] = (P[i,j] + 1 - P[j,i]) / 2，且 S + S^T = 1（对角除外）。"""
    P = np.array([
        [np.nan, 0.8, 0.6],
        [0.4, np.nan, 0.9],
        [0.1, 0.3, np.nan],
    ])
    S = symmetrize(P)
    assert S[0, 1] == pytest.approx((0.8 + 1 - 0.4) / 2)
    assert S[1, 0] == pytest.approx((0.4 + 1 - 0.8) / 2)
    off = ~np.eye(3, dtype=bool)
    assert np.allclose((S + S.T)[off], 1.0)
    assert np.all(np.isnan(np.diag(S)))


def test_wins_from_scores_round_robin():
    """胜场 = 对称化得分 > 0.5 的对手数；恰 0.5 双方都不得分。"""
    S = np.array([
        [np.nan, 0.9, 0.5],
        [0.1, np.nan, 0.7],
        [0.5, 0.3, np.nan],
    ])
    assert wins_from_scores(S).tolist() == [1, 1, 0]


def test_pair_vote_matrix_majority():
    """3 个种子逐无序对多数票；平票用平均 S 破。"""
    S_win = np.array([[np.nan, 0.8], [0.2, np.nan]])   # 0 胜 1
    S_lose = np.array([[np.nan, 0.3], [0.7, np.nan]])  # 1 胜 0
    # 2:1 多数 → 0 胜
    V = pair_vote_matrix([S_win, S_win, S_lose])
    assert V[0, 1] == 1 and V[1, 0] == 0
    # 1:1（一票弃权 S=0.5）→ 平均 S = (0.8+0.5+0.3)/3 > 0.5 → 0 胜
    S_abstain = np.array([[np.nan, 0.5], [0.5, np.nan]])
    V2 = pair_vote_matrix([S_win, S_abstain, S_lose])
    assert V2[0, 1] == 1 and V2[1, 0] == 0


def test_borda_ranking_aggregates_and_breaks_ties_deterministically():
    """跨任务胜场求和降序；并列先按 tiebreak 得分降序、再按 key 升序。"""
    wins_t1 = np.array([2, 1, 0])
    wins_t2 = np.array([0, 1, 2])       # 合计 [2, 2, 2] 全并列
    scores = np.array([0.5, 0.7, 0.5])  # 1 的得分最高；0 与 2 再并列 → key 升序
    keys = ["bbb", "ccc", "aaa"]
    order = borda_ranking([wins_t1, wins_t2], scores, keys)
    assert [keys[i] for i in order] == ["ccc", "aaa", "bbb"]
    # 输入顺序无关：重复调用逐位一致（确定性）
    assert order == borda_ranking([wins_t1, wins_t2], scores, keys)


def test_borda_ranking_prefers_total_wins_over_score():
    wins = np.array([5, 3, 0])
    scores = np.array([0.1, 0.99, 0.99])
    order = borda_ranking([wins], scores, ["a", "b", "c"])
    assert order[0] == 0


# ---------------------------------------------------------------------------
# 脚本助手：候选池 / epoch↔iters
# ---------------------------------------------------------------------------


def test_candidate_pool_31_unique_with_ggs():
    pool = _tz.build_candidate_pool()
    keys = [k for _o, _c, k in pool]
    assert len(pool) == 31
    assert len(set(keys)) == 31, "候选键必须唯一"
    origins = [o for o, _c, _k in pool]
    assert origins.count("shared") == 10
    assert origins.count("random") == 20
    assert origins.count("ggs") == 1
    # shared 10 与 labels_workpoint 目录逐键一致（与源池 label 可比）
    import json
    cat = json.loads((ROOT / "results/labels_workpoint/cls_catalog.json")
                     .read_text(encoding="utf-8"))
    assert [k for o, _c, k in pool if o == "shared"] == \
        [d["cls_key"] for d in cat["shared"]]


def test_candidate_pool_random_prefix_stable():
    """逐个接受制：加大 n 不改变前 n' 个 random 的取值（缓存前缀稳定）。"""
    r10 = [k for o, _c, k in _tz.build_candidate_pool(n_random=10) if o == "random"]
    r20 = [k for o, _c, k in _tz.build_candidate_pool(n_random=20) if o == "random"]
    assert r20[:10] == r10


def test_epoch_iters_roundtrip():
    """任务级网格 {1,2,5,10,20}ep @ spe → iters；全量换算用全量 spe。"""
    assert _tz.grid_iters(spe=6) == [6, 12, 30, 60, 120]
    # 全量 UCI-HAR train 7352 → ⌈7352/128⌉ = 58 steps/ep（M1 口径：40ep=2320）
    assert _tz.full_iters_for_ep(10, n_train=7352) == 580
    assert _tz.full_iters_for_ep(40, n_train=7352) == 2320


# ---------------------------------------------------------------------------
# 目标池口径（铁律 6）
# ---------------------------------------------------------------------------


def test_uci_har_target_entry_has_slice_params():
    reg = load_registry()
    e = reg.get("UCI-HAR")
    assert e.role == ROLE_TARGET
    assert e.n_channels == 9 and e.n_classes == 6
    assert e.n_classes_range == (3, 6) and e.planned_tasks == 8
    kw = build_slice_kwargs(e, reg)   # 能建通用切片参数（切任务走唯一真源）
    assert kw["n_channels_range"] == (5, 9)
    assert kw["n_samples"] == 1024 and kw["seed"] == 0


def test_uci_har_source_load_fails_fast():
    with pytest.raises(TargetPoolLeakageError):
        load_source_dataset("UCI-HAR")


# ---------------------------------------------------------------------------
# 出路 A：跨受试者协议（2026-08-12 拍板）
# ---------------------------------------------------------------------------

from autocls.labeling import (  # noqa: E402
    evaluate_rprime, split_train_eval_by_group,
)


def _toy_groups():
    """6 组 × 各 10 样本、3 类均匀分布 → 组切分的可控夹具。"""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 10)
    y = np.tile(np.array([0, 1, 2]), 20)
    return y, groups, rng


def test_group_split_keeps_groups_whole_and_disjoint():
    y, groups, _ = _toy_groups()
    tr, ev = split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=0)
    assert len(np.intersect1d(tr, ev)) == 0
    assert len(tr) + len(ev) == len(y)
    # 整组进一侧：任一组不得横跨两侧
    assert not (set(groups[tr].tolist()) & set(groups[ev].tolist()))
    # 首次跨越目标即停：eval ≥ 目标 且 去掉最后一组就 < 目标（不过量吞组）
    assert len(ev) >= 0.3 * len(y)
    assert len(ev) - 10 < 0.3 * len(y)   # 夹具组大小恒 10


def test_group_split_covers_all_classes_both_sides():
    y, groups, _ = _toy_groups()
    tr, ev = split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=0)
    assert set(y[tr].tolist()) == set(y[ev].tolist()) == {0, 1, 2}


def test_group_split_deterministic_and_seed_sensitive():
    y, groups, _ = _toy_groups()
    a = split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=7)
    b = split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    outs = {tuple(split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=s)[1])
            for s in range(6)}
    assert len(outs) > 1, "不同 seed 应产生不同留出组"


def test_group_split_fails_when_class_isolated_in_one_group():
    """某类只存在于单一组 → 该组无论在哪一侧都缺类，必须 fail-fast。"""
    groups = np.repeat(np.arange(3), 10)
    y = np.zeros(30, dtype=int)
    y[groups == 2] = 1   # 类 1 只在组 2
    with pytest.raises(ValueError):
        split_train_eval_by_group(y, groups, eval_ratio=0.3, seed=0)


def test_evaluate_rprime_honors_split_indices():
    """split_indices 注入：n_train/n_eval 必须按注入索引计，内部分层切分被绕过。"""
    from autocls.data.task_slicer import SlicedTask

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 32, 2)).astype(np.float32)
    y = np.tile(np.array([0, 1]), 30)
    task = SlicedTask(task_id="toy_00", pool="toy", classes=(0, 1), channels=(0, 1),
                      sample_indices=tuple(range(60)))
    tr = np.arange(0, 40)
    ev = np.arange(40, 60)
    rec = evaluate_rprime(task, _tz._m1.GGS, X, y, weight_seed=0, split_seed=0,
                          iters=0, split_indices=(tr, ev), device="cpu")
    assert rec["n_train"] == 40 and rec["n_eval"] == 20
    assert rec["r_prime"] is not None
