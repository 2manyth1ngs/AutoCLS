"""源池规模曲线（scripts/44_source_size_curve.py）纯函数测试——先于实现（铁律 5）。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib
from dataclasses import asdict

import numpy as np
import pytest

from autocls.labeling import ORIGIN_SHARED, cls_key
from autocls.space.cls_space import CLSSpace

g44 = importlib.import_module("scripts.44_source_size_curve")
B3 = (60, 120, 290)


def test_subsets_enumerate_all_combinations_sorted():
    src = ["d3", "d1", "d2", "d4", "d5", "d6", "d7"]
    s3 = g44.subsets(src, 3)
    assert len(s3) == 35 and all(len(t) == 3 for t in s3)
    assert all(tuple(sorted(t)) == t for t in s3) and s3 == sorted(s3)
    assert g44.subsets(src, 7) == [tuple(sorted(src))]
    assert len({t for t in s3}) == 35


def _rows(space, rng, datasets, n_slices=2, n_shared=4, n_random=3):
    shared = [space.sample_random(rng) for _ in range(n_shared)]
    rows = []
    for d in datasets:
        for s in range(n_slices):
            rnd = [space.sample_random(rng) for _ in range(n_random)]
            for origin, cfgs in ((ORIGIN_SHARED, shared), ("random", rnd)):
                for c in cfgs:
                    for b in B3:
                        rows.append({"dataset": d, "task_id": f"{d}_{s:02d}", "cls_key": cls_key(c),
                                     "cls": asdict(c), "origin": origin, "iters": b,
                                     "r_prime": float(rng.uniform(0.2, 0.9)), "seconds": float(b)})
    return g44._g36.zscore_within(rows), [cls_key(c) for c in shared]


def test_prior_from_sources_uses_only_shared_rows_of_named_sources():
    space = CLSSpace.from_yaml()
    rows, shared = _rows(space, np.random.default_rng(0), ["A", "B", "C"])
    prior = g44.prior_from_sources(rows, ("A", "C"))
    assert set(prior) == set(shared)
    for k in shared:
        z = [r["z"] for r in rows if r["cls_key"] == k and r["dataset"] in ("A", "C") and r["origin"] == ORIGIN_SHARED]
        assert prior[k] == pytest.approx(float(np.mean(z)))
    assert g44.prior_from_sources(rows, ()) == {}


def test_eval_prior_on_target_metrics_hand_computed():
    space = CLSSpace.from_yaml()
    rng = np.random.default_rng(1)
    rows, shared = _rows(space, rng, ["T"], n_slices=1, n_shared=4, n_random=4)
    # 先验：shared[0] 最高 … shared[3] 最低；random 无先验，按 key 升序垫底（36 号 rank_by 语义）
    prior = {k: float(3 - i) for i, k in enumerate(shared)}
    inst = g44.eval_prior_on_target(prior, rows, budgets=B3, ks=(2, 1, 1))
    assert len(inst) == 3 and {r["budget"] for r in inst} == set(B3)
    for r in inst:
        b = r["budget"]
        truth = {x["cls_key"]: x["r_prime"] for x in rows if int(x["iters"]) == b}
        oracle = max(truth.values())
        oracle_key = max(truth, key=truth.get)
        ranked = g44._g36.rank_by(prior, sorted(truth))
        assert ranked[:4] == shared and ranked[4:] == sorted(k for k in truth if k not in shared)
        assert r["oracle"] == pytest.approx(oracle) and r["n_candidates"] == 8 and r["oracle_key"] == oracle_key
        assert r["regret1"] == pytest.approx(oracle - truth[shared[0]])
        assert r["regret3"] == pytest.approx(oracle - max(truth[k] for k in shared[:3]))
        assert r["in_top6"] == (oracle_key in ranked[:6])
        if b == B3[-1]:
            path = g44._g36.sh_path(shared, {k: {bb: truth_b for bb, truth_b in
                                                 ((int(x["iters"]), x["r_prime"]) for x in rows if x["cls_key"] == k)}
                                             for k in shared}, (2, 1, 1), B3)
            assert r["sh_pick"] == path[-1][0] and r["regret_sh"] == pytest.approx(oracle - truth[path[-1][0]])
        else:
            assert r["regret_sh"] is None


def test_curve_rows_and_summary_shapes():
    space = CLSSpace.from_yaml()
    rows, _ = _rows(space, np.random.default_rng(2), ["A", "B", "C", "D"], n_slices=1, n_shared=3, n_random=1)
    out = g44.curve(rows, folds=["A", "B"], max_k=3)
    # 每折训练源 = 其余 3 个：k=1 → 3 子集，k=2 → 3，k=3 → 1
    assert sorted({(r["fold"], r["k"]) for r in out}) == sorted({(f, k) for f in ("A", "B") for k in (1, 2, 3)})
    assert sum(1 for r in out if r["fold"] == "A" and r["k"] == 1) == 3
    assert sum(1 for r in out if r["fold"] == "A" and r["k"] == 3) == 1
    full = {r["fold"]: r for r in out if r["k"] == 3}
    assert all(r["top1_same_as_full"] is True and r["spearman_vs_full"] == pytest.approx(1.0) for r in full.values())
    summ = g44.summarize(out)
    assert set(summ) == {1, 2, 3}
    assert {"regret3_290_mean", "regret3_290_sd_across_subsets", "regret_sh_290_mean", "top1_same_frac", "n_subsets"} <= set(summ[1])
    assert summ[3]["top1_same_frac"] == 1.0 and summ[3]["n_subsets"] == 2
