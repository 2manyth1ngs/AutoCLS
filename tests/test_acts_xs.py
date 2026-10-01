"""WP17-ACTS XS（results/acts/xs/PROTOCOL.md）——先于实现。覆盖：可行集规则、残差聚类选代表、L4 标签、s₀ 合并、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _x():
    return importlib.import_module("scripts.116_acts_xs")


def test_eligible_reasons():
    x = _x()
    kw = dict(targets={"Cricket"}, rejected={"SelfRegulationSCP2"}, target_pool={"Epilepsy"}, l_max=1751)
    ok = {"length_max": "144", "n_total": "440", "equal_length": "True"}
    assert x.eligible_reasons("PEMS-SF", ok, **kw) == []
    assert x.eligible_reasons("Cricket", ok, **kw) and x.eligible_reasons("Epilepsy", ok, **kw) and x.eligible_reasons("SelfRegulationSCP2", ok, **kw)
    assert x.eligible_reasons("Libras", {"length_max": "45", "n_total": "360", "equal_length": "True"}, **kw)
    assert x.eligible_reasons("EigenWorms", {"length_max": "17984", "n_total": "259", "equal_length": "True"}, **kw)
    assert x.eligible_reasons("BasicMotions", {"length_max": "100", "n_total": "80", "equal_length": "True"}, **kw)
    assert x.eligible_reasons("CharacterTrajectories", {"length_max": "182", "n_total": "2858", "equal_length": "False"}, **kw)


def test_select_reps():
    x = _x()
    rng = np.random.default_rng(0)
    base = {g: rng.normal(size=30) for g in "abc"}
    prof = {f"{g}{i}": base[g] + 0.05 * rng.normal(size=30) for g in "abc" for i in range(2)}
    reps = x.select_reps(prof, np.zeros(30), k=3)
    assert len(reps) == 3 and sorted(r[0] for r in reps) == ["a", "b", "c"]          # 每组一个
    few = {k: prof[k] for k in ("c0", "a1", "b0")}
    assert x.select_reps(few, np.zeros(30), k=4) == ["a1", "b0", "c0"]                 # ≤ k → 全选（名字序）


def test_l4_labels():
    x = _x()
    r, fail = x.l4_labels(np.array([0.5, 0.6]), np.array([0.7, 0.6]), np.array([0.4]), 0.55)
    assert np.allclose(r, [0.6, 0.6, 0.4]) and fail.tolist() == [False, False, True]


def test_merge_s0():
    x = _x()
    e = np.ones(4)
    old = {"lens": np.array([2]), "Cricket/0": np.array([1.0, 2.0]), "Ept_target/Cricket": e}
    new = {"lens": np.array([1]), "Cricket/0": np.array([3.0])}
    m = x.merge_s0(old, new)
    assert m["lens"].tolist() == [2, 1] and m["Cricket/0"].tolist() == [1.0, 2.0, 3.0] and np.allclose(m["Ept_target/Cricket"], e)


def test_verdict():
    x = _x()
    assert x.verdict_xs(0.650, 0.640) == "XS-BEATS-BOTH+SRC+BLIND+E"
    assert x.verdict_xs(0.630, 0.635) == "XS-BEATS-TS2VEC"
    assert x.verdict_xs(0.620, 0.610) == "XS-BELOW+E"


def test_cli():
    x = _x()
    a = x.build_parser().parse_args(["--select"])
    assert a.select and not a.analyze
    assert x.build_parser().parse_args(["--analyze"]).analyze
