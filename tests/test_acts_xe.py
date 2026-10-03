"""WP17-ACTS XE（results/acts/xe/PROTOCOL.md）——先于实现。覆盖：E 形式拼接、缺失插补、近亲检索质量、选形式规则、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _x():
    return importlib.import_module("scripts.117_acts_xe")


def test_build_form_and_impute():
    x = _x()
    blocks = {"pt": np.ones((3, 2)), "A": np.zeros((3, 1)), "T": np.full((3, 2), 2.0)}
    assert x.build_form(blocks, "pt+T").shape == (3, 4) and np.allclose(x.build_form(blocks, "pt+T")[:, 2:], 2.0)
    assert x.build_form(blocks, "A").shape == (3, 1)
    M = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    out = x.impute_nan(M, ref_rows=[0, 1])                       # 列均值只用参照行
    assert np.allclose(out, [[1.0, 4.0], [3.0, 4.0], [2.0, 8.0]])


def test_retrieval_score_and_hit():
    x = _x()
    rng = np.random.default_rng(0)
    S = rng.normal(size=(5, 8))                                  # 5 个候选源的均值 E
    q = S[0] + 0.01 * rng.normal(size=(3, 8))                     # 查询任务 ≈ 候选 0
    esim = x.e_similarity(q, S)
    assert esim.shape == (5,) and int(np.argmax(esim)) == 0
    assert abs(x.retrieval_score(esim, esim * 2 + 1) - 1.0) < 1e-12 # 单调同序 → Spearman 1
    assert x.hit_at3(np.array([5, 4, 3, 2, 1.0]), np.array([1, 2, 5, 4, 3.0])) == 1 / 3


def test_pick_form():
    x = _x()
    dims = {"pt+A": 281, "A": 25, "T": 44}
    assert x.pick_form({"pt+A": 0.10, "A": 0.12, "T": 0.30}, dims) == "T"
    assert x.pick_form({"pt+A": 0.20, "A": 0.20, "T": 0.10}, dims) == "pt+A"        # 并列优先现用
    assert x.pick_form({"pt+A": 0.10, "A": 0.30, "T": 0.30}, dims) == "A"           # 并列再取维度少者


def test_verdict():
    x = _x()
    assert x.verdict_xe(0.650, 0.6189) == "XE-BEATS-BOTH+SWAP+BLIND+E"
    assert x.verdict_xe(0.628, 0.6189) == "XE-BEATS-TS2VEC+E"
    assert x.verdict_xe(0.615, 0.6189) == "XE-BELOW"


def test_cli():
    x = _x()
    assert x.build_parser().parse_args(["--select"]).select
    assert x.build_parser().parse_args(["--compute-et", "--parts", "4", "--part", "1"]).part == 1
    assert set(x.FORMS) == {"pt", "A", "T", "pt+A", "pt+T", "A+T", "pt+A+T"}
