"""WP17-ACTS S3-EA（results/acts/s3ea/PROTOCOL.md）——先于实现。覆盖：读数码、换源指派、任务盲 field 模型结构、提名去重、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.92_acts_s3ea")


def _cmp(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    assert g.verdict_s3ea(_cmp(0.012, 0.01, 5), _cmp(0.011, 0.03, 4), 0.001) == "EA-E-CONTRIB+CONTENT+BLIND-PAR"
    assert g.verdict_s3ea(_cmp(0.012, 0.01, 3), _cmp(0.0, 1.0, 0), -0.002) == "EA-E-NULL"           # 同号源不够
    assert g.verdict_s3ea(_cmp(0.004, 0.01, 6), _cmp(0.02, 0.01, 6), 0.0) == "EA-E-NULL+CONTENT+BLIND-PAR"
    assert g.verdict_s3ea(_cmp(-0.015, 0.02, 1), _cmp(0.0, 1.0, 3), 0.01) == "EA-E-HURT+BLIND-PAR"
    assert g.verdict_s3ea(_cmp(-0.015, 0.20, 1), _cmp(0.0, 1.0, 3), -0.01) == "EA-E-NULL"


def test_cross_index_other_source():
    g = _g()
    tasks = ["A_a00", "A_a01", "B_a00", "B_a01", "C_a00"]
    groups = ["A", "A", "B", "B", "C"]
    m = g.cross_index(tasks, groups, ["A", "B", "C"])
    assert [groups[j] for j in m] == ["B", "B", "C", "C", "A"]
    assert m[1] == 3 and m[4] == 0                                    # 同 OA 行号；缺行取该源第 0 行


def test_blind_model_structure():
    import torch
    g = _g()
    m = g.build_blind(20)
    assert m.task_branch == "none" and m.head == "field" and m.set_encoder is None and not hasattr(m, "field_w")
    torch.manual_seed(1)
    a, b = torch.rand(7, 20), torch.rand(7, 20)
    with torch.no_grad():
        assert torch.allclose(m(a, b, None) + m(b, a, None), torch.ones(7), atol=1e-6)


def test_new_tags():
    g = _g()
    noms = {"t1": {"B-lcb": ["k1", "k2"], "X-lcb": ["k2", "k3"]}}
    assert g.new_tags(noms, {"t1|k1|0"}) == ["t1|k1|1", "t1|k2|0", "t1|k2|1", "t1|k3|0", "t1|k3|1"]


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--fold", "Cricket", "--seed", "3"])
    assert a.train and a.fold == "Cricket" and a.seed == 3
