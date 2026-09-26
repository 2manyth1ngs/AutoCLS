"""WP17-ACTS S3-H（results/acts/s3h/PROTOCOL.md）——先于实现。覆盖：ctx 折内标准化、ctx 残差模型起点 = 任务盲模型、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.95_acts_s3h")


def test_standardize_ctx():
    g = _g()
    C = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [100.0, -7.0]])
    groups = np.array(["A", "A", "B", "F"])
    Z = g.standardize_ctx(C, groups, fold="F")
    tr = Z[:3]
    assert np.allclose(tr.mean(0), 0) and np.allclose(tr.std(0), 1)                     # 只用折内训练侧（非留出源）拟合
    assert np.isclose(Z[3, 0], (100.0 - 3.0) / np.std([1.0, 3.0, 5.0]))


def test_ctx_residual_matches_blind():
    import torch
    g = _g(); g92 = importlib.import_module("scripts.92_acts_s3ea"); g93 = importlib.import_module("scripts.93_acts_s3f")
    torch.manual_seed(5); blind = g92.build_blind(20)
    m = g.build_field_ctx(20)
    g93.init_residual(m, blind.state_dict())
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names == {n for n, _ in m.named_parameters() if n.startswith(("set_fc", "field_w"))}
    a, b, c = torch.rand(5, 20), torch.rand(5, 20), torch.rand(5, g.K_CTX)
    with torch.no_grad():
        assert torch.allclose(m(a, b, None, c), blind(a, b, None), atol=1e-6)


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    assert g.verdict_s3h(_c(0.011, 0.02, 4), _c(0.001, 0.5, 4), _c(0.006, 0.03, 5), True, 10) == "S3H-E-CONTRIB+CONTENT+CAT-PAR"
    assert g.verdict_s3h(_c(0.011, 0.02, 3), _c(0.001, 0.5, 4), _c(0.0, 1.0, 3), False, 10) == "S3H-E-NULL"
    assert g.verdict_s3h(_c(-0.012, 0.01, 1), _c(-0.01, 0.1, 1), _c(0.0, 1.0, 3), False, 10) == "S3H-E-HURT"
    assert g.verdict_s3h(_c(0.0, 1.0, 0), _c(0.0, 1.0, 0), _c(0.0, 1.0, 0), False, 0) == "S3H-VAL-INERT"


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--fold", "Cricket", "--seed", "1"])
    assert a.train and a.fold == "Cricket" and a.seed == 1
