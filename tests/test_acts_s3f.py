"""WP17-ACTS S3-F（results/acts/s3f/PROTOCOL.md）——先于实现。覆盖：48 任务考卷、残差初始化（起点 = 任务盲模型、s₀ 冻结）、两种读法、读数码、命令行。零 GPU。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.93_acts_s3f")


def test_exam_tasks():
    g = _g()
    tasks = [f"{s}_a{r:02d}" for s in ("A", "B") for r in range(16 if s == "B" else 8)]
    ex = g.exam_tasks(tasks, ["A"] * 8 + ["B"] * 16, ["A", "B"])
    assert ex == [f"A_a{r:02d}" for r in range(8)] + [f"B_a{r:02d}" for r in range(8)]


def test_residual_init_matches_blind():
    import torch
    g = _g(); g92 = importlib.import_module("scripts.92_acts_s3ea")
    torch.manual_seed(3); blind = g92.build_blind(20)
    m = g.build_field_e(20)
    params = g.init_residual(m, blind.state_dict())
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and not any(n.startswith(("cls_mlp", "score_mlp")) for n in names) and any(n.startswith("field_w") for n in names)
    assert len(params) == sum(1 for p in m.parameters() if p.requires_grad)
    a, b, E = torch.rand(6, 20), torch.rand(6, 20), torch.rand(6, 256)
    with torch.no_grad():
        assert torch.allclose(m(a, b, E), blind(a, b, None), atol=1e-6)


def test_readouts():
    g = _g()
    val = {("t", "k1", 0): 0.5, ("t", "k1", 1): 0.6, ("t", "k2", 0): 0.7, ("t", "k2", 1): 0.4, ("t", "k3", 0): 0.1, ("t", "k3", 1): 0.2}
    p1, m3 = g.readouts("t", ["k1", "k2", "k3"], val, floor=0.0)
    assert np.isclose(p1, 0.4) and np.isclose(m3, (0.55 + 0.55 + 0.15) / 3)
    p1, m3 = g.readouts("t", ["k1", "kx"], val, floor=0.05)                  # 缺格按地板补
    assert np.isclose(p1, 0.6) and np.isclose(m3, (0.55 + 0.05) / 2)


def _c(d, p, npos):
    return {"d": d, "p": p, "npos": npos}


def test_verdict():
    g = _g()
    assert g.verdict_s3f(_c(0.012, 0.01, 5), _c(0.003, 0.3, 4), _c(0.006, 0.02, 4), True, 12) == "S3F-E-CONTRIB+CONTENT+CAT-PAR"
    assert g.verdict_s3f(_c(0.012, 0.01, 5), _c(-0.001, 0.3, 2), _c(0.0, 1.0, 3), False, 12) == "S3F-E-NULL"      # 3 选 1 反向
    assert g.verdict_s3f(_c(-0.02, 0.01, 1), _c(-0.01, 0.1, 1), _c(0.0, 1.0, 3), False, 30) == "S3F-E-HURT"
    assert g.verdict_s3f(_c(0.0, 1.0, 0), _c(0.0, 1.0, 0), _c(0.0, 1.0, 0), True, 0) == "S3F-VAL-INERT+CAT-PAR"


def test_cli():
    a = _g().build_parser().parse_args(["--train", "--fold", "Cricket", "--seed", "2"])
    assert a.train and a.fold == "Cricket" and a.seed == 2
