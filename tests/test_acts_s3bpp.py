"""WP17-ACTS S3-B″（results/acts/s3bpp/PROTOCOL.md）——先于实现。覆盖：多偏移 cross 指派（k=1 = S3-B cross_assign）、内层选 r、读数码、命令行。"""
from __future__ import annotations

import importlib

import numpy as np


def _g():
    return importlib.import_module("scripts.88_acts_s3bpp")


def test_cross_maps():
    g = _g(); g86 = importlib.import_module("scripts.86_acts_s3b")
    tasks = ["A_a00", "A_a01", "B_a00", "C_a00", "C_a01", "C_a02"]
    groups = np.array(["A", "A", "B", "C", "C", "C"]); order = ["A", "B", "C"]
    maps = g.cross_maps(tasks, groups, order, offsets=(1, 2))
    assert maps[0] == g86.cross_assign(tasks, groups, order)
    assert [tasks[j] for j in maps[1]] == ["C_a00", "C_a01", "A_a00", "B_a00", "B_a00", "B_a00"]
    for m in maps:                                  # 不会指回自己的源
        assert all(groups[j] != groups[i] for i, j in enumerate(m))


def test_choose_r():
    g = _g()
    assert g.choose_r({4: 0.012, 8: 0.010, 16: 0.011}) == 8
    assert g.choose_r({4: 0.010, 8: 0.010, 16: 0.011}) == 4          # 平局取小


def test_verdict():
    g = _g()
    f = {"F1": False, "F2": False, "F3": False, "F4": False}
    c = {"F5": False}
    assert g.verdict_fec({"pt": dict(f), "sh": dict(f)}, {"pt+p": dict(c), "sh+p": dict(c)}) == "FEC-NOT-CONFIRMED"
    assert g.verdict_fec({"pt": dict(f), "sh": {**f, "F2": True, "F3": True}}, {"pt+p": dict(c), "sh+p": dict(c)}) == "FEC-VALUE-ONLY"
    assert g.verdict_fec({"pt": {**f, "F1": True}, "sh": {**f, "F2": True}}, {"pt+p": dict(c), "sh+p": dict(c)}) == "FEC-NOT-CONFIRMED"   # F1 ∧ F2 须同一臂
    assert g.verdict_fec({"pt": dict(f), "sh": {**f, "F1": True, "F2": True, "F4": True}}, {"pt+p": {"F5": True}, "sh+p": dict(c)}) == "FEC-CONFIRMED+ADD+SPACE"


def test_cli():
    assert _g().build_parser().parse_args(["--analyze"]).analyze is True
