"""TwoPatterns（UCR 单变量档案）——Simulated 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "TwoPatterns",
    "domain": "Simulated",
    "fs": None,
    "url": "https://zenodo.org/record/11198356",
    "n_subjects": None,
    "note": "合成档案（人工植入两种形状模式）；无真实传感器噪声结构，作对照组用",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
