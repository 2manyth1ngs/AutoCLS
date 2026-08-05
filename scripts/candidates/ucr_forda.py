"""FordA（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "FordA",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11191164",
    "n_subjects": None,
    "note": "发动机噪声二分类（有/无特定症状）；FordA 训练与测试同为正常工况",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
