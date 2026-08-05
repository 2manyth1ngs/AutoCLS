"""FordB（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "FordB",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11191172",
    "n_subjects": None,
    "note": "与 FordA 同源，但测试集在带噪工况下采集——原档案是刻意做的分布漂移对，本项目合并 TRAIN+TEST 后该漂移变成集内异质性",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
