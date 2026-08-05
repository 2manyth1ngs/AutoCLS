"""Wafer（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "Wafer",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11198387",
    "n_subjects": None,
    "note": "半导体蚀刻工序单传感器记录，正常/异常二分类，正类占绝大多数（重度不平衡）",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
