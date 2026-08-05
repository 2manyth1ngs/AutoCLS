"""MixedShapesRegularTrain（UCR 单变量档案）——Image 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "MixedShapesRegularTrain",
    "domain": "Image",
    "fs": None,
    "url": "https://zenodo.org/record/11197803",
    "n_subjects": None,
    "note": "图像轮廓转一维距离序列（非时序传感器）；RegularTrain 变体训练集较大",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
