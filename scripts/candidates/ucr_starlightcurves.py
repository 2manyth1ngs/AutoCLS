"""StarLightCurves（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "StarLightCurves",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11198308",
    "n_subjects": None,
    "note": "天文光变曲线，原生观测不等间隔、档案已重采样对齐到 1024 步；N 与 L 都偏大，npz 体积最大的候选",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
