"""FreezerRegularTrain（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "FreezerRegularTrain",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11191184",
    "n_subjects": None,
    "note": "冰箱用电负荷二分类（厨房/车库两处冰箱）；两类各半、基本平衡",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
