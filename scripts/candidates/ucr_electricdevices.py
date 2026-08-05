"""ElectricDevices（UCR 单变量档案）——Device 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "ElectricDevices",
    "domain": "Device",
    "fs": None,
    "url": "https://zenodo.org/record/11190880",
    "n_subjects": None,
    "note": "英国 Powering the Nation 家电用电调研；一条 = 一天 96 步（15 分钟粒度），7 类家电极不平衡",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
