"""ChlorineConcentration（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "ChlorineConcentration",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11186229",
    "n_subjects": None,
    "note": "EPANET 供水管网仿真的余氯浓度；类间差异极细微，是 UCR 上公认的难档案",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
