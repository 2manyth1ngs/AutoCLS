"""NonInvasiveFetalECGThorax1（UCR 单变量档案）——ECG 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "NonInvasiveFetalECGThorax1",
    "domain": "ECG",
    "fs": None,
    "url": "https://zenodo.org/record/11197817",
    "n_subjects": None,
    "note": "PhysioNet 无创胎儿心电（胸导联 1）；42 类对应不同记录，类数多、每类样本少",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
