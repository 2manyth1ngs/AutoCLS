"""ECG5000（UCR 单变量档案）——ECG 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "ECG5000",
    "domain": "ECG",
    "fs": None,
    "url": "https://zenodo.org/record/11186692",
    "n_subjects": 1,
    "note": "BIDMC 充血性心衰库单个病人的 20 小时记录切出的 5000 个心搏——**同一受试者**，类内相关性远高于跨人数据集；5 类重度不平衡（正常搏占 ~58%）",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
