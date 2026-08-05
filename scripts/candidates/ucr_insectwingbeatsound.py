"""InsectWingbeatSound（UCR 单变量档案）——Sensor 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "InsectWingbeatSound",
    "domain": "Sensor",
    "fs": None,
    "url": "https://zenodo.org/record/11197635",
    "n_subjects": None,
    "note": "昆虫振翅声——档案里的 256 步是**频谱**而非时域波形，时间轴语义与其余候选不同，对时序增强（shift/crop）的意义存疑",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
