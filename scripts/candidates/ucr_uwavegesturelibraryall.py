"""UWaveGestureLibraryAll（UCR 单变量档案）——Motion 域候选。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "UWaveGestureLibraryAll",
    "domain": "Motion",
    "fs": 100,
    "url": "https://zenodo.org/record/11198366",
    "n_subjects": None,
    "note": "手势加速度计；一条 945 步 = X/Y/Z **三轴首尾拼接**（315×3），不是真正的单变量连续信号——crop/jitter 类增强会跨轴边界，写论文时须点明",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
