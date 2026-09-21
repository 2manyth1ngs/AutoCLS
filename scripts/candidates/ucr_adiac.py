"""Adiac（UCR 单变量档案）——Image 域，WP17-ACTS UCR-ADM 扩池源（results/acts/ucradm/PROTOCOL.md §1.1）。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
`min_class_count: 5` = ACTS 切片器每类 ≥ 5 的对应物（调研期默认 64 会剔掉小类，见 prepare_candidates.effective_min_class_count）。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "Adiac",
    "domain": "Image",
    "fs": None,
    "url": "https://zenodo.org/record/11179788",
    "n_subjects": None,
    "note": "硅藻轮廓 37 类，L=176，每类 ≈ 20 条",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
    "min_class_count": 5,       # ACTS 口径：每类 ≥ 5 即可（不剔类）
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
