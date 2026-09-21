"""EOGHorizontalSignal（UCR 单变量档案）——EOG 域，WP17-ACTS UCR-ADM 扩池源（results/acts/ucradm/PROTOCOL.md §1.1）。

已切好样本的档案，不开窗（segmented=True），保留原生长度 L。加载口径见 `_ucr.build_ucr`。
`min_class_count: 5` = ACTS 切片器每类 ≥ 5 的对应物（调研期默认 64 会剔掉小类，见 prepare_candidates.effective_min_class_count）。
"""
from __future__ import annotations

from ._ucr import build_ucr

SPEC = {
    "name": "EOGHorizontalSignal",
    "domain": "EOG",
    "fs": None,
    "url": "https://zenodo.org/record/11190930",
    "n_subjects": None,
    "note": "眼电水平通道 12 类眼动，L=1250；与 EOGVerticalSignal 同记录（兄弟不入池）",
    "segmented": True,          # UCR = 已切好样本，驱动不再开窗
    "min_class_count": 5,       # ACTS 口径：每类 ≥ 5 即可（不剔类）
}


def build():
    return build_ucr(SPEC["name"], SPEC["domain"])
