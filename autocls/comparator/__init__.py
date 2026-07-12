"""比较器 T-CLSC（TECHNICAL_DESIGN.md 模块 6，核心创新）。

任务无关模块（铁律 7）：不 import task_types；优劣方向经 tclsc_engine 的
better_fn 注入，由脚本层绑定 task_types.is_better（铁律 8）。
"""
from .tclsc_engine import ClsRecord, Pair, TCLSCEngine, derangement, dynamic_pairs, with_reversed
from .tclsc_model import TCLSC

__all__ = [
    "ClsRecord",
    "Pair",
    "TCLSC",
    "TCLSCEngine",
    "derangement",
    "dynamic_pairs",
    "with_reversed",
]
