"""任务类型显式分发的公共原语（CLAUDE.md 铁律 7 + 铁律 8 的落点）。

铁律 7「任务类型显式分发」：本模块是全项目 task_type 取值的**唯一真源**。
task 数据结构必须带显式 task_type 字段（当前恒为 classification）；凡按任务类型
不同而行为不同的逻辑必须经 task_type 显式分发，禁止隐式假设"就是分类"。
已知分发点：task_slicer 切片策略（slice_strategies.resolve_slice_strategy）、
task_embed 窗口逻辑、downstream 评估器、early-val 的 R' 优劣方向（本文件
is_better）。任务无关模块（cls_space/cls_encoder/contrastive/encoders/
comparator/search）不得 import 本模块——它们一旦需要 task_type 就说明破坏了纯粹性。

铁律 8「R' 优劣方向抽象」："什么叫更优"随任务类型不同（分类 ACC 大者优；预测
MSE 小者优）。造 label 时 y 的定义禁止写死比较方向，必须经 is_better 判定。

扩展点：新增任务类型 = ①在此登记常量并入 KNOWN_TASK_TYPES；②补 is_better 的
方向分支；③在各分发点加 case（如 slice_strategies 里新增该类型的切片策略并在
resolve_slice_strategy 分发）。核心与既有分支零改动。
"""
from __future__ import annotations

TASK_CLASSIFICATION = "classification"
TASK_FORECASTING = "forecasting"  # 预留：阶段一未实现，仅作分发 stub（关键决策 3）

KNOWN_TASK_TYPES = (TASK_CLASSIFICATION, TASK_FORECASTING)


def validate_task_type(task_type: str) -> None:
    """task_type 合法性检查——各分发点入口调用，未知类型 fail-fast。"""
    if task_type not in KNOWN_TASK_TYPES:
        raise ValueError(
            f"未知 task_type={task_type!r}，合法取值 {KNOWN_TASK_TYPES}"
            "（新任务类型须先在 autocls/task_types.py 登记，铁律 7）"
        )


def is_better(r1: float, r2: float, task_type: str) -> bool:
    """R' 优劣统一比较（铁律 8）：r1 是否**严格**优于 r2，方向由任务类型决定。

    造配对 label 的唯一合法途径：y = is_better(r_a, r_b, task_type)；平局返回
    False（严格优），调用方应弃平局样本而非赋任意方向。
    classification: R'=ACC，大者优；forecasting: R'=MSE，小者优——当前留 stub，
    接入时实现 `return r1 < r2` 并在 tests/test_task_types.py 补用例。
    """
    validate_task_type(task_type)
    if task_type == TASK_CLASSIFICATION:
        return r1 > r2
    raise NotImplementedError(
        f"task_type={task_type!r} 的 R' 比较未实现——forecasting 为小者优"
        "（r1 < r2），接入时在此补分支"
    )
