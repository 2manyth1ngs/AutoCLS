"""task_types 单测（铁律 7「任务类型显式分发」+ 铁律 8「R' 优劣方向抽象」）。

- task_type 取值唯一真源：TASK_CLASSIFICATION / TASK_FORECASTING（预留 stub）；
  各分发点入口用 validate_task_type 对未知类型 fail-fast。
- is_better(r1, r2, task_type)：造 label 时判 y 的唯一合法途径，"什么叫更优"
  由任务类型决定——classification: ACC 大者优；forecasting: MSE 小者优（当前
  留 stub，接入时实现并补用例）。
"""
import pytest

from autocls.task_types import (
    KNOWN_TASK_TYPES,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    is_better,
    validate_task_type,
)


class TestTaskTypeConstants:
    def test_values_and_registry(self):
        assert TASK_CLASSIFICATION == "classification"
        assert TASK_FORECASTING == "forecasting"
        assert TASK_CLASSIFICATION in KNOWN_TASK_TYPES
        assert TASK_FORECASTING in KNOWN_TASK_TYPES

    def test_validate_accepts_known(self):
        for t in KNOWN_TASK_TYPES:
            validate_task_type(t)  # 不抛即通过

    def test_validate_rejects_unknown(self):
        with pytest.raises(ValueError, match="task_type"):
            validate_task_type("segmentation")


class TestIsBetter:
    def test_classification_higher_acc_is_better(self):
        # 分类 R'=ACC，大者优（铁律 8：方向不写死在调用点）
        assert is_better(0.93, 0.90, TASK_CLASSIFICATION) is True
        assert is_better(0.90, 0.93, TASK_CLASSIFICATION) is False

    def test_classification_tie_is_not_better(self):
        # 严格优：平局不算优（配对 label 造样时平局应弃样，不给假方向）
        assert is_better(0.90, 0.90, TASK_CLASSIFICATION) is False

    def test_forecasting_stub_reserved(self):
        # 预测分支留 stub：接入时实现"小者优"（r1 < r2），当前显式拒绝而非静默给错方向
        with pytest.raises(NotImplementedError, match="小者优"):
            is_better(0.1, 0.2, TASK_FORECASTING)

    def test_unknown_task_type_rejected(self):
        with pytest.raises(ValueError, match="task_type"):
            is_better(0.9, 0.8, "segmentation")
