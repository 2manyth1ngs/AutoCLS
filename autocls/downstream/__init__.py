"""下游评估器（铁律 7 分发点：downstream 评估器按 task_type 显式分发）。"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..task_types import TASK_CLASSIFICATION, validate_task_type
from .linear_eval import evaluate_classification

__all__ = ["evaluate_classification", "resolve_evaluator"]


def resolve_evaluator(task_type: str) -> Callable[..., dict]:
    """按 task_type 分发下游评估器（铁律 7 分发点）。

    返回 evaluator(encoder, X_train, y_train, X_test, y_test, *, device=None,
    seed=0) -> dict，dict 必含 'r_prime'（early-val 的 R'）——其优劣方向由
    task_types.is_better 按同一 task_type 判定（铁律 8），调用方不得对 r_prime
    做硬编码方向比较。classification: R'=ACC；forecasting: R'=MSE（未实现，
    接入时在此补分支并在 is_better 补"小者优"）。
    """
    validate_task_type(task_type)
    if task_type == TASK_CLASSIFICATION:

        def _eval_classification(
            encoder: torch.nn.Module,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            *,
            device: Optional[str] = None,
            seed: int = 0,
        ) -> dict:
            m = evaluate_classification(
                encoder, X_train, y_train, X_test, y_test, device=device, seed=seed
            )
            return {**m, "r_prime": m["acc"]}

        return _eval_classification
    raise NotImplementedError(
        f"task_type={task_type!r} 的下游评估器未实现——forecasting 接入时在此"
        "补分支（R'=MSE，小者优）"
    )
