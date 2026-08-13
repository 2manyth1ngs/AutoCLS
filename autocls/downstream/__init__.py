"""下游评估器（铁律 7 分发点：downstream 评估器按 task_type 显式分发）。"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..task_types import TASK_CLASSIFICATION, TASK_FORECASTING, validate_task_type
from .forecast_eval import evaluate_forecasting
from .linear_eval import evaluate_classification

__all__ = ["evaluate_classification", "evaluate_forecasting", "resolve_evaluator"]


def resolve_evaluator(task_type: str) -> Callable[..., dict]:
    """按 task_type 分发下游评估器（铁律 7 分发点）。

    返回的 evaluator 签名随任务类型不同（评估协议本质不同，强行统一签名只会
    造出假抽象），但返回 dict 必含 'r_prime'（early-val 的 R'）——其优劣方向由
    task_types.is_better 按同一 task_type 判定（铁律 8），调用方不得对 r_prime
    做硬编码方向比较。
    - classification: evaluator(encoder, X_train, y_train, X_test, y_test, *,
      device=None, seed=0)；R'=ACC（大者优）。
    - forecasting: evaluator(encoder, data, train_slice, valid_slice, test_slice,
      *, pred_len, n_covariate_cols, scaler=None, padding=200, batch_size=256,
      device=None, seed=0)；R'=norm 尺度 MSE@pred_len（小者优，TS2Vec 官方
      Ridge 协议，见 forecast_eval 模块 docstring）。
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
    if task_type == TASK_FORECASTING:

        def _eval_forecasting(encoder, data, **kwargs) -> dict:
            m = evaluate_forecasting(encoder, data, **kwargs)
            return {**m, "r_prime": m["mse"]}

        return _eval_forecasting
    raise NotImplementedError(
        f"task_type={task_type!r} 已登记但下游评估器未实现——新类型接入时在此补分支"
    )
