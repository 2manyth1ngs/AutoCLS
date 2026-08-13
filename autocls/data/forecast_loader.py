"""预测任务数据装载（铁律 4：先读 TS2Vec 官方真实源码再适配，逐行标注对应行号）。

对照源：yuezhihan/ts2vec `datautils.py`（main 分支，2026-08-13 拉取核对）：
- `_get_time_features`（L125-134）→ time_features：7 个日历协变量
  [minute, hour, dayofweek, day, dayofyear, month, weekofyear]
- `load_forecast_csv`（L137-181）→ load_forecast_csv：date 索引解析、univar 时
  ETT 系取 'OT' 列（L143-144）/ 其余取末列（L148）、ETTh1/h2 按 12/4/4 月切片
  （L151-154）、ETTm1/m2 ×4（L155-158）、其余 0.6/0.2/0.2（L160-162）、
  StandardScaler **只 fit train 切片**（L164）、协变量各自标准化后拼在**前**
  n_covariate_cols 列（L171-174）、ETTh 系 pred_lens=[24,48,168,336,720]（L176-177）。

与官方的差异（全部注明理由，数值不变或不影响本项目路径）：
- 官方 `np.float`（L134）已被 numpy 移除 → np.float64（同一精度）。
- 官方 `dt.weekofyear`（L133）已被 pandas 移除 → isocalendar().week（同值）。
- 不支持 electricity 的"每变量一实例"转置（L166-167）——本轮只做 ETT 类
  单实例长序列；命中该名报 NotImplementedError 而非静默走错分支。
- 新增 make_pretrain_instances：官方把 (1, T, C) 长序列直接喂 TS2Vec.fit（内部按
  max_train_length=3000 切段成 N=2 段、batch=2）；本项目 cl_trainer 是实例批语义
  （STD_BATCH_SIZE=128，铁律 2），故以滑窗把 train 切片切成 (N, inst_len, C) 实例。
  inst_len 默认 256 沿路线 A CROP=256 口径（8GB 显存下 resize 2× 后 sim 矩阵仍
  与既有分类工作负载同量级）。窗口终点绝不越出 train 切片（铁律 6 同款泄漏防线：
  对比预训练不可见 valid/test 时间步；下游 Ridge 的 valid/test 表征由**冻结**
  encoder 事后编码，与官方 encode 全序列语义一致，不构成泄漏）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """(T,) DatetimeIndex → (T, 7) float64 日历协变量（官方 _get_time_features L125-134）。"""
    return np.stack(
        [
            np.asarray(index.minute),
            np.asarray(index.hour),
            np.asarray(index.dayofweek),
            np.asarray(index.day),
            np.asarray(index.dayofyear),
            np.asarray(index.month),
            # 官方 dt.weekofyear（L133）已被 pandas 移除，isocalendar().week 同值
            np.asarray(index.isocalendar().week),
        ],
        axis=1,
    ).astype(np.float64)


def load_forecast_csv(
    csv_path: str | Path, univar: bool = False
) -> tuple[np.ndarray, slice, slice, slice, StandardScaler, list[int], int]:
    """官方 load_forecast_csv（L137-181）：按文件名（stem）分发切片/取列规则。

    返回 (data, train_slice, valid_slice, test_slice, scaler, pred_lens,
    n_covariate_cols)，data 形如 (1, T, n_cov + n_feat)、全部已标准化；
    scaler 只 fit train 切片、只作用于数据列（协变量另配 scaler，与官方一致）。
    """
    csv_path = Path(csv_path)
    name = csv_path.stem
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)  # L138
    dt_embed = time_features(df.index)  # L139
    n_covariate_cols = dt_embed.shape[-1]  # L140

    if univar:  # L142-148
        if name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
            df = df[["OT"]]
        elif name == "electricity":
            raise NotImplementedError(
                "electricity 的每变量一实例转置（官方 L166-167）未适配——本轮只做 ETT 类"
            )
        else:
            df = df.iloc[:, -1:]

    data = df.to_numpy()
    if name in ("ETTh1", "ETTh2"):  # L151-154
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name in ("ETTm1", "ETTm2"):  # L155-158
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
    else:  # L160-162
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])  # L164：只 fit train
    data = scaler.transform(data)
    if name == "electricity":
        raise NotImplementedError("electricity 未适配（同 univar 分支说明）")
    data = np.expand_dims(data, 0)  # L169：(T, C) → (1, T, C)

    if n_covariate_cols > 0:  # L171-174：协变量各自标准化拼在前
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    if name in ("ETTh1", "ETTh2", "electricity"):  # L176-179
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def make_pretrain_instances(
    data: np.ndarray, train_slice: slice, *, inst_len: int = 256, stride: int = 24
) -> np.ndarray:
    """train 切片 → (N, inst_len, C) float32 滑窗实例（本项目新增，见模块 docstring）。

    起点 0, stride, 2·stride, …；终点越出 train 切片的窗直接丢弃（绝不 pad、绝不
    借用 valid/test 时间步）。train 长度 < inst_len 时无窗可切，fail-fast。
    """
    assert data.ndim == 3 and data.shape[0] == 1, f"期望 (1, T, C)，得到 {data.shape}"
    series = data[0, train_slice]
    T = series.shape[0]
    if T < inst_len:
        raise ValueError(f"train 切片长 {T} < inst_len {inst_len}，无窗可切")
    starts = range(0, T - inst_len + 1, stride)
    return np.stack([series[s : s + inst_len] for s in starts]).astype(np.float32)
