"""预测下游评估（AutoCL 附录 B.1"linear regression with l2 norm"= TS2Vec 官方
Ridge 协议；铁律 4：先读真实源码再适配，逐行标注对应文件+行号）。

对照源：yuezhihan/ts2vec（main 分支，2026-08-13 拉取核对）
- tasks/forecasting.py
  - L5-12  generate_pred_samples：feature r_t ↔ label data[t+1..t+Q]（展平 Q·C 维）
  - L14-18 cal_metrics：MSE / MAE
  - L20-30 表征提取：model.encode(data, causal=True, sliding_length=1,
    sliding_padding=200, batch_size=256) —— 每个时间步 t 的表征 = 编码"以 t 结尾、
    长 padding+1 的因果窗"后取**末位**（ts2vec.py L242-256：l=i-padding, r=i+1,
    slicing=slice(padding, padding+1)）。不做任何池化（encoding_window=None）。
    分类协议是 full_series max-pool，预测协议与之不同——这里是逐时间步因果表征。
  - L46    train 侧 drop=padding：前 padding 个训练样本（窗内含填充值）丢弃；
           valid/test 不 drop（其窗口上下文全部真实存在）
  - L62-67 raw 指标 = scaler.inverse_transform 后重算
- tasks/_eval_protocols.py L81-109 fit_ridge：Ridge、alpha 网格 13 档、
  验证集 score=RMSE+MAE 选 alpha、best alpha 在 train 上重拟合

与官方的差异（全部注明，均不影响入判数值）：
1. 左侧缺失上下文用**零填充**：官方 torch_pad_nan 填 NaN、其 TSEncoder 内部把
   NaN 时间步置零；本项目 encoder 无 mask 语义（路线 A"无 mask 中性零填充"口径，
   DilatedCNNEncoder 对 NaN 的处理同为置零），直接零填充等效。受影响的只有全序列
   前 padding 个时间步的表征——它们全部落在 train 切片且被 drop=padding 丢弃
   （官方同款），valid/test 的窗口不含任何填充。
2. raw 指标的逆标准化用显式广播 pred*scale_+mean_（官方 L63-67 对 4 维数组调
   sklearn inverse_transform，依赖旧版 sklearn 不查维度；数学同式）。
3. 无 GridSearchCV/loky：Ridge 全程串行 sklearn 直调，分类协议的 threading
   后端教训不适用于本模块。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 官方 tasks/forecasting.py L21：滑动编码的上下文长度（协议常数，勿改）
SLIDING_PADDING = 200
# 官方 tasks/_eval_protocols.py L98：alpha 网格（协议常数，勿改）
RIDGE_ALPHAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def generate_pred_samples(
    features: np.ndarray, data: np.ndarray, pred_len: int, drop: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(1,T,d) 表征 + (1,T,C) 数据 → 监督样本 (n, d), (n, Q·C)（官方 L5-12 逐行）。

    对齐语义：features[t] 预测 data[t+1 .. t+pred_len]；drop 丢弃前 drop 个样本。
    """
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack(
        [data[:, i : 1 + n + i - pred_len] for i in range(pred_len)], axis=2
    )[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return (
        features.reshape(-1, features.shape[-1]),
        labels.reshape(-1, labels.shape[2] * labels.shape[3]),
    )


def cal_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """MSE/MAE（官方 L14-18 逐行）。"""
    return {
        "MSE": ((pred - target) ** 2).mean(),
        "MAE": np.abs(pred - target).mean(),
    }


def fit_ridge(
    train_features: np.ndarray,
    train_y: np.ndarray,
    valid_features: np.ndarray,
    valid_y: np.ndarray,
    MAX_SAMPLES: int = 100000,
) -> Ridge:
    """官方 _eval_protocols.py L81-109 逐行：验证集 RMSE+MAE 选 alpha 再重拟合。"""
    if train_features.shape[0] > MAX_SAMPLES:  # L83-89（本项目 ETTh1 触不到）
        split = train_test_split(train_features, train_y, train_size=MAX_SAMPLES, random_state=0)
        train_features, train_y = split[0], split[2]
    if valid_features.shape[0] > MAX_SAMPLES:  # L90-96
        split = train_test_split(valid_features, valid_y, train_size=MAX_SAMPLES, random_state=0)
        valid_features, valid_y = split[0], split[2]

    valid_results = []
    for alpha in RIDGE_ALPHAS:  # L100-104
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = RIDGE_ALPHAS[int(np.argmin(valid_results))]  # L105

    lr = Ridge(alpha=best_alpha)  # L107-109：best alpha 只在 train 上重拟合
    lr.fit(train_features, train_y)
    return lr


@torch.no_grad()
def encode_causal_sliding(
    encoder: torch.nn.Module,
    data: np.ndarray,
    *,
    padding: int = SLIDING_PADDING,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """(1,T,C) → (1,T,d) 逐时间步因果表征（官方 encode causal+sliding 语义适配）。

    repr[t] = encoder(零左填充的 data[t-padding .. t])[:, -1]（官方 ts2vec.py
    L242-256：窗口 [i-padding, i+1)、slicing 取末位；差异 1：零填充代替 NaN 填充）。
    encoder 只需满足 (B,T,C)→(B,T,d) 逐时间步输出，因果性由窗口构造保证，
    与 encoder 自身是否因果无关。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert data.ndim == 3 and data.shape[0] == 1, f"期望 (1,T,C)，得到 {data.shape}"
    encoder = encoder.to(device)
    was_training = encoder.training
    encoder.eval()
    x = torch.from_numpy(np.ascontiguousarray(data[0], dtype=np.float32))  # (T, C)
    T, C = x.shape
    padded = torch.cat([torch.zeros(padding, C, dtype=x.dtype), x], dim=0)
    outs = []
    for start in range(0, T, batch_size):
        stop = min(start + batch_size, T)
        # 窗口 i ∈ [start, stop)：padded[i : i+padding+1] 即原序列 [i-padding, i]
        wins = torch.stack([padded[i : i + padding + 1] for i in range(start, stop)])
        z = encoder(wins.to(device))  # (b, padding+1, d)
        outs.append(z[:, -1].cpu())
    if was_training:
        encoder.train()
    return torch.cat(outs, dim=0).unsqueeze(0).numpy()


def evaluate_forecasting(
    encoder: torch.nn.Module,
    data: np.ndarray,
    train_slice: slice,
    valid_slice: slice,
    test_slice: slice,
    *,
    pred_len: int,
    n_covariate_cols: int,
    scaler=None,
    padding: int = SLIDING_PADDING,
    batch_size: int = 256,
    device: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """预训练后 encoder 的下游预测评估（官方 eval_forecasting L20-78 单 horizon 版）。

    返回 {'mse','mae','alpha','n_train_samples','n_eval'[,'mse_raw','mae_raw']}；
    mse/mae 为标准化（norm）尺度——TS2Vec/AutoCL 论文 Table 报告口径，R' 取 mse。
    scaler 给定时按官方 L62-67 补 raw 尺度指标（差异 2：显式广播逆标准化）。
    seed 不参与计算（Ridge 确定性、样本量触不到官方 MAX_SAMPLES 下采样），
    保留参数仅为与分类评估器接口对齐。
    """
    all_repr = encode_causal_sliding(
        encoder, data, padding=padding, batch_size=batch_size, device=device
    )  # 官方 L24-30

    train_repr = all_repr[:, train_slice]  # L33-35
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]  # L37-39：目标只含数据列
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    train_features, train_labels = generate_pred_samples(
        train_repr, train_data, pred_len, drop=padding
    )  # L46
    valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
    test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

    lr = fit_ridge(train_features, train_labels, valid_features, valid_labels)  # L51
    test_pred = lr.predict(test_features)  # L55

    metrics_norm = cal_metrics(test_pred, test_labels)  # L76 'norm'
    rec = {
        "mse": float(metrics_norm["MSE"]),
        "mae": float(metrics_norm["MAE"]),
        "alpha": float(lr.alpha),
        "n_train_samples": int(train_features.shape[0]),
        "n_eval": int(test_features.shape[0]),
    }
    if scaler is not None:  # L62-67：raw 尺度（差异 2：显式广播，数学同式）
        n_feat = test_data.shape[2]
        ori_shape = test_data.shape[0], -1, pred_len, n_feat  # L58-60
        pred4 = test_pred.reshape(ori_shape)
        labels4 = test_labels.reshape(ori_shape)
        pred_raw = pred4 * scaler.scale_ + scaler.mean_
        labels_raw = labels4 * scaler.scale_ + scaler.mean_
        metrics_raw = cal_metrics(pred_raw, labels_raw)  # L77 'raw'
        rec["mse_raw"] = float(metrics_raw["MSE"])
        rec["mae_raw"] = float(metrics_raw["MAE"])
    return rec
