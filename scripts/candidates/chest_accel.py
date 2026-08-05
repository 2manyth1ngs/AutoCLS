"""ChestAccel：单胸挂加速度计活动识别（UCI id 287，7 类，15 受试者）。

来源：Casale et al.，UCI ML Repository。15 个 csv（每受试者一次连续记录），
逗号分隔 5 列：`序号, x, y, z, label`，fs=52 Hz，加速度是**未标定的原始计数值**
（量纲无物理单位，z-score 由驱动统一做，正好抹平）。

连续段口径：每个文件按 label 的**连续同值区间**分段，逐段 `window_stream`——
label 变化点就是活动切换点，跨切换点的窗物理上是混合活动，不该产样本。

已知坑：
- UCI 新式 static/public 直链（…/static/public/287/…）对该数据集 404，只有老式
  machine-learning-databases/00287/ 路径（含 %20 空格转义）还活着。
- label=0 是未标注段 → 统一置 IGNORE_LABEL(-1)，负标签区间直接跳过不开窗。
- 有效类 id 1..7（工作于电脑前 / 起立行走上下楼 / 站立 / 行走 / 上下楼 /
  边走边聊 / 站立聊天），驱动统一重编号 0..k-1。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autocls.data.candidates import (
    IGNORE_LABEL,
    RAW_ROOT,
    extract_archive,
    fetch_archive,
    window_stream,
)

SPEC = {
    "name": "ChestAccel",
    "domain": "HAR",
    "fs": 52,
    # 新式 static/public 直链 404，老式路径是唯一活着的官方地址
    "url": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00287/"
            "Activity%20Recognition%20from%20Single%20Chest-Mounted%20Accelerometer.zip"),
    "n_subjects": 15,
    "note": "未标定计数值；label 0=未标注→-1；每文件按 label 连续区间分段开窗",
}

_RAW_DIR = RAW_ROOT / "chest_accel"

# 按原始类 id 1..7 排（UCI 数据说明原文顺序）
_CLASS_NAMES = [
    "working_at_computer",
    "standing_up_walking_stairs",
    "standing",
    "walking",
    "going_up_down_stairs",
    "walking_and_talking",
    "talking_while_standing",
]


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(T,) 逐步标签 → [(start, end, label), …] 连续同值区间（end 开区间）。"""
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    return [(int(s), int(e), int(labels[s])) for s, e in zip(bounds[:-1], bounds[1:])]


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    zip_path = fetch_archive(SPEC["url"], _RAW_DIR / "chest_accel.zip", timeout=600)
    root = extract_archive(zip_path, _RAW_DIR / "extracted")

    # 档案里是 "Activity Recognition from …/1.csv..15.csv"，按纯数字文件名递归找，
    # 顺便挡掉 README 之类的杂项
    csvs = sorted(
        (p for p in root.rglob("*.csv") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    if len(csvs) != 15:
        raise FileNotFoundError(f"预期 15 个受试者 csv，实际找到 {len(csvs)} 个：{root}")

    xs, ys = [], []
    n_segments = 0
    for csv in csvs:
        df = pd.read_csv(csv, header=None)
        if df.shape[1] != 5:
            raise ValueError(f"{csv} 有 {df.shape[1]} 列，预期 5（序号,x,y,z,label）")
        arr = df.to_numpy(dtype=np.float64)
        x = arr[:, 1:4]                              # 丢序号列，取 x/y/z
        lab = arr[:, 4].astype(np.int64)
        lab = np.where(lab == 0, IGNORE_LABEL, lab)  # 0=未标注 → -1
        # 按 label 连续区间分段：每段内标签恒定，窗不会跨活动切换点
        for s, e, v in _runs(lab):
            if v < 0:
                continue
            Xw, yw = window_stream(x[s:e], lab[s:e])
            if len(Xw):
                xs.append(Xw)
                ys.append(yw)
                n_segments += 1
    if not xs:
        raise ValueError("ChestAccel：所有区间开窗后均无样本")

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    meta = {
        "class_names": _CLASS_NAMES,     # 按原始类 id 1..7 排
        "n_subjects": len(csvs),
        "n_files": len(csvs),
        "n_segments": n_segments,        # 产出过样本的连续区间数
    }
    return X, y, meta
