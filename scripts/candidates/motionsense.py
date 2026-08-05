"""MotionSense：iPhone 6s 裤兜 DeviceMotion 流（6 类日常动作，24 受试者）。

来源：Malekzadeh et al. 2019（repo mmalekzadeh/motion-sense），A_DeviceMotion_data
子集：15 个 trial 目录（dws/ups/wlk/jog/sit/std 各 2-3 次），每目录 24 个受试者 csv。
fs=50 Hz，12 通道 = attitude(3) + gravity(3) + rotationRate(3) + userAcceleration(3)。

连续段口径：**每个 (trial 目录, 受试者 csv) 是一次独立记录**，整段同一活动标签——
按段分别 `window_stream`，不跨段拼接（拼了会造出物理上不存在的过渡窗）。

已知坑：
- csv 第一列是 pandas 落盘带出来的无名行号列，不是数据，必须丢（index_col=0）。
- zip 里带 macOS 的 `__MACOSX/` 目录和 `._sub_*.csv` resource fork，都要跳过，
  不然会把 AppleDouble 二进制当 csv 读。
- 列顺序不做位置假设：按列名显式选取重排，列缺了直接报错而不是错位静默通过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autocls.data.candidates import RAW_ROOT, extract_archive, fetch_archive, window_stream

SPEC = {
    "name": "MotionSense",
    "domain": "HAR",
    "fs": 50,
    "url": "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip",
    "n_subjects": 24,
    "note": "DeviceMotion 12 通道；每 (trial, 受试者) csv 为一连续段；首列是行号要丢",
}

_RAW_DIR = RAW_ROOT / "motionsense"

# 固定通道顺序（= 原 csv 列序）：任何 csv 缺列/改名都会在 reindex 时报错
_COLS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
]

# trial 目录名前缀 → 类 id（id 只求稳定，驱动会统一重编号 0..k-1）
_ACT_IDS = {"dws": 0, "jog": 1, "sit": 2, "std": 3, "ups": 4, "wlk": 5}
_ACT_NAMES = ["downstairs", "jogging", "sitting", "standing", "upstairs", "walking"]


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    zip_path = fetch_archive(SPEC["url"], _RAW_DIR / "A_DeviceMotion_data.zip", timeout=600)
    root = extract_archive(zip_path, _RAW_DIR / "extracted")

    data_dir = root / "A_DeviceMotion_data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} 不存在，档案结构与预期不符")
    trial_dirs = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("__")
    )
    if not trial_dirs:
        raise FileNotFoundError(f"{data_dir} 下没有 trial 目录")

    xs, ys = [], []
    n_csv = 0
    subjects: set[str] = set()
    for td in trial_dirs:
        act = td.name.split("_")[0]
        if act not in _ACT_IDS:
            raise ValueError(f"未知活动前缀 {act!r}（目录 {td.name}），标签映射需要更新")
        lab = _ACT_IDS[act]
        for csv in sorted(td.glob("sub_*.csv")):
            if csv.name.startswith("._"):     # macOS resource fork
                continue
            df = pd.read_csv(csv, index_col=0)   # 首列 = 无名行号列，丢
            missing = [c for c in _COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{csv} 缺列 {missing}，通道顺序假设失效")
            x = df[_COLS].to_numpy(dtype=np.float64)
            # 一个 csv = 一次连续记录，整段同标签；逐步标签常量数组交给 window_stream
            Xw, yw = window_stream(x, np.full(x.shape[0], lab, dtype=np.int64))
            if len(Xw):
                xs.append(Xw)
                ys.append(yw)
            n_csv += 1
            subjects.add(csv.stem)
    if not xs:
        raise ValueError("MotionSense：所有连续段开窗后均无样本")

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    meta = {
        "class_names": _ACT_NAMES,               # 按类 id 0..5 排
        "n_subjects": len(subjects),
        "n_trial_dirs": len(trial_dirs),
        "n_files": n_csv,
    }
    return X, y, meta
