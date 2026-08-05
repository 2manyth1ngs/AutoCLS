"""OPPORTUNITY 活动识别（UCI id 226）adapter —— 换池调研候选。

数据：4 受试者 × (ADL1–5 + Drill) = 24 个 `.dat`，空格分隔 **250 列** @30 Hz，
记录厨房场景的日常活动。本 adapter 只取 **Locomotion** 标签轨（官方 1-indexed 第 244
列 → 0-indexed 243），取值 0=null / 1=Stand / 2=Walk / 4=Sit / 5=Lie。

选它的理由（换池判据）：与 UCI-HAR 同为姿态/移动四分类，但**传感器摆位更多、场景更
自然**（受试者在厨房自由活动而非按指令做动作），随机初始化基线不像 UEA 那批那么容易
贴顶。同时 Locomotion 只有 4 类且 Lie 天然稀少，不平衡比是真实的，不是人为构造。

已知坑（都在下面代码里显式处理，不掩盖）：
1. **大量 NaN**：部分传感器整段掉线、蓝牙丢包。**不插值**——插值会造假值，而被测量
   恰恰是"表征从真实信号里学到什么"。含 NaN 的窗由 `window_stream` 丢弃，丢窗比例
   写进 meta（这个数字本身就是结论的一部分：它决定候选还剩多少可用样本）。
2. 通道不能全要：250 列里有整段近乎全 NaN 的传感器（如某些 run 里的鞋载 IMU）。这里
   固定选 **5 个身上 IMU（BACK / RUA / RLA / LUA / LLA）的 acc+gyro+mag = 45 通道**，
   再按实测 NaN 比例剔掉坏到没法用的列（阈值 `NAN_COL_MAX`），选中列号与各自 NaN
   比例全部写进 meta，便于复核。
3. 标签列取值不连续（1,2,4,5）——不在这里重编号，驱动 `drop_rare_classes` 统一映射
   到 0..k-1（契约：adapter 只负责"把原始格式读对"）。
4. UCI 的 static zip 里还套了一层 `OpportunityUCIDataset.zip`，要二次解压。
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.candidates import (  # noqa: E402
    RAW_ROOT,
    WINDOW_LEN,
    WINDOW_STRIDE,
    extract_archive,
    fetch_archive,
    window_stream,
)

SPEC = {
    "name": "Opportunity",
    "domain": "HAR",
    "fs": 30,
    "url": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    "n_subjects": 4,
    "note": "UCI 226；只用 Locomotion 标签轨（244 列，0=null）与 5 个身上 IMU；"
            "传感器丢包导致大量 NaN，不插值、含 NaN 的窗直接丢",
}

RAW_DIR = RAW_ROOT / "opportunity"
ARCHIVE = RAW_DIR / "opportunity.zip"

# —— 列口径（0-indexed；官方 column_names.txt 是 1-indexed，这里已减 1）——
# 5 个身上 IMU，每个 9 列 acc(x,y,z) + gyro(x,y,z) + mag(x,y,z)；各 IMU 后面紧跟的
# 4 列 Quaternion 不要（四元数与 acc/gyro 冗余，且不同 run 的参考系不一致）。
IMU_BLOCKS = {          # 传感器名 → 该 IMU 9 列的 1-indexed 起始列
    "BACK": 38,
    "RUA": 51,
    "RLA": 64,
    "LUA": 77,
    "LLA": 90,
}
IMU_AXES = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"]
LABEL_COL = 244 - 1     # Locomotion（官方 1-indexed 244）
N_COLS_EXPECTED = 250
NULL_LABEL = 0          # Locomotion 的 null 取值 → 置 IGNORE_LABEL(-1)

# 单列 NaN 比例超过此值就整列剔除：一个整段掉线的传感器会让**每一个**含它的窗都被
# 丢掉（NaN 沿通道传染），留着等于把整个数据集废掉。阈值定在 0.5 而非更严，是为了
# 让"这列到底有多脏"由 meta 里的实测比例说话，而不是被一个紧阈值悄悄筛没。
NAN_COL_MAX = 0.5

CLASS_NAMES = {1: "Stand", 2: "Walk", 4: "Sit", 5: "Lie"}


def _ensure_data() -> Path:
    """下载 + 解压（含内层 OpportunityUCIDataset.zip），返回含 .dat 的根目录。"""
    fetch_archive(SPEC["url"], ARCHIVE, timeout=300)
    out = extract_archive(ARCHIVE, RAW_DIR / "extracted")
    if not list(out.rglob("*.dat")):
        # UCI static 包外面套了一层：内层 zip 才是真正的数据集
        for inner in sorted(out.rglob("*.zip")):
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)
    dats = sorted(out.rglob("S?-*.dat"))
    if not dats:
        raise FileNotFoundError(f"{out} 下未找到 S?-*.dat（解压结构与预期不符）")
    # 列口径全靠 1-indexed 硬编码列号，档案一旦改版会**静默错位**（读到的还是数，只是
    # 换了传感器）。这里用首行列数当格式护栏：对不上就炸，不带着错列号往下跑。
    with open(dats[0], encoding="utf-8") as f:
        n_cols = len(f.readline().split())
    if n_cols != N_COLS_EXPECTED:
        raise ValueError(f"{dats[0].name}: {n_cols} 列 != 期望 {N_COLS_EXPECTED}，列口径失效")
    return out


def _selected_columns() -> tuple[list[int], list[str]]:
    """(0-indexed 列号, 通道名) —— 5 IMU × 9 轴 = 45 通道，顺序固定可复现。"""
    cols, names = [], []
    for sensor, start1 in IMU_BLOCKS.items():
        for k, axis in enumerate(IMU_AXES):
            cols.append(start1 - 1 + k)
            names.append(f"{sensor}_{axis}")
    return cols, names


def _segments(labels: np.ndarray):
    """按标签的连续区间切段：(start, stop) 半开区间，逐段返回。

    铁律：开窗必须按连续段调用 `window_stream`。这里"段"= 同一活动的一次连续执行；
    跨段拼接会造出物理上不存在的活动过渡窗。同一段内标签恒定，因此不会有混合标签窗。
    """
    if labels.size == 0:
        return
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], change, [labels.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(a), int(b)


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    import pandas as pd

    root = _ensure_data()
    files = sorted(root.rglob("S?-*.dat"))
    cols, ch_names = _selected_columns()
    usecols = sorted(set(cols + [LABEL_COL]))

    # —— 第一遍：只统计各候选列的 NaN 比例，据此定最终通道集 ——
    # 先统计再选列（而不是拍脑袋选），是因为哪个传感器掉线随 run 变化，纸面文档不可靠。
    frames = {}
    n_rows_total = 0
    nan_counts = np.zeros(len(cols), dtype=np.int64)
    for f in files:
        df = pd.read_csv(f, sep=r"\s+", header=None, usecols=usecols,
                         dtype=np.float64, na_values=["NaN", "nan"])
        if df.shape[1] != len(usecols):
            raise ValueError(f"{f.name}: 读到 {df.shape[1]} 列，期望 {len(usecols)}")
        arr = df[cols].to_numpy(dtype=np.float64)
        lab = df[LABEL_COL].to_numpy()
        frames[f.name] = (arr, lab)
        nan_counts += np.isnan(arr).sum(axis=0)
        n_rows_total += arr.shape[0]
    nan_ratio = nan_counts / max(1, n_rows_total)

    keep = [i for i, r in enumerate(nan_ratio) if r <= NAN_COL_MAX]
    if not keep:
        raise ValueError("所有候选通道 NaN 比例都超阈值，数据不可用")
    kept_cols = [cols[i] for i in keep]
    kept_names = [ch_names[i] for i in keep]

    # —— 第二遍：按连续段开窗 ——
    Xs, ys = [], []
    n_pure_pos = 0          # 标签合法（≥0）且窗完整落在段内的候选窗数
    n_short_seg = 0         # 短于一个窗、直接产不出样本的段
    n_seg = 0
    subjects = set()
    for fname, (arr, lab_raw) in frames.items():
        subjects.add(fname.split("-")[0])
        x = arr[:, keep]
        lab = np.where(np.isnan(lab_raw), NULL_LABEL, lab_raw).astype(np.int64)
        lab = np.where(lab == NULL_LABEL, -1, lab)      # null → 忽略段
        for a, b in _segments(lab):
            n_seg += 1
            seg_x, seg_lab = x[a:b], lab[a:b]
            n_pos = len(range(0, max(0, seg_lab.size - WINDOW_LEN + 1), WINDOW_STRIDE))
            if seg_lab[0] >= 0:
                n_pure_pos += n_pos
                if n_pos == 0:
                    n_short_seg += 1
            if n_pos == 0:
                continue
            xw, yw = window_stream(seg_x, seg_lab)
            if xw.shape[0]:
                Xs.append(xw)
                ys.append(yw)
    if not Xs:
        raise ValueError("开窗后无样本（NaN 或段长不足）")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    dropped_nan = n_pure_pos - X.shape[0]

    meta = {
        "n_files": len(files),
        "n_subjects": len(subjects),
        "n_segments": n_seg,
        "label_col_0indexed": LABEL_COL,
        "label_track": "Locomotion",
        "class_names": [CLASS_NAMES.get(int(v), str(int(v)))
                        for v in sorted(np.unique(y).tolist())],
        "channels": kept_names,
        "channel_cols_0indexed": kept_cols,
        "channel_nan_ratio": {ch_names[i]: round(float(nan_ratio[i]), 5)
                              for i in range(len(cols))},
        "channels_dropped_high_nan": [ch_names[i] for i in range(len(cols))
                                      if i not in keep],
        "nan_col_max": NAN_COL_MAX,
        "n_windows_candidate": int(n_pure_pos),
        "n_windows_dropped_nan": int(dropped_nan),
        "nan_drop_ratio": round(dropped_nan / max(1, n_pure_pos), 4),
        "n_segments_shorter_than_window": n_short_seg,
        "interpolated": False,
    }
    return X, y, meta
