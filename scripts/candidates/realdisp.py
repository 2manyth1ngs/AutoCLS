"""REALDISP 活动识别（UCI id 305）adapter —— 换池调研候选。

数据：17 受试者 × 9 个 IMU（LC/RC/LT/RT/BACK/LLA/RLA/LUA/RUA）@50 Hz，制表符分隔
**120 列**：0–1 时间戳(秒, 微秒)，2–118 = 9 IMU × 13 列（acc×3, gyro×3, mag×3,
quaternion×4），119 = activityID（0=null，1..33 共 33 类健身动作）。

**只用 `_ideal` 变体**（理想佩戴位置）。`_self`（受试者自行佩戴）与 `_mutual{N}`
（人为位移）是这个数据集的主打卖点——传感器位移鲁棒性。但对本项目它们是**污染**：
把"佩戴位移"混进来会让同一动作的样本分布多峰，难度来源变成传感器摆位而非活动本身，
和"这个数据集对 CLS 差异敏不敏感"这个被测量纠缠在一起。

选它的理由（换池判据）：33 类、类间差异细（"cycling" vs "elliptical bike"、多种
上肢摆动），比 UCI-HAR 的 6 类难得多；随机初始化基线不太可能贴顶。9 个佩戴点也给了
远高于 UCI-HAR 的通道数。

通道口径：**9 IMU × (acc 3 + gyro 3) = 54 通道**。丢 quaternion（与 acc/gyro 冗余、
参考系依赖标定）和 mag（磁力计受室内铁磁干扰、跨受试者不可比），既控规模也去噪。

已知坑：
1. **档案比文档声称的大一个数量级**：UCI 页面给的量级是 ~200 MB，实测 zip **2.67 GB**、
   解压后 **7.04 GB**（47 个成员，含全部 _self/_mutual 变体）。所以这里**只选择性解压
   17 个 `*_ideal.log`（2.58 GB）**，不用整包解压——省 4.5 GB 磁盘，也不落地我们明确
   排除的数据。
2. `_mutual` 文件名带受试者编号（如 `subject3_mutual7.log`），glob 时若只写
   `subject*_*.log` 会把它们捞进来——这里显式只匹配 `*_ideal.log`。
3. 时间戳列不能当特征（单调递增，z-score 后仍是斜坡，会被 encoder 当成最强信号）。
4. activityID=0 是活动间的休息/过渡 → 置 -1 忽略，不产样本。
5. 标签取值 1..33 不从 0 开始——不在这里重编号，驱动 `drop_rare_classes` 统一映射。
6. 窗口数组按 **float32** 累积而非 window_stream 默认的 float64：2.58 GB 原始流开出的窗
   若用 float64 会占 ~2 GB，叠加驱动里 z-score 的中间副本容易把 16 GB 机器打爆。精度上
   无损失——传感器读数有效位远少于 float32，且驱动的 z-score 统计仍在 float64 上做。
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
    fetch_archive,
    window_stream,
)

SPEC = {
    "name": "REALDISP",
    "domain": "HAR",
    "fs": 50,
    "url": "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip",
    "n_subjects": 17,
    "note": "UCI 305；只用 _ideal（理想佩戴）变体，排除 _self/_mutual 的传感器位移；"
            "9 IMU × (acc+gyro) = 54 通道，丢 quaternion 与 mag",
}

RAW_DIR = RAW_ROOT / "realdisp"
ARCHIVE = RAW_DIR / "realdisp.zip"

# —— 列口径（0-indexed）——
TS_COLS = 2                 # 0=秒, 1=微秒：只用来定位，不进特征
BLOCK = 13                  # 每个 IMU 13 列
N_IMU = 9
LABEL_COL = 119
N_COLS_EXPECTED = 120
NULL_LABEL = 0

# 官方文档给的 9 个佩戴点顺序（列块顺序即此顺序）
SENSORS = ["RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC"]
# 每个 13 列块内的偏移：0-2 acc / 3-5 gyro / 6-8 mag / 9-12 quaternion
AXES_KEPT = {0: "accX", 1: "accY", 2: "accZ", 3: "gyroX", 4: "gyroY", 5: "gyroZ"}


def _ensure_data() -> Path:
    """下载 + **只解压 `*_ideal.log`**，返回含 .log 的目录。

    不用 `extract_archive`（它整包解压）：整包 7.04 GB，其中 4.5 GB 是我们明确排除的
    _self/_mutual 变体，落地纯属浪费。逐成员解压并跳过已存在的文件，重跑不重解。
    """
    fetch_archive(SPEC["url"], ARCHIVE, timeout=300)
    out = RAW_DIR / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        members = [i for i in z.infolist() if i.filename.endswith("_ideal.log")]
        if not members:
            raise FileNotFoundError(f"{ARCHIVE} 内未找到 *_ideal.log（档案结构与预期不符）")
        for info in members:
            target = out / Path(info.filename).name
            if target.exists() and target.stat().st_size == info.file_size:
                continue                        # 已解压且完整，跳过
            with z.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 22):
                    dst.write(chunk)            # 分块写：单文件最大 183 MB，不整份进内存
    # 列口径全靠硬编码列号，档案改版会**静默错位**（读到的还是数，只是换了传感器）。
    # 用首行列数当格式护栏：对不上就炸，不带着错列号往下跑。
    first = sorted(out.glob("*_ideal.log"))[0]
    with open(first, encoding="utf-8") as f:
        n_cols = len(f.readline().rstrip("\n").split("\t"))
    if n_cols != N_COLS_EXPECTED:
        raise ValueError(f"{first.name}: {n_cols} 列 != 期望 {N_COLS_EXPECTED}，列口径失效")
    return out


def _selected_columns() -> tuple[list[int], list[str]]:
    """(0-indexed 列号, 通道名) —— 9 IMU × 6 轴 = 54 通道，顺序固定可复现。"""
    cols, names = [], []
    for i in range(N_IMU):
        base = TS_COLS + i * BLOCK
        for off, axis in AXES_KEPT.items():
            cols.append(base + off)
            names.append(f"{SENSORS[i]}_{axis}")
    return cols, names


def _segments(labels: np.ndarray):
    """按 activityID 的连续区间切段：(start, stop) 半开区间。

    铁律：开窗必须按连续段调用 `window_stream`。同一段内标签恒定，因此不会有跨活动的
    混合窗；活动间的 null 段（休息）自成一段且标签为 -1，不产样本。
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
    files = sorted(root.rglob("*_ideal.log"))
    cols, ch_names = _selected_columns()
    usecols = sorted(set(cols + [LABEL_COL]))

    Xs, ys = [], []
    n_pure_pos = 0          # 标签合法（≥0）且窗完整落在段内的候选窗数
    n_short_seg = 0
    n_seg = 0
    n_rows_total = 0
    nan_counts = np.zeros(len(cols), dtype=np.int64)
    for f in files:
        df = pd.read_csv(f, sep="\t", header=None, usecols=usecols,
                         dtype=np.float64, na_values=["NaN", "nan"])
        if df.shape[1] != len(usecols):
            raise ValueError(f"{f.name}: 读到 {df.shape[1]} 列，期望 {len(usecols)}")
        x = df[cols].to_numpy(dtype=np.float64)
        lab = df[LABEL_COL].to_numpy()
        lab = np.where(np.isnan(lab), NULL_LABEL, lab).astype(np.int64)
        lab = np.where(lab == NULL_LABEL, -1, lab)      # null/休息 → 忽略段
        nan_counts += np.isnan(x).sum(axis=0)
        n_rows_total += x.shape[0]

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
                Xs.append(xw.astype(np.float32))     # 见文件头坑 6：float64 会撑爆内存
                ys.append(yw)
        del df, x, lab                                # 单个 .log 最大 183 MB，及时释放
    if not Xs:
        raise ValueError("开窗后无样本（NaN 或段长不足）")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    dropped_nan = n_pure_pos - X.shape[0]
    nan_ratio = nan_counts / max(1, n_rows_total)

    meta = {
        "n_files": len(files),
        "n_subjects": len(files),           # 每受试者恰一份 _ideal
        "n_segments": n_seg,
        "variant": "ideal",
        "label_col_0indexed": LABEL_COL,
        "channels": ch_names,
        "channel_cols_0indexed": cols,
        "channel_nan_ratio_max": round(float(nan_ratio.max()), 6),
        "n_windows_candidate": int(n_pure_pos),
        "n_windows_dropped_nan": int(dropped_nan),
        "nan_drop_ratio": round(dropped_nan / max(1, n_pure_pos), 4),
        "n_segments_shorter_than_window": n_short_seg,
        "interpolated": False,
    }
    return X, y, meta
