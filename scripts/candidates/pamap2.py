"""PAMAP2 adapter（UCI id 231，换池调研候选）。

来源：UCI `pamap2+physical+activity+monitoring.zip` → 只用 `Protocol/subject10X.dat`
（9 个受试者，每人按固定协议做 12 类活动）。Optional 目录里那 5 个受试者的自选活动
不进——它们各人做的活动不同，混进来会让类别分布随受试者漂移。

格式：空格分隔 **54 列** @100 Hz。列布局（readme "Data collection protocol"）：

    0        timestamp(s)
    1        activityID
    2        心率(bpm)
    3-19     IMU hand      20-36  IMU chest      37-53  IMU ankle

每个 IMU 的 17 列依次是：温度×1、acc±16g×3、acc±6g×3、陀螺×3、磁力×3、orientation×4。

通道：3 IMU × (acc16g 3 + gyro 3 + mag 3) = **27 通道**。三处取舍都是数据本身逼的：
- 丢**心率**：只有 ~9 Hz、逐步序列里 ~90% 是 NaN，留着等于让 window_stream 丢掉几乎
  所有窗（铁律：不插值造假值）。
- 丢 **orientation**：官方 readme 明确写 "invalid in this data collection"。
- 丢 **acc±6g**：与 ±16g 量测同一物理量，剧烈活动（跑步/跳绳）时 ±6g 会削顶饱和；
  两份都留只是给 27 通道加一份高度相关的冗余。

已知坑：
1. **嵌套 zip**：外层 zip 里还有一个 `PAMAP2_Dataset.zip`，要二次解压。
2. **activityID=0 是 transient**（活动之间的过渡），不是一类活动，逐步标签置 -1。
3. **IMU 丢包造成 NaN**（无线传输掉帧）。不插值——含 NaN 的窗交给 window_stream 丢，
   丢窗数记进 meta。
4. 同一活动可能被 transient 段打断，故按 activityID **连续区间**分段后分别开窗。
5. **NaN 丢窗与活动强相关**（实测：walking 81% / rope_jumping 80% / nordic_walking 68%
   的窗被丢，而 lying 只丢 3.5%）——动态活动传感器晃动、掉帧多。后果是**落盘后的类别
   不平衡比是丢窗伪像**：按连续段切出的候选窗本身只有 ~5 倍不平衡，丢完变成 ~20 倍。
   meta 的 `nan_drop_by_class` 逐类记下 pure/kept，用统计表判候选时必须看这一栏。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autocls.data.candidates import (  # noqa: E402
    IGNORE_LABEL,
    RAW_ROOT,
    WINDOW_LEN,
    WINDOW_STRIDE,
    extract_archive,
    fetch_archive,
    window_stream,
)

_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"

SPEC = {
    "name": "PAMAP2",
    "domain": "HAR",
    "fs": 100,
    "url": _URL,
    "n_subjects": 9,
    "note": "UCI 231；只用 Protocol，27 通道（3 IMU × acc16g/gyro/mag）@100 Hz，"
            "activityID=0(transient) 置 -1，心率/orientation/acc6g 丢弃",
}

_RAW = RAW_ROOT / "pamap2"
_N_COLS = 54
_COL_ACT = 1

# 三个 IMU 块的起始列（块内第 0 列是温度）
_IMU_BASE = {"hand": 3, "chest": 20, "ankle": 37}


def _imu_cols(base: int) -> list[int]:
    """块内取 acc±16g(1-3) + 陀螺(7-9) + 磁力(10-12)，跳过温度/acc±6g/orientation。"""
    return [base + o for o in (1, 2, 3, 7, 8, 9, 10, 11, 12)]


SENSOR_COLS: list[int] = [c for loc in ("hand", "chest", "ankle")
                          for c in _imu_cols(_IMU_BASE[loc])]
CHANNEL_NAMES: list[str] = [
    f"{loc}_{mod}_{ax}"
    for loc in ("hand", "chest", "ankle")
    for mod in ("acc16g", "gyro", "mag")
    for ax in "xyz"
]

# activityID → 名称（readme "DescriptionOfActivities"）。Protocol 只出现其中 12 个，
# 全表照抄是为了万一某受试者文件里混进别的 ID 也能报出名字而不是数字。
ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 9: "watching_tv", 10: "computer_work",
    11: "car_driving", 12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
    19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
}


def _ensure_raw() -> Path:
    """下载 → 解外层 zip → 解嵌套 PAMAP2_Dataset.zip，返回含 subject10X.dat 的 Protocol 目录。"""
    zip_path = fetch_archive(_URL, _RAW / "pamap2.zip", timeout=300)
    outer = extract_archive(zip_path, _RAW / "extracted")

    protocol = _find_protocol(outer)
    if protocol is not None:
        return protocol

    inner_zips = sorted(outer.glob("**/*.zip"))          # 坑 1：嵌套 zip
    if not inner_zips:
        raise FileNotFoundError(f"{outer} 下既无 Protocol 目录也无嵌套 zip")
    inner = extract_archive(inner_zips[0], _RAW / "extracted_inner")
    protocol = _find_protocol(inner)
    if protocol is None:
        raise FileNotFoundError(f"{inner} 下找不到含 subject10X.dat 的 Protocol 目录")
    return protocol


def _find_protocol(root: Path) -> Path | None:
    """找含 subject*.dat 的 Protocol 目录（大小写与层级都不假定）。"""
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if d.name.lower() == "protocol" and list(d.glob("subject*.dat")):
            return d
    return None


def _contiguous_runs(labels: np.ndarray):
    """逐步标签 → [(start, stop)] 连续同值区间（半开）。

    同一活动被 transient 打断后是**两次独立记录**，必须各自开窗。
    """
    if labels.size == 0:
        return []
    cut = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], cut, [labels.size]))
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))


def _n_starts(T: int, win: int = WINDOW_LEN, stride: int = WINDOW_STRIDE) -> int:
    """长度 T 的段能切出几个窗（与 window_stream 的 range 口径一致）。"""
    return len(range(0, max(0, T - win + 1), stride))


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    protocol = _ensure_raw()
    files = sorted(protocol.glob("subject*.dat"))
    if not files:
        raise FileNotFoundError(f"{protocol} 下没有 subject*.dat")

    xs, ys, subj_of_win = [], [], []
    n_rows = n_rows_transient = 0
    n_starts_naive = n_starts_pure = n_kept = 0
    nan_step_hits = np.zeros(len(SENSOR_COLS), dtype=np.int64)
    pure_by_cls: Counter[int] = Counter()      # 坑 5：逐类记 pure/kept 才看得出丢窗伪像
    kept_by_cls: Counter[int] = Counter()

    for path in files:
        sid = int("".join(ch for ch in path.stem if ch.isdigit()))
        df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64,
                         na_values=["NaN", "nan"])
        if df.shape[1] != _N_COLS:
            raise ValueError(f"{path.name}: 期望 {_N_COLS} 列，实得 {df.shape[1]}")
        arr = df.to_numpy(dtype=np.float64)
        del df

        # 先选列再开窗：心率那种 ~90% 缺失的通道若留着，会把几乎所有窗都判成含 NaN
        x = np.ascontiguousarray(arr[:, SENSOR_COLS])
        lab = arr[:, _COL_ACT].astype(np.int64)
        del arr

        n_rows += lab.size
        n_rows_transient += int((lab == 0).sum())
        nan_step_hits += np.isnan(x).sum(axis=0)
        lab = np.where(lab == 0, IGNORE_LABEL, lab)      # 坑 2：transient → 忽略段

        n_starts_naive += _n_starts(lab.size)            # 无视分段的窗数（只作对照）
        for s, e in _contiguous_runs(lab):
            if lab[s] < 0:
                continue
            seg_x, seg_lab = x[s:e], lab[s:e]
            n_starts_pure += _n_starts(seg_x.shape[0])
            pure_by_cls[int(lab[s])] += _n_starts(seg_x.shape[0])
            w, wy = window_stream(seg_x, seg_lab)
            kept_by_cls[int(lab[s])] += int(w.shape[0])
            if w.shape[0]:
                # float32 落袋：PAMAP2 原始窗数 ~3e4，float64 堆着 concat 峰值近 2 GB，
                # 而驱动最终也要转 float32（契约允许 float32）
                xs.append(w.astype(np.float32))
                ys.append(wy)
                subj_of_win.append(np.full(w.shape[0], sid, dtype=np.int64))
                n_kept += w.shape[0]

    if not xs:
        raise RuntimeError("PAMAP2: 没有解析出任何窗口")

    X = np.concatenate(xs)
    y_raw = np.concatenate(ys)                            # 原始 activityID
    subj = np.concatenate(subj_of_win)

    # 稠密化到 0..k-1（按 activityID 升序）；驱动剔稀类时同样按升序 remap，故 meta 里
    # 的类名顺序对幸存类仍成立（PAMAP2 的 rope_jumping 极稀，常被驱动剔掉）
    present = np.unique(y_raw)
    remap = {int(v): i for i, v in enumerate(present.tolist())}
    y = np.asarray([remap[int(v)] for v in y_raw], dtype=np.int64)

    n_dropped_nan = n_starts_pure - n_kept                # 段内标签全同 → 唯一丢因是 NaN
    meta = {
        "class_names": [ACTIVITY_NAMES.get(int(v), f"activity_{int(v)}") for v in present],
        "class_counts_by_name": {
            ACTIVITY_NAMES.get(int(v), f"activity_{int(v)}"): int((y_raw == v).sum())
            for v in present
        },
        "n_subjects": len(files),
        "n_files": len(files),
        "channels": CHANNEL_NAMES,
        "n_channels_used": len(CHANNEL_NAMES),
        "sensor_cols": SENSOR_COLS,
        "n_rows_total": int(n_rows),
        "transient_step_ratio": round(n_rows_transient / max(1, n_rows), 4),
        "nan_step_ratio_max_channel": round(float(nan_step_hits.max()) / max(1, n_rows), 4),
        "n_windows_naive": int(n_starts_naive),           # 无视分段的窗数（对照）
        "n_windows_pure": int(n_starts_pure),             # 按连续段切出的候选窗
        "n_windows_kept": int(n_kept),
        "n_windows_dropped_nan": int(n_dropped_nan),
        "nan_drop_ratio": round(n_dropped_nan / max(1, n_starts_pure), 4),
        "nan_drop_by_class": {                            # 坑 5：不平衡比多半是这里来的
            ACTIVITY_NAMES.get(int(k), f"activity_{int(k)}"): {
                "pure": int(pure_by_cls[k]), "kept": int(kept_by_cls[k]),
                "nan_drop_ratio": round(
                    (pure_by_cls[k] - kept_by_cls[k]) / max(1, pure_by_cls[k]), 4),
            }
            for k in sorted(pure_by_cls)
        },
        "imbalance_ratio_pure": round(
            max(pure_by_cls.values()) / max(1, min(pure_by_cls.values())), 2),
        # 分段相对"整文件当一条流"损失的窗：transient 段 + 活动边界，非 NaN 所致
        "boundary_transient_drop_ratio": round(
            (n_starts_naive - n_starts_pure) / max(1, n_starts_naive), 4),
        "windows_per_subject": {int(s): int((subj == s).sum()) for s in np.unique(subj)},
    }
    return X, y, meta
