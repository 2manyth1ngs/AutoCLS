"""MHEALTH adapter（UCI id 319，换池调研候选）。

来源：UCI `mhealth+dataset.zip` → `MHEALTHDATASET/mHealth_subject{1..10}.log`。
10 个受试者、12 类活动、所有模态统一 **50 Hz**（README §1），制表符分隔 24 列：
前 23 列传感器（胸部 acc×3 + ECG×2、左踝 acc/gyro/mag×9、右腕 acc/gyro/mag×9），
末列 label 0..12，0 = null。

通道：**23 列全用**。ECG 两导联留着不是疏忽——它和 IMU 是不同物理量、不同频谱
（PAMAP2 的心率之所以丢是因为 9 Hz + 90% 缺失，这里 ECG 是 50 Hz 满格），换池调研
要的正是"随机初始化基线不贴顶"的数据集，混合模态反而是加分项。实际列写进 meta。

已知坑：
1. README 的列表把 "Column 12" 误写成 "Column 13"（左踝磁力计 X）——列**顺序**没错，
   只是文档笔误，按 1..23 顺序取即可。
2. 分隔符文档说 tab，实测确为 tab；这里仍按任意空白切分（`sep=r"\\s+"`），对个别
   行用空格的情况也不会炸。
3. label=0 是 null（活动之间的过渡/静置），逐步标签置 -1 → 不产样本。
4. 同一 label 在文件里会被 null 段切成多段，故按 label **连续区间**分段后分别开窗，
   绝不跨段拼接（会造出物理上不存在的过渡窗）。
"""
from __future__ import annotations

import sys
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

_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"

SPEC = {
    "name": "MHEALTH",
    "domain": "HAR",
    "fs": 50,
    "url": _URL,
    "n_subjects": 10,
    "note": "UCI 319；23 通道（胸 acc+ECG / 左踝 IMU / 右腕 IMU）@50 Hz，label=0 为 null 段已置 -1",
}

_RAW = RAW_ROOT / "mhealth"
_N_COLS = 24            # 23 传感器 + 1 label
_N_SENSOR = 23
_SUBJECTS = range(1, 11)

# label 1..12 → 名称（README §2 的 L1..L12）。索引 i 对应 label i+1。
CLASS_NAMES = [
    "standing_still", "sitting_relaxing", "lying_down", "walking",
    "climbing_stairs", "waist_bends_forward", "frontal_elevation_of_arms",
    "knees_bending", "cycling", "jogging", "running", "jump_front_back",
]

# 通道名（列 1..23，与 README 顺序一致；笔误的 col12 按左踝磁力计 X 补回）
CHANNEL_NAMES = (
    [f"chest_acc_{a}" for a in "xyz"]
    + ["chest_ecg_1", "chest_ecg_2"]
    + [f"lankle_acc_{a}" for a in "xyz"]
    + [f"lankle_gyro_{a}" for a in "xyz"]
    + [f"lankle_mag_{a}" for a in "xyz"]
    + [f"rarm_acc_{a}" for a in "xyz"]
    + [f"rarm_gyro_{a}" for a in "xyz"]
    + [f"rarm_mag_{a}" for a in "xyz"]
)


def _ensure_raw() -> Path:
    """下载 + 解压到 data/candidates/raw/mhealth/extracted，返回 MHEALTHDATASET 目录。"""
    zip_path = fetch_archive(_URL, _RAW / "mhealth.zip", timeout=180)
    out = extract_archive(zip_path, _RAW / "extracted")
    for cand in (out / "MHEALTHDATASET", out):
        if list(cand.glob("mHealth_subject*.log")):
            return cand
    raise FileNotFoundError(f"{out} 下找不到 mHealth_subject*.log")


def _contiguous_runs(labels: np.ndarray):
    """逐步标签 → [(start, stop)] 连续同值区间（半开区间）。

    同一活动被 null 段打断后是**两次独立记录**，必须各自开窗。
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
    root = _ensure_raw()

    xs, ys, subj_of_win = [], [], []
    n_rows = n_rows_null = 0
    n_starts_naive = n_starts_pure = n_kept = 0
    n_files = 0

    for sid in _SUBJECTS:
        path = root / f"mHealth_subject{sid}.log"
        if not path.exists():
            continue
        # sep=r"\s+" 兼容 tab/空格；文件里没有表头
        df = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64,
                         na_values=["NaN", "nan"])
        if df.shape[1] != _N_COLS:
            raise ValueError(f"{path.name}: 期望 {_N_COLS} 列，实得 {df.shape[1]}")
        n_files += 1
        arr = df.to_numpy(dtype=np.float64)
        x = arr[:, :_N_SENSOR]
        lab = arr[:, _N_SENSOR].astype(np.int64)

        n_rows += lab.size
        n_rows_null += int((lab == 0).sum())
        lab = np.where(lab == 0, IGNORE_LABEL, lab)      # null → 忽略段

        n_starts_naive += _n_starts(lab.size)            # 若无视分段能切多少（只作对照）
        for s, e in _contiguous_runs(lab):
            if lab[s] < 0:
                continue
            seg_x, seg_lab = x[s:e], lab[s:e]
            n_starts_pure += _n_starts(seg_x.shape[0])
            w, wy = window_stream(seg_x, seg_lab)
            if w.shape[0]:
                xs.append(w)
                ys.append(wy)
                subj_of_win.append(np.full(w.shape[0], sid, dtype=np.int64))
                n_kept += w.shape[0]

    if not xs:
        raise RuntimeError("MHEALTH: 没有解析出任何窗口")

    X = np.concatenate(xs)
    y_raw = np.concatenate(ys)                            # 原始 label 1..12
    subj = np.concatenate(subj_of_win)

    # 稠密化到 0..k-1（升序，与 CLASS_NAMES 顺序一致）；驱动的剔稀类会再 remap 一次，
    # 但它同样按升序保序，故 meta 里的类名顺序对幸存类仍成立
    present = np.unique(y_raw)
    remap = {int(v): i for i, v in enumerate(present.tolist())}
    y = np.asarray([remap[int(v)] for v in y_raw], dtype=np.int64)

    counts = {CLASS_NAMES[int(v) - 1]: int((y_raw == v).sum()) for v in present}
    n_dropped_nan = n_starts_pure - n_kept                # 段内标签全同，唯一丢因就是 NaN
    meta = {
        "class_names": [CLASS_NAMES[int(v) - 1] for v in present],
        "class_counts_by_name": counts,
        "n_subjects": n_files,
        "n_files": n_files,
        "channels": CHANNEL_NAMES,
        "n_channels_used": len(CHANNEL_NAMES),
        "n_rows_total": int(n_rows),
        "null_step_ratio": round(n_rows_null / max(1, n_rows), 4),
        "n_windows_naive": int(n_starts_naive),           # 无视分段的窗数（对照）
        "n_windows_pure": int(n_starts_pure),             # 按连续段切出的候选窗
        "n_windows_kept": int(n_kept),
        "n_windows_dropped_nan": int(n_dropped_nan),
        "nan_drop_ratio": round(n_dropped_nan / max(1, n_starts_pure), 4),
        # 分段相对"整文件当一条流"损失的窗：null 段 + 活动边界，非 NaN 所致
        "boundary_null_drop_ratio": round(
            (n_starts_naive - n_starts_pure) / max(1, n_starts_naive), 4),
        "windows_per_subject": {int(s): int((subj == s).sum()) for s in np.unique(subj)},
    }
    return X, y, meta
