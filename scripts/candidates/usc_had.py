"""USC-HAD（USC Human Activity Dataset）adapter：14 受试者 × 12 活动 × 5 trial 的髋部 IMU。

来源：USC SIPI 官方档案 `https://sipi.usc.edu/had/USC-HAD.zip`（44.6 MB，实测可达）。
每个 `.mat` 是**一次 trial 的一段连续记录**（单一活动、无 null 段），字段 `sensor_readings`
形状 (T, 6)：acc_x/y/z（单位 g）+ gyro_x/y/z（单位 dps），fs=100 Hz（Readme §1）。
传感器只有一处（front-right-hip），与 PAMAP2/MHEALTH 那种多位置数据集互补。

已知坑（实际加载确认，不是抄论文）：
1. `Subject13/a11t4.mat` 的字段名拼错成 **`activity_numbr`**（840 个文件里唯一一个）。
   只认 `activity_number` 会静默少 1 条 trial，故按 `activity_number → activity_numbr →
   文件名 a<m>t<n>` 三级回退取活动号。
2. `activity` 这个**名字**字段在档案里不统一（同一活动出现 run/running、walking-down/
   walking-downstairs、sit/sitting… 共 24 种写法），**不能拿它当标签**。标签一律取
   `activity_number`（1–12，覆盖完整、每类恰好 70 个 trial），类名用 Readme §3 的规范表。
3. 数值字段全是**字符串**（`activity_number` 是 '1' 这种 <U1），要 int() 转。
4. trial 时长差异很大（T 600–13500 步，中位 3000），逐 trial 开窗即可，不需要额外处理。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.candidates import RAW_ROOT, extract_archive, fetch_archive, window_stream

SPEC = {
    "name": "USC-HAD",
    "domain": "HAR",
    "fs": 100,
    "url": "https://sipi.usc.edu/had/USC-HAD.zip",
    "n_subjects": 14,
    "note": "USC SIPI 官方 zip；髋部单点 IMU(acc 3 + gyro 3)@100Hz，12 活动 × 5 trial × 14 人；"
            "Subject13/a11t4.mat 字段名拼错为 activity_numbr，activity 名字段有 24 种写法故只用编号",
}

# Readme §3 的规范活动表：索引 = activity_number - 1
CLASS_NAMES = [
    "walking_forward", "walking_left", "walking_right", "walking_upstairs",
    "walking_downstairs", "running_forward", "jumping_up", "sitting",
    "standing", "sleeping", "elevator_up", "elevator_down",
]

_FNAME_RE = re.compile(r"a(\d+)t(\d+)\.mat$", re.IGNORECASE)


def _scalar(mat: dict, key: str):
    """MATLAB 标量字段 → python 值（档案里存的是 (1,) 的 <U 字符串数组）。"""
    v = mat.get(key)
    if v is None:
        return None
    arr = np.asarray(v).ravel()
    return arr[0] if arr.size else None


def _activity_number(mat: dict, path: Path) -> int:
    """三级回退取活动号：正规字段 → 拼错字段 → 文件名（见模块坑 1）。"""
    for key in ("activity_number", "activity_numbr"):
        v = _scalar(mat, key)
        if v is not None:
            return int(str(v).strip())
    m = _FNAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"{path}: 既无 activity_number 也无法从文件名解析活动号")
    return int(m.group(1))


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    import scipy.io as sio

    raw = RAW_ROOT / "usc_had"
    zip_path = fetch_archive(SPEC["url"], raw / "USC-HAD.zip", timeout=120)
    root = extract_archive(zip_path, raw / "extracted")

    files = sorted(root.glob("USC-HAD/Subject*/*.mat"))
    if not files:
        raise FileNotFoundError(f"{root} 下没找到 Subject*/*.mat，档案结构与预期不符")

    Xs, ys = [], []
    subjects, trials_used, steps = set(), 0, 0
    for f in files:
        mat = sio.loadmat(str(f))
        sr = mat.get("sensor_readings")
        if sr is None:
            raise ValueError(f"{f}: 缺 sensor_readings 字段")
        sr = np.asarray(sr, dtype=np.float64)
        if sr.ndim != 2 or sr.shape[1] != 6:
            raise ValueError(f"{f}: sensor_readings 应是 (T, 6)，实得 {sr.shape}")
        act = _activity_number(mat, f)
        if not 1 <= act <= 12:
            raise ValueError(f"{f}: 活动号 {act} 超出 1–12")

        # 一个 .mat = 一次 trial = 一段连续记录（单一活动），故整段一次开窗；
        # 逐 trial 调用即"按连续段开窗"，跨 trial 拼接会造出不存在的过渡窗
        labels = np.full(sr.shape[0], act, dtype=np.int64)
        xw, yw = window_stream(sr, labels)
        if xw.shape[0]:
            Xs.append(xw)
            ys.append(yw)
        subjects.add(f.parent.name)
        trials_used += 1
        steps += sr.shape[0]

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = {
        "class_names": CLASS_NAMES,
        "n_subjects": len(subjects),
        "n_files": len(files),
        "n_trials_used": trials_used,
        "n_raw_steps": int(steps),
        "channel_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "sensor_location": "front-right-hip",
        "units": "acc:g / gyro:dps",
    }
    return X, y, meta
