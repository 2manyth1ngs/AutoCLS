"""Daphnet FoG（Freezing of Gait）adapter：帕金森患者步态冻结检测，10 受试者 × 3 处加速度计。

来源：UCI id 245 `https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip`
（21 MB，实测可达）。`dataset_fog_release/dataset/S<pp>R<rr>.txt` 共 17 个记录文件，
空格分隔 11 列 @64 Hz（doc/documentation.html）：

    col 0      时间戳（ms）
    col 1–3    踝/小腿加速度 forward / vertical / lateral [mg]
    col 4–6    大腿加速度   forward / vertical / lateral [mg]
    col 7–9    躯干加速度   forward / vertical / lateral [mg]
    col 10     标注：0=不在实验中（未标注）/ 1=实验中无冻结 / 2=冻结

时间戳实测严格连续（步距恒 15/16 ms，无跳变），故一个文件 = 一次连续记录。

与其它 HAR 候选的差异（这正是把它放进调研的理由）：类别不是"活动种类"而是**病理事件**，
且**严重不平衡**——原始逐步标注里 freeze 只占已标注样本的 ~9.7%（110k / 1.14M 步），
两个受试者（S04、S10）**完全没有 freeze**。此处如实保留该分布，**不做重采样/不做类平衡**：
不平衡本身是这个数据集的性质，掩盖它会让候选统计表失真。

已知坑：
1. label==0 是"不在实验中"（装传感器、debrief 等），**不是**某个类。按铁律置 -1，
   `window_stream` 不会为负标签产样本。文件开头那段 0 标注里传感器读数全是 0，
   若误当成类别会喂进大量常值窗。
2. 必须**按标注的连续区间分段**分别开窗：一个文件里 no-freeze / freeze / 未标注交替
   出现，整文件一次开窗会让窗口起点被前面的段"错位"，且跨标注边界的混合窗虽然会被
   `window_stream` 丢掉，冻结段（多数只有几秒）能出的纯窗会明显变少。
3. 已标注区间内没有全零传感器行、没有 NaN（实测），故无需插值——本项目也禁止插值。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.candidates import (
    IGNORE_LABEL,
    RAW_ROOT,
    extract_archive,
    fetch_archive,
    window_stream,
)

SPEC = {
    "name": "Daphnet",
    "domain": "HAR",
    "fs": 64,
    "url": "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    "n_subjects": 10,
    "note": "UCI id245；踝/大腿/躯干三处三轴加速度(9ch)@64Hz，二分类 no-freeze vs freeze；"
            "标注 0=不在实验中→置 -1；freeze 极稀（占已标注步的 ~9.7%，S04/S10 无 freeze），"
            "不平衡如实保留不重采样",
}

# 标注取值 → 类名（0 已在解析时置 -1，不出现在这里）；驱动会把 {1,2} 重编号为 {0,1}，
# 排序后顺序与本表一致
CLASS_NAMES = ["no_freeze", "freeze"]

_SUBJ_RE = re.compile(r"S(\d+)R(\d+)\.txt$", re.IGNORECASE)


def _label_runs(labels: np.ndarray):
    """逐步标签 → [(start, stop)] 的极大常值区间（左闭右开）。

    每个区间就是一段"同一标注下的连续记录"，开窗以此为单位（见模块坑 2）。
    """
    if labels.size == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [labels.size]))
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    raw = RAW_ROOT / "daphnet"
    zip_path = fetch_archive(SPEC["url"], raw / "daphnet.zip", timeout=120)
    root = extract_archive(zip_path, raw / "extracted")

    files = sorted(root.glob("dataset_fog_release/dataset/S*R*.txt"))
    if not files:
        raise FileNotFoundError(f"{root} 下没找到 dataset_fog_release/dataset/S*R*.txt")

    Xs, ys = [], []
    subjects, steps, segs = set(), 0, 0
    step_counts = {0: 0, 1: 0, 2: 0}
    for f in files:
        arr = np.loadtxt(f, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 11:
            raise ValueError(f"{f}: 应是 11 列（时间 + 9 通道 + 标注），实得 {arr.shape}")
        sig = arr[:, 1:10]                      # 丢时间戳列，只留 9 个加速度通道
        raw_lab = arr[:, 10].astype(np.int64)
        for v in np.unique(raw_lab):
            if int(v) not in step_counts:
                raise ValueError(f"{f}: 未知标注取值 {v}（文档只定义 0/1/2）")
            step_counts[int(v)] += int((raw_lab == v).sum())

        # 0 = 不在实验中 → IGNORE_LABEL，不产样本（坑 1）
        labels = np.where(raw_lab == 0, IGNORE_LABEL, raw_lab)
        for s, e in _label_runs(labels):
            if labels[s] < 0 or e - s < 1:
                continue
            xw, yw = window_stream(sig[s:e], labels[s:e])
            if xw.shape[0]:
                Xs.append(xw)
                ys.append(yw)
            segs += 1

        m = _SUBJ_RE.search(f.name)
        subjects.add(m.group(1) if m else f.stem)
        steps += arr.shape[0]

    if not Xs:
        raise ValueError("Daphnet: 没有产出任何窗口（解析或标注处理有误）")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    labeled = step_counts[1] + step_counts[2]
    meta = {
        "class_names": CLASS_NAMES,
        "n_subjects": len(subjects),
        "n_files": len(files),
        "n_label_segments": segs,
        "n_raw_steps": int(steps),
        "step_label_counts": {"not_in_experiment": step_counts[0],
                              "no_freeze": step_counts[1], "freeze": step_counts[2]},
        "freeze_step_fraction": round(step_counts[2] / labeled, 4) if labeled else None,
        "channel_names": ["ankle_fwd", "ankle_vert", "ankle_lat",
                          "thigh_fwd", "thigh_vert", "thigh_lat",
                          "trunk_fwd", "trunk_vert", "trunk_lat"],
        "units": "mg",
    }
    return X, y, meta
