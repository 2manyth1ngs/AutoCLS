"""HHAR — Heterogeneity Activity Recognition（UCI id 344）adapter。

为什么选它：9 用户 × 6 类日常活动（bike/sit/stand/walk/stairsup/stairsdown）的手机
IMU 连续流，与 UCI-HAR 同域但**不是**同一数据集（目标池的 UCI-HAR 是 id 240、HAPT 是
id 341，铁律 6 均不可碰；本数据集 id 344 由 Stisen et al. 2015 独立采集，无泄漏）。

设计意图与已知坑：
- **采样率不统一是数据集的设计意图**（异构性研究）：4 种手机 × 各 2 台，标称
  50–200 Hz 不等，且不同机型 Creation_Time 时钟基准都不一致。把全部机型混进同一
  个池，"机型/采样率"会成为与活动标签部分相关的隐藏混杂因子，因此走**单机型**路线：
  只取 Phones_accelerometer.csv 里 Model == 'nexus4'（标称 200 Hz、两台设备
  nexus4_1/nexus4_2、9 名用户全覆盖），**不重采样、不插值**——重采样会引入插值伪影，
  影响的正是"表征学到什么"这个被测量本身。
- Creation_Time 是**纳秒**级传感器时间戳（Arrival_Time 才是毫秒墙钟，抖动大，弃用）。
  实测采样率按相邻 Creation_Time 中位差算并写进 meta，不抄论文标称值。
- Phones_accelerometer.csv 约 1300 万行：必须 usecols + dtype + chunksize 分块读并
  在块内先过滤机型，整读会吃掉数 GB 内存。
- gt 缺失/"null"（受试者未在做目标活动）→ 逐步标签置 IGNORE_LABEL(-1)，由
  window_stream 统一丢弃，不产样本。
- 连续段 = 同一 (User, Device) 按 Creation_Time 排序后，相邻差 > 10× 中位差处断开
  （录制中途暂停/换活动间隙会留时间跳变）。跨段/跨用户/跨设备绝不拼接：会造出
  物理上不存在的过渡窗（契约铁律）。
- UCI 打包是**嵌套 zip**：外层 zip 里还有 "Activity recognition exp.zip" 等，解一层
  后若找不到 csv 要再解内层（本模块用 rglob 容错，两种打包布局都能吃）。
- 下载 ~1.1 GB 且 UCI 服务器走 chunked 传输（HEAD 拿不到 Content-Length），慢网时
  fetch_archive 的 .part 断点语义能保证半份下载不会被当缓存命中。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from autocls.data.candidates import (
    IGNORE_LABEL,
    RAW_ROOT,
    WINDOW_LEN,
    extract_archive,
    fetch_archive,
    window_stream,
)

URL = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
RAW_DIR = RAW_ROOT / "HHAR"

# 只用这一个机型（见模块 docstring：单机型避免采样率混杂）。若将来样本不够要扩机型，
# 必须在 meta 的 note 里写明采样率混杂事实与各机型实测采样率，且仍不做重采样。
MODEL = "nexus4"
SOURCE_CSV = "Phones_accelerometer.csv"

# 类别 id 按字母序固定（驱动会重编号 0..k-1，这里只需稳定即可复现实验）
CLASS_NAMES = ["bike", "sit", "stairsdown", "stairsup", "stand", "walk"]
LABEL_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# 时间跳变断段阈值：相邻 Creation_Time 差 > 10× 中位差视为断点
GAP_FACTOR = 10.0

SPEC = {
    "name": "HHAR",
    "domain": "HAR",
    # nexus4 标称 200 Hz（Stisen 2015 Table 1）。真实值以 build() 实测的
    # meta.fs_measured_hz 为准（按相邻 Creation_Time 中位差逐设备算）——若实测
    # 与标称明显不符，以实测为准更新此值
    "fs": 200,
    "url": URL,
    "n_subjects": 9,
    "note": "UCI id 344（Stisen 2015）；单机型 nexus4 加速度计，不重采样；"
            "与目标池 UCI-HAR(id 240)/HAPT(id 341) 无关，无泄漏",
}


def _find_source_csv() -> Path:
    """下载 + 解压（含嵌套 zip 容错），定位 Phones_accelerometer.csv。"""
    archive = fetch_archive(URL, RAW_DIR / "heterogeneity+activity+recognition.zip",
                            timeout=120)
    out = extract_archive(archive, RAW_DIR / "extracted")
    hits = list(out.rglob(SOURCE_CSV))
    if not hits:
        # UCI 外层 zip 内还套着 "Activity recognition exp.zip" 等：逐个解到同名目录
        for nested in sorted(out.rglob("*.zip")):
            extract_archive(nested, nested.with_suffix(""))
        hits = list(out.rglob(SOURCE_CSV))
    if not hits:
        raise FileNotFoundError(f"{SOURCE_CSV} 不在 {out} 下——档案布局变了？")
    return hits[0]


def _read_model_streams(csv_path: Path) -> dict[tuple[str, str], list]:
    """分块读 csv，块内过滤机型后按 (User, Device) 归拢原始流。

    只在这里碰 pandas：13M 行整读会爆内存，块内先 Model 过滤再收集，峰值只有
    单块 + nexus4 子集（~百 MB 量级）。返回 {(user, device): [(t, xyz, lab), ...]}，
    块间顺序保留，最终排序在 _segment_windows 里做。
    """
    import pandas as pd

    usecols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
    # Creation_Time 读成 float64 而非 int64：个别行缺失时 int64 解析直接崩，float64
    # 能吃 NaN 后置过滤。ns 时间戳 ~1.4e18 超出 float64 整数精度（2^53），舍入误差
    # ~256 ns，相对 5e6 ns 的采样间隔完全可忽略，不影响排序与断段。
    dtype = {
        "Creation_Time": np.float64,
        "x": np.float32, "y": np.float32, "z": np.float32,
        "User": "category", "Model": "category", "Device": "category",
        "gt": "category",
    }
    streams: dict[tuple[str, str], list] = {}
    n_scanned = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype,
                             chunksize=1_000_000):
        n_scanned += len(chunk)
        chunk = chunk[chunk["Model"] == MODEL]
        if chunk.empty:
            continue
        # 缺失时间戳的行直接丢（无法排序/断段），正常数据不该有
        chunk = chunk[np.isfinite(chunk["Creation_Time"].to_numpy())]
        if chunk.empty:
            continue
        # gt 缺失(NaN)/"null" → -1：astype(object) 保留 NaN，map 不中者也归 NaN
        lab = (chunk["gt"].astype(object).map(LABEL_IDS)
               .fillna(IGNORE_LABEL).astype(np.int64))
        for (user, device), g in chunk.groupby(["User", "Device"], observed=True):
            streams.setdefault((str(user), str(device)), []).append((
                g["Creation_Time"].to_numpy(np.float64),
                g[["x", "y", "z"]].to_numpy(np.float32),
                lab.loc[g.index].to_numpy(np.int64),
            ))
    streams["__n_scanned__"] = n_scanned  # 顺路带出总行数，避免二次扫描
    return streams


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    csv_path = _find_source_csv()
    streams = _read_model_streams(csv_path)
    n_scanned = streams.pop("__n_scanned__")
    if not streams:
        raise ValueError(f"HHAR: {csv_path} 里没有 Model == {MODEL!r} 的行")

    Xs, ys = [], []
    med_dt_ns: dict[str, list[float]] = {}   # device → 各 (user,device) 流的中位采样间隔
    n_rows_model = 0
    n_segments = 0
    users, devices = set(), set()

    for (user, device), parts in sorted(streams.items()):
        t = np.concatenate([p[0] for p in parts])
        xyz = np.concatenate([p[1] for p in parts])
        lab = np.concatenate([p[2] for p in parts])
        n_rows_model += len(t)
        # 稳定排序：csv 本身按时间写入，块间拼接后仅需一次全排；相同时间戳保持原序
        order = np.argsort(t, kind="stable")
        t, xyz, lab = t[order], xyz[order], lab[order]

        dt = np.diff(t)
        pos = dt[dt > 0]                      # 剔除重复时间戳的 0 差，免得拉低中位数
        if len(pos) == 0:
            continue                          # 退化流（全同时间戳），不可用
        med = float(np.median(pos))
        med_dt_ns.setdefault(device, []).append(med)

        # 断段：时间跳变 > GAP_FACTOR × 中位间隔 → 物理上不连续，必须分开开窗
        breaks = np.where(dt > GAP_FACTOR * med)[0] + 1
        for idx in np.split(np.arange(len(t)), breaks):
            if len(idx) < WINDOW_LEN:
                continue
            xw, yw = window_stream(xyz[idx], lab[idx])
            if len(xw) == 0:
                continue
            n_segments += 1
            Xs.append(xw.astype(np.float32))  # window_stream 出 float64，转 float32 省一半内存
            ys.append(yw)
        users.add(user)
        devices.add(device)

    if not Xs:
        raise ValueError("HHAR: 开窗后无样本——检查机型过滤与断段阈值")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    # 实测采样率（写 meta 不写死：SPEC.fs 只是量级，这里是逐设备真实值）
    fs_measured = {
        dev: round(1e9 / float(np.median(np.asarray(meds))), 2)
        for dev, meds in sorted(med_dt_ns.items())
    }
    meta = {
        "class_names": CLASS_NAMES,
        "n_subjects": len(users),
        "users": sorted(users),
        "devices": sorted(devices),
        "model_selected": MODEL,
        "source_csv": SOURCE_CSV,
        "fs_measured_hz": fs_measured,
        "fs_note": "按相邻 Creation_Time(ns) 中位差实测，逐 (user,device) 流取中位；"
                   "单机型 nexus4，未重采样、未插值",
        "n_rows_scanned": int(n_scanned),
        "n_rows_model": int(n_rows_model),
        "n_segments": int(n_segments),
    }
    return X, y, meta
