"""WISDM 2019（Smartphone & Smartwatch Activity and Biometrics）：手机 6 通道，18 类活动。

来源 UCI ML Repository id=507（Weiss et al. 2019, IEEE Access 7）。51 受试者，每人每种
活动约 3 min，phone/watch 各有 accel/gyro 四个原始流（`raw/<device>/<sensor>/data_<id>_
<sensor>_<device>.txt`），行格式与 WISDM v1.1 同族：`subject,activity,timestamp,x,y,z;`。
活动用字母 A–S 编码（**跳过 N**，共 18 类），映射表见档案内 `activity_key.txt`。

**通道选择：phone accel + phone gyro 对齐成 6 通道**（本机实测可行，非退化方案）：
两个流共用同一设备 uptime 时钟——908 个 (受试者, 活动) 组里 907 组的最近邻匹配率
≥99.75%（中位 99.94%）。对齐规则刻意选成"**只挑真实采样、不合成值**"：
- 以两流中**较粗**（中位采样间隔较大）的那条做参考轴。约三分之一受试者 accel 是
  50 Hz 而 gyro 只有 20–25 Hz；若拿细流做参考，粗流会被最近邻重复取值 → 造出阶梯状
  的假信号（等价于零阶保持插值）。拿粗流做参考则每个参考时刻各取到一条不同的细流
  采样（实测被取样本唯一率 100%），全部是真实测量值。
- 容差 = 0.5 × 参考轴中位间隔（"物理上最近的那一条"），超容差的时刻整条丢弃，
  不插值补（铁律：含 NaN/缺测的窗宁缺勿造）。

已知坑（均为本机实测，计数写进 meta）：
- 受试者 1629 的 phone accel/gyro 文件整份重复了一遍（139346 行 / 69673 唯一），且
  uptime 时间戳跨了 ~145 h 的两段。按 (活动, timestamp) 首次出现去重后自然消解，
  剩下的跨段跳变由分段逻辑切开。全池共去重 133923 行。
- 1642/J 一组的 accel 与 gyro 时间范围**完全不重叠**（差 ~860 s，源数据录制错位），
  对齐后为空 → 该组不产窗口。1641/Q、1644/L 两组根本没有 gyro 文件，同样跳过。
- 各受试者采样率不统一（实测参考轴间隔 19.9 / 39.6 / 50.4 ms 都有），文档声称的
  20 Hz 只是名义值。按预注册口径统一按**步**开窗、不重采样。
- 严格正则解析下 phone 的 accel/gyro 无畸形行（与 v1.1 不同），但解析仍走同一套
  过滤，格式漂移时会被计入 `n_dropped_malformed` 而非静默混入。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from autocls.data.candidates import RAW_ROOT, extract_archive, fetch_archive, window_stream

SPEC = {
    "name": "WISDM2019",
    "domain": "HAR",
    "fs": 20,
    "url": ("https://archive.ics.uci.edu/static/public/507/"
            "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"),
    "n_subjects": 51,
    "note": ("手机 accel+gyro 按时间戳对齐成 6 通道、18 类活动；活动字母跳过 N，"
             "受试者 1629 文件整份重复、1642/J accel-gyro 时间范围不重叠"),
}

# 活动字母 → 类名（档案 activity_key.txt 原文，**没有 N**）。标签 id = 本表插入顺序，
# 与 class_names 下标一致；固定写死而非按出现顺序编号，保证 meta 里的类名对得上 y。
ACTIVITY_KEY = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting", "E": "standing",
    "F": "typing", "G": "teeth", "H": "soup", "I": "chips", "J": "pasta",
    "K": "drinking", "L": "sandwich", "M": "kicking", "O": "catch", "P": "dribbling",
    "Q": "writing", "R": "clapping", "S": "folding",
}
CLASS_NAMES = list(ACTIVITY_KEY.values())
_ACT_ID = {letter: i for i, letter in enumerate(ACTIVITY_KEY)}

# 整行严格匹配（与 wisdm_v1 同族格式）：格式漂移一律整行丢弃并计数
_LINE = re.compile(
    r"^(\d+),([A-Z]),(\d+),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?);?$"
)

GAP_FACTOR = 10       # 断段阈值 = GAP_FACTOR × 该组中位采样间隔
ALIGN_TOL_FRAC = 0.5  # 对齐容差 = 该比例 × 参考轴中位间隔


def _median_step(ts: np.ndarray, fallback_ns: float) -> float:
    """该流的中位采样间隔（ns）。全重复/过短时退回名义间隔，避免阈值退化成 0。"""
    if ts.shape[0] < 2:
        return fallback_ns
    med = float(np.median(np.diff(ts)))
    return med if med > 0 else fallback_ns


def _parse_file(path: Path, counters: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """一个原始 txt → {活动字母: (ts 升序 (T,), 值 (T, 3))}，按 (活动, ts) 去重。"""
    buckets: dict[str, dict[int, tuple[float, float, float]]] = {}
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counters["n_lines_total"] += 1
            m = _LINE.match(line)
            if m is None:
                counters["n_dropped_malformed"] += 1
                continue
            act = m.group(2)
            if act not in _ACT_ID:
                # 出现 key 表外的字母（含被跳过的 N）说明档案变了，宁可炸也不静默造错类
                raise ValueError(f"{path.name}: 未知活动字母 {act!r}")
            ts = int(m.group(3))
            if ts <= 0:
                counters["n_dropped_ts_nonpos"] += 1
                continue
            d = buckets.setdefault(act, {})
            if ts in d:
                counters["n_dropped_duplicate_ts"] += 1
                continue
            d[ts] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    out = {}
    for act, d in buckets.items():
        ts = np.fromiter(sorted(d), dtype=np.int64, count=len(d))
        vals = np.asarray([d[int(t)] for t in ts], dtype=np.float64)
        out[act] = (ts, vals)
    return out


def _align(
    ts_a: np.ndarray, val_a: np.ndarray,
    ts_g: np.ndarray, val_g: np.ndarray,
    fallback_ns: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """accel/gyro 两流 → 公共时间轴 (T,) + 6 通道 (T, 6)，通道序 [ax,ay,az,gx,gy,gz]。

    参考轴取**较粗**的那条（见模块注释：拿细流做参考会把粗流零阶保持成阶梯假信号）；
    另一条只做"取最近的一条真实采样"，超容差的参考时刻整条丢弃。返回保留率供 meta 记录。
    """
    step_a = _median_step(ts_a, fallback_ns)
    step_g = _median_step(ts_g, fallback_ns)
    accel_is_ref = step_a >= step_g
    if accel_is_ref:
        ts_ref, val_ref, ts_oth, val_oth, step = ts_a, val_a, ts_g, val_g, step_a
    else:
        ts_ref, val_ref, ts_oth, val_oth, step = ts_g, val_g, ts_a, val_a, step_g
    if ts_oth.shape[0] == 0 or ts_ref.shape[0] == 0:
        return ts_ref[:0], np.zeros((0, 6)), 0.0

    idx = np.searchsorted(ts_oth, ts_ref)
    lo = np.clip(idx - 1, 0, ts_oth.shape[0] - 1)
    hi = np.clip(idx, 0, ts_oth.shape[0] - 1)
    pick = np.where(np.abs(ts_ref - ts_oth[lo]) <= np.abs(ts_ref - ts_oth[hi]), lo, hi)
    keep = np.abs(ts_ref - ts_oth[pick]) <= ALIGN_TOL_FRAC * step
    other = val_oth[pick]
    merged = (np.concatenate([val_ref, other], axis=1) if accel_is_ref
              else np.concatenate([other, val_ref], axis=1))
    return ts_ref[keep], merged[keep], float(keep.mean())


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    raw_dir = RAW_ROOT / "WISDM2019"
    archive = fetch_archive(SPEC["url"], raw_dir / "wisdm2019.zip")
    outer = extract_archive(archive, raw_dir / "outer")
    # UCI 的包是套娃：外层 zip 里还有一个 wisdm-dataset.zip
    inner_zip = outer / "wisdm-dataset.zip"
    if not inner_zip.exists():
        raise FileNotFoundError(f"未找到内层档案 {inner_zip}——UCI 打包结构可能已变")
    root = extract_archive(inner_zip, raw_dir / "inner") / "wisdm-dataset" / "raw" / "phone"
    accel_dir, gyro_dir = root / "accel", root / "gyro"
    if not accel_dir.is_dir() or not gyro_dir.is_dir():
        raise FileNotFoundError(f"未找到 {accel_dir} / {gyro_dir}")

    fallback_ns = 1e9 / SPEC["fs"]
    counters = {"n_lines_total": 0, "n_dropped_malformed": 0,
                "n_dropped_ts_nonpos": 0, "n_dropped_duplicate_ts": 0}
    xs, ys = [], []
    subjects, keep_rates = [], []
    n_segments = n_missing_gyro = n_empty_after_align = 0

    for fa in sorted(accel_dir.glob("data_*_accel_phone.txt")):
        subj = fa.name.split("_")[1]
        fg = gyro_dir / f"data_{subj}_gyro_phone.txt"
        if not fg.exists():
            n_missing_gyro += 1
            continue
        subjects.append(subj)
        acc = _parse_file(fa, counters)
        gyr = _parse_file(fg, counters)
        for act in sorted(acc):
            if act not in gyr:
                n_missing_gyro += 1        # 该受试者的这一活动没录 gyro
                continue
            ts, stream, rate = _align(*acc[act], *gyr[act], fallback_ns)
            keep_rates.append(rate)
            if ts.shape[0] == 0:
                n_empty_after_align += 1   # 如 1642/J：两流时间范围完全不重叠
                continue
            # 断段：uptime 跳变 > GAP_FACTOR × 该组中位间隔。用组内实测间隔而非名义
            # 20 Hz——各受试者真实采样率 19.9~50.4 ms 不等，固定阈值对快档太松。
            gap = GAP_FACTOR * _median_step(ts, fallback_ns)
            cuts = np.where(np.diff(ts) > gap)[0]
            label = _ACT_ID[act]
            for seg in np.split(stream, cuts + 1):
                if seg.shape[0] == 0:
                    continue
                n_segments += 1
                labels = np.full(seg.shape[0], label, dtype=np.int64)
                wx, wy = window_stream(seg, labels)
                if wx.shape[0]:
                    xs.append(wx)
                    ys.append(wy)

    if not xs:
        raise ValueError("WISDM2019: 解析后无任何有效窗口")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    meta = {
        "class_names": CLASS_NAMES,
        "channel_names": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        "device": "phone",
        "n_subjects_seen": len(subjects),
        **counters,
        "n_segments": n_segments,
        "n_groups_missing_gyro": n_missing_gyro,
        "n_groups_empty_after_align": n_empty_after_align,
        "align_keep_rate_median": round(float(np.median(keep_rates)), 5) if keep_rates else None,
        "align_keep_rate_min": round(float(np.min(keep_rates)), 5) if keep_rates else None,
        "align_tol_frac": ALIGN_TOL_FRAC,
        "note_channels": ("phone accel + phone gyro 对齐成 6 通道；参考轴取较粗的一条流，"
                          "只取真实采样不做零阶保持，容差 0.5×参考间隔"),
    }
    return X, y, meta
