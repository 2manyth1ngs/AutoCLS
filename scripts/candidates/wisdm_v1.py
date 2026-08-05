"""WISDM v1.1（Actitracker）：口袋手机三轴加速度，6 类日常活动。

来源 Fordham WISDM Lab（Kwapisz et al. 2010, SIGKDD Explorations 12(2)）。
36 受试者、单设备（裤兜 Android 手机）、fs=20 Hz、3 通道加速度。原始文件为逐行文本：
`user,activity,timestamp,x,y,z;`（行尾分号，timestamp 为设备 uptime 纳秒）。

已知坑（均为本机实测确认，丢弃计数写进 meta，不静默）：
- 畸形行 ~1.1%（11k+ 行）：z 后多一个逗号（`...,5.13,;`）、两条记录粘连在一行、
  截断缺字段。一律整行丢弃不抢救——量小且抢救规则会引入歧义解析。
- timestamp=0 的记录 ~1.2%（常伴 x=y=z=0）：无法在时间轴上定位，丢弃。
- 重复 timestamp ~54k 对：保留（stable sort 不打乱原文件顺序）；窗口纯度只看
  标签与 NaN，重复采样点不影响。
- 部分受试者实际采样间隔 ~100 ms（10 Hz 设备），与文档声称的 20 Hz 不符；按预注册
  口径统一按**步**开窗、不重采样（见 candidates.py 模块注释：感受野约束是步数）。
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from autocls.data.candidates import RAW_ROOT, extract_archive, fetch_archive, window_stream

SPEC = {
    "name": "WISDM",
    "domain": "HAR",
    "fs": 20,
    "url": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
    "n_subjects": 36,
    "note": "手机口袋三轴加速度 6 类活动；原始 txt 有 ~1% 畸形行与 ts=0 记录，逐行严格正则过滤",
}

# 标签 id = 列表下标（0..5）。固定顺序写死而非按出现顺序编号：meta 的 class_names
# 必须与 y 的语义对得上（save_candidate 的同一约定）。
ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]

# 整行严格匹配：user, activity, timestamp, x, y, z + 可选行尾分号。
# 三类已知畸形（粘连行的中段分号、`z,;` 的多余逗号、截断缺字段）都无法通过全行匹配，
# 一个正则统一挡掉，不用逐类特判。
_LINE = re.compile(
    r"^(\d+),([A-Za-z]+),(\d+),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?),"
    r"(-?[\d.]+(?:[Ee]-?\d+)?);?$"
)

# 断段阈值 = GAP_FACTOR × **该组实测**中位采样间隔（timestamp 单位纳秒）。
# 不用名义 20 Hz 算死值：实测各 (user, activity) 组的中位间隔有 40 / 50 / 100 ms 三档
# （部分受试者的设备只有 10 Hz），固定 500 ms 对慢档太紧、对快档太松。
# 实测跳变与正常间隔差着量级（99.9 分位 330 ms，>1 s 的仅 255 处），阈值不敏感。
GAP_FACTOR = 10
_NOMINAL_STEP_NS = 1e9 / 20      # 组内间隔退化时的兜底（20 Hz 名义值）


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    raw_dir = RAW_ROOT / "WISDM"
    archive = fetch_archive(SPEC["url"], raw_dir / "WISDM_ar_latest.tar.gz")
    extract_archive(archive, raw_dir)
    txt = raw_dir / "WISDM_ar_v1.1" / "WISDM_ar_v1.1_raw.txt"
    if not txt.exists():
        raise FileNotFoundError(f"解压后未找到 {txt}——档案结构可能已变")

    act_id = {a: i for i, a in enumerate(ACTIVITIES)}
    # (user, activity) 分组收集 (ts, x, y, z)。分组键不含"次记录"——原始文件没有
    # session 字段，靠 timestamp 跳变把多次记录切开（见下）。
    groups: dict[tuple[int, int], list[tuple[int, float, float, float]]] = defaultdict(list)
    n_lines = n_malformed = n_ts_nonpos = 0
    with open(txt, encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            m = _LINE.match(line)
            if m is None:
                n_malformed += 1
                continue
            act = m.group(2)
            if act not in act_id:
                # 出现文档外的活动名说明格式漂移，宁可炸也不静默造出错类
                raise ValueError(f"未知活动名 {act!r}（行：{line[:80]}）")
            ts = int(m.group(3))
            if ts <= 0:
                n_ts_nonpos += 1
                continue
            groups[(int(m.group(1)), act_id[act])].append(
                (ts, float(m.group(4)), float(m.group(5)), float(m.group(6)))
            )

    xs, ys = [], []
    n_segments = 0
    for (_user, act), recs in sorted(groups.items()):
        arr = np.asarray(recs, dtype=np.float64)          # (T, 4): ts, x, y, z
        order = np.argsort(arr[:, 0], kind="stable")      # stable：重复 ts 保持原文件顺序
        arr = arr[order]
        # timestamp 跳变 >GAP_FACTOR 倍中位采样间隔 → 断成新的连续段；每段独立开窗
        # （跨段拼接会造出物理上不存在的过渡窗，契约铁律）
        steps = np.diff(arr[:, 0])
        med = float(np.median(steps)) if steps.size else 0.0
        gap = GAP_FACTOR * (med if med > 0 else _NOMINAL_STEP_NS)   # 全重复 ts 时兜底
        gaps = np.where(steps > gap)[0]
        for seg in np.split(arr, gaps + 1):
            if seg.shape[0] == 0:
                continue
            n_segments += 1
            stream = seg[:, 1:4]
            labels = np.full(stream.shape[0], act, dtype=np.int64)
            wx, wy = window_stream(stream, labels)
            if wx.shape[0]:
                xs.append(wx)
                ys.append(wy)

    if not xs:
        raise ValueError("WISDM: 解析后无任何有效窗口")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    meta = {
        "class_names": ACTIVITIES,
        "n_subjects_seen": len({u for u, _ in groups}),
        "n_lines_total": n_lines,
        "n_dropped_malformed": n_malformed,     # 粘连/缺字段/`z,;` 畸形行
        "n_dropped_ts_nonpos": n_ts_nonpos,     # timestamp=0 无法定位，丢弃
        "n_segments": n_segments,
    }
    return X, y, meta
