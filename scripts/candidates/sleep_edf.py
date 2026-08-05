"""Sleep-EDF Expanded（sleep-cassette 子集）→ 30 s 睡眠分期 epoch。

来源：PhysioNet `sleep-edfx` 1.0.0 的 sleep-cassette（SC*）子集。相关工作 TS-TCC / TF-C
都用它做单通道 EEG 睡眠分期，口径是 **EEG Fpz-Cz @100 Hz、30 s epoch、5 类（AASM）**，
本 adapter 照抄该口径。

**已切好样本**：hypnogram 注释天然按 30 s epoch 对齐，样本 = 一个 epoch = 3000 步 × 1 通道。
故 SPEC 带 `segmented=True`，不走 `window_stream` 开窗——再开 128 步窗会把"整段睡眠期的
慢波形态"这个被分类的对象切碎，与文献口径不可比。

已知坑（按踩到的顺序）：
1. **首尾大段 Wake**：SC 记录是全天候佩戴（本例单份 79500 s ≈ 22 h），灯灭前 8.5 h、
   灯亮后 7.5 h 全是 `Sleep stage W`。不裁的话 W 单类就占 60%+，不平衡比爆到 20×，
   分类任务退化成"检测这人是不是醒着"。文献通用做法（TS-TCC 等）是只保留
   **第一个非 Wake epoch 前 30 min ~ 最后一个非 Wake epoch 后 30 min**，本模块照做，
   裁掉的 epoch 数记入 meta 的 `epochs_trimmed`。
2. **PSG 与 Hypnogram 文件名后两位字母不同**（SC4001E0-PSG / SC4001EC-Hypnogram），
   只有前 6 位 `SC4sse`（ss=受试者、e=夜次）能配对，必须按前缀匹配而非直接替换后缀。
3. **注释会越过信号末尾**：末条 `Sleep stage ?` 的 onset 正好等于信号总时长（本例 79500 s），
   duration 还有 6900 s。逐 epoch 展开时必须按 PSG 实际样本数截断，否则索引越界。
4. **Sleep stage 4 并入 N3**：原始标注是 R&K 六级（W/1/2/3/4/R），AASM 把 3 和 4 合成 N3。
   `Movement time` 与 `Sleep stage ?` 是非睡眠期标注，直接丢弃（不是第 6 类）。
5. **PhysioNet 单连接限速 ~120 KB/s**：单份 PSG 46 MB 串行要 7 min，10 份 70 min。
   实测并发 4 路可到 ~300 KB/s，故用线程池并发取档（`fetch_archive` 本身会跳过已下载的，
   重跑不重下）。

下载量控制：只取目录列表里的**前 10 份记录**（约 460 MB）。这 10 份来自 5 名受试者
（每人 2 晚），实际取到的记录名与受试者数记入 meta。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from autocls.data.candidates import RAW_ROOT, fetch_archive

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

SPEC = {
    "name": "SleepEDF",
    "domain": "EEG",
    "fs": 100,                 # EEG Fpz-Cz 实测采样率（EDF 头里读出来的，不是论文声称值）
    "url": BASE,
    "n_subjects": 5,           # 前 10 份记录 = 5 名受试者 × 2 晚；build() 会用实际值覆盖
    "segmented": True,         # 30 s epoch 已是样本，不开窗
    "note": "PhysioNet sleep-edfx 1.0.0 sleep-cassette 前 10 份记录；"
            "单通道 EEG Fpz-Cz@100Hz、30s epoch、AASM 5 类；"
            "坑：首尾各 8h 的 Wake 必须裁（否则 W 占 60%+）、PSG/Hypnogram 后缀不同需按前缀配对",
}

N_RECORDS = 10                 # 只取前 10 份，控制下载量在 ~460 MB
CHANNEL = "EEG Fpz-Cz"         # TS-TCC / TF-C 的标准单通道口径
EPOCH_SEC = 30
FS = 100
EPOCH_LEN = EPOCH_SEC * FS     # 3000 步
PAD_EPOCHS = 60                # 首尾各保留 30 min = 60 个 epoch（文献通用裁剪口径）
DOWNLOAD_WORKERS = 4           # 实测 4 路并发把 PhysioNet 吞吐从 ~120 拉到 ~300 KB/s

# R&K 六级 → AASM 5 类。stage 4 并入 N3；? 与 Movement time 不是类别，丢弃。
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,        # AASM：R&K 的 S3+S4 合成 N3
    "Sleep stage R": 4,
}
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
WAKE = 0
UNLABELED = -1                 # 与 candidates.IGNORE_LABEL 同语义：不产样本


def _list_records() -> list[tuple[str, str]]:
    """抓目录列表 → 按前 6 位 `SC4sse` 配对的 (PSG, Hypnogram) 文件名，取前 N_RECORDS 份。

    不硬编码文件名：Hypnogram 的后两位字母（EC/EH/EJ/EP…）无规律，只能从列表里配。
    """
    import requests

    html = requests.get(BASE, timeout=60).text
    files = sorted(set(re.findall(r"SC4\d{3}[A-Z0-9]+-(?:PSG|Hypnogram)\.edf", html)))
    psg = {f[:6]: f for f in files if f.endswith("-PSG.edf")}
    hyp = {f[:6]: f for f in files if f.endswith("-Hypnogram.edf")}
    keys = sorted(set(psg) & set(hyp))[:N_RECORDS]
    if not keys:
        raise RuntimeError(f"{BASE} 目录列表里没解析出 PSG/Hypnogram 配对（页面结构变了？）")
    return [(psg[k], hyp[k]) for k in keys]


def _download(pairs: list[tuple[str, str]], raw_dir) -> None:
    """并发取档。fetch_archive 命中已存在文件即跳过，故这里天然支持断点重跑。"""
    names = [n for pair in pairs for n in pair]
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        list(ex.map(lambda n: fetch_archive(BASE + n, raw_dir / n, timeout=300), names))


def _epoch_labels(hyp_path, n_epochs_signal: int) -> np.ndarray:
    """hypnogram 注释 → 逐 epoch 标签数组 (n_epochs_signal,)，未标注处为 UNLABELED。

    注释是 (onset_s, duration_s, description) 三元组，duration 恒为 30 的整数倍。
    坑 3：末条注释会越过信号末尾，故按 n_epochs_signal 截断。
    """
    import pyedflib

    r = pyedflib.EdfReader(str(hyp_path))
    try:
        onsets, durations, descs = r.readAnnotations()
    finally:
        r.close()

    labels = np.full(n_epochs_signal, UNLABELED, dtype=np.int64)
    for onset, dur, desc in zip(onsets, durations, descs):
        stage = STAGE_MAP.get(str(desc))
        if stage is None:          # 'Sleep stage ?' / 'Movement time' → 不产样本
            continue
        i0 = int(round(float(onset) / EPOCH_SEC))
        i1 = i0 + int(round(float(dur) / EPOCH_SEC))
        i0, i1 = max(0, i0), min(n_epochs_signal, i1)
        if i1 > i0:
            labels[i0:i1] = stage
    return labels


def _trim_wake(labels: np.ndarray) -> tuple[int, int]:
    """坑 1：返回保留区间 [lo, hi)——第一个非 Wake epoch 前 30 min 到最后一个后 30 min。

    "非 Wake" 指标签是 N1/N2/N3/REM（UNLABELED 不算，它可能就是记录尾部的空白段）。
    整段没有睡眠 epoch 的记录返回空区间，由调用方跳过。
    """
    asleep = np.where((labels > WAKE))[0]
    if asleep.size == 0:
        return 0, 0
    lo = max(0, int(asleep[0]) - PAD_EPOCHS)
    hi = min(labels.shape[0], int(asleep[-1]) + 1 + PAD_EPOCHS)
    return lo, hi


def _read_record(psg_path, hyp_path) -> tuple[np.ndarray, np.ndarray, dict]:
    """单份记录 → (X (n, 3000, 1), y (n,), 该记录的统计)。"""
    import pyedflib

    r = pyedflib.EdfReader(str(psg_path))
    try:
        sig_labels = [s.strip() for s in r.getSignalLabels()]
        if CHANNEL not in sig_labels:
            raise ValueError(f"{psg_path.name} 没有通道 {CHANNEL}，实有 {sig_labels}")
        ch = sig_labels.index(CHANNEL)
        fs = float(r.getSampleFrequency(ch))
        if fs != FS:
            raise ValueError(f"{psg_path.name} 通道 {CHANNEL} 采样率 {fs} != {FS}（不重采样）")
        n_signal = int(r.getNSamples()[ch])
        n_epochs = n_signal // EPOCH_LEN

        labels = _epoch_labels(hyp_path, n_epochs)
        lo, hi = _trim_wake(labels)
        if hi <= lo:
            return (np.zeros((0, EPOCH_LEN, 1)), np.zeros((0,), dtype=np.int64),
                    {"record": psg_path.name, "epochs_total": n_epochs, "epochs_kept": 0,
                     "epochs_trimmed": n_epochs, "epochs_unlabeled": 0})

        # 只读裁剪区间：整通道 79500 s @100 Hz = 8M 点，10 份全读进内存没必要
        raw = r.readSignal(ch, start=lo * EPOCH_LEN, n=(hi - lo) * EPOCH_LEN)
    finally:
        r.close()

    seg_labels = labels[lo:hi]
    x = np.asarray(raw, dtype=np.float64).reshape(hi - lo, EPOCH_LEN, 1)
    keep = seg_labels >= 0
    # 铁律：不插值补缺失。含非有限值的 epoch 直接丢（EDF 理论上不会有，防御性检查）
    finite = np.isfinite(x).all(axis=(1, 2))
    keep &= finite
    stats = {
        "record": psg_path.name,
        "epochs_total": int(n_epochs),
        "epochs_kept": int(keep.sum()),
        "epochs_trimmed": int(n_epochs - (hi - lo)),
        "epochs_unlabeled": int((seg_labels < 0).sum()),
        "epochs_nonfinite": int((~finite).sum()),
    }
    return x[keep], seg_labels[keep], stats


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    raw_dir = RAW_ROOT / "sleep_edf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pairs = _list_records()
    _download(pairs, raw_dir)

    xs, ys, per_record = [], [], []
    for psg, hyp in pairs:
        x, y, st = _read_record(raw_dir / psg, raw_dir / hyp)
        if x.shape[0]:
            xs.append(x)
            ys.append(y)
        per_record.append(st)
    if not xs:
        raise RuntimeError("Sleep-EDF：所有记录都没解析出有效 epoch")

    X = np.concatenate(xs)
    y = np.concatenate(ys)
    # 受试者 = 文件名 SC4ssN 的 ss 段（同一人两晚是两份记录，但只算一名受试者）
    subjects = sorted({p[:5] for p, _ in pairs})
    total_epochs = sum(s["epochs_total"] for s in per_record)
    meta = {
        "class_names": CLASS_NAMES,
        "channel": CHANNEL,
        "epoch_sec": EPOCH_SEC,
        "n_records": len(pairs),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "records": [p for p, _ in pairs],
        "epochs_total": int(total_epochs),
        "epochs_trimmed": int(sum(s["epochs_trimmed"] for s in per_record)),
        "epochs_unlabeled": int(sum(s["epochs_unlabeled"] for s in per_record)),
        "trim_rule": f"第一个非 Wake epoch 前 {PAD_EPOCHS} 个 epoch ~ 最后一个后 {PAD_EPOCHS} 个",
        "per_record": per_record,
        "bytes_downloaded": int(sum(p.stat().st_size for p in raw_dir.glob("*.edf"))),
    }
    return X, y, meta
