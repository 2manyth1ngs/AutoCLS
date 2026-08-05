"""候选数据集通用加载层（换池调研：离开 UEA 重建源池）。

背景：协议探针（`results/protocol_probe/REPORT.md`）证伪了"改下游评估协议"这条出路
——随机初始化的 10 层膨胀 CNN 在 UEA 这批数据集上已经是个够好的特征提取器，换分类器
族是空旋钮。剩下的出路是**换数据集**：按"随机初始化基线不贴顶且训练能拉动"重建源池。

本模块是候选数据集的统一入口。与 `uea.py` 的分工：
- `uea.py`：UEA 档案（已切好样本的 .ts）→ pad + z-score。
- 本模块：**连续传感器流**（WISDM/PAMAP2/MHEALTH/… 的逐时刻记录）→ 定长开窗成样本，
  再 z-score。UCI-HAR 之所以是 (7352, 128, 9) 就是这么来的（2.56 s @50 Hz、50% 重叠）。

数组约定与全项目一致：X (N, T, C) float32 已按通道 z-score；y (N,) int64 恒 0..k-1。

窗口口径（`results/candidate_survey/PROTOCOL.md` 预注册，跑数据前写死）：
win=128 步 @**原生采样率**、stride=64（50% 重叠）。128 = UCI-HAR 的 L——它是唯一验证过
"CLS 差异可见"的正样本（M1 GGS 0.9372）；也高于 L≥50 的固定 encoder 适用边界。
不按秒对齐而按步对齐：encoder 的感受野约束是**步数**（10 层膨胀 CNN，dilation 到 2^9），
与物理时长无关；各数据集采样率不同导致的窗口时长差异（1.28 s @100 Hz ~ 6.4 s @20 Hz）
留作数据集间的天然多样性。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = ROOT / "data" / "candidates"
RAW_ROOT = CANDIDATE_ROOT / "raw"

# 预注册窗口口径（PROTOCOL §1）——改这两个数就是换任务定义，缓存全部作废
WINDOW_LEN = 128
WINDOW_STRIDE = 64

IGNORE_LABEL = -1   # 逐步标签 <0 = 忽略段（Opportunity null / PAMAP2 transient）


def window_stream(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    win: int = WINDOW_LEN,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """连续流 (T, C) + 逐步标签 (T,) → 定长窗 (N, win, C) + 窗标签 (N,)。

    只保留**纯标签窗**：窗内标签全同且 ≥0。跨活动边界的混合窗直接丢弃（HAR 文献通用
    做法），负标签段（null/transient）不产样本。含 NaN 的窗丢弃——原始记录常有丢包
    （PAMAP2 心率通道 ~90% 缺失、IMU 偶发丢帧），插值会造假值、影响的是"表征学得到
    什么"这个被测量本身，故宁缺勿造。

    调用方负责按**连续段**（同一受试者/同一次记录）分别调用：跨段拼接会造出物理上
    不存在的过渡窗。
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2:
        raise ValueError(f"x 必须是 (T, C)，实得 {x.shape}")
    if labels.shape[0] != x.shape[0]:
        raise ValueError(f"labels 长度 {labels.shape[0]} != 时间步 {x.shape[0]}")

    C = x.shape[1]
    starts = range(0, max(0, x.shape[0] - win + 1), stride)
    xs, ys = [], []
    for s in starts:
        lab = labels[s:s + win]
        first = lab[0]
        if first < 0 or not np.all(lab == first):
            continue
        seg = x[s:s + win]
        if not np.isfinite(seg).all():
            continue
        xs.append(seg)
        ys.append(int(first))
    if not xs:
        return np.zeros((0, win, C), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def zscore_channels(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N, T, C) → 按通道 z-score 的 float32。

    统计量在 float64 上算（对齐 datasets.py 对 total_acc 那类"均值≈1g、方差小"通道的
    处理：float32 统计会放大偏差）。常值通道 std<eps 按 1 处理 → 该通道恒 0，不产 NaN。
    """
    Xd = np.asarray(X, dtype=np.float64)
    flat = Xd.reshape(-1, Xd.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return ((Xd - mean) / std).astype(np.float32)


def drop_rare_classes(
    X: np.ndarray, y: np.ndarray, min_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """样本数 <min_count 的类整类剔除，剩余类重新编号为 0..k-1。

    原始 HAR 数据集常有"某活动只有 1-2 个受试者做过"的极稀类（PAMAP2 的
    rope_jumping、Opportunity 的部分手势）——留着会让类别子集切片抽到不成任务的类。
    """
    y = np.asarray(y, dtype=np.int64)
    vals, counts = np.unique(y, return_counts=True)
    keep_vals = vals[counts >= min_count]
    mask = np.isin(y, keep_vals)
    remap = {int(v): i for i, v in enumerate(sorted(keep_vals.tolist()))}
    y2 = np.asarray([remap[int(v)] for v in y[mask]], dtype=np.int64)
    return X[mask], y2


def describe(X: np.ndarray, y: np.ndarray) -> dict:
    """候选统计表的唯一口径——**实际加载后**的真实形状与类别分布。

    上一轮选 UEA 的教训之一是统计来自论文摘要；本函数只吃已加载的数组。
    """
    y = np.asarray(y, dtype=np.int64)
    _vals, counts = np.unique(y, return_counts=True)
    counts = counts.tolist()
    return {
        "n_samples": int(X.shape[0]),
        "seq_len": int(X.shape[1]),
        "n_channels": int(X.shape[2]),
        "n_classes": len(counts),
        "class_counts": counts,
        "imbalance_ratio": float(max(counts) / min(counts)) if counts else float("nan"),
        "chance": float(1.0 / len(counts)) if counts else float("nan"),
    }


def _npz_path(name: str, root: Path) -> Path:
    return Path(root) / f"{name}.npz"


def save_candidate(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
    *,
    root: str | Path = CANDIDATE_ROOT,
) -> Path:
    """落盘 canonical npz + 同名 .json 统计（铁律 3：贵的一步必须缓存）。

    y 必须已是 0..k-1 连续 int64——下游切片与 SVM 都假定连续类 id（见 uea.encode_labels
    的同一约定）。不连续直接报错，不静默重编号：静默重编号会让 adapter 的类别语义
    与 meta 里记的类名对不上。
    """
    y = np.asarray(y, dtype=np.int64)
    uniq = np.unique(y)
    if len(uniq) and uniq.tolist() != list(range(len(uniq))):
        raise ValueError(f"{name}: 标签必须是 0..k-1 连续 int64，实得 {uniq.tolist()[:12]}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    path = _npz_path(name, root)
    np.savez_compressed(path, X=X, y=y)
    (root / f"{name}.json").write_text(
        json.dumps({"name": name, **meta, **describe(X, y)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_candidate(
    name: str, *, root: str | Path = CANDIDATE_ROOT
) -> tuple[np.ndarray, np.ndarray]:
    """读 canonical npz → X (N, T, C) float32、y (N,) int64。"""
    path = _npz_path(name, Path(root))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在——先运行 scripts/prepare_candidates.py --datasets {name}"
        )
    d = np.load(path)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def fetch_archive(url: str, dest: str | Path, *, timeout: int = 60) -> Path:
    """下载到 dest（已存在且非空则跳过——重跑不重下，铁律 3 的同一精神）。

    流式写盘：候选档案从几十 MB 到 1 GB+（HHAR/PTB-XL），整份读进内存不划算。
    """
    import requests

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)      # 原子替换：半份下载不会被当成缓存命中
    return dest


def extract_archive(path: str | Path, out_dir: str | Path) -> Path:
    """解压 .zip / .tar.gz / .tar.bz2 到 out_dir（已解压则直接返回）。"""
    import tarfile
    import zipfile

    path, out_dir = Path(path), Path(out_dir)
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(out_dir)
    else:
        raise ValueError(f"不认识的档案格式：{path}")
    marker.write_text("ok", encoding="utf-8")
    return out_dir


def candidate_meta(name: str, *, root: str | Path = CANDIDATE_ROOT) -> dict:
    path = Path(root) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")
    return json.loads(path.read_text(encoding="utf-8"))
