"""UEA 多变量分类数据加载与预处理（路线 A 探针配套；不属于阶段一核心管线）。

datasets.py 是 HAR/Epilepsy 的专用加载；本模块处理 UEA 30 数据集的通用加载，
数据来自 results/uea_survey 落盘的 `data/uea_cache/<name>/<name>_{TRAIN,TEST}.ts`
（Zenodo 镜像，见 results/uea_survey/REPORT.md §0）。

与 datasets.py 的两个约定差异，本模块负责抹平：
1. aeon 返回 **通道优先** (n, C, L)；本项目约定 (N, T, C)——转置。
2. 不等长数据集（如 SpokenArabicDigits L=4-93）aeon 返回 list[(C, L_i)]——本模块
   右侧零填充到等长（填充值 = z-score 后的通道均值 0，中性）。

z-score（对齐 datasets.py 的按通道标准化，防泄漏语义）：统计量只在**真实时间步**
上计算，填充步不计入——否则短序列多的数据集统计量会被填充零拉偏。float64 统计
后转 float32（对齐 datasets.py 对 total_acc 那类小方差通道的处理）。

crop_len：长序列（Cricket L=1197、超长 EEG）按算力预算裁到定长（REPORT.md §5 的
CROP 杠杆）。裁剪取前缀（截断非重采样），保持与增强 crop 字段的时间语义一致。

npz 直读（WP17-ACTS，实验室服务器 py3.8 env 装不了 aeon）：`export_uea_merged_npz` 把 `load_uea_merged`
的解析结果（X float32 / y int64 / crop_len）落成 `<cache_dir>/<name>/<name>_merged.npz`；`load_uea_merged`
默认先查同名 npz，crop_len 匹配即直读（与本地解析逐位同），否则回落解析；`use_npz=False` 强制解析。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "uea_cache"


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """字符串/任意标签 → 连续 int64 (0..k-1) + 类目录（np.unique 升序）。

    取值仅作类 id，顺序不影响分类；保证连续以满足下游分类头/SVM 要求。
    """
    classes, inv = np.unique(np.asarray(y), return_inverse=True)
    return inv.astype(np.int64), classes


def pad_and_zscore(
    seqs: list[np.ndarray], crop_len: int | None = None, eps: float = 1e-8
) -> np.ndarray:
    """不等长序列列表 (T_i, C) → 等长 (N, T, C) float32，按通道 z-score。

    步骤：① crop_len 给定则各序列截断到前 crop_len 步；② 目标长度 = crop_len 或
    各序列真实长度最大值；③ 统计量只用截断后的真实步（填充不计入）；④ z-score 后
    写入，右侧不足处留零（= 通道均值，中性填充）。常值通道 std<eps 时按 1 处理
    （避免除零 NaN，该通道 z-score 后恒 0）。
    """
    if not seqs:
        raise ValueError("seqs 为空")
    real = _crop_real(seqs, crop_len)
    target = crop_len if crop_len is not None else max(s.shape[0] for s in real)
    mean, std = _fit_channel_stats(real, eps=eps)
    return _apply_pad_zscore(real, target, mean, std)


# pad_and_zscore 的三段拆分（全量协议对齐 M8，2026-09-03）：组合后输出逐位不变
# （tests/test_uea.py::test_pad_and_zscore_refactor_bitwise_golden），拆开只为让
# load_uea_pair(scaler="train") 能"只在 TRAIN 真实步上拟合统计量、套用到 TEST"。

def _crop_real(seqs: list[np.ndarray], crop_len: int | None) -> list[np.ndarray]:
    """各序列转 float64 并按 crop_len 截断前缀（None 不截）。"""
    return [np.asarray(s, dtype=np.float64)[:crop_len] if crop_len else np.asarray(s, dtype=np.float64)
            for s in seqs]


def _fit_channel_stats(real: list[np.ndarray], eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """按通道在**真实步**上拟合 (mean, std)；常值通道 std<eps 按 1 处理。"""
    stacked = np.concatenate(real, axis=0)  # (sum_ti, C) 仅真实步
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def _apply_pad_zscore(
    real: list[np.ndarray], target: int, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """用给定统计量 z-score 并右侧零填充到 target 长度 → (N, target, C) float32。"""
    C = real[0].shape[1]
    out = np.zeros((len(real), target, C), dtype=np.float32)
    for i, s in enumerate(real):
        t = min(s.shape[0], target)
        out[i, :t] = ((s[:t] - mean) / std).astype(np.float32)
    return out


def _to_seq_list(X) -> list[np.ndarray]:
    """aeon 加载结果 → list[(T_i, C)]（通道优先转 (T, C)）。

    aeon 等长返回 ndarray (n, C, L)，不等长返回 list/object 数组 of (C, L_i)。
    """
    if isinstance(X, np.ndarray) and X.ndim == 3:
        return [X[i].T for i in range(X.shape[0])]     # (C, L) → (L, C)
    return [np.asarray(x).T for x in X]                # 逐条 (C, L_i) → (L_i, C)


def _parse_ts_tolerant(path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
    """容错解析 .ts：对"同 case 内各通道长度不一"的文件也能加载（aeon 会 fail-fast）。

    论文/aeon 未给全，按"对齐到 case 内最短真实观测"语义补全：把该 case 各通道**截断
    到最短通道长**（截断非填充，不造假值），得 (T_i, C)。仅 CharacterTrajectories
    这类 21/1422 例 ragged 的档案走此路，其余仍走 aeon（见 _load_raw_seqs 回退）。
    缺失标记 '?' → NaN（本项目源池档案实测无 '?'，仅作稳健处理）。
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "@data":
            start = i + 1
            break
    seqs: list[np.ndarray] = []
    labels: list[str] = []
    for ln in lines[start:]:
        ln = ln.strip()
        if not ln or ln.startswith(("#", "@")):
            continue
        parts = ln.split(":")
        label = parts[-1].strip()
        chans = []
        for cs in parts[:-1]:
            vals = [np.nan if v.strip() == "?" else float(v)
                    for v in cs.split(",") if v.strip() != ""]
            chans.append(np.asarray(vals, dtype=np.float64))
        if not chans:
            continue
        lmin = min(len(c) for c in chans)          # 截断到 case 内最短通道长
        seqs.append(np.stack([c[:lmin] for c in chans], axis=1))  # (lmin, C)
        labels.append(label)
    return seqs, np.asarray(labels)


def _load_raw_seqs(
    name: str, split: str, cache_dir: str | Path
) -> tuple[list[np.ndarray], np.ndarray]:
    """经 aeon 本地解析取一个切分的原始序列列表 (list[(T_i, C)]) + 原始标签 y。

    加载不等长/含缺失一律用最宽松开关；填充/标准化/标签编码由调用方统一做（不依赖
    aeon 的 pad 行为，保证填充值与统计口径可控）。aeon 对"同 case 内通道长度不一"的
    档案 fail-fast（OSError，如 CharacterTrajectories）→ 回退容错解析。
    """
    from aeon.datasets import load_classification  # 惰性导入：核心管线不依赖 aeon

    try:
        X, y = load_classification(
            name, split=split, extract_path=str(cache_dir),
            load_equal_length=False, load_no_missing=False,
        )
        return _to_seq_list(X), np.asarray(y)
    except OSError:
        return _parse_ts_tolerant(Path(cache_dir) / name / f"{name}_{split}.ts")


def load_uea_cached(
    name: str,
    split: str = "TRAIN",
    cache_dir: str | Path = DEFAULT_CACHE,
    crop_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 `data/uea_cache` 经 aeon 本地解析加载一个 UEA 数据集的一个切分。

    返回 X (N, T, C) float32（等长/已填充/已裁剪）、y (N,) int64（0..k-1）。
    """
    seqs, y = _load_raw_seqs(name, split, cache_dir)
    Xt = pad_and_zscore(seqs, crop_len=crop_len)
    yi, _ = encode_labels(y)
    return Xt, yi


NPZ_SUFFIX = "_merged.npz"


def merged_npz_path(name: str, cache_dir: str | Path = DEFAULT_CACHE) -> Path:
    return Path(cache_dir) / name / f"{name}{NPZ_SUFFIX}"


def export_uea_merged_npz(
    name: str, *, cache_dir: str | Path = DEFAULT_CACHE, crop_len: int | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """把 load_uea_merged 的解析结果落成 npz（X float32、y int64、crop_len；-1 = None）。服务器无 aeon 时直读，与本地逐位同。"""
    X, y = load_uea_merged(name, cache_dir=cache_dir, crop_len=crop_len, use_npz=False)
    p = merged_npz_path(name, out_dir or cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=np.ascontiguousarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64),
                        crop_len=np.int64(-1 if crop_len is None else int(crop_len)))
    return p


def load_uea_merged(
    name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    crop_len: int | None = None,
    use_npz: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """加载 TRAIN+TEST 合并的整数据集（路线 A 源池"把整档案当源候选"口径）。

    use_npz（默认 True）：`<cache_dir>/<name>/<name>_merged.npz` 存在且 crop_len 匹配时直读（export_uea_merged_npz 的产物，
    与解析逐位同），否则解析 .ts；False 强制解析。

    合并原始序列后一次性 pad_and_zscore + encode_labels：z-score 统计量与目标长度
    在**合并真实步**上统一计算（比按切分各自处理更一致），标签编码也在合并集上做
    一次（防两切分类别顺序不同导致 id 错配）。不等长两切分各自 L_max 不同也由此
    对齐到统一 T。返回 X (N, T, C) float32、y (N,) int64（0..k-1）。

    注意：本函数不带任何"源/目标"角色判断——泄漏防护（铁律 6）在 source_pool 层做，
    该层拒绝对目标池数据集调用本函数。
    """
    if use_npz:
        p = merged_npz_path(name, cache_dir)
        if p.exists():
            with np.load(p) as z:
                if int(z["crop_len"]) == (-1 if crop_len is None else int(crop_len)):
                    return np.ascontiguousarray(z["X"], dtype=np.float32), np.asarray(z["y"], dtype=np.int64)
    seqs, y = _load_raw_seqs(name, "TRAIN", cache_dir)
    seqs_te, y_te = _load_raw_seqs(name, "TEST", cache_dir)
    seqs = seqs + seqs_te
    y = np.concatenate([y, y_te])
    Xt = pad_and_zscore(seqs, crop_len=crop_len)
    yi, _ = encode_labels(y)
    return Xt, yi


SCALER_MODES = ("split", "train")


def load_uea_pair(
    name: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    crop_len: int | None = None,
    scaler: str = "split",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """官方 TRAIN/TEST 两切分一起加载 → (Xtr, ytr, Xte, yte)（全量协议对齐 M8）。

    scaler="split"（既有口径）：与依次调用 load_uea_cached(name, "TRAIN") /
        load_uea_cached(name, "TEST") **逐位相同**（各切分自己的统计量、自己的
        目标长度、自己的标签编码）。
    scaler="train"（TS2Vec 官方 datautils.load_UEA 语义）：StandardScaler 只在
        TRAIN 上 fit（按通道、全部真实时间步，ddof=0）后 transform 两切分；目标长度
        取两切分真实长度最大值（等长数据集 = L）；标签 id 在两切分**并集**上编码
        （官方在 TRAIN 唯一值上编码——两者在"TEST 类集 ⊆ TRAIN 类集"时一致，并集
        更稳，TEST 缺某类时 id 也不重排）。
    """
    if scaler not in SCALER_MODES:
        raise ValueError(f"未知 scaler={scaler!r}（可选 {SCALER_MODES}）")
    if scaler == "split":
        Xtr, ytr = load_uea_cached(name, "TRAIN", cache_dir=cache_dir, crop_len=crop_len)
        Xte, yte = load_uea_cached(name, "TEST", cache_dir=cache_dir, crop_len=crop_len)
        return Xtr, ytr, Xte, yte
    seqs_tr, y_tr = _load_raw_seqs(name, "TRAIN", cache_dir)
    seqs_te, y_te = _load_raw_seqs(name, "TEST", cache_dir)
    real_tr, real_te = _crop_real(seqs_tr, crop_len), _crop_real(seqs_te, crop_len)
    target = (crop_len if crop_len is not None
              else max(s.shape[0] for s in real_tr + real_te))
    mean, std = _fit_channel_stats(real_tr)            # 只在 TRAIN 真实步上拟合
    Xtr = _apply_pad_zscore(real_tr, target, mean, std)
    Xte = _apply_pad_zscore(real_te, target, mean, std)
    y_all, _classes = encode_labels(np.concatenate([np.asarray(y_tr), np.asarray(y_te)]))
    return Xtr, y_all[: len(y_tr)], Xte, y_all[len(y_tr):]


def segment_train(X: np.ndarray, max_train_length: int) -> np.ndarray:
    """官方 TS2Vec ``fit`` 的 ``max_train_length`` 分段（ts2vec.py:66-69 ``split_with_nan``），在加载层实现、
    trainer 零改动（WP10-UEA30，只对 EigenWorms L=17984 生效）。

    ``sections = L // max_train_length``；``sections < 2`` 原样返回**同一对象**；否则沿时间轴
    ``np.array_split`` 成 sections 段，各段右侧补零到首段长度后沿样本轴拼接 → (N·sections, ceil(L/sections), C)，
    样本 i 的第 j 段在下标 i·sections + j，dtype 不变。只用于预训练输入；截点评估仍编码全长序列（官方 SVM 用未切分的
    train 编码）。论文未给全，按 X 语义补全：官方用 NaN 补齐再由 encoder 置零，本仓库增强链（freq_mask 的 FFT）不耐
    NaN，这里直接补零 = z-score 后的通道均值，与 ``_apply_pad_zscore`` 的填充语义一致。
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X 须为 (N, T, C)，得到 shape={X.shape}")
    sections = X.shape[1] // int(max_train_length)
    if sections < 2:
        return X
    parts = np.array_split(X, sections, axis=1)
    T = parts[0].shape[1]
    out = np.zeros((X.shape[0] * sections, T, X.shape[2]), dtype=X.dtype)
    for j, p in enumerate(parts):
        out[j::sections, : p.shape[1]] = p
    return out
