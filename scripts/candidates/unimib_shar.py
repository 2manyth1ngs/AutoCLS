"""UniMiB SHAR：智能手机加速度计 HAR + 跌倒检测（17 类，已切好样本）。

来源：Micucci et al. 2017（Applied Sciences 7(10):1101）。30 受试者（24 女），
Samsung Galaxy Nexus 裤兜采集，fs=50 Hz。每样本 151 步 ×3 轴，以加速度峰值为中心的
3 s 窗——**档案里已经切好**，不走 `window_stream` 开窗路径（SPEC `segmented=True`，
驱动放行 L=151≠128；151>50 也在 encoder 适用边界内）。

分发地址（两条都试过）：
- 官方 wp-content 直链 `…/uploads/2016/12/UniMiB-SHAR.zip` 已 **404**；官方页
  (http://www.sal.disco.unimib.it/technologies/unimib-shar/) 现在指向 Dropbox 分享链，
  但本机实测 ~1 MB/min 且中途卡死（14.7 MB 后不再增长），不可用。官方站 https
  证书还过期（DH_KEY_TOO_SMALL），探测只能走 http。
- 故实际取 GitHub 镜像上的三个 .mat（共 33.6 MB，5 分钟下完）。镜像不可尽信，因此
  `_check_integrity` 对着论文公开总数硬校验（11771 样本 / 17 类 / 30 受试者），
  对不上直接报错而不是静默产出一份来路不明的数据。

已知坑：
- `acc_data.mat` 是 (11771, 453)，453 = **[x(151) | y(151) | z(151)] 三段块状拼接**，
  不是 [x0,y0,z0,x1,…] 逐步交错。`_verify_axis_layout` 实测这一点，见该函数注释。
- `acc_labels.mat` 是 (11771, 3)：[活动类 id 1..17, 受试者 id 1..30, 试次 1..6]，只取第 1 列。
- `acc_names.mat` 是 (2, 17)：第 0 行长名（'Going upstairs'），第 1 行短名（'GoingUpS'），
  取第 0 行。别 flatten——会得到 34 个混着两种命名的条目。
"""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from autocls.data.candidates import RAW_ROOT, fetch_archive

# GitHub 镜像的 data/ 目录（官方 zip 见模块 docstring：直链 404 + Dropbox 卡死）
_MIRROR = ("https://raw.githubusercontent.com/jacky18008/NCCU_Coding_is_Magic"
           "/master/UniMiB-SHAR/data")

SPEC = {
    "name": "UniMiB-SHAR",
    "domain": "HAR",
    "fs": 50,
    "url": f"{_MIRROR}/acc_data.mat",
    "n_subjects": 30,
    "segmented": True,   # 档案已按峰值切好 151 步样本，驱动跳过 L=128 检查
    "note": ("17 类（9 日常动作 + 8 跌倒）；453 列 = x/y/z 各 151 步块状拼接；"
             "官方直链 404、官方 Dropbox 卡死 → 走 GitHub 镜像 + 论文总数硬校验"),
}

_RAW_DIR = RAW_ROOT / "unimib_shar" / "mirror"
_N_STEPS = 151
_N_CH = 3
# 论文公开总数（Applied Sciences 7(10):1101）——镜像完整性的对账基准
_EXPECT_N = 11771
_EXPECT_K = 17
_EXPECT_SUBJ = 30


def _verify_axis_layout(raw: np.ndarray) -> dict:
    """实证 453 = [x(151)|y(151)|z(151)] 块状拼接，而非 [x0,y0,z0,x1,…] 逐步交错。

    判据（平滑度）：50 Hz 加速度流相邻时刻应连续。把交错当块状（或反之）reshape，
    会把**异通道**的值当成相邻时刻，时间轴一阶差分随之变大。本机实测（前 256 样本）：
    块状 0.565 vs 交错 1.283 —— 交错解释下差分大 2.27×，块状成立。
    要求 ≥1.5× 余量，模棱两可（比值接近 1）时报错而不是赌一把。

    注：合加速度 sqrt(x²+y²+z²) 均值在两种排列下分别是 9.63 / 8.16 m/s²，**都**落在
    1 g 附近——它只能校验单位是 m/s²（而非 g 或原始计数），**不能**区分排列，
    别拿它当排列证据。两个数都记进 meta 备查。
    """
    sub = raw[:256].astype(np.float64)
    block = sub.reshape(-1, _N_CH, _N_STEPS)        # 块状假设：轴 2 是时间
    inter = sub.reshape(-1, _N_STEPS, _N_CH)        # 交错假设：轴 1 是时间
    d_block = float(np.abs(np.diff(block, axis=2)).mean())
    d_inter = float(np.abs(np.diff(inter, axis=1)).mean())
    if not d_inter > 1.5 * d_block:
        raise ValueError(
            f"UniMiB-SHAR 排列判据不成立：交错差分 {d_inter:.3f} 未超块状 "
            f"{d_block:.3f} 的 1.5×，档案格式可能已变，须重查再解析"
        )
    mag = float(np.sqrt((block ** 2).sum(axis=1)).mean())
    if not 5.0 < mag < 20.0:
        raise ValueError(f"UniMiB-SHAR 合加速度均值 {mag:.2f} 不在 1g≈9.8 m/s² 量级，单位存疑")
    return {"diff_block": round(d_block, 4), "diff_interleaved": round(d_inter, 4),
            "diff_ratio": round(d_inter / d_block, 3), "magnitude_mean": round(mag, 3)}


def _check_integrity(raw: np.ndarray, labels: np.ndarray) -> None:
    """对着论文公开总数校验镜像（走第三方副本就必须对账，否则等于信任来路不明的文件）。"""
    if raw.shape != (_EXPECT_N, _N_CH * _N_STEPS):
        raise ValueError(f"acc_data 形状 {raw.shape}，论文口径应为 ({_EXPECT_N}, {_N_CH * _N_STEPS})")
    if labels.shape != (_EXPECT_N, 3):
        raise ValueError(f"acc_labels 形状 {labels.shape}，应为 ({_EXPECT_N}, 3)")
    k, s = len(np.unique(labels[:, 0])), len(np.unique(labels[:, 1]))
    if (k, s) != (_EXPECT_K, _EXPECT_SUBJ):
        raise ValueError(f"镜像类数/受试者数 = {k}/{s}，论文口径 {_EXPECT_K}/{_EXPECT_SUBJ}")


def _load_class_names(mat_path) -> list[str] | None:
    """acc_names.mat 是 (2, 17) cell：第 0 行长名，第 1 行短名——取长名。

    解析失败返回 None：类名只是 meta 附注，不值得为它让整个 build 失败。
    """
    try:
        names = np.asarray(loadmat(mat_path)["acc_names"])
        row = names[0] if names.ndim == 2 and names.shape[0] == 2 else names.flatten()
        flat = [str(np.asarray(v).flat[0]) for v in row]
        return flat if len(flat) == _EXPECT_K else None
    except Exception:
        return None


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    paths = {f: fetch_archive(f"{_MIRROR}/{f}", _RAW_DIR / f, timeout=900)
             for f in ("acc_data.mat", "acc_labels.mat", "acc_names.mat")}

    raw = loadmat(paths["acc_data.mat"])["acc_data"]
    labels = loadmat(paths["acc_labels.mat"])["acc_labels"]
    _check_integrity(raw, labels)
    layout = _verify_axis_layout(raw)

    # (N, 453) → (N, 3, 151) → (N, 151, 3)：全项目 (N, T, C) 约定
    X = raw.astype(np.float64).reshape(-1, _N_CH, _N_STEPS).transpose(0, 2, 1)
    y = labels[:, 0].astype(np.int64)           # 1..17，驱动统一重编号 0..k-1

    meta = {
        "class_names": _load_class_names(paths["acc_names.mat"]),   # 按原始类 id 1..17 排
        "n_subjects": int(len(np.unique(labels[:, 1]))),
        "n_trials_per_subject": int(len(np.unique(labels[:, 2]))),
        "source": "github_mirror(jacky18008/NCCU_Coding_is_Magic)",
        "layout_check": layout,   # 排列判据实测值，落 json 备查
    }
    return X, y, meta
