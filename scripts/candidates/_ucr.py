"""UCR 单变量档案的共享加载助手（换池调研：广撒网候选的 UCR 支线）。

与 HAR 那批**连续传感器流**候选的根本差别：UCR 是**已切好样本**的档案（每条 case
就是一个样本），不需要开窗——故各 UCR adapter 的 SPEC 一律带 `"segmented": True`，
build() 返回档案原生长度，驱动只查 L≥50 的 encoder 适用边界。

加载口径与 `autocls.data.uea.load_uea_merged` 对齐（换池调研要跨 UEA/UCR 比统计，
两边口径不同会让表不可比）：
1. **TRAIN+TEST 合并**当整档案用——UCR 官方切分是固定的一次划分，本项目只把整档案
   当"源候选池的一个数据集"，切分语义在下游任务切片层重建。
2. 标签编码在**合并集**上做一次（`uea.encode_labels`），防两切分类别顺序不同导致 id 错配。
3. 解析走 `uea._load_raw_seqs`（同一个 aeon 调用 + 同一套 ragged 容错回退），不另起炉灶。

唯一的差别：本模块**不做 z-score**（`load_uea_merged` 内联做了）——候选管线的标准化
由驱动 `prepare_candidates.py` 统一负责（adapter 契约），这里返回未标准化 float64。

下载：timeseriesclassification.com 现挂 reCAPTCHA 安全墙（aeon 内置 URL 直接 401），
沿用 `results/uea_survey/scripts/uea_survey.py` §ensure_local 的做法——从 **Zenodo 镜像**
（aeon 的 `tsc_zenodo` 映射）拉 `.ts` 到 aeon 期望的 `<extract_path>/<name>/` 布局，
之后 aeon 纯本地解析、不走网。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from autocls.data.candidates import RAW_ROOT
from autocls.data.uea import _load_raw_seqs, encode_labels

# UCR .ts 缓存根（aeon 期望 <root>/<name>/<name>_{TRAIN,TEST}.ts）
UCR_CACHE = RAW_ROOT / "ucr_cache"

ZENODO_FILE = "https://zenodo.org/record/{zid}/files/{name}{suffix}"

# aeon 1.3.0 的 tsc_zenodo 有错条目：NonInvasiveFetalECGThorax1 指向的 11197817 实为
# MoteStrain（Zenodo API 核实）。这里按数据集名重新查到的正确 record 覆盖。
# 注意这类错配**不会**造成静默串档案：URL 里带 `<name>_TRAIN.ts` 文件名，记录不对就是
# 404（本条正是这么暴露的）；_check_problem_name 再补一道解析后的校验。
ZENODO_OVERRIDE = {
    "NonInvasiveFetalECGThorax1": 11197821,
}


def _check_problem_name(path: Path, name: str) -> None:
    """核对 .ts 头部 `@problemName` 与期望数据集名一致（防镜像映射错配串档案）。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.strip().lower()
            if low.startswith("@problemname"):
                got = line.strip().split(None, 1)[1].strip() if len(line.split()) > 1 else ""
                if got.lower() != name.lower():
                    raise ValueError(f"{path.name}: @problemName={got!r} 与期望 {name!r} 不符")
                return
            if low == "@data":
                break   # 没有 @problemName 就不校验（不是所有 .ts 都写）


def _ensure_local(name: str, cache_dir: Path) -> Path:
    """确保 `<cache_dir>/<name>/<name>_{TRAIN,TEST}.ts` 就位（已存在则不重下，铁律 3）。"""
    import requests
    from aeon.datasets._data_loaders import tsc_zenodo

    ddir = Path(cache_dir) / name
    ddir.mkdir(parents=True, exist_ok=True)
    if name not in ZENODO_OVERRIDE and name not in tsc_zenodo:
        raise KeyError(f"{name}: aeon 的 tsc_zenodo 无此档案，无法走镜像下载")
    zid = ZENODO_OVERRIDE.get(name, tsc_zenodo.get(name))
    for suffix in ("_TRAIN.ts", "_TEST.ts"):
        dst = ddir / f"{name}{suffix}"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = ZENODO_FILE.format(zid=zid, name=name, suffix=suffix)
        print(f"    fetch {url}", flush=True)
        # 用 requests 默认 UA：Zenodo 对伪装成浏览器的 UA 返回 403（uea_survey 踩过）
        with requests.get(url, timeout=600, stream=True) as r:
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dst)   # 原子替换：半份下载不会被当成缓存命中
        print(f"    saved {dst.name}  {dst.stat().st_size / 1e6:.1f} MB", flush=True)
    for suffix in ("_TRAIN.ts", "_TEST.ts"):
        _check_problem_name(ddir / f"{name}{suffix}", name)
    return ddir


def build_ucr(
    name: str, domain: str, *, cache_dir: str | Path = UCR_CACHE
) -> tuple[np.ndarray, np.ndarray, dict]:
    """UCR 单变量档案 → (X, y, meta)，X (N, T, 1) float64 **未标准化**、y (N,) int64。

    等长/单变量/有限值三项都**显式校验后报错**，不静默兜底：UCR 里混着不等长变体
    （…VaryingLength）与含缺失变体（…MissingValues），静默 pad/插值会把"表征学得到
    什么"这个被测量本身污染掉，宁可让该候选整个失败、由驱动记进 prepare_failures.jsonl。
    """
    cache_dir = Path(cache_dir)
    _ensure_local(name, cache_dir)

    seqs_tr, y_tr = _load_raw_seqs(name, "TRAIN", cache_dir)
    seqs_te, y_te = _load_raw_seqs(name, "TEST", cache_dir)
    seqs = list(seqs_tr) + list(seqs_te)          # 整档案 = TRAIN+TEST
    y_raw = np.concatenate([np.asarray(y_tr), np.asarray(y_te)])

    lengths = {int(s.shape[0]) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"{name}: 档案不等长（L∈{sorted(lengths)[:6]}…），UCR 支线只收等长档案")
    channels = {int(s.shape[1]) for s in seqs}
    if channels != {1}:
        raise ValueError(f"{name}: 期望单变量 C=1，实得 C∈{sorted(channels)}")

    X = np.stack([np.asarray(s, dtype=np.float64) for s in seqs])   # (N, T, 1)
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        raise ValueError(f"{name}: 含 {n_bad} 个非有限值（NaN/Inf），不做插值")
    y, classes = encode_labels(y_raw)

    meta = {
        "domain": domain,
        "n_train": int(len(seqs_tr)),
        "n_test": int(len(seqs_te)),
        "class_names": [str(c) for c in classes],
        "source": "UCR archive (Zenodo mirror)",
        "merged_splits": True,
    }
    return X, y, meta
