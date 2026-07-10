"""下载/预处理 HAR 与 Epilepsy（TECHNICAL_DESIGN.md 模块 9 之前的数据准备入口）。

用法（在仓库根目录）：
    python scripts/1_prepare_data.py --dataset har
    python scripts/1_prepare_data.py --dataset epilepsy [--epilepsy-csv 路径]

产物：
    data/HAR/har.npz             (X_train, y_train, X_test, y_test)
    data/Epilepsy/epilepsy.npz   同上
已存在则跳过（幂等，可安全重跑）。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.datasets import load_epilepsy_csv, prepare_har  # noqa: E402

HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def prepare_har_cmd(data_dir: Path) -> None:
    out = data_dir / "HAR" / "har.npz"
    if out.exists():
        print(f"[skip] {out} 已存在")
        return
    raw_root = data_dir / "raw" / "uci_har"
    outer_zip = raw_root / "uci_har.zip"
    inner_dir = raw_root / "inner" / "UCI HAR Dataset"
    if not inner_dir.exists():
        raw_root.mkdir(parents=True, exist_ok=True)
        if not outer_zip.exists():
            print(f"[download] {HAR_URL}")
            urllib.request.urlretrieve(HAR_URL, outer_zip)
        print(f"[extract] {outer_zip}")
        with zipfile.ZipFile(outer_zip) as z:
            z.extractall(raw_root)
        # UCI 的 zip 里再套一层 "UCI HAR Dataset.zip"
        with zipfile.ZipFile(raw_root / "UCI HAR Dataset.zip") as z:
            z.extractall(raw_root / "inner")
    print(f"[preprocess] {inner_dir} -> {out}")
    prepare_har(inner_dir, out)
    print(f"[done] {out}")


def prepare_epilepsy_cmd(data_dir: Path, csv_path: Path | None) -> None:
    import numpy as np

    out = data_dir / "Epilepsy" / "epilepsy.npz"
    if out.exists():
        print(f"[skip] {out} 已存在")
        return
    csv_path = csv_path or data_dir / "raw" / "Epileptic_Seizure_Recognition.csv"
    # UCI 官网已下架该数据集，无稳定直链；需手动下载 CSV（见 datasets.py 提示）
    X_train, y_train, X_test, y_test = load_epilepsy_csv(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"[done] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["har", "epilepsy", "all"], default="har")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--epilepsy-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.dataset in ("har", "all"):
        prepare_har_cmd(args.data_dir)
    if args.dataset in ("epilepsy", "all"):
        prepare_epilepsy_cmd(args.data_dir, args.epilepsy_csv)


if __name__ == "__main__":
    main()
