"""WP10-UEA30 pod 侧：从 Zenodo 直下 UEA 数据集的 <Name>_TRAIN.ts / <Name>_TEST.ts 到 data/uea_cache/<Name>/，并按清单 md5 对账。

    python scripts/pa_pod/uea30_fetch.py <Name> [<Name> ...] [--cache-dir data/uea_cache] [--md5 md5_uea_ts.txt]

逻辑复制自 results/uea_survey/scripts/uea_survey.py:ensure_local（不 import 它）：
- 记录号来自 aeon 的 tsc_zenodo 映射（aeon 1.3.0 定义在 aeon.datasets.tsc_datasets，经 aeon.datasets._data_loaders 再导出）；
- Zenodo 的 TSC 记录只放两份 .ts，没有 <Name>.zip（2026-09-06 实测 RacketSports.zip 404、RacketSports_TRAIN.ts 200，
  /api/records/<id> 的 files 也只有两份 .ts），故直下 .ts：https://zenodo.org/record/<id>/files/<Name>_<SPLIT>.ts；
- requests 默认 UA（Zenodo 对伪装浏览器 UA 回 403）；流式写 <dst>.part 再 os.replace；每份文件重试 3 次、超时 120 s；
- md5：清单（--md5，bundle 的 md5_uea_ts.txt，行格式 "md5  相对路径"，按文件名索引，缓存目录可不同）优先；
  无清单条目时退到 Zenodo API 的 files[].checksum；两者都无只查非空（ensure_local 语义）。
- 两文件已在且 md5 与清单一致 → 跳过；任一下载失败或对不上 → 打印靶名，退出码 1（由 uea30_setup.sh 决定回退到上传的分包）。
只用标准库 + requests（+ aeon 的映射；import 失败 → 退出码 2）。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

try:
    from aeon.datasets._data_loaders import tsc_zenodo
except Exception:  # noqa: BLE001
    try:
        from aeon.datasets.tsc_datasets import tsc_zenodo
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: 无法导入 aeon 的 tsc_zenodo 映射：{type(e).__name__}: {e}", flush=True)
        sys.exit(2)

SPLITS = ("TRAIN", "TEST")
RETRIES = 3
TIMEOUT_S = 120
CHUNK = 1 << 20


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Optional[Path]) -> dict[str, str]:
    """"md5  相对路径" → {basename: md5}；basename <Name>_<SPLIT>.ts 全局唯一。"""
    if path is None:
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[Path(parts[-1].replace("\\", "/")).name] = parts[0].lower()
    return out


def zenodo_checksums(zid: int) -> dict[str, str]:
    """{file_key: md5}，取不到返回 {}（只在清单缺条目时用）。"""
    try:
        r = requests.get(f"https://zenodo.org/api/records/{zid}", timeout=TIMEOUT_S)  # 默认 UA
        r.raise_for_status()
        return {f["key"]: f["checksum"].split(":", 1)[-1].lower()
                for f in r.json().get("files", []) if f.get("checksum")}
    except Exception as e:  # noqa: BLE001
        print(f"    (Zenodo API 校验和不可得：{type(e).__name__}: {e})", flush=True)
        return {}


def download(url: str, dst: Path) -> bool:
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(url, timeout=TIMEOUT_S, stream=True) as r:  # default UA; Zenodo 403s spoofed UA
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
            if tmp.exists():
                tmp.unlink()
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    return False


def check(p: Path, want: Optional[str]) -> tuple[bool, str]:
    if not p.is_file() or p.stat().st_size == 0:
        return False, "missing"
    if want is None:
        return True, "present (no md5 to compare)"
    got = md5_file(p)
    return got == want, ("md5 OK" if got == want else f"md5 MISMATCH got {got} want {want}")


def fetch(name: str, cache: Path, manifest: dict[str, str]) -> bool:
    ddir = cache / name
    ddir.mkdir(parents=True, exist_ok=True)
    files = {s: ddir / f"{name}_{s}.ts" for s in SPLITS}
    want = {s: manifest.get(files[s].name) for s in SPLITS}
    if all(want[s] is not None and check(files[s], want[s])[0] for s in SPLITS):
        print(f"[skip] {name}: 两文件已在且 md5 与清单一致", flush=True)
        return True
    if name not in tsc_zenodo:
        print(f"[FAIL] {name}: 不在 aeon tsc_zenodo 映射", flush=True)
        return False
    zid = int(tsc_zenodo[name])
    api = zenodo_checksums(zid) if any(want[s] is None for s in SPLITS) else {}
    ok = True
    for s in SPLITS:
        dst = files[s]
        expect = want[s] if want[s] is not None else api.get(dst.name)
        good, why = check(dst, expect)
        if good:
            print(f"  [keep] {dst.name}  {why}", flush=True)
            continue
        url = f"https://zenodo.org/record/{zid}/files/{dst.name}"
        print(f"  fetch {url}", flush=True)
        if not download(url, dst):
            print(f"[FAIL] {name}: 下载失败 {dst.name}", flush=True)
            ok = False
            continue
        good, why = check(dst, expect)
        print(f"  saved {dst.name}  {dst.stat().st_size / 1e6:.1f} MB  {why}", flush=True)
        if not good:
            ok = False
    print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="UEA 数据集名（aeon tsc_zenodo 键，如 RacketSports PEMS-SF）")
    ap.add_argument("--cache-dir", default="data/uea_cache")
    ap.add_argument("--md5", default=None, help="bundle 的 md5_uea_ts.txt（原始字节 md5；无则退到 Zenodo API 校验和）")
    args = ap.parse_args(argv)
    manifest = load_manifest(Path(args.md5)) if args.md5 else {}
    cache = Path(args.cache_dir)
    failed = [n for n in args.names if not fetch(n, cache, manifest)]
    if failed:
        print(f"FAILED datasets: {' '.join(failed)}", flush=True)
        return 1
    print(f"all ok: {' '.join(args.names)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
