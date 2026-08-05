"""候选数据集准备驱动（换池调研第1步）：下载 → 解析 → 开窗 → canonical npz + 真实统计。

用法：
    python scripts/prepare_candidates.py --list
    python scripts/prepare_candidates.py --datasets wisdm mhealth
    python scripts/prepare_candidates.py --all --skip-existing
    python scripts/prepare_candidates.py --table            # 汇总统计表（第1步产出）

adapter 契约见 `scripts/candidates/__init__.py`。驱动负责预注册的统一后处理
（`results/candidate_survey/PROTOCOL.md` §1）：

1. **剔稀类**：样本数 <MIN_CLASS_COUNT 的类整类剔除后重新编号 0..k-1。
2. **规模上限**：N>MAX_SAMPLES 时按类分层随机子采样到 MAX_SAMPLES（seed 固定）。
   原始 N 记入 meta 的 `n_samples_raw`，统计表两个都报。
3. **按通道 z-score**（float64 统计 → float32 落盘），与 uea.py / datasets.py 同口径。

这三步写死在驱动里而不是各 adapter 里：8 个 adapter 各写一遍必然发散，而它们直接
决定统计表里的 N/K——第1步的初筛判据就吃这两个数。
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):        # 本机控制台默认 GBK
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.data.candidates import (  # noqa: E402
    CANDIDATE_ROOT,
    WINDOW_LEN,
    WINDOW_STRIDE,
    describe,
    drop_rare_classes,
    save_candidate,
    zscore_channels,
)

# —— 预注册后处理常量（PROTOCOL §1，跑数据前写死）——
MIN_CLASS_COUNT = 64     # 类样本下限：低于此整类剔除（registry min_samples=32，留一倍余量）
MAX_SAMPLES = 30000      # 规模上限：切片每任务只取 1024，30k 足够且 npz <1 GB
SUBSAMPLE_SEED = 0

PKG = "scripts.candidates"
FAIL_LOG = ROOT / "results" / "candidate_survey" / "prepare_failures.jsonl"


def adapter_names() -> list[str]:
    pkg = importlib.import_module(PKG)
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                  if not m.name.startswith("_"))


def stratified_subsample(X, y, cap: int, seed: int):
    """按类比例分层子采样到 cap 条（保持类别分布，不改变不平衡比的量级）。"""
    if X.shape[0] <= cap:
        return X, y, False
    rng = np.random.default_rng(seed)
    keep = []
    vals, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.round(counts / counts.sum() * cap).astype(int))
    for v, q in zip(vals, quota):
        idx = np.where(y == v)[0]
        keep.append(rng.permutation(idx)[:q])
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel], True


def prepare_one(mod_name: str, *, skip_existing: bool) -> dict:
    mod = importlib.import_module(f"{PKG}.{mod_name}")
    spec = dict(mod.SPEC)
    name = spec["name"]
    if skip_existing and (CANDIDATE_ROOT / f"{name}.npz").exists():
        meta = json.loads((CANDIDATE_ROOT / f"{name}.json").read_text(encoding="utf-8"))
        print(f"[{name}] 已存在，跳过（N={meta['n_samples']}）", flush=True)
        return meta

    t0 = time.time()
    X, y, meta = mod.build()
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    # 连续流候选统一开窗到 WINDOW_LEN；已切好样本的档案（segmented=True，如 UniMiB 的
    # 151 步、Sleep-EDF 的 30 s epoch、UCR/UEA）保留档案原生长度，只查 L≥50 的 encoder 边界
    if X.ndim != 3:
        raise ValueError(f"{name}: build() 应返回 (N, T, C)，实得 {X.shape}")
    if spec.get("segmented"):
        if X.shape[1] < 50:
            raise ValueError(f"{name}: 原生长度 {X.shape[1]} < 50，低于 encoder 适用边界")
    elif X.shape[1] != WINDOW_LEN:
        raise ValueError(f"{name}: 流式候选应开窗到 (N, {WINDOW_LEN}, C)，实得 {X.shape}")
    n_raw, k_raw = X.shape[0], len(np.unique(y))

    X, y = drop_rare_classes(X, y, MIN_CLASS_COUNT)
    if X.shape[0] == 0:
        raise ValueError(f"{name}: 剔稀类（<{MIN_CLASS_COUNT}）后无样本")
    X, y, capped = stratified_subsample(X, y, MAX_SAMPLES, SUBSAMPLE_SEED)
    X = zscore_channels(X)

    full = {**spec, **meta,
            "n_samples_raw": n_raw, "n_classes_raw": k_raw,
            "n_classes_dropped": k_raw - len(np.unique(y)),
            "subsampled": capped, "window_len": WINDOW_LEN, "window_stride": WINDOW_STRIDE,
            "min_class_count": MIN_CLASS_COUNT, "seconds": round(time.time() - t0, 1)}
    save_candidate(name, X, y, full)
    d = describe(X, y)
    print(f"[{name}] N={d['n_samples']}(raw {n_raw}) L={d['seq_len']} C={d['n_channels']} "
          f"K={d['n_classes']}(raw {k_raw}) imb={d['imbalance_ratio']:.2f} "
          f"{time.time() - t0:.0f}s", flush=True)
    return {**full, **d}


def table() -> None:
    """汇总已备好的候选统计（第1步的统计表）。"""
    rows = []
    for p in sorted(CANDIDATE_ROOT.glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        print("尚无候选，先跑 --all")
        return
    print(f"{'数据集':<22}{'域':<10}{'N':>7}{'L':>5}{'C':>5}{'K':>4}{'不平衡':>8}"
          f"{'猜测':>7}{'fs':>6}{'受试':>5}")
    for r in sorted(rows, key=lambda r: (r.get("domain", ""), -r.get("n_samples", 0))):
        print(f"{r['name']:<22}{r.get('domain', '?'):<10}{r['n_samples']:>7}{r['seq_len']:>5}"
              f"{r['n_channels']:>5}{r['n_classes']:>4}{r['imbalance_ratio']:>8.2f}"
              f"{r['chance']:>7.3f}{str(r.get('fs', '?')):>6}"
              f"{str(r.get('n_subjects') or '?'):>5}")
    print(f"\n共 {len(rows)} 个候选 → {CANDIDATE_ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None, help="adapter 模块名（小写）")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\n".join(adapter_names()) or "（尚无 adapter）")
        return
    if args.table:
        table()
        return

    names = adapter_names() if args.all else (args.datasets or [])
    if not names:
        ap.error("指定 --datasets / --all / --list / --table 之一")
    ok, bad = [], []
    for n in names:
        try:
            prepare_one(n, skip_existing=args.skip_existing)
            ok.append(n)
        except Exception as e:  # 一个候选失败不该拖垮整批调研
            bad.append((n, f"{type(e).__name__}: {e}"))
            print(f"[{n}] 失败 {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if bad:
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            for n, msg in bad:
                f.write(json.dumps({"adapter": n, "error": msg}, ensure_ascii=False) + "\n")
    print(f"\n成功 {len(ok)}/{len(names)}" + (f"，失败 {[b[0] for b in bad]}" if bad else ""))


if __name__ == "__main__":
    main()
