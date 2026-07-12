"""δ 标定：从 M2 探针缓存分析 early-val R' 的同分噪声量级，为 configs/labels.yaml
的 delta 提供依据（模块 7 配对造 label 的 δ 过滤，动机见 autocls/labeling.py）。

方法：
1. 同分噪声刻画：full acc 几乎相同（|Δfull|<阈值）的 CLS 配对，其 early R'
   差值就是 early-val 对同质量 CLS 给出的伪差值（纯噪声）——看其分位数定噪声上界。
2. 决策曲线（labeling.delta_decision_curve）：候选 δ vs 配对保留比例 / early-full
   方向一致率。δ 取一致率明显抬升、再加大无增益的位置（噪声上界略高处）。

用法（在仓库根目录）：
    python scripts/calibrate_delta.py    # 读 results/m2_probe/cache.jsonl，
                                         # 写 results/delta_calibration/calibration.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.labeling import delta_decision_curve  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402

# m2_probe 旧缓存按 epoch 计量（HAR N=7352, bs=64）：5ep=575 / 40ep=4600 iters
_LEGACY_EPOCH_STEPS = 115
EARLY_ITERS_LEGACY, FULL_ITERS_LEGACY = 575, 4600


def load_early_full(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    recs: dict[tuple[str, int], dict] = {}
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                iters = r.get("iters", r.get("epochs", 0) * _LEGACY_EPOCH_STEPS)
                recs[(r["cls_hash"], iters)] = r
    early, full = [], []
    for h in sorted({h for h, _ in recs}):
        e, f = recs.get((h, EARLY_ITERS_LEGACY)), recs.get((h, FULL_ITERS_LEGACY))
        if e and f and np.isfinite(e["acc"]) and np.isfinite(f["acc"]):
            early.append(e["acc"])
            full.append(f["acc"])
    return np.array(early), np.array(full)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "results" / "m2_probe" / "cache.jsonl")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "delta_calibration")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.0, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03])
    args = ap.parse_args()

    early, full = load_early_full(args.cache)
    print(f"early/full 齐全且有效的 CLS 数: {len(early)}，配对数: {len(early) * (len(early) - 1) // 2}")

    # 1) 同分噪声刻画
    noise_rows = []
    d_early = np.array([abs(early[i] - early[j])
                        for i, j in itertools.combinations(range(len(early)), 2)])
    d_full = np.array([abs(full[i] - full[j])
                       for i, j in itertools.combinations(range(len(full)), 2)])
    print("\n同分噪声（|Δfull|<阈值 的配对，其 |Δearly| 分布 = 纯噪声幅度）：")
    for thr in (0.002, 0.003, 0.005):
        m = d_full < thr
        if m.sum():
            q = np.percentile(d_early[m], [50, 75, 90, 95, 100])
            row = {"full_tie_thr": thr, "n": int(m.sum()),
                   "dEarly_p50": round(q[0], 4), "dEarly_p75": round(q[1], 4),
                   "dEarly_p90": round(q[2], 4), "dEarly_p95": round(q[3], 4),
                   "dEarly_max": round(q[4], 4)}
            noise_rows.append(row)
            print(f"  |dFull|<{thr}: n={row['n']:3d}  |dEarly| p50={row['dEarly_p50']} "
                  f"p90={row['dEarly_p90']} p95={row['dEarly_p95']} max={row['dEarly_max']}")

    # 2) 决策曲线
    rows = delta_decision_curve(early, full, TASK_CLASSIFICATION, deltas=args.deltas)
    print(f"\n{'delta':>7} {'kept':>5} {'kept%':>6} {'agree%':>7}")
    for r in rows:
        print(f"{r['delta']:7.4f} {r['n_kept']:5d} {r['kept_frac'] * 100:5.1f}% "
              f"{r['agreement'] * 100:6.1f}%")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "source_cache": str(args.cache),
        "n_cls": len(early),
        "note": "early=575iters@bs64（旧口径，等量于现行 290@bs128 的样本遍数），full=4600iters@bs64",
        "same_quality_noise": noise_rows,
        "decision_curve": rows,
    }
    path = args.outdir / "calibration.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已存 {path}")


if __name__ == "__main__":
    main()
