"""workpoint_confirm 的幂等种子写入（协议 §7；命令重放事故教训 2026-08-10）。

按键 (task_id, cls_key, iters) 增补：目标文件已有的键一律跳过，只 append 缺失
记录，绝不截断（禁止 open(...,"w")）。重复执行/命令重放无害。

用法（在仓库根目录）：
    python results/workpoint_confirm/seed_fill.py            # Handwriting @60/@120 种子
    python results/workpoint_confirm/seed_fill.py --uwave290 # 回落分支：UWave@290 种子
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results" / "workpoint_confirm" / "uea"
FILL = OUTDIR / "trained_fill.jsonl"
BASE_SRC = ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl"
BASE_DST = OUTDIR / "baseline_cache.jsonl"

# (源文件, 数据集, iters 集合) —— 种子来源
SEEDS = {
    "handwriting": (
        ROOT / "results" / "iters_workpoint" / "uea" / "trained_fill.jsonl",
        "Handwriting", {60, 120},
    ),
    "uwave290": (
        ROOT / "results" / "stage1_rerun_20260804" / "results" / "labels_route_a"
        / "rprime_cache.jsonl",
        "UWaveGestureLibrary", {290},
    ),
}


def _key(r: dict) -> tuple:
    return (r["task_id"], r["cls_key"], int(r["iters"]))


def seed(which: str) -> None:
    src, dataset, iters_set = SEEDS[which]
    existing: set[tuple] = set()
    if FILL.exists():
        for line in FILL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(_key(json.loads(line)))
    picked, skipped = [], 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dataset") != dataset or int(r.get("iters", -1)) not in iters_set:
            continue
        if _key(r) in existing:
            skipped += 1
            continue
        picked.append(line)
        existing.add(_key(r))
    if picked:
        with FILL.open("a", encoding="utf-8") as f:   # append-only，幂等
            for line in picked:
                f.write(line + "\n")
    print(f"[{which}] 源 {src.name}: 新增 {len(picked)} 条，已存在跳过 {skipped} 条")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uwave290", action="store_true",
                    help="回落分支：种入 UWave@290 的 5090 记录（协议 §2-3）")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not BASE_DST.exists():                          # 存在即跳过，幂等
        shutil.copyfile(BASE_SRC, BASE_DST)
        print(f"baseline_cache 复制 → {BASE_DST}")
    else:
        print("baseline_cache 已存在，跳过")
    seed("handwriting")
    if args.uwave290:
        seed("uwave290")
    n = sum(1 for line in FILL.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"trained_fill.jsonl 现有 {n} 条")


if __name__ == "__main__":
    main()
