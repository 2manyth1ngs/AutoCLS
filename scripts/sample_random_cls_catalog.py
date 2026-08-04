"""解耦实验随机 CLS 目录生成（预注册 results/decouple_random_cls/PREREGISTRATION.md §2）。

从 configs/space.yaml（策略空间唯一真源，铁律 1）**均匀随机**采样 n 条 CLS：
不看任何性能信息、不按好坏覆盖挑选、不与旧探针目录去重。旧探针 10 条按 HAR
排名跨好坏挑出（必含退化 CLS），两版闸门判据都被这个构成支配（剂量-反应
Pearson 0.751/0.719）；本目录把采样自由度压缩到 (seed, n) 两个数，均写死于
预注册文档并与目录同 commit——git 历史可验证"采样在前、结果在后"。

采样器直接复用 autocls.labeling._sample_unique（逐个 space.sample_random +
cls_key 去重的接受制），与阶段2 生产随机采样同一循环语义；前缀稳定：
将来加大 n 不改变前 n' 条，既有 (task, cls) → R' 缓存不作废（铁律 3）。

用法：
    python scripts/sample_random_cls_catalog.py            # 写默认目录（已存在则须 --force）
    python scripts/sample_random_cls_catalog.py --check    # 校验已提交目录 == 再生成
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.labeling import _sample_unique, cls_key  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "decouple_random_cls" / "random_cls_catalog.json"
DEFAULT_N = 20
DEFAULT_SEED = 20260804


def build_catalog(space: CLSSpace, *, n: int, seed: int) -> dict:
    """均匀随机 n 条 CLS → 目录 dict。条目只有 cls_key + cls 两个字段——
    刻意不携带任何性能/排名信息（带了就又给下游"按好坏取子集"开了口子）。"""
    sampled = _sample_unique(space, np.random.default_rng(seed), n, set())
    return {
        "source": "configs/space.yaml 均匀随机采样（无任何按性能/覆盖的挑选）",
        "note": ("解耦实验（PREREGISTRATION.md）随机 CLS 目录：sampler = "
                 "autocls.labeling._sample_unique(space, default_rng(seed), n, set())；"
                 "har_rank 字段刻意不存在"),
        "n": n,
        "seed": seed,
        "cls": [{"cls_key": cls_key(c), "cls": asdict(c)} for c in sampled],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="覆盖已存在目录（预注册后不该发生）")
    ap.add_argument("--check", action="store_true", help="只校验已提交目录==再生成，不写文件")
    args = ap.parse_args()

    space = CLSSpace.from_yaml()
    if args.check:
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        regen = build_catalog(space, n=committed["n"], seed=committed["seed"])
        if committed["cls"] != regen["cls"]:
            sys.exit(f"❌ {args.out} 与当前 space.yaml 下的再生成不一致——space.yaml 变过？")
        print(f"✓ 目录与再生成一致（n={committed['n']}, seed={committed['seed']}）")
        return

    if args.out.exists() and not args.force:
        sys.exit(f"❌ {args.out} 已存在。预注册目录不得静默覆盖；确要重生成用 --force。")
    cat = build_catalog(space, n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {args.out}（n={args.n}, seed={args.seed}）")
    for e in cat["cls"]:
        print(" ", e["cls_key"])


if __name__ == "__main__":
    main()
