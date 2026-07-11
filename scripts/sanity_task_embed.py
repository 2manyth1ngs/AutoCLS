"""task_embed 验收（TECHNICAL_DESIGN.md 模块 5）：E 的一致性与区分力 sanity check。

单测（tests/test_task_embed.py）只验证形状/确定性/集合语义；E 是否携带"数据
分布特征"必须在真实 HAR 上验：若所有任务的 E 高度相似（相似度矩阵≈全 1），
后续 T-CLSC 会退化成"忽略任务输入"，必须在此拦截。

A. 一致性：同一任务（完整 HAR train、同 seed）独立编码两次，余弦 ≈1
   （GPU 卷积非确定性容许逐位差异，见 memory：±5e-3 内不算回归）。
B. 区分力：人为构造的不同子任务两两余弦相似度矩阵——
   - full_s0 / full_s1：完整 HAR，不同随机种子（同构任务 → 期望相似度高）
   - cls012 / cls345：只含类别 {0,1,2} / {3,4,5} 的样本（标签仅用于切子集，
     不进入编码）
   - chA / chB：通道 0-3（body_acc xyz + gyro x）/ 通道 5-8（gyro z + total_acc xyz）
   期望：同构任务相似度明显高于类别/通道子集间的相似度。

所有任务共享同一份 TaskSetEncoder 权重（固定 seed 初始化，尚未与 T-CLSC 联合
训练）——跨任务可比的前提。TS2Vec 每任务独立无监督训练（AutoCTS++ 语义）。

用法（在仓库根目录）：
    python scripts/sanity_task_embed.py            # iters=600（AutoCTS++ 大数据默认）
    python scripts/sanity_task_embed.py --iters 200 --device cpu   # 快速冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.datasets import load_har  # noqa: E402
from autocls.task_embed import TaskSetEncoder, embed_task  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # TS2Vec 训练量：AutoCTS++ ts2vec.py 对 size>1e5 的数据默认 n_iters=600
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--sample-num", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)  # STD_BATCH_SIZE
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "HAR" / "har.npz")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "task_embed_sanity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, _, _ = load_har(args.data)  # (7352, 128, 9)；y 只用于切类别子任务
    print(f"HAR train: X={X.shape}, device={args.device}, ts2vec_iters={args.iters}")

    # 共享 SetTransformer 权重（跨任务 E 可比的前提）
    torch.manual_seed(0)
    set_enc = TaskSetEncoder(dim_input=256, nz=256)

    # (任务名, 样本子集, seed)。seed 影响 TS2Vec 初始化/训练随机性与窗口采样。
    m012, m345 = np.isin(y, [0, 1, 2]), np.isin(y, [3, 4, 5])
    tasks = {
        "full_s0":        (X, 0),
        "full_s0_rerun":  (X, 0),  # check A：同任务同 seed 重复编码
        "full_s1":        (X, 1),  # check B：同构任务、不同随机性
        "cls012_s0":      (X[m012], 0),
        "cls345_s0":      (X[m345], 0),
        "chA_s0":         (X[:, :, 0:4], 0),
        "chB_s0":         (X[:, :, 5:9], 0),
    }

    embeds: dict[str, np.ndarray] = {}
    for name, (Xt, seed) in tasks.items():
        t0 = time.time()
        embeds[name] = embed_task(
            Xt, ts2vec_iters=args.iters, sample_num=args.sample_num,
            batch_size=args.batch_size, set_encoder=set_enc,
            device=args.device, seed=seed,
        )
        print(f"  {name:15s} X={Xt.shape}  ‖E‖={np.linalg.norm(embeds[name]):.4f}"
              f"  ({time.time() - t0:.1f}s)")

    # ---- check A：一致性 ----
    cos_a = cosine(embeds["full_s0"], embeds["full_s0_rerun"])
    max_diff = float(np.abs(embeds["full_s0"] - embeds["full_s0_rerun"]).max())
    print(f"\n[A] 同任务同 seed 重复编码：cosine={cos_a:.6f}  max|Δ|={max_diff:.2e}")

    # ---- check B：区分力矩阵 ----
    names = ["full_s0", "full_s1", "cls012_s0", "cls345_s0", "chA_s0", "chB_s0"]
    mat = np.array([[cosine(embeds[a], embeds[b]) for b in names] for a in names])
    print("\n[B] 两两余弦相似度矩阵：")
    print(" " * 12 + "".join(f"{n:>11s}" for n in names))
    for i, a in enumerate(names):
        print(f"{a:>12s}" + "".join(f"{mat[i, j]:11.4f}" for j in range(len(names))))

    same = mat[0, 1]  # full_s0 vs full_s1（同构不同随机性）
    cross = [mat[names.index(p), names.index(q)] for p, q in [
        ("full_s0", "cls012_s0"), ("full_s0", "cls345_s0"),
        ("cls012_s0", "cls345_s0"),
        ("full_s0", "chA_s0"), ("full_s0", "chB_s0"), ("chA_s0", "chB_s0"),
    ]]
    print(f"\n同构任务相似度（full_s0 vs full_s1）：{same:.4f}")
    print(f"异构任务相似度：min={min(cross):.4f} max={max(cross):.4f} "
          f"mean={np.mean(cross):.4f}")

    out = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "check_a": {"cosine": cos_a, "max_abs_diff": max_diff},
        "check_b": {"names": names, "cosine_matrix": mat.tolist(),
                    "same_task": same, "cross_task": cross},
        "embeddings": {k: v.tolist() for k, v in embeds.items()},
    }
    path = args.outdir / "sanity_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n结果已存 {path}")


if __name__ == "__main__":
    main()
