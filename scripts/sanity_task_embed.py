"""task_embed 验收（定稿语义：单变量共享冻结 TS2Vec + 泄漏防护，铁律 6）。

单测（tests/test_task_embed.py）只验证形状/确定性/集合语义/API 防护；E 是否
携带"数据分布特征"必须在真实 HAR 上验：若所有任务的 E 高度相似且结构反常，
后续 T-CLSC 会退化成"忽略任务输入"，必须在此拦截。

流程：fit_shared_ts2vec 在 HAR train（当前源池；task_slicer 落地后应只用
"造 label 的切片"，留出测试切片不参与——铁律 6）上训练一份单变量 TS2Vec，
冻结后编码全部人工子任务。

A. 一致性：同一任务（完整 HAR train、同 seed）独立编码两次，余弦 ≈1
   （GPU 卷积非确定性容许逐位差异，见 memory：±5e-3 内不算回归）。
B. 区分力：人为构造的不同子任务两两余弦相似度矩阵，报告两级——
   - E 级：SetTF(TS2Vec(windows))。SetTF 未训练（M3 起与 T-CLSC 联合训练），
     随机权重把 raw cos 压到 0.99 附近属已知且可接受；
   - TS2Vec 级基线（**关键信号**）：full_series 池化后跨窗口取均值，不过
     SetTF——检验"SetTF 的输入是否携带任务分布信号"。
   任务集：
   - full_s0 / full_s1：完整 HAR，不同窗口采样种子（同构任务 → 期望相似度高）
   - cls012 / cls345：只含类别 {0,1,2} / {3,4,5}（标签仅用于切子集，不进编码；
     期望明显更低，诊断实验 3 量级 ≈0.65）
   - chA / chB：通道 0-3 / 5-8 真子集（单变量编码器原生支持，无零填充；
     期望中等，≈0.92）
   结构性检查：同构任务相似度必须高于一切异构对（旧"每任务重训"路径的
   反常正是同构 < 异构，见 results/task_embed_sanity/diag_shared_ts2vec.txt）。

用法（在仓库根目录）：
    python scripts/sanity_task_embed.py            # iters=600（AutoCTS++ 大数据默认）
    python scripts/sanity_task_embed.py --iters 100 --device cpu   # 快速冒烟
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
from autocls.task_embed import (  # noqa: E402
    TaskSetEncoder,
    embed_task,
    encode_task_windows,
    fit_shared_ts2vec,
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_matrix(vecs: dict[str, np.ndarray], names: list[str], title: str) -> np.ndarray:
    mat = np.array([[cosine(vecs[a], vecs[b]) for b in names] for a in names])
    print(f"\n{title}")
    print(" " * 12 + "".join(f"{n:>11s}" for n in names))
    for i, a in enumerate(names):
        print(f"{a:>12s}" + "".join(f"{mat[i, j]:11.4f}" for j in range(len(names))))
    return mat


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

    # 共享冻结单变量 TS2Vec（铁律 6：只在源池=HAR train 上训练这一次）
    t0 = time.time()
    shared = fit_shared_ts2vec(
        X, n_iters=args.iters, batch_size=args.batch_size, device=args.device, seed=0,
    )
    print(f"shared univar TS2Vec: fitted on (N*C={X.shape[0] * X.shape[2]}, T={X.shape[1]}, 1), "
          f"{args.iters} iters, frozen ({time.time() - t0:.0f}s)")

    # 共享 SetTransformer 权重（跨任务 E 可比的前提；M3 起与 T-CLSC 联合训练）
    torch.manual_seed(0)
    set_enc = TaskSetEncoder(dim_input=shared.output_dims, nz=256)

    # (任务名, 样本子集, seed)。共享编码器已冻结，seed 只影响窗口采样。
    m012, m345 = np.isin(y, [0, 1, 2]), np.isin(y, [3, 4, 5])
    tasks = {
        "full_s0":        (X, 0),
        "full_s0_rerun":  (X, 0),  # check A：同任务同 seed 重复编码
        "full_s1":        (X, 1),  # check B：同构任务、不同窗口采样
        "cls012_s0":      (X[m012], 0),
        "cls345_s0":      (X[m345], 0),
        "chA_s0":         (X[:, :, 0:4], 0),  # 真通道子集：单变量编码器无需零填充
        "chB_s0":         (X[:, :, 5:9], 0),
    }

    embeds: dict[str, np.ndarray] = {}    # E 级：SetTF(TS2Vec)
    baselines: dict[str, np.ndarray] = {}  # TS2Vec 级：full_series 池化跨窗口均值
    for name, (Xt, seed) in tasks.items():
        t0 = time.time()
        embeds[name] = embed_task(
            Xt, shared, sample_num=args.sample_num, set_encoder=set_enc, seed=seed,
        )
        baselines[name] = encode_task_windows(
            Xt, shared, sample_num=args.sample_num, seed=seed,
            encoding_window="full_series",
        ).mean(axis=0)
        print(f"  {name:15s} X={Xt.shape}  ‖E‖={np.linalg.norm(embeds[name]):.4f}"
              f"  ({time.time() - t0:.1f}s)")

    # ---- check A：一致性 ----
    cos_a_e = cosine(embeds["full_s0"], embeds["full_s0_rerun"])
    cos_a_b = cosine(baselines["full_s0"], baselines["full_s0_rerun"])
    max_diff = float(np.abs(embeds["full_s0"] - embeds["full_s0_rerun"]).max())
    print(f"\n[A] 同任务同 seed 重复编码：E 级 cosine={cos_a_e:.6f}  "
          f"TS2Vec 级 cosine={cos_a_b:.6f}  max|ΔE|={max_diff:.2e}")

    # ---- check B：区分力矩阵（两级） ----
    names = ["full_s0", "full_s1", "cls012_s0", "cls345_s0", "chA_s0", "chB_s0"]
    mat_e = print_matrix(embeds, names, "[B] E 级（SetTF 未训练，raw cos 偏高属预期）：")
    mat_b = print_matrix(baselines, names, "[B] TS2Vec 级基线（关键信号）：")

    same = mat_b[names.index("full_s0"), names.index("full_s1")]
    hetero_pairs = [
        ("full_s0", "cls012_s0"), ("full_s0", "cls345_s0"), ("cls012_s0", "cls345_s0"),
        ("full_s0", "chA_s0"), ("full_s0", "chB_s0"), ("chA_s0", "chB_s0"),
    ]
    hetero = {f"{p}~{q}": mat_b[names.index(p), names.index(q)] for p, q in hetero_pairs}
    cls_pair = hetero["cls012_s0~cls345_s0"]
    ch_pair = hetero["chA_s0~chB_s0"]
    structural_ok = same > max(hetero.values())

    print(f"\n同构任务相似度（full_s0 vs full_s1，TS2Vec 级）：{same:.4f}")
    print(f"类别子集间（cls012 vs cls345）：{cls_pair:.4f}（预期 ≈0.65 量级）")
    print(f"通道子集间（chA vs chB）：{ch_pair:.4f}（预期 ≈0.92 量级）")
    print(f"异构对：min={min(hetero.values()):.4f} max={max(hetero.values()):.4f}")
    print(f"结构性检查（同构 > 一切异构对）：{'通过' if structural_ok else '不通过'}")

    out = {
        "semantics": "shared frozen univariate TS2Vec (fit_shared_ts2vec on HAR train)",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "check_a": {"cosine_e": cos_a_e, "cosine_ts2vec": cos_a_b, "max_abs_diff": max_diff},
        "check_b": {
            "names": names,
            "cosine_matrix_e": mat_e.tolist(),
            "cosine_matrix_ts2vec": mat_b.tolist(),
            "same_task_ts2vec": float(same),
            "hetero_ts2vec": {k: float(v) for k, v in hetero.items()},
            "structural_ok": bool(structural_ok),
        },
        "embeddings": {k: v.tolist() for k, v in embeds.items()},
    }
    path = args.outdir / "sanity_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n结果已存 {path}")


if __name__ == "__main__":
    main()
