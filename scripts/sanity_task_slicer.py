"""task_slicer 验收（路线 B：HAR 内 leave-out 的两级划分 + 切片多样性验证）。

单测（tests/test_task_slicer.py）只验证划分/切片/防泄漏接口的正确性；
"切出来的任务是否足够不同"必须在真实 HAR 上用已定稿的冻结共享 TS2Vec 验：
若所有源任务的 E 高度相似（TS2Vec 级 mean > 0.97 拉不开），comparator 会
退化成"忽略任务输入"，须在此拦截并回报讨论。

流程：
1. split_pools 把 HAR train 分层划成源池(70%)/留出池(30%)，切 16 源任务 + 8 留出任务；
2. fit_shared_ts2vec 只用 X[source_indices]（铁律 6：留出池零参与）；
3. 对每个源任务 materialize 后走 embed_task 算 E（共享 SetTF 权重）——同时给
   TS2Vec 级基线（full_series 池化跨窗口均值，不过 SetTF）；
4. 两两余弦矩阵（两级）+ off-diagonal min/mean/max；判据参照 task_embed 定稿：
   TS2Vec 级为关键信号（未训练 SetTF 把 E 级 raw cos 压到 0.99 附近属已知可接受），
   类别差异大的对应落 0.65~0.85 量级、整体 mean 明显 < 0.95；
5. 泄漏自检：两池零交集、任务样本各归其池、TS2Vec 训练数据 == 源池。

用法（在仓库根目录）：
    python scripts/sanity_task_slicer.py            # iters=600（与 task_embed 验收同量）
    python scripts/sanity_task_slicer.py --iters 50 --device cpu   # 快速冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.datasets import load_har  # noqa: E402
from autocls.data.task_slicer import (  # noqa: E402
    assert_tasks_disjoint,
    make_leaveout_split,
    materialize_task,
)
from autocls.task_embed import (  # noqa: E402
    TaskSetEncoder,
    embed_task,
    encode_task_windows,
    fit_shared_ts2vec,
)


def off_diag_stats(mat: np.ndarray) -> dict[str, float]:
    v = mat[~np.eye(len(mat), dtype=bool)]
    return {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=600)  # 与 sanity_task_embed 同口径
    ap.add_argument("--n-source", type=int, default=16)
    ap.add_argument("--n-holdout", type=int, default=8)
    ap.add_argument("--sample-num", type=int, default=100)
    # 多样性探针用：更小的类别子集（如 2 3）→ 子集间重叠更少、分布更"纯"
    ap.add_argument("--n-classes-range", type=int, nargs=2, default=(3, 5))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "HAR" / "har.npz")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "task_slicer_sanity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, _, _ = load_har(args.data)  # (7352, 128, 9)；y 只用于切片定义，不进编码
    sp = make_leaveout_split(
        y, n_source_tasks=args.n_source, n_holdout_tasks=args.n_holdout,
        holdout_ratio=0.3, seed=args.seed,
        n_classes_range=tuple(args.n_classes_range),
    )
    print(f"HAR train: X={X.shape} → 源池 {len(sp.source_indices)} / "
          f"留出池 {len(sp.holdout_indices)}；源任务 {len(sp.source_tasks)} / "
          f"留出任务 {len(sp.holdout_tasks)}")
    for t in sp.source_tasks:
        print(f"  {t.task_id}: classes={t.classes} channels={t.channels} "
              f"n={len(t.sample_indices)}")

    # ---- 泄漏自检（铁律 6）----
    assert len(np.intersect1d(sp.source_indices, sp.holdout_indices)) == 0, "两池索引重叠"
    assert_tasks_disjoint(sp.source_tasks, sp.holdout_tasks)
    src_set = set(sp.source_indices.tolist())
    hold_set = set(sp.holdout_indices.tolist())
    assert all(set(t.sample_indices) <= src_set for t in sp.source_tasks), "源任务样本越池"
    assert all(set(t.sample_indices) <= hold_set for t in sp.holdout_tasks), "留出任务样本越池"

    # TS2Vec 训练数据 = 且仅 = 源池（fit_shared_ts2vec 的 source_data 显式传参）
    ts2vec_train = X[sp.source_indices]
    assert ts2vec_train.shape[0] == len(sp.source_indices)
    print("泄漏自检：两池零交集 / 任务样本各归其池 / TS2Vec 训练数据==源池  全部通过")

    t0 = time.time()
    shared = fit_shared_ts2vec(
        ts2vec_train, n_iters=args.iters, device=args.device, seed=args.seed,
    )
    print(f"shared univar TS2Vec: fitted on (N*C={ts2vec_train.shape[0] * ts2vec_train.shape[2]}, "
          f"T={ts2vec_train.shape[1]}, 1), {args.iters} iters, frozen "
          f"({time.time() - t0:.0f}s)")

    # 共享 SetTF 权重：跨任务 E 可比的前提（M3 起与 T-CLSC 联合训练）
    torch.manual_seed(args.seed)
    set_enc = TaskSetEncoder(dim_input=shared.output_dims, nz=256)

    names = [t.task_id for t in sp.source_tasks]
    embeds, baselines = {}, {}
    for t in sp.source_tasks:
        Xt, _ = materialize_task(t, X, y)  # 标签不进编码，E 是数据分布特征
        t0 = time.time()
        embeds[t.task_id] = embed_task(
            Xt, shared, sample_num=args.sample_num, set_encoder=set_enc, seed=args.seed,
        )
        baselines[t.task_id] = encode_task_windows(
            Xt, shared, sample_num=args.sample_num, seed=args.seed,
            encoding_window="full_series",
        ).mean(axis=0)
        print(f"  {t.task_id}: X={Xt.shape}  ({time.time() - t0:.1f}s)")

    def cosine_matrix(vecs: dict[str, np.ndarray]) -> np.ndarray:
        V = np.stack([vecs[n] for n in names])
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        return V @ V.T

    mat_e, mat_b = cosine_matrix(embeds), cosine_matrix(baselines)
    for title, mat in [("E 级（SetTF 未训练，raw cos 偏高属已知）", mat_e),
                       ("TS2Vec 级基线（关键信号）", mat_b)]:
        print(f"\n[{title}] 两两余弦矩阵：")
        print(" " * 10 + "".join(f"{n[-2:]:>7s}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:>10s}" + "".join(f"{mat[i, j]:7.3f}" for j in range(len(names))))
        s = off_diag_stats(mat)
        print(f"off-diagonal: min={s['min']:.4f} mean={s['mean']:.4f} max={s['max']:.4f}")

    # 类别重叠度 vs 相似度：验证"类别子集是多样性主力维度"在切片上兑现
    by_overlap: dict[str, list[float]] = {}
    for (i, a), (j, b) in combinations(enumerate(sp.source_tasks), 2):
        ja = len(set(a.classes) & set(b.classes)) / len(set(a.classes) | set(b.classes))
        bin_ = "jaccard<0.34" if ja < 0.34 else ("jaccard<0.67" if ja < 0.67 else "jaccard>=0.67")
        by_overlap.setdefault(bin_, []).append(float(mat_b[i, j]))
    print("\n类别重叠度 vs TS2Vec 级相似度（低重叠应更不相似）：")
    for k in sorted(by_overlap):
        v = by_overlap[k]
        print(f"  {k}: n={len(v)} mean={np.mean(v):.4f} min={np.min(v):.4f} max={np.max(v):.4f}")

    stats_b = off_diag_stats(mat_b)
    verdict = "达标" if stats_b["mean"] < 0.95 else "多样性不足（mean>=0.95，需回报讨论）"
    print(f"\n判据（TS2Vec 级 off-diagonal mean < 0.95）：{stats_b['mean']:.4f} → {verdict}")

    out = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pools": {"n_source": len(sp.source_indices), "n_holdout": len(sp.holdout_indices)},
        "leak_checks": "passed",
        "source_tasks": [asdict(t) for t in sp.source_tasks],
        "holdout_tasks": [asdict(t) for t in sp.holdout_tasks],
        "names": names,
        "cosine_matrix_e": mat_e.tolist(),
        "cosine_matrix_ts2vec": mat_b.tolist(),
        "off_diag_e": off_diag_stats(mat_e),
        "off_diag_ts2vec": stats_b,
        "by_class_overlap_ts2vec": {k: v for k, v in by_overlap.items()},
        "verdict": verdict,
    }
    path = args.outdir / "sanity_results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"结果已存 {path}")


if __name__ == "__main__":
    main()
