"""task_embed 区分力诊断（配合 scripts/sanity_task_embed.py 的 check B 失败分析）。

sanity check B 发现：每任务独立重训 TS2Vec 时相似度矩阵≈全 1，且中心化后
"同数据不同种子"的 E 距离不小于"不同任务"间距离——E 被 TS2Vec 训练随机性
（对比表征的基任意性：两次独立训练学到的表征空间相差任意旋转）主导。

本脚本对比两种共享冻结 TS2Vec 方案（所有任务用同一份编码器权重，消除基噪声）：
- multivar：9 通道 TS2Vec 在完整 HAR 上训练一次；通道子任务零填充缺失通道
  （z-score 后 0=均值，与 TSEncoder 掩码语义一致）。
- univar：单变量 TS2Vec（input_dims=1，通道当实例）+ 表征跨通道取均值——
  AutoCTS++ 原味（← exps/generate_task_feature.py L137-140 对 N 个传感器
  的表征取 mean），天然支持任意通道数，通道子任务无需零填充。

各报告两级相似度：
- E 级：SetTF(TS2Vec(windows))。SetTF 未训练（M3 起与 T-CLSC 联合训练），
  随机权重会压缩差异，raw cos 偏高属预期；
- TS2Vec 级基线：full_series 池化后跨窗口取均值，不过 SetTF——检验"SetTF 的
  输入是否携带任务分布信号"（联合训练能否学出区分力的必要条件）。

用法（在仓库根目录）：
    python scripts/diag_task_embed_shared.py            # 两种模式都跑
    python scripts/diag_task_embed_shared.py --mode univar
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.data.datasets import load_har  # noqa: E402
from autocls.task_embed import TaskSetEncoder  # noqa: E402
from autocls.task_embed.ts2vec_wrapper import TS2VecWrapper  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def matrix_report(vecs: dict[str, np.ndarray], title: str, lines: list[str]) -> None:
    names = list(vecs)
    V = np.stack([vecs[n] for n in names])
    mean = V.mean(0)
    lines.append(f"\n== {title} ==")
    for tag, M in [("raw cos", V), ("centered cos", V - mean)]:
        lines.append(f"-- {tag} --")
        lines.append(" " * 8 + "".join(f"{n:>9s}" for n in names))
        for i, a in enumerate(names):
            lines.append(f"{a:>8s}" + "".join(f"{cosine(M[i], M[j]):9.4f}" for j in range(len(names))))


def zero_fill(Xs: np.ndarray, keep: list[int]) -> np.ndarray:
    out = np.zeros_like(Xs)
    out[:, :, keep] = Xs[:, :, keep]
    return out


def run_mode(mode: str, X: np.ndarray, y: np.ndarray, iters: int, device: str,
             lines: list[str]) -> None:
    t0 = time.time()
    if mode == "univar":
        # 通道展开成单变量实例池 (N*C, T, 1)
        pool = X.transpose(0, 2, 1).reshape(-1, X.shape[1], 1)
        w = TS2VecWrapper(input_dims=1, device=device, seed=0)
    else:
        pool = X
        w = TS2VecWrapper(input_dims=X.shape[-1], device=device, seed=0)
    w.fit(pool, n_iters=iters)
    lines.append(f"\n[{mode}] shared TS2Vec fitted on {pool.shape}, "
                 f"{iters} iters, {time.time() - t0:.0f}s")

    torch.manual_seed(0)
    set_enc = TaskSetEncoder(dim_input=256, nz=256).to(device).eval()

    def task_vecs(Xt: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(Xt.shape[0], size=min(100, Xt.shape[0]), replace=False))
        s = Xt[idx]
        if mode == "univar":
            m, T, C = s.shape
            uni = s.transpose(0, 2, 1).reshape(m * C, T, 1)
            reprs = w.encode(uni).reshape(m, C, T, -1).mean(axis=1)  # 跨通道均值
            base = w.encode(uni, encoding_window="full_series").reshape(m, C, -1).mean(axis=1)
        else:
            reprs = w.encode(s)
            base = w.encode(s, encoding_window="full_series")
        with torch.no_grad():
            e = set_enc(torch.from_numpy(reprs).float().to(device)).cpu().numpy()
        return e, base.mean(0)

    m012, m345 = np.isin(y, [0, 1, 2]), np.isin(y, [3, 4, 5])
    if mode == "univar":
        chA, chB = X[:, :, 0:4], X[:, :, 5:9]  # 真通道子集，无需零填充
    else:
        chA, chB = zero_fill(X, [0, 1, 2, 3]), zero_fill(X, [5, 6, 7, 8])
    tasks = {
        "full_a": (X, 0), "full_b": (X, 1),  # 同任务不同窗口采样
        "cls012": (X[m012], 0), "cls345": (X[m345], 0),
        "chA": (chA, 0), "chB": (chB, 0),
    }
    E, B = {}, {}
    for name, (Xt, seed) in tasks.items():
        E[name], B[name] = task_vecs(Xt, seed)

    matrix_report(E, f"{mode}: E = SetTF(TS2Vec)，SetTF 未训练", lines)
    matrix_report(B, f"{mode}: TS2Vec 级基线（full_series 均值，无 SetTF）", lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["multivar", "univar", "both"], default="both")
    ap.add_argument("--iters", type=int, default=600)  # AutoCTS++ 大数据默认 n_iters
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "HAR" / "har.npz")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "task_embed_sanity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X, y, _, _ = load_har(args.data)  # y 只用于切类别子任务
    lines = [f"HAR train X={X.shape}, device={args.device}, iters={args.iters}"]
    modes = ["multivar", "univar"] if args.mode == "both" else [args.mode]
    for m in modes:
        run_mode(m, X, y, args.iters, args.device, lines)

    report = "\n".join(lines)
    print(report)
    out = args.outdir / "diag_shared_ts2vec.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已存 {out}")


if __name__ == "__main__":
    main()
