"""M2 探针（临时脚本）：early-validation 排名相关性验证（TECHNICAL_DESIGN.md M2 闸口）。

回答的问题：k iters 对比预训练的下游 acc 排名，能否代表完整训练（full_iters，与 M1
验收一致）的排名？这是 early-validation 造 label 能否成立的隐含前提。
【口径注记】M2 结论（k:full=1:8 时 ρ=0.85）在 575/4600@bs=64 口径下取得；2026-07
口径更正后标准为 290/2320@bs=128（configs/train.yaml），1:8 比例不变。

训练量计量单位为 iters（参数更新次数，见 configs/train.yaml）：跨任务时 epoch
截断会让 step 数随任务大小漂移，故统一按 iters。早期缓存（epoch 计量，5/40）由
load_cache 按 1 epoch = 115 steps（HAR N=7352, bs=64）自动换算，逐位等价。

协议：
- HAR 上 sample_random() 采样 N 个合法 CLS（固定 seed，可复现）；
- 每个 CLS 跑两次**彼此独立、均从头开始**的预训练（铁律 2：early-val 与最终评估
  禁止混用，故不做 checkpoint 续训）：
    early: max_iters=k    → SVM 下游 acc_early
    full : max_iters=full → SVM 下游 acc_full
  两次共用同一权重初始化种子与数据顺序种子，early 的 k 个 iters 与 full 的前 k 个
  iters 逐位相同——隔离出"训练量"这一个变量，排除种子噪声混淆。
- 每条 (cls, iters) → 结果逐行追加到 results/m2_probe/cache.jsonl（铁律 3），
  --resume 跳过已完成条目；full 结果在后续换 k 重跑时自动复用。
- 输出：Spearman ρ、Kendall τ、top-5 重合度、散点图 png。

用法（在仓库根目录）：
    python scripts/m2_probe_correlation.py                  # k=configs/train.yaml early_iters
    python scripts/m2_probe_correlation.py --resume         # 断点续跑
    python scripts/m2_probe_correlation.py --resume --k 1150  # 追加更大 k（full 缓存复用）
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.datasets import load_har  # noqa: E402
from autocls.downstream.linear_eval import evaluate_classification  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402


def cls_hash(cls_dict: dict) -> str:
    return hashlib.md5(json.dumps(cls_dict, sort_keys=True).encode()).hexdigest()[:12]


def sample_cls_list(space: CLSSpace, n: int, seed: int) -> list:
    """固定 seed 采样 n 个互不相同的合法 CLS（空间 ~3e12，重复概率可忽略，仍去重保险）。"""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        c = space.sample_random(rng)
        h = cls_hash(dataclasses.asdict(c))
        if h in seen:
            continue
        assert space.validate(c), f"sample_random 产出非法 CLS: {c}"
        seen.add(h)
        out.append(c)
    return out


# 迁移前旧缓存按 epoch 计量；仅对该缓存的口径（HAR N=7352, bs=64）成立的换算。
# 5 epoch=575 iters、40 epoch=4600 iters 与新 trainer 同种子逐位等价，可直接复用。
_LEGACY_EPOCH_STEPS = 115


def load_cache(path: Path) -> dict[tuple[str, int], dict]:
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if "iters" not in rec:  # epoch→iters 迁移前的旧记录
                        rec["iters"] = rec["epochs"] * _LEGACY_EPOCH_STEPS
                    done[(rec["cls_hash"], rec["iters"])] = rec
    return done


def run_one(cls_cfg, iters: int, data, run_seed: int, device: str, args) -> dict:
    """从头预训练 iters 次参数更新 + SVM 下游评估。失败（NaN/异常）记录 acc=nan 不中断实验。"""
    X_train, y_train, X_test, y_test = data
    torch.manual_seed(run_seed)  # 控制权重初始化：同一 CLS 的 early/full 同一初始化
    encoder = build_encoder(input_dims=X_train.shape[-1])

    t0 = time.perf_counter()
    rec: dict = {"seconds_pretrain": None, "seconds_eval": None, "error": None}
    try:
        losses = pretrain_encoder(
            encoder, cls_cfg, X_train,
            max_iters=iters, batch_size=args.batch_size, lr=args.lr,
            device=device, seed=run_seed,
        )
        rec["seconds_pretrain"] = round(time.perf_counter() - t0, 1)
        rec["loss_final"] = losses[-1] if losses else None
        t1 = time.perf_counter()
        metrics = evaluate_classification(
            encoder, X_train, y_train, X_test, y_test, device=device, seed=args.seed
        )
        rec["seconds_eval"] = round(time.perf_counter() - t1, 1)
        rec.update(metrics)
    except Exception as e:  # 训练发散（NaN 特征等）视作该 CLS 的坏结果，不中断整体实验
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["acc"], rec["f1"] = float("nan"), float("nan")
    return rec


def analyze(cache: dict, hashes: list[str], k: int, full: int, outdir: Path) -> None:
    """对齐 (acc_early@k, acc_full) 并输出相关性指标 + 散点图 + 判读。"""
    pairs = []  # (idx, hash, acc_early, acc_full)
    for i, h in enumerate(hashes):
        e, f = cache.get((h, k)), cache.get((h, full))
        if e is None or f is None:
            continue
        pairs.append((i, h, e["acc"], f["acc"]))
    valid = [p for p in pairs if np.isfinite(p[2]) and np.isfinite(p[3])]
    n_bad = len(pairs) - len(valid)

    print(f"\n{'=' * 72}\n  k={k} vs full={full}   配对完成 {len(pairs)}/{len(hashes)}"
          f"（其中训练失败剔除 {n_bad}）\n{'=' * 72}")
    print(f"  {'idx':>3}  {'hash':<12}  {'acc_early':>9}  {'acc_full':>8}")
    for i, h, ae, af in pairs:
        print(f"  {i:>3}  {h:<12}  {ae:>9.4f}  {af:>8.4f}")
    if len(valid) < 3:
        print("  有效配对不足 3 个，无法计算相关性")
        return

    from scipy.stats import kendalltau, spearmanr
    idxs = [p[0] for p in valid]
    early = np.array([p[2] for p in valid])
    fullv = np.array([p[3] for p in valid])
    rho, rho_p = spearmanr(early, fullv)
    tau, tau_p = kendalltau(early, fullv)
    top = min(5, len(valid))
    top_e = {idxs[j] for j in np.argsort(-early)[:top]}  # 以 CLS 编号表示，平局按位次截断
    top_f = {idxs[j] for j in np.argsort(-fullv)[:top]}
    overlap = len(top_e & top_f)

    print(f"\n  Spearman ρ  = {rho:.4f}  (p={rho_p:.2e})")
    print(f"  Kendall τ   = {tau:.4f}  (p={tau_p:.2e})")
    print(f"  top-{top} 重合 = {overlap}/{top}  (early {sorted(top_e)} vs full {sorted(top_f)})")

    if rho >= 0.7:
        verdict = f"ρ≥0.7：k={k} 足够，可直接进 M2 第二步（task_slicer + generate_labels）"
    elif rho >= 0.5:
        verdict = f"0.5≤ρ<0.7：k={k} 偏弱，按预案追加更大 k（2×/3×）重测"
    else:
        verdict = f"ρ<0.5：k={k} 不可行，需讨论调整评估协议"
    print(f"  判读：{verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(early, fullv, s=40, alpha=0.8)
    for i, h, ae, af in valid:
        ax.annotate(str(i), (ae, af), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(early.min(), fullv.min()) - 0.02
    hi = max(early.max(), fullv.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"acc_early (k={k} iters)")
    ax.set_ylabel(f"acc_full ({full} iters)")
    ax.set_title(f"M2 probe on HAR (N={len(valid)}): "
                 f"Spearman={rho:.3f}, Kendall={tau:.3f}, top{top}={overlap}/{top}")
    fig.tight_layout()
    png = outdir / f"scatter_k{k}_vs_full{full}.png"
    fig.savefig(png, dpi=150)
    print(f"  散点图：{png}")


def main() -> None:
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=20, help="采样 CLS 数")
    ap.add_argument("--seed", type=int, default=42, help="采样与评估随机种子")
    ap.add_argument("--k", type=int, nargs="+", default=[train_cfg["early_iters"]],
                    help="early-validation 参数更新次数 iters（可多个）")
    ap.add_argument("--full-iters", type=int, default=train_cfg["full_iters"],
                    help="完整预训练参数更新次数（与 M1 验收一致）")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "HAR" / "har.npz")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "m2_probe")
    ap.add_argument("--batch-size", type=int, default=train_cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=train_cfg["lr"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="跳过 cache.jsonl 中已完成条目")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_path = args.outdir / "cache.jsonl"
    if cache_path.exists() and not args.resume:
        sys.exit(f"{cache_path} 已存在：加 --resume 续跑，或删除后重跑")

    space = CLSSpace.from_yaml()
    cls_list = sample_cls_list(space, args.n, args.seed)
    hashes = [cls_hash(dataclasses.asdict(c)) for c in cls_list]
    with open(args.outdir / "cls_list.json", "w", encoding="utf-8") as fh:
        json.dump(
            [{"idx": i, "cls_hash": h, "cls": dataclasses.asdict(c)}
             for i, (h, c) in enumerate(zip(hashes, cls_list))],
            fh, ensure_ascii=False, indent=2,
        )

    data = load_har(args.data)
    cache = load_cache(cache_path)
    todo = [(i, it) for i in range(args.n)
            for it in (*args.k, args.full_iters)
            if (hashes[i], it) not in cache]
    print(f"设备 {args.device} | N={args.n} seed={args.seed} | k={args.k} full={args.full_iters}"
          f" | 已缓存 {len(cache)} 条，待跑 {len(todo)} 条", flush=True)

    for n_done, (i, it) in enumerate(todo):
        rec = run_one(cls_list[i], it, data, run_seed=args.seed * 1000 + i,
                      device=args.device, args=args)
        rec = {"cls_idx": i, "cls_hash": hashes[i], "iters": it, "seed": args.seed,
               **rec, "cls": dataclasses.asdict(cls_list[i])}
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cache[(hashes[i], it)] = rec
        status = f"acc={rec['acc']:.4f}" if rec.get("error") is None else f"失败: {rec['error']}"
        print(f"[{n_done + 1}/{len(todo)}] cls#{i} iters={it:<5d} {status} "
              f"(pretrain {rec.get('seconds_pretrain')}s, eval {rec.get('seconds_eval')}s)",
              flush=True)

    for k in args.k:
        analyze(cache, hashes, k, args.full_iters, args.outdir)


if __name__ == "__main__":
    main()
