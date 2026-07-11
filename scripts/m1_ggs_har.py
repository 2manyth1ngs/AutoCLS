"""M1 验收脚本：GGS（AutoCL Table 5）在 HAR 上完整预训练 + SVM-RBF 下游评估。

GGS 原文：resize=0.2, rescale=0.3, jitter=0.0, point_mask=0.2, freq_mask=0.0,
crop=0.2, order=Order-3, emb_jitter=0.7, emb_mask=0.1, norm=None,
instance=True, temporal=False, cross_scale=False, kernel=5, pool=Avg,
adj_neighbor=False, loss=InfoNCE, sim=Distance(负欧氏), temperature=1.0。

本空间合法化调整：
- order：Table 5 为 1-based 的 Order-3 → 本空间 order=2（0-based 索引）。
- temporal/cross_scale 均关 → 按 space.yaml 约束 kernel/pool/adj_neighbor
  置无效档 None（原文 kernel=5/Avg 在两开关全关时无实际作用路径）。

历史基准：
- 2026-07-10 首次 M1 验收（当时 trainer 按 epoch 计量）：40 epoch、batch_size=128
  （HAR N=7352 → 58 steps/epoch，即 2320 iters）→ ACC 0.9372 / F1 0.9378，
  首/末 pass 平均损失 5.5671 / 2.9563。
  epoch→iters 迁移的逐位回归口径：--iters 2320 --batch-size 128 应精确复现上述值
  （max_iters 为整遍步数的整数倍时与旧 epoch 训练逐位等价）。
- 标准口径（configs/train.yaml）：--iters 4600 --batch-size 64（即默认值）。

用法（在仓库根目录）：
    python scripts/m1_ggs_har.py                                # 标准口径 4600 iters, bs=64
    python scripts/m1_ggs_har.py --iters 2320 --batch-size 128  # 迁移逐位校验
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.datasets import load_har  # noqa: E402
from autocls.downstream.linear_eval import evaluate_classification  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402

GGS = CLS(
    resize=0.2, rescale=0.3, jitter=0.0, point_mask=0.2, freq_mask=0.0,
    crop=0.2, order=2,
    emb_jitter=0.7, emb_mask=0.1, norm="none",
    instance=True, temporal=False, cross_scale=False,
    kernel=None, pool=None, adj_neighbor=None,
    loss_type="info_nce", sim_func="euclidean", temperature=1.0,
)


def main() -> None:
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=train_cfg["full_iters"],
                    help="预训练参数更新次数（默认 configs/train.yaml 的 full_iters）")
    ap.add_argument("--batch-size", type=int, default=train_cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=train_cfg["lr"])
    ap.add_argument("--seed", type=int, default=0, help="权重初始化与数据顺序种子")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "HAR" / "har.npz")
    args = ap.parse_args()

    space = CLSSpace.from_yaml()
    assert space.validate(GGS), "GGS 不在合法空间内！"
    print("GGS 合法性校验通过", flush=True)

    X_train, y_train, X_test, y_test = load_har(args.data)
    print("HAR:", X_train.shape, X_test.shape, "cuda:", torch.cuda.is_available(), flush=True)

    torch.manual_seed(args.seed)
    encoder = build_encoder(input_dims=X_train.shape[-1])  # 走 registry + encoder.yaml
    print("encoder:", type(encoder).__name__, flush=True)

    t0 = time.time()
    losses = pretrain_encoder(
        encoder, GGS, X_train,
        max_iters=args.iters, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, verbose=True,
    )
    t1 = time.time()
    res = evaluate_classification(encoder, X_train, y_train, X_test, y_test)
    t2 = time.time()

    summary = {
        "cls": "GGS (AutoCL Table 5, 本空间合法化)",
        "max_iters": args.iters,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pretrain_seconds": round(t1 - t0, 1),
        "eval_seconds": round(t2 - t1, 1),
        "first_pass_loss": round(losses[0], 4),
        "last_pass_loss": round(losses[-1], 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
