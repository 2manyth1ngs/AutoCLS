"""显存预检：证明峰值显存与 `max_iters` 无关（full-iters 判别实验 §8）。

判别实验要把 iters 从 290 拉到 2320。若显存随 iters 涨，L=256 的
InsectWingbeatSound 会在长跑中途 OOM，白烧几小时。

机制上不该涨（`pretrain_encoder` 每步循环体相同，`pass_losses` 只 append float，
无计算图引用），但"不该"要用实测顶住——本项目已经栽过一次"按 runtime.yaml 峰值
模型低估 1.85×"（记忆条目"显存放大器是 resize"）。

做法：同一 (任务, CLS, 初始化) 跑不同 iters，比较 `torch.cuda.max_memory_allocated`。
先扫 10 条 CLS 在短 iters 下的峰值挑出最重的一条（resize 会把 T 翻倍，是已知放大器），
再对它跑到**真实目标 iters**。

判据不是"峰值逐位相同"。增强参数每 batch 重新采样（crop 长度、resize 倍率都是随机
量），故 peak 是"跑过的所有 step 里最大的那一步"——即随机抽样的**极大值统计量**，
步数越多越可能撞到极端 draw，会随 log(iters) 缓慢上抬而非随 iters 线性增长。
故判据取两条：
  A. **不随 iters 标度**：peak 增幅 ≪ iters 增幅（阈值：iters ×10 时增幅 <5%）
  B. **绝对装得下**：真实目标 iters 下的峰值 < 单卡 80%
只有 A 成立才谈得上"外推安全"，但本脚本直接测到目标 iters，B 是实测不是外推。

用法：
    python scripts/vram_precheck_full_iters.py --dataset InsectWingbeatSound \
        --registry configs/candidate_pool.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import load_probe_cls, run_seed_for, split_task_train_eval  # noqa: E402

PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"


def peak_mib(task, cls_cfg, X, y, *, iters: int, device: str) -> tuple[float, float]:
    """跑 iters 步预训练，返回 (峰值显存 MiB, 墙钟 s)。口径与 evaluate_rprime 的
    训练段逐字段相同（同 materialize/split/build_encoder/种子派生）。"""
    Xt, yt = materialize_task(task, X, y)
    tr_idx, _ = split_task_train_eval(yt, task.task_type, eval_ratio=0.3,
                                      seed=run_seed_for(task.task_id, "train_eval_split"))
    w_seed = run_seed_for(task.task_id, "vram_precheck")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    torch.manual_seed(w_seed)
    encoder = build_encoder(input_dims=Xt.shape[-1])
    pretrain_encoder(encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
                     batch_size=128, lr=1e-3, device=device, seed=w_seed)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    del encoder
    torch.cuda.empty_cache()
    return peak, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--scan-iters", type=int, default=60, help="扫 10 条 CLS 用的短 iters")
    ap.add_argument("--long-iters", type=int, default=600,
                    help="中间点，用于测'峰值是否随 iters 标度'（判据 A）")
    ap.add_argument("--target-iters", type=int, default=2320,
                    help="真实实验训练量，直接实测其峰值（判据 B，非外推）")
    ap.add_argument("--only-cls", default=None,
                    help="跳过扫描，直接用这条 cls_key（已知最重时省 2 min）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    registry = load_registry(Path(args.registry)) if args.registry else load_registry()
    entry = registry.get(args.dataset)
    X, y = load_source_dataset(args.dataset, registry)
    _, tasks = slice_source_tasks(entry, registry, y)
    task = tasks[0]
    cls_list, _ = load_probe_cls(PROBE_CATALOG, 10)

    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"{torch.cuda.get_device_name(0)} 总显存 {total:.0f} MiB | "
          f"{args.dataset} L={entry.l_enc} 任务 {task.task_id}", flush=True)

    scan = []
    if args.only_cls:
        cls_cfg, key = next((c, k) for c, k in cls_list if k.startswith(args.only_cls))
        print(f"\n[1] 跳过扫描，指定 CLS = {key[:8]}", flush=True)
        pk_short, _ = peak_mib(task, cls_cfg, X, y, iters=args.scan_iters, device=args.device)
        scan = [(pk_short, key, cls_cfg)]
    else:
        print(f"\n[1] 扫 10 条 CLS @ {args.scan_iters} iters，挑峰值最大的一条", flush=True)
        for i, (c, k) in enumerate(cls_list):
            pk, dt = peak_mib(task, c, X, y, iters=args.scan_iters, device=args.device)
            scan.append((pk, k, c))
            print(f"  [{i+1}/10] {k[:8]} resize={c.resize:<4} crop={c.crop:<4} "
                  f"peak={pk:8.1f} MiB ({dt:.1f}s)", flush=True)
        scan.sort(reverse=True, key=lambda r: r[0])
        pk_short, key, cls_cfg = scan[0]

    print(f"\n[2] 最重 CLS = {key[:8]}（@{args.scan_iters} iters {pk_short:.1f} MiB）；"
          f"测 {args.long_iters} 与 {args.target_iters} iters", flush=True)
    pk_long, dt_long = peak_mib(task, cls_cfg, X, y, iters=args.long_iters, device=args.device)
    pk_tgt, dt_tgt = peak_mib(task, cls_cfg, X, y, iters=args.target_iters, device=args.device)

    # 判据 A：峰值增幅 ≪ iters 增幅。峰值是随机增强 draw 的极大值统计量，会随步数
    # 缓慢上抬；真正要排除的是"随 iters 线性增长"（那才会在长跑中途 OOM）。
    grow = (pk_long - pk_short) / pk_short
    iters_grow = args.long_iters / args.scan_iters
    not_scaling = grow < 0.05
    # 判据 B：真实目标 iters 下实测装得下（非外推）
    frac = pk_tgt / total
    fits = frac < 0.80
    ok = not_scaling and fits

    print(f"\n  peak({args.scan_iters:>5} iters) = {pk_short:8.1f} MiB")
    print(f"  peak({args.long_iters:>5} iters) = {pk_long:8.1f} MiB ({dt_long:.1f}s)")
    print(f"  peak({args.target_iters:>5} iters) = {pk_tgt:8.1f} MiB ({dt_tgt:.1f}s)  ← 真实实验口径")
    print(f"\n  判据A 不随 iters 标度：iters ×{iters_grow:.0f} → 峰值 {grow*100:+.2f}% "
          f"{'✅' if not_scaling else '❌ 疑似线性增长'}")
    print(f"  判据B 装得下：{pk_tgt:.0f} / {total:.0f} MiB = {frac*100:.1f}% "
          f"{'✅' if fits else '❌'}")
    verdict = (f"PASS：full={args.target_iters} 实测峰值 {pk_tgt:.0f} MiB（占卡 {frac*100:.1f}%），"
               f"且峰值不随 iters 标度，不会 OOM" if ok else
               "FAIL：峰值随 iters 增长或装不下，full 跑前必须先解决")
    print(f"\n{verdict}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "dataset": args.dataset, "l_enc": entry.l_enc, "task_id": task.task_id,
            "gpu": torch.cuda.get_device_name(0), "total_mib": total,
            "scan_iters": args.scan_iters, "long_iters": args.long_iters,
            "target_iters": args.target_iters,
            "scan": [{"cls_key": k, "peak_mib": p, "resize": c.resize, "crop": c.crop}
                     for p, k, c in scan],
            "heaviest_cls": key, "peak_short_mib": pk_short, "peak_long_mib": pk_long,
            "peak_target_mib": pk_tgt, "growth_frac": grow, "target_frac_of_card": frac,
            "not_scaling_with_iters": not_scaling, "fits": fits, "verdict": verdict,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {args.out}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
