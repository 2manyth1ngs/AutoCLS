"""预训练 T-CLSC 并做 M3 验收：留出任务配对准确率 + E 消融三档（TECHNICAL_DESIGN.md 模块 6）。

改自 AutoCTS++ exps/random_search.py mode='pretrain'（CLAUDE.md 复用映射：
random_search.py → 3_pretrain_tclsc.py）。域逻辑在 autocls/：
- 模型/引擎：comparator/tclsc_model.py + tclsc_engine.py（任务无关，铁律 7）
- 缓存桥接：labeling.comparator_records / comparator_eval_pairs（方向经
  is_better，铁律 8；引擎的动态配对经注入 better_fn 取得同一方向语义）

数据（scripts/2_generate_labels.py 产出）：
- 训练：源池 rprime_cache.jsonl（16 源任务，动态配对 + curriculum）
- 测试：留出池 rprime_cache_holdout.jsonl（8 留出任务，静态双向评估对）
- 任务特征：task_features/{tid}.npz 的 windows_repr (m,T,D)，SetTF 输入，
  与 T-CLSC 联合训练；SetTF 从 set_encoder_init.pt 初始化（与已存 E 同源）

验收（M3，机制验证）：
1. BCE 下降 + 留出任务配对准确率显著 > 0.5（随机基线）；
2. E 消融（不重训、只改测试输入）：基线 / shuffle E（任务错位）/ zero E 三个
   准确率——大幅下降 ⇒ 真 task-aware；几乎不变 ⇒ 退化为忽略 E 的通用 CLS
   排序器（很可能是 16 个同源任务多样性不足的必然结果，印证转路线 A）。

【定位（路线 B）】机制工程验证，非 zero-shot 证据：留出任务与源任务同分布
（同一 HAR 切片协议），只能验证"泛化到新任务切片"，不能验证"泛化到新数据
集"。真 zero-shot 结果由路线 A（多 UEA 数据集、源/目标池分离）产出。

用法（在仓库根目录）：
    python scripts/3_pretrain_tclsc.py                 # 训练 + 评估 + 消融
    python scripts/3_pretrain_tclsc.py --device cpu    # 无 GPU 时
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.labeling import (  # noqa: E402
    comparator_eval_pairs,
    comparator_records,
    load_rprime_cache,
)
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import TaskSetEncoder  # noqa: E402
from autocls.task_types import is_better  # noqa: E402


def load_task_windows(feat_dir: Path, task_ids: set[str]) -> dict[str, np.ndarray]:
    """逐任务加载 windows_repr (m,T,D)（2_generate_labels.py ensure_task_features 落盘）。"""
    out = {}
    for tid in sorted(task_ids):
        path = feat_dir / f"{tid}.npz"
        if not path.exists():
            sys.exit(f"缺任务特征 {path}：先跑 scripts/2_generate_labels.py（对应池）")
        out[tid] = np.load(path)["windows_repr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pretrain_tclsc.yaml")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "results" / "labels")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "tclsc")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    task_type = labels_cfg["task_type"]
    delta = labels_cfg["delta"]          # δ 唯一真源：labels.yaml（训练/评估同参）
    iters = train_cfg["early_iters"]     # 铁律 2：label 一律 early-val 口径
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- 数据：源池训练 / 留出池测试（铁律 6：任务零重叠）----
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    src_cache = load_rprime_cache(args.labels_dir / "rprime_cache.jsonl")
    hold_cache = load_rprime_cache(args.labels_dir / "rprime_cache_holdout.jsonl")
    if not src_cache or not hold_cache:
        sys.exit("缺 R' 缓存：先跑 scripts/2_generate_labels.py（source 与 holdout 两池）")

    train_records = comparator_records(src_cache, encoder, iters=iters)
    train_pairs = comparator_eval_pairs(
        src_cache, encoder, task_type=task_type, delta=delta, iters=iters)
    test_pairs = comparator_eval_pairs(
        hold_cache, encoder, task_type=task_type, delta=delta, iters=iters)

    train_tids = {r.task_id for r in train_records}
    test_tids = {p.task_id for p in test_pairs}
    assert not (train_tids & test_tids), "训练/测试任务重叠（铁律 6）"

    task_windows = load_task_windows(args.labels_dir / "task_features",
                                     train_tids | test_tids)
    dim_input = next(iter(task_windows.values())).shape[-1]

    # ---- 模型：SetTF 从 set_encoder_init.pt 初始化（与已存 E 同一初始化），
    #      与 T-CLSC 联合训练 ----
    torch.manual_seed(cfg["seed"])
    set_enc = TaskSetEncoder(dim_input=dim_input, nz=labels_cfg["nz"])
    init_path = args.labels_dir / "set_encoder_init.pt"
    if init_path.exists():
        set_enc.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
    model = TCLSC(cls_dim=encoder.dim, set_encoder=set_enc,
                  embedding_dim=cfg["embedding_dim"], fz=cfg["fz"])
    engine = TCLSCEngine(model, task_windows, device=args.device,
                         lr=cfg["lr"], batch_size=cfg["batch_size"], seed=cfg["seed"])

    def better_fn(r1: float, r2: float) -> bool:
        # 铁律 8：方向唯一合法途径 is_better；引擎保持任务无关（铁律 7）
        return is_better(r1, r2, task_type)

    n_shared = sum(r.origin == "shared" for r in train_records)
    print(f"训练记录 {len(train_records)} 条（shared {n_shared} / random "
          f"{len(train_records) - n_shared}），任务 {len(train_tids)} 个 | "
          f"静态评估对：训练 {len(train_pairs)} / 留出 {len(test_pairs)}（含双向）| "
          f"cls_dim={encoder.dim} nz={labels_cfg['nz']} device={args.device}", flush=True)

    # ---- 训练（动态配对 + curriculum，← random_search.py mode='pretrain'）----
    t0 = time.time()
    history = engine.train(train_records, epochs=cfg["epochs"],
                           curriculum_epochs=cfg["curriculum_epochs"],
                           better_fn=better_fn, delta=delta)
    train_seconds = round(time.time() - t0, 1)
    for h in history[:: max(1, cfg["epochs"] // 10)]:
        print(f"  epoch {h['epoch']:>3} [{h['phase']:^6}] pairs={h['n_pairs']:>3} "
              f"loss={h['loss']:.4f} acc={h['acc']:.3f}", flush=True)

    # ---- 验收 1：BCE 下降 + 留出配对准确率 ----
    res_train = engine.evaluate(train_pairs)
    res_test = engine.evaluate(test_pairs)
    # 显著性：双向样本同源不独立，二项检验用单向（无序对）子集
    fwd = test_pairs[: len(test_pairs) // 2]
    res_fwd = engine.evaluate(fwd)
    btest = binomtest(round(res_fwd["acc"] * len(fwd)), len(fwd), 0.5,
                      alternative="greater")

    # ---- 验收 2：E 消融（只改测试输入，不重训）----
    res_shuffle = engine.evaluate(test_pairs, e_mode="shuffle", seed=cfg["seed"])
    res_zero = engine.evaluate(test_pairs, e_mode="zero")

    print(f"\n=== 验收 1（机制能训起来；训练 {train_seconds}s）===")
    print(f"BCE: 首 epoch {history[0]['loss']:.4f} → 末 epoch {history[-1]['loss']:.4f}")
    print(f"配对准确率: 训练集 {res_train['acc']:.4f} (n={res_train['n']}) | "
          f"留出任务 {res_test['acc']:.4f} (n={res_test['n']})")
    print(f"显著性(单向 n={len(fwd)}, H0=0.5): p={btest.pvalue:.2e}")
    print("\n=== 验收 2（E 消融，task-aware 检验）===")
    print(f"基线      : {res_test['acc']:.4f}")
    print(f"shuffle E : {res_shuffle['acc']:.4f}  (Δ={res_test['acc'] - res_shuffle['acc']:+.4f})")
    print(f"zero E    : {res_zero['acc']:.4f}  (Δ={res_test['acc'] - res_zero['acc']:+.4f})")

    report = {
        "config": {**cfg, "delta": delta, "iters": iters, "task_type": task_type,
                   "device": args.device},
        "data": {
            "n_train_records": len(train_records),
            "n_train_records_shared": n_shared,
            "n_train_tasks": len(train_tids),
            "n_test_tasks": len(test_tids),
            "n_train_pairs_directed": len(train_pairs),
            "n_test_pairs_directed": len(test_pairs),
        },
        "acceptance_1": {
            "bce_first_epoch": history[0]["loss"],
            "bce_last_epoch": history[-1]["loss"],
            "train_pair_acc": res_train,
            "test_pair_acc": res_test,
            "binom_p_one_sided": btest.pvalue,
            "train_seconds": train_seconds,
        },
        "acceptance_2_e_ablation": {
            "baseline": res_test, "shuffle_E": res_shuffle, "zero_E": res_zero,
        },
        "scope_note": (
            "路线 B 机制工程验证，非 zero-shot 证据：留出任务与源任务同分布"
            "（同一 HAR 切片协议），只验证'泛化到新任务切片'；真 zero-shot"
            "（跨数据集）结果由路线 A（多 UEA 数据集、源/目标池分离）产出。"
            "E 消融若无下降 ⇒ 模型退化为忽略 E 的通用 CLS 排序器，"
            "印证 16 个同源任务多样性不足、转路线 A 的必要性。"
        ),
    }
    (args.outdir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.outdir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"model": model.state_dict(), "cls_dim": encoder.dim,
                "nz": labels_cfg["nz"], "dim_input": dim_input,
                "embedding_dim": cfg["embedding_dim"], "fz": cfg["fz"]},
               args.outdir / "tclsc.pt")
    print(f"\n报告/历史/权重 → {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
