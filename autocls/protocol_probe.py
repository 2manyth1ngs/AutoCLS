"""下游评估协议探针的纯逻辑（协议锁 `results/protocol_probe/PROTOCOL.md`）。

**为什么要这个量**：基线探针（results/baseline_probe/REPORT.md）证明贴顶是源池
主流（16 个里 5 个基线 >0.90），且 `n_classes_range` 加难口径这个旋钮已拧到头
（全类别档只降 0.011–0.040）。因果链最后落在一句话上：

    这批 UEA 对「随机卷积特征 + 时间维 max-pool + RBF SVM」这个**下游评估协议**
    本身就太容易（AWR 25 类、随机猜测 0.040，零训练表征拿到 0.990）。

本探针测剩下那个旋钮：其余口径逐项不变，只换下游评估协议，能否同时做到
①空间大 ②空间真被用上 ③不破坏唯一无争议可用的源（Handwriting）。三者缺一不可
——MotorImagery 是反例：基线 0.511、名义 headroom 0.489，290 iters 训完还是 0.514。

本模块只放**判据与特征通路**；协议实现在 downstream/protocols.py（分类专用）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .baseline_probe import probe_scope
from .data.source_pool import SourcePoolRegistry

__all__ = [
    "G1_BASELINE_MAX", "G2_MIN_ABOVE", "G3_MIN_ABOVE", "G3_MIN_BEST_GAIN",
    "G4_BEST_GAIN_FLOOR", "GATE1_ITERS", "GATE2_ITERS", "GATE2A_DATASETS",
    "GATE2B_DATASETS", "GATE2_N_TASKS", "GATE2_N_CLS", "N_BASELINE_SEEDS",
    "compute_task_features", "gate1_scope", "gate1_verdict", "gate2_verdict",
    "gate3_verdict", "overall_verdict",
]

# 训练量（铁律 2：max_iters 必须显式传参，early/full 禁止混用）
GATE1_ITERS = 0        # 闸门1：不训练，测天花板
GATE2_ITERS = 290      # 闸门2：configs/train.yaml 的 early 口径 = 阶段1 canonical
N_BASELINE_SEEDS = 5   # 与基线探针同：5 个随机初始化，用于零假设 ±2σ 带

# 闸门2 范围（PROTOCOL §4.1）。MotorImagery 不参与判定却占本闸 55% 算力，
# 推迟到 2a 通过后再补；判定表只需 2a。
GATE2A_DATASETS = ("ArticularyWordRecognition", "NATOPS", "Handwriting")
GATE2B_DATASETS = ("MotorImagery",)
GATE2_N_TASKS, GATE2_N_CLS = 2, 10   # 与阶段1 rprime_cache 逐字段同口径

# ---------------------------------------------------------------------------
# 判据阈值（PROTOCOL.md §4.3 写死；来源全部独立于本次结果）
# ---------------------------------------------------------------------------

# G1：AWR 新基线 < 0.70。0.70 逐字沿用 baseline_probe/PROTOCOL.md §3.1 的"离顶"
# 阈值（headroom > 6δ，典型 δ=0.05），非为本实验新造。
G1_BASELINE_MAX = 0.70

# G2：AWR 与 NATOPS 均需 ↑≥6/20 且 ↑≥↓。6/20 = 旧协议下 Handwriting（唯一无争议
# 可用源）15/20 的 40%——"至少要到无争议样本的四成"；"↑≥↓"排除"只是让训练更
# 容易破坏表征"（旧协议 NATOPS 就是 0↑/16↓）。
G2_MIN_ABOVE = 6

# G3：对照不被破坏。12/20 = 旧协议 15/20 的 80%（换协议必然抬噪声，允许适度下降，
# 但不允许失去"无争议"地位）；0.10 = 2× 典型 δ（旧协议实测 best_gain +0.397）。
G3_MIN_ABOVE = 12
G3_MIN_BEST_GAIN = 0.10

# G4：幅度可用，沿用 baseline_probe 的 usable 门槛（= 典型 δ；低于 δ 的增益产不出
# 存活 label）。
G4_BEST_GAIN_FLOOR = 0.05

# §4.4 判定表的四个取值
VERDICT_WORKS = "protocol_works"      # 推广到 16 个 + 重标定 δ
VERDICT_FALSE_FIX = "false_fix"       # 天花板降了但 headroom 用不上（MotorImagery 型）
VERDICT_NO_EFFECT = "no_effect"       # 对任何合理下游协议都太容易
VERDICT_BROKE = "broke_control"       # 造出假空间却毁了唯一可用的源


def gate1_scope(registry: SourcePoolRegistry) -> list[str]:
    """闸门1 范围 = 源池 ∪ reserve（16 个）。目标池绝不入内（铁律 6）。"""
    return probe_scope(registry)


def gate1_verdict(awr_baseline: float) -> dict:
    """G1 天花板：AWR 新基线 **严格小于** 0.70。

    用 AWR 单点判定是故意的：它是最极端的贴顶反例（旧协议 0.990、25 类、随机猜测
    0.040），取它作必要条件比取"4 个的均值"更难被侥幸通过。
    """
    return {"gate": "G1", "awr_baseline": float(awr_baseline),
            "threshold": G1_BASELINE_MAX,
            "passed": bool(awr_baseline < G1_BASELINE_MAX)}


def gate2_verdict(per_dataset: dict[str, tuple[int, int]]) -> dict:
    """G2 空间真被用上：每个数据集都要 ↑≥6/20 **且** ↑≥↓。

    per_dataset: {数据集: (超出 ±2σ 带的条数, 跌破的条数)}。任一数据集不过即不过
    ——"只有 AWR 过而 NATOPS 不过"说明效应挑数据集，不足以支撑推广到 16 个。
    """
    detail, failed = {}, []
    for name, (above, below) in sorted(per_dataset.items()):
        ok = above >= G2_MIN_ABOVE and above >= below
        detail[name] = {"n_above": int(above), "n_below": int(below), "passed": ok}
        if not ok:
            failed.append(name)
    return {"gate": "G2", "min_above": G2_MIN_ABOVE, "detail": detail,
            "failed": failed, "passed": not failed and bool(per_dataset)}


def gate3_verdict(above: int, best_gain: Optional[float]) -> dict:
    """G3 对照不被破坏：Handwriting ↑≥12/20 且 best_gain≥0.10。

    best_gain 缺失（无配对 trained R'）→ 判不过，不臆断。
    """
    ok = (best_gain is not None and above >= G3_MIN_ABOVE
          and best_gain >= G3_MIN_BEST_GAIN)
    return {"gate": "G3", "n_above": int(above),
            "best_gain": None if best_gain is None else float(best_gain),
            "min_above": G3_MIN_ABOVE, "min_best_gain": G3_MIN_BEST_GAIN,
            "passed": bool(ok)}


def overall_verdict(g1: bool, g2: bool, g3: bool) -> str:
    """PROTOCOL.md §4.4 判定表（先于数据写死）。

    | G1 | G2 | G3 | 结论 |
    |----|----|----|------|
    | ✅ | ✅ | ✅ | protocol_works —— 推广到 16 个，重标定 per-dataset δ |
    | ✅ | ❌ | —  | false_fix —— 天花板降了但 headroom 用不上 |
    | ❌ | —  | —  | no_effect —— 这批 UEA 对任何合理下游协议都太容易 |
    | ✅ | ✅ | ❌ | broke_control —— 造出假空间却毁了唯一可用的源 |
    """
    if not g1:
        return VERDICT_NO_EFFECT
    if not g2:
        return VERDICT_FALSE_FIX
    return VERDICT_WORKS if g3 else VERDICT_BROKE


# ---------------------------------------------------------------------------
# 特征通路：evaluate_rprime 的前半段（到 max-pool 特征为止），落盘供多协议复用
# ---------------------------------------------------------------------------


def compute_task_features(
    task,
    cls_cfg,
    X: np.ndarray,
    y: np.ndarray,
    *,
    weight_seed: int,
    split_seed: int,
    iters: int,
    eval_ratio: float = 0.3,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> dict:
    """(task, cls, iters, 种子) → max-pool 后的 train/eval 特征（+ 标签、耗时）。

    **逐行镜像 `labeling.evaluate_rprime` 的前半段**（同 materialize_task、同
    split_task_train_eval、同 torch.manual_seed→build_encoder→pretrain_encoder、
    同 encode_features），只是不接分类器 —— 分类器由 downstream/protocols.py 在
    落盘的特征上离线跑。`tests/test_protocol_probe.py::
    test_compute_task_features_reproduces_evaluate_rprime` 逐位断言两条通路同值，
    防止静默 fork。

    为什么要这条通路（PROTOCOL §6.1）：既有 260 条 canonical R' 只存了 {acc,f1,C}，
    表征与权重都没落盘，换协议无法直接重算；但训练按 weight_seed 可复现，故
    **重跑一次 + dump 特征，可一次喂饱 A/B/B′/C 全部变体**，以后再加协议零成本。

    **例外：显存/内存不足抛 RprimeOOM，绝不静默记结论**（与 evaluate_rprime 同规矩）。
    """
    import time

    import torch

    from .contrastive.cl_trainer import pretrain_encoder
    from .data.task_slicer import materialize_task
    from .downstream.linear_eval import encode_features
    from .encoders.registry import build_encoder
    from .labeling import split_task_train_eval
    from .oom_guard import RprimeOOM, is_oom_error

    Xt, yt = materialize_task(task, X, y)
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=eval_ratio, seed=split_seed,
    )
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)  # 权重初始化可复现
        encoder = build_encoder(input_dims=Xt.shape[-1])
        pretrain_encoder(
            encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
            batch_size=batch_size, lr=lr, device=device, seed=weight_seed,
        )
        train_feats = encode_features(encoder, Xt[tr_idx], device=device)
        eval_feats = encode_features(encoder, Xt[ev_idx], device=device)
    except Exception as e:  # noqa: BLE001 —— 与 evaluate_rprime 同语义
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        return {"error": f"{type(e).__name__}: {e}", "train_feats": None,
                "eval_feats": None, "y_train": None, "y_eval": None,
                "n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)),
                "seconds": round(time.perf_counter() - t0, 1),
                "weight_seed": int(weight_seed), "split_seed": int(split_seed)}
    return {
        "error": None,
        "train_feats": train_feats, "eval_feats": eval_feats,
        "y_train": yt[tr_idx], "y_eval": yt[ev_idx],
        "n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)),
        "feat_dim": int(train_feats.shape[1]),
        "seconds": round(time.perf_counter() - t0, 1),
        "weight_seed": int(weight_seed), "split_seed": int(split_seed),
    }
