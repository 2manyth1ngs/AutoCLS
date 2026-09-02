"""噪声感知软 label（实验 2；results/soft_label_tclsc/PROTOCOL.md §1–§3；DESIGN_NOTES §1.3-1）。

硬 δ 的替代：不弃样，每个配对给 P(a ≻ b | Δ) = F_η(Δ)——F_η 为同 CLS 跨 weight-seed
重复差（k=3 噪声探针）的分布（Student-t MLE 或经验 CDF），Δ_true 取平先验。
训练用软 BCE（目标 p）或按信息权 w=|2p−1| 加权的硬 BCE；评估给加权准确率。

任务无关模块（铁律 7）：方向经注入的 margin_fn(r_a, r_b)（>0 ⇔ a 优；正式脚本绑定
task_types 的方向语义，铁律 8）；概率经注入的 prob_fn(task_id, key_a, key_b, delta)
（噪声模型 × 任务/制式尺度的组合知识留在调用方）。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t

from .tclsc_engine import ClsRecord, Pair, TCLSCEngine

MarginFn = Callable[[float, float], float]
ProbFn = Callable[[str, str, str, float], float]


# ---------------------------------------------------------------- 噪声模型


def symmetrize(diffs: Sequence[float]) -> np.ndarray:
    """|d| 与 −|d| 合并：重复差按定义关于 0 对称，对称化后拟合 loc=0 的分布。"""
    a = np.abs(np.asarray(diffs, dtype=np.float64))
    return np.concatenate([a, -a])


def fit_student_t(x: Sequence[float], df_bounds: tuple[float, float] = (2.0, 100.0)) -> tuple[float, float]:
    """loc=0 的 Student-t MLE：对 df 在 df_bounds 内一维搜索，每个 df 下 scale 用 scipy 拟合。"""
    x = np.asarray(x, dtype=np.float64)
    assert len(x) >= 4, "拟合至少需要 4 个样本"

    def nll(df: float) -> float:
        _, _, scale = student_t.fit(x, df, floc=0, fdf=df)
        return -float(student_t.logpdf(x, df, 0.0, scale).sum())

    res = minimize_scalar(nll, bounds=df_bounds, method="bounded",
                          options={"xatol": 1e-2})
    df = float(np.clip(res.x, *df_bounds))
    _, _, scale = student_t.fit(x, df, floc=0, fdf=df)
    return df, float(scale)


@dataclass(frozen=True)
class NoiseModel:
    """η 的分布：kind ∈ {student_t, empirical}；scale 按制式（"nd" 非破坏 / "dest" 破坏）。

    empirical 时 samples 为对称化重复差（scale=1 口径），cdf 按 scale 缩放样本后线性插值。
    """

    kind: str
    df: Optional[float]
    scale: dict[str, float]
    samples: Optional[np.ndarray]

    def cdf(self, delta: np.ndarray, scale: float) -> np.ndarray:
        d = np.asarray(delta, dtype=np.float64)
        if self.kind == "student_t":
            return student_t.cdf(d, self.df, 0.0, scale)
        if self.kind == "empirical":
            s = np.sort(np.asarray(self.samples, dtype=np.float64) * scale)
            n = len(s)
            # 对称化样本的经验 CDF（中点连续版：F(x) = (#s<x + 0.5·#s==x)/n，线性插值保单调）
            grid_y = (np.arange(n) + 0.5) / n
            out = np.interp(d, s, grid_y, left=0.0, right=1.0)
            return np.clip(out, 0.0, 1.0)
        raise ValueError(f"未知 kind={self.kind!r}")

    def to_json(self) -> dict:
        return {"kind": self.kind, "df": self.df, "scale": dict(self.scale),
                "n_samples": None if self.samples is None else int(len(self.samples))}


def fit_noise_model(
    replicate_diffs: dict[str, Sequence[float]], *, kind: str, min_regime_n: int = 5,
) -> NoiseModel:
    """按制式拟合。replicate_diffs = {"nd": [...], "dest": [...]}（dest 可缺/可少）。

    Student-t：df 在合并样本上拟合并共用；scale 各制式单独拟合（dest 样本 < min_regime_n
    时并入 nd 单制式）。empirical：samples 取合并对称化重复差（scale=1），各制式 scale
    取 |d| 中位数之比（nd=1），dest 不足时单制式。
    """
    nd = list(replicate_diffs.get("nd", []))
    dest = list(replicate_diffs.get("dest", []))
    two_regime = len(dest) >= min_regime_n
    all_diffs = nd + dest
    if kind == "student_t":
        df, _ = fit_student_t(symmetrize(all_diffs))
        if two_regime:
            _, _, s_nd = student_t.fit(symmetrize(nd), df, floc=0, fdf=df)
            _, _, s_dest = student_t.fit(symmetrize(dest), df, floc=0, fdf=df)
            scale = {"nd": float(s_nd), "dest": float(s_dest)}
        else:
            _, _, s_all = student_t.fit(symmetrize(all_diffs), df, floc=0, fdf=df)
            scale = {"nd": float(s_all)}
        return NoiseModel(kind="student_t", df=float(df), scale=scale, samples=None)
    if kind == "empirical":
        samples = symmetrize(all_diffs)
        if two_regime:
            m_nd = float(np.median(np.abs(nd)))
            m_dest = float(np.median(np.abs(dest)))
            ratio = (m_dest / m_nd) if m_nd > 0 else 1.0
            scale = {"nd": 1.0, "dest": float(max(ratio, 1e-6))}
        else:
            scale = {"nd": 1.0}
        return NoiseModel(kind="empirical", df=None, scale=scale, samples=samples)
    raise ValueError(f"未知 kind={kind!r}")


def task_scale_factor(*, n_eval: int, n_ref: float, kappa: float) -> float:
    """两成分尺度因子 √(κ·n_ref/n_eval + (1−κ))：计数成分 ∝ 1/√n_eval，地板不随 n 变。"""
    assert 0.0 <= kappa <= 1.0 and n_eval > 0 and n_ref > 0
    return float(np.sqrt(kappa * n_ref / n_eval + (1.0 - kappa)))


def pair_probability(delta: float, model: NoiseModel, scale_a: float, scale_b: float) -> float:
    """P(a ≻ b | Δ) = F_η(Δ)，混合制式 s_pair = √((s_a² + s_b²)/2)（协议 §1）。"""
    s = float(np.sqrt((scale_a ** 2 + scale_b ** 2) / 2.0))
    return float(model.cdf(np.array([delta]), s)[0])


def info_weight(p):
    """信息权 |2p − 1|（p=0.5 零权，p∈{0,1} 满权）。"""
    return np.abs(2.0 * np.asarray(p, dtype=np.float64) - 1.0) if np.ndim(p) else abs(2.0 * float(p) - 1.0)


# ---------------------------------------------------------------- 配对


@dataclass(frozen=True)
class SoftPair:
    """一条软 label 样本：p = P(a ≻ b)，w = 样本权，y_hard = 观测方向（Δ=0 时按 0 记，w 应为 0）。"""

    task_id: str
    key_a: str
    key_b: str
    vec_a: np.ndarray
    vec_b: np.ndarray
    p: float
    w: float
    y_hard: int
    kind: str


def _make_soft(a: ClsRecord, b: ClsRecord, margin_fn: MarginFn, prob_fn: ProbFn) -> SoftPair:
    d = float(margin_fn(a.r, b.r))
    p = float(prob_fn(a.task_id, a.key, b.key, d))
    return SoftPair(task_id=a.task_id, key_a=a.key, key_b=b.key, vec_a=a.vec, vec_b=b.vec,
                    p=p, w=float(info_weight(p)), y_hard=int(d > 0),
                    kind="-".join(sorted((a.origin, b.origin))))


def soft_pairs_all(records: Sequence[ClsRecord], margin_fn: MarginFn, prob_fn: ProbFn) -> list[SoftPair]:
    """静态穷举：每任务全部无序对（δ=0，不弃样；平局 p=0.5/w=0 保留）。评估集口径。"""
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    out: list[SoftPair] = []
    for _tid, rs in sorted(by_task.items()):
        for a, b in combinations(rs, 2):
            if a.key == b.key:
                continue
            out.append(_make_soft(a, b, margin_fn, prob_fn))
    return out


def soft_dynamic_pairs(
    records: Sequence[ClsRecord], margin_fn: MarginFn, prob_fn: ProbFn, *,
    rng: np.random.Generator, origins: Optional[tuple[str, ...]] = None,
) -> list[SoftPair]:
    """每 epoch 动态配对（与 tclsc_engine.dynamic_pairs 同构：任务内乱序副本逐位配对），
    但不做 δ/平局弃样。origins 给出时只用该来源记录（curriculum）。"""
    if origins is not None:
        records = [r for r in records if r.origin in origins]
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    out: list[SoftPair] = []
    for _tid, recs in by_task.items():
        partners = list(recs)
        rng.shuffle(partners)  # type: ignore[arg-type]
        for a, b in zip(recs, partners):
            if a.key == b.key:
                continue
            out.append(_make_soft(a, b, margin_fn, prob_fn))
    return out


def hard_dynamic_pairs_soft(
    records: Sequence[ClsRecord], margin_fn: MarginFn, *, delta,
    rng: np.random.Generator, origins: Optional[tuple[str, ...]] = None,
) -> list[SoftPair]:
    """硬 δ 口径的动态配对（语义与 tclsc_engine.dynamic_pairs 逐条一致：自配对/|Δ|<δ/
    严格平局三类跳过），但产出带 cls_key 的 SoftPair（p=y，w=1）——翻转注入按无序对
    哈希需要 key。delta 为标量或 {task_id: δ}。"""
    if origins is not None:
        records = [r for r in records if r.origin in origins]
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    out: list[SoftPair] = []
    for tid, recs in by_task.items():
        d = delta[tid] if isinstance(delta, dict) else float(delta)
        partners = list(recs)
        rng.shuffle(partners)  # type: ignore[arg-type]
        for a, b in zip(recs, partners):
            if a.key == b.key:
                continue
            m = float(margin_fn(a.r, b.r))
            if abs(m) < d or m == 0.0:
                continue
            y = int(m > 0)
            out.append(SoftPair(task_id=a.task_id, key_a=a.key, key_b=b.key, vec_a=a.vec,
                                vec_b=b.vec, p=float(y), w=1.0, y_hard=y,
                                kind="-".join(sorted((a.origin, b.origin)))))
    return out


def hard_to_soft(pairs: Sequence[Pair]) -> list[SoftPair]:
    """硬 label 配对 → SoftPair（p=y，w=1）：让硬 δ 臂与软臂走同一训练循环。"""
    return [SoftPair(task_id=p.task_id, key_a=getattr(p, "key_a", ""), key_b=getattr(p, "key_b", ""),
                     vec_a=p.vec_a, vec_b=p.vec_b, p=float(p.y), w=1.0, y_hard=int(p.y), kind=p.kind)
            for p in pairs]


def with_reversed_soft(pairs: Sequence[SoftPair]) -> list[SoftPair]:
    """追加反向样本 (b, a, 1−p, 1−y)。"""
    return list(pairs) + [
        replace(p, key_a=p.key_b, key_b=p.key_a, vec_a=p.vec_b, vec_b=p.vec_a,
                p=1.0 - p.p, y_hard=1 - p.y_hard) for p in pairs
    ]


def flip_decision(seed: int, task_id: str, key_a: str, key_b: str, rate: float) -> bool:
    """确定性翻转：md5(seed, task, 无序对) 前 8 位十六进制 / 2³² < rate。跨 epoch 一致、a/b 对称。"""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    lo, hi = sorted((key_a, key_b))
    h = hashlib.md5(f"flip:{seed}:{task_id}:{lo}:{hi}".encode()).hexdigest()[:8]
    return int(h, 16) / 2 ** 32 < rate


def apply_flips(pairs: Sequence[SoftPair], *, seed: int, rate: float) -> list[SoftPair]:
    """按 flip_decision 翻转 y_hard 与 p（w 不变）。rate=0 原样返回。"""
    if rate <= 0:
        return list(pairs)
    return [replace(p, y_hard=1 - p.y_hard, p=1.0 - p.p)
            if flip_decision(seed, p.task_id, p.key_a, p.key_b, rate) else p
            for p in pairs]


# ---------------------------------------------------------------- 加权任务盲天花板


def weighted_task_blind_ceiling(
    votes: dict[tuple[str, str], Sequence[tuple[str, int, float]]],
    task_filter: Optional[Callable[[str], bool]] = None,
) -> dict:
    """votes[(a,b)] = [(task_id, sign∈{0,1}, w)]。任务盲预测器每对只能选一个方向：
    天花板 = Σ_pairs max_d Σ_t w·1[sign=d] / Σ_pairs Σ_t w。零权票不计。"""
    maj = tot = 0.0
    n_votes = 0
    for vs in votes.values():
        sel = [(s, w) for t, s, w in vs if (task_filter is None or task_filter(t)) and w > 0]
        if not sel:
            continue
        pos = sum(w for s, w in sel if s == 1)
        neg = sum(w for s, w in sel if s == 0)
        maj += max(pos, neg)
        tot += pos + neg
        n_votes += len(sel)
    return {"ceiling": (maj / tot) if tot > 0 else None, "mass": tot, "n_votes": n_votes}


# ---------------------------------------------------------------- 引擎


class SoftTCLSCEngine(TCLSCEngine):
    """在 TCLSCEngine 上加软/加权 BCE 训练（train_soft）；评估/预测沿用基类。"""

    def train_soft(
        self, sampler: Callable[[int, np.random.Generator], Sequence[SoftPair]], *, epochs: int,
    ) -> list[dict]:
        """每 epoch 由 sampler(epoch, rng) 给出 SoftPair 列表；loss = Σ w·BCE(out, p) / Σ w。
        w 全零的 batch 跳过（不更新）。"""
        history: list[dict] = []
        for epoch in range(epochs):
            pairs = list(sampler(epoch, self.rng))
            self.rng.shuffle(pairs)  # type: ignore[arg-type]
            self.model.train()
            losses, n_seen, w_sum = [], 0, 0.0
            for batch in self._batches(pairs):
                w = torch.as_tensor([p.w for p in batch], dtype=torch.float32, device=self.device)
                if float(w.sum()) <= 0:
                    continue
                a, b, E, ctx, _ = self._encode_batch(batch, grad=True)  # 与硬 label 同一条编码路径
                target = torch.as_tensor([p.p for p in batch], dtype=torch.float32, device=self.device)
                out = self.model(a, b, E, ctx)
                bce = torch.nn.functional.binary_cross_entropy(out, target, reduction="none")
                loss = (w * bce).sum() / w.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                n_seen += len(batch)
                w_sum += float(w.sum())
            history.append({
                "epoch": epoch, "n_pairs": n_seen,
                "w_mean": (w_sum / n_seen) if n_seen else float("nan"),
                "loss": float(np.mean(losses)) if losses else float("nan"),
            })
        return history
