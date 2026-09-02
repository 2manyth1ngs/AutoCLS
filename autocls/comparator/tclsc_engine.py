"""T-CLSC 训练引擎（TECHNICAL_DESIGN.md 模块 6；改自 AutoCTS++ AHC 训练机制）。

来源（reference/AutoCTS_plusplus，逐段标注）：
- 动态配对 ← utils.py::AHC_DataLoader_linear.generate_pairs（L1047-1056）：每个
  epoch 把任务内记录与同任务随机伙伴重新配对、按指标定 y——等价于每轮从
  O(n²) 配对空间重采样 O(n) 条，比静态全配对多样且省显存。
- 训练/评估循环 ← NAS_Net/AHC/ahc_engine.py::train_task_oriented_ahc / evaluate
  （L11-70）：BCE + round(output) 计配对准确率。
- curriculum（先易后难）← AutoCTS++ 3.2.4：shared CLS（全任务共用，跨任务可
  直接对比同一 CLS 的排名差异）先训，random CLS（每任务独立的难知识）后入。
  此处实现为：前 curriculum_epochs 个 epoch 只用 origin=='shared' 的记录配对。

与原码差异：
1. 原码按 (task, arch, mae) 三元组配对，方向硬编码 `<`（MAE 小者优）；此处
   优劣方向经注入的 better_fn 判定——comparator 是任务无关模块（铁律 7），
   不 import task_types；正式脚本必须绑定 task_types.is_better（铁律 8）。
2. 增加 δ 弃样：|r_a - r_b| < delta 的近平局是同分噪声，不给二值 label
   （与 labeling.make_pairs 同参数同语义，δ 标定见 configs/labels.yaml）。
3. 任务特征 = windows_repr (m, T, D)，经模型持有的 SetTF 联合训练编码为 E
   （原码 task_feature 语义）；同批去重后只前向一次/任务。
4. 增加 E 消融评估（e_mode）：机制验证的关键——区分"真 task-aware"与
   "退化为忽略 E 的通用 CLS 排序器"（详见 evaluate docstring）。

【定位（路线 B）】本引擎当前用于 HAR 同分布切片任务的机制验证（M3）：留出
任务与源任务同分布，只能验证"泛化到新任务切片"，不构成 zero-shot（跨数据
集）证据——真 zero-shot 由路线 A（多 UEA 数据集、源/目标池分离）产出。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from .tclsc_model import TCLSC

BetterFn = Callable[[float, float], bool]


@dataclass(frozen=True)
class ClsRecord:
    """一条 (task, cls) → R' 记录（rprime 缓存的引擎侧视图）。"""

    task_id: str
    key: str      # cls_key（自配对防护与追溯用）
    origin: str   # labeling.ORIGIN_SHARED / ORIGIN_RANDOM（curriculum 分组）
    vec: np.ndarray  # CLSEncoder.encode(cls)
    r: float      # early-val R'


@dataclass(frozen=True)
class Pair:
    """一条训练/评估样本：y = int(better_fn(r_a, r_b))。"""

    task_id: str
    vec_a: np.ndarray
    vec_b: np.ndarray
    y: int
    kind: str  # "shared-shared" / "random-shared" / "random-random"


DeltaSpec = Union[float, Mapping[str, float]]


def _delta_of(delta: DeltaSpec, task_id: str) -> float:
    """δ 解析：标量全局同值；映射按 task_id 查表（per-dataset δ 的引擎侧形态
    ——引擎不知道数据集概念，调用方把数据集 δ 展开成 per-task 表，铁律 7）。
    映射缺任务即 KeyError fail-fast，绝不静默回退。"""
    if isinstance(delta, Mapping):
        return delta[task_id]
    return delta


def dynamic_pairs(
    records: Sequence[ClsRecord],
    better_fn: BetterFn,
    *,
    delta: DeltaSpec,
    rng: np.random.Generator,
    origins: Optional[tuple[str, ...]] = None,
) -> list[Pair]:
    """每 epoch 动态配对（← AHC_DataLoader_linear.generate_pairs）。

    任务内把记录序列与其乱序副本逐位配对；自配对、|Δr|<delta 近平局、双向
    均不优（严格平局）三类跳过。origins 给出时只用该来源的记录（curriculum
    第一阶段传 ('shared',)）。delta 可为 {task_id: δ} 映射（_delta_of）。
    """
    if origins is not None:
        records = [r for r in records if r.origin in origins]
    by_task: dict[str, list[ClsRecord]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)

    pairs: list[Pair] = []
    for tid, recs in by_task.items():
        d = _delta_of(delta, tid)
        partners = list(recs)
        rng.shuffle(partners)  # type: ignore[arg-type]
        for a, b in zip(recs, partners):
            if a.key == b.key:
                continue
            if abs(a.r - b.r) < d:
                continue
            a_better = better_fn(a.r, b.r)
            if not (a_better or better_fn(b.r, a.r)):
                continue  # 严格平局：不赋假方向（与 labeling.make_pairs 一致）
            pairs.append(Pair(
                task_id=a.task_id, vec_a=a.vec, vec_b=b.vec,
                y=int(a_better), kind="-".join(sorted((a.origin, b.origin))),
            ))
    return pairs


def with_reversed(pairs: Sequence[Pair]) -> list[Pair]:
    """追加反向样本 (b, a, 1-y)（← random_search.py::generate_task_pairs 双向出样；
    y 翻转对任何任务类型的二值比较都成立，不引入方向假设）。"""
    return list(pairs) + [
        replace(p, vec_a=p.vec_b, vec_b=p.vec_a, y=1 - p.y) for p in pairs
    ]


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """无不动点排列（E 消融 shuffle 档：任务 A 的 E 必须配给别的任务）。"""
    if n < 2:
        raise ValueError(f"错位排列需要 n>=2，得到 n={n}")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


class TCLSCEngine:
    """持有模型 + 各任务 windows_repr，提供 curriculum 训练与 E 消融评估。"""

    def __init__(
        self,
        model: TCLSC,
        task_windows: dict[str, np.ndarray],
        *,
        device: str = "cpu",
        lr: float = 1e-4,   # ← random_search.py --ahc_lr 默认 1e-4
        batch_size: int = 64,  # ← random_search.py --batch_size 默认 64
        seed: int = 0,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.task_windows = {
            tid: torch.as_tensor(w, dtype=torch.float32, device=device)
            for tid, w in task_windows.items()
        }
        self.criterion = nn.BCELoss()  # ← random_search.py L479
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ 训练

    def train(
        self,
        records: Sequence[ClsRecord],
        *,
        epochs: int,
        curriculum_epochs: int,
        better_fn: BetterFn,
        delta: DeltaSpec,
    ) -> list[dict]:
        """动态配对 + curriculum 训练，返回逐 epoch 历史（loss/配对准确率）。

        ← ahc_engine.py::train_task_oriented_ahc（每 epoch shuffle 重配对 +
        BCE + round 计准确率）；curriculum ← AutoCTS++ 3.2.4。
        """
        history: list[dict] = []
        for epoch in range(epochs):
            phase = "shared" if epoch < curriculum_epochs else "all"
            origins = ("shared",) if phase == "shared" else None
            pairs = dynamic_pairs(records, better_fn, delta=delta,
                                  rng=self.rng, origins=origins)
            self.rng.shuffle(pairs)  # type: ignore[arg-type]

            self.model.train()
            losses, n_correct, n_seen = [], 0, 0
            for batch in self._batches(pairs):
                a, b, E, y = self._materialize(batch, grad=True)
                out = self.model(a, b, E)
                loss = self.criterion(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                n_correct += (torch.round(out) == y).sum().item()
                n_seen += len(batch)
            history.append({
                "epoch": epoch, "phase": phase, "n_pairs": n_seen,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "acc": n_correct / n_seen if n_seen else float("nan"),
            })
        return history

    # ------------------------------------------------------------------ 评估

    def _build_e_map(
        self, tids: list[str], *, e_mode: str, seed: int,
        e_assignment: Optional[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        """按 e_mode 给每个测试任务配 E（normal/shuffle/assign/zero；语义见 evaluate）。"""
        if e_mode not in ("normal", "shuffle", "assign", "zero"):
            raise ValueError(f"未知 e_mode={e_mode!r}（可选 normal/shuffle/assign/zero）")
        if (e_assignment is not None) != (e_mode == "assign"):
            raise ValueError("e_assignment 当且仅当 e_mode='assign' 时给出")
        if e_mode == "assign":
            missing = [t for t in tids if t not in e_assignment]
            if missing:
                raise ValueError(f"e_assignment 缺任务映射: {missing}")
            unknown = sorted({e_assignment[t] for t in tids} - set(self.task_windows))
            if unknown:
                raise ValueError(f"e_assignment 目标任务无特征: {unknown}")
            src = sorted({e_assignment[t] for t in tids})
            rows = self.model.encode_tasks([self.task_windows[t] for t in src])
            idx = {t: i for i, t in enumerate(src)}
            return {t: rows[idx[e_assignment[t]]] for t in tids}
        e_rows = self.model.encode_tasks([self.task_windows[t] for t in tids])
        if e_mode == "zero":
            e_rows = torch.zeros_like(e_rows)
        elif e_mode == "shuffle":
            e_rows = e_rows[torch.as_tensor(
                derangement(len(tids), np.random.default_rng(seed)), device=self.device)]
        return {t: e_rows[i] for i, t in enumerate(tids)}

    def predict(
        self, pairs: Sequence, *, e_mode: str = "normal", seed: int = 0,
        e_assignment: Optional[dict[str, str]] = None,
    ) -> np.ndarray:
        """逐样本 P(a 优于 b)（numpy, 与 pairs 同序）。pairs 只需有 task_id/vec_a/vec_b
        （Pair 或 soft_labels.SoftPair 均可）；E 消融语义同 evaluate。"""
        return self._predict_tensor(pairs, e_mode=e_mode, seed=seed,
                                    e_assignment=e_assignment).cpu().numpy()

    def _predict_tensor(self, pairs: Sequence, *, e_mode: str, seed: int,
                        e_assignment: Optional[dict[str, str]]) -> torch.Tensor:
        tids = sorted({p.task_id for p in pairs})
        self.model.eval()
        outs = []
        with torch.no_grad():
            e_map = self._build_e_map(tids, e_mode=e_mode, seed=seed, e_assignment=e_assignment)
            for batch in self._batches(pairs):
                a = self._stack([p.vec_a for p in batch])
                b = self._stack([p.vec_b for p in batch])
                E = torch.stack([e_map[p.task_id] for p in batch])
                outs.append(self.model(a, b, E))
        return torch.cat(outs) if outs else torch.zeros(0, device=self.device)

    def evaluate(
        self, pairs: Sequence[Pair], *, e_mode: str = "normal", seed: int = 0,
        e_assignment: Optional[dict[str, str]] = None, return_correct: bool = False,
    ) -> dict:
        """配对准确率评估（← ahc_engine.py::evaluate），带 E 消融开关。

        e_mode（机制验证的关键，不重训、只改测试输入）：
        - normal：每任务用自己的 E；
        - shuffle：任务间错位（derangement）——任务 A 的样本配任务 B 的 E。
          若准确率不掉，说明模型没在看 E；
        - assign：按调用方给的 e_assignment（task_id → 取谁的 E）逐任务指定，
          目标任务只需在 task_windows 里（可指向测试集外的任务）。分组消融
          （如"数据集内错位"/"跨数据集对调"）的分组知识留在调用方——引擎
          保持任务无关（铁律 7），不 import 任何数据集概念；
        - zero：E 置零（set_fc 只剩常数支路）。
        判读：消融后大幅下降 → 真 task-aware；几乎不变 → 退化为忽略 E 的
        通用 CLS 排序器（很可能是源任务多样性不足的必然结果，印证转路线 A）。
        """
        out = self._predict_tensor(pairs, e_mode=e_mode, seed=seed, e_assignment=e_assignment)
        n = len(pairs)
        with torch.no_grad():
            y = torch.as_tensor([float(p.y) for p in pairs], device=self.device)
            loss = self.criterion(out, y).item() if n else float("nan")
            hit = (torch.round(out) == y)
        res = {
            "e_mode": e_mode, "n": n,
            "acc": (hit.sum().item() / n) if n else float("nan"),
            "loss": loss,
        }
        if return_correct:
            res["correct"] = hit.cpu().numpy().astype(bool)
        return res

    # ------------------------------------------------------------------ 内部

    def _batches(self, pairs: Sequence[Pair]):
        for i in range(0, len(pairs), self.batch_size):
            yield pairs[i:i + self.batch_size]

    def _stack(self, vecs: list[np.ndarray]) -> torch.Tensor:
        return torch.as_tensor(np.stack(vecs), dtype=torch.float32, device=self.device)

    def _materialize(self, batch: Sequence[Pair], *, grad: bool):
        """一个 batch → (cls_a, cls_b, E, y)。任务去重后每任务只过一次 SetTF
        （训练时梯度经 index 展开回传，与逐样本前向数学等价）。"""
        tids = sorted({p.task_id for p in batch})
        e_unique = self.model.encode_tasks([self.task_windows[t] for t in tids])
        idx = {t: i for i, t in enumerate(tids)}
        E = e_unique[torch.as_tensor([idx[p.task_id] for p in batch], device=self.device)]
        a = self._stack([p.vec_a for p in batch])
        b = self._stack([p.vec_b for p in batch])
        y = torch.as_tensor([float(p.y) for p in batch], device=self.device)
        return a, b, E, y
