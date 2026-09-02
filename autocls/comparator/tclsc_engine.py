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
   （原码 task_feature 语义）；同批按**窗口键**去重后只前向一次/键。
4. 增加 E 消融评估（e_mode）：机制验证的关键——区分"真 task-aware"与
   "退化为忽略 E 的通用 CLS 排序器"（详见 evaluate docstring）。
5. 任务 id 与窗口键解耦（windows_key_of）+ 任务设定向量（context_of）：一个
   引擎任务可以是"同一份窗口 + 一段设定"（原码 (数据集, P, Q) 的类比），多个
   引擎任务共用一份 windows_repr、只差 ctx。ctx 的语义（预算 one-hot 等）与
   id 的拼法全留在调用方——引擎只当它们是字符串与浮点向量（铁律 7）。

【定位（路线 B）】本引擎当前用于 HAR 同分布切片任务的机制验证（M3）：留出
任务与源任务同分布，只能验证"泛化到新任务切片"，不构成 zero-shot（跨数据
集）证据——真 zero-shot 由路线 A（多 UEA 数据集、源/目标池分离）产出。
"""
from __future__ import annotations

import contextlib
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
    """持有模型 + 各窗口键的 windows_repr，提供 curriculum 训练与 E/ctx 消融评估。"""

    def __init__(
        self,
        model: TCLSC,
        task_windows: dict[str, np.ndarray],
        *,
        device: str = "cpu",
        lr: float = 1e-4,   # ← random_search.py --ahc_lr 默认 1e-4
        batch_size: int = 64,  # ← random_search.py --batch_size 默认 64
        seed: int = 0,
        windows_key_of: Optional[dict[str, str]] = None,
        context_of: Optional[dict[str, np.ndarray]] = None,
    ):
        """task_windows 按**窗口键**索引；windows_key_of 给出 任务 id → 窗口键
        （缺省恒等，即任务 id 就是窗口键），context_of 给出 任务 id → ctx (ctx_dim,)
        （缺省不传 ctx）。两张表的键都只是字符串，引擎不解析其结构（铁律 7）。"""
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.task_windows = {
            key: torch.as_tensor(w, dtype=torch.float32, device=device)
            for key, w in task_windows.items()
        }
        self.windows_key_of = dict(windows_key_of) if windows_key_of is not None else None
        self.context_of, self.ctx_dim = self._prepare_context(context_of, device)
        if model.ctx_dim != self.ctx_dim:
            raise ValueError(
                f"模型 ctx_dim={model.ctx_dim} 与 context_of 维度 {self.ctx_dim} 不一致")
        self.criterion = nn.BCELoss()  # ← random_search.py L479
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _prepare_context(
        context_of: Optional[dict[str, np.ndarray]], device: str,
    ) -> tuple[Optional[dict[str, torch.Tensor]], int]:
        """context_of → {任务: (ctx_dim,) 张量} + ctx_dim；各任务维度必须一致。"""
        if context_of is None:
            return None, 0
        if not context_of:
            raise ValueError("context_of 不得为空映射（不用 ctx 就传 None）")
        out: dict[str, torch.Tensor] = {}
        dim: Optional[int] = None
        for tid, v in context_of.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"context_of[{tid!r}] 应为一维 (ctx_dim,)，得到 {arr.shape}")
            if dim is None:
                dim = int(arr.shape[0])
            elif int(arr.shape[0]) != dim:
                raise ValueError(f"context_of 各任务维度必须一致：{tid!r} 为 {arr.shape[0]}，前为 {dim}")
            out[tid] = torch.as_tensor(arr, device=device)
        return out, int(dim)

    def _windows_key(self, tid: str) -> str:
        """任务 id → 窗口键（缺省恒等；给了表就必须查得到，绝不静默回退）。"""
        if self.windows_key_of is None:
            return tid
        try:
            return self.windows_key_of[tid]
        except KeyError:
            raise KeyError(f"windows_key_of 缺任务映射: {tid!r}") from None

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
                a, b, E, ctx, y = self._encode_batch(batch, grad=True)
                out = self.model(a, b, E, ctx)
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
    ) -> Optional[dict[str, torch.Tensor]]:
        """按 e_mode 给每个测试任务配 E（normal/shuffle/assign/zero；语义见 evaluate）。

        SetTF 按**窗口键**去重：共用一份 windows_repr 的多个引擎任务只前向一次、
        拿到同一行 E；错位（shuffle）也在窗口键上做——同一切片的各设定拿到同一
        份别家 E（否则"错位"会在共用窗口的任务之间空转）。
        三条"消融必须真的动了东西"的守卫，宁可报错也不给假阴性：
        1. 任务支路根本不吃 E 的模型（task_branch ctx/none）只接受 e_mode='normal'
           ——否则消融结果会与 normal 逐位相同却被贴上 shuffle/zero 的标签；
        2. assign 的来源解析到与接收方同一个窗口键 = 恒等消融（budget_assignment
           那类"同切片换设定"的映射误传进来就是这形状），直接报错；
        3. 唯一窗口键 < 2 无法错位（提前报错，别等 derangement 抛不知所云的 n>=2）。
        **seed 无效的边界**：唯一窗口键恰为 2 时错位排列只有一种（对调），
        e_mode='shuffle' 的结果与 seed 无关——调用方不得在这种规模上报 seed 方差。
        """
        if e_mode not in ("normal", "shuffle", "assign", "zero"):
            raise ValueError(f"未知 e_mode={e_mode!r}（可选 normal/shuffle/assign/zero）")
        if (e_assignment is not None) != (e_mode == "assign"):
            raise ValueError("e_assignment 当且仅当 e_mode='assign' 时给出")
        if not self.model.uses_E:
            if e_mode != "normal":
                raise ValueError(
                    f"模型 task_branch={self.model.task_branch!r} 的任务支路不吃 E，"
                    f"不能做 e_mode={e_mode!r} 消融（结果会与 normal 完全相同）")
            return None
        if e_mode == "assign":
            missing = [t for t in tids if t not in e_assignment]
            if missing:
                raise ValueError(f"e_assignment 缺任务映射: {missing}")
            donors = {t: e_assignment[t] for t in tids}
            if self.windows_key_of is not None:
                unmapped = sorted({d for d in donors.values() if d not in self.windows_key_of})
                if unmapped:
                    raise ValueError(f"e_assignment 目标任务不在 windows_key_of: {unmapped}")
            src = {t: self._windows_key(d) for t, d in donors.items()}
            unknown = sorted(set(src.values()) - set(self.task_windows))
            if unknown:
                raise ValueError(f"e_assignment 目标任务无特征: {unknown}")
            identity = sorted(t for t in tids if src[t] == self._windows_key(t))
            if identity:
                raise ValueError(
                    f"e_assignment 的来源与接收方同窗口键（恒等消融，E 实际没换）: {identity}")
        else:
            src = {t: self._windows_key(t) for t in tids}
        keys = sorted(set(src.values()))
        if e_mode == "shuffle" and len(keys) < 2:
            raise ValueError(f"唯一窗口键只有 {len(keys)} 个，无法错位（shuffle 需 >=2 个窗口键）")
        rows = self.model.encode_tasks([self.task_windows[k] for k in keys])
        if e_mode == "zero":
            rows = torch.zeros_like(rows)
        elif e_mode == "shuffle":
            rows = rows[torch.as_tensor(
                derangement(len(keys), np.random.default_rng(seed)), device=self.device)]
        idx = {k: i for i, k in enumerate(keys)}
        return {t: rows[idx[src[t]]] for t in tids}

    def _build_ctx_map(
        self, tids: list[str], *, ctx_mode: str, ctx_assignment: Optional[dict[str, str]],
    ) -> Optional[dict[str, torch.Tensor]]:
        """按 ctx_mode 给每个测试任务配任务设定向量（normal/assign/zero；语义见 evaluate）。"""
        if ctx_mode not in ("normal", "assign", "zero"):
            raise ValueError(f"未知 ctx_mode={ctx_mode!r}（可选 normal/assign/zero）")
        if (ctx_assignment is not None) != (ctx_mode == "assign"):
            raise ValueError("ctx_assignment 当且仅当 ctx_mode='assign' 时给出")
        if self.context_of is None:
            if ctx_mode != "normal":
                raise ValueError(f"未提供 context_of，不能做 ctx_mode={ctx_mode!r} 消融")
            return None
        if ctx_mode == "zero":
            zero = torch.zeros(self.ctx_dim, device=self.device)
            return {t: zero for t in tids}
        if ctx_mode == "assign":
            missing = [t for t in tids if t not in ctx_assignment]
            if missing:
                raise ValueError(f"ctx_assignment 缺任务映射: {missing}")
            donors = {t: ctx_assignment[t] for t in tids}
        else:
            donors = {t: t for t in tids}
        unknown = sorted(set(donors.values()) - set(self.context_of))
        if unknown:
            raise ValueError(f"ctx 来源任务无设定向量: {unknown}")
        return {t: self.context_of[d] for t, d in donors.items()}

    def predict(
        self, pairs: Sequence, *, e_mode: str = "normal", seed: int = 0,
        e_assignment: Optional[dict[str, str]] = None,
        ctx_mode: str = "normal", ctx_assignment: Optional[dict[str, str]] = None,
    ) -> np.ndarray:
        """逐样本 P(a 优于 b)（numpy, 与 pairs 同序）。pairs 只需有 task_id/vec_a/vec_b
        （Pair 或 soft_labels.SoftPair 均可）；E/ctx 消融语义同 evaluate。"""
        return self._predict_tensor(
            pairs, e_mode=e_mode, seed=seed, e_assignment=e_assignment,
            ctx_mode=ctx_mode, ctx_assignment=ctx_assignment).cpu().numpy()

    def _predict_tensor(
        self, pairs: Sequence, *, e_mode: str, seed: int,
        e_assignment: Optional[dict[str, str]],
        ctx_mode: str = "normal", ctx_assignment: Optional[dict[str, str]] = None,
    ) -> torch.Tensor:
        tids = sorted({p.task_id for p in pairs})
        self.model.eval()
        outs = []
        with torch.no_grad():
            # 错位/指派必须在整份测试集上算一次（按 batch 重算会变成 batch 内错位）
            e_map = self._build_e_map(tids, e_mode=e_mode, seed=seed, e_assignment=e_assignment)
            ctx_map = self._build_ctx_map(tids, ctx_mode=ctx_mode, ctx_assignment=ctx_assignment)
            for batch in self._batches(pairs):
                a, b, E, ctx, _ = self._encode_batch(
                    batch, grad=False, e_map=e_map, ctx_map=ctx_map)
                outs.append(self.model(a, b, E, ctx))
        return torch.cat(outs) if outs else torch.zeros(0, device=self.device)

    def evaluate(
        self, pairs: Sequence[Pair], *, e_mode: str = "normal", seed: int = 0,
        e_assignment: Optional[dict[str, str]] = None,
        ctx_mode: str = "normal", ctx_assignment: Optional[dict[str, str]] = None,
        return_correct: bool = False,
    ) -> dict:
        """配对准确率评估（← ahc_engine.py::evaluate），带 E / ctx 消融开关。

        e_mode（机制验证的关键，不重训、只改测试输入）：
        - normal：每任务用自己的 E；
        - shuffle：窗口键间错位（derangement）——切片 A 的样本配切片 B 的 E，
          共用一份窗口的多个任务拿到同一份别家 E。若准确率不掉，说明模型没在看 E。
          **测试集只有 2 个唯一窗口键时错位排列唯一（对调），结果与 seed 无关**，
          此规模下不得报 seed 方差；<2 个窗口键直接报错（无法错位）；
        - assign：按调用方给的 e_assignment（task_id → 取谁的 E）逐任务指定，
          目标任务的窗口键只需在 task_windows 里（可指向测试集外的任务）。
          ctx 不受影响（仍用各任务自己的），两条支路可独立消融。来源与接收方
          同窗口键（恒等消融）直接报错。分组消融（如"数据集内错位"/"跨数据集
          对调"）的分组知识留在调用方——引擎保持任务无关（铁律 7），不 import
          任何数据集概念；
        - zero：E 置零（set_fc 只剩 ctx / 常数支路）。
        任务支路不吃 E 的模型（task_branch ctx/none）只接受 e_mode='normal'——
        非 normal 档会报错，不会返回一个"贴着 shuffle 标签、值同 normal"的结果
        （与 ctx 侧对称：没有 context_of 就不能做 ctx_mode 消融）。
        ctx_mode（与 e_mode 正交，仅在给了 context_of 时可用）：
        - normal：每任务用自己的设定向量；
        - assign：按 ctx_assignment（task_id → 取谁的 ctx）指定（如"换预算"）；
        - zero：ctx 置零。
        判读：消融后大幅下降 → 真 task-aware；几乎不变 → 退化为忽略 E 的
        通用 CLS 排序器（很可能是源任务多样性不足的必然结果，印证转路线 A）。
        """
        out = self._predict_tensor(pairs, e_mode=e_mode, seed=seed, e_assignment=e_assignment,
                                   ctx_mode=ctx_mode, ctx_assignment=ctx_assignment)
        n = len(pairs)
        with torch.no_grad():
            y = torch.as_tensor([float(p.y) for p in pairs], device=self.device)
            loss = self.criterion(out, y).item() if n else float("nan")
            hit = (torch.round(out) == y)
        res = {
            "e_mode": e_mode, "ctx_mode": ctx_mode, "n": n,
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

    def _encode_batch(
        self, batch: Sequence, *, e_mode: str = "normal", seed: int = 0,
        e_assignment: Optional[dict[str, str]] = None,
        ctx_mode: str = "normal", ctx_assignment: Optional[dict[str, str]] = None,
        grad: bool = True,
        e_map: Optional[dict[str, torch.Tensor]] = None,
        ctx_map: Optional[dict[str, torch.Tensor]] = None,
    ):
        """一个 batch → (cls_a, cls_b, E, ctx, y)（软/硬 label 训练与预测共用）。

        窗口键去重后每键只过一次 SetTF（训练时梯度经 stack 展开回传，与逐样本
        前向数学等价）。e_map/ctx_map 给出时直接查表——评估侧的错位/指派必须在
        整份测试集上算好，不能按 batch 重算；缺省则按本 batch 的任务现算（训练侧）。
        E/ctx 在模型不吃该支路时为 None；y 在样本无 y 字段时为 None
        （SoftPair 的目标是 p，由调用方自行组装）。
        """
        tids = sorted({p.task_id for p in batch})
        with contextlib.nullcontext() if grad else torch.no_grad():
            if e_map is None:
                e_map = self._build_e_map(tids, e_mode=e_mode, seed=seed,
                                          e_assignment=e_assignment)
            if ctx_map is None:
                ctx_map = self._build_ctx_map(tids, ctx_mode=ctx_mode,
                                              ctx_assignment=ctx_assignment)
            a = self._stack([p.vec_a for p in batch])
            b = self._stack([p.vec_b for p in batch])
            E = None if e_map is None else torch.stack([e_map[p.task_id] for p in batch])
            ctx = None if ctx_map is None else torch.stack([ctx_map[p.task_id] for p in batch])
            y = (torch.as_tensor([float(p.y) for p in batch], device=self.device)
                 if hasattr(batch[0], "y") else None)
        return a, b, E, ctx, y
