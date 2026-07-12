"""T-CLSC 训练 label 生成的领域逻辑（TECHNICAL_DESIGN.md 模块 7；scripts/2_generate_labels.py 的可测核心）。

照 AutoCTS++ 3.2.4（exps/generate_seeds.py noisy_seeds 模式）：每任务的候选 =
L 个 shared CLS（全任务共用同一批，固定 seed 采样——T-CLSC 借它直接对比"同一
CLS 在不同任务上的排名差异"，是学习任务相似性的关键信号）+ L 个 random CLS
（每任务独立采样，更难的知识）；shared/random 二分同时是后续 curriculum
learning 的先易后难依据，配对记录 kind 字段供其分组。

任务类型显式分发（铁律 7）：本模块是任务类型相关的（train/eval 划分方式、
R' 优劣方向都随 task_type 变），允许 import task_types；配对方向唯一合法途径
是 is_better（铁律 8），本模块不出现任何 `r1 > r2` 式硬编码方向。

δ 过滤（独立于 is_better 的平局弃样）：M2 第一步发现高分簇内 early-val 分辨力
弱（簇内 ρ=0.42）——full acc 只差 0.001~0.003 的配对在 early R' 上的差值是同分
噪声，给二值 label 等于拿噪声当监督。is_better 只处理"完全相等"，"几乎相等"
由 |R'_a - R'_b| < δ 弃样处理；δ 从 results/m2_probe/cache.jsonl 标定
（scripts/calibrate_delta.py + delta_decision_curve），存 configs/labels.yaml。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from .comparator.tclsc_engine import ClsRecord, Pair, with_reversed
from .space.cls_encoder import CLSEncoder
from .space.cls_space import CLS, CLSSpace
from .task_types import TASK_CLASSIFICATION, is_better, validate_task_type

ORIGIN_SHARED = "shared"
ORIGIN_RANDOM = "random"


def cls_key(cls_cfg: CLS) -> str:
    """CLS 的稳定短哈希（缓存/配对引用键；与 m2 探针 cls_hash 同构）。"""
    payload = json.dumps(asdict(cls_cfg), sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# CLS 采样（AutoCTS++ 3.2.4 shared + random）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLSSampling:
    """采样结果：shared 全任务共用；random[task_id] 每任务独立。"""

    shared: tuple[CLS, ...]
    random: dict[str, tuple[CLS, ...]]


def _sample_unique(
    space: CLSSpace, rng: np.random.Generator, n: int, taken: set[str]
) -> tuple[CLS, ...]:
    """逐个采样直到集齐 n 个 key 不在 taken 中的合法 CLS（就地扩充 taken）。

    逐个接受制保证前缀稳定：同一 rng 流下，加大 n 不改变前 n' 个的取值——
    小规模试跑与全量跑因此共享 (task, cls) → R' 缓存（铁律 3）。
    """
    out: list[CLS] = []
    while len(out) < n:
        c = space.sample_random(rng)
        k = cls_key(c)
        if k in taken:
            continue
        taken.add(k)
        out.append(c)
    return tuple(out)


def sample_cls_sets(
    space: CLSSpace,
    task_ids: list[str],
    *,
    n_shared: int,
    n_random: int,
    seed: int,
) -> CLSSampling:
    """采样 L 个 shared（全任务共用）+ 每任务 L 个 random CLS。

    随机流隔离：shared 用 default_rng(seed)；第 i 个任务的 random 用
    default_rng([seed, 1+i])（SeedSequence 派生，互相独立）——任一批的大小
    变化不影响其他批的取值。random 批与 shared 去重（同一任务出现重复 CLS
    会造出 R' 自比配对），任务间 random 允许偶然重合（不同任务上是不同样本）。
    """
    shared = _sample_unique(space, np.random.default_rng(seed), n_shared, set())
    shared_keys = {cls_key(c) for c in shared}
    random: dict[str, tuple[CLS, ...]] = {}
    for i, tid in enumerate(task_ids):
        rng = np.random.default_rng([seed, 1 + i])
        random[tid] = _sample_unique(space, rng, n_random, set(shared_keys))
    return CLSSampling(shared=shared, random=random)


# ---------------------------------------------------------------------------
# 任务内 train/eval 划分（铁律 7 分发点：R' 评估协议随任务类型变）
# ---------------------------------------------------------------------------


def split_task_train_eval(
    y: np.ndarray, task_type: str, *, eval_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """把一个切片任务的样本划成 (train, eval) 两侧索引（任务内局部索引，升序）。

    train 侧：cl_trainer 对比预训练 + 下游分类器拟合；eval 侧：R' 评估。
    classification: 按类分层（保证每类两侧都有，SVM 才能训练/评估）；
    forecasting: 应按时间后缀划分防未来信息泄漏——未实现，接入时在此补分支。
    """
    validate_task_type(task_type)
    if task_type == TASK_CLASSIFICATION:
        rng = np.random.default_rng(seed)
        tr_parts, ev_parts = [], []
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            rng.shuffle(idx)
            n_ev = min(max(int(round(len(idx) * eval_ratio)), 1), len(idx) - 1)
            ev_parts.append(idx[:n_ev])
            tr_parts.append(idx[n_ev:])
        return np.sort(np.concatenate(tr_parts)), np.sort(np.concatenate(ev_parts))
    raise NotImplementedError(
        f"task_type={task_type!r} 的任务内 train/eval 划分未实现——forecasting "
        "应按时间后缀划分（防未来信息泄漏），接入时在此补分支"
    )


# ---------------------------------------------------------------------------
# 配对造 label（铁律 8：方向只经 is_better；δ 过滤同分噪声）
# ---------------------------------------------------------------------------


def make_pairs(
    rprime: dict[str, float],
    task_type: str,
    *,
    delta: float,
    origin: Optional[dict[str, str]] = None,
) -> tuple[list[dict], dict]:
    """把一个任务的 {cls_key: R'} 配成二值 label 样本。

    每个无序对 (a, b)（a 在 rprime 中的插入序靠前）：
    - 任一方 R' 非有限（训练失败 NaN）→ 该 CLS 整体剔除；
    - |R'_a - R'_b| < delta → δ 弃样（同分噪声，见模块 docstring）；
    - 双向 is_better 均 False（严格平局，delta=0 时可达）→ 弃样，不赋假方向；
    - 否则 y = is_better(R'_a, R'_b, task_type)。
    origin 给出每个 key 的来源（shared/random）时，配对记 kind 字段
    （"shared-shared"/"random-shared"/"random-random"，供 curriculum 分组）。
    返回 (pairs, stats)；stats 计数各弃样原因，供落盘报告 δ 过滤比例。
    """
    validate_task_type(task_type)
    valid = [k for k, v in rprime.items() if np.isfinite(v)]
    n_failed = len(rprime) - len(valid)

    pairs: list[dict] = []
    n_delta_dropped = n_tie_dropped = 0
    for a, b in combinations(valid, 2):
        ra, rb = rprime[a], rprime[b]
        if abs(ra - rb) < delta:
            n_delta_dropped += 1
            continue
        a_better = is_better(ra, rb, task_type)
        b_better = is_better(rb, ra, task_type)
        if not (a_better or b_better):
            n_tie_dropped += 1
            continue
        rec = {"a": a, "b": b, "y": int(a_better), "r_a": ra, "r_b": rb}
        if origin is not None:
            rec["kind"] = "-".join(sorted((origin[a], origin[b])))
        pairs.append(rec)

    n_candidates = len(valid) * (len(valid) - 1) // 2
    stats = {
        "n_cls": len(rprime),
        "n_failed_cls": n_failed,
        "n_candidates": n_candidates,
        "n_delta_dropped": n_delta_dropped,
        "n_tie_dropped": n_tie_dropped,
        "n_kept": len(pairs),
    }
    return pairs, stats


def delta_decision_curve(
    early: np.ndarray,
    full: np.ndarray,
    task_type: str,
    *,
    deltas: list[float],
) -> list[dict]:
    """δ 标定决策曲线（scripts/calibrate_delta.py 的核心）。

    对每个候选 δ：保留 |ΔR'_early| ≥ δ 且 early 非平局的配对，统计保留比例与
    early/full 方向一致率（方向均经 is_better，铁律 8；full 平局的配对方向
    无定义，不计一致率分母）。δ 应取在同分噪声上界略高处：一致率明显抬升、
    再加大无增益的位置。
    """
    validate_task_type(task_type)
    early, full = np.asarray(early, dtype=float), np.asarray(full, dtype=float)
    idx_pairs = [
        (i, j)
        for i, j in combinations(range(len(early)), 2)
        if np.isfinite(early[i]) and np.isfinite(early[j])
        and np.isfinite(full[i]) and np.isfinite(full[j])
    ]
    rows = []
    for delta in deltas:
        n_kept = n_comparable = n_agree = 0
        for i, j in idx_pairs:
            if abs(early[i] - early[j]) < delta:
                continue
            e_ij = is_better(early[i], early[j], task_type)
            if not (e_ij or is_better(early[j], early[i], task_type)):
                continue  # early 平局：本就造不出 label，不计入保留
            n_kept += 1
            f_ij = is_better(full[i], full[j], task_type)
            if not (f_ij or is_better(full[j], full[i], task_type)):
                continue  # full 平局：方向无定义
            n_comparable += 1
            n_agree += int(e_ij == f_ij)
        rows.append({
            "delta": delta,
            "n_pairs": len(idx_pairs),
            "n_kept": n_kept,
            "kept_frac": n_kept / len(idx_pairs) if idx_pairs else float("nan"),
            "n_agree": n_agree,
            "agreement": n_agree / n_comparable if n_comparable else float("nan"),
        })
    return rows


# ---------------------------------------------------------------------------
# 比较器桥接：R' 缓存 → T-CLSC 引擎数据结构（scripts/3_pretrain_tclsc.py 的
# 可测核心）。方向语义（is_better/make_pairs）留在本模块，comparator 保持
# 任务无关（铁律 7/8）。
# ---------------------------------------------------------------------------


def comparator_records(
    cache: dict[tuple[str, str, int], dict], encoder: CLSEncoder, *, iters: int
) -> list[ClsRecord]:
    """缓存 → ClsRecord 列表：只取指定 iters 口径（铁律 2）的有限 R' 记录。

    失败 CLS（r_prime 为 None/NaN）剔除；vec 用 CLSEncoder 现算（缓存里存的是
    cls 字段字典，向量化规则以 space.yaml 当前定义为准）。
    """
    out: list[ClsRecord] = []
    for (tid, key, it), rec in cache.items():
        if it != iters:
            continue
        r = rec.get("r_prime")
        if r is None or not np.isfinite(r):
            continue
        out.append(ClsRecord(
            task_id=tid, key=key, origin=rec["origin"],
            vec=encoder.encode(CLS(**rec["cls"])), r=float(r),
        ))
    return out


def comparator_eval_pairs(
    cache: dict[tuple[str, str, int], dict],
    encoder: CLSEncoder,
    *,
    task_type: str,
    delta: float,
    iters: int,
) -> list[Pair]:
    """缓存 → 静态双向评估对：逐任务 make_pairs（δ 过滤 + is_better 方向，
    铁律 8）后接反向出样（← random_search.py::generate_task_pairs 双向语义）。

    与训练侧动态配对不同，评估对固定且穷举全部有效无序对——留出任务的
    配对准确率必须在同一集合上可复现比较（含 E 消融三档）。
    """
    recs = comparator_records(cache, encoder, iters=iters)
    by_task: dict[str, list[ClsRecord]] = {}
    for r in recs:
        by_task.setdefault(r.task_id, []).append(r)

    pairs: list[Pair] = []
    for tid, rs in by_task.items():
        rprime = {r.key: r.r for r in rs}
        origin = {r.key: r.origin for r in rs}
        vec = {r.key: r.vec for r in rs}
        raw, _ = make_pairs(rprime, task_type, delta=delta, origin=origin)
        pairs += [Pair(task_id=tid, vec_a=vec[p["a"]], vec_b=vec[p["b"]],
                       y=p["y"], kind=p["kind"]) for p in raw]
    return with_reversed(pairs)


# ---------------------------------------------------------------------------
# 落盘缓存与可复现种子（铁律 3：(task, cls) → R' 必须缓存 + --resume）
# ---------------------------------------------------------------------------


def append_jsonl(path: str | Path, rec: dict) -> None:
    """逐行追加（每完成一条立即落盘，中途挂掉零丢失）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_rprime_cache(path: str | Path) -> dict[tuple[str, str, int], dict]:
    """读 R' 缓存：{(task_id, cls_key, iters): record}。iters 入键防 early/full 混用（铁律 2）。"""
    path = Path(path)
    done: dict[tuple[str, str, int], dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 进程被杀可能留下半行：跳过（该条视作未完成，重跑）
                done[(rec["task_id"], rec["cls_key"], rec["iters"])] = rec
    return done


def run_seed_for(task_id: str, key: str, base: int = 0) -> int:
    """(task, cls) 的确定性运行种子：与执行顺序/续跑无关，同缓存键同权重初始化。"""
    h = hashlib.md5(f"{base}:{task_id}:{key}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)
