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


def load_probe_cls(
    catalog_path: str | Path, n_cls: int
) -> tuple[list[tuple[CLS, str]], dict[str, int]]:
    """探针 CLS 目录前 n_cls 个 → [(CLS, cls_key)] + {cls_key: har_rank}。

    step0 预探针与 2b 造 label 的 shared CLS 共用此唯一入口（路线 A 拍板：shared 10
    = 探针目录前 10）——step0 的 canonical R'（键 = 探针 cls_key + early_iters +
    canonical 种子）在阶段2 因此逐字段命中复用。
    """
    cat = json.loads(Path(catalog_path).read_text(encoding="utf-8"))["cls"][:n_cls]
    cls_list = [(CLS(**cd["cls"]), cd["cls_key"]) for cd in cat]
    # 解耦实验的随机目录（scripts/sample_random_cls_catalog.py）刻意不带 har_rank
    # ——随机样本没有排名，带了就又给"按好坏取子集"开口子。只收录有该字段的条目。
    har_rank = {cd["cls_key"]: cd["har_rank"] for cd in cat if "har_rank" in cd}
    return cls_list, har_rank


def sample_random_cls(
    space: CLSSpace, task_id: str, *, n: int, seed: int, exclude_keys: set[str]
) -> tuple[CLS, ...]:
    """路线 A 每任务 random CLS：rng 由 (seed, task_id 哈希) 派生。

    与路线 B sample_cls_sets 的按池内位置派生不同——跨数据集源池成员随 step0
    过闸浮动（踢/留/reserve 回填），按位置派生会让既有 (task, cls) → R' 缓存整体
    失效（铁律 3）；按 task_id 派生则与池组成无关。与 shared（探针 CLS）去重；
    逐个接受制保证前缀稳定（加大 n 不改变前 n' 个取值，小规模试跑与全量共享缓存）。
    """
    rng = np.random.default_rng([seed, run_seed_for(task_id, "random_cls")])
    return _sample_unique(space, rng, n, set(exclude_keys))


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


def split_train_eval_by_group(
    y: np.ndarray, groups: np.ndarray, *, eval_ratio: float, seed: int,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """按分组（如受试者）划 (train, eval)：**整组进一侧**，杜绝组内孪生泄漏。

    动机（目标池 zero-shot 评估 2026-08-11 发现，出路 A 拍板 2026-08-12）：
    UCI-HAR 窗口 50% 重叠 + 分层随机切分让 eval 侧每个窗口在 train 侧有同受试者
    孪生近邻——随机初始化基线被抬 +0.070 直接贴顶，R' 测量饱和。组切分与官方
    test（跨受试者）的划分语义对齐。

    算法（数据前写死）：组随机排列后依序累入 eval，**首次达到目标样本数即停**
    （过冲上界 = 最大组的大小，保证 train 侧规模可控——steps/epoch 换算依赖）；
    校验每类两侧都有 ≥1 样本（SVM 训练/评估前提），不满足换派生 seed 重试
    max_attempts 次后 ValueError fail-fast（如某类只存在于单一组时无解）。
    返回升序局部索引，接口与 split_task_train_eval 对齐。
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    assert len(y) == len(groups), f"y {len(y)} 与 groups {len(groups)} 长度不一致"
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError(f"组数 {len(uniq)} < 2，无法组切分")
    target = eval_ratio * len(y)
    classes = set(np.unique(y).tolist())
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        perm = rng.permutation(uniq)
        ev_groups, n_ev = [], 0
        for g in perm:
            ev_groups.append(g)
            n_ev += int(np.sum(groups == g))
            if n_ev >= target:
                break
        ev_mask = np.isin(groups, ev_groups)
        if ev_mask.all():
            continue  # 组太少/太大吞掉全部样本：换排列
        tr_idx = np.where(~ev_mask)[0]
        ev_idx = np.where(ev_mask)[0]
        if set(y[tr_idx].tolist()) == classes and set(y[ev_idx].tolist()) == classes:
            return np.sort(tr_idx), np.sort(ev_idx)
    raise ValueError(
        f"组切分 {max_attempts} 次尝试均无法保证每类两侧覆盖"
        "（可能某类只存在于单一组）——数据不支持该协议，fail-fast"
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


def shared_seed_for(task_id: str, rep: int) -> int:
    """共享种子（common random numbers）：同任务全 CLS 共用的第 rep 个权重种子。

    与生产口径 run_seed_for(task_id, cls_key) 的关键差异：**不含 cls_key**——同一
    任务上所有 CLS 从同一初始化出发做配对比较，seed 主效应 g 在 ΔR' 中完全抵消，
    只剩 cls×seed 交互 ε（results/shared_seed_probe/PROTOCOL.md 的方差模型）。
    派生串带 "sharedseed:" 前缀，与 canonical（cls_key）/噪声探针（"{key}:noise{rep}"）
    两条既有种子流天然不相交，历史缓存零污染。"""
    return run_seed_for(task_id, f"sharedseed:{rep}")


def mark_inflight(path: str | Path, task_id: str, cls_key_: str, iters: int) -> None:
    """开跑一条 R' 前打点（Part A-3 看门狗协议）：进程在这条上被杀时，看门狗读此
    文件累计 (task, cls) 崩溃次数，≥FAIL_LIMIT 写永久隔离记录防死循环。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"{task_id}\t{cls_key_}\t{iters}", encoding="utf-8")


def clear_inflight(path: str | Path) -> None:
    """该条 R' 已落盘缓存 → 清空打点（正常完成不算崩溃）。"""
    p = Path(path)
    if p.exists():
        p.write_text("", encoding="utf-8")


# ---------------------------------------------------------------------------
# 单条 (task, cls) → R' 的共享评估助手（scripts/2_generate_labels.py 与
# scripts/step0_snr_gate.py 共用，避免第 3 份 fork）。weight_seed/split_seed 显式
# 传入：全量造 label 用 (run_seed_for(tid,key), run_seed_for(tid,'train_eval_split'))
# 的 canonical 口径；step0 噪声测量则每个种子派生不同的 weight/split（探针协议）。
# torch/encoder 惰性导入——保持 labeling 纯函数（make_pairs 等）可无 GPU 依赖测试。
# ---------------------------------------------------------------------------


def device_label(device: Optional[str]) -> str:
    """实际执行设备的硬件标识（如 "NVIDIA GeForce RTX 4060 Laptop GPU"）。

    R' 缓存记录此前没有任何硬件字段，跨机器的 seconds/R' 锚点无法区分——
    full-iters 判别实验的 5.6× 估时事故有一半来自把 5090 的耗时锚点套在 4060 上。
    设备解析与 pretrain_encoder 同规则（None → 有 CUDA 用 CUDA）。
    """
    import platform

    import torch

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(dev.index if dev.index is not None else 0)
    return f"cpu:{platform.processor() or platform.machine()}"


def evaluate_rprime(
    task,
    cls_cfg: CLS,
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
    split_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    use_swa: bool = False,
) -> dict:
    """对 (task, cls) 跑 iters 对比预训练 + 任务内分层 train/eval 下游评估得 R'。

    use_swa（默认关）：TS2Vec 官方语义的等权权重平均（见 pretrain_encoder 文档），
    评估用平均权重。状态记入返回记录的 "swa" 字段——调用方必须将其纳入缓存键，
    与历史（无 SWA）记录隔离。

    split_indices: 显式 (tr_idx, ev_idx)（任务内局部索引）——给出时绕过内部
    分层切分（出路 A 跨受试者协议：调用方用 split_train_eval_by_group 产出组
    切分注入；split_seed 仍原样落盘作切分来源标识）。None = 原分层切分。

    通道数按任务适配（build_encoder(input_dims=|channels|)）；task_type 分发评估器与
    划分方式（铁律 7）。训练发散/异常记 r_prime=NaN + error，不中断（对齐既有
    run_one 语义）。返回逐条缓存所需字段（r_prime/n_train/n_eval/seconds/error 等）。

    **例外：显存/内存不足抛 RprimeOOM，绝不记 NaN**（autocls/oom_guard.py）。NaN 是
    "这条 CLS 训不出好表征"的结论、会被写进缓存且 --resume 不再重试；OOM 只是这次
    没跑成，记成结论 = 静默永久丢 label。调用方须走 oom_guard.guarded_run。
    """
    import time

    import torch

    from .oom_guard import RprimeOOM, is_oom_error

    from .contrastive.cl_trainer import pretrain_encoder
    from .data.task_slicer import materialize_task
    from .downstream import resolve_evaluator
    from .encoders.registry import build_encoder

    Xt, yt = materialize_task(task, X, y)
    if split_indices is not None:
        tr_idx, ev_idx = split_indices
    else:
        tr_idx, ev_idx = split_task_train_eval(
            yt, task.task_type, eval_ratio=eval_ratio, seed=split_seed,
        )
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)  # 权重初始化可复现
        encoder = build_encoder(input_dims=Xt.shape[-1])
        pretrain_encoder(
            encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
            batch_size=batch_size, lr=lr, device=device, seed=weight_seed,
            use_swa=use_swa,
        )
        evaluator = resolve_evaluator(task.task_type)  # 铁律 7 分发点
        metrics = evaluator(
            encoder, Xt[tr_idx], yt[tr_idx], Xt[ev_idx], yt[ev_idx],
            device=device, seed=weight_seed,
        )
        rec.update(metrics)
        rec["r_prime"] = float(metrics["r_prime"])
    except Exception as e:  # 训练发散等视作该 CLS 的坏结果，不中断实验
        if is_oom_error(e):
            # 显存/内存不足：与这条 CLS 的好坏无关，不得落盘成结论。抛出前尽量把本
            # 进程占的显存还回去，好让同卡其他 worker 与串行重试有余量。
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 —— 无 CUDA 或已崩，清理失败不掩盖 OOM
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["r_prime"] = float("nan")
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec.update({
        "n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)),
        "weight_seed": int(weight_seed), "split_seed": int(split_seed),
        "device_name": device_label(device), "swa": bool(use_swa),
    })
    return rec
