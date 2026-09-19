"""WP17-ACTS 公共层：配置 / acts 空间 / 剪枝 / 目录 / 格 / 切片数组 / 标签 / 读数量——83（S1/S2 语料）、84（S3）、85（S4）共用的唯一真源。

协议 results/acts/s1/PROTOCOL.md；真源 configs/acts.yaml；方案 docs/AUTOCTS_STRICT_REPRO_PLAN.md。

铁律落点：
- 铁律 1：档位与合法性只经 CLSSpace（configs/space.yaml）；acts.yaml 只 **追加** 规范化约束（`acts_space`，opt-in）
  与剪枝谓词（`is_pruned`），默认 `CLSSpace.from_yaml()` 逐位不变；
- 铁律 2：iters 由 35 号 `resolve_iters_from_size`（协议 `iters_rule`）算出并显式落盘，截点是同一轨迹；
- 铁律 3：格键 = 35 号 `cache_key`（dataset 位填任务 id），runs.jsonl + resume + verify；
- 铁律 6：本模块不知道靶是谁；切片只吃 y / C / L（`acts_slicer`），`target_sizes_29` 只读 uea30.yaml 里的尺寸做 sanity；
- 铁律 7 / 8：不引入数据集概念到空间 / 比较器；R′ 优劣方向留给 task_types.is_better（S3 配对时用）。
"""
from __future__ import annotations

import hashlib
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.task_slicer import SlicedTask, materialize_task  # noqa: E402
from autocls.labeling import cls_key, run_seed_for, split_task_train_eval  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace, DEFAULT_SPACE_YAML  # noqa: E402

ACTS_YAML = ROOT / "configs" / "acts.yaml"
UEA30_YAML = ROOT / "configs" / "uea30.yaml"
OUT_ROOT = ROOT / "results" / "acts"
STAGE = "s1_labels"
ARM_SHARED, ARM_RANDOM, ARM_ANCHOR = "shared", "random", "anchor"
# 六个增强比率字段名（与 cl_trainer._AUG_FIELDS 同；字段名不是档位，铁律 1 不涉及）
AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop")
# 保真度残差分组：(名, lo, hi]，PROTOCOL §1.5
AUGMAX_GROUPS = (("le0.5", -math.inf, 0.5), ("0.5-0.8", 0.5, 0.8), ("gt0.8", 0.8, math.inf))
SANITY_AXES = ("C", "L", "n", "k")


# ---------------------------------------------------------------- 配置 / 空间
def _deep_merge(base: Mapping, over: Mapping) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def load_acts_config(path: str | Path | None = None, profile: Optional[str] = None) -> dict:
    """acts.yaml；profile（如 "s1p"）= `profiles.<name>` 逐键深合并到基线上（基线不动，S1 可复现），并写 cfg["profile"]。"""
    cfg = yaml.safe_load(Path(path or ACTS_YAML).read_text(encoding="utf-8"))
    if not profile:
        return cfg
    prof = (cfg.get("profiles") or {}).get(profile)
    if prof is None:
        raise KeyError(f"acts.yaml 无 profiles.{profile}")
    merged = _deep_merge({k: v for k, v in cfg.items() if k != "profiles"}, prof)
    merged["profile"] = str(profile)
    return merged


def acts_space(cfg: Optional[Mapping] = None, space_yaml: str | Path = DEFAULT_SPACE_YAML) -> CLSSpace:
    """space.yaml 字段 + space.yaml 约束 + acts.yaml normalize.constraints → 新 CLSSpace（opt-in；默认空间不动）。"""
    cfg = cfg or load_acts_config()
    sc = yaml.safe_load(Path(space_yaml).read_text(encoding="utf-8"))
    sc["constraints"] = list(sc.get("constraints", [])) + list((cfg.get("normalize") or {}).get("constraints") or [])
    return CLSSpace(sc)


# ---------------------------------------------------------------- 剪枝（PROTOCOL §1.2）
def _match(value: Any, target: Any) -> bool:
    if isinstance(value, bool) or isinstance(target, bool):
        return value is target
    if isinstance(value, (int, float)) and isinstance(target, (int, float)):
        return math.isclose(float(value), float(target), rel_tol=1e-9, abs_tol=1e-12)
    return value == target


def prune_rules(cfg: Optional[Mapping] = None) -> list[dict]:
    cfg = cfg or load_acts_config()
    return list((cfg.get("prune") or {}).get("rules") or [])


def is_pruned(cls_cfg: CLS, cfg: Optional[Mapping] = None) -> bool:
    """任一规则的全部条件成立 → 剪掉（S1：τ=100；info_nce ∧ τ=10）。"""
    for rule in prune_rules(cfg):
        if all(_match(getattr(cls_cfg, cond["field"]), cond["equals"]) for cond in rule["all"]):
            return True
    return False


def pruned_share(space: CLSSpace, cfg: Mapping, *, n: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([is_pruned(space.sample_random(rng), cfg) for _ in range(int(n))]))


def normalization_stats(cfg: Mapping, *, n: int = 20000, seed: int = 0) -> dict:
    """均匀采样（默认空间流）下 kernel=0 规范化的改写份额与 distinct key 比值（PROTOCOL §1.3 报数）。"""
    dspace, aspace = CLSSpace.from_yaml(), acts_space(cfg)
    rng = np.random.default_rng(seed)
    raw_keys, can_keys, rewritten = [], [], 0
    for _ in range(int(n)):
        c = dspace.sample_random(rng)
        v = {name: getattr(c, name) for name in aspace.all_field_names}
        v.update(aspace.inactive_overrides(v))
        cc = CLS(**v)
        raw_keys.append(cls_key(c))
        can_keys.append(cls_key(cc))
        rewritten += int(cc != c)
    return {"n": int(n), "rewritten_share": rewritten / n,
            "distinct_ratio_sampled": len(set(can_keys)) / max(1, len(set(raw_keys))),   # n ≪ 空间大小时 ≈ 1，无信息，只留档
            "space_size_ratio": aspace.space_size() / dspace.space_size()}             # 规范形计数 39 / 49（解析值）


# ---------------------------------------------------------------- 目录（PROTOCOL §1.2 / §1.4 锚）
def _sample_catalog(space: CLSSpace, cfg: Mapping, rng: np.random.Generator, n: int, taken: set[str]) -> list[tuple[CLS, str]]:
    out: list[tuple[CLS, str]] = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 200_000:
            raise RuntimeError("采样 200k 次仍凑不齐目录（剪枝 / 去重过严？）")
        c = space.sample_random(rng)
        if not space.validate(c) or is_pruned(c, cfg):
            continue
        k = cls_key(c)
        if k in taken:
            continue
        taken.add(k)
        out.append((c, k))
    return out


def shared_catalog(space: CLSSpace, cfg: Mapping, n: Optional[int] = None) -> list[tuple[CLS, str]]:
    """全部任务共用的 L 条：固定 seed 的均匀采样（= AutoCTS++ sample_arch(use_seed=True)），逐个接受制 → 前缀稳定。"""
    n = int(cfg["candidates"]["n_shared"] if n is None else n)
    rng = np.random.default_rng(int(cfg["candidates"]["shared_seed"]))
    return _sample_catalog(space, cfg, rng, n, set())


def random_catalog(space: CLSSpace, cfg: Mapping, task_id: str, *, exclude_keys: Iterable[str], n: Optional[int] = None) -> list[tuple[CLS, str]]:
    """每任务 L 条：rng 由 (random_seed, task_id 哈希) 派生（他们 seed(None) 不可复现，语义同）；与 shared 及任务内去重。"""
    n = int(cfg["candidates"]["n_random"] if n is None else n)
    rng = np.random.default_rng([int(cfg["candidates"]["random_seed"]), run_seed_for(task_id, "acts_random")])
    return _sample_catalog(space, cfg, rng, n, set(exclude_keys))


def aug_max(cls_cfg: CLS) -> float:
    return float(max(float(getattr(cls_cfg, f)) for f in AUG_FIELDS))


def pick_anchors(shared: Sequence[tuple[CLS, str]], *, n: int = 2, aug_max_le: float = 0.8) -> list[tuple[CLS, str]]:
    """噪声锚 = shared 里按索引顺序前 n 个 aug_max ≤ 阈值的配置（确定性规则，不是剪枝；PROTOCOL §1.4）。"""
    out = [(c, k) for c, k in shared if aug_max(c) <= float(aug_max_le)][: int(n)]
    if len(out) < int(n):
        raise ValueError(f"shared 里 aug_max ≤ {aug_max_le} 的配置不足 {n} 条（只有 {len(out)}）")
    return out


# ---------------------------------------------------------------- 标签（PROTOCOL §1.4）
def best_of_checkpoints(row: Optional[Mapping]) -> Optional[float]:
    """R′ = 三截点 acc 的最大值；error / 缺截点 / 非有限 → None（交给地板）。"""
    if not row or row.get("error") or not row.get("checkpoint_results"):
        return None
    accs = [c.get("acc") for c in row["checkpoint_results"]]
    if any(a is None or not np.isfinite(float(a)) for a in accs):
        return None
    return float(max(float(a) for a in accs))


def majority_rate(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.bincount(y).max() / len(y)) if len(y) else float("nan")


def floor_of(majority: float, probe_accs: Sequence[float]) -> float:
    """失败地板 = max(多数类占比, iters=0 探针均值)。"""
    accs = [float(a) for a in probe_accs if a is not None and np.isfinite(float(a))]
    return float(max(float(majority), float(np.mean(accs)) if accs else float(majority)))


def label_of(best: Optional[float], floor: float) -> tuple[float, bool]:
    """(R′, failed)：None 或 ≤ 地板 → (地板, True)，并列最末；否则 (best, False)。"""
    if best is None or float(best) <= float(floor):
        return float(floor), True
    return float(best), False


# ---------------------------------------------------------------- 格（PROTOCOL §1.4；执行体 = 39.execute）
def estimate_cell_seconds(cell: Mapping, cfg: Mapping) -> float:
    cm = cfg["cost_model"]
    tc = float(cell["T"]) * float(cell["C"])
    pre = float(cell["iters"]) * (float(cm["iter_s_intercept"]) + float(cm["iter_s_per_tc"]) * tc)
    ev = 3.0 * (float(cm["eval_s_intercept"]) + float(cm["eval_s_per_n"]) * float(cell["n_train"]))
    return float(pre + ev)


def build_cells(tasks_meta: Mapping[str, Mapping], shared: Sequence[tuple[CLS, str]], randoms: Mapping[str, Sequence[tuple[CLS, str]]],
                anchors: Sequence[tuple[CLS, str]], protos: Mapping, cfg: Mapping, *, iters_override: Optional[int] = None) -> list[dict]:
    """每任务：shared → random（seed 0）→ 锚 × anchors.seeds；格字典 = 35._cell（dataset 位 = 任务 id）+ 截点 + 任务尺寸。"""
    g35 = importlib.import_module("scripts.35_protocol_alignment")
    g39 = importlib.import_module("scripts.39_sh_deploy")
    pname = str(cfg["protocol"])
    proto = protos[pname]
    anchor_seeds = [int(s) for s in cfg["labels"]["anchors"]["seeds"]]
    cells: list[dict] = []
    for tid, m in tasks_meta.items():
        sizes = {tid: (int(m["n_train"]), int(m["T"]), int(m["C"]))}

        def mk(origin: str, c: CLS, k: str, seed: int) -> dict:
            cell = g35._cell(STAGE, tid, pname, proto, sizes, arm_names=[origin], cls_key_=k, cls=c, seed=seed, iters=iters_override)
            cell["checkpoints"] = list(g39.checkpoints_for(cell["iters"]))
            cell.update({"task_id": tid, "origin": origin, "twin_of": None,
                         "n_train": int(m["n_train"]), "T": int(m["T"]), "C": int(m["C"])})
            g39.check_extension_consistency(cell)
            return cell

        for c, k in shared:
            cells.append(mk(ARM_SHARED, c, k, 0))
        for c, k in randoms[tid]:
            cells.append(mk(ARM_RANDOM, c, k, 0))
        for c, k in anchors:
            for s in anchor_seeds:
                cells.append(mk(ARM_ANCHOR, c, k, s))
    keys = [c["key"] for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("cache_key 撞键")
    return cells


# ---------------------------------------------------------------- 切片数组（PROTOCOL §1.1；契约同 45.make_loader）
def split_seed_for(task_id: str) -> int:
    return run_seed_for(task_id, "train_eval_split")


def task_arrays(task: SlicedTask, X: np.ndarray, y: np.ndarray, *, eval_ratio: float, seed: int) -> dict:
    """物化切片（通道 / 时间窗 / 样本）→ 70/30 分层 → 训练部分按通道标准化后套用到评估部分（= 官方 scaler: train）。"""
    Xt, yt = materialize_task(task, X, y)
    tr, ev = split_task_train_eval(yt, task.task_type, eval_ratio=float(eval_ratio), seed=int(seed))
    Xtr_raw = Xt[tr].astype(np.float64)
    flat = Xtr_raw.reshape(-1, Xtr_raw.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = ((Xtr_raw - mu) / sd).astype(np.float32)
    Xte = ((Xt[ev].astype(np.float64) - mu) / sd).astype(np.float32)
    ytr, yte = yt[tr], yt[ev]
    return {"Xtr": Xtr, "ytr": ytr, "Xte_train": Xte, "Xte_split": Xte, "yte": yte,
            "train_idx": tr, "eval_idx": ev, "majority": majority_rate(yte), "mean": mu, "std": sd}


def split_md5_of(arrays: Mapping) -> str:
    """切片 train/eval 划分的指纹（十进制索引串的 md5，与 numpy 版本 / 字节序无关）：跨机器部署自检用。"""
    s = ",".join(str(int(i)) for i in arrays["train_idx"]) + "|" + ",".join(str(int(i)) for i in arrays["eval_idx"])
    return hashlib.md5(s.encode()).hexdigest()


# ---------------------------------------------------------------- S1′：分层 5 折 CV 切片数组（PROTOCOL s1p §1.2）
def stratified_folds(y: np.ndarray, *, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """任务内固定的分层 K 折（sklearn StratifiedKFold，shuffle + random_state=seed）→ [(train_idx, test_idx)]。"""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    return [(np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)) for tr, te in skf.split(np.zeros(len(y)), y)]


def fold_md5_of(folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> str:
    s = "|".join(",".join(str(int(i)) for i in te) for _tr, te in folds)
    return hashlib.md5(s.encode()).hexdigest()


def cv_fold_seed_for(task_id: str, key: str = "cv_folds") -> int:
    return run_seed_for(task_id, key)


def task_arrays_cv(task: SlicedTask, X: np.ndarray, y: np.ndarray, *, n_folds: int, seed: int) -> dict:
    """整切片：物化 → 按通道在全部样本上标准化（预训练集 = 整个切片）→ 固定分层 K 折。"""
    Xt, yt = materialize_task(task, X, y)
    raw = Xt.astype(np.float64)
    flat = raw.reshape(-1, raw.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = ((raw - mu) / sd).astype(np.float32)
    folds = stratified_folds(yt, n_folds=n_folds, seed=seed)
    return {"X": Xs, "y": yt, "folds": folds, "fold_md5": fold_md5_of(folds), "majority": majority_rate(yt), "mean": mu, "std": sd}


def cv_eval(feats: np.ndarray, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]], grid: str, seed: int) -> dict:
    """每折在训练折特征上跑官方网格（35._svm_variants → fit_svm(grid)）、测试折出 acc / f1；返回折均值与逐折值。"""
    g35 = importlib.import_module("scripts.35_protocol_alignment")
    accs, f1s, cs, cvs = [], [], [], []
    for tr, te in folds:
        v = g35._svm_variants(feats[tr], y[tr], {"cv": feats[te]}, y[te], (grid,), seed)[f"cv/{grid}"]
        accs.append(float(v["acc"])); f1s.append(float(v["f1"])); cs.append(v.get("svm_c")); cvs.append(v.get("svm_cv_score"))
    return {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s)), "fold_acc": accs, "fold_f1": f1s, "svm_c": cs, "svm_cv_score": cvs}


# ---------------------------------------------------------------- S1′：终点标签 / 两种子均值 / 噪声 / 准入（PROTOCOL s1p §1.3–1.4）
def endpoint_of(row: Optional[Mapping]) -> Optional[float]:
    """R′ = 终点截点 acc；error / 缺截点 / 非有限 → None。"""
    if not row or row.get("error") or not row.get("checkpoint_results"):
        return None
    a = row["checkpoint_results"][-1].get("acc")
    return float(a) if a is not None and np.isfinite(float(a)) else None


def label_multi_seed(rows: Sequence[Optional[Mapping]], floor: float) -> tuple[float, bool]:
    """多种子终点均值后套地板；任一种子缺 → 地板（failed）。"""
    vals = [endpoint_of(r) for r in rows]
    if not vals or any(v is None for v in vals):
        return float(floor), True
    return label_of(float(np.mean(vals)), floor)


def noise_ms_of_mean(z0: np.ndarray, z1: np.ndarray) -> float:
    """两种子均值标签的噪声均方 = mean((z0 − z1)²) / 4（单种子噪声 /2 的一半）。"""
    d = (np.asarray(z0, dtype=float) - np.asarray(z1, dtype=float)).ravel()
    d = d[np.isfinite(d)]
    return float(np.mean(d ** 2) / 4.0) if d.size else float("nan")


def admission_rho(s0: np.ndarray, s1: np.ndarray, failed0: np.ndarray, failed1: np.ndarray, *, min_survivors: int) -> dict:
    """源准入量：两种子都存活的配置上 Spearman(seed0, seed1)；存活数 < min_survivors → nan。"""
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    surv = ~np.asarray(failed0, bool) & ~np.asarray(failed1, bool) & np.isfinite(s0) & np.isfinite(s1)
    n = int(surv.sum())
    if n < int(min_survivors) or np.std(s0[surv]) == 0 or np.std(s1[surv]) == 0:
        return {"n_survivors": n, "rho": float("nan")}
    return {"n_survivors": n, "rho": float(spearmanr(s0[surv], s1[surv]).correlation)}


def make_cell(tid: str, m: Mapping, origin: str, c: CLS, k: str, seed: int, protos: Mapping, cfg: Mapping, *,
              iters: Optional[int] = None, fractions: Optional[Sequence[float]] = None, extra: Optional[Mapping] = None) -> dict:
    """单格构造（S1′ 计划用；S1 的 build_cells 保持原样）：35._cell + 截点（可自定义分位）+ 任务尺寸 + eval 模式。"""
    g35 = importlib.import_module("scripts.35_protocol_alignment")
    g39 = importlib.import_module("scripts.39_sh_deploy")
    pname = str(cfg["protocol"])
    sizes = {tid: (int(m["n_train"]), int(m["T"]), int(m["C"]))}
    cell = g35._cell(STAGE, tid, pname, protos[pname], sizes, arm_names=[origin], cls_key_=k, cls=c, seed=int(seed), iters=iters)
    cell["checkpoints"] = list(g39.checkpoints_for(cell["iters"], fractions) if fractions else g39.checkpoints_for(cell["iters"]))
    cell.update({"task_id": tid, "origin": origin, "twin_of": None, "n_train": int(m["n_train"]), "T": int(m["T"]), "C": int(m["C"]),
                 "eval": str((cfg.get("eval") or {}).get("mode", "holdout"))})
    if extra:
        cell.update(dict(extra))
    g39.check_extension_consistency(cell)
    return cell


def load_dataset_full(name: str) -> tuple[np.ndarray, np.ndarray]:
    """源数据集 TRAIN+TEST 合并、**原生长度**（不用注册表 crop_len；部署靶就是原生长度）。注册表分派同 pq_common.REG_OF。"""
    pq = importlib.import_module("scripts.pq_common")
    from autocls.data.source_pool import load_registry, load_source_dataset
    reg = pq.registry_for(name) if name in pq.REG_OF else load_registry()     # PQ 池外的源（如 SCP1）走默认注册表 configs/source_pool.yaml
    return load_source_dataset(name, reg, crop_len=None)


def make_task_loader(tasks: Mapping[str, SlicedTask], dataset_of: Mapping[str, str], load_dataset: Callable[[str], tuple[np.ndarray, np.ndarray]],
                     *, eval_ratio: float) -> Callable:
    """39.execute 的 loader：loader(task_id, har_npz) → arrays。单槽缓存（数据集一份 + 任务一份），换任务即释放。"""
    slot: dict = {}

    def loader(task_id: str, har_npz=None) -> dict:
        if slot.get("task") == task_id:
            return slot["arrays"]
        name = dataset_of[task_id]
        if slot.get("dataset") != name:
            X, y = load_dataset(name)
            slot.clear()
            slot.update({"dataset": name, "X": X, "y": y})
            print(f"[data] {name}: X={X.shape} classes={len(np.unique(y))}", flush=True)
        t = tasks[task_id]
        arrays = task_arrays(t, slot["X"], slot["y"], eval_ratio=eval_ratio, seed=split_seed_for(task_id))
        slot.update({"task": task_id, "arrays": arrays})
        print(f"[data] {task_id}: train={arrays['Xtr'].shape} eval={arrays['Xte_train'].shape} "
              f"classes={len(np.unique(arrays['ytr']))} size={arrays['Xtr'].size} window={t.time_window}", flush=True)
        return arrays

    return loader


# ---------------------------------------------------------------- 读数量（PROTOCOL §1.5 / §2）
def split_half_rho(Z: np.ndarray, *, half_size: int, n_draws: int, seed: int) -> float:
    """随机抽 2·half_size 个任务分两半，两半画像均值的 Spearman，n_draws 次取均值。"""
    Z = np.asarray(Z, dtype=float)
    nt = Z.shape[0]
    if nt < 2 * int(half_size):
        raise ValueError(f"任务数 {nt} < 2 × half_size={half_size}")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_draws)):
        perm = rng.permutation(nt)[: 2 * int(half_size)]
        a, b = Z[perm[: int(half_size)]].mean(0), Z[perm[int(half_size):]].mean(0)
        r = spearmanr(a, b).correlation
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def interaction_share(Z: np.ndarray, *, noise_ms: float) -> dict:
    """二因素（任务 × 配置）交互均方 MS_int = SS_int / ((n_t−1)(n_c−1))；真实交互份额 = (MS_int − MS_noise) / MS_int。"""
    Z = np.asarray(Z, dtype=float)
    nt, nc = Z.shape
    resid = Z - Z.mean(0, keepdims=True) - Z.mean(1, keepdims=True) + Z.mean()
    ss = float((resid ** 2).sum())
    df = max(1, (nt - 1) * (nc - 1))
    ms_int = ss / df
    share = (ms_int - float(noise_ms)) / ms_int if ms_int > 0 else float("nan")
    return {"ms_int": ms_int, "ss_int": ss, "df": df, "ms_noise": float(noise_ms), "real_share": float(share),
            "n_tasks": int(nt), "n_cfg": int(nc)}


def anchor_noise_ms(z0: np.ndarray, z1: np.ndarray) -> float:
    """两个种子的 z 之差 → 单格噪声均方 = mean(d²) / 2。"""
    d = (np.asarray(z0, dtype=float) - np.asarray(z1, dtype=float)).ravel()
    d = d[np.isfinite(d)]
    return float(np.mean(d ** 2) / 2.0) if d.size else float("nan")


def _zs(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def fidelity_metrics(traj: Mapping[str, Mapping[str, tuple[float, float]]], augmax: Mapping[str, float]) -> dict:
    """保真度：逐任务 Spearman(早截点, 终点)、recall@3（早 top-3 ∩ 终 top-10）、残差 z_final − z_early 按 aug_max 分组。"""
    rho: dict[str, float] = {}
    rec: dict[str, float] = {}
    resid: dict[str, dict[str, float]] = {g: {} for g, _lo, _hi in AUGMAX_GROUPS}
    for task, d in traj.items():
        keys = list(d)
        e = np.array([float(d[k][0]) for k in keys])
        f = np.array([float(d[k][1]) for k in keys])
        ok = np.isfinite(e) & np.isfinite(f)
        if int(ok.sum()) < 3:
            continue
        keys = [k for k, o in zip(keys, ok) if o]
        e, f = e[ok], f[ok]
        r = spearmanr(e, f).correlation if (e.std() > 0 and f.std() > 0) else float("nan")
        rho[task] = float(r) if np.isfinite(r) else float("nan")
        top3 = set(np.argsort(-e, kind="stable")[:3].tolist())
        top10 = set(np.argsort(-f, kind="stable")[:10].tolist())
        rec[task] = len(top3 & top10) / 3.0
        res = _zs(f) - _zs(e)
        for g, lo, hi in AUGMAX_GROUPS:
            sel = [i for i, k in enumerate(keys) if k in augmax and lo < float(augmax[k]) <= hi]
            if sel:
                resid[g][task] = float(res[sel].mean())
    rv = np.array([v for v in rho.values() if np.isfinite(v)])
    return {"rho_per_task": rho,
            "rho_median": float(np.median(rv)) if rv.size else float("nan"),
            "rho_q25": float(np.percentile(rv, 25)) if rv.size else float("nan"),
            "rho_q75": float(np.percentile(rv, 75)) if rv.size else float("nan"),
            "recall3_per_task": rec, "recall3_mean": float(np.mean(list(rec.values()))) if rec else float("nan"),
            "residual_by_augmax": resid,
            "residual_by_augmax_mean": {g: (float(np.mean(list(v.values()))) if v else float("nan")) for g, v in resid.items()},
            "n_tasks": len(rho)}


def champion_retention(rows: Iterable[Mapping], cfg: Mapping) -> dict:
    """官方口径目录（stage s1_catalog、seed 0、非 error）每靶冠军是否落在剪后空间（G-A0）。"""
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("stage") != "s1_catalog" or int(r.get("seed", 0)) != 0 or r.get("error") or r.get("acc") is None:
            continue
        by[r["dataset"]].append(r)
    champions: dict[str, dict] = {}
    lost: list[tuple[str, str]] = []
    for d, rs in sorted(by.items()):
        best = max(rs, key=lambda r: float(r["acc"]))
        pr = is_pruned(CLS(**best["cls"]), cfg)
        champions[d] = {"cls_key": best["cls_key"], "acc": float(best["acc"]), "pruned": bool(pr)}
        if pr:
            lost.append((d, best["cls_key"]))
    n = len(by)
    kept = n - len(lost)
    return {"n": n, "kept": kept, "rate": (kept / n if n else float("nan")), "lost": lost, "champions": champions}


def verdict_s1(gates: Mapping[str, bool]) -> str:
    if not gates.get("G-A0", False):
        return "ACTS-S1-PRUNE"
    if not gates.get("G-A1", False):
        return "ACTS-S1-SAME"
    if not gates.get("G-A2", False):
        return "ACTS-S1-NOISE"
    return "ACTS-S1-GO"


# ---------------------------------------------------------------- 切片 sanity（PROTOCOL §1.7-5）
def target_sizes_29(path: str | Path = UEA30_YAML) -> dict[str, dict]:
    """29 个官方 UEA 靶的 (C, L, n_train, 类别数)：uea30.yaml 里 run: true 的 23 个 + class LODO 的 6 个（HAR / InsectWingbeat 不入）。"""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, d in cfg["datasets"].items():
        if name == "HAR" or not (d.get("run") or d.get("class") == "LODO"):
            continue
        out[name] = {"C": int(d["C"]), "L": int(d["L"]), "n": int(d["n_train"]), "k": int(d["n_classes"]), "class": str(d.get("class"))}
    return out


def coverage_table(subsets: Sequence[Mapping], targets: Mapping[str, Mapping]) -> dict:
    """每靶四轴是否落在子集云 [min, max] 内 + 到最近子集的 log 距离；汇总云外靶数。"""
    S = np.array([[float(s[a]) for a in SANITY_AXES] for s in subsets], dtype=float)
    lo, hi = S.min(0), S.max(0)
    logS = np.log10(S)
    per: dict[str, dict] = {}
    outside_by_axis = {a: 0 for a in SANITY_AXES}
    outside_any = 0
    for name, t in targets.items():
        v = np.array([float(t[a]) for a in SANITY_AXES])
        inside = {a: bool(lo[i] <= v[i] <= hi[i]) for i, a in enumerate(SANITY_AXES)}
        d = float(np.min(np.sqrt(((logS - np.log10(v)) ** 2).sum(1))))
        per[name] = {"inside": inside, "nearest_log_dist": d, "values": {a: float(t[a]) for a in SANITY_AXES}}
        if not all(inside.values()):
            outside_any += 1
        for a, ok in inside.items():
            outside_by_axis[a] += int(not ok)
    return {"per_target": per,
            "cloud_range": {a: [float(lo[i]), float(hi[i])] for i, a in enumerate(SANITY_AXES)},
            "summary": {"n_targets": len(targets), "n_subsets": int(S.shape[0]), "outside_any": outside_any, "outside_by_axis": outside_by_axis}}


__all__ = [
    "ACTS_YAML", "OUT_ROOT", "STAGE", "ARM_SHARED", "ARM_RANDOM", "ARM_ANCHOR", "AUG_FIELDS", "AUGMAX_GROUPS", "SANITY_AXES",
    "load_acts_config", "acts_space", "prune_rules", "is_pruned", "pruned_share", "normalization_stats",
    "shared_catalog", "random_catalog", "aug_max", "pick_anchors",
    "best_of_checkpoints", "majority_rate", "floor_of", "label_of",
    "estimate_cell_seconds", "build_cells", "split_seed_for", "task_arrays", "split_md5_of", "load_dataset_full", "make_task_loader",
    "split_half_rho", "interaction_share", "anchor_noise_ms", "fidelity_metrics", "champion_retention", "verdict_s1",
    "target_sizes_29", "coverage_table",
    "stratified_folds", "fold_md5_of", "cv_fold_seed_for", "task_arrays_cv", "cv_eval", "endpoint_of", "label_multi_seed",
    "noise_ms_of_mean", "admission_rho", "make_cell",
]
