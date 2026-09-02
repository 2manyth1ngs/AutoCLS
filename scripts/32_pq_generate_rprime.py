"""MX-3-PQ 真值网格：多截点 R'（60/120/290）+ 噪声探针 + δ 标定 + 收官校验。

协议：.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/
interfaces.md「Budgets / ids」「Task 1 → Task 5」「Task 5（脚本 32）」。
公共装载/切片/CLS 目录/种子/δ 公式一律走 `scripts/pq_common.py`（唯一真源，Task 4）。

网格：8 数据集 × 前 6 切片（SCP2 只有 3，共 45 切片）× 20 CLS（10 shared + 10 random）
+ 每切片 1 个 GGS 锚 → 每条**一次预训练轨迹**在 60/120/290 三个 checkpoint 上各评一次
R'（`autocls.labeling.evaluate_rprime_trajectory`，Task 1）。主缓存 45×20×3=2700 行
+ GGS 45×3=135 行；噪声探针 8×2 切片×10 shared×2 rep×3 = 960 行。

铁律落点：
- 铁律 2：预算（iters）显式进缓存键 (task_id, cls_key, iters)，early/full 口径不动；
  轨迹的 `checkpoint_iters` 由 evaluate_rprime_trajectory 署名、本脚本原样带上
  （guarded_run_multi 会核对，不符即炸）；
- 铁律 3：所有 (task, cls, budget) → R' 落盘 + `--resume` 贯穿；一条轨迹的 3 行由
  `guarded_run_multi` 原子落盘（半条轨迹会让两条不同权重轨迹共享一个 trajectory_id）；
- 铁律 6：切片一律从源池/候选池注册表取（pq_common.load_pq_datasets 自带目标池自检）；
- 铁律 8：本脚本只造 R'，"谁更优"的方向语义留在 task_types（33 号配对时才用）。

**--resume 的四态**（interfaces.md「Task 1 → Task 5」）：
- done：3 个预算齐全 → 跳过；
- terminal：任一预算的行 `r_prime=null` 且 `error` 有值 → **整条轨迹终结，绝不重试**
  （看门狗把毒条目隔离时只在 max budget 留一条 null 行，重试会无限循环）；
- partial：有行但不齐且无 error → `--verify` 报出、`--fill --redo-partial` 才重跑；
- missing：一行没有 → 跑。

用法（各阶段幂等，顺序：tasks → fill/ggs/noise-probe → calibrate → verify →
reproduce-check）：
    python scripts/32_pq_generate_rprime.py --tasks
    python scripts/32_pq_generate_rprime.py --fill --ggs --noise-probe --resume
    python scripts/32_pq_generate_rprime.py --calibrate --apply-delta
    python scripts/32_pq_generate_rprime.py --verify
    python scripts/32_pq_generate_rprime.py --reproduce-check
    python scripts/32_pq_generate_rprime.py --smoke --outdir results/mx3_pq_smoke

退出码：1 = --verify 判红；3 = inflight 互锁；4 = --apply-delta 覆盖度不足；
5 = 复现校验超差（PROTOCOL §8，除非 --allow-reproduction-breach）。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402

from autocls.labeling import evaluate_rprime_trajectory  # noqa: E402
from autocls.oom_guard import (  # noqa: E402
    ON_OOM_RECORD, ON_OOM_SKIP, guarded_run_multi,
)
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

ORIGIN_GGS = "ggs"                    # 主缓存里 GGS 锚的 origin（shared/random 之外）
DEFAULT_OUTDIR = pq.OUT
# smoke 默认落在子目录：33 号的 smoke 会向 results/mx3_pq_smoke **捧造一份合成
# rprime_cache.jsonl**（device_name="smoke"，18 切片 × 8 CLS × 60/120/290 = 432 行），
# 与本脚本跑出来的真行 schema 完全一样，混在同一份文件里只能靠 device_name 分辨
# （fix 轮 2 实测，见报告）。链路要串在一起时显式传 --outdir 即可。
SMOKE_OUTDIR = pq.OUT.with_name("mx3_pq_smoke") / "gen32"
SMOKE_DATASETS = ("Handwriting", "UniMiB-SHAR")
SMOKE_CHECKPOINTS = (4, 8, 12)

# 定义漂移参照（切片 classes 必须与既有 canonical R' 缓存的来源逐位一致，否则
# "同一个 Handwriting_00" 在两处指的是不同任务，缓存复用就是静默串味）。
STAGE1_META = ROOT / ("results/stage1_rerun_20260804/results/step0_snr_gate/"
                      "tasks_meta.d")
STAGE1_NOISE = ROOT / ("results/stage1_rerun_20260804/results/step0_snr_gate/"
                       "noise_cache.jsonl")
WORKPOINT_TASKS = ROOT / "results" / "labels_workpoint" / "tasks.json"
DRIFT_SLICES = ("_00", "_01")         # 参照只覆盖前两个切片

# 轨迹四态
DONE, TERMINAL, PARTIAL, MISSING = "done", "terminal", "partial", "missing"

# stage1 k3 @290 一致性判读带（比值落在带外只告警，不拦路）
RATIO_BAND = (0.7, 1.4)

# ---- 复现校验（PROTOCOL §8「复现校验（预注册容差）」，阈值先于数据写死） ----
# 本次 @290 的 canonical 行 vs 旧缓存里同 (task_id, cls_key, weight_seed) 的行。
# 旧缓存的 @290 是**同一条轨迹定义**（同切片、同 CLS、同权重种子），只是跑在别的
# 机器/别的批次；对不上就说明这一轮的 R' 与历史数字不是一个口径，晋级/净空等
# 既有结论全部失效——§8 点名"不过 → 停下诊断再进比较器阶段"。
REPRO_BUDGET = 290
REPRO_SOURCES = (
    # 5090 对 5090（GGS 45 格、Cricket/Ethanol/HMD 02–05 × 10、SCP2_02 × 10）
    ("source_pool_v2", ROOT / "results" / "source_pool_v2" / "rprime_cache.jsonl"),
    # 混机（HW/PS/UWave/UM 的 4060 与 5090 各半，按 device_name 分桶）
    ("wide_panel", ROOT / "results" / "wide_panel" / "rprime_cache.jsonl"),
)
# 桶 → (容差, 命中率下限, 配对中位 |Δ| 上限 or None)
REPRO_TOL = {
    "same_5090": {"tol": 0.02, "min_within": 0.90, "max_median": 0.01},
    "mixed": {"tol": 0.03, "min_within": 0.85, "max_median": None},
}


class TaskMeta(NamedTuple):
    """行合成只需要的任务字段（SlicedTask 的子集；单测可直接构造）。"""

    task_id: str
    tag: str
    task_type: str
    classes: tuple


class Cell(NamedTuple):
    """网格的一格：(切片, 来源, CLS, cls_key)。一格 = 一条轨迹 = len(budgets) 行。"""

    slice_id: str
    origin: str
    cls: CLS
    cls_key: str


class NoiseCell(NamedTuple):
    """噪声探针的一格：主网格的一格 + 第 rep 个额外 weight seed。"""

    slice_id: str
    origin: str
    cls: CLS
    cls_key: str
    rep: int


class Job(NamedTuple):
    """一条待跑轨迹的全部输入（fill / ggs / noise-probe 共用同一个执行体）。"""

    dataset: str
    task: object
    X: np.ndarray
    y: np.ndarray
    origin: str
    cls: CLS
    cls_key: str
    weight_seed: int
    split_seed: int
    seed: int          # 缓存行的 seed 字段：canonical=0，噪声探针=rep


# ---------------------------------------------------------------- 小工具
def read_rows(path: str | Path) -> list[dict]:
    """读 jsonl（不存在 → 空表；pq.read_jsonl 对缺文件抛错，分阶段产物需先判存在）。"""
    p = Path(path)
    return pq.read_jsonl(p) if p.exists() else []


def cache_index(rows: Iterable[dict]) -> dict[tuple[str, str, int], dict]:
    """行表 → {(task_id, cls_key, iters): 行}（同键后行覆盖前行，同 load_rprime_cache）。"""
    return {(r["task_id"], r["cls_key"], int(r["iters"])): r for r in rows}


def index_by_seed(rows: Iterable[dict]) -> dict[int, dict]:
    """噪声探针专用：{seed: 该 seed 的 cache_index}。

    噪声行的键必须带 seed——同一 (切片, CLS, 预算) 的 rep1/rep2 是**两条独立轨迹**，
    塞进同一个 (task_id, cls_key, iters) 索引会互相覆盖，于是 rep2 落盘后 rep1
    看起来也"已完成"（--resume 静默少跑一半重复对，δ 直接标偏）。
    """
    out: dict[int, dict] = defaultdict(dict)
    for r in rows:
        out[int(r.get("seed", 0))][
            (r["task_id"], r["cls_key"], int(r["iters"]))] = r
    return out


def _p95(values: Sequence[float]) -> float:
    """P95 索引规则：与 2c calibrate / 7_swa `_p95` / pq_common.calibrate_delta 同式。

    `sorted[min(n-1, round(0.95*(n-1)))]`——**不是** numpy 的插值分位数。此处只为
    把 P95 本身写进报表；δ 仍由 pq_common.calibrate_delta 算（公式唯一真源）。
    """
    ds = sorted(abs(float(v)) for v in values)
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


# ---------------------------------------------------------------- 行合成
def build_rows(traj: dict[int, dict], *, dataset: str, task, cls_key: str,
               origin: str, cls_cfg: CLS, weight_seed: int, seed: int = 0) -> list[dict]:
    """{iters: rec} → 缓存行列表（按 iters 升序）。

    `**rec` 原样带出 evaluate_rprime_trajectory 的全部字段，其中 `checkpoint_iters`
    **必须逐字保留**：guarded_run_multi 用它核对"这行属于哪条轨迹"，改写等于把
    不一致抹平成谎言（Task 1 报告修复轮 Finding 2）。`trajectory_id` 让同一条轨迹的
    N 行可追溯到同一次预训练——budget 间比较的前提。
    """
    return [make_row_for(b, traj[b], dataset=dataset, task=task, cls_key=cls_key,
                         origin=origin, cls_cfg=cls_cfg, weight_seed=weight_seed,
                         seed=seed)
            for b in sorted(traj)]


def make_row_for(iters: int, rec: dict, *, dataset: str, task, cls_key: str,
                 origin: str, cls_cfg: CLS, weight_seed: int, seed: int = 0) -> dict:
    """单截点行（guarded_run_multi 的 make_row 回调体；2c make_row 同风格）。"""
    rp = rec.get("r_prime", float("nan"))
    finite = rp is not None and np.isfinite(rp)
    return {"dataset": dataset, "task_id": task.task_id, "tag": task.tag,
            "task_type": task.task_type, "classes": list(task.classes),
            "cls_key": cls_key, "origin": origin, "seed": int(seed),
            "iters": int(iters),
            "trajectory_id": f"{task.task_id}:{cls_key}:{int(weight_seed)}",
            **rec,
            "r_prime": float(rp) if finite else None,
            "cls": dataclasses.asdict(cls_cfg)}


def row_builder(*, dataset: str, task, cls_key: str, origin: str, cls_cfg: CLS,
                weight_seed: int, seed: int = 0) -> Callable[[int, dict], dict]:
    def _make(iters: int, rec: dict) -> dict:
        return make_row_for(iters, rec, dataset=dataset, task=task, cls_key=cls_key,
                            origin=origin, cls_cfg=cls_cfg, weight_seed=weight_seed,
                            seed=seed)

    return _make


# ---------------------------------------------------------------- 四态 / resume
def trajectory_state(index: dict, slice_id: str, cls_key: str,
                     budgets: Sequence[int]) -> str:
    """(切片, CLS) 在缓存里的状态：done / terminal / partial / missing。

    terminal 优先于 partial：`r_prime=null` 且 `error` 有值的行是**结论**
    （训练发散 = 这条 CLS 的属性；看门狗隔离 = 这条轨迹反复崩），重试只会
    无限循环——interfaces.md 点名要求当作整条轨迹终结。
    """
    rows = [index.get((slice_id, cls_key, int(b))) for b in budgets]
    present = [r for r in rows if r is not None]
    if not present:
        return MISSING
    if any(r.get("r_prime") is None and r.get("error") for r in present):
        return TERMINAL
    return DONE if len(present) == len(tuple(budgets)) else PARTIAL


def pending_cells(cells: Sequence, index, budgets: Sequence[int], *,
                  resume: bool, redo_partial: bool,
                  retry_terminal: bool = False) -> list:
    """待跑格（不 --resume 时全跑：与 2b/2c 同语义，pod 上一律带 --resume）。

    `index` 是 cache_index 的结果；噪声探针传 `index_by_seed` 的结果（按 rep 分表，
    本函数用 cell.rep 取对应那张）。terminal 格默认**永不重试**；
    `retry_terminal=True` 是操作方显式要求的修复动作（先剔旧行再重跑）。
    """
    if not resume:
        return list(cells)
    out = []
    for c in cells:
        st = trajectory_state(_index_for(index, c), c.slice_id, c.cls_key, budgets)
        if (st == MISSING or (st == PARTIAL and redo_partial)
                or (st == TERMINAL and retry_terminal)):
            out.append(c)
    return out


def _index_for(index, cell) -> dict:
    rep = getattr(cell, "rep", None)
    return index[rep] if rep is not None else index


def cell_states(cells: Sequence, index, budgets: Sequence[int]) -> dict[str, list]:
    """按状态分组（verify / fill 打印共用；index 语义同 pending_cells）。"""
    out: dict[str, list] = defaultdict(list)
    for c in cells:
        st = trajectory_state(_index_for(index, c), c.slice_id, c.cls_key, budgets)
        out[st].append(c)
    return out


# ---------------------------------------------------------------- 网格计划
def build_plan(names: Sequence[str] = pq.PQ_DATASETS, *, space: CLSSpace | None = None,
               n_slices: int | None = None, n_cls: int | dict[str, int] | None = None,
               include_ggs: bool = True) -> list[Cell]:
    """全网格计划（**零数据加载**：只读注册表 + CLSSpace，verify/计数可离线用）。

    每切片 = pq.catalog_for_slice 的 20 个（10 shared 在前、10 random 在后）+ 1 个
    GGS 锚（origin="ggs"）。`n_slices`/`n_cls` 只给 smoke 用：取每数据集前 n 个切片、
    **每个 origin 各取前 n 个** CLS（n_cls=1 → 1 shared + 1 random + GGS）。

    注意 `n_cls` 是**按 origin 分组**截的，不是整个目录取前 n：后者会让 smoke
    只有 shared，于是 33 号的 wo_shared（只用 random）与 soft 两臂拿不到任何材料，
    32→33→34 的链路根本跑不通（fix 轮 2）。两组都是"前缀截断"：shared 来自
    探针目录前 10 的前 n，random 来自同一条 seed=42 / task_id 派生的采样流（逐个
    接受制，加大 n 不改变前 n′ 个取值）——故 smoke 的格子与全量网格**逐键同缓存**。

    `n_cls` 可以是整数（每个 origin 同一个上限），也可以是
    `{origin: n}`（各 origin 不同上限；smoke 用 `{shared: 2, random: 1}`——两个
    shared 才能给 33 号课程期凑出 shared–shared 配对，fix 轮 3）。dict 里
    没列的 origin 上限为 0（整组排除）。
    """
    space = space or CLSSpace.from_yaml()
    limits: dict[str, int] | None = None
    if n_cls is not None:
        limits = ({o: int(n_cls) for o in (pq.ORIGIN_SHARED, pq.ORIGIN_RANDOM)}
                  if isinstance(n_cls, int) else {k: int(v) for k, v in n_cls.items()})
    shared = None
    ggs = pq.ggs_cls() if include_ggs else None
    ggs_key = pq.cls_key(ggs) if include_ggs else None
    plan: list[Cell] = []
    for name in names:
        slices = pq.pq_slice_ids(name)
        if n_slices is not None:
            slices = slices[:n_slices]
        for sid in slices:
            entries = pq.catalog_for_slice(space, sid, shared)
            if shared is None:       # 探针目录只读一次（10 个 shared 全网格共用）
                shared = [(c, k) for org, c, k in entries if org == pq.ORIGIN_SHARED]
            if limits is not None:
                taken: dict[str, int] = defaultdict(int)
                kept = []
                for org, c, k in entries:
                    if taken[org] < limits.get(org, 0):
                        taken[org] += 1
                        kept.append((org, c, k))
                entries = kept
            plan += [Cell(sid, org, c, k) for org, c, k in entries]
            if include_ggs:
                plan.append(Cell(sid, ORIGIN_GGS, ggs, ggs_key))
    # 缓存键是 (task_id, cls_key, iters)——同切片撞键（例如 GGS 恰好被 random 抽中）
    # 会让两个 origin 写同一条键，--verify 的"零重复键"随即变成假红。实测全网格零撞。
    seen_cells = [(c.slice_id, c.cls_key) for c in plan]
    if len(seen_cells) != len(set(seen_cells)):
        dup = sorted({x for x in seen_cells if seen_cells.count(x) > 1})
        raise ValueError(f"网格内 (切片, cls_key) 撞键 {dup}——缓存键会重叠")
    return plan


def noise_plan(plan: Sequence[Cell], *, n_slices: int, reps: Sequence[int],
               origins: Sequence[str] = (pq.ORIGIN_SHARED,)) -> list[NoiseCell]:
    """噪声探针格：每数据集前 n_slices 个切片 × 指定 origin 的 CLS × reps。"""
    keep: dict[str, list[str]] = defaultdict(list)
    for c in plan:
        ds = pq.dataset_of(c.slice_id)
        if c.slice_id not in keep[ds]:
            keep[ds].append(c.slice_id)
    wanted = {s for sl in keep.values() for s in sl[:n_slices]}
    return [NoiseCell(c.slice_id, c.origin, c.cls, c.cls_key, int(rep))
            for c in plan if c.slice_id in wanted and c.origin in tuple(origins)
            for rep in reps]


def slice_index(datasets: dict) -> dict[str, tuple]:
    """{切片 id: (数据集名, SlicedTask, X, y)}。"""
    return {t.task_id: (ds.name, t, ds.X, ds.y)
            for ds in datasets.values() for t in ds.slices}


# ---------------------------------------------------------------- 定义漂移闸
def _reference_classes(dataset: str, *, meta_dir: Path,
                       workpoint: Path) -> Optional[dict[str, tuple]]:
    """既有落盘里该数据集前两个切片的 classes（无参照返回 None）。"""
    if dataset == "UniMiB-SHAR":
        if not Path(workpoint).exists():
            return None
        raw = json.loads(Path(workpoint).read_text(encoding="utf-8"))
        tasks = raw.get("source_tasks", [])
        return {t["task_id"]: tuple(t["classes"]) for t in tasks
                if pq.dataset_of(t["task_id"]) == dataset}
    path = Path(meta_dir) / f"{dataset}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {t["task_id"]: tuple(t["classes"]) for t in raw.get("tasks", [])}


def check_definition_drift(classes_by_slice: dict[str, tuple], dataset: str, *,
                           meta_dir: Path = STAGE1_META,
                           workpoint: Path = WORKPOINT_TASKS) -> list[tuple]:
    """本次切出的 classes vs 既有落盘（阶段1 / 工作点）。返回不符的切片清单。

    只对 `_00/_01` 对账（参照只覆盖这两个）。**缺参照只告警**（新数据集、或该池
    从未跑过阶段1）；**对不上直接判死**——切片定义漂了却复用同一个 task_id，
    既有 canonical R' 缓存会静默串味（铁律 3 的前提是键指的是同一个任务）。
    """
    ref = _reference_classes(dataset, meta_dir=meta_dir, workpoint=workpoint)
    if not ref:
        print(f"⚠️ [{dataset}] 无参照定义（阶段1/工作点落盘未找到）→ 跳过漂移对账",
              flush=True)
        return []
    bad = []
    for sid, classes in sorted(classes_by_slice.items()):
        if not sid.endswith(DRIFT_SLICES) or sid not in ref:
            continue
        if tuple(classes) != tuple(ref[sid]):
            bad.append((sid, tuple(classes), tuple(ref[sid])))
    if not bad:
        n = sum(1 for s in classes_by_slice if s.endswith(DRIFT_SLICES) and s in ref)
        print(f"[{dataset}] 定义对账通过（{n} 个切片 classes 与既有落盘一致）",
              flush=True)
    return bad


# ---------------------------------------------------------------- 阶段：tasks
def read_json(path: Path) -> Optional[dict]:
    """读 json：不存在 → None；**存在但解析不了 → 抛错**。

    撕裂的文件绝不能当成 None：合并逻辑会把它当"还没有既有内容"，于是把
    45 个切片的 tasks.json 一把换成本次这 6 个（fix 轮 1 IMPORTANT 3）。
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{p} 不是合法 JSON（可能是被杀留下的半截文件）：{e}。"
            "先人工检查/删除它再重跑，绝不能当成'空'去合并") from e


def write_json_atomic(path: Path, payload: dict) -> None:
    """临时文件 + os.replace 落盘：并发读方要么看到旧全量、要么看到新全量。

    直接 write_text 的窗口里文件是空的/半截的，别的进程此刻读到就是上面那种
    "撕裂读"；而本文件是 8 个进程共享的（铁律 3 的产物清单）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def merge_payload(prev: Optional[dict], payload: dict, *, list_keys=("pool",),
                  dict_keys=("tasks", "random")) -> dict:
    """把本次结果并进既有内容（`--datasets` 子集跑时**绝不把别的数据集抹掉**）。

    pod 上 8 个数据集各一个进程、共用一份 tasks.json/cls_catalog.json；直接覆写会
    让最后一个进程把文件缩成它自己那 6 个切片，33/34 装配时才发现少了 39 个切片。
    合并规则：列表键取并集（保序）、字典键新值覆盖同键，其余标量取本次值。
    """
    if not payload:
        raise ValueError("payload 为空——不允许把既有内容并进一个空表（会整份清零）")
    if not prev:
        return payload
    out = dict(payload)
    for k in list_keys:
        if isinstance(prev.get(k), list) and isinstance(payload.get(k), list):
            out[k] = prev[k] + [v for v in payload[k] if v not in prev[k]]
    for k in dict_keys:
        if isinstance(prev.get(k), dict) and isinstance(payload.get(k), dict):
            out[k] = {**prev[k], **payload[k]}
    return out


def _write_merged(path: Path, prev: Optional[dict], payload: dict) -> dict:
    merged = merge_payload(prev, payload)
    write_json_atomic(path, merged)
    return merged


def stage_tasks(outdir: Path, datasets: dict, plan: Sequence[Cell]) -> None:
    """写 tasks.json + cls_catalog.json（子集跑时并入既有），并做定义漂移对账。

    切片 classes 与既有落盘不符 → 退出码 2（缓存键指的必须是同一个任务，铁律 3）。
    对账在落盘**之前**：漂了就不该留下一份带漂移定义的 tasks.json 供人误用。
    """
    bad_all = []
    for name, ds in datasets.items():
        bad_all += [(name, *b) for b in check_definition_drift(
            {t.task_id: tuple(t.classes) for t in ds.slices}, name)]
    if bad_all:
        for name, sid, got, ref in bad_all:
            print(f"❌ [{name}] {sid} classes 漂移：本次 {got} ≠ 既有 {ref}", flush=True)
        sys.exit(2)

    path = outdir / "tasks.json"
    prev = read_json(path)                       # 撕裂即抛错，绝不当成"没有既有内容"
    stage = path.with_name("tasks.json.new")     # 本次 payload 先落临时文件，
    pq.write_tasks_json(datasets, stage)         # 免得 write_tasks_json 的非原子写
    payload = read_json(stage)                   # 在 path 上开一个"空文件"的窗口
    stage.unlink()
    merged = _write_merged(path, prev, payload)
    print(f"→ {path}（本次 {sum(len(d.slices) for d in datasets.values())} 切片，"
          f"累计 {len(merged.get('tasks', {}))}）", flush=True)

    shared, per_slice, ggs = [], {}, None
    seen = set()
    for c in plan:
        if c.origin == pq.ORIGIN_SHARED and c.cls_key not in seen:
            seen.add(c.cls_key)
            shared.append({"cls_key": c.cls_key, "cls": dataclasses.asdict(c.cls)})
        elif c.origin == pq.ORIGIN_RANDOM:
            per_slice.setdefault(c.slice_id, []).append(
                {"cls_key": c.cls_key, "cls": dataclasses.asdict(c.cls)})
        elif c.origin == ORIGIN_GGS and ggs is None:
            ggs = {"cls_key": c.cls_key, "cls": dataclasses.asdict(c.cls)}
    cat_path = outdir / "cls_catalog.json"
    cat = _write_merged(cat_path, read_json(cat_path),
                        {"shared": shared, "random": per_slice, "ggs": ggs})
    print(f"→ {cat_path}（shared {len(cat['shared'])} / random "
          f"{sum(len(v) for v in cat['random'].values())} 切片累计 / "
          f"ggs {'1' if cat.get('ggs') else '0'}）", flush=True)


# ---------------------------------------------------------------- 阶段：跑轨迹
def make_jobs(cells: Sequence, sidx: dict) -> list[Job]:
    """格 → Job（canonical 种子；NoiseCell 走 noise_seed，split 仍 canonical）。"""
    jobs = []
    for c in cells:
        if c.slice_id not in sidx:
            # 计划里的切片必然在装载结果里（两者都由同一份 names 派生，
            # load_pq_datasets 还断言过 task_id 与 pq_slice_ids 逐位一致）。
            # 静默跳过 = 那一格永远不跑、--verify 才发现，故直接炸。
            raise KeyError(f"{c.slice_id} 不在本次装载的切片里（计划与装载不一致）")
        name, task, X, y = sidx[c.slice_id]
        w_seed, sp_seed = pq.canonical_seeds(c.slice_id, c.cls_key)
        rep = getattr(c, "rep", 0)
        if rep:
            w_seed = pq.noise_seed(c.slice_id, c.cls_key, rep)
        jobs.append(Job(name, task, X, y, c.origin, c.cls, c.cls_key,
                        w_seed, sp_seed, rep))
    return jobs


def run_jobs(jobs: Sequence[Job], *, budgets: Sequence[int], cache_path: Path,
             inflight: Path, on_oom: str, oom_log: Path, eval_ratio: float,
             batch_size: int, lr: float, device: str, label: str) -> int:
    """跑一批轨迹（每条 = 一次 guarded_run_multi = N 行原子落盘）。返回 OOM 条数。"""
    if not jobs:
        print(f"[{label}] 无待跑轨迹", flush=True)
        return 0
    # threading 后端：fit_svm 的 GridSearchCV(n_jobs=-1) 在 Windows 上逐条 spawn
    # loky 进程池（8–15s/条，占 99% 耗时；R' 逐位同值）——长期记忆教训，勿删
    with parallel_backend("threading", n_jobs=1):
        return _run_loop(jobs, budgets=budgets, cache_path=cache_path,
                         inflight=inflight, on_oom=on_oom, oom_log=oom_log,
                         eval_ratio=eval_ratio, batch_size=batch_size, lr=lr,
                         device=device, label=label)


def _run_loop(jobs, *, budgets, cache_path, inflight, on_oom, oom_log, eval_ratio,
              batch_size, lr, device, label) -> int:
    t0, n_oom = time.time(), 0
    budgets = tuple(int(b) for b in budgets)
    for i, jb in enumerate(jobs):
        make_row = row_builder(dataset=jb.dataset, task=jb.task, cls_key=jb.cls_key,
                               origin=jb.origin, cls_cfg=jb.cls,
                               weight_seed=jb.weight_seed, seed=jb.seed)
        rows = guarded_run_multi(
            evaluate_rprime_trajectory, make_row,
            cache_path=cache_path, inflight_path=inflight, task_id=jb.task.task_id,
            cls_key=jb.cls_key, iters_list=budgets, on_oom=on_oom,
            oom_log_path=oom_log,
            context={"dataset": jb.dataset, "origin": jb.origin, "seed": jb.seed},
            log=lambda m, _i=i: print(f"[{label} {_i + 1}/{len(jobs)}] {m}", flush=True),
            task=jb.task, cls_cfg=jb.cls, X=jb.X, y=jb.y, weight_seed=jb.weight_seed,
            split_seed=jb.split_seed, checkpoints=budgets, eval_ratio=eval_ratio,
            batch_size=batch_size, lr=lr, device=device)
        if rows is None:
            n_oom += 1
            continue
        vals = ["null" if r["r_prime"] is None else f"{r['r_prime']:.4f}" for r in rows]
        eta = (time.time() - t0) / (i + 1) * (len(jobs) - i - 1)
        print(f"[{label} {i + 1}/{len(jobs)}] {jb.dataset}/{jb.task.task_id} "
              f"{jb.origin}:{jb.cls_key[:8]} seed={jb.seed} R'={'/'.join(vals)} "
              f"({rows[-1].get('trajectory_seconds')}s) ETA {eta / 60:.1f}min",
              flush=True)
    print(f"[{label}] 完成 {len(jobs)} 条轨迹，墙钟 {(time.time() - t0) / 60:.1f} min"
          + (f"，OOM {n_oom}（--resume 重试）" if n_oom else ""), flush=True)
    return n_oom


class InflightBusy(RuntimeError):
    """有 inflight 打点非空 → 可能有别的进程在追加缓存，禁止任何重写型操作。"""


def prune_partial(cache_path: Path, cells: Sequence, *, outdir: Path,
                  extra: Path | None = None, reason: str = "--redo-partial") -> int:
    """把指定格的行从缓存里剔掉（`--redo-partial` / `--retry-terminal` 前置）。

    append-only 缓存原则上不重写，但半条/毒轨迹重跑后会与旧行**同键重复**——
    `--verify` 的"零重复键"是硬闸，且旧行来自另一条被打断的轨迹，留着就是脏数据。
    落盘前先备份 `.bak-<时间戳>`（重放事故教训：绝不无痕截断）。
    受害键带 seed：噪声探针同格的另一个 rep 是独立轨迹，不能被连坐删掉。

    **互锁（本函数自己查，不靠调用方）**：outdir 下任何名字含 "inflight" 的文件
    只要非空就抛 InflightBusy。理由是本操作"读全表 → 内存过滤 → os.replace"，
    而 pod 上 8 个数据集进程共用一份 rprime_cache.jsonl 并发追加——重写会把
    replace 期间别的进程追加的行整段吞掉，且吞得无声无息。宁可要求操作方先确认
    无进程在跑、手动清空打点文件（消息里列出是哪几个）。
    """
    busy = busy_inflight(outdir, extra)
    if busy:
        raise InflightBusy(
            f"{reason} 要重写缓存，需独占；这些 inflight 打点非空："
            f"{[str(p) for p in busy]}。确认无其他进程在跑后清空它们再重试。")
    victims = {(c.slice_id, c.cls_key, int(getattr(c, "rep", 0))) for c in cells}
    if not victims or not Path(cache_path).exists():
        return 0
    rows = read_rows(cache_path)
    keep = [r for r in rows
            if (r["task_id"], r["cls_key"], int(r.get("seed", 0))) not in victims]
    dropped = len(rows) - len(keep)
    if not dropped:
        return 0
    bak = Path(cache_path).with_suffix(f".jsonl.bak-{time.strftime('%Y%m%d_%H%M%S')}")
    bak.write_bytes(Path(cache_path).read_bytes())
    tmp = Path(cache_path).with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                   encoding="utf-8")
    tmp.replace(cache_path)
    print(f"⚠️ {reason}：剔除 {dropped} 行（{len(victims)} 格），备份 {bak.name}",
          flush=True)
    return dropped


def inflight_files(outdir: Path, extra: Path | None = None) -> list[Path]:
    """outdir 下所有 inflight 打点文件（每进程一份，见 configs/runtime.yaml）。

    匹配**任何名字里含 "inflight" 的文件**，不是固定的 `inflight*.txt`：
    并发档给每个数据集起的名字是 `pq_inflight_<ds>.txt`，只认前缀的话
    `--verify` 的"inflight 为空"闸与 prune 的互锁会对它们全部失明（fix 轮 1
    CRITICAL 1）。打点文件必须落在 outdir 里，否则本函数同样看不见——
    run.sh/finish.sh 的 `--inflight` 一律指向 $OUT。
    """
    found = sorted(p for p in Path(outdir).glob("*inflight*") if p.is_file())
    if extra is not None and Path(extra).exists() and Path(extra) not in found:
        found.append(Path(extra))
    return found


def busy_inflight(outdir: Path, extra: Path | None = None) -> list[Path]:
    """非空的 inflight 打点文件 = 有进程在跑（或上次被杀留下的残迹）。

    **不排除本进程自己那份**：区分"我上一轮被杀留下的"与"隔壁进程正在跑的"在
    文件层面做不到，而猜错的代价是把别人正在追加的行整段吞掉。残留由操作方显式
    清空（refuse 消息会列出路径）。
    """
    return sorted((p for p in inflight_files(outdir, extra)
                   if p.read_text(encoding="utf-8", errors="replace").strip()),
                  key=str)


# ---------------------------------------------------------------- 阶段：δ 标定
def calibrate(outdir: Path, *, budgets: Sequence[int] = pq.BUDGETS,
              reps: Sequence[int] = (1, 2),
              bands_of: Callable[[str], dict] | None = None,
              datasets: Sequence[str] | None = None) -> dict:
    """按 (数据集, 预算) 标定 δ 与 δ_nd，落盘 delta_calibration.json。

    一个"格"= (切片, cls_key)；重复对 = {canonical(seed 0), rep1, rep2} 里两两组合。
    **必须三点齐全且有限**才纳入（缺一点的格不进 diffs——半套重复对会把 P95 拉偏）。
    量子下限取该数据集全部参与记录 `1/n_eval` 的**最大值**（pq_common.calibrate_delta
    的 docstring 点名：取平均/单条会给出偏小的下限，静默放行同分噪声配对）。

    δ_nd = 同式，但剔掉 strict 破坏格（canonical R' < 基线均值 − 带，基线来自
    `pq_common.baseline_bands`）——破坏 CLS 的轨迹噪声比正常 CLS 大一个档，
    两制式 δ 白拿 ~40% 存活率（memory: 噪声结构定稿）。
    """
    budgets = tuple(int(b) for b in budgets)
    reps = tuple(int(r) for r in reps)
    bands_of = bands_of if bands_of is not None else pq.baseline_bands
    idx = cache_index(read_rows(outdir / "rprime_cache.jsonl"))
    noise = read_rows(outdir / "noise_probe.jsonl")
    noise_idx = {(r["task_id"], r["cls_key"], int(r["iters"]), int(r["seed"])): r
                 for r in noise}
    cells = sorted({(r["task_id"], r["cls_key"]) for r in noise})

    by_ds: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sid, key in cells:
        ds = pq.dataset_of(sid)
        if datasets is None or ds in tuple(datasets):
            by_ds[ds].append((sid, key))

    out: dict[str, dict] = {}
    for ds, ds_cells in sorted(by_ds.items()):
        bands, bands_ok = {}, True
        try:
            bands = bands_of(ds)
        except Exception as e:                    # noqa: BLE001 —— 缺基线只降级
            bands_ok = False
            print(f"⚠️ [{ds}] 基线带不可用（{e}）→ δ_nd 记 null", flush=True)
        # 有基线但缺本数据集某个切片，同样判"不可用"：那些格的破坏与否无从判定，
        # 硬算出来的 δ_nd 会把它们当"正常"算进去，读者却以为已剃干净（fix 轮 1 之 6）。
        if bands_ok and any(sid not in bands for sid, _k in ds_cells):
            bands_ok = False
            print(f"⚠️ [{ds}] 基线带缺部分探针切片 → δ_nd 记 null", flush=True)
        per_b = {}
        for b in budgets:
            diffs, diffs_nd, quanta = [], [], []
            n_cells = n_incomplete = n_strict = 0
            for sid, key in ds_cells:
                vals, ok = [], True
                for rep in (0,) + reps:
                    rec = (idx.get((sid, key, b)) if rep == 0
                           else noise_idx.get((sid, key, b, rep)))
                    rp = rec.get("r_prime") if rec else None
                    if rp is None or not np.isfinite(rp):
                        ok = False
                        break
                    vals.append(float(rp))
                    if rec.get("n_eval"):
                        quanta.append(1.0 / float(rec["n_eval"]))
                if not ok:
                    n_incomplete += 1
                    continue
                n_cells += 1
                pair = [abs(vals[i] - vals[j])
                        for i in range(len(vals)) for j in range(i + 1, len(vals))]
                diffs += pair
                mean_band = bands.get(sid)
                strict = (mean_band is not None
                          and vals[0] < mean_band[0] - mean_band[1])
                if strict:
                    n_strict += 1
                else:
                    diffs_nd += pair
            if not diffs:
                print(f"⚠️ [{ds}@{b}] 无完整重复对（缺 {n_incomplete} 格）→ 跳过",
                      flush=True)
                continue
            quantum = max(quanta) if quanta else 0.0
            rec_out = {
                "delta": pq.calibrate_delta(diffs, quantum),
                "p95": _p95(diffs), "quantum_floor": 2.0 * quantum,
                "n_diffs": len(diffs), "diff_median": statistics.median(diffs),
                "diff_max": max(diffs), "n_cells": n_cells,
                "n_cells_incomplete": n_incomplete, "n_cells_strict": n_strict,
                "delta_nd": (pq.calibrate_delta(diffs_nd, quantum)
                             if (bands_ok and diffs_nd) else None),
                "n_diffs_nd": len(diffs_nd) if bands_ok else 0,
            }
            per_b[str(b)] = rec_out
            print(f"[{ds}@{b}] 格 {n_cells}（缺 {n_incomplete}，strict {n_strict}）"
                  f" 重复对 {len(diffs)} | P95={rec_out['p95']:.4f} "
                  f"量子下限={rec_out['quantum_floor']:.4f} → δ={rec_out['delta']:.3f}"
                  f"（δ_nd={rec_out['delta_nd']}，中位 "
                  f"{rec_out['diff_median']:.4f}）", flush=True)
        if per_b:
            out[ds] = per_b

    # 按数据集并入既有标定（`--datasets` 子集跑时绝不把别的数据集截掉——
    # 8 个进程各标各的，最后一个覆写会让 delta_calibration.json 只剩它自己，
    # 而 --apply-delta 会照着这份残表回填 labels.yaml。fix 轮 1 IMPORTANT 2）。
    path = outdir / "delta_calibration.json"
    merged = {**(read_json(path) or {}), **out}
    write_json_atomic(path, merged)
    print(f"→ {path}（本次 {len(out)} 数据集，累计 {len(merged)}）", flush=True)
    return merged


def delta_table(result: dict) -> dict[str, dict[int, float]]:
    """标定结果 → {数据集: {预算: δ}}（--apply-delta 与 YAML 块渲染共用）。"""
    return {ds: {int(b): float(rec["delta"]) for b, rec in sorted(
        per_b.items(), key=lambda kv: int(kv[0]))} for ds, per_b in result.items()}


def print_delta_yaml(table: dict[str, dict[int, float]]) -> None:
    print("\n# 回填 configs/labels.yaml（--apply-delta 可自动写入）:")
    print("  delta_per_dataset_budget:")
    for line in _delta_block_lines(table):
        print(line)


def _delta_block_lines(table: dict[str, dict[int, float]]) -> list[str]:
    """δ 表的 YAML 值块（不含键行；缩进 4/6，与 labels.yaml 现有风格一致）。"""
    order = [d for d in pq.PQ_DATASETS if d in table]
    order += [d for d in sorted(table) if d not in order]
    lines = []
    for ds in order:
        lines.append(f"    {ds}:")
        for b, v in sorted(table[ds].items()):
            lines.append(f"      {int(b)}: {float(v):.3f}")
    return lines


def delta_coverage_gaps(table: dict[str, dict[int, float]],
                        budgets: Sequence[int] = pq.BUDGETS) -> list[str]:
    """回填前的覆盖度闸：8 个数据集 × 全部预算档一个都不能缺。

    labels.yaml 的 δ 表是 33 号配对的唯一真源（缺项即 KeyError fail-fast）。
    子集跑标出来的半张表若被回填，33 号会拿"看起来齐全"的表跑起来，直到某个
    折才炸——或者更糟，别人以为那些数字是全量标定的。
    """
    gaps = []
    for ds in pq.PQ_DATASETS:
        if ds not in table:
            gaps.append(f"{ds}（整个数据集缺）")
            continue
        miss = [int(b) for b in budgets if int(b) not in table[ds]]
        if miss:
            gaps.append(f"{ds}@{miss}")
    return gaps


def apply_delta(labels_path: str | Path, table: dict[str, dict[int, float]]) -> None:
    """把 δ 表写进 configs/labels.yaml 的 `pq.delta_per_dataset_budget`。

    **只替换那一个键的值块**，其余行（含注释、行尾风格）逐字节不动——YAML
    round-trip 会重排注释与引号，对一份满是协议注释的配置是不可接受的损失。
    块的范围判定：键行之后，凡缩进 > 2 或空行都属于该块（labels.yaml 里紧跟的
    是缩进 2 的注释行，天然是边界）。幂等：重复执行结果逐字节相同。
    """
    path = Path(labels_path)
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    key = "  delta_per_dataset_budget:"
    hits = [i for i, ln in enumerate(lines) if ln.rstrip("\r\n").startswith(key)]
    if len(hits) != 1:
        raise ValueError(f"{path} 里 `{key.strip()}` 出现 {len(hits)} 次（应恰好 1 次）")
    i = hits[0]
    nl = lines[i][len(lines[i].rstrip("\r\n")):] or "\n"
    j = i + 1
    while j < len(lines):
        body = lines[j].rstrip("\r\n")
        # 只吞"缩进 > 2 的非空行"。空行**不吞**：它属于文件的排版，吞掉就违反
        # "除该块外逐字节不变"（fix 轮 1 之 7）；我们自己写的块里也没有空行。
        if body.strip() and body.startswith("   "):
            j += 1
            continue
        break
    block = [key + nl] + [ln + nl for ln in _delta_block_lines(table)]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(lines[:i] + block + lines[j:]))
    print(f"→ 已回填 {path}: pq.delta_per_dataset_budget（{len(table)} 数据集）",
          flush=True)


def stage1_consistency(result: dict, *, path: Path = STAGE1_NOISE, budget: int = 290,
                       slices: Sequence[str] = DRIFT_SLICES) -> list[tuple]:
    """本次 @290 的 P95 vs 阶段1 k3 缓存（seeds 1–3，切片 00/01）的比值。

    两边都是 k=3 → 每格 3 个重复对，口径可比；差别只在种子来源（本次含
    canonical seed）。落在 [0.7, 1.4] 之外只告警——噪声本就随天花板距离浮动，
    但一个数量级的偏差意味着口径搞错了（比如切片/划分不同）。
    """
    if not Path(path).exists():
        print(f"⚠️ 阶段1 参照缺失（{path}）→ 跳过 @{budget} 一致性对账", flush=True)
        return []
    rows = [r for r in read_rows(path)
            if int(r.get("iters", 0)) == budget and r.get("r_prime") is not None
            and str(r["task_id"]).endswith(tuple(slices))]
    ref_cells: dict[str, dict] = defaultdict(dict)
    for r in rows:
        ref_cells[pq.dataset_of(r["task_id"])].setdefault(
            (r["task_id"], r["cls_key"]), {})[int(r["seed"])] = float(r["r_prime"])
    out = []
    print(f"\n@{budget} 与阶段1 k3 缓存的 P95 比值（判读带 {RATIO_BAND}）:")
    for ds, per_b in sorted(result.items()):
        rec = per_b.get(str(budget))
        if rec is None or ds not in ref_cells:
            continue
        ref_diffs = []
        for vals in ref_cells[ds].values():
            v = list(vals.values())
            ref_diffs += [abs(v[i] - v[j]) for i in range(len(v))
                          for j in range(i + 1, len(v))]
        if not ref_diffs:
            continue
        ref = _p95(ref_diffs)
        ratio = rec["p95"] / ref if ref else float("inf")
        flag = "" if RATIO_BAND[0] <= ratio <= RATIO_BAND[1] else "  ⚠️ 带外"
        out.append((ds, rec["p95"], ref, ratio, bool(flag)))
        print(f"  {ds:<26} 本次 {rec['p95']:.4f} / 阶段1 {ref:.4f} = "
              f"{ratio:.2f}{flag}")
    return out


# ---------------------------------------------------------------- 阶段：复现校验
def _is_5090(device_name) -> bool:
    return "5090" in str(device_name or "")


def repro_bucket(new_device, old_device) -> str:
    """一对行落哪个容差桶：两端都是 5090 → 严档，其余（含未知设备）→ 混机档。

    §8 把两份参照分别写作"5090 对 5090"与"混机"，但 wide_panel 里两种机器各占一半，
    所以判据只能逐行按 `device_name` 分——按文件名一刀切会把其中 167 条 5090↔5090
    的对子也放宽到 0.03，白丢一半的分辨力。
    """
    return ("same_5090" if _is_5090(new_device) and _is_5090(old_device)
            else "mixed")


def _repro_index(rows: Iterable[dict], *, budget: int,
                 canonical_only: bool) -> dict[tuple, dict]:
    """{(task_id, cls_key, weight_seed): 行}——只留该预算档的有限 R'。

    `canonical_only`（新缓存侧）额外要求 `seed == 0`：噪声探针的 rep 行是**另一条
    轨迹**（另一个 weight_seed 本就对不上，但 seed 字段是更直接的判据），混进来
    只会让"重叠数"看着变大。旧缓存没有 seed 字段（全是 canonical），不设此闸。
    """
    out: dict[tuple, dict] = {}
    for r in rows:
        if int(r.get("iters", -1)) != int(budget):
            continue
        if canonical_only and int(r.get("seed") or 0) != 0:
            continue
        rp = r.get("r_prime")
        if rp is None or not np.isfinite(rp):
            continue
        ws = r.get("weight_seed")
        if ws is None:
            continue
        out[(r["task_id"], r["cls_key"], int(ws))] = r
    return out


def _repro_bucket_stats(pairs: Sequence[tuple[float, float]], rule: dict) -> dict:
    """一个桶的判定：命中率 + 配对中位 |Δ| + 符号均值（后者只报，不设门）。"""
    diffs = [new - old for new, old in pairs]
    absd = sorted(abs(d) for d in diffs)
    n_within = sum(1 for d in absd if d <= rule["tol"])
    frac = n_within / len(absd)
    median = statistics.median(absd)
    ok = frac >= rule["min_within"] and (rule["max_median"] is None
                                         or median <= rule["max_median"])
    return {"n": len(absd), "n_within": n_within, "frac_within": frac,
            "median_abs": median, "signed_mean": statistics.fmean(diffs),
            "max_abs": absd[-1], "tol": rule["tol"],
            "min_within": rule["min_within"], "max_median": rule["max_median"],
            "ok": bool(ok)}


def reproduce_check(outdir: Path, *, sources: Sequence[tuple[str, Path]] = None,
                    budget: int = REPRO_BUDGET,
                    tolerances: dict = None) -> dict:
    """PROTOCOL §8 复现校验：新 @290 canonical 行 vs 旧缓存，落盘 reproduction_check.json。

    连接键 = (task_id, cls_key, weight_seed)——**同一条轨迹的三要素**。只比 R'：
    seconds/device 本就该不同，acc/f1 是 R' 的同源读数。

    判定（阈值来自 §8，跑完不改）：
    - 5090↔5090：≥90% 的对子 |ΔR'| ≤ 0.02 **且** 配对中位 |Δ| ≤ 0.01；
    - 混机：≥85% 的对子 |ΔR'| ≤ 0.03（§8 没给中位门，故不设）。
    档位**逐对按 device_name 分**（wide_panel 里 4060/5090 各占一半，按文件一刀切
    会把其中 167 条 5090↔5090 也放宽到 0.03）；每个源另出一份 §8 原文口径的
    `protocol_reading`（按文件定档、全部对子合并）作对照，只报不判。

    **零重叠不是通过**：一个都没连上意味着切片定义/种子口径已经漂了（或者参照
    缓存没上传），这时"没有反例"恰恰是最坏的信号，故与超差同样判红。
    """
    outdir = Path(outdir)
    sources = tuple(sources if sources is not None else REPRO_SOURCES)
    tolerances = tolerances if tolerances is not None else REPRO_TOL
    new_rows = read_rows(outdir / "rprime_cache.jsonl")
    new_idx = _repro_index(new_rows, budget=budget, canonical_only=True)
    print(f"\n复现校验（@{budget}，PROTOCOL §8）：本次 canonical 行 {len(new_idx)}")

    per_source: dict[str, dict] = {}
    hard: list[str] = []
    for name, path in sources:
        path = Path(path)
        entry: dict = {"path": str(path), "n_old_rows": 0, "n_matched": 0,
                       "buckets": {}, "failures": [], "ok": False}
        if not path.exists():
            entry["failures"].append(
                f"缺参照缓存 {path}（setup.sh 的 data_pq.tar.gz 清单里有它，"
                "没解出来说明 bundle 漏打或路径不对）")
            per_source[name] = entry
            hard.append(f"{name}: 缺参照缓存")
            print(f"  ❌ [{name}] 缺参照缓存 {path}")
            continue
        old_idx = _repro_index(read_rows(path), budget=budget,
                               canonical_only=False)
        entry["n_old_rows"] = len(old_idx)
        buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for key, old in old_idx.items():
            new = new_idx.get(key)
            if new is None:
                continue
            buckets[repro_bucket(new.get("device_name"),
                                 old.get("device_name"))].append(
                (float(new["r_prime"]), float(old["r_prime"])))
        entry["n_matched"] = sum(len(v) for v in buckets.values())
        if not entry["n_matched"]:
            entry["failures"].append(
                f"与 {path} 零重叠（旧 @{budget} 行 {len(old_idx)}，本次 canonical "
                f"{len(new_idx)}）——切片/CLS/权重种子口径可能已漂，"
                "**不得读作通过**")
            per_source[name] = entry
            hard.append(f"{name}: 零重叠")
            print(f"  ❌ [{name}] 零重叠（旧 {len(old_idx)} 行）")
            continue
        for bname, pairs in sorted(buckets.items()):
            st = _repro_bucket_stats(pairs, tolerances[bname])
            entry["buckets"][bname] = st
            flag = "✅" if st["ok"] else "❌"
            print(f"  {flag} [{name}/{bname}] n={st['n']} 命中 "
                  f"{st['n_within']}/{st['n']}={st['frac_within']:.3f}"
                  f"（≥{st['min_within']:.2f} @tol {st['tol']}）| 中位 |Δ|="
                  f"{st['median_abs']:.4f}"
                  + (f"（≤{st['max_median']}）" if st["max_median"] else "")
                  + f" | 符号均值 {st['signed_mean']:+.4f} | 最大 |Δ|="
                    f"{st['max_abs']:.4f}")
            if not st["ok"]:
                entry["failures"].append(
                    f"{bname}: 命中 {st['frac_within']:.3f} / 中位 "
                    f"{st['median_abs']:.4f}（门 ≥{st['min_within']} @"
                    f"{st['tol']}，中位 ≤{st['max_median']}）")
        # 逐源汇总读数（§8 是**按文件**给阈值的：source_pool_v2 全 5090 走严档，
        # wide_panel 含 4060 走混机档）。判红仍以逐对分桶为准——分桶只会更严，
        # 绝不放宽；两个读数不一致时，操作方看得见分歧再决定要不要
        # --allow-reproduction-breach，而不是被一个数字蒙住。
        all_pairs = [p for v in buckets.values() for p in v]
        srule = "mixed" if "mixed" in buckets else "same_5090"
        entry["protocol_reading"] = {
            "regime": srule, **_repro_bucket_stats(all_pairs, tolerances[srule])}
        entry["ok"] = not entry["failures"]
        if not entry["ok"]:
            hard.append(f"{name}: " + "；".join(entry["failures"]))
        per_source[name] = entry

    rep = {"budget": int(budget), "ok": not hard,
           "n_new_canonical": len(new_idx), "sources": per_source,
           "hard_failures": hard,
           "thresholds": {k: dict(v) for k, v in tolerances.items()}}
    write_json_atomic(outdir / "reproduction_check.json", rep)
    print("✅ 复现校验通过" if not hard
          else f"❌ 复现校验失败: {'；'.join(hard)}", flush=True)
    print(f"→ {outdir / 'reproduction_check.json'}", flush=True)
    return rep


# ---------------------------------------------------------------- 阶段：verify
def verify(outdir: Path, plan: Sequence[Cell], noise_cells: Sequence[NoiseCell],
           budgets: Sequence[int], *, inflight_extra: Path | None = None,
           oom_log: Path | None = None, allow_terminal: bool = False) -> dict:
    """收官校验：计数 / null / 重复键 / 设备混杂 / 半条轨迹 / 毒轨迹 / inflight / OOM。

    硬失败（退出码非 0）：缺格、半条轨迹、重复键、null 却无 error、噪声缺格、
    inflight 非空、**毒轨迹**（除非 --allow-terminal）。
    只告警：device_name 混杂（跨机器 R' 有 ±0.005 级差，成本锚也会失真）、OOM 记录。

    **计数与行顺序无关**（fix 轮 3 复查）：每个数字要么是对 rows 的 sum/len，
    要么是把 rows 建成字典后**按 plan 顺序**查表（plan 由注册表 + CLS 目录确定）。
    唯一受顺序影响的是 `cache_index` 对**重复键**的"后行覆盖前行"（与
    labeling.load_rprime_cache 同约定）：同一键一好一坏时，状态取決于哪行在后。
    但重复键本身就是硬失败，那一跑已经是红的，不存在"靠顺序蒙混过关"。
    """
    outdir = Path(outdir)
    budgets = tuple(int(b) for b in budgets)
    rows = read_rows(outdir / "rprime_cache.jsonl")
    idx = cache_index(rows)
    keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
    n_dup = len(keys) - len(set(keys))
    n_null_no_error = sum(1 for r in rows
                          if r.get("r_prime") is None and not r.get("error"))
    devices = sorted({r.get("device_name") for r in rows if r.get("device_name")})

    states = cell_states(plan, idx, budgets)
    main_cells = [c for c in plan if c.origin != ORIGIN_GGS]
    ggs_cells = [c for c in plan if c.origin == ORIGIN_GGS]
    n_main = sum(1 for r in rows if r.get("origin") != ORIGIN_GGS)
    n_ggs = sum(1 for r in rows if r.get("origin") == ORIGIN_GGS)

    nrows = read_rows(outdir / "noise_probe.jsonl")
    nidx = {(r["task_id"], r["cls_key"], int(r["iters"]), int(r["seed"])): r
            for r in nrows}
    # 噪声缓存的同键重复与主缓存同罪（键**带 seed**：rep1/rep2 是两条独立轨迹，
    # 不算重复）。同一 (切片, CLS, 预算, rep) 出现两次 = 那条轨迹跑了两遍，
    # δ 会把同一份差值算进 P95 两次，标定值静默偏移，而主缓存那道闸对本文件失明。
    n_noise_dup = len(nrows) - len(nidx)
    n_noise_missing = sum(
        1 for c in noise_cells for b in budgets
        if (c.slice_id, c.cls_key, b, c.rep) not in nidx)
    n_noise_null = sum(1 for r in nrows
                       if r.get("r_prime") is None and not r.get("error"))
    # 噪声格同样会中毒（整条 error 行）。它们不算"缺行"，却让那个格
    # 凑不齐重复对 → δ 站在更少的样本上。与主网格同一道闸、同一个旗标。
    nstates = cell_states(noise_cells, index_by_seed(nrows), budgets)

    busy = busy_inflight(outdir, inflight_extra)
    ooms = read_rows(oom_log or outdir / "oom_failures.jsonl")

    print(f"\nrprime_cache: {len(rows)} 行（主 {n_main}/期望 "
          f"{len(main_cells) * len(budgets)}，GGS {n_ggs}/期望 "
          f"{len(ggs_cells) * len(budgets)}）| 唯一键 {len(set(keys))} | "
          f"重复 {n_dup} | null 无 error {n_null_no_error}")
    print(f"轨迹状态: done {len(states[DONE])} / partial {len(states[PARTIAL])} / "
          f"terminal {len(states[TERMINAL])} / missing {len(states[MISSING])}"
          f"（共 {len(plan)} 格）｜噪声毒轨迹 {len(nstates[TERMINAL])}")
    print(f"noise_probe: {len(nrows)} 行 / 期望 {len(noise_cells) * len(budgets)}"
          f"（缺 {n_noise_missing}，重复 {n_noise_dup}，"
          f"null 无 error {n_noise_null}）")
    print(f"device_name: {devices if devices else '（无）'}"
          + ("  ⚠️ 跨设备混杂" if len(devices) > 1 else ""))
    if ooms:
        per_ds = defaultdict(int)
        for r in ooms:
            per_ds[r.get("dataset", "?")] += 1
        print(f"⚠️ oom_failures: {len(ooms)} 条 {dict(per_ds)}")
    for c in states[PARTIAL][:20]:
        print(f"  ⚠️ 半条轨迹 {c.slice_id}/{c.cls_key[:8]}（--fill --redo-partial 重跑）")
    terminal_ids = ([f"{c.slice_id}:{c.cls_key}" for c in states[TERMINAL]]
                    + [f"{c.slice_id}:{c.cls_key}:noise{c.rep}"
                       for c in nstates[TERMINAL]])
    for c in list(states[TERMINAL])[:20]:
        row = next((idx.get((c.slice_id, c.cls_key, b)) for b in budgets
                    if idx.get((c.slice_id, c.cls_key, b))), None)
        err = (row or {}).get("error", "")
        print(f"  ❌ 毒轨迹 {row.get('trajectory_id') if row else c.slice_id}"
              f"（整条无 R'）{str(err)[:80]}")
    for c in states[MISSING][:20]:
        print(f"  ❌ 缺格 {c.slice_id}/{c.origin}:{c.cls_key[:8]}")
    for p in busy:
        print(f"  ❌ inflight 非空: {p}（有进程在跑，或上次被杀留下打点）")

    hard = []
    if states[MISSING]:
        hard.append(f"缺格 {len(states[MISSING])}")
    if states[PARTIAL]:
        hard.append(f"半条轨迹 {len(states[PARTIAL])}")
    if n_dup:
        hard.append(f"重复键 {n_dup}")
    if n_noise_dup:
        hard.append(f"噪声重复键 {n_noise_dup}")
    if n_null_no_error or n_noise_null:
        hard.append(f"null 无 error {n_null_no_error + n_noise_null}")
    if n_noise_missing:
        hard.append(f"噪声缺格 {n_noise_missing}")
    if busy:
        hard.append(f"inflight 非空 {len(busy)}")
    # 毒轨迹 = 那几格永远没有 R'（非 OOM 异常写下的 error 行）。当"告警"报
    # 会让 finish.sh 打印"verify 通过"而网格实际缺格（fix 轮 1 IMPORTANT 4），
    # 故默认判红；已人工确认的才用 --allow-terminal 放行。
    n_terminal_all = len(states[TERMINAL]) + len(nstates[TERMINAL])
    if n_terminal_all and not allow_terminal:
        hard.append(f"毒轨迹 {n_terminal_all}（排查后 --retry-terminal "
                    "重跑，或确认不可恢复后 --verify --allow-terminal）")
    print("✅ verify 通过" if not hard else f"❌ verify 失败: {'；'.join(hard)}",
          flush=True)
    return {"ok": not hard, "hard_failures": hard, "n_rows": len(rows),
            "n_main": n_main, "n_ggs": n_ggs, "n_noise": len(nrows),
            "n_duplicate": n_dup, "n_noise_duplicate": n_noise_dup,
            "n_null_no_error": n_null_no_error,
            "n_missing": len(states[MISSING]), "n_partial": len(states[PARTIAL]),
            "n_terminal": n_terminal_all, "n_done": len(states[DONE]),
            "n_terminal_main": len(states[TERMINAL]),
            "n_terminal_noise": len(nstates[TERMINAL]),
            "terminal_ids": terminal_ids,
            "n_noise_missing": n_noise_missing, "devices": devices,
            "inflight": [str(p) for p in busy], "n_oom": len(ooms)}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=None,
                    help=f"产物目录（默认 {DEFAULT_OUTDIR}；--smoke 默认 {SMOKE_OUTDIR}）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=f"只跑这些数据集（默认全部 {len(pq.PQ_DATASETS)} 个）")
    ap.add_argument("--inflight", type=Path, default=None,
                    help="看门狗打点文件（并发时每进程一份，默认 <outdir>/inflight.txt）")
    ap.add_argument("--on-oom", choices=(ON_OOM_SKIP, ON_OOM_RECORD),
                    default=ON_OOM_SKIP)
    ap.add_argument("--oom-log", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--redo-partial", action="store_true",
                    help="重跑半条轨迹（先剔旧行并备份，防同键重复）")
    ap.add_argument("--retry-terminal", action="store_true",
                    help="重跑毒轨迹（整条 error 行）：先剔旧行并备份。"
                         "永不自动发生，必须操作方显式要求")
    ap.add_argument("--allow-terminal", action="store_true",
                    help="--verify 时把毒轨迹当已知损失放行（默认判红）")
    ap.add_argument("--checkpoints", nargs="+", type=int, default=None,
                    help="覆盖预算档（仅 smoke/非生产 --outdir；默认 configs 的 budgets）")
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--ggs", action="store_true")
    ap.add_argument("--noise-probe", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--apply-delta", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reproduce-check", action="store_true",
                    help="PROTOCOL §8 复现校验：新 @290 行 vs source_pool_v2 / "
                         "wide_panel 旧缓存（不过则非零退出）")
    ap.add_argument("--allow-reproduction-breach", action="store_true",
                    help="复现校验超差时仍返回 0（须人工确认后显式使用）")
    ap.add_argument("--smoke", action="store_true",
                    help="2 数据集 × 切片 00 × 5 CLS（2 shared/2 random/GGS）"
                         " × checkpoints(4,8,12) 全流程")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    stages = (args.tasks, args.fill, args.ggs, args.noise_probe, args.calibrate,
              args.verify, args.smoke, args.reproduce_check)
    if not any(stages):
        ap.error("至少指定一个阶段（--tasks/--fill/--ggs/--noise-probe/"
                 "--calibrate/--verify/--reproduce-check/--smoke）")

    if args.apply_delta and not args.calibrate:
        ap.error("--apply-delta 必须与 --calibrate 同跑（回填的是本次标定值）")
    if args.apply_delta and args.smoke:
        ap.error("--smoke 的 δ 是玩具口径（checkpoints 4/8/12），绝不回填 labels.yaml")
    # --checkpoints 改的是**缓存键**（铁律 2 的 iters 口径）。作用在生产目录上，
    # 跑出来的行与全量网格同 (task_id, cls_key) 却是另一个训练量，混进
    # rprime_cache.jsonl 就再也分不出来。只在 smoke 或另开 --outdir 时放行。
    if args.checkpoints and not args.smoke:
        prod = (args.outdir is None
                or Path(args.outdir).resolve() == DEFAULT_OUTDIR)
        if prod:
            print(f"❌ --checkpoints 会改 iters 口径，不得作用在真值网格目录 "
                  f"{DEFAULT_OUTDIR}——请加 --smoke 或换一个 --outdir", flush=True)
            sys.exit(2)     # 与定义漂移/smoke 误写同码：配置层硬错，开跑前就断

    cfg = pq.load_pq_config()
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    eval_ratio = labels_cfg["eval_ratio"]

    if args.smoke:
        if args.outdir is not None and Path(args.outdir).resolve() == DEFAULT_OUTDIR:
            print(f"❌ --smoke 不得写进真值网格目录 {DEFAULT_OUTDIR}"
                  f"（smoke 的 checkpoints 与全量不同，混进去就是脏缓存）", flush=True)
            sys.exit(2)     # 与定义漂移同码：配置层硬错，开跑前就断
        outdir = Path(args.outdir) if args.outdir is not None else SMOKE_OUTDIR
        names = SMOKE_DATASETS
        budgets = tuple(args.checkpoints or SMOKE_CHECKPOINTS)
        # 2 shared + 2 random + GGS = 每切片 5 条。**两边都必须是 2**：配对是
        # 同切片内两两组，只有 1 条时那一类根本组不出对——shared 不够则 33 号
        # 课程期无 shared–shared，random 不够则 wo_shared 臂（只吃 random）根本不训练
        # （fix 轮 3 + 追加）。
        n_slices = 1
        n_cls = {pq.ORIGIN_SHARED: 2, pq.ORIGIN_RANDOM: 2}
        noise_slices, noise_reps = 1, (1,)
        # 噪声探针只跑 shared（与全量口径一致）：重复对是拿来标 δ 的，
        # 混进 GGS/random 会把不同性质的格摄进同一个 P95。
        noise_origins = (pq.ORIGIN_SHARED,)
        do = dict(tasks=True, fill=True, ggs=True, noise=True, calibrate=True,
                  verify=True)
        resume, redo_partial = True, args.redo_partial
        retry_terminal = args.retry_terminal
    else:
        outdir = Path(args.outdir) if args.outdir is not None else DEFAULT_OUTDIR
        names = tuple(args.datasets or pq.PQ_DATASETS)
        bad = [n for n in names if n not in pq.PQ_DATASETS]
        if bad:
            ap.error(f"未知数据集 {bad}（合法：{pq.PQ_DATASETS}）")
        budgets = tuple(args.checkpoints or pq.BUDGETS)
        n_slices = n_cls = None
        noise_slices = int(cfg["noise_probe"]["slices"])
        noise_reps = tuple(cfg["noise_probe"]["reps"])
        noise_origins = (pq.ORIGIN_SHARED,)
        do = dict(tasks=args.tasks, fill=args.fill, ggs=args.ggs,
                  noise=args.noise_probe, calibrate=args.calibrate,
                  verify=args.verify)
        resume, redo_partial = args.resume, args.redo_partial
        retry_terminal = args.retry_terminal

    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "rprime_cache.jsonl"
    noise_path = outdir / "noise_probe.jsonl"
    inflight = Path(args.inflight) if args.inflight else outdir / "inflight.txt"
    oom_log = Path(args.oom_log) if args.oom_log else outdir / "oom_failures.jsonl"

    print(f"[32] outdir={outdir} 数据集={list(names)} budgets={list(budgets)} "
          f"device={args.device} resume={resume}", flush=True)
    plan = build_plan(names, n_slices=n_slices, n_cls=n_cls)
    ncells = noise_plan(plan, n_slices=noise_slices, reps=noise_reps,
                        origins=noise_origins)

    datasets = None
    if do["tasks"] or do["fill"] or do["ggs"] or do["noise"]:
        datasets = pq.load_pq_datasets(names)
        sidx = slice_index(datasets)

    if do["tasks"]:
        stage_tasks(outdir, datasets, plan)

    if (do["fill"] or do["ggs"] or do["noise"]) and not resume:
        if cache_path.exists() or noise_path.exists():
            print("⚠️ 未带 --resume 且缓存非空：已完成的格会被**重跑并追加同键行**"
                  "（--verify 的零重复键会判红）。要续跑请加 --resume。", flush=True)

    if do["fill"] or do["ggs"]:
        idx = cache_index(read_rows(cache_path))
        want = ([c for c in plan if c.origin in (pq.ORIGIN_SHARED, pq.ORIGIN_RANDOM)]
                if do["fill"] else [])
        want += [c for c in plan if c.origin == ORIGIN_GGS] if do["ggs"] else []
        # shared 全跑完再跑 random（2c 同序：先把可跨数据集比较的那批填满）
        order = {pq.ORIGIN_SHARED: 0, pq.ORIGIN_RANDOM: 1, ORIGIN_GGS: 2}
        want.sort(key=lambda c: (order[c.origin], c.slice_id))
        todo = pending_cells(want, idx, budgets, resume=resume,
                             redo_partial=redo_partial,
                             retry_terminal=retry_terminal)
        st = cell_states(want, idx, budgets)
        print(f"R' 网格: 计划 {len(want)} 格 | done {len(st[DONE])} / partial "
              f"{len(st[PARTIAL])} / terminal {len(st[TERMINAL])} → 待跑 {len(todo)}",
              flush=True)
        victims = ((st[PARTIAL] if redo_partial else [])
                   + (st[TERMINAL] if retry_terminal else []))
        if victims:
            try:
                prune_partial(cache_path, victims, outdir=outdir, extra=inflight,
                              reason="--redo-partial/--retry-terminal")
            except InflightBusy as e:
                print(f"❌ {e}", flush=True)
                return 3
        run_jobs(make_jobs(todo, sidx), budgets=budgets, cache_path=cache_path,
                 inflight=inflight, on_oom=args.on_oom, oom_log=oom_log,
                 eval_ratio=eval_ratio, batch_size=train_cfg["batch_size"],
                 lr=train_cfg["lr"], device=args.device, label="fill")

    if do["noise"]:
        nidx = index_by_seed(read_rows(noise_path))
        todo = pending_cells(ncells, nidx, budgets, resume=resume,
                             redo_partial=redo_partial,
                             retry_terminal=retry_terminal)
        st = cell_states(ncells, nidx, budgets)
        print(f"噪声探针: 计划 {len(ncells)} 格 | done {len(st[DONE])} / partial "
              f"{len(st[PARTIAL])} / terminal {len(st[TERMINAL])} → 待跑 {len(todo)}",
              flush=True)
        victims = ((st[PARTIAL] if redo_partial else [])
                   + (st[TERMINAL] if retry_terminal else []))
        if victims:
            try:
                prune_partial(noise_path, victims, outdir=outdir, extra=inflight,
                              reason="--redo-partial/--retry-terminal")
            except InflightBusy as e:
                print(f"❌ {e}", flush=True)
                return 3
        run_jobs(make_jobs(todo, sidx), budgets=budgets, cache_path=noise_path,
                 inflight=inflight, on_oom=args.on_oom, oom_log=oom_log,
                 eval_ratio=eval_ratio, batch_size=train_cfg["batch_size"],
                 lr=train_cfg["lr"], device=args.device, label="probe")

    if do["calibrate"]:
        res = calibrate(outdir, budgets=budgets, reps=noise_reps, datasets=names)
        table = delta_table(res)
        print_delta_yaml(table)
        if 290 in budgets:
            stage1_consistency(res)
        if args.apply_delta:
            gaps = delta_coverage_gaps(table, budgets)
            if gaps:
                print(f"❌ --apply-delta：标定表未覆盖全网格，拒绝回填。缺：{gaps}",
                      flush=True)
                print("   labels.yaml 的 δ 表是 33 号配对的唯一真源；半张表回填下去，",
                      flush=True)
                print("   读者会以为那是全量标定。先把缺的那几个数据集跑完再来。",
                      flush=True)
                return 4
            apply_delta(ROOT / "configs" / "labels.yaml", table)

    if do["verify"]:
        rep = verify(outdir, plan, ncells, budgets, inflight_extra=inflight,
                     oom_log=oom_log, allow_terminal=args.allow_terminal)
        if not rep["ok"]:
            return 1

    if args.reproduce_check:
        rep = reproduce_check(outdir)
        if not rep["ok"]:
            if not args.allow_reproduction_breach:
                print("   §8：复现校验不过 → **停下诊断，不得进比较器阶段**。"
                      "确认无碍才用 --allow-reproduction-breach 放行", flush=True)
                return 5
            print("   ⚠️ --allow-reproduction-breach：超差已被显式放行，"
                  "报表须声明本轮 R' 与历史口径的偏差", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
