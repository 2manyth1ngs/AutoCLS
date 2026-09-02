"""MX-3-PQ 公共装载/切片/配对/δ/噪声模型——scripts/32–34 共用的唯一真源。

协议：.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/
interfaces.md「Budgets / ids」「Task 4 → Tasks 5, 6, 7」。

MX-3-PQ = 8 数据集 × 7 折 leave-one-dataset-out。每个数据集取注册表切出的**前 6 个**
切片（SelfRegulationSCP2 按注册表 planned_tasks 只有 3），每切片 20 个 CLS（10 shared
探针目录 + 10 random），每条轨迹在 60/120/290 三个预算 checkpoint 上评估 R'——比较器
看到的"任务"因此是 `slice@budget`（engine task id），预算以 one-hot ctx 进模型。

本模块是数据集清单 / 切片 id / CLS 目录 / 种子 / 折划分 / δ 表 / 消融指派 /
(dataset, budget) 噪声模型的**唯一落点**：32（造 R'）、33（训比较器 + 消融）、
34（进化搜索）一律从这里取，禁止各自 fork 一份口径。

铁律落点：
- 铁律 1：CLS 取值集合只经 CLSSpace（configs/space.yaml），本模块零硬编码档位；
- 铁律 2：预算（iters）显式进 engine task id 与缓存键，early/full 口径不混用；
  configs/train.yaml 的 early_iters/full_iters 不动，budgets 是新增的第三口径；
- 铁律 3：(task, cls) → R' 缓存键 = (slice, cls_key, budget)；切片一律走
  `slice_source_tasks(entry, reg, y)[1][:N_SLICES]`——**绝不传 n_tasks=6**：n_tasks
  改变会改切片 RNG 的前缀流，既有 canonical R' 缓存整体失效；
- 铁律 6：考侧数据集绝不进训练侧（fold_split 保证）；加载入口对目标池 fail-fast
  （load_source_dataset 自带），本模块再补一道注册表 target 自检；
- 铁律 7/8：本模块只做装配，"谁更优"的方向语义仍留在 labeling/task_types，
  比较器与 task_embed 不因本模块引入任何数据集概念（ctx 只是通用向量）。
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.soft_labels import (  # noqa: E402
    NoiseModel, fit_noise_model, pair_probability, task_scale_factor,
)
from autocls.comparator.tclsc_engine import ClsRecord, derangement  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, DatasetEntry, SourcePoolRegistry, load_registry,
    load_source_dataset, slice_source_tasks,
)
from autocls.data.task_slicer import SlicedTask, materialize_task  # noqa: E402
from autocls.labeling import (  # noqa: E402
    ORIGIN_RANDOM, ORIGIN_SHARED, cls_key, comparator_records, load_probe_cls,
    load_rprime_cache, run_seed_for, sample_random_cls, split_task_train_eval,
)
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
# jsonl 读取复用 e_ablation_common（不再 fork 第三份）：注意它对**不存在的文件**
# 抛 FileNotFoundError（非返回空表）——调用方读 noise_probe.jsonl 这类分阶段产物前
# 需自己判 exists()。
from scripts.e_ablation_common import read_jsonl  # noqa: E402

# ---------------------------------------------------------------- 常量（协议写死）
# 源池 v2 定稿 8（configs/source_pool.yaml 文件头）：核心 4 + 晋级 3 + EEG 低权 1。
PQ_DATASETS = (
    "Handwriting", "UniMiB-SHAR", "PhonemeSpectra", "UWaveGestureLibrary",
    "EthanolConcentration", "Cricket", "HandMovementDirection", "SelfRegulationSCP2",
)
# SCP2 = EEG 低权：只供 label 与全局先验，不作 LOO 折、不入靶池（源池 v2 REPORT）。
TRAINING_ONLY = ("SelfRegulationSCP2",)
FOLD_TARGETS = tuple(d for d in PQ_DATASETS if d not in TRAINING_ONLY)   # 7 折

BUDGETS = (60, 120, 290)     # 同一条轨迹的 checkpoint（非三次独立训练），铁律 2
N_SLICES = 6                 # 每数据集取注册表切片的前 6 个（SCP2 只有 3）
N_SHARED = 10                # shared CLS = 探针目录前 10（与 2b/2c 同口径）
N_RANDOM = 10                # random CLS = 每切片 task_id 派生
CLS_SEED = 42                # labels.yaml route_a.cls_seed 同值（缓存跨脚本复用）

SHARED_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
OUT = ROOT / "results" / "mx3_pq"

# 双注册表分派（scripts/25_mx2_endtoend.py:72 同款）：UniMiB-SHAR 在候选池注册表，
# 其余 7 个在 configs/source_pool.yaml（None = 默认注册表）。两文件各是各池唯一真源。
REG_OF: dict[str, Optional[str]] = {
    **{name: None for name in PQ_DATASETS},
    "UniMiB-SHAR": "configs/candidate_pool.yaml",
}

# 基线缓存（iters=0 随机初始化 R'，5 种子）：strict 破坏制式的判据来源。
BASELINE_UEA = ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl"
BASELINE_CANDIDATE = (
    ROOT / "results" / "candidate_survey" / "baseline" / "baseline_cache.jsonl")
BASELINE_OF: dict[str, Path] = {"UniMiB-SHAR": BASELINE_CANDIDATE}

DELTA_QUANTUM_STEP = 0.005   # δ 上取整步长（2c calibrate 同式）
# S2 划分（折内训练/留出切片）：6 切片 → 00–03 训 / 04–05 留；SCP2 3 切片 → 00–01 / 02。
S2_TEST_COUNT = {6: 2, 3: 1}

KAPPA = 0.595                # 计数成分方差份额（soft_label PROTOCOL §1，31 号脚本同值）
MIN_REGIME_N = 5             # 破坏制式 cell 数下限（按 cell 计，非按差值计）


# ---------------------------------------------------------------- engine task id
def engine_task_id(slice_id: str, budget: int) -> str:
    """切片 + 预算 → 比较器看到的任务 id（`Handwriting_03@120`）。

    预算入 id 而非另开字段：比较器/引擎是任务无关的（铁律 7），"同一切片的不同
    训练量是不同任务"这件事只能靠 id 表达，windows_key_of 再把它映回同一份窗口表征。
    """
    return f"{slice_id}@{int(budget)}"


def split_engine_task_id(tid: str) -> tuple[str, int]:
    """`Handwriting_03@120` → ("Handwriting_03", 120)；无 `@` 则 fail-fast。"""
    if "@" not in tid:
        raise ValueError(f"{tid!r} 不是 engine task id（缺 '@预算'）")
    slice_id, budget = tid.rsplit("@", 1)
    return slice_id, int(budget)


def slice_of(slice_or_tid: str) -> str:
    """engine task id 或切片 id → 切片 id（幂等：已是切片 id 时原样返回）。"""
    return slice_or_tid.split("@", 1)[0]


def dataset_of(slice_or_tid: str) -> str:
    """engine task id 或切片 id → 数据集名（先剥 `@预算`，再剥 `_NN` 序号）。"""
    return slice_of(slice_or_tid).rsplit("_", 1)[0]


# ---------------------------------------------------------------- ctx（预算编码）
def ctx_dim(budgets: Sequence[int] = BUDGETS) -> int:
    """ctx 向量维度 = 预算档数（one-hot 编码，configs/labels.yaml: pq.ctx_encoding）。"""
    return len(tuple(budgets))


def ctx_vector(budget: int, budgets: Sequence[int] = BUDGETS) -> np.ndarray:
    """预算 → one-hot ctx（float32）。比较器只看到一个通用向量，不知道它叫"预算"。"""
    bs = tuple(int(b) for b in budgets)
    b = int(budget)
    if b not in bs:
        raise ValueError(f"未知预算 {b}（合法档 {bs}）")
    v = np.zeros(len(bs), dtype=np.float32)
    v[bs.index(b)] = 1.0
    return v


# ---------------------------------------------------------------- 配置
def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_pq_config(
    labels_path: str | Path | None = None, train_path: str | Path | None = None
) -> dict:
    """读 configs/labels.yaml 的 `pq:` 段 + configs/train.yaml 的 `budgets`，并对账。

    对账（fail-fast，防 split-brain）：两份 YAML 的 budgets 与本模块 BUDGETS 必须
    逐位一致；datasets/training_only/n_slices_per_dataset 与常量一致。任何一处漂移
    都会让 32/33/34 三脚本的缓存键与折划分对不上，宁可开跑前炸。
    """
    labels_cfg = _read_yaml(Path(labels_path) if labels_path else ROOT / "configs" / "labels.yaml")
    train_cfg = _read_yaml(Path(train_path) if train_path else ROOT / "configs" / "train.yaml")
    assert "pq" in labels_cfg, "configs/labels.yaml 缺 pq: 段（MX-3-PQ 配置真源）"
    cfg = labels_cfg["pq"]
    assert cfg["budgets"] == list(BUDGETS), \
        f"labels.yaml pq.budgets={cfg['budgets']} ≠ pq_common.BUDGETS={list(BUDGETS)}"
    assert train_cfg.get("budgets") == list(BUDGETS), \
        f"train.yaml budgets={train_cfg.get('budgets')} ≠ pq_common.BUDGETS={list(BUDGETS)}"
    assert cfg["datasets"] == list(PQ_DATASETS), "labels.yaml pq.datasets 与 PQ_DATASETS 不一致"
    assert cfg["training_only"] == list(TRAINING_ONLY), "pq.training_only 与 TRAINING_ONLY 不一致"
    assert cfg["n_slices_per_dataset"] == N_SLICES, "pq.n_slices_per_dataset 与 N_SLICES 不一致"
    assert cfg["ctx_encoding"] == "onehot", f"未实现的 ctx 编码 {cfg['ctx_encoding']!r}"
    return cfg


def _labels_cfg() -> dict:
    return _read_yaml(ROOT / "configs" / "labels.yaml")


# ---------------------------------------------------------------- 注册表 / 切片
_REGISTRY_CACHE: dict[str, SourcePoolRegistry] = {}


def registry_for(name: str) -> SourcePoolRegistry:
    """按 REG_OF 分派注册表（进程内缓存；两文件各是各池的唯一真源）。"""
    if name not in REG_OF:
        raise ValueError(f"{name!r} 不在 MX-3-PQ 池（{PQ_DATASETS}）")
    rel = REG_OF[name]
    cache_key = rel or "__default__"
    if cache_key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[cache_key] = load_registry(ROOT / rel) if rel else load_registry()
    return _REGISTRY_CACHE[cache_key]


def n_slices_for(name: str) -> int:
    """该数据集在 PQ 里的切片数 = min(注册表 planned_tasks, N_SLICES)。

    SCP2 的 planned_tasks=3 < 6，取注册表值（协议：**不覆盖**注册表，切片定义
    必须与 step0/2b/2c 的 canonical R' 缓存逐字段一致）。
    """
    return min(registry_for(name).get(name).planned_tasks, N_SLICES)


def pq_slice_ids(name: str) -> list[str]:
    """该数据集参与 PQ 的切片 id 列表（task_slicer 的 f"{pool}_{i:02d}" 约定）。"""
    return [f"{name}_{i:02d}" for i in range(n_slices_for(name))]


def all_pq_slices(names: Sequence[str] = PQ_DATASETS) -> list[str]:
    """给定数据集集合的全部 PQ 切片 id（零数据加载，只读注册表）。"""
    return [s for n in names for s in pq_slice_ids(n)]


@dataclass(eq=False)
class PQDataset:
    """一个 PQ 数据集的装载结果（X/y 全量 + 注册表登记项 + 前 N_SLICES 个切片）。"""

    name: str
    X: np.ndarray
    y: np.ndarray
    entry: Optional[DatasetEntry]
    registry: Optional[SourcePoolRegistry]
    slices: list[SlicedTask]


def load_pq_datasets(names: Sequence[str] = PQ_DATASETS) -> dict[str, PQDataset]:
    """装载 PQ 数据集 + 切片（**只在 smoke/pod 路径调用**，需要 aeon 与 uea_cache）。

    切片口径（铁律 3）：`slice_source_tasks(entry, reg, y)` 切满注册表 planned_tasks
    后取前 N_SLICES 个——绝不改 n_tasks，否则切片 RNG 前缀变、既有 R' 缓存全失效。
    铁律 6 自检：该数据集不得是所在注册表的目标池成员（load_source_dataset 亦 fail-fast）。
    """
    out: dict[str, PQDataset] = {}
    for name in names:
        reg = registry_for(name)
        assert name not in reg.names(ROLE_TARGET), \
            f"{name} 是目标池成员，绝不走源加载路径（铁律 6）"
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _eff, tasks = slice_source_tasks(entry, reg, y)
        slices = list(tasks[:N_SLICES])
        assert [t.task_id for t in slices] == pq_slice_ids(name), (
            f"{name} 切片 id 与注册表预期不符："
            f"{[t.task_id for t in slices]} vs {pq_slice_ids(name)}"
        )
        out[name] = PQDataset(name=name, X=X, y=y, entry=entry, registry=reg,
                              slices=slices)
    return out


# ---------------------------------------------------------------- tasks.json
def write_tasks_json(datasets, path: str | Path, *, eval_ratio: float | None = None) -> Path:
    """落盘任务定义（SlicedTask 全字段 + canonical 划分下的 n_train/n_eval）。

    n_train/n_eval 走 canonical split_seed（canonical_seeds 的第二个分量，全 CLS 共用），
    与 evaluate_rprime 内部的划分逐位一致——δ 的量子下限 1/n_eval 因此可离线算。
    """
    path = Path(path)
    ratio = eval_ratio if eval_ratio is not None else _labels_cfg()["eval_ratio"]
    items = list(datasets.values()) if isinstance(datasets, dict) else list(datasets)
    tasks: dict[str, dict] = {}
    for ds in items:
        for t in ds.slices:
            _Xt, yt = materialize_task(t, ds.X, ds.y)
            tr, ev = split_task_train_eval(
                yt, t.task_type, eval_ratio=ratio,
                seed=canonical_seeds(t.task_id, "__split__")[1])
            tasks[t.task_id] = {**dataclasses.asdict(t), "dataset": ds.name,
                                "n_train": int(len(tr)), "n_eval": int(len(ev))}
    payload = {"pool": [ds.name for ds in items], "budgets": list(BUDGETS),
               "n_slices_per_dataset": N_SLICES, "eval_ratio": ratio, "tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_tasks_json(path: str | Path) -> dict[str, dict]:
    """tasks.json → {slice_id: task_dict}（33/34 无需重切片即可拿到任务定义）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]


# ---------------------------------------------------------------- CLS 目录 / 种子
def catalog_for_slice(
    space: CLSSpace, slice_id: str, shared: list[tuple[CLS, str]] | None = None
) -> list[tuple[str, CLS, str]]:
    """每切片 20 个候选：10 shared（探针目录前 10）+ 10 random（task_id 派生）。

    与 2b/2c 的 build_candidates 同口径（同目录、同 seed、同去重规则）——step0/工作点
    既有 canonical R' 因此逐键命中；random 按 task_id 派生（与池组成无关，铁律 3）。
    返回 [(origin, CLS, cls_key)]，shared 在前。
    """
    shared_list = shared if shared is not None else load_probe_cls(SHARED_CATALOG, N_SHARED)[0]
    shared_keys = {k for _c, k in shared_list}
    rnd = sample_random_cls(space, slice_id, n=N_RANDOM, seed=CLS_SEED,
                            exclude_keys=shared_keys)
    return ([(ORIGIN_SHARED, c, k) for c, k in shared_list]
            + [(ORIGIN_RANDOM, c, cls_key(c)) for c in rnd])


def ggs_cls() -> CLS:
    """AutoCL Table 5 的 GGS（本空间合法化版）——真值网格里的强默认锚。

    惰性导入 scripts/m1_ggs_har：该模块导入期会拉起 torch/encoder registry，
    pq_common 作为库模块不在导入期承担这份开销。
    """
    from scripts.m1_ggs_har import GGS

    return GGS


def canonical_seeds(slice_id: str, key: str) -> tuple[int, int]:
    """canonical (weight_seed, split_seed)：2c canonical_seeds 逐字同式。

    split_seed 不含 cls_key —— 同一切片全部 CLS 共用一个 train/eval 划分，
    配对比较才在同一评估集上进行。
    """
    return run_seed_for(slice_id, key), run_seed_for(slice_id, "train_eval_split")


def noise_seed(slice_id: str, key: str, rep: int) -> int:
    """噪声探针的第 rep 个额外 weight seed（2c 约定 f"{cls_key}:noise{rep}"）。

    与 canonical（cls_key）流天然不相交，历史缓存零污染。split 仍走 canonical。
    """
    return run_seed_for(slice_id, f"{key}:noise{rep}")


# ---------------------------------------------------------------- 记录装配
def _as_slice_ids(obj: Any) -> list[str]:
    """把各种"任务集合"形状统一摊平成切片 id 列表。

    接受的形状（33/34 手头真实拿得到的那几种）：
    - `"Handwriting_03"` / `"Handwriting_03@120"`（engine tid 先剥 `@预算`）；
    - `SlicedTask` / `PQDataset`；
    - **`load_tasks_json` 的输出 `{slice_id: task_dict}` → 取键**（值是任务字典，
      递归进去会摊到 `classes: [0, 1]` 的元素 `0` 上炸掉——这条是本函数最容易踩的坑）；
    - 单条任务字典（`dataclasses.asdict(SlicedTask)` / tasks.json 的值）→ 取 `task_id`；
    - `{数据集: [切片 id]}` / `{名字: PQDataset}` → 递归值；
    - 上述任意组合的 list/tuple/set/生成器。

    其它叶子类型（int、None、…）**不**回落到"当可迭代对象试试"——直接 ValueError
    报清楚收到了什么，免得在深层递归里抛出无从追溯的 TypeError。
    """
    if isinstance(obj, str):
        return [slice_of(obj)]
    if isinstance(obj, PQDataset):
        return [t.task_id for t in obj.slices]
    if isinstance(obj, SlicedTask):
        return [obj.task_id]
    if isinstance(obj, Mapping):
        if "task_id" in obj:                 # 单条任务字典
            return [slice_of(str(obj["task_id"]))]
        values = list(obj.values())
        if values and all(isinstance(v, Mapping) for v in values):
            return [slice_of(str(k)) for k in obj]   # {slice_id: task_dict}：键即切片 id
        return [s for v in values for s in _as_slice_ids(v)]
    if hasattr(obj, "__iter__"):
        return [s for x in obj for s in _as_slice_ids(x)]
    raise ValueError(
        f"无法解释成切片 id：{obj!r}（{type(obj).__name__}）——接受 str / "
        "SlicedTask / PQDataset / tasks.json 任务字典（或 {slice_id: 任务字典}），"
        "以及它们的列表/字典"
    )


def build_records(
    cache: dict[tuple[str, str, int], dict],
    encoder: CLSEncoder,
    *,
    budgets: Sequence[int] = BUDGETS,
    slices: Iterable | None = None,
    origins: Sequence[str] = (ORIGIN_SHARED, ORIGIN_RANDOM),
) -> list[ClsRecord]:
    """R' 缓存 → ClsRecord（task_id 改写成 engine task id）。

    每个预算各调一次 comparator_records（它按 iters 精确取，铁律 2）；缓存里缺的
    预算自然缺席（不补 NaN，也不报错——32 分批灌数据时 33 仍可先跑已有预算）。

    **`origins` 默认剔掉 GGS**（协议 §2 的目录 = 10 shared + 10 random；GGS 是
    34 号的**外部锚臂**，不是比较器的材料）：留着它每个 (切片, 预算) 就成了 21 条
    CLS，配对数与计划（训练 2340 / 考侧 360）对不上，而且 GGS 在真值里常年贴顶
    ——等于白送比较器一条"永远赢"的样本。要做诊断/消融时显式放行即可。
    """
    keep = None if slices is None else set(_as_slice_ids(slices))
    keep_origin = None if origins is None else set(origins)
    out: list[ClsRecord] = []
    for b in budgets:
        for r in comparator_records(cache, encoder, iters=int(b)):
            if keep is not None and r.task_id not in keep:
                continue
            if keep_origin is not None and r.origin not in keep_origin:
                continue
            out.append(dataclasses.replace(r, task_id=engine_task_id(r.task_id, b)))
    return out


# ------------------------------------------------- 窗口条数（settf_windows 旋钮）
def n_instances_from_meta(meta: Any) -> Optional[int]:
    """task_features npz 的 `META` → 该切片的实例数 N（META[3] = log10 N）。

    形状契约见 interfaces.md「Task 6 → Task 7」：META = [log10 T, log10 C,
    n_classes, log10 N]。给不出（缺 META / 长度不足）时返回 None。
    """
    if meta is None:
        return None
    arr = np.asarray(meta, dtype=float).reshape(-1)
    if arr.size < 4 or not np.isfinite(arr[3]):
        return None
    return int(round(float(10.0 ** arr[3])))


def expected_window_count(settf_windows: int,
                          n_instances: Optional[int] = None) -> int:
    """npz 里**应该**有多少个窗口 = min(settf_windows, 切片实例数)。

    `task_embed.encode_task_windows` 取 `m = min(sample_num, N)`——源池里
    Cricket_02 只有 90 个实例，settf_windows=100 时它**只可能**编出 90 行。
    所以"预注册旋钮是否真的生效"的判据不是 `== settf_windows`（那会让这一格
    永远判红），而是这个封顶值；N ≥ 旋钮的 44/45 格上二者等价。
    N 未知（老 npz 无 META）时退回旋钮本身。
    """
    w = int(settf_windows)
    return w if n_instances is None else min(w, int(n_instances))


def windows_key_of(records: Iterable[ClsRecord]) -> dict[str, str]:
    """{engine_tid: 切片 id}——3 个预算共用同一份 SetTF 窗口表征（编码只做一次）。"""
    return {r.task_id: slice_of(r.task_id) for r in records}


def context_of(
    records: Iterable[ClsRecord], budgets: Sequence[int] = BUDGETS
) -> dict[str, np.ndarray]:
    """{engine_tid: ctx one-hot}——预算是模型唯一能看到的"这条轨迹训了多久"。"""
    return {r.task_id: ctx_vector(split_engine_task_id(r.task_id)[1], budgets)
            for r in records}


# ---------------------------------------------------------------- 折 / 划分
def fold_split(target: str) -> tuple[list[str], str]:
    """7 折 leave-one-dataset-out 的一折：(训练侧 7 个数据集, 考侧 1 个)。

    铁律 6：考侧数据集从"离线一切训练"（TS2Vec fit / label / δ 标定 / 比较器训练）
    里整体抠除，只在考试时前向编码。SCP2 是低权 EEG，永远在训练侧、绝不作靶。
    """
    if target in TRAINING_ONLY:
        raise ValueError(
            f"{target!r} 是 training-only 数据集（源池 v2：只供 label 与全局先验，"
            "不作 LOO 折、不入靶池），不能当折的考侧")
    if target not in FOLD_TARGETS:
        raise ValueError(f"{target!r} 不是合法折靶（合法：{FOLD_TARGETS}）")
    return [d for d in PQ_DATASETS if d != target], target


def s2_split(datasets) -> tuple[list[str], list[str]]:
    """折内 S2 划分：每数据集前 4 切片训 / 后 2 留（SCP2 3 切片 → 前 2 训 / 后 1 留）。

    入参可以是切片 id 序列、{数据集: 切片}、PQDataset/SlicedTask 容器——一律摊平后
    按数据集分组。切片数不在 {6, 3} 之外时 fail-fast（协议只定义了这两档）。
    """
    by_ds: dict[str, list[str]] = defaultdict(list)
    for s in _as_slice_ids(datasets):
        by_ds[dataset_of(s)].append(s)
    train: list[str] = []
    test: list[str] = []
    for _d, sl in sorted(by_ds.items()):
        sl = sorted(set(sl))
        if len(sl) not in S2_TEST_COUNT:
            raise ValueError(
                f"{_d} 有 {len(sl)} 个切片，协议只定义了 {sorted(S2_TEST_COUNT)} 两档")
        n_test = S2_TEST_COUNT[len(sl)]
        train += sl[:len(sl) - n_test]
        test += sl[len(sl) - n_test:]
    return sorted(train), sorted(test)


# ---------------------------------------------------------------- δ
def calibrate_delta(diffs: Sequence[float], quantum: float) -> float:
    """δ = max(P95(|重复对差|), 2×量子) 上取整到 0.005（2c calibrate 逐字同式）。

    P95 索引规则与 2c / 7_swa_noise_probe._p95 完全一致：sorted 后取
    `min(n-1, int(round(0.95*(n-1))))`（**不是** numpy 的插值分位数）——换规则会让
    历史标定值对不上，δ 是 label 存活率的直接开关。

    **`quantum` 必须是参与标定的全部记录里 1/n_eval 的最大值**（2c 写作
    `2.0 * max(quanta)`，quanta 收集自该数据集所有贡献记录）：量子下限的意义是
    "ACC 的最小可分辨步长"，取单条记录（或平均）会给出偏小的下限，进而静默放行
    一批本该被 δ 弃掉的同分噪声配对。本函数只做 `max(P95, 2×quantum)`，取 max
    的责任在调用方。
    """
    ds = sorted(abs(float(d)) for d in diffs)
    if not ds:
        raise ValueError("无可用重复对，δ 标定失败（先跑 --noise-probe）")
    p95 = ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]
    raw = max(p95, 2.0 * float(quantum))
    return float(np.ceil(raw / DELTA_QUANTUM_STEP) * DELTA_QUANTUM_STEP)


def load_delta_calibration(path: str | Path) -> dict[tuple[str, int], float]:
    """delta_calibration.json → {(dataset, budget): δ}（预算键 JSON 里是字符串）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(d, int(b)): float(rec["delta"])
            for d, per_b in raw.items() for b, rec in per_b.items()}


def delta_map_for_fold(
    calibration: dict[tuple[str, int], float],
    slices: Iterable,
    budgets: Sequence[int] = BUDGETS,
) -> dict[str, float]:
    """{(dataset, budget): δ} + 切片 → {engine_tid: δ}（引擎按任务查表，铁律 7）。

    缺 (dataset, budget) 直接 KeyError——拿 290 口径 δ 去配 60 口径 label 是
    2c 已写死要防的事故，宁可开跑前炸。
    """
    out: dict[str, float] = {}
    for s in _as_slice_ids(slices):
        d = dataset_of(s)
        for b in budgets:
            b = int(b)
            if (d, b) not in calibration:
                raise KeyError(
                    f"δ 标定缺 ({d}, {b})——先跑 32 号 --calibrate 并回填 "
                    "configs/labels.yaml: pq.delta_per_dataset_budget")
            out[engine_task_id(s, b)] = float(calibration[(d, b)])
    return out


# ---------------------------------------------------------------- 基线带
def baseline_bands(name: str, path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """{切片 id: (基线均值, 带半宽 2·max(σ, 量子))}（iters=0 随机初始化 R' 的 5 种子）。

    与 scripts/e_ablation_common.baseline_bands / 2c build_baseline_bands 同式，
    只换缓存来源：UEA 走 results/baseline_probe，UniMiB-SHAR（候选池 npz）走
    results/candidate_survey/baseline。strict 破坏判据 = R' < 均值 − 带。
    数据集名按 dataset_of 精确匹配（`+c` 覆盖档任务不混入）；本数据集的 PQ 切片
    缺任何一个 → ValueError（缺基线就没有制式判据，噪声模型会静默降级）。
    """
    src = Path(path) if path is not None else BASELINE_OF.get(name, BASELINE_UEA)
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(src):
        if r.get("r_prime") is not None and dataset_of(r["task_id"]) == name:
            rows_by_task[r["task_id"]].append(r)
    out: dict[str, tuple[float, float]] = {}
    for tid, rows in rows_by_task.items():
        vals = [float(r["r_prime"]) for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        quantum = max((1.0 / r["n_eval"] for r in rows if r.get("n_eval")), default=0.0)
        out[tid] = (statistics.fmean(vals), 2.0 * max(std, quantum))
    missing = [s for s in pq_slice_ids(name) if s not in out]
    if missing:
        raise ValueError(f"{src} 缺 {name} 的基线切片 {missing}（无法定 strict 破坏判据）")
    return out


# ---------------------------------------------------------------- E/ctx 消融指派
def n_derangements(n: int) -> int:
    """n 个元素的无不动点排列个数 !n（子阶乘，递推 !n = (n−1)·(!(n−1) + !(n−2))）。

    within 档判读要用它：可用排列少于消融种子数时，"多种子"给不出新的随机性
    ——n=2 只有 1 个（互换），n=3 有 2 个，n=4 起才 ≥9。
    """
    if n < 0:
        raise ValueError(f"n 必须 ≥0，得到 {n}")
    if n < 2:
        return 1 - n       # !0 = 1, !1 = 0
    prev, cur = 1, 0       # (!0, !1)
    for k in range(2, n + 1):
        prev, cur = cur, (k - 1) * (prev + cur)
    return cur


def within_assignment_slices(
    test_tids: Iterable[str], seed: int, *, strict: bool = False
) -> dict[str, str]:
    """within 档：**切片级** derangement，同一排列施加到全部预算（4b within 的 PQ 版）。

    杀"这条轨迹属于哪个切片"的信号、留数据集/域级信号；供体保留本对自己的预算，
    所以它单独消掉的只是切片身份，不动预算通道。要求全部考侧切片同属一个数据集。

    **退化警告（本实验必踩）**：k 个切片只有 !k 个无不动点排列——k=2 时 **!2 = 1**，
    即互换是唯一解，`seed` 完全无作用。MX-3-PQ 的 S2 留出恰好是每数据集 2 个切片
    （04–05），所以在留出切片上跑 5 个消融种子会得到 5 个**逐位相同**的结果：报表
    里那份"零方差"是排列空间的退化，不是"E 通路稳定"的证据，绝不能当误差棒用。
    k=3 也才 2 个排列（同 budget_assignment 的处境）。

    `strict=True`：切片数 < 3（即 !k < 2、种子给不出第二种排列）时直接 ValueError
    ——Task 6/7 若要按种子报方差，应传 strict=True 让它在开跑前炸，而不是事后
    解释一张全 0 的方差表。默认 False（允许 k=2 单点跑，但调用方须自证已知情）。
    """
    tids = sorted(test_tids)
    slices = sorted({slice_of(t) for t in tids})
    dss = {dataset_of(s) for s in slices}
    assert len(dss) == 1, f"within 档要求考侧切片同属一个数据集，得到 {sorted(dss)}"
    if len(slices) < 2:
        raise ValueError(f"within derangement 需要 ≥2 切片，得到 {slices}")
    if strict and len(slices) < 3:
        raise ValueError(
            f"strict：{len(slices)} 个切片只有 {n_derangements(len(slices))} 个无不动点"
            f"排列（{slices}），seed 无作用——多种子会报出不存在的方差。"
            "要么改用 ≥3 切片的考侧集合，要么 strict=False 并在报表里声明单点。"
        )
    rng = np.random.default_rng([seed, 202608])
    perm = derangement(len(slices), rng)
    donor = {s: slices[perm[i]] for i, s in enumerate(slices)}
    return {t: engine_task_id(donor[slice_of(t)], split_engine_task_id(t)[1])
            for t in tids}


def cross_assignment_slices(
    test_tids: Iterable[str], train_tids: Iterable[str], seed: int
) -> dict[str, str]:
    """cross 档：供体切片从**别的数据集**里均匀抽（4b cross 的 PQ 版），预算不变。

    杀域级信号。供体池来自训练侧任务（train_tids，可传 engine tid 或切片 id）。
    """
    by_ds: dict[str, list[str]] = defaultdict(list)
    for s in sorted({slice_of(t) for t in _as_slice_ids(train_tids)}):
        by_ds[dataset_of(s)].append(s)
    rng = np.random.default_rng([seed, 202609])
    out: dict[str, str] = {}
    for t in sorted(test_tids):
        s, b = split_engine_task_id(t)
        others = [x for d, sl in sorted(by_ds.items()) if d != dataset_of(s) for x in sl]
        assert others, f"{t} 没有跨数据集供体候选"
        out[t] = engine_task_id(others[rng.integers(len(others))], b)
    return out


# (60,120,290) 的两个 derangement，写成 budgets 下标的像：
#   seed%2=0 → (120,290,60)、seed%2=1 → (290,60,120)
_BUDGET_DERANGEMENTS = ((1, 2, 0), (2, 0, 1))


def budget_assignment(
    test_tids: Iterable[str], seed: int, budgets: Sequence[int] = BUDGETS
) -> tuple[dict[str, str], bool]:
    """预算档消融：切片不变、**预算错位**（本对的 ctx/E 换成同切片另一预算的）。

    3 个预算只有 2 个 derangement（写死在 _BUDGET_DERANGEMENTS，seed%2 选一），
    所以种子 0/1 之外无新的无不动点排列可用——种子 ≥2 退化为"随机非恒等排列"，
    可能带不动点，由第二个返回值 is_derangement 如实标注（判读时必须区分：带不动点
    的排列有一部分对根本没被消融，Δacc 会被系统性低估）。
    """
    bs = tuple(int(b) for b in budgets)
    if len(bs) == 3 and seed in (0, 1):
        perm = _BUDGET_DERANGEMENTS[seed % 2]
    else:
        rng = np.random.default_rng([seed, 202609])
        while True:
            perm = tuple(int(v) for v in rng.permutation(len(bs)))
            if any(p != i for i, p in enumerate(perm)):
                break
    is_derangement = all(p != i for i, p in enumerate(perm))
    donor_budget = {bs[i]: bs[p] for i, p in enumerate(perm)}
    out = {}
    for t in sorted(test_tids):
        s, b = split_engine_task_id(t)
        out[t] = engine_task_id(s, donor_budget[b])
    return out, is_derangement


# ---------------------------------------------------------------- 噪声模型
class ProbModelPQ:
    """(dataset, budget) 粒度的噪声模型 → prob_fn(engine_tid, key_a, key_b, delta)。

    scripts/31_soft_label_tclsc.ProbModel 的推广：31 每个数据集一个模型（单一工作点
    iters），PQ 的一条轨迹有 3 个预算 checkpoint、噪声尺度随训练量变（step0.5 实测的
    iters U 形地板），故模型键 = (dataset, budget)。其余口径与 31 逐字一致：
    - 制式（"dest" 破坏 / "nd" 非破坏）由 strict_map 给出，破坏制式 **cell 数**
      < MIN_REGIME_N 时并入单制式（按 cell 计，非按差值计）；
    - 任务尺度因子 κ=0.595 的两成分模型（计数成分 ∝ 1/√n_eval + 不随 n 变的地板），
      n_ref 取该 (dataset, budget) 噪声探针任务的平均 n_eval；
    - 配对尺度 s_pair = √((s_a²+s_b²)/2)（soft_labels.pair_probability）。

    入参 cache_rows / noise_rows 是 rprime_cache.jsonl / noise_probe.jsonl 的行；
    strict_map 键为 (engine_tid, cls_key)——制式随预算变，故不能只按切片键。
    """

    def __init__(
        self,
        cache_rows: Sequence[dict],
        noise_rows: Sequence[dict],
        strict_map: dict[tuple[str, str], bool],
        budgets: Sequence[int] = BUDGETS,
        kind: str = "student_t",
    ):
        self.kind = kind
        self.budgets = tuple(int(b) for b in budgets)
        self.strict = dict(strict_map)
        self.n_eval: dict[str, int] = {}
        canonical: dict[tuple[str, str], float] = {}
        for r in cache_rows:
            b = int(r["iters"])
            if b not in self.budgets:
                continue
            tid = engine_task_id(r["task_id"], b)
            if r.get("n_eval"):
                self.n_eval[tid] = int(r["n_eval"])
            rp = r.get("r_prime")
            if rp is None or not np.isfinite(rp) or int(r.get("seed", 0)) != 0:
                continue
            canonical[(tid, r["cls_key"])] = float(rp)
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in noise_rows:
            b = int(r["iters"])
            rp = r.get("r_prime")
            if b not in self.budgets or rp is None or not np.isfinite(rp):
                continue
            cells[(engine_task_id(r["task_id"], b), r["cls_key"])].append(float(rp))

        self.models: dict[tuple[str, int], NoiseModel] = {}
        self.factor: dict[str, float] = {}
        self.calib: dict[str, dict] = {}
        datasets = sorted({dataset_of(tid) for tid, _k in canonical})
        for d in datasets:
            for b in self.budgets:
                diffs: dict[str, list[float]] = {"nd": [], "dest": []}
                n_cells = {"nd": 0, "dest": 0}
                noise_tids: set[str] = set()
                for (tid, key), extra in cells.items():
                    if dataset_of(tid) != d or split_engine_task_id(tid)[1] != b:
                        continue
                    if (tid, key) not in canonical:
                        continue
                    vals = [canonical[(tid, key)]] + extra
                    regime = "dest" if self.strict[(tid, key)] else "nd"
                    n_cells[regime] += 1
                    noise_tids.add(tid)
                    for i, j in combinations(range(len(vals)), 2):
                        diffs[regime].append(vals[i] - vals[j])
                if not (diffs["nd"] or diffs["dest"]):
                    continue     # 该 (dataset, budget) 没有噪声探针材料：不建模型
                if n_cells["dest"] < MIN_REGIME_N:
                    diffs = {"nd": diffs["nd"] + diffs["dest"], "dest": []}
                self.models[(d, b)] = fit_noise_model(
                    diffs, kind=kind, min_regime_n=MIN_REGIME_N)
                n_ref = float(np.mean([self.n_eval[t] for t in noise_tids]))
                for tid, ne in self.n_eval.items():
                    if dataset_of(tid) == d and split_engine_task_id(tid)[1] == b:
                        self.factor[tid] = task_scale_factor(
                            n_eval=ne, n_ref=n_ref, kappa=KAPPA)
                self.calib[f"{d}@{b}"] = {
                    "n_cells": dict(n_cells),
                    "n_diffs": {k: len(v) for k, v in diffs.items()},
                    "abs_diff_median": {
                        k: (float(np.median(np.abs(v))) if v else None)
                        for k, v in diffs.items()},
                    "abs_diff_p95": {
                        k: (float(np.percentile(np.abs(v), 95)) if v else None)
                        for k, v in diffs.items()},
                    "n_ref": n_ref, "noise_tasks": sorted(noise_tids),
                    "model": self.models[(d, b)].to_json(),
                }

    def scale_of(self, engine_tid: str, key: str) -> float:
        m = self.models[(dataset_of(engine_tid), split_engine_task_id(engine_tid)[1])]
        regime = "dest" if (self.strict[(engine_tid, key)] and "dest" in m.scale) else "nd"
        return m.scale[regime] * self.factor[engine_tid]

    def __call__(self, engine_tid: str, key_a: str, key_b: str, delta: float) -> float:
        model = self.models[(dataset_of(engine_tid), split_engine_task_id(engine_tid)[1])]
        return pair_probability(delta, model, self.scale_of(engine_tid, key_a),
                                self.scale_of(engine_tid, key_b))

    def to_json(self) -> dict:
        return {"kind": self.kind, "kappa": KAPPA, "min_regime_n": MIN_REGIME_N,
                "budgets": list(self.budgets), "per_dataset_budget": self.calib}


__all__ = [
    "PQ_DATASETS", "TRAINING_ONLY", "FOLD_TARGETS", "BUDGETS", "N_SLICES",
    "N_SHARED", "N_RANDOM", "CLS_SEED", "SHARED_CATALOG", "OUT", "REG_OF",
    "BASELINE_UEA", "BASELINE_CANDIDATE", "BASELINE_OF", "KAPPA", "MIN_REGIME_N",
    "ROOT", "read_jsonl", "load_rprime_cache",
    "engine_task_id", "split_engine_task_id", "slice_of", "dataset_of",
    "ctx_vector", "ctx_dim", "load_pq_config", "registry_for", "n_slices_for",
    "pq_slice_ids", "all_pq_slices", "PQDataset", "load_pq_datasets",
    "write_tasks_json", "load_tasks_json", "catalog_for_slice", "ggs_cls",
    "canonical_seeds", "noise_seed", "build_records", "windows_key_of",
    "n_instances_from_meta", "expected_window_count",
    "context_of", "fold_split", "s2_split", "calibrate_delta",
    "load_delta_calibration", "delta_map_for_fold", "baseline_bands",
    "within_assignment_slices", "cross_assignment_slices", "budget_assignment",
    "n_derangements",
    "ProbModelPQ",
]
