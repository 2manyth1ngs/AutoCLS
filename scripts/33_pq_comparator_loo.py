"""MX-3-PQ 阶段 33：折特征 + 比较器臂 + 消融 + 判读（results/mx3_pq/PROTOCOL.md §3–§6）。

四段（铁律 3：逐折/逐跑落盘 + --resume 幂等）：
    python scripts/33_pq_comparator_loo.py --fit-folds --resume
        轻 GPU：7 折 + all8 各 fit 一份共享 TS2Vec（600 iters / bs 32 / cap 4096，
        **只在该折训练侧数据集** fit，铁律 6 按折断言）+ 编码全部 45 切片的窗口表征
        （settf_windows 个窗口 → pool_time_axis 封顶 settf_t_max）→ folds/<折>/task_features/
    python scripts/33_pq_comparator_loo.py --fit-per-dataset
        副臂 A1（AutoCTS++ 式）：每数据集在**自身**无标签数据上各 fit 一份 TS2Vec
        （含目标 → iron_law_6_compliant: false，单独报表，绝不进主表）
    python scripts/33_pq_comparator_loo.py --train --arms ... --folds ... --resume
        10 个臂 × 折 × 种子；训练用硬 δ 动态配对（A7 走软 label），评估在目标数据集
        的 δ 过滤单向对 + 反向；五档消融（within / cross / budget / zero_E / zero_ctx）
    python scripts/33_pq_comparator_loo.py --analyze
        逐 (臂, 折) 中位种子档位 + pooled Wilcoxon + 判读码 + 预测对账 + 成本表

冒烟（零 GPU、零真实数据集）：
    python scripts/33_pq_comparator_loo.py --smoke --outdir results/mx3_pq_smoke --epochs 2

臂表（协议 §3；名字即 runs/ 文件名前缀，勿改）：
    A0 main / A1 autocts_style / A2 wo_ts2vec / A3 wo_settf / A4 wo_shared /
    A5 cls_only / A6 cls_b / A7 soft / A8 onehot_oracle(S2) / A0-S2 main_s2(S2)

铁律落点：
- 铁律 1/7/8：CLS 取值、"谁更优"的方向、任务无关性全部留在 space/task_types/comparator，
  本脚本只做装配；数据集 one-hot（A8）**只在本脚本内**构造，绝不进 autocls/**；
- 铁律 2：预算显式进 engine task id 与缓存键，δ 按 (数据集, 预算) 查表；
- 铁律 6：折 TS2Vec 的 fit 池不含考侧数据集（assert_fold_fit_pool + meta 指纹校验）。
  唯一豁免是 all8 折（S2 上界臂 A8/main_s2 的对照，本就不是 zero-shot，单独报）。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder  # noqa: E402
from autocls.comparator.soft_labels import (  # noqa: E402
    SoftTCLSCEngine, soft_dynamic_pairs,
)
from autocls.comparator.tclsc_engine import Pair, with_reversed  # noqa: E402
from autocls.labeling import load_rprime_cache, make_pairs  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import (  # noqa: E402
    TS2VecWrapper, TaskSetEncoder, encode_task_windows, pool_time_axis, task_windows,
)
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
# 档位/达标线/McNemar 与 C2（4b）逐字同源——换口径会让本实验与既有 E 判负结论不可比。
from scripts.e_ablation_common import (  # noqa: E402
    PASS_DELTA, PASS_SIG_COUNT, mcnemar_one_sided, read_jsonl, tier_of,
)

# ---------------------------------------------------------------- 常量（协议写死）
OUT = pq.OUT
ALL8_FOLD = "all8"          # S2 上界臂的 TS2Vec fit 池（含全部 8 个数据集，非 zero-shot）
S2_FOLD = "S2"              # S2 划分的"折名"（runs/S2/、models/S2/）
TS2VEC_OUTPUT_DIMS = 256    # = SetTF dim_input（ahc_engine.py 原码取值）
# 执行偏差沿用 16_feasibility_loo.py:57-72：TS2Vec fit bs 128→32（长序列折池 T 大，
# temporal loss 的 B×(2T)² 在 8 GB 卡上 OOM）。折间单变量原则不受影响，600 iters 不变。
TS2VEC_FIT_BS = 32
N_ABLATION_SEEDS = 5        # 协议 §4：每档 5 个 assignment 种子
DEFAULT_TRIO = ("Handwriting", "UniMiB-SHAR", "Cricket")   # A2/A4/A7 默认折覆盖
WINDOWS_CALIBRATION_MINUTES = 25.0   # 预注册旋钮：>25 min/跑 → settf_windows 100→32
SMOKE_DATASETS = ("Handwriting", "UniMiB-SHAR", "Cricket")


# ---------------------------------------------------------------- 臂表
@dataclass(frozen=True)
class ArmSpec:
    """一个臂的全部构造知识（协议 §3 臂表的可执行形态）。

    task_branch/encoder/e_source/ctx_kind 四元组决定模型形状与特征来源；
    label_mode 决定训练配对（hard = δ 过滤动态配对，soft = ProbModelPQ 概率目标）。
    """

    name: str
    split: str                  # "S1"（7 折 LOO）/ "S2"（oracle 上界单折）
    task_branch: str            # TCLSC.TASK_BRANCHES
    encoder: Optional[str]      # "settf" / "meanpool" / None
    e_source: Optional[str]     # "fold" / "per_dataset" / "unfit" / None（绑定枚举）
    ctx_kind: str               # "budget" / "dataset_budget" / "none"
    label_mode: str = "hard"    # "hard" / "soft"
    random_only: bool = False   # A4：只用 random-10 训练（且关 curriculum）
    seeds: tuple = (0,)
    folds: tuple = ()           # 空 = S1 全部折靶 / S2 单折
    # 特征目录名覆盖：S1 臂用折靶名（None），S2 的 main_s2 用 all8（fit 于全 8 数据集）。
    # e_source 必须留在绑定枚举内，"用了哪个折的特征"另经 feature_fold 表达——Task 7
    # 只拿到 fold="S2" 时无法反推目录，故 meta 里显式写死。
    feature_fold: Optional[str] = None
    # True / False（A1 每数据集自 fit，非合规）/ "s2_oracle"（S2 上界，本就不是 zero-shot）
    iron_law_6_compliant: bool | str = True


E_SOURCES = ("fold", "per_dataset", "unfit", None)   # interfaces 绑定枚举


class ArmUnavailable(RuntimeError):
    """这份缓存喂不出该臂（训练记录为空、δ 过滤后考侧无配对、软 label 材料不足…）。

    不是 bug 而是"材料不够"：--train 捕获后**跳过该 (臂, 折, 种子) 并大声打印**，
    记进 `<outdir>/skipped.json`，其余臂照跑。典型触发是薄冒烟缓存（如 2 CLS/切片
    全是 shared → A4 wo_shared 的 random-only 训练集为空）；真实网格 20 CLS/切片不会。

    **例外见 _require_material：主臂 main 在 S1 上永不静默跳过**——它是判读表的
    唯一数据源，缺了整轮结论就没了，宁可当场炸。
    """


class SoftLabelUnavailable(ArmUnavailable):
    """A7 soft 臂的噪声模型材料不足（探针 cell/重复对太少、或全落同一制式）。"""


ARMS: dict[str, ArmSpec] = {
    s.name: s for s in (
        ArmSpec("main", "S1", "E+ctx", "settf", "fold", "budget", seeds=(0, 1, 2)),
        ArmSpec("autocts_style", "S1", "E+ctx", "settf", "per_dataset", "budget",
                iron_law_6_compliant=False),
        ArmSpec("wo_ts2vec", "S1", "E+ctx", "settf", "unfit", "budget",
                folds=DEFAULT_TRIO),
        ArmSpec("wo_settf", "S1", "E+ctx", "meanpool", "fold", "budget"),
        ArmSpec("wo_shared", "S1", "E+ctx", "settf", "fold", "budget",
                random_only=True, folds=DEFAULT_TRIO),
        ArmSpec("cls_only", "S1", "none", None, None, "none"),
        ArmSpec("cls_b", "S1", "ctx", None, None, "budget"),
        ArmSpec("soft", "S1", "E+ctx", "settf", "fold", "budget",
                label_mode="soft", folds=DEFAULT_TRIO),
        ArmSpec("onehot_oracle", "S2", "ctx", None, None, "dataset_budget",
                iron_law_6_compliant="s2_oracle"),
        ArmSpec("main_s2", "S2", "E+ctx", "settf", "fold", "budget",
                feature_fold=ALL8_FOLD, iron_law_6_compliant="s2_oracle"),
    )
}
ALL_ARMS = tuple(ARMS)
assert all(s.e_source in E_SOURCES for s in ARMS.values()), \
    f"e_source 必须落在 interfaces 绑定枚举 {E_SOURCES} 内"


def feature_fold_of(spec: ArmSpec, fold: str) -> Optional[str]:
    """该 (臂, 折) 实际用的**特征折目录名**（Task 7 据此定位 npz）。

    S1 折内 E → 折靶名；S2 的 main_s2 → "all8"；per_dataset / unfit / 无 E 的臂 → None
    （前两者的特征不按折组织，后者根本不吃 E）。
    """
    if spec.e_source != "fold":
        return None
    return spec.feature_fold or fold


def default_folds(spec: ArmSpec) -> tuple:
    """该臂的默认折覆盖：S2 臂单折；S1 臂用 spec.folds 或全部 7 个折靶。"""
    if spec.split == "S2":
        return (S2_FOLD,)
    return spec.folds or tuple(pq.FOLD_TARGETS)


def ctx_dim_for(spec: ArmSpec, budgets: Sequence[int], datasets: Sequence[str]) -> int:
    """ctx 维度：预算 one-hot / (数据集 ⊕ 预算) one-hot / 无 ctx。"""
    if spec.ctx_kind == "budget":
        return len(tuple(budgets))
    if spec.ctx_kind == "dataset_budget":
        return len(tuple(datasets)) + len(tuple(budgets))
    return 0


def context_for(spec: ArmSpec, tids: Sequence[str], budgets: Sequence[int],
                datasets: Sequence[str]) -> Optional[dict[str, np.ndarray]]:
    """{engine_tid: ctx}——A8 的数据集 one-hot **只在本脚本构造**（铁律 7：
    autocls/** 不得出现数据集概念，引擎眼里 ctx 只是一段浮点向量）。"""
    if spec.ctx_kind == "none":
        return None
    bs = tuple(budgets)
    if spec.ctx_kind == "budget":
        return {t: pq.ctx_vector(pq.split_engine_task_id(t)[1], bs) for t in tids}
    order = list(datasets)
    out: dict[str, np.ndarray] = {}
    for t in tids:
        slc, b = pq.split_engine_task_id(t)
        v = np.zeros(len(order) + len(bs), dtype=np.float32)
        v[order.index(pq.dataset_of(slc))] = 1.0
        v[len(order) + bs.index(int(b))] = 1.0
        out[t] = v
    return out


def filter_train_records(spec: ArmSpec, records: Sequence) -> list:
    """A4 `wo_shared`：训练侧只留 origin=='random' 的记录（shared 目录全部抠掉）。"""
    if not spec.random_only:
        return list(records)
    return [r for r in records if r.origin == "random"]


def curriculum_for(spec: ArmSpec, curriculum_epochs: int) -> int:
    """A4 没有 shared 记录，curriculum（先易后难）无从谈起 → 关闭。"""
    return 0 if spec.random_only else int(curriculum_epochs)


def build_model(spec: ArmSpec, *, cls_dim: int, ctx_dim: int, nz: int, dim_input: int,
                cfg: dict, seed: int, init_path: Optional[Path] = None,
                standardizer: Optional[tuple] = None) -> TCLSC:
    """臂 → TCLSC（构造顺序与 4b run_one 一致：manual_seed → SetTF(init) → TCLSC）。

    torch.manual_seed(seed) 必须在建 SetTF 之前：`set_encoder_init.pt` 全臂共用
    （协议 §3 折间/臂间单变量），比较器主体的初始化则跟随 seed。
    """
    torch.manual_seed(seed)
    if spec.encoder == "settf":
        enc = TaskSetEncoder(dim_input=dim_input, nz=nz)
        if init_path is not None:
            enc.load_state_dict(torch.load(init_path, map_location="cpu",
                                           weights_only=True))
    elif spec.encoder == "meanpool":
        mu, sd = standardizer if standardizer is not None else (None, None)
        enc = MeanPoolTaskEncoder(dim_input=dim_input, mean=mu, std=sd)
    else:
        enc = None
    return TCLSC(cls_dim=cls_dim, set_encoder=enc, embedding_dim=cfg["embedding_dim"],
                 fz=cfg["fz"], ctx_dim=ctx_dim, task_branch=spec.task_branch)


def load_pq_model(path, *, device: str = "cpu") -> tuple[TCLSC, dict]:
    """models/<折>/<臂>_s<种子>.pt → (模型, meta)（interfaces「Task 6 → Task 7」同格式）。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    kind = meta["set_encoder"]
    if kind == "settf":
        enc = TaskSetEncoder(dim_input=TS2VEC_OUTPUT_DIMS, nz=meta["nz"])
    elif kind == "meanpool":
        enc = MeanPoolTaskEncoder(dim_input=TS2VEC_OUTPUT_DIMS)
    else:
        enc = None
    model = TCLSC(cls_dim=meta["cls_dim"], set_encoder=enc,
                  embedding_dim=meta["embedding_dim"], fz=meta["fz"],
                  ctx_dim=meta["ctx_dim"], task_branch=meta["task_branch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


# ================================================================ 阶段 A：折特征
def assert_fold_fit_pool(target: str, fit_datasets: Sequence[str]) -> None:
    """铁律 6：折 TS2Vec 的 fit 池绝不含考侧数据集（all8 折为 S2 上界，显式豁免）。"""
    if target == ALL8_FOLD:
        return
    assert target not in set(fit_datasets), \
        f"铁律 6 违例：考侧 {target} 出现在 fit 池 {sorted(fit_datasets)} 内"


def get_set_encoder_init(path: Path, *, nz: int, dim_input: int = TS2VEC_OUTPUT_DIMS,
                         seed: int = 0) -> TaskSetEncoder:
    """SetTF 全局一份 init（协议 §3：全臂共用；16_feasibility_loo 同语义）。"""
    torch.manual_seed(seed)
    enc = TaskSetEncoder(dim_input=dim_input, nz=nz)
    if Path(path).exists():
        enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), path)
    return enc


def encode_slice_feature(X: np.ndarray, task_type: str, shared: TS2VecWrapper,
                         set_enc, *, sample_num: int, seed: int, t_max: int,
                         n_classes: int, allow_unfit: bool = False) -> dict:
    """一个切片 → task_features npz 的三张表（interfaces「Task 6 → Task 7」形状契约）。

    windows_repr (m, min(T, t_max), 256) float16：SetTF 的输入。时间轴封顶经
    pool_time_axis——注意力代价 ∝ T²，Ethanol/SCP2 级长度不封顶会显存爆（协议 §3）。
    E_init (256,) float32：冻结 init SetTF 的前向（诊断/Task 7 用，训练时 SetTF 联合训）。
    META (4,)：[log10 T, log10 C, n_classes, log10 N]。
    """
    windows = task_windows(X, task_type)          # 铁律 7 分发点
    reprs = encode_task_windows(windows, shared, sample_num=sample_num, seed=seed,
                                allow_unfit=allow_unfit)
    reprs = pool_time_axis(reprs, t_max)
    with torch.no_grad():
        e = set_enc(torch.as_tensor(np.asarray(reprs), dtype=torch.float32))
    meta = np.array([np.log10(X.shape[1]), np.log10(X.shape[2]), float(n_classes),
                     np.log10(X.shape[0])], dtype=np.float32)
    return {"windows_repr": np.asarray(reprs).astype(np.float16),
            "E_init": e.cpu().numpy().astype(np.float32).reshape(-1), "META": meta}


def _fit_ts2vec(pool: np.ndarray, ra: dict, device: str):
    from autocls.task_embed import fit_shared_ts2vec
    return fit_shared_ts2vec(pool, n_iters=ra["ts2vec_iters"], device=device,
                             seed=ra["embed_seed"], batch_size=TS2VEC_FIT_BS)


def get_fold_ts2vec(fold_dir: Path, target: str, fit_datasets: list[str], data: dict,
                    ra: dict, device: str) -> TS2VecWrapper:
    """折 TS2Vec：只在训练侧数据集 fit（铁律 6 按折）；已落盘则校验 fit 池指纹后加载。"""
    _b2 = importlib.import_module("scripts.2b_generate_labels_route_a")
    fit_pool = sorted(fit_datasets)
    assert_fold_fit_pool(target, fit_pool)
    path = fold_dir / "shared_ts2vec.pt"
    meta_path = fold_dir / "shared_ts2vec.meta.json"
    if path.exists():
        meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else None)
        if meta is None or meta.get("fit_datasets") != fit_pool:
            sys.exit(f"{path} 的 fit 池 {meta and meta.get('fit_datasets')} ≠ 本折 "
                     f"{fit_pool}：口径漂移，确认后删除该折目录重跑")
        assert_fold_fit_pool(target, meta["fit_datasets"])
        return TS2VecWrapper.load(path, device=device)
    fold_dir.mkdir(parents=True, exist_ok=True)
    per_ds = [(name, data[name].X, None, None) for name in fit_pool]
    pool = _b2.build_ts2vec_pool(per_ds, ra["ts2vec_max_instances_per_dataset"],
                                 ra["embed_seed"])
    t0 = time.time()
    w = _fit_ts2vec(pool, ra, device)
    w.save(path)
    meta_path.write_text(json.dumps({
        "fold": target, "fit_datasets": fit_pool,
        "n_instances": int(pool.shape[0]), "t_max": int(pool.shape[1]),
        "n_iters": ra["ts2vec_iters"], "batch_size": TS2VEC_FIT_BS,
        "embed_seed": ra["embed_seed"],
        "max_instances_per_dataset": ra["ts2vec_max_instances_per_dataset"],
        "device_name": (torch.cuda.get_device_name(0)
                        if device.startswith("cuda") else "cpu"),
        "seconds": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{target}] TS2Vec fit {pool.shape} {time.time() - t0:.0f}s → {path}",
          flush=True)
    return w


def _encode_all_slices(feat_dir: Path, data: dict, shared, set_enc, *, ra: dict,
                       sample_num: int, t_max: int, names: Sequence[str],
                       allow_unfit: bool = False) -> int:
    """把 names 里各数据集的全部 PQ 切片编码进 feat_dir（已存在的 npz 跳过 = --resume）。"""
    from autocls.data.task_slicer import materialize_task
    feat_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        ds = data[name]
        for t in ds.slices:
            path = feat_dir / f"{t.task_id}.npz"
            if path.exists():
                n += 1
                continue
            Xt, _yt = materialize_task(t, ds.X, ds.y)
            feat = encode_slice_feature(
                Xt, t.task_type, shared, set_enc, sample_num=sample_num,
                seed=ra["embed_seed"], t_max=t_max, n_classes=len(t.classes),
                allow_unfit=allow_unfit)
            np.savez_compressed(path, **feat)
            n += 1
    return n


def cmd_fit_folds(outdir: Path, folds: Optional[Sequence[str]], device: str,
                  settf_windows: Optional[int]) -> None:
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text("utf-8"))
    ra, pq_cfg = labels_cfg["route_a"], labels_cfg["pq"]
    sample_num = int(settf_windows or pq_cfg["settf_windows"])
    t_max = int(pq_cfg["settf_t_max"])
    data = pq.load_pq_datasets()
    names = list(pq.PQ_DATASETS)
    set_enc = get_set_encoder_init(outdir / "set_encoder_init.pt", nz=ra["nz"],
                                   seed=ra["embed_seed"]).to("cpu").eval()
    targets = list(folds) if folds else list(pq.FOLD_TARGETS) + [ALL8_FOLD]
    for target in targets:
        fold_dir = outdir / "folds" / target
        fit_pool = names if target == ALL8_FOLD else pq.fold_split(target)[0]
        shared = get_fold_ts2vec(fold_dir, target, fit_pool, data, ra, device)
        n = _encode_all_slices(fold_dir / "task_features", data, shared, set_enc,
                               ra=ra, sample_num=sample_num, t_max=t_max, names=names)
        # A2「w/o TS2Vec」：同一批切片改用**未训练**的随机 TS2Vec 编码（allow_unfit）
        unfit = TS2VecWrapper(input_dims=1, output_dims=TS2VEC_OUTPUT_DIMS,
                              device=device, seed=ra["embed_seed"])
        unfit.freeze()
        n_u = _encode_all_slices(fold_dir / "task_features_unfit", data, unfit, set_enc,
                                 ra=ra, sample_num=sample_num, t_max=t_max, names=names,
                                 allow_unfit=True)
        print(f"[{target}] 特征 {n} + 未训练 {n_u} 齐", flush=True)
        del shared, unfit
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def cmd_fit_per_dataset(outdir: Path, device: str,
                        settf_windows: Optional[int]) -> None:
    """副臂 A1（AutoCTS++ 式）：每数据集在自身数据上 fit 一份 TS2Vec（含目标）。

    **明标 non-compliant**：这条通路故意违反铁律 6（目标数据集参与了自己的 E 训练），
    只用于回答 Q5「AutoCTS++ 式 E vs 铁律 6 E」，单独报表、不进主表。
    """
    _b2 = importlib.import_module("scripts.2b_generate_labels_route_a")
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text("utf-8"))
    ra, pq_cfg = labels_cfg["route_a"], labels_cfg["pq"]
    sample_num = int(settf_windows or pq_cfg["settf_windows"])
    t_max = int(pq_cfg["settf_t_max"])
    data = pq.load_pq_datasets()
    set_enc = get_set_encoder_init(outdir / "set_encoder_init.pt", nz=ra["nz"],
                                   seed=ra["embed_seed"]).to("cpu").eval()
    ts_dir, feat_dir = outdir / "per_dataset_ts2vec", outdir / "per_dataset_features"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for name in pq.PQ_DATASETS:
        path, meta_path = ts_dir / f"{name}.pt", ts_dir / f"{name}.meta.json"
        if path.exists():
            w = TS2VecWrapper.load(path, device=device)
        else:
            pool = _b2.build_ts2vec_pool([(name, data[name].X, None, None)],
                                         ra["ts2vec_max_instances_per_dataset"],
                                         ra["embed_seed"])
            t0 = time.time()
            w = _fit_ts2vec(pool, ra, device)
            w.save(path)
            meta_path.write_text(json.dumps({
                "protocol": "autocts_style_per_dataset", "fit_datasets": [name],
                "iron_law_6_compliant": False, "n_instances": int(pool.shape[0]),
                "t_max": int(pool.shape[1]), "n_iters": ra["ts2vec_iters"],
                "batch_size": TS2VEC_FIT_BS, "embed_seed": ra["embed_seed"],
                "device_name": (torch.cuda.get_device_name(0)
                                if device.startswith("cuda") else "cpu"),
                "seconds": round(time.time() - t0, 1),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        n = _encode_all_slices(feat_dir, data, w, set_enc, ra=ra,
                               sample_num=sample_num, t_max=t_max, names=[name])
        print(f"[{name}] per-dataset TS2Vec + {n} 切片特征（non-compliant 副臂）",
              flush=True)
        del w
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ================================================================ 阶段 B：配对/消融
def margin_for(task_type: str):
    """铁律 8：方向经 task_types；分类 = R'_a − R'_b（>0 ⇔ a 优）。"""
    if task_type == TASK_CLASSIFICATION:
        return lambda ra, rb: ra - rb
    raise NotImplementedError(f"task_type={task_type!r} 的 margin 未定义")


def forward_pairs_pq(records: Sequence, task_type: str,
                     delta_by_tid: dict[str, float]) -> list[Pair]:
    """静态单向评估对（4b forward_pairs_by_task 的 PQ 版：δ 按 **engine task** 查表）。

    反向样本在按任务过滤**之后**加（with_reversed），保证 forward/reverse 结构
    可用于 McNemar——正反向同属一个无序对，独立计入会虚增显著性。
    """
    by_task: dict[str, list] = defaultdict(list)
    for r in records:
        by_task[r.task_id].append(r)
    pairs: list[Pair] = []
    for tid, rs in sorted(by_task.items()):
        rprime = {r.key: r.r for r in rs}
        origin = {r.key: r.origin for r in rs}
        vec = {r.key: r.vec for r in rs}
        raw, _stats = make_pairs(rprime, task_type, delta=delta_by_tid[tid],
                                 origin=origin)
        pairs += [Pair(task_id=tid, vec_a=vec[p["a"]], vec_b=vec[p["b"]], y=p["y"],
                       kind=p["kind"]) for p in raw]
    return pairs


def _slices_by_dataset(tids: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, set] = defaultdict(set)
    for t in tids:
        out[pq.dataset_of(t)].add(pq.slice_of(t))
    return {d: sorted(s) for d, s in out.items()}


def within_map(test_tids: Sequence[str], seed: int, *, strict: bool,
               eligible: Sequence[str]) -> dict[str, str]:
    """within 档指派：**逐数据集**做切片级 derangement 后合并。

    pq.within_assignment_slices 要求考侧同属一个数据集（S1 天然满足）；S2 的考侧
    横跨 8 个数据集，故按数据集分组分别求错位再并表——切片身份被打乱、数据集身份
    与预算都保持不变，语义与 S1 的 within 一致。
    """
    out: dict[str, str] = {}
    for d, _sl in sorted(_slices_by_dataset(test_tids).items()):
        if d not in eligible:
            continue
        tids = [t for t in test_tids if pq.dataset_of(t) == d]
        out.update(pq.within_assignment_slices(tids, seed, strict=strict))
    return out


def _agg(rows: list[dict], **flags) -> dict:
    """档位聚合（4b agg 同式）+ tier_of。达标线 PASS_DELTA/PASS_SIG_COUNT 来自 C2。

    n_rows=1 的档（zero_E/zero_ctx 无随机性、退化 within 只有唯一错位排列）**不可能**
    达到 pass（n_sig ≥ 4 于 5 种子）——这是如实反映"没有多重随机性可报"，
    绝不为了凑档位复制 5 行相同结果当方差用。
    """
    deltas = [r["delta_acc"] for r in rows]
    n_sig = sum(r["mcnemar"]["p"] < 0.05 for r in rows)
    block = {
        "rows": rows, "n_rows": len(rows),
        "mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)), "n_sig": int(n_sig),
        "pass": bool(np.mean(deltas) >= PASS_DELTA and n_sig >= PASS_SIG_COUNT),
    }
    block["tier"] = tier_of(block)
    block.update(flags)
    return block


def _normal_block(base: dict, base_fwd: np.ndarray, directed: Sequence[Pair],
                  n_fwd: int, budgets: Sequence[int]) -> dict:
    """normal 档：整体 acc/loss + 单向对二项检验 + 按预算/配对种类分层。"""
    acc_fwd = float(np.mean(base_fwd)) if n_fwd else float("nan")
    p = (float(binomtest(int(base_fwd.sum()), n_fwd, 0.5,
                         alternative="greater").pvalue) if n_fwd else float("nan"))
    by_budget = {}
    for b in budgets:
        m = np.array([pq.split_engine_task_id(directed[i].task_id)[1] == int(b)
                      for i in range(n_fwd)], dtype=bool)
        by_budget[str(int(b))] = {"n": int(m.sum()),
                                  "acc": float(base_fwd[m].mean()) if m.any()
                                  else float("nan")}
    kinds = np.array([directed[i].kind for i in range(n_fwd)])
    by_kind = {k: {"n": int((kinds == k).sum()),
                   "acc": float(base_fwd[kinds == k].mean())}
               for k in sorted(set(kinds.tolist()))}
    return {"acc": base["acc"], "loss": base["loss"], "n": len(directed),
            "n_forward": n_fwd, "acc_forward": acc_fwd, "binom_p_forward": p,
            "by_budget": by_budget, "by_kind": by_kind}


def ablation_block(engine, directed: Sequence[Pair], n_fwd: int,
                   test_tids: Sequence[str], train_tids: Sequence[str],
                   budgets: Sequence[int], *, cross_channel: Optional[str] = None,
                   n_seeds: int = N_ABLATION_SEEDS) -> tuple[dict, dict]:
    """五档消融（协议 §4）：不重训、只改测试输入。返回 (档位块, 逐档单向 correct 向量)。

    - within：目标内切片错位（b 不变）——任务级；
    - cross：**决定性**——E（或数据集身份 ctx）换成训练数据集随机切片的，b 不变；
    - budget：切片内 b 错位（ctx_mode='assign'；E 不动，因为同切片各预算共用一份窗口）；
    - zero_E / zero_ctx：置零。
    任务支路不吃 E 的臂（task_branch ctx/none）跳过全部 E 档并写 null——引擎会对
    "贴着 shuffle 标签、值同 normal"的假消融直接报错，这里提前按臂形状跳过。
    cross_channel='ctx'：ctx 本身编码数据集身份时（A8），跨数据集换的是 ctx 而非 E。
    """
    uses_E = bool(engine.model.uses_E)
    uses_ctx = engine.context_of is not None
    if cross_channel is None:
        cross_channel = "E" if uses_E else None
    test_tids, train_tids = sorted(test_tids), sorted(train_tids)
    base = engine.evaluate(directed, return_correct=True)
    base_correct = np.asarray(base["correct"], dtype=bool)
    base_fwd = base_correct[:n_fwd]
    correct: dict[str, np.ndarray] = {"normal": base_fwd}
    blocks: dict[str, Optional[dict]] = {
        "normal": _normal_block(base, base_fwd, directed, n_fwd, budgets)}

    def row(seed: Optional[int], *, keep: Optional[set] = None, **kw):
        if keep is None:
            pairs, idx, nf = directed, None, n_fwd
        else:
            fwd_idx = [i for i in range(n_fwd) if directed[i].task_id in keep]
            idx = np.array(fwd_idx + [i + n_fwd for i in fwd_idx], dtype=int)
            pairs, nf = [directed[i] for i in idx], len(fwd_idx)
        b_corr = base_correct if idx is None else base_correct[idx]
        res = engine.evaluate(pairs, return_correct=True, **kw)
        abl = np.asarray(res["correct"], dtype=bool)
        base_acc = float(np.mean(b_corr)) if len(b_corr) else float("nan")
        r = {"seed": seed, "acc": res["acc"], "loss": res["loss"],
             "base_acc": base_acc, "delta_acc": base_acc - res["acc"],
             "n_directed": len(pairs), "n_forward": nf,
             "mcnemar": mcnemar_one_sided(b_corr[:nf], abl[:nf])}
        return r, abl[:nf], (None if idx is None else idx[:nf])

    # -------- within（切片级错位；退化规模只报单行，见 _agg docstring）
    groups = _slices_by_dataset(test_tids)
    eligible = [d for d, sl in groups.items() if len(sl) >= 2]
    if not uses_E or not eligible:
        blocks["within"] = None
    else:
        strict = min(len(groups[d]) for d in eligible) >= 3
        keep = ({t for t in test_tids if pq.dataset_of(t) in eligible}
                if len(eligible) < len(groups) else None)
        rows = []
        for s in range(n_seeds if strict else 1):
            amap = within_map(test_tids, s, strict=strict, eligible=eligible)
            r, c, idx = row(s, keep=keep, e_mode="assign", e_assignment=amap)
            rows.append(r)
            correct[f"within_s{s}"] = c
            if idx is not None:
                correct[f"within_s{s}_idx"] = idx
        blocks["within"] = _agg(
            rows, degenerate=not strict, channel="E",
            excluded_datasets=sorted(set(groups) - set(eligible)))

    # -------- cross（决定性档）
    if cross_channel is None or not train_tids:
        blocks["cross"] = None
    else:
        rows = []
        for s in range(n_seeds):
            amap = pq.cross_assignment_slices(test_tids, train_tids, s)
            kw = ({"e_mode": "assign", "e_assignment": amap} if cross_channel == "E"
                  else {"ctx_mode": "assign", "ctx_assignment": amap})
            r, c, _idx = row(s, **kw)
            rows.append(r)
            correct[f"cross_s{s}"] = c
        blocks["cross"] = _agg(rows, channel=cross_channel)

    # -------- budget（P/Q 对应物；3 预算只有 2 个 derangement，逐行标注）
    if not uses_ctx or len(tuple(budgets)) < 2:
        blocks["budget"] = None
    else:
        rows = []
        for s in range(n_seeds):
            amap, is_der = pq.budget_assignment(test_tids, s, budgets)
            r, c, _idx = row(s, ctx_mode="assign", ctx_assignment=amap)
            r["is_derangement"] = bool(is_der)
            rows.append(r)
            correct[f"budget_s{s}"] = c
        blocks["budget"] = _agg(rows, channel="ctx")

    # -------- zero 档（无随机性：单行 + deterministic 标注）
    if uses_E:
        r, c, _idx = row(None, e_mode="zero")
        correct["zero_E"] = c
        blocks["zero_E"] = _agg([r], deterministic=True, channel="E")
    else:
        blocks["zero_E"] = None
    if uses_ctx:
        r, c, _idx = row(None, ctx_mode="zero")
        correct["zero_ctx"] = c
        blocks["zero_ctx"] = _agg([r], deterministic=True, channel="ctx")
    else:
        blocks["zero_ctx"] = None
    return blocks, correct


# ================================================================ 阶段 B：装配
class PQContext:
    """一次 --train 会话的共享装载（缓存 / δ 表 / 配置 / CLS 编码器）。"""

    def __init__(self, outdir: Path, *, device: str, epochs: Optional[int] = None,
                 settf_windows: Optional[int] = None,
                 budgets: Optional[Sequence[int]] = None):
        self.outdir = Path(outdir)
        self.device = device
        labels_cfg = yaml.safe_load(
            (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
        self.task_type = labels_cfg["task_type"]
        self.ra, self.pq_cfg = labels_cfg["route_a"], labels_cfg["pq"]
        cfg = yaml.safe_load(
            (ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))
        # --epochs 覆盖时 curriculum 一并夹住：epochs=2 而 curriculum=40 会让整跑停在
        # "只见 shared 配对"的第一阶段，random 记录一条都没进过训练（冒烟/标定失真）。
        # 夹到 **epochs 的一半**（下限 1）而不是 epochs 本身：夹成 epochs 时全部 epoch
        # 仍在 curriculum 阶段，薄目录（每切片 1 个 shared）里凑不出 shared–shared 对，
        # 于是每个 epoch 配出 0 对、模型一步没训——冒烟看着"跑通"其实什么都没学。
        self.cfg = dict(cfg)
        if epochs:
            self.cfg["epochs"] = int(epochs)
            self.cfg["curriculum_epochs"] = min(int(cfg["curriculum_epochs"]),
                                                max(1, int(epochs) // 2))
        self.nz = int(self.ra["nz"])
        self.settf_windows = int(settf_windows or self.pq_cfg["settf_windows"])
        self.settf_t_max = int(self.pq_cfg["settf_t_max"])
        self.space = CLSSpace.from_yaml()
        self.encoder = CLSEncoder(self.space)
        cache_path = self.outdir / "rprime_cache.jsonl"
        if not cache_path.exists():
            sys.exit(f"缺 {cache_path}：先跑 scripts/32_pq_generate_rprime.py")
        self.cache = load_rprime_cache(cache_path)
        present = sorted({k[2] for k in self.cache})
        self.budgets = (tuple(int(b) for b in budgets) if budgets else
                        (pq.BUDGETS if set(pq.BUDGETS) <= set(present)
                         else tuple(present)))
        self.slices_by_ds = _slices_by_dataset(sorted({k[0] for k in self.cache}))
        self.calibration = pq.load_delta_calibration(
            self.outdir / "delta_calibration.json")
        self.init_path = self.outdir / "set_encoder_init.pt"
        # 只缓存"当前特征目录"的窗口表征（键 = 切片 id）；换目录即清空，见 windows()
        self._windows_cache: dict[str, np.ndarray] = {}
        self._windows_dir: Optional[str] = None

    # ---------------------------------------------------------- 划分 / 记录
    def split_for(self, spec: ArmSpec, fold: str):
        if spec.split == "S1":
            _require_material(
                spec, fold, fold in self.slices_by_ds,
                f"折靶在缓存里没有任何切片（现有 {sorted(self.slices_by_ds)}）")
            train_ds, test_ds = pq.fold_split(fold)
            train_ds = [d for d in train_ds if d in self.slices_by_ds]
            train_slices = [s for d in train_ds for s in self.slices_by_ds[d]]
            test_slices = list(self.slices_by_ds[test_ds])
        else:
            all_slices = [s for d in sorted(self.slices_by_ds)
                          for s in self.slices_by_ds[d]]
            try:
                train_slices, test_slices = pq.s2_split(all_slices)
            except ValueError as exc:
                # s2_split 只定义了 6 切片 / 3 切片两档；薄缓存（每数据集 1 个切片）
                # 给不出 S2 划分——S2 两臂是上界对照，缺席只是少一列，跳过即可。
                _require_material(spec, fold, False, f"S2 划分不可用（{exc}）")
            train_ds, test_ds = sorted(self.slices_by_ds), S2_FOLD
        return train_ds, test_ds, sorted(train_slices), sorted(test_slices)

    def records(self, slices: Sequence[str]) -> list:
        return pq.build_records(self.cache, self.encoder, budgets=self.budgets,
                                slices=list(slices))

    def delta_map(self, slices: Sequence[str]) -> dict[str, float]:
        return pq.delta_map_for_fold(self.calibration, list(slices), self.budgets)

    def robust_delta_map(self, train_ds: Sequence[str],
                         test_slices: Sequence[str]) -> tuple[dict, dict]:
        """稳健读数：考侧对改用**训练数据集** δ 的中位数（不看目标自己的标定）。"""
        med = {}
        for b in self.budgets:
            vals = [self.calibration[(d, int(b))] for d in train_ds
                    if (d, int(b)) in self.calibration]
            if vals:
                med[int(b)] = float(np.median(vals))
        table = {pq.engine_task_id(s, b): med[int(b)]
                 for s in test_slices for b in self.budgets if int(b) in med}
        return med, table

    # ---------------------------------------------------------- 特征
    def feature_dir(self, spec: ArmSpec, fold: str) -> Optional[Path]:
        if spec.e_source == "fold":
            return self.outdir / "folds" / feature_fold_of(spec, fold) / "task_features"
        if spec.e_source == "unfit":
            return (self.outdir / "folds" / (spec.feature_fold or fold)
                    / "task_features_unfit")
        if spec.e_source == "per_dataset":
            return self.outdir / "per_dataset_features"
        return None

    def windows(self, spec: ArmSpec, fold: str,
                slices: Sequence[str]) -> dict[str, np.ndarray]:
        """{切片: windows_repr}（float32；3 个预算共用一份，SetTF 前向只做一次）。

        进程内只缓存**当前一个特征目录**：45 切片 × (100, 152, 256) float32 ≈ 0.65 GiB，
        7 折 + all8 + per_dataset + unfit 十余个目录全留会吃掉 ~8 GiB 宿主内存。
        目录一换就整表清空（换回来会重读盘，代价远小于常驻）。
        """
        feat_dir = self.feature_dir(spec, fold)
        if feat_dir is None:
            return {}
        if str(feat_dir) != self._windows_dir:      # 目录切换即驱逐（内存上限 = 1 个目录）
            self._windows_cache.clear()
            self._windows_dir = str(feat_dir)
        if spec.e_source == "fold":     # 铁律 6 的使用点自检：折 meta 不得含考侧
            ffold = feature_fold_of(spec, fold)
            meta_path = self.outdir / "folds" / ffold / "shared_ts2vec.meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                assert_fold_fit_pool(ffold, meta.get("fit_datasets", []))
        out = {}
        for s in sorted(set(slices)):
            path = feat_dir / f"{s}.npz"
            if not path.exists():
                sys.exit(f"缺特征 {path}：先跑 --fit-folds / --fit-per-dataset")
            if s not in self._windows_cache:
                with np.load(path) as npz:
                    self._windows_cache[s] = npz["windows_repr"].astype(np.float32)
            out[s] = self._windows_cache[s]
        return out

    # ---------------------------------------------------------- 软 label 材料
    def prob_model(self, train_ds: Sequence[str]):
        """(数据集, 预算) 噪声模型（A7 soft 臂的软目标）；材料不足时抛 SoftLabelUnavailable。"""
        noise_path = self.outdir / "noise_probe.jsonl"
        if not noise_path.exists():
            raise SoftLabelUnavailable(
                f"缺 {noise_path}：A7 soft 臂需要噪声探针（32 --noise-probe）")
        keep = set(train_ds)
        cache_rows = [r for r in self.cache.values()
                      if pq.dataset_of(r["task_id"]) in keep]
        noise_rows = [r for r in read_jsonl(noise_path)
                      if pq.dataset_of(r["task_id"]) in keep]
        bands: dict[str, dict] = {}
        strict: dict[tuple, bool] = {}
        for r in cache_rows:
            d = pq.dataset_of(r["task_id"])
            if d not in bands:
                bands[d] = pq.baseline_bands(d)
            rp = r.get("r_prime")
            if rp is None or not np.isfinite(rp):
                continue
            mean, band = bands[d][r["task_id"]]
            strict[(pq.engine_task_id(r["task_id"], int(r["iters"])), r["cls_key"])] = \
                bool(float(rp) < mean - band)
        try:
            pm = pq.ProbModelPQ(cache_rows, noise_rows, strict, self.budgets)
            # 覆盖度自检：ProbModelPQ 对"没有噪声探针材料的 (数据集, 预算)"是静默
            # `continue`，模型表里就没这个键——直到训练第一批配对才 KeyError 炸在
            # soft_labels._make_soft 里。这里提前把它变成可跳过的信号。
            need = {(pq.dataset_of(r["task_id"]), int(r["iters"])) for r in cache_rows
                    if int(r["iters"]) in self.budgets
                    and r.get("r_prime") is not None and np.isfinite(r["r_prime"])}
            missing = sorted(need - set(pm.models))
            if missing:
                raise SoftLabelUnavailable(
                    f"噪声探针未覆盖 (数据集, 预算) {missing}：这些格的软 label 无从算起")
            return pm
        except SoftLabelUnavailable:
            raise
        except Exception as exc:      # noqa: BLE001 —— 见下方 SoftLabelUnavailable
            # 噪声探针材料不足时 fit_noise_model 会以各种形状炸（样本 <4、某制式空样本、
            # 某 (数据集, 预算) 无 cell）——冒烟缓存（1 切片 × 2 CLS）必然踩到。
            # 统一收敛成一个可跳过的信号，别让 A7 的材料问题拖垮整轮 --train。
            raise SoftLabelUnavailable(
                f"噪声模型不可拟合（{type(exc).__name__}: {exc}）：噪声探针 cell/重复对"
                "太少或全落在同一制式") from exc


def _require_material(spec: ArmSpec, fold: str, ok: bool, reason: str) -> None:
    """材料自检：不足则跳过该臂——**主臂 main 在 S1 上例外，仍然硬炸**。

    判读表（协议 §5 全部五个码）只读 A0 = main 的 S1 折；它要是被静默跳过，
    analysis 会安静地给出一个建立在空气上的 AMBIGUOUS。其余臂缺席只是少一列对照，
    如实记进 skipped.json 即可。
    """
    if ok:
        return
    if spec.name == "main" and spec.split == "S1":
        raise AssertionError(f"main/{fold}（主臂，判读表唯一数据源）: {reason}")
    # 只带 reason：打印方 (cmd_train / analyze) 负责加 "<臂>/<折>" 前缀，别叠两遍
    raise ArmUnavailable(reason)


def _history_loss(history: list[dict], idx: int) -> float:
    return float(history[idx]["loss"]) if history else float("nan")


def run_arm(ctx: PQContext, spec: ArmSpec, fold: str, seed: int, *,
            save: bool = True) -> tuple[dict, dict]:
    """训一个 (臂, 折, 种子) 并出全部消融档 → (run json, correct 向量表)。

    save=False 供 --calibrate-windows 的计时跑用：那一跑只有 1 epoch，权重绝不能
    留在 models/ 里冒充正式产物（Task 7 会照单加载）。
    """
    train_ds, test_ds, train_slices, test_slices = ctx.split_for(spec, fold)
    train_records = filter_train_records(spec, ctx.records(train_slices))
    test_records = ctx.records(test_slices)
    _require_material(spec, fold, bool(train_records),
                      "训练记录为空（缓存缺该折数据集，或该臂的 origin 过滤把记录清空了"
                      "——A4 wo_shared 只吃 random-10，薄缓存里可能一条都没有）")
    assert not (set(train_slices) & set(test_slices)), "训练/考侧切片重叠"

    delta_train = ctx.delta_map(train_slices)
    delta_test = ctx.delta_map(test_slices)
    fwd = forward_pairs_pq(test_records, ctx.task_type, delta_test)
    _require_material(spec, fold, bool(fwd),
                      f"δ 过滤后考侧无单向对（{len(test_records)} 条考侧记录，"
                      "同切片 CLS 太少或 R' 差都小于 δ）")
    directed = with_reversed(fwd)
    all_records = list(train_records) + list(test_records)
    wk = pq.windows_key_of(all_records)
    tids = sorted(wk)
    ctx_map = context_for(spec, tids, ctx.budgets, pq.PQ_DATASETS)
    cdim = ctx_dim_for(spec, ctx.budgets, pq.PQ_DATASETS)

    windows = ctx.windows(spec, fold, sorted({pq.slice_of(t) for t in tids}))
    dim_input = (int(next(iter(windows.values())).shape[-1]) if windows
                 else TS2VEC_OUTPUT_DIMS)
    standardizer = None
    if spec.encoder == "meanpool":
        # 统计量只用**训练切片**拟合（留出切片套同一统计量，防泄漏；模块 docstring 语义）
        standardizer = MeanPoolTaskEncoder.fit_standardizer(
            windows, sorted({pq.slice_of(t) for t in train_slices}))
    cls_dim = int(train_records[0].vec.shape[0])
    model = build_model(spec, cls_dim=cls_dim, ctx_dim=cdim,
                        nz=ctx.nz, dim_input=dim_input, cfg=ctx.cfg, seed=seed,
                        init_path=(ctx.init_path if spec.encoder == "settf" else None),
                        standardizer=standardizer)
    engine_cls = SoftTCLSCEngine if spec.label_mode == "soft" else TCLSCEngine
    engine = engine_cls(model, windows, device=ctx.device, lr=ctx.cfg["lr"],
                        batch_size=ctx.cfg["batch_size"], seed=seed,
                        windows_key_of=wk, context_of=ctx_map)

    cur = curriculum_for(spec, ctx.cfg["curriculum_epochs"])
    t0 = time.time()
    if spec.label_mode == "soft":
        margin = margin_for(ctx.task_type)
        prob_fn = ctx.prob_model(train_ds)

        def sampler(epoch: int, rng: np.random.Generator):
            origins = ("shared",) if epoch < cur else None
            pairs = soft_dynamic_pairs(train_records, margin, prob_fn, rng=rng,
                                       origins=origins)
            return [dataclasses.replace(p, w=1.0) for p in pairs]   # 软目标，全权

        history = engine.train_soft(sampler, epochs=ctx.cfg["epochs"])
    else:
        history = engine.train(train_records, epochs=ctx.cfg["epochs"],
                               curriculum_epochs=cur,
                               better_fn=lambda a, b: is_better(a, b, ctx.task_type),
                               delta=delta_train)
    train_seconds = round(time.time() - t0, 1)
    # 训完立刻落权重（消融段崩了也不必重训）；格式 = interfaces 的 Task 7 绑定契约
    if save:
        save_model(ctx.outdir / "models" / fold / f"{spec.name}_s{seed}.pt", model,
                   spec, fold=fold, seed=seed, cls_dim=cls_dim, ctx_dim=cdim,
                   cfg=ctx.cfg, nz=ctx.nz, settf_windows=ctx.settf_windows,
                   settf_t_max=ctx.settf_t_max, budgets=ctx.budgets,
                   ctx_encoding=ctx.pq_cfg["ctx_encoding"])

    cross_channel = "ctx" if spec.ctx_kind == "dataset_budget" else None
    train_tids = [pq.engine_task_id(s, b) for s in train_slices for b in ctx.budgets
                  if pq.engine_task_id(s, b) in wk]
    blocks, correct = ablation_block(engine, directed, len(fwd), tids_of(test_records),
                                     train_tids, ctx.budgets,
                                     cross_channel=cross_channel)

    med, robust_table = ctx.robust_delta_map(train_ds, test_slices)
    fwd_r = forward_pairs_pq(test_records, ctx.task_type, robust_table)
    robust = {"delta_by_budget": {str(k): v for k, v in med.items()},
              "n_forward": len(fwd_r)}
    if fwd_r:
        rob = engine.evaluate(with_reversed(fwd_r), return_correct=True)
        robust.update(acc=rob["acc"], loss=rob["loss"],
                      acc_forward=float(np.mean(rob["correct"][:len(fwd_r)])))
    else:
        # 稳健读数是**副读数**：训练侧 δ 中位数比目标自己的 δ 严时可能一条对都不剩，
        # 空 pairs 会让 SetTF 的 torch.stack 炸（引擎不接受空批）。副读数缺席就缺席，
        # 绝不因此判死整个臂——主读数 normal 仍在。
        robust.update(acc=None, loss=None, acc_forward=None,
                      note="训练侧 δ 中位数过严：考侧无单向对，稳健读数缺席")

    res = {
        "fold": fold, "split": spec.split, "arm": spec.name, "seed": seed,
        "train_datasets": list(train_ds), "test_dataset": test_ds,
        "budgets": [int(b) for b in ctx.budgets],
        "settings": {
            "task_branch": spec.task_branch, "ctx_dim": cdim,
            "ctx_encoding": ctx.pq_cfg["ctx_encoding"], "ctx_kind": spec.ctx_kind,
            "encoder": spec.encoder, "e_source": spec.e_source,
            "feature_fold": feature_fold_of(spec, fold),
            "label_mode": spec.label_mode, "random_only": spec.random_only,
            "settf_windows": ctx.settf_windows,
            "settf_windows_actual": (int(next(iter(windows.values())).shape[0])
                                     if windows else None),
            "settf_t_max": ctx.settf_t_max, "nz": ctx.nz, "dim_input": dim_input,
            "epochs": ctx.cfg["epochs"], "curriculum_epochs": cur,
            "lr": ctx.cfg["lr"], "batch_size": ctx.cfg["batch_size"],
            "iron_law_6_compliant": spec.iron_law_6_compliant,
            "delta_train": _delta_by_dataset_budget(delta_train),
            "delta_test": _delta_by_dataset_budget(delta_test),
            "cross_channel": cross_channel or ("E" if model.uses_E else None),
        },
        "n_train_records": len(train_records), "n_test_records": len(test_records),
        "n_test_pairs_forward": len(fwd), "n_test_pairs_directed": len(directed),
        "train_seconds": train_seconds, "device": ctx.device,
        "bce_first": _history_loss(history, 0), "bce_last": _history_loss(history, -1),
        # 每 epoch 实际配出的训练对数：薄缓存下可能为 0（每任务只剩 1 条记录 → 无从配对），
        # 那种 run 的 normal 只是未训练模型的读数，判读时必须看这个字段。
        "train_pairs_last": (int(history[-1]["n_pairs"]) if history else 0),
        "robust_pooled_delta": robust,
        **blocks,
    }
    return res, correct


def tids_of(records: Sequence) -> list[str]:
    return sorted({r.task_id for r in records})


def _delta_by_dataset_budget(delta_map: dict[str, float]) -> dict[str, float]:
    """{engine_tid: δ} → {"数据集@预算": δ}（落盘紧凑形态，45×3 条不必逐切片存）。"""
    out = {}
    for tid, d in sorted(delta_map.items()):
        slc, b = pq.split_engine_task_id(tid)
        out[f"{pq.dataset_of(slc)}@{b}"] = float(d)
    return out


def save_model(path: Path, model: TCLSC, spec: ArmSpec, *, fold: str, seed: int,
               cls_dim: int, ctx_dim: int, cfg: dict, nz: int, settf_windows: int,
               settf_t_max: int, budgets: Sequence[int], ctx_encoding: str) -> None:
    """interfaces「Task 6 → Task 7 saved-model format」——meta 键集是绑定契约。

    `e_source` 只取绑定枚举 ("fold"|"per_dataset"|"unfit"|null)；`feature_fold`
    （控制方补充键）给出**实际用的特征折目录名**——S2 臂的 fold 是 "S2"，光看它
    找不到 `folds/all8/task_features/`，Task 7 据此键定位。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": {
        "arm": spec.name, "seed": seed, "fold": fold,
        "task_branch": spec.task_branch, "ctx_dim": ctx_dim, "cls_dim": cls_dim,
        "embedding_dim": cfg["embedding_dim"], "fz": cfg["fz"], "nz": nz,
        "set_encoder": spec.encoder, "e_source": spec.e_source,
        "feature_fold": feature_fold_of(spec, fold),
        "settf_windows": settf_windows, "settf_t_max": settf_t_max,
        "budgets": [int(b) for b in budgets], "ctx_encoding": ctx_encoding,
    }}, path)


def _fmt_block(name: str, blk: Optional[dict]) -> str:
    if blk is None:
        return f"{name} —"
    return (f"{name} Δ̄={blk['mean_delta']:+.4f} sig {blk['n_sig']}/{blk['n_rows']} "
            f"{blk['tier']}")


def load_skipped(outdir) -> list[dict]:
    """`<outdir>/skipped.json` → [{arm, fold, seed, reason}]（缺文件= 空表）。"""
    path = Path(outdir) / "skipped.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("skipped", [])


def _write_skipped(outdir: Path, skips: list[dict], attempted: set) -> None:
    """合并历史与本轮：本轮**试过**的组合以本轮结果为准（跑成了就把旧记录抹掉），
    没试过的组合保留历史（分片/续跑时别把别的分片的跳过记录吃掉）。"""
    keep = [r for r in load_skipped(outdir)
            if (r.get("arm"), r.get("fold"), r.get("seed")) not in attempted]
    payload = sorted(keep + skips, key=lambda r: (r["arm"], r["fold"], r["seed"]))
    (Path(outdir) / "skipped.json").write_text(
        json.dumps({"skipped": payload, "n": len(payload)}, indent=2,
                   ensure_ascii=False), encoding="utf-8")


def cmd_train(outdir: Path, *, arms: Sequence[str], folds: Optional[Sequence[str]],
              seeds: Optional[Sequence[int]], resume: bool, device: str,
              epochs: Optional[int], settf_windows: Optional[int]) -> None:
    ctx = PQContext(outdir, device=device, epochs=epochs, settf_windows=settf_windows)
    print(f"缓存 {len(ctx.cache)} 行 | 预算 {ctx.budgets} | 数据集 "
          f"{sorted(ctx.slices_by_ds)} | device={device}", flush=True)
    skips: list[dict] = []          # 本轮跳过的 (臂, 折, 种子)，逐条即时落盘
    attempted: set = set()          # 本轮试过的组合：成功的要把历史 skip 记录清掉
    for name in arms:
        spec = ARMS[name]
        arm_folds = ((folds if spec.split == "S1" else (S2_FOLD,)) if folds
                     else default_folds(spec))
        arm_seeds = seeds if seeds is not None else spec.seeds
        for fold in arm_folds:
            for seed in arm_seeds:
                out_path = outdir / "runs" / fold / f"{name}_s{seed}.json"
                attempted.add((name, fold, int(seed)))
                if resume and out_path.exists():
                    print(f"[{fold}/{name}_s{seed}] 已有结果，跳过", flush=True)
                    continue
                try:
                    res, correct = run_arm(ctx, spec, fold, seed)
                except ArmUnavailable as exc:
                    reason = str(exc)
                    print(f"[SKIP] {name}/{fold}: {reason}（seed {seed}）"
                          "——本臂跳过，其余臂继续；见 skipped.json", flush=True)
                    skips.append({"arm": name, "fold": fold, "seed": int(seed),
                                  "reason": reason})
                    _write_skipped(outdir, skips, attempted)
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                               default=float), encoding="utf-8")
                np.savez_compressed(
                    out_path.with_name(f"{name}_s{seed}_correct.npz"), **correct)
                _write_skipped(outdir, skips, attempted)   # 这次跑成了 → 清历史 skip
                print(f"[{fold}/{name}_s{seed}] {res['train_seconds']}s | "
                      f"normal {res['normal']['acc']:.4f} | "
                      f"{_fmt_block('within', res['within'])} | "
                      f"{_fmt_block('cross', res['cross'])} | "
                      f"{_fmt_block('budget', res['budget'])}", flush=True)


def project_run_minutes(train_seconds: float, total_seconds: float,
                        full_epochs: int) -> float:
    """1 epoch 计时 → 整跑分钟数：**只有训练部分乘 epoch 数**。

    一次标定跑 = 1 个 epoch 的训练 + 一次性开销（≈16 次消融评估、δ 过滤配对、
    稳健读数）。把整跑时间乘 200 会把一次性开销也放大 200 倍，直接把预注册旋钮
    （>25 min/跑 → windows 32）推向错误的一侧，故公式为
    `train_seconds × full_epochs + (total_seconds − train_seconds)`。
    """
    overhead = float(total_seconds) - float(train_seconds)
    return (float(train_seconds) * int(full_epochs) + overhead) / 60.0


def cmd_calibrate_windows(outdir: Path, device: str,
                          settf_windows: Optional[int]) -> None:
    """预注册旋钮标定：折 Handwriting 的 A0 s0 跑一个 epoch，外推整跑墙钟。"""
    ctx = PQContext(outdir, device=device, epochs=1, settf_windows=settf_windows)
    spec, fold = ARMS["main"], "Handwriting"
    full_epochs = yaml.safe_load(
        (ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))["epochs"]
    t0 = time.time()
    res, _correct = run_arm(ctx, spec, fold, 0, save=False)
    one = time.time() - t0
    train_s = float(res["train_seconds"])
    proj = project_run_minutes(train_s, one, full_epochs)
    print(f"[标定] 折 {fold} main s0：整跑 {one:.1f}s = 训练 1 epoch {train_s:.1f}s "
          f"+ 一次性开销（消融/配对）{one - train_s:.1f}s")
    print(f"[标定] {full_epochs} epoch 外推 ≈ {proj:.1f} min/跑 "
          f"（= {train_s:.1f}×{full_epochs} + {one - train_s:.1f}，消融只算一次）")
    print(f"[预注册规则] > {WINDOWS_CALIBRATION_MINUTES:.0f} min/跑 → 全臂统一 "
          f"--settf-windows 32（由操作者执行，脚本不自动改）", flush=True)


# ================================================================ 阶段 C：判读
def _wilcoxon_greater(x: Sequence[float]) -> Optional[float]:
    """单侧 Wilcoxon：H1 = 中位差 > 0。全零差 → 1.0（scipy 会报错，与 16 同款守卫）。"""
    a = np.asarray(list(x), dtype=float)
    if a.size == 0:
        return None
    if np.all(a == 0):
        return 1.0
    return float(wilcoxon(a, alternative="greater").pvalue)


def verdict_code(*, cross_tiers: Sequence[Optional[str]],
                 budget_tiers: Sequence[Optional[str]], pooled_p: Optional[float],
                 q3_delta: Optional[float], q3_p: Optional[float],
                 a8_cross_tier: Optional[str]) -> dict:
    """协议 §5 判读表（先于数据写死；阈值跑完不改）。

    表序即优先序：E-ALIVE → E-DETECT → LABEL-BLIND → E-PATHWAY → B-DEAD → AMBIGUOUS。
    B-DEAD 与 LABEL-BLIND 可同时成立（模型既不用 E 也不用 b）——code 按表序给出，
    但 conditions 里五个布尔值全报，REPORT 须一并引用。
    """
    n_pass = sum(t == "pass" for t in cross_tiers)
    n_det = sum(t in ("pass", "detectable") for t in cross_tiers)
    n_zero = sum(t == "zero" for t in cross_tiers)
    n_budget_pass = sum(t == "pass" for t in budget_tiers)
    n_budget_zero = sum(t == "zero" for t in budget_tiers)
    q3_ok = (q3_delta is not None and q3_p is not None
             and q3_delta >= 0.0 and q3_p < 0.05)
    cond = {
        "e_alive": bool(n_pass >= 5 and pooled_p is not None and pooled_p < 0.05
                        and q3_ok),
        "e_detect": bool(n_det >= 4),
        "label_blind": bool(n_zero >= 5 and a8_cross_tier == "zero"),
        "e_pathway": bool(n_zero >= 5 and a8_cross_tier == "pass"),
        "b_dead": bool(n_budget_zero >= 5),
    }
    counts = {"cross_pass": n_pass, "cross_detectable_or_better": n_det,
              "cross_zero": n_zero, "budget_pass": n_budget_pass,
              "budget_zero": n_budget_zero, "n_folds": len(list(cross_tiers))}
    for key, code in (("e_alive", "E-ALIVE"), ("e_detect", "E-DETECT"),
                      ("label_blind", "LABEL-BLIND"), ("e_pathway", "E-PATHWAY"),
                      ("b_dead", "B-DEAD")):
        if cond[key]:
            return {"code": code, "conditions": cond, "counts": counts, "unmet": [],
                    "pooled_p": pooled_p, "q3_delta": q3_delta, "q3_p": q3_p,
                    "a8_cross_tier": a8_cross_tier}
    unmet = [
        f"E-ALIVE: cross pass {n_pass}/5 需 ≥5；pooled p={pooled_p}；Q3 Δ={q3_delta} "
        f"p={q3_p}",
        f"E-DETECT: cross ≥detectable {n_det} 需 ≥4",
        f"LABEL-BLIND/E-PATHWAY: cross zero {n_zero} 需 ≥5；A8 cross={a8_cross_tier}",
        f"B-DEAD: budget zero {n_budget_zero} 需 ≥5",
    ]
    return {"code": "AMBIGUOUS", "conditions": cond, "counts": counts, "unmet": unmet,
            "pooled_p": pooled_p, "q3_delta": q3_delta, "q3_p": q3_p,
            "a8_cross_tier": a8_cross_tier}


def _median_seed_run(runs: list[dict]) -> dict:
    """A0 每折取 3 种子中 within Δ̄ 的**中位**种子（协议 §5）；within 缺席时按种子取中。"""
    if all(r.get("within") for r in runs):
        ordered = sorted(runs, key=lambda r: r["within"]["mean_delta"])
    else:
        ordered = sorted(runs, key=lambda r: r["seed"])
    return ordered[len(ordered) // 2]


def _pooled_acc(runs: list[dict]) -> Optional[float]:
    if not runs:
        return None
    accs = [r["normal"]["acc"] for r in runs]
    ns = [r.get("n_test_pairs_directed") for r in runs]
    if all(n for n in ns):
        return float(np.average(accs, weights=ns))
    return float(np.mean(accs))


def analyze(outdir) -> dict:
    """runs/**.json → analysis.json（逐折档位 / pooled 检验 / 判读码 / 预测对账 / 成本）。"""
    outdir = Path(outdir)
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((outdir / "runs").glob("*/*.json"))]
    skipped = load_skipped(outdir)
    by_arm: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_arm[r["arm"]][r["fold"]].append(r)

    per_arm_fold: dict[str, dict[str, dict]] = {}
    for arm, folds in by_arm.items():
        per_arm_fold[arm] = {}
        for fold, rs in folds.items():
            chosen = _median_seed_run(rs)
            settings = chosen.get("settings") or {}
            per_arm_fold[arm][fold] = {
                "seed": chosen["seed"], "n_seeds": len(rs),
                "normal_acc": chosen["normal"]["acc"],
                # 主表只收 True 的臂：A1 是 False（每数据集自 fit），S2 两臂是
                # "s2_oracle"（上界对照，本就不是 zero-shot）——判读引用必须带这个标。
                "iron_law_6_compliant": settings.get("iron_law_6_compliant"),
                "feature_fold": settings.get("feature_fold"),
                "tiers": {k: (tier_of(chosen[k]) if chosen.get(k) else None)
                          for k in ("within", "cross", "budget")},
                "mean_delta": {k: (chosen[k]["mean_delta"] if chosen.get(k) else None)
                               for k in ("within", "cross", "budget")},
                "train_seconds": chosen.get("train_seconds"),
            }

    a0 = per_arm_fold.get("main", {})
    folds_a0 = sorted(a0)
    cross_tiers = [a0[f]["tiers"]["cross"] for f in folds_a0]
    budget_tiers = [a0[f]["tiers"]["budget"] for f in folds_a0]
    cross_deltas = [a0[f]["mean_delta"]["cross"] for f in folds_a0
                    if a0[f]["mean_delta"]["cross"] is not None]
    pooled_p = _wilcoxon_greater(cross_deltas)

    def _paired(arm_a: str, arm_b: str) -> tuple[list[float], Optional[float]]:
        A, B = per_arm_fold.get(arm_a, {}), per_arm_fold.get(arm_b, {})
        common = sorted(set(A) & set(B))
        d = [A[f]["normal_acc"] - B[f]["normal_acc"] for f in common]
        return d, _wilcoxon_greater(d)

    q3_d, q3_p = _paired("main", "cls_b")
    a1_d, a1_p = _paired("autocts_style", "main")
    a4_d, _a4_p = _paired("main", "wo_shared")
    a8_cross = (per_arm_fold.get("onehot_oracle", {}).get(S2_FOLD, {})
                .get("tiers", {}).get("cross"))
    verdict = verdict_code(cross_tiers=cross_tiers, budget_tiers=budget_tiers,
                           pooled_p=pooled_p,
                           q3_delta=(float(np.mean(q3_d)) if q3_d else None),
                           q3_p=q3_p, a8_cross_tier=a8_cross)

    a0_runs = [r for r in runs if r["arm"] == "main"]
    pooled_normal = _pooled_acc(a0_runs)
    a1_cross = [v["tiers"]["cross"] for v in per_arm_fold.get("autocts_style", {}).values()]
    calib_path = outdir / "delta_calibration.json"
    p9 = {"hit": None, "n_datasets_ge": None, "note": "缺 delta_calibration.json"}
    if calib_path.exists():
        raw = json.loads(calib_path.read_text(encoding="utf-8"))
        n_ge = sum(1 for d, per_b in raw.items()
                   if "290" in per_b and "120" in per_b
                   and per_b["290"]["delta"] >= per_b["120"]["delta"])
        p9 = {"hit": bool(n_ge >= 6), "value": n_ge, "n_datasets_ge": n_ge,
              "n_datasets": len(raw)}

    predictions = {
        "P1": {"desc": "A0 pooled normal acc ∈ [0.70, 0.85]", "value": pooled_normal,
               "hit": (None if pooled_normal is None
                       else bool(0.70 <= pooled_normal <= 0.85)),
               "n_runs_pooled": len(a0_runs),
               "note": "pooled = A0 **全部种子**按单向对数加权（3×7 = 21 跑）；"
                       "P2/P3 的档位则取每折 within Δ̄ 的中位种子，两者口径不同"},
        "P2": {"desc": "A0 cross 档位 zero 于 ≥5/7 折",
               "value": sum(t == "zero" for t in cross_tiers),
               "hit": bool(sum(t == "zero" for t in cross_tiers) >= 5)},
        "P3": {"desc": "A0 budget 档位 pass 于 ≥5/7 折",
               "value": sum(t == "pass" for t in budget_tiers),
               "hit": bool(sum(t == "pass" for t in budget_tiers) >= 5)},
        "P4": {"desc": "Q3：A0 − A6 ≥ 0.01 且 p<0.05",
               "value": [float(np.mean(q3_d)) if q3_d else None, q3_p],
               "hit": bool(q3_d and np.mean(q3_d) >= 0.01 and q3_p is not None
                           and q3_p < 0.05)},
        "P5": {"desc": "A1 cross ≥ detectable 于 ≥4/7 折",
               "value": sum(t in ("pass", "detectable") for t in a1_cross),
               "hit": bool(sum(t in ("pass", "detectable") for t in a1_cross) >= 4)},
        "P6": {"desc": "A8 cross pass（S2）", "value": a8_cross,
               "hit": (None if a8_cross is None else bool(a8_cross == "pass"))},
        "P7": {"desc": "D1/D2（端到端 regret）", "value": None, "hit": None,
               "note": "由 scripts/34_pq_search.py 产出"},
        "P8": {"desc": "D3（进化 top-3）", "value": None, "hit": None,
               "note": "由 scripts/34_pq_search.py 产出"},
        "P9": {"desc": "δ@290 ≥ δ@120 于 ≥6/8 数据集", **p9},
        "P10": {"desc": "A4 比 A0 低 ≥0.03 于 ≥5/7 覆盖折",
                "value": sum(1 for d in a4_d if d >= 0.03), "n_covered": len(a4_d),
                "hit": bool(sum(1 for d in a4_d if d >= 0.03) >= 5),
                "note": "阈值 ≥5 是**绝对折数**（协议 §6 原文）：A4 默认只覆盖 3 折"
                        "（HW/UM/Cricket），此时 hit 在数学上不可达——n_covered <5 时"
                        "应读 value/n_covered 的比例，不读 hit"},
    }

    cost: dict[str, dict] = {}
    for arm, folds in by_arm.items():
        secs = [r.get("train_seconds") or 0.0 for rs in folds.values() for r in rs]
        cost[arm] = {"n_runs": len(secs), "total_seconds": float(np.sum(secs)),
                     "mean_seconds": float(np.mean(secs)) if secs else None}

    fold_meta = {}
    for meta_path in sorted((outdir / "folds").glob("*/shared_ts2vec.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fold = meta_path.parent.name
        fold_meta[fold] = {"fit_datasets": meta.get("fit_datasets"),
                           "target_excluded": bool(fold == ALL8_FOLD
                                                   or fold not in
                                                   set(meta.get("fit_datasets", [])))}

    out = {
        "protocol": "results/mx3_pq/PROTOCOL.md",
        "per_arm_fold": per_arm_fold,
        "tests": {
            "n_folds_A0": len(folds_a0), "folds_A0": folds_a0,
            "cross_tiers": cross_tiers, "budget_tiers": budget_tiers,
            "pooled_cross_delta_p": pooled_p,
            "pooled_cross_delta_mean": float(np.mean(cross_deltas))
            if cross_deltas else None,
            "Q3_A0_minus_A6_mean": float(np.mean(q3_d)) if q3_d else None,
            "Q3_p": q3_p, "Q3_n_folds": len(q3_d),
            "A1_minus_A0_mean": float(np.mean(a1_d)) if a1_d else None,
            "A1_gt_A0_p": a1_p, "A1_n_folds": len(a1_d),
            "A0_pooled_normal_acc": pooled_normal,
        },
        "verdict": verdict, "predictions": predictions, "cost": cost,
        "fold_meta_check": fold_meta,
        # 被跳过的 (臂, 折, 种子)：per_arm_fold 里**不伪造**这些条目，缺席即在此列表
        "skipped": skipped, "n_skipped": len(skipped),
        "skipped_arms": sorted({r["arm"] for r in skipped}),
    }
    return out


def cmd_analyze(outdir: Path) -> dict:
    res = analyze(outdir)
    path = Path(outdir) / "analysis.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                    encoding="utf-8")
    print(f"\n===== MX-3-PQ 比较器（{res['tests']['n_folds_A0']} 折 A0）=====")
    print(f"{'臂':<16}{'折':<24}{'种子':>4}{'normal':>9}"
          f"{'within':>12}{'cross':>12}{'budget':>12}")
    for arm in sorted(res["per_arm_fold"]):
        for fold, e in sorted(res["per_arm_fold"][arm].items()):
            t = e["tiers"]
            print(f"{arm:<16}{fold:<24}{e['seed']:>4}{e['normal_acc']:>9.4f}"
                  f"{str(t['within']):>12}{str(t['cross']):>12}"
                  f"{str(t['budget']):>12}")
    for r in res["skipped"]:
        print(f"[SKIP] {r['arm']}/{r['fold']} s{r['seed']}: {r['reason']}")
    t = res["tests"]
    print(f"\npooled cross Δ p={t['pooled_cross_delta_p']} | "
          f"Q3(A0−A6) Δ̄={t['Q3_A0_minus_A6_mean']} p={t['Q3_p']} | "
          f"A1−A0 Δ̄={t['A1_minus_A0_mean']} p={t['A1_gt_A0_p']}")
    print(f"[判读] → {res['verdict']['code']}  {res['verdict']['counts']}")
    for k in sorted(res["predictions"]):
        p = res["predictions"][k]
        print(f"  {k} {p['desc']}: hit={p['hit']} value={p.get('value')}")
    print(f"→ {path}", flush=True)
    return res


# ================================================================ 冒烟
def synth_smoke_inputs(outdir: Path, *, datasets: Sequence[str], n_slices: int,
                       n_cls: int, budgets: Sequence[int], seed: int = 0,
                       origins: Sequence[str] = ("shared", "random")) -> None:
    """Task 5 冒烟缓存缺席时的**合成回落**：R' 缓存 / 噪声探针 / δ 表 / tasks.json。

    **不产特征**——窗口合成在 synth_smoke_features（复用 Task 5 缓存时同样要跑）。
    `origins=("shared",)` 复现 32 号冒烟的形状（目录前缀 = 全 shared，零 random）：
    该形状下 A4 wo_shared 的训练集为空，用来验证"跳过而不是中断整轮"。

    合成 R' 刻意可分（同切片相邻 CLS 差 0.02 > δ=0.01），且**锚在该切片真实的
    iters=0 基线带上**（起点 = 均值 − 带 − 0.05）——这样最低几个 CLS 落在 strict
    破坏带内、其余在带外，A7 soft 臂的两制式噪声模型才有材料（全同制式会让
    fit_noise_model 拿到空样本）。只用于跑通链路、**绝不用于任何判读**：
    落盘 device_name="smoke" 以便与真实缓存一眼区分。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    space = CLSSpace.from_yaml()
    rng = np.random.default_rng(seed)
    bs = [int(b) for b in budgets]
    bands: dict[str, tuple] = {}
    for d in datasets:
        try:
            bands.update(pq.baseline_bands(d))
        except (ValueError, FileNotFoundError):
            pass          # 无基线缓存的数据集：退回中性锚点，制式退化为单一 nd
    rows, noise_rows, tasks = [], [], {}
    for di, d in enumerate(datasets):
        for i in range(n_slices):
            slc = f"{d}_{i:02d}"
            cat = pq.catalog_for_slice(space, slc)
            if tuple(origins) == ("shared",):
                chosen = cat[:n_cls]                     # 32 号冒烟形状：全 shared
            else:
                n_sh = max(1, n_cls // 2)
                chosen = cat[:n_sh] + cat[pq.N_SHARED:pq.N_SHARED + (n_cls - n_sh)]
            tasks[slc] = {"task_id": slc, "dataset": d, "task_type": "classification",
                          "classes": [0, 1], "n_train": 100, "n_eval": 300}
            mean, band = bands.get(slc, (0.50, 0.02))
            lo = float(np.clip(mean - band - 0.05, 0.02, 0.88))
            # 名次按切片随机置换：否则 R' 与目录顺序单调，单向对的 y 会**全为同值**，
            # binom/McNemar 退化成常数，冒烟就查不出配对/消融的接线错误。
            rank = rng.permutation(len(chosen))
            for j, (origin, cls_obj, key) in enumerate(chosen):
                for bi, b in enumerate(bs):
                    r = lo + 0.02 * int(rank[j]) + 0.006 * bi + 0.002 * di
                    rows.append({
                        "dataset": d, "task_id": slc, "tag": "smoke",
                        "task_type": "classification", "classes": [0, 1],
                        "cls_key": key, "origin": origin, "seed": 0, "iters": b,
                        "checkpoint_iters": bs, "trajectory_id": f"{slc}:{key}:0",
                        "r_prime": round(r, 6), "acc": round(r, 6), "f1": round(r, 6),
                        "svm_c": 1, "n_train": 100, "n_eval": 300, "weight_seed": 0,
                        "split_seed": 0, "device_name": "smoke", "swa": False,
                        "seconds": 1.0, "trajectory_seconds": 3.0, "error": None,
                        "cls": dataclasses.asdict(cls_obj)})
                    if i < 2 and origin == "shared":
                        for rep in (1, 2):
                            noise_rows.append({
                                **rows[-1], "seed": rep,
                                "r_prime": round(r + 0.004 * rep - 0.002, 6)})
    (outdir / "rprime_cache.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    (outdir / "noise_probe.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in noise_rows),
        encoding="utf-8")
    (outdir / "delta_calibration.json").write_text(json.dumps(
        {d: {str(b): {"delta": 0.01, "p95": 0.008, "quantum_floor": 0.0067,
                      "n_diffs": 24, "diff_median": 0.004, "delta_nd": 0.01}
             for b in bs} for d in datasets}, indent=2), encoding="utf-8")
    (outdir / "tasks.json").write_text(json.dumps(
        {"pool": list(datasets), "budgets": bs, "n_slices_per_dataset": n_slices,
         "eval_ratio": 0.3, "tasks": tasks}, indent=2, ensure_ascii=False),
        encoding="utf-8")


def synth_smoke_features(outdir: Path, *, slices: Sequence[str], folds: Sequence[str],
                         settf_windows: int, t_max: int, nz: int, seed: int = 0) -> int:
    """合成窗口表征（已存在的 npz 跳过 = 幂等）。

    与缓存合成**分开**：复用 Task 5 的 `--smoke` 产物时缓存已在盘上、特征却一个没有，
    如果特征生成挂在"缓存缺席"分支里，冒烟会直接撞上 windows() 的"缺特征"fail-fast。
    故 run_smoke 无论缓存来自哪里都调用本函数。返回写出/已存在的 npz 总数。
    """
    rng = np.random.default_rng(seed)
    set_enc = get_set_encoder_init(outdir / "set_encoder_init.pt", nz=nz).eval()
    dirs = [outdir / "per_dataset_features"]
    for f in list(folds) + [ALL8_FOLD]:
        dirs += [outdir / "folds" / f / "task_features",
                 outdir / "folds" / f / "task_features_unfit"]
    n = 0
    for dpath in dirs:
        dpath.mkdir(parents=True, exist_ok=True)
        for slc in sorted(set(slices)):
            n += 1
            path = dpath / f"{slc}.npz"
            if path.exists():
                continue
            w = rng.standard_normal((settf_windows, t_max,
                                     TS2VEC_OUTPUT_DIMS)).astype(np.float32)
            with torch.no_grad():
                e = set_enc(torch.as_tensor(w)).numpy().astype(np.float32)
            np.savez_compressed(path, windows_repr=w.astype(np.float16), E_init=e,
                                META=np.array([2.0, 0.5, 2.0, 2.0], dtype=np.float32))
    return n


def run_smoke(outdir, *, epochs: int = 2, arms: Optional[Sequence[str]] = None,
              folds: Optional[Sequence[str]] = None,
              seeds: Optional[Sequence[int]] = (0,),
              datasets: Sequence[str] = SMOKE_DATASETS, n_slices: int = 6,
              n_cls: int = 8, budgets: Optional[Sequence[int]] = None,
              device: str = "cpu", settf_windows: int = 4, t_max: int = 8) -> dict:
    """端到端冒烟：（缺则合成）缓存 + 合成窗口 → --train → --analyze。

    优先复用 Task 5 的 `--smoke` 产物（`<outdir>/rprime_cache.jsonl` 存在即直接用，
    预算档由缓存自身的 iters 决定）；缺席时按上面的合成回落自造一份。
    """
    outdir = Path(outdir)
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    nz = int(labels_cfg["route_a"]["nz"])
    cache_path = outdir / "rprime_cache.jsonl"
    if not cache_path.exists():
        synth_smoke_inputs(outdir, datasets=datasets, n_slices=n_slices, n_cls=n_cls,
                           budgets=budgets or pq.BUDGETS)
    else:
        print(f"[smoke] 复用已有 {cache_path}", flush=True)
    # 特征无论缓存来自哪里都要有：切片清单从缓存自身读（Task 5 的冒烟切片可能与
    # 本地 datasets/n_slices 默认值不同），折目录清单取 fold_list ∪ all8。
    slices = sorted({k[0] for k in load_rprime_cache(cache_path)})
    # 折靶也从缓存推导：32 号冒烟只覆盖它自己那几个数据集，写死 Handwriting 会在
    # 缓存不含该数据集时直接判死整条链（本轮 review 的复现路径）。
    fold_list = (list(folds) if folds else
                 [d for d in sorted({pq.dataset_of(x) for x in slices})
                  if d in pq.FOLD_TARGETS])
    n_feat = synth_smoke_features(outdir, slices=slices, folds=fold_list,
                                  settf_windows=settf_windows, t_max=t_max, nz=nz)
    print(f"[smoke] 切片 {len(slices)} | 合成特征槽位 {n_feat}", flush=True)
    arm_list = list(arms) if arms else list(ALL_ARMS)
    cmd_train(outdir, arms=arm_list, folds=fold_list, seeds=seeds, resume=False,
              device=device, epochs=epochs, settf_windows=settf_windows)
    return cmd_analyze(outdir)


# ================================================================ 入口
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-folds", action="store_true")
    ap.add_argument("--fit-per-dataset", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibrate-windows", action="store_true",
                    help="折 HW 的 A0 s0 单 epoch 计时 → 外推整跑（预注册旋钮标定）")
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ALL_ARMS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 pretrain_tclsc.yaml")
    ap.add_argument("--settf-windows", type=int, default=None,
                    help="覆盖 labels.yaml: pq.settf_windows（预注册旋钮）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    args = ap.parse_args()
    if not any((args.fit_folds, args.fit_per_dataset, args.train, args.analyze,
                args.smoke, args.calibrate_windows)):
        ap.error("至少指定 --fit-folds / --fit-per-dataset / --train / --analyze / "
                 "--smoke / --calibrate-windows 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.fit_folds:
        cmd_fit_folds(args.outdir, args.folds, args.device, args.settf_windows)
    if args.fit_per_dataset:
        cmd_fit_per_dataset(args.outdir, args.device, args.settf_windows)
    if args.calibrate_windows:
        cmd_calibrate_windows(args.outdir, args.device, args.settf_windows)
    if args.smoke:
        run_smoke(args.outdir, epochs=args.epochs or 2, arms=args.arms,
                  folds=args.folds,
                  seeds=tuple(args.seeds) if args.seeds else (0,),
                  device=args.device, settf_windows=args.settf_windows or 4)
    if args.train:
        cmd_train(args.outdir, arms=args.arms or list(ALL_ARMS), folds=args.folds,
                  seeds=args.seeds, resume=args.resume, device=args.device,
                  epochs=args.epochs, settf_windows=args.settf_windows)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
