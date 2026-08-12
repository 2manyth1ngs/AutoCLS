"""目标池 zero-shot 评估：UCI-HAR（TECHNICAL_DESIGN.md 模块 8 / M4——pipeline 最后一环）。

主张口径（出路 X，决定性实验 C2 后）：T-CLSC = 任务无关的通用 CLS 质量先验；
本脚本回答"它的 zero-shot 排名在未见数据集上顶不顶用"。E 支路保留走完整流程，
消融如实报告（源池内 Δ=0.0000，目标池异质性更大，观察一次）。

zero-shot 边界（对照 AutoCTS++；写进 REPORT 防审稿质疑）：
- 目标数据集只用于【评估】：绝不参与共享 TS2Vec 训练（铁律 6，指纹校验 + fail-fast
  自检）、绝不产生 T-CLSC 训练 label；
- 工作点标定在目标 train 上跑 shared-CLS 短训练（≈10 min）——AutoCTS++ 的 zero-shot
  同样要在目标任务上生成任务表示 + top-K train on D_train；"zero-shot"指"不为该
  任务重新搜索"，不是"完全不碰目标数据"；
- **测试集纪律**：标定/排名/真值 sweep 只用官方 train 切分（含内部 80/20 val）；
  官方 test 只在 --final 阶段被 top-3 / GGS 对照 / winner 补种子 / 桥接触碰。

阶段（按序，全部幂等，R' 类阶段支持 --resume；缓存键含 iters，铁律 2/3）：
    python scripts/5_target_zero_shot.py --prep               # 切任务+基线+任务特征
    python scripts/5_target_zero_shot.py --calibrate --resume # 工作点标定（inert 即停）
    python scripts/5_target_zero_shot.py --tclsc --resume     # 重训 R1-R3 mixed 存档
    python scripts/5_target_zero_shot.py --rank               # 比较器排名 + E 消融
    python scripts/5_target_zero_shot.py --sweep --resume     # 全池 31 候选真值 sweep(val)
    python scripts/5_target_zero_shot.py --final --resume     # top-3/GGS/winner 测试评估+桥接
    python scripts/5_target_zero_shot.py --verify             # 收官校验 + 泄漏自检
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from autocls.baseline_probe import (  # noqa: E402
    BEST_GAIN_FLOOR, CEILING_BASELINE, T3_ABOVE_FRACTION, T3_ABOVE_OVER_BELOW,
    baseline_seed_specs, inert_cls, summarize_dataset, summarize_task,
)
from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.comparator.tclsc_engine import derangement, with_reversed  # noqa: E402
from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.datasets import load_har  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, TargetPoolLeakageError, load_registry, load_source_dataset,
    slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.downstream import evaluate_classification  # noqa: E402
from autocls.downstream.linear_eval import encode_features, fit_svm  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    append_jsonl, cls_key, comparator_records, device_label, evaluate_rprime,
    load_probe_cls, load_rprime_cache, run_seed_for, split_task_train_eval,
    split_train_eval_by_group,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run, is_oom_error  # noqa: E402
from autocls.search.rank import (  # noqa: E402
    borda_ranking, pair_vote_matrix, symmetrize, wins_from_scores,
)
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import TS2VecWrapper, TaskSetEncoder  # noqa: E402
from autocls.task_types import is_better  # noqa: E402
from autocls.workpoint import (  # noqa: E402
    destruction_participation, no_workpoint_verdict, select_workpoint,
    steps_per_epoch,
)

# 数字开头模块走 importlib（与 2c 同法）
_r2b = importlib.import_module("scripts.2b_generate_labels_route_a")
_4b = importlib.import_module("scripts.4b_decisive_e_ablation")
_m1 = importlib.import_module("scripts.m1_ggs_har")

TARGET = "UCI-HAR"
DEFAULT_OUTDIR = ROOT / "results" / "target_zero_shot"
LABELS_DIR = ROOT / "results" / "labels_workpoint"
DECISIVE_RUNS = ROOT / "results" / "decisive_e_ablation" / "runs"
HAR_NPZ = ROOT / "data" / "HAR" / "har.npz"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"

EP_GRID = (1, 2, 5, 10, 20)     # 用户拍板的标定网格（epoch 语义，任务级换算成 iters）
N_PROBE_TASKS = 2               # 标定口径 = 前 2 任务（workpoint_confirm PROTOCOL §2 同款）
N_SHARED_CLS = 10
N_BASELINE_SEEDS = 5            # 随机初始化基线种子数（源池探针同口径）
N_FULLBASE_SEEDS = 3            # 全量随机初始化 val 基线种子数
RANDOM_POOL_SEED = 777          # 候选池 random 段种子（与任务/数据无关，用户确认）
N_RANDOM_POOL = 20
GGS_ORIGIN = "ggs"
TOP_K = 3                       # 沿用 AutoCTS++；top-10 免费记录作诊断
N_ABL_SEEDS = 5                 # E 消融 shuffle 档种子数（decisive 同款）
VAL_RATIO = 0.2                 # 真值 sweep 的 train 内 80/20 val（测试集纪律）
VAL_SPLIT_SEED = 20260811
DESTRUCTION_DELTAS = (0.05, 0.03)  # 破坏参与率诊断口径（confirm §4 主口径 + 敏感性）
M1_GGS_2320 = {"original": 0.9372, "rerun_4060": 0.9348}  # M1 锚点（桥接对照引用）
WORKPOINT_MARGIN = 0.02         # select_workpoint 让分窗（confirm 协议 §2）

# --- 出路 A：跨受试者协议（2026-08-12 拍板；PROTOCOL 见 subject_protocol/PROTOCOL.md）---
# 动机：train 内分层切分受试者 100% 重叠 + 窗口 50% 重叠 → 测量饱和（首轮停机报告 §4）。
# 协议模式下任务内 train/eval 与 sweep 的 val 全部按受试者整组切分；判据/网格/候选池
# 与首轮逐字不变。协议特定缓存进 subject_protocol/ 子目录（切分不同 → R' 不可混用，
# 铁律 3）；task_features / tclsc / ranking / candidates / encoders 切分无关，两协议共享。
SUBJECT_TXT = ROOT / "data" / "raw" / "uci_har" / "inner" / "UCI HAR Dataset" / \
    "train" / "subject_train.txt"
VAL_HOLDOUT_SUBJECTS = (8, 17, 21, 25)   # 预注册固定（diag rng seed=0 抽出，停机报告 §9-1）
SUBJECT_SUBDIR = "subject_protocol"


# ---------------------------------------------------------------------------
# 可测助手（tests/test_target_zero_shot.py）
# ---------------------------------------------------------------------------


def grid_iters(spe: int) -> list[int]:
    """标定网格：epoch 档 × 任务级 steps/epoch → iters（铁律 2：iters 计量）。"""
    return [ep * spe for ep in EP_GRID]


def full_iters_for_ep(ep: float, n_train: int) -> int:
    """工作点 epoch → 全量数据集 iters（steps/epoch = ⌈n_train/128⌉，M1 口径）。"""
    return int(round(ep * steps_per_epoch(n_train)))


def build_candidate_pool(
    n_shared: int = N_SHARED_CLS, n_random: int = N_RANDOM_POOL,
    seed: int = RANDOM_POOL_SEED,
) -> list[tuple[str, object, str]]:
    """候选池 31 = 10 shared（labels_workpoint 同款，与源池 label 逐键可比）
    + 20 random（seed=777 与任务无关；逐个接受制前缀稳定，铁律 3）
    + GGS（AutoCL Table 5，文献级好配置的直接诊断，标记 origin='ggs'）。"""
    space = CLSSpace.from_yaml()
    shared = load_probe_cls(PROBE_CATALOG, n_shared)[0]
    ggs_k = cls_key(_m1.GGS)
    taken = {k for _c, k in shared} | {ggs_k}
    rng = np.random.default_rng(seed)
    rnd: list[tuple[object, str]] = []
    while len(rnd) < n_random:   # 逐个接受制：加大 n 不改变前缀（labeling._sample_unique 同语义）
        c = space.sample_random(rng)
        k = cls_key(c)
        if k in taken:
            continue
        taken.add(k)
        rnd.append((c, k))
    return ([("shared", c, k) for c, k in shared]
            + [("random", c, k) for c, k in rnd]
            + [(GGS_ORIGIN, _m1.GGS, ggs_k)])


# ---------------------------------------------------------------------------
# 公用
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 被杀留下的半行
    return out


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_target_train() -> tuple[np.ndarray, np.ndarray]:
    """目标 train 侧（7352）。测试集纪律：test 只经 load_target_test（--final 专用）。"""
    X_train, y_train, _X_test, _y_test = load_har(HAR_NPZ)
    return X_train, y_train


def load_target_test() -> tuple[np.ndarray, np.ndarray]:
    """官方 test（2947）——仅 --final 阶段调用（触碰清单见 stage_verify）。"""
    _X, _y, X_test, y_test = load_har(HAR_NPZ)
    return X_test, y_test


def target_tasks(reg, y_train: np.ndarray):
    """目标任务切片：注册表唯一真源的通用切片（非 HAR 动/静分层——那是路线 B 遗留）。

    slice_source_tasks 是纯切片函数（只吃 y，不走加载路径），对目标沿用不触碰
    铁律 6——数据加载仍由 load_target_train 专用路径承担，load_source_dataset
    对 UCI-HAR 保持 fail-fast（stage_prep 有自检）。
    """
    entry = reg.get(TARGET)
    assert entry.role == ROLE_TARGET, f"{TARGET} 应为目标池成员"
    _, tasks = slice_source_tasks(entry, reg, y_train)
    return tasks


def val_split(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """真值 sweep 的 train 内分层 80/20（fit 侧, val 侧）；固定种子全阶段一致。"""
    return split_task_train_eval(
        y_train, "classification", eval_ratio=VAL_RATIO, seed=VAL_SPLIT_SEED)


def data_dir(outdir: Path, subject: bool) -> Path:
    """协议特定缓存目录：subject 模式进子目录（切分不同 → R' 不可混用，铁律 3）。"""
    return outdir / SUBJECT_SUBDIR if subject else outdir


def load_train_subjects(n_expected: int) -> np.ndarray:
    subjects = np.loadtxt(SUBJECT_TXT, dtype=np.int64)
    assert len(subjects) == n_expected, \
        f"subject_train.txt {len(subjects)} 行 ≠ train {n_expected}（顺序对齐前提破坏）"
    return subjects


def subject_task_split(task, subjects: np.ndarray, yt: np.ndarray,
                       eval_ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    """任务内跨受试者切分（canonical：seed 由 task_id 派生，与执行顺序无关）。"""
    gidx = np.asarray(task.sample_indices)
    seed = run_seed_for(task.task_id, "subject_eval_split")
    tr, ev = split_train_eval_by_group(
        yt, subjects[gidx], eval_ratio=eval_ratio, seed=seed)
    return tr, ev, seed


def subject_val_split(subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sweep 真值的跨受试者 val：留出组预注册固定 VAL_HOLDOUT_SUBJECTS。"""
    mask = np.isin(subjects, VAL_HOLDOUT_SUBJECTS)
    assert mask.any() and not mask.all(), "留出受试者不在 train 中或吞掉全部样本"
    return np.where(~mask)[0], np.where(mask)[0]


def probe_spe(tasks, X_train, y_train, eval_ratio: float,
              subjects: np.ndarray | None = None) -> int:
    """前 2 任务的 steps/epoch（须唯一，否则 epoch 网格不可换算——fail-fast）。

    subjects 给出时按跨受试者切分计 n_train（出路 A 协议）。"""
    spes = set()
    for t in tasks[:N_PROBE_TASKS]:
        _Xt, yt = materialize_task(t, X_train, y_train)
        if subjects is not None:
            tr_idx, _ev, _s = subject_task_split(t, subjects, yt, eval_ratio)
        else:
            tr_idx, _ = split_task_train_eval(
                yt, t.task_type, eval_ratio=eval_ratio,
                seed=run_seed_for(t.task_id, "train_eval_split"))
        spes.add(steps_per_epoch(len(tr_idx)))
    if len(spes) != 1:
        sys.exit(f"探针任务 steps/epoch 不唯一 {spes}，epoch 网格不可换算")
    return spes.pop()


def load_workpoint(outdir: Path) -> dict:
    path = outdir / "workpoint.json"
    if not path.exists():
        sys.exit("缺 workpoint.json：先跑 --calibrate")
    wp = json.loads(path.read_text(encoding="utf-8"))
    if wp.get("selection") is None:
        sys.exit("工作点标定判 inert（无可确认点）——按协议停下报告，不进后续阶段")
    return wp


def build_tclsc_skeleton(space: CLSSpace, dim_input: int, cfg: dict, nz: int) -> TCLSC:
    set_enc = TaskSetEncoder(dim_input=dim_input, nz=nz)
    return TCLSC(cls_dim=CLSEncoder(space).dim, set_encoder=set_enc,
                 embedding_dim=cfg["embedding_dim"], fz=cfg["fz"])


def pair_prob_matrix(model: TCLSC, vecs: torch.Tensor, e_row: torch.Tensor) -> np.ndarray:
    """全部有序对一批前向：P[i,j] = P(候选 i 优于 j | E)。对角 NaN。"""
    n = vecs.shape[0]
    ii, jj = np.where(~np.eye(n, dtype=bool))
    with torch.no_grad():
        out = model(vecs[ii], vecs[jj], e_row.expand(len(ii), -1)).cpu().numpy()
    P = np.full((n, n), np.nan)
    P[ii, jj] = out
    return P


# ---------------------------------------------------------------------------
# 阶段 1：--prep 切任务 + 随机初始化基线 + 任务特征（冻结 TS2Vec 只编码）
# ---------------------------------------------------------------------------


def stage_prep(outdir: Path, reg, labels_cfg: dict, device: str, resume: bool,
               subject: bool = False) -> None:
    ddir = data_dir(outdir, subject)
    ddir.mkdir(parents=True, exist_ok=True)
    # 泄漏自检①（铁律 6）：目标走源加载路径必须 fail-fast
    try:
        load_source_dataset(TARGET, reg)
        sys.exit("泄漏自检失败：load_source_dataset(UCI-HAR) 未 fail-fast！")
    except TargetPoolLeakageError:
        print("泄漏自检①通过：load_source_dataset(UCI-HAR) fail-fast", flush=True)

    X_train, y_train = load_target_train()
    tasks = target_tasks(reg, y_train)
    eval_ratio = labels_cfg["eval_ratio"]
    subjects = load_train_subjects(len(y_train)) if subject else None
    print(f"{TARGET}: X_train={X_train.shape} → {len(tasks)} 任务（通用切片"
          f"{'，跨受试者协议' if subject else ''}）", flush=True)

    def task_split(t, yt):
        """协议分发：subject 模式整组切分（注入），否则 canonical 分层切分。"""
        if subject:
            tr, ev, seed = subject_task_split(t, subjects, yt, eval_ratio)
            return tr, ev, seed
        seed = run_seed_for(t.task_id, "train_eval_split")
        tr, ev = split_task_train_eval(yt, t.task_type, eval_ratio=eval_ratio,
                                       seed=seed)
        return tr, ev, seed

    task_meta = []
    for t in tasks:
        _Xt, yt = materialize_task(t, X_train, y_train)
        tr_idx, ev_idx, sp_seed = task_split(t, yt)
        row = {**dataclasses.asdict(t), "n_train": len(tr_idx),
               "n_eval": len(ev_idx), "spe": steps_per_epoch(len(tr_idx)),
               "split_seed": sp_seed}
        if subject:
            gidx = np.asarray(t.sample_indices)
            row["eval_subjects"] = sorted(set(subjects[gidx][ev_idx].tolist()))
        task_meta.append(row)
    (ddir / "tasks.json").write_text(json.dumps({
        "dataset": TARGET, "n_tasks": len(tasks), "slice": "RandomClassSubsetSlicer",
        "split_protocol": "subject_disjoint" if subject else "stratified",
        "tasks": task_meta}, indent=2, ensure_ascii=False), encoding="utf-8")

    # 任务特征：共享 TS2Vec 只 load 不 fit（铁律 6 核心；指纹校验 fit 池 = 源池）。
    # 特征不含标签/切分 → 两协议共享（subject 模式仅校验已存在，不重算）。
    feat_dir = outdir / "task_features"
    if subject:
        missing = [t.task_id for t in tasks
                   if not (feat_dir / f"{t.task_id}.npz").exists()]
        assert not missing, f"任务特征缺失 {missing}：先跑首轮 --prep"
        print("任务特征复用首轮产物（切分无关）", flush=True)
    else:
        meta = json.loads((LABELS_DIR / "shared_ts2vec.meta.json").read_text(encoding="utf-8"))
        assert TARGET not in meta["fit_datasets"], "共享 TS2Vec fit 池含目标数据集！（铁律 6）"
        assert sorted(meta["fit_datasets"]) == ["Handwriting", "UniMiB-SHAR"], \
            f"fit 池 {meta['fit_datasets']} ≠ 决定性实验源池"
        shared = TS2VecWrapper.load(LABELS_DIR / "shared_ts2vec.pt", device=device)
        assert shared.frozen and shared.n_iters > 0, "共享 TS2Vec 必须已冻结（铁律 6）"
        init_path = LABELS_DIR / "set_encoder_init.pt"
        assert init_path.exists(), "缺 set_encoder_init.pt（决定性实验产物）"
        ra = labels_cfg["route_a"]
        torch.manual_seed(ra["embed_seed"])
        set_enc = TaskSetEncoder(dim_input=shared.output_dims, nz=ra["nz"])
        set_enc.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
        feat_dir.mkdir(exist_ok=True)
        for t in tasks:
            _r2b.ensure_task_features(t, X_train, y_train, shared, set_enc, ra, feat_dir)
        print("任务特征就绪（冻结 TS2Vec 只编码，零训练）", flush=True)

    # 随机初始化基线：8 任务 × 5 种子 iters=0（split 恒 canonical——subject 模式为
    # 任务级组切分、否则分层切分；weight 逐种子派生。源池探针同口径）
    cache_path = ddir / "baseline_cache.jsonl"
    done = {(r["task_id"], r["seed"]) for r in read_jsonl(cache_path)}
    work = [(t, s) for t in tasks for s in range(N_BASELINE_SEEDS)]
    todo = [w for w in work if not (resume and (w[0].task_id, w[1]) in done)]
    print(f"基线: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    base_cls = inert_cls()
    split_cache: dict[str, tuple] = {}   # task_id → (tr, ev, seed)（组切分复用）

    def split_of(t):
        if t.task_id not in split_cache:
            _Xt, yt = materialize_task(t, X_train, y_train)
            split_cache[t.task_id] = task_split(t, yt)
        return split_cache[t.task_id]

    # threading 后端：fit_svm 的 GridSearchCV(n_jobs=-1) 在 Windows 上逐条 spawn
    # loky 进程池（占 99% 耗时；R' 逐位同值）——长期记忆教训，勿删
    with parallel_backend("threading", n_jobs=1):
        for i, (t, s) in enumerate(todo):
            w_seed, _canon_sp = baseline_seed_specs(t.task_id, s)
            tr_idx, ev_idx, sp_seed = split_of(t)

            def make_row(rec, _t=t, _s=s):
                rp = rec["r_prime"]
                return {"dataset": TARGET, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": "baseline", "seed": _s, "iters": 0, **rec,
                        "split_protocol": "subject_disjoint" if subject else "stratified",
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            full = guarded_run(
                lambda: evaluate_rprime(t, base_cls, X_train, y_train,
                                        weight_seed=w_seed, split_seed=sp_seed,
                                        iters=0, eval_ratio=eval_ratio,
                                        split_indices=(tr_idx, ev_idx) if subject else None,
                                        device=device),
                make_row, cache_path=cache_path, inflight=ddir / "inflight.txt",
                task_id=t.task_id, cls_key="baseline", iters=0,
                on_oom=ON_OOM_SKIP, oom_log=ddir / "oom_failures.jsonl",
                context={"dataset": TARGET, "baseline_seed": s},
                log=lambda m: print(f"[base {i + 1}/{len(todo)}] {m}", flush=True))
            if full is not None:
                print(f"[base {i + 1}/{len(todo)}] {t.task_id} s{s} "
                      f"R'={full['r_prime']}", flush=True)

        if subject:
            # 全量跨受试者随机初始化 val 基线：诊断已按同一留出组测过（3 种子），
            # 逐字段复用，不重烧 GPU
            diag = json.loads((outdir / "subject_leak_diag.json").read_text(encoding="utf-8"))
            dj = diag["subject_disjoint_random_init_val"]
            assert tuple(dj["held_out_subjects"]) == VAL_HOLDOUT_SUBJECTS
            full_base, n_fit, n_val = dj["seeds"], dj["n_fit"], dj["n_val"]
            print(f"[full-base] 复用 subject_leak_diag: {full_base}", flush=True)
        else:
            # 全量随机初始化 val 基线（贴顶判断的全量口径；只用 train/val，不碰 test）
            fit_idx, val_idx = val_split(y_train)
            n_fit, n_val = len(fit_idx), len(val_idx)
            full_base = []
            for s in range(N_FULLBASE_SEEDS):
                torch.manual_seed(s)
                enc = build_encoder(input_dims=X_train.shape[-1])
                f_fit = encode_features(enc, X_train[fit_idx], device=device)
                f_val = encode_features(enc, X_train[val_idx], device=device)
                clf = fit_svm(f_fit, y_train[fit_idx], seed=s)
                acc = float(np.mean(clf.predict(f_val) == y_train[val_idx]))
                full_base.append(acc)
                print(f"[full-base s{s}] val_acc={acc:.4f}", flush=True)

    rows_by_task = defaultdict(list)
    for r in read_jsonl(cache_path):
        rows_by_task[r["task_id"]].append(r)
    n_classes_of = {t.task_id: len(t.classes) for t in tasks}
    per_task = {tid: summarize_task(rows, n_classes=n_classes_of[tid])
                for tid, rows in rows_by_task.items()}
    summary = {
        "split_protocol": "subject_disjoint" if subject else "stratified",
        "per_task": per_task,
        "dataset": summarize_dataset(per_task),
        "full_random_init_val": {"seeds": full_base,
                                 "mean": statistics.fmean(full_base),
                                 "n_fit": int(n_fit), "n_val": int(n_val)},
        "shared_ts2vec_md5": file_md5(LABELS_DIR / "shared_ts2vec.pt"),
        "shared_ts2vec_fit_pool": json.loads(
            (LABELS_DIR / "shared_ts2vec.meta.json").read_text(encoding="utf-8")
        )["fit_datasets"],
    }
    (ddir / "prep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    d = summary["dataset"]
    print(f"\n基线汇总: 任务级 mean={d['baseline_mean']:.4f} "
          f"[{d['task_min']:.4f}, {d['task_max']:.4f}] tier={d['tier']} | "
          f"全量随机初始化 val={statistics.fmean(full_base):.4f}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 2：--calibrate 工作点标定（{1,2,5,10,20}ep × 10 shared × 前 2 任务）
# ---------------------------------------------------------------------------


def stage_calibrate(outdir: Path, reg, labels_cfg: dict, train_cfg: dict,
                    device: str, resume: bool, subject: bool = False) -> None:
    ddir = data_dir(outdir, subject)
    ddir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_target_train()
    tasks = target_tasks(reg, y_train)
    probe = tasks[:N_PROBE_TASKS]
    eval_ratio = labels_cfg["eval_ratio"]
    subjects = load_train_subjects(len(y_train)) if subject else None
    spe = probe_spe(tasks, X_train, y_train, eval_ratio, subjects=subjects)
    iters_grid = grid_iters(spe)
    shared_cls = load_probe_cls(PROBE_CATALOG, N_SHARED_CLS)[0]
    print(f"标定: spe={spe} → iters 网格 {iters_grid}（= {list(EP_GRID)} ep"
          f"{'，跨受试者协议' if subject else ''}）", flush=True)

    split_cache: dict[str, tuple] = {}

    def split_of(t):
        """subject 模式的任务级组切分（canonical、任务内恒定）；否则 None。"""
        if not subject:
            return None, run_seed_for(t.task_id, "train_eval_split")
        if t.task_id not in split_cache:
            _Xt, yt = materialize_task(t, X_train, y_train)
            split_cache[t.task_id] = subject_task_split(t, subjects, yt, eval_ratio)
        tr, ev, seed = split_cache[t.task_id]
        return (tr, ev), seed

    fill_path = ddir / "trained_fill.jsonl"
    cache = load_rprime_cache(fill_path)
    work = [(t, c, k, it) for it in iters_grid for t in probe for c, k in shared_cls]
    todo = [w for w in work if not (resume and (w[0].task_id, w[2], w[3]) in cache)]
    print(f"标定 fill: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：同 stage_prep 处注释（loky spawn 99% 耗时教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (t, c, k, it) in enumerate(todo):
            w_seed = run_seed_for(t.task_id, k)
            split_idx, sp_seed = split_of(t)

            def make_row(rec, _t=t, _k=k, _c=c, _it=it):
                rp = rec["r_prime"]
                return {"dataset": TARGET, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "origin": "shared", "seed": 0, "iters": _it,
                        **rec, "r_prime": None if not np.isfinite(rp) else float(rp),
                        "split_protocol": "subject_disjoint" if subject else "stratified",
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(t, c, X_train, y_train, weight_seed=w_seed,
                                        split_seed=sp_seed, iters=it,
                                        eval_ratio=eval_ratio,
                                        batch_size=train_cfg["batch_size"],
                                        lr=train_cfg["lr"], device=device,
                                        split_indices=split_idx),
                make_row, cache_path=fill_path, inflight=ddir / "inflight.txt",
                task_id=t.task_id, cls_key=k, iters=it,
                on_oom=ON_OOM_SKIP, oom_log=ddir / "oom_failures.jsonl",
                context={"dataset": TARGET},
                log=lambda m: print(f"[cal {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            cache[(t.task_id, k, it)] = full
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[cal {i + 1}/{len(todo)}] {t.task_id}@{it} {k[:8]} "
                  f"R'={full['r_prime']:.4f} ({full['seconds']}s) "
                  f"ETA {eta / 60:.1f}min", flush=True)

    # 臂汇总（iters_workpoint_scan.collect 同式）→ 选点（confirm 协议 §2）
    base_rows = defaultdict(list)
    for r in read_jsonl(ddir / "baseline_cache.jsonl"):
        if r["task_id"] in {t.task_id for t in probe}:
            base_rows[r["task_id"]].append(r)
    by_id = {t.task_id: t for t in probe}
    arm_rows = N_SHARED_CLS * N_PROBE_TASKS
    arms = {}
    for it in iters_grid:
        per_task = {}
        for tid, rows in base_rows.items():
            trained = [cache[(tid, k, it)]["r_prime"] for _c, k in shared_cls
                       if (tid, k, it) in cache]
            per_task[tid] = summarize_task(rows, n_classes=len(by_id[tid].classes),
                                           trained=trained)
        d = summarize_dataset(per_task)
        n_tr, n_up, n_dn = (d["n_trained_total"], d["n_trained_above_null"],
                            d["n_trained_below_null"])
        bg = d["best_gain_max"]
        arm = {
            "epoch": it / spe, "n_trained": n_tr, "complete": n_tr >= arm_rows,
            "baseline": d["baseline_mean"],
            "realized_gain": d["realized_gain_mean"], "best_gain": bg,
            "n_above": n_up, "n_below": n_dn, "n_band": n_tr - n_up - n_dn,
            "T1": d["baseline_mean"] <= CEILING_BASELINE,
            "T2": bool(bg is not None and bg >= BEST_GAIN_FLOOR),
            "T3": bool(n_tr > 0 and n_up >= T3_ABOVE_FRACTION * n_tr
                       and n_up >= T3_ABOVE_OVER_BELOW * n_dn),
            "verdict": d["verdict"],
        }
        arm["usable"] = bool(arm["T1"] and arm["T2"] and arm["T3"] and arm["complete"])
        arms[it] = arm

    complete = {it: a for it, a in arms.items() if a["complete"]}
    selection = select_workpoint(complete, margin=WORKPOINT_MARGIN)
    quantum = max((1.0 / r["n_eval"] for rows in base_rows.values() for r in rows
                   if r.get("n_eval")), default=None)
    nw = (no_workpoint_verdict(complete, quantum=quantum)
          if quantum and complete else None)

    # 破坏参与率诊断（δ=0.05 主口径 + 0.03 敏感性；目标上不造 label，仅诊断）
    destruction = {}
    if selection:
        it = selection["iters"]
        bands = {tid: (per["baseline_mean"], per["null_band"])
                 for tid, per in ((tid, summarize_task(rows, n_classes=len(by_id[tid].classes)))
                                  for tid, rows in base_rows.items())}
        pt = {tid: {k: cache[(tid, k, it)]["r_prime"] for _c, k in shared_cls
                    if (tid, k, it) in cache} for tid in base_rows}
        destruction = {str(dl): destruction_participation(pt, bands, delta=dl)
                       for dl in DESTRUCTION_DELTAS}

    wp = {"dataset": TARGET, "spe": spe, "ep_grid": list(EP_GRID),
          "iters_grid": iters_grid, "margin": WORKPOINT_MARGIN,
          "split_protocol": "subject_disjoint" if subject else "stratified",
          "arms": {str(k): v for k, v in arms.items()},
          "selection": selection, "no_workpoint": nw,
          "destruction_at_selection": destruction,
          "workpoint_ep": (selection["iters"] / spe) if selection else None}
    (ddir / "workpoint.json").write_text(
        json.dumps(wp, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'iters':>6}{'ep':>5}{'基线':>8}{'实得':>8}{'最好':>8}"
          f"{'↑':>4}{'↓':>4}{'带内':>5}  T1 T2 T3 usable")
    tick = lambda b: "✅" if b else "❌"
    for it in iters_grid:
        a = arms[it]
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else "  —  "
        print(f"{it:>6}{a['epoch']:>5.0f}{a['baseline']:>8.3f}"
              f"{fmt(a['realized_gain']):>8}{fmt(a['best_gain']):>8}"
              f"{a['n_above']:>4}{a['n_below']:>4}{a['n_band']:>5}  "
              f"{tick(a['T1'])} {tick(a['T2'])} {tick(a['T3'])} {tick(a['usable'])}")
    if selection is None:
        print("\n⚠ 无可确认工作点（usable 臂为空）——UCI-HAR 判 inert，按协议停下报告；"
              f"no_workpoint 诊断: {nw}", flush=True)
    else:
        print(f"\n工作点: iters={selection['iters']}"
              f"（{selection['iters'] / spe:.0f}ep）best_gain={selection['best_gain']:+.3f} "
              f"候选臂 {selection['candidates']}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 3a：--tclsc 重训 R1-R3 mixed（decisive 同种子同数据）并存 checkpoint
# ---------------------------------------------------------------------------


def stage_tclsc(outdir: Path, labels_cfg: dict, cfg: dict, device: str,
                resume: bool) -> None:
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    wp = labels_cfg["workpoint"]
    iters_map, delta_map = wp["iters_per_dataset"], wp["delta_per_dataset"]
    task_type = labels_cfg["task_type"]
    cache = load_rprime_cache(LABELS_DIR / "rprime_cache.jsonl")
    records = []
    for d in _4b.DATASETS:
        rs = comparator_records(cache, encoder, iters=iters_map[d])
        assert all(_4b.ds_of(r.task_id) == d for r in rs)
        records += rs
    all_tids = sorted({r.task_id for r in records})
    assert len(all_tids) == 28, f"应有 28 源任务，得到 {len(all_tids)}"
    fwd_pairs = _4b.forward_pairs_by_task(records, task_type, delta_map)
    feat_dir = LABELS_DIR / "task_features"
    task_windows = {t: np.load(feat_dir / f"{t}.npz")["windows_repr"]
                    for t in all_tids}
    init_path = LABELS_DIR / "set_encoder_init.pt"
    dim_input = next(iter(task_windows.values())).shape[-1]
    tdir = outdir / "tclsc"
    tdir.mkdir(exist_ok=True)

    metas = {}
    for spec in _4b.build_runs(all_tids)[:3]:   # 仅 R1-R3 mixed（decisive 主判定跑）
        run_id, seed = spec["run_id"], spec["seed"]
        out_path = tdir / f"{run_id}.pt"
        meta_path = tdir / f"{run_id}.json"
        if resume and out_path.exists() and meta_path.exists():
            metas[run_id] = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"[{run_id}] 已有 checkpoint，跳过", flush=True)
            continue
        train_tids, test_tids = spec["train"], spec["test"]
        train_records = [r for r in records if r.task_id in train_tids]
        fwd = [p for p in fwd_pairs if p.task_id in test_tids]
        directed = with_reversed(fwd)
        delta_by_task = {t: delta_map[_4b.ds_of(t)] for t in (train_tids | test_tids)}

        # 训练序列与 4b.run_one 逐行同构（同种子同数据 → 复现到 GPU 非确定性内）
        torch.manual_seed(seed)
        set_enc = TaskSetEncoder(dim_input=dim_input, nz=labels_cfg["nz"])
        set_enc.load_state_dict(torch.load(init_path, map_location="cpu",
                                           weights_only=True))
        model = TCLSC(cls_dim=encoder.dim, set_encoder=set_enc,
                      embedding_dim=cfg["embedding_dim"], fz=cfg["fz"])
        engine = TCLSCEngine(model, task_windows, device=device,
                             lr=cfg["lr"], batch_size=cfg["batch_size"], seed=seed)
        t0 = time.time()
        history = engine.train(train_records, epochs=cfg["epochs"],
                               curriculum_epochs=cfg["curriculum_epochs"],
                               better_fn=lambda a, b: is_better(a, b, task_type),
                               delta=delta_by_task)
        base = engine.evaluate(directed)
        ref_path = DECISIVE_RUNS / f"{run_id}.json"
        ref_acc = (json.loads(ref_path.read_text(encoding="utf-8"))["normal"]["acc"]
                   if ref_path.exists() else None)
        torch.save(model.state_dict(), out_path)
        meta = {"run_id": run_id, "seed": seed,
                "train_seconds": round(time.time() - t0, 1),
                "bce_last": history[-1]["loss"],
                "normal_acc": base["acc"], "ref_acc": ref_acc,
                "acc_delta_vs_ref": (base["acc"] - ref_acc) if ref_acc else None,
                "n_train_records": len(train_records),
                "model_md5": file_md5(out_path)}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        metas[run_id] = meta
        flag = ("" if ref_acc is None or abs(base["acc"] - ref_acc) <= 0.01
                else " ⚠ 偏离参考 >0.01")
        print(f"[{run_id}] {meta['train_seconds']}s normal={base['acc']:.4f} "
              f"(decisive 参考 {ref_acc}){flag}", flush=True)
    (tdir / "meta.json").write_text(json.dumps(metas, indent=2, ensure_ascii=False),
                                    encoding="utf-8")


# ---------------------------------------------------------------------------
# 阶段 3b：--rank 比较器排名（Round-Robin × 3 种子多数票 → Borda）+ E 消融
# ---------------------------------------------------------------------------


def _rank_once(models, vecs_t, e_rows_per_model, keys) -> tuple[list[int], list, np.ndarray]:
    """给定每模型每任务的 E 行 → (Borda 排名, 每任务胜场, tiebreak 得分)。"""
    n_tasks = len(e_rows_per_model[0])
    wins_list, s_accum = [], []
    for ti in range(n_tasks):
        S_per_model = []
        for mi, model in enumerate(models):
            P = pair_prob_matrix(model, vecs_t, e_rows_per_model[mi][ti])
            S_per_model.append(symmetrize(P))
        V = pair_vote_matrix(S_per_model)
        wins_list.append(V.sum(axis=1))
        s_accum.append(np.stack(S_per_model))
    all_S = np.concatenate(s_accum)          # (n_tasks*n_models, n, n)
    import warnings as _w
    with np.errstate(invalid="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)  # 对角全 NaN 的空切片均值
        scores = np.nanmean(np.nanmean(all_S, axis=0), axis=1)
    order = borda_ranking(wins_list, scores, keys)
    return order, wins_list, scores


def stage_rank(outdir: Path, labels_cfg: dict, cfg: dict, device: str) -> None:
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space)
    pool = build_candidate_pool()
    keys = [k for _o, _c, k in pool]
    origins = {k: o for o, _c, k in pool}
    ggs_k = cls_key(_m1.GGS)
    (outdir / "candidates.json").write_text(json.dumps(
        [{"origin": o, "cls_key": k, "cls": dataclasses.asdict(c)}
         for o, c, k in pool], indent=2, ensure_ascii=False), encoding="utf-8")
    vecs = torch.as_tensor(np.stack([enc.encode(c) for _o, c, _k in pool]),
                           dtype=torch.float32, device=device)

    tids = sorted(json.loads((outdir / "tasks.json").read_text(encoding="utf-8"))
                  ["tasks"], key=lambda t: t["task_id"])
    tids = [t["task_id"] for t in tids]
    windows = {tid: torch.as_tensor(
        np.load(outdir / "task_features" / f"{tid}.npz")["windows_repr"],
        dtype=torch.float32, device=device) for tid in tids}

    tdir = outdir / "tclsc"
    run_ids = ["R1_mixed_s0", "R2_mixed_s1", "R3_mixed_s2"]
    models = []
    for rid in run_ids:
        m = build_tclsc_skeleton(space, next(iter(windows.values())).shape[-1],
                                 cfg, labels_cfg["nz"])
        m.load_state_dict(torch.load(tdir / f"{rid}.pt", map_location="cpu",
                                     weights_only=True))
        models.append(m.to(device).eval())

    # 每模型每任务的 E（trained SetTF；E 支路走完整流程——用户拍板④）
    e_rows = [[m.encode_tasks([windows[tid]])[0] for tid in tids] for m in models]

    order, wins_list, scores = _rank_once(models, vecs, e_rows, keys)

    # 每模型单独排名（种子稳定性：Kendall τ）
    per_model_orders = []
    for mi, m in enumerate(models):
        w_list, s_sum = [], np.zeros(len(keys))
        for ti in range(len(tids)):
            S = symmetrize(pair_prob_matrix(m, vecs, e_rows[mi][ti]))
            w_list.append(wins_from_scores(S))
            with np.errstate(invalid="ignore"):
                s_sum += np.nanmean(S, axis=1)
        per_model_orders.append(borda_ranking(w_list, s_sum / len(tids), keys))

    def rank_pos(o):  # 排名序 → 每候选名次向量（Kendall τ 用）
        pos = np.empty(len(o), dtype=int)
        for r, i in enumerate(o):
            pos[i] = r
        return pos

    taus = [float(kendalltau(rank_pos(per_model_orders[a]),
                             rank_pos(per_model_orders[b])).statistic)
            for a in range(3) for b in range(a + 1, 3)]

    # E 消融（用户拍板④：目标池观察一次）：shuffle=目标任务间错位 ×5 种子；zero
    npos = rank_pos(order)
    ablations = {}
    for mode in ["shuffle", "zero"]:
        rows = []
        seeds = range(N_ABL_SEEDS) if mode == "shuffle" else [0]
        for s in seeds:
            if mode == "shuffle":
                perm = derangement(len(tids), np.random.default_rng([s, 20260811]))
                e_abl = [[e_rows[mi][perm[ti]] for ti in range(len(tids))]
                         for mi in range(3)]
            else:
                zero = torch.zeros_like(e_rows[0][0])
                e_abl = [[zero for _ in tids] for _ in range(3)]
            o_abl, _w, _s = _rank_once(models, vecs, e_abl, keys)
            rows.append({
                "seed": s,
                "kendall_tau_vs_normal": float(kendalltau(npos, rank_pos(o_abl)).statistic),
                "top3_overlap": len(set(o_abl[:TOP_K]) & set(order[:TOP_K])),
                "top1_same": o_abl[0] == order[0],
                "ggs_rank": [keys[i] for i in o_abl].index(ggs_k) + 1,
                "top10": [keys[i] for i in o_abl[:10]],
            })
        ablations[mode] = rows

    ranking = {
        "n_candidates": len(keys), "n_tasks": len(tids), "models": run_ids,
        "order": [{"rank": r + 1, "cls_key": keys[i], "origin": origins[keys[i]],
                   "borda_wins": int(sum(int(w[i]) for w in wins_list)),
                   "mean_score": float(scores[i])}
                  for r, i in enumerate(order)],
        "top3": [keys[i] for i in order[:TOP_K]],
        "top10": [keys[i] for i in order[:10]],
        "ggs_rank": [keys[i] for i in order].index(ggs_k) + 1,
        "per_task_wins": {tids[ti]: {keys[i]: int(wins_list[ti][i])
                                     for i in range(len(keys))}
                          for ti in range(len(tids))},
        "per_model_kendall_tau": taus,
        "per_model_top3": [[keys[i] for i in o[:TOP_K]] for o in per_model_orders],
        "ablations": ablations,
    }
    (outdir / "ranking.json").write_text(
        json.dumps(ranking, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n比较器排名（31 候选 × {len(tids)} 任务 × 3 种子多数票 → Borda）：")
    for row in ranking["order"][:10]:
        print(f"  #{row['rank']:>2} {row['cls_key'][:8]} [{row['origin']}] "
              f"wins={row['borda_wins']} score={row['mean_score']:.3f}")
    print(f"GGS 名次: #{ranking['ggs_rank']}/31 | 种子间 τ={taus} | "
          f"E 消融 shuffle τ={[r['kendall_tau_vs_normal'] for r in ablations['shuffle']]} "
          f"zero τ={ablations['zero'][0]['kendall_tau_vs_normal']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 4：--sweep 全池真值（31 候选 @工作点全量预训练 + val 评估；测试集不碰）
# ---------------------------------------------------------------------------


def pretrain_full(cls_cfg, seed: int, iters: int, X_train: np.ndarray,
                  train_cfg: dict, device: str):
    torch.manual_seed(seed)
    encoder = build_encoder(input_dims=X_train.shape[-1])
    pretrain_encoder(encoder, cls_cfg, X_train, max_iters=iters,
                     batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                     device=device, seed=seed)
    return encoder


def encoder_path(outdir: Path, key: str, iters: int, seed: int) -> Path:
    return outdir / "encoders" / f"{key}_it{iters}_s{seed}.pt"


def ensure_pretrained(outdir: Path, cls_cfg, key: str, seed: int, iters: int,
                      X_train, train_cfg, device: str):
    """预训练 encoder（已存在则加载）——sweep 与 final 共用，保证一条 (cls, seed,
    iters) 只训一次（铁律 3 精神：最贵一步落盘复用）。"""
    p = encoder_path(outdir, key, iters, seed)
    encoder = build_encoder(input_dims=X_train.shape[-1])
    if p.exists():
        encoder.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        return encoder, True
    encoder = pretrain_full(cls_cfg, seed, iters, X_train, train_cfg, device)
    p.parent.mkdir(exist_ok=True)
    torch.save(encoder.state_dict(), p)
    return encoder, False


def stage_sweep(outdir: Path, train_cfg: dict, device: str, resume: bool,
                subject: bool = False) -> None:
    ddir = data_dir(outdir, subject)
    wp = load_workpoint(ddir)
    X_train, y_train = load_target_train()
    ep = wp["workpoint_ep"]
    iters = full_iters_for_ep(ep, n_train=len(y_train))
    pool = build_candidate_pool()
    ggs_k = cls_key(_m1.GGS)
    if subject:
        fit_idx, val_idx = subject_val_split(load_train_subjects(len(y_train)))
    else:
        fit_idx, val_idx = val_split(y_train)
    print(f"sweep: 工作点 {ep:.0f}ep → 全量 {iters} iters | "
          f"fit/val = {len(fit_idx)}/{len(val_idx)}"
          f"（{'跨受试者' if subject else 'train 内分层'}切分，test 不碰）", flush=True)

    cache_path = ddir / "sweep_cache.jsonl"
    done = {(r["cls_key"], r["iters"], r["seed"]) for r in read_jsonl(cache_path)}
    # 31 × seed0 + GGS 补种子 1/2（对照误差棒；winner 补种子在 --final 判明后跑）
    work = [(o, c, k, 0) for o, c, k in pool] + \
           [(GGS_ORIGIN, _m1.GGS, ggs_k, s) for s in (1, 2)]
    todo = [w for w in work if not (resume and (w[2], iters, w[3]) in done)]
    print(f"sweep: 计划 {len(work)} 跑，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：同 stage_prep 处注释（loky spawn 99% 耗时教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (org, c, k, seed) in enumerate(todo):
            t1 = time.time()
            try:
                encoder, cached = ensure_pretrained(outdir, c, k, seed, iters,
                                                    X_train, train_cfg, device)
                f_fit = encode_features(encoder, X_train[fit_idx], device=device)
                f_val = encode_features(encoder, X_train[val_idx], device=device)
                clf = fit_svm(f_fit, y_train[fit_idx], seed=seed)
                val_acc = float(np.mean(clf.predict(f_val) == y_train[val_idx]))
            except Exception as e:  # noqa: BLE001
                if is_oom_error(e):
                    torch.cuda.empty_cache()
                    print(f"[sweep {i + 1}/{len(todo)}] {k[:8]} s{seed} OOM，跳过"
                          f"（--resume 重试）: {e}", flush=True)
                    continue
                raise
            rec = {"dataset": TARGET, "scope": "full_train", "cls_key": k,
                   "origin": org, "iters": iters, "workpoint_ep": ep, "seed": seed,
                   "val_acc": val_acc, "svm_c": float(clf.C),
                   "split_protocol": "subject_disjoint" if subject else "stratified",
                   "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
                   "seconds": round(time.time() - t1, 1),
                   "encoder_cached": cached, "device_name": device_label(device)}
            append_jsonl(cache_path, rec)
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[sweep {i + 1}/{len(todo)}] {org}:{k[:8]} s{seed} "
                  f"val={val_acc:.4f} ({rec['seconds']}s) ETA {eta / 60:.1f}min",
                  flush=True)

    rows = [r for r in read_jsonl(cache_path)
            if r["iters"] == iters and r["seed"] == 0]
    rows.sort(key=lambda r: (-r["val_acc"], r["cls_key"]))
    ranking = json.loads((outdir / "ranking.json").read_text(encoding="utf-8"))
    comp_order = [r["cls_key"] for r in ranking["order"]]
    val_order = [r["cls_key"] for r in rows]
    comp_pos = {k: i for i, k in enumerate(comp_order)}
    val_accs = {r["cls_key"]: r["val_acc"] for r in rows}
    tau = float(kendalltau([comp_pos[k] for k in val_order],
                           list(range(len(val_order)))).statistic)
    summary = {
        "iters": iters, "workpoint_ep": ep,
        "split_protocol": "subject_disjoint" if subject else "stratified",
        "val_ranking": [{"rank": i + 1, "cls_key": r["cls_key"],
                         "origin": r["origin"], "val_acc": r["val_acc"],
                         "comparator_rank": comp_pos[r["cls_key"]] + 1}
                        for i, r in enumerate(rows)],
        "true_best": {"cls_key": val_order[0], "val_acc": rows[0]["val_acc"],
                      "comparator_rank": comp_pos[val_order[0]] + 1},
        "ggs_val_rank": val_order.index(ggs_k) + 1 if ggs_k in val_order else None,
        "kendall_tau_comparator_vs_val": tau,
        "top3_val_accs": {k: val_accs.get(k) for k in ranking["top3"]},
    }
    (ddir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tb = summary["true_best"]
    print(f"\n真值(val)最优: {tb['cls_key'][:8]} val={tb['val_acc']:.4f} → "
          f"比较器名次 #{tb['comparator_rank']} | τ(比较器,val)={tau:.3f}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 5：--final top-3/GGS/winner 测试评估 + @2320 桥接（测试集唯一触碰点）
# ---------------------------------------------------------------------------


def stage_final(outdir: Path, train_cfg: dict, device: str, resume: bool,
                subject: bool = False) -> None:
    ddir = data_dir(outdir, subject)
    wp = load_workpoint(ddir)
    ranking = json.loads((outdir / "ranking.json").read_text(encoding="utf-8"))
    X_train, y_train = load_target_train()
    X_test, y_test = load_target_test()   # ← 测试集全流程唯一加载点
    ep = wp["workpoint_ep"]
    iters = full_iters_for_ep(ep, n_train=len(y_train))
    pool = {k: (o, c) for o, c, k in build_candidate_pool()}
    ggs_k = cls_key(_m1.GGS)

    evals_path = ddir / "final_evals.jsonl"
    done = {(r["cls_key"], r["iters"], r["seed"]) for r in read_jsonl(evals_path)}

    def test_eval(key: str, seed: int, it: int) -> dict:
        """预训练（缓存命中则复用 sweep 的 encoder）+ M1 口径测试评估。"""
        if resume and (key, it, seed) in done:
            return next(r for r in read_jsonl(evals_path)
                        if (r["cls_key"], r["iters"], r["seed"]) == (key, it, seed))
        org, c = pool[key]
        t0 = time.time()
        encoder, cached = ensure_pretrained(outdir, c, key, seed, it,
                                           X_train, train_cfg, device)
        res = evaluate_classification(encoder, X_train, y_train, X_test, y_test,
                                      device=device, seed=seed)
        rec = {"dataset": TARGET, "scope": "official_test", "cls_key": key,
               "origin": org, "iters": it, "seed": seed,
               "test_acc": res["acc"], "test_f1": res["f1"], "svm_c": res["svm_c"],
               "seconds": round(time.time() - t0, 1), "encoder_cached": cached,
               "device_name": device_label(device)}
        append_jsonl(evals_path, rec)
        done.add((key, it, seed))
        print(f"[final] {org}:{key[:8]} s{seed}@{it} test={res['acc']:.4f} "
              f"({rec['seconds']}s)", flush=True)
        return rec

    # threading 后端：同 stage_prep 处注释（loky spawn 99% 耗时教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        top3 = ranking["top3"]
        top3_evals = {k: test_eval(k, 0, iters) for k in top3}
        ggs_evals = {s: test_eval(ggs_k, s, iters) for s in (0, 1, 2)}
        winner = max(top3, key=lambda k: top3_evals[k]["test_acc"])
        winner_evals = {0: top3_evals[winner]}
        for s in (1, 2):
            winner_evals[s] = test_eval(winner, s, iters)
        # 桥接：winner @2320（M1 口径）；winner==GGS 时直接引用 M1 锚点不重跑
        if winner == ggs_k:
            bridge = {"cls_key": winner, "iters": 2320, "seed": 0,
                      "test_acc": M1_GGS_2320["rerun_4060"],
                      "source": "results/m1_rerun/runA（GGS 即 winner，引用 M1 复跑锚点）"}
        else:
            bridge = test_eval(winner, 0, 2320)

    ggs_accs = [ggs_evals[s]["test_acc"] for s in (0, 1, 2)]
    win_accs = [winner_evals[s]["test_acc"] for s in (0, 1, 2)]
    final = {
        "workpoint_ep": ep, "iters": iters,
        "top3": [{"cls_key": k, "origin": pool[k][0],
                  "comparator_rank": top3.index(k) + 1,
                  "test_acc": top3_evals[k]["test_acc"]} for k in top3],
        "winner": {"cls_key": winner, "origin": pool[winner][0],
                   "test_acc_seed0": winner_evals[0]["test_acc"],
                   "test_acc_seeds": win_accs,
                   "test_acc_mean": statistics.fmean(win_accs)},
        "ggs_at_workpoint": {"test_acc_seeds": ggs_accs,
                             "test_acc_mean": statistics.fmean(ggs_accs)},
        "bridge_2320": bridge,
        "m1_anchor": M1_GGS_2320,
        "delta_winner_vs_ggs_workpoint":
            statistics.fmean(win_accs) - statistics.fmean(ggs_accs),
        "n_test_set_evals": len(read_jsonl(evals_path)),
        "split_protocol": wp.get("split_protocol", "stratified"),
    }
    (ddir / "final.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntop-3 测试: {[(k[:8], round(top3_evals[k]['test_acc'], 4)) for k in top3]}")
    print(f"winner {winner[:8]} mean={statistics.fmean(win_accs):.4f} {win_accs}")
    print(f"GGS@工作点 mean={statistics.fmean(ggs_accs):.4f} {ggs_accs}")
    print(f"桥接 @2320: {bridge['test_acc']:.4f} | M1 锚点 {M1_GGS_2320}", flush=True)


# ---------------------------------------------------------------------------
# 阶段 6：--verify 收官校验 + 泄漏自检（铁律 6 最后一道关）
# ---------------------------------------------------------------------------


def stage_verify(outdir: Path, reg, subject: bool = False) -> None:
    ddir = data_dir(outdir, subject)
    # 1. 源加载路径 fail-fast（复跑自检）
    try:
        load_source_dataset(TARGET, reg)
        leak_ok = False
    except TargetPoolLeakageError:
        leak_ok = True
    # 2. 共享 TS2Vec 指纹：与 labels_workpoint 落盘一致 + fit 池不含目标
    meta = json.loads((LABELS_DIR / "shared_ts2vec.meta.json").read_text(encoding="utf-8"))
    prep = json.loads((ddir / "prep_summary.json").read_text(encoding="utf-8"))
    ts2vec_ok = (prep["shared_ts2vec_md5"] == file_md5(LABELS_DIR / "shared_ts2vec.pt")
                 and TARGET not in meta["fit_datasets"])
    # 3. T-CLSC 只吃源池数据：meta 里记录的训练记录数 + checkpoint hash 未变
    tmeta = json.loads((outdir / "tclsc" / "meta.json").read_text(encoding="utf-8"))
    tclsc_ok = all(m["model_md5"] == file_md5(outdir / "tclsc" / f"{rid}.pt")
                   for rid, m in tmeta.items())
    # 4. 测试集触碰清单：只有 final_evals.jsonl 的 official_test 记录
    finals = read_jsonl(ddir / "final_evals.jsonl")
    test_scopes = {r.get("scope") for r in finals}
    # 5. 缓存完整性
    print(f"== 收官校验（{'跨受试者协议' if subject else '分层协议'}）==")
    for name, path, keyf in [
        ("baseline_cache", ddir / "baseline_cache.jsonl",
         lambda r: (r["task_id"], r["seed"], r["iters"])),
        ("trained_fill", ddir / "trained_fill.jsonl",
         lambda r: (r["task_id"], r["cls_key"], r["iters"])),
        ("sweep_cache", ddir / "sweep_cache.jsonl",
         lambda r: (r["cls_key"], r["iters"], r["seed"])),
        ("final_evals", ddir / "final_evals.jsonl",
         lambda r: (r["cls_key"], r["iters"], r["seed"])),
    ]:
        rows = read_jsonl(path)
        ks = [keyf(r) for r in rows]
        n_dev = sum(1 for r in rows if r.get("device_name"))
        n_null = sum(1 for r in rows if r.get("r_prime", r.get("val_acc",
                     r.get("test_acc"))) is None)
        print(f"  {name}: {len(rows)} 行 | 唯一键 {len(set(ks))}"
              f"{'' if len(ks) == len(set(ks)) else ' ⚠重复'} | null {n_null} | "
              f"device {n_dev}/{len(rows)}")
    print("== 泄漏自检（铁律 6）==")
    print(f"  ① load_source_dataset(UCI-HAR) fail-fast: {'✅' if leak_ok else '❌'}")
    print(f"  ② 共享 TS2Vec 指纹一致 & fit 池不含目标: {'✅' if ts2vec_ok else '❌'}")
    print(f"  ③ T-CLSC checkpoint 未被改动（只训过源池 pairs）: {'✅' if tclsc_ok else '❌'}")
    print(f"  ④ 测试集触碰: {len(finals)} 次评估，scope={sorted(test_scopes)}（应仅 "
          f"official_test；标定/排名/sweep 均只用 train）")
    print(f"  ⑤ 本脚本不含 fit_shared_ts2vec / fit(...) 调用——TS2Vec 只 load "
          f"(TS2VecWrapper.load)，见 stage_prep")


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--subject", action="store_true",
                    help="出路 A 跨受试者协议：任务内与 sweep val 全按受试者整组切分，"
                         "缓存进 subject_protocol/ 子目录（判据/网格/候选池逐字不变）")
    for st in ("prep", "calibrate", "tclsc", "rank", "sweep", "final", "verify"):
        ap.add_argument(f"--{st}", action="store_true")
    args = ap.parse_args()
    if not any(getattr(args, s) for s in
               ("prep", "calibrate", "tclsc", "rank", "sweep", "final", "verify")):
        ap.error("至少指定一个阶段")

    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    tclsc_cfg = yaml.safe_load((ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))
    reg = load_registry()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.subject and (args.tclsc or args.rank):
        ap.error("--tclsc/--rank 与切分协议无关（产物两协议共享），不接受 --subject")

    if args.prep:
        stage_prep(args.outdir, reg, labels_cfg, args.device, args.resume,
                   subject=args.subject)
    if args.calibrate:
        stage_calibrate(args.outdir, reg, labels_cfg, train_cfg, args.device,
                        args.resume, subject=args.subject)
    if args.tclsc:
        stage_tclsc(args.outdir, labels_cfg, tclsc_cfg, args.device, args.resume)
    if args.rank:
        stage_rank(args.outdir, labels_cfg, tclsc_cfg, args.device)
    if args.sweep:
        stage_sweep(args.outdir, train_cfg, args.device, args.resume,
                    subject=args.subject)
    if args.final:
        stage_final(args.outdir, train_cfg, args.device, args.resume,
                    subject=args.subject)
    if args.verify:
        stage_verify(args.outdir, reg, subject=args.subject)


if __name__ == "__main__":
    main()
