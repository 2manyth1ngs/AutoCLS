"""全量标准协议对齐（results/protocol_alignment/）：让本栈在 UEA 官方 train/test 划分
（SVM-RBF 下游）上的绝对精度追平多数已发表模型。

背景（2026-09-03 审计，REPORT.md §0）：全量锚 Handwriting 0.34–0.39 vs TS2Vec 0.515 的差距
定位在**训练口径**（2320 iters@128 = 150 样本训 ≈1160 遍、bs 128 每遍 2 步且无 drop_last、
无 SWA、TS2Vec 式目标从未跑过、train/test 各自 z-score、SVM 网格偏窄），不是 encoder、
也不是零样本选择（UCI-HAR 上 GGS 0.935 ≈ AutoCL 0.937）。

协议真源：configs/train.yaml `protocols:`（legacy_full = 冻结的 25 号 cmd_anchor 口径；
ts2vec_official = 官方口径：iters 200 if N·T·C≤1e5 else 600、bs 8、drop_last、SWA、
scaler 只在 TRAIN 拟合、官方 SVM 网格）。所有新开关默认关，旧缓存逐位不变。

阶段（均接受 --datasets/--seeds/--resume/--device；结果统一写 runs.jsonl 按 key 缓存）：
    python scripts/35_protocol_alignment.py --dry-run [--stages ...]   # 工单数 + 估时（零 GPU）
    python scripts/35_protocol_alignment.py --eval-audit --resume       # A1 iters=0 基线 + A2 旧口径重跑
    python scripts/35_protocol_alignment.py --reproduce --resume        # HW × TS2VEC_EQUIV × 4 格 × 3 种子（P1）
    python scripts/35_protocol_alignment.py --bs-scan --resume          # HW × bs∈{8,32,128} 等遍 × 3 种子
    python scripts/35_protocol_alignment.py --official-ts2vec --resume  # vendored 官方 fit 循环（P1′）
    python scripts/35_protocol_alignment.py --ladder --resume           # 仅 P1 不过：逐成分阶梯
    python scripts/35_protocol_alignment.py --targets --resume          # 六靶 × 臂 × 3 种子（P2）
    python scripts/35_protocol_alignment.py --catalog --resume          # 可选：目录全量重扫（用户 go 后）
    python scripts/35_protocol_alignment.py --analyze                   # 零 GPU：analysis.json + tables.md

预注册判线（REPORT.md §0 先于任何 GPU 跑写死）：
    P1  TS2VEC_EQUIV @ ts2vec_official 三种子均值 ≥ 0.485 且每种子 ≥ 0.465
    P1′ 官方 wrapper 同线（P1′ 不过 = 数据/评估侧问题，先修再上阶梯）
    P2  头条 = zs_top3 三种子均值 ≥ TS2Vec − 0.03 的数据集数（次读数 ≥ 该行文献中位数）
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_m1 = importlib.import_module("scripts.m1_ggs_har")

# 噪声压制：sklearn 在 cv=5 × 26 类 × 30 样本/折时逐折报"类数 > 样本数 50%"（无碍，
# 官方网格本来就这么跑）；torch.optim.swa_utils 的 deprecation 提示与本实验无关。
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.metrics.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.model_selection.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:  # 官方网格的 C=inf 在 max_iter=1e7 上限处提前终止是预期行为（linear_eval 注释），不刷屏
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:  # pragma: no cover
    pass

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.uea import load_uea_pair  # noqa: E402
from autocls.downstream.linear_eval import encode_features, fit_svm  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import append_jsonl, cls_key, device_label  # noqa: E402
from autocls.oom_guard import RprimeOOM, is_oom_error  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, validate_task_type  # noqa: E402

OUTDIR_DEFAULT = ROOT / "results" / "protocol_alignment"
MX3_DIR = ROOT / "results" / "mx3_pq"
MX2_ANCHOR = ROOT / "results" / "mx2_endtoend" / "anchor_cache.jsonl"
TRAIN_YAML = ROOT / "configs" / "train.yaml"
TASK_TYPE = TASK_CLASSIFICATION          # 铁律 7：本脚本只做分类，硬断言
validate_task_type(TASK_TYPE)

TARGETS = ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra",
           "EthanolConcentration", "Cricket", "HandMovementDirection")
SEEDS = (0, 1, 2)
PICK_BUDGET = 290                        # zero-shot pick 读 rankings 的 @290 口径
PARITY_MARGIN = 0.03
P1_MEAN, P1_EACH = 0.485, 0.465
LADDER_ADOPT = 0.03
BS_SCAN = (8, 32, 128)
BS_SCAN_DATASET = "Handwriting"
CHAMPION_KEY = "f48e5da1d899"            # MX-2 dir_A 冠军（anchor_cache.jsonl）
OFFICIAL_KEY = "official_ts2vec"         # engine=ts2vec_wrapper 的伪 cls_key
RANDOM_INIT_KEY = "random_init"          # A1 iters=0 的伪 cls_key
OFFICIAL_OUTPUT_DIMS = 320               # 官方 train.py --repr-dims 320

# TS2Vec 论文 Table 10（UEA 全量协议 acc；results/mx3_pq/REPORT.md §8.3 转录）
TABLE10: dict[str, dict[str, float]] = {
    "Handwriting":          {"TS2Vec": 0.515, "T-Loss": 0.451, "TNC": 0.249, "TS-TCC": 0.498, "TST": 0.225, "DTW": 0.286},
    "UWaveGestureLibrary":  {"TS2Vec": 0.906, "T-Loss": 0.875, "TNC": 0.759, "TS-TCC": 0.753, "TST": 0.575, "DTW": 0.903},
    "PhonemeSpectra":       {"TS2Vec": 0.233, "T-Loss": 0.222, "TNC": 0.207, "TS-TCC": 0.252, "TST": 0.085, "DTW": 0.151},
    "EthanolConcentration": {"TS2Vec": 0.308, "T-Loss": 0.205, "TNC": 0.297, "TS-TCC": 0.285, "TST": 0.262, "DTW": 0.323},
    "Cricket":              {"TS2Vec": 0.972, "T-Loss": 0.972, "TNC": 0.958, "TS-TCC": 0.917, "TST": 1.000, "DTW": 1.000},
    "HandMovementDirection": {"TS2Vec": 0.338, "T-Loss": 0.351, "TNC": 0.324, "TS-TCC": 0.243, "TST": 0.243, "DTW": 0.231},
}

# TS2Vec 目标在本空间内的等价表示：只开 crop（p 只当开关，共享段由 crop 字段固定）、
# instance+temporal、kernel 2 max（层级由 kernel 驱动，pair_construction.py:194-221）、
# cross_scale 关、dot、τ=1、InfoNCE。已知残差（不加开关）：losses 取 mean 除以 2d−1 vs
# 官方 /d·α=0.5（有效 lr ×≈0.94）；hierarchical_pool ceil_mode vs 官方 floor。
TS2VEC_EQUIV = CLS(
    resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0,
    emb_jitter=0.0, emb_mask=0.0, norm="none",
    instance=True, temporal=True, cross_scale=False, kernel=2, pool="max", adj_neighbor=False,
    loss_type="info_nce", sim_func="dot", temperature=1.0,
)
# AutoCL Table 5 忠实版 GGS（kernel 5 / avg）：本实现里两开关全关时 kernel 会加多尺度
# instance 项，空间闸门判非法（铁律 1）——仅作文献对照白名单，space_legal=False，永不作候选。
GGS_T5 = replace(_m1.GGS, kernel=5, pool="avg", adj_neighbor=False)
OUTSIDE_SPACE_ARMS = frozenset({"ggs_table5"})

LADDER: tuple[tuple[str, dict], ...] = (
    ("ladder_crop_ts2vec", {"crop_mode": "ts2vec"}),
    ("ladder_input_mask", {"input_mask": "binomial"}),
    ("ladder_repr_dropout", {"repr_dropout": 0.1}),
    ("ladder_all", {"crop_mode": "ts2vec", "input_mask": "binomial", "repr_dropout": 0.1}),
)
OPTION_KEYS = ("batch_size", "lr", "drop_last", "swa", "crop_mode", "scaler", "svm_grid",
               "input_mask", "repr_dropout", "engine", "aug_learn")
# WP17-ACTS AL：aug_learn 只在 overrides 显式给出时出现在 options 里（run_options 不设默认），cache_key 只哈希
# options 中实际存在的 OPTION_KEYS → 所有旧格的键逐字节不变（tests/test_learned_aug.py 钉死）。
_SPACE = CLSSpace.from_yaml()            # 铁律 1：合法性只问 space.yaml
ENGINES = ("cl_trainer", "ts2vec_wrapper")
STAGES = ("eval_audit", "reproduce", "bs_scan", "official", "ladder", "targets", "catalog")


# ================================================================ 协议 / 纯函数
def load_protocols(train_yaml: str | Path = TRAIN_YAML) -> dict:
    """读 train.yaml `protocols:`；legacy_full 与同文件 full_iters/batch_size/lr 漂移即 fail-fast。"""
    cfg = yaml.safe_load(Path(train_yaml).read_text(encoding="utf-8"))
    protos = cfg.get("protocols")
    if not protos or "legacy_full" not in protos or "ts2vec_official" not in protos:
        raise ValueError(f"{train_yaml} 缺 protocols.{{legacy_full,ts2vec_official}} 块")
    lf = protos["legacy_full"]
    for pk, tk in (("iters", "full_iters"), ("batch_size", "batch_size"), ("lr", "lr")):
        if lf.get(pk) != cfg.get(tk):
            raise ValueError(
                f"protocols.legacy_full.{pk}={lf.get(pk)} ≠ {tk}={cfg.get(tk)}："
                "legacy_full 必须是冻结的 cmd_anchor 口径（铁律 2）")
    for name, p in protos.items():
        if ("iters" in p) == ("iters_rule" in p):
            raise ValueError(f"protocols.{name} 须恰有 iters 或 iters_rule 之一")
    return {k: dict(v) for k, v in protos.items()}


def ts2vec_default_iters(size: int) -> int:
    """官方 ts2vec.py:77：n_iters = 200 if train_data.size <= 100000 else 600。"""
    return 200 if int(size) <= 100_000 else 600


def resolve_iters_from_size(protocol: dict, size: int) -> int:
    if "iters" in protocol:
        return int(protocol["iters"])
    rule = protocol.get("iters_rule")
    if rule == "ts2vec_size":
        return ts2vec_default_iters(size)
    raise ValueError(f"未知 iters_rule={rule!r}")


def resolve_iters(protocol: dict, X) -> int:
    """X 只用 .size（N·T·C）；返回显式 int（铁律 2：仍由调用方显式传入 pretrain_encoder）。"""
    return resolve_iters_from_size(protocol, int(X.size))


def equal_epoch_iters(base_iters: int, base_bs: int, bs: int, n_train: int,
                      drop_last: bool) -> int:
    """等遍换算：iters_bs = base_iters × steps(bs)/steps(base_bs)，steps 按 drop_last 口径。"""
    def steps(b: int) -> int:
        return (n_train // b) if drop_last else math.ceil(n_train / b)
    if steps(bs) < 1 or steps(base_bs) < 1:
        raise ValueError(f"n_train={n_train} 下 bs={bs}/{base_bs} 无完整 batch")
    return max(1, int(round(base_iters * steps(bs) / steps(base_bs))))


def run_options(protocol: dict, overrides: Optional[dict] = None) -> dict:
    """协议 → 规范化选项字典（缓存键成分）。overrides 只允许 OPTION_KEYS。"""
    opts = {
        "batch_size": int(protocol["batch_size"]), "lr": float(protocol["lr"]),
        "drop_last": bool(protocol["drop_last"]), "swa": bool(protocol["swa"]),
        "crop_mode": str(protocol["crop_mode"]), "scaler": str(protocol["scaler"]),
        "svm_grid": str(protocol["svm_grid"]),
        "input_mask": "none", "repr_dropout": 0.0, "engine": "cl_trainer",
    }
    for k, v in (overrides or {}).items():
        if k not in OPTION_KEYS:
            raise ValueError(f"未知选项 {k!r}（可选 {OPTION_KEYS}）")
        opts[k] = v
    if opts["engine"] not in ENGINES:
        raise ValueError(f"未知 engine={opts['engine']!r}")
    return opts


def cache_key(dataset: str, protocol: str, cls_key_: str, seed: int, options: dict,
              iters: int, arm: str = "") -> str:
    """规范 JSON 的 sha1[:16]。arm 入键：bs_scan_8 与 repro_official 计算完全相同，若共用
    一行则先跑的阶段独占该行、后跑的阶段在 --analyze 里缺格；分开跑一次 = 免费的同机
    复现对照（GPU 非确定性量级可直接读出）。"""
    payload = json.dumps({"dataset": dataset, "protocol": protocol, "cls_key": cls_key_,
                          "seed": int(seed), "iters": int(iters), "arm": str(arm),
                          "options": {k: options[k] for k in OPTION_KEYS if k in options}},
                         sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_done(path: str | Path) -> dict[str, dict]:
    """runs.jsonl → {key: row}（半行跳过；error 行也算已跑——OOM 从不落盘）。"""
    path = Path(path)
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["key"]] = r
    return done


def pending(cells: list[dict], done: dict) -> list[dict]:
    return [c for c in cells if c["key"] not in done]


def _borda(rank_lists: list[list[str]], keys: list[str]) -> dict[str, int]:
    n = len(keys)
    score = {k: 0 for k in keys}
    for ranked in rank_lists:
        pos = {k: i for i, k in enumerate([k for k in ranked if k in score])}
        for k in keys:
            if k in pos:
                score[k] += n - 1 - pos[k]
    return score


def zero_shot_picks(rankings: dict, catalog: dict, dataset: str, budget: int = PICK_BUDGET,
                    arm: str = "main") -> dict:
    """以 dataset 为折的 rankings[dataset][budget] 各切片排名 → 10 个 shared 键上的 Borda。

    铁律 6：只读目标数据集自己的折（该折的比较器/TS2Vec/δ 都不含它）。
    返回 top1 / top3 / per_slice_top1（各切片总榜第一，可为 random 键）/ borda /
    cls_only_top1（A5 臂同法，缺席为 None）。
    """
    per_b = rankings[dataset][str(budget)]
    shared_keys = [d["cls_key"] for d in catalog["shared"]]
    slices = sorted(per_b)
    main_lists = [per_b[s][arm] for s in slices]
    score = _borda(main_lists, shared_keys)
    order = sorted(shared_keys, key=lambda k: (-score[k], k))
    cls_only_top1 = None
    if all("cls_only" in per_b[s] for s in slices) and slices:
        sc = _borda([per_b[s]["cls_only"] for s in slices], shared_keys)
        cls_only_top1 = sorted(shared_keys, key=lambda k: (-sc[k], k))[0]
    return {"dataset": dataset, "budget": int(budget), "arm": arm, "n_slices": len(slices),
            "top1": order[0], "top3": order[:3], "borda": score,
            "per_slice_top1": [main_lists[i][0] for i in range(len(slices))],
            "cls_only_top1": cls_only_top1}


def parity_thresholds(row: dict) -> dict:
    vals = sorted(float(v) for v in row.values())
    return {"ts2vec": float(row["TS2Vec"]), "line": float(row["TS2Vec"]) - PARITY_MARGIN,
            "median": float(np.median(vals))}


def verdict(mean_acc: float, row: dict) -> str:
    if mean_acc is None or not np.isfinite(mean_acc):
        return "missing"
    th = parity_thresholds(row)
    if mean_acc >= th["line"]:
        return "parity"
    if mean_acc >= th["median"]:
        return "median"
    return "below"


def p1_pass(accs: Sequence[float]) -> bool:
    accs = [float(a) for a in accs if a is not None and np.isfinite(a)]
    return len(accs) == len(SEEDS) and float(np.mean(accs)) >= P1_MEAN and min(accs) >= P1_EACH


def summarize_seeds(rows: list[dict]) -> dict:
    accs = [float(r["acc"]) for r in rows if r.get("acc") is not None]
    f1s = [float(r["f1"]) for r in rows if r.get("f1") is not None]
    cvs = [r.get("svm_cv_score") for r in rows]
    cvs = [float(c) for c in cvs if c is not None]
    return {"n": len(accs), "mean": float(np.mean(accs)) if accs else float("nan"),
            "std": float(np.std(accs)) if accs else float("nan"), "accs": accs,
            "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "cv_mean": float(np.mean(cvs)) if len(cvs) == len(accs) and cvs else None}


def top3_selection(keys: list[str], per_cls: dict) -> tuple[str, Optional[str]]:
    """best-of-3（测试集均值最大，MX-2 tclsc3 同口径上界）与 cv 选择（SVM 交叉验证分最大，无泄漏）。"""
    best = max(keys, key=lambda k: (per_cls[k]["mean"], k))
    if all(per_cls[k].get("cv_mean") is not None for k in keys):
        cv_sel = max(keys, key=lambda k: (per_cls[k]["cv_mean"], k))
    else:
        cv_sel = None
    return best, cv_sel


# ================================================================ 工单
def _size_of(sizes: dict, dataset: str) -> int:
    n, t, c = sizes[dataset]
    return int(n) * int(t) * int(c)


def _cell(stage: str, dataset: str, protocol_name: str, protocol: dict, sizes: dict, *,
          arm_names: list[str], cls_key_: str, cls: Optional[CLS], seed: int,
          overrides: Optional[dict] = None, iters: Optional[int] = None,
          space_legal: bool = True) -> dict:
    opts = run_options(protocol, overrides)
    it = int(iters) if iters is not None else resolve_iters_from_size(protocol, _size_of(sizes, dataset))
    return {"key": cache_key(dataset, protocol_name, cls_key_, seed, opts, it, arm=arm_names[0]),
            "stage": stage, "dataset": dataset, "protocol": protocol_name,
            "arm": arm_names[0], "arm_names": list(arm_names), "cls_key": cls_key_,
            "cls": cls.to_dict() if cls is not None else None, "space_legal": bool(space_legal),
            "seed": int(seed), "iters": it, "options": opts}


def audit_baseline_worklist(protos: dict, sizes: dict, datasets=TARGETS, seeds=SEEDS) -> list[dict]:
    return [_cell("eval_audit", d, "ts2vec_official", protos["ts2vec_official"], sizes,
                  arm_names=["audit_iters0"], cls_key_=RANDOM_INIT_KEY, cls=None, seed=s, iters=0)
            for d in datasets for s in seeds]


def audit_legacy_worklist(protos: dict, sizes: dict, *, champion_key: str, champion_cls: CLS,
                          dataset: str = "Handwriting") -> list[dict]:
    p = protos["legacy_full"]
    return [_cell("eval_audit", dataset, "legacy_full", p, sizes, arm_names=["audit_champion"],
                  cls_key_=champion_key, cls=champion_cls, seed=0),
            _cell("eval_audit", dataset, "legacy_full", p, sizes, arm_names=["audit_ggs"],
                  cls_key_=cls_key(_m1.GGS), cls=_m1.GGS, seed=0)]


def reproduce_worklist(protos: dict, sizes: dict, datasets=("Handwriting",), seeds=SEEDS) -> list[dict]:
    grids = (("repro_official", "ts2vec_official", {}),
             ("repro_official_noswa", "ts2vec_official", {"swa": False}),
             ("repro_legacy", "legacy_full", {}),
             ("repro_legacy_swa", "legacy_full", {"swa": True}))
    k = cls_key(TS2VEC_EQUIV)
    return [_cell("reproduce", d, pn, protos[pn], sizes, arm_names=[arm], cls_key_=k,
                  cls=TS2VEC_EQUIV, seed=s, overrides=ov)
            for d in datasets for arm, pn, ov in grids for s in seeds]


def bs_scan_worklist(protos: dict, sizes: dict, seeds=SEEDS, dataset: str = BS_SCAN_DATASET) -> list[dict]:
    p = protos["ts2vec_official"]
    n_train = int(sizes[dataset][0])
    base_iters = resolve_iters_from_size(p, _size_of(sizes, dataset))
    k = cls_key(TS2VEC_EQUIV)
    out = []
    for bs in BS_SCAN:
        it = equal_epoch_iters(base_iters, int(p["batch_size"]), bs, n_train, bool(p["drop_last"]))
        for s in seeds:
            out.append(_cell("bs_scan", dataset, "ts2vec_official", p, sizes,
                             arm_names=[f"bs_scan_{bs}"], cls_key_=k, cls=TS2VEC_EQUIV, seed=s,
                             overrides={"batch_size": bs}, iters=it))
    return out


def ladder_worklist(protos: dict, sizes: dict, seeds=SEEDS, dataset: str = "Handwriting") -> list[dict]:
    p = protos["ts2vec_official"]
    k = cls_key(TS2VEC_EQUIV)
    return [_cell("ladder", dataset, "ts2vec_official", p, sizes, arm_names=[arm], cls_key_=k,
                  cls=TS2VEC_EQUIV, seed=s, overrides=ov)
            for arm, ov in LADDER for s in seeds]


def parse_ladder_overrides(spec: Optional[str]) -> Optional[dict]:
    """--ladder-overrides 取值：阶梯 rung 名（ladder_input_mask 等）或 JSON 字典；空/None → None。"""
    if not spec:
        return None
    for name, ov in LADDER:
        if spec == name:
            return dict(ov)
    ov = json.loads(spec)
    if not isinstance(ov, dict) or not ov:
        raise ValueError(f"--ladder-overrides 须为 rung 名或非空 JSON 字典，得到 {spec!r}")
    run_options({"batch_size": 8, "lr": 1e-3, "drop_last": True, "swa": True, "crop_mode": "fixed",
                 "scaler": "train", "svm_grid": "ts2vec"}, ov)   # 只校验键合法
    return ov


def adopted_ladder_rung(ladder_summary: dict) -> Optional[str]:
    """REPORT §0.7：+ladder 第二遍用**实测过的单一配置** = 采纳（Δ≥+0.03）的 rung 里 Δ 最大者。"""
    cands = [(v["delta"], k) for k, v in ladder_summary.items() if v.get("adopt")]
    if not cands:
        return None
    return max(cands)[1]


def official_worklist(protos: dict, sizes: dict, datasets=TARGETS, seeds=SEEDS) -> list[dict]:
    p = protos["ts2vec_official"]
    return [_cell("official", d, "ts2vec_official", p, sizes, arm_names=["official_ts2vec"],
                  cls_key_=OFFICIAL_KEY, cls=None, seed=s, overrides={"engine": "ts2vec_wrapper"})
            for d in datasets for s in seeds]


def target_arms(picks: dict, key2cls: dict) -> list[dict]:
    """臂 → 唯一 CLS 列表（按 CLS 内容的规范键去重；同一 CLS 的多个臂名合并）。"""
    ordered: list[tuple[str, CLS]] = [("ggs", _m1.GGS), ("ggs_table5", GGS_T5),
                                      ("ts2vec_equiv", TS2VEC_EQUIV),
                                      ("zs_top1", key2cls[picks["top1"]])]
    ordered += [("zs_top3", key2cls[k]) for k in picks["top3"]]
    if picks.get("cls_only_top1") is not None:
        ordered.append(("zs_clsonly_top1", key2cls[picks["cls_only_top1"]]))
    merged: dict[str, dict] = {}
    for arm, c in ordered:
        k = cls_key(c)
        if k not in merged:
            # space_legal 是 CLS 自身的属性（铁律 1：由 space.yaml 判），不随臂名变
            merged[k] = {"cls_key": k, "cls": c, "arms": [], "space_legal": bool(_SPACE.validate(c))}
        if arm not in merged[k]["arms"]:
            merged[k]["arms"].append(arm)
        if arm in OUTSIDE_SPACE_ARMS and merged[k]["space_legal"]:
            raise ValueError(f"白名单臂 {arm} 的 CLS 竟在空间内——白名单定义漂移")
    return list(merged.values())


def targets_worklist(protos: dict, sizes: dict, picks_by_ds: dict, key2cls: dict,
                     datasets=TARGETS, seeds=SEEDS, overrides: Optional[dict] = None,
                     suffix: str = "") -> list[dict]:
    """六靶工单。overrides/suffix（REPORT §0.7 补充）：阶梯采纳成分后的第二遍
    （臂名加 "+ladder" 后缀、stage="targets_ladder"），主口径（无 overrides）不变。"""
    p = protos["ts2vec_official"]
    out = []
    stage = "targets_ladder" if overrides else "targets"
    for d in datasets:
        for a in target_arms(picks_by_ds[d], key2cls):
            for s in seeds:
                out.append(_cell(stage, d, "ts2vec_official", p, sizes,
                                 arm_names=[x + suffix for x in a["arms"]],
                                 cls_key_=a["cls_key"], cls=a["cls"], seed=s,
                                 overrides=overrides, space_legal=a["space_legal"]))
    return out


def catalog_worklist(protos: dict, sizes: dict, catalog: dict, datasets=TARGETS,
                     seed: int = 0) -> list[dict]:
    p = protos["ts2vec_official"]
    out = []
    for d in datasets:
        ents = list(catalog["shared"]) + list(catalog["random"].get(f"{d}_00", []))
        for e in ents:
            c = CLS(**e["cls"])
            out.append(_cell("catalog", d, "ts2vec_official", p, sizes,
                             arm_names=["catalog"], cls_key_=e["cls_key"], cls=c, seed=seed))
    return out


# ================================================================ 装配
def load_mx3_picks() -> tuple[dict, dict, dict]:
    """rankings.json + cls_catalog.json → (picks_by_ds, key2cls, catalog)。"""
    rankings = json.loads((MX3_DIR / "rankings.json").read_text(encoding="utf-8"))
    catalog = json.loads((MX3_DIR / "cls_catalog.json").read_text(encoding="utf-8"))
    key2cls = {d["cls_key"]: CLS(**d["cls"]) for d in catalog["shared"]}
    for lst in catalog["random"].values():
        key2cls.update({d["cls_key"]: CLS(**d["cls"]) for d in lst})
    key2cls[catalog["ggs"]["cls_key"]] = CLS(**catalog["ggs"]["cls"])
    picks = {d: zero_shot_picks(rankings, catalog, d, budget=PICK_BUDGET) for d in TARGETS
             if d in rankings}
    return picks, key2cls, catalog


_ARRAYS: dict[str, dict] = {}


def load_arrays(dataset: str) -> dict:
    """(Xtr, ytr, Xte_split, Xte_train)：Xtr 在两种 scaler 下逐位相同（TRAIN 统计量 = 自身），
    只有 TEST 的标准化随 scaler 变；等长数据集目标长度也相同。"""
    if dataset not in _ARRAYS:
        Xtr_t, ytr_t, Xte_t, yte_t = load_uea_pair(dataset, scaler="train")
        Xtr_s, ytr_s, Xte_s, yte_s = load_uea_pair(dataset, scaler="split")
        assert Xtr_t.tobytes() == Xtr_s.tobytes(), f"{dataset}: TRAIN 在两 scaler 下应逐位相同"
        assert np.array_equal(ytr_t, ytr_s) and np.array_equal(yte_t, yte_s), \
            f"{dataset}: 标签编码跨 scaler 不一致（TEST 缺类？）"
        _ARRAYS[dataset] = {"Xtr": Xtr_t, "ytr": ytr_t, "Xte_split": Xte_s, "Xte_train": Xte_t,
                            "yte": yte_t}
        print(f"[data] {dataset}: train={Xtr_t.shape} test={Xte_t.shape} "
              f"classes={len(np.unique(ytr_t))} size={Xtr_t.size}", flush=True)
    return _ARRAYS[dataset]


def sizes_from_arrays(datasets=TARGETS) -> dict:
    return {d: tuple(int(v) for v in load_arrays(d)["Xtr"].shape) for d in datasets}


SIZES_SURVEY = {  # results/uea_survey/uea_survey.csv（n_train, L, C）——dry-run 免加载
    "Handwriting": (150, 152, 3), "UWaveGestureLibrary": (120, 315, 3),
    "PhonemeSpectra": (3315, 217, 11), "EthanolConcentration": (261, 1751, 3),
    "Cricket": (108, 1197, 6), "HandMovementDirection": (160, 400, 10),
}


# ================================================================ GPU 入口
def _svm_variants(feats_tr, ytr, feats_te: dict, yte, grids: Sequence[str], seed: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    out = {}
    for g in grids:
        clf = fit_svm(feats_tr, ytr, seed=seed, grid=g)
        cv = getattr(clf, "cv_score_", None)
        for sc_name, fte in feats_te.items():
            pred = clf.predict(fte)
            out[f"{sc_name}/{g}"] = {
                "acc": float(accuracy_score(yte, pred)),
                "f1": float(f1_score(yte, pred, average="macro")),
                "svm_c": float(clf.C), "svm_cv_score": (float(cv) if cv is not None else None)}
    return out


def run_cell(cell: dict, arrays: dict, device: str, audit: str = "all") -> dict:
    """唯一 GPU 入口：预训练（或 iters=0 只编码）→ 三组特征 → SVM 变体 → 行。OOM 抛 RprimeOOM。"""
    opts, seed, iters = cell["options"], int(cell["seed"]), int(cell["iters"])
    Xtr, ytr, yte = arrays["Xtr"], arrays["ytr"], arrays["yte"]
    C = Xtr.shape[-1]
    t0 = time.perf_counter()
    try:
        if opts["engine"] == "ts2vec_wrapper":
            from autocls.task_embed.ts2vec_wrapper import TS2VecWrapper
            w = TS2VecWrapper(input_dims=C, output_dims=OFFICIAL_OUTPUT_DIMS, hidden_dims=64,
                              depth=10, device=device, lr=opts["lr"],
                              batch_size=opts["batch_size"], seed=seed)
            losses = w.fit(Xtr, n_iters=iters) if iters > 0 else []
            enc_fn = lambda X: w.encode(X, encoding_window="full_series")  # noqa: E731
        else:
            cls = CLS(**cell["cls"]) if cell["cls"] is not None else None
            torch.manual_seed(seed)
            ov = {}
            if opts["input_mask"] != "none":
                ov["input_mask"] = opts["input_mask"]
            if float(opts["repr_dropout"]) > 0:
                ov["repr_dropout"] = float(opts["repr_dropout"])
            enc = build_encoder(input_dims=C, overrides=ov or None)
            if iters > 0:
                assert cls is not None, "iters>0 需要 CLS"
                losses = pretrain_encoder(
                    enc, cls, Xtr, max_iters=iters, batch_size=opts["batch_size"],
                    lr=opts["lr"], device=device, seed=seed, use_swa=opts["swa"],
                    drop_last=opts["drop_last"], crop_mode=opts["crop_mode"])
            else:
                losses = []
                enc = enc.to(device)
            enc_fn = lambda X: encode_features(enc, X, device=device)  # noqa: E731
        t1 = time.perf_counter()
        feats_tr = enc_fn(Xtr)
        feats_te = {"split": enc_fn(arrays["Xte_split"]), "train": enc_fn(arrays["Xte_train"])}
        grids = ("legacy", "ts2vec") if audit == "all" else (opts["svm_grid"],)
        variants = _svm_variants(feats_tr, ytr, feats_te, yte, grids, seed)
        t2 = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        return {**cell, "acc": None, "f1": None, "svm_c": None, "svm_cv_score": None,
                "eval_variants": None, "pretrain_seconds": round(time.perf_counter() - t0, 1),
                "eval_seconds": None, "device_name": device_label(device),
                "first_pass_loss": None, "last_pass_loss": None,
                "error": f"{type(e).__name__}: {e}"}
    own = variants[f"{opts['scaler']}/{opts['svm_grid']}"]
    return {**cell, "acc": own["acc"], "f1": own["f1"], "svm_c": own["svm_c"],
            "svm_cv_score": own["svm_cv_score"], "eval_variants": variants,
            "pretrain_seconds": round(t1 - t0, 1), "eval_seconds": round(t2 - t1, 1),
            "device_name": device_label(device),
            "first_pass_loss": (float(losses[0]) if losses else None),
            "last_pass_loss": (float(losses[-1]) if losses else None),
            "n_train": int(Xtr.shape[0]), "n_test": int(len(yte)), "error": None}


def execute(cells: list[dict], args, tag: str) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs, inflight, oom_log = outdir / "runs.jsonl", outdir / "inflight.txt", outdir / "oom_failures.jsonl"
    done = load_done(runs) if args.resume else {}
    todo = pending(cells, done)
    print(f"[{tag}] 计划 {len(cells)} 格，已有 {len(cells) - len(todo)}，待跑 {len(todo)}", flush=True)
    # joblib 后端：本机 Windows 上 loky spawn 占 R' 单条 99% 耗时（memory），故默认 threading；
    # Linux pod 上 GridSearchCV(n_jobs=-1) 用 loky 才有真并行（PhonemeSpectra cv=5×10C 是耗时大头）。
    backend_kw = {"n_jobs": 1} if args.joblib_backend == "threading" else {}
    with parallel_backend(args.joblib_backend, **backend_kw):
        for i, cell in enumerate(todo):
            arrays = load_arrays(cell["dataset"])
            who = (f"{cell['dataset']}/{cell['arm']}/{cell['cls_key'][:8]} s{cell['seed']} "
                   f"iters={cell['iters']} bs={cell['options']['batch_size']}")
            inflight.write_text(f"{cell['key']}\t{who}", encoding="utf-8")
            try:
                row = run_cell(cell, arrays, args.device, audit=args.audit)
            except RprimeOOM as e:
                inflight.write_text("", encoding="utf-8")
                append_jsonl(oom_log, {**cell, "error": str(e), "device_name": device_label(args.device)})
                print(f"[{tag} {i + 1}/{len(todo)}] ⚠️ OOM {who} → 不写缓存：{e}", flush=True)
                continue
            append_jsonl(runs, row)
            inflight.write_text("", encoding="utf-8")
            acc = row["acc"]
            print(f"[{tag} {i + 1}/{len(todo)}] {who} acc={'NaN' if acc is None else f'{acc:.4f}'} "
                  f"({row['pretrain_seconds']}s+{row['eval_seconds']}s)"
                  + (f" ERROR {row['error']}" if row["error"] else ""), flush=True)
    print(f"[{tag}] 收官", flush=True)


# ================================================================ 阶段
def _datasets(args) -> tuple[str, ...]:
    return tuple(args.datasets) if args.datasets else TARGETS


def _seeds(args) -> tuple[int, ...]:
    return tuple(int(s) for s in args.seeds) if args.seeds else SEEDS


def _champion() -> tuple[str, CLS]:
    _picks, key2cls, _cat = load_mx3_picks()
    return CHAMPION_KEY, key2cls[CHAMPION_KEY]


def build_stage_cells(stage: str, protos: dict, sizes: dict, args) -> list[dict]:
    ds, seeds = _datasets(args), _seeds(args)
    if stage == "eval_audit":
        ck, cc = _champion()
        cells = audit_baseline_worklist(protos, sizes, datasets=ds, seeds=seeds)
        if "Handwriting" in ds:
            cells += audit_legacy_worklist(protos, sizes, champion_key=ck, champion_cls=cc)
        return cells
    if stage == "reproduce":
        return reproduce_worklist(protos, sizes, datasets=tuple(d for d in ds if d == "Handwriting")
                                  or ("Handwriting",), seeds=seeds)
    if stage == "bs_scan":
        return bs_scan_worklist(protos, sizes, seeds=seeds)
    if stage == "official":
        return official_worklist(protos, sizes, datasets=ds, seeds=seeds)
    if stage == "ladder":
        return ladder_worklist(protos, sizes, seeds=seeds)
    if stage == "targets":
        picks, key2cls, _cat = load_mx3_picks()
        (Path(args.outdir)).mkdir(parents=True, exist_ok=True)
        (Path(args.outdir) / "picks.json").write_text(
            json.dumps(picks, indent=2, ensure_ascii=False), encoding="utf-8")
        ov = parse_ladder_overrides(getattr(args, "ladder_overrides", None))
        return targets_worklist(protos, sizes, picks, key2cls, datasets=ds, seeds=seeds,
                                overrides=ov, suffix=("+ladder" if ov else ""))
    if stage == "catalog":
        _picks, _k2c, cat = load_mx3_picks()
        return catalog_worklist(protos, sizes, cat, datasets=ds, seed=seeds[0])
    raise ValueError(stage)


def _est_seconds(cell: dict, anchors: dict) -> float:
    """估时：优先用 runs.jsonl 里同 (dataset, engine, bs≈) 的实测秒/iter；否则按 bs·T 粗估。"""
    n, t, c = SIZES_SURVEY.get(cell["dataset"], (150, 152, 3))
    bs = cell["options"]["batch_size"]
    key = (cell["dataset"], cell["options"]["engine"])
    per_iter = anchors.get(key)
    if per_iter is None:
        per_iter = 9e-7 * bs * t * (1.0 if cell["options"]["engine"] == "cl_trainer" else 1.2)
        per_iter = max(per_iter, 0.004)
    svm = 1.0 if n < 1000 else 60.0
    return cell["iters"] * per_iter + 3 * svm


def cmd_dry_run(args, protos: dict) -> None:
    sizes = SIZES_SURVEY
    done = load_done(Path(args.outdir) / "runs.jsonl")
    anchors: dict = {}
    acc: dict = defaultdict(list)
    for r in done.values():
        if r.get("pretrain_seconds") and r.get("iters"):
            acc[(r["dataset"], r["options"]["engine"])].append(r["pretrain_seconds"] / r["iters"])
    anchors = {k: float(np.median(v)) for k, v in acc.items()}
    stages = args.stages or list(STAGES)
    total = 0.0
    for st in stages:
        cells = build_stage_cells(st, protos, sizes, args)
        todo = pending(cells, done)
        est = sum(_est_seconds(c, anchors) for c in todo)
        total += est
        print(f"{st:11s} 计划 {len(cells):4d}  待跑 {len(todo):4d}  估时 {est / 60:7.1f} min", flush=True)
    print(f"合计估时 {total / 3600:.2f} h（估时锚：{'实测' if anchors else '粗估'}）", flush=True)


# ================================================================ --analyze
def _rows_by(rows: list[dict], stage: str) -> list[dict]:
    return [r for r in rows if r["stage"] == stage and r.get("error") is None]


def _group(rows: list[dict], keyf) -> dict:
    g: dict = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    return g


def _variant_means(rows: list[dict]) -> dict:
    out: dict = {}
    if not rows or not rows[0].get("eval_variants"):
        return out
    for v in rows[0]["eval_variants"]:
        accs = [r["eval_variants"][v]["acc"] for r in rows if r.get("eval_variants")]
        out[v] = {"mean": float(np.mean(accs)), "n": len(accs), "accs": accs}
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def cmd_analyze(args) -> None:
    outdir = Path(args.outdir)
    rows_all = list(load_done(outdir / "runs.jsonl").values())
    rows = [r for r in rows_all if r.get("error") is None]
    errors = [r for r in rows_all if r.get("error")]
    out: dict = {"n_rows": len(rows_all), "n_errors": len(errors),
                 "errors": [{"key": r["key"], "dataset": r["dataset"], "arm": r["arm"],
                             "seed": r["seed"], "error": r["error"]} for r in errors]}
    md: list[str] = ["# protocol_alignment tables（--analyze 自动生成）", ""]

    # §1 评估审计
    a1 = _group([r for r in _rows_by(rows, "eval_audit") if r["iters"] == 0], lambda r: r["dataset"])
    audit = {}
    md += ["## §1 评估审计：iters=0 随机 encoder 基线（各数据集 3 种子均值；2×2 = scaler × svm_grid）", "",
           "| 数据集 | split/legacy | train/legacy | split/ts2vec | train/ts2vec | Δscaler(ts2vec) | Δgrid(train) |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for d in TARGETS:
        if d not in a1:
            continue
        vm = _variant_means(a1[d])
        audit[d] = vm
        g = lambda k: vm.get(k, {}).get("mean", float("nan"))  # noqa: E731
        md.append(f"| {d} | {g('split/legacy'):.3f} | {g('train/legacy'):.3f} | {g('split/ts2vec'):.3f} | "
                  f"{g('train/ts2vec'):.3f} | {g('train/ts2vec') - g('split/ts2vec'):+.3f} | "
                  f"{g('train/ts2vec') - g('train/legacy'):+.3f} |")
    a2 = [r for r in _rows_by(rows, "eval_audit") if r["iters"] > 0]
    a2_out = {}
    if a2:
        md += ["", "### A2 旧口径重跑（legacy_full，seed 0）vs MX-2 anchor_cache（5090）", "",
               "| 臂 | 本机 acc | 本机 device | 5090 seed0 | 5090 3 种子均值 | Δ(本机−5090 s0) |", "|---|---:|---|---:|---:|---:|"]
        anchor = defaultdict(list)
        if MX2_ANCHOR.exists():
            for line in MX2_ANCHOR.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rr = json.loads(line)
                    anchor[rr["arm"]].append(rr)
        for r in a2:
            arm = "champion" if r["arm"] == "audit_champion" else "ggs"
            s0 = next((x["acc"] for x in anchor.get(arm, []) if x["seed"] == 0), None)
            m3 = float(np.mean([x["acc"] for x in anchor[arm]])) if anchor.get(arm) else None
            a2_out[arm] = {"acc": r["acc"], "device": r["device_name"], "anchor_seed0": s0,
                           "anchor_mean3": m3, "delta_seed0": (r["acc"] - s0) if s0 is not None else None,
                           "within_0.02": (abs(r["acc"] - s0) <= 0.02) if s0 is not None else None,
                           "pretrain_seconds": r["pretrain_seconds"]}
            d_s0 = "—" if s0 is None else f"{r['acc'] - s0:+.4f}"
            md.append(f"| {arm} | {r['acc']:.4f} | {r['device_name']} | "
                      f"{'—' if s0 is None else f'{s0:.4f}'} | {'—' if m3 is None else f'{m3:.4f}'} | "
                      f"{d_s0} |")
    out["audit"] = {"iters0": audit, "legacy_rerun": a2_out}

    # §2 复现阶梯（HW × TS2VEC_EQUIV × 4 格）+ bs 扫描
    rep = _group(_rows_by(rows, "reproduce"), lambda r: r["arm"])
    rep_out = {arm: summarize_seeds(rs) for arm, rs in rep.items()}
    md += ["", "## §2 复现：Handwriting × TS2VEC_EQUIV（3 种子 mean±std [各种子]）", "",
           "| 格 | 协议 | swa | acc | f1 | 各种子 |", "|---|---|---|---:|---:|---|"]
    for arm in ("repro_official", "repro_official_noswa", "repro_legacy", "repro_legacy_swa"):
        if arm in rep_out:
            s = rep_out[arm]
            r0 = rep[arm][0]
            md.append(f"| {arm} | {r0['protocol']} | {r0['options']['swa']} | {s['mean']:.4f}±{s['std']:.4f} | "
                      f"{s['f1_mean']:.4f} | {', '.join(f'{a:.3f}' for a in s['accs'])} |")
    p1 = p1_pass(rep_out["repro_official"]["accs"]) if "repro_official" in rep_out else None
    attribution = {}
    if {"repro_official", "repro_official_noswa", "repro_legacy", "repro_legacy_swa"} <= set(rep_out):
        m = {k: v["mean"] for k, v in rep_out.items()}
        attribution = {"swa_effect_at_official": m["repro_official"] - m["repro_official_noswa"],
                       "swa_effect_at_legacy": m["repro_legacy_swa"] - m["repro_legacy"],
                       "regime_effect_no_swa": m["repro_official_noswa"] - m["repro_legacy"],
                       "regime_effect_with_swa": m["repro_official"] - m["repro_legacy_swa"],
                       "total": m["repro_official"] - m["repro_legacy"]}
        md += ["", f"归因：SWA@official {attribution['swa_effect_at_official']:+.4f}，SWA@legacy "
                   f"{attribution['swa_effect_at_legacy']:+.4f}，训练口径（无 SWA）"
                   f"{attribution['regime_effect_no_swa']:+.4f}，合计 {attribution['total']:+.4f}"]
    md += ["", f"**P1**（official 三种子均值 ≥ {P1_MEAN} 且每种子 ≥ {P1_EACH}）："
               f"{'过' if p1 else ('不过' if p1 is not None else '未跑')}"]
    out["reproduce"] = {"arms": rep_out, "P1": p1, "attribution": attribution}

    bs = _group(_rows_by(rows, "bs_scan"), lambda r: r["options"]["batch_size"])
    bs_out = {}
    if bs:
        md += ["", "### bs 扫描：Handwriting × TS2VEC_EQUIV × 等遍（≈11 遍）× 3 种子", "",
               "| bs | iters | acc | 各种子 |", "|---:|---:|---:|---|"]
        for b in sorted(bs):
            s = summarize_seeds(bs[b])
            bs_out[str(b)] = {**s, "iters": bs[b][0]["iters"]}
            md.append(f"| {b} | {bs[b][0]['iters']} | {s['mean']:.4f}±{s['std']:.4f} | "
                      f"{', '.join(f'{a:.3f}' for a in s['accs'])} |")
    out["bs_scan"] = bs_out

    # §3 官方地面真值
    off = _group(_rows_by(rows, "official"), lambda r: r["dataset"])
    off_out = {d: summarize_seeds(rs) for d, rs in off.items()}
    if off_out:
        md += ["", "## §3 官方代码（vendored TS2Vec fit 循环）于本机", "",
               "| 数据集 | 本机 official | 论文 TS2Vec | Δ | 各种子 |", "|---|---:|---:|---:|---|"]
        for d in TARGETS:
            if d in off_out:
                s = off_out[d]
                md.append(f"| {d} | {s['mean']:.4f}±{s['std']:.4f} | {TABLE10[d]['TS2Vec']:.3f} | "
                          f"{s['mean'] - TABLE10[d]['TS2Vec']:+.4f} | {', '.join(f'{a:.3f}' for a in s['accs'])} |")
    p1p = p1_pass(off_out["Handwriting"]["accs"]) if "Handwriting" in off_out else None
    md += ["", f"**P1′**（官方 wrapper 于 Handwriting 同线）：{'过' if p1p else ('不过' if p1p is not None else '未跑')}"]
    out["official"] = {"per_dataset": off_out, "P1_prime": p1p}

    # §2b 阶梯
    lad = _group(_rows_by(rows, "ladder"), lambda r: r["arm"])
    lad_out = {}
    if lad:
        base = rep_out.get("repro_official", {}).get("mean")
        md += ["", "## §2b 阶梯（在 ts2vec_official 上逐成分；采纳线 ≥ +0.03）", "",
               "| 成分 | acc | Δ vs official | 采纳 |", "|---|---:|---:|---|"]
        for arm, _ov in LADDER:
            if arm in lad:
                s = summarize_seeds(lad[arm])
                delta = (s["mean"] - base) if base is not None else None
                adopt = bool(delta is not None and delta >= LADDER_ADOPT)
                lad_out[arm] = {**s, "delta": delta, "adopt": adopt}
                md.append(f"| {arm} | {s['mean']:.4f}±{s['std']:.4f} | "
                          f"{'—' if delta is None else f'{delta:+.4f}'} | {'✓' if adopt else '✗'} |")
    out["ladder"] = lad_out

    # §4 六靶表
    tg = _rows_by(rows, "targets")
    picks_path = outdir / "picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path.exists() else {}
    targets_out = {}
    if tg:
        md += ["", "## §4 六靶（ts2vec_official 口径，3 种子均值；列 = 论文 / 官方代码本机 / 本栈各臂）", "",
               "| 数据集 | TS2Vec 论文 | 官方本机 | TS2VEC_EQUIV | GGS | GGS_T5* | zs_top1 | zs_top3(best) | zs_top3(cv) | 判定(top3) |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        by_ds = _group(tg, lambda r: r["dataset"])
        for d in TARGETS:
            if d not in by_ds:
                continue
            per_cls = {k: summarize_seeds(rs) for k, rs in _group(by_ds[d], lambda r: r["cls_key"]).items()}
            arm_of = {}
            for r in by_ds[d]:
                for a in r["arm_names"]:
                    arm_of[a] = r["cls_key"]
            def m(a):  # noqa: E306
                k = arm_of.get(a)
                return per_cls[k]["mean"] if k else float("nan")
            # picks 里的 key 与目录 key 同源（cls_key(CLS)），直接用
            top3_keys = [k for k in picks.get(d, {}).get("top3", []) if k in per_cls]
            best, cv_sel = (top3_selection(top3_keys, per_cls) if top3_keys else (None, None))
            row = TABLE10[d]
            zs3 = per_cls[best]["mean"] if best else float("nan")
            zs3cv = per_cls[cv_sel]["mean"] if cv_sel else float("nan")
            v = verdict(zs3, row)
            targets_out[d] = {"per_cls": per_cls, "arm_of": arm_of, "top3_best": best, "top3_cv": cv_sel,
                              "zs_top1": m("zs_top1"), "zs_top3_best": zs3, "zs_top3_cv": zs3cv,
                              "ts2vec_equiv": m("ts2vec_equiv"), "ggs": m("ggs"), "ggs_table5": m("ggs_table5"),
                              "official": off_out.get(d, {}).get("mean"),
                              "thresholds": parity_thresholds(row), "verdict_top3": v,
                              "verdict_top1": verdict(m("zs_top1"), row),
                              "verdict_top3_cv": verdict(zs3cv, row),
                              "verdict_equiv": verdict(m("ts2vec_equiv"), row)}
            offm = off_out.get(d, {}).get("mean")
            md.append(f"| {d} | {row['TS2Vec']:.3f} | {'—' if offm is None else f'{offm:.3f}'} | "
                      f"{m('ts2vec_equiv'):.3f} | {m('ggs'):.3f} | {m('ggs_table5'):.3f} | {m('zs_top1'):.3f} | "
                      f"{zs3:.3f} | {zs3cv:.3f} | {v} |")
        n_parity = sum(1 for d in targets_out if targets_out[d]["verdict_top3"] == "parity")
        n_median = sum(1 for d in targets_out if targets_out[d]["verdict_top3"] in ("parity", "median"))
        n_parity_cv = sum(1 for d in targets_out if targets_out[d]["verdict_top3_cv"] == "parity")
        md += ["", f"*GGS_T5 出空间（space_legal=false），仅文献对照。",
               f"**P2**：zs_top3(best-of-3) ≥ TS2Vec−{PARITY_MARGIN} 的数据集数 = {n_parity}/{len(targets_out)}"
               f"（≥ 文献中位数：{n_median}/{len(targets_out)}；cv 选择口径 parity：{n_parity_cv}/{len(targets_out)}）"]
        out["targets"] = {"per_dataset": targets_out, "P2_parity_count": n_parity,
                          "median_count": n_median, "P2_cv_parity_count": n_parity_cv,
                          "n_datasets": len(targets_out)}

    # §4b +ladder 第二遍（若跑）
    tgl = _rows_by(rows, "targets_ladder")
    out["ladder_adopted_rung"] = adopted_ladder_rung(lad_out) if lad_out else None
    if tgl:
        md += ["", f"## §4b 六靶 +ladder 第二遍（ts2vec_official + 采纳成分 "
                   f"{out['ladder_adopted_rung']}；REPORT §0.7；次读数，非头条）", "",
               "| 数据集 | TS2Vec 论文 | TS2VEC_EQUIV+L | GGS+L | zs_top1+L | zs_top3(best)+L | zs_top3(cv)+L | 判定(top3) |",
               "|---|---:|---:|---:|---:|---:|---:|---|"]
        tl_out = {}
        for d, rs in _group(tgl, lambda r: r["dataset"]).items():
            per_cls = {k: summarize_seeds(v) for k, v in _group(rs, lambda r: r["cls_key"]).items()}
            arm_of = {a: r["cls_key"] for r in rs for a in r["arm_names"]}

            def m2(a, _p=per_cls, _a=arm_of):
                return _p[_a[a]]["mean"] if a in _a else float("nan")

            top3_keys = [k for k in picks.get(d, {}).get("top3", []) if k in per_cls]
            best, cv_sel = (top3_selection(top3_keys, per_cls) if top3_keys else (None, None))
            zs3 = per_cls[best]["mean"] if best else float("nan")
            zs3cv = per_cls[cv_sel]["mean"] if cv_sel else float("nan")
            row = TABLE10[d]
            tl_out[d] = {"per_cls": per_cls, "zs_top1": m2("zs_top1+ladder"), "zs_top3_best": zs3,
                         "zs_top3_cv": zs3cv, "ts2vec_equiv": m2("ts2vec_equiv+ladder"),
                         "ggs": m2("ggs+ladder"), "ggs_table5": m2("ggs_table5+ladder"),
                         "verdict_top3": verdict(zs3, row), "verdict_top3_cv": verdict(zs3cv, row),
                         "verdict_equiv": verdict(m2("ts2vec_equiv+ladder"), row)}
        for d in TARGETS:
            if d in tl_out:
                t = tl_out[d]
                md.append(f"| {d} | {TABLE10[d]['TS2Vec']:.3f} | {t['ts2vec_equiv']:.3f} | {t['ggs']:.3f} | "
                          f"{t['zs_top1']:.3f} | {t['zs_top3_best']:.3f} | {t['zs_top3_cv']:.3f} | {t['verdict_top3']} |")
        n_par = sum(1 for t in tl_out.values() if t["verdict_top3"] == "parity")
        n_med = sum(1 for t in tl_out.values() if t["verdict_top3"] in ("parity", "median"))
        md += ["", f"+ladder：zs_top3 parity {n_par}/{len(tl_out)}，≥中位数 {n_med}/{len(tl_out)}"]
        out["targets_ladder"] = {"per_dataset": tl_out, "parity_count": n_par, "median_count": n_med}

    # §5 目录 vs R'@290 Spearman
    cat = _rows_by(rows, "catalog")
    cat_out = {}
    if cat:
        rprime = defaultdict(list)
        rp = MX3_DIR / "rprime_cache.jsonl"
        if rp.exists():
            for line in rp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rr.get("iters") == PICK_BUDGET and rr.get("r_prime") is not None:
                    rprime[(rr["dataset"], rr["cls_key"])].append(float(rr["r_prime"]))
        md += ["", "## §5 目录（ts2vec_official 口径，seed 0）vs 切片 R'@290 的 Spearman", "",
               "| 数据集 | n | ρ(all) | ρ(shared-10) | oracle acc | oracle key |", "|---|---:|---:|---:|---:|---|"]
        for d, rs in _group(cat, lambda r: r["dataset"]).items():
            shared = {x["cls_key"] for x in json.loads((MX3_DIR / "cls_catalog.json").read_text(encoding="utf-8"))["shared"]}
            pairs = [(r["acc"], float(np.mean(rprime[(d, r["cls_key"])])), r["cls_key"])
                     for r in rs if (d, r["cls_key"]) in rprime]
            rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else float("nan")
            ps = [p for p in pairs if p[2] in shared]
            rho_s = _spearman([p[0] for p in ps], [p[1] for p in ps]) if len(ps) >= 3 else float("nan")
            orc = max(rs, key=lambda r: r["acc"])
            cat_out[d] = {"n": len(pairs), "rho_all": rho, "rho_shared": rho_s,
                          "oracle_acc": orc["acc"], "oracle_key": orc["cls_key"],
                          "needs_relabel": bool(np.isfinite(rho_s) and rho_s < 0.5)}
            md.append(f"| {d} | {len(pairs)} | {rho:.3f} | {rho_s:.3f} | {orc['acc']:.3f} | {orc['cls_key']} |")
    out["catalog"] = cat_out

    (outdir / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    (outdir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    print(f"→ {outdir / 'analysis.json'} / tables.md", flush=True)


# ================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stages", nargs="*", choices=STAGES, help="dry-run 只算这些阶段")
    ap.add_argument("--eval-audit", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--bs-scan", action="store_true")
    ap.add_argument("--official-ts2vec", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="*", choices=TARGETS)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--audit", choices=("all", "protocol"), default="all",
                    help="all=每跑记 2×2 评估变体；protocol=只记协议自身网格（PhonemeSpectra 省时）")
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT), help="冒烟务必改到别处（铁律 3）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ladder-overrides", default=None,
                    help="--targets 的第二遍（REPORT §0.7）：rung 名（如 ladder_input_mask）或 JSON 字典；"
                         "臂名加 +ladder 后缀、stage=targets_ladder；不传 = 主口径")
    ap.add_argument("--joblib-backend", choices=("threading", "loky"), default="threading",
                    help="GridSearchCV 并行后端：Windows 本机用 threading（loky spawn 太慢）；Linux pod 用 loky")
    args = ap.parse_args()

    protos = load_protocols(TRAIN_YAML)
    space = CLSSpace.from_yaml()
    assert space.validate(TS2VEC_EQUIV) and space.validate(_m1.GGS)
    assert not space.validate(GGS_T5), "GGS_T5 应在空间外（白名单臂）"

    if args.dry_run:
        cmd_dry_run(args, protos)
    plan = [("eval_audit", args.eval_audit), ("reproduce", args.reproduce), ("bs_scan", args.bs_scan),
            ("official", args.official_ts2vec), ("ladder", args.ladder), ("targets", args.targets),
            ("catalog", args.catalog)]
    ran = False
    for st, on in plan:
        if not on:
            continue
        ran = True
        sizes = sizes_from_arrays(_datasets(args) if st in ("eval_audit", "official", "targets", "catalog")
                                  else ("Handwriting",))
        cells = build_stage_cells(st, protos, sizes, args)
        execute(cells, args, st)
    if args.analyze:
        cmd_analyze(args)
        ran = True
    if not (ran or args.dry_run):
        ap.error("至少指定一个阶段 / --dry-run / --analyze")


if __name__ == "__main__":
    main()
