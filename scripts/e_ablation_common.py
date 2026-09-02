"""实验 1 / 实验 2 共用：工作点数据装载 + C2 同口径 E 消融块 + 基线带（制式）。

装载/划分/消融的口径全部逐字沿用 scripts/4b_decisive_e_ablation.py（协议
results/decisive_e_ablation/PROTOCOL.md §4–§5）——本模块只把 4b 的 run_one 拆成
"训练（调用方决定 set_encoder 与训练循环）"与"消融评估（此处）"两段，便于换 E 通路 /
换 label 模型而不动评估口径。
"""
from __future__ import annotations

import importlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import Pair, with_reversed  # noqa: E402
from autocls.labeling import comparator_records, load_rprime_cache  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402

_c2 = importlib.import_module("scripts.4b_decisive_e_ablation")
DATASETS = _c2.DATASETS
HOLDOUT = _c2.HOLDOUT
N_ABLATION_SEEDS = _c2.N_ABLATION_SEEDS
PASS_DELTA = _c2.PASS_DELTA
PASS_SIG_COUNT = _c2.PASS_SIG_COUNT
ds_of = _c2.ds_of
forward_pairs_by_task = _c2.forward_pairs_by_task
within_assignment = _c2.within_assignment
cross_assignment = _c2.cross_assignment
mcnemar_one_sided = _c2.mcnemar_one_sided
build_runs = _c2.build_runs

LABELS_DIR = ROOT / "results" / "labels_workpoint"
# 基线缓存（iters=0 随机初始化 R'）：strict 破坏制式 = R' < 均值 − 2·max(σ, 量子)
# （autocls.workpoint.destruction_participation 同式；来源同 2c SPEC）
BASELINE_CACHE = {
    "Handwriting": ROOT / "results" / "iters_workpoint" / "uea" / "baseline_cache.jsonl",
    "UniMiB-SHAR": ROOT / "results" / "iters_workpoint" / "candidates" / "baseline_cache.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class WorkpointData:
    """28 任务工作点数据的一次性装载（与 4b main 的 --train 段逐字同口径）。"""

    def __init__(self, labels_dir: Path = LABELS_DIR):
        self.labels_dir = labels_dir
        labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
        self.cfg = yaml.safe_load((ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))
        self.task_type = labels_cfg["task_type"]
        wp = labels_cfg["workpoint"]
        self.iters_map, self.delta_map = wp["iters_per_dataset"], wp["delta_per_dataset"]
        self.nz = labels_cfg["nz"]
        space = CLSSpace.from_yaml()
        self.encoder = CLSEncoder(space)
        cache = load_rprime_cache(labels_dir / "rprime_cache.jsonl")
        self.cache = cache
        records = []
        for d in DATASETS:
            rs = comparator_records(cache, self.encoder, iters=self.iters_map[d])
            assert all(ds_of(r.task_id) == d for r in rs), f"iters={self.iters_map[d]} 捞到非 {d} 记录"
            records += rs
        self.records = records
        self.all_tids = sorted({r.task_id for r in records})
        assert len(self.all_tids) == 28, f"应有 28 任务，得到 {len(self.all_tids)}"
        self.fwd_pairs = forward_pairs_by_task(records, self.task_type, self.delta_map)
        feat_dir = labels_dir / "task_features"
        self.task_windows = {t: np.load(feat_dir / f"{t}.npz")["windows_repr"] for t in self.all_tids}
        self.task_E_init = {t: np.load(feat_dir / f"{t}.npz")["E"] for t in self.all_tids}
        self.init_path = labels_dir / "set_encoder_init.pt"
        self.n_eval = {(rec["task_id"]): int(rec["n_eval"]) for rec in cache.values()}
        self.cls_dim = records[0].vec.shape[0]
        self.dim_input = next(iter(self.task_windows.values())).shape[-1]

    def delta_by_task(self, tids) -> dict[str, float]:
        return {t: self.delta_map[ds_of(t)] for t in tids}

    def records_in(self, tids: set[str]):
        return [r for r in self.records if r.task_id in tids]

    def hard_test_pairs(self, test_tids: set[str]) -> tuple[list[Pair], int]:
        """C2 硬评估集：δ 过滤单向对（按任务过滤后）+ 反向；返回 (directed, n_fwd)。"""
        fwd = [p for p in self.fwd_pairs if p.task_id in test_tids]
        return with_reversed(fwd), len(fwd)


def baseline_bands(name: str) -> dict[str, tuple[float, float]]:
    """task_id → (基线均值, 带半宽 2·max(σ, 量子))；与 2c build_baseline_bands 同式。"""
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(BASELINE_CACHE[name]):
        if r.get("r_prime") is not None and r["task_id"].startswith(name):
            rows_by_task[r["task_id"]].append(r)
    out = {}
    for tid, rows in rows_by_task.items():
        vals = [float(r["r_prime"]) for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        quantum = max((1.0 / r["n_eval"] for r in rows if r.get("n_eval")), default=0.0)
        out[tid] = (statistics.fmean(vals), 2.0 * max(std, quantum))
    return out


def strict_destroyed(data: WorkpointData) -> dict[tuple[str, str], bool]:
    """(task_id, cls_key) → 是否 strict 破坏（R' < 均值 − 带）。"""
    bands = {}
    for d in DATASETS:
        bands.update(baseline_bands(d))
    out = {}
    for r in data.records:
        mean, band = bands[r.task_id]
        out[(r.task_id, r.key)] = bool(r.r < mean - band)
    return out


def ablation_block(engine, directed: list[Pair], n_fwd: int, test_tids: set[str],
                   all_tids: list[str]) -> dict:
    """4b run_one 的评估段（normal / within×5 / cross×5 / zero + McNemar + 达标）。"""
    base = engine.evaluate(directed, return_correct=True)
    base_fwd = base["correct"][:n_fwd]

    def ablate(mode: str) -> list[dict]:
        rows = []
        for s in range(N_ABLATION_SEEDS):
            amap = (within_assignment(sorted(test_tids), s) if mode == "within"
                    else cross_assignment(sorted(test_tids), all_tids, s))
            r = engine.evaluate(directed, e_mode="assign", e_assignment=amap, return_correct=True)
            rows.append({"seed": s, "acc": r["acc"], "loss": r["loss"],
                         "delta_acc": base["acc"] - r["acc"],
                         "mcnemar": mcnemar_one_sided(base_fwd, r["correct"][:n_fwd])})
        return rows

    within_rows, cross_rows = ablate("within"), ablate("cross")
    zero = engine.evaluate(directed, e_mode="zero", return_correct=True)

    def agg(rows: list[dict]) -> dict:
        deltas = [r["delta_acc"] for r in rows]
        n_sig = sum(r["mcnemar"]["p"] < 0.05 for r in rows)
        return {"mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)), "n_sig": n_sig,
                "pass": bool(np.mean(deltas) >= PASS_DELTA and n_sig >= PASS_SIG_COUNT)}

    fwd_acc = float(np.mean(base_fwd)) if n_fwd else float("nan")
    btest = (binomtest(int(round(fwd_acc * n_fwd)), n_fwd, 0.5, alternative="greater").pvalue
             if n_fwd else float("nan"))
    return {
        "n_test_pairs_directed": len(directed), "n_test_pairs_forward": n_fwd,
        "normal": {"acc": base["acc"], "loss": base["loss"], "binom_p_forward": float(btest)},
        "within": {"rows": within_rows, **agg(within_rows)},
        "cross": {"rows": cross_rows, **agg(cross_rows)},
        "zero": {"acc": zero["acc"], "loss": zero["loss"], "delta_acc": base["acc"] - zero["acc"],
                 "mcnemar": mcnemar_one_sided(base_fwd, zero["correct"][:n_fwd])},
    }


def tier_of(block: dict) -> str:
    """协议档位：zero / detectable / pass（实验 1 §4 F3 与实验 2 §5 Q1 共用口径）。"""
    d, n = block["mean_delta"], block["n_sig"]
    if block["pass"]:
        return "pass"
    if d >= 0.005 and n >= 3:
        return "detectable"
    if d < 0.005 and n <= 1:
        return "zero"
    return "ambiguous"


def load_set_encoder_init(data: WorkpointData):
    from autocls.task_embed import TaskSetEncoder
    enc = TaskSetEncoder(dim_input=data.dim_input, nz=data.nz)
    enc.load_state_dict(torch.load(data.init_path, map_location="cpu", weights_only=True))
    return enc
