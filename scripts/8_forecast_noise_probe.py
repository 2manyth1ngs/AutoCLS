"""预测任务 R' 噪声探针（ETTh1；判据 results/forecast_noise_probe/PROTOCOL.md 先行 commit）。

测"预测 R'（TS2Vec Ridge 协议、norm 尺度 MSE@48）跨 weight_seed 的重复噪声"。
分类路线的死因链已定位为噪声（工作点 P95 0.110 ≈ 任务特异信号 3-5×，decisive C2）；
预测被寄望于'Ridge 无 SVM 决策边界随机层'——这是推理不是实测，本探针先测噪声。

口径对齐 labels_workpoint 噪声探针（可比性）：10 shared CLS（同一探针目录，
results/route_a_probe/probe_cls_catalog.json 前 10）× k=3 weight_seed（rep0=canonical、
rep1/2=噪声种子，派生式与 2c/7_swa 逐字一致）；iters 扫 {30, 60, 290}（预测工作点
未知：30/60=分类两数据集工作点、290=early 全局口径、官方 TS2Vec ETTh1 训 200 iters
落带内）。MSE 有量纲 → 主判据用相对量（|ΔMSE|/cell 均值 的 P95），阈值见 PROTOCOL.md。

无 threading 后端：Ridge 全程串行 sklearn 直调，无 GridSearchCV/loky（分类协议的
spawn 教训不适用）；OMP=1 仍设（造 label CPU-bound 教训，须在 numpy/torch 导入前）。

用法（幂等、--resume 贯穿）：
    python scripts/8_forecast_noise_probe.py --run --resume
    python scripts/8_forecast_noise_probe.py --analyze
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.forecast_loader import load_forecast_csv, make_pretrain_instances  # noqa: E402
from autocls.downstream import resolve_evaluator  # noqa: E402
from autocls.downstream.forecast_eval import SLIDING_PADDING  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import device_label, load_probe_cls, run_seed_for  # noqa: E402
from autocls.oom_guard import (  # noqa: E402
    ON_OOM_SKIP, RprimeOOM, guarded_run, is_oom_error,
)
from autocls.task_types import TASK_FORECASTING  # noqa: E402

DATASET = "ETTh1"
TASK_ID = "ETTh1_univar_Q48"        # 单变量（AutoCL Table 4 口径）+ 单 horizon Q=48
CSV_PATH = ROOT / "data" / "ETT" / "ETTh1.csv"
PRED_LEN = 48                        # AutoCL B.3：预测的验证指标 = MSE (48 horizons)
INST_LEN = 256                       # 预训练滑窗实例长（路线 A CROP=256 口径）
STRIDE = 24                          # 1 天（小时数据）：train 8640 步 → 350 实例
ITERS_LEVELS = (30, 60, 290)
NOISE_REPS = (1, 2)                  # canonical(=0) 外的 2 个 weight seed → k=3
N_SHARED = 10
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
DEFAULT_OUTDIR = ROOT / "results" / "forecast_noise_probe"

# 判据阈值（PROTOCOL.md §4 写死；边界值归中间档）
GREEN_MAX = 0.05
RED_MIN = 0.15


# ---------------------------------------------------------------- 计划与种子
def seeds_for(cls_key_: str, rep: int) -> int:
    """rep0=canonical；rep1/2=噪声种子——派生式与 2c/7_swa 逐字一致。

    无 split_seed：train/valid/test 为 TS2Vec 官方固定时间切片（12/4/4 月），
    不存在切分随机性。
    """
    if rep == 0:
        return run_seed_for(TASK_ID, cls_key_)
    return run_seed_for(TASK_ID, f"{cls_key_}:noise{rep}")


def plan_work(cls_keys: list[str]) -> list[tuple[int, str, int]]:
    """(iters, cls_key, rep) 全网格；小 iters 档先跑（最快出首批可分析数据）。"""
    return [(it, k, rep)
            for it in sorted(ITERS_LEVELS)
            for k in cls_keys
            for rep in (0, *NOISE_REPS)]


# ---------------------------------------------------------------- 数据与单条评估
@dataclass
class DataPack:
    data: np.ndarray          # (1, T, 8)：7 协变量 + z 后 OT
    train_slice: slice
    valid_slice: slice
    test_slice: slice
    scaler: object
    n_covariate_cols: int
    instances: np.ndarray     # (N, INST_LEN, 8) 预训练滑窗实例（只出自 train 切片）


def load_pack() -> DataPack:
    data, tr, va, te, scaler, pred_lens, n_cov = load_forecast_csv(CSV_PATH, univar=True)
    assert PRED_LEN in pred_lens, f"Q={PRED_LEN} 不在官方 pred_lens {pred_lens}"
    inst = make_pretrain_instances(data, tr, inst_len=INST_LEN, stride=STRIDE)
    return DataPack(data, tr, va, te, scaler, n_cov, inst)


def evaluate_one(
    pack: DataPack, cls_cfg, *, iters: int, weight_seed: int,
    batch_size: int, lr: float, device: str,
) -> dict:
    """单条 (cls, iters, seed) → 预训练 + Ridge 评估。错误语义与 evaluate_rprime
    逐字对齐：训练发散记 NaN 落盘；OOM 抛 RprimeOOM 绝不落盘（oom_guard 四规矩）。"""
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)
        encoder = build_encoder(input_dims=pack.data.shape[-1])
        pretrain_encoder(
            encoder, cls_cfg, pack.instances, max_iters=iters,
            batch_size=batch_size, lr=lr, device=device, seed=weight_seed,
        )
        evaluator = resolve_evaluator(TASK_FORECASTING)  # 铁律 7 分发点
        metrics = evaluator(
            encoder, pack.data,
            train_slice=pack.train_slice, valid_slice=pack.valid_slice,
            test_slice=pack.test_slice, pred_len=PRED_LEN,
            n_covariate_cols=pack.n_covariate_cols, scaler=pack.scaler,
            device=device,
        )
        rec.update(metrics)
        rec["r_prime"] = float(metrics["r_prime"])
    except Exception as e:
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 —— 无 CUDA/已崩，清理失败不掩盖 OOM
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["r_prime"] = float("nan")
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec["device_name"] = device_label(device)
    return rec


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 进程被杀留下的半行：视作未完成
    return out


def run(outdir: Path, device: str, resume: bool) -> None:
    shared_cls = load_probe_cls(PROBE_CATALOG, N_SHARED)[0]
    by_key = dict((k, c) for c, k in shared_cls)
    pack = load_pack()
    print(f"{DATASET}: data {pack.data.shape}, train 实例 {pack.instances.shape}", flush=True)

    cache_path = outdir / "probe_cache.jsonl"
    done = {(r["dataset"], r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in read_jsonl(cache_path) if r.get("r_prime") is not None}
    work = plan_work([k for _, k in shared_cls])
    todo = [w for w in work
            if not (resume and (DATASET, TASK_ID, w[1], w[0], w[2]) in done)]
    print(f"预测噪声探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)

    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    t0 = time.time()
    for i, (iters, key, rep) in enumerate(todo):
        w_seed = seeds_for(key, rep)
        cls_cfg = by_key[key]

        def make_row(rec, _k=key, _c=cls_cfg, _rep=rep, _it=iters):
            rp = rec["r_prime"]
            return {"dataset": DATASET, "task_id": TASK_ID,
                    "task_type": TASK_FORECASTING, "cls_key": _k, "seed": _rep,
                    "iters": _it, "weight_seed": seeds_for(_k, _rep),
                    "pred_len": PRED_LEN, "padding": SLIDING_PADDING,
                    "inst_len": INST_LEN, "stride": STRIDE,
                    "split": "ts2vec_official_12_4_4", **rec,
                    "r_prime": None if not np.isfinite(rp) else float(rp),
                    "cls": dataclasses.asdict(_c)}

        full = guarded_run(
            lambda: evaluate_one(pack, cls_cfg, iters=iters, weight_seed=w_seed,
                                 batch_size=train_cfg["batch_size"],
                                 lr=train_cfg["lr"], device=device),
            make_row, cache_path=cache_path, inflight=outdir / "inflight.txt",
            task_id=TASK_ID, cls_key=key, iters=iters,
            on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
            context={"dataset": DATASET, "noise_rep": rep},
            log=lambda m: print(f"[fc {i + 1}/{len(todo)}] {m}", flush=True))
        if full is None:
            continue
        eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
        print(f"[fc {i + 1}/{len(todo)}] iters={iters} {key[:8]} rep{rep} "
              f"MSE={full['r_prime']} ({full['seconds']}s) ETA {eta / 60:.1f}min",
              flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 分析（纯函数，单测钉死）
def pctl(diffs: list[float], q: float) -> float:
    """与 2c calibrate/7_swa 同式：sorted 后取 round(q*(n-1)) 索引。"""
    ds = sorted(diffs)
    return ds[min(len(ds) - 1, int(round(q * (len(ds) - 1))))]


def load_cells(rows: list[dict]) -> dict:
    """(iters, cls_key) → {rep: r_prime}（同键后行覆盖前行；失败条不进 cell）。"""
    cells: dict = defaultdict(dict)
    for r in rows:
        if r.get("r_prime") is None:
            continue
        cells[(r["iters"], r["cls_key"])][r["seed"]] = float(r["r_prime"])
    return dict(cells)


def _summary(diffs: list[float]) -> dict:
    return {"median": statistics.median(diffs), "p90": pctl(diffs, 0.90),
            "p95": pctl(diffs, 0.95), "max": max(diffs)}


def noise_stats(*, probe_cells: dict, iters: int) -> dict:
    """一个 iters 档的噪声统计：绝对/相对重复差 + per-cell CV。

    相对差分母 = cell 三 rep 均值（比配对均值稳，预注册口径）；CV 用 ddof=1。
    rep 不满 2 的 cell 无重复差可算，整体剔除（计入 n_cells 的分母之外）。
    """
    abs_diffs, rel_diffs, cell_rows = [], [], []
    for (it, key), reps in sorted(probe_cells.items()):
        if it != iters or len(reps) < 2:
            continue
        vals = [reps[r] for r in sorted(reps)]
        mean = float(np.mean(vals))
        pairs = [abs(vals[i] - vals[j])
                 for i in range(len(vals)) for j in range(i + 1, len(vals))]
        abs_diffs += pairs
        rel_diffs += [d / mean for d in pairs]
        cv = float(np.std(vals, ddof=1) / mean) if len(vals) >= 2 else None
        cell_rows.append({"cls_key": key, "mean_mse": mean, "n_reps": len(vals),
                          "cv": cv,
                          "max_abs_diff": max(pairs), "max_rel_diff": max(pairs) / mean})
    if not abs_diffs:
        return {"n_cells": 0, "n_pairs": 0, "cells": []}
    cvs = [c["cv"] for c in cell_rows if c["cv"] is not None]
    return {
        "n_cells": len(cell_rows), "n_pairs": len(abs_diffs),
        "abs": _summary(abs_diffs), "rel": _summary(rel_diffs),
        "cv_median": statistics.median(cvs), "cv_max": max(cvs),
        "heavy_tail_p95_over_median": (
            float("inf") if _summary(rel_diffs)["median"] == 0
            else _summary(rel_diffs)["p95"] / _summary(rel_diffs)["median"]),
        "cells": cell_rows,
    }


def verdict_of(rel_p95: float) -> dict:
    """单档判定（PROTOCOL.md §4；边界值归中间档，宁保守不放行）。"""
    if rel_p95 < GREEN_MAX:
        v = "green"
    elif rel_p95 > RED_MIN:
        v = "red"
    else:
        v = "mid"
    return {"rel_p95": rel_p95, "verdict": v}


def overall_verdict(rel_p95_by_iters: dict[int, float]) -> dict:
    """总判定 = 各档中 rel P95 最低者（将来工作点会取噪声最低且信号可用的档）。"""
    best_iters = min(rel_p95_by_iters, key=rel_p95_by_iters.get)
    return {"best_iters": best_iters,
            "per_level": {it: verdict_of(v)["verdict"]
                          for it, v in rel_p95_by_iters.items()},
            **verdict_of(rel_p95_by_iters[best_iters])}


def analyze(outdir: Path) -> dict:
    rows = read_jsonl(outdir / "probe_cache.jsonl")
    cells = load_cells(rows)
    res: dict = {"n_rows": len(rows)}
    rel_by_iters: dict[int, float] = {}
    for it in ITERS_LEVELS:
        st = noise_stats(probe_cells=cells, iters=it)
        res[f"iters_{it}"] = st
        if st["n_cells"]:
            rel_by_iters[it] = st["rel"]["p95"]
            # CLS 质量与噪声的相关（诊断，不参与判定；UM 曾见 ρ≈0.52）
            from scipy.stats import spearmanr

            means = [c["mean_mse"] for c in st["cells"]]
            cvs = [c["cv"] for c in st["cells"]]
            rho = spearmanr(means, cvs)
            st["spearman_meanmse_cv"] = {"rho": float(rho.statistic),
                                         "p": float(rho.pvalue)}
    if rel_by_iters:
        res["verdict"] = overall_verdict(rel_by_iters)
    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 判据打印（PROTOCOL.md 写死的阈值）----
    for it in ITERS_LEVELS:
        st = res.get(f"iters_{it}", {})
        if not st.get("n_cells"):
            print(f"[iters={it}] 无完整 cell", flush=True)
            continue
        print(f"[iters={it}] cells={st['n_cells']} pairs={st['n_pairs']} | "
              f"abs med={st['abs']['median']:.4f} P95={st['abs']['p95']:.4f} "
              f"max={st['abs']['max']:.4f} | rel med={st['rel']['median']:.3%} "
              f"P95={st['rel']['p95']:.3%} max={st['rel']['max']:.3%} | "
              f"CV med={st['cv_median']:.3%} max={st['cv_max']:.3%} | "
              f"重尾 P95/med={st['heavy_tail_p95_over_median']:.1f}", flush=True)
    if "verdict" in res:
        v = res["verdict"]
        label = {"green": "绿灯（噪声远小于分类）", "mid": "中间档（看信号侧）",
                 "red": "红灯（预测路线核心假设不成立）"}[v["verdict"]]
        print(f"\n[主判据] 最优档 iters={v['best_iters']} 相对重复差 P95="
              f"{v['rel_p95']:.3%} → {label}", flush=True)
        print("[对照] 分类 HW 工作点：abs P95=0.110（ACC 尺度相对 ≈12-20%）、"
              "required-k≈11", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not (args.run or args.analyze):
        ap.error("至少指定 --run 或 --analyze")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.run:
        run(args.outdir, args.device, args.resume)
    if args.analyze:
        analyze(args.outdir)


if __name__ == "__main__":
    main()
