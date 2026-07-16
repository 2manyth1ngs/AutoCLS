"""路线 A pre-flight 探针（只验证，不造全量 label、不上云）。

在烧云 GPU 前，验证两个会让路线 A 失败的隐含前提：

风险1 — R' 可靠性：δ=0.02 在 HAR(N=7352, test 充裕) 上标定；UEA 多数数据集切成
任务后 test 只有几十个，ACC 噪声 σ≈0.5/√n_test 可能 >> δ。噪声 >> δ 时 δ 过滤不仅
无效、反而"筛选出噪声"（把纯噪声大差值当可信标签），比较器退回学全局平均排名。

风险2 — 短序列 × 固定 encoder 架构失配：dilated CNN 感受野为 L≥100 设计；PenDigits
L=8、SpokenArabicDigits 真实 L=4-93 上多数膨胀层被 padding 主导，增强（crop/freq_mask/
resize）可能失效，所有 CLS 的 R' 挤在一起 → 对训练比较器零贡献。

探针协议（4 代表性源数据集，覆盖不同 N/C 与序列长度）：
  PenDigits(L8, N/C=1099)  SpokenArabicDigits(L4-93, N/C=880)
  PhonemeSpectra(L217, N/C=171)  Cricket(L1197, N/C=15)
每数据集：切 2-3 任务（RandomClassSubsetSlicer 通用切片，非 HAR 动/静分层）×
探针 CLS（10 个跨 HAR 好坏排名，results/route_a_probe/probe_cls_catalog.json）×
3-5 个随机种子重复 early-val(290 iters)。逐条落盘缓存 + --resume（铁律 3）。

噪声测量口径（关键）：同一 (task, seed) 下所有 CLS 共用同一 train/eval 划分
（对齐生产：造 label 时 CLS 在同一任务上竞争、划分固定），划分随 seed 变。据此
可算两种噪声：① 每 (task,cls) 跨 seed 的 R' std（含划分方差的保守上界，作"噪声
vs δ"表头）；② 每 task 跨 seed 的 (R'_a − R'_b) std（δ 真正要盖过的成对标签噪声，
含真实协方差，是金标准）。

铁律遵循：early-val 一律 290 iters（train.yaml，铁律 2）；R' 方向经 task_types
（此处只测数值不造 y，方向无关）；不训练 TS2Vec、不算 E（探针只要 R'，不构成
zero-shot 证据）。本脚本是探针，产物落 results/route_a_probe/，与生产 labels 隔离。

用法：
  # 1) 增强生效性诊断（CPU，秒级）——回答风险2的"增强是否实际生效"
  python scripts/probe_route_a.py --diagnose-augs
  # 2) smoke：每数据集只跑 1 task×2 cls×1 seed，低 iters，验证四条链路都通
  python scripts/probe_route_a.py --run --smoke
  # 3) 全量（脱离会话跑，后台会被杀，见 memory）：
  python scripts/probe_route_a.py --run --resume
  # 4) 分析已落盘缓存 → 判定表 + analysis.json
  python scripts/probe_route_a.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.contrastive.augmentations import ORDERS, augment_two_views  # noqa: E402
from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.task_slicer import materialize_task, slice_pool_tasks  # noqa: E402
from autocls.data.slice_strategies import RandomClassSubsetSlicer  # noqa: E402
from autocls.data.uea import load_uea_cached  # noqa: E402
from autocls.downstream import resolve_evaluator  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    append_jsonl,
    run_seed_for,
    split_task_train_eval,
)
from autocls.space.cls_space import CLS  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402

OUTDIR = ROOT / "results" / "route_a_probe"
CATALOG = OUTDIR / "probe_cls_catalog.json"
CACHE = OUTDIR / "rprime_cache.jsonl"
AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop")


# ---------------------------------------------------------------------------
# 探针数据集配置：class 子集大小、通道全取、样本上限、CROP、任务/种子数
# 覆盖 N/C 从 1099 到 15、L 从 8 到 1197（裁 256）。cost∝L 与批大小；小 N/C 的
# Cricket 是风险1最坏情况（n_eval 极小→噪声爆），长 L 的 PhonemeSpectra/Cricket 是
# 风险2的"正常长度参照"（感受野足够，CLS 应能拉开）。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DsConfig:
    name: str
    n_classes_range: tuple[int, int]
    n_tasks: int
    n_seeds: int
    crop_len: int | None = None
    n_samples: int = 768        # 每任务采样上限（大数据集限成本；小的取全部）
    min_samples: int = 24       # 探针放宽（生产 min_samples=128 是 batch 口径，此处不适用）
    n_cls: int = 8              # 用前 n_cls 个探针 CLS（8 足够测 spread，省 20% 成本）


# 类别子集刻意取大（off-ceiling）：smoke 发现 2-3 类的 digits/audio-digits/Cricket
# 在 30 iters 就已 R'≈1.0（任务过易），290 iters 下所有 CLS 会全挤在天花板 → 既测
# 不出 CLS 区分度、噪声也被 ceiling 压没，会把"任务过易"误判成"短序列 encoder 失效"。
# 用更难的多类任务把 R' 拉到判别带（≈0.5-0.9），才能同时测噪声与 CLS 分辨力。
# PhonemeSpectra(39 类) 小子集即 0.68 off-ceiling + n_eval 大，是"正常长度可靠参照"。
DATASETS: dict[str, DsConfig] = {
    "PenDigits":          DsConfig("PenDigits",          (6, 10), n_tasks=3, n_seeds=5),
    "SpokenArabicDigits": DsConfig("SpokenArabicDigits", (6, 10), n_tasks=3, n_seeds=4),
    "PhonemeSpectra":     DsConfig("PhonemeSpectra",     (3, 6),  n_tasks=2, n_seeds=4),
    "Cricket":            DsConfig("Cricket",            (6, 10), n_tasks=2, n_seeds=5,
                                   crop_len=256, n_samples=2048, min_samples=20),
}


def load_probe_cls() -> list[dict]:
    """读 10 个探针 CLS（cls_key + cls 字段字典 + HAR 参照排名）。"""
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    return cat["cls"]


def load_dataset_full(cfg: DsConfig) -> tuple[np.ndarray, np.ndarray]:
    """源数据集 = TRAIN+TEST 全量（探针把整个数据集当源池候选，非 zero-shot 评估）。

    合并给 Cricket 这类小数据集尽量多样本（N/C 按全量算）；不等长/长序列由
    load_uea_cached 统一 pad/crop。
    """
    Xtr, ytr = load_uea_cached(cfg.name, split="TRAIN", crop_len=cfg.crop_len)
    Xte, yte = load_uea_cached(cfg.name, split="TEST", crop_len=cfg.crop_len)
    # 两切分独立 pad 到各自 L_max：合并前对齐到较大的 T
    T = max(Xtr.shape[1], Xte.shape[1])
    Xtr = _pad_T(Xtr, T)
    Xte = _pad_T(Xte, T)
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    return X, y


def _pad_T(X: np.ndarray, T: int) -> np.ndarray:
    if X.shape[1] == T:
        return X
    out = np.zeros((X.shape[0], T, X.shape[2]), dtype=X.dtype)
    out[:, : X.shape[1]] = X
    return out


def slice_tasks(cfg: DsConfig, X: np.ndarray, y: np.ndarray, seed: int = 0):
    """用通用分类切片（RandomClassSubsetSlicer）切 n_tasks 个任务，全通道。

    channels 全取（n_channels_range=(C,C)）以隔离"类别子集×长度"效应，避免通道子集
    引入额外混淆。min_samples 放宽到探针档（见 DsConfig）。
    """
    C = X.shape[2]
    strat = RandomClassSubsetSlicer(n_classes_range=cfg.n_classes_range)
    return slice_pool_tasks(
        y, np.arange(len(y)), pool_name=cfg.name, n_tasks=cfg.n_tasks,
        task_type=TASK_CLASSIFICATION, strategy=strat,
        n_channels_range=(C, C), n_total_channels=C,
        n_samples=cfg.n_samples, min_samples=cfg.min_samples, seed=seed,
    )


# ---------------------------------------------------------------------------
# 单条 (dataset, task, cls, seed) → R'
# 划分对 (task, seed) 固定（所有 CLS 共用）；init/aug/svm 用 (task, cls, seed) 派生
# ---------------------------------------------------------------------------
def run_one(task, cls_cfg: CLS, X, y, *, seed: int, iters: int, device: str) -> dict:
    from autocls.labeling import cls_key
    key = cls_key(cls_cfg)
    Xt, yt = materialize_task(task, X, y)
    split_seed = run_seed_for(task.task_id, f"split:{seed}")       # 同 (task,seed) 全 CLS 共用
    run_seed = run_seed_for(task.task_id, f"{key}:{seed}")         # init/aug/eval
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=0.3, seed=split_seed,
    )
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(run_seed)
        encoder = build_encoder(input_dims=Xt.shape[-1])
        pretrain_encoder(
            encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
            batch_size=128, lr=1e-3, device=device, seed=run_seed,
        )
        evaluator = resolve_evaluator(task.task_type)
        metrics = evaluator(encoder, Xt[tr_idx], yt[tr_idx], Xt[ev_idx], yt[ev_idx],
                            device=device, seed=run_seed)
        rec.update(metrics)
        rec["r_prime"] = float(metrics["r_prime"])
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["r_prime"] = float("nan")
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec.update({"n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)),
                "split_seed": split_seed, "run_seed": run_seed})
    return rec


def cache_key(rec: dict) -> tuple:
    return (rec["dataset"], rec["task_id"], rec["cls_key"], rec["seed"])


def load_cache() -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[cache_key(r)] = r
    return done


def run_probe(smoke: bool, resume: bool, device: str) -> None:
    from autocls.labeling import cls_key
    OUTDIR.mkdir(parents=True, exist_ok=True)
    iters = 30 if smoke else 290
    probe_cls = load_probe_cls()

    if CACHE.exists() and not resume and not smoke:
        sys.exit(f"{CACHE} 已存在：加 --resume 续跑，或删除后重跑")
    cache = load_cache()

    # 落盘任务目录（可复现 + 分析时取 classes/tag/n_ch/L）
    tasks_meta: dict = {}
    # 每数据集先备好 (tasks, cls 列表, n_seeds)，再按 seed-major 拼 worklist
    per_ds: list[tuple] = []
    max_seeds = 0
    for name, cfg in DATASETS.items():
        n_tasks = 1 if smoke else cfg.n_tasks
        n_seeds = 1 if smoke else cfg.n_seeds
        n_cls = 2 if smoke else cfg.n_cls
        t0 = time.time()
        X, y = load_dataset_full(cfg)
        tasks = slice_tasks(cfg, X, y)[:n_tasks]
        print(f"[{name}] X={X.shape} y_cls={len(np.unique(y))} → {len(tasks)} tasks "
              f"({time.time()-t0:.1f}s load+slice)", flush=True)
        tasks_meta[name] = {
            "L": int(X.shape[1]), "n_ch": int(X.shape[2]), "crop_len": cfg.crop_len,
            "tasks": [{"task_id": t.task_id, "classes": list(t.classes),
                       "n_samples": len(t.sample_indices), "tag": t.tag} for t in tasks],
        }
        cls_list = [(CLS(**cd["cls"]), cls_key(CLS(**cd["cls"]))) for cd in probe_cls[:n_cls]]
        per_ds.append((name, tasks, cls_list, n_seeds))
        max_seeds = max(max_seeds, n_seeds)
        cfg_runtime[name] = (X, y, {t.task_id: t for t in tasks})

    # seed-major：外层 seed，内层数据集/任务/CLS——中途停止也能覆盖全部 4 数据集
    # （首轮 seed=0 即得每 (task,cls) 一点 → CLS spread 全覆盖；后续 seed 加深噪声分辨）
    worklist: list[tuple] = []
    for s in range(max_seeds):
        for name, tasks, cls_list, n_seeds in per_ds:
            if s >= n_seeds:
                continue
            for t in tasks:
                for c, key in cls_list:
                    worklist.append((name, t, c, key, s))

    (OUTDIR / "tasks_meta.json").write_text(
        json.dumps(tasks_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    todo = [w for w in worklist if (w[0], w[1].task_id, w[3], w[4]) not in cache]
    print(f"设备 {device} | iters={iters} | 计划 {len(worklist)} 条，已缓存 "
          f"{len(worklist)-len(todo)}，待跑 {len(todo)}", flush=True)

    for i, (name, task, c, key, s) in enumerate(todo):
        X, y, _ = cfg_runtime[name]
        rec = run_one(task, c, X, y, seed=s, iters=iters, device=device)
        full = {"dataset": name, "task_id": task.task_id, "tag": task.tag,
                "classes": list(task.classes), "cls_key": key, "seed": s,
                "iters": iters, **rec}
        if not np.isfinite(full["r_prime"]):
            full["r_prime"] = None
        # smoke 写独立缓存，不污染全量
        target = (OUTDIR / "smoke_cache.jsonl") if smoke else CACHE
        append_jsonl(target, full)
        cache[cache_key(full)] = full
        shown = full["r_prime"]
        st = f"R'={shown:.4f}" if shown is not None else f"FAIL {rec['error']}"
        print(f"[{i+1}/{len(todo)}] {name}/{task.task_id} {key[:8]} s{s} {st} "
              f"({rec['seconds']}s, n_ev={rec['n_eval']})", flush=True)
    print("探针运行完成。", flush=True)


cfg_runtime: dict = {}


# ---------------------------------------------------------------------------
# 增强生效性诊断（风险2）：在各数据集真实 L（post-pad/crop）上，逐增强测实际改变量
# ---------------------------------------------------------------------------
def diagnose_augs(device: str = "cpu") -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    report: dict = {}
    for name, cfg in DATASETS.items():
        X, y = load_dataset_full(cfg)
        # 取一个 2 类子集的一批真实样本
        cls0 = np.unique(y)[:2]
        idx = np.where(np.isin(y, cls0))[0][:64]
        xb = torch.from_numpy(X[idx].astype(np.float32))  # (B, T, C)
        B, T, C = xb.shape
        info: dict = {"L": T, "n_ch": C, "augs": {}}

        # 单增强隔离测：p=0.5（freq/point/jitter/rescale/resize）与 crop p=0.5,0.9
        for aug in ("resize", "rescale", "jitter", "point_mask", "freq_mask"):
            params = {k: 0.0 for k in AUG_FIELDS}
            params[aug] = 0.5
            v1, v2, s1, s2 = augment_two_views(xb, params, order=0, generator=gen)
            # 与原序列共享段对比改变量（resize 改长，单独处理）
            if aug == "resize":
                info["augs"][aug] = {"p": 0.5, "T_out": int(v1.shape[1]),
                                     "changed": int(v1.shape[1]) != T,
                                     "note": "极短序列上长度量子大，round 后常无变化" if T < 20 else ""}
            else:
                rel = _rel_change(xb, v1)
                info["augs"][aug] = {"p": 0.5, "mean_rel_change": round(rel, 4),
                                     "effective": rel > 1e-4}

        # freq_mask 频率分辨：n_freq 与实际可掩频点
        n_freq = T // 2 + 1
        info["augs"]["freq_mask"]["n_freq"] = int(n_freq)

        # crop：共享段长度与是否退化（两视图是否真的不同）
        for p in (0.5, 0.9):
            params = {k: 0.0 for k in AUG_FIELDS}
            params["crop"] = p
            v1, v2, s1, s2 = augment_two_views(xb, params, order=0, generator=gen)
            shared_len = s1.stop - s1.start
            info["augs"][f"crop@{p}"] = {
                "T_shared": int(shared_len), "v1_len": int(v1.shape[1]),
                "v2_len": int(v2.shape[1]),
                "views_differ": bool(not torch.equal(v1, v2) or v1.shape != v2.shape),
                "degenerate": bool(shared_len >= T),  # 共享段=全长 → crop 无效果
            }
        report[name] = info
        print(f"[{name}] L={T} ch={C}: "
              f"freq n_freq={n_freq}, "
              f"crop@0.9 T_shared={report[name]['augs']['crop@0.9']['T_shared']}", flush=True)

    (OUTDIR / "aug_diagnostic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n增强诊断 → {OUTDIR/'aug_diagnostic.json'}", flush=True)


def _rel_change(x0: torch.Tensor, x1: torch.Tensor) -> float:
    """两张量在共享前缀长度上的相对改变量（对增强是否'实际改动数据'的度量）。"""
    T = min(x0.shape[1], x1.shape[1])
    a, b = x0[:, :T], x1[:, :T]
    denom = a.abs().mean().item() + 1e-8
    return (a - b).abs().mean().item() / denom


# ---------------------------------------------------------------------------
# 分析：噪声 vs δ、所需 k、重标定 δ、CLS 区分度、成对标签噪声、可靠性判定
# ---------------------------------------------------------------------------
def analyze() -> None:
    cache = load_cache()
    if not cache:
        sys.exit(f"{CACHE} 无数据，先跑 --run")
    recs = list(cache.values())
    tasks_meta = json.loads((OUTDIR / "tasks_meta.json").read_text(encoding="utf-8"))

    # 组织：per dataset -> per task -> per cls -> {seed: r'}
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in recs:
        rp = r["r_prime"]
        data[r["dataset"]][r["task_id"]][r["cls_key"]][r["seed"]] = (
            float(rp) if rp is not None else np.nan)

    delta = 0.02
    out: dict = {"delta_reference": delta, "datasets": {}}
    print(f"\n{'='*92}\n路线 A 探针分析（δ_ref={delta}）\n{'='*92}")

    for ds, tasks in data.items():
        meta = tasks_meta.get(ds, {})
        # ① 每 (task,cls) 跨 seed 的 R' std（含划分方差，保守）
        rprime_stds, cls_means_per_task = [], {}
        # ② 成对 (R'_a - R'_b) 跨 seed std（共享划分，真实成对噪声）
        pair_stds, pair_sign_stable = [], []
        # CLS 区分度：每 task 各 CLS 的 seed 均值，spread = max-min / std
        cls_spread_per_task = []
        fail = 0
        for tid, clsmap in tasks.items():
            seeds_all = sorted({s for cm in clsmap.values() for s in cm})
            cls_seed_mean = {}
            for key, sm in clsmap.items():
                vals = np.array([sm[s] for s in sorted(sm)], dtype=float)
                fail += int(np.isnan(vals).sum())
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 2:
                    rprime_stds.append(vals.std(ddof=1))
                if len(vals) >= 1:
                    cls_seed_mean[key] = np.nanmean(
                        np.array([sm.get(s, np.nan) for s in seeds_all], dtype=float))
            cls_means_per_task[tid] = cls_seed_mean
            # CLS 区分度
            means = np.array([m for m in cls_seed_mean.values() if np.isfinite(m)])
            if len(means) >= 2:
                cls_spread_per_task.append({
                    "task_id": tid, "range": float(means.max() - means.min()),
                    "std": float(means.std(ddof=1)), "n_cls": int(len(means)),
                    "min": float(means.min()), "max": float(means.max())})
            # 成对噪声：对每对 CLS，用共享 seed 上的差值序列求 std + 符号稳定性
            keys = list(clsmap.keys())
            for a, b in combinations(keys, 2):
                diffs = []
                for s in seeds_all:
                    va, vb = clsmap[a].get(s, np.nan), clsmap[b].get(s, np.nan)
                    if np.isfinite(va) and np.isfinite(vb):
                        diffs.append(va - vb)
                diffs = np.array(diffs, dtype=float)
                if len(diffs) >= 2:
                    pair_stds.append(diffs.std(ddof=1))
                    signs = np.sign(diffs[np.abs(diffs) > 1e-9])
                    if len(signs) >= 2:
                        pair_sign_stable.append(float(np.all(signs == signs[0])))

        rprime_std = float(np.mean(rprime_stds)) if rprime_stds else float("nan")
        rprime_std_p90 = float(np.percentile(rprime_stds, 90)) if rprime_stds else float("nan")
        pair_std = float(np.mean(pair_stds)) if pair_stds else float("nan")
        sign_stable = float(np.mean(pair_sign_stable)) if pair_sign_stable else float("nan")
        k_needed = int(np.ceil((rprime_std / delta) ** 2)) if rprime_std == rprime_std else None
        delta_recal = round(3 * rprime_std, 3)  # δ 取噪声上界略高处（≈3σ）
        # 区分度信号比：CLS 均值 spread 相对 seed 噪声
        spreads = [c["std"] for c in cls_spread_per_task]
        cls_spread_std = float(np.mean(spreads)) if spreads else float("nan")
        snr = cls_spread_std / rprime_std if rprime_std and rprime_std == rprime_std else float("nan")

        verdict = _verdict(ds, rprime_std, delta, snr, sign_stable)
        out["datasets"][ds] = {
            "L": meta.get("L"), "n_ch": meta.get("n_ch"),
            "n_tasks": len(tasks), "n_fail_runs": fail,
            "rprime_std_mean": round(rprime_std, 4),
            "rprime_std_p90": round(rprime_std_p90, 4),
            "noise_vs_delta": round(rprime_std / delta, 2) if rprime_std == rprime_std else None,
            "k_needed_for_noise_lt_delta": k_needed,
            "delta_recalibrated": delta_recal,
            "pairwise_diff_std_mean": round(pair_std, 4),
            "pair_sign_stability": round(sign_stable, 3),
            "cls_spread_std_mean": round(cls_spread_std, 4),
            "cls_spread_snr": round(snr, 2) if snr == snr else None,
            "cls_spread_per_task": cls_spread_per_task,
            "n_eval_per_task": {tid: _task_n_eval(recs, ds, tid) for tid in tasks},
            "verdict": verdict,
        }
        print(f"\n[{ds}] L={meta.get('L')} ch={meta.get('n_ch')} tasks={len(tasks)} "
              f"fails={fail}")
        print(f"  R' seed-std(mean/p90) = {rprime_std:.4f}/{rprime_std_p90:.4f}  "
              f"noise/δ = {rprime_std/delta:.2f}x  k≈{k_needed}  δ_recal≈{delta_recal}")
        print(f"  成对差 std = {pair_std:.4f}  符号稳定率 = {sign_stable:.2f}  "
              f"CLS 区分度 std = {cls_spread_std:.4f}  SNR = {snr:.2f}")
        print(f"  判定: {verdict}")

    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n分析 → {OUTDIR/'analysis.json'}")


def _task_n_eval(recs, ds, tid) -> int:
    for r in recs:
        if r["dataset"] == ds and r["task_id"] == tid and r.get("n_eval"):
            return int(r["n_eval"])
    return 0


def _verdict(ds, std, delta, snr, sign_stable) -> str:
    if std != std:
        return "数据不足"
    reliable_noise = std <= delta
    good_snr = snr == snr and snr >= 1.0
    stable = sign_stable == sign_stable and sign_stable >= 0.7
    if reliable_noise and good_snr and stable:
        return "可靠：噪声≤δ 且 CLS 可分（可用于造 label）"
    if not good_snr:
        return "不可靠：CLS 区分度被噪声淹没（SNR<1，短序列/小样本失效，剔除或降权）"
    if not reliable_noise:
        return f"需多种子平均：噪声>{'%.0f'%(std/delta)}×δ，须 k 折降噪或重标定 δ"
    return "边界：符号稳定率偏低，谨慎"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="跑 R' 主循环")
    ap.add_argument("--diagnose-augs", action="store_true", help="增强生效性诊断（CPU）")
    ap.add_argument("--analyze", action="store_true", help="分析已落盘缓存")
    ap.add_argument("--smoke", action="store_true", help="每数据集 1task×2cls×1seed 低 iters 验链路")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.diagnose_augs:
        diagnose_augs()
    if args.run:
        run_probe(smoke=args.smoke, resume=args.resume, device=args.device)
    if args.analyze:
        analyze()
    if not (args.diagnose_augs or args.run or args.analyze):
        ap.error("至少指定 --run / --diagnose-augs / --analyze 之一")


if __name__ == "__main__":
    main()
