"""排查 B：encoder 架构消融——随机初始化基线（iters=0）由哪些架构维度支撑。

背景（排查 A，2026-08-13）：autocls/encoders/dilated_cnn.py 与 TS2Vec 官方实现
（github.com/zhihanyue/ts2vec models/{dilated_conv,encoder}.py）逐项对照确认：
eval 前向路径功能等价（两边都无自定义初始化=PyTorch 默认、结构逐行同构、
分类协议同为时间维 max-pool + SVM-RBF）；差异全在训练期（binomial mask 与
repr_dropout 有意移除、无 SWA 平均、SVM C 网格缩小）。故"基线贴顶"若与
encoder 有关，只能是**架构本身的性质**（深度/宽度/pooling/残差），不是实现偏差。

本脚本一次一因子扫这些维度，测 iters=0 随机初始化基线（CLS 完全不参与）：
- UCI-HAR 全量 6 类 9 通道，跨受试者 val（留出组 VAL_HOLDOUT_SUBJECTS），
  参照 0.8911（subject_leak_diag 3 种子均值）。只碰 train 侧，test 触碰数不变。
- Handwriting（源池对照、headroom 大户 best_gain +0.397）全量 26 类分层 70/30。
  对照语义：拉低 UCI-HAR 基线的配置若把 Handwriting 拉向 chance(≈0.038)，
  是把 encoder 削弱到什么都表示不了，不是"修好了过强"。

判读（先于数据写死，用户拍板）：
- 某配置把 UCI-HAR 跨受试者基线从 0.891 拉到 <0.75 → 找到"encoder 过强"的
  具体架构来源，值得排查 C（该配置下 GGS 有无增益）；
- 全部配置 >0.85 → encoder 架构不是贴顶主因，"数据集×评估栈天花板"结论坐实。

口径隔离（铁律 3）：消融行独立落盘 results/encoder_ablation/ablation_cache.jsonl，
键 = (dataset, config, seed)，绝不写入任何历史 R' 缓存。
flatten pooling（无池化直接展平）不测：HAR T=128 → 40960 维 RBF 核，单次
kernel 计算量约为 320 维的 128×，且展平不具平移不变性、与协议 A 不可比。

用法：
    python scripts/diag_encoder_ablation.py --run --resume
    python scripts/diag_encoder_ablation.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402

from autocls.data.source_pool import load_registry, load_source_dataset  # noqa: E402
from autocls.downstream.linear_eval import fit_svm  # noqa: E402
from autocls.encoders.dilated_cnn import ConvBlock, DilatedCNNEncoder  # noqa: E402
from autocls.labeling import device_label, split_task_train_eval  # noqa: E402

_tz = importlib.import_module("scripts.5_target_zero_shot")

OUTDIR = ROOT / "results" / "encoder_ablation"
CACHE = OUTDIR / "ablation_cache.jsonl"
N_SEEDS = 5
HW_EVAL_RATIO = 0.3          # 项目通用任务内划分比（configs/labels.yaml: eval_ratio）
HW_SPLIT_SEED = 0
HAR_REF, HW_CHANCE = 0.8911, 1 / 26

# 架构基准 = configs/encoder.yaml 的 dilated_cnn 默认（AutoCL B.3 / TS2Vec 同款）。
BASE_ARCH = {"depth": 10, "output_dims": 320, "hidden_dims": 64, "kernel_size": 3,
             "residual": True}

# 一次一因子网格（base 首位；测试保证每项恰偏离 base 一个因子）。
CONFIGS = [
    {"name": "base",     **BASE_ARCH, "pooling": "max"},
    {"name": "depth6",   **{**BASE_ARCH, "depth": 6}, "pooling": "max"},
    {"name": "depth3",   **{**BASE_ARCH, "depth": 3}, "pooling": "max"},
    {"name": "depth1",   **{**BASE_ARCH, "depth": 1}, "pooling": "max"},
    {"name": "dims128",  **{**BASE_ARCH, "output_dims": 128}, "pooling": "max"},
    {"name": "dims64",   **{**BASE_ARCH, "output_dims": 64}, "pooling": "max"},
    {"name": "dims32",   **{**BASE_ARCH, "output_dims": 32}, "pooling": "max"},
    {"name": "pool_mean", **BASE_ARCH, "pooling": "mean"},
    {"name": "pool_last", **BASE_ARCH, "pooling": "last"},
    {"name": "nores",    **{**BASE_ARCH, "residual": False}, "pooling": "max"},
]


def _no_res_forward(self: ConvBlock, x: torch.Tensor) -> torch.Tensor:
    """ConvBlock 去残差路径：gelu→conv1→gelu→conv2（不加 residual/projector）。"""
    x = F.gelu(x)
    x = self.conv1(x)
    x = F.gelu(x)
    return self.conv2(x)


def build_variant(input_dims: int, arch: dict) -> DilatedCNNEncoder:
    """按消融配置构造 encoder 变体。

    residual=False 只 monkeypatch 各 ConvBlock 实例的 forward，模块树不动——
    与同种子 base 逐位同权重（RNG 消耗序列一致），保证单因子对照。
    """
    enc = DilatedCNNEncoder(
        input_dims=input_dims, output_dims=arch["output_dims"],
        hidden_dims=arch["hidden_dims"], depth=arch["depth"],
        kernel_size=arch["kernel_size"])
    if not arch.get("residual", True):
        for m in enc.feature_extractor:
            if isinstance(m, ConvBlock):
                m.forward = types.MethodType(_no_res_forward, m)
    return enc


@torch.no_grad()
def encode_multi_pooled(
    encoder: torch.nn.Module, X: np.ndarray, poolings: tuple[str, ...],
    device: str = "cuda", batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """(N,T,C) → {pooling: (N,d)}：一次前向同时导出多路时间维 pooling。

    max 路与 linear_eval.encode_features 逐位一致（协议 A）；mean/last 为消融维度。
    """
    encoder = encoder.to(device)
    encoder.eval()
    out: dict[str, list[np.ndarray]] = {p: [] for p in poolings}
    for start in range(0, X.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(
            X[start:start + batch_size], dtype=np.float32))
        z = encoder(x.to(device))  # (B, T, d)
        for p in poolings:
            if p == "max":
                f = z.max(dim=1).values
            elif p == "mean":
                f = z.mean(dim=1)
            elif p == "last":
                f = z[:, -1]
            else:
                raise ValueError(f"未知 pooling: {p}")
            out[p].append(f.cpu().numpy())
    return {p: np.concatenate(v, axis=0) for p, v in out.items()}


def load_datasets() -> dict[str, dict]:
    """两个测床：UCI-HAR（跨受试者 val）+ Handwriting（分层 70/30 对照）。"""
    X_har, y_har = _tz.load_target_train()
    subjects = _tz.load_train_subjects(len(y_har))
    fit_h, val_h = _tz.subject_val_split(subjects)

    reg = load_registry()
    X_hw, y_hw = load_source_dataset("Handwriting", reg)
    fit_w, val_w = split_task_train_eval(
        y_hw, "classification", eval_ratio=HW_EVAL_RATIO, seed=HW_SPLIT_SEED)
    return {
        "UCI-HAR": {"X": X_har, "y": y_har, "fit": fit_h, "val": val_h,
                    "split": "subject_disjoint", "ref": HAR_REF},
        "Handwriting": {"X": X_hw, "y": y_hw, "fit": fit_w, "val": val_w,
                        "split": f"stratified{HW_EVAL_RATIO}", "ref": None},
    }


def _read_rows() -> list[dict]:
    if not CACHE.exists():
        return []
    rows = []
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run(device: str, resume: bool, n_seeds: int) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_datasets()
    done = {(r["dataset"], r["config"], r["seed"]) for r in _read_rows()}

    # 按 (arch, seed) 归组：同权重一次前向导出全部所需 pooling（省编码时间，
    # 且 pooling 对照严格配对同一份随机权重）。
    arch_names: dict[str, list[dict]] = {}
    for cfg in CONFIGS:
        arch_key = json.dumps({k: cfg[k] for k in BASE_ARCH}, sort_keys=True)
        arch_names.setdefault(arch_key, []).append(cfg)

    plan = [(ds, ak, s) for ds in data for ak in arch_names for s in range(n_seeds)]
    total = sum(len(cfgs) for cfgs in arch_names.values()) * len(data) * n_seeds
    print(f"设备 {device} | iters=0（不训练）| {len(CONFIGS)} 配置 × {n_seeds} 种子 × "
          f"{len(data)} 数据集 = {total} 行，已缓存 {len(done)}", flush=True)

    t0, n_run = time.time(), 0
    with parallel_backend("threading", n_jobs=1):  # loky spawn 教训（勿删）
        for ds_name, arch_key, seed in plan:
            arch = json.loads(arch_key)
            cfgs = [c for c in arch_names[arch_key]
                    if not (resume and (ds_name, c["name"], seed) in done)]
            if not cfgs:
                continue
            d = data[ds_name]
            torch.manual_seed(seed)  # 与 5_target_zero_shot 全量基线同派生方式
            enc = build_variant(d["X"].shape[-1], arch)
            poolings = tuple(dict.fromkeys(c["pooling"] for c in cfgs))
            t_enc = time.time()
            f_fit = encode_multi_pooled(enc, d["X"][d["fit"]], poolings, device=device)
            f_val = encode_multi_pooled(enc, d["X"][d["val"]], poolings, device=device)
            enc_s = time.time() - t_enc
            for cfg in cfgs:
                t1 = time.time()
                clf = fit_svm(f_fit[cfg["pooling"]], d["y"][d["fit"]], seed=seed)
                acc = float(np.mean(
                    clf.predict(f_val[cfg["pooling"]]) == d["y"][d["val"]]))
                row = {"dataset": ds_name, "config": cfg["name"], "seed": seed,
                       "iters": 0, "val_acc": acc, "svm_c": float(clf.C),
                       "pooling": cfg["pooling"],
                       "arch": {k: cfg[k] for k in BASE_ARCH},
                       "split": d["split"], "n_fit": int(len(d["fit"])),
                       "n_val": int(len(d["val"])),
                       "seconds": round(enc_s / len(cfgs) + time.time() - t1, 1),
                       "device_name": device_label(device)}
                with CACHE.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_run += 1
                print(f"[{n_run}] {ds_name}/{cfg['name']} s{seed} "
                      f"acc={acc:.4f} ({row['seconds']}s)", flush=True)
    print(f"\n跑完 {n_run} 行，墙钟 {(time.time() - t0) / 60:.1f} min → {CACHE}",
          flush=True)


def analyze() -> None:
    rows = _read_rows()
    if not rows:
        sys.exit("无缓存行：先 --run")
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by_key.setdefault((r["dataset"], r["config"]), []).append(r["val_acc"])

    summary: dict[str, dict] = {}
    order = [c["name"] for c in CONFIGS]
    for ds in ("UCI-HAR", "Handwriting"):
        base = by_key.get((ds, "base"))
        print(f"\n{'=' * 74}\n{ds}（iters=0 随机初始化基线，"
              f"{'参照 0.8911 跨受试者' if ds == 'UCI-HAR' else 'chance≈0.038'}）\n"
              f"{'=' * 74}")
        print(f"{'配置':<11}{'n':>3}{'均值':>9}{'±std':>8}{'Δvs base':>10}")
        for name in order:
            accs = by_key.get((ds, name))
            if not accs:
                continue
            mean = statistics.fmean(accs)
            std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            delta = mean - statistics.fmean(base) if base else float("nan")
            summary.setdefault(ds, {})[name] = {
                "n": len(accs), "mean": round(mean, 4), "std": round(std, 4),
                "delta_vs_base": round(delta, 4)}
            print(f"{name:<11}{len(accs):>3}{mean:>9.4f}{std:>8.4f}{delta:>+10.4f}")

    har = summary.get("UCI-HAR", {})
    below = [n for n, v in har.items() if v["mean"] < 0.75]
    verdict = ("found_weakening_config" if below else
               "architecture_not_the_cause" if all(v["mean"] > 0.85 for v in har.values())
               else "gray_zone")
    out = {"criterion": {"har_ref": HAR_REF, "weak_threshold": 0.75,
                         "solid_threshold": 0.85},
           "verdict": verdict, "below_075": below, "summary": summary}
    (OUTDIR / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n判定: {verdict}"
          + (f"（<0.75 的配置: {below}）" if below else "")
          + f"\n分析 → {OUTDIR / 'summary.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.run:
        run(args.device, args.resume, args.seeds)
    if args.analyze:
        analyze()
    if not (args.run or args.analyze):
        ap.error("至少指定 --run / --analyze 之一")


if __name__ == "__main__":
    main()
