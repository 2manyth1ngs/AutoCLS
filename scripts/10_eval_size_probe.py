"""评估集规模探针（协议 results/eval_size_probe/PROTOCOL.md，先于数据写死）。

检验"R' 噪声 ∝ 1/√n_eval"（跨数据集线索：HW/UM 的 med 与 P95 噪声比 1.45/1.47
≈ √(307/140)=1.48，但只有两个点且混着域/L/iters）。本探针在数据集内直接测缩放
指数：与 δ=0.110 标定逐字同口径重跑 HW@60 前 2 任务 × 10 shared CLS × 3 种子
（canonical + noise1/2），评估端额外落盘**逐样本判对向量**——嵌套评估子集的
|ΔR'| 分布全部离线可算，一次 GPU 跑完。副产品：跨种子逐点翻转率 q、翻转方向
不平衡（区分"计数噪声"与"系统性表征发散"）。

用法（幂等、--resume 贯穿）：
    python scripts/10_eval_size_probe.py --run --resume
    python scripts/10_eval_size_probe.py --analyze
"""
from __future__ import annotations

import os

# 造 label CPU-bound 教训：不设 OMP=1 慢 4.2×（须在 numpy/torch 导入前生效）
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import dataclasses
import importlib
import json
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
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

from autocls.labeling import (  # noqa: E402
    device_label, load_probe_cls, run_seed_for, split_task_train_eval,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run, RprimeOOM, is_oom_error  # noqa: E402

_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")

DEFAULT_OUTDIR = ROOT / "results" / "eval_size_probe"
WORKPOINT_DIR = ROOT / "results" / "labels_workpoint"
DATASET = "Handwriting"
REPS = (0, 1, 2)                       # canonical + 2c noise1/2（δ=0.110 同口径）
FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.125)
FIT_FRACTIONS = (1.0, 0.75, 0.5, 0.25)  # 1/8 档量子(1/16≈0.06)支配中位，只报不拟合
N_SUBSET_DRAWS = 200
SUBSET_RNG_SEED = 20260828
UM_REF = {"n_eval": 307, "med": 0.020, "p95": 0.075}  # labels_workpoint 发布标定


# ---------------------------------------------------------------- 采集
def seeds_for(task_id: str, cls_key_: str, rep: int) -> tuple[int, int]:
    """与 2c 噪声探针逐字同口径：rep0=canonical；rep1/2=噪声种子（split 恒 canonical）。"""
    if rep == 0:
        return _c2.canonical_seeds(task_id, cls_key_)
    return (run_seed_for(task_id, f"{cls_key_}:noise{rep}"),
            run_seed_for(task_id, "train_eval_split"))


def eval_cell(task, cls_cfg, X, y, *, weight_seed: int, split_seed: int,
              iters: int, eval_ratio: float, batch_size: int, lr: float,
              device) -> dict:
    """evaluate_rprime 同序列（materialize→split→manual_seed→build→pretrain→
    encode→fit_svm→predict），额外返回逐样本判对向量（升序 ev_idx 对齐；split
    canonical 故同任务所有 (cls, seed) 的 eval 集与顺序逐位相同）。R' 必须与
    labels_workpoint 既有值逐位复现（--analyze 的有效性闸核对）。"""
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.data.task_slicer import materialize_task
    from autocls.downstream.linear_eval import encode_features, fit_svm
    from autocls.encoders.registry import build_encoder

    Xt, yt = materialize_task(task, X, y)
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=eval_ratio, seed=split_seed)
    rec: dict = {"seconds": None, "error": None, "r_prime": float("nan")}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(weight_seed)
        encoder = build_encoder(input_dims=Xt.shape[-1])
        pretrain_encoder(encoder, cls_cfg, Xt[tr_idx], max_iters=iters,
                         batch_size=batch_size, lr=lr, device=device,
                         seed=weight_seed)
        train_feats = encode_features(encoder, Xt[tr_idx], device=device)
        eval_feats = encode_features(encoder, Xt[ev_idx], device=device)
        clf = fit_svm(train_feats, yt[tr_idx], seed=weight_seed)
        correct = (clf.predict(eval_feats) == yt[ev_idx]).astype(int)
        rec.update({"r_prime": float(correct.mean()), "acc": float(correct.mean()),
                    "svm_c": float(clf.C),
                    "correct": "".join(str(int(c)) for c in correct)})
    except Exception as e:  # noqa: BLE001 —— 训练发散记 NaN；OOM 抛出不落结论
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    rec.update({"n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)),
                "weight_seed": int(weight_seed), "split_seed": int(split_seed),
                "device_name": device_label(device)})
    return rec


def run(outdir: Path, per_ds: list, iters_map: dict, shared_cls: list,
        train_cfg: dict, labels_cfg: dict, device: str, resume: bool) -> None:
    cache_path = outdir / "probe_cache.jsonl"
    done = {(r["task_id"], r["cls_key"], r["iters"], r["seed"])
            for r in _c2.read_jsonl(cache_path) if r.get("r_prime") is not None}
    work = [(name, X, y, t, c, k, rep, iters_map[name])
            for name, X, y, tasks in per_ds if name == DATASET
            for t in tasks[:_c2.NOISE_TASKS]
            for c, k in shared_cls
            for rep in REPS]
    todo = [w for w in work
            if not (resume and (w[3].task_id, w[5], w[7], w[6]) in done)]
    print(f"评估集规模探针: 计划 {len(work)} 条，待跑 {len(todo)}", flush=True)
    t0 = time.time()
    # threading 后端：fit_svm 的 GridSearchCV 逐条 spawn loky 占 99% 耗时（长期教训，勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, (name, X, y, t, c, k, rep, iters) in enumerate(todo):
            w_seed, sp_seed = seeds_for(t.task_id, k, rep)

            def make_row(rec, _n=name, _t=t, _k=k, _c=c, _rep=rep, _it=iters):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "seed": _rep, "iters": _it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: eval_cell(t, c, X, y, weight_seed=w_seed,
                                  split_seed=sp_seed, iters=iters,
                                  eval_ratio=labels_cfg["eval_ratio"],
                                  batch_size=train_cfg["batch_size"],
                                  lr=train_cfg["lr"], device=device),
                make_row, cache_path=cache_path, inflight=outdir / "inflight.txt",
                task_id=t.task_id, cls_key=k, iters=iters,
                on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                context={"dataset": name, "noise_rep": rep},
                log=lambda m: print(f"[evsize {i + 1}/{len(todo)}] {m}", flush=True))
            if full is None:
                continue
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[evsize {i + 1}/{len(todo)}] {t.task_id} {k[:8]} rep{rep} "
                  f"R'={full['r_prime']} ({full['seconds']}s) ETA {eta / 60:.1f}min",
                  flush=True)
    print(f"run 完成，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 纯分析函数
def _p95(vals) -> float:
    """与 2c calibrate 同式：最近秩不插值。"""
    ds = sorted(vals)
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


def parse_correct(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int8) - ord("0")


def flip_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """跨种子逐点翻转结构：q=判对不一致率；net=|Δacc|；方向不平衡
    |f₊−f₋|/(f₊+f₋)（1=单向系统发散，0=对称计数噪声）。"""
    d = a.astype(np.int16) - b.astype(np.int16)
    fp, fm = int((d > 0).sum()), int((d < 0).sum())
    return {"q": float((d != 0).mean()), "net": abs(float(d.mean())),
            "flips_pos": fp, "flips_neg": fm,
            "imbalance": abs(fp - fm) / (fp + fm) if fp + fm else None}


def subset_noise_values(vectors_by_cls: dict, n_e: int, fractions, n_draws: int,
                        rng: np.random.Generator) -> dict:
    """{f: {cls: |Δacc| 值数组}}。同一 f 的子集抽签全 cls 共用（common random
    numbers）；f=1 只有全集一份。counterfactual 语义：把每个子集当作"评估集本来
    就这么大"的一次实现，跨 cell × 抽签合并成该规模下的噪声分布。"""
    out: dict = {}
    for f in fractions:
        n_f = int(round(f * n_e))
        draws = ([np.arange(n_e)] if f >= 1.0 else
                 [rng.choice(n_e, size=n_f, replace=False) for _ in range(n_draws)])
        per: dict = {}
        for k, reps in vectors_by_cls.items():
            vals = []
            for i, j in combinations(range(len(reps)), 2):
                d = reps[i].astype(np.int16) - reps[j].astype(np.int16)
                vals += [abs(float(d[S].mean())) for S in draws]
            per[k] = np.array(vals)
        out[f] = per
    return out


def fit_beta(points: list[tuple[float, float]]) -> float:
    """log med = a − β·log n 的最小二乘斜率取负。points=[(n, med), ...]。"""
    x = np.log([p[0] for p in points])
    y = np.log([p[1] for p in points])
    slope = np.polyfit(x, y, 1)[0]
    return float(-slope)


def scaling_verdict(beta: float) -> str:
    """主判据档位（协议 §4）：[0.35,0.65] 坐实 / [0.2,0.35) 混合拍板 /
    <0.2 推翻 / >0.65 异常（量化伪影排查后拍板）。"""
    if beta > 0.65:
        return "anomalous"
    if beta >= 0.35:
        return "confirmed"
    if beta >= 0.20:
        return "mixed"
    return "refuted"


def bitwise_check(rows: list[dict]) -> dict:
    """60 条 R' ↔ labels_workpoint 既有值逐位比对（有效性闸：同种子同口径必须
    复现；长进程上下文底噪 ≤0.02 为已知豁免带）。"""
    ref: dict = {}
    for r in _c2.read_jsonl(WORKPOINT_DIR / "rprime_cache.jsonl"):
        if r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], r["iters"], 0)] = float(r["r_prime"])
    for r in _c2.read_jsonl(WORKPOINT_DIR / "noise_probe.jsonl"):
        if r.get("r_prime") is not None:
            ref[(r["task_id"], r["cls_key"], r["iters"], r["seed"])] = float(r["r_prime"])
    n_match = n_total = n_missing = 0
    max_diff, mismatches = 0.0, []
    for r in rows:
        if r.get("r_prime") is None:
            continue
        n_total += 1
        key = (r["task_id"], r["cls_key"], r["iters"], r["seed"])
        if key not in ref:
            n_missing += 1
            continue
        d = abs(float(r["r_prime"]) - ref[key])
        max_diff = max(max_diff, d)
        if d == 0.0:
            n_match += 1
        else:
            mismatches.append({"key": list(key), "new": r["r_prime"], "ref": ref[key]})
    return {"n_total": n_total, "n_match": n_match, "n_missing_ref": n_missing,
            "max_abs_diff": max_diff, "mismatches": mismatches[:10],
            "passed": n_total == 60 and n_match >= 48 and max_diff <= 0.02}


# ---------------------------------------------------------------- 分析主流程
def analyze(outdir: Path, n_shared: int) -> dict:
    rows = _c2.read_jsonl(outdir / "probe_cache.jsonl")
    shared_keys = [k for _, k in load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]]
    cells: dict = defaultdict(dict)
    for r in rows:
        if r.get("r_prime") is not None and r.get("correct"):
            cells[(r["task_id"], r["cls_key"])][r["seed"]] = parse_correct(r["correct"])
    task_ids = sorted({tid for tid, _ in cells})
    res: dict = {"protocol": "results/eval_size_probe/PROTOCOL.md"}

    # ---- 有效性闸（协议 §5）----
    keys = [(r["task_id"], r["cls_key"], r["iters"], r["seed"]) for r in rows]
    n_eval_of = {tid: {len(v) for (t, _), sv in cells.items() if t == tid
                       for v in sv.values()} for tid in task_ids}
    bw = bitwise_check(rows)
    validity = {
        "n_rows": len(rows),
        "n_finite": sum(1 for r in rows if r.get("r_prime") is not None),
        "n_dup_keys": len(keys) - len(set(keys)),
        "n_tasks": len(task_ids),
        "cells_complete": all(len(sv) == 3 for sv in cells.values()),
        "vector_len_consistent": {t: sorted(s) for t, s in n_eval_of.items()},
        "bitwise": bw,
        "devices": sorted({r.get("device_name", "?") for r in rows}),
    }
    validity["passed"] = (
        validity["n_finite"] == 60 and validity["n_tasks"] == 2
        and validity["cells_complete"]
        and all(len(s) == 1 for s in n_eval_of.values()) and bw["passed"])
    res["validity"] = validity

    # ---- 翻转结构（协议 §6：q、方向不平衡）----
    flips = []
    for (tid, k), sv in cells.items():
        for i, j in combinations(sorted(sv), 2):
            flips.append({"task_id": tid, "cls_key": k, "seeds": [i, j],
                          **flip_stats(sv[i], sv[j])})
    qs = [f["q"] for f in flips]
    imb = [f["imbalance"] for f in flips if f["imbalance"] is not None]
    nets = sorted(flips, key=lambda f: -f["net"])
    res["flip_structure"] = {
        "q_median": statistics.median(qs), "q_p25": float(np.percentile(qs, 25)),
        "q_p75": float(np.percentile(qs, 75)),
        "imbalance_median": statistics.median(imb),
        "imbalance_top6_by_net": [
            {k: f[k] for k in ("task_id", "cls_key", "net", "q", "imbalance")}
            for f in nets[:6]],
        "per_pair": flips,
    }

    # ---- 子集缩放（协议 §3/§4 主判据）----
    rng = np.random.default_rng(SUBSET_RNG_SEED)
    per_task_vals: dict = {}
    for tid in task_ids:
        n_e = next(iter(n_eval_of[tid]))
        by_cls = {k: [sv[s] for s in sorted(sv)]
                  for (t, k), sv in cells.items() if t == tid}
        per_task_vals[tid] = (n_e, subset_noise_values(
            by_cls, n_e, FRACTIONS, N_SUBSET_DRAWS, rng))

    curve = []
    for tid, (n_e, vals) in per_task_vals.items():
        for f in FRACTIONS:
            pool = np.concatenate(list(vals[f].values()))
            curve.append({"task_id": tid, "fraction": f,
                          "n_f": int(round(f * n_e)),
                          "median": float(np.median(pool)),
                          "p95": float(_p95(pool.tolist())), "n_vals": len(pool)})
    res["scaling_curve"] = curve

    fit_pts_med = [(c["n_f"], c["median"]) for c in curve
                   if c["fraction"] in FIT_FRACTIONS]
    fit_pts_p95 = [(c["n_f"], c["p95"]) for c in curve
                   if c["fraction"] in FIT_FRACTIONS]
    beta_med = fit_beta(fit_pts_med)
    beta_p95 = fit_beta(fit_pts_p95)
    beta_per_task = {
        tid: fit_beta([(c["n_f"], c["median"]) for c in curve
                       if c["task_id"] == tid and c["fraction"] in FIT_FRACTIONS])
        for tid in task_ids}

    # bootstrap（cls 为可交换单元，500 次重采样，联合 8 点拟合）
    boot = []
    brng = np.random.default_rng(SUBSET_RNG_SEED + 1)
    for _ in range(500):
        pick = [shared_keys[i] for i in brng.integers(0, len(shared_keys),
                                                      len(shared_keys))]
        pts = []
        for tid, (n_e, vals) in per_task_vals.items():
            for f in FIT_FRACTIONS:
                pool = np.concatenate([vals[f][k] for k in pick if k in vals[f]])
                pts.append((int(round(f * n_e)), float(np.median(pool))))
        boot.append(fit_beta(pts))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # UM 外推（诊断，不判定）：联合拟合截距外推到 n=307
    x = np.log([p[0] for p in fit_pts_med])
    y = np.log([p[1] for p in fit_pts_med])
    slope, intercept = np.polyfit(x, y, 1)
    pred_um_med = float(np.exp(intercept + slope * np.log(UM_REF["n_eval"])))

    verdict = scaling_verdict(beta_med)
    res["verdict"] = {
        "beta_median": beta_med, "beta_median_ci95": ci,
        "beta_p95": beta_p95, "beta_per_task": beta_per_task,
        "band": verdict,
        "um_extrapolation": {"predicted_med_at_307": pred_um_med,
                             "published_med": UM_REF["med"],
                             "ratio": pred_um_med / UM_REF["med"]},
    }

    # ---- 预测对账（协议 §7）----
    q_med = res["flip_structure"]["q_median"]
    res["predictions"] = {
        "P1_beta_med_in_band": {"value": beta_med, "hit": 0.35 <= beta_med <= 0.65},
        "P2_tail_flatter": {"beta_p95": beta_p95, "beta_med": beta_med,
                            "hit": beta_p95 <= beta_med - 0.15},
        "P3_q_median_in_band": {"value": q_med, "hit": 0.06 <= q_med <= 0.18},
        "P4_um_extrapolation_30pct": {
            "ratio": res["verdict"]["um_extrapolation"]["ratio"],
            "hit": 0.7 <= res["verdict"]["um_extrapolation"]["ratio"] <= 1.3},
    }

    (outdir / "analysis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ---- 控制台判读 ----
    v = validity
    print(f"\n[有效性] 行 {v['n_rows']}/finite {v['n_finite']}/60 "
          f"dup={v['n_dup_keys']} 逐位复现 {bw['n_match']}/{bw['n_total']} "
          f"(max|Δ|={bw['max_abs_diff']:.4f}) devices={v['devices']} "
          f"→ {'PASS' if v['passed'] else 'FAIL'}", flush=True)
    print(f"[翻转结构] q med={q_med:.3f} [p25 {res['flip_structure']['q_p25']:.3f}, "
          f"p75 {res['flip_structure']['q_p75']:.3f}] | 方向不平衡 med="
          f"{res['flip_structure']['imbalance_median']:.3f}", flush=True)
    for f in nets[:6]:
        print(f"  [大 net 对] {f['task_id']} {f['cls_key'][:8]} s{f['seeds']} "
              f"net={f['net']:.3f} q={f['q']:.3f} 不平衡={f['imbalance']:.2f}",
              flush=True)
    for c in curve:
        tag = "" if c["fraction"] in FIT_FRACTIONS else "（只报不拟合）"
        print(f"[缩放 {c['task_id']} f={c['fraction']}] n={c['n_f']} "
              f"med={c['median']:.4f} p95={c['p95']:.4f}{tag}", flush=True)
    print(f"\n[主判据] β_med={beta_med:.3f} CI95=[{ci[0]:.3f},{ci[1]:.3f}] "
          f"per-task={ {t: round(b, 3) for t, b in beta_per_task.items()} } "
          f"| β_p95={beta_p95:.3f} → {verdict.upper()}", flush=True)
    ue = res["verdict"]["um_extrapolation"]
    print(f"[UM 外推] 预测 med@307={ue['predicted_med_at_307']:.4f} vs 发布 "
          f"{ue['published_med']:.3f}（比 {ue['ratio']:.2f}）", flush=True)
    for name, p in res["predictions"].items():
        print(f"[预测 {name}] {'✓' if p['hit'] else '✗'} {p}", flush=True)
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

    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(
        (ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    n_shared = labels_cfg["route_a"]["n_shared_cls"]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.run:
        per_ds = _c2.load_pool(labels_cfg)
        shared_cls = load_probe_cls(_c2.PROBE_CATALOG, n_shared)[0]
        run(args.outdir, per_ds, iters_map, shared_cls, train_cfg, labels_cfg,
            args.device, args.resume)
    if args.analyze:
        analyze(args.outdir, n_shared)


if __name__ == "__main__":
    main()
