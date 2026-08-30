"""宽任务面板 @290：任务级两道闸（协议 results/wide_panel/PROTOCOL.md）。

序贯设计：迷你期 = UM+CT（本地过夜）→ 期中止损检查（只止损不宣胜）→
二期补 UWave/PS/HW → 全面板判定。闸 1 = T2 置换存在性（scripts/21 机器）；
闸 2 = 试点 B 赢家单发（P|三态核插值 vs 等权锚，禁改）。

    python scripts/24_wide_panel.py --run --resume [--datasets ...] [--max-tasks N] [--max-cls N]
    python scripts/24_wide_panel.py --analyze [--outdir ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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

fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")
gp = importlib.import_module("scripts.22_gate2_pilot")
tp = importlib.import_module("scripts.21_tasklevel_precheck")
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.baseline_probe import baseline_seed_specs, inert_cls  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.labeling import (  # noqa: E402
    evaluate_rprime, load_probe_cls, load_rprime_cache, run_seed_for,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402

OUTDIR_DEFAULT = ROOT / "results" / "wide_panel"
# 面板组成（协议 §1 写死）：selection = 试点选拔数据集，闸 2 主池排除
PANEL = {
    "UniMiB-SHAR": {"registry": "configs/candidate_pool.yaml", "role": "selection"},
    "CharacterTrajectories": {"registry": None, "role": "primary"},
    "UWaveGestureLibrary": {"registry": None, "role": "primary"},
    "PhonemeSpectra": {"registry": None, "role": "primary"},
    "Handwriting": {"registry": None, "role": "primary"},
}
N_TASKS_PANEL = 8
MINI = ("UniMiB-SHAR", "CharacterTrajectories")
ITERS = 290
N_BASE_SEEDS = 5
K_ANCHOR, N_REPS = 3, 200          # 闸 2 赢家参数（试点 B 逐字，禁改）
GATE1_ALPHA = 0.01                  # Bonferroni ×5（协议 §3）
EXISTING_TRAINED = (
    ROOT / "results" / "stage1_rerun_20260804" / "results" / "labels_route_a"
    / "rprime_cache.jsonl",
    ROOT / "results" / "baseline_probe" / "trained_fill.jsonl",
)
EXISTING_BASELINES = (
    ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl",
    ROOT / "results" / "iters_workpoint" / "candidates" / "baseline_cache.jsonl",
    ROOT / "results" / "iters_workpoint" / "uea" / "baseline_cache.jsonl",
)


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def interim_futility(*, p_um, dir_um, dir_ct) -> bool:
    """期中止损（协议 §2 写死）：UM 闸 1 口径迁移失败，或闸 2 方向双负。"""
    if p_um is not None and p_um >= 0.10:
        return True
    if dir_um is not None and dir_ct is not None and dir_um < 0 and dir_ct < 0:
        return True
    return False


def primary_pool() -> list:
    return [d for d, s in PANEL.items() if s["role"] == "primary"]


def baseline_done_keys(rows: list, *, split_of) -> set:
    """既有基线的 (tid, seed) 有效键：iters=0、canonical 划分、r' 有限。"""
    done = set()
    for r in rows:
        tid = r.get("task_id")
        if (tid is None or r.get("iters") != 0 or r.get("r_prime") is None
                or r.get("split_seed") != split_of(tid)):
            continue
        done.add((tid, r.get("seed")))
    return done


# ---------------------------------------------------------------- 数据装配
def read_rows(paths) -> list:
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            out += [json.loads(x) for x in
                    p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def load_panel_data(datasets: list, max_tasks: int):
    per_ds = []
    for name in datasets:
        spec = PANEL[name]
        reg = (load_registry(ROOT / spec["registry"]) if spec["registry"]
               else load_registry())
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _, tasks = slice_source_tasks(entry, reg, y)
        acts = tasks[:max_tasks]
        print(f"[{name}] X={X.shape} → 面板任务 {len(acts)}", flush=True)
        per_ds.append((name, X, y, acts))
    return per_ds


# ---------------------------------------------------------------- 阶段 1：run
def cmd_run(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base_cache = outdir / "baseline_cache.jsonl"
    cell_cache = outdir / "rprime_cache.jsonl"
    inflight = outdir / "inflight.txt"
    oom_log = outdir / "oom_failures.jsonl"
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml")
                               .read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                                .read_text(encoding="utf-8"))
    datasets = args.datasets or list(PANEL)
    per_ds = load_panel_data(datasets, args.max_tasks)
    shared_cls = load_probe_cls(fl.PROBE_CATALOG, 10)[0][: args.max_cls]

    # ---- 基线补测（既有缓存 + 本面板缓存的 (tid, seed) 有效键跳过）----
    split_of = lambda tid: run_seed_for(tid, "train_eval_split")  # noqa: E731
    done_b = baseline_done_keys(
        read_rows(list(EXISTING_BASELINES) + [base_cache]), split_of=split_of)
    from autocls.labeling import cls_key as _ck
    cls0 = inert_cls()
    k0 = _ck(cls0)
    todo_b = [(n, X, y, t, s) for n, X, y, tasks in per_ds for t in tasks
              for s in range(N_BASE_SEEDS) if (t.task_id, s) not in done_b]
    print(f"基线：待补 {len(todo_b)} 条（既有键 {len(done_b)} 复用）", flush=True)
    with parallel_backend("threading", n_jobs=1):
        for i, (name, X, y, t, s) in enumerate(todo_b):
            w_seed, sp_seed = baseline_seed_specs(t.task_id, s)

            def mk(rec, _n=name, _t=t, _s=s):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": k0, "seed": _s, "iters": 0, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            guarded_run(lambda: evaluate_rprime(t, cls0, X, y, weight_seed=w_seed,
                                                split_seed=sp_seed, iters=0,
                                                device=args.device),
                        mk, cache_path=base_cache, inflight=inflight,
                        task_id=t.task_id, cls_key=k0, iters=0,
                        on_oom=ON_OOM_SKIP, oom_log=oom_log,
                        context={"dataset": name, "seed": s},
                        log=lambda m: print(f"[base {i + 1}/{len(todo_b)}] {m}",
                                            flush=True))

    # ---- @290 格（既有 stage1 k1 canonical 复用：同 tid/cls/iters 且 classes 一致）----
    done_c = set()
    cur_classes = {t.task_id: list(t.classes) for _n, _X, _y, ts in per_ds
                   for t in ts}
    for path in list(EXISTING_TRAINED) + [cell_cache]:
        for k, rec in load_rprime_cache(path).items():
            tid, key, it = k
            if it == ITERS and rec.get("r_prime") is not None and \
                    list(rec.get("classes", [])) == cur_classes.get(tid):
                done_c.add((tid, key))
    work = [(n, X, y, t, c, key) for n, X, y, tasks in per_ds for t in tasks
            for c, key in shared_cls if (t.task_id, key) not in done_c]
    print(f"@290 格：计划 {sum(len(ts) for _n, _x, _y, ts in per_ds) * len(shared_cls)}"
          f"，复用+已跑 {len(done_c)}，待跑 {len(work)}", flush=True)
    bs, lr = train_cfg["batch_size"], train_cfg["lr"]
    evr = labels_cfg["eval_ratio"]
    t0 = time.time()
    with parallel_backend("threading", n_jobs=1):
        for i, (name, X, y, t, c, key) in enumerate(work):
            w_seed = run_seed_for(t.task_id, key)
            sp_seed = run_seed_for(t.task_id, "train_eval_split")

            def mk(rec, _n=name, _t=t, _k=key, _c=c):
                rp = rec["r_prime"]
                import dataclasses
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": _k, "iters": ITERS, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(t, c, X, y, weight_seed=w_seed,
                                        split_seed=sp_seed, iters=ITERS,
                                        eval_ratio=evr, batch_size=bs, lr=lr,
                                        device=args.device),
                mk, cache_path=cell_cache, inflight=inflight,
                task_id=t.task_id, cls_key=key, iters=ITERS,
                on_oom=ON_OOM_SKIP, oom_log=oom_log,
                context={"dataset": name},
                log=lambda m: print(f"[{i + 1}/{len(work)}] {m}", flush=True))
            if full is not None and full.get("r_prime") is not None:
                eta = (time.time() - t0) / (i + 1) * (len(work) - i - 1)
                print(f"[{i + 1}/{len(work)}] {name}/{t.task_id} {key[:8]} "
                      f"R'={full['r_prime']:.4f} ({full['seconds']}s) "
                      f"ETA {eta / 60:.0f}min", flush=True)
    print(f"run 收官：墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- 阶段 2：analyze
def cmd_analyze(args) -> None:
    outdir = Path(args.outdir)
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    keys = sorted(shared_keys)
    bands, classes, rej_b = dm.collect_baselines(
        list(EXISTING_BASELINES) + [outdir / "baseline_cache.jsonl"],
        exclude_plus_c=True)
    trained, rej_t = dm.collect_trained(
        list(EXISTING_TRAINED) + [outdir / "rprime_cache.jsonl"],
        shared_keys, bands, classes, lambda ds: ITERS)

    flags: dict = defaultdict(dict)
    for (tid, key), rp in trained.items():
        ds = fl.dataset_of(tid)
        if ds in PANEL and int(tid.rsplit("_", 1)[1]) < N_TASKS_PANEL:
            mean, band = bands[tid]
            flags[(ds, tid)][key] = fl.LABELS.index(fl.tri_state(rp, mean, band))

    per_ds: dict = {}
    for ds in PANEL:
        tids = sorted(t for (d, t) in flags if d == ds
                      and len(flags[(d, t)]) == len(keys))
        entry = {"n_complete_tasks": len(tids)}
        if len(tids) >= 4:
            prof = {t: np.array([flags[(ds, t)][k] for k in keys]) for t in tids}
            mat = np.array([[prof[t][i] for t in tids] for i in range(len(keys))])
            entry["gate1"] = tp.perm_test(mat, n_perm=tp.N_PERM, seed=tp.SEED)
            # 闸 2：数据集内 LOTO（P 特征核插值 vs 等权锚，试点 B 赢家逐字）
            P = {t: np.array([bands[t][0], 1.0 / len(classes[t]), bands[t][1]])
                 for t in tids}
            errs_k, errs_u = [], []
            for held in tids:
                rest = [t for t in tids if t != held]
                M = np.stack([P[t] for t in rest])
                mu, sd = lo.standardize_fit(M)
                z = {t: (P[t] - mu) / sd for t in tids}
                dists = {t: float(np.linalg.norm(z[t] - z[held])) for t in rest}
                sigma = float(np.median([np.linalg.norm(z[a] - z[b])
                                         for i, a in enumerate(rest)
                                         for b in rest[i + 1:]]))
                ek, eu = [], []
                for rep in range(N_REPS):
                    anchors = gp.sample_anchors(
                        tids, held=held, k=min(K_ANCHOR, len(rest)),
                        seed=run_seed_for(f"wp:{held}", f"rep{rep}"))
                    wk = gp.kernel_w({a: dists[a] for a in anchors}, sigma)
                    ek.append(gp.profile_brier(
                        gp.anchor_predict(prof, anchors, wk), prof[held]))
                    eu.append(gp.profile_brier(
                        gp.anchor_predict(prof, anchors,
                                          {a: 1.0 for a in anchors}),
                        prof[held]))
                errs_k.append(float(np.mean(ek)))
                errs_u.append(float(np.mean(eu)))
            entry["gate2"] = {
                "tasks": tids, "err_kernel": errs_k, "err_uniform": errs_u,
                "direction": 1 if np.median(errs_k) < np.median(errs_u) else -1,
                "p": lo.paired_wilcoxon_less(np.array(errs_k), np.array(errs_u))}
        per_ds[ds] = entry

    # 汇总判定
    present = [d for d in PANEL if per_ds[d].get("gate1")]
    gate1_pass = [d for d in present
                  if per_ds[d]["gate1"]["p"] < GATE1_ALPHA]
    prim = [d for d in primary_pool() if per_ds[d].get("gate2")]
    pooled = None
    if prim:
        ek = np.concatenate([per_ds[d]["gate2"]["err_kernel"] for d in prim])
        eu = np.concatenate([per_ds[d]["gate2"]["err_uniform"] for d in prim])
        pooled = {"n_tasks": len(ek),
                  "p": lo.paired_wilcoxon_less(ek, eu),
                  "direction": 1 if np.median(ek) < np.median(eu) else -1}

    interim = set(present) <= set(MINI)
    fut = None
    if interim and "UniMiB-SHAR" in present:
        fut = interim_futility(
            p_um=per_ds["UniMiB-SHAR"]["gate1"]["p"],
            dir_um=per_ds["UniMiB-SHAR"].get("gate2", {}).get("direction"),
            dir_ct=per_ds.get("CharacterTrajectories", {})
                   .get("gate2", {}).get("direction"))

    out = {"protocol": "results/wide_panel/PROTOCOL.md",
           "rejects": {"baseline": rej_b, "trained": rej_t},
           "per_dataset": per_ds,
           "gate1_pass": gate1_pass, "gate2_pooled_primary": pooled,
           "interim": interim, "interim_futility": fut}
    (outdir / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print(f"\n===== 宽面板 analyze（{'期中' if interim else '全面板'}）=====")
    for d in PANEL:
        e = per_ds[d]
        if not e.get("gate1"):
            print(f"[{d}] 完整任务 {e['n_complete_tasks']}（<4，未入判定）")
            continue
        g1, g2 = e["gate1"], e["gate2"]
        print(f"[{d}] 任务 {e['n_complete_tasks']} | 闸1 T2={g1['t2_obs']:.3f} "
              f"p={g1['p']:.4f} | 闸2 方向={'+' if g2['direction'] > 0 else '-'} "
              f"p={g2['p']:.4f} 中位 {np.median(g2['err_kernel']):.4f}/"
              f"{np.median(g2['err_uniform']):.4f}")
    print(f"闸1 过 α={GATE1_ALPHA}: {gate1_pass or '无'}")
    if pooled:
        print(f"闸2 主池 pooled: n={pooled['n_tasks']} 方向="
              f"{'+' if pooled['direction'] > 0 else '-'} p={pooled['p']:.4f}")
    if fut is not None:
        print(f"[期中止损检查] → {'触发（关闭面板）' if fut else '未触发（可进二期）'}")
    print(f"→ {outdir / 'analysis.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="面板子集（迷你期 = UniMiB-SHAR CharacterTrajectories）")
    ap.add_argument("--max-tasks", type=int, default=N_TASKS_PANEL)
    ap.add_argument("--max-cls", type=int, default=10)
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT),
                    help="冒烟务必改到别处（铁律 3）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.datasets:
        bad = set(args.datasets) - set(PANEL)
        if bad:
            sys.exit(f"{sorted(bad)} 不在面板（协议 §1）")
    if torch.get_num_threads() > 1 and args.run:
        print("⚠️ 建议 OMP_NUM_THREADS=1 启动（CPU-bound 教训）", flush=True)
    if args.run:
        cmd_run(args)
    if args.analyze:
        cmd_analyze(args)
    if not (args.run or args.analyze):
        ap.error("至少指定 --run / --analyze")


if __name__ == "__main__":
    main()
