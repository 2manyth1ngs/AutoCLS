"""源池 v2 定稿：晋级确认 + 靶池净空（协议 results/source_pool_v2/PROTOCOL.md）。

判据先于补格数据写死：晋级 = 确认存活率≥40% ∧ 非贴顶 ∧ 非死平（锁位
SAD/Ethanol，余位按存活率降序，≤4）；EEG 低权 = SCP2 ≥30%；靶池三档 =
可评×净空。n=2 初筛数字禁止作为晋级依据。

    python scripts/27_source_pool_v2.py --screen              # 零 GPU：初筛表重算对账
    python scripts/27_source_pool_v2.py --run --resume        # GPU 补格（pod 开通后）
    python scripts/27_source_pool_v2.py --analyze             # 终判（满格硬闸）
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
dm = importlib.import_module("scripts.14_destruction_matrix")
wp = importlib.import_module("scripts.24_wide_panel")
_m1 = importlib.import_module("scripts.m1_ggs_har")

from autocls.baseline_probe import baseline_seed_specs, inert_cls  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.labeling import (  # noqa: E402
    cls_key, evaluate_rprime, load_probe_cls, load_rprime_cache, run_seed_for,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402

OUTDIR_DEFAULT = ROOT / "results" / "source_pool_v2"
ITERS = 290
N_BASE_SEEDS = 5
# ---- 协议 §2 固定项 ----
CORE = ("Handwriting", "UniMiB-SHAR", "PhonemeSpectra", "UWaveGestureLibrary")
CONFIRM_TASKS = {                       # 确认轮名单 → n 口径（SCP2 例外 n=3）
    "SpokenArabicDigits": 6, "NATOPS": 6, "EthanolConcentration": 6,
    "HandMovementDirection": 6, "Cricket": 6, "SelfRegulationSCP2": 3,
}
CULLED = ("CharacterTrajectories", "ArticularyWordRecognition", "Heartbeat")
LOCK_PRIORITY = ("SpokenArabicDigits", "EthanolConcentration")   # 合格必入
BACKUPS = ("NATOPS", "HandMovementDirection", "Cricket")
EEG_REP = "SelfRegulationSCP2"
UM = "UniMiB-SHAR"
UM_REGISTRY = "configs/candidate_pool.yaml"   # UM 走候选注册表（宽面板同口径）
# ---- 协议 §4 判读常数 ----
SURV_MIN, EEG_SURV_MIN = 0.40, 0.30
RANGE_MIN, CEIL_MAX = 0.10, 0.99
EVAL_GAP_MIN, HEADROOM_MIN = 0.05, 0.05
PROMO_CAP = 4
GGS_N_MAX = 8


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def dataset_survival(tasks: dict, delta: float) -> dict:
    """数据集 δ 存活率（协议 §3）：全部任务 C(k,2) 配对池化，|ΔR'|>δ 严格。"""
    vals_all: list = []
    n_surv = n_pairs = 0
    for _tid, rmap in tasks.items():
        rs = [rmap[k] for k in sorted(rmap)]
        vals_all += rs
        for i in range(len(rs)):
            for jj in range(i + 1, len(rs)):
                n_pairs += 1
                n_surv += abs(rs[i] - rs[jj]) > delta
    return {"survival": n_surv / n_pairs if n_pairs else 0.0,
            "n_pairs": n_pairs, "n_tasks": len(tasks),
            "max_r": max(vals_all), "range": max(vals_all) - min(vals_all)}


def _eligible(s: dict) -> bool:
    return (s["survival"] >= SURV_MIN and s["max_r"] < CEIL_MAX
            and s["range"] >= RANGE_MIN)


def promotion_verdict(stats: dict) -> list:
    """协议 §4 晋级判：锁位（SAD/Ethanol 合格必入）→ 备选按存活率降序 → ≤4。"""
    elig = {ds for ds, s in stats.items() if _eligible(s)}
    promoted = [ds for ds in LOCK_PRIORITY if ds in elig]
    promoted += sorted((ds for ds in BACKUPS if ds in elig),
                       key=lambda d: (-stats[d]["survival"], d))
    return promoted[:PROMO_CAP]


def eeg_verdict(s: dict) -> bool:
    """协议 §4 EEG 低权判：SCP2 确认存活率 ≥30% ∧ 全距 ≥0.10。"""
    return bool(s["survival"] >= EEG_SURV_MIN and s["range"] >= RANGE_MIN)


def target_verdict(m: dict) -> str:
    """协议 §4 靶池三档：可评∧净空=常规靶；可评∧¬净空=展示靶；¬可评=out。"""
    if m["evalgap_med"] < EVAL_GAP_MIN or m["max_r"] >= CEIL_MAX:
        return "out"
    return "regular" if m["headroom_med"] >= HEADROOM_MIN else "showcase"


def ggs_task_count(planned: int) -> int:
    return min(GGS_N_MAX, planned)


# ---------------------------------------------------------------- 数据收集
def collect_rprime_map(paths, keys_wanted: set, iters: int) -> tuple[dict, dict]:
    """{(tid, key): R'} 轻收集器（存活率/净空不需基线带）：canonical 划分 +
    classes 一致性（先到先得钉死）+ iters 校验 + 剔 +c 档标任务。"""
    out: dict = {}
    classes: dict = {}
    rejects: dict = defaultdict(int)
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid, key = r.get("task_id"), r.get("cls_key")
            if tid is None or r.get("r_prime") is None:
                rejects["null"] += 1
                continue
            if "+" in tid:
                rejects["plus_tag"] += 1
                continue
            if key not in keys_wanted:
                rejects["not_wanted"] += 1
                continue
            if r.get("iters") != iters:
                rejects["iters"] += 1
                continue
            if r.get("split_seed") != dm.canonical_split(tid):
                rejects["split"] += 1
                continue
            if tid in classes and list(classes[tid]) != list(r.get("classes", [])):
                rejects["classes_conflict"] += 1
                continue
            classes.setdefault(tid, r.get("classes", []))
            out.setdefault((tid, key), float(r["r_prime"]))
    return out, dict(rejects)


# UM 不在 labels.yaml delta_per_dataset（非路线 A 池）；初筛（memory
# source-pool-screen-20260831）对其用 0.05（δ=0.05 逐位复现 68%），非全局
# 回退 0.02——同口径钉死防移动球门。UM 是核心源不入确认判，仅影响对账表。
DELTA_OVERRIDE = {UM: 0.05}


def delta_table() -> tuple[dict, float]:
    cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                         .read_text(encoding="utf-8"))
    return {**cfg["delta_per_dataset"], **DELTA_OVERRIDE}, cfg["delta"]


def all_trained_paths(outdir: Path) -> list:
    return (list(wp.EXISTING_TRAINED)
            + [wp.OUTDIR_DEFAULT / "rprime_cache.jsonl",
               outdir / "rprime_cache.jsonl"])


# ---------------------------------------------------------------- --screen
def cmd_screen(args) -> None:
    """零 GPU：既有缓存重算初筛表（§1 已见读数的机器对账，不构成任何判定）。"""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shared_keys = {k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0]}
    dmap, ddef = delta_table()
    trained, rejects = collect_rprime_map(
        all_trained_paths(outdir), shared_keys, ITERS)
    per_ds: dict = defaultdict(lambda: defaultdict(dict))
    for (tid, key), r in trained.items():
        per_ds[fl.dataset_of(tid)][tid][key] = r
    table = {}
    for ds, tasks in sorted(per_ds.items()):
        full = {tid: m for tid, m in tasks.items() if len(m) == len(shared_keys)}
        if not full:
            continue
        delta = dmap.get(ds, ddef)
        table[ds] = {**dataset_survival(full, delta), "delta": delta}
    out = {"protocol": "results/source_pool_v2/PROTOCOL.md",
           "note": "初筛表机器对账（已见读数重算，不构成判定；协议 §1）",
           "rejects": rejects, "table": table}
    (outdir / "screen.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"{'数据集':<28}{'任务':>4}{'配对':>6}{'存活率':>8}{'δ':>6}"
          f"{'max':>7}{'全距':>7}")
    for ds, s in sorted(table.items(), key=lambda kv: -kv[1]["survival"]):
        print(f"{ds:<28}{s['n_tasks']:>4}{s['n_pairs']:>6}"
              f"{s['survival']:>8.2f}{s['delta']:>6.2f}"
              f"{s['max_r']:>7.3f}{s['range']:>7.3f}")
    print(f"→ {outdir / 'screen.json'}", flush=True)


# ---------------------------------------------------------------- --run
def build_plan(max_tasks_cap: int | None = None) -> list:
    """确认轮 6 + 核心 4 的加载与切片；按 l_enc 升序（断点落长 L 端）。"""
    plan = []
    for name in list(CONFIRM_TASKS) + list(CORE):
        reg = (load_registry(ROOT / UM_REGISTRY) if name == UM
               else load_registry())
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _, tasks = slice_source_tasks(entry, reg, y)
        n_conf = CONFIRM_TASKS.get(name, 0)
        n_ggs = ggs_task_count(entry.planned_tasks)
        if max_tasks_cap:
            n_conf, n_ggs = min(n_conf, max_tasks_cap), min(n_ggs, max_tasks_cap)
        n_need = max(n_conf, n_ggs)
        if len(tasks) < n_need:
            sys.exit(f"[{name}] 切片任务 {len(tasks)} < 需要 {n_need}")
        plan.append({"name": name, "entry": entry, "X": X, "y": y,
                     "tasks": tasks[:n_need], "n_conf": n_conf, "n_ggs": n_ggs})
        print(f"[{name}] X={X.shape} 确认 {n_conf} / GGS {n_ggs} 任务"
              f"（l_enc={entry.l_enc}）", flush=True)
    plan.sort(key=lambda d: d["entry"].l_enc)
    return plan


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
    from joblib import parallel_backend
    shared_cls = load_probe_cls(fl.PROBE_CATALOG, 10)[0]
    ggs_k = cls_key(_m1.GGS)
    plan = build_plan(args.max_tasks)

    # ---- 基线补测（确认任务；既有键复用，协议 §3）----
    split_of = lambda tid: run_seed_for(tid, "train_eval_split")  # noqa: E731
    done_b = wp.baseline_done_keys(
        wp.read_rows(list(wp.EXISTING_BASELINES) + [base_cache]),
        split_of=split_of)
    cls0 = inert_cls()
    k0 = cls_key(cls0)
    todo_b = [(d, t, s) for d in plan for t in d["tasks"][: d["n_conf"]]
              for s in range(N_BASE_SEEDS) if (t.task_id, s) not in done_b]
    print(f"基线：待补 {len(todo_b)} 条（既有键 {len(done_b)} 复用）", flush=True)
    with parallel_backend("threading", n_jobs=1):
        for i, (d, t, s) in enumerate(todo_b if not args.dry_run else []):
            w_seed, sp_seed = baseline_seed_specs(t.task_id, s)

            def mk(rec, _d=d, _t=t, _s=s):
                rp = rec["r_prime"]
                return {"dataset": _d["name"], "task_id": _t.task_id,
                        "tag": _t.tag, "task_type": _t.task_type,
                        "classes": list(_t.classes), "cls_key": k0,
                        "seed": _s, "iters": 0, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            guarded_run(lambda: evaluate_rprime(t, cls0, d["X"], d["y"],
                                                weight_seed=w_seed,
                                                split_seed=sp_seed, iters=0,
                                                device=args.device),
                        mk, cache_path=base_cache, inflight=inflight,
                        task_id=t.task_id, cls_key=k0, iters=0,
                        on_oom=ON_OOM_SKIP, oom_log=oom_log,
                        context={"dataset": d["name"], "seed": s},
                        log=lambda m: print(f"[base {i + 1}/{len(todo_b)}] {m}",
                                            flush=True))

    # ---- @290 格：确认（shared 10）+ GGS（复用键 classes 校验，24 同式）----
    done_c = set()
    cur_classes = {t.task_id: list(t.classes) for d in plan for t in d["tasks"]}
    for path in all_trained_paths(outdir):
        for k, rec in load_rprime_cache(path).items():
            tid, key, it = k
            if it == ITERS and rec.get("r_prime") is not None and \
                    list(rec.get("classes", [])) == cur_classes.get(tid):
                done_c.add((tid, key))
    work = []
    for d in plan:
        for t in d["tasks"][: d["n_conf"]]:
            work += [(d, t, c, key, "confirm") for c, key in shared_cls
                     if (t.task_id, key) not in done_c]
        for t in d["tasks"][: d["n_ggs"]]:
            if (t.task_id, ggs_k) not in done_c:
                work.append((d, t, _m1.GGS, ggs_k, "ggs"))
    n_plan = sum(d["n_conf"] * len(shared_cls) + d["n_ggs"] for d in plan)
    print(f"@290 格：计划 {n_plan}，复用+已跑 {n_plan - len(work)}，"
          f"待跑 {len(work)}", flush=True)
    if args.dry_run:
        per: dict = defaultdict(lambda: defaultdict(int))
        for d, _t, _c, _key, origin in work:
            per[d["name"]][origin] += 1
        for name, c in per.items():
            print(f"  [{name}] confirm {c.get('confirm', 0)} / "
                  f"ggs {c.get('ggs', 0)}")
        print("dry-run：工单构建完毕，未执行任何评估", flush=True)
        return
    bs, lr = train_cfg["batch_size"], train_cfg["lr"]
    evr = labels_cfg["eval_ratio"]
    t0 = time.time()
    with parallel_backend("threading", n_jobs=1):
        for i, (d, t, c, key, origin) in enumerate(work):
            w_seed = run_seed_for(t.task_id, key)
            sp_seed = run_seed_for(t.task_id, "train_eval_split")

            def mk(rec, _d=d, _t=t, _k=key, _c=c, _o=origin):
                rp = rec["r_prime"]
                import dataclasses
                return {"dataset": _d["name"], "task_id": _t.task_id,
                        "tag": _t.tag, "task_type": _t.task_type,
                        "classes": list(_t.classes), "cls_key": _k,
                        "iters": ITERS, "origin": _o, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp),
                        "cls": dataclasses.asdict(_c)}

            full = guarded_run(
                lambda: evaluate_rprime(t, c, d["X"], d["y"],
                                        weight_seed=w_seed, split_seed=sp_seed,
                                        iters=ITERS, eval_ratio=evr,
                                        batch_size=bs, lr=lr,
                                        device=args.device),
                mk, cache_path=cell_cache, inflight=inflight,
                task_id=t.task_id, cls_key=key, iters=ITERS,
                on_oom=ON_OOM_SKIP, oom_log=oom_log,
                context={"dataset": d["name"], "origin": origin},
                log=lambda m: print(f"[{i + 1}/{len(work)}] {m}", flush=True))
            if full is not None and full.get("r_prime") is not None:
                eta = (time.time() - t0) / (i + 1) * (len(work) - i - 1)
                print(f"[{i + 1}/{len(work)}] {d['name']}/{t.task_id} "
                      f"{origin} {key[:8]} R'={full['r_prime']:.4f} "
                      f"({full['seconds']}s) ETA {eta / 60:.0f}min", flush=True)
    print(f"run 收官：墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- --analyze
def grid_tids(trained: dict, ds: str, n: int) -> list:
    tids = sorted({tid for (tid, _k) in trained
                   if fl.dataset_of(tid) == ds
                   and tid.rsplit("_", 1)[1].isdigit()
                   and int(tid.rsplit("_", 1)[1]) < n})
    return tids


def cmd_analyze(args) -> None:
    outdir = Path(args.outdir)
    shared_keys = sorted(k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0])
    ggs_k = cls_key(_m1.GGS)
    dmap, ddef = delta_table()
    trained, rejects = collect_rprime_map(
        all_trained_paths(outdir), set(shared_keys) | {ggs_k}, ITERS)

    # ---- 确认轮硬闸 + 存活率（协议 §4）----
    confirm_stats: dict = {}
    for ds, n in CONFIRM_TASKS.items():
        tids = grid_tids(trained, ds, n)
        missing = [f"{ds}: 任务数 {len(tids)}/{n}"] if len(tids) != n else []
        tasks = {}
        for tid in tids:
            m = {k: trained[(tid, k)] for k in shared_keys
                 if (tid, k) in trained}
            if len(m) != len(shared_keys):
                missing.append(f"{tid}: {len(m)}/{len(shared_keys)} CLS")
            tasks[tid] = m
        if missing:
            sys.exit("确认轮缺格（硬闸，先 --run --resume 补齐）：\n  "
                     + "\n  ".join(missing))
        confirm_stats[ds] = {**dataset_survival(tasks, dmap.get(ds, ddef)),
                             "delta": dmap.get(ds, ddef)}
    cand_stats = {ds: s for ds, s in confirm_stats.items() if ds != EEG_REP}
    promoted = promotion_verdict(cand_stats)
    eeg_ok = eeg_verdict(confirm_stats[EEG_REP])

    # ---- 靶池（核心 + 晋级者中 planned≥8；网格∩GGS 任务，协议 §4）----
    target: dict = {}
    for ds in list(CORE) + promoted:
        reg = load_registry(ROOT / UM_REGISTRY) if ds == UM else load_registry()
        if reg.get(ds).planned_tasks < 8:
            continue
        n_grid = 8 if ds in CORE else CONFIRM_TASKS[ds]
        tids = grid_tids(trained, ds, n_grid)
        gaps, heads, max_r = [], [], 0.0
        missing = [f"{ds}: 网格任务 {len(tids)}/{n_grid}"] \
            if len(tids) != n_grid else []
        for tid in tids:
            rs = [trained[(tid, k)] for k in shared_keys if (tid, k) in trained]
            if len(rs) != len(shared_keys):
                missing.append(f"{tid}: 网格 {len(rs)}/{len(shared_keys)}")
                continue
            if (tid, ggs_k) not in trained:
                missing.append(f"{tid}: 缺 GGS@290 格")
                continue
            max_r = max(max_r, max(rs))
            gaps.append(max(rs) - median(rs))
            heads.append(max(rs) - trained[(tid, ggs_k)])
        if missing:
            sys.exit("靶池缺格（硬闸）：\n  " + "\n  ".join(missing))
        m = {"evalgap_med": float(median(gaps)),
             "headroom_med": float(median(heads)), "max_r": float(max_r)}
        target[ds] = {**m, "verdict": target_verdict(m), "n_tasks": len(tids)}

    regular = [d for d, v in target.items() if v["verdict"] == "regular"]
    preds = {
        "PS1": {"claim": "SAD 确认 ≥40%", "prior": 0.70,
                "hit": confirm_stats["SpokenArabicDigits"]["survival"] >= SURV_MIN},
        "PS2": {"claim": "Ethanol 确认 ≥40%", "prior": 0.60,
                "hit": confirm_stats["EthanolConcentration"]["survival"] >= SURV_MIN},
        "PS3": {"claim": "晋级总数 ≥2", "prior": 0.75, "hit": len(promoted) >= 2},
        "PS4": {"claim": "HW @290 净空 <0.05", "prior": 0.85,
                "hit": target["Handwriting"]["headroom_med"] < HEADROOM_MIN},
        "PS5": {"claim": "SCP2 确认 ≥30%", "prior": 0.60,
                "hit": confirm_stats[EEG_REP]["survival"] >= EEG_SURV_MIN},
        "PS6": {"claim": "常规靶 ≥2", "prior": 0.60, "hit": len(regular) >= 2},
    }

    out = {"protocol": "results/source_pool_v2/PROTOCOL.md",
           "rejects": rejects,
           "confirm_stats": confirm_stats,
           "promoted": promoted, "eeg_rep_kept": eeg_ok,
           "source_pool_v2": list(CORE) + promoted
           + ([EEG_REP] if eeg_ok else []),
           "culled": list(CULLED),
           "target_pool": target,
           "target_regular": regular,
           "predictions": preds}
    (outdir / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print("\n===== 源池 v2 终判（协议判读带执行）=====")
    print(f"{'数据集':<26}{'n':>3}{'存活率':>8}{'δ':>6}{'max':>7}{'全距':>7}  判")
    for ds, s in sorted(confirm_stats.items(), key=lambda kv: -kv[1]["survival"]):
        tag = ("晋级" if ds in promoted else
               ("低权入池" if ds == EEG_REP and eeg_ok else "落选"))
        print(f"{ds:<26}{s['n_tasks']:>3}{s['survival']:>8.2f}{s['delta']:>6.2f}"
              f"{s['max_r']:>7.3f}{s['range']:>7.3f}  {tag}")
    print(f"\n源池 v2 = 核心 {list(CORE)} + 晋级 {promoted}"
          f"{' + 低权 ' + EEG_REP if eeg_ok else ''}；剔除 {list(CULLED)}")
    print(f"\n{'靶池候选':<26}{'可评gap':>8}{'净空':>8}{'max':>7}  档")
    for ds, v in target.items():
        print(f"{ds:<26}{v['evalgap_med']:>8.3f}{v['headroom_med']:>8.3f}"
              f"{v['max_r']:>7.3f}  {v['verdict']}")
    for k, p in preds.items():
        print(f"{k}: {p['claim']} → {'✓' if p['hit'] else '✗'}")
    print(f"→ {outdir / 'analysis.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="--run 只建工单不执行（装配/复用键/预算本地预检）")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="冒烟用：每数据集任务数上限")
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT),
                    help="冒烟务必改到别处（铁律 3）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.run and args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.get_num_threads() > 1:
            print("⚠️ 建议 OMP_NUM_THREADS=1 启动（CPU-bound 教训）", flush=True)
    if args.screen:
        cmd_screen(args)
    if args.run:
        cmd_run(args)
    if args.analyze:
        cmd_analyze(args)
    if not (args.screen or args.run or args.analyze):
        ap.error("至少指定 --screen / --run / --analyze")


if __name__ == "__main__":
    main()
