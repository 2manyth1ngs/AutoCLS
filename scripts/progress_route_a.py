"""路线 A 进度自查（Part A-4；只读，本地/云上随时可跑，不动任何缓存）。

四块信息（方案 B-4）：
① step0 过闸：每数据集 verdict（pass/retry_ceiling/fail_mismatch/fail_unstable）
   + SNR + R̄ + 恢复档（recommended_n_classes_range）+ pending_retry；
② 存活任务总数（tested 骨干 ∪ analysis 通过者）对照任务数下限闸 floor=100；
③ 阶段2 label 进度：rprime_cache.jsonl 完成/隔离/失败条数 vs 计划总数
   （存活任务 × 20 CLS）、per-dataset 完成率、pairs.jsonl 对数、δ 弃样比例；
④ 看门狗：重启次数（worker_r*.log 计数）、毒条目计数（watchdog_state.tsv）、
   inflight 残留、watchdog.log 尾部。

用法：
    python scripts/progress_route_a.py                     # 默认目录
    python scripts/progress_route_a.py --labels-dir results/labels_route_a_smoke
    python scripts/progress_route_a.py --smoke             # 读 analysis_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from autocls.data.source_pool import dataset_of_task_id, load_registry  # noqa: E402

DEFAULT_STEP0 = ROOT / "results" / "step0_snr_gate"
DEFAULT_LABELS = ROOT / "results" / "labels_route_a"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 被杀进程可能留半行
    return out


def _watchdog_state(outdir: Path) -> dict:
    state_path = outdir / "watchdog_state.tsv"
    poison = Counter()
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                poison[line] += 1
    inflight_path = outdir / "inflight.txt"
    inflight = inflight_path.read_text(encoding="utf-8") if inflight_path.exists() else ""
    log = outdir / "watchdog.log"
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:] if log.exists() else []
    return {
        "restarts": len(list(outdir.glob("worker_*.log"))),  # 含跨看门狗重启的时间戳日志
        "poison": dict(poison),
        "inflight": inflight,
        "log_tail": tail,
    }


def collect_progress(
    *,
    step0_dir: Path = DEFAULT_STEP0,
    labels_dir: Path = DEFAULT_LABELS,
    analysis_name: str = "analysis.json",
) -> dict:
    """汇总进度为纯数据（打印在 main；本函数可单测）。"""
    registry = load_registry()
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    ra = labels_cfg.get("route_a", {})
    n_cls_per_task = ra.get("n_shared_cls", 10) + ra.get("n_random_cls", 10)
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    early_iters = train_cfg["early_iters"]

    # ① step0 过闸
    step0 = None
    apath = Path(step0_dir) / analysis_name
    if apath.exists():
        a = json.loads(apath.read_text(encoding="utf-8"))
        step0 = {
            "datasets": {ds: {k: d.get(k) for k in (
                "verdict", "passed", "snr", "rprime_grand_mean", "pending_retry",
                "recommended_n_classes_range", "suggested_delta")}
                for ds, d in a.get("datasets", {}).items()},
            "gate": a.get("gate"),
        }

    # ②③ 存活池 + 阶段2 进度（计划 = 存活任务 × 20 CLS）
    passed = {ds for ds, d in (step0 or {}).get("datasets", {}).items() if d.get("passed")}
    pool = [e for e in registry.source if e.tested or e.name in passed]
    planned_tasks = {e.name: e.planned_tasks for e in pool}
    planned_records = sum(planned_tasks.values()) * n_cls_per_task if pool else None

    recs = _read_jsonl(Path(labels_dir) / "rprime_cache.jsonl")
    done_keys, quarantined, failed = set(), 0, 0
    per_ds_done: dict[str, set] = defaultdict(set)
    for r in recs:
        if r.get("iters") != early_iters:
            continue  # 非 early 口径（如 smoke 30 iters）不计
        key = (r["task_id"], r["cls_key"])
        if r.get("r_prime") is None:
            if "watchdog" in (r.get("error") or ""):
                quarantined += 1
            else:
                failed += 1
            continue
        if key not in done_keys:
            done_keys.add(key)
            per_ds_done[dataset_of_task_id(r["task_id"])].add(key)

    pairs_path = Path(labels_dir) / "pairs.jsonl"
    summary_path = Path(labels_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    labels = {
        "cache_exists": (Path(labels_dir) / "rprime_cache.jsonl").exists(),
        "n_records": len(recs),
        "n_done": len(done_keys),
        "n_quarantined": quarantined,
        "n_failed": failed,
        "planned_records": planned_records,
        "per_dataset": {ds: {
            "done": len(per_ds_done.get(ds, ())),
            "planned": planned_tasks.get(ds, None) and planned_tasks[ds] * n_cls_per_task,
        } for ds in sorted(set(per_ds_done) | set(planned_tasks))},
        "n_pairs": len(_read_jsonl(pairs_path)) if pairs_path.exists() else None,
        "delta_drop_frac": summary.get("delta_drop_frac") if summary else None,
    }

    # ④ 看门狗（两阶段各自目录）
    watchdog = {
        "step0": _watchdog_state(Path(step0_dir)),
        "labels": _watchdog_state(Path(labels_dir)),
    }
    return {"step0": step0, "labels": labels, "watchdog": watchdog}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step0-dir", type=Path, default=DEFAULT_STEP0)
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--smoke", action="store_true", help="读 analysis_smoke.json")
    args = ap.parse_args()

    p = collect_progress(step0_dir=args.step0_dir, labels_dir=args.labels_dir,
                         analysis_name="analysis_smoke.json" if args.smoke else "analysis.json")

    print(f"{'='*88}\n① step0 过闸（{args.step0_dir}）\n{'—'*88}")
    if p["step0"] is None:
        print("  analysis.json 不存在：step0 未跑或未 --analyze")
    else:
        for ds, d in p["step0"]["datasets"].items():
            rng = d.get("recommended_n_classes_range")
            print(f"  {ds:<26} {d.get('verdict', '?'):<14} SNR={d.get('snr')} "
                  f"R̄={d.get('rprime_grand_mean')} δ≈{d.get('suggested_delta')}"
                  + (f" 恢复档 c{rng[0]}-{rng[1]}" if rng else "")
                  + (" ⚠️pending_retry" if d.get("pending_retry") else ""))
        g = p["step0"]["gate"] or {}
        mark = "✅" if g.get("ok") else "❌"
        print(f"\n② 任务数下限闸：{mark} {g.get('count')}/{g.get('floor')}  {g.get('message', '')}")

    lab = p["labels"]
    print(f"\n{'—'*88}\n③ 阶段2 label 进度（{args.labels_dir}）\n{'—'*88}")
    if not lab["cache_exists"]:
        print("  rprime_cache.jsonl 不存在：阶段2 未开跑")
    else:
        frac = (f"（{lab['n_done'] / lab['planned_records']:.1%}）"
                if lab["planned_records"] else "")
        print(f"  完成 {lab['n_done']}/{lab['planned_records'] or '?'} 条{frac} | "
              f"看门狗隔离 {lab['n_quarantined']} | 训练失败 {lab['n_failed']} | "
              f"缓存总行 {lab['n_records']}")
        for ds, d in lab["per_dataset"].items():
            pl = d["planned"]
            bar = f"{d['done']}/{pl}" if pl else f"{d['done']}/?"
            print(f"    {ds:<26} {bar}")
        print(f"  pairs: {lab['n_pairs']} 对 | δ 弃样比例: {lab['delta_drop_frac']}")

    print(f"\n{'—'*88}\n④ 看门狗\n{'—'*88}")
    for stage in ("step0", "labels"):
        wd = p["watchdog"][stage]
        print(f"  [{stage}] 重启 {wd['restarts']} 次 | 毒条目 "
              f"{len(wd['poison'])} 种（{sum(wd['poison'].values())} 次崩）"
              + (f" | ⚠️ inflight 残留: {wd['inflight']}" if wd["inflight"].strip() else ""))
        for line, n in wd["poison"].items():
            print(f"      {line.replace(chr(9), ' / ')}  ×{n}")
        for ln in wd["log_tail"][-3:]:
            print(f"      log| {ln}")


if __name__ == "__main__":
    main()
