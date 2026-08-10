"""workpoint confirm 分析（协议 results/workpoint_confirm/PROTOCOL.md）。

判据代码全部复用：T1/T2/T3 与臂汇总 = iters_workpoint_scan.collect 原样；
确认层级 = autocls.workpoint.confirm_verdict（A/B/C，n=60）；破坏参与 =
workpoint_followup.destruction_table（δ=0.05 主 / 0.03 敏感）。本轮唯一新增
判定是选点纯函数 select_workpoint（0.02 让分窗，先测后写，协议 §2）。

用法：
    python scripts/workpoint_confirm.py                      # 段1 + 选点
    python scripts/workpoint_confirm.py --uwave-confirm 30   # 段2：UWave 确认臂
    python scripts/workpoint_confirm.py --verify             # 收官三重校验之满格/零重复
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.workpoint import confirm_verdict, select_workpoint, wilson_interval  # noqa: E402
from scripts.iters_workpoint_scan import collect, read_jsonl  # noqa: E402
from scripts.workpoint_followup import destruction_table  # noqa: E402

OUTDIR = "results/workpoint_confirm/uea"
BASE = f"{OUTDIR}/baseline_cache.jsonl"
FILL = f"{OUTDIR}/trained_fill.jsonl"
STAGE1_290 = "results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl"

HW_ARMS = [120, 60]                 # 主判定臂在前（协议 §3）
UW_SCAN_ARMS = [3, 6, 15, 30, 60]   # {1,2,5,10,20} ep × 3 steps/ep（任务 00–01）
TIER_RANK = {"confirmed": 2, "marginal": 1, "reversed": 0}

fmt = lambda v, w=8: (f"{v:>+{w}.3f}" if isinstance(v, (int, float)) else f"{'—':>{w}}")
tick = lambda b: "✅" if b else "❌"


def confirm_arms(name: str, expected: list[int], *, trained: list[str]) -> dict:
    """60 条确认口径：n_tasks=6 / arm_rows=60 / allow_mixed_spe（协议 §1 epoch 错位）。"""
    r = collect(name, {"registry": None, "baseline": BASE, "trained": trained,
                       "expected": expected},
                n_tasks=6, arm_rows=60, allow_mixed_spe=True)
    for iters in expected:
        a = r["arms"].get(iters)
        if a and a["n_trained"]:
            a["confirm"] = confirm_verdict(n_above=a["n_above"], n_below=a["n_below"],
                                           n_trained=a["n_trained"], t1=a["T1"], t2=a["T2"])
    return r


def print_confirm(name: str, r: dict, workpoint_iters: list[int]) -> None:
    print(f"{'数据集':<21}{'iters':>6}{'条数':>5}{'基线':>7}{'最好':>8}{'实得':>8}"
          f"{'↑':>4}{'↓':>4}{'带内':>5}{'Wilson90':>16}  层级")
    for iters in sorted(r["arms"]):
        a = r["arms"][iters]
        if not a["n_trained"] or "confirm" not in a:
            continue
        cv = a["confirm"]
        wl = (f"[{cv['wilson_low']:.3f},{cv['wilson_high']:.3f}]"
              if cv["wilson_low"] is not None else "—")
        role = "主判定臂" if iters == workpoint_iters[0] else "段内臂"
        print(f"{name:<21}{iters:>6}{a['n_trained']:>5}{a['baseline']:>7.3f}"
              f"{fmt(a['best_gain'])}{fmt(a['realized_gain'])}{a['n_above']:>4}"
              f"{a['n_below']:>4}{a['n_band']:>5}{wl:>16}  {cv['tier']}（{role}）")
    if r["missing_arms"]:
        print(f"{'':<21}⚠ 缺臂: {r['missing_arms']}")


def pool_tier(r: dict) -> str:
    """入池判定 = 全部预注册臂的最高层级（协议 §3：工作点是段不是点）。"""
    tiers = [a["confirm"]["tier"] for a in r["arms"].values()
             if a.get("n_trained") and "confirm" in a]
    return max(tiers, key=TIER_RANK.get) if tiers else "no_data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uwave-confirm", type=int, default=None,
                    help="段2：UWave 确认臂的 iters（由段1 select_workpoint 给出）")
    ap.add_argument("--out", default="results/workpoint_confirm/confirm.json")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        rows = read_jsonl(ROOT / FILL)
        keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
        dupes = {k: n for k, n in Counter(keys).items() if n > 1}
        nulls = [k for r, k in zip(rows, keys) if r.get("r_prime") is None]
        per = Counter((r["dataset"], int(r["iters"])) for r in rows)
        no_dev = sum(1 for r in rows if not r.get("device_name"))
        print(f"总条数 {len(rows)} ｜ 重复键 {len(dupes)} ｜ null r_prime {len(nulls)}"
              f" ｜ 无 device_name {no_dev}（UWave@290 种子 20 条为 5090 旧格式，预期内）")
        for (ds, it), n in sorted(per.items()):
            print(f"  {ds:<22} iters={it:>4}  n={n}")
        if dupes:
            print(f"⚠ 重复键样例: {list(dupes)[:5]}")
        return

    payload: dict = {}

    # ---- Handwriting 确认（两臂，主判定臂 @120）------------------------------
    print(f"{'='*104}\nHandwriting 确认（60 条 = 10 CLS × 前 6 任务；判据协议 §3，B 不入池）\n{'='*104}")
    hw = confirm_arms("Handwriting", HW_ARMS, trained=[FILL])
    print_confirm("Handwriting", hw, HW_ARMS)
    hw_pool = pool_tier(hw)
    print(f"\nHandwriting 入池判定（两臂最高层级）: {hw_pool}\n")
    payload["handwriting"] = {"arms": hw, "pool_tier": hw_pool}

    # ---- UWave 扫描 + @290 参照 + 选点 --------------------------------------
    print(f"{'='*104}\nUWave 工作点扫描（20 条/臂 = 10 CLS × 前 2 任务；@290 = 5090 历史参照臂）\n{'='*104}")
    uw_scan = collect("UWaveGestureLibrary",
                      {"registry": None, "baseline": BASE, "trained": [FILL],
                       "expected": UW_SCAN_ARMS}, n_tasks=2, arm_rows=20)
    uw_290 = collect("UWaveGestureLibrary",
                     {"registry": None, "baseline": BASE, "trained": [STAGE1_290],
                      "expected": [290]}, n_tasks=2, arm_rows=20)
    merged = dict(uw_scan["arms"])
    merged.update(uw_290["arms"])
    print(f"{'iters':>6}{'epoch':>7}{'条数':>5}{'基线':>7}{'实得':>8}{'最好':>8}"
          f"{'↑':>4}{'↓':>4}{'带内':>5}  {'T1':<3}{'T2':<3}{'T3':<3} 判定")
    for iters in sorted(merged):
        a = merged[iters]
        if not a["n_trained"]:
            continue
        ep = f"{a['epoch']:>7.1f}" if a.get("epoch") is not None else f"{'—':>7}"
        print(f"{iters:>6}{ep}{a['n_trained']:>5}{a['baseline']:>7.3f}"
              f"{fmt(a['realized_gain'])}{fmt(a['best_gain'])}{a['n_above']:>4}"
              f"{a['n_below']:>4}{a['n_band']:>5}  "
              f"{tick(a['T1']):<3}{tick(a['T2']):<3}{tick(a['T3']):<3} {a['verdict']}")
    sel = select_workpoint({it: {"usable": a["verdict"] == "usable",
                                 "best_gain": a["best_gain"]}
                            for it, a in merged.items() if a["n_trained"]})
    if sel is None:
        print("\n选点：候选集空 → 按协议 §2-3 回落 @290 确认（seed_fill.py --uwave290）")
    else:
        print(f"\n选点（0.02 让分窗）：iters={sel['iters']}"
              f"（best_gain {sel['best_gain']:+.3f}；top {sel['top_gain']:+.3f}；"
              f"候选臂 {sel['candidates']}）")
    payload["uwave_scan"] = {"arms": merged, "selection": sel or {"fallback": 290}}

    # ---- UWave 确认（段2）---------------------------------------------------
    uw_confirm = None
    if args.uwave_confirm is not None:
        print(f"\n{'='*104}\nUWave 确认（60 条 @iters={args.uwave_confirm}）\n{'='*104}")
        uw_confirm = confirm_arms("UWaveGestureLibrary", [args.uwave_confirm],
                                  trained=[FILL])
        print_confirm("UWaveGestureLibrary", uw_confirm, [args.uwave_confirm])
        uw_pool = pool_tier(uw_confirm)
        print(f"\nUWave 入池判定: {uw_pool}\n")
        payload["uwave_confirm"] = {"arms": uw_confirm, "pool_tier": uw_pool,
                                    "iters": args.uwave_confirm}

    # ---- 破坏参与率（如实报告，不设硬门槛）----------------------------------
    specs = {("Handwriting", it): {"registry": None, "n_tasks": 6,
                                   "baseline": BASE, "trained": [FILL]}
             for it in HW_ARMS}
    for it in UW_SCAN_ARMS:
        specs[("UWaveGestureLibrary", it)] = {"registry": None, "n_tasks": 2,
                                              "baseline": BASE, "trained": [FILL]}
    specs[("UWaveGestureLibrary", 290)] = {"registry": None, "n_tasks": 2,
                                           "baseline": BASE, "trained": [STAGE1_290]}
    if args.uwave_confirm is not None and args.uwave_confirm != 290:
        specs[("UWaveGestureLibrary", args.uwave_confirm)] = {
            "registry": None, "n_tasks": 6, "baseline": BASE, "trained": [FILL]}
    dtable = destruction_table(specs)
    print(f"{'='*104}\n破坏参与率逐臂（δ=0.05 主口径；strict = 至少一方 < 基线−null_band）\n{'='*104}")
    print(f"{'数据集':<21}{'iters':>6}{'条数':>5}{'存活对':>10}{'存活率':>8}"
          f"{'strict参与':>11}{'loose参与':>10}")
    pct = lambda v: f"{v:>7.0%}" if v is not None else f"{'—':>7}"
    for row in sorted(dtable, key=lambda r: (r["dataset"], r["iters"])):
        m = row["main"]
        print(f"{row['dataset']:<21}{row['iters']:>6}{m['n_cls']:>5}"
              f"{m['n_survive']:>5}/{m['n_pairs']:<4}{pct(m['survive_rate'])}"
              f" {m['n_strict']:>3}/{m['n_survive']:<3}{pct(m['strict_share'])}"
              f" {m['n_loose']:>3}/{m['n_survive']:<3}{pct(m['loose_share'])}")
    payload["destruction"] = dtable
    payload["wilson_threshold_60"] = {"low_37": wilson_interval(37, 60)[0],
                                      "low_36": wilson_interval(36, 60)[0]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
