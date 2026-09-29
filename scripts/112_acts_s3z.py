"""WP17-ACTS S3-Z（results/acts/s3z/PROTOCOL.md）：独立搜索者翻倍（S3-Y 之上只改一项）。

成员 5 → 10（新训种子 5–9，设置同 S3-Y；种子 0–4 复用 S3-Y 模型），搜索者 6 → 11（10 个成员各自 + 10 成员集成均值），
其余（Ks、按分数进化、每搜索者前 2 并集、3 折 CV 选 1、3 种子最终）与 S3-Y 逐项相同；搜索 / 计划 / 选择直接调用 111。
阶段：
  --prep          本地：meta.json（= S3-Y）+ runs/runs.jsonl 预置 S3-Y 已跑格（同键复用）
  --search / --plan cv|final / --select   服务器：= 111 同名阶段（MEMBERS = 0..9）
  --analyze       本地：S3-Y 口径读数 + 选中者来源、搜索者曲线 m = 1…11、读数码
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3z"
S3Y = ROOT / "results" / "acts" / "s3y"
MEMBERS10 = tuple(range(10))
OLD_SEARCHERS = tuple(f"m{i}" for i in range(5))
PREV = 0.6330                                        # S3-Y NEW 部署均值


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


def configure():
    y = _g("111_acts_s3y")
    y.MEMBERS = MEMBERS10
    return y


# ================================================================ 纯函数
def searcher_curve(picks: Mapping[str, Sequence[str]], s0: Mapping[str, float], max_combos: Optional[int] = None, seed: int = 0) -> dict:
    """m 个搜索者的并集里 s0 最高值，对全部 m 子集取平均；组合数 ≤ 1000 时精确枚举，否则随机抽 max_combos 个。"""
    names = list(picks); rng = np.random.default_rng(seed); out = {}
    for m in range(1, len(names) + 1):
        combos = list(itertools.combinations(names, m))
        if max_combos and len(combos) > max(int(max_combos), 1000):
            combos = [combos[j] for j in rng.choice(len(combos), size=int(max_combos), replace=False)]
        out[m] = float(np.mean([max(s0[k] for n in c for k in picks[n]) for c in combos]))
    return out


def pick_origin(origins: Sequence[str]) -> str:
    """S3-Y 已有成员（m0–m4）提名过 → old（该配置在 S3-Y 池里）；否则 new（新成员或 10 成员集成）。"""
    return "old" if any(o in OLD_SEARCHERS for o in origins) else "new"


def verdict_s3z(new: float, new_b: float, prev: float = PREV) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    code = "S3Z-BEATS-BOTH" if (new > inf and new > ts) else ("S3Z-BEATS-TS2VEC" if new > ts else "S3Z-BELOW")
    code += "+UP" if new >= prev + 0.010 - 1e-12 else ("+DOWN" if new <= prev - 0.010 + 1e-12 else "+FLAT")
    return code + ("+E" if new >= new_b + 0.010 - 1e-12 else "")


# ================================================================ 阶段
def cmd_prep(args) -> None:
    outdir = Path(args.outdir); (outdir / "runs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(S3Y / "meta.json", outdir / "meta.json")
    shutil.copy2(S3Y / "runs" / "runs.jsonl", outdir / "runs" / "runs.jsonl")
    n = sum(1 for _ in open(outdir / "runs" / "runs.jsonl", encoding="utf-8"))
    print(f"[prep] meta.json 同 S3-Y；预置 S3-Y 已跑格 {n} 行", flush=True)


def cmd_search(args) -> None:
    configure().cmd_search(args)


def cmd_plan(args) -> None:
    configure().cmd_plan(args)


def cmd_select(args) -> None:
    configure().cmd_select(args)


def cmd_analyze(args) -> dict:
    y = configure(); v = _g("108_acts_s3v")
    outdir = Path(args.outdir)
    o = y.cmd_analyze(args)                                            # S3-Y 口径读数（写 analysis.json / tables.md，下面覆盖）
    rows = v._rows(outdir); sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
    prev = json.loads((S3Y / "analysis.json").read_text(encoding="utf-8"))
    D = v.DATASETS; ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    o["s3y"] = {a: {d: prev["per_arm"][a][d]["test"] for d in D} for a in ("NEW", "NEW_B")}
    o["s3y_means"] = {a: float(np.mean(list(o["s3y"][a].values()))) for a in o["s3y"]}
    o["origin"], o["curve"] = {}, {}
    for a in y.ARMS_Y:
        orig, curves = {}, []
        for d in D:
            sr = json.loads((outdir / "search" / f"{a}_{d}.json").read_text(encoding="utf-8")); t = v.off_tid(d)
            og = {e["cls_key"]: e["origins"] for e in sr["union"]}
            orig[d] = {"pick": pick_origin(og[sel[a][d]["pick"]]), "searchers": og[sel[a][d]["pick"]]}
            s0 = {e["cls_key"]: rows[(t, e["cls_key"], 0)] for e in sr["union"]}
            curves.append(searcher_curve(sr["searchers"], s0))
        o["origin"][a] = orig
        o["curve"][a] = {m: float(np.mean([c[m] for c in curves])) for m in curves[0]}
    m = o["means"]
    o["verdict"] = verdict_s3z(m["NEW_test"], m["NEW_B_test"])
    o["predictions"] = {"P1": m["NEW_test"] > PREV, "P2": m["NEW_test"] > inf and m["NEW_test"] > ts, "P3": m["NEW_U_best_s0"] > 0.6479,
                        "P4": m["NEW_sel_loss_s0"] <= 0.010, "P5": m["NEW_test"] >= m["NEW_B_test"] + 0.010,
                        "P6": sum(o["origin"]["NEW"][d]["pick"] == "new" for d in D) >= 2}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# S3-Z 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值；成员 10、搜索者 11）", "",
         "| 数据集 | NEW（S3-Z） | NEW（S3-Y） | NEW_B（S3-Z） | NEW_B（S3-Y） | NEW 纯比较器 | NEW 的 U 最好（种子 0） | NEW 选中来源 | TS2Vec | InfoTS_s |",
         "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for d in D:
        n, b = o["per_arm"]["NEW"][d], o["per_arm"]["NEW_B"][d]
        L.append(f"| {d} | {n['test']:.3f} | {o['s3y']['NEW'][d]:.3f} | {b['test']:.3f} | {o['s3y']['NEW_B'][d]:.3f} | {n['comp_test']:.3f} | "
                 f"{n['U_best_s0']:.3f}（|U| {n['U_n']}） | {o['origin']['NEW'][d]['pick']}（{'/'.join(o['origin']['NEW'][d]['searchers'])}） | "
                 f"{v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    L.append(f"| **均值** | **{m['NEW_test']:.4f}** | {o['s3y_means']['NEW']:.4f} | {m['NEW_B_test']:.4f} | {o['s3y_means']['NEW_B']:.4f} | "
             f"{m['NEW_comp_test']:.4f} | {m['NEW_U_best_s0']:.4f} | | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/6", "",
          f"- CV 选择损失（种子 0）：NEW {m['NEW_sel_loss_s0']:+.4f}、NEW_B {m['NEW_B_sel_loss_s0']:+.4f}；U 平均（种子 0）NEW {m['NEW_U_mean_s0']:.4f}",
          f"- S3-X 池考卷（10 成员集成）：NEW Spearman {m['NEW_xpool_sp']:+.3f}、regret@3 {m['NEW_xpool_r3']:.4f}；NEW_B {m['NEW_B_xpool_sp']:+.3f} / {m['NEW_B_xpool_r3']:.4f}",
          "- 搜索者数量曲线（池内最好，种子 0，子集平均）：",
          "  - NEW：" + "；".join(f"m={k} {x:.4f}" for k, x in o["curve"]["NEW"].items()),
          "  - NEW_B：" + "；".join(f"m={k} {x:.4f}" for k, x in o["curve"]["NEW_B"].items())]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = configure().build_parser()
    ap.description = __doc__
    ap.set_defaults(outdir=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("prep", cmd_prep), ("search", cmd_search), ("select", cmd_select), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    if args.plan:
        cmd_plan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
