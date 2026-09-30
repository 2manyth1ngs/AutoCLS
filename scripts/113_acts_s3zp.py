"""WP17-ACTS S3-Z′（results/acts/s3zp/PROTOCOL.md）：两阶段选择——S3-Z 的候选池不变，只降低选择噪声。

第一阶段 = S3-Z 已跑的 3 折 CV（折种子 = S3-Y / S3-Z，训练种子 0）取前 4；
第二阶段 = 前 4 各再做 2 次重复 3 折 CV（新折划分、训练种子 r = 1 / 2）；按 9 次折评估均值选 1；
决赛 4 条都补齐官方种子 0 / 1 / 2 TEST（描述；种子 0 复用 S3-Z）。不训练比较器、不搜索。
阶段：
  --prep     本地：meta.json（S3-Z 任务 + 重复折任务）、finalists.json、runs/plan_W.json、预置 S3-Z 已跑格
  --analyze  本地：两阶段 / 单阶段 / 决赛 oracle、读数码
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3zp"
S3Z = ROOT / "results" / "acts" / "s3z"
K_FIN = 4
REPS = (1, 2)
K_TCV = 3
FINAL_SEEDS = (0, 1, 2)
PREV = 0.6279                                        # S3-Z NEW（同池单阶段选择）部署均值
ARMS = ("NEW", "NEW_B")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def rep_tid(d: str, i: int, r: int) -> str:
    return f"{d}_whole_tcv{i}" if r == 0 else f"{d}_whole_tcv{i}_r{r}"


def rep_fold_seed(d: str, r: int) -> int:
    from autocls.labeling import run_seed_for
    return int(run_seed_for(d, "s3y_tcv" if r == 0 else f"s3zp_tcv_r{r}"))


def finalists(keys: Sequence[str], cv1: Mapping[str, float], ens: Mapping[str, float], k: int = K_FIN) -> list:
    """第一阶段 3 折 CV 均值取前 k；平局取集成分数高者，再按原顺序。"""
    ks = [x for x in keys if cv1.get(x) is not None]
    return sorted(ks, key=lambda x: (-round(float(cv1[x]), 12), -float(ens.get(x, -np.inf)), keys.index(x)))[:k]


def two_stage_pick(fin: Sequence[str], cv9: Mapping[str, Sequence[float]], cv1: Mapping[str, float], ens: Mapping[str, float]) -> str:
    """9 次折评估均值最高；平局依次取第一阶段均值高者、集成分数高者、决赛名单靠前者。"""
    return max(fin, key=lambda x: (round(float(np.mean(cv9[x])), 12), round(float(cv1[x]), 12), float(ens.get(x, -np.inf)), -list(fin).index(x)))


def verdict_s3zp(new: float, new_b: float, prev: float = PREV) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    code = "S3ZP-BEATS-BOTH" if (new > inf and new > ts) else ("S3ZP-BEATS-TS2VEC" if new > ts else "S3ZP-BELOW")
    code += "+UP" if new >= prev + 0.010 - 1e-12 else ("+DOWN" if new <= prev - 0.010 + 1e-12 else "+FLAT")
    return code + ("+E" if new >= new_b + 0.010 - 1e-12 else "")


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    ac, v, g35 = _g("acts_common"), _g("108_acts_s3v"), _g("35_protocol_alignment")
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); (outdir / "runs").mkdir(parents=True, exist_ok=True)
    tm = json.loads((S3Z / "meta.json").read_text(encoding="utf-8"))["tasks_meta"]
    for d in v.DATASETS:
        o = tm[v.off_tid(d)]; _X, y = ac.load_dataset_full(d)
        for r in REPS:
            seed = rep_fold_seed(d, r); fid = ac.tcv_folds(np.asarray(y)[:int(o["n_train_official"])], K_TCV, seed)
            for i in range(K_TCV):
                tm[rep_tid(d, i, r)] = {"dataset": d, "T": o["T"], "C": o["C"], "n_train_official": o["n_train_official"], "n_test_official": o["n_test_official"],
                                        "n_train": int((fid != i).sum()), "n_eval": int((fid == i).sum()), "eval_mode": "tcv",
                                        "tcv_k": K_TCV, "tcv_fold": i, "tcv_seed": seed, "tag": f"whole_tcv{i}_r{r}"}
    (outdir / "meta.json").write_text(json.dumps({"datasets": list(v.DATASETS), "tasks_meta": tm}, ensure_ascii=False, indent=1), encoding="utf-8")
    sel = json.loads((S3Z / "select.json").read_text(encoding="utf-8")); fin_all, ent = {}, []
    for a in ARMS:
        fin_all[a] = {}
        for d in v.DATASETS:
            sr = json.loads((S3Z / "search" / f"{a}_{d}.json").read_text(encoding="utf-8"))
            keys = [e["cls_key"] for e in sr["union"]]; ens = {e["cls_key"]: e["ens"] for e in sr["union"]}; byk = {e["cls_key"]: e["cls"] for e in sr["union"]}
            fin = finalists(keys, sel[a][d]["cv"], ens)
            assert fin[0] == sel[a][d]["pick"], f"{a} {d}：决赛第 1 名应等于 S3-Z 单阶段选中者"
            fin_all[a][d] = [{"cls_key": k, "cls": byk[k], "cv1": sel[a][d]["cv"][k], "ens": ens[k]} for k in fin]
            for k in fin:
                ent += [(rep_tid(d, i, r), k, byk[k], r) for r in REPS for i in range(K_TCV)] + [(v.off_tid(d), k, byk[k], s) for s in FINAL_SEEDS]
    (outdir / "finalists.json").write_text(json.dumps(fin_all, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols(); cells, seen = [], set()
    for tid, k, c, s in ent:
        mode = tm[tid]["eval_mode"]
        cell = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**c), k, s, protos, cfg, extra={"eval": mode, "whole_mode": mode})
        if cell["key"] not in seen:
            seen.add(cell["key"]); cells.append(cell)
    plan = {"phase": "W", "profile": "s1pp", "n_cells": len(cells), "n_tasks": len(tm), "probe_tasks": list(tm), "protocol": "ts2vec_official",
            "tasks_meta": tm, "cells": cells}
    (outdir / "runs" / "plan_W.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(S3Z / "runs" / "runs.jsonl", outdir / "runs" / "runs.jsonl")
    done = {json.loads(l)["key"] for l in open(outdir / "runs" / "runs.jsonl", encoding="utf-8")}
    print(f"[prep] 决赛 {len(ARMS)} × {len(v.DATASETS)} × {K_FIN}；计划 {len(cells)} 格，复用 {len(seen & done)}，新跑 {len(seen - done)}", flush=True)


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    from scipy.stats import spearmanr
    v = _g("108_acts_s3v")
    outdir = Path(args.outdir); rows = v._rows(outdir); fin_all = json.loads((outdir / "finalists.json").read_text(encoding="utf-8"))
    D = v.DATASETS; ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    o: dict = {"per_arm": {}, "means": {}}
    for a in ARMS:
        res = {}
        for d in D:
            F = fin_all[a][d]; fin = [e["cls_key"] for e in F]; cv1 = {e["cls_key"]: e["cv1"] for e in F}; ens = {e["cls_key"]: e["ens"] for e in F}
            cv9 = {k: [rows[(rep_tid(d, i, r), k, r)] for r in (0,) + REPS for i in range(K_TCV)] for k in fin}
            T3 = {k: float(np.mean([rows[(v.off_tid(d), k, s)] for s in FINAL_SEEDS])) for k in fin}
            p2 = two_stage_pick(fin, cv9, cv1, ens); p1 = fin[0]; orc = max(fin, key=lambda k: T3[k])
            sr = json.loads((S3Z / "search" / f"{a}_{d}.json").read_text(encoding="utf-8"))
            s0 = {e["cls_key"]: rows[(v.off_tid(d), e["cls_key"], 0)] for e in sr["union"]}
            res[d] = {"pick2": p2, "pick1": p1, "oracle4": orc, "test2": T3[p2], "test1": T3[p1], "test_oracle4": T3[orc], "changed": p2 != p1,
                      "fin_has_Ubest": max(s0, key=s0.get) in fin, "cv9": {k: float(np.mean(x)) for k, x in cv9.items()}, "T3": T3,
                      "sp_cv9_T3": float(spearmanr([np.mean(cv9[k]) for k in fin], [T3[k] for k in fin]).correlation)}
        o["per_arm"][a] = res
        for k in ("test2", "test1", "test_oracle4"):
            o["means"][f"{a}_{k}"] = float(np.mean([res[d][k] for d in D]))
        o["means"][f"{a}_changed"] = int(sum(res[d]["changed"] for d in D)); o["means"][f"{a}_fin_has_Ubest"] = int(sum(res[d]["fin_has_Ubest"] for d in D))
        o["means"][f"{a}_sp"] = float(np.nanmean([res[d]["sp_cv9_T3"] for d in D]))
    m = o["means"]
    o["verdict"] = verdict_s3zp(m["NEW_test2"], m["NEW_B_test2"])
    o["predictions"] = {"P1": m["NEW_test2"] > PREV, "P2": m["NEW_test2"] > 0.6330, "P3": m["NEW_test2"] > inf and m["NEW_test2"] > ts,
                        "P4": m["NEW_changed"] + m["NEW_B_changed"] >= 3, "P5": m["NEW_B_test2"] > 0.6377,
                        "P6": m["NEW_fin_has_Ubest"] + m["NEW_B_fin_has_Ubest"] >= 8}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# S3-Z′ 部署口径读数（S3-Z 候选池；官方 TEST 3 种子均值）", "",
         "| 数据集 | NEW 两阶段 | NEW 单阶段（S3-Z） | NEW 决赛 oracle | NEW_B 两阶段 | NEW_B 单阶段 | NEW_B 决赛 oracle | TS2Vec | InfoTS_s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        n, b = o["per_arm"]["NEW"][d], o["per_arm"]["NEW_B"][d]
        L.append(f"| {d} | {n['test2']:.3f}{' *' if n['changed'] else ''} | {n['test1']:.3f} | {n['test_oracle4']:.3f} | {b['test2']:.3f}{' *' if b['changed'] else ''} | "
                 f"{b['test1']:.3f} | {b['test_oracle4']:.3f} | {v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    L.append(f"| **均值** | **{m['NEW_test2']:.4f}** | {m['NEW_test1']:.4f} | {m['NEW_test_oracle4']:.4f} | {m['NEW_B_test2']:.4f} | {m['NEW_B_test1']:.4f} | "
             f"{m['NEW_B_test_oracle4']:.4f} | {ts:.4f} | {inf:.4f} |")
    L += ["", "\\* = 两阶段改变了选择", "", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/6", "",
          f"- 两阶段改变选择：NEW {m['NEW_changed']}/6、NEW_B {m['NEW_B_changed']}/6；决赛名单含 U 种子 0 最好者：NEW {m['NEW_fin_has_Ubest']}/6、NEW_B {m['NEW_B_fin_has_Ubest']}/6",
          f"- 决赛内 9 次 CV 均值与 3 种子 TEST 的 Spearman（逐集平均）：NEW {m['NEW_sp']:+.2f}、NEW_B {m['NEW_B_sp']:+.2f}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prep:
        cmd_prep(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
