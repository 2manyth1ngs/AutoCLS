"""WP17-ACTS S3-X（results/acts/s3x/PROTOCOL.md）：21 条全空间提名全跑——是“找不到好配置”还是“排不对”。

每个 UEA 集的候选池 = S3-V 五臂 + S3-C-D A / D 的全空间搜索前 3（共 21 条，互不相同）。
阶段：
  --prep     本地：pool.json；plan_W.json（官方种子 0 / 1 / 2 + D_val 种子 0）；复用 S3-V / S3-C-D 已跑格 → runs/runs.jsonl
  --score    服务器：--arm <比较器> → scores/<arm>.json（每个数据集 21 条池的概率矩阵与循环赛胜场）
  --analyze  本地：R1 / R2、描述项、读数码
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from math import comb
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3x"
S3V = ROOT / "results" / "acts" / "s3v"
S3CD = ROOT / "results" / "acts" / "s3cd"
S3C = ROOT / "results" / "acts" / "s3c"
SEEDS = (0, 1, 2)
ARM_ORDER = ("FULL", "WO_SETTF", "WO_TS2VEC", "WO_SHARED", "BLIND", "A", "D")
RANKERS = {a: ("s3v" if a not in ("A", "D") else "s3cd") for a in ARM_ORDER}      # 提名来自哪次部署
FULL_REF = 0.5987


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def build_pool(searches: Mapping[str, Mapping[str, Sequence[Mapping]]], arms: Sequence[str], datasets: Sequence[str]) -> dict:
    """{数据集: [{cls_key, cls, arms}]}：按臂顺序并入各臂提名，同键合并来源。"""
    out = {}
    for d in datasets:
        lst, idx = [], {}
        for a in arms:
            for e in searches.get(a, {}).get(d, []):
                k = e["cls_key"]
                if k in idx:
                    lst[idx[k]]["arms"].append(a)
                else:
                    idx[k] = len(lst); lst.append({"cls_key": k, "cls": e["cls"], "arms": [a]})
        out[d] = lst
    return out


def import_rows(cells: Sequence[Mapping], old: Sequence[Mapping]) -> tuple[list, int, int]:
    """已跑格复用：键相同直接复用；键不同但（任务，配置键，种子，评估口径）相同则改写键后复用；出错行不复用。"""
    by_key = {r["key"]: r for r in old if not r.get("error")}
    loose = {}
    for r in old:
        if not r.get("error"):
            loose.setdefault((r.get("task_id"), r.get("cls_key"), int(r.get("seed", 0)), r.get("eval")), r)
    rows, nd, nl = [], 0, 0
    for c in cells:
        if c["key"] in by_key:
            rows.append(dict(by_key[c["key"]])); nd += 1
            continue
        r = loose.get((c["task_id"], c["cls_key"], int(c["seed"]), c.get("eval")))
        if r is not None:
            r2 = dict(r); r2["key"] = c["key"]; r2["reused_from"] = r["key"]
            rows.append(r2); nl += 1
    return rows, nd, nl


def pool_regret(scores: np.ndarray, truth: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    truth = np.asarray(truth, float)
    return float(truth.max() - truth[top].max())


def random_k_regret(truth: np.ndarray, k: int = 3) -> float:
    """21 选 k 的期望 regret（精确）：P(max = 第 j 小) = C(j−1, k−1) / C(n, k)。"""
    v = np.sort(np.asarray(truth, float)); n = len(v)
    if k >= n:
        return 0.0
    e = sum(comb(j - 1, k - 1) / comb(n, k) * v[j - 1] for j in range(k, n + 1))
    return float(v[-1] - e)


def honest_best(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """按种子 0、1 均值选 1，报种子 2（去赢家诅咒）。"""
    j = int(np.argmax((np.asarray(s0, float) + np.asarray(s1, float)) / 2))
    return float(np.asarray(s2, float)[j])


def self_pref(scores: np.ndarray, own: Sequence[int], k: int = 3) -> int:
    top = set(np.argsort(-np.asarray(scores, float), kind="stable")[:k].tolist())
    return int(len(top & set(int(i) for i in own)))


def verdict_s3x(find_val: float, full_r3: float, rand_r3: float) -> str:
    ts = float(np.mean(list(_g("108_acts_s3v").TS2VEC.values())))
    find = "FIND-OK" if find_val > ts else ("FIND-PARTIAL" if find_val > FULL_REF else "FIND-FAIL")
    rank = "RANK-OK" if full_r3 <= 0.5 * rand_r3 else ("RANK-FAIL" if full_r3 >= 0.8 * rand_r3 else "RANK-WEAK")
    return f"S3X-{find}+{rank}"


# ================================================================ prep（本地）
def _searches() -> dict:
    v = _g("108_acts_s3v"); out = {}
    for a in ARM_ORDER:
        src = S3V if RANKERS[a] == "s3v" else S3CD
        out[a] = {d: json.loads((src / "search" / f"{a}_{d}.json").read_text(encoding="utf-8"))["nominated"] for d in v.DATASETS}
    return out


def cmd_prep(args) -> None:
    ac, v, g35 = _g("acts_common"), _g("108_acts_s3v"), _g("35_protocol_alignment")
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pool = build_pool(_searches(), ARM_ORDER, v.DATASETS)
    (outdir / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = json.loads((S3V / "meta.json").read_text(encoding="utf-8")); tm = meta["tasks_meta"]
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols()
    cells, seen = [], set()
    for d in v.DATASETS:
        for e in pool[d]:
            for tid, s in [(v.off_tid(d), s) for s in SEEDS] + [(v.val_tid(d), 0)]:
                mode = tm[tid]["eval_mode"]
                c = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**e["cls"]), e["cls_key"], s, protos, cfg, extra={"eval": mode, "whole_mode": mode})
                if c["key"] not in seen:
                    seen.add(c["key"]); cells.append(c)
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    plan = {"phase": "W", "profile": "s1pp", "n_cells": len(cells), "n_tasks": len(tm), "probe_tasks": list(tm), "protocol": "ts2vec_official",
            "tasks_meta": tm, "cells": cells}
    (rd / "plan_W.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    old = []
    for p in (S3V / "runs" / "runs.jsonl", S3CD / "runs" / "runs.jsonl"):
        old += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, nd, nl = import_rows(cells, old)
    with open(rd / "runs.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] 池 {[len(pool[d]) for d in v.DATASETS]}；计划 {len(cells)} 格；复用 {nd} 直接 + {nl} 改键；新跑 {len(cells) - nd - nl}", flush=True)


# ================================================================ 池内打分（服务器）
def cmd_score(args) -> None:
    import torch
    g89, v = _g("89_acts_s3c"), _g("108_acts_s3v")
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    ac = _g("acts_common")
    outdir, a, dev = Path(args.outdir), args.arm, args.device
    op = outdir / "scores" / f"{a}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[score] skip {a}"); return
    pool = json.loads((outdir / "pool.json").read_text(encoding="utf-8"))
    enc = CLSEncoder(ac.acts_space(ac.load_acts_config(profile="s2")), mode="binned")
    res = {}
    for d in v.DATASETS:
        V = np.stack([enc.encode(CLS(**e["cls"])) for e in pool[d]]).astype(np.float32); keys = [e["cls_key"] for e in pool[d]]
        if a in ("A", "D"):                           # S3-C 单模型（= 90.cmd_search 的加载方式）
            m = g89.build_model(a, enc.dim, dev); m.load_state_dict(torch.load(S3C / "models" / f"{a}_{d}.pt", map_location=dev)); m.eval()
            sc = g89.Scorer(m, a, dev); T = None
            if a == "A":
                with np.load(S3CD / "E" / f"{d}.npz") as z:
                    W = z["windows_repr"]
                with torch.no_grad():
                    T = sc.task_vec(W)
            P = sc.pmat(V, T)
        else:                                         # S3-V 各臂的 3 种子集成（= 108._ensemble）
            scs, Ts = v._ensemble(a, d, dev, enc.dim, S3V)
            P = np.mean([s.pmat(V, T) for s, T in zip(scs, Ts)], 0)
        order = g89.rr_order(P, keys)
        wins = np.zeros(len(keys)); wins[order] = np.arange(len(keys))[::-1]          # 循环赛名次分（高 = 靠前）
        res[d] = {"keys": keys, "P": P.round(5).tolist(), "rank_score": wins.tolist()}
        print(f"[score] {a} {d} 前 3 {[keys[j][:8] for j in order[:3]]}", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    from scipy.stats import spearmanr
    v = _g("108_acts_s3v")
    outdir = Path(args.outdir); rows = v._rows(outdir); pool = json.loads((outdir / "pool.json").read_text(encoding="utf-8"))
    D = v.DATASETS; ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    o: dict = {"per_ds": {}, "rankers": {}}
    TR, VAL = {}, {}
    for d in D:
        keys = [e["cls_key"] for e in pool[d]]
        S = np.array([[rows[(v.off_tid(d), k, s)] for s in SEEDS] for k in keys])
        TR[d], VAL[d] = S, np.array([rows[(v.val_tid(d), k, 0)] for k in keys])
        T3 = S.mean(1)
        o["per_ds"][d] = {"T3_best": float(T3.max()), "T3_mean": float(T3.mean()), "T3_median": float(np.median(T3)),
                          "honest": honest_best(S[:, 0], S[:, 1], S[:, 2]), "rand3_regret": random_k_regret(T3, 3),
                          "val_pick_T3": float(T3[int(np.argmax(VAL[d]))]), "val_r3": pool_regret(VAL[d], T3, 3),
                          "n_beat_ts2vec": int((T3 > v.TS2VEC[d]).sum()), "n_beat_infots": int((T3 > v.INFOTS_S[d]).sum()),
                          "best_key": keys[int(np.argmax(T3))], "best_arms": pool[d][int(np.argmax(T3))]["arms"]}
    o["honest_mean"] = float(np.mean([o["per_ds"][d]["honest"] for d in D]))
    o["T3_best_mean"] = float(np.mean([o["per_ds"][d]["T3_best"] for d in D]))
    o["rand3_regret"] = float(np.mean([o["per_ds"][d]["rand3_regret"] for d in D]))
    o["val_pick_mean"] = float(np.mean([o["per_ds"][d]["val_pick_T3"] for d in D]))
    o["val_r3"] = float(np.mean([o["per_ds"][d]["val_r3"] for d in D]))
    for a in ARM_ORDER:
        p = outdir / "scores" / f"{a}.json"
        if not p.exists():
            continue
        sc = json.loads(p.read_text(encoding="utf-8")); r = {"sp": [], "r3": [], "self": [], "own_best": []}
        for d in D:
            keys = [e["cls_key"] for e in pool[d]]; assert sc[d]["keys"] == keys
            s = np.array(sc[d]["rank_score"]); T3 = TR[d].mean(1)
            own = [i for i, e in enumerate(pool[d]) if a in e["arms"]]
            r["sp"].append(float(spearmanr(s, T3).correlation)); r["r3"].append(pool_regret(s, T3, 3))
            r["self"].append(self_pref(s, own, 3)); r["own_best"].append(float(T3[own].max()))
        o["rankers"][a] = {"sp": float(np.mean(r["sp"])), "r3": float(np.mean(r["r3"])), "self": float(np.mean(r["self"])),
                           "own_best_mean": float(np.mean(r["own_best"])), "per_ds_sp": dict(zip(D, r["sp"])), "per_ds_r3": dict(zip(D, r["r3"]))}
    full = o["rankers"].get("FULL")
    o["verdict"] = verdict_s3x(o["honest_mean"], full["r3"], o["rand3_regret"]) if full else None
    if full:
        o["predictions"] = {"P1": o["honest_mean"] > ts, "P2": full["r3"] >= 0.8 * o["rand3_regret"], "P3": full["sp"] < 0.2,
                            "P4": o["val_pick_mean"] > ts, "P5": float(np.mean([x["self"] for x in o["rankers"].values()])) >= 2.0}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# S3-X 读数（每个数据集 21 条全空间提名，官方 TEST 3 种子）", "",
         "| 数据集 | 21 条均值 | 中位 | 最好（T3） | 诚实最好（种子 2） | D_val 选 1 | 超 TS2Vec 条数 | TS2Vec | InfoTS_s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        q = o["per_ds"][d]
        L.append(f"| {d} | {q['T3_mean']:.3f} | {q['T3_median']:.3f} | {q['T3_best']:.3f} | {q['honest']:.3f} | {q['val_pick_T3']:.3f} | "
                 f"{q['n_beat_ts2vec']}/21 | {v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    L.append(f"| **均值** | {np.mean([o['per_ds'][d]['T3_mean'] for d in D]):.4f} | | **{o['T3_best_mean']:.4f}** | **{o['honest_mean']:.4f}** | "
             f"{o['val_pick_mean']:.4f} | | {ts:.4f} | {inf:.4f} |")
    L += ["", "| 比较器 | 池内 Spearman | 池内 regret@3 | 自己 3 条进池内前 3 | 自己 3 条里最好（T3） |", "|---|---:|---:|---:|---:|"]
    for a, x in o["rankers"].items():
        L.append(f"| {a} | {x['sp']:+.3f} | {x['r3']:.4f} | {x['self']:.2f} | {x['own_best_mean']:.4f} |")
    L += [f"| 随机 3 条（期望） | 0 | {o['rand3_regret']:.4f} | | |", f"| D_val 排序（测量） | | {o['val_r3']:.4f} | | |", "",
          f"读数码：**{o['verdict']}**" + (f"；预测 {sum(o['predictions'].values())}/5" if full else "")]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "score", "analyze", "resume"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prep:
        cmd_prep(args)
    if args.score:
        cmd_score(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
