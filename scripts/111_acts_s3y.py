"""WP17-ACTS S3-Y（results/acts/s3y/PROTOCOL.md）：结构 + 选择方式同时改，直接上部署口径。

结构：field 头（反对称 + 低秩，单配置分数 s(c, E)）+ 顶部排序权重（105 --head field --loss lambda）+ 每折 5 种子。
选择：按分数搜索（不用硬阈值胜场）；5 个成员各自 + 集成均值共 6 个搜索者各取前 2 → 并集 U；目标官方 TRAIN 分层 3 折 CV 选 1。
阶段：
  --prep          本地：meta.json（官方任务 + 3 个 tcv 任务 / 数据集）
  --search        服务器：--arm NEW|NEW_B --dataset D → search/<arm>_<D>.json（U、COMP-ONLY、各搜索者、S3-X 池考卷打分）
  --plan cv       服务器：U 每条 3 折 CV + 官方种子 0 → runs/plan_W.json
  --select        服务器：CV 均值选 1 → select.json
  --plan final    服务器：并入选中者与 COMP-ONLY 的官方种子 1 / 2
  --analyze       本地
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3y"
S3V = ROOT / "results" / "acts" / "s3v"
S3CD = ROOT / "results" / "acts" / "s3cd"
S3X = ROOT / "results" / "acts" / "s3x"
ARMS_Y = {"NEW": {"arm": "E", "head": "field"}, "NEW_B": {"arm": "B", "head": "field"}}
MEMBERS = (0, 1, 2, 3, 4)
K_TCV = 3
TOP_PER = 2
FINAL_SEEDS = (0, 1, 2)
FULL_REF = 0.5987


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


def tcv_tid(d: str, i: int) -> str:
    return f"{d}_whole_tcv{i}"


# ================================================================ 纯函数
def union_candidates(lists: Sequence[Sequence[str]]) -> list:
    out, seen = [], set()
    for lst in lists:
        for k in lst:
            if k not in seen:
                seen.add(k); out.append(k)
    return out


def select_cv(keys: Sequence[str], cv: Mapping[str, Sequence[float]], ens: Mapping[str, float]) -> str:
    """3 折 CV 均值最高；平局取集成分数高者，再按键序。"""
    cand = [k for k in keys if k in cv and len(cv[k])]
    return max(cand, key=lambda k: (round(float(np.mean(cv[k])), 12), float(ens.get(k, -np.inf)), -keys.index(k)))


def norm_stats(s: np.ndarray) -> tuple[float, float]:
    s = np.asarray(s, float); sd = float(s.std())
    return float(s.mean()), (sd if sd > 1e-12 else 1.0)


def score_evolve(fitness: Callable[[np.ndarray], np.ndarray], init: Sequence, space, enc, cfg, rng: np.random.Generator,
                 kp: int, gens: int, p_cross: float = 0.8, p_mut: float = 0.2) -> list:
    """按分数的进化（精英保留）：种群 kp，每代 kp 个后代（交叉 / 变异 / 复制，过结构性剪枝），并入后按适应度截断回 kp。"""
    ac = _g("acts_common")
    from autocls.labeling import cls_key
    from autocls.search.evolution import _make_offspring
    accept = lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg)  # noqa: E731

    def rank(cs):
        uniq, seen = [], set()
        for c in cs:
            k = cls_key(c)
            if k not in seen:
                seen.add(k); uniq.append(c)
        keys = [cls_key(c) for c in uniq]
        f = np.asarray(fitness(np.stack([enc.encode(c) for c in uniq]).astype(np.float32)), float)
        return [uniq[i] for i in sorted(range(len(uniq)), key=lambda i: (-f[i], keys[i]))]

    pop = rank(list(init))[:kp]
    for _g_ in range(int(gens)):
        seen = {cls_key(c) for c in pop}; kids = []
        for _ in range(kp):
            ch = _make_offspring(pop, rng, space, p_cross, p_mut, accept=accept)
            if ch is None or cls_key(ch) in seen:
                continue
            seen.add(cls_key(ch)); kids.append(ch)
        pop = rank(pop + kids)[:kp]
    return pop


def verdict_s3y(new: float, new_b: float) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    if new > inf and new > ts:
        code = "S3Y-BEATS-BOTH"
    elif new > ts:
        code = "S3Y-BEATS-TS2VEC"
    elif new >= FULL_REF + 0.010:
        code = "S3Y-UP"
    elif new <= FULL_REF - 0.010:
        code = "S3Y-DOWN"
    else:
        code = "S3Y-FLAT"
    return code + ("+E" if new >= new_b + 0.010 - 1e-12 else "")


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    ac, v = _g("acts_common"), _g("108_acts_s3v")
    from autocls.labeling import run_seed_for
    tm0 = json.loads((S3V / "meta.json").read_text(encoding="utf-8"))["tasks_meta"]; tm = {}
    for d in v.DATASETS:
        o = tm0[v.off_tid(d)]; tm[v.off_tid(d)] = o
        _X, y = ac.load_dataset_full(d)
        seed = int(run_seed_for(d, "s3y_tcv")); fid = ac.tcv_folds(np.asarray(y)[:int(o["n_train_official"])], K_TCV, seed)
        for i in range(K_TCV):
            tm[tcv_tid(d, i)] = {"dataset": d, "T": o["T"], "C": o["C"], "n_train_official": o["n_train_official"], "n_test_official": o["n_test_official"],
                                 "n_train": int((fid != i).sum()), "n_eval": int((fid == i).sum()), "eval_mode": "tcv",
                                 "tcv_k": K_TCV, "tcv_fold": i, "tcv_seed": seed, "tag": f"whole_tcv{i}"}
        print(f"[prep] {d}: TRAIN {o['n_train_official']} → 3 折验证 {[int((fid == i).sum()) for i in range(K_TCV)]}", flush=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "meta.json").write_text(json.dumps({"datasets": list(v.DATASETS), "tasks_meta": tm}, ensure_ascii=False, indent=1), encoding="utf-8")


# ================================================================ 搜索（服务器）
def cmd_search(args) -> None:
    import torch
    ac, g89, v, r105 = _g("acts_common"), _g("89_acts_s3c"), _g("108_acts_s3v"), _g("105_acts_s3r")
    from autocls.labeling import cls_key, run_seed_for
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    outdir, a, d, dev = Path(args.outdir), args.arm, args.dataset, args.device
    op = outdir / "search" / f"{a}_{d}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {a}_{d}"); return
    A = ARMS_Y[a]; t0 = time.time()
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    W = None
    if A["arm"] == "E":
        with np.load(Path(args.s3cd) / "E" / f"{d}.npz") as z:
            W = z["windows_repr"]
    models, Ts = [], []
    for s in MEMBERS:
        m = r105._build(A["arm"], enc.dim, s, dev, head=A["head"])
        m.load_state_dict(torch.load(outdir / "models" / f"{A['arm']}_s{s}_{d}.pt", map_location=dev)); m.eval()
        with torch.no_grad():
            Ts.append(None if W is None else g89.Scorer(m, "A", dev).task_vec(W))
        models.append(m)

    def raw_scores(V: np.ndarray) -> np.ndarray:                     # (成员, 配置)
        out = []
        with torch.no_grad():
            for m, T in zip(models, Ts):
                ss = []
                for st in range(0, len(V), 65536):
                    Vt = torch.as_tensor(np.asarray(V[st:st + 65536], np.float32), device=dev)
                    ss.append(m.score(Vt, None if T is None else T.unsqueeze(0).expand(len(Vt), -1)).float().cpu().numpy())
                out.append(np.concatenate(ss))
        return np.stack(out)

    tid = v.off_tid(d)
    cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(tid, "s3c_ks")), args.ks or g89.KS, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32)
    S = raw_scores(Vc); st = [norm_stats(S[q]) for q in range(len(MEMBERS))]

    def zs(V):
        R = raw_scores(V)
        return np.stack([(R[q] - st[q][0]) / st[q][1] for q in range(len(MEMBERS))])

    Zc = np.stack([(S[q] - st[q][0]) / st[q][1] for q in range(len(MEMBERS))])
    gens = args.gens if args.gens is not None else g89.GENS
    searchers = [("ens", lambda V: zs(V).mean(0), Zc.mean(0))] + [(f"m{q}", (lambda q: (lambda V: zs(V)[q]))(q), Zc[q]) for q in range(len(MEMBERS))]
    picks, byk = {}, {}
    for name, fit, f0 in searchers:
        init = [cands[j][0] for j in np.argsort(-f0, kind="stable")[:g89.KP]]
        fin = score_evolve(fit, init, space, enc, cfg, np.random.default_rng(run_seed_for(tid, f"s3y_evo_{name}")), g89.KP, gens, g89.P_CROSS, g89.P_MUT)
        picks[name] = [cls_key(c) for c in fin[:TOP_PER]]
        for c in fin[:TOP_PER]:
            byk[cls_key(c)] = c
    U = union_candidates([picks[n] for n, _f, _z in searchers])
    Vu = np.stack([enc.encode(byk[k]) for k in U]).astype(np.float32); Zu = zs(Vu)
    ens = dict(zip(U, Zu.mean(0).tolist()))
    xpool = {}
    xp = S3X / "pool.json"
    if xp.exists():
        pool = json.loads(xp.read_text(encoding="utf-8"))[d]
        Zx = zs(np.stack([enc.encode(CLS(**e["cls"])) for e in pool]).astype(np.float32))
        xpool = dict(zip([e["cls_key"] for e in pool], Zx.mean(0).tolist()))
    res = {"union": [{"cls_key": k, "cls": byk[k].to_dict(), "origins": [n for n in picks if k in picks[n]], "ens": ens[k]} for k in U],
           "comp_only": picks["ens"][0], "searchers": picks, "xpool": xpool, "ks": len(cands), "gens": gens, "seconds": time.time() - t0}
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[search] {a}_{d}：|U| = {len(U)}，COMP-ONLY {picks['ens'][0][:8]}，{time.time() - t0:.0f}s", flush=True)


# ================================================================ 计划 / 选择
def _search(outdir: Path, a: str, d: str) -> dict:
    return json.loads((outdir / "search" / f"{a}_{d}.json").read_text(encoding="utf-8"))


def _arms_ready(outdir: Path) -> list:
    v = _g("108_acts_s3v")
    return [a for a in ARMS_Y if all((outdir / "search" / f"{a}_{d}.json").exists() for d in v.DATASETS)]


def cmd_plan(args) -> None:
    ac, v, g35 = _g("acts_common"), _g("108_acts_s3v"), _g("35_protocol_alignment")
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); tm = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))["tasks_meta"]
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols()
    ent = []
    arms = _arms_ready(outdir)
    for a in arms:
        for d in v.DATASETS:
            for e in _search(outdir, a, d)["union"]:
                ent += [(tcv_tid(d, i), e["cls_key"], e["cls"], 0) for i in range(K_TCV)] + [(v.off_tid(d), e["cls_key"], e["cls"], 0)]
    if args.plan == "final":
        sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
        for a in arms:
            for d in v.DATASETS:
                byk = {e["cls_key"]: e["cls"] for e in _search(outdir, a, d)["union"]}
                for k in {sel[a][d]["pick"], sel[a][d]["comp_only"]}:
                    ent += [(v.off_tid(d), k, byk[k], s) for s in FINAL_SEEDS[1:]]
    cells, seen = [], set()
    for tid, k, c, s in ent:
        mode = tm[tid]["eval_mode"]
        cell = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**c), k, s, protos, cfg, extra={"eval": mode, "whole_mode": mode})
        if cell["key"] not in seen:
            seen.add(cell["key"]); cells.append(cell)
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    plan = {"phase": "W", "profile": "s1pp", "n_cells": len(cells), "n_tasks": len(tm), "probe_tasks": list(tm), "protocol": "ts2vec_official",
            "tasks_meta": tm, "cells": cells}
    (rd / "plan_W.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print(f"[plan {args.plan}] 臂 {arms}：{len(cells)} 格", flush=True)


def cmd_select(args) -> None:
    v = _g("108_acts_s3v")
    outdir = Path(args.outdir); rows = v._rows(outdir); sel = {}
    for a in _arms_ready(outdir):
        sel[a] = {}
        for d in v.DATASETS:
            sr = _search(outdir, a, d); keys = [e["cls_key"] for e in sr["union"]]; ens = {e["cls_key"]: e["ens"] for e in sr["union"]}
            cv = {k: [rows[(tcv_tid(d, i), k, 0)] for i in range(K_TCV) if (tcv_tid(d, i), k, 0) in rows] for k in keys}
            sel[a][d] = {"pick": select_cv(keys, cv, ens), "comp_only": sr["comp_only"], "cv": {k: float(np.mean(x)) if x else None for k, x in cv.items()},
                         "cv_n": {k: len(x) for k, x in cv.items()}}
    (outdir / "select.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[select] {json.dumps({a: {d[:6]: s['pick'][:8] for d, s in x.items()} for a, x in sel.items()}, ensure_ascii=False)}", flush=True)


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    from scipy.stats import spearmanr
    v = _g("108_acts_s3v"); x = _g("110_acts_s3x")
    outdir = Path(args.outdir); rows = v._rows(outdir); sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
    D = v.DATASETS; ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    xrows = v._rows(S3X); xpool = json.loads((S3X / "pool.json").read_text(encoding="utf-8"))
    o: dict = {"per_arm": {}, "means": {}}
    for a in ARMS_Y:
        res = {}
        for d in D:
            sr = _search(outdir, a, d); t = v.off_tid(d); keys = [e["cls_key"] for e in sr["union"]]; ens = np.array([e["ens"] for e in sr["union"]])
            s0 = np.array([rows[(t, k, 0)] for k in keys]); pk, co = sel[a][d]["pick"], sel[a][d]["comp_only"]
            xk = [e["cls_key"] for e in xpool[d]]; T3x = np.array([np.mean([xrows[(t, k, s)] for s in FINAL_SEEDS]) for k in xk])
            sx = np.array([sr["xpool"][k] for k in xk])
            res[d] = {"pick": pk, "test": float(np.mean([rows[(t, pk, s)] for s in FINAL_SEEDS])), "test_seeds": [rows[(t, pk, s)] for s in FINAL_SEEDS],
                      "comp_only": co, "comp_test": float(np.mean([rows[(t, co, s)] for s in FINAL_SEEDS])),
                      "U_n": len(keys), "U_best_s0": float(s0.max()), "U_mean_s0": float(s0.mean()), "sel_loss_s0": float(s0.max() - rows[(t, pk, 0)]),
                      "sp_U": float(spearmanr(ens, s0).correlation) if len(keys) > 2 else None,
                      "xpool_sp": float(spearmanr(sx, T3x).correlation), "xpool_r3": x.pool_regret(sx, T3x, 3), "xpool_rand3": x.random_k_regret(T3x, 3)}
        o["per_arm"][a] = res
        for k in ("test", "comp_test", "U_best_s0", "U_mean_s0", "sel_loss_s0", "xpool_sp", "xpool_r3", "xpool_rand3"):
            o["means"][f"{a}_{k}"] = float(np.mean([res[d][k] for d in D]))
    o["verdict"] = verdict_s3y(o["means"]["NEW_test"], o["means"]["NEW_B_test"])
    o["predictions"] = {"P1": o["means"]["NEW_test"] > FULL_REF, "P2": o["means"]["NEW_test"] > ts, "P3": o["means"]["NEW_test"] > inf,
                        "P4": o["means"]["NEW_test"] >= o["means"]["NEW_B_test"] + 0.010, "P5": o["means"]["NEW_sel_loss_s0"] <= 0.010,
                        "P6": o["means"]["NEW_xpool_sp"] > 0.2, "P7": o["means"]["NEW_comp_test"] > FULL_REF}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# S3-Y 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值）", "",
         "| 数据集 | NEW | NEW_B | NEW 纯比较器 | NEW_B 纯比较器 | NEW U 最好（种子 0） | TS2Vec | InfoTS_s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        n, b = o["per_arm"]["NEW"][d], o["per_arm"]["NEW_B"][d]
        L.append(f"| {d} | {n['test']:.3f} | {b['test']:.3f} | {n['comp_test']:.3f} | {b['comp_test']:.3f} | {n['U_best_s0']:.3f}（|U| {n['U_n']}） | "
                 f"{v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    m = o["means"]
    L.append(f"| **均值** | **{m['NEW_test']:.4f}** | {m['NEW_B_test']:.4f} | {m['NEW_comp_test']:.4f} | {m['NEW_B_comp_test']:.4f} | {m['NEW_U_best_s0']:.4f} | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/7", "",
          f"- CV 选择损失（U 最好 − 选中，种子 0）：NEW {m['NEW_sel_loss_s0']:+.4f}、NEW_B {m['NEW_B_sel_loss_s0']:+.4f}",
          f"- U 种子 0 平均：NEW {m['NEW_U_mean_s0']:.4f}、NEW_B {m['NEW_B_U_mean_s0']:.4f}",
          f"- S3-X 池考卷（21 条、3 种子真值）：NEW Spearman {m['NEW_xpool_sp']:+.3f}、regret@3 {m['NEW_xpool_r3']:.4f}；"
          f"NEW_B {m['NEW_B_xpool_sp']:+.3f} / {m['NEW_B_xpool_r3']:.4f}；随机 3 条 {m['NEW_xpool_rand3']:.4f}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "search", "select", "analyze", "resume"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--plan", choices=("cv", "final"))
    ap.add_argument("--arm", choices=tuple(ARMS_Y))
    ap.add_argument("--dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--s3cd", default=str(S3CD))
    ap.add_argument("--outdir", default=str(OUT))
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
