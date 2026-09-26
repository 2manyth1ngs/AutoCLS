"""WP17-ACTS S3-J（results/acts/s3j/PROTOCOL.md）：轴瓶颈 T-CLSC + L3 源侧对抗扩充一轮。

阶段：
  --aug-search    服务器：--fold → search_aug/<fold>.json（S3-I 集成用 E 坐标在 30 个训练源任务上搜索，前 5 条 + 预测胜率）
  --plan-aug      服务器：→ runs_aug/plan_X.json（seed 0；去重至 S3-I）
  --train         服务器：--fold --seed → I′（= 96.cmd_train，扩充 = S3-G ∪ 本轮，按配置去重）
  --search        服务器：--fold → search/<fold>.json（= 96.cmd_search：O′ / A′ / AX′）
  --plan-runs     服务器：→ runs/plan_X.json（seed 0 / 1）
  --analyze       本地
  --smoke         本地冒烟
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3j"
PREV = ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h", "s3i")
AUG_TOP = 5
EXAM = ("O-lcb", "A-lcb", "AX-lcb")


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def union_aug(*augs: Mapping[int, Sequence[tuple]]) -> dict:
    """{任务下标: [(cls_key, 向量, 终点)]} 若干份 → {任务下标: (向量矩阵, 终点)}；同任务同配置只计一次（先到者为准）。"""
    out: dict[int, tuple] = {}
    keys: dict[int, list] = {}
    for a in augs:
        for i, lst in a.items():
            for k, v, r in lst:
                if k in {x[0] for x in keys.get(i, [])}:
                    continue
                keys.setdefault(i, []).append((k, np.asarray(v, np.float32), float(r)))
    for i, lst in keys.items():
        out[i] = (np.stack([x[1] for x in lst]), np.array([x[2] for x in lst], float))
    return out


def verdict_s3j(dO_m3, dO_p1, dA_m3, dA_p1, dL_m3, dX_m3) -> str:
    ok = lambda m3, p1: m3["d"] >= 0.010 and m3["p"] < 0.05 and m3["npos"] >= 4 and p1["d"] > 0  # noqa: E731
    sfx = lambda m3: m3["d"] >= 0.005 and m3["p"] < 0.05 and m3["npos"] >= 4  # noqa: E731
    code = "S3J-ARCH-OK" if ok(dO_m3, dO_p1) else "S3J-ARCH-NULL"
    if ok(dA_m3, dA_p1):
        code += "+E-WIN"
    elif dA_m3["d"] <= -0.010 and dA_m3["p"] < 0.05:
        code += "+E-HURT"
    else:
        code += "+E-NULL"
    if sfx(dL_m3):
        code += "+L3"
    if sfx(dX_m3):
        code += "+CONTENT"
    return code


# ================================================================ 公用
def _dirs(args) -> list[Path]:
    return [Path(getattr(args, k)) for k in PREV] + [Path(args.outdir)]


def _keyed(path: Path, val: Mapping, tasks: Sequence[str], enc) -> dict:
    from autocls.space.cls_space import CLS
    out = {}
    if not path.exists():
        return out
    for t, lst in json.loads(path.read_text(encoding="utf-8")).items():
        rows = [(e["cls_key"], enc.encode(CLS(**e["cls"])), val[(t, e["cls_key"], 0)]) for e in lst if (t, e["cls_key"], 0) in val]
        if rows:
            out[tasks.index(t)] = rows
    return out


def _aug_fn(args):
    def fn(meta, enc):
        val = _g("94_acts_s3g")._val_map(_dirs(args)); tasks = meta["tasks"]
        a = union_aug(_keyed(Path(args.s3g) / "search_aug" / f"{args.fold}.json", val, tasks, enc),
                      _keyed(Path(args.outdir) / "search_aug" / f"{args.fold}.json", val, tasks, enc))
        return a
    return fn


def _ens_i(args, fold: str, enc, dev):
    """S3-I 轴瓶颈集成（模型 + 各成员的 E 坐标）。"""
    import torch
    g96 = _g("96_acts_s3i"); s3i = Path(args.s3i)
    models, Ahats = [], []
    for sd in (args.seeds or _g("91_acts_s3e").ENS_SEEDS):
        m = g96.AxisComparator(enc.dim).to(dev); m.load_state_dict(torch.load(s3i / "models" / f"I_s{sd}_{fold}.pt", map_location=dev)); m.eval(); models.append(m)
        with np.load(s3i / "train" / f"I_s{sd}_{fold}.npz") as z:
            Ahats.append(torch.as_tensor(z["Ahat"], device=dev))
    return g96.AxisEnsemble(models, dev), Ahats


# ================================================================ 扩充（服务器）
def cmd_aug_search(args) -> None:
    g89, g91, g92, g94 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "94_acts_s3g"))
    from autocls.space.cls_space import CLS
    meta, arr, cfg, space, enc = g94._ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search_aug" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[aug-search] skip {fold}"); return
    ens, Ahats = _ens_i(args, fold, enc, dev)
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    tasks = meta["tasks"]; res, t0 = {}, time.time()
    for t in g94.aug_tasks(tasks, meta["groups"], meta["order"], fold, meta["val"][fold]):
        i = tasks.index(t); Ta = [A[i] for A in Ahats]
        rec = g92._search_task(ens, Ta, t, (("aug", g91.LAM, None),), space, enc, cfg, ks, gens, top=AUG_TOP)
        have = set(meta["keys"][t]); Vsh = arr[f"{t}/vec"][:g89.N_SH]; lst = []
        for e in rec["aug"]:
            if e["cls_key"] in have:
                continue
            e["pred_beat"] = g94._pred_beat(ens, Ta, enc.encode(CLS(**e["cls"])), Vsh); lst.append(e)
        res[t] = lst
        print(f"[aug-search] {fold} {t} {[x['cls_key'] for x in lst]} pred {[round(x['pred_beat'], 2) for x in lst]} {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def cmd_plan_aug(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search_aug").glob("*.json")):
        for t, lst in json.loads(f.read_text(encoding="utf-8")).items():
            noms.setdefault(t, {})[f.stem] = [e["cls_key"] for e in lst]
            byk.update({e["cls_key"]: e["cls"] for e in lst})
    _g("94_acts_s3g")._plan(args, noms, byk, (0,), "runs_aug", dirs=_dirs(args))


# ================================================================ 重训 / 考卷（服务器）
def cmd_train(args) -> None:
    _g("96_acts_s3i").cmd_train(args, aug_fn=_aug_fn(args))


def cmd_search(args) -> None:
    _g("96_acts_s3i").cmd_search(args)


def cmd_plan_runs(args) -> None:
    noms, byk = {}, {}
    for f in sorted((Path(args.outdir) / "search").glob("*.json")):
        for t, rec in json.loads(f.read_text(encoding="utf-8")).items():
            noms[t] = {a: [e["cls_key"] for e in rec[a]] for a in EXAM}
            for a in EXAM:
                byk.update({e["cls_key"]: e["cls"] for e in rec[a]})
    _g("94_acts_s3g")._plan(args, noms, byk, (0, 1), "runs", dirs=_dirs(args))


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import rankdata, wilcoxon
    from autocls.search.rank import symmetrize, wins_from_scores
    g84, g89, g91, g92, g93, g94, g96 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g", "96_acts_s3i"))
    outdir = Path(args.outdir); P_ = {k: Path(getattr(args, k)) for k in PREV}
    meta, arr = g89.load_data(P_["s3c"]); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    val = g94._val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    # ---- 扩充高估
    aug = []
    for s in g89.FOLDS:
        for t, lst in js(outdir / "search_aug" / f"{s}.json").items():
            i = tasks.index(t)
            for e in lst:
                v = val.get((t, e["cls_key"], 0))
                if v is not None:
                    aug.append((e["pred_beat"], g94.true_beat(v, end0[i]), v > end0[i].max()))
    aug_s = {"n": len(aug), "pred": float(np.mean([a[0] for a in aug])), "true": float(np.mean([a[1] for a in aug])),
             "over": float(np.mean([a[0] - a[1] for a in aug])), "beat_best_frac": float(np.mean([a[2] for a in aug]))}
    # ---- 考卷
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    load = lambda d, s: js(d / "search" / f"{s}.json")  # noqa: E731
    sj = {s: load(outdir, s) for s in g89.FOLDS}; si = {s: load(P_["s3i"], s) for s in g89.FOLDS}
    sgg = {s: load(P_["s3g"], s) for s in g89.FOLDS}; shh = {s: load(P_["s3h"], s) for s in g89.FOLDS}
    sf = {s: load(P_["s3f"], s) for s in g89.FOLDS}; sc_ = {s: js(P_["s3c"] / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("O′-lcb", "A′-lcb", "AX′-lcb", "O-lcb", "A-lcb", "AX-lcb", "B′-lcb", "H-lcb", "R")
    noms = {}
    for t, s in zip(ex, sg):
        f = sf[s][t]
        noms[t] = {"O′-lcb": sj[s][t]["O-lcb"], "A′-lcb": sj[s][t]["A-lcb"], "AX′-lcb": sj[s][t]["AX-lcb"],
                   "O-lcb": si[s][t]["O-lcb"], "A-lcb": si[s][t]["A-lcb"], "AX-lcb": si[s][t]["AX-lcb"],
                   "B′-lcb": sgg[s][t]["B′-lcb"], "H-lcb": shh[s][t]["H-lcb"], "R": f["R"] if f["new_row"] else sc_[s][t]["random"]}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    cat_p1, cat_m3, orc = [], [], []
    over = {"O′": [], "A′": []}
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        for arm, key in (("O-lcb", "O′"), ("A-lcb", "A′")):
            for e in sj[s][t][arm]:
                over[key].append(e["pred_beat"] - g94.true_beat((val.get((t, e["cls_key"], 0), fl) + val.get((t, e["cls_key"], 1), fl)) / 2, ref))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
        orc.append(float(end1[i, int(np.argmax(end0[i]))]))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"], P1["oracle"] = np.array(cat_p1), np.array(cat_m3), np.array(orc)

    def cmp(x, y):
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    pairs = (("O′-lcb", "B′-lcb"), ("A′-lcb", "B′-lcb"), ("A′-lcb", "A-lcb"), ("A′-lcb", "AX′-lcb"), ("O′-lcb", "O-lcb"),
             ("O′-lcb", "cat3"), ("A′-lcb", "cat3"), ("A′-lcb", "H-lcb"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    code = verdict_s3j(C["O′-lcb−B′-lcb"]["m3"], C["O′-lcb−B′-lcb"]["p1"], C["A′-lcb−B′-lcb"]["m3"], C["A′-lcb−B′-lcb"]["p1"],
                       C["A′-lcb−A-lcb"]["m3"], C["A′-lcb−AX′-lcb"]["m3"])
    # ---- 排序
    held = [i for i, g in enumerate(groups) if g in g89.FOLDS]; hg = groups[held]
    zj = {s: [np.load(outdir / "train" / f"I_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    zb2 = {s: [np.load(P_["s3g"] / "train" / f"blind_s{sd}_{s}.npz") for sd in g91.ENS_SEEDS] for s in g89.FOLDS}
    rO, spO = g96._rank(zj, "P", meta, arr, held); rA, spA = g96._rank(zj, "PA", meta, arr, held); rB, spB = g92._rank_readout(zb2, meta, arr, held)

    def wins_of(P):
        P = np.array(P, float); np.fill_diagonal(P, 0.5)
        return wins_from_scores(symmetrize(P)).astype(float)
    rx = np.zeros(len(held))
    for jj, i in enumerate(held):
        t = tasks[i]; keys = meta["keys"][t]; xs = []
        for k in range(16):
            Wx = np.mean([wins_of(z[f"XA/{t}"][k]) for z in zj[groups[i]]], axis=0)
            ox = sorted(range(N), key=lambda j: (-Wx[j], keys[j]))[:3]
            xs.append(end1[i, int(np.argmax(end0[i]))] - end1[i, ox[int(np.argmax(end0[i, ox]))]])
        rx[jj] = float(np.mean(xs))
    ref_i = js(P_["s3i"] / "analysis.json")
    rank = {"O′": {"regret3": float(rO.mean()), "sp": float(np.nanmedian(spO))}, "A′": {"regret3": float(rA.mean()), "sp": float(np.nanmedian(spA))},
            "B′": {"regret3": float(rB.mean()), "sp": float(np.nanmedian(spB))}, "S3-I": ref_i["rank"],
            "A′ cross": {"d": float((rx - rA).mean()), "ci": g84.cluster_bootstrap_ci(rx - rA, hg), "p_source": g84.source_wilcoxon(rx - rA, hg), "npos": g89.n_sources_positive(rx - rA, hg)}}
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "n_tasks": len(ex), "aug": aug_s, "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "missing": miss, "over": {k: float(np.mean(v)) for k, v in over.items()}, "over_ref_s3i": ref_i["over"], "rank": rank, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-J 读数（{code}）", "", "## 扩充（S3-I 集成在训练源上的 E 坐标搜索提名）", "",
         f"- {aug_s['n']} 条有真值；预测胜率 {f(aug_s['pred'])}、真值胜率 {f(aug_s['true'])}、高估 {aug_s['over']:+.3f}；超过任务目录最好者 {f(aug_s['beat_best_frac'])}", "",
         f"## 考卷（{len(ex)} 任务）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3", "oracle"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'].get(a))} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", f"自提名高估：O′ {out['over']['O′']:+.3f} / A′ {out['over']['A′']:+.3f}（S3-I：O {ref_i['over']['O']:+.3f} / A {ref_i['over']['A']:+.3f}）", "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    L += ["", "## 排序（56 留出任务）", "",
          f"- regret@3：O′ {f(rank['O′']['regret3'])} / A′ {f(rank['A′']['regret3'])} / B′ {f(rank['B′']['regret3'])}（S3-I：O {f(ref_i['rank']['O']['regret3'])} / A {f(ref_i['rank']['A']['regret3'])}）",
          f"- random-20 中位 Sp：O′ {f(rank['O′']['sp'])} / A′ {f(rank['A′']['sp'])} / B′ {f(rank['B′']['sp'])}（S3-I：O {f(ref_i['rank']['O']['sp'])} / A {f(ref_i['rank']['A']['sp'])}）",
          f"- A′ 换源 Δ（16 偏移）{rank['A′ cross']['d']:+.4f} [{f(rank['A′ cross']['ci'][0])}, {f(rank['A′ cross']['ci'][1])}]，源级 p {rank['A′ cross']['p_source']:.3f}，同号 {rank['A′ cross']['npos']}/6"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89, g96 = _g("89_acts_s3c"), _g("96_acts_s3i")
    od = Path(args.outdir); sgd, sid = od / "_s3g", od / "_s3i"; (sgd / "search_aug").mkdir(parents=True, exist_ok=True)
    fold = g89.FOLDS[0]
    (sgd / "search_aug" / f"{fold}.json").write_text("{}", encoding="utf-8")
    base = dict(vars(args)); base.update({"synthetic": True, "max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "s3g": str(sgd), "s3i": str(sid), "seeds": [0, 1]})
    for sd in (0, 1):
        g96.cmd_train(types.SimpleNamespace(**{**base, "outdir": str(sid), "fold": fold, "seed": sd}))
    ns = types.SimpleNamespace(**{**base, "fold": fold})
    cmd_aug_search(ns); cmd_plan_aug(ns)
    plan = json.loads((od / "runs_aug" / "plan_X.json").read_text(encoding="utf-8")); rng = np.random.default_rng(0)
    with open(od / "runs_aug" / "runs.jsonl", "w", encoding="utf-8") as fh:            # 合成真值（只为走通并入路径）
        for c in plan["cells"]:
            fh.write(json.dumps({"key": c["key"], "checkpoint_results": [{"acc": float(rng.uniform(0.2, 0.9))}]}) + "\n")
    for sd in (0, 1):
        cmd_train(types.SimpleNamespace(**{**base, "fold": fold, "seed": sd}))
    cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in EXAM)
    print(f"[smoke] OK（扩充格 {plan['n_cells']}）", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("aug-search", "plan-aug", "train", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    for k in PREV:
        ap.add_argument(f"--{k}", default=str(ROOT / "results" / "acts" / k))
    ap.add_argument("--s3b", default=str(ROOT / "results" / "acts" / "s3b"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("smoke", cmd_smoke), ("aug_search", cmd_aug_search), ("plan_aug", cmd_plan_aug), ("train", cmd_train), ("search", cmd_search),
                     ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
