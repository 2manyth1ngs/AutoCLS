"""WP17-ACTS S3-W（results/acts/s3w/PROTOCOL.md）：保守式比较器——训练时内置全空间搜索，以部署精度为目标。

训练在 105（--cons α）；本脚本负责扩充考卷、扩充池评估、α 选择与部署（部署复用 108，臂 = CONS / CONS_B）。
阶段：
  --prep-aug      本地：S3-G / S3-J 扩充提名中有 seed-0 真值者（按任务 × 配置键去重、剔已标）→ augpool.npz / augpool.json
  --eval-ext      服务器：--src <105 输出目录> --tag <名> --arm E|B → ext/<tag>_<arm>.json
                  （逐折 3 种子 P 均值；每个扩充任务的扩充池 regret@3、高估、扩充 vs shared 判对率、前 3 中扩充配置数）
  --select-alpha  服务器：训练源扩充任务的 regret@3 逐折选 α → alpha.json；选中模型与训练记录复制到 sel_E / sel_B
  --search / --plan val|final / --select   服务器：= 108 同名阶段（同 Ks / 进化种子、同 D_val、同种子）
  --analyze       本地：部署表、机制门、读数码
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

OUT = ROOT / "results" / "acts" / "s3w"
S3C = ROOT / "results" / "acts" / "s3c"
S3B = ROOT / "results" / "acts" / "s3b"
S3V = ROOT / "results" / "acts" / "s3v"
AUG_EXPS = ("s3g", "s3j")
ALPHAS = (0.3, 1.0, 3.0)
SEEDS = (0, 1, 2)
N_SH = 30
FULL_REF = 0.5987                                   # S3-V FULL（α = 0）部署均值
ARMS_W = {
    "CONS": {"dir": "results/acts/s3w/sel_E", "arm": "E", "e_mode": "settf", "input": "E"},
    "CONS_B": {"dir": "results/acts/s3w/sel_B", "arm": "B", "e_mode": "settf", "input": None},
}


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


def atag(a: float) -> str:
    return f"a{a:g}"


# ================================================================ 纯函数
def regret_ext(P: np.ndarray, keys: Sequence[str], s0: np.ndarray, k: int = 3) -> float:
    """扩充池 regret@k：循环赛前 k 里 seed-0 真值最好者 vs 池内最好者。"""
    o = _g("89_acts_s3c").rr_order(P, keys)[:k]
    s0 = np.asarray(s0, float)
    return float(s0.max() - s0[o].max())


def over_rate(Psh: np.ndarray, v_aug: np.ndarray, end0_sh: np.ndarray) -> float:
    """高估（= S3-G）：预测胜率（P(扩充 ≻ shared) 全体均值）− 真值胜率（seed-0 真值高于 shared end0 的比例，按扩充配置平均）。"""
    end0_sh = np.asarray(end0_sh, float)
    true = float(np.mean([np.mean(float(v) > end0_sh) for v in np.asarray(v_aug, float)]))
    return float(np.mean(Psh)) - true


def acc_aug_sh(Psh: np.ndarray, v_aug: np.ndarray, end0_sh: np.ndarray) -> float:
    """扩充 × shared 配对的判对率（真值平局剔除）。"""
    d = np.asarray(v_aug, float)[:, None] - np.asarray(end0_sh, float)[None, :]
    m = d != 0
    return float(np.mean((np.asarray(Psh)[m] > 0.5) == (d[m] > 0))) if m.any() else float("nan")


def dedupe_aug(ents: Sequence[tuple], labeled: Mapping[str, set]) -> list:
    """(任务, 配置键, …) 先到先留；与该任务已标配置同键者剔除。"""
    seen, out = set(), []
    for e in ents:
        t, k = e[0], e[1]
        if (t, k) in seen or k in labeled.get(t, set()):
            continue
        seen.add((t, k)); out.append(e)
    return out


def select_alpha(scores: Mapping[float, float], tol: float = 0.002) -> float:
    """regret 最小；与最小值相差 ≤ tol 的取较小 α。"""
    m = min(scores.values())
    return min(a for a, v in scores.items() if v <= m + tol + 1e-12)


def verdict_s3w(cons: float, cons_b: float, mech: bool) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    if cons > inf and cons > ts:
        code = "S3W-BEATS-BOTH"
    elif cons > ts:
        code = "S3W-BEATS-TS2VEC"
    elif cons >= FULL_REF + 0.010:
        code = "S3W-UP"
    elif cons <= FULL_REF - 0.010:
        code = "S3W-DOWN"
    else:
        code = "S3W-FLAT"
    if cons >= cons_b + 0.010 - 1e-12:
        code += "+E"
    return code + ("+MECH" if mech else "")


# ================================================================ 扩充考卷（本地）
def cmd_prep_aug(args) -> None:
    ac, g89 = _g("acts_common"), _g("89_acts_s3c")
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    meta, _arr = g89.load_data(S3C); tasks = meta["tasks"]
    cfg = ac.load_acts_config(profile="s2"); enc = CLSEncoder(ac.acts_space(cfg), mode="binned")
    ents, miss = [], 0
    for exp in AUG_EXPS:
        d = ROOT / "results" / "acts" / exp
        plan = json.loads((d / "runs_aug" / "plan_X.json").read_text(encoding="utf-8"))
        rows = {}
        for line in (d / "runs_aug" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line); rows[r["key"]] = r
        truth = {(c["task_id"], c["cls_key"]): ac.endpoint_of(rows.get(c["key"])) for c in plan["cells"] if int(c["seed"]) == 0}
        for f in sorted((d / "search_aug").glob("*.json")):
            for t, lst in json.loads(f.read_text(encoding="utf-8")).items():
                for e in lst:
                    v = truth.get((t, e["cls_key"]))
                    if v is None:
                        miss += 1; continue
                    ents.append((t, e["cls_key"], enc.encode(CLS(**e["cls"])).astype(np.float32), float(v), exp, f.stem))
    ents = dedupe_aug(ents, {t: set(meta["keys"][t]) for t in tasks})
    by: dict = {}
    for e in ents:
        by.setdefault(e[0], []).append(e)
    npz, js = {}, {}
    for t, lst in by.items():
        npz[f"V/{t}"] = np.stack([e[2] for e in lst]); npz[f"S/{t}"] = np.array([e[3] for e in lst], float)
        js[t] = {"keys": [e[1] for e in lst], "exp": [e[4] for e in lst], "by_fold": [e[5] for e in lst]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "augpool.npz", **npz)
    (OUT / "augpool.json").write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    print(f"[prep-aug] {len(by)} 个任务、{len(ents)} 条扩充配置（缺真值 {miss}）；每任务 {min(len(v) for v in by.values())}–{max(len(v) for v in by.values())}", flush=True)


def load_aug(outdir: Path = OUT) -> dict:
    js = json.loads((Path(outdir) / "augpool.json").read_text(encoding="utf-8"))
    with np.load(Path(outdir) / "augpool.npz") as z:
        return {t: {"V": z[f"V/{t}"], "S": z[f"S/{t}"], **v} for t, v in js.items()}


# ================================================================ 扩充池评估（服务器）
def cmd_eval_ext(args) -> None:
    import torch
    g89, r105 = _g("89_acts_s3c"), _g("105_acts_s3r")
    meta, arr = g89.load_data(S3C); tasks = meta["tasks"]; groups = meta["groups"]
    outdir, src, arm, dev = Path(args.outdir), Path(args.src), args.arm, args.device
    op = outdir / "ext" / f"{args.tag}_{arm}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[eval-ext] skip {op.name}"); return
    aug = load_aug(outdir); at = sorted(aug)
    W = dict(zip(at, g89.load_windows("A", "", at, S3B, synthetic=args.synthetic))) if arm == "E" else {}
    dim = arr[f"{at[0]}/vec"].shape[1]
    folds = args.folds or list(g89.FOLDS); seeds = args.seeds or list(SEEDS)
    res: dict = {}
    for f in folds:
        Ps: dict = {t: [] for t in at}
        for s in seeds:
            m = r105._build(arm, dim, s, dev); m.load_state_dict(torch.load(src / "models" / f"{arm}_s{s}_{f}.pt", map_location=dev)); m.eval()
            sc = g89.Scorer(m, "A" if arm == "E" else "D", dev)
            with torch.no_grad():
                for t in at:
                    T = sc.task_vec(W[t], None) if arm == "E" else None
                    Ps[t].append(sc.pmat(np.vstack([arr[f"{t}/vec"], aug[t]["V"]]), T))
        res[f] = {}
        for t in at:
            i = tasks.index(t); n_lab = len(arr[f"{t}/vec"]); P = np.mean(Ps[t], 0)
            keys = list(meta["keys"][t]) + list(aug[t]["keys"])
            s0 = np.concatenate([arr["end0"][i], arr[f"{t}/R0"], aug[t]["S"]])
            top3 = g89.rr_order(P, keys)[:3]
            Psh = P[n_lab:, :N_SH]
            res[f][t] = {"src": groups[i], "held": groups[i] == f, "n_aug": int(len(aug[t]["S"])),
                         "r3": regret_ext(P, keys, s0, 3), "over": over_rate(Psh, aug[t]["S"], arr["end0"][i]),
                         "acc": acc_aug_sh(Psh, aug[t]["S"], arr["end0"][i]), "aug_in_top3": int(sum(j >= n_lab for j in top3)),
                         "pred_aug": float(np.mean(Psh))}
        h = [v for v in res[f].values() if v["held"]]
        print(f"[eval-ext] {args.tag}_{arm} {f}: 训练源 regret {np.mean([v['r3'] for v in res[f].values() if not v['held']]):.4f}；"
              f"留出 regret {np.mean([v['r3'] for v in h]) if h else float('nan'):.4f} 高估 {np.mean([v['over'] for v in h]) if h else float('nan'):+.3f}", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_select_alpha(args) -> None:
    outdir = Path(args.outdir); out: dict = {}
    folds = args.folds or list(_g("89_acts_s3c").FOLDS); seeds = args.seeds or list(SEEDS)
    for arm in ("E", "B"):
        if not all((outdir / "ext" / f"{atag(a)}_{arm}.json").exists() for a in ALPHAS):
            continue
        ext = {a: json.loads((outdir / "ext" / f"{atag(a)}_{arm}.json").read_text(encoding="utf-8")) for a in ALPHAS}
        out[arm] = {}
        sel = outdir / f"sel_{arm}"
        for d in ("models", "train"):
            (sel / d).mkdir(parents=True, exist_ok=True)
        for f in folds:
            scores = {a: float(np.mean([v["r3"] for v in ext[a][f].values() if not v["held"]])) for a in ALPHAS}
            a_star = select_alpha(scores)
            out[arm][f] = {"alpha": a_star, "scores": {atag(a): v for a, v in scores.items()}}
            for s in seeds:
                tg = f"{arm}_s{s}_{f}"
                shutil.copy2(outdir / atag(a_star) / "models" / f"{tg}.pt", sel / "models" / f"{tg}.pt")
                shutil.copy2(outdir / atag(a_star) / "train" / f"{tg}.npz", sel / "train" / f"{tg}.npz")
    (outdir / "alpha.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[select-alpha] {json.dumps({a: {f[:6]: v['alpha'] for f, v in d.items()} for a, d in out.items()}, ensure_ascii=False)}", flush=True)


# ================================================================ 部署（服务器；= 108，臂换成 CONS / CONS_B）
def _v108():
    v = _g("108_acts_s3v")
    v.ARMS = ARMS_W
    return v


def cmd_search(args) -> None:
    _v108().cmd_search(args)


def cmd_plan(args) -> None:
    _v108().cmd_plan(args)


def cmd_select(args) -> None:
    _v108().cmd_select(args)


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    from scipy.stats import wilcoxon  # noqa: F401
    v = _v108(); g89 = _g("89_acts_s3c")
    outdir = Path(args.outdir); rows = v._rows(outdir); sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
    D = v.DATASETS; o: dict = {"per_arm": {}, "means": {}}
    for a in ARMS_W:
        res = {}
        for d in D:
            t = v.off_tid(d); ks = sel[a][d]["ranked"]; k = sel[a][d]["pick"]
            te = [rows[(t, kk, 0)] for kk in ks]; ps = [rows[(t, k, s)] for s in SEEDS]
            res[d] = {"pick": k, "test_seeds": ps, "test": float(np.mean(ps)), "val": sel[a][d]["val"], "top3_s0": te,
                      "sel_loss": float(max(te) - rows[(t, k, 0)])}
        o["per_arm"][a] = res; o["means"][a] = float(np.mean([res[d]["test"] for d in D]))
        o["means"][a + "_top3avg"] = float(np.mean([np.mean(res[d]["top3_s0"]) for d in D]))
        o["means"][a + "_top3best"] = float(np.mean([max(res[d]["top3_s0"]) for d in D]))
    s3v = json.loads((S3V / "analysis.json").read_text(encoding="utf-8"))
    ref = {a: {d: s3v["per_arm"][a][d]["test"] for d in D} for a in ("FULL", "BLIND")}
    o["ref"] = {a: float(np.mean(list(ref[a].values()))) for a in ref}
    # 机制门：留出 12 任务，选中 α vs S3-R′（α = 0）
    alpha = json.loads((outdir / "alpha.json").read_text(encoding="utf-8")); o["alpha"] = alpha
    mech = {}
    for arm in ("E", "B"):
        r0 = json.loads((outdir / "ext" / f"ref_{arm}.json").read_text(encoding="utf-8"))
        rc = {f: json.loads((outdir / "ext" / f"{atag(alpha[arm][f]['alpha'])}_{arm}.json").read_text(encoding="utf-8"))[f] for f in g89.FOLDS}
        H0 = [x for f in g89.FOLDS for x in r0[f].values() if x["held"]]; HC = [x for f in g89.FOLDS for x in rc[f].values() if x["held"]]
        m = {k: (float(np.mean([x[k] for x in H0])), float(np.mean([x[k] for x in HC]))) for k in ("r3", "over", "acc", "aug_in_top3", "pred_aug")}
        m["pass"] = bool(m["over"][1] <= m["over"][0] - 0.05 and m["r3"][1] <= m["r3"][0] + 0.005)
        m["n_tasks"] = len(HC)
        mech[arm] = m
    o["mech"] = mech
    o["verdict"] = verdict_s3w(o["means"]["CONS"], o["means"]["CONS_B"], mech["E"]["pass"])
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    o["predictions"] = {"P1": o["means"]["CONS"] > FULL_REF, "P2": o["means"]["CONS"] > ts, "P3": o["means"]["CONS"] > inf,
                        "P4": mech["E"]["pass"], "P5": o["means"]["CONS"] >= o["means"]["CONS_B"] + 0.010,
                        "P6": o["means"]["CONS_top3avg"] > 0.578, "P7": sum(alpha["E"][f]["alpha"] == 3.0 for f in g89.FOLDS) >= 3}
    # 训练记录（选中 α）
    tr = {}
    for arm in ("E", "B"):
        for f in g89.FOLDS:
            for s in SEEDS:
                p = outdir / f"sel_{arm}" / "train" / f"{arm}_s{s}_{f}.npz"
                if p.exists():
                    with np.load(p) as z:
                        ch = json.loads(str(z["cons_hist"]))
                        tr[f"{arm}_s{s}_{f}"] = {"best_epoch": int(z["best_epoch"]), "epochs": len(json.loads(str(z["hist"]))),
                                                 "pred_adv_first": ch[0]["pred_adv"] if ch else None, "pred_adv_last": ch[-1]["pred_adv"] if ch else None,
                                                 "base_rate": ch[-1]["base_rate"] if ch else None, "cons_sec": float(sum(c["sec"] for c in ch))}
    o["train"] = tr
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # 表
    L = ["# S3-W 部署口径读数（官方 TEST，D_val 选 1，3 种子均值）", "",
         "| 数据集 | CONS | CONS_B | FULL（S3-V） | 任务盲（S3-V） | TS2Vec | InfoTS_s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        L.append(f"| {d} | {o['per_arm']['CONS'][d]['test']:.3f} | {o['per_arm']['CONS_B'][d]['test']:.3f} | {ref['FULL'][d]:.3f} | {ref['BLIND'][d]:.3f} | "
                 f"{v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    L.append(f"| **均值** | **{o['means']['CONS']:.4f}** | {o['means']['CONS_B']:.4f} | {o['ref']['FULL']:.4f} | {o['ref']['BLIND']:.4f} | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/7", "",
          "前 3 提名 TEST（种子 0）平均 / 最好：" + "；".join(f"{a} {o['means'][a + '_top3avg']:.3f} / {o['means'][a + '_top3best']:.3f}" for a in ARMS_W), "",
          "机制门（留出 12 任务，α = 0 → 选中 α）："]
    for arm in ("E", "B"):
        m = mech[arm]
        L.append(f"- {arm}：扩充池 regret@3 {m['r3'][0]:.4f} → {m['r3'][1]:.4f}；高估 {m['over'][0]:+.3f} → {m['over'][1]:+.3f}；"
                 f"扩充 vs shared 判对率 {m['acc'][0]:.3f} → {m['acc'][1]:.3f}；前 3 中扩充配置 {m['aug_in_top3'][0]:.2f} → {m['aug_in_top3'][1]:.2f}（{'过' if m['pass'] else '不过'}）")
    L += ["", "选中 α：" + "；".join(f"{arm} " + " / ".join(f"{f[:6]} {alpha[arm][f]['alpha']:g}" for f in g89.FOLDS) for arm in ("E", "B"))]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep_aug", "eval_ext", "select_alpha", "search", "select", "analyze", "resume", "synthetic"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--plan", choices=("val", "final"))
    ap.add_argument("--arm")
    ap.add_argument("--dataset")
    ap.add_argument("--src")
    ap.add_argument("--tag")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("prep_aug", cmd_prep_aug), ("eval_ext", cmd_eval_ext), ("select_alpha", cmd_select_alpha), ("search", cmd_search),
                     ("select", cmd_select), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    if args.plan:
        cmd_plan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
