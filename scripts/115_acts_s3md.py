"""WP17-ACTS S3-MD（results/acts/s3md/PROTOCOL.md）：源专家混合做进 T-CLSC，接部署，三个消融臂。

主干 = S3-Z NEW_B 的 10 个成员（field 头任务盲 s₀）；每成员每折对训练源残差（任务内 z − 任务内标准化 s₀）岭回归得源专家 (w_j, c_j)；
靶分数 = 2 万候选上标准化的 s₀ + γ · Σ_j π_j (w_j·x + c_j)，π = softmax(E_pt ⊕ E_A 余弦 / τ)（ME）或均匀（UNI）；BLIND（γ = 0）复用 S3-Z NEW_B。
阶段：
  --prep        本地：meta.json（= S3-Y）+ runs/runs.jsonl 预置 S3-Z 已跑格
  --score-src   服务器 GPU：10 成员 × 6 折对 160 个源任务已标配置打分 + 6 靶整集 E_pt 均值 → s0src.npz
  --prep-ea     CPU：6 靶官方 TRAIN 上的 E_A（response_profile）→ ea_targets.npz
  --fit         CPU：内层留一源选 (α_w, τ, γ)、每成员拟合源专家、靶的 π → experts.npz / fit.json
  --search      服务器 GPU：--arm ME|UNI --dataset D（= S3-Z 搜索，成员适应度加专家项）
  --plan cv|final / --select   = 111 同名阶段
  --analyze     本地
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3md"
S3Z = ROOT / "results" / "acts" / "s3z"
S3Y = ROOT / "results" / "acts" / "s3y"
S3X = ROOT / "results" / "acts" / "s3x"
S3CD = ROOT / "results" / "acts" / "s3cd"
ARMS_MD = {"ME": {"uniform": False}, "UNI": {"uniform": True}}
MEMBERS = tuple(range(10))
BLIND_REF = 0.6377                                    # S3-Z NEW_B 部署均值
FINAL_SEEDS = (0, 1, 2)


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def task_z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-12 else (v - v.mean()) / sd


def fit_experts(X: Sequence[np.ndarray], Z: Sequence[np.ndarray], S0: Sequence[np.ndarray], groups: np.ndarray, sources: Sequence[str],
                alpha: float, idx: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """每个源：其切片的残差（z − 任务内标准化 s₀）做岭回归 → (W: J×d, c: J)。"""
    m = _g("114_acts_s3me"); W, c = [], []
    for s in sources:
        ii = [i for i in idx if groups[i] == s]
        w, b = m.ridge_fit(np.vstack([X[i] for i in ii]), np.concatenate([np.asarray(Z[i], float) - task_z(S0[i]) for i in ii]), alpha)
        W.append(w); c.append(b)
    return np.array(W), np.array(c)


def member_score(zs0: np.ndarray, X: np.ndarray, W: np.ndarray, c: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    zs0 = np.asarray(zs0, float)
    if gamma == 0:
        return zs0
    return zs0 + float(gamma) * (np.asarray(X, float) @ (np.asarray(pi) @ W) + float(np.asarray(pi) @ c))


def target_pi(e_t: np.ndarray, Es: np.ndarray, groups: np.ndarray, sources: Sequence[str], tau: float, uniform: bool) -> np.ndarray:
    m = _g("114_acts_s3me")
    if uniform:
        return np.full(len(sources), 1.0 / len(sources))
    S = m.source_E(Es, groups, sources)
    sims = (S @ e_t) / (np.linalg.norm(S, axis=1) * np.linalg.norm(e_t) + 1e-12)
    return m.softmax_weights(sims, tau)


def verdict_s3md(me: float, uni: float, blind: float = BLIND_REF) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    code = "S3MD-BEATS-BOTH" if (me > inf and me > ts) else ("S3MD-BEATS-TS2VEC" if me > ts else "S3MD-BELOW")
    return code + ("+E" if me > uni else "") + ("+EXP" if uni > blind else "")


def _std_E(E: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-9
    return (E - mu) / sd, mu, sd


# ================================================================ 阶段
def cmd_prep(args) -> None:
    outdir = Path(args.outdir); (outdir / "runs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(S3Y / "meta.json", outdir / "meta.json")
    shutil.copy2(S3Z / "runs" / "runs.jsonl", outdir / "runs" / "runs.jsonl")
    print("[prep] meta.json 同 S3-Y；预置 S3-Z 已跑格", flush=True)


def cmd_score_src(args) -> None:
    import torch
    g89, r105, v = _g("89_acts_s3c"), _g("105_acts_s3r"), _g("108_acts_s3v")
    outdir = Path(args.outdir); meta, arr = g89.load_data(_g("105_acts_s3r").S3C); tasks = meta["tasks"]; dev = args.device
    Vs = [np.asarray(arr[f"{t}/vec"], np.float32) for t in tasks]; lens = np.array([len(x) for x in Vs]); Vall = np.vstack(Vs)
    out = {"lens": lens}
    for d in (args.folds or list(v.DATASETS)):
        for s in (args.members or MEMBERS):
            m = r105._build("B", Vall.shape[1], s, dev, head="field")
            m.load_state_dict(torch.load(Path(args.models) / f"B_s{s}_{d}.pt", map_location=dev)); m.eval()
            with torch.no_grad():
                out[f"{d}/{s}"] = m.score(torch.as_tensor(Vall, device=dev)).float().cpu().numpy()
        p = Path(args.s3cd) / "E" / f"{d}.npz"
        if p.exists():
            with np.load(p) as z:
                out[f"Ept_target/{d}"] = np.asarray(z["windows_repr"], np.float64).mean(axis=(0, 1))
        print(f"[score-src] {d} 完成", flush=True)
    np.savez_compressed(outdir / "s0src.npz", **out)


def cmd_prep_ea(args) -> None:
    ac, v = _g("acts_common"), _g("108_acts_s3v")
    from autocls.data.mech_features import RESPONSE_PROFILE_COLS, response_profile
    from autocls.task_types import TASK_CLASSIFICATION
    outdir = Path(args.outdir); tm = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))["tasks_meta"]; out = {}
    for d in (args.folds or list(v.DATASETS)):
        t0 = time.time(); X, y = ac.load_dataset_full(d); n = int(tm[v.off_tid(d)]["n_train_official"])
        wa = ac.whole_arrays_official(X, y, n_train=n); r = response_profile(wa["Xtr"], wa["ytr"], task_type=TASK_CLASSIFICATION)
        out[d] = np.array([float(r[k]) for k in RESPONSE_PROFILE_COLS])
        print(f"[prep-ea] {d} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(outdir / "ea_targets.npz", **out)


def _split(arr: np.ndarray, lens: np.ndarray) -> list:
    return np.split(np.asarray(arr, float), np.cumsum(lens)[:-1])


def cmd_fit(args) -> None:
    from scipy.stats import spearmanr
    m114, v = _g("114_acts_s3me"), _g("108_acts_s3v")
    outdir = Path(args.outdir); data = m114.load_all(); g = data["groups"]; order = data["order"]; X, Z = data["X"], data["Z"]
    s0 = np.load(outdir / "s0src.npz"); lens = s0["lens"]; eat = np.load(outdir / "ea_targets.npz")
    members = args.members or MEMBERS; grid = list(itertools.product(m114.ALPHA_W, m114.TAUS, m114.GAMMAS))
    res_npz, fit = {}, {}
    for d in (args.folds or list(v.DATASETS)):
        tr = g != d; train_src = [s for s in order if s != d]; train_idx = [i for i in range(len(g)) if tr[i]]
        Ep, mup, sdp = _std_E(data["Ept"], tr); Ea, mua, sda = _std_E(data["EA"], tr); Ecat = np.hstack([Ep, Ea])
        e_t = np.concatenate([(s0[f"Ept_target/{d}"] - mup) / sdp, (eat[d] - mua) / sda])
        S0m = {mm: _split(s0[f"{d}/{mm}"], lens) for mm in members}
        S0e = [np.mean([task_z(S0m[mm][i]) for mm in members], axis=0) for i in range(len(g))]
        fit[d] = {}
        for arm, A in ARMS_MD.items():
            sc = {k: [] for k in grid}
            for ih in train_src:
                isrc = [s for s in train_src if s != ih]; iidx = [i for i in train_idx if g[i] != ih]
                ex = {aw: fit_experts(X, Z, S0e, g, isrc, aw, iidx) for aw in m114.ALPHA_W}
                for i in [i for i in train_idx if g[i] == ih]:
                    zs = task_z(S0e[i])
                    for tau in m114.TAUS:
                        pi = target_pi(Ecat[i], Ecat, g, isrc, tau, A["uniform"])
                        for aw in m114.ALPHA_W:
                            for gam in m114.GAMMAS:
                                sc[(aw, tau, gam)].append(spearmanr(member_score(zs, X[i], *ex[aw], pi, gam), Z[i]).correlation)
            k = m114.select_grid({kk: float(np.nanmean(vv)) for kk, vv in sc.items()})
            pi = target_pi(e_t, Ecat, g, train_src, k[1], A["uniform"])
            Ws, cs = zip(*[fit_experts(X, Z, S0m[mm], g, train_src, k[0], train_idx) for mm in members])
            res_npz[f"{arm}/{d}/W"] = np.array(Ws); res_npz[f"{arm}/{d}/c"] = np.array(cs); res_npz[f"{arm}/{d}/pi"] = pi
            res_npz[f"{arm}/{d}/gamma"] = np.array(k[2])
            top = np.argsort(-pi)[:3]
            fit[d][arm] = {"alpha_w": k[0], "tau": k[1], "gamma": k[2], "inner_score": float(np.nanmean(sc[k])),
                           "pi_top3": [[train_src[j], float(pi[j])] for j in top], "sources": train_src}
            print(f"[fit] {d} {arm}：α_w {k[0]:g} τ {k[1]:g} γ {k[2]:g}；π 前 3 " + "，".join(f"{train_src[j]} {pi[j]:.2f}" for j in top), flush=True)
    np.savez_compressed(outdir / "experts.npz", **res_npz)
    (outdir / "fit.json").write_text(json.dumps(fit, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_search(args) -> None:
    import torch
    ac, g89, v, r105, y = _g("acts_common"), _g("89_acts_s3c"), _g("108_acts_s3v"), _g("105_acts_s3r"), _g("111_acts_s3y")
    from autocls.labeling import cls_key, run_seed_for
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    outdir, a, d, dev = Path(args.outdir), args.arm, args.dataset, args.device
    op = outdir / "search" / f"{a}_{d}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {a}_{d}"); return
    t0 = time.time(); cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    ex = np.load(outdir / "experts.npz"); W, cw, pi, gam = ex[f"{a}/{d}/W"], ex[f"{a}/{d}/c"], ex[f"{a}/{d}/pi"], float(ex[f"{a}/{d}/gamma"])
    members = list(args.members or MEMBERS); models = []
    for s in members:
        m = r105._build("B", enc.dim, s, dev, head="field")
        m.load_state_dict(torch.load(Path(args.models) / f"B_s{s}_{d}.pt", map_location=dev)); m.eval(); models.append(m)

    def raw(V):
        out = []
        with torch.no_grad():
            for m in models:
                out.append(np.concatenate([m.score(torch.as_tensor(np.asarray(V[s:s + 65536], np.float32), device=dev)).float().cpu().numpy()
                                           for s in range(0, len(V), 65536)]))
        return np.stack(out)

    tid = v.off_tid(d)
    cands = ac._sample_catalog(space, cfg, np.random.default_rng(run_seed_for(tid, "s3c_ks")), args.ks or g89.KS, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32)
    S = raw(Vc); st = [y.norm_stats(S[q]) for q in range(len(members))]

    def total(V):
        R = raw(V)
        return np.stack([member_score((R[q] - st[q][0]) / st[q][1], V, W[q], cw[q], pi, gam) for q in range(len(members))])

    Tc = total(Vc); gens = args.gens if args.gens is not None else g89.GENS
    searchers = [("ens", lambda V: total(V).mean(0), Tc.mean(0))] + [(f"m{q}", (lambda q: (lambda V: total(V)[q]))(q), Tc[q]) for q in range(len(members))]
    picks, byk = {}, {}
    for name, fn, f0 in searchers:
        init = [cands[j][0] for j in np.argsort(-f0, kind="stable")[:g89.KP]]
        fin = y.score_evolve(fn, init, space, enc, cfg, np.random.default_rng(run_seed_for(tid, f"s3y_evo_{name}")), g89.KP, gens, g89.P_CROSS, g89.P_MUT)
        picks[name] = [cls_key(c) for c in fin[:y.TOP_PER]]
        for c in fin[:y.TOP_PER]:
            byk[cls_key(c)] = c
    U = y.union_candidates([picks[n] for n, _f, _z in searchers])
    Tu = total(np.stack([enc.encode(byk[k]) for k in U]).astype(np.float32)); ens = dict(zip(U, Tu.mean(0).tolist()))
    xpool = {}
    if (S3X / "pool.json").exists():
        pool = json.loads((S3X / "pool.json").read_text(encoding="utf-8"))[d]
        Tx = total(np.stack([enc.encode(CLS(**e["cls"])) for e in pool]).astype(np.float32)); xpool = dict(zip([e["cls_key"] for e in pool], Tx.mean(0).tolist()))
    res = {"union": [{"cls_key": k, "cls": byk[k].to_dict(), "origins": [n for n in picks if k in picks[n]], "ens": ens[k]} for k in U],
           "comp_only": picks["ens"][0], "searchers": picks, "xpool": xpool, "gamma": gam, "ks": len(cands), "gens": gens, "seconds": time.time() - t0}
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[search] {a}_{d}：|U| = {len(U)}，γ {gam:g}，COMP-ONLY {picks['ens'][0][:8]}，{time.time() - t0:.0f}s", flush=True)


def _y():
    y = _g("111_acts_s3y"); y.ARMS_Y = {k: {} for k in ARMS_MD}
    return y


def cmd_plan(args) -> None:
    _y().cmd_plan(args)


def cmd_select(args) -> None:
    _y().cmd_select(args)


def cmd_analyze(args) -> dict:
    v = _g("108_acts_s3v")
    outdir = Path(args.outdir); rows = v._rows(outdir); sel = json.loads((outdir / "select.json").read_text(encoding="utf-8"))
    D = v.DATASETS; ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    z = json.loads((S3Z / "analysis.json").read_text(encoding="utf-8")); yy = json.loads((S3Y / "analysis.json").read_text(encoding="utf-8"))
    ref = {"BLIND": {d: z["per_arm"]["NEW_B"][d]["test"] for d in D}, "S3Z_NEW": {d: z["per_arm"]["NEW"][d]["test"] for d in D},
           "S3Y_NEW": {d: yy["per_arm"]["NEW"][d]["test"] for d in D}}
    o: dict = {"per_arm": {}, "means": {}, "ref": {k: float(np.mean(list(x.values()))) for k, x in ref.items()},
               "fit": json.loads((outdir / "fit.json").read_text(encoding="utf-8"))}
    for a in ARMS_MD:
        r = {}
        for d in D:
            sr = json.loads((outdir / "search" / f"{a}_{d}.json").read_text(encoding="utf-8")); t = v.off_tid(d)
            keys = [e["cls_key"] for e in sr["union"]]; s0 = np.array([rows[(t, k, 0)] for k in keys])
            pk, co = sel[a][d]["pick"], sel[a][d]["comp_only"]
            r[d] = {"pick": pk, "test": float(np.mean([rows[(t, pk, s)] for s in FINAL_SEEDS])), "comp_test": float(np.mean([rows[(t, co, s)] for s in FINAL_SEEDS])),
                    "U_n": len(keys), "U_best_s0": float(s0.max()), "U_mean_s0": float(s0.mean()), "sel_loss_s0": float(s0.max() - rows[(t, pk, 0)]),
                    "origins": [e["origins"] for e in sr["union"] if e["cls_key"] == pk][0]}
        o["per_arm"][a] = r
        for k in ("test", "comp_test", "U_best_s0", "U_mean_s0", "sel_loss_s0"):
            o["means"][f"{a}_{k}"] = float(np.mean([r[d][k] for d in D]))
    m = o["means"]
    o["verdict"] = verdict_s3md(m["ME_test"], m["UNI_test"], o["ref"]["BLIND"])
    o["predictions"] = {"P1": m["ME_test"] > o["ref"]["BLIND"], "P2": m["ME_test"] > m["UNI_test"], "P3": m["UNI_test"] > o["ref"]["BLIND"],
                        "P4": m["ME_test"] > inf and m["ME_test"] > ts, "P5": m["ME_test"] > o["ref"]["S3Z_NEW"], "P6": m["ME_U_best_s0"] > 0.6593}
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    L = ["# S3-MD 部署口径读数（官方 TEST，3 折 CV 选 1，3 种子均值）", "",
         "| 数据集 | ME（E 挑源） | UNI（均匀） | BLIND（S3-Z NEW_B） | S3-Z NEW（field E） | S3-Y NEW | ME 纯比较器 | ME 的 U 最好（种子 0） | TS2Vec | InfoTS_s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        a, b = o["per_arm"]["ME"][d], o["per_arm"]["UNI"][d]
        L.append(f"| {d} | {a['test']:.3f} | {b['test']:.3f} | {ref['BLIND'][d]:.3f} | {ref['S3Z_NEW'][d]:.3f} | {ref['S3Y_NEW'][d]:.3f} | {a['comp_test']:.3f} | "
                 f"{a['U_best_s0']:.3f} | {v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} |")
    L.append(f"| **均值** | **{m['ME_test']:.4f}** | {m['UNI_test']:.4f} | {o['ref']['BLIND']:.4f} | {o['ref']['S3Z_NEW']:.4f} | {o['ref']['S3Y_NEW']:.4f} | "
             f"{m['ME_comp_test']:.4f} | {m['ME_U_best_s0']:.4f} | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/6", "",
          f"- CV 选择损失（种子 0）：ME {m['ME_sel_loss_s0']:+.4f}、UNI {m['UNI_sel_loss_s0']:+.4f}；U 平均：ME {m['ME_U_mean_s0']:.4f}、UNI {m['UNI_U_mean_s0']:.4f}",
          "- ME 各折选中超参与 π 前 3：" + "；".join(f"{d[:6]} γ {o['fit'][d]['ME']['gamma']:g} τ {o['fit'][d]['ME']['tau']:g} → " +
                                                 "/".join(f"{s[:8]} {p:.2f}" for s, p in o['fit'][d]['ME']['pi_top3']) for d in D)]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "score_src", "prep_ea", "fit", "search", "select", "analyze", "resume"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--plan", choices=("cv", "final"))
    ap.add_argument("--arm", choices=tuple(ARMS_MD))
    ap.add_argument("--dataset")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--members", nargs="*", type=int, default=None)
    ap.add_argument("--models", default=str(S3Z / "models"))
    ap.add_argument("--s3cd", default=str(S3CD))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("prep", cmd_prep), ("score_src", cmd_score_src), ("prep_ea", cmd_prep_ea), ("fit", cmd_fit), ("search", cmd_search),
                     ("select", cmd_select), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    if args.plan:
        cmd_plan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
