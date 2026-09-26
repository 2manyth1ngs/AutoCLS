"""WP17-ACTS S3-M（results/acts/s3m/PROTOCOL.md）：E 可预测的偏好轴（降秩回归）+ 重训轴瓶颈 T-CLSC。

阶段：
  --prep          本地：RRR 选 (E, λ, K)（训练侧留一个源 EV）→ axes.npz（EP_oof / EP_fit / ET）+ prep.json（门 1）
  --train         服务器：--kind EP|ET --fold --seed → models/<kind>_s<seed>_<fold>.pt、train/<kind>_s<seed>_<fold>.npz
  --rank          服务器：R1 / R2 / R3（EP、EP-Z、ET）→ rank.json（门 2）
  --search        服务器：--fold → search/<fold>.json（EP-lcb / EPX-lcb / ET-lcb）
  --plan-runs     服务器：→ runs/plan_X.json
  --analyze       本地
  --smoke         本地冒烟
"""
from __future__ import annotations

import argparse
import copy
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

OUT = ROOT / "results" / "acts" / "s3m"
PREV = ("s3c", "s3e", "s3ea", "s3f", "s3g", "s3h", "s3i", "s3j", "s3l")
E_VARS = ("sh4", "pt4", "pt4+sh4")
LAMS = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
KS = (1, 2, 3)
KINDS = ("EP", "ET")
EXAM = ("EP-lcb", "EPX-lcb", "ET-lcb")
P_REF_R2SP = 0.376          # S3-K 中现行部署 P 的 R2 池内 Spearman（门 2）


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def rrr_fit(U: np.ndarray, R: np.ndarray, lam: float, k: int) -> dict:
    """降秩回归：C = (UᵀU + λI)⁻¹UᵀR（中心化），Q = 拟合值 UC 的前 k 个右奇异向量。"""
    U, R = np.asarray(U, float), np.asarray(R, float)
    Um, Rm = U.mean(0), R.mean(0); Uc, Rc = U - Um, R - Rm
    C = np.linalg.solve(Uc.T @ Uc + float(lam) * np.eye(U.shape[1]), Uc.T @ Rc)
    _u, _s, Vt = np.linalg.svd(Uc @ C, full_matrices=False)
    return {"Um": Um, "Rm": Rm, "C": C, "Q": Vt[:k].T.copy()}


def rrr_coords(fit: Mapping, U: np.ndarray, Q: Optional[np.ndarray] = None) -> np.ndarray:
    return (np.asarray(U, float) - fit["Um"]) @ fit["C"] @ (fit["Q"] if Q is None else Q)


def loso_ev(U: np.ndarray, R: np.ndarray, groups: Sequence, lam: float, k: int, mask: Optional[np.ndarray] = None) -> float:
    """训练侧留一个源：R̂ = Rm + coords·Qᵀ；EV = 1 − Σ‖R − R̂‖² / Σ‖R‖²。"""
    U, R, groups = np.asarray(U, float), np.asarray(R, float), np.asarray(groups)
    rows = np.ones(len(U), bool) if mask is None else np.asarray(mask, bool)
    sse = ss = 0.0
    for s in sorted(set(groups[rows].tolist())):
        fit_r = rows & (groups != s); te = rows & (groups == s)
        f = rrr_fit(U[fit_r], R[fit_r], lam, k)
        Rh = f["Rm"] + rrr_coords(f, U[te]) @ f["Q"].T
        sse += float(((R[te] - Rh) ** 2).sum()); ss += float((R[te] ** 2).sum())
    return 1.0 - sse / ss


def select_rrr(tab: Mapping[tuple, float]) -> Optional[tuple]:
    """(E, λ, K) → EV；取最大（平局取小 K、大 λ、E 名字典序）；最大 EV ≤ 0 → None。"""
    best = max(tab.values())
    if best <= 0:
        return None
    return sorted((k for k, v in tab.items() if v >= best - 1e-12), key=lambda k: (k[2], -k[1], k[0]))[0]


def verdict_s3m(dE_m3: Mapping, dE_p1: Mapping, dG_m3: Mapping, dX_m3: Mapping) -> str:
    sfx = lambda m3: m3["d"] >= 0.005 and m3["p"] < 0.05 and m3["npos"] >= 4  # noqa: E731
    if dE_m3["d"] >= 0.010 and dE_m3["p"] < 0.05 and dE_m3["npos"] >= 4 and dE_p1["d"] > 0:
        code = "S3M-E-WIN"
    elif dE_m3["d"] <= -0.010 and dE_m3["p"] < 0.05:
        code = "S3M-E-HURT"
    else:
        code = "S3M-E-NULL"
    return code + ("+GAIN" if sfx(dG_m3) else "") + ("+CONTENT" if sfx(dX_m3) else "")


# ================================================================ prep（本地）
def cmd_prep(args, e_vars: Sequence[str] = E_VARS, feat=None) -> None:
    """feat：可选 (var, fold, groups, em_sh, em_pt) → U 钩子（S3-N 加入 E_A 用；缺省 = 99.e_features，S3-M 原样）。"""
    g84, g85, g86, g89, g99 = (_g(m) for m in ("84_acts_s3a", "85_acts_s3ap", "86_acts_s3b", "89_acts_s3c", "99_acts_s3l"))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    meta, _arr = g89.load_data(Path(args.s3c)); tasks = meta["tasks"]
    D = g85.load_all()["pool"]; assert [str(t) for t in D["tasks"]] == tasks
    Z = np.asarray(D["Z"], float); groups = np.asarray(D["groups"])
    s3b = ROOT / "results/acts/s3b"
    em_sh = g86._e_means(s3b, "sh", tasks, list(g89.FOLDS)); em_pt = g86._e_means(s3b, "pt", tasks, [""])[""]
    tab, svd_tab, orc_tab, Rs, Us = {}, {}, {}, {}, {}
    t0 = time.time()
    for fold in g89.FOLDS:
        tr = groups != fold
        R = g84.residual(Z, Z[tr].mean(0)); Rs[fold] = R
        for var in e_vars:
            U = (feat or g99.e_features)(var, fold, groups, em_sh, em_pt); Us[(fold, var)] = U
            for lam in LAMS:
                for k in KS:
                    tab.setdefault((var, lam, k), []).append(loso_ev(U, R, groups, lam, k, mask=tr))
        # 对照：SVD 轴 + 逐分量岭回归（sh4，α 按 MSE）与真实投影上界
        U = Us[(fold, "sh4")]
        for k in KS:
            sse = ss = sse_o = 0.0
            for s in sorted(set(groups[tr].tolist())):
                fr = tr & (groups != s); te = tr & (groups == s)
                mu_in = R[fr].mean(0); _u, _s, Vt = np.linalg.svd(R[fr] - mu_in, full_matrices=False); B = Vt[:k].T
                A = (R[fr] - mu_in) @ B
                ah = np.stack([g84.ridge_predict(g84.ridge_fit(U[fr], A[:, j], g84.choose_alpha(U[fr], A[:, j], groups[fr], LAMS)), U[te]) for j in range(k)], 1)
                sse += float(((R[te] - mu_in - ah @ B.T) ** 2).sum()); ss += float((R[te] ** 2).sum())
                sse_o += float(((R[te] - mu_in - ((R[te] - mu_in) @ B) @ B.T) ** 2).sum())
            svd_tab.setdefault(k, []).append(1 - sse / ss); orc_tab.setdefault(k, []).append(1 - sse_o / ss)
        print(f"[prep] {fold} {time.time() - t0:.0f}s", flush=True)
    ev = {k: float(np.mean(v)) for k, v in tab.items()}
    star = select_rrr(ev)
    out = {"star": list(star) if star else None, "gate1": star is not None,
           "ev_table": {f"{k[0]}|{k[1]:g}|{k[2]}": v for k, v in ev.items()},
           "svd_ridge_ev": {k: float(np.mean(v)) for k, v in svd_tab.items()}, "oracle_proj_ev": {k: float(np.mean(v)) for k, v in orc_tab.items()}}
    if star:
        var, lam, K = star; npz = {}
        for fold in g89.FOLDS:
            tr = groups != fold; R = Rs[fold]; U = Us[(fold, var)]
            full = rrr_fit(U[tr], R[tr], lam, K); Q = full["Q"]
            ep_fit = rrr_coords(full, U); et = (R - full["Rm"]) @ Q
            ep_oof = np.full((len(tasks), K), np.nan)
            for s in sorted(set(groups[tr].tolist())):
                fr = tr & (groups != s); te = groups == s
                ep_oof[te] = rrr_coords(rrr_fit(U[fr], R[fr], lam, K), U[te], Q)       # 内层映射、全训练侧轴
            sd_ep = ep_oof[tr].std(0); sd_et = et[tr].std(0)
            sd_ep = np.where(sd_ep > 1e-12, sd_ep, 1.0); sd_et = np.where(sd_et > 1e-12, sd_et, 1.0)
            npz[f"{fold}/EP_oof"] = (ep_oof / sd_ep).astype(np.float32); npz[f"{fold}/EP_fit"] = (ep_fit / sd_ep).astype(np.float32)
            npz[f"{fold}/ET"] = (et / sd_et).astype(np.float32)
            hi = ~tr
            out.setdefault("heldout_rho", {})[fold] = [float(np.corrcoef(ep_fit[hi, j], et[hi, j])[0, 1]) for j in range(K)]
        np.savez_compressed(outdir / "axes.npz", **npz)
    (outdir / "prep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(ev.items(), key=lambda kv: -kv[1])[:8]
    L = [f"# S3-M 第一道门（{'通过' if star else '未过 → S3M-NO-AXIS'}；所选 = {star}）", "",
         "| E | λ | K | 训练侧留一个源 EV |", "|---|---:|---:|---:|"] + [f"| {k[0]} | {k[1]:g} | {k[2]} | {v:+.4f} |" for k, v in top]
    L += ["", "对照（sh4，同口径 EV）：SVD 轴 + 岭回归 " + " / ".join(f"K={k} {v:+.4f}" for k, v in out["svd_ridge_ev"].items())
          + "；真实投影上界 " + " / ".join(f"K={k} {v:+.4f}" for k, v in out["oracle_proj_ev"].items())]
    if star:
        L.append("留出任务上 EP 与真实投影的相关（逐折，逐轴）：" + "；".join(f"{f} " + "/".join(f"{x:+.2f}" for x in v) for f, v in out["heldout_rho"].items()))
    (outdir / "prep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 训练（服务器）
def _aug_union(args, fold: str, meta: Mapping, enc) -> dict:
    g94, g97 = _g("94_acts_s3g"), _g("97_acts_s3j")
    val = g94._val_map([Path(getattr(args, k)) for k in PREV] + [Path(args.outdir)]); tasks = meta["tasks"]
    return g97.union_aug(g97._keyed(Path(args.s3g) / "search_aug" / f"{fold}.json", val, tasks, enc),
                         g97._keyed(Path(args.s3j) / "search_aug" / f"{fold}.json", val, tasks, enc))


def _coords(outdir: Path, fold: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(训练侧 teacher 坐标, 留出 / 搜索坐标)。"""
    with np.load(outdir / "axes.npz") as z:
        if kind == "EP":
            return np.asarray(z[f"{fold}/EP_oof"]), np.asarray(z[f"{fold}/EP_fit"])
        return np.asarray(z[f"{fold}/ET"]), np.asarray(z[f"{fold}/ET"])


def cmd_train(args) -> None:
    import torch
    g88, g89, g91, g94, g96 = (_g(m) for m in ("88_acts_s3bpp", "89_acts_s3c", "91_acts_s3e", "94_acts_s3g", "96_acts_s3i"))
    outdir = Path(args.outdir)
    meta, arr, _cfg, _space, enc = g94._ctx(args)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, seed, kind, dev = args.fold, int(args.seed), args.kind, args.device
    tag = f"{kind}_s{seed}_{fold}"
    for d in ("train", "models"):
        (outdir / d).mkdir(parents=True, exist_ok=True)
    op = outdir / "train" / f"{tag}.npz"
    if op.exists() and args.resume:
        print(f"[train] skip {tag}"); return
    vs = meta["val"][fold]
    tr = [i for i, g in enumerate(groups) if g not in (fold, vs)]
    va = [i for i, g in enumerate(groups) if g == vs]
    te = [i for i, g in enumerate(groups) if g == fold]
    V = [arr[f"{t}/vec"] for t in tasks]
    R, NS, SIG = g91._labels(arr, tasks, "L4")
    aug = _aug_union(args, fold, meta, enc); V, R, NS = g94.merge_aug(V, R, NS, aug)
    Ct, Ch = _coords(outdir, fold, kind); K = Ct.shape[1]
    Tt = torch.as_tensor(np.nan_to_num(Ct), dtype=torch.float32, device=dev); Th = torch.as_tensor(Ch, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    model = g96.AxisComparator(V[0].shape[1], k=K).to(dev); sc = g96.AxisScorer(model, dev)
    opt = torch.optim.Adam(model.parameters(), lr=g89.LR); rng = np.random.default_rng(seed)

    def weighted(i, trip):
        out = []
        for a, b, y in trip:
            w = g91.l4_weight(R[i][a], R[i][b], SIG[i], NS[i][a], NS[i][b])
            if w > 0:
                out.append((i, a, b, y, w))
        return out

    val_pairs = [p for i in va for p in weighted(i, g89.all_ordered_pairs(R[i]))]

    def batch_loss(batch):
        va_ = torch.as_tensor(np.stack([V[b[0]][b[1]] for b in batch]), dtype=torch.float32, device=dev)
        vb_ = torch.as_tensor(np.stack([V[b[0]][b[2]] for b in batch]), dtype=torch.float32, device=dev)
        y = torch.as_tensor([float(b[3]) for b in batch], dtype=torch.float32, device=dev)
        w = torch.as_tensor([float(b[4]) for b in batch], dtype=torch.float32, device=dev)
        p = sc.forward(va_, vb_, Tt[[b[0] for b in batch]]).clamp(1e-6, 1 - 1e-6)
        return -(w * (y * torch.log(p) + (1 - y) * torch.log(1 - p))).sum() / w.sum().clamp_min(1e-9)

    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(args.max_epochs or g89.MAX_EPOCHS):
        model.train()
        pairs = [p for i in tr for p in weighted(i, g89.linear_pairs(R[i], rng))]
        perm = rng.permutation(len(pairs)); tot, nb = 0.0, 0
        for s in range(0, len(perm), g89.BATCH):
            loss = batch_loss([pairs[j] for j in perm[s:s + g89.BATCH]])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for s in range(0, len(val_pairs), 4096):
                b = val_pairs[s:s + 4096]; vl += float(batch_loss(b)) * len(b); vn += len(b)
        vl /= max(vn, 1); hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl})
        if vl < best - 1e-9:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= g89.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    cmaps = g88.cross_maps(tasks, groups, order); Z0 = torch.zeros(K, device=dev)
    out = {"hist": json.dumps(hist), "te": np.array(te), "va": np.array(va), "K": K}
    for i in te + va:
        out[f"P/{tasks[i]}"] = sc.pmat(V[i], Th[i] if i in te else Tt[i]).astype(np.float32)
    for i in te:
        out[f"PZ/{tasks[i]}"] = sc.pmat(V[i], Z0).astype(np.float32)
        out[f"XA/{tasks[i]}"] = np.stack([sc.pmat(V[i][:g89.N_SH], Th[m[i]]) for m in cmaps]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best val {best:.4f} epochs {len(hist)} K {K} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 排序门（服务器 / 本地）
def _pool(args, meta, enc) -> dict:
    from autocls.space.cls_space import CLS
    g93, g94 = _g("93_acts_s3f"), _g("94_acts_s3g")
    dirs = [Path(getattr(args, k)) for k in PREV]
    val = g94._val_map(dirs); cls_of = {}
    for d in [ROOT / "results/acts/s2"] + dirs:
        for sub, name in (("", "plan_S.json"), ("runs", "plan_X.json"), ("runs_aug", "plan_X.json")):
            p = (d / sub / name) if sub else (d / name)
            if p.exists():
                for c in json.loads(p.read_text(encoding="utf-8"))["cells"]:
                    cls_of[(c["task_id"], c["cls_key"])] = c["cls"]
    tasks = meta["tasks"]; out = {}
    for t in g93.exam_tasks(tasks, np.array(meta["groups"]), _g("89_acts_s3c").FOLDS):
        keys = sorted({k for (tt, k, s) in val if tt == t and s == 0 and (t, k, 1) in val and (t, k) in cls_of})
        out[t] = (np.stack([enc.encode(CLS(**cls_of[(t, k)])) for k in keys]).astype(np.float32),
                  np.array([val[(t, k, 0)] for k in keys]), np.array([val[(t, k, 1)] for k in keys]))
    return out


def cmd_rank(args) -> None:
    import torch
    from scipy.stats import spearmanr
    g89, g91, g94, g96, g98, g99 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "94_acts_s3g", "96_acts_s3i", "98_acts_s3k", "99_acts_s3l"))
    outdir = Path(args.outdir)
    meta, arr, _cfg, _space, enc = g94._ctx(args)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]; N = g89.N_SH
    folds = args.folds or list(g89.FOLDS)
    pool = _pool(args, meta, enc)
    res = {}
    for kind, cond in (("EP", "EP"), ("EP", "EP-Z"), ("ET", "ET")):
        r1, r3, r2, r2s = [], [], [], []
        for fold in folds:
            _ct, ch = _coords(outdir, fold, kind); K = ch.shape[1]
            ms = []
            for sd in (args.seeds or g91.ENS_SEEDS):
                m = g96.AxisComparator(enc.dim, k=K); m.load_state_dict(torch.load(outdir / "models" / f"{kind}_s{sd}_{fold}.pt", map_location="cpu")); m.eval(); ms.append(m)

            def heads(V):
                with torch.no_grad():
                    return [m.heads(torch.as_tensor(np.asarray(V, np.float32))).numpy() for m in ms]
            for i in np.where(groups == fold)[0]:
                t = tasks[i]; c = np.zeros(K) if cond == "EP-Z" else ch[i]
                w = np.mean([g99.wins_linear(H, c) for H in heads(arr[f"{t}/vec"][:N])], 0)
                r1.append(g98.pool_regret(w, end0[i], end1[i]))
                R0 = arr[f"{t}/R0"]
                if len(R0) >= 10:
                    r3.append(float(spearmanr(np.mean([g99.wins_linear(H, c) for H in heads(arr[f"{t}/vec"][N:N + len(R0)])], 0), R0)[0]))
                if t in pool:
                    V, v0, v1 = pool[t]; w = np.mean([g99.wins_linear(H, c) for H in heads(V)], 0)
                    r2.append(g98.pool_regret(w, v0, v1)); r2s.append(float(spearmanr(w, (v0 + v1) / 2)[0]))
        res[cond] = {"R1": float(np.mean(r1)), "R2": float(np.mean(r2)), "R2_sp": float(np.mean(r2s)), "R3": float(np.nanmedian(r3)), "n_R2": len(r2)}
    ok = bool(res["EP"]["R2_sp"] > P_REF_R2SP and res["EP"]["R2_sp"] - res["EP-Z"]["R2_sp"] >= 0.02)
    out = {"gate2": ok, "ranking": res, "ref_S3K": json.loads((ROOT / "results/acts/s3k/analysis.json").read_text(encoding="utf-8"))["summary"]}
    (outdir / "rank.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"gate2": ok, **{k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()}}, ensure_ascii=False), flush=True)


# ================================================================ 搜索（服务器）
def cmd_search(args) -> None:
    import torch
    g89, g91, g92, g93, g94, g96 = (_g(m) for m in ("89_acts_s3c", "91_acts_s3e", "92_acts_s3ea", "93_acts_s3f", "94_acts_s3g", "96_acts_s3i"))
    from autocls.space.cls_space import CLS
    meta, arr, cfg, space, enc = g94._ctx(args)
    outdir, fold, dev = Path(args.outdir), args.fold, args.device
    op = outdir / "search" / f"{fold}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {fold}"); return
    seeds = args.seeds or g91.ENS_SEEDS
    ens, C = {}, {}
    for kind in KINDS:
        _ct, ch = _coords(outdir, fold, kind); K = ch.shape[1]; C[kind] = torch.as_tensor(ch, dtype=torch.float32, device=dev)
        ms = []
        for sd in seeds:
            m = g96.AxisComparator(enc.dim, k=K).to(dev); m.load_state_dict(torch.load(outdir / "models" / f"{kind}_s{sd}_{fold}.pt", map_location=dev)); m.eval(); ms.append(m)
        ens[kind] = g96.AxisEnsemble(ms, dev)
    tasks = meta["tasks"]; groups = meta["groups"]; cx = g92.cross_index(tasks, groups, meta["order"])
    ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    lcb = lambda name: ((name, g91.LAM, None),)  # noqa: E731
    res, t0 = {}, time.time()
    for t in g93.exam_tasks(tasks, groups, [fold]):
        i = tasks.index(t); n = len(seeds); Vsh = arr[f"{t}/vec"][:g89.N_SH]
        Tep, Tx, Tet = [C["EP"][i]] * n, [C["EP"][cx[i]]] * n, [C["ET"][i]] * n
        rec = {"x_task": tasks[cx[i]]}
        rec.update(g92._search_task(ens["EP"], Tep, t, lcb("EP-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens["EP"], Tx, t, lcb("EPX-lcb"), space, enc, cfg, ks, gens))
        rec.update(g92._search_task(ens["ET"], Tet, t, lcb("ET-lcb"), space, enc, cfg, ks, gens))
        for arm, e_, Ts in (("EP-lcb", ens["EP"], Tep), ("ET-lcb", ens["ET"], Tet)):
            for e in rec[arm]:
                e["pred_beat"] = g94._pred_beat(e_, Ts, enc.encode(CLS(**e["cls"])), Vsh)
        res[t] = rec
        print(f"[search] {fold} {t} " + " ".join(f"{a} {[x['cls_key'] for x in rec[a]]}" for a in EXAM) + f" {time.time() - t0:.0f}s", flush=True)
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")


def _dirs(args) -> list[Path]:
    return [Path(getattr(args, k)) for k in PREV] + [Path(args.outdir)]


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
    g84, g89, g93, g94 = (_g(m) for m in ("84_acts_s3a", "89_acts_s3c", "93_acts_s3f", "94_acts_s3g"))
    outdir = Path(args.outdir); P_ = {k: Path(getattr(args, k)) for k in PREV}
    meta, arr = g89.load_data(P_["s3c"]); tasks = meta["tasks"]; groups = np.array(meta["groups"]); end0, end1 = arr["end0"], arr["end1"]
    val = g94._val_map(_dirs(args))
    js = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    ex = g93.exam_tasks(tasks, groups, g89.FOLDS); sg = np.array([groups[tasks.index(t)] for t in ex])
    sm = {s: js(outdir / "search" / f"{s}.json") for s in g89.FOLDS}; sj = {s: js(P_["s3j"] / "search" / f"{s}.json") for s in g89.FOLDS}
    sgg = {s: js(P_["s3g"] / "search" / f"{s}.json") for s in g89.FOLDS}; sf = {s: js(P_["s3f"] / "search" / f"{s}.json") for s in g89.FOLDS}
    sc_ = {s: js(P_["s3c"] / "search" / f"A_{s}.json") for s in g89.FOLDS}
    arms = ("EP-lcb", "EPX-lcb", "ET-lcb", "A′-lcb", "O′-lcb", "B′-lcb", "R")
    noms = {t: {"EP-lcb": sm[s][t]["EP-lcb"], "EPX-lcb": sm[s][t]["EPX-lcb"], "ET-lcb": sm[s][t]["ET-lcb"], "A′-lcb": sj[s][t]["A-lcb"], "O′-lcb": sj[s][t]["O-lcb"],
                "B′-lcb": sgg[s][t]["B′-lcb"], "R": sf[s][t]["R"] if sf[s][t]["new_row"] else sc_[s][t]["random"]} for t, s in zip(ex, sg)}
    P1, M3, miss = {a: [] for a in arms}, {a: [] for a in arms}, {a: 0 for a in arms}
    over = {"EP": [], "ET": []}; cat_p1, cat_m3 = [], []
    for t, s in zip(ex, sg):
        i = tasks.index(t); fl = float(np.nanmin(end0[i])); ref = (end0[i] + end1[i]) / 2
        for a in arms:
            ks_ = [x["cls_key"] for x in noms[t][a]]
            miss[a] += int(any((t, k, sd) not in val for k in ks_ for sd in (0, 1)))
            p1, m3 = g93.readouts(t, ks_, val, fl); P1[a].append(p1); M3[a].append(m3)
        for arm, key in (("EP-lcb", "EP"), ("ET-lcb", "ET")):
            for e in sm[s][t][arm]:
                over[key].append(e["pred_beat"] - g94.true_beat((val.get((t, e["cls_key"], 0), fl) + val.get((t, e["cls_key"], 1), fl)) / 2, ref))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        cat_p1.append(float(end1[i, o[int(np.argmax(end0[i, o]))]])); cat_m3.append(float(np.mean((end0[i, o] + end1[i, o]) / 2)))
    P1 = {k: np.array(v) for k, v in P1.items()}; M3 = {k: np.array(v) for k, v in M3.items()}
    P1["cat3"], M3["cat3"] = np.array(cat_p1), np.array(cat_m3)

    def cmp(x, y):
        d = np.asarray(x) - np.asarray(y)
        p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        return {"d": float(d.mean()), "p": p, "npos": g89.n_sources_positive(d, sg), "ci": g84.cluster_bootstrap_ci(d, sg)}
    pairs = (("EP-lcb", "B′-lcb"), ("EP-lcb", "A′-lcb"), ("EP-lcb", "EPX-lcb"), ("ET-lcb", "B′-lcb"), ("ET-lcb", "O′-lcb"), ("EP-lcb", "cat3"), ("O′-lcb", "B′-lcb"))
    C = {f"{a}−{b}": {"m3": cmp(M3[a], M3[b]), "p1": cmp(P1[a], P1[b])} for a, b in pairs}
    code = verdict_s3m(C["EP-lcb−B′-lcb"]["m3"], C["EP-lcb−B′-lcb"]["p1"], C["EP-lcb−A′-lcb"]["m3"], C["EP-lcb−EPX-lcb"]["m3"])
    per_src = {a: {s: float(M3[a][sg == s].mean()) for s in g89.FOLDS} for a in list(arms) + ["cat3"]}
    out = {"verdict": code, "prep": js(outdir / "prep.json")["star"], "rank": js(outdir / "rank.json")["ranking"],
           "final_p1": {k: float(v.mean()) for k, v in P1.items()}, "final_m3": {k: float(v.mean()) for k, v in M3.items()},
           "cmp": C, "missing": miss, "over": {k: float(np.mean(v)) for k, v in over.items()}, "per_source_m3": per_src}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    g = lambda x: f"{x['d']:+.3f}（p {x['p']:.3f}，{x['npos']}/6）"  # noqa: E731
    L = [f"# S3-M 读数（{code}；轴 = {out['prep']}）", "", "| 臂 | 3 选 1 | 3 条平均 | 缺格任务 |", "|---|---:|---:|---:|"]
    for a in list(arms) + ["cat3"]:
        L.append(f"| {a} | {f(out['final_p1'][a])} | {f(out['final_m3'][a])} | {miss.get(a, 0)} |")
    L += ["", "| 对比 | 3 条平均 | 3 选 1 |", "|---|---|---|"] + [f"| {k} | {g(v['m3'])} | {g(v['p1'])} |" for k, v in C.items()]
    L += ["", f"自提名高估：EP {out['over']['EP']:+.3f} / ET {out['over']['ET']:+.3f}（S3-J：A′ +0.151 / O′ +0.098）", "",
          "## 逐源（3 条平均）", "", "| 源 | " + " | ".join(list(arms) + ["cat3"]) + " |", "|---|" + "---:|" * (len(arms) + 1)]
    for s in g89.FOLDS:
        L.append(f"| {s} | " + " | ".join(f(per_src[a][s]) for a in list(arms) + ["cat3"]) + " |")
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


# ================================================================ 冒烟（本地）
def cmd_smoke(args) -> None:
    g89 = _g("89_acts_s3c")
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    fold = g89.FOLDS[0]
    base = dict(vars(args)); base.update({"max_epochs": 1, "resume": False, "ks": 1000, "gens": 2, "seeds": [0, 1], "folds": [fold]})
    if (OUT / "axes.npz").exists() and od != OUT:
        import shutil
        for n in ("axes.npz", "prep.json"):
            shutil.copy(OUT / n, od / n)
    else:
        cmd_prep(types.SimpleNamespace(**base))
    for kind in KINDS:
        for sd in (0, 1):
            cmd_train(types.SimpleNamespace(**{**base, "kind": kind, "fold": fold, "seed": sd}))
    cmd_rank(types.SimpleNamespace(**base))
    ns = types.SimpleNamespace(**{**base, "fold": fold}); cmd_search(ns); cmd_plan_runs(ns)
    r = json.loads((od / "search" / f"{fold}.json").read_text(encoding="utf-8"))
    assert len(r) == 8 and all(len(v[a]) == 3 for v in r.values() for a in EXAM)
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "train", "rank", "search", "plan-runs", "analyze", "smoke"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--kind", default="EP", choices=KINDS)
    ap.add_argument("--fold", default="Handwriting")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--device", default=None)
    for k in PREV:
        ap.add_argument(f"--{k}", default=str(ROOT / "results" / "acts" / k))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("prep", cmd_prep), ("smoke", cmd_smoke), ("train", cmd_train), ("rank", cmd_rank), ("search", cmd_search),
                     ("plan_runs", cmd_plan_runs), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
