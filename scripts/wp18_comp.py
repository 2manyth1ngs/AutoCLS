"""WP18 比较器：T-CLSC field 头（反对称、单配置分数）+ LambdaRank 顶部权重 + 软配对权重（= S3-Z NEW_B 的训练口径），
在扩展空间（旧 116 维 ⊕ 新轴 one-hot）上训练；源专家混合（E 只做相似度，= S3-MD ME）提供任务感知项。

纯函数 + 训练循环，供 scripts/122_wp18.py 调用。配对构造 / 权重 / 早停复用 105 的纯函数。
"""
from __future__ import annotations

import copy
import importlib
import time
from typing import Mapping, Optional, Sequence

import numpy as np

LR, MAX_EPOCHS, PATIENCE = 1e-4, 40, 5


def _g(name):
    return importlib.import_module(f"scripts.{name}")


def task_sigma(D: Mapping) -> np.ndarray:
    """每任务单次评估噪声 σ：两种子都有的可用配置上 std(a0 − a1)/√2 的近似 = 旧 S3 sig；
    这里没有逐种子值时用池中位数（只影响软配对权重的尺度）。"""
    return np.full(len(D["tasks"]), float(D.get("sigma_pool", 0.02)))


def holdout(n: int, task_id: str, frac: float = 0.25) -> np.ndarray:
    from autocls.labeling import run_seed_for
    h = int(n * frac)
    if h == 0:
        return np.array([], int)
    rng = np.random.default_rng(run_seed_for(task_id, "wp18_val"))
    return np.sort(rng.choice(int(n), size=h, replace=False))


def build_pairs(D: Mapping, idx: Sequence[int], sig: np.ndarray, lam: bool = True):
    r105 = _g("105_acts_s3r")
    TP, VP = {}, {}
    for i in idx:
        r, f, ns = D["r"][i], D["fail"][i], D["ns"][i]
        held = holdout(len(r), D["tasks"][i])
        use = np.ones(len(r), bool); use[held] = False
        TP[i] = r105.pair_table(r, f, sig[i], ns, use)
        if lam:
            a, b, y, w, k = TP[i]
            TP[i] = (a, b, y, r105.apply_top(w, r105.top_weights(r, f, use, a, b)), k)
        if len(held):
            VP[i] = r105.val_pairs(r, f, sig[i], ns, held)
            if lam:
                a, b, y, w, k = VP[i]
                VP[i] = (a, b, y, r105.apply_top(w, r105.top_weights(r, f, np.ones(len(r), bool), a, b)), k)
    return TP, VP


def train_member(D: Mapping, idx: Sequence[int], seed: int, device: str = "cuda", *, max_epochs: int = MAX_EPOCHS,
                 patience: int = PATIENCE, lr: float = LR, log=print):
    """任务盲 field 头比较器（TCLSC task_branch='none'）；早停 = 验证加权配对准确率（S3-R′ 口径）。"""
    import torch
    from autocls.comparator.tclsc_model import TCLSC
    r105 = _g("105_acts_s3r")
    torch.manual_seed(seed)
    dim = D["X"][idx[0]].shape[1]
    model = TCLSC(dim, None, task_branch="none", head="field").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sig = task_sigma(D)
    TP, VP = build_pairs(D, idx, sig)
    Vt = {i: torch.as_tensor(np.asarray(D["X"][i], np.float32), device=device) for i in idx}
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep, hist = -np.inf, None, 0, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        tot, totw = 0.0, 0.0
        for batch in r105.task_batches({i: len(TP[i][0]) for i in idx if len(TP[i][0])}, rng):
            losses, wsum = [], 0.0
            for i, sel in batch:
                a, b, y, w, _k = (x[sel] for x in TP[i])
                p = model(Vt[i][torch.as_tensor(a, device=device)], Vt[i][torch.as_tensor(b, device=device)], None)
                wt = torch.as_tensor(w, dtype=torch.float32, device=device)
                losses.append(torch.nn.functional.binary_cross_entropy(p.float(), torch.as_tensor(y, dtype=torch.float32, device=device),
                                                                       weight=wt, reduction="sum"))
                wsum += float(w.sum())
            loss = torch.stack(losses).sum() / max(wsum, 1e-12)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * wsum; totw += wsum
        model.eval()
        ps, ys, ws = [], [], []
        with torch.no_grad():
            for i, (a, b, y, w, _k) in VP.items():
                if not len(a):
                    continue
                p = model(Vt[i][torch.as_tensor(a, device=device)], Vt[i][torch.as_tensor(b, device=device)], None).float().cpu().numpy()
                ps.append(p); ys.append(y); ws.append(w)
        va = r105.weighted_acc(np.concatenate(ps), np.concatenate(ys), np.concatenate(ws)) if ps else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(totw, 1e-12), "val_acc": va})
        if va > best + 1e-9:
            best, bad, best_state, best_ep = va, 0, copy.deepcopy(model.state_dict()), ep
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    log(f"[member s{seed}] best val acc {best:.4f} @ ep {best_ep} / {len(hist)} epochs, {time.time() - t0:.0f}s")
    return model, {"best_val_acc": float(best), "best_epoch": int(best_ep), "hist": hist}


def score(model, X: np.ndarray, device: str = "cuda", bs: int = 8192) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(model.score(torch.as_tensor(np.asarray(X[s:s + bs], np.float32), device=device)).float().cpu().numpy())
    return np.concatenate(out)


# ================================================================ 源专家（E 只做相似度）
def zlabels(D: Mapping, q: float) -> list:
    ac = _g("acts_common")
    return [ac.clip_z(r, q)[0] for r in D["r"]]


def fit_inner(D: Mapping, Z: Sequence, S0e: Sequence, Ecat: np.ndarray, sources: Sequence[str], idx: Sequence[int], *,
              uniform: bool, alphas=None, taus=None, gammas=None) -> tuple:
    """内层留一源：选 (α_w, τ, γ) 使留出源任务上 Spearman(专家修正后分数, z) 均值最大（= 115.cmd_fit 内层）。"""
    from scipy.stats import spearmanr
    m114, g115 = _g("114_acts_s3me"), _g("115_acts_s3md")
    alphas = alphas or m114.ALPHA_W; taus = taus or m114.TAUS; gammas = gammas or m114.GAMMAS
    g = D["groups"]
    grid = [(a, t, gm) for a in alphas for t in taus for gm in gammas]
    sc = {k: [] for k in grid}
    for ih in sources:
        isrc = [s for s in sources if s != ih]
        iidx = [i for i in idx if g[i] != ih]
        ex = {a: g115.fit_experts(D["X"], Z, S0e, g, isrc, a, iidx) for a in alphas}
        for i in [i for i in idx if g[i] == ih]:
            zs = g115.task_z(S0e[i])
            for t in taus:
                pi = g115.target_pi(Ecat[i], Ecat, g, isrc, t, uniform)
                for a in alphas:
                    for gm in gammas:
                        sc[(a, t, gm)].append(spearmanr(g115.member_score(zs, D["X"][i], *ex[a], pi, gm), Z[i]).correlation)
    means = {k: float(np.nanmean(v)) for k, v in sc.items()}
    return m114.select_grid(means), means


# ================================================================ 轴 × E 交互项（FV：新轴 one-hot ⊗ 视图地标，岭回归在源残差上拟合）
def axis_slice(xs, axis: str = "view") -> slice:
    start = xs.enc.dim
    for n, a in xs.axes.items():
        w = len(a["choices"])
        if n == axis:
            return slice(start, start + w)
        start += w
    raise KeyError(axis)


def fv_design(X: np.ndarray, e: np.ndarray, sl: slice) -> np.ndarray:
    """(n, d) 配置 × (k,) 任务描述 → (n, w·(k+1))：轴 one-hot 与 [1, e] 的外积（含轴主效应）。"""
    oh = np.asarray(X, float)[:, sl]
    ee = np.concatenate([[1.0], np.asarray(e, float)])
    return (oh[:, :, None] * ee[None, None, :]).reshape(len(oh), -1)


def fit_fv(X: Sequence, Z: Sequence, S0e: Sequence, Ev: np.ndarray, idx: Sequence[int], sl: slice, alpha: float):
    """残差（任务内 z − 任务内标准化 s₀）对 fv_design 岭回归（无截距，主效应在设计里）→ 系数 B。"""
    g115 = _g("115_acts_s3md")
    A = np.vstack([fv_design(X[i], Ev[i], sl) for i in idx])
    y = np.concatenate([np.asarray(Z[i], float) - g115.task_z(S0e[i]) for i in idx])
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)


def fv_term(X: np.ndarray, e: np.ndarray, B: np.ndarray, sl: slice) -> np.ndarray:
    return fv_design(X, e, sl) @ B


def fit_fv_inner(D: Mapping, Z: Sequence, S0e: Sequence, Ev: np.ndarray, sources: Sequence[str], idx: Sequence[int], sl: slice,
                 alphas=(1.0, 10.0, 100.0, 1000.0), gammas=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    """内层留一源：(α, γ) 使留出源任务上 Spearman(task_z(s₀) + γ·FV, z) 均值最大；平局取较小 γ、较大 α。"""
    from scipy.stats import spearmanr
    g115 = _g("115_acts_s3md")
    g = D["groups"]
    sc = {(a, gm): [] for a in alphas for gm in gammas}
    for ih in sources:
        tr = [i for i in idx if g[i] != ih]
        te = [i for i in idx if g[i] == ih]
        for a in alphas:
            B = fit_fv(D["X"], Z, S0e, Ev, tr, sl, a)
            for i in te:
                zs, t = g115.task_z(S0e[i]), fv_term(D["X"][i], Ev[i], B, sl)
                for gm in gammas:
                    sc[(a, gm)].append(spearmanr(zs + gm * t, Z[i]).correlation)
    means = {k: float(np.nanmean(v)) for k, v in sc.items()}
    best = max(means, key=lambda k: (round(means[k], 12), -k[1], k[0]))
    return best, means
