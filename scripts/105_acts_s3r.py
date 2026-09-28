"""WP17-ACTS S3-R（results/acts/s3r/PROTOCOL.md）：按 AutoCTS++ 实际做法重训 T-CLSC。

改三处、结构不变（E 臂 = S3-C 严格复现 A：E_pt + SetTF + concat；B 臂 = 同设置任务盲）：
  ① 标签清洗：L4 值；失败（≤ 探针地板）对失败剔除，失败对可用权重 0.25 且不回传 SetTF，可用对可用 L4 权重；
  ② 验证 = 同任务留出 ⌊n_random/4⌋ 条 random 配置（← AutoCTS++ get_valid_test_seeds：同任务留出架构），早停 patience 5；
  ③ 任务内全有序对（← AHC_DataLoader quadratic），按任务分组成批（4 任务 × 64 对）。
用法：--prep（本地，写 labels.npz）| --train --arm E|B --fold F --seed S（服务器）| --analyze（本地）| --smoke
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3r"
S3B = ROOT / "results" / "acts" / "s3b"
S3C = ROOT / "results" / "acts" / "s3c"
N_SH = 30
FW_W = 0.25
PER_TASK, TPB = 64, 4
LR, MAX_EPOCHS, PATIENCE, SUB_WIN = 1e-4, 40, 5, 32
SEEDS = (0, 1, 2)
ARMS = ("E", "B")
OFFS_IN = (1, 2, 3, 4)
RAW_DIR = ROOT / "results" / "acts" / "s3v" / "raw"


# ================================================================ 纯函数
def val_holdout(n_random: int, task_id: str) -> np.ndarray:
    """训练任务的验证留出：⌊n_random/4⌋ 条 random（下标在 [30, 30 + n_random)），按任务固定种子。"""
    h = int(n_random) // 4
    if h == 0:
        return np.array([], int)
    from autocls.labeling import run_seed_for
    rng = np.random.default_rng(run_seed_for(task_id, "s3r_val"))
    return np.sort(N_SH + rng.choice(int(n_random), size=h, replace=False))


def _pairs(r: np.ndarray, fail: np.ndarray, sig: float, ns: np.ndarray, keep: np.ndarray, gate: float = 0.0):
    """keep (n, n) 布尔：候选有序对。返回 (a, b, y, w, kind)，kind 0 = 可用对可用、1 = 失败对可用；失败对失败与权重 0 的剔除。"""
    r = np.asarray(r, float); fail = np.asarray(fail, bool); ns = np.asarray(ns, float)
    a, b = np.where(keep & ~np.eye(len(r), dtype=bool))
    fa, fb = fail[a], fail[b]
    d = np.abs(r[a] - r[b]); s = float(sig) * np.sqrt(1.0 / ns[a] + 1.0 / ns[b])
    w_ww = np.where(s <= 0, 1.0, np.minimum(d / np.where(s > 0, 2.0 * s, 1.0), 1.0))          # = 91.l4_weight
    w_ww = np.where(d == 0, 0.0, w_ww)
    if gate > 0:                                                                          # S3-T：硬门槛 |Δ| > c·SE，进入者权重 1
        w_ww = np.where(d > float(gate) * s, 1.0, 0.0)
    fw = fa ^ fb
    w = np.where(fw, FW_W, w_ww)
    y = np.where(fw, fb, r[a] > r[b]).astype(int)
    m = ~(fa & fb) & (w > 0)
    return a[m], b[m], y[m], w[m].astype(float), fw[m].astype(int)


def pair_table(r, fail, sig, ns, use, gate: float = 0.0):
    use = np.asarray(use, bool)
    return _pairs(r, fail, sig, ns, use[:, None] & use[None, :], gate)


def val_pairs(r, fail, sig, ns, held, allowed=None):
    h = np.zeros(len(r), bool); h[np.asarray(held, int)] = True
    keep = h[:, None] | h[None, :]
    if allowed is not None:                                                      # S3-V w/o shared：验证配对也不含 shared
        a = np.asarray(allowed, bool); keep = keep & a[:, None] & a[None, :]
    return _pairs(r, fail, sig, ns, keep)


def train_use_mask(n_total: int, held, no_shared: bool = False) -> np.ndarray:
    """训练配置掩码：去掉验证留出；S3-V w/o shared 再去掉前 N_SH 条 shared。"""
    use = np.ones(int(n_total), bool); use[np.asarray(held, int)] = False
    if no_shared:
        use[:N_SH] = False
    return use


def task_batches(by_task: Mapping[int, int], rng: np.random.Generator, *, per_task: int = PER_TASK, tasks_per_batch: int = TPB) -> list:
    """每任务配对乱序切成 ≤ per_task 的块，块整体乱序，连续 tasks_per_batch 块成一批：[(task, idx), ...]。"""
    chunks = []
    for t, n in by_task.items():
        p = rng.permutation(int(n))
        chunks += [(t, p[s:s + per_task]) for s in range(0, int(n), per_task)]
    order = rng.permutation(len(chunks))
    chunks = [chunks[j] for j in order]
    return [chunks[s:s + tasks_per_batch] for s in range(0, len(chunks), tasks_per_batch)]


def weighted_acc(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p, y, w = np.asarray(p, float), np.asarray(y, int), np.asarray(w, float)
    return float(np.sum(w * ((p > 0.5).astype(int) == y)) / max(np.sum(w), 1e-12))


def verdict_s3r(gin: Mapping, gval: Mapping, gcon: Mapping) -> str:
    if not gin["ok"]:
        return "S3R-E-UNUSED"
    code = "S3R-E-WIN" if gval["ok"] else "S3R-E-FITS"
    return code + ("+CONTENT" if gcon["ok"] else "")


def improved(cur: float, best: float, mode: str) -> bool:
    """早停判据：loss（S3-R）取小，acc（S3-R′）取大；都要求严格变好。"""
    return cur < best - 1e-9 if mode == "loss" else cur > best + 1e-9


def base_rates(r: np.ndarray, use: np.ndarray, n_sh: int = N_SH) -> np.ndarray:
    """S3-W 基准率：t(c) = 训练用 random 配置（下标 ≥ n_sh、use 为真、不含 c 自身）中值高于 c 的比例，平局记 0.5；无 random → nan。"""
    r = np.asarray(r, float); use = np.asarray(use, bool)
    rnd = np.where(use & (np.arange(len(r)) >= n_sh))[0]
    out = np.full(len(r), np.nan)
    for c in range(len(r)):
        o = rnd[rnd != c]
        if len(o):
            out[c] = float(np.mean((r[o] > r[c]) + 0.5 * (r[o] == r[c])))
    return out


def cons_pairs(n_adv: int, refs: np.ndarray, t: np.ndarray, offset: int):
    """S3-W 保守配对：对抗配置（扩展下标 offset..offset+n_adv−1）× 参考配置 c，双向；P(x ≻ c) 目标 t(c)，P(c ≻ x) 目标 1 − t(c)。"""
    refs = np.asarray(refs, int); t = np.asarray(t, float)
    xs = offset + np.arange(int(n_adv))
    X, C = np.repeat(xs, len(refs)), np.tile(refs, len(xs))
    a = np.concatenate([X, C]); b = np.concatenate([C, X])
    y = np.concatenate([t[C], 1.0 - t[C]])
    adv_first = np.r_[np.ones(len(X), bool), np.zeros(len(X), bool)]
    return a, b, y, adv_first


def cons_schedule(tr: Sequence[int], epoch: int, period: int, seed: int = 0) -> list:
    """S3-W 对抗集刷新日程：epoch 0 全部训练任务；之后按固定排列分 period 组轮换。"""
    tr = list(tr)
    if epoch == 0 or period <= 1:
        return tr
    perm = np.random.default_rng(seed).permutation(len(tr))
    g = (int(epoch) - 1) % int(period)
    return [tr[j] for j in sorted(perm[g::int(period)])]


def verdict_gate(d: float, p: float, npos: int) -> str:
    """S3-T 主门：S3-R′ E 臂 − 本档 E 臂的 regret@3 差。"""
    if d >= 0.003 and p < 0.05 and npos >= 4:
        return "S3T-GATE-HELPS"
    if d <= -0.003 and p < 0.05:
        return "S3T-GATE-HURTS"
    return "S3T-GATE-NULL"


def signflip_p(d: np.ndarray) -> float:
    return importlib.import_module("scripts.104_acts_s3q").signflip_p(d)


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g89 = importlib.import_module("scripts.89_acts_s3c"); g91 = importlib.import_module("scripts.91_acts_s3e")
    P = g84.load_s2()["pool"]
    meta, arr = g89.load_data(S3C)
    tasks = meta["tasks"]
    assert [str(t) for t in P["tasks"]] == tasks
    R, NS, SIG = g91._labels(arr, tasks, "L4")
    pf = np.asarray(P["pfloor"], float)
    out = {"sig": np.array(SIG, float), "pfloor": pf}
    nf = []
    for i, t in enumerate(tasks):
        fail = R[i] <= pf[i] + 1e-12
        out[f"{t}/r"] = R[i]; out[f"{t}/ns"] = NS[i]; out[f"{t}/fail"] = fail
        out[f"{t}/held"] = val_holdout(len(R[i]) - N_SH, t)
        nf.append(fail.mean())
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "labels.npz", **out)
    print(f"[prep] {len(tasks)} 任务 → labels.npz；失败占比 {np.mean(nf):.3f}", flush=True)


def load_raw_windows(raw_dir: Path, tasks, synthetic: bool = False) -> list:
    """S3-V w/o TS2Vec：每任务原始窗口 (100, ≤152, C)（108 --prep-raw 产出，与 E_pt 同一批窗口）。"""
    if synthetic:
        rng = np.random.default_rng(0)
        return [rng.normal(size=(20, 16, 2)).astype(np.float16) for _ in tasks]
    out = []
    for t in tasks:
        with np.load(Path(raw_dir) / f"{t}.npz") as z:
            out.append(z["raw"])
    return out


def load_labels(outdir: Path) -> dict:
    f = Path(outdir) / "labels.npz"
    with np.load(f if f.exists() else OUT / "labels.npz") as z:                # 冒烟目录回退到正式 labels
        return {k: z[k] for k in z.files}


# ================================================================ 训练（服务器）
def _build(arm: str, cls_dim: int, seed: int, device: str, e_mode: str = "settf", std=None):
    import torch
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.task_embed import TaskSetEncoder
    torch.manual_seed(seed)
    if arm == "E" and e_mode == "mean":              # S3-V w/o Set-Transformer：TS2Vec 表示均值池化（训练任务标准化）
        from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder
        m = TCLSC(cls_dim, MeanPoolTaskEncoder(256, *(std or (None, None))), task_branch="E+ctx", head="concat")
    elif arm == "E" and e_mode == "raw":             # S3-V w/o TS2Vec：原始窗口逐时间步 MLP + SetTF
        from autocls.comparator.raw_mlp_encoder import RawMLPSetEncoder
        m = TCLSC(cls_dim, RawMLPSetEncoder(nz=256, hidden=64), task_branch="E+ctx", head="concat")
    elif arm == "E":                                  # = 89.build_model("A")，只换种子
        m = TCLSC(cls_dim, TaskSetEncoder(dim_input=256, nz=256), task_branch="E+ctx", head="concat")
    else:                                             # = 89.build_model("D")
        m = TCLSC(cls_dim, None, task_branch="none", head="concat")
    return m.to(device)


def _cons_init(meta: Mapping, tasks: Sequence[str], R, F, H, tr: Sequence[int], no_shared: bool) -> dict:
    """S3-W：搜索空间 / 编码器（= 部署搜索同款）+ 每个训练任务的基准率、参考下标、已标键。"""
    ac = importlib.import_module("scripts.acts_common")
    from autocls.space.cls_encoder import CLSEncoder
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg)
    TB, REF = {}, {}
    for i in tr:
        use = train_use_mask(len(R[i]), H[i], no_shared)
        TB[i] = base_rates(R[i], use, N_SH)
        REF[i] = np.where(use & ~np.asarray(F[i], bool) & np.isfinite(TB[i]))[0]
    return {"cfg": cfg, "space": space, "enc": CLSEncoder(space, mode="binned"), "TB": TB, "REF": REF,
            "lab": {i: set(meta["keys"][tasks[i]]) for i in tr}}


def _cons_search(sc, T, cs: Mapping, i: int, rng: np.random.Generator, cands, Vc: np.ndarray, keys: Sequence[str], gens: int, top: int):
    """S3-W 内层搜索（= 108.cmd_search 同一流程、规模缩小）：分块循环赛 → 进化 → 前 top 条未标配置的向量。"""
    import torch
    ac = importlib.import_module("scripts.acts_common"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.search.evolution import evolutionary_search
    from autocls.space.cls_space import CLS
    space, enc, cfg = cs["space"], cs["enc"], cs["cfg"]
    init = g89.chunk_select(len(cands), lambda idx: sc.pmat(Vc[np.asarray(idx)], T), keys)

    def score_pairs(a, b):
        with torch.no_grad():
            at = torch.as_tensor(np.asarray(a, np.float32), device=sc.dev); bt = torch.as_tensor(np.asarray(b, np.float32), device=sc.dev)
            Tb = None if T is None else T.unsqueeze(0).expand(len(at), -1)
            return sc.forward(at, bt, Tb).float().cpu().numpy()

    evo = evolutionary_search(score_pairs, space, enc, ks=0, kp=g89.KP, generations=gens, p_cross=g89.P_CROSS, p_mut=g89.P_MUT, top_k=g89.KP,
                              batch_size=65536, rng=rng, seed_population=[cands[j][0] for j in init],
                              accept=lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg))
    out = [e for e in evo if e["cls_key"] not in cs["lab"][i]][:top]
    return np.stack([enc.encode(CLS(**e["cls"])) for e in out]).astype(np.float32) if out else None


def cmd_train(args) -> None:
    import torch
    g89 = importlib.import_module("scripts.89_acts_s3c"); g88 = importlib.import_module("scripts.88_acts_s3bpp")
    outdir = Path(args.outdir); L = load_labels(outdir)
    meta, arr = g89.load_data(S3C)
    tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]
    fold, arm, seed, dev = args.fold, args.arm, int(args.seed), args.device
    tag = f"{arm}_s{seed}_{fold}"
    for d in ("train", "models"):
        (outdir / d).mkdir(parents=True, exist_ok=True)
    op = outdir / "train" / f"{tag}.npz"
    if op.exists() and args.resume:
        print(f"[train] skip {tag}"); return
    tr = [i for i, g in enumerate(groups) if g != fold]
    te = [i for i, g in enumerate(groups) if g == fold]
    if args.max_tasks_per_src:                        # 冒烟用：每源只留前 n 个任务
        keep = []
        for s in dict.fromkeys(groups.tolist()):
            keep += [i for i in range(len(tasks)) if groups[i] == s][: args.max_tasks_per_src]
        tr = [i for i in tr if i in keep]; te = [i for i in te if i in keep]
    V = [arr[f"{t}/vec"] for t in tasks]
    R = [L[f"{t}/r"] for t in tasks]; F = [L[f"{t}/fail"] for t in tasks]; NSd = [L[f"{t}/ns"] for t in tasks]
    H = [L[f"{t}/held"] for t in tasks]; SIG = L["sig"]
    for i in range(len(tasks)):
        assert len(R[i]) == len(V[i]), f"{tasks[i]} 标签与配置条数不一致"
    e_mode = getattr(args, "e_mode", "settf") or "settf"; no_shared = bool(getattr(args, "no_shared", False))
    if arm == "E" and e_mode == "raw":
        W = load_raw_windows(Path(getattr(args, "raw_dir", RAW_DIR)), tasks, synthetic=args.synthetic)
    else:
        W = g89.load_windows("A", fold, tasks, S3B, synthetic=args.synthetic) if arm == "E" else None
    std = None
    if arm == "E" and e_mode == "mean":
        from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder
        std = MeanPoolTaskEncoder.fit_standardizer({str(i): W[i] for i in tr}, [str(i) for i in tr])   # 只用训练任务（铁律 6）
    model = _build(arm, V[0].shape[1], seed, dev, e_mode, std); sc = g89.Scorer(model, "A" if arm == "E" else "D", dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    gate = float(getattr(args, "gate", 0.0) or 0.0)
    TP, TPr, VP = {}, {}, {}
    for i in tr:
        use = train_use_mask(len(R[i]), H[i], no_shared)
        TP[i] = pair_table(R[i], F[i], SIG[i], NSd[i], use, gate=gate)                 # 训练表（S3-T 门槛）
        TPr[i] = TP[i] if gate == 0 else pair_table(R[i], F[i], SIG[i], NSd[i], use)  # 样本内读数表（S3-R′ 软权重口径）
        if len(H[i]):
            VP[i] = val_pairs(R[i], F[i], SIG[i], NSd[i], H[i], allowed=(np.arange(len(R[i])) >= N_SH) if no_shared else None)
    n_pairs = int(sum(len(v[0]) for v in TP.values()))
    Vt = {i: torch.as_tensor(np.asarray(V[i], np.float32), device=dev) for i in set(tr) | set(te)}

    def tvec(i, sub):
        return None if arm == "B" else sc.task_vec(W[i], None, rng, sub)

    def fwd(i, a, b, T, detach_mask=None):
        Tb = None
        if T is not None:
            Tb = T.unsqueeze(0).expand(len(a), -1)
            if detach_mask is not None:
                Tb = torch.where(detach_mask[:, None], Tb.detach(), Tb)
        return sc.forward(Vt[i][a], Vt[i][b], Tb)

    def eval_pairs(i, pairs, T, kind_sel=None):
        a, b, y, w, k = pairs
        m = np.ones(len(a), bool) if kind_sel is None else (k == kind_sel)
        if not m.any():
            return np.zeros(0), y[m], w[m]
        with torch.no_grad():
            p = fwd(i, torch.as_tensor(a[m], device=dev), torch.as_tensor(b[m], device=dev), T).float().cpu().numpy()
        return p, y[m], w[m]

    mode = getattr(args, "stop", "loss") or "loss"
    best, best_state, bad, hist, seen, best_ep = (np.inf if mode == "loss" else -np.inf), None, 0, [], 0, 0
    t0 = time.time()
    max_ep = args.max_epochs or MAX_EPOCHS
    # ---- S3-W 保守项（cons = 0 时不执行任何新代码，与 S3-R′ 逐位相同）
    cons = float(getattr(args, "cons", 0.0) or 0.0)
    cons_ks, cons_gens, cons_top = int(getattr(args, "cons_ks", 2000)), int(getattr(args, "cons_gens", 3)), int(getattr(args, "cons_top", 5))
    cons_ref, cons_period = int(getattr(args, "cons_ref", 16)), int(getattr(args, "cons_period", 3))
    cons_hist, ADV, NADV = [], {}, {}
    if cons > 0:
        ac = importlib.import_module("scripts.acts_common")
        cs = _cons_init(meta, tasks, R, F, H, tr, no_shared)
        cons_rng = np.random.default_rng([seed, 7717])
    for ep in range(max_ep):
        if cons > 0:                                  # 每 epoch 开始：当前模型在（轮换的）训练任务上跑全空间搜索 → 对抗集
            tc = time.time(); model.eval()
            cands = ac._sample_catalog(cs["space"], cs["cfg"], np.random.default_rng([seed, 9973, ep]), cons_ks, set())
            Vc = np.stack([cs["enc"].encode(c) for c, _k in cands]).astype(np.float32); ckeys = [k for _c, k in cands]
            sched = cons_schedule(tr, ep, cons_period, seed)
            pa, br = [], []
            with torch.no_grad():
                for i in sched:
                    Ti = tvec(i, False)
                    xa = _cons_search(sc, Ti, cs, i, np.random.default_rng([seed, 9973, ep, i]), cands, Vc, ckeys, cons_gens, cons_top)
                    if xa is None:
                        ADV.pop(i, None); continue
                    ADV[i] = torch.as_tensor(np.vstack([np.asarray(V[i], np.float32), xa]), device=dev); NADV[i] = len(xa)
                    if len(cs["REF"][i]):
                        ca, cb, cy, af = cons_pairs(len(xa), cs["REF"][i], cs["TB"][i], len(V[i]))
                        Tb = None if Ti is None else Ti.unsqueeze(0).expand(int(af.sum()), -1)
                        pa.append(float(sc.forward(ADV[i][ca[af]], ADV[i][cb[af]], Tb).float().mean())); br.append(float(cy[af].mean()))
            model.train()
            cons_hist.append({"epoch": ep, "n_tasks_refreshed": len(sched), "n_tasks_adv": len(ADV), "pred_adv": float(np.mean(pa)) if pa else float("nan"),
                              "base_rate": float(np.mean(br)) if br else float("nan"), "sec": round(time.time() - tc, 1)})
            print(f"[cons] {tag} ep {ep} 刷新 {len(sched)} 任务：对抗集预测胜率 {cons_hist[-1]['pred_adv']:.3f} vs 基准率 {cons_hist[-1]['base_rate']:.3f} {cons_hist[-1]['sec']:.0f}s", flush=True)
        model.train()
        tot, totw = 0.0, 0.0
        for batch in task_batches({i: len(TP[i][0]) for i in tr}, rng):
            losses, wsum, lcs = [], 0.0, []
            for i, idx in batch:
                a, b, y, w, k = (x[idx] for x in TP[i])
                T = tvec(i, True)
                dm = torch.as_tensor(k == 1, device=dev) if T is not None else None
                p = fwd(i, torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev), T, dm)
                wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
                losses.append(torch.nn.functional.binary_cross_entropy(p.float(), torch.as_tensor(y, dtype=torch.float32, device=dev), weight=wt, reduction="sum"))
                wsum += float(w.sum()); seen += len(a)
                if cons > 0 and i in ADV and len(cs["REF"][i]):          # 保守配对：E detach，只约束比较头与配置支路
                    refs = cons_rng.choice(cs["REF"][i], size=min(cons_ref, len(cs["REF"][i])), replace=False)
                    ca, cb, cy, _af = cons_pairs(NADV[i], refs, cs["TB"][i], len(V[i]))
                    Tb = None if T is None else T.detach().unsqueeze(0).expand(len(ca), -1)
                    pc = sc.forward(ADV[i][torch.as_tensor(ca, device=dev)], ADV[i][torch.as_tensor(cb, device=dev)], Tb)
                    lcs.append(torch.nn.functional.binary_cross_entropy(pc.float(), torch.as_tensor(cy, dtype=torch.float32, device=dev)))
            lmain = torch.stack(losses).sum() / max(wsum, 1e-12)
            loss = lmain + cons * torch.stack(lcs).mean() if lcs else lmain
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(lmain.detach()) * wsum; totw += wsum
        model.eval()
        vl, vw, vacc_p, vacc_y, vacc_w = 0.0, 0.0, [], [], []
        with torch.no_grad():
            for i, pr in VP.items():
                p, y, w = eval_pairs(i, pr, tvec(i, False))
                vl += float(np.sum(-w * (y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))); vw += float(w.sum())
                vacc_p.append(p); vacc_y.append(y); vacc_w.append(w)
        vl /= max(vw, 1e-12)
        va = weighted_acc(np.concatenate(vacc_p), np.concatenate(vacc_y), np.concatenate(vacc_w)) if vacc_p else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(totw, 1e-12), "val_loss": vl, "val_acc": va, "pairs_seen": seen, "sec": round(time.time() - t0, 1)})
        print(f"[train] {tag} ep {ep} train {hist[-1]['train_loss']:.4f} val {vl:.4f} acc {va:.3f} {time.time() - t0:.0f}s", flush=True)
        cur = vl if mode == "loss" else va
        if improved(cur, best, mode):
            best, bad, best_state, best_ep = cur, 0, copy.deepcopy(model.state_dict()), ep
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    torch.save(model.state_dict(), outdir / "models" / f"{tag}.pt")
    # ---- 样本内 / 验证（本 E vs 换 E；换 E = 训练源内 cross_maps 偏移 1–4）
    out = {"hist": json.dumps(hist), "te": np.array(te), "tr": np.array(tr), "n_pairs_epoch": n_pairs, "best_epoch": best_ep, "stop": mode, "gate": gate,
           "e_mode": e_mode, "no_shared": no_shared, "cons": cons, "cons_hist": json.dumps(cons_hist),
           "cons_cfg": json.dumps({"ks": cons_ks, "gens": cons_gens, "top": cons_top, "ref": cons_ref, "period": cons_period})}
    with torch.no_grad():
        Tall = None if arm == "B" else torch.stack([sc.task_vec(W[i], None) for i in range(len(tasks))])
        tr_tasks = [tasks[i] for i in tr]
        cm_in = g88.cross_maps(tr_tasks, groups[tr], [s for s in order if s in set(groups[tr].tolist())], OFFS_IN) if arm == "E" else []
        rows = []
        for j, i in enumerate(tr):
            Ti = None if Tall is None else Tall[i]
            row = []
            for kind in (0, 1):
                p, y, w = eval_pairs(i, TPr[i], Ti, kind)
                row += [weighted_acc(p, y, w) if len(p) else np.nan, float(w.sum())]
                xs = [weighted_acc(*eval_pairs(i, TPr[i], Tall[tr[m[j]]], kind)) for m in cm_in] if (arm == "E" and len(p)) else [np.nan]
                row.append(float(np.mean(xs)))
            if i in VP:
                p, y, w = eval_pairs(i, VP[i], Ti)
                row += [weighted_acc(p, y, w), float(w.sum())]
                xs = [weighted_acc(*eval_pairs(i, VP[i], Tall[tr[m[j]]])) for m in cm_in] if arm == "E" else [np.nan]
                row.append(float(np.mean(xs)))
            else:
                row += [np.nan, 0.0, np.nan]
            rows.append(row)
        out["insample"] = np.array(rows, float)      # 列：ww acc, ww w, ww acc×E, fw acc, fw w, fw acc×E, val acc, val w, val acc×E
        # ---- 留出源（= S3-C 考卷）
        if Tall is not None:
            out["T_all"] = Tall.float().cpu().numpy()
        cm_out = g88.cross_maps(tasks, groups, order)
        for i in te:
            Ti = None if Tall is None else Tall[i]
            out[f"P/{tasks[i]}"] = sc.pmat(V[i], Ti).astype(np.float32)
            if Tall is not None:
                out[f"X/{tasks[i]}"] = np.stack([sc.pmat(V[i][:N_SH], Tall[m[i]]) for m in cm_out]).astype(np.float32)
    np.savez_compressed(op, **out)
    print(f"[train] {tag} done best {mode} {best:.4f} @ ep {best_ep} epochs {len(hist)} pairs/epoch {n_pairs} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 分析（本地）
INS_COLS = ("ww", "ww_w", "ww_x", "fw", "fw_w", "fw_x", "val", "val_w", "val_x")


def _ins(z) -> dict:
    A = z["insample"]
    return {c: A[:, j] for j, c in enumerate(INS_COLS)}


def _wmean(x, w):
    m = np.isfinite(x) & (w > 0)
    return float(np.sum(x[m] * w[m]) / max(np.sum(w[m]), 1e-12))


def cmd_analyze(args) -> dict:
    from scipy.stats import spearmanr, wilcoxon
    g84 = importlib.import_module("scripts.84_acts_s3a"); g86 = importlib.import_module("scripts.86_acts_s3b"); g89 = importlib.import_module("scripts.89_acts_s3c")
    outdir = Path(args.outdir); meta, arr = g89.load_data(S3C)
    tasks = meta["tasks"]; groups = np.array(meta["groups"])
    P2 = g84.load_s2()["pool"]; end0, end1 = P2["end0"], P2["end1"]
    folds = list(g89.FOLDS)
    Z = {(a, s, f): np.load(outdir / "train" / f"{a}_s{s}_{f}.npz") for a in ARMS for s in SEEDS for f in folds
         if (outdir / "train" / f"{a}_s{s}_{f}.npz").exists()}
    o: dict = {"n_models": len(Z)}
    # ---- 训练过程
    tr_stats = {}
    for a in ARMS:
        ep, be, tl, vl, va, seen, npe = [], [], [], [], [], [], []
        for s in SEEDS:
            for f in folds:
                if (a, s, f) not in Z:
                    continue
                zz = Z[(a, s, f)]; h = json.loads(str(zz["hist"]))
                b = int(zz["best_epoch"]) if "best_epoch" in zz.files else int(np.argmin([x["val_loss"] for x in h]))   # S3-R 旧文件按损失
                ep.append(len(h)); be.append(b); tl.append(h[b]["train_loss"]); vl.append(h[b]["val_loss"]); va.append(h[b]["val_acc"])
                seen.append(h[b]["pairs_seen"]); npe.append(int(Z[(a, s, f)]["n_pairs_epoch"]))
        tr_stats[a] = {"epochs": ep, "best_epoch": be, "best_epoch_median": float(np.median(be)) if be else None,
                       "train_loss_best": float(np.mean(tl)) if tl else None, "val_loss_best": float(np.mean(vl)) if vl else None,
                       "val_acc_best": float(np.mean(va)) if va else None, "pairs_seen_median": float(np.median(seen)) if seen else None,
                       "pairs_per_epoch_median": float(np.median(npe)) if npe else None}
    o["training"] = tr_stats
    # ---- 样本内
    din, fold_d, ins = [], {f: [] for f in folds}, {}
    for (a, s, f), z in Z.items():
        I = _ins(z)
        row = {"ww": _wmean(I["ww"], I["ww_w"]), "fw": _wmean(I["fw"], I["fw_w"]), "val": _wmean(I["val"], I["val_w"])}
        if a == "E":
            row.update({"ww_x": _wmean(I["ww_x"], I["ww_w"]), "fw_x": _wmean(I["fw_x"], I["fw_w"]), "val_x": _wmean(I["val_x"], I["val_w"])})
            d = row["ww"] - row["ww_x"]; din.append(d); fold_d[f].append(d)
        ins[f"{a}_s{s}_{f}"] = row
    din = np.array(din)
    fm = np.array([np.mean(v) for v in fold_d.values() if v])
    gin = {"d": float(din.mean()) if len(din) else float("nan"), "p": signflip_p(din) if len(din) else 1.0,
           "fold_pos": int((fm > 0).sum()), "n_fold": int(len(fm))}
    gin["ok"] = bool(len(din) and gin["d"] >= 0.02 and gin["p"] < 0.05 and gin["fold_pos"] >= 5)
    eb = [ins[f"E_s{s}_{f}"]["ww"] - ins[f"B_s{s}_{f}"]["ww"] for s in SEEDS for f in folds if f"E_s{s}_{f}" in ins and f"B_s{s}_{f}" in ins]
    o["insample"] = {"G_IN": gin, "E_minus_B_ww": float(np.mean(eb)) if eb else None,
                     "mean": {k: float(np.mean([v[k] for n, v in ins.items() if n.startswith(a + "_") and k in v]))
                              for a in ARMS for k in ("ww", "fw", "val", "ww_x", "fw_x", "val_x")
                              if any(k in v for n, v in ins.items() if n.startswith(a + "_"))} if ins else {},
                     "by_arm": {a: {k: float(np.mean([v[k] for n, v in ins.items() if n.startswith(a + "_") and k in v]))
                                    for k in ("ww", "fw", "val", "ww_x", "fw_x", "val_x") if any(k in v for n, v in ins.items() if n.startswith(a + "_"))}
                                for a in ARMS}}
    o["insample"].pop("mean")
    # ---- 留出源（3 种子 P 取均值 → RR）
    held = [i for i, g in enumerate(groups) if g in folds]; hg = groups[held]
    res = {}
    for a in ARMS:
        r3, r1, sp, rx = (np.full(len(held), np.nan) for _ in range(4))
        cosE = []
        for jj, i in enumerate(held):
            t = tasks[i]; f = groups[i]
            zs = [Z[(a, s, f)] for s in SEEDS if (a, s, f) in Z and f"P/{t}" in Z[(a, s, f)].files]
            if not zs:
                continue
            P = np.mean([z[f"P/{t}"] for z in zs], 0); keys = meta["keys"][t]
            o_ = g89.rr_order(P[:N_SH, :N_SH], keys[:N_SH]); S = np.zeros((1, N_SH)); S[0, o_] = -np.arange(N_SH)
            r3[jj] = g84.regret_at_k(S, end0[[i]], end1[[i]], 3)[0]; r1[jj] = g84.regret_at_k(S, end0[[i]], end1[[i]], 1)[0]
            oa = g89.rr_order(P, keys); rk = np.empty(len(keys)); rk[oa] = np.arange(len(keys))
            R0 = arr[f"{t}/R0"]
            if len(R0) >= 10:
                sp[jj] = float(spearmanr(-rk[N_SH:N_SH + len(R0)], R0)[0])
            if a == "E" and all(f"X/{t}" in z.files for z in zs):
                Xm = np.mean([z[f"X/{t}"] for z in zs], 0); xs = []
                for Px in Xm:
                    ox = g89.rr_order(Px, keys[:N_SH]); Sx = np.zeros((1, N_SH)); Sx[0, ox] = -np.arange(N_SH)
                    xs.append(g84.regret_at_k(Sx, end0[[i]], end1[[i]], 3)[0])
                rx[jj] = float(np.mean(xs))
        row = {"regret3": float(np.nanmean(r3)), "regret1": float(np.nanmean(r1)), "sp_median": float(np.nanmedian(sp)), "_r3": r3, "_rx": rx}
        if a == "E":
            Es, gs = [], []
            for f in folds:
                if ("E", 0, f) in Z and "T_all" in Z[("E", 0, f)].files:
                    T = Z[("E", 0, f)]["T_all"]; idx = [i for i in held if groups[i] == f]
                    Es.append(T[idx]); gs += [f] * len(idx)
            if Es:
                row["cos_within"], row["cos_between"] = g86.cosine_within_between(np.vstack(Es), np.array(gs))
        res[a] = row

    def gate(d):
        d = np.asarray(d, float); m = np.isfinite(d); d, g = d[m], hg[m]
        p = 1.0 if np.all(np.abs(d) < 1e-15) else float(wilcoxon(d).pvalue)
        pos = sum(np.mean(d[g == s]) > 0 for s in folds if (g == s).any())
        return {"d": float(d.mean()), "p": p, "src_pos": int(pos), "ci": g84.cluster_bootstrap_ci(d, g, n_boot=10000),
                "ok": bool(d.mean() >= 0.003 and p < 0.05 and pos >= 4)}
    gval = gate(res["B"]["_r3"] - res["E"]["_r3"]) if "E" in res and "B" in res else {"ok": False}
    gcon = gate(res["E"]["_rx"] - res["E"]["_r3"]) if "E" in res and np.isfinite(res["E"]["_rx"]).any() else {"ok": False}
    for a in res:
        res[a].pop("_r3"); res[a].pop("_rx")
    s3c = json.loads((S3C / "analysis.json").read_text(encoding="utf-8"))
    o["heldout"] = {"arms": res, "G_OUT_value": gval, "G_OUT_content": gcon,
                    "s3c_ref": {k: s3c.get("rank", {}).get(k, {}).get("regret3") for k in ("A", "D")}}
    o["verdict"] = verdict_s3r(gin, gval, gcon)
    ref_d = o["heldout"]["s3c_ref"].get("D")
    o["predictions"] = {
        "P1": gin["ok"],
        "P2": bool(o["insample"]["E_minus_B_ww"] is not None and o["insample"]["E_minus_B_ww"] >= 0.02),
        "P3": bool(all(tr_stats[a]["best_epoch_median"] is not None and tr_stats[a]["best_epoch_median"] >= 5 for a in ARMS)),
        "P4": gval["ok"], "P5": gcon["ok"],
        "P6": bool(res.get("E", {}).get("cos_within", 0) >= 0.9),
        "P7": bool(ref_d is not None and res.get("B", {}).get("regret3", 1) <= ref_d),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"verdict": o["verdict"], "G_IN": gin, "E": {k: v for k, v in res.get("E", {}).items()}, "B": res.get("B"),
                      "G_OUT_value": gval, "G_OUT_content": gcon, "training": tr_stats, "predictions": o["predictions"]},
                     ensure_ascii=False, indent=1, default=float), flush=True)
    return o


def heldout_scores(outdir: Path, arm: str) -> dict:
    """留出 56 任务：3 种子 P 均值 → RR → regret@3；shared-30 可用对可用加权准确率（S3-R′ 软权重口径）与好半区准确率。"""
    g84 = importlib.import_module("scripts.84_acts_s3a"); g89 = importlib.import_module("scripts.89_acts_s3c")
    meta, _ = g89.load_data(S3C); tasks = meta["tasks"]; groups = np.array(meta["groups"])
    P2 = g84.load_s2()["pool"]; e0, e1 = P2["end0"], P2["end1"]; L = load_labels(OUT)
    out = {}
    for f in g89.FOLDS:
        zs = [np.load(Path(outdir) / "train" / f"{arm}_s{s}_{f}.npz") for s in SEEDS]
        for i in [i for i in range(len(tasks)) if groups[i] == f]:
            t = tasks[i]; keys = meta["keys"][t]
            P = np.mean([z[f"P/{t}"] for z in zs], 0)
            o_ = g89.rr_order(P[:N_SH, :N_SH], keys[:N_SH]); S = np.zeros((1, N_SH)); S[0, o_] = -np.arange(N_SH)
            r, fl, ns = L[f"{t}/r"], L[f"{t}/fail"], L[f"{t}/ns"]
            use = np.zeros(len(r), bool); use[:N_SH] = True
            a, b, y, w, k = pair_table(r, fl, L["sig"][i], ns, use); m = k == 0
            ok = use & ~fl
            top = (r >= np.median(r[ok])) if ok.any() else np.zeros(len(r), bool)
            mt = m & top[a] & top[b]
            out[i] = {"src": f, "r3": float(g84.regret_at_k(S, e0[[i]], e1[[i]], 3)[0]),
                      "acc": weighted_acc(P[a[m], b[m]], y[m], w[m]) if m.any() else np.nan,
                      "acc_top": weighted_acc(P[a[mt], b[mt]], y[mt], w[mt]) if mt.any() else np.nan}
    return out


def cmd_compare(outdir: Path, ref: Path) -> dict:
    """S3-T 主门 G-GATE：ref（S3-R′）E 臂 − 本档 E 臂的留出 regret@3；另报可用对准确率。"""
    from scipy.stats import wilcoxon
    g84 = importlib.import_module("scripts.84_acts_s3a"); g89 = importlib.import_module("scripts.89_acts_s3c")
    A, R_ = heldout_scores(outdir, "E"), heldout_scores(ref, "E")
    idx = sorted(A); src = np.array([A[i]["src"] for i in idx])
    d = np.array([R_[i]["r3"] - A[i]["r3"] for i in idx])
    p = 1.0 if np.all(np.abs(d) < 1e-15) else float(wilcoxon(d).pvalue)
    npos = int(sum(d[src == s].mean() > 0 for s in g89.FOLDS))
    da = np.array([A[i]["acc"] - R_[i]["acc"] for i in idx]); dt = np.array([A[i]["acc_top"] - R_[i]["acc_top"] for i in idx])
    o = {"G_GATE": {"d": float(d.mean()), "p": p, "src_pos": npos, "ci": g84.cluster_bootstrap_ci(d, src, n_boot=10000)},
         "verdict_gate": verdict_gate(float(d.mean()), p, npos),
         "regret3": {"this_E": float(np.mean([A[i]["r3"] for i in idx])), "ref_E": float(np.mean([R_[i]["r3"] for i in idx]))},
         "acc": {"this_E": float(np.nanmean([A[i]["acc"] for i in idx])), "ref_E": float(np.nanmean([R_[i]["acc"] for i in idx])),
                 "diff_p": float(wilcoxon(da[np.isfinite(da)]).pvalue)},
         "acc_top": {"this_E": float(np.nanmean([A[i]["acc_top"] for i in idx])), "ref_E": float(np.nanmean([R_[i]["acc_top"] for i in idx])),
                     "diff_p": float(wilcoxon(dt[np.isfinite(dt)]).pvalue)},
         "per_src_d": {s: float(d[src == s].mean()) for s in g89.FOLDS}}
    (Path(outdir) / "compare.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(o, ensure_ascii=False, indent=1, default=float), flush=True)
    return o


# ================================================================ 冒烟 / CLI
def cmd_smoke(args) -> None:
    """合成窗口 + 每源 2 任务 + 1 epoch，两臂各跑一次 train，再核输出键。"""
    import types
    base = dict(outdir=args.outdir, device=args.device, resume=False, synthetic=True, max_epochs=1, max_tasks_per_src=2, fold="Cricket", seed=0, stop=args.stop, gate=args.gate)
    for arm in ARMS:
        cmd_train(types.SimpleNamespace(**base, arm=arm))
        with np.load(Path(args.outdir) / "train" / f"{arm}_s0_Cricket.npz") as z:
            assert "insample" in z.files and any(k.startswith("P/") for k in z.files)
            assert (arm == "B") or any(k.startswith("X/") for k in z.files)
    print("[smoke] ok", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "train", "analyze", "smoke", "resume", "synthetic"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--stop", choices=("loss", "acc"), default="loss")
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--e-mode", choices=("settf", "mean", "raw"), default="settf")
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--cons", type=float, default=0.0)                  # S3-W 保守项权重 α（0 = S3-R′）
    ap.add_argument("--cons-ks", type=int, default=2000)
    ap.add_argument("--cons-gens", type=int, default=3)
    ap.add_argument("--cons-top", type=int, default=5)
    ap.add_argument("--cons-ref", type=int, default=16)
    ap.add_argument("--cons-period", type=int, default=3)
    ap.add_argument("--compare-ref", default=None)
    ap.add_argument("--max-tasks-per-src", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prep:
        cmd_prep(args)
    if args.train:
        cmd_train(args)
    if args.smoke:
        cmd_smoke(args)
    if args.analyze:
        cmd_analyze(args)
        if args.compare_ref:
            cmd_compare(Path(args.outdir), Path(args.compare_ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
