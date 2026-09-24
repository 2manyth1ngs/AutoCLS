"""WP17-ACTS S3-B（results/acts/s3b/PROTOCOL.md）：任务嵌入 E（E_pt 每任务 TS2Vec / E_sh 折共享 TS2Vec × mean / settf × {E, E ⊕ ④}）预测轴坐标 a(t)。

阶段（服务器只需 tasks.json + 源数据 + targets.npz，不需要标签缓存）：
  --prep                 本地：由 S2 缓存算每折 μ / B / A / ④ 探针特征 / end0·end1 → results/acts/s3b/targets.npz
  --extract-pt           GPU：每任务 fit 单变量 TS2Vec（seed 0）→ E_pt/{task}.npz（--parts / --part 分片，--resume）
  --fit-shared           GPU：每折（17 + ALL）在训练源整集上 fit 共享 TS2Vec → E_sh/{fold}/{task}.npz
  --train-settf          GPU：--arm pt|sh --fold <源|ALL> [--combo] → settf/{arm}{+p}/{fold}.json（3 seed 平均 â、验证源选 K、cross-E）
  --analyze              本地：mean 臂（84 管线）+ settf 臂读数、门、表
  --smoke                本地冒烟（小 iters / 2 任务 / 2 折 × 3 epoch）
纯函数不含数据集概念（铁律 7）；TS2Vec / SetTF 取自 autocls/task_embed（AutoCTS++ 对应项见其注释，铁律 4）。
"""
from __future__ import annotations

import argparse
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

from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3b"
S2_DIR = ROOT / "results" / "acts" / "s2"
N_WINDOWS = 100
T_MAX = 152
REPR = 256
PT_BATCH = 16                 # TS2Vec 原码默认 batch
SH_BATCH = 32                 # = MX-3-PQ TS2VEC_FIT_BS
SH_ITERS = 600
SH_MAX_PER_DS = 4096
SH_MAX_LEN = 1000              # 共享池切段上限（TS2Vec max_train_length 语义；原码 3000，显存按 B·T² 走，共享 4090 上取 1000，PROTOCOL §8 记偏差）
SETTF = dict(nz=256, max_epochs=100, patience=10, lr=1e-4, batch_tasks=16, sub_windows=32)
SEEDS = (0, 1, 2)
E_ARMS = ("pt_mean", "sh_mean", "pt_settf", "sh_settf")
C_ARMS = ("pt_mean+p", "sh_mean+p", "pt_settf+p", "sh_settf+p")


# ================================================================ 纯函数
def ts2vec_iters(size: int) -> int:
    """TS2Vec 原码 n_iters 规则：train_data.size ≤ 1e5 → 200，否则 600。"""
    return 200 if int(size) <= 100_000 else 600


def window_idx(n: int, task_id: str, m: int = N_WINDOWS) -> np.ndarray:
    from autocls.labeling import run_seed_for
    if n <= m:
        return np.arange(n)
    rng = np.random.default_rng(run_seed_for(task_id, "s3b_windows"))
    return np.sort(rng.choice(n, size=m, replace=False))


def mean_pool_E(reprs: np.ndarray) -> np.ndarray:
    return np.asarray(reprs, dtype=np.float64).mean(axis=(0, 1))


def cosine_within_between(E: np.ndarray, groups: Sequence) -> tuple[float, float]:
    E = np.asarray(E, float); g = np.asarray(groups)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    C = En @ En.T
    same = g[:, None] == g[None, :]
    off = ~np.eye(len(E), dtype=bool)
    return float(C[same & off].mean()), float(C[~same].mean())


def _row(task_id: str) -> int:
    return int(task_id.rsplit("_a", 1)[1])


def cross_assign(tasks: Sequence[str], groups: Sequence, order: Sequence[str]) -> list[int]:
    """cross-E：每任务改用循环下一个源同 OA 行号任务（不存在取该源第 0 行）。"""
    groups = np.asarray(groups)
    by = {s: {_row(t): i for i, t in enumerate(tasks) if groups[i] == s} for s in order}
    out = []
    for i, t in enumerate(tasks):
        nxt = order[(list(order).index(groups[i]) + 1) % len(order)]
        rows = by[nxt]
        out.append(rows.get(_row(t), rows[min(rows)]))
    return out


def val_source(order: Sequence[str], heldout: str) -> str:
    order = list(order)
    return order[(order.index(heldout) + 1) % len(order)]


def fit_pool_sources(pool: Sequence[str], heldout: str) -> list[str]:
    return [s for s in pool if s != heldout]


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def verdict_s3b(E: Mapping[str, Mapping], C: Mapping[str, Mapping]) -> str:
    """§3：主门（E-only 臂，Holm 到 4 臂）；+REPLACE（对 ④ 非劣效上界 ≤ 0.005）；+ADD（E ⊕ ④ 优于 ④，Holm 到该组臂数）。"""
    names = list(E)
    adj = holm([E[k]["p_vs_land"] for k in names])
    win = any(E[k]["rho_diff_ci_lo"] > 0 and adj[i] < 0.05 and E[k]["delta_vs_land"] > 0 for i, k in enumerate(names))
    code = "S3B-E-WIN" if win else "S3B-E-NOT-BETTER"
    if any(E[k]["vs_probe_ub"] <= 0.005 for k in names):
        code += "+REPLACE"
    cn = list(C)
    cadj = holm([C[k]["p_vs_probe"] for k in cn])
    if any(C[k]["delta_vs_probe"] > 0 and cadj[i] < 0.05 for i, k in enumerate(cn)):
        code += "+ADD"
    return code


def split_long(pool: np.ndarray, max_len: int) -> np.ndarray:
    """← ts2vec.py fit 的 max_train_length 分支（split_with_nan）：T > max_len 时沿时间切成 ceil(T/max_len) 段（尾段 NaN 右填充），丢弃全 NaN 段。"""
    pool = np.asarray(pool, np.float32)
    m, t, c = pool.shape
    if t <= max_len:
        return pool
    k = int(np.ceil(t / max_len))
    pad = np.full((m, k * max_len, c), np.nan, np.float32); pad[:, :t] = pool
    seg = pad.reshape(m, k, max_len, c).reshape(m * k, max_len, c)
    keep = ~np.all(np.isnan(seg), axis=(1, 2))
    return seg[keep]


def subsample(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= k:
        return x
    return x[np.sort(rng.choice(x.shape[0], size=k, replace=False))]


# ================================================================ SetTF 回归头
def train_settf(reprs: Sequence[np.ndarray], a: np.ndarray, tr: np.ndarray, va: np.ndarray, *, extra: Optional[np.ndarray], seed: int,
                dim_input: int = REPR, nz: int = 256, max_epochs: int = 100, patience: int = 10, lr: float = 1e-4,
                batch_tasks: int = 16, sub_windows: int = 32, device: str = "cpu") -> dict:
    """TaskSetEncoder（← ahc_engine.py intra sabPF + inter sabP）+ Linear 头回归 a(t)（标准化目标，MSE）。
    训练时每任务每步子采 sub_windows 个窗口（方案 §8-6 (i)，AutoCTS++ 无此项）；验证 / 预测用全部窗口。验证 MSE 早停，保留最好权重。"""
    import torch
    from autocls.task_embed import TaskSetEncoder
    tr, va = np.asarray(tr), np.asarray(va)
    a = np.asarray(a, float)
    am, asd = a[tr].mean(0), a[tr].std(0)
    asd = np.where(asd > 1e-12, asd, 1.0)
    y = torch.tensor((a - am) / asd, dtype=torch.float32)
    if extra is not None:
        ex = np.asarray(extra, float)
        em, esd = ex[tr].mean(0), ex[tr].std(0)
        esd = np.where(esd > 1e-12, esd, 1.0)
        xe = torch.tensor((ex - em) / esd, dtype=torch.float32)
    torch.manual_seed(int(seed))
    enc = TaskSetEncoder(dim_input=dim_input, nz=nz)
    head_in = nz + (0 if extra is None else xe.shape[1])
    head = torch.nn.Linear(head_in, a.shape[1])
    enc.to(device); head.to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(int(seed))

    def forward(idx, sub):
        xs = [torch.as_tensor(np.asarray(subsample(reprs[i], sub_windows, rng) if sub else reprs[i]), dtype=torch.float32, device=device) for i in idx]
        h = enc.set_encode(xs)
        if extra is not None:
            h = torch.cat([h, xe[idx].to(device)], dim=1)
        return head(h), h

    def eval_mse(idx):
        enc.eval(); head.eval()
        with torch.no_grad():
            out = torch.cat([forward(idx[j:j + batch_tasks], False)[0] for j in range(0, len(idx), batch_tasks)])
            return float(((out.cpu() - y[idx]) ** 2).mean())

    best, best_state, best_ep, bad, losses, vlosses = np.inf, None, -1, 0, [], []
    for ep in range(int(max_epochs)):
        enc.train(); head.train()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for j in range(0, len(perm), batch_tasks):
            idx = perm[j:j + batch_tasks]
            out, _ = forward(idx, True)
            loss = ((out - y[idx].to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        losses.append(tot / max(nb, 1))
        v = eval_mse(va) if len(va) else losses[-1]
        vlosses.append(v)
        if v < best - 1e-9:
            best, best_ep, bad = v, ep, 0
            best_state = ([p.detach().cpu().clone() for p in enc.state_dict().values()], [p.detach().cpu().clone() for p in head.state_dict().values()])
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        enc.load_state_dict(dict(zip(enc.state_dict().keys(), best_state[0])))
        head.load_state_dict(dict(zip(head.state_dict().keys(), best_state[1])))

    def predict(idx, swap: Optional[Sequence[int]] = None):
        """swap[i] = 用第 swap[i] 个任务的窗口代替第 idx[i] 个（cross-E）；返回原尺度 â。"""
        idx = np.asarray(idx)
        src = idx if swap is None else np.asarray(swap)
        enc.eval(); head.eval()
        outs, embs = [], []
        with torch.no_grad():
            for j in range(0, len(idx), batch_tasks):
                xs = [torch.as_tensor(np.asarray(reprs[i]), dtype=torch.float32, device=device) for i in src[j:j + batch_tasks]]
                h = enc.set_encode(xs)
                embs.append(h.cpu().numpy())
                if extra is not None:
                    h = torch.cat([h, xe[idx[j:j + batch_tasks]].to(device)], dim=1)
                outs.append(head(h).cpu().numpy())
        predict.last_emb = np.vstack(embs)
        return np.vstack(outs) * asd + am

    return {"predict": predict, "train_loss": losses, "val_loss": vlosses, "best_epoch": best_ep, "head_in": head_in}


# ================================================================ 数据（服务器侧：tasks.json + 源数据）
def task_meta() -> tuple[list[str], dict, list[str], list[str]]:
    """S3-B 任务表：训练池 160（S2 pool_after_gates，Ethanol 前 8 行）+ HMD / SCP2 各 8。返回 (任务, meta, 池源序, 额外任务)。"""
    g84 = importlib.import_module("scripts.84_acts_s3a")
    tj = json.loads((S2_DIR / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    an = json.loads((S2_DIR / "analysis_S.json").read_text(encoding="utf-8"))
    order = list(tj)
    pool = g84.pool_tasks(order, tj, an["pool_after_gates"], an["reverse_demoted_to_8"])
    extra = [t for t in order if tj[t]["dataset"] in ("HandMovementDirection", "SelfRegulationSCP2")]
    srcs = list(dict.fromkeys(tj[t]["dataset"] for t in pool))
    return pool + extra, tj, srcs, extra


_DS: dict = {}


def dataset(name: str):
    ac = importlib.import_module("scripts.acts_common")
    if name not in _DS:
        _DS[name] = ac.load_dataset_full(name)
    return _DS[name]


def task_X(tid: str, meta: Mapping) -> np.ndarray:
    """= ac.task_arrays_cv 的 X：物化切片 → 按通道在全部样本上 z-score。"""
    g83 = importlib.import_module("scripts.83_acts_corpus")
    from autocls.data.task_slicer import materialize_task
    t = g83.tasks_from_meta({tid: meta[tid]})[tid]
    X, y = dataset(meta[tid]["dataset"])
    Xt, _ = materialize_task(t, X, y)
    raw = Xt.astype(np.float64)
    flat = raw.reshape(-1, raw.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32)


def encode_windows(w, X: np.ndarray, idx: np.ndarray, chunk: int = 10) -> np.ndarray:
    """窗口 → 单变量逐通道编码 → 跨通道均值（Eq.10）→ 时间轴池化到 T_MAX；fp16。"""
    from autocls.task_embed import pool_time_axis
    out = []
    for j in range(0, len(idx), chunk):
        s = X[idx[j:j + chunk]]
        m, t, c = s.shape
        r = w.encode(s.transpose(0, 2, 1).reshape(m * c, t, 1), batch_size=256)
        r = r.reshape(m, c, *r.shape[1:]).mean(axis=1)
        out.append(pool_time_axis(r, T_MAX).astype(np.float16))
    return np.concatenate(out)


def _device_name(device: str) -> str:
    import torch
    return torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"


def _parts(items: Sequence, parts: int, part: int) -> list:
    return [x for i, x in enumerate(items) if parts <= 1 or i % parts == part]


def cmd_extract_pt(args) -> None:
    from autocls.task_embed import TS2VecWrapper
    tasks, meta, _srcs, _ = task_meta()
    d = Path(args.outdir) / "E_pt"; d.mkdir(parents=True, exist_ok=True)
    todo = _parts(tasks if not args.tasks else args.tasks, args.parts, args.part)
    todo = sorted(todo, key=lambda t: meta[t]["dataset"])            # 同数据集相邻，少重载
    for tid in todo:
        p = d / f"{tid}.npz"
        if p.exists() and args.resume:
            continue
        X = task_X(tid, meta)
        n, t, c = X.shape
        pool = X.transpose(0, 2, 1).reshape(n * c, t, 1)
        iters = ts2vec_iters(pool.size) if args.iters_cap is None else min(ts2vec_iters(pool.size), args.iters_cap)
        t0 = time.time()
        split_len = 0
        try:
            w = TS2VecWrapper(input_dims=1, output_dims=REPR, hidden_dims=64, depth=10, device=args.device, batch_size=PT_BATCH, seed=0)
            w.fit(pool, n_iters=iters)
        except Exception as e:                                   # 严格口径 OOM → 按 max_train_length 语义切段到 SH_MAX_LEN 重训（npz 记 split_len）
            if "out of memory" not in str(e).lower():
                raise
            import torch
            w = None; torch.cuda.empty_cache()
            split_len = SH_MAX_LEN
            w = TS2VecWrapper(input_dims=1, output_dims=REPR, hidden_dims=64, depth=10, device=args.device, batch_size=PT_BATCH, seed=0)
            w.fit(split_long(pool, SH_MAX_LEN), n_iters=iters)
            print(f"[pt] {tid} OOM → split_len {SH_MAX_LEN}", flush=True)
        w.freeze()
        t1 = time.time()
        r = encode_windows(w, X, window_idx(n, tid))
        t2 = time.time()
        np.savez_compressed(p, windows_repr=r, E_mean=mean_pool_E(r), fit_seconds=t1 - t0, encode_seconds=t2 - t1, iters=iters,
                            n_pool=pool.shape[0], split_len=split_len, device=_device_name(args.device))
        print(f"[pt] {tid} X={X.shape} iters={iters} fit {t1 - t0:.0f}s enc {t2 - t1:.0f}s", flush=True)
        del w
        import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def cmd_fit_shared(args) -> None:
    from autocls.task_embed import fit_shared_ts2vec, TS2VecWrapper
    b2 = importlib.import_module("scripts.2b_generate_labels_route_a")
    tasks, meta, srcs, _ = task_meta()
    folds = args.folds or (srcs + ["ALL"])
    folds = _parts(folds, args.parts, args.part)
    for fold in folds:
        fd = Path(args.outdir) / "E_sh" / fold; fd.mkdir(parents=True, exist_ok=True)
        wp = fd / "shared_ts2vec.pt"
        fit_ds = fit_pool_sources(srcs, fold)
        assert fold not in fit_ds and "HandMovementDirection" not in fit_ds and "SelfRegulationSCP2" not in fit_ds
        if wp.exists() and args.resume:
            w = TS2VecWrapper.load(wp, device=args.device)
        else:
            per_ds = []
            for s in fit_ds:
                X, _y = dataset(s)
                raw = X.astype(np.float64); flat = raw.reshape(-1, raw.shape[-1])
                mu, sd = flat.mean(0), flat.std(0); sd = np.where(sd < 1e-8, 1.0, sd)
                per_ds.append((s, ((raw - mu) / sd).astype(np.float32), None, None))
            pool = split_long(b2.build_ts2vec_pool(per_ds, SH_MAX_PER_DS if args.max_per_ds is None else args.max_per_ds, 0), SH_MAX_LEN)
            t0 = time.time()
            w = fit_shared_ts2vec(pool, n_iters=SH_ITERS if args.iters_cap is None else min(SH_ITERS, args.iters_cap), device=args.device, seed=0, batch_size=SH_BATCH)
            w.save(wp)
            (fd / "meta.json").write_text(json.dumps({"fold": fold, "fit_datasets": fit_ds, "n_instances": int(pool.shape[0]), "t_max": int(pool.shape[1]),
                                                      "n_iters": int(w.n_iters), "batch_size": SH_BATCH, "seconds": round(time.time() - t0, 1),
                                                      "device_name": _device_name(args.device)}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[sh] {fold} fit {pool.shape} {time.time() - t0:.0f}s", flush=True)
            del pool, per_ds
        todo = sorted(tasks if not args.tasks else args.tasks, key=lambda t: meta[t]["dataset"])
        for tid in todo:
            p = fd / f"{tid}.npz"
            if p.exists() and args.resume:
                continue
            X = task_X(tid, meta)
            t1 = time.time()
            r = encode_windows(w, X, window_idx(X.shape[0], tid))
            np.savez_compressed(p, windows_repr=r, E_mean=mean_pool_E(r), encode_seconds=time.time() - t1, device=_device_name(args.device))
        print(f"[sh] {fold} encoded {len(todo)}", flush=True)


# ================================================================ 本地准备：每折目标
def cmd_prep(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a")
    g85 = importlib.import_module("scripts.85_acts_s3ap")
    tasks, meta, srcs, extra = task_meta()
    data = g85.load_all()
    D = data["pool"]
    allb = data["all"]
    pos = {t: i for i, t in enumerate(allb["tasks"])}
    ex = [pos[t] for t in extra]
    assert list(D["tasks"]) == tasks[:len(D["tasks"])]
    Z = np.vstack([D["Z"], allb["Z"][ex]]); end0 = np.vstack([D["end0"], allb["end0"][ex]]); end1 = np.vstack([D["end1"], allb["end1"][ex]])
    acc25 = np.vstack([D["acc25"], allb["acc25"][ex]]); pf = np.concatenate([D["pfloor"], allb["pfloor"][ex]]); maj = np.concatenate([D["maj"], allb["maj"][ex]])
    groups = np.array([meta[t]["dataset"] for t in tasks])
    n_pool = len(D["tasks"])
    b_ref = importlib.import_module("scripts.85_acts_s3ap").orient_bref(D)
    probe = g84.probe_feature(acc25, pf, maj)
    out = {"tasks": np.array(tasks), "groups": groups, "n_pool": n_pool, "order": np.array(srcs), "b_ref": b_ref,
           "Z": Z, "end0": end0, "end1": end1}
    for fold in srcs + ["ALL"]:
        tr = np.array([i for i in range(n_pool) if groups[i] != fold])
        ctx = g84._fold(Z, tr, 3, b_ref)
        vs = val_source(srcs, fold) if fold != "ALL" else srcs[0]
        out[f"{fold}/tr"] = tr; out[f"{fold}/mu"] = ctx["mu"]; out[f"{fold}/B"] = ctx["B"]; out[f"{fold}/A"] = ctx["A"]
        out[f"{fold}/X4"] = probe({**ctx, "train_idx": tr}); out[f"{fold}/val"] = np.array(vs)
    np.savez_compressed(Path(args.outdir) / "targets.npz", **out)
    print(f"[prep] {len(tasks)} tasks, {len(srcs)} folds + ALL → targets.npz", flush=True)


def _load_targets(outdir: Path) -> dict:
    with np.load(Path(outdir) / "targets.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _reprs(outdir: Path, arm: str, fold: str, tasks: Sequence[str]) -> list[np.ndarray]:
    d = Path(outdir) / ("E_pt" if arm == "pt" else f"E_sh/{fold}")
    out = []
    for t in tasks:
        with np.load(d / f"{t}.npz") as z:
            out.append(z["windows_repr"])
    return out


def cmd_train_settf(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a")
    T = _load_targets(Path(args.outdir))
    tasks = [str(t) for t in T["tasks"]]; groups = T["groups"].astype(str); order = [str(s) for s in T["order"]]
    n_pool = int(T["n_pool"]); fold = args.fold
    tag = args.arm + ("+p" if args.combo else "")
    od = Path(args.outdir) / "settf" / tag; od.mkdir(parents=True, exist_ok=True)
    op = od / f"{fold}.json"
    if op.exists() and args.resume:
        print(f"[settf] skip {op}"); return
    reprs = _reprs(Path(args.outdir), args.arm, fold if args.arm == "sh" else "", tasks)
    tr_all = T[f"{fold}/tr"]; vs = str(T[f"{fold}/val"])
    va = np.array([i for i in tr_all if groups[i] == vs]); tr = np.array([i for i in tr_all if groups[i] != vs])
    te = np.array([i for i in range(n_pool) if groups[i] == fold]) if fold != "ALL" else np.arange(n_pool, len(tasks))
    A = T[f"{fold}/A"][:, :3]; mu, B = T[f"{fold}/mu"], T[f"{fold}/B"]
    extra = T[f"{fold}/X4"] if args.combo else None
    cfg = dict(SETTF); cfg.update({k: v for k, v in (("max_epochs", args.max_epochs),) if v is not None})
    swap = None
    if fold != "ALL":
        cm = cross_assign(tasks[:n_pool], groups[:n_pool], order)
        swap = [cm[i] for i in te]
    preds, preds_x, embs, info = [], [], [], []
    t0 = time.time()
    for sd in (args.seeds or SEEDS):
        r = train_settf(reprs, A, tr, va, extra=extra, seed=int(sd), device=args.device, **cfg)
        allidx = np.arange(len(tasks))
        preds.append(r["predict"](allidx)); embs.append(r["predict"].last_emb)
        if swap is not None:
            preds_x.append(r["predict"](te, swap=swap))
        info.append({"seed": int(sd), "best_epoch": r["best_epoch"], "train_loss": r["train_loss"], "val_loss": r["val_loss"]})
        print(f"[settf] {tag} {fold} seed {sd} best_ep {r['best_epoch']} val {min(r['val_loss']):.4f} {time.time() - t0:.0f}s", flush=True)
    P = np.mean(preds, axis=0)
    regs = {}
    for K in (1, 2, 3):
        S = g84.scores_from(mu, B[:, :K], P[va][:, :K])
        regs[K] = float(np.mean(g84.regret_at_k(S, T["end0"][va], T["end1"][va], 3)))
    best = min(regs.values()); K = int(min(k for k, v in regs.items() if v <= best + 1e-12))
    res = {"fold": fold, "arm": tag, "val_source": vs, "K": K, "val_regret": regs, "a_hat": P.tolist(), "emb": np.mean(embs, axis=0).tolist(),
           "te": te.tolist(), "a_hat_cross": (np.mean(preds_x, axis=0).tolist() if preds_x else None), "seeds": info,
           "seconds": time.time() - t0, "device": _device_name(args.device), "cfg": cfg}
    op.write_text(json.dumps(res), encoding="utf-8")


# ================================================================ 分析（本地）
def _ci(x, groups):
    return importlib.import_module("scripts.84_acts_s3a").cluster_bootstrap_ci(np.asarray(x, float), groups)


def _e_means(outdir: Path, arm: str, tasks: Sequence[str], folds: Sequence[str]) -> dict:
    """E_mean：pt → {'': (n,256)}；sh → {fold: (n,256)}（服务器回传的 _means.npz 优先，否则逐任务 npz）。"""
    out = {}
    for f in ([""] if arm == "pt" else folds):
        d = Path(outdir) / ("E_pt" if arm == "pt" else f"E_sh/{f}")
        mp = d / "_means.npz"
        if mp.exists():
            with np.load(mp) as z:
                m = dict(zip(z["tasks"].astype(str), z["E_mean"]))
            out[f] = np.array([m[t] for t in tasks])
        else:
            out[f] = np.array([np.load(d / f"{t}.npz")["E_mean"] for t in tasks])
    return out


def cmd_pack_means(args) -> None:
    """服务器侧：把每目录逐任务 E_mean / 秒数汇成 _means.npz（回传用，不拉 windows_repr）。"""
    base = Path(args.outdir)
    for d in [base / "E_pt"] + sorted((base / "E_sh").glob("*")):
        if not d.is_dir():
            continue
        fs = sorted(p for p in d.glob("*.npz") if not p.name.startswith("_"))
        if not fs:
            continue
        ts, em, fsec, esec = [], [], [], []
        for p in fs:
            with np.load(p) as z:
                ts.append(p.stem); em.append(z["E_mean"]); fsec.append(float(z["fit_seconds"]) if "fit_seconds" in z.files else 0.0); esec.append(float(z["encode_seconds"]))
        np.savez(d / "_means.npz", tasks=np.array(ts), E_mean=np.array(em), fit_seconds=np.array(fsec), encode_seconds=np.array(esec))
        print(f"[pack] {d} {len(ts)}", flush=True)


def _settf_arm(outdir: Path, tag: str, T: Mapping, folds: Sequence[str]) -> Optional[dict]:
    g84 = importlib.import_module("scripts.84_acts_s3a")
    n_pool = int(T["n_pool"]); groups = T["groups"].astype(str)[:n_pool]
    a_hat = np.zeros((n_pool, 3)); a_true = np.zeros((n_pool, 3)); r1 = np.zeros(n_pool); r3 = np.zeros(n_pool); r3x = np.full(n_pool, np.nan)
    Ks, embs = [], np.zeros((n_pool, SETTF["nz"]))
    for f in folds:
        p = Path(outdir) / "settf" / tag / f"{f}.json"
        if not p.exists():
            return None
        R = json.loads(p.read_text(encoding="utf-8"))
        te = np.array(R["te"]); P = np.array(R["a_hat"]); K = int(R["K"]); Ks.append(K)
        mu, B, A = T[f"{f}/mu"], T[f"{f}/B"], T[f"{f}/A"]
        a_hat[te] = P[te]; a_true[te] = A[te, :3]; embs[te] = np.array(R["emb"])[te]
        S = g84.scores_from(mu, B[:, :K], P[te][:, :K])
        r1[te] = g84.regret_at_k(S, T["end0"][te], T["end1"][te], 1); r3[te] = g84.regret_at_k(S, T["end0"][te], T["end1"][te], 3)
        if R.get("a_hat_cross") is not None:
            Sx = g84.scores_from(mu, B[:, :K], np.array(R["a_hat_cross"])[:, :K])
            r3x[te] = g84.regret_at_k(Sx, T["end0"][te], T["end1"][te], 3)
    ext = None
    pa = Path(outdir) / "settf" / tag / "ALL.json"
    if pa.exists():
        R = json.loads(pa.read_text(encoding="utf-8")); ext = np.array(R["a_hat"])[n_pool:, 0]
    return {"a_hat": a_hat, "a_true": a_true, "regret1": r1, "regret3": r3, "regret3_cross": r3x, "k_chosen": Ks, "emb": embs, "extra_a1": ext}


def cmd_analyze(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a")
    g85 = importlib.import_module("scripts.85_acts_s3ap")
    outdir = Path(args.outdir)
    T = _load_targets(outdir)
    tasks = [str(t) for t in T["tasks"]]; n_pool = int(T["n_pool"]); order = [str(s) for s in T["order"]]
    groups = T["groups"].astype(str)[:n_pool]
    data = g85.load_all(); D = data["pool"]; b_ref = T["b_ref"]
    assert list(D["tasks"]) == tasks[:n_pool]
    res_all = {}
    # mean 臂 + 参照（S3-A 同一管线）
    em_pt = _e_means(outdir, "pt", tasks, order)[""]
    em_sh = _e_means(outdir, "sh", tasks, order + ["ALL"])
    cur = {"s": None}

    def sh_feat(combo: bool):
        pf = g84.probe_feature(D["acc25"], D["pfloor"], D["maj"])
        def f(ctx):
            missing = set(order) - set(groups[ctx["train_idx"]].tolist())
            if len(missing) == 1:
                cur["s"] = next(iter(missing))
            assert cur["s"] in missing
            X = em_sh[cur["s"]][:n_pool]
            return np.column_stack([X, pf(ctx)]) if combo else X
        return f

    pf = g84.probe_feature(D["acc25"], D["pfloor"], D["maj"])
    feats = {"land25": g84.land25_feature(D["Z25"]), "probe": pf, "pt_mean": em_pt[:n_pool],
             "pt_mean+p": (lambda ctx: np.column_stack([em_pt[:n_pool], pf(ctx)]))}
    res = g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], feats, b_ref=b_ref)
    for k in ("sh_mean", "sh_mean+p"):
        res.update({k: g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], {k: sh_feat(k.endswith("+p"))}, b_ref=b_ref)[k]})
    blind = res["_blind"]["regret3"]
    c0 = {"land25": res["land25"]["mean_regret3"], "probe": res["probe"]["mean_regret3"], "blind": float(blind.mean())}
    arms = {}
    for k in ("pt_mean", "sh_mean", "pt_mean+p", "sh_mean+p"):
        arms[k] = {"a_hat": res[k]["a_hat"], "a_true": res[k]["a_true"], "regret1": res[k]["regret1"], "regret3": res[k]["regret3"], "k_chosen": res[k]["k_chosen"]}
    for tag in ("pt_settf", "sh_settf", "pt_settf+p", "sh_settf+p"):
        a = _settf_arm(outdir, tag.replace("_settf", ""), T, order)
        if a is not None:
            arms[tag] = a
    ref = {"land25": res["land25"], "probe": res["probe"]}
    rows, E_gate, C_gate = {}, {}, {}
    for k, a in arms.items():
        r = {"rho_a1": float(spearmanr(a["a_hat"][:, 0], a["a_true"][:, 0])[0]), "regret1": float(np.mean(a["regret1"])), "regret3": float(np.mean(a["regret3"])),
             "delta_vs_blind": float(np.mean(blind - a["regret3"])), "k_chosen": list(a["k_chosen"])}
        r["rho_ci"] = g84.cluster_bootstrap_ci(np.column_stack([a["a_hat"][:, 0], a["a_true"][:, 0]]), groups, stat=lambda x: float(spearmanr(x[:, 0], x[:, 1])[0]))
        dl = ref["land25"]["regret3"] - a["regret3"]
        r["delta_vs_land"], r["delta_vs_land_ci"], r["p_vs_land"] = float(dl.mean()), _ci(dl, groups), g84.source_wilcoxon(dl, groups)
        st = np.column_stack([a["a_hat"][:, 0], ref["land25"]["a_hat"][:, 0], a["a_true"][:, 0]])
        r["rho_diff_vs_land"] = r["rho_a1"] - ref["land25"]["rho_a1"]
        r["rho_diff_ci"] = g84.cluster_bootstrap_ci(st, groups, stat=lambda x: float(spearmanr(x[:, 0], x[:, 2])[0] - spearmanr(x[:, 1], x[:, 2])[0]))
        dp = a["regret3"] - ref["probe"]["regret3"]
        r["vs_probe"], r["vs_probe_ci"] = float(dp.mean()), _ci(dp, groups)
        r["p_vs_probe"] = g84.source_wilcoxon(-dp, groups)
        if "regret3_cross" in a and np.isfinite(a["regret3_cross"]).all():
            dx = a["regret3_cross"] - a["regret3"]
            r["cross_delta"], r["cross_ci"], r["cross_p"] = float(dx.mean()), _ci(dx, groups), g84.source_wilcoxon(dx, groups)
        if "emb" in a:
            r["cos_within"], r["cos_between"] = cosine_within_between(a["emb"], groups)
        rows[k] = r
        if k in E_ARMS:
            E_gate[k] = {"rho_diff_ci_lo": r["rho_diff_ci"][0], "p_vs_land": r["p_vs_land"], "delta_vs_land": r["delta_vs_land"], "vs_probe_ub": r["vs_probe_ci"][1]}
        if k in C_ARMS:
            C_gate[k] = {"delta_vs_probe": -r["vs_probe"], "p_vs_probe": r["p_vs_probe"]}
    for nm, em in (("pt_mean", em_pt[:n_pool]), ("sh_mean(ALL)", em_sh["ALL"][:n_pool])):
        rows.setdefault(nm.split("(")[0], {})["E_cos_within_between"] = cosine_within_between(em, groups)
    code = verdict_s3b(E_gate, C_gate) if len(E_gate) == 4 and len(C_gate) == 4 else "INCOMPLETE(" + ",".join(sorted(set(E_ARMS + C_ARMS) - set(arms))) + ")"
    # 成本
    rows_s2 = importlib.import_module("scripts.83_acts_corpus")._g35.load_done(S2_DIR / "runs.jsonl")
    plan = json.loads((S2_DIR / "plan_S.json").read_text(encoding="utf-8"))
    cell_sec = {}
    for c in plan["cells"]:
        r_ = rows_s2.get(c["key"])
        if c["arm"] == "shared" and r_ and not r_.get("error"):
            cell_sec.setdefault(c["task_id"], []).append(float(r_.get("pretrain_seconds") or 0) + float(r_.get("eval_seconds") or 0))
    cost = {}
    mp = outdir / "E_pt" / "_means.npz"
    if mp.exists():
        with np.load(mp) as z:
            ts = z["tasks"].astype(str); fs, es = z["fit_seconds"], z["encode_seconds"]
            eq_pt = [(fs[i] + es[i]) / np.median(cell_sec[t]) for i, t in enumerate(ts) if t in cell_sec]
            eq_enc = [es[i] / np.median(cell_sec[t]) for i, t in enumerate(ts) if t in cell_sec]
        cost = {"pt_median_equiv": float(np.median(eq_pt)), "pt_mean_equiv": float(np.mean(eq_pt)), "encode_median_equiv": float(np.median(eq_enc))}
    # HMD / SCP2 表征（mean 臂 17 源全训 ridge；settf 用 ALL.json）
    rep = {}
    ctx = g84._fold(np.vstack([D["Z"], T["Z"][n_pool:]]), np.arange(n_pool), 3, b_ref)
    for nm, X in (("pt_mean", em_pt), ("sh_mean", em_sh["ALL"])):
        a = g84.choose_alpha(X[:n_pool], ctx["A"][:n_pool, 0], groups, g84.ALPHAS)
        m = g84.ridge_fit(X[:n_pool], ctx["A"][:n_pool, 0], a)
        pr = g84.ridge_predict(m, X[n_pool:])
        eg = T["groups"].astype(str)[n_pool:]
        rep[nm] = {s: float(pr[eg == s].mean()) for s in dict.fromkeys(eg.tolist())}
    for tag in ("pt_settf", "sh_settf"):
        if tag in arms and arms[tag]["extra_a1"] is not None:
            eg = T["groups"].astype(str)[n_pool:]
            rep[tag] = {s: float(arms[tag]["extra_a1"][eg == s].mean()) for s in dict.fromkeys(eg.tolist())}
    eg = T["groups"].astype(str)[n_pool:]
    rep["true"] = {s: float(ctx["A"][n_pool:, 0][eg == s].mean()) for s in dict.fromkeys(eg.tolist())}
    rep["train_q25"] = float(np.quantile(ctx["A"][:n_pool, 0], 0.25))
    out = {"verdict": code, "c0": c0, "ref": {k: {"rho_a1": v["rho_a1"], "regret3": v["mean_regret3"]} for k, v in ref.items()},
           "blind_regret3": float(blind.mean()), "rows": rows, "cost": cost, "representation": rep}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


def _f(x, nd=3):
    try:
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-B 读数（{o['verdict']}）", "", f"C0：③ {_f(o['c0']['land25'], 4)} / ④ {_f(o['c0']['probe'], 4)} / 任务盲 {_f(o['c0']['blind'], 4)}（S3-A 0.0058 / 0.0098 / 0.0213）", "",
         f"参照：③ land25 ρ {_f(o['ref']['land25']['rho_a1'], 2)} regret@3 {_f(o['ref']['land25']['regret3'])}；④ ρ {_f(o['ref']['probe']['rho_a1'], 2)} regret@3 {_f(o['ref']['probe']['regret3'])}；任务盲 {_f(o['blind_regret3'])}", "",
         "| 臂 | ρ_a1 [CI] | ρ − ρ③ [CI] | regret@1 | regret@3 | vs ③ Δ [CI] p | vs ④（E − ④）[CI] | cross-E Δ [CI] p | 余弦 集内/跨集 | K=1/2/3 |",
         "|---|---|---|---:|---:|---|---|---|---|---|"]
    for k, r in o["rows"].items():
        if "rho_a1" not in r:
            continue
        kc = [r["k_chosen"].count(K) for K in (1, 2, 3)]
        cr = f"{_f(r['cross_delta'])} [{_f(r['cross_ci'][0])}, {_f(r['cross_ci'][1])}] {_f(r['cross_p'], 3)}" if "cross_delta" in r else "—"
        cs = f"{_f(r['cos_within'], 2)}/{_f(r['cos_between'], 2)}" if "cos_within" in r else (f"E {_f(r['E_cos_within_between'][0], 2)}/{_f(r['E_cos_within_between'][1], 2)}" if "E_cos_within_between" in r else "—")
        L.append(f"| {k} | {_f(r['rho_a1'], 2)} [{_f(r['rho_ci'][0], 2)}, {_f(r['rho_ci'][1], 2)}] | {_f(r['rho_diff_vs_land'], 2)} [{_f(r['rho_diff_ci'][0], 2)}, {_f(r['rho_diff_ci'][1], 2)}] | "
                 f"{_f(r['regret1'])} | {_f(r['regret3'])} | {_f(r['delta_vs_land'])} [{_f(r['delta_vs_land_ci'][0])}, {_f(r['delta_vs_land_ci'][1])}] {_f(r['p_vs_land'], 3)} | "
                 f"{_f(r['vs_probe'])} [{_f(r['vs_probe_ci'][0])}, {_f(r['vs_probe_ci'][1])}] | {cr} | {cs} | {kc[0]}/{kc[1]}/{kc[2]} |")
    if o.get("cost"):
        L += ["", f"成本：E_pt fit+encode 中位 {_f(o['cost']['pt_median_equiv'], 2)} / 均值 {_f(o['cost']['pt_mean_equiv'], 2)} 全跑当量；encode 中位 {_f(o['cost']['encode_median_equiv'], 3)}"]
    rep = o["representation"]
    L += ["", f"HMD / SCP2 â₁（训练任务 a₁ 下四分位 {_f(rep['train_q25'], 2)}）：" + "；".join(f"{k} {', '.join(f'{s} {_f(v, 2)}' for s, v in d.items())}" for k, d in rep.items() if isinstance(d, dict))]
    return "\n".join(L) + "\n"


# ================================================================ 冒烟
def cmd_smoke(args) -> None:
    import types
    tasks, meta, srcs, _ = task_meta()
    small = sorted(tasks, key=lambda t: meta[t]["n_train"] * meta[t]["T"] * meta[t]["C"])[:2]
    base = dict(vars(args)); base.update({"tasks": small, "iters_cap": 5, "resume": False, "parts": 1, "part": 0})
    cmd_extract_pt(types.SimpleNamespace(**base))
    for t in small:
        with np.load(Path(args.outdir) / "E_pt" / f"{t}.npz") as z:
            assert z["windows_repr"].dtype == np.float16 and z["windows_repr"].shape[0] == min(100, meta[t]["n_train"]) and z["windows_repr"].shape[2] == REPR
            assert z["windows_repr"].shape[1] <= T_MAX
    sh = dict(base); sh.update({"folds": [srcs[0]], "max_per_ds": 64})
    cmd_fit_shared(types.SimpleNamespace(**sh))
    m = json.loads((Path(args.outdir) / "E_sh" / srcs[0] / "meta.json").read_text(encoding="utf-8"))
    assert srcs[0] not in m["fit_datasets"]
    # SetTF：合成 reprs 上 2 折 × 1 seed × 3 epoch（真实 targets 维度）
    T = _load_targets(Path(args.outdir))
    rng = np.random.default_rng(0)
    reprs = [rng.normal(size=(20, 16, REPR)).astype(np.float16) for _ in T["tasks"]]
    for fold in [str(s) for s in T["order"]][:2]:
        tr_all = T[f"{fold}/tr"]; vs = str(T[f"{fold}/val"]); g = T["groups"].astype(str)
        va = np.array([i for i in tr_all if g[i] == vs]); tr = np.array([i for i in tr_all if g[i] != vs])
        r = train_settf(reprs, T[f"{fold}/A"][:, :3], tr, va, extra=T[f"{fold}/X4"], seed=0, device=args.device, **{**SETTF, "max_epochs": 3, "patience": 3})
        assert len(r["train_loss"]) == 3 and r["predict"](np.arange(4)).shape == (4, 3)
        print(f"[smoke] settf {fold} loss {r['train_loss']}", flush=True)
    print("[smoke] OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "extract-pt", "fit-shared", "train-settf", "analyze", "smoke", "pack-means"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm", choices=("pt", "sh"), default="pt")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--combo", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters-cap", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--max-per-ds", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--max-epochs", type=int, default=None, help="仅冒烟用")
    ap.add_argument("--device", default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.prep:
        cmd_prep(args)
    if args.smoke:
        cmd_smoke(args)
    if args.extract_pt:
        cmd_extract_pt(args)
    if args.fit_shared:
        cmd_fit_shared(args)
    if args.pack_means:
        cmd_pack_means(args)
    if args.train_settf:
        cmd_train_settf(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
