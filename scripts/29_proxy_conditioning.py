"""代理条件化：逐格零训练代理读交互项（协议 results/proxy_conditioning/PROTOCOL.md）。

对每个 (任务, CLS) 把该 CLS 的视图生成管线（输入增强 → 随机初始化编码器 →
嵌入变换）施加到任务训练划分上，量类别可分性变化 fid = Δg1（无对比预训练）。
决定性检验 D1x = 扣除任务/CLS 两个主效应后的残差 Spearman（读交互 0.448 份额），
陷阱对照 = pooled 相关。第 0 步先验杀门只看代理自身交互份额，不看结果变量。

    python scripts/29_proxy_conditioning.py --audit
    python scripts/29_proxy_conditioning.py --proxy --resume
    python scripts/29_proxy_conditioning.py --step0
    python scripts/29_proxy_conditioning.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
dm = importlib.import_module("scripts.14_destruction_matrix")
gg = importlib.import_module("scripts.17_feasibility_loo_b")
br = importlib.import_module("scripts.28_behavior_recon")

from autocls.contrastive.augmentations import augment_two_views  # noqa: E402
from autocls.contrastive.emb_transform import apply_emb_transform  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.downstream.linear_eval import encode_features  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    append_jsonl, device_label, load_probe_cls, run_seed_for,
    split_task_train_eval,
)

OUTDIR = ROOT / "results" / "proxy_conditioning"
CACHE = OUTDIR / "proxy_cache.jsonl"
ITERS = 290                     # 结果变量口径（代理侧不训练）
N_CLS = 10
N_REPS = 5                      # 协议 §3：每任务 5 个编码器种子，跨 CLS 配对
EVAL_RATIO = 0.3
AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop")
PANEL_DS = br.PRIMARY_DS + (br.SUPP_DS,)
N_TASKS_PANEL, N_TASKS_OTHER = 8, 2
# 判读常数（协议 §4/§5，先于计算写死）
KILL_SHARE, GREEN_SHARE = 0.10, 0.25
ALPHA_D1X, RHO_ESTABLISH, RHO_EDGE = 0.05, 0.15, 0.05
N_PERM, PERM_SEED = 10_000, 20260831
D2X_RHO, D2X_ALPHA = 0.5, 0.05


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def two_way_residuals(M: np.ndarray) -> np.ndarray:
    """加性两因素残差 r_ij = v_ij − v_i· − v_·j + v··（协议 §5）。"""
    M = np.asarray(M, float)
    return (M - M.mean(axis=1, keepdims=True)
            - M.mean(axis=0, keepdims=True) + M.mean())


def interaction_share(M: np.ndarray) -> float:
    """两因素 SS 分解的交互份额（与 dm.ss_decomposition 同式）。"""
    M = np.asarray(M, float)
    total = float(((M - M.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    return float((two_way_residuals(M) ** 2).sum() / total)


def verdict_step0(share: float) -> str:
    """第 0 步杀门三档（协议 §4；边界含在高档一侧）。"""
    if share < KILL_SHARE:
        return "kill"
    return "amber" if share < GREEN_SHARE else "green"


def verdict_d1x(p: float, rho: float, *, amber: bool) -> str:
    """D1x 判读带（协议 §5；琥珀档不接受边缘行）。"""
    if p >= ALPHA_D1X:
        return "not_established"
    if rho >= RHO_ESTABLISH:
        return "established"
    if rho >= RHO_EDGE and not amber:
        return "edge"
    return "not_established"


def permute_within_tasks(R: np.ndarray, rng) -> np.ndarray:
    """任务内置换（保任务块结构、破坏交互配对；协议 §5 零分布）。"""
    P = np.array(R, float, copy=True)
    for i in range(P.shape[0]):
        P[i] = P[i][rng.permutation(P.shape[1])]
    return P


def residual_spearman_perm(X: np.ndarray, Y: np.ndarray, *, n_perm: int,
                           seed: int) -> tuple[float, float]:
    """D1x：两向残差 Spearman + 任务内置换零分布（单侧 H1: ρ>0）。"""
    from scipy.stats import spearmanr
    rx, ry = two_way_residuals(X), two_way_residuals(Y)
    rho = float(spearmanr(rx.ravel(), ry.ravel()).statistic)
    rng = np.random.default_rng([seed, 20260901])
    flat_x = rx.ravel()
    n_ge = sum(1 for _ in range(n_perm)
               if spearmanr(flat_x,
                            permute_within_tasks(ry, rng).ravel()).statistic >= rho)
    return rho, (1 + n_ge) / (n_perm + 1)


def aug_params(cls_cfg) -> dict:
    """6 个输入增强字段（cl_trainer._AUG_FIELDS 逐字同源）。"""
    return {n: getattr(cls_cfg, n) for n in AUG_FIELDS}


def geom(V: np.ndarray, y: np.ndarray) -> np.ndarray:
    """类几何 5 维（scripts/17 逐字复用，单位球面口径）。"""
    return gg.class_geometry(gg.l2_normalize_rows(V), y)


def _pool_encode(enc, x: torch.Tensor, cls_cfg, gen, device) -> np.ndarray:
    """增强视图 → 编码 → 嵌入变换 → 时间维 max-pool（对比目标实际看到的表征）。"""
    enc.eval()
    with torch.no_grad():
        z = enc(x.to(device))
        z = apply_emb_transform(z, cls_cfg, gen)
        return z.max(dim=1).values.cpu().numpy()


def proxy_cell(Xtr: np.ndarray, ytr: np.ndarray, cls_cfg, *, tid: str,
               cls_key: str, n_reps: int = N_REPS, device: str = "cpu") -> dict:
    """单格代理（协议 §3）：R 个随机初始化编码器上的 Δ 几何均值。"""
    params, order = aug_params(cls_cfg), cls_cfg.order
    xt = torch.from_numpy(np.asarray(Xtr, dtype=np.float32))
    fids, varis, dg2, dg3, dg4, g1c, g1a, dts = [], [], [], [], [], [], [], []
    for s in range(n_reps):
        torch.manual_seed(run_seed_for(tid, f"proxy_encoder:{s}"))
        enc = build_encoder(input_dims=xt.shape[-1]).to(device)
        gen = torch.Generator()
        gen.manual_seed(run_seed_for(f"proxy:{tid}", f"{cls_key}:{s}"))
        Z0 = encode_features(enc, Xtr, device=device)
        v1, v2, _s1, _s2 = augment_two_views(xt, params, order, gen)
        Z1 = _pool_encode(enc, v1, cls_cfg, gen, device)
        Z2 = _pool_encode(enc, v2, cls_cfg, gen, device)
        G0, G1 = geom(Z0, ytr), geom(Z1, ytr)
        n1 = Z1 / np.maximum(np.linalg.norm(Z1, axis=1, keepdims=True), 1e-12)
        n2 = Z2 / np.maximum(np.linalg.norm(Z2, axis=1, keepdims=True), 1e-12)
        g1c.append(G0[0]); g1a.append(G1[0])
        fids.append(G1[0] - G0[0])
        dg2.append(G1[1] - G0[1]); dg3.append(G1[2] - G0[2]); dg4.append(G1[3] - G0[3])
        varis.append(1.0 - float((n1 * n2).sum(axis=1).mean()))
        dts.append((v1.shape[1] - xt.shape[1]) / xt.shape[1])
    return {"fid": float(np.mean(fids)), "var": float(np.mean(varis)),
            "d_g2": float(np.mean(dg2)), "d_g3": float(np.mean(dg3)),
            "d_g4": float(np.mean(dg4)), "g1_clean": float(np.mean(g1c)),
            "g1_aug": float(np.mean(g1a)), "dT": float(np.mean(dts))}


# ---------------------------------------------------------------- 装配
def resolve_registry(ds: str, regs: dict):
    """注册表分派（协议 §3 钉源）：crop 钉源数据集必须走 crop256 注册表，
    与其 R' 来源（candidate_survey/baseline_crop256）逐字一致；其余源池优先、
    候选池兜底。三表 slice_seed/n_samples 相同，切片定义可比。"""
    if ds in br.PIN_CROP:
        return regs["crop256"], "crop256"
    for name in ("source", "cand"):
        if ds in regs[name].names():
            return regs[name], name
    raise KeyError(f"{ds} 不在任何注册表")


def load_registries() -> dict:
    return {"source": load_registry(),
            "cand": load_registry(ROOT / "configs" / "candidate_pool.yaml"),
            "crop256": load_registry(ROOT / "configs"
                                     / "candidate_pool_crop256.yaml")}


def target_tasks(asm: dict) -> dict:
    """{dataset: [task_id...]}：面板 5 个各 8 任务，其余各 2（协议 §3）。"""
    Q, _src = br.build_rows(asm)
    out = {}
    for ds in sorted(Q):
        n = N_TASKS_PANEL if ds in PANEL_DS else N_TASKS_OTHER
        tids = [f"{ds}_{i:02d}" for i in range(n)]
        if all(t in asm["bands"] for t in tids):
            out[ds] = tids
    return out


def outcome_matrix(asm: dict, tasks: dict, keys: list) -> tuple:
    """结果变量 y = 既有 k=1 canonical R' @290；返回 (行名, 矩阵, 缺格)。"""
    names, rows, missing = [], [], []
    for ds in sorted(tasks):
        for tid in tasks[ds]:
            vals = [asm["k1"].get((tid, k)) for k in keys]
            if any(v is None for v in vals):
                missing.append(tid)
                continue
            names.append(tid)
            rows.append(vals)
    return names, np.array(rows, float), missing


def load_proxy_cache() -> dict:
    out = {}
    for line in (CACHE.read_text(encoding="utf-8").splitlines()
                 if CACHE.exists() else []):
        if line.strip():
            r = json.loads(line)
            out[(r["task_id"], r["cls_key"])] = r
    return out


# ---------------------------------------------------------------- 子命令
def cmd_audit(asm: dict, keys: list) -> None:
    tasks = target_tasks(asm)
    names, Y, missing = outcome_matrix(asm, tasks, keys)
    done = load_proxy_cache()
    audit = {"protocol": "results/proxy_conditioning/PROTOCOL.md", "iters": ITERS,
             "n_datasets": len(tasks), "n_tasks": len(names),
             "n_cells": len(names) * len(keys),
             "tasks_per_dataset": {d: len(t) for d, t in sorted(tasks.items())},
             "missing_outcome_tasks": missing,
             "proxy_cells_done": len(done),
             "expected_tasks": 80, "expected_cells": 800}
    (OUTDIR / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"数据集 {len(tasks)} | 任务 {len(names)}（预期 80）| 格 "
          f"{len(names) * len(keys)}（预期 800）| 代理已算 {len(done)}")
    print(f"缺结果任务: {missing or '无'}")
    if len(names) != 80:
        print("⚠️ 任务数偏离预期，先查装配")
    print(f"→ {OUTDIR / 'audit.json'}", flush=True)


def cmd_proxy(asm: dict, keys: list, args) -> None:
    tasks = target_tasks(asm)
    names, _Y, _m = outcome_matrix(asm, tasks, keys)
    wanted = set(names)
    done = load_proxy_cache() if args.resume else {}
    cls_by_key = {k: c for c, k in load_probe_cls(fl.PROBE_CATALOG, N_CLS)[0]}
    regs = load_registries()
    todo = sum(1 for ds in tasks for t in tasks[ds] if t in wanted
               for k in keys if (t, k) not in done)
    print(f"代理待算 {todo} 格（已有 {len(done)}）| device={args.device}", flush=True)
    t0, n_done = time.time(), 0
    for ds in sorted(tasks):
        tids = [t for t in tasks[ds] if t in wanted]
        if all((t, k) in done for t in tids for k in keys):
            continue
        registry, reg_name = resolve_registry(ds, regs)
        entry = registry.get(ds)
        X, y = load_source_dataset(ds, registry)
        _e, sliced = slice_source_tasks(entry, registry, y)
        by_id = {t.task_id: t for t in sliced}
        print(f"[{ds}] 注册表={reg_name} X={X.shape} 任务 {len(tids)}", flush=True)
        for tid in tids:
            task = by_id[tid]
            expect = list(asm["classes"].get(tid, []))
            if list(task.classes) != expect:
                sys.exit(f"[{tid}] classes 不一致：切片 {list(task.classes)} vs "
                         f"缓存 {expect} —— 注册表/切片口径错，中止（协议 §3）")
            Xt, yt = materialize_task(task, X, y)
            tr, _ev = split_task_train_eval(
                yt, task.task_type, eval_ratio=EVAL_RATIO,
                seed=run_seed_for(tid, "train_eval_split"))
            Xtr, ytr = Xt[tr], yt[tr]
            for k in keys:
                if (tid, k) in done:
                    continue
                m = proxy_cell(Xtr, ytr, cls_by_key[k], tid=tid, cls_key=k,
                               n_reps=N_REPS, device=args.device)
                append_jsonl(CACHE, {"dataset": ds, "task_id": tid, "cls_key": k,
                                     "n_train": int(len(tr)), "n_reps": N_REPS,
                                     "iters_outcome": ITERS,
                                     "device_name": device_label(args.device), **m})
                n_done += 1
                if n_done % 50 == 0:
                    el = time.time() - t0
                    print(f"[{n_done}/{todo}] {tid} {k[:8]} fid={m['fid']:+.4f} "
                          f"ETA {el / n_done * (todo - n_done) / 60:.0f}min", flush=True)
    print(f"代理收官：{n_done} 新格，墙钟 {(time.time() - t0) / 60:.1f} min", flush=True)


def _matrices(asm: dict, keys: list) -> tuple:
    tasks = target_tasks(asm)
    names, Y, _m = outcome_matrix(asm, tasks, keys)
    cache = load_proxy_cache()
    ok = [t for t in names if all((t, k) in cache for k in keys)]
    idx = [names.index(t) for t in ok]
    P = {f: np.array([[cache[(t, k)][f] for k in keys] for t in ok], float)
         for f in ("fid", "var", "d_g2", "d_g3", "d_g4", "dT")}
    return ok, Y[idx], P, cache


def cmd_step0(asm: dict, keys: list) -> None:
    ok, _Y, P, _c = _matrices(asm, keys)
    share = interaction_share(P["fid"])
    verdict = verdict_step0(share)
    dec = {f: {"inter": interaction_share(M),
               "task": float(((M.mean(axis=1, keepdims=True) - M.mean()) ** 2).sum()
                             * M.shape[1] / max(((M - M.mean()) ** 2).sum(), 1e-12)),
               "cls": float(((M.mean(axis=0, keepdims=True) - M.mean()) ** 2).sum()
                            * M.shape[0] / max(((M - M.mean()) ** 2).sum(), 1e-12))}
           for f, M in P.items()}
    out = {"protocol": "results/proxy_conditioning/PROTOCOL.md",
           "n_tasks": len(ok), "n_cells": len(ok) * len(keys),
           "fid_interaction_share": share, "decomposition": dec,
           "bands": {"kill": KILL_SHARE, "green": GREEN_SHARE},
           "verdict": verdict}
    (OUTDIR / "step0_share.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n===== 第 0 步：代理自身交互份额（{len(ok)} 任务 × {len(keys)} CLS）=====")
    for f, d in dec.items():
        print(f"{f:<6} 任务 {d['task']:.3f} | CLS {d['cls']:.3f} | 交互 {d['inter']:.3f}")
    print(f"[杀门判读] fid 交互份额 {share:.3f} → {verdict}"
          + ("（停，不跑第 1 步）" if verdict == "kill" else ""))
    print(f"→ {OUTDIR / 'step0_share.json'}", flush=True)


def cmd_analyze(asm: dict, keys: list) -> None:
    from scipy.stats import spearmanr
    s0p = OUTDIR / "step0_share.json"
    if not s0p.exists():
        sys.exit("先跑 --step0（杀门先行，协议 §4）")
    s0 = json.loads(s0p.read_text(encoding="utf-8"))
    if s0["verdict"] == "kill":
        sys.exit("第 0 步杀档：不跑 D1x（协议 §4）")
    amber = s0["verdict"] == "amber"
    ok, Y, P, cache = _matrices(asm, keys)
    X = P["fid"]

    rho, p = residual_spearman_perm(X, Y, n_perm=N_PERM, seed=PERM_SEED)
    verdict = verdict_d1x(p, rho, amber=amber)
    pooled = float(spearmanr(X.ravel(), Y.ravel()).statistic)
    half = float(spearmanr((X - X.mean(axis=0, keepdims=True)).ravel(),
                           (Y - Y.mean(axis=0, keepdims=True)).ravel()).statistic)

    # S-std：任务内标准化残差
    rx, ry = two_way_residuals(X), two_way_residuals(Y)
    sx = rx / np.maximum(rx.std(axis=1, keepdims=True), 1e-12)
    sy = ry / np.maximum(ry.std(axis=1, keepdims=True), 1e-12)
    s_std = float(spearmanr(sx.ravel(), sy.ravel()).statistic)

    sens = {}
    for f in ("var", "d_g2", "d_g3", "d_g4"):
        r_f, p_f = residual_spearman_perm(P[f], Y, n_perm=2000, seed=PERM_SEED)
        sens[f] = {"rho": r_f, "p": p_f}

    # 长度归因诊断（协议 §6）
    r_dT = two_way_residuals(P["dT"])
    len_rho = float(spearmanr(r_dT.ravel(), rx.ravel()).statistic)
    beta = float(np.polyfit(r_dT.ravel(), rx.ravel(), 1)[0]) \
        if r_dT.std() > 1e-12 else 0.0
    partial = float(spearmanr((rx - beta * r_dT).ravel(), ry.ravel()).statistic)

    # D2x 数据集级 + D3x 格级判别
    ds_of = {t: fl.dataset_of(t) for t in ok}
    per_ds_fid, per_ds_destroy = defaultdict(list), defaultdict(list)
    for i, t in enumerate(ok):
        mean, band = asm["bands"][t]
        for j, k in enumerate(keys):
            per_ds_fid[ds_of[t]].append(X[i, j])
            per_ds_destroy[ds_of[t]].append(
                int(fl.tri_state(Y[i, j], mean, band) == "destroy"))
    dss = sorted(per_ds_fid)
    d2_rho = float(spearmanr([np.mean(per_ds_fid[d]) for d in dss],
                             [np.mean(per_ds_destroy[d]) for d in dss]).statistic)
    d2_p = float(spearmanr([np.mean(per_ds_fid[d]) for d in dss],
                           [np.mean(per_ds_destroy[d]) for d in dss]).pvalue)
    flat_destroy, flat_fid = [], []
    for i, t in enumerate(ok):
        mean, band = asm["bands"][t]
        for j, k in enumerate(keys):
            flat_destroy.append(int(fl.tri_state(Y[i, j], mean, band) == "destroy"))
            flat_fid.append(X[i, j])
    flat_destroy, flat_fid = np.array(flat_destroy), np.array(flat_fid)
    pos, neg = flat_fid[flat_destroy == 1], flat_fid[flat_destroy == 0]
    auc = float((np.less.outer(pos, neg).mean()
                 + 0.5 * np.equal.outer(pos, neg).mean())) if len(pos) and len(neg) \
        else float("nan")

    out = {"protocol": "results/proxy_conditioning/PROTOCOL.md",
           "n_tasks": len(ok), "n_cells": len(ok) * len(keys),
           "step0": {"share": s0["fid_interaction_share"], "verdict": s0["verdict"]},
           "D1x": {"rho_residual": rho, "p_perm": p, "n_perm": N_PERM,
                   "verdict": verdict},
           "trap_controls": {"pooled_rho": pooled, "cls_centered_rho": half},
           "sensitivity": {"S_std_rho": s_std, **sens},
           "length_diagnostic": {"rho_dT_vs_fid_resid": len_rho,
                                 "partial_rho_after_dT": partial},
           "D2x": {"rho": d2_rho, "p": d2_p, "n_datasets": len(dss),
                   "pass": bool(d2_rho >= D2X_RHO and d2_p < D2X_ALPHA)},
           "D3x": {"auc_destroy": auc, "n_destroy": int(flat_destroy.sum()),
                   "n_total": int(len(flat_destroy))}}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print(f"\n===== 代理条件化 analyze（{len(ok)} 任务 × {len(keys)} CLS）=====")
    print(f"D1x 交互残差 ρ={rho:+.4f} 置换 p={p:.4f} → {verdict}"
          + ("（琥珀档：不接受边缘）" if amber else ""))
    print(f"陷阱对照: pooled ρ={pooled:+.4f} | 仅扣 CLS 主效应 ρ={half:+.4f}")
    print(f"敏感性: S-std ρ={s_std:+.4f} | " +
          " ".join(f"{f} ρ={v['rho']:+.3f}(p={v['p']:.3f})" for f, v in sens.items()))
    print(f"长度诊断: ρ(ΔT残差, fid残差)={len_rho:+.4f} | 偏出后 ρ={partial:+.4f}")
    print(f"D2x 数据集级: ρ={d2_rho:+.4f} p={d2_p:.4f} (n={len(dss)}) → "
          f"{'过' if out['D2x']['pass'] else '不过'}")
    print(f"D3x 格级 AUC(destroy)={auc:.4f}（{int(flat_destroy.sum())}/"
          f"{len(flat_destroy)} 破坏格）")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not any((args.audit, args.proxy, args.step0, args.analyze)):
        ap.error("至少指定一个子命令")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    asm = br.assemble()
    keys = br.load_shared_keys()
    if args.audit:
        cmd_audit(asm, keys)
    if args.proxy:
        cmd_proxy(asm, keys, args)
    if args.step0:
        cmd_step0(asm, keys)
    if args.analyze:
        cmd_analyze(asm, keys)


if __name__ == "__main__":
    main()
