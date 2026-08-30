"""MX-1b 探针条件化可行性预测：16 折 LOO（协议 results/mx1b_feasibility/PROTOCOL.md）。

与 MX-1（scripts/16）的唯一差异 = 特征：P 组（iters=0 探针统计，3 维）⊕
G 组（折 TS2Vec 表征空间类别几何，5 维，标签感知零训练）；E 降格为增量臂。
折结构 / label / 折 TS2Vec / MLP 超参 / 判据机器全部逐字复用 scripts/16
（协议 §4：禁止为 MX-1b 另调任何超参）。

    python scripts/17_feasibility_loo_b.py --features --resume   # 轻 GPU：复用折 TS2Vec 重编码
    python scripts/17_feasibility_loo_b.py --train --resume      # CPU
    python scripts/17_feasibility_loo_b.py --analyze             # 判读带 §5（含勿复活分支）
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import encode_task_windows, task_windows  # noqa: E402

OUTDIR = ROOT / "results" / "mx1b_feasibility"
BFOLDS = OUTDIR / "folds"
ANALYSIS_PATH = OUTDIR / "analysis.json"
LABELS = lo.LABELS
ARM_KEYS = ("bm", "be", "bp", "bg", "bm_a1")
FEAT_ARMS = ("bm", "be", "bp", "bg")     # 需独立训练的臂（bm_a1 复用 bm 模型）
SCP_PAIR = ("SelfRegulationSCP1", "SelfRegulationSCP2")   # 协议 §6 R3b 展品


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def sample_indices(n: int, sample_num: int, seed: int) -> np.ndarray:
    """镜像 encode_task_windows 的采样（task_embed/__init__.py L121-123 逐字同式），
    G 组窗口标签必须与其选中的窗口逐位同源（tests 用恒等替身断言）。"""
    rng = np.random.default_rng(seed)
    m = min(sample_num, n)
    return np.sort(rng.choice(n, size=m, replace=False))


def l2_normalize_rows(V: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化（协议 §3：几何在单位球面上算；零行护栏）。"""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norms, 1e-12)


def class_geometry(V: np.ndarray, y: np.ndarray) -> np.ndarray:
    """G 组 5 维（协议 §3 公式写死）。<2 个在场类别 → 全零向量（审计计数）。

    g1 NCC 准确率 | g2 log1p(Fisher 比) | g3 质心 silhouette |
    g4 min 类间质心距 / mean 类内距 | g5 = g1 − 1/C
    """
    classes = np.unique(y)
    if len(classes) < 2:
        return np.zeros(5)
    cents = np.stack([V[y == c].mean(axis=0) for c in classes])
    grand = V.mean(axis=0)
    d2cents = np.linalg.norm(V[:, None, :] - cents[None, :, :], axis=2)  # (n, C)
    own = np.array([np.where(classes == c)[0][0] for c in y])
    g1 = float((d2cents.argmin(axis=1) == own).mean())
    n_k = np.array([(y == c).sum() for c in classes], dtype=float)
    s_w = float(sum(((V[y == c] - cents[i]) ** 2).sum()
                    for i, c in enumerate(classes)))
    s_b = float((n_k[:, None] * (cents - grand) ** 2).sum())
    g2 = float(np.log1p(s_b / max(s_w, 1e-12)))
    a = d2cents[np.arange(len(y)), own]
    d_other = d2cents.copy()
    d_other[np.arange(len(y)), own] = np.inf
    b = d_other.min(axis=1)
    g3 = float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12)))
    inter = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
    inter[np.eye(len(classes), dtype=bool)] = np.inf
    g4 = float(inter.min() / max(a.mean(), 1e-12))
    g5 = g1 - 1.0 / len(classes)
    return np.array([g1, g2, g3, g4, g5])


def probe_vec(row: dict) -> np.ndarray:
    """P 组 3 维：base_mean / chance=1/C / band（labels.jsonl 既有字段）。"""
    return np.array([row["base_mean"], 1.0 / len(row["classes"]), row["band"]])


def assemble_features(P: np.ndarray, G: np.ndarray, E: np.ndarray) -> dict:
    """臂 → 任务特征向量（协议 §4 表）。"""
    return {"bm": np.concatenate([P, G]), "be": np.concatenate([P, G, E]),
            "bp": np.asarray(P), "bg": np.asarray(G)}


def verdict_mx1b(p_d1b: float, p_d2b: float) -> str:
    """判读带（协议 §5；¬D1b 且不边缘 → 勿复活条款生效）。"""
    if p_d1b < 0.05:
        return "established" if p_d2b < 0.05 else "feature_signal_unproven"
    if p_d1b < 0.15:
        return "edge"
    return "not_established_norevive"


# ---------------------------------------------------------------- 阶段 1：features
def cmd_features(device: str) -> None:
    ra = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                        .read_text(encoding="utf-8"))["route_a"]
    rows = lo.load_panel()
    datasets, tids_by_ds, classes_by_tid = lo.panel_index(rows)
    base_by_tid = {}
    for r in rows:                      # base_mean/band 同任务各 cls 行相同，取首见
        base_by_tid.setdefault(r["task_id"], r)
    data = lo.load_datasets_and_tasks(datasets, tids_by_ds, classes_by_tid)
    n_degenerate = 0

    for heldout, train_ds in lo.fold_split(datasets):
        out_dir = BFOLDS / heldout / "probe_features"
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [t for name in datasets for t in data[name][2]
                if not (out_dir / f"{t.task_id}.npz").exists()]
        if not todo:
            print(f"[{heldout}] 特征已齐，跳过", flush=True)
            continue
        # 复用 MX-1 的折 TS2Vec（已存在 → 只加载 + 指纹校验，绝不重训）
        shared = lo.get_fold_ts2vec(lo.FOLDS_DIR / heldout, heldout,
                                    train_ds, data, ra, device)
        t0 = time.time()
        for name in datasets:
            X, y, tasks = data[name]
            for t in tasks:
                path = out_dir / f"{t.task_id}.npz"
                if path.exists():
                    continue
                Xt, yt = materialize_task(t, X, y)
                windows = task_windows(Xt, t.task_type)
                reprs = encode_task_windows(windows, shared,
                                            sample_num=ra["sample_num"],
                                            seed=ra["embed_seed"])
                idx = sample_indices(len(windows), ra["sample_num"],
                                     ra["embed_seed"])
                ysel = np.asarray(yt)[idx]
                V = l2_normalize_rows(reprs.mean(axis=1))
                G = class_geometry(V, ysel)
                if not G.any():
                    n_degenerate += 1
                np.savez_compressed(
                    path, G=G.astype(np.float32),
                    P=probe_vec(base_by_tid[t.task_id]).astype(np.float32),
                    V=V.astype(np.float32), y=ysel,
                    n_classes_present=len(np.unique(ysel)))
        print(f"[{heldout}] 特征 {len(list(out_dir.glob('*.npz')))}/"
              f"{len(datasets) * 2} 齐 ({time.time() - t0:.0f}s)", flush=True)
        del shared
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    n_ok = sum(1 for d, _ in lo.fold_split(datasets)
               if len(list((BFOLDS / d / "probe_features").glob("*.npz")))
               == len(datasets) * 2)
    print(f"\nfeatures 收官：{n_ok}/{len(datasets)} 折齐全 | "
          f"退化任务（<2 在场类别→G 全零）: {n_degenerate}", flush=True)


# ---------------------------------------------------------------- 阶段 2：train
def _load_task_feats(heldout: str, tids: list[str]) -> dict:
    """{tid: {arm: 特征向量}}——P/G 取本折 probe_features，E 取 MX-1 折特征。"""
    out = {}
    for tid in tids:
        pb = BFOLDS / heldout / "probe_features" / f"{tid}.npz"
        eb = lo.FOLDS_DIR / heldout / "task_features" / f"{tid}.npz"
        assert pb.exists(), f"缺 {pb}：先跑 --features"
        assert eb.exists(), f"缺 {eb}：MX-1 折特征应已存在"
        with np.load(pb) as z:
            P, G = z["P"], z["G"]
        with np.load(eb) as z:
            E = z["E"]
        out[tid] = assemble_features(P, G, E)
    return out


def cmd_train(resume: bool) -> None:
    rows = lo.load_panel()
    datasets, tids_by_ds, _cls = lo.panel_index(rows)
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space)
    vcls = {key: enc.encode(c)
            for c, key in load_probe_cls(lo._b2.PROBE_CATALOG, 10)[0]}

    for heldout, _train_ds in lo.fold_split(datasets):
        pred_path = BFOLDS / heldout / "predictions.jsonl"
        test_cells = [r for r in rows if r["dataset"] == heldout]
        if len(test_cells) < lo.MIN_FOLD_CELLS:
            print(f"[{heldout}] cell 不足，折剔除", flush=True)
            continue
        expected = len(ARM_KEYS) * len(test_cells)
        if resume and pred_path.exists() and len(lo.read_jsonl(pred_path)) == expected:
            print(f"[{heldout}] predictions 齐全，跳过", flush=True)
            continue
        all_tids = sorted(t for ds in datasets for t in tids_by_ds[ds])
        feats = _load_task_feats(heldout, all_tids)
        train_cells = [r for r in rows if r["dataset"] != heldout]
        train_tids = sorted({c["task_id"] for c in train_cells})
        test_tids = sorted({c["task_id"] for c in test_cells})
        ytr = np.array([LABELS.index(c["label"]) for c in train_cells])
        w = lo.class_weights([c["label"] for c in train_cells])
        t0 = time.time()
        arms_out: dict = {}

        def build_X(cells, z):
            return np.stack([np.concatenate([z[c["task_id"]], vcls[c["cls_key"]]])
                             for c in cells])

        for arm in FEAT_ARMS:
            Etr = np.stack([feats[t][arm] for t in train_tids])
            mu, sd = lo.standardize_fit(Etr)    # 只喂训练折（泄漏契约同 MX-1）
            z = {t: lo.standardize_apply(feats[t][arm], mu, sd) for t in all_tids}
            models = [lo.train_mlp(build_X(train_cells, z), ytr, w,
                                   seed=run_seed_for(f"mx1b:{heldout}:{arm}", f"m{s}"))
                      for s in range(lo.N_MODEL_SEEDS)]

            def avg_pred(X):
                return np.mean([lo.predict_proba(m, X) for m in models], axis=0)

            arms_out[arm] = {(c["task_id"], c["cls_key"]): p
                             for c, p in zip(test_cells, avg_pred(build_X(test_cells, z)))}
            if arm == "bm":     # D2b 消融：特征向量 cross-assignment（4b 同语义）
                acc = np.zeros((len(test_cells), len(LABELS)))
                for a_s in range(lo.N_ASSIGN_SEEDS):
                    mapping = lo.cross_assignment_mx1(
                        test_tids, train_tids, fl.dataset_of,
                        seed=run_seed_for(f"mx1b:{heldout}:bm", f"assign{a_s}"))
                    za = dict(z)
                    za.update({t: z[mapping[t]] for t in test_tids})
                    acc += avg_pred(build_X(test_cells, za))
                arms_out["bm_a1"] = {(c["task_id"], c["cls_key"]): p
                                     for c, p in zip(test_cells,
                                                     acc / lo.N_ASSIGN_SEEDS)}

        label_of = {(c["task_id"], c["cls_key"]): c["label"] for c in test_cells}
        tmp = pred_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for arm in ARM_KEYS:
                for (tid, key), p in sorted(arms_out[arm].items()):
                    fh.write(json.dumps({
                        "fold": heldout, "arm": arm, "task_id": tid,
                        "cls_key": key, "p": [round(float(x), 6) for x in p],
                        "label": label_of[(tid, key)]}, ensure_ascii=False) + "\n")
        tmp.replace(pred_path)
        print(f"[{heldout}] {len(ARM_KEYS)} 臂 × {len(test_cells)} cell "
              f"({time.time() - t0:.0f}s) → {pred_path}", flush=True)


# ---------------------------------------------------------------- 阶段 3：analyze
def cmd_analyze() -> None:
    rows = lo.load_panel()
    datasets, tids_by_ds, _cls = lo.panel_index(rows)
    global_destroy = lo.observed_rates(rows)[LABELS.index("destroy")]
    di = LABELS.index("destroy")

    per_fold: dict = {}
    for heldout, _train in lo.fold_split(datasets):
        pred_path = BFOLDS / heldout / "predictions.jsonl"
        test_cells = [r for r in rows if r["dataset"] == heldout]
        if len(test_cells) < lo.MIN_FOLD_CELLS or not pred_path.exists():
            per_fold[heldout] = {"excluded": True}
            continue
        by_arm: dict = defaultdict(dict)
        for r in lo.read_jsonl(pred_path):
            by_arm[r["arm"]][(r["task_id"], r["cls_key"])] = np.array(r["p"])
        train_cells = [r for r in rows if r["dataset"] != heldout]
        p_obs = lo.observed_rates(test_cells)
        b0 = lo.b0_rates(train_cells)
        b1 = lo.b1_rates(train_cells)
        keys = [(c["task_id"], c["cls_key"]) for c in test_cells]
        entry = {
            "excluded": False, "n_test_cells": len(test_cells),
            "obs_rates": [round(float(x), 4) for x in p_obs],
            "floor": lo.noise_floor(p_obs, n_cells=len(test_cells),
                                    n_sim=lo.FLOOR_N_SIM,
                                    seed=run_seed_for(heldout, "mx1_floor")),
            "err_b0": lo.err_tv(b0, p_obs), "err": {}, "pred_rates": {},
        }
        for arm in ARM_KEYS:
            p_ds = np.mean([by_arm[arm][k] for k in keys], axis=0)
            entry["err"][arm] = lo.err_tv(p_ds, p_obs)
            entry["pred_rates"][arm] = [round(float(x), 4) for x in p_ds]
        entry["brier_bm"] = float(np.mean(
            [lo.brier(by_arm["bm"][k], c["label"])
             for k, c in zip(keys, test_cells)]))
        entry["brier_b1"] = float(np.mean(
            [lo.brier(b1[c["cls_key"]], c["label"]) for c in test_cells]))
        per_fold[heldout] = entry

    active = [d for d in sorted(per_fold) if not per_fold[d]["excluded"]]
    arr = lambda key: np.array([per_fold[d][key] for d in active])  # noqa: E731
    earr = lambda arm: np.array([per_fold[d]["err"][arm] for d in active])  # noqa: E731

    tests = {
        "n_active_folds": len(active),
        "D1b_p": lo.paired_wilcoxon_less(earr("bm"), arr("err_b0")),
        "D2b_p": lo.paired_wilcoxon_less(earr("bm"), earr("bm_a1")),
        "D3b_p": lo.paired_wilcoxon_less(earr("be"), earr("bm")),
        "S1b_p": lo.paired_wilcoxon_less(arr("brier_bm"), arr("brier_b1")),
        "R2b_bp_p": lo.paired_wilcoxon_less(earr("bp"), arr("err_b0")),
        "R2b_bg_p": lo.paired_wilcoxon_less(earr("bg"), arr("err_b0")),
        "D1b_win_folds": int((earr("bm") < arr("err_b0")).sum()),
        "err_median": {arm: float(np.median(earr(arm))) for arm in ARM_KEYS},
        "err_b0_median": float(np.median(arr("err_b0"))),
        "floor_median": float(np.median(arr("floor"))),
    }
    k1_ds = {r["dataset"] for r in rows if r["r_source"] == "k1"}
    k3_active = [d for d in active if d not in k1_ds]
    idx = [active.index(d) for d in k3_active]
    tests["R1b_folds"] = k3_active
    tests["R1b_D1b_p"] = lo.paired_wilcoxon_less(earr("bm")[idx],
                                                 arr("err_b0")[idx])
    pred_destroy = [per_fold[d]["pred_rates"]["bm"][di] for d in active]
    obs_destroy = [per_fold[d]["obs_rates"][di] for d in active]
    tests["P6b_spearman"] = dm.spearman(pred_destroy, obs_destroy)
    # R3b：SCP1/SCP2 展品——预测破坏率之差与观测同向（协议 §6）
    if all(d in active for d in SCP_PAIR):
        d_pred = (per_fold[SCP_PAIR[1]]["pred_rates"]["bm"][di]
                  - per_fold[SCP_PAIR[0]]["pred_rates"]["bm"][di])
        d_obs = (per_fold[SCP_PAIR[1]]["obs_rates"][di]
                 - per_fold[SCP_PAIR[0]]["obs_rates"][di])
        tests["R3b_scp_pair"] = {"pred_diff": round(d_pred, 4),
                                 "obs_diff": round(d_obs, 4),
                                 "same_sign": bool(d_pred * d_obs > 0)}
    delta = earr("bm_a1") - earr("bm")
    extremeness = np.abs(np.array(obs_destroy) - global_destroy)
    tests["D2b_heterogeneity_spearman"] = dm.spearman(list(delta),
                                                      list(extremeness))
    # 诊断：g1..g5（各数据集取自己折的特征）与破坏率单变量 Spearman
    g_diag = {}
    g_mat = {d: np.mean([np.load(BFOLDS / d / "probe_features" / f"{t}.npz")["G"]
                         for t in tids_by_ds[d]], axis=0) for d in active}
    for i, gname in enumerate(["g1_ncc", "g2_fisher", "g3_sil", "g4_margin",
                               "g5_ncc_lift"]):
        g_diag[gname] = dm.spearman([float(g_mat[d][i]) for d in active],
                                    obs_destroy)
    tests["g_univariate_spearman"] = g_diag

    verdict = verdict_mx1b(tests["D1b_p"], tests["D2b_p"])
    out = {"protocol": "results/mx1b_feasibility/PROTOCOL.md",
           "per_fold": per_fold, "tests": tests, "verdict": verdict}
    ANALYSIS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                        default=float), encoding="utf-8")

    print(f"\n===== MX-1b LOO（{len(active)} 折）=====")
    print(f"{'留出数据集':<28}{'观测(提/惰/破)':>16}{'地板':>7}{'B0':>7}"
          f"{'bm':>7}{'a1':>7}{'be':>7}{'bp':>7}{'bg':>7}")
    for d in active:
        e = per_fold[d]
        obs = "/".join(f"{v:.2f}" for v in e["obs_rates"])
        print(f"{d:<28}{obs:>16}{e['floor']:>7.3f}{e['err_b0']:>7.3f}"
              f"{e['err']['bm']:>7.3f}{e['err']['bm_a1']:>7.3f}"
              f"{e['err']['be']:>7.3f}{e['err']['bp']:>7.3f}{e['err']['bg']:>7.3f}")
    t = tests
    print(f"\nD1b(bm<B0) p={t['D1b_p']:.4f}（{t['D1b_win_folds']}/{len(active)} 折胜）"
          f" | D2b(bm<a1) p={t['D2b_p']:.4f} | D3b(be<bm,E增量) p={t['D3b_p']:.4f}"
          f" | S1b p={t['S1b_p']:.4f}")
    print(f"R1b(剔k1) D1b p={t['R1b_D1b_p']:.4f} | R2b: bp p={t['R2b_bp_p']:.4f} "
          f"bg p={t['R2b_bg_p']:.4f}")
    print(f"P6b Spearman(bm 预测破坏率,观测)={t['P6b_spearman'][0]:.3f} "
          f"(p={t['P6b_spearman'][1]:.3f})")
    if "R3b_scp_pair" in t:
        r3 = t["R3b_scp_pair"]
        print(f"R3b SCP 展品: 预测差 {r3['pred_diff']:+.3f} vs 观测差 "
              f"{r3['obs_diff']:+.3f} → 同向={r3['same_sign']}")
    print("g 单变量 Spearman(↔破坏率): "
          + "  ".join(f"{k}={v[0]:+.2f}" for k, v in g_diag.items()))
    print(f"\n[判读] → {verdict}")
    print(f"→ {ANALYSIS_PATH}", flush=True)


# ---------------------------------------------------------------- 入口
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.features:
        cmd_features(args.device)
    if args.train:
        cmd_train(args.resume)
    if args.analyze:
        cmd_analyze()
    if not (args.features or args.train or args.analyze):
        ap.error("至少指定 --features / --train / --analyze 之一")


if __name__ == "__main__":
    main()
