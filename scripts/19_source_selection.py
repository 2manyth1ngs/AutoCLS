"""MX-2a E 相似度选源：task-feature 作比较器迁移课程（协议 results/mx2a_source_selection/PROTOCOL.md）。

零 GPU（CPU 分钟级）。@290 面板 320 格 → per-task δ 配对 → PairNet
（cls 向量对，E 不进输入——M3/C2 已判死输入用法）；处理变量 = 训练配对的
来源构成（uniform / E-top-k / E-kernel / 随机-top-k 对照 / P⊕G-top-k 诊断）。

用法：python scripts/19_source_selection.py
"""
from __future__ import annotations

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

from autocls.data.source_pool import delta_for  # noqa: E402
from autocls.labeling import load_probe_cls, make_pairs, run_seed_for  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402

OUTDIR = ROOT / "results" / "mx2a_source_selection"
MX1B_FOLDS = ROOT / "results" / "mx1b_feasibility" / "folds"
K_PRIMARY, K_SENS = 5, (3, 8)        # 协议 §3 写死
N_SEL_SEEDS = 5                      # WR 随机对照的选择种子数
N_MODEL_SEEDS = 5
EPOCHS, LR, WD = 300, 1e-3, 1e-3     # 沿用 MX-1 §4.2 同族，禁调


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def dataset_distances(task_feats: dict, train_ds: dict, heldout_tids: list) -> dict:
    """留出数据集到各训练数据集的特征距离（标准化统计量只来自训练任务）。"""
    train_tids = sorted(t for ts in train_ds.values() for t in ts)
    M = np.stack([np.asarray(task_feats[t], dtype=float) for t in train_tids])
    mu, sd = lo.standardize_fit(M)
    z = lambda t: lo.standardize_apply(  # noqa: E731
        np.asarray(task_feats[t], dtype=float), mu, sd)
    held = np.mean([z(t) for t in sorted(heldout_tids)], axis=0)
    return {ds: float(np.linalg.norm(np.mean([z(t) for t in ts], axis=0) - held))
            for ds, ts in train_ds.items()}


def top_k_sources(dists: dict, k: int) -> list:
    """距离升序前 k（并列按名字，确定性）。"""
    return [ds for ds, _ in sorted(dists.items(), key=lambda kv: (kv[1], kv[0]))[:k]]


def kernel_weights(dists: dict, sigma: float) -> dict:
    return {ds: float(np.exp(-d * d / (2 * sigma * sigma)))
            for ds, d in dists.items()}


def train_pairwise_median(task_feats: dict, train_ds: dict) -> float:
    """训练数据集两两距离的中位数（W2 的 σ，协议 §3 写死）。"""
    train_tids = sorted(t for ts in train_ds.values() for t in ts)
    M = np.stack([np.asarray(task_feats[t], dtype=float) for t in train_tids])
    mu, sd = lo.standardize_fit(M)
    means = {ds: np.mean([lo.standardize_apply(
        np.asarray(task_feats[t], dtype=float), mu, sd) for t in ts], axis=0)
        for ds, ts in train_ds.items()}
    names = sorted(means)
    dd = [float(np.linalg.norm(means[a] - means[b]))
          for i, a in enumerate(names) for b in names[i + 1:]]
    return float(np.median(dd))


def random_sources(pool: list, k: int, seed: int) -> list:
    rng = np.random.default_rng([seed, 20260832])
    return sorted(rng.choice(sorted(pool), size=k, replace=False).tolist())


def pairs_for_task(rows: list, *, delta: float) -> tuple[list, dict]:
    """一个任务的 label 行 → δ 配对（make_pairs 原味，classification 方向）。"""
    rprime = {r["cls_key"]: r["r_prime"] for r in rows}
    return make_pairs(rprime, "classification", delta=delta)


class PairNet(torch.nn.Module):
    """v_a⊕v_b → [128, 64] → 1 logit（协议 §3；E 不进输入）。"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_pairnet(Xa, Xb, y, w, *, seed: int, epochs: int = EPOCHS) -> PairNet:
    torch.manual_seed(seed)
    X = torch.as_tensor(np.concatenate([Xa, Xb], axis=1), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    wt = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    model = PairNet(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = torch.nn.BCEWithLogitsLoss(weight=wt)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(model(X), yt)
        loss.backward()
        opt.step()
    return model.eval()


def pairnet_proba(model: PairNet, Xa, Xb) -> np.ndarray:
    X = torch.as_tensor(np.concatenate([Xa, Xb], axis=1), dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def verdict_mx2a(p_d1d: float, p_d2d: float, p_w2: float, p_w1pg: float) -> str:
    """判读带（协议 §4）。"""
    if p_d1d < 0.05:
        return "established" if p_d2d < 0.05 else "fewer_sources_effect"
    if p_d1d < 0.15 or p_w2 < 0.05 or p_w1pg < 0.05:
        return "edge"
    return "not_established"


# ---------------------------------------------------------------- 主流程
def _load_feats(fold: str, tids: list, kind: str) -> dict:
    """kind='E'（MX-1 折特征）或 'PG'（MX-1b probe_features 的 P⊕G）。"""
    out = {}
    for tid in tids:
        if kind == "E":
            with np.load(lo.FOLDS_DIR / fold / "task_features" / f"{tid}.npz") as z:
                out[tid] = z["E"]
        else:
            with np.load(MX1B_FOLDS / fold / "probe_features" / f"{tid}.npz") as z:
                out[tid] = np.concatenate([z["P"], z["G"]])
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    labels_cfg = yaml.safe_load(
        (ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    rows = lo.load_panel()
    datasets, tids_by_ds, _cls = lo.panel_index(rows)
    by_task: dict = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space)
    vcls = {key: enc.encode(c)
            for c, key in load_probe_cls(lo._b2.PROBE_CATALOG, 10)[0]}

    # δ 配对（per-dataset δ 查表；弃样计数如实）
    task_pairs: dict = {}
    drop_stats: dict = defaultdict(int)
    for tid, trows in by_task.items():
        ds = trows[0]["dataset"]
        pairs, stats = pairs_for_task(trows, delta=delta_for(ds, labels_cfg))
        task_pairs[tid] = pairs
        for k in ("n_candidates", "n_delta_dropped", "n_kept"):
            drop_stats[k] += stats[k]
    print(f"配对: 候选 {drop_stats['n_candidates']} | δ 弃样 "
          f"{drop_stats['n_delta_dropped']} | 产出 {drop_stats['n_kept']}",
          flush=True)

    def arrs(tids_sel):
        Xa, Xb, y, ds_of_pair = [], [], [], []
        for tid in sorted(tids_sel):
            for p in task_pairs[tid]:
                Xa.append(vcls[p["a"]])
                Xb.append(vcls[p["b"]])
                y.append(int(p["y"]))
                ds_of_pair.append(fl.dataset_of(tid))
        return (np.stack(Xa), np.stack(Xb), np.array(y, dtype=float), ds_of_pair)

    per_fold: dict = {}
    for heldout, train_ds_names in lo.fold_split(datasets):
        t0 = time.time()
        train_map = {d: tids_by_ds[d] for d in train_ds_names}
        held_tids = tids_by_ds[heldout]
        all_tids = sorted(t for d in datasets for t in tids_by_ds[d])
        featE = _load_feats(heldout, all_tids, "E")
        featPG = _load_feats(heldout, all_tids, "PG")
        dE = dataset_distances(featE, train_map, held_tids)
        dPG = dataset_distances(featPG, train_map, held_tids)
        sigma = train_pairwise_median(featE, train_map)
        Xa_te, Xb_te, y_te, _ = arrs(held_tids)

        def run_arm(sources, weights, arm):
            Xa, Xb, y, dsp = arrs(t for d in sources for t in tids_by_ds[d])
            w = np.array([weights.get(d, 1.0) for d in dsp])
            w = w / w.mean()
            probs = np.mean([pairnet_proba(
                train_pairnet(Xa, Xb, y, w,
                              seed=run_seed_for(f"mx2a:{heldout}:{arm}", f"m{s}")),
                Xa_te, Xb_te) for s in range(N_MODEL_SEEDS)], axis=0)
            return float(((probs > 0.5) == y_te.astype(bool)).mean())

        entry = {"n_test_pairs": len(y_te), "nearest_E": top_k_sources(dE, 3),
                 "acc": {}}
        entry["acc"]["W0"] = run_arm(train_ds_names, {}, "W0")
        entry["acc"]["W1"] = run_arm(top_k_sources(dE, K_PRIMARY), {}, "W1")
        entry["acc"]["W2"] = run_arm(train_ds_names, kernel_weights(dE, sigma), "W2")
        entry["acc"]["W1pg"] = run_arm(top_k_sources(dPG, K_PRIMARY), {}, "W1pg")
        for k in K_SENS:
            entry["acc"][f"W1_k{k}"] = run_arm(top_k_sources(dE, k), {}, f"W1k{k}")
        wr = [run_arm(random_sources(train_ds_names, K_PRIMARY, seed=s), {},
                      f"WR{s}") for s in range(N_SEL_SEEDS)]
        entry["acc"]["WR"] = float(np.mean(wr))
        entry["acc_wr_each"] = [round(v, 4) for v in wr]
        per_fold[heldout] = entry
        print(f"[{heldout}] test_pairs={len(y_te)} "
              + " ".join(f"{k}={v:.3f}" for k, v in entry["acc"].items())
              + f" ({time.time() - t0:.0f}s)", flush=True)

    active = sorted(per_fold)
    acc = lambda arm: np.array([per_fold[d]["acc"][arm] for d in active])  # noqa: E731
    tests = {
        "D1d_p": lo.paired_wilcoxon_less(acc("W0"), acc("W1")),
        "D2d_p": lo.paired_wilcoxon_less(acc("WR"), acc("W1")),
        "W2_vs_W0_p": lo.paired_wilcoxon_less(acc("W0"), acc("W2")),
        "W1pg_vs_W0_p": lo.paired_wilcoxon_less(acc("W0"), acc("W1pg")),
        "P3d_w1pg_ge_w1_folds": int((acc("W1pg") >= acc("W1")).sum()),
        "P4d_w0_mean": float(acc("W0").mean()),
        "acc_mean": {arm: float(acc(arm).mean())
                     for arm in ("W0", "W1", "W2", "WR", "W1pg",
                                 *(f"W1_k{k}" for k in K_SENS))},
        "sens_k": {f"k{k}": lo.paired_wilcoxon_less(acc("W0"), acc(f"W1_k{k}"))
                   for k in K_SENS},
        "pair_stats": dict(drop_stats),
    }
    verdict = verdict_mx2a(tests["D1d_p"], tests["D2d_p"],
                           tests["W2_vs_W0_p"], tests["W1pg_vs_W0_p"])
    out = {"protocol": "results/mx2a_source_selection/PROTOCOL.md",
           "per_fold": per_fold, "tests": tests, "verdict": verdict}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    t = tests
    print(f"\n===== MX-2a 选源（{len(active)} 折）=====")
    print("均值 acc: " + "  ".join(f"{k}={v:.4f}" for k, v in t["acc_mean"].items()))
    print(f"D1d(W1>W0) p={t['D1d_p']:.4f} | D2d(W1>WR) p={t['D2d_p']:.4f} | "
          f"W2 p={t['W2_vs_W0_p']:.4f} | W1pg p={t['W1pg_vs_W0_p']:.4f}")
    print(f"P3d: W1pg≥W1 折数 {t['P3d_w1pg_ge_w1_folds']}/16 | "
          f"P4d: W0 均值 {t['P4d_w0_mean']:.4f} | 敏感性 "
          + "  ".join(f"{k}:p={v:.3f}" for k, v in t["sens_k"].items()))
    print(f"\n[判读] → {verdict}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
