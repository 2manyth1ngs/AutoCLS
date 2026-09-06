"""MX-4 WP6 描述性探针（零 GPU，**非预注册**）：机制性数据集描述量能否解释"掩码收益随数据集变"。

响应量：S1 每对孪生在 100% 截点的 Δ = acc(开) − acc(关)（seed 0，125 对）；S0 的 36 对 3 种子 Δ 作旁证。
配置侧：mask_strength = point_mask + emb_mask（双重掩码假说），aug_strength = 8 个比率之和。
数据集侧（全部不训练、只读 data/uea_cache 的 train 划分）：
  T / C / n_train / n_classes；lag1_ac（一阶自相关）；interp_r2（奇数时刻由相邻偶数时刻线性插值的 R²，时间冗余）；
  mask_1nn_retention（遮掉 50% 时间步后 1-NN 留一准确率 / 不遮的准确率，遮掩下标签信息保留率）；
  headroom = 1 − 随机 encoder 基线 acc（protocol_alignment A1 审计）；passes = 官方 iters × bs / n_train（训练遍数）。
读数：6 个数据集上各描述量与"每数据集平均 Δ / 干净配置 Δ / 正向比例"的 Spearman（n=6，只描述）；
      配对级 LODO：ridge 用 (mask_strength) vs (mask_strength ⊕ 数据集描述量) 预测留出数据集的 Δ 符号（|Δ| ≥ 0.02）。
用法：python results/mx4_mask/mech_probe.py  → results/mx4_mask/mech_probe.{json,md}
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
g40 = importlib.import_module("scripts.40_mask_field_sim")
g35 = importlib.import_module("scripts.35_protocol_alignment")
OUT = ROOT / "results" / "mx4_mask"
TARGETS = list(g40._g39.TARGETS)
RANDOM_ENC_ACC = {"Handwriting": 0.251, "UWaveGestureLibrary": 0.668, "PhonemeSpectra": 0.202,
                  "EthanolConcentration": 0.293, "Cricket": 0.968, "HandMovementDirection": 0.234}   # PA §1 A1
OFFICIAL_ITERS = {"Handwriting": 200}
BS = 8
FLOOR = 0.02
SHORT = {"Handwriting": "HW", "UWaveGestureLibrary": "UWave", "PhonemeSpectra": "PS",
         "EthanolConcentration": "Ethanol", "Cricket": "Cricket", "HandMovementDirection": "HMD"}


# ---------------------------------------------------------------- 数据集描述量
def _zs(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def lag1_autocorr(X):                     # X (N, T, C)
    Z = _zs(X)
    a, b = Z[:, :-1, :], Z[:, 1:, :]
    num = (a * b).mean(axis=1)
    return float(np.nanmean(num))


def interp_r2(X):
    Z = _zs(X)
    odd = Z[:, 1:-1:2, :]
    pred = 0.5 * (Z[:, 0:-2:2, :][:, :odd.shape[1], :] + Z[:, 2::2, :][:, :odd.shape[1], :])
    resid = ((odd - pred) ** 2).mean()
    return float(1.0 - resid / (odd.var() + 1e-8))


def _nn1_loo_acc(F, y):
    F = F.astype(np.float32)
    n = F.shape[0]
    sq = (F ** 2).sum(axis=1)
    acc_hits = 0
    chunk = 512
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = sq[s:e, None] + sq[None, :] - 2.0 * F[s:e] @ F.T
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        nn = d.argmin(axis=1)
        acc_hits += int((y[nn] == y[s:e]).sum())
    return acc_hits / n


def mask_1nn_retention(X, y, seed=0):
    Z = _zs(X)
    N, T, C = Z.shape
    rng = np.random.default_rng(seed)
    keep = rng.random((N, T, 1)) < 0.5          # 与 encoder 的 binomial mask 同分布（keep p=0.5）
    base = _nn1_loo_acc(Z.reshape(N, -1), y)
    masked = _nn1_loo_acc((Z * keep).reshape(N, -1), y)
    return float(masked / base) if base > 0 else float("nan"), float(base), float(masked)


def descriptors(d):
    arr = g35.load_arrays(d)
    X, y = np.asarray(arr["Xtr"], dtype=np.float32), np.asarray(arr["ytr"])
    N, T, C = X.shape
    iters = OFFICIAL_ITERS.get(d, 600)
    ret, base, masked = mask_1nn_retention(X, y)
    return {"T": int(T), "C": int(C), "n_train": int(N), "n_classes": int(len(set(y.tolist()))),
            "lag1_ac": lag1_autocorr(X), "interp_r2": interp_r2(X),
            "mask_1nn_retention": ret, "nn1_base": base, "nn1_masked": masked,
            "headroom": float(1.0 - RANDOM_ENC_ACC[d]), "passes": float(iters * BS / N)}


# ---------------------------------------------------------------- 响应量
def s1_pairs():
    off, on = g40.load_wp3_rows(), g40.load_mask_rows()
    cls_of = {}
    pairs = []
    for d in TARGETS:
        ro, rn = g40._s1_seed0(off, d), g40._s1_seed0(on, d)
        acc_off = {r["cls_key"]: float(r["acc"]) for r in ro if r.get("acc") is not None}
        for r in ro:
            cls_of[r["cls_key"]] = r["cls"]
        for r in rn:
            k = r.get("twin_of")
            if k in acc_off and r.get("acc") is not None:
                c = cls_of[k]
                pairs.append({"dataset": d, "cls_key": k, "delta": float(r["acc"]) - acc_off[k],
                              "mask_strength": float(c["point_mask"]) + float(c["emb_mask"]),
                              "aug_strength": float(sum(float(c[f] or 0.0) for f in g40.AUG_FIELDS)),
                              "arm": r["arm"]})
    return pairs


def s0_pairs():
    from scripts.e_ablation_common import read_jsonl
    return g40.s0_deltas(read_jsonl(g40.PA_RUNS))


# ---------------------------------------------------------------- 分析
def lodo_sign(pairs, feats, floor=FLOOR):
    """留一数据集：ridge(Δ ~ feats) → 留出数据集配对 Δ 的符号准确率（|Δ| ≥ floor）。"""
    from sklearn.linear_model import Ridge
    hits = n = 0
    per = {}
    for d in TARGETS:
        tr = [p for p in pairs if p["dataset"] != d]
        te = [p for p in pairs if p["dataset"] == d and abs(p["delta"]) >= floor]
        if not te:
            per[d] = None
            continue
        Xtr = np.array([[p[f] for f in feats] for p in tr]); ytr = np.array([p["delta"] for p in tr])
        Xte = np.array([[p[f] for f in feats] for p in te]); yte = np.array([p["delta"] for p in te])
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        pr = m.predict(Xte)
        h = int((np.sign(pr) == np.sign(yte)).sum())
        per[d] = (h, len(te))
        hits += h; n += len(te)
    return (hits / n if n else None), n, per


def main():
    desc = {d: descriptors(d) for d in TARGETS}
    pairs = s1_pairs()
    s0 = s0_pairs()
    for p in pairs:
        p.update({k: desc[p["dataset"]][k] for k in ("lag1_ac", "interp_r2", "mask_1nn_retention", "headroom", "passes")})
        p["logT"] = float(np.log10(desc[p["dataset"]]["T"]))
    by_d = defaultdict(list)
    for p in pairs:
        by_d[p["dataset"]].append(p)
    clean = {e["dataset"]: e["delta"] for e in s0 if e["cls_key"].startswith("3c231edc")}
    top1 = {e["dataset"]: e["delta"] for e in s0 if e["cls_key"].startswith("9092e36e")}
    resp = {}
    for d in TARGETS:
        ds = [p["delta"] for p in by_d[d]]
        resp[d] = {"mean_delta": float(np.mean(ds)), "median_delta": float(np.median(ds)),
                   "frac_positive": float(np.mean([x > 0 for x in ds])), "n_pairs": len(ds),
                   "delta_clean_3seed": clean.get(d), "delta_top1_3seed": top1.get(d),
                   "mean_delta_light": float(np.mean([p["delta"] for p in by_d[d] if p["mask_strength"] <= 0.3] or [np.nan])),
                   "mean_delta_heavy": float(np.mean([p["delta"] for p in by_d[d] if p["mask_strength"] >= 0.8] or [np.nan]))}
    feats = ["T", "C", "n_train", "n_classes", "lag1_ac", "interp_r2", "mask_1nn_retention", "headroom", "passes"]
    targets = ["mean_delta", "frac_positive", "delta_clean_3seed", "mean_delta_light"]
    corr = {}
    for f in feats:
        corr[f] = {}
        for t in targets:
            xs = [desc[d][f] for d in TARGETS]; ys = [resp[d][t] for d in TARGETS]
            corr[f][t] = float(spearmanr(xs, ys).correlation)
    # 配对级：配置强度与 Δ（全池）
    ms = [p["mask_strength"] for p in pairs]; ag = [p["aug_strength"] for p in pairs]; dl = [p["delta"] for p in pairs]
    pooled = {"n_pairs": len(pairs), "spearman_mask_strength": float(spearmanr(ms, dl).correlation),
              "spearman_aug_strength": float(spearmanr(ag, dl).correlation),
              "mean_delta": float(np.mean(dl)), "frac_positive": float(np.mean([x > 0 for x in dl])),
              "n_eligible": int(sum(abs(x) >= FLOOR for x in dl))}
    lodo = {}
    for name, fs in [("config_only", ["mask_strength", "aug_strength"]),
                     ("config+lag1", ["mask_strength", "aug_strength", "lag1_ac"]),
                     ("config+interp", ["mask_strength", "aug_strength", "interp_r2"]),
                     ("config+retention", ["mask_strength", "aug_strength", "mask_1nn_retention"]),
                     ("config+headroom", ["mask_strength", "aug_strength", "headroom"]),
                     ("config+passes", ["mask_strength", "aug_strength", "passes"]),
                     ("config+logT", ["mask_strength", "aug_strength", "logT"]),
                     ("config+all", ["mask_strength", "aug_strength", "lag1_ac", "interp_r2", "mask_1nn_retention", "headroom", "passes", "logT"])]:
        acc, n, per = lodo_sign(pairs, fs)
        lodo[name] = {"sign_acc": acc, "n": n, "per_dataset": per}
    on_acc = float(np.mean([x > 0 for x in dl if abs(x) >= FLOOR])) if pooled["n_eligible"] else None
    lodo["always_on"] = {"sign_acc": on_acc, "n": pooled["n_eligible"]}
    lodo["always_off"] = {"sign_acc": (1 - on_acc) if on_acc is not None else None, "n": pooled["n_eligible"]}
    out = {"note": "非预注册描述性探针；n=6 数据集，只报相关与 LODO 符号准确率，不作显著性宣称",
           "descriptors": desc, "response": resp, "spearman_n6": corr, "pooled_pairs": pooled, "lodo_sign": lodo}
    (OUT / "mech_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # markdown
    L = ["## 数据集描述量（train 划分，不训练）", "",
         "| 靶 | T | C | n_train | 类 | lag1_ac | interp_r2 | 1-NN 保留率（遮 50%） | headroom | passes |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in TARGETS:
        e = desc[d]
        L.append(f"| {SHORT[d]} | {e['T']} | {e['C']} | {e['n_train']} | {e['n_classes']} | {e['lag1_ac']:.3f} | {e['interp_r2']:.3f} | "
                 f"{e['mask_1nn_retention']:.3f}（{e['nn1_base']:.2f}→{e['nn1_masked']:.2f}） | {e['headroom']:.3f} | {e['passes']:.1f} |")
    L += ["", "## 每数据集掩码收益（S1 seed 0，100% 截点；S0 3 种子旁证）", "",
          "| 靶 | 对数 | Δ 均值 | Δ 中位 | 正向比例 | Δ 轻掩码配置 | Δ 重掩码配置 | Δ 干净配置(S0) | Δ 先验第一(S0) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in TARGETS:
        r = resp[d]
        f = lambda v: ("—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}")
        L.append(f"| {SHORT[d]} | {r['n_pairs']} | {f(r['mean_delta'])} | {f(r['median_delta'])} | {r['frac_positive']:.2f} | "
                 f"{f(r['mean_delta_light'])} | {f(r['mean_delta_heavy'])} | {f(r['delta_clean_3seed'])} | {f(r['delta_top1_3seed'])} |")
    L += ["", "## Spearman（n = 6 数据集，只描述）", "", "| 描述量 | vs Δ 均值 | vs 正向比例 | vs Δ 干净配置 | vs Δ 轻掩码 |", "|---|---:|---:|---:|---:|"]
    for fe in feats:
        L.append(f"| {fe} | " + " | ".join(f"{corr[fe][t]:+.2f}" for t in targets) + " |")
    L += ["", f"配对级（{pooled['n_pairs']} 对）：Spearman(mask_strength, Δ) = {pooled['spearman_mask_strength']:+.3f}，"
          f"Spearman(aug_strength, Δ) = {pooled['spearman_aug_strength']:+.3f}，Δ 均值 {pooled['mean_delta']:+.4f}，正向 {pooled['frac_positive']:.2f}，"
          f"|Δ| ≥ {FLOOR} 的 {pooled['n_eligible']} 对", "",
          "## 留一数据集符号预测（ridge，|Δ| ≥ 0.02 的配对）", "", "| 特征集 | 符号准确率 | n | 逐靶（命中/对数） |", "|---|---:|---:|---|"]
    for name, v in lodo.items():
        per = v.get("per_dataset")
        if per:
            parts = []
            for d in TARGETS:
                cell = "—" if per[d] is None else f"{per[d][0]}/{per[d][1]}"
                parts.append(f"{SHORT[d]} {cell}")
            pd = " ".join(parts)
        else:
            pd = "—"
        acc_s = "—" if v["sign_acc"] is None else f"{v['sign_acc']:.2f}"
        L.append(f"| {name} | {acc_s} | {v['n']} | {pd} |")
    (OUT / "mech_probe.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
