"""WP16 SP1 检验：剪空间 S⁺——任务盲"明显错误配置"代理，定义 AutoCTS++ 式全空间搜索的采样域。协议 results/tclsc_sp/PROTOCOL.md。零 GPU。

子命令（按执行顺序；`--all` 一次做完）：
  --labels    源侧 @290 行（mx3_pq + SX 缓存）→ labels.jsonl（within-切片 z、z-bad、可评切片的 strict）
  --fit       LODO 7 折（SCP2 只训练）：FieldSurrogate(hgb) on binned 116 维拟合 z；τ 只用训练折（内层 OOF）；held-out 全部 / random 配置两种口径；
              全数据模型 → S⁺ 占空间（10 万条均匀样本）、采样器自检、字段置换重要性、ridge 对照 → fit.json + sp_model.joblib + tau.json
  --analyze   门 G-S1/2/3、读数码、预测、tables.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_RANDOM, ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g74 = importlib.import_module("scripts.74_tclsc_slices")
OUT = ROOT / "results" / "tclsc_sp"
SOURCE, SX = pq.OUT, _g74.OUT
BUDGET = 290
ZBAD = -1.0                     # z-bad：切片内 z < −1
GOOD_Z = 0.5                    # "好配置"：z ≥ +0.5（召回口径）
MARGIN = 0.02                   # 可评切片：最好 shared − (基线均值 + 带) ≥ 0.02
PURITY = 0.10                   # τ 的纯度目标（训练折 kept 配置 z-bad 率）
FOLDS = tuple(pq.FOLD_TARGETS)
N_SPACE = 100_000
G1_AUC, G1_RHO, G2_RATE, G3_FRAC, G3_RECALL = 0.85, 0.5, 0.10, 0.20, 0.90
SEED = 0


# ---------------------------------------------------------------- 标签
def load_rows() -> list[dict]:
    rows = _g32.read_rows(SOURCE / "rprime_cache.jsonl") + _g32.read_rows(SX / "rprime_cache.jsonl")
    return [r for r in rows if r.get("r_prime") is not None and r.get("error") is None and int(r["iters"]) == BUDGET]


def load_bands() -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for d in pq.PQ_DATASETS:
        try:
            bands.update(pq.baseline_bands(d))
        except Exception as exc:  # noqa: BLE001 —— 缺基线的数据集只是不可评
            print(f"[bands] {d}: {str(exc)[:90]}", flush=True)
    return bands


def is_evaluable(rows: Sequence[dict], task_id: str, band: tuple[float, float], *, margin: float = MARGIN) -> bool:
    """可评切片：最好 shared R' − (基线均值 + 带) ≥ margin（训练能明显超过不训练）。"""
    best = max((float(r["r_prime"]) for r in rows if r["task_id"] == task_id and r["origin"] == ORIGIN_SHARED), default=None)
    if best is None:
        return False
    m, b = band
    return bool(best - (m + b) >= margin)


def make_labels(rows: Sequence[dict], bands: Mapping[str, tuple[float, float]], *, margin: float = MARGIN) -> list[dict]:
    """每行加 z（within-切片）、zbad、evaluable、strict（不可评 → None）。"""
    out = []
    ev_cache: dict[str, bool] = {}
    for r in _g36.zscore_within(rows):
        t = r["task_id"]
        if t not in ev_cache:
            ev_cache[t] = bool(t in bands and is_evaluable(rows, t, bands[t], margin=margin))
        r["zbad"] = bool(r["z"] < ZBAD)
        r["evaluable"] = ev_cache[t]
        if ev_cache[t]:
            m, b = bands[t]
            r["strict"] = bool(float(r["r_prime"]) < m - b)
        else:
            r["strict"] = None
        out.append(r)
    return out


def cmd_labels(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    bands = load_bands()
    lab = make_labels(rows, bands)
    with open(outdir / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in lab:
            fh.write(json.dumps({k: r[k] for k in ("task_id", "dataset", "cls_key", "origin", "iters", "r_prime", "cls", "z", "zbad", "evaluable", "strict")}, ensure_ascii=False) + "\n")
    ev = sorted({r["task_id"] for r in lab if r["evaluable"]})
    by_ds: dict[str, dict] = {}
    for r in lab:
        d = by_ds.setdefault(r["dataset"], {"n": 0, "zbad": 0, "eval_slices": set(), "strict_n": 0, "strict": 0})
        d["n"] += 1
        d["zbad"] += int(r["zbad"])
        if r["evaluable"]:
            d["eval_slices"].add(r["task_id"])
            d["strict_n"] += 1
            d["strict"] += int(bool(r["strict"]))
    summ = {"rows": len(lab), "slices": len({r["task_id"] for r in lab}), "configs": len({r["cls_key"] for r in lab}), "evaluable_slices": ev,
            "zbad_rate": float(np.mean([r["zbad"] for r in lab])), "zbad_rate_random": float(np.mean([r["zbad"] for r in lab if r["origin"] == ORIGIN_RANDOM])),
            "strict_rate_evaluable": float(np.mean([r["strict"] for r in lab if r["evaluable"]])) if ev else None,
            "by_dataset": {d: {"n": v["n"], "zbad_rate": v["zbad"] / v["n"], "evaluable_slices": len(v["eval_slices"]),
                               "strict_rate": (v["strict"] / v["strict_n"]) if v["strict_n"] else None} for d, v in sorted(by_ds.items())}}
    (outdir / "labels_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "by_dataset"}, ensure_ascii=False), flush=True)
    return summ


def read_labels(outdir: Path) -> list[dict]:
    return _g32.read_rows(outdir / "labels.jsonl")


# ---------------------------------------------------------------- 代理 / τ / 采样
_ENC: dict[int, CLSEncoder] = {}


def encoder(space: CLSSpace) -> CLSEncoder:
    k = id(space)
    if k not in _ENC:
        _ENC[k] = CLSEncoder(space, mode="binned")
    return _ENC[k]


def encode(space: CLSSpace, cfgs: Sequence[CLS]) -> np.ndarray:
    enc = encoder(space)
    return np.stack([enc.encode(c) for c in cfgs]).astype(np.float64)


def fit_quality(X: np.ndarray, z: np.ndarray, *, seed: int = SEED, kind: str = "hgb") -> FieldSurrogate:
    return FieldSurrogate(kind, bagging=1, seed=seed).fit(np.asarray(X, float), np.asarray(z, float))


def predict_quality(model: FieldSurrogate, X: np.ndarray) -> np.ndarray:
    return model.predict(np.asarray(X, float))[0]


def choose_tau(zhat: np.ndarray, zbad: np.ndarray, *, target: float = PURITY) -> float:
    """使 kept = {ẑ ≥ τ} 的 z-bad 率 ≤ target 的最小 τ（候选 = 观测到的 ẑ 值）；无解 → 最大 ẑ。"""
    zhat, zbad = np.asarray(zhat, float), np.asarray(zbad, bool)
    order = np.argsort(-zhat)                      # ẑ 降序：前缀 = kept 集
    bad_cum = np.cumsum(zbad[order])
    n = np.arange(1, len(zhat) + 1)
    ok = bad_cum / n <= target
    if not ok.any():
        return float(zhat.max())
    last = int(np.max(np.where(ok)[0]))            # 满足纯度的最大前缀 → 最小 τ
    return float(zhat[order][last])


def sample_splus(space: CLSSpace, model: FieldSurrogate, tau: float, *, n: int, rng: np.random.Generator, batch: int = 256, max_batches: int = 400) -> list[CLS]:
    """S⁺ 均匀采样 = 均匀采样 + 拒绝（ẑ < τ 丢弃）；同 rng 逐位确定。"""
    out: list[CLS] = []
    for _ in range(max_batches):
        cand = [space.sample_random(rng) for _ in range(batch)]
        zh = predict_quality(model, encode(space, cand))
        out += [c for c, v in zip(cand, zh) if v >= tau]
        if len(out) >= n:
            return out[:n]
    raise RuntimeError(f"S⁺ 采样 {max_batches} 批仍不足 {n} 条（τ={tau:.3f} 太严）")


def space_fraction(space: CLSSpace, model: FieldSurrogate, tau: float, *, n: int = N_SPACE, seed: int = SEED, batch: int = 4096) -> float:
    rng = np.random.default_rng([seed, 7801])
    kept = 0
    done = 0
    while done < n:
        m = min(batch, n - done)
        cand = [space.sample_random(rng) for _ in range(m)]
        kept += int((predict_quality(model, encode(space, cand)) >= tau).sum())
        done += m
    return kept / n


# ---------------------------------------------------------------- 指标
def _auc(score: np.ndarray, pos: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    pos = np.asarray(pos, bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return None
    return float(roc_auc_score(pos, score))


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    from scipy.stats import spearmanr
    if len(a) < 3:
        return None
    return float(spearmanr(a, b).correlation)


def fold_metrics(rows: Sequence[dict], zhat: np.ndarray, tau: float) -> dict:
    """held-out 一个口径的读数：AUC（−ẑ 判 z-bad）、Spearman、kept 的纯度 / 覆盖 / 好配置召回。"""
    z = np.array([r["z"] for r in rows]); zbad = np.array([r["zbad"] for r in rows], bool)
    keep = zhat >= tau
    strict_rows = [(r, k) for r, k in zip(rows, keep) if r["evaluable"]]
    good = z >= GOOD_Z
    return {"n": len(rows), "auc": _auc(-zhat, zbad), "spearman": _spearman(zhat, z), "kept_frac": float(keep.mean()),
            "zbad_rate": float(zbad[keep].mean()) if keep.any() else None,
            "strict_rate": (float(np.mean([bool(r["strict"]) for r, k in strict_rows if k])) if any(k for _r, k in strict_rows) else None),
            "n_strict": int(sum(1 for _r, k in strict_rows if k)),
            "recall_good": float(keep[good].mean()) if good.any() else None, "n_good": int(good.sum())}


def _pooled(parts: Sequence[tuple[Sequence[dict], np.ndarray, float]]) -> dict:
    rows = [r for rs, _z, _t in parts for r in rs]
    zh = np.concatenate([np.asarray(z) for _r, z, _t in parts])
    keep = np.concatenate([np.asarray(z) >= t for _r, z, t in parts])
    z = np.array([r["z"] for r in rows]); zbad = np.array([r["zbad"] for r in rows], bool)
    strict = [(bool(r["strict"]), k) for r, k in zip(rows, keep) if r["evaluable"]]
    good = z >= GOOD_Z
    return {"n": len(rows), "auc": _auc(-zh, zbad), "spearman": _spearman(zh, z), "kept_frac": float(keep.mean()),
            "zbad_rate": float(zbad[keep].mean()) if keep.any() else None,
            "strict_rate": float(np.mean([s for s, k in strict if k])) if any(k for _s, k in strict) else None, "n_strict": int(sum(1 for _s, k in strict if k)),
            "recall_good": float(keep[good].mean()) if good.any() else None, "n_good": int(good.sum())}


def oof_predictions(rows: Sequence[dict], X: np.ndarray, datasets: Sequence[str], *, seed: int, kind: str = "hgb") -> np.ndarray:
    """训练集内的 LODO OOF 预测（用于在训练折上选 τ，不看 held-out）。"""
    ds = np.array([r["dataset"] for r in rows])
    z = np.array([r["z"] for r in rows])
    oof = np.full(len(rows), np.nan)
    for d in sorted(set(datasets)):
        te = ds == d
        if te.all() or not te.any():
            continue
        m = fit_quality(X[~te], z[~te], seed=seed, kind=kind)
        oof[te] = predict_quality(m, X[te])
    return oof


def field_importance(space: CLSSpace, model: FieldSurrogate, X: np.ndarray, z: np.ndarray, *, seed: int) -> dict[str, float]:
    """按字段块置换的 Spearman 下降（held-out）。"""
    rng = np.random.default_rng([seed, 7802])
    base = _spearman(predict_quality(model, X), z) or 0.0
    out = {}
    for f, sl in encoder(space).field_slices.items():
        Xp = X.copy()
        Xp[:, sl] = Xp[rng.permutation(len(X))][:, sl]
        out[f] = float(base - (_spearman(predict_quality(model, Xp), z) or 0.0))
    return out


def cmd_fit(outdir: Path, *, seed: int = SEED, n_space: int = N_SPACE) -> dict:
    import joblib
    rows = read_labels(outdir)
    space = CLSSpace.from_yaml()
    X = encode(space, [CLS(**r["cls"]) for r in rows])
    z = np.array([r["z"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    res: dict = {"folds": {}, "pooled": {}, "ridge": {}, "importance": {}}
    parts_all, parts_rand = [], []
    imp_acc: dict[str, list] = {}
    for target in FOLDS:
        te = ds == target
        tr_rows = [r for r, t in zip(rows, te) if not t]
        oof = oof_predictions(tr_rows, X[~te], [r["dataset"] for r in tr_rows], seed=seed)
        ok = ~np.isnan(oof)
        tau = choose_tau(oof[ok], np.array([r["zbad"] for r in tr_rows], bool)[ok], target=PURITY)
        model = fit_quality(X[~te], z[~te], seed=seed)
        zh = predict_quality(model, X[te])
        te_rows = [r for r, t in zip(rows, te) if t]
        rand = np.array([r["origin"] == ORIGIN_RANDOM for r in te_rows], bool)
        fm_all = fold_metrics(te_rows, zh, tau)
        fm_rand = fold_metrics([r for r, k in zip(te_rows, rand) if k], zh[rand], tau)
        parts_all.append((te_rows, zh, tau))
        parts_rand.append(([r for r, k in zip(te_rows, rand) if k], zh[rand], tau))
        for f, v in field_importance(space, model, X[te], z[te], seed=seed).items():
            imp_acc.setdefault(f, []).append(v)
        ridge = fit_quality(X[~te], z[~te], seed=seed, kind="ridge")
        zr = predict_quality(ridge, X[te])
        res["ridge"][target] = {"auc_random": _auc(-zr[rand], np.array([r["zbad"] for r in te_rows], bool)[rand]), "spearman_random": _spearman(zr[rand], z[te][rand])}
        res["folds"][target] = {"tau": tau, "all": fm_all, "random": fm_rand, "n_train": int((~te).sum())}
        print(f"[fit {target}] τ {tau:+.3f} | all AUC {fm_all['auc']} ρ {fm_all['spearman']} kept {fm_all['kept_frac']:.2f} zbad {fm_all['zbad_rate']} strict {fm_all['strict_rate']} recall {fm_all['recall_good']} "
              f"| random AUC {fm_rand['auc']} ρ {fm_rand['spearman']} zbad {fm_rand['zbad_rate']} recall {fm_rand['recall_good']}", flush=True)
    res["pooled"] = {"all": _pooled(parts_all), "random": _pooled(parts_rand)}
    res["ridge"]["pooled_random"] = {"auc": _auc(-np.concatenate([predict_quality(fit_quality(X[ds != t], z[ds != t], seed=seed, kind="ridge"), X[(ds == t) & np.array([r["origin"] == ORIGIN_RANDOM for r in rows])]) for t in FOLDS]),
                                                np.concatenate([np.array([r["zbad"] for r in rows], bool)[(ds == t) & np.array([r["origin"] == ORIGIN_RANDOM for r in rows])] for t in FOLDS]))}
    res["importance"] = {f: float(np.mean(v)) for f, v in sorted(imp_acc.items(), key=lambda kv: -np.mean(kv[1]))}
    # 全数据模型 + τ（全部 8 集的 LODO OOF 上选）+ S⁺ 占空间 + 采样器自检
    oof_all = oof_predictions(rows, X, [r["dataset"] for r in rows], seed=seed)
    ok = ~np.isnan(oof_all)
    tau_all = choose_tau(oof_all[ok], np.array([r["zbad"] for r in rows], bool)[ok], target=PURITY)
    final = fit_quality(X, z, seed=seed)
    frac = space_fraction(space, final, tau_all, n=n_space, seed=seed)
    samp = sample_splus(space, final, tau_all, n=50, rng=np.random.default_rng([seed, 7803]))
    zh_samp = predict_quality(final, encode(space, samp))
    shared = pq.load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    known = [(k, c) for c, k in shared] + [(pq.cls_key(pq.ggs_cls()), pq.ggs_cls())]
    zh_known = predict_quality(final, encode(space, [c for _k, c in known]))
    res["final"] = {"tau": tau_all, "space_fraction": frac, "n_space": n_space, "sampler_check": {"n": len(samp), "min_zhat": float(zh_samp.min()), "all_valid": all(space.validate(c) for c in samp)},
                    "known_in_splus": {k[:8]: bool(v >= tau_all) for (k, _c), v in zip(known, zh_known)}, "known_zhat": {k[:8]: float(v) for (k, _c), v in zip(known, zh_known)},
                    "oof_kept_frac": float((oof_all[ok] >= tau_all).mean()), "oof_zbad_rate_kept": float(np.array([r["zbad"] for r in rows], bool)[ok][oof_all[ok] >= tau_all].mean())}
    # τ–纯度–覆盖曲线（全数据 OOF）
    zb = np.array([r["zbad"] for r in rows], bool)[ok]; zz = z[ok]; oo = oof_all[ok]
    curve = []
    for q in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        t = float(np.quantile(oo, q))
        keep = oo >= t
        curve.append({"quantile": q, "tau": t, "kept_frac_labeled": float(keep.mean()), "zbad_rate": float(zb[keep].mean()), "recall_good": float(keep[zz >= GOOD_Z].mean())})
    res["curve"] = curve
    joblib.dump(final, outdir / "sp_model.joblib")
    (outdir / "tau.json").write_text(json.dumps({"tau": tau_all, "space_fraction": frac, "purity_target": PURITY, "zbad": ZBAD, "encoder": "binned", "seed": seed}, indent=1), encoding="utf-8")
    (outdir / "fit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[final] τ {tau_all:+.3f} S⁺ 占空间 {frac:.3f} known_in_splus {res['final']['known_in_splus']}", flush=True)
    return res


# ---------------------------------------------------------------- 门 / 读数码
def gate_s1(m: Mapping) -> bool:
    return bool(m.get("auc") is not None and m["auc"] >= G1_AUC and m.get("spearman") is not None and m["spearman"] >= G1_RHO)


def gate_s2(m: Mapping) -> bool:
    if m.get("zbad_rate") is None or m["zbad_rate"] > G2_RATE:
        return False
    return bool(m.get("strict_rate") is None or m["strict_rate"] <= G2_RATE)


def gate_s3(m: Mapping) -> bool:
    return bool(m.get("fraction") is not None and m["fraction"] >= G3_FRAC and m.get("recall_good") is not None and m["recall_good"] >= G3_RECALL)


def verdict_sp(g: Mapping[str, bool]) -> str:
    if not (g["G-S1"] and g["G-S2"]):
        return "SP-DEAD"
    return "SP-GO" if g["G-S3"] else "SP-TIGHT"


def cmd_analyze(outdir: Path) -> dict:
    fit = json.loads((outdir / "fit.json").read_text(encoding="utf-8"))
    ls = json.loads((outdir / "labels_summary.json").read_text(encoding="utf-8"))
    pr, pa = fit["pooled"]["random"], fit["pooled"]["all"]
    g = {"G-S1": gate_s1(pr), "G-S2": gate_s2(pa), "G-S3": gate_s3({"fraction": fit["final"]["space_fraction"], "recall_good": pa["recall_good"]})}
    imp = list(fit["importance"])
    preds = {"P1": {"desc": "AUC ≥ 0.85（random 口径）", "value": pr["auc"], "hit": bool(pr["auc"] is not None and pr["auc"] >= G1_AUC)},
             "P2": {"desc": "S⁺ 占空间 20–50%", "value": fit["final"]["space_fraction"], "hit": bool(0.2 <= fit["final"]["space_fraction"] <= 0.5)},
             "P3": {"desc": "置换重要性前 2 = temperature, sim_func", "value": imp[:3], "hit": bool(set(imp[:2]) == {"temperature", "sim_func"})},
             "P4": {"desc": "G-S2", "hit": g["G-S2"]}, "P5": {"desc": "好配置召回 ≥ 0.9", "value": pa["recall_good"], "hit": bool(pa["recall_good"] is not None and pa["recall_good"] >= G3_RECALL)},
             "P6": {"desc": "LODO Spearman（random）≥ 0.5", "value": pr["spearman"], "hit": bool(pr["spearman"] is not None and pr["spearman"] >= G1_RHO)}}
    a = {"protocol": "results/tclsc_sp/PROTOCOL.md", "labels": {k: v for k, v in ls.items() if k != "by_dataset"}, "by_dataset": ls["by_dataset"], "pooled": fit["pooled"], "folds": fit["folds"],
         "final": fit["final"], "curve": fit["curve"], "importance": fit["importance"], "ridge": fit["ridge"], "gates": g, "verdict": verdict_sp(g), "predictions": preds}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], g, flush=True)
    return a


def markdown_tables(a: dict) -> str:
    f = lambda v, d=3: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    L = [f"# SP1 检验 tables：剪空间 S⁺", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", "",
         f"标签：{a['labels']}", "", "| 数据集 | n | z-bad 率 | 可评切片 | strict 率 |", "|---|---:|---:|---:|---:|"]
    for d, v in a["by_dataset"].items():
        L.append(f"| {d} | {v['n']} | {f(v['zbad_rate'], 2)} | {v['evaluable_slices']} | {f(v['strict_rate'], 2)} |")
    L += ["", "## LODO pooled", "", "| 口径 | n | AUC | Spearman | kept | z-bad 率 | strict 率 (n) | 好配置召回 (n) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, m in a["pooled"].items():
        L.append(f"| {k} | {m['n']} | {f(m['auc'])} | {f(m['spearman'])} | {f(m['kept_frac'], 2)} | {f(m['zbad_rate'])} | {f(m['strict_rate'])} ({m['n_strict']}) | {f(m['recall_good'])} ({m['n_good']}) |")
    L += ["", "## 逐折（random 口径）", "", "| 折 | τ | AUC | Spearman | kept | z-bad | strict (n) | 召回 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, v in a["folds"].items():
        m = v["random"]; s = v["all"]
        L.append(f"| {t} | {v['tau']:+.3f} | {f(m['auc'])} | {f(m['spearman'])} | {f(s['kept_frac'], 2)} | {f(s['zbad_rate'])} | {f(s['strict_rate'])} ({s['n_strict']}) | {f(s['recall_good'])} |")
    fin = a["final"]
    L += ["", f"## 全数据模型：τ {fin['tau']:+.3f}，S⁺ 占空间 {fin['space_fraction']:.3f}（{fin['n_space']} 样本），OOF kept {fin['oof_kept_frac']:.2f} / z-bad {fin['oof_zbad_rate_kept']:.3f}；信任域在 S⁺：{fin['known_in_splus']}", "",
          "| τ 分位 | τ | 标签内 kept | z-bad 率 | 好配置召回 |", "|---|---:|---:|---:|---:|"] + [f"| {c['quantile']} | {c['tau']:+.3f} | {c['kept_frac_labeled']:.2f} | {c['zbad_rate']:.3f} | {c['recall_good']:.3f} |" for c in a["curve"]]
    L += ["", f"字段置换重要性（Spearman 下降，held-out 均值）：{ {k: round(v, 3) for k, v in a['importance'].items()} }", "", f"ridge 对照（random）：{a['ridge'].get('pooled_random')}", "", f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for fl in ("labels", "fit", "analyze", "all"):
        ap.add_argument(f"--{fl}", action="store_true")
    ap.add_argument("--n-space", type=int, default=N_SPACE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.labels or args.all:
        cmd_labels(args.outdir)
    if args.fit or args.all:
        cmd_fit(args.outdir, seed=args.seed, n_space=args.n_space)
    if args.analyze or args.all:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
