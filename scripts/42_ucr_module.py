"""WP7-UCR：在 UCR 语料上训练"CLS × 任务特征"打分模块 f(CLS, φ(数据集)) → within-dataset z，冻结后插入主管线。

协议：results/ucr_module/PROTOCOL.md（U1 语料内留一 / U2 主管线 7 折 / U3 官方口径）。
φ = 14 维：40 号的 12 维（meta 4 / iters=0 landmark 2 / 信号 6）⊕ 经典分解的趋势强度、季节强度
（TS-ARM 口径的 STL 强度公式：1 − Var(残差)/Var(成分+残差)；此处用移动平均趋势 + 相位均值季节的经典分解，
周期取周期图峰值、限 [2, T/4]；不依赖 statsmodels）。
臂：hgb（主，FFORMA 型 GBDT ⊕ φ）/ ridge_int（CLS 字段 one-hot × φ 显式交互）/ twin_knn（TS-ARM 型孪生匹配）/
hgb_shuffle×n（φ 置换对照：测试时换成 derangement 供体数据集的 φ）/ pooled（任务盲：shared 均值 + 任务盲 HGB）。
复用 36/37/40/41 的纯函数（importlib），不改它们与 autocls 模块（铁律 7）。

用法：
  python scripts/42_ucr_module.py --phi          # φ_ucr / φ_uea / φ_official → results/ucr_module/
  python scripts/42_ucr_module.py --lodo         # U1
  python scripts/42_ucr_module.py --fit          # 冻结 module.pkl（全部 UCR）
  python scripts/42_ucr_module.py --uea          # U2（7 折，臂表见协议）
  python scripts/42_ucr_module.py --official     # U3（6 靶）
  python scripts/42_ucr_module.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g37 = importlib.import_module("scripts.37_surrogate_loo")
_g40 = importlib.import_module("scripts.40_task_feature_scorer")

# ---------------------------------------------------------------- 常量
OUT = ROOT / "results" / "ucr_module"
RPRIME = ROOT / "results" / "ucr_rprime"
BUDGETS = tuple(pq.BUDGETS)
PHI_COLS = tuple(_g40.PHI_COLS) + ("trend_strength", "season_strength")   # 14
PHI_GROUPS = {"meta": tuple(_g40.PHI_META), "base": tuple(_g40.PHI_BASE), "sig": tuple(_g40.PHI_SIG),
              "stl": ("trend_strength", "season_strength")}
INT_FIELDS = tuple(_g40.INT_FIELDS)
RIDGE_ALPHA = _g40.RIDGE_ALPHA
TOP_N, K_TOP = _g40.TOP_N, _g40.K_TOP
N_SHUFFLE = 5
ALPHA = 0.05
U1_MAIN = "hgb"
U2_MAIN = "ucr_direct"


# ---------------------------------------------------------------- φ：经典分解强度
def _period(x: np.ndarray) -> int:
    T = len(x)
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    if P.shape[0] <= 1:
        return 2
    k = int(np.argmax(P[1:])) + 1
    p = int(round(T / k))
    return int(min(max(p, 2), max(T // 4, 2)))


def _strength_one(x: np.ndarray) -> tuple[float, float]:
    T = len(x)
    if T < 8 or x.var() <= 1e-12:
        return 0.0, 0.0
    p = _period(x)
    w = p if p % 2 == 1 else p + 1
    pad = w // 2
    xp = np.pad(x, pad, mode="reflect")
    trend = np.convolve(xp, np.ones(w) / w, mode="valid")[:T]
    detr = x - trend
    phase = np.arange(T) % p
    seasonal = np.zeros(T)
    for ph in range(p):
        m = phase == ph
        seasonal[m] = detr[m].mean()
    rem = detr - seasonal
    v_rem = rem.var()
    v_ts = (x - seasonal).var()
    v_ss = detr.var()
    ts = max(0.0, 1.0 - v_rem / v_ts) if v_ts > 1e-12 else 0.0
    ss = max(0.0, 1.0 - v_rem / v_ss) if v_ss > 1e-12 else 0.0
    return float(ts), float(ss)


def decomp_strength(X: np.ndarray) -> dict[str, float]:
    """(n, T, C) → 趋势强度 / 季节强度：逐 (样本, 通道) 经典分解后取中位。"""
    X = np.asarray(X, dtype=float)
    n, T, C = X.shape
    ts, ss = [], []
    for i in range(n):
        for c in range(C):
            a, b = _strength_one(X[i, :, c])
            ts.append(a)
            ss.append(b)
    return {"trend_strength": float(np.median(ts)) if ts else 0.0,
            "season_strength": float(np.median(ss)) if ss else 0.0}


def phi_vec(d: Mapping[str, float], cols: Sequence[str] = PHI_COLS) -> np.ndarray:
    return np.array([float(d[c]) for c in cols], dtype=float)


def phi_stats(phi_by_task: Mapping[str, Mapping], tasks: Sequence[str], cols: Sequence[str] = PHI_COLS):
    M = np.stack([phi_vec(phi_by_task[t], cols) for t in tasks])
    mu, sd = M.mean(axis=0), M.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


# ---------------------------------------------------------------- 模块
_ENC_CACHE: dict = {}


def _space_enc():
    """CLSSpace / 编码器进程内缓存（每次 from_yaml + 建编码器约 10 ms，U1/U2 会调用上万次）。"""
    if "enc" not in _ENC_CACHE:
        space = CLSSpace.from_yaml()
        _ENC_CACHE["enc"] = (space, CLSEncoder(space), CLSEncoder(space, mode="binned"))
    return _ENC_CACHE["enc"]


class UcrModule:
    """f(CLS, φ) → within-dataset z。kind ∈ {"hgb", "ridge_int"}。冻结 = pickle（含 φ 标准化统计量、训练 φ、shared 偏好表）。"""

    def __init__(self, kind: str = "hgb", *, cols: Sequence[str] = PHI_COLS, seed: int = 0,
                 int_fields: Sequence[str] = INT_FIELDS):
        if kind not in ("hgb", "ridge_int"):
            raise ValueError(kind)
        self.kind, self.cols, self.seed, self.int_fields = kind, tuple(cols), int(seed), tuple(int_fields)
        self.model = None
        self.blind = None
        self.mu = self.sd = None
        self.train_phi: dict[str, np.ndarray] = {}
        self.train_dataset_of: dict[str, str] = {}
        self.profiles: dict[str, dict[str, float]] = {}
        self.pooled_prior: dict[str, float] = {}
        self.n_datasets = 0

    # ---- 特征
    def _X(self, rows: Sequence[dict], Phi: np.ndarray, enc_b: CLSEncoder) -> np.ndarray:
        Xb = _g36.features(rows, enc_b, max(BUDGETS))
        if self.kind == "hgb":
            return np.hstack([Xb, Phi])
        Ps = (Phi - self.mu) / self.sd
        return np.hstack([Xb, Ps, _g40.interaction_columns(Xb, Ps, enc_b.field_slices, self.int_fields)])

    def fit(self, rows: Sequence[dict], phi_by_task: Mapping[str, Mapping]) -> "UcrModule":
        _space, enc_s, enc_b = _space_enc()
        rows_z = _g36.zscore_within(rows)
        tasks = sorted({r["task_id"] for r in rows_z})
        self.mu, self.sd = phi_stats(phi_by_task, tasks, self.cols)
        self.train_phi = {t: phi_vec(phi_by_task[t], self.cols) for t in tasks}
        self.train_dataset_of = {r["task_id"]: r["dataset"] for r in rows_z}
        self.n_datasets = len({r["dataset"] for r in rows_z})
        Phi = np.stack([self.train_phi[r["task_id"]] for r in rows_z])
        z = np.array([float(r["z"]) for r in rows_z])
        if self.kind == "hgb":
            self.model = FieldSurrogate("hgb", seed=self.seed).fit(self._X(rows_z, Phi, enc_b), z)
        else:
            self.model = FieldSurrogate("ridge", seed=self.seed, alpha=RIDGE_ALPHA).fit(self._X(rows_z, Phi, enc_b), z)
        self.blind = FieldSurrogate("hgb", seed=self.seed).fit(_g36.features(rows_z, enc_s, max(BUDGETS)), z)
        shared = [r for r in rows_z if r["origin"] == ORIGIN_SHARED]
        self.pooled_prior = _g36.prior_scores(shared)
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in shared:
            acc[r["dataset"]][r["cls_key"]].append(float(r["z"]))
        self.profiles = {d: {k: float(np.mean(v)) for k, v in kv.items()} for d, kv in acc.items()}
        return self

    def predict_with_phi(self, rows: Sequence[dict], phi: np.ndarray | Mapping[str, float]) -> np.ndarray:
        _space, _enc_s, enc_b = _space_enc()
        v = phi_vec(phi, self.cols) if isinstance(phi, Mapping) else np.asarray(phi, dtype=float)
        Phi = np.repeat(v[None, :], len(rows), axis=0)
        return self.model.predict(self._X(rows, Phi, enc_b))[0]

    def predict(self, rows: Sequence[dict], phi_by_task: Mapping[str, Mapping]) -> np.ndarray:
        _space, _enc_s, enc_b = _space_enc()
        Phi = np.stack([phi_vec(phi_by_task[r["task_id"]], self.cols) for r in rows])
        return self.model.predict(self._X(rows, Phi, enc_b))[0]

    def blind_predict(self, rows: Sequence[dict]) -> np.ndarray:
        _space, enc_s, _enc_b = _space_enc()
        return self.blind.predict(_g36.features(rows, enc_s, max(BUDGETS)))[0]

    def mean_prediction(self, rows: Sequence[dict]) -> np.ndarray:
        """f̄(c) = f 在全部训练数据集 φ 上的均值（每数据集一个 φ，切片同数据集共用）。"""
        by_ds: dict[str, np.ndarray] = {}
        for t, v in self.train_phi.items():
            by_ds.setdefault(self.train_dataset_of[t], v)
        return np.mean([self.predict_with_phi(rows, v) for v in by_ds.values()], axis=0)

    def dev_scores(self, rows: Sequence[dict], phi: np.ndarray | Mapping[str, float]) -> np.ndarray:
        return self.predict_with_phi(rows, phi) - self.mean_prediction(rows)

    def twin_phi(self) -> dict[str, np.ndarray]:
        """数据集级标准化 φ（孪生匹配用）。"""
        out: dict[str, np.ndarray] = {}
        for t, v in self.train_phi.items():
            out.setdefault(self.train_dataset_of[t], (v - self.mu) / self.sd)
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "UcrModule":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def twin_knn_scores(phi_by_ds: Mapping[str, np.ndarray], profiles: Mapping[str, Mapping[str, float]],
                    q: np.ndarray, keys: Sequence[str], *, k: int = 3) -> dict[str, float]:
    """TS-ARM 型孪生匹配：φ 空间 k 个最近数据集的 shared 偏好均值。"""
    q = np.asarray(q, dtype=float)
    order = sorted(phi_by_ds, key=lambda d: (float(np.sum((np.asarray(phi_by_ds[d]) - q) ** 2)), d))[:k]
    out = {}
    for key in keys:
        vals = [profiles[d][key] for d in order if key in profiles.get(d, {})]
        out[key] = float(np.mean(vals)) if vals else float("-inf")
    return out


# ---------------------------------------------------------------- 名单指标（零样本，无 SH）
def zero_shot_metrics(keys: Sequence[str], truth: Mapping[str, float], ranked: Sequence[str], *,
                      top_n: int = TOP_N, k_top: int = K_TOP) -> dict:
    ranked = list(ranked)
    oracle = max(float(v) for v in truth.values())
    ok = _g40.oracle_key(truth)
    sc = {k: -i for i, k in enumerate(ranked)}
    rho = spearmanr([sc[k] for k in keys], [truth[k] for k in keys]).correlation
    return {"oracle_in_top": ok in ranked[:top_n], "regret1": oracle - float(truth[ranked[0]]),
            "regret3": oracle - max(float(truth[k]) for k in ranked[:k_top]),
            "regret_top": oracle - max(float(truth[k]) for k in ranked[:top_n]),
            "rho": float(rho) if rho == rho else 0.0,
            "pair_acc": _g37.pair_accuracy(sc, truth, 0.0)[0]}


def _derangement(items: Sequence[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    items = list(items)
    while True:
        perm = rng.permutation(len(items))
        if all(int(perm[i]) != i for i in range(len(items))):
            return {items[i]: items[int(perm[i])] for i in range(len(items))}


# ---------------------------------------------------------------- U1：语料内留一数据集
def lodo_ucr(rows: Sequence[dict], phi_by_task: Mapping[str, Mapping], *, n_shuffle: int = N_SHUFFLE,
             seed: int = 0, folds: Optional[Sequence[str]] = None, k_twin: int = 3) -> dict:
    rows_z = _g36.zscore_within(rows)
    datasets = sorted({r["dataset"] for r in rows_z})
    task_of = {r["dataset"]: r["task_id"] for r in rows_z}
    folds = list(folds) if folds else datasets
    derange = [_derangement(datasets, 1000 * seed + 17 * i + 1) for i in range(n_shuffle)]
    inst, donors = [], {}
    for d in folds:
        train = [r for r in rows_z if r["dataset"] != d]
        test = [r for r in rows_z if r["dataset"] == d]
        hgb = UcrModule("hgb", seed=seed).fit(train, phi_by_task)
        ridge = UcrModule("ridge_int", seed=seed).fit(train, phi_by_task)
        phi_t = phi_vec(phi_by_task[task_of[d]])
        q_std = (phi_t - hgb.mu) / hgb.sd
        twin_phi = hgb.twin_phi()
        donors[d] = [dr[d] for dr in derange]
        for b in BUDGETS:
            rows_sb = sorted((r for r in test if int(r["iters"]) == b), key=lambda r: r["cls_key"])
            if not rows_sb:
                continue
            keys = [r["cls_key"] for r in rows_sb]
            truth = {r["cls_key"]: float(r["r_prime"]) for r in rows_sb}
            shared_keys = [k for k, r in zip(keys, rows_sb) if r["origin"] == ORIGIN_SHARED]
            blind = dict(zip(keys, hgb.blind_predict(rows_sb).tolist()))
            pooled = {k: (hgb.pooled_prior[k] if k in hgb.pooled_prior else blind[k]) for k in keys}
            twin = dict(blind)
            twin.update(twin_knn_scores(twin_phi, hgb.profiles, q_std, shared_keys, k=k_twin))
            sc = {"pooled": pooled, "blind": blind,
                  "hgb": dict(zip(keys, hgb.predict_with_phi(rows_sb, phi_t).tolist())),
                  "ridge_int": dict(zip(keys, ridge.predict_with_phi(rows_sb, phi_t).tolist())),
                  "twin_knn": twin}
            for i, dr in enumerate(derange):
                sc[f"hgb_shuffle_{i}"] = dict(zip(keys, hgb.predict_with_phi(
                    rows_sb, phi_vec(phi_by_task[task_of[dr[d]]])).tolist()))
            metrics = {a: zero_shot_metrics(keys, truth, _g36.rank_by(v, keys)) for a, v in sc.items()}
            ok = _g40.oracle_key(truth)
            inst.append({"fold": d, "dataset": d, "budget": b, "oracle": ok,
                         "oracle_origin": next(r["origin"] for r in rows_sb if r["cls_key"] == ok),
                         "n_candidates": len(keys), "metrics": metrics})
        print(f"[U1 {d}] regret@3 pooled {np.mean([r['metrics']['pooled']['regret3'] for r in inst if r['dataset'] == d]):.4f} "
              f"hgb {np.mean([r['metrics']['hgb']['regret3'] for r in inst if r['dataset'] == d]):.4f}", flush=True)
    return {"per_instance": inst, "summary": summarize_u1(inst), "tests": tests_u1(inst),
            "meta": {"folds": folds, "n_datasets": len(datasets), "shuffle_donors": donors, "seed": seed}}


def _block(rows: Sequence[dict]) -> dict:
    arms = list(rows[0]["metrics"]) if rows else []
    out = {"n": len(rows)}
    for a in arms:
        ms = [r["metrics"][a] for r in rows]
        out[a] = {"oracle_in_top": float(np.mean([m["oracle_in_top"] for m in ms])),
                  "regret1": float(np.mean([m["regret1"] for m in ms])), "regret3": float(np.mean([m["regret3"] for m in ms])),
                  "regret_top": float(np.mean([m["regret_top"] for m in ms])), "rho": float(np.mean([m["rho"] for m in ms])),
                  "pair_acc": float(np.mean([m["pair_acc"] for m in ms if m["pair_acc"] is not None]))}
    return out


def summarize_u1(inst: Sequence[dict]) -> dict:
    out = {"pooled_all": _block(inst)}
    for b in BUDGETS:
        out[f"pooled_{b}"] = _block([r for r in inst if int(r["budget"]) == b])
    return out


def _wil(inst: Sequence[dict], a: str, b: str, field: str = "regret3") -> dict:
    x = np.array([r["metrics"][a][field] for r in inst], dtype=float)
    y = np.array([r["metrics"][b][field] for r in inst], dtype=float)
    return {"n": len(inst), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "p": float(_g36.paired_wilcoxon_less(x, y)), "n_better": int(np.sum(x < y)), "n_worse": int(np.sum(x > y))}


def tests_u1(inst: Sequence[dict]) -> dict:
    shuf = sorted(k for k in inst[0]["metrics"] if k.startswith("hgb_shuffle_"))
    inst2 = []
    for r in inst:
        m = dict(r["metrics"])
        m["shuffle_mean"] = {"regret3": float(np.mean([r["metrics"][s]["regret3"] for s in shuf])),
                             "regret1": float(np.mean([r["metrics"][s]["regret1"] for s in shuf]))}
        inst2.append({**r, "metrics": m})
    return {"hgb_vs_pooled": _wil(inst2, "hgb", "pooled"), "hgb_vs_shuffle_mean": _wil(inst2, "hgb", "shuffle_mean"),
            "ridge_vs_pooled": _wil(inst2, "ridge_int", "pooled"), "twin_vs_pooled": _wil(inst2, "twin_knn", "pooled"),
            "hgb_vs_pooled_r1": _wil(inst2, "hgb", "pooled", "regret1")}


def gate_u1(tests: Mapping[str, Mapping], *, alpha: float = ALPHA) -> dict:
    a, b = tests["hgb_vs_pooled"], tests["hgb_vs_shuffle_mean"]
    return {"vs_pooled": a, "vs_shuffle": b,
            "pass": bool(a["p"] < alpha and a["mean_a"] < a["mean_b"] and b["p"] < alpha and b["mean_a"] < b["mean_b"])}


def verdict_code(g: Mapping[str, Mapping]) -> str:
    if not g["U1"]["pass"]:
        return "UCR-U1-DEAD"
    if g.get("A1", {}).get("pass") and g.get("A2", {}).get("pass"):
        return "UCR-TF-WIN"
    return "UCR-U1-ONLY"


# ---------------------------------------------------------------- φ 计算
def load_jsonl(path: Path) -> list[dict]:
    from scripts.e_ablation_common import read_jsonl
    return read_jsonl(path) if Path(path).exists() else []


def phi_from_arrays(Xtr: np.ndarray, ytr: np.ndarray, *, base_mean: float, base_band: float) -> dict:
    T, C = int(Xtr.shape[1]), int(Xtr.shape[2])
    return {**_g40.phi_meta(T, C, int(len(np.unique(ytr))), int(len(ytr))), **_g40.phi_base(base_mean, base_band),
            **_g40.signal_stats(Xtr), **decomp_strength(Xtr)}


def cmd_phi(rprime: Path, outdir: Path) -> None:
    """φ_ucr（tasks.json 的每个整数据集任务）、φ_uea（45 切片：40 号 12 维 ⊕ 分解 2 维）、φ_official（6 靶）。"""
    from autocls.data.task_slicer import materialize_task
    from autocls.labeling import run_seed_for, split_task_train_eval
    _g41 = importlib.import_module("scripts.41_ucr_rprime")
    outdir.mkdir(parents=True, exist_ok=True)
    # --- UCR
    tasks = json.loads((rprime / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    base_rows = load_jsonl(rprime / "baseline_cache.jsonl")
    base: dict[str, list[dict]] = defaultdict(list)
    for r in base_rows:
        if r.get("r_prime") is not None:
            base[r["task_id"]].append(r)
    reg = _g41.registry()
    path = outdir / "phi_ucr.jsonl"
    done = {r["task_id"] for r in load_jsonl(path)}
    with open(path, "a", encoding="utf-8") as fh:
        for tid, spec in tasks.items():
            if tid in done:
                continue
            if len(base.get(tid, [])) < 2:
                print(f"[phi] {tid}: 基线不足，跳过", flush=True)
                continue
            X, y = _g41.load_dataset(spec["dataset"], reg)
            t = _g41.task_from_json(spec)
            Xt, yt = materialize_task(t, X, y)
            tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(tid, "train_eval_split"))
            vals = [float(r["r_prime"]) for r in base[tid]]
            band = 2.0 * max(float(np.std(vals, ddof=1)), 1.0 / max(int(base[tid][0].get("n_eval") or 1), 1))
            row = {"task_id": tid, "dataset": spec["dataset"], **phi_from_arrays(Xt[tr], yt[tr], base_mean=float(np.mean(vals)), base_band=band)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[phi] {tid}: " + " ".join(f"{c}={row[c]:.3f}" for c in ("log10_T", "base_mean", "spec_conc", "trend_strength", "season_strength")), flush=True)
    # --- UEA 45 切片：40 号 12 维 ⊕ 分解 2 维
    uea_path = outdir / "phi_uea.jsonl"
    if not uea_path.exists():
        phi12 = _g40.load_phi(_g40.OUT / "task_features.jsonl")
        ds = pq.load_pq_datasets(pq.PQ_DATASETS)
        with open(uea_path, "w", encoding="utf-8") as fh:
            for name, d in ds.items():
                for t in d.slices:
                    Xt, yt = materialize_task(t, d.X, d.y)
                    tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(t.task_id, "train_eval_split"))
                    row = {**phi12[t.task_id], **decomp_strength(Xt[tr])}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"[phi uea] {t.task_id}: trend={row['trend_strength']:.3f} season={row['season_strength']:.3f}", flush=True)
    # --- 官方口径 6 靶
    off_path = outdir / "phi_official.json"
    if not off_path.exists():
        from autocls.data.uea import load_uea_pair
        pa_rows = load_jsonl(_g40.PA_RUNS)
        out = {}
        for d in _g40.OFFICIAL_TARGETS:
            base12 = _g40.official_phi(d, pa_rows)
            Xtr, _ytr, _a, _b = load_uea_pair(d, scaler="train")
            out[d] = {**base12, **decomp_strength(Xtr)}
            print(f"[phi official] {d}: trend={out[d]['trend_strength']:.3f} season={out[d]['season_strength']:.3f}", flush=True)
        off_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- U2：主管线 7 折
def _phi_map(rows: Sequence[dict]) -> dict[str, dict]:
    return {r["task_id"]: r for r in rows}


def fold_uea(target: str, cat_z: Sequence[dict], phi_uea: Mapping[str, Mapping], ucr_rows: Sequence[dict],
             phi_ucr: Mapping[str, Mapping], frozen: UcrModule, delta: Mapping, *, seed: int = 0,
             n_shuffle: int = N_SHUFFLE) -> tuple[list[dict], dict]:
    """臂：hybrid / s0z / ucr_direct / ucr_dev（hybrid + dev）/ ucr_dev_z（s0z + dev）/ ucr_direct_shuffle×n / ucr_plus_source_direct / ucr_plus_source_dev_z。"""
    _space, enc_s, _enc_b = _space_enc()
    train = [r for r in cat_z if r["dataset"] != target]
    test = [r for r in cat_z if r["dataset"] == target]
    prior, blind = _g40.fit_s0(train, enc_s, seed=seed)
    # UCR ⊕ 源池（训练折 7 个 UEA 数据集的切片行，φ 用 phi_uea）
    plus = UcrModule("hgb", seed=seed).fit(list(ucr_rows) + list(train), {**phi_ucr, **phi_uea})
    shuffles = [_g40.derange_by_dataset(phi_uea, seed=1000 * seed + 17 * i + 1) for i in range(n_shuffle)]
    twin = frozen.twin_phi()
    out = []
    for s in sorted({r["task_id"] for r in test}):
        rows_s = [r for r in test if r["task_id"] == s]
        traj, _ = _g36.trajectories(rows_s)
        phi_t = phi_vec(phi_uea[s])
        q = (phi_t - frozen.mu) / frozen.sd
        d_twin = sorted(float(np.sqrt(np.sum((v - q) ** 2))) for v in twin.values())[0]
        for b in BUDGETS:
            rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
            keys = [r["cls_key"] for r in rows_sb]
            truth = {k: float(traj[k][b]) for k in keys}
            stages = [x for x in BUDGETS if x <= b]
            s0z = _g40.s0_scores(rows_sb, prior, blind, enc_s)
            hyb = _g40.with_bias(s0z, rows_sb)
            direct = dict(zip(keys, frozen.predict_with_phi(rows_sb, phi_t).tolist()))
            dev = dict(zip(keys, frozen.dev_scores(rows_sb, phi_t).tolist()))
            pdirect = dict(zip(keys, plus.predict_with_phi(rows_sb, phi_t).tolist()))
            pdev = dict(zip(keys, plus.dev_scores(rows_sb, phi_t).tolist()))
            sc = {"hybrid": hyb, "s0z": dict(s0z), "ucr_direct": direct,
                  "ucr_dev": {k: hyb[k] + dev[k] for k in keys}, "ucr_dev_z": {k: s0z[k] + dev[k] for k in keys},
                  "ucr_plus_source_direct": pdirect, "ucr_plus_source_dev_z": {k: s0z[k] + pdev[k] for k in keys}}
            for i, shp in enumerate(shuffles):
                sc[f"ucr_direct_shuffle_{i}"] = dict(zip(keys, frozen.predict_with_phi(rows_sb, phi_vec(shp[s])).tolist()))
            dl = float(delta[(target, b)])
            metrics = {}
            for a, v in sc.items():
                m = _g40.list_metrics(keys, truth, traj, stages, _g36.rank_by(v, keys))
                m["pair_acc"] = _g37.pair_accuracy(v, truth, dl)[0]
                metrics[a] = m
            ok = _g40.oracle_key(truth)
            origin = {r["cls_key"]: r["origin"] for r in rows_sb}
            out.append({"fold": target, "dataset": target, "slice": s, "budget": b, "oracle": ok,
                        "oracle_origin": origin[ok], "twin_distance": d_twin, "metrics": metrics})
    return out, {"twin_distance": {s: r["twin_distance"] for r in out for s in [r["slice"]]}}


def gates_u2(inst: Sequence[dict], *, main: str = U2_MAIN, alpha: float = ALPHA) -> dict:
    r290 = [r for r in inst if int(r["budget"]) == 290]
    top_main = sum(bool(r["metrics"][main]["oracle_in_top"]) for r in r290)
    top_hyb = sum(bool(r["metrics"]["hybrid"]["oracle_in_top"]) for r in r290)
    w = _wil(inst, main, "hybrid")
    A1 = {"in_top_main_290": top_main, "in_top_hybrid_290": top_hyb, "need": top_hyb + _g40.A1_TOP_GAIN, **{f"r3_{k}": v for k, v in w.items()},
          "pass": bool(top_main >= top_hyb + _g40.A1_TOP_GAIN and w["p"] < alpha)}
    shuf = sorted(k for k in inst[0]["metrics"] if k.startswith("ucr_direct_shuffle_"))
    gain_real = w["mean_b"] - w["mean_a"]
    gain_shuf = [w["mean_b"] - float(np.mean([r["metrics"][s]["regret3"] for r in inst])) for s in shuf]
    w2 = _wil(inst, main, "s0z")
    A2 = {"vs_s0z": w2, "gain_real": gain_real, "gain_shuffle": gain_shuf, "gain_shuffle_mean": float(np.mean(gain_shuf)),
          "pass": bool(w2["p"] < alpha and gain_real >= 2 * float(np.mean(gain_shuf)))}
    return {"A1": A1, "A2": A2}


# ---------------------------------------------------------------- U3：官方口径（描述性）
def replay_official_module(d: str, cand: Mapping, s1_rows: Sequence[dict], phi_d: Mapping, frozen: UcrModule) -> dict:
    traj = _g40.trajectory_from_rows(s1_rows)
    budgets = sorted({b for v in traj.values() for b in v})
    bf = budgets[-1]
    keys = sorted(k for k in traj if all(b in traj[k] for b in budgets))
    cls_of = {e["cls_key"]: e["cls"] for e in list(cand["shared"]) + [cand["ggs"]] + list(cand.get("nominated", []))}
    prior = {k: float(v) for k, v in cand["prior"].items()}
    nom = {e["cls_key"]: float(e["pred_z"]) for e in cand.get("nominated", [])}
    s0 = {k: (prior[k] if k in prior else nom[k]) for k in keys}
    hybrid = {k: s0[k] + (_g40.HYBRID_BIAS if k in prior else 0.0) for k in keys}
    truth = {k: float(traj[k][bf]) for k in keys}
    pseudo = [{"task_id": d, "dataset": d, "cls_key": k, "cls": cls_of[k], "iters": max(BUDGETS),
               "origin": ORIGIN_SHARED if k in prior else "random"} for k in keys]
    direct = dict(zip(keys, frozen.predict_with_phi(pseudo, phi_d).tolist()))
    dev = dict(zip(keys, frozen.dev_scores(pseudo, phi_d).tolist()))
    sc = {"hybrid": hybrid, "s0z": dict(s0), "ucr_direct": direct,
          "ucr_dev": {k: hybrid[k] + dev[k] for k in keys}, "ucr_dev_z": {k: s0[k] + dev[k] for k in keys}}
    ggs_key = cand["ggs"]["cls_key"]
    out = {}
    for a, v in sc.items():
        rk = _g36.rank_by(v, keys)
        m = _g40.list_metrics(keys, truth, traj, budgets, rk)
        m["acc"] = float(truth[m["sh_pick"]]) if m["sh_pick"] else None
        m["top6"] = rk[:6]
        m["ggs_in_top6"] = ggs_key in rk[:6]
        m["ggs_rank"] = rk.index(ggs_key) + 1
        out[a] = m
    return {"dataset": d, "budgets": budgets, "oracle": _g40.oracle_key(truth), "oracle_acc": max(truth.values()),
            "ggs_key": ggs_key, "ggs_acc": truth.get(ggs_key), "arms": out}


# ---------------------------------------------------------------- 子命令
def cmd_lodo(rprime: Path, outdir: Path, *, seed: int, n_shuffle: int) -> dict:
    rows = [r for r in load_jsonl(rprime / "rprime_cache.jsonl") if r.get("error") is None and r.get("r_prime") is not None]
    phi = _phi_map(load_jsonl(outdir / "phi_ucr.jsonl"))
    have = {r["task_id"] for r in rows} & set(phi)
    rows = [r for r in rows if r["task_id"] in have]
    res = lodo_ucr(rows, phi, n_shuffle=n_shuffle, seed=seed)
    res["gate"] = gate_u1(res["tests"])
    (outdir / "u1_lodo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    s = res["summary"]["pooled_all"]
    print(f"[U1] n_datasets={res['meta']['n_datasets']} regret@3 pooled {s['pooled']['regret3']:.4f} hgb {s['hgb']['regret3']:.4f} "
          f"ridge {s['ridge_int']['regret3']:.4f} twin {s['twin_knn']['regret3']:.4f}；gate {'PASS' if res['gate']['pass'] else 'FAIL'}", flush=True)
    return res


def cmd_fit(rprime: Path, outdir: Path, *, seed: int) -> UcrModule:
    rows = [r for r in load_jsonl(rprime / "rprime_cache.jsonl") if r.get("error") is None and r.get("r_prime") is not None]
    phi = _phi_map(load_jsonl(outdir / "phi_ucr.jsonl"))
    rows = [r for r in rows if r["task_id"] in phi]
    m = UcrModule("hgb", seed=seed).fit(rows, phi)
    m.save(outdir / "module.pkl")
    print(f"[fit] 冻结 module.pkl：{m.n_datasets} 数据集，{len(rows)} 行", flush=True)
    return m


def cmd_uea(rprime: Path, outdir: Path, *, seed: int, n_shuffle: int, folds: Sequence[str]) -> dict:
    cat, ggs, probe, _rk = _g36.load_inputs(pq.OUT)
    cat_z = _g36.zscore_within(cat)
    delta = pq.load_delta_calibration(pq.OUT / "delta_calibration.json")
    phi_uea = _phi_map(load_jsonl(outdir / "phi_uea.jsonl"))
    phi_ucr = _phi_map(load_jsonl(outdir / "phi_ucr.jsonl"))
    ucr_rows = [r for r in load_jsonl(rprime / "rprime_cache.jsonl") if r.get("error") is None and r.get("r_prime") is not None
                and r["task_id"] in phi_ucr]
    frozen = UcrModule.load(outdir / "module.pkl")
    inst, meta = [], {}
    for t in folds:
        rows, m = fold_uea(t, cat_z, phi_uea, ucr_rows, phi_ucr, frozen, delta, seed=seed, n_shuffle=n_shuffle)
        inst += rows
        meta[t] = m
        r290 = [r for r in rows if int(r["budget"]) == 290]
        print(f"[U2 {t}] 入前6 @290 hybrid {sum(r['metrics']['hybrid']['oracle_in_top'] for r in r290)}/6 "
              f"ucr_direct {sum(r['metrics']['ucr_direct']['oracle_in_top'] for r in r290)}/6 "
              f"ucr_dev_z {sum(r['metrics']['ucr_dev_z']['oracle_in_top'] for r in r290)}/6；twin 距离 "
              f"{np.mean([r['twin_distance'] for r in rows]):.2f}", flush=True)
    res = {"per_instance": inst, "meta": meta, "gates": gates_u2(inst), "summary": _g40.summarize_a(inst)}
    (outdir / "u2_uea.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def cmd_official(outdir: Path) -> dict:
    frozen = UcrModule.load(outdir / "module.pkl")
    phi_off = json.loads((outdir / "phi_official.json").read_text(encoding="utf-8"))
    cands = json.loads((_g40.DEPLOY / "candidates.json").read_text(encoding="utf-8"))
    runs = [r for r in load_jsonl(_g40.DEPLOY / "runs.jsonl") if r.get("stage") == "s1_catalog" and int(r.get("seed", 0)) == 0
            and r.get("error") is None]
    out = {}
    for d in _g40.OFFICIAL_TARGETS:
        if d not in cands:
            continue
        out[d] = replay_official_module(d, cands[d], [r for r in runs if r["dataset"] == d], phi_off[d], frozen)
        a = out[d]["arms"]
        print(f"[U3 {d}] oracle {out[d]['oracle_acc']:.3f}；SH hybrid {a['hybrid']['acc']:.3f} direct {a['ucr_direct']['acc']:.3f} "
              f"dev_z {a['ucr_dev_z']['acc']:.3f}；GGS 名次 hybrid {a['hybrid']['ggs_rank']} direct {a['ucr_direct']['ggs_rank']}", flush=True)
    (outdir / "u3_official.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def cmd_analyze(outdir: Path) -> dict:
    u1 = json.loads((outdir / "u1_lodo.json").read_text(encoding="utf-8"))
    g = {"U1": u1["gate"]}
    u2p, u3p = outdir / "u2_uea.json", outdir / "u3_official.json"
    u2 = json.loads(u2p.read_text(encoding="utf-8")) if u2p.exists() else None
    u3 = json.loads(u3p.read_text(encoding="utf-8")) if u3p.exists() else None
    if u2:
        g.update(u2["gates"])
    analysis = {"protocol": "results/ucr_module/PROTOCOL.md", "gates": g, "verdict": verdict_code(g),
                "u1_summary": u1["summary"], "u1_tests": u1["tests"], "u1_meta": {k: v for k, v in u1["meta"].items() if k != "shuffle_donors"},
                "u2_summary": u2["summary"] if u2 else None, "u2_twin": {t: m["twin_distance"] for t, m in u2["meta"].items()} if u2 else None,
                "u3": u3}
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(analysis), encoding="utf-8")
    print(markdown_tables(analysis))
    return analysis


def markdown_tables(a: Mapping) -> str:
    L = [f"# WP7-UCR 结果表（读数码 **{a['verdict']}**）", "", "## U1 语料内留一（regret@3 / regret@1 / 入前 6 率 / 配对 acc）", ""]
    for k in ("pooled_all", "pooled_60", "pooled_120", "pooled_290"):
        s = a["u1_summary"][k]
        L.append(f"### {k}（n={s['n']}）")
        L.append("| 臂 | regret@3 | regret@1 | 入前6 | pair_acc | ρ |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for arm, v in s.items():
            if arm == "n":
                continue
            L.append(f"| {arm} | {v['regret3']:.4f} | {v['regret1']:.4f} | {v['oracle_in_top']:.3f} | {v['pair_acc']:.3f} | {v['rho']:.3f} |")
        L.append("")
    t = a["u1_tests"]
    L.append("配对 Wilcoxon（regret@3）：" + "; ".join(f"{k}: {v['mean_a']:.4f} vs {v['mean_b']:.4f} p={v['p']:.2e} ({v['n_better']}/{v['n_worse']})" for k, v in t.items()))
    L.append(f"\n**U1 门**：{'✓' if a['gates']['U1']['pass'] else '✗'}")
    if a.get("u2_summary"):
        s = a["u2_summary"]
        L += ["", "## U2 主管线 7 折（pooled 126 / @290 42）", "", "| 臂 | 入前6率 | 入前6 @290 | regret@1 | regret@3 | SH6-3-3 @290 |", "|---|---:|---:|---:|---:|---:|"]
        n290 = s["by_budget"]["290"]["n"]
        for arm in s["pooled"]:
            if arm == "n":
                continue
            p, q = s["pooled"][arm], s["by_budget"]["290"][arm]
            L.append(f"| {arm} | {p['oracle_in_top']:.3f} | {q['n_in_top']}/{n290} | {p['regret1']:.4f} | {p['regret3']:.4f} | {q['sh_regret']:.4f} |")
        g = a["gates"]
        L.append(f"\nA1 {'✓' if g['A1']['pass'] else '✗'}（入前6 {g['A1']['in_top_main_290']} vs 需 {g['A1']['need']}；r3 p={g['A1']['r3_p']:.2e}）；"
                 f"A2 {'✓' if g['A2']['pass'] else '✗'}（vs s0z p={g['A2']['vs_s0z']['p']:.2e}；gain {g['A2']['gain_real']:.4f} vs shuffle {g['A2']['gain_shuffle_mean']:.4f}）")
        L.append("孪生距离（靶到最近 UCR 数据集，标准化 φ）：" + json.dumps({t: {s_: round(v, 2) for s_, v in m.items()} for t, m in a["u2_twin"].items()}, ensure_ascii=False))
    if a.get("u3"):
        L += ["", "## U3 官方口径（seed 0 acc@100%）", "", "| 靶 | oracle | hybrid | ucr_direct | ucr_dev_z | GGS 名次 hyb/direct/dev_z |", "|---|---:|---:|---:|---:|---|"]
        for d, t in a["u3"].items():
            ar = t["arms"]
            L.append(f"| {d} | {t['oracle_acc']:.3f} | {ar['hybrid']['acc']:.3f} | {ar['ucr_direct']['acc']:.3f} | {ar['ucr_dev_z']['acc']:.3f} | "
                     f"{ar['hybrid']['ggs_rank']}/{ar['ucr_direct']['ggs_rank']}/{ar['ucr_dev_z']['ggs_rank']} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("phi", "lodo", "fit", "uea", "official", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--rprime", default=str(RPRIME))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--folds", nargs="*", default=list(pq.FOLD_TARGETS))
    args = ap.parse_args(argv)
    rprime, outdir = Path(args.rprime), Path(args.outdir)
    ran = False
    if args.phi:
        cmd_phi(rprime, outdir); ran = True
    if args.lodo:
        cmd_lodo(rprime, outdir, seed=args.seed, n_shuffle=args.n_shuffle); ran = True
    if args.fit:
        cmd_fit(rprime, outdir, seed=args.seed); ran = True
    if args.uea:
        cmd_uea(rprime, outdir, seed=args.seed, n_shuffle=args.n_shuffle, folds=args.folds); ran = True
    if args.official:
        cmd_official(outdir); ran = True
    if args.analyze:
        cmd_analyze(outdir); ran = True
    if not ran:
        ap.print_help()


if __name__ == "__main__":
    main()
