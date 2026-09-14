"""WP16-TCLSC-L1：T-CLSC 的 CLS 支路——编码 × 目标函数 分解（协议 results/tclsc_l1/PROTOCOL.md）。

问题：同一批 δ 标签上 T-CLSC main 0.735 vs 字段代理 hgb_binned 0.780（WP2）。把差距拆成两个可单独改的因子，
MLP 结构固定：
- 编码：scalar 32 维（8 个 ratio + temperature 是 1 维标量，MLP 得从一维里学非单调档位效应）→ binned 116 维
  按档 one-hot（进 `cls_mlp` 第一层 Linear ≡ AutoCTS++ 每 op 一张 `nn.Embedding` 表求和）；
- 目标函数：δ 存活配对 BCE（每折 ≈ 800 对、只学方向）→ within-(task, budget) z 回归（2340 行）。

子命令（都零新 R'，`results/mx3_pq/` 只读）：
  --c0        硬门：pod 版 rankings.json["main"] 复算 == 0.735 / 0.074；ref 三臂与 WP2 analysis.json 逐位一致
  --prepare   变体目录 results/tclsc_l1/{scalar,binned}/ 只复制输入（缓存 / δ / 目录），幂等
  --train     本地 `33 --train --arms cls_only --encoder-mode <mode>`（3 种子）
  --rank      变体目录的模型 → rankings.json（34 号 rank_candidates 多数票，向量按 meta.encoder_mode 编码）
  --analyze   L1c 回归臂拟合 + 全臂同口径指标 + 门 G1–G3′ + 读数码 + 预测对账 → analysis.json / tables.md

数据流：rprime_cache → zscore_within（36 号）→ 指标 pair_accuracy / regret（37 号纯函数逐位复用）。
任务无关（铁律 7）：本文件不 import task_types；方向由标签侧（33 的 is_better 注入 / 36 的 z）给定。
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.space.cls_encoder import CLSEncoder, MODES  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g34 = importlib.import_module("scripts.34_pq_search")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g37 = importlib.import_module("scripts.37_surrogate_loo")
zscore_within = _g36.zscore_within
features = _g36.features
rank_by = _g36.rank_by
expected_max_of_k = _g36.expected_max_of_k
paired_wilcoxon_less = _g36.paired_wilcoxon_less
pair_accuracy = _g37.pair_accuracy
scores_from_ranking = _g37.scores_from_ranking

SOURCE = pq.OUT                                   # results/mx3_pq（只读）
OUT = ROOT / "results" / "tclsc_l1"
WP2_ANALYSIS = ROOT / "results" / "mx4_surrogate" / "analysis.json"
BUDGETS = pq.BUDGETS
TOP_GOOD = _g37.TOP_GOOD
K_TOP = _g37.K_TOP
VARIANT_INPUTS = ("rprime_cache.jsonl", "delta_calibration.json", "cls_catalog.json")
VARIANT_OPTIONAL = ("tasks.json",)
ENCODINGS = ("scalar", "binned")
POD_ARMS = ("main", "cls_only")                   # rankings.json 里本协议要读的臂

# 门（协议 §4，先于数据）
C0_ACC, C0_REGRET1, C0_TOL = 0.735, 0.074, 1e-3
G1_MIN, G1_FOLDS = 0.020, 5
G2_MIN, G2_REF = 0.020, 0.735
G3_MIN = 0.765
G3P_MIN = 0.020
LOCAL_BAND = 0.02
# L1c 回归臂固定超参（协议 §3，不调）
MLPZ_HIDDEN, MLPZ_EPOCHS, MLPZ_LR, MLPZ_WD = 128, 300, 1e-3, 1e-4
MLPZ_SEEDS = (0, 1, 2)

_SPACE: Optional[CLSSpace] = None
_ENC: dict[str, CLSEncoder] = {}


def space() -> CLSSpace:
    global _SPACE
    if _SPACE is None:
        _SPACE = CLSSpace.from_yaml()
    return _SPACE


def encoder(mode: str) -> CLSEncoder:
    if mode not in _ENC:
        _ENC[mode] = CLSEncoder(space(), mode=mode)
    return _ENC[mode]


# ---------------------------------------------------------------- meta → 编码器
def encoder_for_meta(meta: Mapping, spc: Optional[CLSSpace] = None) -> CLSEncoder:
    """打分用的 CLS 编码器：按 meta["encoder_mode"] 选，与 meta["cls_dim"] 互校。

    老存档（pod 版）没有 encoder_mode 键：只有 cls_dim 等于 scalar 维数时才认作 scalar；
    116 维而无键不猜（宁可炸——喂错编码模型不报错、排名全错）。
    """
    spc = spc or space()
    dim = int(meta["cls_dim"])
    mode = meta.get("encoder_mode")
    if mode is None:
        if dim != CLSEncoder(spc).dim:
            raise ValueError(f"meta 无 encoder_mode 且 cls_dim={dim} ≠ scalar 维数 {CLSEncoder(spc).dim}，不猜")
        mode = "scalar"
    if mode not in MODES:
        raise ValueError(f"未知 encoder_mode {mode!r}（合法 {MODES}）")
    enc = CLSEncoder(spc, mode=mode)
    if enc.dim != dim:
        raise ValueError(f"encoder_mode={mode} 应为 {enc.dim} 维，meta cls_dim={dim}")
    return enc


# ---------------------------------------------------------------- 指标（WP2 同口径）
def instance_metrics(scores: Mapping[str, Mapping[str, float]], truth: Mapping[str, float],
                     delta: float) -> tuple[dict, dict]:
    """一个 (切片, 预算) 实例：{臂: 得分表} → (metrics, regret)，与 37 号 fold_eval 的单实例逐位同式。"""
    keys = sorted(truth)
    vals = list(truth.values())
    oracle = max(vals)
    metrics, regret = {}, {"oracle": 0.0, "random1": oracle - float(np.mean(vals)),
                           "random3": oracle - float(expected_max_of_k(vals, K_TOP))}
    for arm, sc in scores.items():
        acc, n = pair_accuracy(sc, truth, delta)
        acc_top, n_top = pair_accuracy(sc, truth, delta, top_k=TOP_GOOD)
        ranked = rank_by(sc, keys)
        metrics[arm] = {"pair_acc": acc, "n_pairs": n, "acc_top10": acc_top, "n_top10_pairs": n_top,
                        "rho": float(spearmanr([sc[k] for k in keys], [truth[k] for k in keys]).correlation)}
        regret[arm + "1"] = oracle - truth[ranked[0]]
        regret[arm + "3"] = oracle - max(truth[k] for k in ranked[:K_TOP])
    return metrics, regret


def _instances(rows: Sequence[dict]) -> dict[tuple[str, str, int], list[dict]]:
    g: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["task_id"], int(r["iters"]))].append(r)
    return g


def eval_rankings(rankings: Mapping, rows: Sequence[dict], delta: Mapping, *,
                  arms: Sequence[str], rename: Optional[Mapping[str, str]] = None) -> list[dict]:
    """rankings.json 结构 {靶: {预算: {切片: {臂: [key…]}}}} → WP2 形状的实例行（rank→score 口径）。

    某实例缺某臂 → 该臂不进 metrics（不伪造）；全臂缺席的实例整行跳过。
    """
    rename = dict(rename or {})
    out = []
    for (ds, s, b), rs in sorted(_instances(rows).items()):
        per = rankings.get(ds, {}).get(str(b), {}).get(s, {})
        scores = {rename.get(a, a): scores_from_ranking(per[a]) for a in arms if a in per}
        if not scores:
            continue
        truth = {r["cls_key"]: float(r["r_prime"]) for r in rs}
        dl = float(delta[(ds, int(b))])
        m, reg = instance_metrics(scores, truth, dl)
        out.append({"fold": ds, "dataset": ds, "slice": s, "budget": int(b), "delta": dl,
                    "metrics": m, "regret": reg})
    return out


def pooled_acc(rows: Sequence[dict], arm: str) -> float:
    v = [r["metrics"][arm]["pair_acc"] for r in rows
         if arm in r["metrics"] and r["metrics"][arm]["pair_acc"] is not None]
    return float(np.mean(v)) if v else float("nan")


def c0_check(rows: Sequence[dict], *, arm: str = "main", expected_acc: float = C0_ACC,
             expected_regret1: float = C0_REGRET1, tol: float = C0_TOL) -> dict:
    """硬门：pod 版 main 臂在本口径下的 pooled 配对 acc / regret@1 复现 WP2 数字。"""
    acc = pooled_acc(rows, arm)
    reg = float(np.mean([r["regret"][arm + "1"] for r in rows if arm + "1" in r["regret"]]))
    ok = abs(acc - expected_acc) <= tol and abs(reg - expected_regret1) <= tol
    return {"arm": arm, "pair_acc": acc, "regret1": reg, "expected": [expected_acc, expected_regret1],
            "tol": tol, "pass": bool(ok)}


# ---------------------------------------------------------------- L1c：MLP 回归臂
class MLPZ:
    """与 `TCLSC.cls_mlp` 同形的主干（Linear→ReLU→Linear）+ ReLU→Linear(1)，MSE 回归 within-task z。

    只问"同一个 MLP 换目标函数会怎样"，超参按协议 §3 写死不调；CPU 单进程下按种子逐位确定。
    """

    def __init__(self, in_dim: int, hidden: int = MLPZ_HIDDEN, *, seed: int = 0,
                 epochs: int = MLPZ_EPOCHS, lr: float = MLPZ_LR, weight_decay: float = MLPZ_WD):
        self.seed, self.epochs, self.lr, self.weight_decay = int(seed), int(epochs), float(lr), float(weight_decay)
        torch.manual_seed(self.seed)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 1)
        self._params = list(self.trunk.parameters()) + list(self.head.parameters())

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.trunk(X))).squeeze(1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPZ":
        Xt = torch.as_tensor(np.asarray(X, np.float32))
        yt = torch.as_tensor(np.asarray(y, np.float32))
        opt = torch.optim.Adam(self._params, lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.mean((self._forward(Xt) - yt) ** 2)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._forward(torch.as_tensor(np.asarray(X, np.float32))).numpy().astype(np.float64)


def lodo_split(rows: Sequence[dict], target: str) -> tuple[list[dict], list[dict]]:
    """留一数据集：训练侧绝不含靶（铁律 6）。"""
    train = [r for r in rows if r["dataset"] != target]
    test = [r for r in rows if r["dataset"] == target]
    return train, test


def fit_mlpz_arms(train: Sequence[dict], enc: Mapping[str, CLSEncoder], *,
                  seeds: Sequence[int] = MLPZ_SEEDS, epochs: int = MLPZ_EPOCHS) -> dict[str, list[MLPZ]]:
    y = np.array([r["z"] for r in train])
    out = {}
    for mode in ENCODINGS:
        X = features(train, enc[mode], max(BUDGETS))
        out[f"mlp_z_{mode}"] = [MLPZ(X.shape[1], seed=s, epochs=epochs).fit(X, y) for s in seeds]
    return out


def fold_eval_mlpz(target: str, cat_z: Sequence[dict], delta: Mapping, enc: Mapping[str, CLSEncoder], *,
                   seeds: Sequence[int] = MLPZ_SEEDS, epochs: int = MLPZ_EPOCHS) -> list[dict]:
    """一折：靶外拟合两臂 → 靶上每个 (切片, 预算) 实例的 WP2 形状行（种子预测取均值）。"""
    train, test = lodo_split(cat_z, target)
    models = fit_mlpz_arms(train, enc, seeds=seeds, epochs=epochs)
    out = []
    for (ds, s, b), rs in sorted(_instances(test).items()):
        rs = sorted(rs, key=lambda r: r["cls_key"])
        keys = [r["cls_key"] for r in rs]
        truth = {r["cls_key"]: float(r["r_prime"]) for r in rs}
        scores = {}
        for arm, ms in models.items():
            mode = arm.rsplit("_", 1)[1]
            X = features(rs, enc[mode], max(BUDGETS))
            pred = np.mean([m.predict(X) for m in ms], axis=0)
            scores[arm] = dict(zip(keys, pred.tolist()))
        dl = float(delta[(ds, int(b))])
        m, reg = instance_metrics(scores, truth, dl)
        out.append({"fold": target, "dataset": ds, "slice": s, "budget": int(b), "delta": dl,
                    "metrics": m, "regret": reg})
    return out


# ---------------------------------------------------------------- 本地模型 → 排名
def rank_local_models(models_by_arm: Mapping[str, Sequence[tuple]], catalog: Mapping[str, Sequence[tuple]],
                      *, budgets: Sequence[int] = BUDGETS, device: str = "cpu",
                      ctx_rows: Optional[Mapping[tuple[str, int], np.ndarray]] = None) -> dict:
    """{臂: [(TCLSC, meta)]} × {切片: [(origin, CLS, key)]} → {预算: {切片: {臂: [key…]}}}。

    与 34 号 cmd_rank 同一聚合（rank_candidates：多种子对称化 → 多数票）；差别只有一条——
    候选向量按**每个模型自己的 meta.encoder_mode** 编码（先全部互校，再打分）。只支持不吃 E 的臂
    （cls_only / cls_b / WP16-L3 的 ctx 臂）；吃 E 的臂走 34 号（需折特征）。
    ctx：缺省为预算 one-hot（cls_b 形状）；`ctx_rows[(切片, 预算)]` 给出时按表取（L3 的数据集⊕预算 token，
    消融时调用方直接传错位后的表），维度与每个模型的 ctx_dim 互校。
    """
    encs: dict[str, list[CLSEncoder]] = {}
    for arm, mm in models_by_arm.items():
        encs[arm] = []
        for model, meta in mm:
            enc = encoder_for_meta(meta)
            in_features = int(model.cls_mlp[0].in_features)
            if in_features != enc.dim:
                raise ValueError(f"[{arm}] 模型第一层 in_features={in_features} ≠ {enc.mode} 编码 {enc.dim} 维")
            if _g34.uses_E(model):
                raise NotImplementedError(f"[{arm}] 吃 E 的臂请走 scripts/34_pq_search.py --rank")
            cd = int(getattr(model, "ctx_dim", 0))
            if cd > 0:
                if ctx_rows is None and cd != len(budgets):
                    raise ValueError(f"[{arm}] 模型 ctx_dim={cd} 不是预算 one-hot 形状，须传 ctx_rows")
                if ctx_rows is not None:
                    bad = [k for k, v in ctx_rows.items() if int(np.asarray(v).shape[0]) != cd]
                    if bad:
                        raise ValueError(f"[{arm}] ctx_rows 维度与模型 ctx_dim={cd} 不符：{bad[:3]}")
            model.eval()
            encs[arm].append(enc)
    vec_cache: dict[tuple[str, str], torch.Tensor] = {}

    def vecs(mode: str, s: str) -> torch.Tensor:
        if (mode, s) not in vec_cache:
            vec_cache[(mode, s)] = torch.as_tensor(
                np.stack([encoder(mode).encode(c) for _o, c, _k in catalog[s]]),
                dtype=torch.float32, device=device)
        return vec_cache[(mode, s)]

    out: dict[str, dict] = {}
    for b in budgets:
        ctx_budget = torch.as_tensor(np.asarray(pq.ctx_vector(int(b), budgets)), dtype=torch.float32, device=device)
        per_b: dict[str, dict] = {}
        for s in sorted(catalog):
            keys = [k for _o, _c, k in catalog[s]]
            if ctx_rows is not None:
                if (s, int(b)) not in ctx_rows:
                    raise KeyError(f"ctx_rows 缺 ({s}, {int(b)})")
                ctx_row = torch.as_tensor(np.asarray(ctx_rows[(s, int(b))], np.float32), device=device)
            else:
                ctx_row = ctx_budget
            per_arm: dict[str, list[str]] = {}
            for arm, mm in models_by_arm.items():
                modes = {e.mode for e in encs[arm]}
                if len(modes) != 1:
                    raise ValueError(f"[{arm}] 同一臂的种子编码模式不一致 {sorted(modes)}")
                mode = next(iter(modes))
                ranked, _S = _g34.rank_candidates([m for m, _ in mm], [None] * len(mm),
                                                  vecs(mode, s), keys, ctx_row=ctx_row)
                per_arm[arm] = ranked
            per_b[s] = per_arm
        out[str(int(b))] = per_b
    return out


# ---------------------------------------------------------------- 变体目录
def prepare_variant_dir(source: Path, dst: Path) -> list[str]:
    """只把输入文件（缓存 / δ / 目录 [+ tasks.json]）复制进变体目录；已存在且同长度则不动（幂等）。"""
    source, dst = Path(source), Path(dst)
    missing = [n for n in VARIANT_INPUTS if not (source / n).exists()]
    if missing:
        raise FileNotFoundError(f"{source} 缺输入 {missing}")
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    names = VARIANT_INPUTS + tuple(x for x in VARIANT_OPTIONAL if (source / x).exists())
    for n in names:
        src, tgt = source / n, dst / n
        if tgt.exists() and tgt.stat().st_size == src.stat().st_size:
            continue
        shutil.copyfile(src, tgt)
        copied.append(n)
    return copied


# ---------------------------------------------------------------- 门 / 读数码
def gates(pooled: Mapping[str, Mapping[str, float]], by_fold: Mapping[str, Mapping[str, float]]) -> dict:
    """G1 编码（cls_only）/ G2 编码带到主臂 / G3 目标函数 / G3′ 回归目标下的编码边际（协议 §4）。"""
    def acc(arm: str) -> Optional[float]:
        v = pooled.get(arm, {}).get("pair_acc")
        return None if v is None or not np.isfinite(v) else float(v)

    def n_better(a: str, b: str, ref: Optional[float] = None) -> Optional[int]:
        n = 0
        for f, d in by_fold.items():
            if a not in d:
                return None
            base = d[b] if ref is None else d.get(b, ref)
            n += int(d[a] > base)
        return n

    g: dict[str, dict] = {}
    sc, bi = acc("cls_only_scalar"), acc("cls_only_binned")
    d1 = None if sc is None or bi is None else bi - sc
    nb1 = n_better("cls_only_binned", "cls_only_scalar")
    g["G1"] = {"delta": d1, "n_folds_better": nb1, "n_folds": len(by_fold), "min_delta": G1_MIN, "min_folds": G1_FOLDS,
               "pass": (None if d1 is None or nb1 is None else bool(d1 >= G1_MIN and nb1 >= G1_FOLDS))}
    mb = acc("main_binned")
    if mb is None:
        g["G2"] = {"delta": None, "n_folds_better": None, "pass": None, "note": "阶段 2 未跑（main_binned 缺席）"}
    else:
        d2 = mb - G2_REF
        nb2 = n_better("main_binned", "main_pod", ref=G2_REF)
        g["G2"] = {"delta": d2, "n_folds_better": nb2, "n_folds": len(by_fold), "ref": G2_REF,
                   "pass": bool(d2 >= G2_MIN and (nb2 or 0) >= G1_FOLDS)}
    zb, zs = acc("mlp_z_binned"), acc("mlp_z_scalar")
    g["G3"] = {"value": zb, "min": G3_MIN, "pass": None if zb is None else bool(zb >= G3_MIN)}
    d3 = None if zb is None or zs is None else zb - zs
    g["G3p"] = {"delta": d3, "min": G3P_MIN, "pass": None if d3 is None else bool(d3 >= G3P_MIN)}
    return g


def verdict_code(g: Mapping[str, Mapping]) -> str:
    g1, g3 = bool(g["G1"]["pass"]), bool(g["G3"]["pass"])
    return {(True, True): "L1-ENC+OBJ", (True, False): "L1-ENC-ONLY",
            (False, True): "L1-OBJ-ONLY", (False, False): "L1-FLAT"}[(g1, g3)]


# ---------------------------------------------------------------- 汇总
def merge_rows(*groups: Sequence[dict]) -> list[dict]:
    """按 (fold, slice, budget) 合并多组实例行的 metrics / regret（各组臂名互不重复）。"""
    merged: dict[tuple[str, str, int], dict] = {}
    for rows in groups:
        for r in rows:
            k = (r["fold"], r["slice"], int(r["budget"]))
            if k not in merged:
                merged[k] = {"fold": r["fold"], "dataset": r["dataset"], "slice": r["slice"],
                             "budget": int(r["budget"]), "delta": r["delta"], "metrics": {}, "regret": {}}
            merged[k]["metrics"].update(r["metrics"])
            merged[k]["regret"].update(r["regret"])
    return [merged[k] for k in sorted(merged)]


def summarize(rows: Sequence[dict]) -> dict:
    arms = sorted({a for r in rows for a in r["metrics"]})

    def block(rs: Sequence[dict], arm: str) -> dict:
        have = [r for r in rs if arm in r["metrics"]]
        accs = [r["metrics"][arm]["pair_acc"] for r in have if r["metrics"][arm]["pair_acc"] is not None]
        tops = [r["metrics"][arm]["acc_top10"] for r in have if r["metrics"][arm]["acc_top10"] is not None]
        return {"pair_acc": float(np.mean(accs)) if accs else float("nan"),
                "acc_top10": float(np.mean(tops)) if tops else float("nan"),
                "rho": float(np.mean([r["metrics"][arm]["rho"] for r in have])) if have else float("nan"),
                "regret1": float(np.mean([r["regret"][arm + "1"] for r in have])) if have else float("nan"),
                "regret3": float(np.mean([r["regret"][arm + "3"] for r in have])) if have else float("nan"),
                "n": len(have)}

    pooled = {arm: block(rows, arm) for arm in arms}
    pooled["random"] = {"regret1": float(np.mean([r["regret"]["random1"] for r in rows])),
                        "regret3": float(np.mean([r["regret"]["random3"] for r in rows]))}
    by_fold = {}
    for f in sorted({r["fold"] for r in rows}):
        rs = [r for r in rows if r["fold"] == f]
        by_fold[f] = {arm: block(rs, arm)["pair_acc"] for arm in arms if any(arm in r["metrics"] for r in rs)}
    by_budget = {str(b): {arm: block([r for r in rows if r["budget"] == b], arm) for arm in arms} for b in BUDGETS}
    return {"arms": arms, "pooled": pooled, "by_fold_pair_acc": by_fold, "by_budget": by_budget}


def paired_test(rows: Sequence[dict], a: str, b: str) -> dict:
    """配对 Wilcoxon（H1：b 的配对 acc 高于 a）与逐实例胜负，只取两臂都有存活对的实例。"""
    xs, ys = [], []
    for r in rows:
        ma, mb = r["metrics"].get(a), r["metrics"].get(b)
        if ma and mb and ma["pair_acc"] is not None and mb["pair_acc"] is not None:
            xs.append(ma["pair_acc"])
            ys.append(mb["pair_acc"])
    x, y = np.array(xs), np.array(ys)
    if not x.size:
        return {"a": a, "b": b, "n": 0, "p": None}
    return {"a": a, "b": b, "n": int(x.size), "p": float(paired_wilcoxon_less(x, y)),
            "wins_b": int(np.sum(y > x)), "losses_b": int(np.sum(y < x)), "ties": int(np.sum(y == x)),
            "mean_delta": float(np.mean(y - x))}


# ---------------------------------------------------------------- I/O
def load_source(source: Path = SOURCE):
    rows = [r for r in read_jsonl(source / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] != "ggs"]
    delta = pq.load_delta_calibration(source / "delta_calibration.json")
    rankings = json.loads((source / "rankings.json").read_text(encoding="utf-8"))
    return rows, delta, rankings


def cmd_c0(source: Path, outdir: Path, folds: Sequence[str]) -> dict:
    rows, delta, rankings = load_source(source)
    cat_z = zscore_within(rows)
    pod_rows = eval_rankings(rankings, cat_z, delta, arms=list(POD_ARMS))
    c0 = c0_check(pod_rows)
    ref = {}
    ref_match = None
    if WP2_ANALYSIS.exists():
        wp2 = json.loads(WP2_ANALYSIS.read_text(encoding="utf-8"))["summary"]["pooled"]
        enc = {m: encoder(m) for m in ENCODINGS}
        inst = []
        for t in folds:
            inst += _g37.fold_eval(t, cat_z, delta, rankings, enc, seed=0)
        s37 = _g37.summarize(inst)["pooled"]
        ref = {arm: {"pair_acc": s37[arm]["pair_acc"], "wp2": wp2[arm]["pair_acc"]}
               for arm in ("ridge_scalar", "ridge_binned", "hgb_scalar", "hgb_binned", "tclsc")}
        ref_match = all(abs(v["pair_acc"] - v["wp2"]) <= 1e-9 for v in ref.values())
    res = {"c0": c0, "ref": ref, "ref_match": ref_match,
           "pass": bool(c0["pass"] and (ref_match is None or ref_match)),
           "cls_only_pod": {"pair_acc": pooled_acc(pod_rows, "cls_only")}}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "c0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"C0 main: pair_acc {c0['pair_acc']:.4f} regret@1 {c0['regret1']:.4f} → {'PASS' if c0['pass'] else 'FAIL'}"
          f" | ref 逐位 {ref_match} | cls_only_pod {res['cls_only_pod']['pair_acc']:.4f}", flush=True)
    return res


def cmd_prepare(source: Path, outdir: Path, modes: Sequence[str]) -> None:
    for m in modes:
        copied = prepare_variant_dir(source, outdir / m)
        print(f"[{m}] {outdir / m}：复制 {copied or '无（已就位）'}", flush=True)


def cmd_train(outdir: Path, modes: Sequence[str], *, seeds: Sequence[int], folds: Optional[Sequence[str]],
              device: str, epochs: Optional[int], resume: bool) -> None:
    m33 = importlib.import_module("scripts.33_pq_comparator_loo")
    for m in modes:
        d = outdir / m
        if not (d / "rprime_cache.jsonl").exists():
            raise FileNotFoundError(f"{d} 未准备：先 --prepare")
        print(f"===== [{m}] 33 --train --arms cls_only --encoder-mode {m} seeds={list(seeds)} =====", flush=True)
        m33.cmd_train(d, arms=["cls_only"], folds=list(folds) if folds else None, seeds=list(seeds),
                      resume=resume, device=device, epochs=epochs, settf_windows=None,
                      encoder_mode=m)


def cmd_rank(source: Path, outdir: Path, modes: Sequence[str], *, folds: Sequence[str], seeds: Sequence[int],
             device: str = "cpu", arms: Sequence[str] = ("cls_only",)) -> None:
    for m in modes:
        d = outdir / m
        rankings: dict[str, dict] = {}
        for target in folds:
            models_by_arm = {}
            for arm in arms:
                paths = [d / "models" / target / f"{arm}_s{s}.pt" for s in seeds]
                have = [p for p in paths if p.exists()]
                if len(have) != len(paths):
                    print(f"⚠️ [{m}/{target}/{arm}] 模型 {len(have)}/{len(paths)}，整臂跳过", flush=True)
                    continue
                models_by_arm[arm] = [_g34.load_pq_model(p, map_location=device, expect_fold=target,
                                                         expect_budgets=BUDGETS) for p in have]
            if not models_by_arm:
                continue
            catalog, _ggs, _k = _g34.load_catalog(source, space(), slices=pq.pq_slice_ids(target))
            cat_t = {s: catalog[s] for s in pq.pq_slice_ids(target) if s in catalog}
            rankings[target] = rank_local_models(models_by_arm, cat_t, budgets=BUDGETS, device=device)
            print(f"[{m}/{target}] 排名 {len(cat_t)} 切片 × {len(BUDGETS)} 预算 × {sorted(models_by_arm)}", flush=True)
        (d / "rankings.json").write_text(json.dumps(rankings, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{m}] → {d / 'rankings.json'}（{len(rankings)} 靶）", flush=True)


def predictions(summary: dict, g: dict, c0: Optional[dict]) -> dict:
    p = summary["pooled"]
    bf = summary["by_fold_pair_acc"]

    def acc(arm):
        v = p.get(arm, {}).get("pair_acc")
        return None if v is None or not np.isfinite(v) else v

    out = {"P1": {"desc": "C0 硬门过", "hit": None if c0 is None else bool(c0["pass"])},
           "P2": {"desc": "G1 过（编码 +0.02 且 5/7）", "hit": g["G1"]["pass"]},
           "P3": {"desc": "G3 过（mlp_z_binned ≥ 0.765）", "hit": g["G3"]["pass"]},
           "P4": {"desc": "G3′ 过", "hit": g["G3p"]["pass"]},
           "P5": {"desc": "mlp_z_binned ≥ hgb_binned 0.780", "hit": None if acc("mlp_z_binned") is None
                  else bool(acc("mlp_z_binned") >= 0.780)}}
    sl, sp = acc("cls_only_scalar"), acc("cls_only_pod")
    out["P6"] = {"desc": "C0-local 在 0.02 带内", "value": None if sl is None or sp is None else sl - sp,
                 "hit": None if sl is None or sp is None else bool(abs(sl - sp) <= LOCAL_BAND)}
    d_enc = {f: d["cls_only_binned"] - d["cls_only_scalar"] for f, d in bf.items()
             if "cls_only_binned" in d and "cls_only_scalar" in d}
    if d_enc and g["G1"]["delta"] is not None:
        hit = all(d_enc.get(f, -np.inf) > g["G1"]["delta"] for f in ("PhonemeSpectra", "EthanolConcentration"))
    else:
        hit = None
    out["P7"] = {"desc": "编码收益集中在 PS / Ethanol（两折 Δ 都高于 pooled Δ）", "value": d_enc, "hit": hit}
    return out


def markdown_tables(a: dict) -> str:
    p = a["summary"]["pooled"]
    arms = [x for x in a["summary"]["arms"]]
    lines = [f"# WP16-TCLSC-L1 tables（{a['n_instances']} 实例）", "",
             f"读数码 **{a['verdict']}**", "",
             "| 臂 | 配对 acc | top-10 内 acc | Spearman | regret@1 | regret@3 | n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in arms:
        b = p[arm]
        lines.append(f"| {arm} | {b['pair_acc']:.3f} | {b['acc_top10']:.3f} | {b['rho']:.3f} | {b['regret1']:.3f} | "
                     f"{b['regret3']:.3f} | {b['n']} |")
    lines.append(f"| random | — | — | — | {p['random']['regret1']:.3f} | {p['random']['regret3']:.3f} | — |")
    lines += ["", "## 逐折配对 acc", "", "| 折 | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for f, d in a["summary"]["by_fold_pair_acc"].items():
        lines.append(f"| {f} | " + " | ".join(f"{d[x]:.3f}" if x in d else "—" for x in arms) + " |")
    lines += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        val = {kk: vv for kk, vv in v.items() if kk != "pass"}
        lines.append(f"| {k} | {json.dumps(val, ensure_ascii=False)} | {v['pass']} |")
    lines += ["", "## 配对检验（Wilcoxon，H1：后者更高）", ""]
    for t in a["tests"]:
        lines.append(f"- {t['a']} → {t['b']}: n={t['n']} p={t.get('p')} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')} "
                     f"Δ̄={t.get('mean_delta')}")
    lines += ["", "## 预测对账", ""]
    for k, v in a["predictions"].items():
        lines.append(f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else ""))
    return "\n".join(lines) + "\n"


def cmd_analyze(source: Path, outdir: Path, folds: Sequence[str], *, seeds: Sequence[int] = MLPZ_SEEDS,
                epochs: int = MLPZ_EPOCHS, skip_mlpz: bool = False) -> dict:
    rows, delta, rankings = load_source(source)
    cat_z = zscore_within(rows)
    groups = [eval_rankings(rankings, cat_z, delta, arms=list(POD_ARMS),
                            rename={"main": "main_pod", "cls_only": "cls_only_pod"})]
    for m in ENCODINGS:
        rp = outdir / m / "rankings.json"
        if rp.exists():
            rk = json.loads(rp.read_text(encoding="utf-8"))
            groups.append(eval_rankings(rk, cat_z, delta, arms=["cls_only", "main"],
                                        rename={"cls_only": f"cls_only_{m}", "main": f"main_{m}"}))
        else:
            print(f"⚠️ 缺 {rp}（先 --train / --rank），{m} 编码的 L1a 臂缺席", flush=True)
    if not skip_mlpz:
        enc = {m: encoder(m) for m in ENCODINGS}
        for t in folds:
            fr = fold_eval_mlpz(t, cat_z, delta, enc, seeds=seeds, epochs=epochs)
            groups.append(fr)
            print(f"[mlp_z {t}] scalar {pooled_acc(fr, 'mlp_z_scalar'):.3f} binned {pooled_acc(fr, 'mlp_z_binned'):.3f}",
                  flush=True)
    inst = merge_rows(*groups)
    summary = summarize(inst)
    g = gates(summary["pooled"], summary["by_fold_pair_acc"])
    tests = [paired_test(inst, a, b) for a, b in (("cls_only_scalar", "cls_only_binned"),
                                                   ("mlp_z_scalar", "mlp_z_binned"),
                                                   ("cls_only_binned", "mlp_z_binned"),
                                                   ("cls_only_pod", "cls_only_scalar"))]
    c0_path = outdir / "c0.json"
    c0 = json.loads(c0_path.read_text(encoding="utf-8")) if c0_path.exists() else None
    a = {"protocol": "results/tclsc_l1/PROTOCOL.md", "folds": list(folds), "n_instances": len(inst),
         "summary": summary, "gates": g, "verdict": verdict_code(g), "tests": tests,
         "c0": c0, "per_instance": inst}
    a["predictions"] = predictions(summary, g, c0)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"verdict {a['verdict']} | " + " | ".join(
        f"{arm} {summary['pooled'][arm]['pair_acc']:.3f}" for arm in summary["arms"]), flush=True)
    return a


# ---------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--modes", nargs="+", default=list(ENCODINGS), choices=list(ENCODINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", default=list(pq.FOLD_TARGETS))
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 33 号训练 epoch（只用于冒烟）")
    ap.add_argument("--mlpz-epochs", type=int, default=MLPZ_EPOCHS)
    ap.add_argument("--skip-mlpz", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.c0, args.prepare, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --c0 / --prepare / --train / --rank / --analyze 之一")
    if args.c0:
        res = cmd_c0(args.source, args.outdir, args.folds)
        if not res["pass"]:
            sys.exit("C0 硬门不过：口径未对齐，不往下跑")
    if args.prepare:
        cmd_prepare(args.source, args.outdir, args.modes)
    if args.train:
        cmd_train(args.outdir, args.modes, seeds=args.seeds, folds=None if args.folds == list(pq.FOLD_TARGETS) else args.folds,
                  device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.source, args.outdir, args.modes, folds=args.folds, seeds=args.seeds, device="cpu")
    if args.analyze:
        cmd_analyze(args.source, args.outdir, args.folds, seeds=tuple(args.seeds), epochs=args.mlpz_epochs,
                    skip_mlpz=args.skip_mlpz)


if __name__ == "__main__":
    main()
