"""MX-4 WP6：第 20 字段 input_mask 的零 GPU 分析（S0 / K1 / S3 源侧 LODO / 部署臂）。

协议：results/mx4_mask/PROTOCOL.md。沿 38 号的模式 importlib 复用 36（先验 / SH 路径）、37（配对 acc /
代理拟合 / 重要性）、39（部署轨迹 / 臂 / 论文参考），36/37/39 自身的 --analyze 输出逐位不变。

四个子命令：
  --s0      protocol_alignment/runs.jsonl 里既有 none/binomial 配对 → 每 (dataset, legacy key) 的 3 种子 Δacc，
            按 point_mask / emb_mask 档位与增强总强度制表（描述性，PROTOCOL §0）
  --k1      官方口径孪生 Δ（results/mx4_mask/runs.jsonl vs results/mx4_deploy/runs.jsonl）与切片协议孪生 Δ
            （source_twins vs results/mx3_pq）在 |Δ_official| ≥ 地板 的 (dataset, config) 对上的符号一致率 → 门 K1
  --source  7 折 LODO：prior19 / prior20 / gbdt19 / gbdt20，实例候选 = legacy 20 ∪ 该 (切片, 预算) 已有孪生；
            配对 acc、regret@1/@3、SH6-3-3、孪生方向准确率（对照 always-on/off）、分组置换重要性、交互读数 → 门 K2；
            C0：legacy-only 复算逐位等于 WP1 / WP2
  --deploy  掩码关半边 = WP3 runs，掩码开半边 = mx4_mask runs；42 目录上的臂：prior_off（C0 = WP3）、interleave、
            allon、measured(12@25%)、prior20 / gbdt20（S2 后）；3 种子均值；门 K3 / K4；读数码；picks.json

孪生的 z 用同 (task_id, iters) 的 legacy 20 条 mean/sd 换算（legacy z 逐位不变）；孪生行绝不进 results/mx3_pq。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g37 = importlib.import_module("scripts.37_surrogate_loo")
_g39 = importlib.import_module("scripts.39_sh_deploy")

OUT = ROOT / "results" / "mx4_mask"
MX3 = pq.OUT
WP3 = ROOT / "results" / "mx4_deploy"
PA_RUNS = ROOT / "results" / "protocol_alignment" / "runs.jsonl"
SRC_TWINS = OUT / "source_twins" / "rprime_cache.jsonl"
BUDGETS = pq.BUDGETS
TWIN = {"input_mask": "binomial"}
EXT_FIELD = "input_mask"
AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop", "emb_jitter", "emb_mask")
S0_STAGES = ("targets", "targets_ladder")      # 干净配对；ladder 阶段混有 crop/dropout 开关，不用
K1_FLOOR, K1_MIN_N, K1_RATE = 0.02, 15, 0.7
K2_DIR_ACC, K2_MARGIN, ALPHA = 0.70, 0.10, 0.05
PARITY_MARGIN, K3_MIN, K4_MIN, K4_TOL, K4_COST = 0.03, 5, 5, 0.01, 0.75
BASE_ARM, MAIN_ARM, MEASURED_ARM = "SH6-3-3_prior_off", "SH6-3-3_gbdt20", "SH12-3-3_measured"
RANK_ARMS = ("prior_off", "interleave", "allon", "prior20", "gbdt20")
K_TOP, TOP6, SH_MAIN, SH_LONG = 3, 6, (6, 3, 3), (10, 4, 3)
FINAL_SEEDS = _g39.FINAL_SEEDS
HYBRID_BIAS = _g39.HYBRID_BIAS


# ---------------------------------------------------------------- 纯函数：S0
def s0_deltas(rows: Sequence[dict], stages: Sequence[str] = S0_STAGES) -> list[dict]:
    """protocol_alignment 行 → 每 (dataset, legacy cls_key) 的 Δacc = mean_seed(binomial − none)。"""
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cls_of: dict[tuple, dict] = {}
    for r in rows:
        if r.get("error") or r.get("stage") not in stages or r.get("acc") is None:
            continue
        m = (r.get("options") or {}).get(EXT_FIELD, "none")
        acc[(r["dataset"], r["cls_key"], int(r.get("seed", 0)))][m].append(float(r["acc"]))
        cls_of.setdefault((r["dataset"], r["cls_key"]), r.get("cls") or {})
    per: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for (d, k, _s), by in acc.items():
        if "none" in by and "binomial" in by:
            per[(d, k)].append((float(np.mean(by["none"])), float(np.mean(by["binomial"]))))
    out = []
    for (d, k), vals in sorted(per.items()):
        c = cls_of[(d, k)]
        off = float(np.mean([v[0] for v in vals]))
        on = float(np.mean([v[1] for v in vals]))
        out.append({"dataset": d, "cls_key": k, "n_seeds": len(vals), "acc_off": off, "acc_on": on,
                    "delta": on - off, "cls": c,
                    "aug_strength": float(sum(float(c.get(f) or 0.0) for f in AUG_FIELDS))})
    return out


# ---------------------------------------------------------------- 纯函数：孪生 z
def twin_z(legacy_rows: Sequence[dict], twin_rows: Sequence[dict]) -> list[dict]:
    """孪生行的 z = (R′ − mean_legacy) / sd_legacy，同 (task_id, iters) 的 legacy 组统计（不改入参）。"""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in legacy_rows:
        if r.get("r_prime") is not None:
            groups[(r["task_id"], int(r["iters"]))].append(float(r["r_prime"]))
    stats = {g: (float(np.mean(v)), float(np.std(v))) for g, v in groups.items()}
    out = []
    for t in twin_rows:
        g = (t["task_id"], int(t["iters"]))
        if g not in stats or t.get("r_prime") is None:
            continue
        mu, sd = stats[g]
        out.append({**t, "z": (float(t["r_prime"]) - mu) / sd if sd > 0 else 0.0})
    return out


# ---------------------------------------------------------------- 纯函数：K1
def _official_final_acc(rows: Sequence[dict], *, by_twin_of: bool) -> dict[tuple, float]:
    out = {}
    for r in rows:
        if r.get("stage") != "s1_catalog" or int(r.get("seed", 0)) != 0 or r.get("error") or r.get("acc") is None:
            continue
        k = r.get("twin_of") if by_twin_of else r["cls_key"]
        if k is None:
            continue
        out[(r["dataset"], k)] = float(r["acc"])
    return out


def k1_table(off_rows: Sequence[dict], on_rows: Sequence[dict], legacy_rows: Sequence[dict],
             twin_rows: Sequence[dict], *, floor: float = K1_FLOOR,
             dataset_of: Callable[[str], str] = pq.dataset_of) -> dict:
    """官方 Δ（孪生 − legacy，seed 0，100% 截点）vs 切片 Δ（同配置该数据集全部孪生格均值）的符号一致率。"""
    off = _official_final_acc(off_rows, by_twin_of=False)
    on = _official_final_acc(on_rows, by_twin_of=True)
    leg = {(r["task_id"], r["cls_key"], int(r["iters"])): float(r["r_prime"])
           for r in legacy_rows if r.get("r_prime") is not None}
    slice_d: dict[tuple, list[float]] = defaultdict(list)
    for t in twin_rows:
        if t.get("r_prime") is None or not t.get("twin_of"):
            continue
        key = (t["task_id"], t["twin_of"], int(t["iters"]))
        if key in leg:
            slice_d[(dataset_of(t["task_id"]), t["twin_of"])].append(float(t["r_prime"]) - leg[key])
    pairs = []
    for d, k in sorted(set(off) & set(on) & set(slice_d)):
        do = on[(d, k)] - off[(d, k)]
        ds = float(np.mean(slice_d[(d, k)]))
        elig = bool(abs(do) >= floor)
        pairs.append({"dataset": d, "cls_key": k, "delta_official": do, "delta_slice": ds,
                      "n_slice_cells": len(slice_d[(d, k)]), "eligible": elig,
                      "agree": (bool(np.sign(do) == np.sign(ds)) if elig else None)})
    elig = [p for p in pairs if p["eligible"]]
    n_agree = int(sum(1 for p in elig if p["agree"]))
    rho = (float(spearmanr([p["delta_official"] for p in pairs], [p["delta_slice"] for p in pairs]).correlation)
           if len(pairs) >= 3 else None)
    return {"floor": float(floor), "n_pairs_total": len(pairs), "n_pairs_eligible": len(elig),
            "n_agree": n_agree, "agreement_rate": (n_agree / len(elig) if elig else None),
            "spearman_all": rho, "pairs": pairs}


def k1_pass(t: Mapping, *, rate: float = K1_RATE, min_n: int = K1_MIN_N) -> bool:
    r = t.get("agreement_rate")
    return bool(r is not None and r >= rate and int(t.get("n_pairs_eligible", 0)) >= min_n)


# ---------------------------------------------------------------- 纯函数：孪生方向准确率 / 排名
def twin_direction_accuracy(scores: Mapping[str, float], truth: Mapping[str, float],
                            pairs: Sequence[tuple[str, str]], delta: float) -> tuple[Optional[float], int]:
    """δ 存活孪生对上 sign(score_t − score_k) == sign(R′_t − R′_k) 的比例；无存活对 → (None, 0)。"""
    hits = n = 0
    for k, t in pairs:
        if k not in truth or t not in truth or k not in scores or t not in scores:
            continue
        d = float(truth[t]) - float(truth[k])
        if abs(d) < delta:
            continue
        n += 1
        s = float(scores[t]) - float(scores[k])
        if s != 0 and np.sign(s) == np.sign(d):
            hits += 1
    return (hits / n if n else None), n


def constant_scores(pairs: Sequence[tuple[str, str]], sign: float) -> dict[str, float]:
    """always-on（sign=+1）/ always-off（sign=−1）对照的得分表。"""
    out = {k: 0.0 for k, _ in pairs}
    out.update({t: float(sign) for _, t in pairs})
    return out


def interleave(ranked: Sequence[str], twin_map: Mapping[str, str]) -> list[str]:
    out = []
    for k in ranked:
        out.append(k)
        if k in twin_map:
            out.append(twin_map[k])
    return out


def all_on(ranked: Sequence[str], twin_map: Mapping[str, str]) -> list[str]:
    return [twin_map[k] for k in ranked if k in twin_map]


def measured_ranking(ranked: Sequence[str], twin_map: Mapping[str, str], top: int = TOP6) -> list[str]:
    """先验前 top 条各带孪生 → 第一档同时量两版（SH12-3-3_measured 的起跑名单）。"""
    return interleave(list(ranked)[:int(top)], twin_map)


# ---------------------------------------------------------------- 纯函数：门
def run_gates_mask(per: Mapping[str, dict], *, k2: Optional[Mapping], k1: Optional[Mapping],
                   margin: float = PARITY_MARGIN) -> dict:
    ds = list(per)
    parity = {d: bool(per[d].get("champion_mean") is not None
                      and per[d]["champion_mean"] >= per[d]["paper_ts2vec"] - margin) for d in ds}
    k3 = {"n_parity": int(sum(parity.values())), "n": len(ds), "parity": parity,
          "hw_parity": parity.get("Handwriting"), "pass": bool(sum(parity.values()) >= K3_MIN)}
    within = {d: bool(per[d].get("measured_mean") is not None and per[d].get("champion_mean") is not None
                      and per[d]["measured_mean"] - per[d]["champion_mean"] <= K4_TOL) for d in ds}
    cost_ok = bool(all(float(per[d].get("rung0_cost_ratio", 1.0)) <= K4_COST for d in ds))
    k4 = {"n_within": int(sum(within.values())), "n": len(ds), "within": within, "cost_ok": cost_ok,
          "pass": bool(sum(within.values()) >= K4_MIN and cost_ok)}
    return {"K1": (dict(k1) if k1 is not None else None), "K2": (dict(k2) if k2 is not None else None),
            "K3": k3, "K4": k4}


def verdict_code_mask(g: Mapping[str, dict]) -> str:
    if not g["K3"]["pass"]:
        code = "MASK-DEAD"
    elif g.get("K2") and g["K2"].get("pass"):
        code = "MASK-WIN"
    else:
        code = "MASK-SPACE"
    if g.get("K1") is not None and not g["K1"].get("pass"):
        code += "-NOLEARN"
    return code


# ---------------------------------------------------------------- 装载
def load_wp3_rows() -> list[dict]:
    return read_jsonl(WP3 / "runs.jsonl") if (WP3 / "runs.jsonl").exists() else []


def load_mask_rows(outdir: Path = OUT) -> list[dict]:
    p = outdir / "runs.jsonl"
    return read_jsonl(p) if p.exists() else []


def load_source_twins(path: Path = SRC_TWINS) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in read_jsonl(path) if r.get("error") is None and r.get("r_prime") is not None]


def wp3_rankings(c: Mapping, keys: Sequence[str]) -> dict[str, list[str]]:
    """与 39.cmd_analyze 同一套排名（prior / hybrid / tclsc），C0 的前提。"""
    nom = {e["cls_key"]: float(e["pred_z"]) for e in c.get("nominated", [])}
    prior = {k: float(v) for k, v in c["prior"].items()}
    hybrid = {k: (prior[k] + HYBRID_BIAS if k in prior else nom.get(k, float("-inf"))) for k in keys}
    return {"prior": _g36.rank_by(prior, keys), "hybrid": _g36.rank_by(hybrid, keys),
            "tclsc": [k for k in c.get("tclsc_order", []) if k in keys]}


def _s1_seed0(rows: Sequence[dict], d: str) -> list[dict]:
    return [r for r in rows if r.get("stage") == "s1_catalog" and r.get("error") is None
            and r["dataset"] == d and int(r.get("seed", 0)) == 0]


def deploy_target_off(d: str, rows: Sequence[dict], cands: Optional[Mapping] = None) -> dict:
    """WP3 的每靶臂表复算（39.deploy_arms 同码）——C0 锚。"""
    cands = cands or _g39.load_candidates(WP3)
    traj, secs = _g39.trajectory_from_rows(_s1_seed0(rows, d))
    budgets = sorted({b for v in traj.values() for b in v})
    keys = sorted(k for k in traj if all(b in traj[k] for b in budgets))
    rank = wp3_rankings(cands[d], keys)
    return _g39.deploy_arms(keys, traj, rank, budgets=budgets, random_seed=0, seconds=secs)


# ---------------------------------------------------------------- C0：源侧
def c0_source(folds: Sequence[str] = pq.FOLD_TARGETS, *, seed: int = 0) -> dict:
    """legacy-only 复算：36.fold_instances / 37.fold_eval 同码同输入 → 与 WP1 / WP2 analysis.json 逐位。"""
    space = CLSSpace.from_yaml()
    cat, ggs, _probe, rankings = _g36.load_inputs(MX3)
    cat_z = _g36.zscore_within(cat)
    enc_scalar = CLSEncoder(space)
    wp1_rows = []
    for target in folds:
        inst, _meta = _g36.fold_instances(target, cat_z, ggs, rankings, enc_scalar, seed=seed)
        wp1_rows += inst
    rows2, _evo, delta, rankings2 = _g37.load_inputs(MX3)
    cat_z2 = _g36.zscore_within(rows2)
    enc = {"scalar": CLSEncoder(space), "binned": CLSEncoder(space, mode="binned")}
    fold_acc = {}
    for target in folds:
        fr = _g37.fold_eval(target, cat_z2, delta, rankings2, enc, seed=seed)
        accs = [r["metrics"]["hgb_binned"]["pair_acc"] for r in fr if r["metrics"]["hgb_binned"]["pair_acc"] is not None]
        fold_acc[target] = float(np.mean(accs))
    return {"wp1_rows": wp1_rows, "wp2_fold_pair_acc": fold_acc}


# ---------------------------------------------------------------- S3：源侧 LODO（孪生臂）
def _features(rows: Sequence[dict], enc: CLSEncoder) -> np.ndarray:
    return _g36.features(rows, enc, max(BUDGETS))


def _traj_of(rows: Sequence[dict]) -> dict[str, dict[int, float]]:
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        traj[r["cls_key"]][int(r["iters"])] = float(r["r_prime"])
    return dict(traj)


def analyze_source(*, mx3: Path = MX3, twins_path: Path = SRC_TWINS, seed: int = 0,
                   folds: Sequence[str] = pq.FOLD_TARGETS) -> dict:
    space = CLSSpace.from_yaml()
    cat, _ggs, _probe, _rankings = _g36.load_inputs(mx3)
    legacy = _g36.zscore_within(cat)
    delta = pq.load_delta_calibration(mx3 / "delta_calibration.json")
    twin_rows = [r for r in load_source_twins(twins_path) if r.get("origin") != "ggs"]
    twins = twin_z(legacy, twin_rows)
    if not twins:
        raise SystemExit(f"无孪生行：{twins_path}（S2 未跑）")
    enc19 = CLSEncoder(space, mode="binned")
    enc20 = CLSEncoder(space, mode="binned", fields="all")
    per_inst: list[dict] = []
    dir_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)   # arm -> [(hits, n)] per instance
    for target in folds:
        tr_leg = [r for r in legacy if r["dataset"] != target]
        tr_tw = [r for r in twins if r["dataset"] != target]
        prior19 = _g36.prior_scores([r for r in tr_leg if r["origin"] == ORIGIN_SHARED])
        prior20 = _g36.prior_scores([r for r in tr_leg + tr_tw if r["origin"] == ORIGIN_SHARED])
        g19 = FieldSurrogate("hgb", seed=seed).fit(_features(tr_leg, enc19), np.array([r["z"] for r in tr_leg]))
        g20 = FieldSurrogate("hgb", seed=seed).fit(_features(tr_leg + tr_tw, enc20),
                                                   np.array([r["z"] for r in tr_leg + tr_tw]))
        te_leg = [r for r in legacy if r["dataset"] == target]
        te_tw = [r for r in twins if r["dataset"] == target]
        for s in sorted({r["task_id"] for r in te_leg}):
            rows_s = [r for r in te_leg if r["task_id"] == s] + [r for r in te_tw if r["task_id"] == s]
            traj = _traj_of(rows_s)
            for b in BUDGETS:
                rows_sb = sorted((r for r in rows_s if int(r["iters"]) == b), key=lambda r: r["cls_key"])
                keys = [r["cls_key"] for r in rows_sb]
                truth = {r["cls_key"]: float(r["r_prime"]) for r in rows_sb}
                dl = float(delta[(target, b)])
                pairs = [(r["twin_of"], r["cls_key"]) for r in rows_sb if r.get("twin_of") in truth]
                scores = {
                    "prior19": {k: prior19.get(k, float("-inf")) for k in keys},
                    "prior20": {k: prior20.get(k, float("-inf")) for k in keys},
                    "gbdt19": dict(zip(keys, g19.predict(_features(rows_sb, enc19))[0])),
                    "gbdt20": dict(zip(keys, g20.predict(_features(rows_sb, enc20))[0])),
                }
                oracle = max(truth.values())
                stages = tuple(x for x in BUDGETS if x <= b)
                inst = {"fold": target, "slice": s, "budget": b, "n_candidates": len(keys),
                        "n_twins": len(pairs), "delta": dl, "metrics": {}, "regret": {"oracle": 0.0}}
                for arm, sc in scores.items():
                    acc, n = _g37.pair_accuracy(sc, truth, dl)
                    ranked = _g36.rank_by(sc, keys)
                    inst["metrics"][arm] = {"pair_acc": acc, "n_pairs": n}
                    inst["regret"][arm + "1"] = oracle - truth[ranked[0]]
                    inst["regret"][arm + "3"] = oracle - max(truth[k] for k in ranked[:K_TOP])
                    if len(stages) == 3:
                        path = _g36.sh_path(ranked, traj, SH_MAIN, stages)
                        inst["regret"][f"SH6-3-3_{arm}"] = oracle - truth[_g36.sh_pick(path[-1], traj, b)]
                    if pairs:
                        da, dn = twin_direction_accuracy(sc, truth, pairs, dl)
                        inst["metrics"][arm]["twin_dir_acc"], inst["metrics"][arm]["twin_dir_n"] = da, dn
                        if dn:
                            dir_pairs[arm].append((int(round(da * dn)), dn))
                if pairs:
                    for name, sign in (("always_on", +1.0), ("always_off", -1.0)):
                        da, dn = twin_direction_accuracy(constant_scores(pairs, sign), truth, pairs, dl)
                        inst["metrics"][name] = {"twin_dir_acc": da, "twin_dir_n": dn}
                        if dn:
                            dir_pairs[name].append((int(round(da * dn)), dn))
                inst["regret"]["random1"] = oracle - float(np.mean(list(truth.values())))
                per_inst.append(inst)
    arms = ("prior19", "prior20", "gbdt19", "gbdt20")
    pooled = {}
    for arm in arms:
        accs = [r["metrics"][arm]["pair_acc"] for r in per_inst if r["metrics"][arm]["pair_acc"] is not None]
        pooled[arm] = {"pair_acc": float(np.mean(accs)),
                       "regret1": float(np.mean([r["regret"][arm + "1"] for r in per_inst])),
                       "regret3": float(np.mean([r["regret"][arm + "3"] for r in per_inst])),
                       "sh633_regret290": float(np.mean([r["regret"][f"SH6-3-3_{arm}"] for r in per_inst
                                                         if f"SH6-3-3_{arm}" in r["regret"]]))}
    direction = {name: {"acc": (sum(h for h, _ in v) / sum(n for _, n in v) if v else None),
                        "n_pairs": int(sum(n for _, n in v))}
                 for name, v in dir_pairs.items()}
    x = np.array([r["regret"]["gbdt201"] for r in per_inst])
    y = np.array([r["regret"]["prior201"] for r in per_inst])
    p_reg = float(_g36.paired_wilcoxon_less(x, y))
    sub = [r for r in per_inst if r["n_twins"] >= 20]
    dir_g, dir_on, dir_off = (direction.get(k, {}).get("acc") for k in ("gbdt20", "always_on", "always_off"))
    k2 = {"twin_dir_acc_gbdt20": dir_g, "twin_dir_acc_always_on": dir_on, "twin_dir_acc_always_off": dir_off,
          "n_pairs": direction.get("gbdt20", {}).get("n_pairs"),
          "regret1_gbdt20": float(x.mean()), "regret1_prior20": float(y.mean()), "p_regret1": p_reg,
          "n_instances": len(per_inst), "n_instances_full_twins": len(sub),
          "pass": bool(dir_g is not None and dir_g >= K2_DIR_ACC
                       and dir_g >= max(v for v in (dir_on, dir_off) if v is not None) + K2_MARGIN
                       and p_reg < ALPHA and float(x.mean()) < float(y.mean()))}
    # 字段主效应（全池拟合，含孪生）+ 交互读数
    all_rows = legacy + twins
    g_all = FieldSurrogate("hgb", seed=seed).fit(_features(all_rows, enc20), np.array([r["z"] for r in all_rows]))
    imp = _g37.grouped_permutation_importance(g_all, _features(all_rows, enc20), np.array([r["z"] for r in all_rows]),
                                              enc20.field_slices, seed=seed)
    interaction = interaction_reading(g_all, enc20, legacy, twins, space)
    return {"protocol": "results/mx4_mask/PROTOCOL.md", "n_instances": len(per_inst), "pooled": pooled,
            "twin_direction": direction, "K2": k2,
            "grouped_importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
            "interaction": interaction, "per_instance": per_inst}


def interaction_reading(model: FieldSurrogate, enc20: CLSEncoder, legacy: Sequence[dict],
                        twins: Sequence[dict], space: CLSSpace) -> dict:
    """每条 legacy 配置：预测 Δz（孪生 − legacy，b=290）vs 真 Δz（同 (task, iters) 均值），按 point_mask / emb_mask 档位分层。"""
    from autocls.space.cls_space import twin as _twin
    zl = {(r["task_id"], int(r["iters"]), r["cls_key"]): float(r["z"]) for r in legacy}
    true_d: dict[str, list[float]] = defaultdict(list)
    cls_of: dict[str, dict] = {}
    for t in twins:
        k = t.get("twin_of")
        key = (t["task_id"], int(t["iters"]), k)
        if key in zl:
            true_d[k].append(float(t["z"]) - zl[key])
    for r in legacy:
        cls_of.setdefault(r["cls_key"], r["cls"])
    keys = sorted(true_d)
    if not keys:
        return {}
    rows_l = [{"cls": cls_of[k], "iters": max(BUDGETS)} for k in keys]
    rows_t = [{"cls": {**cls_of[k], **TWIN}, "iters": max(BUDGETS)} for k in keys]
    pred = model.predict(_features(rows_t, enc20))[0] - model.predict(_features(rows_l, enc20))[0]
    per = [{"cls_key": k, "pred_delta_z": float(pred[i]), "true_delta_z": float(np.mean(true_d[k])),
            "n": len(true_d[k]), "point_mask": cls_of[k]["point_mask"], "emb_mask": cls_of[k]["emb_mask"]}
           for i, k in enumerate(keys)]
    by: dict[str, dict] = {}
    for field in ("point_mask", "emb_mask"):
        g: dict[float, list[dict]] = defaultdict(list)
        for p in per:
            g[float(p[field])].append(p)
        by[field] = {str(lvl): {"n": len(v), "pred_delta_z": float(np.mean([p["pred_delta_z"] for p in v])),
                                "true_delta_z": float(np.mean([p["true_delta_z"] for p in v]))}
                     for lvl, v in sorted(g.items())}
    rho = float(spearmanr([p["pred_delta_z"] for p in per], [p["true_delta_z"] for p in per]).correlation) if len(per) >= 3 else None
    return {"n_configs": len(per), "spearman_pred_vs_true_delta": rho, "by_level": by, "per_config": per}


# ---------------------------------------------------------------- 部署臂（42 目录）
def mask_arms(keys: Sequence[str], traj: Mapping[str, Mapping[int, float]], rankings: Mapping[str, Sequence[str]],
              twin_map: Mapping[str, str], *, budgets: Sequence[int], seconds=None, random_seed: int = 0) -> dict:
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    keys = [k for k in keys if all(b in traj[k] for b in stages)]
    truth = {k: float(traj[k][bf]) for k in keys}
    on_keys = set(twin_map.values())
    arms: dict[str, float] = {"oracle42": max(truth.values()),
                              "oracle_off": max(v for k, v in truth.items() if k not in on_keys),
                              "oracle_on": (max(v for k, v in truth.items() if k in on_keys) if on_keys & set(truth) else None)}
    cost: dict[str, int] = {}
    secs: Optional[dict[str, float]] = {} if seconds is not None else None
    picks: dict[str, str] = {}

    def zero_shot(name, ranked, top):
        chosen = [k for k in ranked if k in truth][:top]
        if not chosen:
            return
        best = sorted(chosen, key=lambda k: (-truth[k], k))[0]
        arms[name], cost[name], picks[name] = truth[best], top * bf, best
        if secs is not None:
            secs[name] = float(sum(float(seconds[k][bf]) for k in chosen))

    def sh(name, ranked, ks, bud):
        ranked = [k for k in ranked if k in truth]
        if len(ranked) < 1:
            return
        path = _g36.sh_path(ranked, traj, ks, bud)
        pick = _g36.sh_pick(path[-1], traj, bf)
        arms[name], picks[name], cost[name] = truth[pick], pick, _g36.path_cost_iters(path, bud)
        if secs is not None:
            secs[name] = _g36.path_cost_seconds(path, bud, seconds)

    for name, ranked in rankings.items():
        zero_shot(f"{name}1", ranked, 1)
        zero_shot(f"{name}3", ranked, K_TOP)
        sh(f"SH6-3-3_{name}", ranked, SH_MAIN, stages)
        sh(f"SH10-4-3_{name}", ranked, SH_LONG, stages)
    measured = measured_ranking(rankings["prior_off"], twin_map, TOP6)
    sh(MEASURED_ARM, measured, (len(measured), 3, 3), stages)
    arms["random1"] = float(np.mean(list(truth.values())))
    cost["random1"], cost["catalog_direct"] = bf, len(keys) * bf
    regret = {k: (arms["oracle42"] - v if v is not None else None) for k, v in arms.items()}
    return {"arms": arms, "regret": regret, "cost_iters": cost, "cost_seconds": secs, "picks": picks,
            "n_candidates": len(keys), "n_on": len(on_keys & set(truth)), "budgets": stages}


def _three_seed(k: str, d: str, acc_seed0: Optional[float], s2_rows: Sequence[dict]) -> Optional[float]:
    if acc_seed0 is None or k is None:
        return None
    extra = [float(r["acc"]) for r in s2_rows if r["dataset"] == d and r["cls_key"] == k and r.get("error") is None]
    return float(np.mean([acc_seed0] + extra)) if len(extra) >= len(FINAL_SEEDS) else None


def analyze_deploy(outdir: Path = OUT, *, datasets: Sequence[str] = _g39.TARGETS,
                   source_analysis: Optional[Mapping] = None, k1: Optional[Mapping] = None,
                   seed: int = 0) -> dict:
    cands = _g39.load_candidates(outdir)
    off_rows, on_rows = load_wp3_rows(), load_mask_rows(outdir)
    s2 = [r for r in off_rows + on_rows if r.get("stage") == "s2_finals" and r.get("error") is None]
    # 源侧模型（S2 后）：prior20 / gbdt20 按折拟合
    src_twins = [r for r in load_source_twins() if r.get("origin") != "ggs"]
    have_source = bool(src_twins)
    space = CLSSpace.from_yaml()
    enc20 = CLSEncoder(space, mode="binned", fields="all")
    if have_source:
        cat, _g, _p, _r = _g36.load_inputs(MX3)
        legacy = _g36.zscore_within(cat)
        twins_src = twin_z(legacy, src_twins)
    per_target, picks_out = {}, {}
    for d in datasets:
        if d not in cands:
            continue
        c = cands[d]
        rows_off, rows_on = _s1_seed0(off_rows, d), _s1_seed0(on_rows, d)
        if not rows_off or not rows_on:
            print(f"[deploy {d}] 缺 S1 行（off {len(rows_off)} / on {len(rows_on)}），跳过", flush=True)
            continue
        traj_off, secs_off = _g39.trajectory_from_rows(rows_off)
        traj_on, secs_on = _g39.trajectory_from_rows(rows_on)
        traj = {**traj_off, **traj_on}
        secs = {**secs_off, **secs_on}
        budgets = sorted({b for v in traj_off.values() for b in v})
        keys_off = sorted(k for k in traj_off if all(b in traj_off[k] for b in budgets))
        twin_map = {r["twin_of"]: r["cls_key"] for r in rows_on if r.get("twin_of") in traj_off
                    and all(b in traj_on.get(r["cls_key"], {}) for b in budgets)}
        keys = keys_off + sorted(twin_map.values())
        base = wp3_rankings(c, keys_off)
        rankings: dict[str, list[str]] = {"prior_off": base["prior"],
                                          "interleave": interleave(base["prior"], twin_map),
                                          "allon": all_on(base["prior"], twin_map)}
        cls_of = {r["cls_key"]: r["cls"] for r in rows_off + rows_on}
        if have_source:
            tr_leg = [r for r in legacy if r["dataset"] != d]
            tr_tw = [r for r in twins_src if r["dataset"] != d]
            prior20 = _g36.prior_scores([r for r in tr_leg + tr_tw if r["origin"] == ORIGIN_SHARED])
            g20 = FieldSurrogate("hgb", seed=seed).fit(_features(tr_leg + tr_tw, enc20),
                                                       np.array([r["z"] for r in tr_leg + tr_tw]))
            feat_rows = [{"cls": cls_of[k], "iters": max(BUDGETS)} for k in keys]
            pred = dict(zip(keys, g20.predict(_features(feat_rows, enc20))[0]))
            rankings["gbdt20"] = _g36.rank_by(pred, keys)
            # prior20：shared 及其孪生用源池均值 z（+偏置置顶）；GGS 沿 WP3 先验；其余（提名及孪生）按 gbdt20
            ggs_key = c["ggs"]["cls_key"]
            p20 = {}
            for k in keys:
                if k in prior20:
                    p20[k] = prior20[k] + HYBRID_BIAS
                elif k == ggs_key and k in c["prior"]:
                    p20[k] = float(c["prior"][k]) + HYBRID_BIAS
                else:
                    p20[k] = pred[k]
            rankings["prior20"] = _g36.rank_by(p20, keys)
        row = mask_arms(keys, traj, rankings, twin_map, budgets=budgets, seconds=secs, random_seed=seed)
        bf = budgets[-1]
        acc_final = {k: traj[k][bf] for k in keys}
        main = MAIN_ARM if MAIN_ARM in row["arms"] else "SH6-3-3_interleave"
        champ, meas, basek = row["picks"].get(main), row["picks"].get(MEASURED_ARM), row["picks"].get(BASE_ARM)
        rung0_ratio = (SH_MAIN[0] * budgets[0]) / (len(measured_ranking(base["prior"], twin_map)) * budgets[0])
        per_target[d] = {
            "paper_ts2vec": _g39.paper_ts2vec(d), "paper_median": _g39.paper_median(d),
            "budgets": budgets, "n_candidates": len(keys), "n_on": row["n_on"], "main_arm": main,
            "champion_key": champ, "champion_seed0": row["arms"].get(main),
            "champion_mean": _three_seed(champ, d, row["arms"].get(main), s2),
            "champion_is_on": bool(champ in set(twin_map.values())) if champ else None,
            "measured_key": meas, "measured_seed0": row["arms"].get(MEASURED_ARM),
            "measured_mean": _three_seed(meas, d, row["arms"].get(MEASURED_ARM), s2),
            "base_key": basek, "base_seed0": row["arms"].get(BASE_ARM),
            "base_mean": _three_seed(basek, d, row["arms"].get(BASE_ARM), s2),
            "rung0_cost_ratio": float(rung0_ratio),
            "catalog_max": max(acc_final.values()), "catalog_max_key": max(acc_final, key=lambda k: (acc_final[k], k)),
            "arms": row["arms"], "regret": row["regret"], "cost_iters": row["cost_iters"],
            "cost_seconds": row["cost_seconds"], "picks": row["picks"],
        }
        # finals 待跑：各主要臂冠军里还没 3 种子的
        need = {}
        for arm in [main, MEASURED_ARM, "SH6-3-3_interleave", "SH6-3-3_allon", "SH10-4-3_prior_off",
                    "SH10-4-3_interleave"] + [f"SH10-4-3_{a}" for a in ("gbdt20", "prior20")]:
            k = row["picks"].get(arm)
            if k and _three_seed(k, d, row["arms"].get(arm), s2) is None:
                need[arm] = k
        picks_out[d] = need
        print(f"[deploy {d}] 候选 {len(keys)}（开 {row['n_on']}）oracle42 {row['arms']['oracle42']:.3f} "
              f"base {row['arms'].get(BASE_ARM, float('nan')):.3f} main({main}) {row['arms'].get(main, float('nan')):.3f} "
              f"measured {row['arms'].get(MEASURED_ARM, float('nan')):.3f}", flush=True)
    per_eff = {d: {**v, "champion_mean": (v["champion_mean"] if v["champion_mean"] is not None else v["champion_seed0"]),
                   "measured_mean": (v["measured_mean"] if v["measured_mean"] is not None else v["measured_seed0"])}
               for d, v in per_target.items()}
    k2 = (source_analysis or {}).get("K2") if source_analysis else None
    gates = run_gates_mask(per_eff, k2=k2, k1=k1) if per_target else None
    analysis = {"protocol": "results/mx4_mask/PROTOCOL.md", "n_targets": len(per_target),
                "n_targets_with_3_seeds": int(sum(v["champion_mean"] is not None for v in per_target.values())),
                "have_source_twins": have_source, "per_target": per_target, "gates": gates,
                "verdict": (verdict_code_mask(gates) if gates else None)}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis_deploy.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "picks.json").write_text(json.dumps(picks_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables_deploy.md").write_text(deploy_tables(analysis), encoding="utf-8")
    if gates:
        print(f"verdict {analysis['verdict']}；K3 {gates['K3']['n_parity']}/{gates['K3']['n']}（HW {gates['K3']['hw_parity']}）；"
              f"K4 {gates['K4']['n_within']}/{gates['K4']['n']}", flush=True)
    return analysis


def deploy_tables(a: Mapping) -> str:
    per = a.get("per_target") or {}
    if not per:
        return "（无靶）\n"
    arm_names = sorted({k for v in per.values() for k in v["arms"]})
    lines = ["## 六靶头条（100% 截点 test acc；冠军列 = seed0 / 3 种子）", "",
             "| 靶 | 论文 | 目录 oracle42 / off / on | base SH6-3-3_prior_off | 主臂 | measured 12@25% | 追平 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for d, v in per.items():
        ar = v["arms"]
        fmt = lambda x: ("—" if x is None else f"{x:.3f}")
        par = "✓" if (v["champion_mean"] or v["champion_seed0"] or 0) >= v["paper_ts2vec"] - PARITY_MARGIN else "✗"
        lines.append(f"| {d} | {v['paper_ts2vec']:.3f} | {fmt(ar['oracle42'])} / {fmt(ar['oracle_off'])} / {fmt(ar['oracle_on'])} "
                     f"| {fmt(v['base_seed0'])} / {fmt(v['base_mean'])} | {v['main_arm']} {fmt(v['champion_seed0'])} / {fmt(v['champion_mean'])}"
                     f"{'（开）' if v['champion_is_on'] else ''} | {fmt(v['measured_seed0'])} / {fmt(v['measured_mean'])} | {par} |")
    lines += ["", "## 各靶臂表（seed 0，acc@100%）", "", "| 臂 | " + " | ".join(per) + " |", "|---|" + "---:|" * len(per)]
    for arm in arm_names:
        lines.append(f"| {arm} | " + " | ".join(("—" if per[d]["arms"].get(arm) is None else f"{per[d]['arms'][arm]:.3f}") for d in per) + " |")
    g = a.get("gates")
    if g:
        lines += ["", f"读数码 **{a['verdict']}**；K3 {g['K3']['n_parity']}/{g['K3']['n']}；K4 {g['K4']['n_within']}/{g['K4']['n']}"
                  + (f"；K2 {'过' if g['K2'] and g['K2'].get('pass') else '未过/未测'}")
                  + (f"；K1 {'过' if g['K1'] and g['K1'].get('pass') else '未过/未测'}")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 子命令
def cmd_s0(outdir: Path) -> dict:
    rows = read_jsonl(PA_RUNS)
    d = s0_deltas(rows)
    lines = ["| 靶 | 配置 | 种子 | acc 关 | acc 开 | Δ | point_mask | emb_mask | crop | freq_mask | 增强总强度 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in sorted(d, key=lambda e: e["delta"]):
        c = e["cls"]
        lines.append(f"| {e['dataset']} | {e['cls_key'][:8]} | {e['n_seeds']} | {e['acc_off']:.3f} | {e['acc_on']:.3f} | "
                     f"{e['delta']:+.3f} | {c.get('point_mask')} | {c.get('emb_mask')} | {c.get('crop')} | {c.get('freq_mask')} | "
                     f"{e['aug_strength']:.2f} |")
    deltas = np.array([e["delta"] for e in d])
    strength = np.array([e["aug_strength"] for e in d])
    rho = float(spearmanr(strength, deltas).correlation) if len(d) >= 3 else None
    out = {"n_pairs": len(d), "delta_mean": float(deltas.mean()) if len(d) else None,
           "n_positive": int((deltas > 0).sum()), "spearman_strength_vs_delta": rho, "pairs": d}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "s0_alignment.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "s0_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[s0] {len(d)} 对；Δ 均值 {out['delta_mean']:+.4f}；正 {out['n_positive']}/{len(d)}；"
          f"Spearman(强度, Δ) {rho}", flush=True)
    return out


def cmd_k1(outdir: Path) -> dict:
    cat, _g, _p, _r = _g36.load_inputs(MX3)
    ggs_rows = [r for r in read_jsonl(MX3 / "rprime_cache.jsonl") if r.get("origin") == "ggs" and r.get("error") is None]
    t = k1_table(load_wp3_rows(), load_mask_rows(outdir), list(cat) + ggs_rows, load_source_twins())
    t["pass"] = k1_pass(t)
    (outdir / "k1.json").write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[k1] 对 {t['n_pairs_total']}，|Δ_official|≥{t['floor']} 的 {t['n_pairs_eligible']}，一致 {t['n_agree']}"
          f"（{t['agreement_rate']}）；Spearman {t['spearman_all']}；K1 {'过' if t['pass'] else '不过'}", flush=True)
    return t


def cmd_c0(outdir: Path) -> dict:
    wp1 = json.loads((ROOT / "results" / "mx4_prior_sh" / "analysis.json").read_text(encoding="utf-8"))
    wp2 = json.loads((ROOT / "results" / "mx4_surrogate" / "analysis.json").read_text(encoding="utf-8"))
    wp3 = json.loads((WP3 / "analysis.json").read_text(encoding="utf-8"))
    c0 = c0_source()
    ref = {(r["fold"], r["slice"], r["budget"]): r for r in wp1["per_instance"]}
    def _mismatch(r: dict) -> bool:
        e = ref[(r["fold"], r["slice"], r["budget"])]["regret"]
        if r["regret"]["prior1"] != e["prior1"]:
            return True
        return "SH6-3-3_prior" in e and r["regret"].get("SH6-3-3_prior") != e["SH6-3-3_prior"]   # SH 只在 @290

    bad1 = [(r["fold"], r["slice"], r["budget"]) for r in c0["wp1_rows"] if _mismatch(r)]
    bad2 = {t: (c0["wp2_fold_pair_acc"][t], wp2["summary"]["by_fold_pair_acc"][t]["hgb_binned"])
            for t in c0["wp2_fold_pair_acc"] if abs(c0["wp2_fold_pair_acc"][t] - wp2["summary"]["by_fold_pair_acc"][t]["hgb_binned"]) > 1e-9}
    rows = load_wp3_rows()
    bad3 = {}
    for d, refd in wp3["per_target"].items():
        got = deploy_target_off(d, rows)
        if got["arms"]["SH6-3-3_prior"] != refd["arms"]["SH6-3-3_prior"] or got["picks"]["SH6-3-3_prior"] != refd["picks"]["SH6-3-3_prior"]:
            bad3[d] = (got["arms"]["SH6-3-3_prior"], refd["arms"]["SH6-3-3_prior"])
    out = {"wp1_mismatch": bad1, "wp2_mismatch": bad2, "wp3_mismatch": bad3,
           "pass": not (bad1 or bad2 or bad3), "n_wp1_rows": len(c0["wp1_rows"])}
    (outdir / "c0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[c0] WP1 {len(c0['wp1_rows'])} 实例不匹配 {len(bad1)}；WP2 折不匹配 {len(bad2)}；WP3 靶不匹配 {len(bad3)} → "
          f"{'过' if out['pass'] else '不过'}", flush=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s0", action="store_true")
    ap.add_argument("--k1", action="store_true")
    ap.add_argument("--c0", action="store_true")
    ap.add_argument("--source", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.s0:
        cmd_s0(outdir)
    if args.c0:
        cmd_c0(outdir)
    if args.k1:
        cmd_k1(outdir)
    src = None
    if args.source:
        src = analyze_source(seed=args.seed)
        (outdir / "analysis_source.json").write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
        k2 = src["K2"]
        print(f"[source] pooled: " + "; ".join(f"{a} acc {v['pair_acc']:.3f} r1 {v['regret1']:.4f} r3 {v['regret3']:.4f} "
                                               f"SH {v['sh633_regret290']:.4f}" for a, v in src["pooled"].items()), flush=True)
        print(f"[source] 孪生方向 acc gbdt20 {k2['twin_dir_acc_gbdt20']} / on {k2['twin_dir_acc_always_on']} / off "
              f"{k2['twin_dir_acc_always_off']}（n {k2['n_pairs']}）；regret@1 gbdt20 {k2['regret1_gbdt20']:.4f} vs prior20 "
              f"{k2['regret1_prior20']:.4f} p={k2['p_regret1']:.3g} → K2 {'过' if k2['pass'] else '不过'}", flush=True)
    if args.deploy:
        if src is None and (outdir / "analysis_source.json").exists():
            src = json.loads((outdir / "analysis_source.json").read_text(encoding="utf-8"))
        k1 = json.loads((outdir / "k1.json").read_text(encoding="utf-8")) if (outdir / "k1.json").exists() else None
        analyze_deploy(outdir, source_analysis=src, k1=k1, seed=args.seed)
    if not any((args.s0, args.k1, args.c0, args.source, args.deploy)):
        ap.print_help()


if __name__ == "__main__":
    main()
