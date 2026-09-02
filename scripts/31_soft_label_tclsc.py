"""实验 2：噪声感知软 label 重训 T-CLSC + shuffle 消融 + shared-only 对照 + 翻转注入曲线
（results/soft_label_tclsc/PROTOCOL.md；DESIGN_NOTES §1.3-1 / §1.4 / §6-2）。

四段（全部零新增 R'）：
  --noise-model     §1 噪声模型（per-dataset × 制式 × 两成分；Student-t + 经验 CDF）→ noise_model.json
                    + §2 信息加权任务盲天花板（含硬 δ / δ=0 复现校验）→ ceiling.json
  --train --resume  §3 训练臂（H0/S1/S2/S1e/Cshared/F(f)/X）× 种子，每跑 runs/<arm>_s<seed>.json
  --verdict         §5 判读表 → verdict.json

用法：python scripts/31_soft_label_tclsc.py --noise-model --train --resume --verdict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
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

from autocls.comparator import TCLSC  # noqa: E402
from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder  # noqa: E402
from autocls.comparator.soft_labels import (  # noqa: E402
    NoiseModel, SoftTCLSCEngine, apply_flips, fit_noise_model, hard_dynamic_pairs_soft,
    info_weight, pair_probability, soft_dynamic_pairs, soft_pairs_all, task_scale_factor,
    weighted_task_blind_ceiling, with_reversed_soft,
)
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts.e_ablation_common import (  # noqa: E402
    DATASETS, WorkpointData, ablation_block, build_runs, cross_assignment, ds_of,
    load_set_encoder_init, read_jsonl, strict_destroyed, tier_of, within_assignment,
)

OUT = ROOT / "results" / "soft_label_tclsc"
KAPPA = 0.595            # 计数成分方差份额（协议 §1；UM@30 两成分分解）
MIN_REGIME_N = 5
MID_LO, MID_HI = 0.6, 0.97
N_ABL = 5
N_BOOT = 1000
FLIP_RATES = (0.1, 0.2, 0.3, 0.4)

# 臂表（协议 §3）：(arm, seeds, label 模式, 噪声模型 kind, origins 固定?, flip, encoder)
ARMS = (
    [("H0", s, "hard", "student_t", None, 0.0, "settf") for s in (0, 1, 2)]
    + [("S1", s, "soft", "student_t", None, 0.0, "settf") for s in (0, 1, 2)]
    + [("S2", s, "weighted", "student_t", None, 0.0, "settf") for s in (0, 1, 2)]
    + [("S1e", 0, "soft", "empirical", None, 0.0, "settf")]
    + [("Cshared", s, "hard", "student_t", ("shared",), 0.0, "settf") for s in (0, 1, 2)]
    + [(f"F{int(f * 100)}", s, "hard", "student_t", None, f, "settf") for f in FLIP_RATES for s in (0, 1)]
    + [("X", s, "soft", "student_t", None, 0.0, "meanpool") for s in (0, 1, 2)]
)


def margin_for(task_type: str):
    """铁律 8：方向经 task_types；分类 = R'_a − R'_b（>0 ⇔ a 优）。"""
    if task_type == TASK_CLASSIFICATION:
        return lambda ra, rb: ra - rb
    raise NotImplementedError(f"task_type={task_type!r} 的 margin 未定义（forecasting 应为 rb − ra）")


# ---------------------------------------------------------------- §1 噪声模型


class ProbModel:
    """噪声模型 × 任务尺度 × 制式 → prob_fn(task_id, key_a, key_b, delta)。"""

    def __init__(self, data: WorkpointData, kind: str):
        self.kind = kind
        self.strict = strict_destroyed(data)
        noise = read_jsonl(data.labels_dir / "noise_probe.jsonl")
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        canonical = {(r.task_id, r.key): r.r for r in data.records}
        for r in noise:
            if r.get("r_prime") is not None and np.isfinite(r["r_prime"]):
                cells[(r["task_id"], r["cls_key"])].append(float(r["r_prime"]))
        self.models: dict[str, NoiseModel] = {}
        self.factor: dict[str, float] = {}
        self.calib: dict[str, dict] = {}
        for d in DATASETS:
            diffs = {"nd": [], "dest": []}
            n_cells = {"nd": 0, "dest": 0}
            noise_tasks = set()
            for (tid, key), extra in cells.items():
                if ds_of(tid) != d or (tid, key) not in canonical:
                    continue
                vals = [canonical[(tid, key)]] + extra
                regime = "dest" if self.strict[(tid, key)] else "nd"
                n_cells[regime] += 1
                noise_tasks.add(tid)
                for i, j in combinations(range(len(vals)), 2):
                    diffs[regime].append(vals[i] - vals[j])
            if n_cells["dest"] < MIN_REGIME_N:  # 协议 §1：破坏制式 cell < 5 → 单制式（按 cell 计，非按差值计）
                diffs = {"nd": diffs["nd"] + diffs["dest"], "dest": []}
            self.models[d] = fit_noise_model(diffs, kind=kind, min_regime_n=MIN_REGIME_N)
            n_ref = float(np.mean([data.n_eval[t] for t in noise_tasks]))
            for t in data.all_tids:
                if ds_of(t) == d:
                    self.factor[t] = task_scale_factor(n_eval=data.n_eval[t], n_ref=n_ref, kappa=KAPPA)
            self.calib[d] = {"n_cells": n_cells, "n_diffs": {k: len(v) for k, v in diffs.items()},
                             "abs_diff_median": {k: (float(np.median(np.abs(v))) if v else None)
                                                 for k, v in diffs.items()},
                             "abs_diff_p95": {k: (float(np.percentile(np.abs(v), 95)) if v else None)
                                              for k, v in diffs.items()},
                             "n_ref": n_ref, "noise_tasks": sorted(noise_tasks),
                             "model": self.models[d].to_json(),
                             "factor_range": [min(self.factor[t] for t in data.all_tids if ds_of(t) == d),
                                              max(self.factor[t] for t in data.all_tids if ds_of(t) == d)]}

    def scale_of(self, task_id: str, key: str) -> float:
        m = self.models[ds_of(task_id)]
        regime = "dest" if (self.strict[(task_id, key)] and "dest" in m.scale) else "nd"
        return m.scale[regime] * self.factor[task_id]

    def __call__(self, task_id: str, key_a: str, key_b: str, delta: float) -> float:
        return pair_probability(delta, self.models[ds_of(task_id)],
                                self.scale_of(task_id, key_a), self.scale_of(task_id, key_b))

    def to_json(self) -> dict:
        return {"kind": self.kind, "kappa": KAPPA, "min_regime_n": MIN_REGIME_N,
                "per_dataset": self.calib,
                "n_strict_cells": {d: sum(1 for (t, _k), v in self.strict.items() if v and ds_of(t) == d)
                                   for d in DATASETS}}


def stage_noise_model(data: WorkpointData) -> tuple[ProbModel, ProbModel]:
    pm_t, pm_e = ProbModel(data, "student_t"), ProbModel(data, "empirical")
    margin = margin_for(data.task_type)
    # p 分布（全 5320 无序对）
    all_soft = soft_pairs_all(data.records, margin, pm_t)
    pmax = np.array([max(p.p, 1 - p.p) for p in all_soft])
    band = {"n_pairs": len(all_soft),
            "share_pmax_lt_0.6": float(np.mean(pmax < MID_LO)),
            "share_mid_0.6_0.97": float(np.mean((pmax >= MID_LO) & (pmax < MID_HI))),
            "share_pmax_ge_0.97": float(np.mean(pmax >= MID_HI)),
            "mass_share_mid": float(info_weight(pmax)[(pmax >= MID_LO) & (pmax < MID_HI)].sum()
                                    / info_weight(pmax).sum())}
    # 硬 δ 存活对的等效翻转率 f_eff（协议 §5 Q3b）
    # 用 δ 判据直接从 all_soft 取存活对（同 make_pairs 口径：|Δ| ≥ δ 且非平局）
    rec_r = {(r.task_id, r.key): r.r for r in data.records}
    f_eff_vals = []
    for p in all_soft:
        d = margin(rec_r[(p.task_id, p.key_a)], rec_r[(p.task_id, p.key_b)])
        if abs(d) >= data.delta_map[ds_of(p.task_id)] and d != 0:
            f_eff_vals.append(1 - max(p.p, 1 - p.p))
    f_eff = {"mean": float(np.mean(f_eff_vals)), "n_surviving": len(f_eff_vals),
             "p90": float(np.percentile(f_eff_vals, 90))}
    out = {"student_t": pm_t.to_json(), "empirical": pm_e.to_json(), "p_band": band, "f_eff_hard_delta": f_eff}
    (OUT / "noise_model.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for d in DATASETS:
        c = pm_t.calib[d]
        print(f"[noise] {d}: cells nd/dest={c['n_cells']} | |d| med nd={c['abs_diff_median']['nd']:.4f} "
              f"dest={c['abs_diff_median']['dest']} | t: {c['model']} | factor {c['factor_range']}", flush=True)
    print(f"[noise] p 分带: {band} | f_eff(硬δ存活)={f_eff}", flush=True)
    return pm_t, pm_e


# ---------------------------------------------------------------- §2 加权天花板


def stage_ceiling(data: WorkpointData, pm_t: ProbModel, pm_e: ProbModel) -> dict:
    margin = margin_for(data.task_type)
    shared = sorted({r.key for r in data.records if r.origin == "shared"})
    assert len(shared) == 10, f"shared CLS 应为 10，得到 {len(shared)}"
    rec_r = {(r.task_id, r.key): r.r for r in data.records}

    def votes_with(weight_fn):
        votes = defaultdict(list)
        for t in data.all_tids:
            for a, b in combinations(shared, 2):
                ra, rb = rec_r.get((t, a)), rec_r.get((t, b))
                if ra is None or rb is None:
                    continue
                d = margin(ra, rb)
                if d == 0:
                    continue  # 严格平局：与 make_pairs 同（软权本来也为 0）
                votes[(a, b)].append((t, int(d > 0), float(weight_fn(t, a, b, d))))
        return votes

    def report(votes) -> dict:
        r = {"pooled": weighted_task_blind_ceiling(votes)}
        for d in DATASETS:
            r[d] = weighted_task_blind_ceiling(votes, task_filter=lambda t, d=d: ds_of(t) == d)
        return r

    def band_fn(pm, lo, hi):
        def f(t, a, b, d):
            p = pm(t, a, b, d)
            pm_ = max(p, 1 - p)
            return abs(2 * p - 1) if lo <= pm_ < hi else 0.0
        return f

    res = {
        "hard_delta": report(votes_with(lambda t, a, b, d: 1.0 if abs(d) >= data.delta_map[ds_of(t)] else 0.0)),
        "delta_zero": report(votes_with(lambda t, a, b, d: 1.0)),
        "soft_student_t": report(votes_with(lambda t, a, b, d: abs(2 * pm_t(t, a, b, d) - 1))),
        "soft_empirical": report(votes_with(lambda t, a, b, d: abs(2 * pm_e(t, a, b, d) - 1))),
        "soft_student_t_midband": report(votes_with(band_fn(pm_t, MID_LO, MID_HI))),
        "soft_student_t_highband": report(votes_with(band_fn(pm_t, MID_HI, 1.01))),
        "soft_student_t_lowband": report(votes_with(band_fn(pm_t, 0.0, MID_LO))),
    }
    hd, dz = res["hard_delta"]["pooled"]["ceiling"], res["delta_zero"]["pooled"]["ceiling"]
    res["reproduction_check"] = {
        "hard_delta_pooled": hd, "expected": 0.9525, "pass": bool(abs(hd - 0.9525) < 2e-3),
        "delta_zero_pooled": dz, "expected_zero": 0.7473, "pass_zero": bool(abs(dz - 0.7473) < 2e-3),
        "hard_HW": res["hard_delta"]["Handwriting"]["ceiling"], "hard_UM": res["hard_delta"]["UniMiB-SHAR"]["ceiling"],
    }
    (OUT / "ceiling.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        if k == "reproduction_check":
            continue
        print(f"[ceiling] {k:26s} pooled {v['pooled']['ceiling']:.4f} (mass {v['pooled']['mass']:.1f}) | "
              f"HW {v['Handwriting']['ceiling']:.4f} | UM {v['UniMiB-SHAR']['ceiling']:.4f}", flush=True)
    print(f"[ceiling] 复现校验 {res['reproduction_check']}", flush=True)
    return res


# ---------------------------------------------------------------- §3 训练臂


def make_sampler(arm_cfg, data: WorkpointData, train_tids: set[str], prob_fn, margin):
    arm, seed, label_mode, _kind, fixed_origins, flip, _enc = arm_cfg
    recs = data.records_in(train_tids)
    delta = data.delta_by_task(train_tids)
    cur = data.cfg["curriculum_epochs"]

    def sampler(epoch: int, rng: np.random.Generator):
        origins = fixed_origins if fixed_origins is not None else (("shared",) if epoch < cur else None)
        if label_mode == "hard":
            pairs = hard_dynamic_pairs_soft(recs, margin, delta=delta, rng=rng, origins=origins)
            return apply_flips(pairs, seed=seed, rate=flip) if flip > 0 else pairs
        pairs = soft_dynamic_pairs(recs, margin, prob_fn, rng=rng, origins=origins)
        if label_mode == "soft":
            return [p.__class__(**{**p.__dict__, "w": 1.0}) for p in pairs]
        if label_mode == "weighted":
            return [p.__class__(**{**p.__dict__, "p": float(p.y_hard)}) for p in pairs]
        raise ValueError(label_mode)

    return sampler


def weighted_acc(probs: np.ndarray, pairs) -> dict:
    y = np.array([p.y_hard for p in pairs], dtype=float)
    w = np.array([p.w for p in pairs], dtype=float)
    pt = np.array([p.p for p in pairs], dtype=float)
    hit = (np.round(probs) == y).astype(float)
    pmax = np.maximum(pt, 1 - pt)
    mid = (pmax >= MID_LO) & (pmax < MID_HI)
    high = pmax >= MID_HI
    eps = 1e-7
    ll = -(pt * np.log(probs + eps) + (1 - pt) * np.log(1 - probs + eps))
    out = {"acc_w": float((w * hit).sum() / w.sum()) if w.sum() else float("nan"),
           "soft_logloss": float(ll.mean()),
           "mid_n": int(mid.sum()), "high_n": int(high.sum())}
    if mid.any():
        out["mid_acc"] = float(hit[mid].mean())
        out["mid_acc_norm"] = out["mid_acc"] / float(pmax[mid].mean())
        out["mid_acc_w"] = float((w[mid] * hit[mid]).sum() / w[mid].sum())
    if high.any():
        out["high_acc"] = float(hit[high].mean())
    return out


def bootstrap_delta(w, hit_a, hit_b, n_fwd: int, seed: int = 0) -> dict:
    """无序对级 bootstrap：Δ = acc_w(a) − acc_w(b)，正反向样本同随一个无序对。"""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(2 * n_fwd)
    w2, ha, hb = w[idx_all], hit_a[idx_all], hit_b[idx_all]
    deltas = []
    for _ in range(N_BOOT):
        u = rng.integers(0, n_fwd, size=n_fwd)
        sel = np.concatenate([u, u + n_fwd])
        ws = w2[sel]
        if ws.sum() <= 0:
            continue
        deltas.append(float(((ws * ha[sel]).sum() - (ws * hb[sel]).sum()) / ws.sum()))
    deltas = np.array(deltas)
    point = float(((w2 * ha).sum() - (w2 * hb).sum()) / w2.sum())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def soft_block(engine, soft_directed, n_fwd, test_tids, all_tids) -> dict:
    y = np.array([p.y_hard for p in soft_directed], dtype=float)
    w = np.array([p.w for p in soft_directed], dtype=float)
    p_norm = engine.predict(soft_directed)
    hit_norm = (np.round(p_norm) == y).astype(float)
    res = {"n_directed": len(soft_directed), "n_forward": n_fwd, "normal": weighted_acc(p_norm, soft_directed)}
    for mode in ("within", "cross"):
        rows = []
        for s in range(N_ABL):
            amap = (within_assignment(sorted(test_tids), s) if mode == "within"
                    else cross_assignment(sorted(test_tids), all_tids, s))
            p_abl = engine.predict(soft_directed, e_mode="assign", e_assignment=amap)
            hit_abl = (np.round(p_abl) == y).astype(float)
            m = weighted_acc(p_abl, soft_directed)
            rows.append({"seed": s, **m, "delta_acc_w": res["normal"]["acc_w"] - m["acc_w"],
                         "delta_mid_acc": res["normal"].get("mid_acc", np.nan) - m.get("mid_acc", np.nan),
                         "boot": bootstrap_delta(w, hit_norm, hit_abl, n_fwd, seed=s)})
        res[mode] = {"rows": rows,
                     "mean_delta_acc_w": float(np.mean([r["delta_acc_w"] for r in rows])),
                     "mean_delta_mid_acc": float(np.nanmean([r["delta_mid_acc"] for r in rows])),
                     "n_ci_excl_zero_positive": int(sum(r["boot"]["ci95"][0] > 0 for r in rows))}
    p_zero = engine.predict(soft_directed, e_mode="zero")
    res["zero"] = {**weighted_acc(p_zero, soft_directed),
                   "delta_acc_w": res["normal"]["acc_w"] - weighted_acc(p_zero, soft_directed)["acc_w"]}
    return res


def kind_subset_acc(engine, directed, n_fwd) -> dict:
    """硬评估集按配对种类（shared-shared / random 参与）分组 acc（Q2）。"""
    probs = engine.predict(directed)
    y = np.array([p.y for p in directed], dtype=float)
    hit = np.round(probs) == y
    kinds = np.array([p.kind for p in directed])
    out = {}
    for k in sorted(set(kinds)):
        m = kinds == k
        out[k] = {"n": int(m.sum()), "acc": float(hit[m].mean())}
    rnd = kinds != "shared-shared"
    out["random_involved"] = {"n": int(rnd.sum()), "acc": float(hit[rnd].mean()) if rnd.any() else float("nan")}
    if "shared-shared" in out and rnd.any():
        out["gap_random_minus_shared"] = out["random_involved"]["acc"] - out["shared-shared"]["acc"]
    return out


def run_arm(arm_cfg, data: WorkpointData, pm_t: ProbModel, pm_e: ProbModel, device: str) -> dict:
    arm, seed, label_mode, kind, fixed_origins, flip, enc_kind = arm_cfg
    spec = [s for s in build_runs(data.all_tids) if s["run_id"] == f"R{seed + 1}_mixed_s{seed}"][0]
    train_tids, test_tids = spec["train"], spec["test"]
    assert not (train_tids & test_tids)
    margin = margin_for(data.task_type)
    prob_train = pm_e if kind == "empirical" else pm_t
    torch.manual_seed(seed)
    set_enc = (MeanPoolTaskEncoder(dim_input=data.dim_input) if enc_kind == "meanpool"
               else load_set_encoder_init(data))
    model = TCLSC(cls_dim=data.cls_dim, set_encoder=set_enc,
                  embedding_dim=data.cfg["embedding_dim"], fz=data.cfg["fz"])
    engine = SoftTCLSCEngine(model, data.task_windows, device=device, lr=data.cfg["lr"],
                             batch_size=data.cfg["batch_size"], seed=seed)
    sampler = make_sampler(arm_cfg, data, train_tids, prob_train, margin)
    t0 = time.time()
    hist = engine.train_soft(sampler, epochs=data.cfg["epochs"])
    secs = round(time.time() - t0, 1)
    # 硬评估集（C2 同口径）
    directed, n_fwd = data.hard_test_pairs(test_tids)
    hard = ablation_block(engine, directed, n_fwd, test_tids, data.all_tids)
    # 软评估集（δ=0，权重统一用 Student-t 模型，跨臂可比）
    soft_fwd = soft_pairs_all(data.records_in(test_tids), margin, pm_t)
    soft_directed = with_reversed_soft(soft_fwd)
    soft = soft_block(engine, soft_directed, len(soft_fwd), test_tids, data.all_tids)
    return {"arm": arm, "seed": seed, "label_mode": label_mode, "noise_kind": kind,
            "fixed_origins": fixed_origins, "flip_rate": flip, "encoder": enc_kind,
            "train_tids": sorted(train_tids), "test_tids": sorted(test_tids),
            "n_train_records": len(data.records_in(train_tids)), "train_seconds": secs, "device": device,
            "loss_first": hist[0]["loss"], "loss_last": hist[-1]["loss"],
            "train_pairs_per_epoch_last": hist[-1]["n_pairs"], "train_w_mean_last": hist[-1]["w_mean"],
            "hard": hard, "soft": soft, "hard_by_kind": kind_subset_acc(engine, directed, n_fwd)}


def fmt(res: dict) -> str:
    h, s = res["hard"], res["soft"]
    return (f"{res['train_seconds']}s | hard normal {h['normal']['acc']:.4f} "
            f"within Δ̄={h['within']['mean_delta']:+.4f} sig {h['within']['n_sig']}/5 "
            f"cross Δ̄={h['cross']['mean_delta']:+.4f} sig {h['cross']['n_sig']}/5 "
            f"zero Δ={h['zero']['delta_acc']:+.4f} | soft acc_w {s['normal']['acc_w']:.4f} "
            f"mid {s['normal'].get('mid_acc', float('nan')):.4f} "
            f"within Δw̄={s['within']['mean_delta_acc_w']:+.4f} ci+ {s['within']['n_ci_excl_zero_positive']}/5")


def stage_train(data, pm_t, pm_e, device, resume, only) -> None:
    runs_dir = OUT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for cfg in ARMS:
        rid = f"{cfg[0]}_s{cfg[1]}"
        if only and rid not in only and cfg[0] not in only:
            continue
        out_path = runs_dir / f"{rid}.json"
        if resume and out_path.exists():
            print(f"[{rid}] 已有结果，跳过", flush=True)
            continue
        print(f"[{rid}] {cfg[2]}/{cfg[3]} flip={cfg[5]} enc={cfg[6]} ...", flush=True)
        res = run_arm(cfg, data, pm_t, pm_e, device)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{rid}] {fmt(res)}", flush=True)


# ---------------------------------------------------------------- §5 判读


def stage_verdict() -> dict:
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))}
    ceiling = json.loads((OUT / "ceiling.json").read_text(encoding="utf-8"))
    noise = json.loads((OUT / "noise_model.json").read_text(encoding="utf-8"))

    def arm_runs(arm):
        return [r for k, r in runs.items() if k.startswith(f"{arm}_s")]

    def summarize(arm):
        rs = arm_runs(arm)
        if not rs:
            return None
        med = sorted(rs, key=lambda r: r["hard"]["within"]["mean_delta"])[len(rs) // 2]
        h = med["hard"]
        soft_w = med["soft"]["within"]
        return {"n_runs": len(rs), "median_seed": med["seed"],
                "hard_normal_acc_mean": float(np.mean([r["hard"]["normal"]["acc"] for r in rs])),
                "hard_normal_acc_median_run": h["normal"]["acc"],
                "within": {k: h["within"][k] for k in ("mean_delta", "n_sig", "pass")},
                "cross": {k: h["cross"][k] for k in ("mean_delta", "n_sig", "pass")},
                "within_tier": tier_of(h["within"]), "cross_tier": tier_of(h["cross"]),
                "zero_delta": h["zero"]["delta_acc"],
                "soft_acc_w_mean": float(np.mean([r["soft"]["normal"]["acc_w"] for r in rs])),
                "soft_mid_acc_mean": float(np.mean([r["soft"]["normal"].get("mid_acc", np.nan) for r in rs])),
                "soft_within_delta_acc_w": soft_w["mean_delta_acc_w"],
                "soft_within_ci_pos": soft_w["n_ci_excl_zero_positive"],
                "soft_within_tier_detectable": bool(soft_w["n_ci_excl_zero_positive"] >= 3
                                                    and soft_w["mean_delta_acc_w"] > 0)}

    arms = {a: summarize(a) for a in ("H0", "S1", "S2", "S1e", "Cshared", "X")}
    # Q1
    q1 = {}
    for a in ("S1", "S2"):
        s = arms[a]
        if s is None:
            q1[a] = "incomplete"
            continue
        tier = s["within_tier"]
        if tier == "zero" and s["soft_within_tier_detectable"]:
            tier = "detectable(soft-metric)"
        q1[a] = tier
    q1_read = ("zero" if all(v == "zero" for v in q1.values()) else
               ("pass" if any(v == "pass" for v in q1.values()) else "detectable/ambiguous"))
    # Q2
    q2 = None
    cs = arm_runs("Cshared")
    if cs:
        gaps = [r["hard_by_kind"].get("gap_random_minus_shared", np.nan) for r in cs]
        q2 = {"gap_mean": float(np.nanmean(gaps)), "gaps": gaps,
              "shared_acc": [r["hard_by_kind"]["shared-shared"]["acc"] for r in cs],
              "random_acc": [r["hard_by_kind"]["random_involved"]["acc"] for r in cs],
              "assumption_holds": bool(np.nanmean(gaps) >= -0.05)}
    # Q3a
    st = ceiling["soft_student_t"]["pooled"]["ceiling"]
    q3a = {"soft_student_t_pooled": st, "hard_delta_pooled": ceiling["hard_delta"]["pooled"]["ceiling"],
           "delta_zero_pooled": ceiling["delta_zero"]["pooled"]["ceiling"],
           "midband_pooled": ceiling["soft_student_t_midband"]["pooled"]["ceiling"],
           "midband_mass_share": (ceiling["soft_student_t_midband"]["pooled"]["mass"]
                                  / ceiling["soft_student_t"]["pooled"]["mass"]),
           "reads": "soft label 集含被 δ 弃掉的任务依赖质量" if st < 0.95 else "中带也任务盲",
           "reproduction": ceiling["reproduction_check"]}
    # Q3b flip 曲线（seed 0/1 与 H0 同种子）
    curve = {}
    h0 = [r for r in arm_runs("H0") if r["seed"] in (0, 1)]
    if h0:
        curve[0.0] = float(np.mean([r["hard"]["normal"]["acc"] for r in h0]))
    for f in FLIP_RATES:
        rs = arm_runs(f"F{int(f * 100)}")
        if rs:
            curve[f] = float(np.mean([r["hard"]["normal"]["acc"] for r in rs]))
    f_star = None
    if 0.0 in curve and len(curve) > 1:
        base = curve[0.0]
        xs = sorted(curve)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            d0, d1 = base - curve[x0], base - curve[x1]
            if d1 >= 0.02:
                f_star = x0 + (x1 - x0) * (0.02 - d0) / (d1 - d0) if d1 != d0 else x1
                break
        if f_star is None:
            f_star = f">{xs[-1]}"
    f_eff = noise["f_eff_hard_delta"]["mean"]
    q3b = {"curve": {str(k): v for k, v in curve.items()}, "f_star": f_star, "f_eff_hard_delta": f_eff,
           "reads": (None if not isinstance(f_star, float) and not isinstance(f_star, str) else
                     ("label 噪声不是比较器瓶颈（f* > f_eff）" if (isinstance(f_star, str) or f_star > f_eff)
                      else "label 噪声在比较器可感范围（f* ≤ f_eff）"))}
    # X
    x = arms["X"]
    x_read = None
    if x and arms["S1"]:
        order = {"zero": 0, "ambiguous": 0.5, "detectable": 1, "pass": 2}
        x_read = {"X_tier": x["within_tier"], "S1_tier": arms["S1"]["within_tier"],
                  "X_vs_S1": ("higher" if order[x["within_tier"]] > order[arms["S1"]["within_tier"]]
                              else "lower" if order[x["within_tier"]] < order[arms["S1"]["within_tier"]] else "same")}
    verdict = {"arms": arms, "Q1": {"per_arm": q1, "read": q1_read}, "Q2": q2, "Q3a": q3a, "Q3b": q3b, "X": x_read,
               "flip_runs_detail": {k: {"seed": r["seed"], "acc": r["hard"]["normal"]["acc"]}
                                    for k, r in runs.items() if k.startswith("F")}}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in verdict.items() if k != "arms"}, indent=2, ensure_ascii=False))
    for a, s in arms.items():
        if s:
            print(f"[verdict] {a:8s} n={s['n_runs']} normal {s['hard_normal_acc_mean']:.4f} | within {s['within']['mean_delta']:+.4f} "
                  f"sig {s['within']['n_sig']} → {s['within_tier']} | cross {s['cross']['mean_delta']:+.4f} → {s['cross_tier']} | "
                  f"soft acc_w {s['soft_acc_w_mean']:.4f} mid {s['soft_mid_acc_mean']:.4f} within Δw {s['soft_within_delta_acc_w']:+.4f} ci+ {s['soft_within_ci_pos']}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--noise-model", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="只跑指定 arm 或 arm_s<seed>")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="仅冒烟：覆盖训练 epoch 数")
    ap.add_argument("--outdir", type=Path, default=None, help="仅冒烟：改写输出目录")
    args = ap.parse_args()
    global OUT
    if args.outdir is not None:
        OUT = args.outdir
    if not any((args.noise_model, args.train, args.verdict)):
        ap.error("至少指定一个阶段")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.noise_model or args.train:
        data = WorkpointData()
        if args.epochs is not None:
            data.cfg["epochs"] = args.epochs
            data.cfg["curriculum_epochs"] = min(data.cfg["curriculum_epochs"], args.epochs // 2)
        print(f"记录 {len(data.records)} / 任务 {len(data.all_tids)} | 单向硬评估对全池 {len(data.fwd_pairs)} | "
              f"δ={data.delta_map} | device={args.device}", flush=True)
        pm_t, pm_e = stage_noise_model(data)
        if args.noise_model:
            stage_ceiling(data, pm_t, pm_e)
        if args.train:
            stage_train(data, pm_t, pm_e, args.device, args.resume, args.only)
    if args.verdict:
        stage_verdict()


if __name__ == "__main__":
    main()
