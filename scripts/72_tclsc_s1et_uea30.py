"""WP16 U 检验：官方口径 UEA 靶外 23 靶——E_T T-CLSC 名单 + SH6-3-3 vs 任务盲 T-CLSC 名单（消融）vs 先验名单。协议 results/tclsc_s1et_uea30/PROTOCOL.md。

子命令：
  --train-all8  源池 8 集全部 45 切片（mx3_pq）训 blind_diff_hard / ft60_hard（无折，3 种子）→ models/all8/
  --features    23 靶官方 TRAIN：GGS 探针 @60 的 E_T（22 维）+ φ 部分列 → et_uea30.jsonl
  --simulate    三名单（限信任域 shared-10 + GGS，E_T ctx 裁 ±4）× SH6-3-3 于 WP10 目录（seed 0）→ sim.json + picks.json（缺 seeds {1,2} 的冠军，45 号 --finals 用）
  --analyze     合并 finals（results/uea30 + 本目录）→ 3 种子、门 C0 / G-U1 / G-U2 / G-U3、读数码 → analysis.json / tables.md
finals：python scripts/45_uea30_deploy.py --finals --resume --outdir results/tclsc_s1et_uea30（pod）。
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import TCLSCEngine  # noqa: E402
from autocls.data.mech_features import REP_PROFILE_COLS, rep_cols_at  # noqa: E402
from autocls.space.cls_space import CLS  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g39 = importlib.import_module("scripts.39_sh_deploy")
_g40 = importlib.import_module("scripts.40_task_feature_scorer")
_g45 = importlib.import_module("scripts.45_uea30_deploy")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g69 = importlib.import_module("scripts.69_tclsc_s1et")
PHI_COLS = tuple(_m33.PHI_COLS)
OUT = ROOT / "results" / "tclsc_s1et_uea30"
UEA30 = ROOT / "results" / "uea30"
SOURCE = pq.OUT
ALL8_FOLD = "all8"
ARMS = ("blind_diff_hard", "ft60_hard")
LISTS = ("prior", "blind_tclsc", "et_tclsc")
SH_KS = (6, 3, 3)
CLIP = 4.0
DEPLOY_BUDGET = 290
SEEDS = (0, 1, 2)
FINAL_SEEDS = {1, 2}
ALPHA = 0.05
MATCH_MARGIN = 0.03
EIGEN_CROP = 3000


# ---------------------------------------------------------------- all8 训练
def _all8_records(rows: Sequence[dict]):
    enc = _g60.enc()
    return [_g60.ClsRecord(task_id=pq.engine_task_id(r["task_id"], int(r["iters"])), key=r["cls_key"], origin=r["origin"],
                                 vec=enc.encode(CLS(**r["cls"])), r=float(r["r_prime"])) for r in rows if r["origin"] != "ggs"]


def train_all8(outdir: Path, table: Mapping[str, Mapping], *, seeds: Sequence[int], device: str, epochs: Optional[int],
               arms: Sequence[str] = ARMS, source: Path = SOURCE) -> None:
    """源池全部切片（无折）训 blind_diff_hard / ft60_hard；ctx = 切片表（E_T@60 ⊕ φ）在全部切片上标准化。"""
    rows, delta = _g57.load_source(source)
    recs = _all8_records(rows)
    slices = sorted({r["task_id"] for r in rows})
    c = dict(_g60.cfg())
    if epochs:
        c["epochs"], c["curriculum_epochs"] = int(epochs), min(int(c["curriculum_epochs"]), max(1, int(epochs) // 2))
    cls_dim = int(recs[0].vec.shape[0])
    delta_map = pq.delta_map_for_fold(delta, slices, pq.BUDGETS)
    for arm in arms:
        spec = _m33.arm_spec(arm)
        ctx_map, meta_extra, cdim = None, None, 0
        if spec.ctx_kind == "table_budget":
            cols = tuple(spec.ctx_cols)
            mu, sd = _m33.table_standardizer(table, slices, cols)
            ctx_map = {}
            for s in slices:
                x, _n = _m33.table_row_vector(table[s], cols, mu, sd)
                for b in pq.BUDGETS:
                    ctx_map[pq.engine_task_id(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), pq.BUDGETS)]).astype(np.float32)
            meta_extra = {"ctx_cols": list(cols), "ctx_mu": [float(v) for v in mu], "ctx_sd": [float(v) for v in sd], "n_nan": {}}
            cdim = len(cols) + len(pq.BUDGETS)
        for seed in seeds:
            p = outdir / "models" / ALL8_FOLD / f"{arm}_s{seed}.pt"
            if p.exists():
                continue
            model = _m33.build_model(spec, cls_dim=cls_dim, ctx_dim=cdim, nz=256, dim_input=256, cfg=c, seed=seed)
            engine = TCLSCEngine(model, {}, device=device, lr=c["lr"], batch_size=c["batch_size"], seed=seed, context_of=ctx_map)
            t0 = time.time()
            hist = engine.train(recs, epochs=c["epochs"], curriculum_epochs=c["curriculum_epochs"],
                                better_fn=lambda a, b: is_better(a, b, TASK_CLASSIFICATION), delta=delta_map)
            _m33.save_model(p, model, spec, fold=ALL8_FOLD, seed=seed, cls_dim=cls_dim, ctx_dim=cdim, cfg=c, nz=256, settf_windows=0, settf_t_max=0,
                            budgets=pq.BUDGETS, ctx_encoding="onehot", encoder_mode="binned", head=spec.head, ctx_kind=spec.ctx_kind, phi_stats=meta_extra)
            run = {"fold": ALL8_FOLD, "arm": arm, "seed": seed, "n_train_records": len(recs), "train_seconds": round(time.time() - t0, 1),
                   "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"], "train_pairs_last": hist[-1]["n_pairs"], "ctx_dim": cdim}
            rp = outdir / "runs" / ALL8_FOLD / f"{arm}_s{seed}.json"
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[all8/{arm}_s{seed}] {run['train_seconds']}s bce {run['bce_first']:.4f}→{run['bce_last']:.4f} pairs {run['train_pairs_last']}", flush=True)


def cmd_train_all8(*, seeds: Sequence[int], device: str, epochs: Optional[int]) -> None:
    table = {r["task_id"]: r for r in read_jsonl(ROOT / "results" / "tclsc_s1et" / "uea" / "ctx_uea.jsonl")}
    train_all8(OUT, table, seeds=seeds, device=device, epochs=epochs)


# ---------------------------------------------------------------- 特征
def phi_partial(Xtr: np.ndarray, ytr: np.ndarray) -> dict:
    """40 号 official_phi 的数组部分（meta + signal_stats）；base_mean / base_band 无 iters0 基线 → None。"""
    return {**_g40.phi_meta(Xtr.shape[1], Xtr.shape[2], int(len(np.unique(ytr))), Xtr.shape[0]), "base_mean": None, "base_band": None,
            **_g40.signal_stats(np.asarray(Xtr, np.float32))}


def cmd_features(*, device: str, resume: bool, datasets: Optional[Sequence[str]]) -> None:
    cfg = _g45.load_cfg()
    targets = list(datasets) if datasets else [d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "et_uea30.jsonl"
    done = {r["dataset"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
    with open(path, "a", encoding="utf-8") as fh:
        for d in targets:
            if d in done:
                continue
            arr = _g45.uea30_arrays(d, cfg)
            Xtr, ytr = np.asarray(arr["Xtr"], np.float32), np.asarray(arr["ytr"])
            m = ((cfg.get("datasets") or {}).get(d) or {}).get("max_train_length")
            if m and Xtr.shape[1] > int(m):
                Xtr = Xtr[:, : int(m)]
            t0 = time.time()
            prof, _acc = _g69.probe_profiles(Xtr, ytr, TASK_CLASSIFICATION, device=device)
            row = _g69.feature_row(f"{d}_official", d, prof, seconds=time.time() - t0, acc290=None)
            row.update(phi_partial(Xtr, ytr))
            row.update({"n_train": int(len(ytr)), "T_used": int(Xtr.shape[1]), "C": int(Xtr.shape[2])})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[et uea30] {d}: n={len(ytr)} T={Xtr.shape[1]} C={Xtr.shape[2]} {row['seconds']}s rinfo@60 {row['rinfo@60']:+.2f}", flush=True)


# ---------------------------------------------------------------- 目录 / 名单 / SH
def target_catalog(rows: Sequence[dict], target: str):
    """(traj{key: {截点: acc}}, finals{key: [seed0, s2…]}, budgets, keys_off = 非孪生且截点齐全的键)。"""
    s1 = [r for r in rows if r["dataset"] == target and r.get("stage") == "s1_catalog" and r.get("error") is None and r.get("checkpoint_results")]
    traj, _secs = _g39.trajectory_from_rows(s1)
    budgets = tuple(sorted({b for r in s1 if not r.get("twin_of") for b in traj.get(r["cls_key"], {})}))
    keys_off = sorted(k for k in {r["cls_key"] for r in s1 if not r.get("twin_of")} if all(b in traj.get(k, {}) for b in budgets))
    finals: dict[str, list[float]] = {}
    for r in s1:
        finals.setdefault(r["cls_key"], []).insert(0, float(r["acc"]))
    for r in rows:
        if r["dataset"] == target and r.get("stage") == "s2_finals" and r.get("error") is None:
            finals.setdefault(r["cls_key"], []).append(float(r["acc"]))
    return {k: dict(v) for k, v in traj.items()}, finals, budgets, keys_off


def sh_trust(order: Sequence[str], traj: Mapping, allowed: Sequence[str], budgets: Sequence[int], *, ks: Sequence[int] = SH_KS) -> dict:
    """信任域 = allowed（shared-10 + GGS 且截点齐全）；名单只保留其中的键，顺序不变。"""
    ranked = [k for k in order if k in set(allowed)]
    final, cost = _g36.successive_halving(ranked, traj, ks, budgets)
    champ = _g36.sh_pick(final, traj, budgets[-1])
    return {"champion": champ, "acc": float(traj[champ][int(budgets[-1])]), "cost_iters": int(cost), "list": ranked[: int(ks[0])]}


def picks_for(champs: Mapping[str, Mapping[str, str]], have: Mapping[tuple, set]) -> dict:
    """{靶: {SH6-3-3_<名单>: 冠军}}——只列缺 seeds {1,2} 的。"""
    out: dict = {}
    for d, per in champs.items():
        for name, k in per.items():
            if not FINAL_SEEDS <= set(have.get((d, k), set())):
                out.setdefault(d, {})[f"SH6-3-3_{name}"] = k
    return out


def _load_all8(arm: str, device: str = "cpu"):
    paths = [OUT / "models" / ALL8_FOLD / f"{arm}_s{s}.pt" for s in SEEDS]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"all8/{arm} 模型不全（先 --train-all8）")
    return [_g34.load_pq_model(p, map_location=device, expect_fold=ALL8_FOLD, expect_budgets=pq.BUDGETS) for p in paths]


def cmd_simulate(*, device: str = "cpu") -> dict:
    rows = list(read_jsonl(UEA30 / "runs.jsonl"))
    cands = _g39.load_candidates(UEA30)
    et_rows = {r["dataset"]: r for r in read_jsonl(OUT / "et_uea30.jsonl")}
    blind, ft = _load_all8("blind_diff_hard", device), _load_all8("ft60_hard", device)
    meta0 = ft[0][1]
    cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
    have: dict[tuple, set] = {}
    for r in rows:
        if r.get("stage") == "s2_finals" and r.get("error") is None:
            have.setdefault((r["dataset"], r["cls_key"]), set()).add(int(r["seed"]))
    if (OUT / "runs.jsonl").exists():
        for r in read_jsonl(OUT / "runs.jsonl"):
            if r.get("stage") == "s2_finals" and r.get("error") is None:
                have.setdefault((r["dataset"], r["cls_key"]), set()).add(int(r["seed"]))
    per, champs = {}, {}
    for d in sorted(et_rows):
        traj, finals, budgets, keys_off = target_catalog(rows, d)
        c = cands[d]
        keys_cls = {e["cls_key"]: CLS(**e["cls"]) for e in list(c["shared"]) + [c["ggs"]] if e["cls_key"] in keys_off}
        slice_id = f"{d}_official"
        cat = {slice_id: [("official", cl, k) for k, cl in sorted(keys_cls.items())]}
        rk_b = _g56.rank_local_models({"blind_diff_hard": blind}, cat, budgets=pq.BUDGETS, device=device)[str(DEPLOY_BUDGET)][slice_id]["blind_diff_hard"]
        et = {cc: float(et_rows[d][f"{cc}@60"]) for cc in REP_PROFILE_COLS}
        phi = {cc: et_rows[d].get(cc) for cc in PHI_COLS}
        table = {**{f"{cc}@60": v for cc, v in et.items()}, **phi}
        x, n_nan = _m33.table_row_vector(table, cols, mu, sd)
        x = np.clip(x, -CLIP, CLIP)
        rows_ctx = {(slice_id, int(b)): np.concatenate([x, pq.ctx_vector(int(b), pq.BUDGETS)]).astype(np.float32) for b in pq.BUDGETS}
        rk_e = _g56.rank_local_models({"ft60_hard": ft}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_ctx)[str(DEPLOY_BUDGET)][slice_id]["ft60_hard"]
        orders = {"prior": list(c["prior_order"]), "blind_tclsc": rk_b, "et_tclsc": rk_e}
        oracle = max(sorted(keys_cls), key=lambda k: traj[k][budgets[-1]])          # 信任域内 oracle（孪生不参与）
        res = {"budgets": list(budgets), "oracle": {"key": oracle, "acc": float(traj[oracle][budgets[-1]])}, "n_nan": int(n_nan),
               "ctx_abs_max_raw": float(np.abs(_m33.table_row_vector(table, cols, mu, sd)[0]).max()), "lists": {}}
        for name, order in orders.items():
            r = sh_trust(order, traj, sorted(keys_cls), budgets)
            r["oracle_rank"] = (r["list"].index(oracle) + 1) if oracle in r["list"] else None
            r["hit6"] = oracle in r["list"]
            r["finals_seed0"] = finals.get(r["champion"], [])
            res["lists"][name] = r
            champs.setdefault(d, {})[name] = r["champion"]
        per[d] = res
    picks = picks_for(champs, have)
    sim = {"targets": per, "picks_pending": picks, "clip": CLIP}
    (OUT / "sim.json").write_text(json.dumps(sim, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (OUT / "candidates.json").exists():
        (OUT / "candidates.json").write_text((UEA30 / "candidates.json").read_text(encoding="utf-8"), encoding="utf-8")
    n_cells = sum(len(v) for v in picks.values())
    same = sum(1 for d in per if len({per[d]["lists"][n]["champion"] for n in LISTS}) == 1)
    print(f"[simulate] {len(per)} 靶；三名单冠军全同 {same}；待补 finals {n_cells} 格（{len(picks)} 靶）；seed0 均值 " +
          " / ".join(f"{n} {np.mean([per[d]['lists'][n]['acc'] for d in per]):.3f}" for n in LISTS), flush=True)
    return sim


# ---------------------------------------------------------------- 门 / 分析
def gates_uea30(mean_et: float, mean_blind: float, n_not_worse: int, mean_prior: float, hit_et: int, hit_prior: int, v3: Mapping) -> dict:
    return {"GU1": {"et": mean_et, "blind": mean_blind, "n_not_worse": n_not_worse, "pass": bool(mean_et >= mean_blind + 0.005 and n_not_worse >= 15)},
            "GU2": {"et": mean_et, "prior": mean_prior, "hit_et": hit_et, "hit_prior": hit_prior, "pass": bool(mean_et >= mean_prior and hit_et >= hit_prior)},
            "GU3": {"blind_3seed": v3.get("mean_a"), "et_3seed": v3.get("mean_b"), "p": v3.get("p"),
                    "pass": (None if v3.get("p") is None else bool(v3["p"] < ALPHA and v3["mean_b"] > v3["mean_a"]))}}


def verdict_uea30(g: Mapping) -> str:
    code = "UEA30-ET-FLAT" if not g["GU1"]["pass"] else ("UEA30-ET-WIN" if g["GU3"]["pass"] else "UEA30-ET-BETTER")
    return code + ("+PRIOR" if g["GU2"]["pass"] else "-PRIOR")


def cmd_analyze() -> dict:
    sim = json.loads((OUT / "sim.json").read_text(encoding="utf-8"))
    rows = list(read_jsonl(UEA30 / "runs.jsonl")) + ([r for r in read_jsonl(OUT / "runs.jsonl")] if (OUT / "runs.jsonl").exists() else [])
    cfg = _g45.load_cfg()
    ref = json.loads((UEA30 / "analysis.json").read_text(encoding="utf-8"))["per_target"]
    per = sim["targets"]
    targets = sorted(per)
    out = {}
    for d in targets:
        _t, finals, _b, _k = target_catalog(rows, d)
        paper = ((cfg["datasets"].get(d) or {}).get("literature") or {}).get("TS2Vec")
        row = {"class": (cfg["datasets"].get(d) or {}).get("class"), "paper_ts2vec": paper, "oracle": per[d]["oracle"]["acc"], "lists": {}}
        for n in LISTS:
            r = dict(per[d]["lists"][n])
            fs = finals.get(r["champion"], [])
            r["finals"] = fs
            r["mean3"] = float(np.mean(fs)) if len(fs) >= 3 else None
            r["match_seed0"] = (bool(r["acc"] >= paper - MATCH_MARGIN) if paper is not None else None)
            r["match_3seed"] = (bool(r["mean3"] >= paper - MATCH_MARGIN) if (paper is not None and r["mean3"] is not None) else None)
            row["lists"][n] = r
        row["c0_prior_sh6"] = {"ref": ref.get(d, {}).get("arms", {}).get("SH6-3-3_prior"), "got": row["lists"]["prior"]["acc"]}
        out[d] = row
    c0 = all(abs(o["c0_prior_sh6"]["ref"] - o["c0_prior_sh6"]["got"]) <= 1e-6 for o in out.values() if o["c0_prior_sh6"]["ref"] is not None)
    mean0 = {n: float(np.mean([out[d]["lists"][n]["acc"] for d in targets])) for n in LISTS}
    hit = {n: int(sum(out[d]["lists"][n]["hit6"] for d in targets)) for n in LISTS}
    n_not_worse = int(sum(out[d]["lists"]["et_tclsc"]["acc"] >= out[d]["lists"]["blind_tclsc"]["acc"] - 1e-9 for d in targets))
    have3 = [d for d in targets if all(out[d]["lists"][n]["mean3"] is not None for n in ("blind_tclsc", "et_tclsc"))]
    if len(have3) == len(targets):
        x = np.array([out[d]["lists"]["blind_tclsc"]["mean3"] for d in targets]); y = np.array([out[d]["lists"]["et_tclsc"]["mean3"] for d in targets])
        v3 = {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "p": float(_g56.paired_wilcoxon_less(x, y)), "n": len(targets),
              "wins": int(np.sum(y > x)), "losses": int(np.sum(y < x)), "ties": int(np.sum(y == x))}
    else:
        v3 = {"mean_a": None, "mean_b": None, "p": None, "n_have": len(have3)}
    g = gates_uea30(mean0["et_tclsc"], mean0["blind_tclsc"], n_not_worse, mean0["prior"], hit["et_tclsc"], hit["prior"], v3)
    mean3 = {n: (float(np.mean([out[d]["lists"][n]["mean3"] for d in targets])) if all(out[d]["lists"][n]["mean3"] is not None for d in targets) else None) for n in LISTS}
    match3 = {n: (int(sum(out[d]["lists"][n]["match_3seed"] for d in targets if out[d]["lists"][n]["match_3seed"] is not None)),
                  int(sum(out[d]["lists"][n]["match_3seed"] is not None for d in targets))) for n in LISTS}
    same = int(sum(len({out[d]["lists"][n]["champion"] for n in LISTS}) == 1 for d in targets))
    by_class: dict = {}
    for d in targets:
        by_class.setdefault(out[d]["class"], []).append(d)
    cls_means = {k: {n: float(np.mean([out[d]["lists"][n]["acc"] for d in ds])) for n in LISTS} for k, ds in sorted(by_class.items())}
    e_cls = by_class.get("E", [])
    n_e_ge = int(sum(out[d]["lists"]["et_tclsc"]["acc"] >= out[d]["lists"]["blind_tclsc"]["acc"] - 1e-9 for d in e_cls))
    a = {"protocol": "results/tclsc_s1et_uea30/PROTOCOL.md", "n_targets": len(targets), "targets": out, "mean_seed0": mean0, "mean_3seed": mean3,
         "hit6": hit, "match_3seed": match3, "n_same_champion": same, "by_class": cls_means, "c0": c0, "gates": g, "verdict": verdict_uea30(g),
         "predictions": {"P1": {"desc": "C0", "hit": c0}, "P2": {"desc": "G-U1", "hit": g["GU1"]["pass"]}, "P3": {"desc": "G-U2", "hit": g["GU2"]["pass"]},
                         "P4": {"desc": "G-U3", "hit": g["GU3"]["pass"]}, "P5": {"desc": "三名单冠军全同 ≥ 12/23", "value": same, "hit": bool(same >= 12)},
                         "P6": {"desc": "E 类 E_T ≥ 任务盲 ≥ 4/5", "value": [n_e_ge, len(e_cls)], "hit": bool(n_e_ge >= 4)}}}
    (OUT / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"{a['verdict']} | seed0 " + " / ".join(f"{n} {mean0[n]:.3f}" for n in LISTS) + " | hit6 " + " / ".join(f"{hit[n]}" for n in LISTS)
          + f" | 3seed p={v3.get('p')} | 同冠军 {same}/{len(targets)}", flush=True)
    return a


def markdown_tables(a: dict) -> str:
    names = {"prior": "先验", "blind_tclsc": "任务盲 T-CLSC", "et_tclsc": "E_T T-CLSC"}
    L = [f"# U 检验 tables：UEA 靶外 {a['n_targets']} 靶，限信任域 SH6-3-3（seed 0；括号 3 种子均值）", "", f"读数码 **{a['verdict']}**", "",
         "| 靶 | 类 | TS2Vec 论文 | oracle | " + " | ".join(names[n] for n in LISTS) + " | E_T − 任务盲 | hit@6（先验/盲/E_T） |", "|---|---|---:|---:|" + "---:|" * len(LISTS) + "---:|---|"]
    for d, o in a["targets"].items():
        cells = []
        for n in LISTS:
            r = o["lists"][n]
            cells.append(f"{r['acc']:.3f}" + (f" ({r['mean3']:.3f})" if r["mean3"] is not None else ""))
        paper = "—" if o["paper_ts2vec"] is None else "%.3f" % o["paper_ts2vec"]
        L.append(f"| {d} | {o['class']} | {paper} | {o['oracle']:.3f} | " + " | ".join(cells)
                 + f" | {o['lists']['et_tclsc']['acc'] - o['lists']['blind_tclsc']['acc']:+.3f} | " + "/".join("✓" if o["lists"][n]["hit6"] else "✗" for n in LISTS) + " |")
    L.append("| **均值** | | | | " + " | ".join(f"{a['mean_seed0'][n]:.3f}" + (f" ({a['mean_3seed'][n]:.3f})" if a["mean_3seed"][n] is not None else "") for n in LISTS)
             + f" | {a['mean_seed0']['et_tclsc'] - a['mean_seed0']['blind_tclsc']:+.3f} | " + "/".join(str(a["hit6"][n]) for n in LISTS) + " |")
    L += ["", "## 按 WP10 类别的 seed 0 均值", "", "| 类 | " + " | ".join(names[n] for n in LISTS) + " |", "|---|" + "---:|" * len(LISTS)]
    for k, m in a["by_class"].items():
        L.append(f"| {k} | " + " | ".join(f"{m[n]:.3f}" for n in LISTS) + " |")
    g = a["gates"]
    L += ["", f"追平 TS2Vec（3 种子）：" + " / ".join(f"{names[n]} {a['match_3seed'][n][0]}/{a['match_3seed'][n][1]}" for n in LISTS), f"三名单冠军全同：{a['n_same_champion']}/{a['n_targets']}", "",
          "## 门", "", f"- C0 先验 SH6-3-3 seed 0 逐位：{a['c0']}", f"- G-U1 消融：E_T {g['GU1']['et']:.3f} vs 任务盲 {g['GU1']['blind']:.3f}，不低于 {g['GU1']['n_not_worse']}/{a['n_targets']} → {g['GU1']['pass']}",
          f"- G-U2 vs 先验：E_T {g['GU2']['et']:.3f} vs 先验 {g['GU2']['prior']:.3f}，hit@6 {g['GU2']['hit_et']} vs {g['GU2']['hit_prior']} → {g['GU2']['pass']}",
          f"- G-U3 3 种子配对：任务盲 {g['GU3']['blind_3seed']} → E_T {g['GU3']['et_3seed']}，p={g['GU3']['p']} → {g['GU3']['pass']}", "", "## 预测对账", ""]
    L += [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-all8", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.train_all8, args.features, args.simulate, args.analyze)):
        build_parser().error("至少指定一个子命令")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.train_all8:
        cmd_train_all8(seeds=args.seeds, device=args.device, epochs=args.epochs)
    if args.features:
        cmd_features(device=args.device, resume=args.resume, datasets=args.datasets)
    if args.simulate:
        cmd_simulate(device="cpu")
    if args.analyze:
        cmd_analyze()


if __name__ == "__main__":
    main()
