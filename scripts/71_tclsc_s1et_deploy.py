"""WP16 D 检验：官方口径零样本名单模拟——E_T T-CLSC（ft60）给六靶官方目录排名 → SH，与先验 + SH、任务盲 T-CLSC 名单、TS2Vec 论文数同表。零新 R'。
协议 results/tclsc_s1et_deploy/PROTOCOL.md。

子命令：--features（六靶官方 TRAIN 上 GGS 探针 @60/@290 的 E_T → et_official.jsonl）/ --simulate（排名 + SH 模拟 + 门 → analysis.json / tables.md）。
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

from autocls.data.mech_features import REP_PROFILE_COLS, rep_cols_at  # noqa: E402
from autocls.space.cls_space import CLS  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g69 = importlib.import_module("scripts.69_tclsc_s1et")
PHI_COLS = tuple(_m33.PHI_COLS)
OUT = ROOT / "results" / "tclsc_s1et_deploy"
DEPLOY = ROOT / "results" / "mx4_deploy"
PHI_OFFICIAL = ROOT / "results" / "ucr_module" / "phi_official.json"
ET_MODELS = ROOT / "results" / "tclsc_s1et" / "uea" / "models"
BLIND_MODELS = ROOT / "results" / "tclsc_s1phi" / "binned" / "models"
TARGETS = ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra", "EthanolConcentration", "Cricket", "HandMovementDirection")
TS2VEC_PAPER = {"Handwriting": 0.515, "UWaveGestureLibrary": 0.906, "PhonemeSpectra": 0.233, "EthanolConcentration": 0.308, "Cricket": 0.972, "HandMovementDirection": 0.338}
WP3_SH633_SEED0 = {"Handwriting": 0.419, "UWaveGestureLibrary": 0.931, "PhonemeSpectra": 0.251, "EthanolConcentration": 0.274, "Cricket": 1.000, "HandMovementDirection": 0.378}
MATCH_MARGIN = 0.03
SH_ARMS = {"SH6-3-3": (6, 3, 3), "SH10-4-3": (10, 4, 3)}
DEPLOY_BUDGET = 290
SEEDS = (0, 1, 2)
LISTS = ("prior", "blind_tclsc", "et_tclsc")
CLIP = 4.0                       # D′：ctx 裁剪（A″ 护栏）
WP3_FINALS_DIR = ROOT / "results" / "mx4_mask"      # WP3 + WP6 的 s2_finals（3 种子）也在这里


def restrict_list(order: Sequence[str], allowed) -> list[str]:
    """D′ 信任域：只保留 allowed（shared-10 + GGS）里的键，顺序不变。"""
    allowed = set(allowed)
    return [k for k in order if k in allowed]


# ---------------------------------------------------------------- 官方目录轨迹
def catalog_traj(rows: Sequence[dict], target: str) -> tuple[dict, dict, tuple]:
    """{cls_key: {截点: acc}}（s1_catalog seed 0）、{cls_key: [seed0 acc, s2_finals accs…]}、截点元组。"""
    traj, finals, budgets = {}, {}, None
    for r in rows:
        if r["dataset"] != target or r.get("error"):
            continue
        if r["stage"] == "s1_catalog":
            traj[r["cls_key"]] = {int(c["iters"]): float(c["acc"]) for c in r["checkpoint_results"]}
            budgets = tuple(int(b) for b in r["checkpoints"])
            finals.setdefault(r["cls_key"], []).insert(0, float(r["acc"]))
        elif r["stage"] == "s2_finals":
            finals.setdefault(r["cls_key"], []).append(float(r["acc"]))
    return traj, finals, budgets


def sh_on_list(ranked: Sequence[str], traj: Mapping, *, ks: Sequence[int], budgets: Sequence[int]) -> dict:
    ranked = [k for k in ranked if k in traj]
    final, cost = _g36.successive_halving(ranked, traj, ks, budgets)
    champ = _g36.sh_pick(final, traj, budgets[-1])
    return {"champion": champ, "acc": float(traj[champ][int(budgets[-1])]), "cost_iters": int(cost), "list": list(ranked[:int(ks[0])]), "final": list(final)}


# ---------------------------------------------------------------- E_T（官方 TRAIN）
def cmd_features(*, device: str, resume: bool) -> None:
    from autocls.data.uea import load_uea_pair
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "et_official.jsonl"
    done = {r["dataset"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
    with open(path, "a", encoding="utf-8") as fh:
        for d in TARGETS:
            if d in done:
                continue
            Xtr, ytr, _Xte, _yte = load_uea_pair(d, scaler="train")
            t0 = time.time()
            prof, _acc = _g69.probe_profiles(np.asarray(Xtr, np.float32), np.asarray(ytr), TASK_CLASSIFICATION, device=device)
            row = _g69.feature_row(f"{d}_official", d, prof, seconds=time.time() - t0, acc290=None)
            row["n_train"] = int(len(ytr)); row["T"] = int(Xtr.shape[1]); row["C"] = int(Xtr.shape[2])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[et official] {d}: n={len(ytr)} T={Xtr.shape[1]} {row['seconds']}s rinfo@60 {row['rinfo@60']:+.2f} rep_highpass05@60 {row['rep_highpass05@60']:+.2f}", flush=True)


def official_ctx_rows(target: str, et: Mapping[str, float], phi: Mapping[str, float], cols: Sequence[str], mu: np.ndarray, sd: np.ndarray,
                      *, budgets: Sequence[int] = pq.BUDGETS, clip: Optional[float] = None) -> dict[tuple[str, int], np.ndarray]:
    """{(靶_official, 预算): 标准化 (E_T ⊕ φ) ⊕ 预算 one-hot}——列名按 cols（`rep_<op>@60` 取 et 的 `rep_<op>`）；clip 给定时裁到 ±clip（D′）。"""
    table = {**{f"{c}@60": v for c, v in et.items()}, **{f"{c}@290": v for c, v in et.items()}, **phi}
    x, _n = _m33.table_row_vector(table, cols, np.asarray(mu, np.float64), np.asarray(sd, np.float64))
    if clip is not None:
        x = np.clip(x, -float(clip), float(clip))
    return {(f"{target}_official", int(b)): np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32) for b in budgets}


# ---------------------------------------------------------------- 模拟
def _load(models_dir: Path, target: str, arm: str, device: str = "cpu"):
    paths = [models_dir / target / f"{arm}_s{s}.pt" for s in SEEDS]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"{target}/{arm} 模型不全：{[str(p) for p in paths if not p.exists()]}")
    return [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in paths]


def rank_official(target: str, cands: Mapping, et_row: Mapping, phi_row: Mapping, *, device: str = "cpu", clip: Optional[float] = None) -> dict:
    """三份名单：prior（candidates.prior_order）/ blind_tclsc / et_tclsc（预算档 DEPLOY_BUDGET，三种子多数票）。"""
    keys_cls = {c["cls_key"]: CLS(**c["cls"]) for c in cands["shared"]} | {c["cls_key"]: CLS(**c["cls"]) for c in cands["nominated"]}
    keys_cls[cands["ggs"]["cls_key"]] = CLS(**cands["ggs"]["cls"])
    slice_id = f"{target}_official"
    cat = {slice_id: [("official", c, k) for k, c in sorted(keys_cls.items())]}
    out = {"prior": list(cands["prior_order"])}
    blind = _load(BLIND_MODELS, target, "blind_diff_hard", device)
    rk = _g56.rank_local_models({"blind_diff_hard": blind}, cat, budgets=pq.BUDGETS, device=device)
    out["blind_tclsc"] = rk[str(DEPLOY_BUDGET)][slice_id]["blind_diff_hard"]
    ft = _load(ET_MODELS, target, "ft60_hard", device)
    meta0 = ft[0][1]
    cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
    et = {c: float(et_row[f"{c}@60"]) for c in REP_PROFILE_COLS}
    phi = {c: float(phi_row[c]) for c in PHI_COLS}
    rows_ctx = official_ctx_rows(target, et, phi, cols, mu, sd, clip=clip)
    z = rows_ctx[(slice_id, DEPLOY_BUDGET)][: len(cols)]
    rk = _g56.rank_local_models({"ft60_hard": ft}, cat, budgets=pq.BUDGETS, device=device, ctx_rows=rows_ctx)
    out["et_tclsc"] = rk[str(DEPLOY_BUDGET)][slice_id]["ft60_hard"]
    out["_diag"] = {"ctx_abs_max": float(np.abs(z).max()), "ctx_n_over3": int((np.abs(z) > 3).sum()), "n_candidates": len(keys_cls), "ggs_key": cands["ggs"]["cls_key"]}
    return out


def gates_deploy(mean_et: float, mean_prior: float, n_not_worse: int, n_match: int) -> dict:
    return {"GD1": {"mean_et": mean_et, "mean_prior": mean_prior, "n_not_worse": n_not_worse, "pass": bool(mean_et >= mean_prior + 0.005 and n_not_worse >= 4)},
            "GD2": {"n_match": n_match, "pass": bool(n_match >= 5)}}


def verdict_deploy(g: Mapping) -> str:
    if not g["GD1"]["pass"]:
        return "DEPLOY-ET-FLAT"
    return "DEPLOY-ET-WIN" if g["GD2"]["pass"] else "DEPLOY-ET-BETTER"


def cmd_simulate(*, device: str = "cpu") -> dict:
    rows = list(read_jsonl(DEPLOY / "runs.jsonl"))
    cands_all = json.loads((DEPLOY / "candidates.json").read_text(encoding="utf-8"))
    et_rows = {r["dataset"]: r for r in read_jsonl(OUT / "et_official.jsonl")}
    phi_all = json.loads(PHI_OFFICIAL.read_text(encoding="utf-8"))
    per: dict = {}
    for d in TARGETS:
        traj, finals, budgets = catalog_traj(rows, d)
        lists = rank_official(d, cands_all[d], et_rows[d], phi_all[d], device=device)
        oracle_key = max(traj, key=lambda k: traj[k][budgets[-1]])
        res = {"budgets": list(budgets), "oracle": {"key": oracle_key, "acc": float(traj[oracle_key][budgets[-1]])}, "lists": {}, "diag": lists["_diag"], "ggs_rank": {}}
        for name in LISTS:
            ranked = lists[name]
            entry = {"order": list(ranked), "oracle_rank": (ranked.index(oracle_key) + 1) if oracle_key in ranked else None,
                     "ggs_rank": (ranked.index(lists["_diag"]["ggs_key"]) + 1) if lists["_diag"]["ggs_key"] in ranked else None}
            for sh, ks in SH_ARMS.items():
                r = sh_on_list(ranked, traj, ks=ks, budgets=budgets)
                r["finals_3seed"] = finals.get(r["champion"])
                r["finals_mean"] = float(np.mean(finals[r["champion"]])) if finals.get(r["champion"]) and len(finals[r["champion"]]) >= 3 else None
                r["regret"] = float(traj[oracle_key][budgets[-1]] - r["acc"])
                r["match_ts2vec"] = bool(r["acc"] >= TS2VEC_PAPER[d] - MATCH_MARGIN)
                entry[sh] = r
            res["lists"][name] = entry
        per[d] = res
    summary = {}
    for sh in SH_ARMS:
        for name in LISTS:
            accs = [per[d]["lists"][name][sh]["acc"] for d in TARGETS]
            summary[f"{name}|{sh}"] = {"mean": float(np.mean(accs)), "n_match": int(sum(per[d]["lists"][name][sh]["match_ts2vec"] for d in TARGETS)),
                                       "mean_regret": float(np.mean([per[d]["lists"][name][sh]["regret"] for d in TARGETS])),
                                       "mean_cost": float(np.mean([per[d]["lists"][name][sh]["cost_iters"] for d in TARGETS]))}
    c0 = {d: {"ref": WP3_SH633_SEED0[d], "got": per[d]["lists"]["prior"]["SH6-3-3"]["acc"]} for d in TARGETS}
    c0_pass = all(abs(v["ref"] - v["got"]) <= 0.0015 for v in c0.values())
    n_not_worse = sum(per[d]["lists"]["et_tclsc"]["SH6-3-3"]["acc"] >= per[d]["lists"]["prior"]["SH6-3-3"]["acc"] - 1e-9 for d in TARGETS)
    g = gates_deploy(summary["et_tclsc|SH6-3-3"]["mean"], summary["prior|SH6-3-3"]["mean"], n_not_worse, summary["et_tclsc|SH6-3-3"]["n_match"])
    a = {"protocol": "results/tclsc_s1et_deploy/PROTOCOL.md", "targets": per, "summary": summary, "c0": {"per": c0, "pass": c0_pass}, "gates": g, "verdict": verdict_deploy(g)}
    a["predictions"] = {"P1": {"desc": "C0 逐位", "hit": c0_pass}, "P2": {"desc": "G-D1", "hit": g["GD1"]["pass"]}, "P3": {"desc": "G-D2", "hit": g["GD2"]["pass"]},
                        "P4": {"desc": "HW 名单前 6 含 GGS", "value": per["Handwriting"]["lists"]["et_tclsc"]["ggs_rank"],
                               "hit": bool((per["Handwriting"]["lists"]["et_tclsc"]["ggs_rank"] or 99) <= 6)},
                        "P5": {"desc": "E_T ≥ 任务盲 T-CLSC（SH6-3-3 均值）", "value": [summary["et_tclsc|SH6-3-3"]["mean"], summary["blind_tclsc|SH6-3-3"]["mean"]],
                               "hit": bool(summary["et_tclsc|SH6-3-3"]["mean"] >= summary["blind_tclsc|SH6-3-3"]["mean"])}}
    (OUT / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"] + " | " + " | ".join(f"{k} {v['mean']:.3f} ({v['n_match']}/6)" for k, v in summary.items()), flush=True)
    return a



def cmd_simulate_trust(*, device: str = "cpu") -> dict:
    """D′：信任域（prior_order 11 条）+ ctx 裁 ±CLIP + SH6-3-3；冠军缺 3 种子的写 finals/picks.json 供 39 --finals。"""
    rows = list(read_jsonl(DEPLOY / "runs.jsonl"))
    rows_fin = rows + [r for r in read_jsonl(WP3_FINALS_DIR / "runs.jsonl") if r.get("stage") == "s2_finals"]
    fin_dir = OUT / "finals"
    if (fin_dir / "runs.jsonl").exists():
        rows_fin += [r for r in read_jsonl(fin_dir / "runs.jsonl") if r.get("stage") == "s2_finals"]
    cands_all = json.loads((DEPLOY / "candidates.json").read_text(encoding="utf-8"))
    et_rows = {r["dataset"]: r for r in read_jsonl(OUT / "et_official.jsonl")}
    phi_all = json.loads(PHI_OFFICIAL.read_text(encoding="utf-8"))
    per, picks = {}, {}
    for d in TARGETS:
        traj, _f, budgets = catalog_traj(rows, d)
        _t, finals, _b = catalog_traj(rows_fin, d)
        allowed = set(cands_all[d]["prior_order"])
        lists = rank_official(d, cands_all[d], et_rows[d], phi_all[d], device=device, clip=CLIP)
        lists_raw = rank_official(d, cands_all[d], et_rows[d], phi_all[d], device=device, clip=None)
        oracle_key = max(traj, key=lambda k: traj[k][budgets[-1]])
        res = {"lists": {}, "oracle": {"key": oracle_key, "acc": float(traj[oracle_key][budgets[-1]])},
               "clip_changed_top6": restrict_list(lists["et_tclsc"], allowed)[:6] != restrict_list(lists_raw["et_tclsc"], allowed)[:6]}
        for name in LISTS:
            ranked = restrict_list(lists[name], allowed)
            r = sh_on_list(ranked, traj, ks=SH_ARMS["SH6-3-3"], budgets=budgets)
            fs = finals.get(r["champion"], [])
            r["finals"] = fs
            r["finals_mean"] = float(np.mean(fs)) if len(fs) >= 3 else None
            r["match_seed0"] = bool(r["acc"] >= TS2VEC_PAPER[d] - MATCH_MARGIN)
            r["match_3seed"] = (bool(r["finals_mean"] >= TS2VEC_PAPER[d] - MATCH_MARGIN) if r["finals_mean"] is not None else None)
            r["oracle_rank"] = (ranked.index(oracle_key) + 1) if oracle_key in ranked else None
            res["lists"][name] = r
            if name == "et_tclsc" and len(fs) < 3:
                picks.setdefault(d, {})["SH6-3-3_et_tr"] = r["champion"]
        per[d] = res
    summary = {n: {"mean_seed0": float(np.mean([per[d]["lists"][n]["acc"] for d in TARGETS])),
                   "n_match_seed0": int(sum(per[d]["lists"][n]["match_seed0"] for d in TARGETS)),
                   "mean_3seed": (float(np.mean([per[d]["lists"][n]["finals_mean"] for d in TARGETS])) if all(per[d]["lists"][n]["finals_mean"] is not None for d in TARGETS) else None),
                   "n_match_3seed": (int(sum(per[d]["lists"][n]["match_3seed"] for d in TARGETS)) if all(per[d]["lists"][n]["match_3seed"] is not None for d in TARGETS) else None)}
               for n in LISTS}
    et, pr = summary["et_tclsc"], summary["prior"]
    g1 = bool(et["mean_seed0"] >= pr["mean_seed0"] + 0.01)
    hw3 = per["Handwriting"]["lists"]["et_tclsc"]["finals_mean"]
    g2 = None if (et["n_match_3seed"] is None or hw3 is None) else bool(et["n_match_3seed"] >= 5 and hw3 >= 0.485)
    verdict = "DEPLOY-ET-TR-WIN" if (g1 and g2) else ("DEPLOY-ET-TR-BETTER" if g1 else "DEPLOY-ET-TR-FLAT")
    if g1 and g2 is None:
        verdict += "?finals"
    a = {"protocol": "results/tclsc_s1et_deploy/PROTOCOL.md §8", "clip": CLIP, "targets": per, "summary": summary,
         "gates": {"GD1p": {"et": et["mean_seed0"], "prior": pr["mean_seed0"], "pass": g1}, "GD2p": {"n_match_3seed": et["n_match_3seed"], "hw_3seed": hw3, "pass": g2}},
         "verdict": verdict, "picks_pending": picks}
    a["predictions"] = {"P6": {"desc": "G-D′1", "hit": g1}, "P7": {"desc": "G-D′2", "hit": g2},
                        "P8": {"desc": "裁剪不改任何靶的前 6", "hit": not any(per[d]["clip_changed_top6"] for d in TARGETS)},
                        "P9": {"desc": "HW 冠军 3 种子 ≥ 0.50", "value": hw3, "hit": None if hw3 is None else bool(hw3 >= 0.50)}}
    (OUT / "analysis_trust.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    if picks:
        fin_dir.mkdir(parents=True, exist_ok=True)
        (fin_dir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
        if not (fin_dir / "candidates.json").exists():
            (fin_dir / "candidates.json").write_text((DEPLOY / "candidates.json").read_text(encoding="utf-8"), encoding="utf-8")
    names = {"prior": "先验名单", "blind_tclsc": "任务盲 T-CLSC 限域", "et_tclsc": "E_T T-CLSC 限域"}
    L = [f"# D′：限信任域（shared-10 + GGS）+ ctx 裁 ±{CLIP:g} + SH6-3-3（官方口径，零新 R'）", "", f"读数码 **{verdict}**", "",
         "| 靶 | TS2Vec 论文 | 目录 oracle | " + " | ".join(names[n] for n in LISTS) + " |", "|---|---:|---:|" + "---:|" * len(LISTS)]
    for d in TARGETS:
        cells = []
        for n in LISTS:
            r = per[d]["lists"][n]
            cells.append(f"{r['acc']:.3f}" + (f" ({r['finals_mean']:.3f})" if r["finals_mean"] is not None else " (—)") + (" ✓" if r["match_seed0"] else " ✗"))
        L.append(f"| {d} | {TS2VEC_PAPER[d]:.3f} | {per[d]['oracle']['acc']:.3f} | " + " | ".join(cells) + " |")
    L.append("| **均值 (3 种子) / 追平 seed0** | | | " + " | ".join(
        f"{summary[n]['mean_seed0']:.3f}" + (f" ({summary[n]['mean_3seed']:.3f})" if summary[n]["mean_3seed"] is not None else " (—)") + f" / {summary[n]['n_match_seed0']}/6" for n in LISTS) + " |")
    L += ["", f"裁剪改变前 6 名单的靶：{[d for d in TARGETS if per[d]['clip_changed_top6']]}", f"待补 finals：{picks}",
          f"门：G-D′1 {g1}（{et['mean_seed0']:.3f} vs {pr['mean_seed0']:.3f} + 0.01）；G-D′2 {g2}（3 种子追平 {et['n_match_3seed']}/6，HW {hw3}）", ""]
    (OUT / "tables_trust.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return a


def markdown_tables(a: dict) -> str:
    per, s = a["targets"], a["summary"]
    names = {"prior": "先验名单", "blind_tclsc": "任务盲 T-CLSC 名单", "et_tclsc": "E_T T-CLSC 名单"}
    L = [f"# D 检验：官方口径零样本名单模拟（六靶 21 条官方目录，seed 0，零新 R'）", "", f"读数码 **{a['verdict']}**", ""]
    for sh in SH_ARMS:
        L += [f"## {sh} 冠军 acc（100% 官方 iters，seed 0；括号 = 3 种子均值若有）", "",
              "| 靶 | TS2Vec 论文 | 目录 oracle | " + " | ".join(names[n] for n in LISTS) + " | E_T − 先验 |", "|---|---:|---:|" + "---:|" * len(LISTS) + "---:|"]
        for d in TARGETS:
            cells = []
            for n in LISTS:
                r = per[d]["lists"][n][sh]
                cells.append(f"{r['acc']:.3f}" + (f" ({r['finals_mean']:.3f})" if r["finals_mean"] is not None else "") + (" ✓" if r["match_ts2vec"] else " ✗"))
            L.append(f"| {d} | {TS2VEC_PAPER[d]:.3f} | {per[d]['oracle']['acc']:.3f} | " + " | ".join(cells)
                     + f" | {per[d]['lists']['et_tclsc'][sh]['acc'] - per[d]['lists']['prior'][sh]['acc']:+.3f} |")
        L.append(f"| **均值 / 追平数** | | | " + " | ".join(f"{s[f'{n}|{sh}']['mean']:.3f} ({s[f'{n}|{sh}']['n_match']}/6)" for n in LISTS)
                 + f" | {s[f'et_tclsc|{sh}']['mean'] - s[f'prior|{sh}']['mean']:+.3f} |")
        L.append("")
    L += ["## 名单诊断", "", "| 靶 | oracle 在各名单的位次（先验 / 任务盲 / E_T） | GGS 位次（先验 / 任务盲 / E_T） | E_T ctx 出界 |max z| / >3 的列数 |", "|---|---|---|---|"]
    for d in TARGETS:
        ls = per[d]["lists"]
        L.append(f"| {d} | {ls['prior']['oracle_rank']} / {ls['blind_tclsc']['oracle_rank']} / {ls['et_tclsc']['oracle_rank']} | "
                 f"{ls['prior']['ggs_rank']} / {ls['blind_tclsc']['ggs_rank']} / {ls['et_tclsc']['ggs_rank']} | {per[d]['diag']['ctx_abs_max']:.1f} / {per[d]['diag']['ctx_n_over3']} |")
    L += ["", "## 门", "", f"- C0 先验 SH6-3-3 seed 0 逐位：{a['c0']['pass']}（" + ", ".join(f"{d[:4]} {v['got']:.3f} vs {v['ref']:.3f}" for d, v in a["c0"]["per"].items()) + "）",
          f"- G-D1：E_T 均值 {a['gates']['GD1']['mean_et']:.3f} vs 先验 {a['gates']['GD1']['mean_prior']:.3f}，不低于先验 {a['gates']['GD1']['n_not_worse']}/6 → {a['gates']['GD1']['pass']}",
          f"- G-D2：追平 {a['gates']['GD2']['n_match']}/6 → {a['gates']['GD2']['pass']}", "", "## 预测对账", ""]
    L += [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--trust", action="store_true", help="D′：限信任域 + 裁剪 + SH6-3-3（写 finals/picks.json）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.features or args.simulate or args.trust):
        build_parser().error("至少指定 --features / --simulate 之一")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.features:
        cmd_features(device=args.device, resume=args.resume)
    if args.simulate:
        cmd_simulate(device="cpu")
    if args.trust:
        cmd_simulate_trust(device="cpu")


if __name__ == "__main__":
    main()
