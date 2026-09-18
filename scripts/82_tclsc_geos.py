"""WP16 GEO-S 检验：几何子空间搜索——UEA 29 官方靶 × 18 损失几何家族（干净基），干净家族净空 / E_T 按靶选家族 / E 排序 + SH 部署值。协议 results/tclsc_geos/PROTOCOL.md。

子命令：
  --plan                        18 家族网格 × 29 靶（+ 两锚家族 seed 1）→ plan.json（580 格）
  --run [--datasets ...]        39 号 execute（ts2vec_official，截点 25/50/100%）→ runs.jsonl（--resume）
  --verify / --smoke            对账 / 本地冒烟（1 靶 2 家族 8 iters，写 smoke/）
  --analyze                     S1 净空 / S2 E_T 预测（81 号同式）/ S3 SH6-3-3 截点模拟 / 门 / 读数码 / 表
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import cls_key  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g39 = importlib.import_module("scripts.39_sh_deploy")
_g45 = importlib.import_module("scripts.45_uea30_deploy")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g81 = importlib.import_module("scripts.81_tclsc_geo")
OUT = ROOT / "results" / "tclsc_geos"
PROTOCOL = "ts2vec_official"
STAGE, ARM = "s1_catalog", "geo"
CLEAN_TEMPLATE = dict(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=2, emb_jitter=0.0, emb_mask=0.0,
                      instance=True, temporal=True, cross_scale=True, kernel=3, pool="avg", adj_neighbor=True)
TEMPERATURE = {"triplet": 1.0, "info_nce": 0.1}        # WP15 惯例：B_T τ 1.0 / B_I τ 0.1
ANCHORS = (("info_nce", "euclidean", "none"), ("triplet", "dot", "none"))
PRIOR_FAMILY, GGS_FAMILY = ("triplet", "dot", "none"), ("info_nce", "euclidean", "none")
SH_K = (6, 3, 3)
G1_GAIN, G1_FRAC, G2_GAIN, G3_DELTA, ALPHA = 0.015, 1 / 3, 0.005, 0.005, 0.05
EEG_LIKE = _g81.EEG_LIKE
SEED = 0


# ---------------------------------------------------------------- 网格 / 格
def family_grid(space: CLSSpace) -> list[tuple[tuple, CLS, str]]:
    out = []
    for lt, sf, nm in itertools.product(space.choices("loss_type"), space.choices("sim_func"), space.choices("norm")):
        d = dict(CLEAN_TEMPLATE, loss_type=lt, sim_func=sf, norm=nm, temperature=TEMPERATURE[lt])
        d.update(space.inactive_overrides(d))
        c = CLS(**d)
        if not space.validate(c):
            raise ValueError(f"家族 {(lt, sf, nm)} 不合法")
        out.append(((lt, sf, nm), c, cls_key(c)))
    return out


def targets29() -> list[str]:
    return sorted(_g81.feature_table())


def build_cells(*, targets: Optional[Sequence[str]] = None, seeds_anchor: Sequence[int] = (0, 1), iters: Optional[int] = None) -> list[dict]:
    cfg = _g45.load_cfg()
    tg = list(targets) if targets else targets29()
    protos = _g35.load_protocols()
    p = protos[PROTOCOL]
    sizes = _g45.sizes_from_cfg(cfg, tg)
    space = CLSSpace.from_yaml()
    cells = []
    for d in tg:
        for fam, c, k in family_grid(space):
            for s in ([0] + [x for x in seeds_anchor if x != 0]) if fam in ANCHORS else [0]:
                cell = _g35._cell(STAGE, d, PROTOCOL, p, sizes, arm_names=[ARM], cls_key_=k, cls=c, seed=s, iters=iters)
                cell["checkpoints"] = list(_g39.checkpoints_for(cell["iters"]))
                cell.update({"family": str(fam), "twin_of": None})
                _g39.check_extension_consistency(cell)
                cells.append(cell)
    keys = [c["key"] for c in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("cache_key 撞键")
    return cells


def cmd_plan(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    plan = {"n_cells": len(cells), "targets": targets29(), "families": [str(f) for f, _c, _k in family_grid(CLSSpace.from_yaml())], "cells": cells}
    (outdir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    by_iters = Counter(c["iters"] for c in cells)
    print(f"[plan] {len(cells)} 格 = {len(plan['targets'])} 靶 × (18 + {len(ANCHORS)} 锚家族 seed 1)；iters {dict(by_iters)}", flush=True)
    return plan


def load_plan(outdir: Path) -> dict:
    return json.loads((outdir / "plan.json").read_text(encoding="utf-8"))


def split_targets(n_parts: int, targets: Optional[Sequence[str]] = None) -> list[list[str]]:
    """按 uea30.yaml 的 4090 锚（pretrain_s + 3 eval_s；缺锚按 n_train × L × C 估）贪心均分成 n_parts 份（pod 多进程用）。"""
    cfg = _g45.load_cfg()
    tg = list(targets) if targets else targets29()
    est = {}
    for d in tg:
        dc = cfg["datasets"].get(d, {})
        anc = dc.get("anchor_4090") or {}
        est[d] = float(anc.get("pretrain_s", 0)) + 3 * float(anc.get("eval_s", 0)) if anc else float(dc.get("n_train", 300)) * float(dc.get("L", 150)) * float(dc.get("C", 3)) / 2e5
    parts: list[list[str]] = [[] for _ in range(n_parts)]
    load = [0.0] * n_parts
    for d in sorted(tg, key=lambda x: -est[x]):
        i = int(np.argmin(load))
        parts[i].append(d); load[i] += est[d]
    return parts


def cmd_run(args, outdir: Path) -> None:
    sel = set(args.datasets) if args.datasets else None
    if getattr(args, "parts", 0):
        sel = set(split_targets(int(args.parts))[int(args.part)])
    cells = [c for c in load_plan(outdir)["cells"] if sel is None or c["dataset"] in sel]
    cells = _g45.drop_repeated_oom(cells, outdir / "oom_failures.jsonl")
    if sys.platform.startswith("win"):
        importlib.import_module("scripts.43_bo_nominate").install_windows_eval_timeout()
    cfg = _g45.load_cfg()
    ns = argparse.Namespace(outdir=str(outdir), resume=args.resume, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    _g39.execute(cells, ns, "GEOS", loader=_g45.make_loader(cfg, args.cache_dir))


def cmd_verify(outdir: Path) -> dict:
    plan = load_plan(outdir)
    rows = _g35.load_done(outdir / "runs.jsonl") if (outdir / "runs.jsonl").exists() else {}
    want = {c["key"]: c for c in plan["cells"]}
    missing = [k for k in want if k not in rows]
    bad = [k for k, r in rows.items() if k in want and (r.get("error") or len(r.get("checkpoint_results") or []) != len(want[k]["checkpoints"]))]
    v = {"planned": len(want), "have": sum(1 for k in want if k in rows), "missing": len(missing), "bad": len(bad), "ok": not missing and not bad, "missing_examples": missing[:5]}
    print(f"[verify] planned {v['planned']} have {v['have']} missing {v['missing']} bad {v['bad']} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


def cmd_smoke(args, outdir: Path, *, dataset: str = "Handwriting", iters: int = 8) -> bool:
    sm = outdir / "smoke"
    sm.mkdir(parents=True, exist_ok=True)
    cells = build_cells(targets=[dataset], seeds_anchor=(0,), iters=iters)[:2]
    if sys.platform.startswith("win"):
        importlib.import_module("scripts.43_bo_nominate").install_windows_eval_timeout()
    cfg = _g45.load_cfg()
    ns = argparse.Namespace(outdir=str(sm), resume=False, device=args.device, joblib_backend=args.joblib_backend, har_npz=None)
    _g39.execute(cells, ns, "smoke", loader=_g45.make_loader(cfg, args.cache_dir))
    rows = list(_g35.load_done(sm / "runs.jsonl").values())
    ok = len(rows) >= 2 and all(r.get("error") is None and len(r.get("checkpoint_results") or []) == 3 for r in rows)
    print(f"smoke {'OK' if ok else 'FAILED'}: {[(r['family'], round(r['acc'], 3)) for r in rows]}", flush=True)
    return ok


# ---------------------------------------------------------------- 分析
def rows_ok(outdir: Path) -> list[dict]:
    return [r for r in _g35.load_done(outdir / "runs.jsonl").values() if r.get("error") is None and r.get("acc") is not None]


def utility_table(rows: Sequence[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    acc: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if int(r.get("seed", 0)) == 0:
            acc[r["dataset"]][str(r["family"])] = float(r["acc"])
    ds = sorted(acc)
    fams = sorted({f for d in acc.values() for f in d})
    U = np.full((len(ds), len(fams)), np.nan)
    for i, d in enumerate(ds):
        top = max(acc[d].values())
        for j, f in enumerate(fams):
            if f in acc[d]:
                U[i, j] = acc[d][f] - top
    return U, ds, fams


def trajectories(rows: Sequence[dict]) -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for r in rows:
        if int(r.get("seed", 0)) == 0:
            out[r["dataset"]][str(r["family"])] = {int(c["iters"]): float(c["acc"]) for c in r["checkpoint_results"]}
    return out


def sh_sim(traj: Mapping[str, Mapping[int, float]], *, order: Sequence[str], k: Sequence[int] = SH_K) -> float:
    """SH 截点模拟：起跑 k0 条 @ckpt0 → 留 k1 @ckpt1 → 留 k2 @ckpt2 → 取最终最好。"""
    ck = sorted(next(iter(traj.values())).keys())
    alive = [f for f in order if f in traj][: k[0]]
    for stage in range(1, len(k)):
        alive = sorted(alive, key=lambda f: -traj[f][ck[stage - 1]])[: k[stage]]
    return max(traj[f][ck[-1]] for f in alive)


def noise_sigma(rows: Sequence[dict]) -> dict[str, float]:
    by: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], str(r["family"]))][int(r.get("seed", 0))] = float(r["acc"])
    sig: dict[str, list] = defaultdict(list)
    for (d, _f), s in by.items():
        if 0 in s and 1 in s:
            sig[d].append(abs(s[0] - s[1]) / np.sqrt(2))
    return {d: float(np.mean(v)) for d, v in sig.items()}


def lodo_pred_matrix(X: np.ndarray, U: np.ndarray, *, alphas=_g81.ALPHAS) -> np.ndarray:
    """81 号 lodo_predict 的同式，但返回每靶的预测效用向量（用于 S3 排序）。"""
    n = X.shape[0]
    Uf = np.where(np.isnan(U), np.nanmin(U, axis=1, keepdims=True), U)
    P = np.zeros_like(U)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_a, best_r = alphas[0], -np.inf
        for a in alphas:
            reg = []
            for kk in tr:
                tr2 = [j for j in tr if j != kk]
                pred = _g81._ridge_fit_predict(X[tr2], Uf[tr2], X[[kk]], a)[0]
                pred = np.where(np.isnan(U[kk]), -np.inf, pred)
                reg.append(U[kk, int(np.argmax(pred))])
            if np.mean(reg) > best_r:
                best_r, best_a = float(np.mean(reg)), a
        P[i] = _g81._ridge_fit_predict(X[tr], Uf[tr], X[[i]], best_a)[0]
    return P


def gate_s1(m: Mapping) -> bool:
    return bool(m["oracle_minus_fixed"] >= G1_GAIN and m["frac_over_noise"] >= G1_FRAC)


def gate_s2(m: Mapping) -> bool:
    return bool(m["gain_vs_fixed"] >= G2_GAIN and m["p_gain"] < ALPHA)


def gate_s3(m: Mapping) -> bool:
    return bool(m["delta"] >= G3_DELTA and m["p"] < ALPHA)


def verdict_geos(g: Mapping[str, bool]) -> str:
    if not g["G-S1"]:
        return "GEOS-FLAT"
    if not g["G-S2"]:
        return "GEOS-HEADROOM-ONLY"
    return "GEOS-GO" if g["G-S3"] else "GEOS-RANK-ONLY"


def cmd_analyze(outdir: Path, *, n_perm: int = 1000) -> dict:
    rows = rows_ok(outdir)
    U, ds, fams = utility_table(rows)
    traj = trajectories(rows)
    sig = noise_sigma(rows)
    med = float(np.median(list(sig.values()))) if sig else 0.01
    n = len(ds)
    champ = [fams[int(np.nanargmax(U[i]))] for i in range(n)]
    fixed_fold = [_g81.fixed_best(U, train_idx=[j for j in range(n) if j != i]) for i in range(n)]
    oracle_minus_fixed = np.array([0.0 - U[i, fixed_fold[i]] for i in range(n)])
    j_prior, j_ggs = (fams.index(str(PRIOR_FAMILY)) if str(PRIOR_FAMILY) in fams else None), (fams.index(str(GGS_FAMILY)) if str(GGS_FAMILY) in fams else None)
    s1 = {"n": n, "oracle_minus_fixed": float(oracle_minus_fixed.mean()), "frac_over_noise": float(np.mean([oracle_minus_fixed[i] > 2 * sig.get(ds[i], med) for i in range(n)])),
          "oracle_minus_prior": float(np.nanmean([-U[i, j_prior] for i in range(n)])) if j_prior is not None else None,
          "oracle_minus_ggs": float(np.nanmean([-U[i, j_ggs] for i in range(n)])) if j_ggs is not None else None,
          "fixed_family_all": fams[_g81.fixed_best(U, train_idx=list(range(n)))], "champion_counts": dict(Counter(champ)),
          "two_family_share": float(np.mean([c in (str(PRIOR_FAMILY), str(GGS_FAMILY)) for c in champ])), "sigma_median": med,
          "per_dataset": {ds[i]: {"champion": champ[i], "oracle_minus_fixed": float(oracle_minus_fixed[i]), "sigma": sig.get(ds[i]), "acc_best": float(max(traj[ds[i]][f][max(traj[ds[i]][f])] for f in traj[ds[i]]))} for i in range(n)}}
    # 目录家族冠军（GEO-E）vs 干净冠军一致率
    geo_e = _g81.OUT / "analysis.json"
    if geo_e.exists():
        ge = json.loads(geo_e.read_text(encoding="utf-8"))
        cat_champ = {d: str(tuple(v["champion"])) for d, v in ge["arms"]["et60"]["per_dataset"].items()}
        agree = [cat_champ[d] == champ[i] for i, d in enumerate(ds) if d in cat_champ]
        s1["catalog_champion_agreement"] = float(np.mean(agree)) if agree else None
    # S2：E_T 预测（81 号同式）
    table = _g81.feature_table()
    s2: dict = {}
    P_by_arm = {}
    for arm, cols in _g81.ARMS.items():
        X, ok = _g81.feature_matrix(table, ds, cols)
        if len(ok) < 10:
            s2[arm] = {"n": len(ok), "skipped": True}; continue
        Xa, Ua = X[ok], U[ok]
        res = _g81.lodo_predict(Xa, Ua)
        ev = _g81.evaluate_picks(Ua, res["picks"], fixed_by_fold=res["fixed_by_fold"])
        pb = _g81.permutation_both(Xa, Ua, n_perm=n_perm)
        maj = max(Counter(champ[i] for i in ok).values()) / len(ok)
        s2[arm] = {"n": len(ok), "hit_rate": res["hit_rate"], "majority": maj, "gain_vs_fixed": ev["gain_vs_fixed"], "mean_regret": ev["mean_regret"], "fixed_regret": ev["fixed_regret"],
                   "wins": ev["wins"], "losses": ev["losses"], "p_gain": pb["gain"]["p"], "p_hit": pb["hit"]["p"],
                   "picks": {ds[i]: fams[res["picks"][k]] for k, i in enumerate(ok)}}
        P_by_arm[arm] = (ok, lodo_pred_matrix(Xa, Ua))
        print(f"[S2 {arm}] n {len(ok)} hit {res['hit_rate']:.2f} (maj {maj:.2f}) gain {ev['gain_vs_fixed']:+.4f} p {pb['gain']['p']:.3f}", flush=True)
    # S3：SH6-3-3 截点模拟
    s3: dict = {}
    fixed_order = {i: [fams[j] for j in np.argsort(-np.nanmean(np.where(np.isnan(U[[jj for jj in range(n) if jj != i]]), np.nanmin(U, axis=1, keepdims=True)[[jj for jj in range(n) if jj != i]], U[[jj for jj in range(n) if jj != i]]), axis=0))] for i in range(n)}
    sh = {"fixed": [], "prior_first": [], "oracle": [], "e": []}
    for i, d in enumerate(ds):
        sh["fixed"].append(sh_sim(traj[d], order=fixed_order[i]))
        sh["prior_first"].append(sh_sim(traj[d], order=[str(PRIOR_FAMILY)] + [f for f in fixed_order[i] if f != str(PRIOR_FAMILY)]))
        sh["oracle"].append(sh_sim(traj[d], order=[fams[j] for j in np.argsort(-np.where(np.isnan(U[i]), -np.inf, U[i]))]))
    if PRIMARY_ARM in P_by_arm:
        ok, P = P_by_arm[PRIMARY_ARM]
        e_acc = {ds[i]: sh_sim(traj[ds[i]], order=[fams[j] for j in np.argsort(-np.where(np.isnan(U[i]), -np.inf, P[k]))]) for k, i in enumerate(ok)}
        sh["e"] = [e_acc.get(d) for d in ds]
        pairs = [(sh["e"][i], sh["fixed"][i]) for i in range(n) if sh["e"][i] is not None]
        x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        s3 = {"n": len(pairs), "e": float(x.mean()), "fixed": float(y.mean()), "prior_first": float(np.mean(sh["prior_first"])), "oracle": float(np.mean(sh["oracle"])),
              "delta": float((x - y).mean()), "p": float(_g56.paired_wilcoxon_less(y, x)), "wins": int(np.sum(x > y)), "losses": int(np.sum(x < y)),
              "per_dataset": {ds[i]: {"e": sh["e"][i], "fixed": sh["fixed"][i], "prior_first": sh["prior_first"][i], "oracle": sh["oracle"][i]} for i in range(n)}}
    pa = s2.get(PRIMARY_ARM, {})
    g = {"G-S1": gate_s1(s1), "G-S2": bool(pa and not pa.get("skipped") and gate_s2(pa)), "G-S3": bool(s3 and gate_s3(s3))}
    eeg = [s1["per_dataset"][d]["oracle_minus_fixed"] for d in ds if d in EEG_LIKE]; non = [s1["per_dataset"][d]["oracle_minus_fixed"] for d in ds if d not in EEG_LIKE]
    a = {"protocol": "results/tclsc_geos/PROTOCOL.md", "families": fams, "s1": s1, "s2": s2, "s3": s3, "gates": g, "verdict": verdict_geos(g),
         "c0": {"rows": len(rows), "anchor_seed_diff_median": float(np.median([abs(v) * np.sqrt(2) for v in sig.values()])) if sig else None},
         "predictions": {"P1": {"desc": "G-S1", "hit": g["G-S1"]}, "P2": {"desc": "G-S2", "hit": g["G-S2"]}, "P3": {"desc": "G-S3", "hit": g["G-S3"]},
                         "P4": {"desc": "两大家族占冠军 ≥ 0.6", "value": s1["two_family_share"], "hit": bool(s1["two_family_share"] >= 0.6)},
                         "P5": {"desc": "目录冠军家族一致率 ≤ 0.6", "value": s1.get("catalog_champion_agreement"), "hit": None if s1.get("catalog_champion_agreement") is None else bool(s1["catalog_champion_agreement"] <= 0.6)},
                         "P6": {"desc": "净空集中在 EEG / 传感器类靶", "value": (float(np.mean(eeg)) if eeg else None, float(np.mean(non)) if non else None), "hit": bool(eeg and non and np.mean(eeg) > np.mean(non))}}}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], g, flush=True)
    return a


PRIMARY_ARM = _g81.PRIMARY


def markdown_tables(a: dict) -> str:
    f = lambda v, d=4: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    s1, s2, s3 = a["s1"], a["s2"], a["s3"]
    L = [f"# GEO-S 检验 tables：几何子空间（{s1['n']} 靶 × {len(a['families'])} 家族，干净基，官方口径）", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", "",
         f"## S1 干净净空：家族 oracle − 固定最好家族 **{s1['oracle_minus_fixed']:+.4f}**（超 2σ 靶 {s1['frac_over_noise']:.2f}，σ 中位 {s1['sigma_median']:.3f}）；− 先验家族 {f(s1['oracle_minus_prior'])}；− GGS 家族 {f(s1['oracle_minus_ggs'])}；固定最好家族 {s1['fixed_family_all']}；两大家族占冠军 {s1['two_family_share']:.2f}；目录冠军一致率 {f(s1.get('catalog_champion_agreement'), 2)}", "",
         f"冠军家族分布：{s1['champion_counts']}", "", "| 靶 | 冠军家族 | oracle − 固定 | σ | 最好 acc |", "|---|---|---:|---:|---:|"]
    for d, v in sorted(s1["per_dataset"].items(), key=lambda kv: -kv[1]["oracle_minus_fixed"]):
        L.append(f"| {d} | {v['champion']} | {v['oracle_minus_fixed']:+.3f} | {f(v['sigma'], 3)} | {v['acc_best']:.3f} |")
    L += ["", "## S2 E 预测家族（LODO 岭，81 号同式）", "", "| 臂 | n | 命中 (猜多数) | 增益 vs 固定 | 胜:负 | p_gain | p_hit |", "|---|---:|---:|---:|---|---:|---:|"]
    for arm, m in s2.items():
        if m.get("skipped"):
            L.append(f"| {arm} | {m['n']} | 跳过 | | | | |"); continue
        L.append(f"| {arm} | {m['n']} | {m['hit_rate']:.2f} ({m['majority']:.2f}) | **{m['gain_vs_fixed']:+.4f}** | {m['wins']}:{m['losses']} | {m['p_gain']:.3f} | {m['p_hit']:.3f} |")
    if s3:
        L += ["", f"## S3 SH6-3-3 截点模拟（{s3['n']} 靶）：E 排序 {s3['e']:.4f} vs 固定顺序 {s3['fixed']:.4f}（Δ {s3['delta']:+.4f}，p {s3['p']:.3f}，{s3['wins']}:{s3['losses']}）；先验优先 {s3['prior_first']:.4f}；oracle {s3['oracle']:.4f}", ""]
    L += [f"C0：{a['c0']}", "", f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for fl in ("plan", "run", "verify", "smoke", "analyze", "resume"):
        ap.add_argument(f"--{fl}", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--parts", type=int, default=0, help="pod 多进程：把 29 靶按估时贪心均分成 N 份")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--joblib-backend", choices=("threading", "loky"), default="threading")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.plan:
        cmd_plan(args.outdir)
    if args.smoke:
        cmd_smoke(args, args.outdir)
    if args.run:
        cmd_run(args, args.outdir)
    if args.verify:
        cmd_verify(args.outdir)
    if args.analyze:
        cmd_analyze(args.outdir, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
