"""WP17-ACTS S3-B″（results/acts/s3bpp/PROTOCOL.md）：field-E 确认版（零 GPU，五项改动一起做）。

在 S3-B′（87）同一数据 / 同一 field 结构上改五处：α 网格扩到 10⁶；r ∈ {4, 8, 16} 内层按 regret@3 选；16 次偏移 cross 指派；
加 field-E ⊕ ④（u 扩一列 a₁^④）；E-only / 组合臂各自 Holm（F3 用 Bonferroni 97.5% CI）。
用法：python scripts/88_acts_s3bpp.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import wilcoxon  # noqa: E402

OUT = ROOT / "results" / "acts" / "s3bpp"
S3B = ROOT / "results" / "acts" / "s3b"
ALPHAS = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
RS = (4, 8, 16)
OFFSETS = tuple(range(1, 17))


# ================================================================ 纯函数
def cross_maps(tasks: Sequence[str], groups: Sequence, order: Sequence[str], offsets: Sequence[int] = OFFSETS) -> list[list[int]]:
    """偏移 k：每任务改用池顺序中偏移 k 的源的同 OA 行号任务（不存在取该源第 0 行）；k = 1 与 86.cross_assign 相同。"""
    g86 = importlib.import_module("scripts.86_acts_s3b")
    groups = np.asarray(groups); order = list(order)
    by = {s: {g86._row(t): i for i, t in enumerate(tasks) if groups[i] == s} for s in order}
    out = []
    for k in offsets:
        m = []
        for i, t in enumerate(tasks):
            nxt = order[(order.index(groups[i]) + k) % len(order)]
            rows = by[nxt]
            m.append(rows.get(g86._row(t), rows[min(rows)]))
        out.append(m)
    return out


def choose_r(inner_regret: Mapping[int, float]) -> int:
    best = min(inner_regret.values())
    return int(min(r for r, v in inner_regret.items() if v <= best + 1e-12))


def verdict_fec(E: Mapping[str, Mapping[str, bool]], C: Mapping[str, Mapping[str, bool]]) -> str:
    if any(v["F1"] and v["F2"] for v in E.values()):
        code = "FEC-CONFIRMED"
    elif any(v["F2"] and v["F3"] for v in E.values()):
        code = "FEC-VALUE-ONLY"
    else:
        code = "FEC-NOT-CONFIRMED"
    if any(v["F5"] for v in C.values()):
        code += "+ADD"
    if any(v["F4"] for v in E.values()):
        code += "+SPACE"
    return code


# ================================================================ 折内：全部 r 一次算完
def run_arm_multi(D: Mapping, phi_sh: np.ndarray, E_of, extra_of, order: Sequence[str], cmaps: Sequence[Sequence[int]]) -> dict:
    """E_of(fold) → (n, 256)；extra_of(fold) → (n,) 或 None（组合臂的 a₁^④ 列：训练任务 = 折内投影，留出任务 = ④ â₁）。
    返回每 r 的外层预测（shared / random / 16 次 cross 的 shared）、每折每 r 的 α 与内层 regret、每折 r*。"""
    g84 = importlib.import_module("scripts.84_acts_s3a"); g87 = importlib.import_module("scripts.87_acts_s3bp")
    groups = D["groups"]; n = len(groups)
    res = {r: {"pred_sh": np.zeros((n, 30)), "pred_rd": [None] * n, "pred_sh_x": np.zeros((len(cmaps), n, 30))} for r in RS}
    info = {}
    for s in order:
        te = np.where(groups == s)[0]; tr = np.where(groups != s)[0]
        E = E_of(s); ex = extra_of(s) if extra_of else None
        fold_info = {"alpha": {}, "inner_regret": {}}
        for r in RS:
            f = g87.pca_fit(E[tr], r)
            U = g87.pca_apply(f, E)
            if ex is not None:
                U = np.column_stack([U, ex])
            rows_of = {i: np.vstack([phi_sh, D["rand"][i]["phi"]]) for i in range(n)}
            allX = np.vstack([g87.field_design(rows_of[i], np.repeat(U[[i]], len(rows_of[i]), axis=0)) for i in tr])
            cm, cs = allX.mean(0), allX.std(0); cs = np.where(cs > 1e-12, cs, 1.0)
            del allX
            blk = g87._fold_blocks(D, phi_sh, U, cm, cs, tr)
            per_g = {}
            for i in tr:
                st = g87._stats(*blk[i])
                per_g[groups[i]] = st if groups[i] not in per_g else g87._add(per_g[groups[i]], st)
            tot = None
            for st in per_g.values():
                tot = st if tot is None else g87._add(tot, st)
            err = np.zeros(len(ALPHAS)); reg = [[] for _ in ALPHAS]
            for g2, st in per_g.items():
                sols = g87._solve_all(g87._sub(tot, st), ALPHAS)
                for i in tr[groups[tr] == g2]:
                    X, y = blk[i]
                    for j, (w, b) in enumerate(sols):
                        p = X @ w + b
                        err[j] += float(np.sum((p - y) ** 2))
                        reg[j].append(float(g84.regret_at_k(p[None, :30], D["end0"][[i]], D["end1"][[i]], 3)[0]))
            best = err.min(); ai = max(j for j in range(len(ALPHAS)) if err[j] <= best * (1 + 1e-9))
            fold_info["alpha"][r] = ALPHAS[ai]; fold_info["inner_regret"][r] = float(np.mean(reg[ai]))
            w, b = g87._solve_all(tot, [ALPHAS[ai]])[0]
            nE = U.shape[1] - (0 if ex is None else 1)
            for i in te:
                X = (g87.field_design(rows_of[i], np.repeat(U[[i]], len(rows_of[i]), axis=0)) - cm) / cs
                p = X @ w + b
                res[r]["pred_sh"][i] = p[:30]; res[r]["pred_rd"][i] = p[30:]
                for ci, m in enumerate(cmaps):
                    ux = U[[i]].copy(); ux[0, :nE] = U[m[i], :nE]           # 只换 E 部分；组合臂的 ④ 列保留本任务
                    Xs = (g87.field_design(phi_sh, np.repeat(ux, 30, axis=0)) - cm) / cs
                    res[r]["pred_sh_x"][ci, i] = Xs @ w + b
        fold_info["r_star"] = choose_r(fold_info["inner_regret"])
        info[s] = fold_info
        print(f"  [{s}] alpha {fold_info['alpha']} inner {({k: round(v, 4) for k, v in fold_info['inner_regret'].items()})} r* {fold_info['r_star']}", flush=True)
    # r* 组合
    sel = {"pred_sh": np.zeros((n, 30)), "pred_rd": [None] * n, "pred_sh_x": np.zeros((len(cmaps), n, 30))}
    for s, fi in info.items():
        te = np.where(groups == s)[0]; r = fi["r_star"]
        sel["pred_sh"][te] = res[r]["pred_sh"][te]; sel["pred_sh_x"][:, te] = res[r]["pred_sh_x"][:, te]
        for i in te:
            sel["pred_rd"][i] = res[r]["pred_rd"][i]
    return {"by_r": res, "sel": sel, "info": info}


# ================================================================ 统计
def _stats_row(D, P, blind3, probe3, groups) -> dict:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g87 = importlib.import_module("scripts.87_acts_s3bp")
    sc = g87.score_arm(D, P, "pred_sh", "pred_rd")
    r3 = sc["r3"]
    xr = np.mean([g84.regret_at_k(P["pred_sh_x"][c], D["end0"], D["end1"], 3) for c in range(P["pred_sh_x"].shape[0])], axis=0)
    worse = np.mean([g84.regret_at_k(P["pred_sh_x"][c], D["end0"], D["end1"], 3) > r3 + 1e-12 for c in range(P["pred_sh_x"].shape[0])], axis=0)
    dcross, dblind, dprobe = xr - r3, blind3 - r3, r3 - probe3
    ok = np.isfinite(sc["sp"])
    return {"sp_median": float(np.nanmedian(sc["sp"])), "sp_mean": float(np.nanmean(sc["sp"])), "sp_ci": g84.cluster_bootstrap_ci(sc["sp"][ok], groups[ok]),
            "regret1": float(sc["r1"].mean()), "regret3": float(r3.mean()),
            "vs_blind": float(dblind.mean()), "vs_blind_ci": g84.cluster_bootstrap_ci(dblind, groups), "p_vs_blind": g84.source_wilcoxon(dblind, groups),
            "vs_probe": float(dprobe.mean()), "vs_probe_ci95": g84.cluster_bootstrap_ci(dprobe, groups), "vs_probe_ci975": g84.cluster_bootstrap_ci(dprobe, groups, level=0.975),
            "p_probe_better": g84.source_wilcoxon(-dprobe, groups),
            "cross_d": float(dcross.mean()), "cross_ci": g84.cluster_bootstrap_ci(dcross, groups), "p_cross": g84.source_wilcoxon(dcross, groups), "cross_worse_frac": float(worse.mean()),
            "d50": float(np.nanmean(sc["d50"])), "_r3": r3}


def cmd_analyze(outdir: Path = OUT) -> dict:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    g86 = importlib.import_module("scripts.86_acts_s3b"); g87 = importlib.import_module("scripts.87_acts_s3bp")
    data = g85.load_all(); D = data["pool"]; phi_sh = data["phi_sh"]
    groups = D["groups"]; n = len(groups); tasks = list(D["tasks"]); order = list(dict.fromkeys(groups.tolist()))
    b_ref = g85.orient_bref(D)
    T = g86._load_targets(S3B); all_tasks = [str(t) for t in T["tasks"]]
    assert all_tasks[:n] == tasks
    em_pt = g86._e_means(S3B, "pt", all_tasks, order)[""][:n]
    em_sh = {f: v[:n] for f, v in g86._e_means(S3B, "sh", all_tasks, order).items()}
    cmaps = cross_maps(tasks, groups, order)
    # 参照 + C0
    ref = g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], {"probe": g84.probe_feature(D["acc25"], D["pfloor"], D["maj"]), "land25": g84.land25_feature(D["Z25"])}, b_ref=b_ref)
    blind3, probe3 = ref["_blind"]["regret3"], ref["probe"]["regret3"]
    c0r = g87.c0_refs(D, phi_sh, b_ref, ref["probe"]["a_hat"])
    s3ap = json.loads((ROOT / "results/acts/s3ap/analysis.json").read_text(encoding="utf-8"))["task1"]["summary"]
    s3bp = json.loads((ROOT / "results/acts/s3bp/analysis.json").read_text(encoding="utf-8"))
    c0 = {"mu_sp_median": (float(np.nanmedian(c0r["mu"]["sp"])), s3ap["mu|hat"]["sp_median"]), "L_sp_median": (float(np.nanmedian(c0r["L"]["sp"])), s3ap["L|hat"]["sp_median"]),
          **g87.c0_shared_refs(float(blind3.mean()), float(probe3.mean()), float(ref["land25"]["mean_regret3"]))}
    rep = g87.run_arm(D, phi_sh, (lambda s: em_sh[s]), 8, order, g86.cross_assign(tasks, groups, order))       # S3-B′ 复现臂（87 原路径，α ≤ 1e4）
    c0["s3bp_sh_r8"] = (float(g84.regret_at_k(rep["pred_sh"], D["end0"], D["end1"], 3).mean()), s3bp["rows"]["sh_r8"]["regret3"])
    c0_ok = all(abs(a - b) < 1e-12 for a, b in c0.values())
    print(f"[s3bpp] C0 {'OK' if c0_ok else 'FAIL'} {c0}", flush=True)
    # a₁^④ 列：训练任务 = 折内投影 a₁，留出任务 = ④ â₁
    def extra_of(s):
        tr = np.where(groups != s)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        col = ctx["A"][:, 0].copy()
        te = groups == s
        col[te] = ref["probe"]["a_hat"][te, 0]
        return col
    arms = {"pt": ((lambda s: em_pt), None), "sh": ((lambda s: em_sh[s]), None),
            "pt+p": ((lambda s: em_pt), extra_of), "sh+p": ((lambda s: em_sh[s]), extra_of)}
    rows, fixed, info_all = {}, {}, {}
    for name, (E_of, ex_of) in arms.items():
        print(f"[s3bpp] arm {name}", flush=True)
        R = run_arm_multi(D, phi_sh, E_of, ex_of, order, cmaps)
        row = _stats_row(D, R["sel"], blind3, probe3, groups)
        row["r_star"] = [R["info"][s]["r_star"] for s in order]
        row["alpha_star"] = [R["info"][s]["alpha"][R["info"][s]["r_star"]] for s in order]
        row["alpha_all"] = {r: [R["info"][s]["alpha"][r] for s in order] for r in RS}
        rows[name] = row
        fixed[name] = {r: _stats_row(D, R["by_r"][r], blind3, probe3, groups) for r in RS}
        info_all[name] = R["info"]
        print(f"[s3bpp] {name} r3 {row['regret3']:.4f} cross {row['cross_d']:+.4f} r* {row['r_star']}", flush=True)
    # 门（Holm / Bonferroni）
    holm = g86.holm
    pb = holm([rows[k]["p_vs_blind"] for k in ("pt", "sh")]); pc = holm([rows[k]["p_cross"] for k in ("pt", "sh")])
    gates_E = {}
    for j, k in enumerate(("pt", "sh")):
        r = rows[k]
        gates_E[k] = {"F1": bool(r["cross_d"] > 0 and pc[j] < 0.05), "F2": bool(r["vs_blind"] >= 0.010 and pb[j] < 0.05),
                      "F3": bool(r["vs_probe_ci975"][1] <= 0.005), "F4": bool(r["sp_median"] >= 0.50)}
        r["p_vs_blind_holm"], r["p_cross_holm"] = float(pb[j]), float(pc[j])
    padd = holm([rows[k]["p_probe_better"] for k in ("pt+p", "sh+p")])
    gates_C = {}
    for j, k in enumerate(("pt+p", "sh+p")):
        rows[k]["p_add_holm"] = float(padd[j])
        gates_C[k] = {"F5": bool(-rows[k]["vs_probe"] > 0 and padd[j] < 0.05)}
    code = verdict_fec(gates_E, gates_C)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    out = {"verdict": code, "c0": {k: list(v) for k, v in c0.items()}, "c0_ok": c0_ok, "gates_E": gates_E, "gates_C": gates_C,
           "rows": {k: strip(v) for k, v in rows.items()}, "fixed_r": {k: {r: strip(v) for r, v in d.items()} for k, d in fixed.items()},
           "info": info_all, "blind3": float(blind3.mean()), "probe3": float(probe3.mean()), "land3": float(ref["land25"]["mean_regret3"])}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    md = tables_markdown(out)
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return out


def _f(x, nd=3):
    try:
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def tables_markdown(o: Mapping) -> str:
    L = [f"# S3-B″ 读数（{o['verdict']}）", "", f"C0 {'✓' if o['c0_ok'] else '✗'}：" + "；".join(f"{k} {_f(a, 5)} vs {_f(b, 5)}" for k, (a, b) in o["c0"].items()), "",
         f"参照（shared-30，S3-A 口径）：任务盲 {_f(o['blind3'])}；④ {_f(o['probe3'])}；③ {_f(o['land3'])}", "",
         "| 臂（r 内层选） | r* 分布 4/8/16 | α* ≥ 1e5 折数 | random-20 中位 Sp | regret@1 | regret@3 | vs 任务盲 [CI] p（Holm） | E − ④ [97.5% CI] | cross-E（16 次）Δ [CI] p（Holm）/ 更差份额 | Δ50 |",
         "|---|---|---:|---:|---:|---:|---|---|---|---:|"]
    for k, r in o["rows"].items():
        rc = [r["r_star"].count(x) for x in RS]
        big = sum(1 for a in r["alpha_star"] if a >= 1e5)
        ph = f" ({_f(r['p_vs_blind_holm'], 3)})" if "p_vs_blind_holm" in r else ""
        pch = f" ({_f(r['p_cross_holm'], 3)})" if "p_cross_holm" in r else ""
        L.append(f"| {k} | {rc[0]}/{rc[1]}/{rc[2]} | {big}/17 | {_f(r['sp_median'], 2)} | {_f(r['regret1'])} | {_f(r['regret3'])} | "
                 f"{_f(r['vs_blind'])} [{_f(r['vs_blind_ci'][0])}, {_f(r['vs_blind_ci'][1])}] {_f(r['p_vs_blind'], 3)}{ph} | "
                 f"{_f(r['vs_probe'])} [{_f(r['vs_probe_ci975'][0])}, {_f(r['vs_probe_ci975'][1])}] | "
                 f"{_f(r['cross_d'])} [{_f(r['cross_ci'][0])}, {_f(r['cross_ci'][1])}] {_f(r['p_cross'], 3)}{pch} / {_f(r['cross_worse_frac'], 2)} | {_f(r['d50'])} |")
    L += ["", "r 固定（只报）：regret@3 / cross Δ", "", "| 臂 | r=4 | r=8 | r=16 |", "|---|---|---|---|"]
    for k, d in o["fixed_r"].items():
        L.append(f"| {k} | " + " | ".join(f"{_f(d[str(r)]['regret3'] if str(r) in d else d[r]['regret3'])} / {_f(d[str(r)]['cross_d'] if str(r) in d else d[r]['cross_d'])}" for r in RS) + " |")
    L += ["", "门：" + "；".join(f"{a}: " + ", ".join(f"{g} {'✓' if b else '✗'}" for g, b in v.items()) for a, v in {**o["gates_E"], **o["gates_C"]}.items())]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        cmd_analyze(Path(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
