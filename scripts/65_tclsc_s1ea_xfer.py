"""WP16-TCLSC 阶段 1 A″ 跨语料部署检验：UCR 84 全训的响应画像比较器 zero-shot 排 UEA 切片。协议 results/tclsc_s1ea_xfer/PROTOCOL.md。

子命令：
  --train    UCR 84 无折全训（train = 84 集、test = ∅）3 种子：u_blind / u_phi（60 号训练器）+ u_eaphi（63 号训练器）→ models/fold0/
  --rank     UEA 8 集 45 切片：normal / cross×3（供体 = 其他 UEA 数据集切片）/ zero（E_A 列或 φ 列置 0）→ rankings_xfer.json
  --analyze  126 实例（7 折靶）与 UEA 自训参照臂（s1phi blind / φ，A′ eaphi）配对；门 C0 / G-X1 / G-X2 / G-X3；读数码 + 后缀；SCP2 附录

跨语料护栏（协议 §2）：训练侧 σ 在下限的列（UCR 恒定的 log10_C / chan_cv）在目标侧置 0；标准化值裁到 ±CLIP。ctx 只是浮点向量（铁律 7）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.mech_features import RESPONSE_PROFILE_COLS  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(_m33.PHI_COLS)
BUDGETS = pq.BUDGETS
OUT = ROOT / "results" / "tclsc_s1ea_xfer"
SOURCE = pq.OUT
UEA_EA = ROOT / "results" / "tclsc_s1ea_confirm" / "uea" / "ea_uea.jsonl"
S1PHI_RANK = ROOT / "results" / "tclsc_s1phi" / "binned" / "rankings_s1phi.json"
S1PHI_ANALYSIS = ROOT / "results" / "tclsc_s1phi" / "analysis.json"
CONF_RANK = ROOT / "results" / "tclsc_s1ea_confirm" / "uea" / "rankings_uea.json"
CONF_ANALYSIS = ROOT / "results" / "tclsc_s1ea_confirm" / "analysis.json"
ARMS = ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")
CTX_ARMS = ("u_phi_diff_hard", "u_eaphi_diff_hard")
REF_ARMS = ("blind_diff_hard", "phi_diff_hard", "eaphi_diff_hard")
FOLDS = tuple(pq.FOLD_TARGETS)
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
FOLD = "fold0"
CLIP = 4.0
DEAD_MULT = 10.0
ALPHA = 0.05
C0_TOL = 0.0002


# ---------------------------------------------------------------- 护栏 + ctx 行
def dead_cols(sd: np.ndarray) -> np.ndarray:
    """训练侧 σ 在下限（≤ DEAD_MULT × PHI_SD_FLOOR）的列：目标侧置 0（UCR 恒定列在 UEA 上非零会被 1/σ 放大到无意义）。"""
    return np.asarray(sd, np.float64) <= DEAD_MULT * _m33.PHI_SD_FLOOR


def xfer_ctx_rows(table: Mapping[str, Mapping], cols: Sequence[str], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                  cross_map: Optional[Mapping[str, str]] = None, zero_cols: Sequence[str] = (),
                  budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    """{(切片, 预算): 护栏后的标准化列(供体或自己) ⊕ 预算 one-hot}。"""
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    dead = dead_cols(sd)
    zero_idx = [i for i, c in enumerate(cols) if c in set(zero_cols)]
    out = {}
    for s in slices:
        src = cross_map[s] if cross_map is not None else s
        x, _n = _m33.table_row_vector(table[src], cols, mu, sd)
        x = np.clip(x, -CLIP, CLIP)
        x[dead] = 0.0
        if zero_idx:
            x[zero_idx] = 0.0
        for b in budgets:
            out[(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


# ---------------------------------------------------------------- 训练（无折）
def train_all(rows: Sequence[dict], ea: Mapping, phi: Mapping, outdir: Path, *, seeds: Sequence[int], device: str,
              epochs: Optional[int], resume: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    folds = [[]]                                     # test = ∅ → train = 全部数据集
    log = outdir / "train.log"
    for arm in ARMS:
        for seed in seeds:
            p = outdir / "models" / FOLD / f"{arm}_s{seed}.pt"
            if resume and p.exists():
                continue
            if arm in _g63.ARMS:
                run = _g63.train_fold(rows, ea, phi, folds, 0, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
            else:
                run = _g60.train_fold(rows, phi, folds, 0, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
            line = f"[{FOLD}/{arm}_s{seed}] {run['train_seconds']}s | bce {run['bce_first']:.4f}→{run['bce_last']:.4f} | pairs {run['train_pairs_last']} | n_train {run['n_train_records']}"
            print(line, flush=True)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool) -> None:
    rows, ea, phi = _g63._load_all()
    train_all(rows, ea, phi, outdir, seeds=seeds, device=device, epochs=epochs, resume=resume)


# ---------------------------------------------------------------- 排名
def load_models(outdir: Path, *, seeds: Sequence[int], device: str = "cpu") -> dict[str, list]:
    out = {}
    for a in ARMS:
        paths = [outdir / "models" / FOLD / f"{a}_s{s}.pt" for s in seeds]
        if all(p.exists() for p in paths):
            out[a] = [_g34.load_pq_model(p, map_location=device, expect_fold=FOLD, expect_budgets=BUDGETS) for p in paths]
        else:
            print(f"⚠️ [{a}] 模型不全，跳过", flush=True)
    return out


def _ctx_spec(arm: str, meta: Mapping) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    """(列, μ, σ, zero 时置零的列)。"""
    if arm == "u_phi_diff_hard":
        return PHI_COLS, np.array(meta["phi_mu"]), np.array(meta["phi_sd"]), PHI_COLS
    if arm == "u_eaphi_diff_hard":
        cols = tuple(meta["ctx_cols"])
        return cols, np.array(meta["ctx_mu"]), np.array(meta["ctx_sd"]), EA_COLS
    raise KeyError(arm)


def rank_target(models: Mapping[str, Sequence[tuple]], cat: Mapping[str, Sequence[tuple]], table: Mapping[str, Mapping],
                test_slices: Sequence[str], *, donor_slices: Sequence[str], device: str = "cpu") -> dict:
    """一个靶：normal（三臂）/ cross×3 / zero（ctx 臂）。目录只取考侧切片（rank_local_models 按目录里的切片打分）。"""
    cat = {s: cat[s] for s in test_slices}
    per: dict = {"normal": {}, "cross": {}, "zero": {}}

    def merge(dst: dict, rk: dict) -> None:
        for b, pb in rk.items():
            for s, pa in pb.items():
                dst.setdefault(b, {}).setdefault(s, {}).update(pa)

    blind = {a: m for a, m in models.items() if a not in CTX_ARMS}
    if blind:
        merge(per["normal"], _g56.rank_local_models(blind, cat, budgets=BUDGETS, device=device))
    for a in CTX_ARMS:
        if a not in models:
            continue
        cols, mu, sd, zcols = _ctx_spec(a, models[a][0][1])
        rows_fn = lambda cm=None, zero=False, cols=cols, mu=mu, sd=sd, zcols=zcols: xfer_ctx_rows(
            table, cols, test_slices, mu, sd, cross_map=cm, zero_cols=(zcols if zero else ()))
        one = {a: models[a]}
        merge(per["normal"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn()))
        merge(per["zero"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
        for cs in CROSS_SEEDS:
            cmap = _g58.cross_slice_map(test_slices, donor_slices, cs)
            per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
            merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
    return per


def cmd_rank(outdir: Path, *, seeds: Sequence[int], device: str = "cpu") -> None:
    models = load_models(outdir, seeds=seeds, device=device)
    table = {r["task_id"]: r for r in read_jsonl(UEA_EA)}
    rows, _delta = _g57.load_source(SOURCE)
    all_slices = sorted({r["task_id"] for r in rows})
    out = {}
    for target in sorted({pq.dataset_of(s) for s in all_slices}):
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        catalog, _g, _k = _g34.load_catalog(SOURCE, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        out[target] = rank_target(models, cat, table, test_slices, donor_slices=all_slices, device=device)
        print(f"[xfer/{target}] {len(test_slices)} 切片", flush=True)
    (outdir / "rankings_xfer.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def merge_refs(own: Mapping[str, Mapping], refs: Sequence[tuple[Mapping, Sequence[str]]]) -> dict:
    """把若干参照排名表（按靶名对折）的指定臂的 normal 排名并进本实验表；参照缺的靶跳过（SCP2 无参照）。"""
    out = {f: {"normal": {b: {s: dict(pa) for s, pa in pb.items()} for b, pb in per["normal"].items()},
               "cross": per["cross"], "zero": per["zero"]} for f, per in own.items()}
    for ref, arms in refs:
        for f, per in out.items():
            if f not in ref:
                continue
            for b, pb in ref[f]["normal"].items():
                for s, pa in pb.items():
                    keep = {a: pa[a] for a in arms if a in pa}
                    if keep:
                        per["normal"].setdefault(b, {}).setdefault(s, {}).update(keep)
    return out


# ---------------------------------------------------------------- 门 / 读数码
def _lower(v: Mapping) -> bool:
    return bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])


def gates_xfer(v_x1: Mapping, v_x2: Mapping, v_x3: Mapping) -> dict:
    """v_x1 = blind(UEA) → u_eaphi；v_x2 = u_blind(UCR) → u_eaphi（H1 b<a）；v_x3 = u_eaphi normal → cross（H1 cross 更差）。"""
    worse = bool(v_x3.get("p") is not None and v_x3["p"] < ALPHA and v_x3["mean_b"] > v_x3["mean_a"])
    return {"GX1": {"metric": "regret3", "blind_uea": v_x1.get("mean_a"), "u_eaphi": v_x1.get("mean_b"), "p": v_x1.get("p"), "pass": _lower(v_x1)},
            "GX2": {"metric": "regret3", "u_blind_ucr": v_x2.get("mean_a"), "u_eaphi": v_x2.get("mean_b"), "p": v_x2.get("p"), "pass": _lower(v_x2)},
            "GX3": {"metric": "regret3", "normal": v_x3.get("mean_a"), "cross": v_x3.get("mean_b"), "p": v_x3.get("p"), "pass": worse},
            "thresholds": {"alpha": ALPHA}}


def verdict_xfer(g: Mapping) -> str:
    code = {(True, True): "XFER-ALIVE", (True, False): "XFER-CORPUS", (False, True): "XFER-RELATIVE", (False, False): "XFER-DEAD"}[
        (bool(g["GX1"]["pass"]), bool(g["GX2"]["pass"]))]
    return code + ("+CONTENT" if g["GX3"]["pass"] else "-CONTENT")


# ---------------------------------------------------------------- 分析
def cmd_analyze(outdir: Path) -> dict:
    rk_own = json.loads((outdir / "rankings_xfer.json").read_text(encoding="utf-8"))
    rk_all = merge_refs(rk_own, [(json.loads(S1PHI_RANK.read_text(encoding="utf-8")), ("blind_diff_hard", "phi_diff_hard")),
                                 (json.loads(CONF_RANK.read_text(encoding="utf-8")), ("eaphi_diff_hard",))])
    rk = {t: per for t, per in rk_all.items() if t in FOLDS}
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    arms = list(REF_ARMS) + list(ARMS)
    main = _g64._block(rk, cat_z, delta, arms, list(CTX_ARMS), eval_fn=_g58._eval_all)
    normal = main["per_instance_normal"]
    pairs = [("blind_diff_hard", "u_eaphi_diff_hard"), ("u_blind_diff_hard", "u_eaphi_diff_hard"), ("blind_diff_hard", "u_blind_diff_hard"),
             ("phi_diff_hard", "u_phi_diff_hard"), ("blind_diff_hard", "u_phi_diff_hard"), ("eaphi_diff_hard", "u_eaphi_diff_hard"),
             ("u_phi_diff_hard", "u_eaphi_diff_hard"), ("u_blind_diff_hard", "u_phi_diff_hard")]
    main["value_tests"] = {k: [_g58.paired_regret(normal, a, b, key=k) for a, b in pairs if a in main["arms"] and b in main["arms"]] for k in ("1", "3")}
    find = lambda tests, a, b: next((t for t in tests if t["a"] == a and t["b"] == b), {"p": None, "mean_a": None, "mean_b": None})
    v1 = find(main["value_tests"]["3"], "blind_diff_hard", "u_eaphi_diff_hard")
    v2 = find(main["value_tests"]["3"], "u_blind_diff_hard", "u_eaphi_diff_hard")
    v3 = main["regret_ablation"].get("u_eaphi_diff_hard", {}).get("3", {}).get("cross", {"p": None, "mean_a": None, "mean_b": None})
    g = gates_xfer(v1, v2, v3)
    # C0：参照臂 regret@3 与来源 analysis 一致
    c0 = {}
    s1 = json.loads(S1PHI_ANALYSIS.read_text(encoding="utf-8"))["pooled"] if S1PHI_ANALYSIS.exists() else {}
    cf = json.loads(CONF_ANALYSIS.read_text(encoding="utf-8"))["uea"]["pooled"] if CONF_ANALYSIS.exists() else {}
    for a, src in (("blind_diff_hard", s1), ("phi_diff_hard", s1), ("eaphi_diff_hard", cf)):
        ref_v, got = src.get(a, {}).get("regret3"), main["pooled"].get(a, {}).get("regret3")
        c0[a] = {"ref": ref_v, "got": got, "pass": None if (ref_v is None or got is None) else bool(abs(ref_v - got) <= C0_TOL)}
    c0["pass"] = all(v["pass"] for k, v in c0.items() if k != "pass") if all(v["pass"] is not None for k, v in c0.items() if k != "pass") else None
    # SCP2 附录（无参照臂）
    appendix = None
    extra = {t: per for t, per in rk_all.items() if t not in FOLDS}
    if extra:
        ex_rows = _g58._eval_all(extra, cat_z, delta, list(ARMS), "normal")
        appendix = {"targets": sorted(extra), "n_instances": len(ex_rows),
                    "regret3": {a: float(np.mean([r["regret"][a + "3"] for r in ex_rows if a + "3" in r["regret"]])) for a in ARMS if any(a + "3" in r["regret"] for r in ex_rows)}}
    a = {"protocol": "results/tclsc_s1ea_xfer/PROTOCOL.md", "main": main, "appendix_scp2": appendix, "c0": c0, "gates": g, "verdict": verdict_xfer(g)}
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    pu = main["pooled"]
    print(f"{a['verdict']} | " + " | ".join(f"{arm} M0 {pu[arm]['pair_acc']:.3f} r1 {pu[arm]['regret1']:.4f} r3 {pu[arm]['regret3']:.4f}" for arm in main["arms"]), flush=True)
    return a


def predictions(a: dict) -> dict:
    g, pu = a["gates"], a["main"]["pooled"]
    find = lambda tests, x, y: next((t for t in tests if t["a"] == x and t["b"] == y), {"p": None, "mean_a": None, "mean_b": None})
    v5 = find(a["main"]["value_tests"]["3"], "blind_diff_hard", "u_blind_diff_hard")
    v7 = find(a["main"]["value_tests"]["1"], "blind_diff_hard", "u_eaphi_diff_hard")
    m0 = pu.get("u_eaphi_diff_hard", {}).get("pair_acc")
    return {"P1": {"desc": "C0", "hit": a["c0"].get("pass")},
            "P2": {"desc": "G-X1 过", "hit": g["GX1"]["pass"]}, "P3": {"desc": "G-X2 过", "hit": g["GX2"]["pass"]}, "P4": {"desc": "G-X3 过", "hit": g["GX3"]["pass"]},
            "P5": {"desc": "u_blind(UCR) regret@3 ≥ blind(UEA)", "value": [v5.get("mean_a"), v5.get("mean_b")],
                   "hit": None if v5.get("mean_b") is None else bool(v5["mean_b"] >= v5["mean_a"])},
            "P6": {"desc": "u_eaphi M0 ≥ 0.75", "value": m0, "hit": None if m0 is None else bool(m0 >= 0.75)},
            "P7": {"desc": "u_eaphi regret@1 < blind(UEA)（p<0.05）", "value": [v7.get("mean_a"), v7.get("mean_b"), v7.get("p")], "hit": _lower(v7)}}


def markdown_tables(a: dict) -> str:
    f = _g64._fmt
    blk = a["main"]
    L = [f"# WP16-TCLSC 阶段 1 A″ 跨语料部署检验 tables（UEA 7 靶 {blk['n_instances']} 实例）", "", f"读数码 **{a['verdict']}**", "",
         "| 臂 | 训练侧 | M1 | M2 | M0 | regret@1 | regret@3 |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm, b in blk["pooled"].items():
        L.append(f"| {arm} | {'UCR 84' if arm.startswith('u_') else 'UEA 7 折'} | {f(b.get('acc_top10'), 3)} | {f(b.get('wacc'), 3)} | {f(b.get('pair_acc'), 3)} | {f(b['regret1'])} | {f(b['regret3'])} |")
    L += ["", "## 价值（配对 Wilcoxon，H1 b < a）", ""]
    for k in ("3", "1"):
        for t in blk["value_tests"][k]:
            L.append(f"- regret@{k} {t['a']} → {t['b']}: n={t['n']} {f(t['mean_a'])} → {f(t['mean_b'])} p={t['p']:.3g} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
    L += ["", "## 消融下的 regret（H1 消融更差）", ""]
    for arm, kk in blk["regret_ablation"].items():
        for k, d in kk.items():
            for kind, t in d.items():
                if t.get("p") is not None:
                    L.append(f"- {arm} regret@{k} {kind}: {f(t['mean_a'])} → {f(t['mean_b'])} p={t['p']:.3g} 更差/更好/平 {t['worse']}/{t['better']}/{t['ties']}")
    L += ["", "## 消融 Δ（配对精度）", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
    for kind in ("cross", "zero"):
        for arm, dd in blk["deltas"][kind].items():
            for metric, d in dd.items():
                if d.get("mean_delta") is not None:
                    L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | {d['wins']}/{d['losses']}/{d['ties']} |")
    L += ["", "## 逐靶 regret@3", ""]
    for arm, bf in blk["by_fold_regret3"].items():
        L.append(f"- {arm}: " + ", ".join(f"{fo} {v:.4f}" for fo, v in bf.items()))
    if a.get("appendix_scp2"):
        ap = a["appendix_scp2"]
        L += ["", f"## 附录：{ap['targets']}（{ap['n_instances']} 实例，无参照臂）regret@3", ""] + [f"- {arm}: {v:.4f}" for arm, v in ap["regret3"].items()]
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    for k, v in a["c0"].items():
        if k != "pass":
            L.append(f"| C0 {k} | ref {v['ref']} got {v['got']} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.train:
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds, device="cpu")
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
