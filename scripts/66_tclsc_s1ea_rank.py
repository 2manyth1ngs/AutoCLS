"""WP16-TCLSC 阶段 1 R 检验：瓶颈是"E 被要求预测高秩对象"还是 E 本身——field 头（低秩字段加性任务项）在 UEA n=7 与 UCR n=72 对照。
协议 results/tclsc_s1ea_rank/PROTOCOL.md。

子命令：
  --train    UEA：变体目录 uea/（+ 复制 A′ 的 ea_uea.jsonl）→ 33 号训 fw4_eaphi_hard / fw_eaphi_hard；UCR：63 号训 u_fw4 / u_fw（种子 0/1/2）
  --rank     UEA：64 号 rank_uea(arms=)；UCR：63 号 cmd_rank（缺模型的臂自动跳过）
  --analyze  两考场与参照臂（UEA：s1phi blind / φ + A′ eaphi_diff；UCR：s1ucr u_blind / u_phi + s1ea u_eaphi_diff）合并配对；门 C0 / G-R1 / G-R2 / G-R3；读数码
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pq_common as pq  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
BUDGETS = pq.BUDGETS
OUT = ROOT / "results" / "tclsc_s1ea_rank"
SOURCE = pq.OUT
CONF = ROOT / "results" / "tclsc_s1ea_confirm"
S1PHI = ROOT / "results" / "tclsc_s1phi"
S1UCR = ROOT / "results" / "tclsc_s1ucr"
S1EA = ROOT / "results" / "tclsc_s1ea"
UEA_ARMS = ("fw4_eaphi_hard", "fw_eaphi_hard")
UEA_REFS = ("blind_diff_hard", "phi_diff_hard", "eaphi_diff_hard")
UCR_ARMS = ("u_fw4_eaphi_hard", "u_fw_eaphi_hard")
UCR_REFS = ("u_blind_diff_hard", "u_phi_diff_hard", "u_eaphi_diff_hard")
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
ALPHA = 0.05
C0_TOL = 0.0002


# ---------------------------------------------------------------- 训练 / 排名
def cmd_train(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, which: Sequence[str]) -> None:
    if "uea" in which:
        d = outdir / "uea"
        _g56.prepare_variant_dir(SOURCE, d)
        src = CONF / "uea" / "ea_uea.jsonl"
        if not (d / "ea_uea.jsonl").exists():
            shutil.copy(src, d / "ea_uea.jsonl")
        _m33.cmd_train(d, arms=list(UEA_ARMS), folds=None, seeds=list(seeds), resume=resume, device=device, epochs=epochs,
                       settf_windows=None, encoder_mode="binned", ctx_file=d / "ea_uea.jsonl")
    if "ucr" in which:
        (outdir / "ucr").mkdir(parents=True, exist_ok=True)          # 63 号 cmd_train 假定目录已存在（其 main 里才 mkdir）
        _g63.cmd_train(outdir / "ucr", arms=list(UCR_ARMS), seeds=list(seeds), folds_sel=None, device=device, epochs=epochs, resume=resume)


def cmd_rank(outdir: Path, *, seeds: Sequence[int]) -> None:
    rk = _g64.rank_uea(outdir, seeds=seeds, device="cpu", arms=UEA_ARMS)
    (outdir / "uea" / "rankings_uea.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    if (outdir / "ucr" / "folds.json").exists():
        _g63.cmd_rank(outdir / "ucr", seeds=seeds)


# ---------------------------------------------------------------- 门 / 读数码
def _lower(v: Mapping) -> bool:
    return bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])


def gates_rank(v_r1: Mapping, v_r2: Mapping, v_r3: Mapping) -> dict:
    """v_r1 = blind(UEA) → fw4（H1 b<a）；v_r2 = fw4 normal → cross（H1 cross 更差）；v_r3 = u_blind → u_fw4（UCR）。"""
    worse = bool(v_r2.get("p") is not None and v_r2["p"] < ALPHA and v_r2["mean_b"] > v_r2["mean_a"])
    return {"GR1": {"metric": "regret3", "blind": v_r1.get("mean_a"), "fw4": v_r1.get("mean_b"), "p": v_r1.get("p"), "pass": _lower(v_r1)},
            "GR2": {"metric": "regret3", "normal": v_r2.get("mean_a"), "cross": v_r2.get("mean_b"), "p": v_r2.get("p"), "pass": worse},
            "GR3": {"metric": "regret3", "u_blind": v_r3.get("mean_a"), "u_fw4": v_r3.get("mean_b"), "p": v_r3.get("p"), "pass": _lower(v_r3)},
            "thresholds": {"alpha": ALPHA}}


def verdict_rank(g: Mapping) -> str:
    r1, r2, r3 = g["GR1"]["pass"], g["GR2"]["pass"], g["GR3"]["pass"]
    if r1:
        return ("RANK-BOTTLENECK" + ("+RETAINED" if r3 else "-TRADEOFF")) if r2 else "RANK-EXISTENCE"
    return "E-OR-N" if r3 else "HEAD-TOO-WEAK"


# ---------------------------------------------------------------- 分析
def _c0(pooled: Mapping, refs: Mapping[str, Optional[float]]) -> dict:
    out = {}
    for a, ref_v in refs.items():
        got = pooled.get(a, {}).get("regret3")
        out[a] = {"ref": ref_v, "got": got, "pass": None if (ref_v is None or got is None) else bool(abs(ref_v - got) <= C0_TOL)}
    vals = [v["pass"] for v in out.values()]
    out["pass"] = all(vals) if all(v is not None for v in vals) else None
    return out


def _load_json(p: Path) -> Optional[dict]:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cmd_analyze(outdir: Path) -> dict:
    find = lambda tests, a, b: next((t for t in tests if t["a"] == a and t["b"] == b), {"p": None, "mean_a": None, "mean_b": None})
    # --- UEA（n=7）
    rk_uea = _g64.merge_ref(json.loads((outdir / "uea" / "rankings_uea.json").read_text(encoding="utf-8")),
                            json.loads((S1PHI / "binned" / "rankings_s1phi.json").read_text(encoding="utf-8")), arms=("blind_diff_hard", "phi_diff_hard"))
    rk_uea = _g64.merge_ref(rk_uea, json.loads((CONF / "uea" / "rankings_uea.json").read_text(encoding="utf-8")), arms=("eaphi_diff_hard",))
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    uea = _g64._block(rk_uea, cat_z, delta, list(UEA_REFS) + list(UEA_ARMS), list(UEA_ARMS), eval_fn=_g58._eval_all)
    nu = uea["per_instance_normal"]
    pairs_u = [(r, a) for a in UEA_ARMS for r in UEA_REFS] + [("fw_eaphi_hard", "fw4_eaphi_hard")]
    uea["value_tests"] = {k: [_g58.paired_regret(nu, a, b, key=k) for a, b in pairs_u if a in uea["arms"] and b in uea["arms"]] for k in ("1", "3")}
    s1phi_an, conf_an = _load_json(S1PHI / "analysis.json"), _load_json(CONF / "analysis.json")
    c0_uea = _c0(uea["pooled"], {"blind_diff_hard": (s1phi_an or {}).get("pooled", {}).get("blind_diff_hard", {}).get("regret3"),
                                 "phi_diff_hard": (s1phi_an or {}).get("pooled", {}).get("phi_diff_hard", {}).get("regret3"),
                                 "eaphi_diff_hard": ((conf_an or {}).get("uea", {}).get("pooled", {}).get("eaphi_diff_hard", {}) or {}).get("regret3")})
    # --- UCR（n=72）
    ucr, c0_ucr = None, {"pass": None}
    p_ucr = outdir / "ucr" / "rankings_ea.json"
    if p_ucr.exists():
        own = json.loads(p_ucr.read_text(encoding="utf-8"))
        rk_ucr = _g63.merge_ref_rankings(own, json.loads((S1UCR / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_blind_diff_hard", "u_phi_diff_hard"))
        rk_ucr = _g63.merge_ref_rankings(rk_ucr, json.loads((S1EA / "rankings_ea.json").read_text(encoding="utf-8")), arms=("u_eaphi_diff_hard",))
        rows_u, _ea, _phi = _g63._load_all()
        cat_u = _g56.zscore_within(rows_u)
        delta_u = {(d, int(b)): _g60.DELTA_UCR for d in {r["dataset"] for r in rows_u} for b in BUDGETS}
        ucr = _g64._block(rk_ucr, cat_u, delta_u, list(UCR_REFS) + list(UCR_ARMS), list(UCR_ARMS), eval_fn=_g64._eval_ucr)
        nc = ucr["per_instance_normal"]
        pairs_c = [(r, a) for a in UCR_ARMS for r in UCR_REFS] + [("u_fw_eaphi_hard", "u_fw4_eaphi_hard")]
        ucr["value_tests"] = {k: [_g58.paired_regret(nc, a, b, key=k) for a, b in pairs_c if a in ucr["arms"] and b in ucr["arms"]] for k in ("1", "3")}
        s1ucr_an, s1ea_an = _load_json(S1UCR / "analysis.json"), _load_json(S1EA / "analysis.json")
        pu = (s1ucr_an or {}).get("views", {}).get("all20", {}).get("pooled", {})
        c0_ucr = _c0(ucr["pooled"], {"u_blind_diff_hard": pu.get("u_blind_diff_hard", {}).get("regret3"), "u_phi_diff_hard": pu.get("u_phi_diff_hard", {}).get("regret3"),
                                     "u_eaphi_diff_hard": (s1ea_an or {}).get("pooled", {}).get("u_eaphi_diff_hard", {}).get("regret3")})
    v1 = find(uea["value_tests"]["3"], "blind_diff_hard", "fw4_eaphi_hard")
    v2 = uea["regret_ablation"].get("fw4_eaphi_hard", {}).get("3", {}).get("cross", {"p": None, "mean_a": None, "mean_b": None})
    v3 = find(ucr["value_tests"]["3"], "u_blind_diff_hard", "u_fw4_eaphi_hard") if ucr else {"p": None, "mean_a": None, "mean_b": None}
    g = gates_rank(v1, v2, v3)
    a = {"protocol": "results/tclsc_s1ea_rank/PROTOCOL.md", "uea": uea, "ucr": ucr, "c0": {"uea": c0_uea, "ucr": c0_ucr}, "gates": g, "verdict": verdict_rank(g)}
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"{a['verdict']} | UEA " + " | ".join(f"{arm} r3 {uea['pooled'][arm]['regret3']:.4f}" for arm in uea["arms"])
          + (" | UCR " + " | ".join(f"{arm} r3 {ucr['pooled'][arm]['regret3']:.4f}" for arm in ucr["arms"]) if ucr else ""), flush=True)
    return a


def predictions(a: dict) -> dict:
    g = a["gates"]
    find = lambda tests, x, y: next((t for t in tests if t["a"] == x and t["b"] == y), {"p": None, "mean_a": None, "mean_b": None})
    v5 = find(a["uea"]["value_tests"]["3"], "eaphi_diff_hard", "fw4_eaphi_hard")
    pu = a["uea"]["pooled"]
    fw, fw4 = pu.get("fw_eaphi_hard", {}).get("regret3"), pu.get("fw4_eaphi_hard", {}).get("regret3")
    m1 = (a["ucr"] or {}).get("pooled", {}).get("u_fw4_eaphi_hard", {}).get("acc_top10")
    c0 = a["c0"]
    return {"P1": {"desc": "C0", "hit": None if c0["uea"]["pass"] is None else bool(c0["uea"]["pass"] and (c0["ucr"]["pass"] is not False))},
            "P2": {"desc": "G-R1 过", "hit": g["GR1"]["pass"]}, "P3": {"desc": "G-R2 过", "hit": g["GR2"]["pass"]}, "P4": {"desc": "G-R3 过", "hit": g["GR3"]["pass"]},
            "P5": {"desc": "UEA fw4 regret@3 < eaphi_diff（p<0.05）", "value": [v5.get("mean_a"), v5.get("mean_b"), v5.get("p")], "hit": _lower(v5)},
            "P6": {"desc": "UEA fw（不限秩）regret@3 > fw4", "value": [fw, fw4], "hit": None if fw is None or fw4 is None else bool(fw > fw4)},
            "P7": {"desc": "UCR fw4 M1 ≥ 0.60", "value": m1, "hit": None if m1 is None else bool(m1 >= 0.60)}}


def markdown_tables(a: dict) -> str:
    f = _g64._fmt
    L = ["# WP16-TCLSC 阶段 1 R 检验 tables（field 头）", "", f"读数码 **{a['verdict']}**", ""]
    for title, blk in (("UEA 7 折 LODO（n=7，126 实例）", a["uea"]), ("UCR 84 分组 7 折（n=72，252 实例）", a.get("ucr"))):
        if not blk:
            continue
        L += [f"## {title}", "", "| 臂 | M1 | M2 | M0 | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|"]
        for arm, b in blk["pooled"].items():
            L.append(f"| {arm} | {f(b.get('acc_top10'), 3)} | {f(b.get('wacc'), 3)} | {f(b.get('pair_acc'), 3)} | {f(b['regret1'])} | {f(b['regret3'])} |")
        L += ["", "### 价值（配对 Wilcoxon，H1 b < a）", ""]
        for k in ("3", "1"):
            for t in blk["value_tests"][k]:
                L.append(f"- regret@{k} {t['a']} → {t['b']}: n={t['n']} {f(t['mean_a'])} → {f(t['mean_b'])} p={t['p']:.3g} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
        L += ["", "### 消融下的 regret（H1 消融更差）", ""]
        for arm, kk in blk["regret_ablation"].items():
            for k, d in kk.items():
                for kind, t in d.items():
                    if t.get("p") is not None:
                        L.append(f"- {arm} regret@{k} {kind}: {f(t['mean_a'])} → {f(t['mean_b'])} p={t['p']:.3g} 更差/更好/平 {t['worse']}/{t['better']}/{t['ties']}")
        L += ["", "### 消融 Δ（配对精度）", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
        for kind in ("cross", "zero"):
            for arm, dd in blk["deltas"][kind].items():
                for metric, d in dd.items():
                    if d.get("mean_delta") is not None:
                        L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | {d['wins']}/{d['losses']}/{d['ties']} |")
        L += ["", "### 逐折 regret@3", ""] + [f"- {arm}: " + ", ".join(f"{fo} {v:.4f}" for fo, v in bf.items()) for arm, bf in blk["by_fold_regret3"].items()] + [""]
    L += ["## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    for site in ("uea", "ucr"):
        for k, v in a["c0"][site].items():
            if k != "pass":
                L.append(f"| C0 {site} {k} | ref {v['ref']} got {v['got']} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--which", nargs="+", default=["uea", "ucr"], choices=["uea", "ucr"])
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
        cmd_train(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, which=args.which)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
