"""R 检验 §9 消融补充：数据集特征（E_A ⊕ φ12）对 field 头的贡献——训练侧对照臂 fwb（field 头 + 只有预算 one-hot）与推理侧 zero_all / noise。
协议 results/tclsc_s1ea_rank/PROTOCOL.md §9。产物 results/tclsc_s1ea_rank/ablate/。

子命令：--train（fwb_hard @ UEA via 33；u_fwb_hard @ UCR via 60）/ --rank（fwb 排名 + 已训 fw 的 zero_all / noise 行）/ --analyze。
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
_g66 = importlib.import_module("scripts.66_tclsc_s1ea_rank")
EA_COLS = tuple(RESPONSE_PROFILE_COLS)
PHI_COLS = tuple(_m33.PHI_COLS)
BUDGETS = pq.BUDGETS
RANK_OUT = ROOT / "results" / "tclsc_s1ea_rank"
OUT = RANK_OUT / "ablate"
SOURCE = pq.OUT
MODES = ("zero_all", "noise")
NOISE_SEEDS = (0, 1, 2)
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
ALPHA = 0.05


# ---------------------------------------------------------------- 推理侧消融行
def ablate_rows(table: Mapping[str, Mapping], cols: Sequence[str], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                mode: str, seed: int = 0, budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    """zero_all：特征列全 0；noise：每切片一个 N(0,1) 向量（各预算同一向量，按 seed 逐位确定）。预算 one-hot 不动。"""
    if mode not in MODES:
        raise ValueError(f"mode 须为 {MODES}，收到 {mode!r}")
    out = {}
    for s in sorted(slices):
        if mode == "zero_all":
            x = np.zeros(len(cols), dtype=np.float64)
        else:
            rng = np.random.default_rng([int(seed), abs(hash(s)) % (2 ** 31), 202609])
            x = rng.standard_normal(len(cols))
        for b in budgets:
            out[(s, int(b))] = np.concatenate([x, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


# ---------------------------------------------------------------- 训练 / 排名
def cmd_train(*, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool) -> None:
    d = RANK_OUT / "uea"
    _m33.cmd_train(d, arms=["fwb_hard"], folds=None, seeds=list(seeds), resume=resume, device=device, epochs=epochs,
                   settf_windows=None, encoder_mode="binned")
    (RANK_OUT / "ucr").mkdir(parents=True, exist_ok=True)
    _g60.cmd_train(RANK_OUT / "ucr", arms=["u_fwb_hard"], seeds=list(seeds), folds_sel=None, device=device, epochs=epochs, resume=resume)


def rank_uea(*, seeds: Sequence[int], device: str = "cpu") -> dict:
    d = RANK_OUT / "uea"
    table = {r["task_id"]: r for r in read_jsonl(d / "ea_uea.jsonl")}
    rows, _delta = _g57.load_source(SOURCE)
    all_slices = sorted({r["task_id"] for r in rows})
    out = {}
    for target in FOLDS:
        test_slices = [s for s in all_slices if pq.dataset_of(s) == target]
        catalog, _g, _k = _g34.load_catalog(SOURCE, _g56.space(), slices=test_slices)
        cat = {s: catalog[s] for s in test_slices}
        per: dict = {"normal": {}, "ablate": {}}
        fwb = _g64._load_arm_models(d, target, "fwb_hard", seeds, device)
        if fwb:
            per["normal"] = _g56.rank_local_models({"fwb_hard": fwb}, cat, budgets=BUDGETS, device=device)
        fw = _g64._load_arm_models(d, target, "fw_eaphi_hard", seeds, device)
        if fw:
            meta0 = fw[0][1]
            cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
            one = {"fw_eaphi_hard": fw}
            per["ablate"]["zero_all"] = _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device,
                                                              ctx_rows=ablate_rows(table, cols, test_slices, mu, sd, mode="zero_all"))
            for ns in NOISE_SEEDS:
                per["ablate"][f"noise{ns}"] = _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device,
                                                                    ctx_rows=ablate_rows(table, cols, test_slices, mu, sd, mode="noise", seed=ns))
        out[target] = per
        print(f"[ablate uea/{target}]", flush=True)
    return out


def rank_ucr(*, seeds: Sequence[int], device: str = "cpu") -> dict:
    d = RANK_OUT / "ucr"
    rows, ea, phi = _g63._load_all()
    folds = json.loads((d / "folds.json").read_text(encoding="utf-8"))
    table = _g63.merge_tables(ea, phi)
    out = {}
    for idx in range(len(folds)):
        fold, test = f"fold{idx}", list(folds[idx])
        test_t = [f"{x}_00" for x in sorted(test)]
        cat: dict = {}
        for x in test:
            cat.update(_g63._g60.catalog_for(rows, x))
        per: dict = {"normal": {}, "ablate": {}, "test": test}
        a = _g60.rank_fold(rows, phi, folds, idx, arms=["u_fwb_hard"], seeds=seeds, outdir=d, device=device)
        per["normal"] = a["normal"]
        paths = [d / "models" / fold / f"u_fw_eaphi_hard_s{s}.pt" for s in seeds]
        if all(p.exists() for p in paths):
            fw = [_g34.load_pq_model(p, map_location=device, expect_fold=fold, expect_budgets=BUDGETS) for p in paths]
            meta0 = fw[0][1]
            cols, mu, sd = tuple(meta0["ctx_cols"]), np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
            tab_s = {s: table[pq.dataset_of(s)] for s in test_t}
            one = {"u_fw_eaphi_hard": fw}
            per["ablate"]["zero_all"] = _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device,
                                                              ctx_rows=ablate_rows(tab_s, cols, test_t, mu, sd, mode="zero_all"))
            for ns in NOISE_SEEDS:
                per["ablate"][f"noise{ns}"] = _g56.rank_local_models(one, cat, budgets=BUDGETS, device=device,
                                                                    ctx_rows=ablate_rows(tab_s, cols, test_t, mu, sd, mode="noise", seed=ns))
        out[fold] = per
        print(f"[ablate ucr/{fold}]", flush=True)
    return out


def cmd_rank(*, seeds: Sequence[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rankings_uea.json").write_text(json.dumps(rank_uea(seeds=seeds), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "rankings_ucr.json").write_text(json.dumps(rank_ucr(seeds=seeds), ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 分析
def _rows_for(rk_normal: Mapping, cat_z, delta, arms, eval_fn):
    return eval_fn({t: {"normal": per, "cross": {}, "zero": {}, **({"test": tst} if tst else {})} for t, (per, tst) in rk_normal.items()},
                   cat_z, delta, arms, "normal")


def _site(rk_ab: Mapping, rk_main: Mapping, cat_z, delta, *, arms_main: Sequence[str], fw: str, fwb: str, eval_fn) -> dict:
    """一考场：normal 行（参照臂 + fw + fwb）与 fw 的 zero_all / noise 行 → 配对读数。"""
    def normal_of(rk):
        return {t: (per["normal"], per.get("test")) for t, per in rk.items()}
    base = _rows_for(normal_of(rk_main), cat_z, delta, arms_main, eval_fn)
    fwb_rows = _rows_for(normal_of(rk_ab), cat_z, delta, [fwb], eval_fn)
    key = lambda r: (r["slice"], r["budget"])
    fb = {key(r): r for r in fwb_rows}
    for r in base:
        if key(r) in fb:
            r["regret"].update({k: v for k, v in fb[key(r)]["regret"].items() if k.startswith(fwb)})
            r["metrics"].update({k: v for k, v in fb[key(r)]["metrics"].items() if k == fwb})
    abl = {}
    for mode in ["zero_all"] + [f"noise{ns}" for ns in NOISE_SEEDS]:
        rk_m = {t: ({b: {s: {fw: pa[fw]} for s, pa in pb.items()} for b, pb in per["ablate"].get(mode, {}).items()}, per.get("test")) for t, per in rk_ab.items()}
        abl[mode] = {key(r): r for r in _rows_for(rk_m, cat_z, delta, [fw], eval_fn)}
    pooled = {}
    for a in arms_main + [fwb]:
        if any(a + "3" in r["regret"] for r in base):
            pooled[a] = {"regret1": float(np.mean([r["regret"][a + "1"] for r in base if a + "1" in r["regret"]])),
                         "regret3": float(np.mean([r["regret"][a + "3"] for r in base if a + "3" in r["regret"]])),
                         "pair_acc": _g57._pooled(base, a, "pair_acc"), "acc_top10": _g57._pooled(base, a, "acc_top10")}
    for mode, m in abl.items():
        rs = list(m.values())
        pooled[f"{fw}@{mode}"] = {"regret1": float(np.mean([r["regret"][fw + "1"] for r in rs])), "regret3": float(np.mean([r["regret"][fw + "3"] for r in rs])),
                                  "pair_acc": _g57._pooled(rs, fw, "pair_acc"), "acc_top10": _g57._pooled(rs, fw, "acc_top10")}
    noise_mean = {k: np.mean([abl[f"noise{ns}"][k]["regret"][fw + "3"] for ns in NOISE_SEEDS]) for k in abl["noise0"]}
    noise_mean1 = {k: np.mean([abl[f"noise{ns}"][k]["regret"][fw + "1"] for ns in NOISE_SEEDS]) for k in abl["noise0"]}
    pooled[f"{fw}@noise"] = {"regret3": float(np.mean(list(noise_mean.values()))), "regret1": float(np.mean(list(noise_mean1.values())))}
    tests = {}
    blind = arms_main[0]
    tests["structure: blind→fwb"] = _g58.paired_regret(base, blind, fwb)
    tests["features: fwb→fw"] = _g58.paired_regret(base, fwb, fw)
    tests["blind→fw"] = _g58.paired_regret(base, blind, fw)
    for k in ("3", "1"):
        x = np.array([r["regret"][fw + k] for r in base]); z = np.array([abl["zero_all"][key(r)]["regret"][fw + k] for r in base])
        n = np.array([np.mean([abl[f"noise{ns}"][key(r)]["regret"][fw + k] for ns in NOISE_SEEDS]) for r in base])
        tests[f"fw vs zero_all regret@{k}"] = {"normal": float(x.mean()), "ablated": float(z.mean()), "p_worse": float(_g56.paired_wilcoxon_less(x, z)),
                                              "worse": int(np.sum(z > x)), "better": int(np.sum(z < x)), "ties": int(np.sum(z == x))}
        tests[f"fw vs noise regret@{k}"] = {"normal": float(x.mean()), "ablated": float(n.mean()), "p_worse": float(_g56.paired_wilcoxon_less(x, n)),
                                           "worse": int(np.sum(n > x)), "better": int(np.sum(n < x)), "ties": int(np.sum(n == x))}
    tests["features: fwb→fw regret@1"] = _g58.paired_regret(base, fwb, fw, key="1")
    tests["structure: blind→fwb regret@1"] = _g58.paired_regret(base, blind, fwb, key="1")
    return {"n_instances": len(base), "pooled": pooled, "tests": tests}


def cmd_analyze() -> dict:
    rk_ab_u = json.loads((OUT / "rankings_uea.json").read_text(encoding="utf-8"))
    rk_u = _g64.merge_ref(json.loads((RANK_OUT / "uea" / "rankings_uea.json").read_text(encoding="utf-8")),
                          json.loads((_g66.S1PHI / "binned" / "rankings_s1phi.json").read_text(encoding="utf-8")), arms=("blind_diff_hard", "phi_diff_hard"))
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    uea = _site(rk_ab_u, rk_u, cat_z, delta, arms_main=["blind_diff_hard", "phi_diff_hard", "fw_eaphi_hard"], fw="fw_eaphi_hard", fwb="fwb_hard", eval_fn=_g58._eval_all)
    rk_ab_c = json.loads((OUT / "rankings_ucr.json").read_text(encoding="utf-8"))
    rk_c = _g63.merge_ref_rankings(json.loads((RANK_OUT / "ucr" / "rankings_ea.json").read_text(encoding="utf-8")),
                                   json.loads((_g66.S1UCR / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_blind_diff_hard", "u_phi_diff_hard"))
    rows_u, _ea, _phi = _g63._load_all()
    cat_c = _g56.zscore_within(rows_u)
    delta_c = {(d, int(b)): _g60.DELTA_UCR for d in {r["dataset"] for r in rows_u} for b in BUDGETS}
    ucr = _site(rk_ab_c, rk_c, cat_c, delta_c, arms_main=["u_blind_diff_hard", "u_phi_diff_hard", "u_fw_eaphi_hard"], fw="u_fw_eaphi_hard", fwb="u_fwb_hard", eval_fn=_g64._eval_ucr)
    a = {"protocol": "results/tclsc_s1ea_rank/PROTOCOL.md §9", "uea": uea, "ucr": ucr}
    (OUT / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print("done", flush=True)
    return a


def markdown_tables(a: dict) -> str:
    L = ["# R 检验 §9 消融：数据集特征对 field 头的贡献", ""]
    for title, blk in (("UEA n=7（126 实例）", a["uea"]), ("UCR n=72（252 实例）", a["ucr"])):
        L += [f"## {title}", "", "| 臂 / 消融 | M1 | M0 | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|"]
        for arm, b in blk["pooled"].items():
            m1 = "—" if b.get("acc_top10") is None else f"{b['acc_top10']:.3f}"
            m0 = "—" if b.get("pair_acc") is None else f"{b['pair_acc']:.3f}"
            L.append(f"| {arm} | {m1} | {m0} | {b['regret1']:.4f} | {b['regret3']:.4f} |")
        L += ["", "### 配对检验", ""]
        for k, t in blk["tests"].items():
            if "mean_a" in t:
                L.append(f"- {k}: n={t['n']} {t['mean_a']:.4f} → {t['mean_b']:.4f} p={t['p']:.3g}（H1 更低）胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
            else:
                L.append(f"- {k}: {t['normal']:.4f} → {t['ablated']:.4f} p_worse={t['p_worse']:.3g} 更差/更好/平 {t['worse']}/{t['better']}/{t['ties']}")
        L.append("")
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
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --train / --rank / --analyze 之一")
    if args.train:
        cmd_train(seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(seeds=args.seeds)
    if args.analyze:
        cmd_analyze()


if __name__ == "__main__":
    main()
