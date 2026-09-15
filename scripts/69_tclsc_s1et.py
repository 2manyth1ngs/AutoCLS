"""WP16-TCLSC 阶段 1 T 检验：转导型响应画像 E_T（目标训练划分上 GGS 探针短训的表示空间）vs 原始空间 E_A，field 头。协议 results/tclsc_s1et/PROTOCOL.md。

子命令：
  --features   UEA 45 切片 + UCR 84 集：GGS 探针 pretrain_encoder(max_iters=290, checkpoints=(60, 290))，每截点 rep_profile → et_uea.jsonl / et_ucr.jsonl
  --train      UEA：变体目录 uea/（+ E_A ⊕ φ ⊕ E_T 合并表）→ 33 号 ft60 / ft290 / ftea；UCR：63 号 u_ft60 / u_ft290 / u_ftea
  --rank       UEA：64 号 rank_uea(arms=)（表 = 合并表）；UCR：63 号 cmd_rank
  --analyze    与参照臂（UEA：s1phi blind、R 检验 fw、§9 fwb；UCR：u_blind / u_fw / u_fwb）合并配对；门 C0 / G-T1 / G-T2 / G-T3；读数码
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.mech_features import REP_PROFILE_COLS, RESPONSE_SEEDS, rep_cols_at, rep_profile, torch_encode_fn  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g66 = importlib.import_module("scripts.66_tclsc_s1ea_rank")
BUDGETS = pq.BUDGETS
OUT = ROOT / "results" / "tclsc_s1et"
SOURCE = pq.OUT
CONF = ROOT / "results" / "tclsc_s1ea_confirm"
RANK = ROOT / "results" / "tclsc_s1ea_rank"
PROBE_BUDGETS = (60, 290)
PROBE_ITERS = 290
PROBE_SEED = 0
UEA_ARMS = ("ft60_hard", "ft290_hard", "ftea_hard")
UEA_REFS = ("blind_diff_hard", "fw_eaphi_hard", "fwb_hard")
UCR_ARMS = ("u_ft60_hard", "u_ft290_hard", "u_ftea_hard")
UCR_REFS = ("u_blind_diff_hard", "u_fw_eaphi_hard", "u_fwb_hard")
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
ALPHA = 0.05
C0_TOL = 0.0002


# ---------------------------------------------------------------- 特征
def feature_row(task_id: str, dataset: str, per_budget: Mapping[int, Mapping[str, float]], *, seconds: float, acc290: Optional[float]) -> dict:
    row = {"task_id": task_id, "dataset": dataset, "probe": "ggs", "probe_seed": PROBE_SEED, "seconds": round(float(seconds), 1), "acc290_probe": acc290}
    for b, prof in per_budget.items():
        for c in REP_PROFILE_COLS:
            row[f"{c}@{int(b)}"] = float(prof[c])
        row[f"rep_dim@{int(b)}"] = int(prof.get("rep_dim", 0))
    return row


def probe_profiles(Xtr: np.ndarray, ytr: np.ndarray, task_type: str, *, device: str, eval_fn=None) -> tuple[dict, Optional[float]]:
    """GGS 探针在训练划分上训到 290 iters，@60 / @290 各取一次 rep_profile。eval_fn(enc) → acc290（可选，对账用）。"""
    import copy
    from autocls import labeling
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.encoders.registry import build_encoder
    cls = pq.ggs_cls()
    torch.manual_seed(PROBE_SEED)
    enc = build_encoder(input_dims=Xtr.shape[-1], overrides=labeling.extension_overrides(cls) or None)
    out: dict[int, dict] = {}
    acc = {"v": None}

    def _cb(step: int, e) -> None:
        snap = copy.deepcopy(e).to(device).eval()
        out[int(step)] = rep_profile(torch_encode_fn(snap, device=device), Xtr, ytr, task_type=task_type, seeds=RESPONSE_SEEDS)
        if int(step) == PROBE_ITERS and eval_fn is not None:
            acc["v"] = float(eval_fn(snap))

    pretrain_encoder(enc, cls, Xtr, max_iters=PROBE_ITERS, batch_size=128, lr=1e-3, device=device, seed=PROBE_SEED,
                     checkpoints=PROBE_BUDGETS, on_checkpoint=_cb)
    return out, acc["v"]


def cmd_features(*, device: str, resume: bool, which: Sequence[str]) -> None:
    from autocls.data.task_slicer import materialize_task
    from autocls.labeling import run_seed_for, split_task_train_eval
    OUT.mkdir(parents=True, exist_ok=True)
    if "uea" in which:
        path = OUT / "et_uea.jsonl"
        done = {r["task_id"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
        ds = pq.load_pq_datasets(pq.PQ_DATASETS)
        with open(path, "a", encoding="utf-8") as fh:
            for name, d in ds.items():
                for t in d.slices:
                    if t.task_id in done:
                        continue
                    Xt, yt = materialize_task(t, d.X, d.y)
                    tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(t.task_id, "train_eval_split"))
                    t0 = time.time()
                    prof, _acc = probe_profiles(Xt[tr], yt[tr], t.task_type, device=device)
                    row = feature_row(t.task_id, name, prof, seconds=time.time() - t0, acc290=None)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                    print(f"[et uea] {t.task_id}: {row['seconds']}s rep_highpass05@60 {row['rep_highpass05@60']:+.2f} rinfo@60 {row['rinfo@60']:+.2f} @290 {row['rinfo@290']:+.2f}", flush=True)
    if "ucr" in which:
        path = OUT / "et_ucr.jsonl"
        done = {r["task_id"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
        tasks = json.loads((ROOT / "results" / "ucr_rprime" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        _g41 = importlib.import_module("scripts.41_ucr_rprime")
        reg = _g41.registry()
        with open(path, "a", encoding="utf-8") as fh:
            for tid, spec in tasks.items():
                if tid in done:
                    continue
                X, y = _g41.load_dataset(spec["dataset"], reg)
                t = _g41.task_from_json(spec)
                Xt, yt = materialize_task(t, X, y)
                tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(tid, "train_eval_split"))
                t0 = time.time()
                prof, _acc = probe_profiles(Xt[tr], yt[tr], t.task_type, device=device)
                row = feature_row(tid, spec["dataset"], prof, seconds=time.time() - t0, acc290=None)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
                print(f"[et ucr] {spec['dataset']}: {row['seconds']}s rep_highpass05@60 {row['rep_highpass05@60']:+.2f} rinfo@60 {row['rinfo@60']:+.2f} @290 {row['rinfo@290']:+.2f}", flush=True)


# ---------------------------------------------------------------- 训练 / 排名
def _merged_uea_table_file(d: Path) -> Path:
    """E_A ⊕ φ（A′ 的 ea_uea.jsonl）⊕ E_T（et_uea.jsonl）合并成 33 号的 --ctx-file。"""
    ea = {r["task_id"]: r for r in read_jsonl(CONF / "uea" / "ea_uea.jsonl")}
    et = {r["task_id"]: r for r in read_jsonl(OUT / "et_uea.jsonl")}
    missing = sorted(set(ea) - set(et))
    if missing:
        raise RuntimeError(f"et_uea.jsonl 缺切片 {missing[:5]}…（先跑 --features）")
    p = d / "ctx_uea.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for s, r in ea.items():
            fh.write(json.dumps({**r, **{k: v for k, v in et[s].items() if k not in ("task_id", "dataset")}}, ensure_ascii=False) + "\n")
    return p


def cmd_train(*, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, which: Sequence[str]) -> None:
    if "uea" in which:
        d = OUT / "uea"
        _g56.prepare_variant_dir(SOURCE, d)
        ctx = _merged_uea_table_file(d)
        _m33.cmd_train(d, arms=list(UEA_ARMS), folds=None, seeds=list(seeds), resume=resume, device=device, epochs=epochs,
                       settf_windows=None, encoder_mode="binned", ctx_file=ctx)
    if "ucr" in which:
        (OUT / "ucr").mkdir(parents=True, exist_ok=True)
        _g63.cmd_train(OUT / "ucr", arms=list(UCR_ARMS), seeds=list(seeds), folds_sel=None, device=device, epochs=epochs, resume=resume)


def cmd_rank(*, seeds: Sequence[int]) -> None:
    d = OUT / "uea"
    if not (d / "ea_uea.jsonl").exists():
        shutil.copy(d / "ctx_uea.jsonl", d / "ea_uea.jsonl")          # 64 号 rank_uea 读 uea/ea_uea.jsonl 作 ctx 表
    rk = _g64.rank_uea(OUT, seeds=seeds, device="cpu", arms=UEA_ARMS)
    (d / "rankings_uea.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    if (OUT / "ucr" / "folds.json").exists():
        _g63.cmd_rank(OUT / "ucr", seeds=seeds)


# ---------------------------------------------------------------- 门 / 读数码
def _lower(v: Mapping) -> bool:
    return bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])


def gates_et(v_t1: Mapping, v_t2: Mapping, v_t3: Mapping) -> dict:
    """v_t1 = fw → ft60；v_t2 = blind → ft60（H1 b<a）；v_t3 = ft60 normal → cross（H1 cross 更差）。"""
    worse = bool(v_t3.get("p") is not None and v_t3["p"] < ALPHA and v_t3["mean_b"] > v_t3["mean_a"])
    return {"GT1": {"metric": "regret3", "fw": v_t1.get("mean_a"), "ft60": v_t1.get("mean_b"), "p": v_t1.get("p"), "pass": _lower(v_t1)},
            "GT2": {"metric": "regret3", "blind": v_t2.get("mean_a"), "ft60": v_t2.get("mean_b"), "p": v_t2.get("p"), "pass": _lower(v_t2)},
            "GT3": {"metric": "regret3", "normal": v_t3.get("mean_a"), "cross": v_t3.get("mean_b"), "p": v_t3.get("p"), "pass": worse},
            "thresholds": {"alpha": ALPHA}}


def verdict_et(g: Mapping) -> str:
    t1, t2, t3 = g["GT1"]["pass"], g["GT2"]["pass"], g["GT3"]["pass"]
    code = "ET-DEAD" if not t2 else ("ET-ALIVE" if t1 else "ET-EQUIV")
    return code + ("+CONTENT" if t3 else "-CONTENT")


# ---------------------------------------------------------------- 分析
def _refs_uea():
    rk = json.loads((RANK / "uea" / "rankings_uea.json").read_text(encoding="utf-8"))
    rk = _g64.merge_ref(rk, json.loads((_g66.S1PHI / "binned" / "rankings_s1phi.json").read_text(encoding="utf-8")), arms=("blind_diff_hard",))
    ab = json.loads((RANK / "ablate" / "rankings_uea.json").read_text(encoding="utf-8"))
    return _g64.merge_ref(rk, ab, arms=("fwb_hard",))


def cmd_analyze() -> dict:
    find = lambda tests, a, b: next((t for t in tests if t["a"] == a and t["b"] == b), {"p": None, "mean_a": None, "mean_b": None})
    rk_u = _g64.merge_ref(json.loads((OUT / "uea" / "rankings_uea.json").read_text(encoding="utf-8")), _refs_uea(), arms=UEA_REFS)
    rows, delta = _g57.load_source(SOURCE)
    cat_z = _g56.zscore_within(rows)
    uea = _g64._block(rk_u, cat_z, delta, list(UEA_REFS) + list(UEA_ARMS), list(UEA_ARMS), eval_fn=_g58._eval_all)
    nu = uea["per_instance_normal"]
    pairs_u = [(r, a) for a in UEA_ARMS for r in UEA_REFS] + [("ft60_hard", "ft290_hard"), ("ft60_hard", "ftea_hard")]
    uea["value_tests"] = {k: [_g58.paired_regret(nu, a, b, key=k) for a, b in pairs_u if a in uea["arms"] and b in uea["arms"]] for k in ("1", "3")}
    ucr = None
    p_ucr = OUT / "ucr" / "rankings_ea.json"
    if p_ucr.exists():
        rk_c = _g63.merge_ref_rankings(json.loads(p_ucr.read_text(encoding="utf-8")), json.loads((_g66.S1UCR / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_blind_diff_hard",))
        rk_c = _g63.merge_ref_rankings(rk_c, json.loads((RANK / "ucr" / "rankings_ea.json").read_text(encoding="utf-8")), arms=("u_fw_eaphi_hard",))
        rk_c = _g63.merge_ref_rankings(rk_c, json.loads((RANK / "ablate" / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_fwb_hard",))
        rows_u, _ea, _phi = _g63._load_all()
        cat_c = _g56.zscore_within(rows_u)
        delta_c = {(d, int(b)): _g60.DELTA_UCR for d in {r["dataset"] for r in rows_u} for b in BUDGETS}
        ucr = _g64._block(rk_c, cat_c, delta_c, list(UCR_REFS) + list(UCR_ARMS), list(UCR_ARMS), eval_fn=_g64._eval_ucr)
        nc = ucr["per_instance_normal"]
        pairs_c = [(r, a) for a in UCR_ARMS for r in UCR_REFS] + [("u_ft60_hard", "u_ft290_hard"), ("u_ft60_hard", "u_ftea_hard")]
        ucr["value_tests"] = {k: [_g58.paired_regret(nc, a, b, key=k) for a, b in pairs_c if a in ucr["arms"] and b in ucr["arms"]] for k in ("1", "3")}
    v1 = find(uea["value_tests"]["3"], "fw_eaphi_hard", "ft60_hard")
    v2 = find(uea["value_tests"]["3"], "blind_diff_hard", "ft60_hard")
    v3 = uea["regret_ablation"].get("ft60_hard", {}).get("3", {}).get("cross", {"p": None, "mean_a": None, "mean_b": None})
    g = gates_et(v1, v2, v3)
    refs = {"blind_diff_hard": 0.0294, "fw_eaphi_hard": 0.0269}
    c0 = {a: {"ref": r, "got": uea["pooled"].get(a, {}).get("regret3"), "pass": None if a not in uea["pooled"] else bool(abs(uea["pooled"][a]["regret3"] - r) <= 0.0003)} for a, r in refs.items()}
    # 探针特征诊断：@60 vs @290 列相关、与 E_A 同名列相关
    et = {r["task_id"]: r for r in read_jsonl(OUT / "et_uea.jsonl")}
    ea = {r["task_id"]: r for r in read_jsonl(CONF / "uea" / "ea_uea.jsonl")}
    diag = {}
    from scipy.stats import spearmanr
    for c in REP_PROFILE_COLS:
        x60 = [et[s][f"{c}@60"] for s in et]; x290 = [et[s][f"{c}@290"] for s in et]
        diag[c] = {"rho_60_290": float(spearmanr(x60, x290).correlation)}
        raw = c.replace("rep_", "ret_").replace("rcos_", "sim_")
        if raw in ea[next(iter(ea))]:
            diag[c]["rho_60_raw"] = float(spearmanr(x60, [ea[s][raw] for s in et]).correlation)
    a = {"protocol": "results/tclsc_s1et/PROTOCOL.md", "uea": uea, "ucr": ucr, "c0": c0, "gates": g, "verdict": verdict_et(g), "feature_diag": diag}
    a["predictions"] = predictions(a)
    (OUT / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"{a['verdict']} | UEA " + " | ".join(f"{arm} r3 {uea['pooled'][arm]['regret3']:.4f}" for arm in uea["arms"])
          + (" | UCR " + " | ".join(f"{arm} r3 {ucr['pooled'][arm]['regret3']:.4f}" for arm in ucr["arms"]) if ucr else ""), flush=True)
    return a


def predictions(a: dict) -> dict:
    g = a["gates"]; pu = a["uea"]["pooled"]
    find = lambda tests, x, y: next((t for t in tests if t["a"] == x and t["b"] == y), {"p": None, "mean_a": None, "mean_b": None})
    r60, r290, rea, rfw = (pu.get(k, {}).get("regret3") for k in ("ft60_hard", "ft290_hard", "ftea_hard", "fw_eaphi_hard"))
    v7 = find((a["ucr"] or {"value_tests": {"3": []}})["value_tests"]["3"], "u_blind_diff_hard", "u_ft60_hard")
    return {"P1": {"desc": "C0", "hit": all(v["pass"] for v in a["c0"].values() if v["pass"] is not None)},
            "P2": {"desc": "G-T1 过", "hit": g["GT1"]["pass"]}, "P3": {"desc": "G-T2 过", "hit": g["GT2"]["pass"]}, "P4": {"desc": "G-T3 过", "hit": g["GT3"]["pass"]},
            "P5": {"desc": "UEA ft290 < ft60", "value": [r60, r290], "hit": None if r60 is None or r290 is None else bool(r290 < r60)},
            "P6": {"desc": "UEA ftea < min(ft60, fw)", "value": [rea, r60, rfw], "hit": None if None in (rea, r60, rfw) else bool(rea < min(r60, rfw))},
            "P7": {"desc": "UCR u_ft60 < u_blind（p<0.05）", "value": [v7.get("mean_a"), v7.get("mean_b"), v7.get("p")], "hit": _lower(v7)}}


def markdown_tables(a: dict) -> str:
    f = _g64._fmt
    L = ["# WP16 T 检验 tables（转导型 E_T，field 头）", "", f"读数码 **{a['verdict']}**", ""]
    for title, blk in (("UEA n=7（126 实例）", a["uea"]), ("UCR n=72（252 实例）", a.get("ucr"))):
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
        L += ["", "### 逐折 regret@3", ""] + [f"- {arm}: " + ", ".join(f"{fo} {v:.4f}" for fo, v in bf.items()) for arm, bf in blk["by_fold_regret3"].items()] + [""]
    L += ["## 特征诊断（UEA 45 切片）：@60 vs @290 的 Spearman、@60 vs 原始空间同名列", "", "| 列 | ρ(60,290) | ρ(60,raw) |", "|---|---:|---:|"]
    for c, d in a["feature_diag"].items():
        L.append(f"| {c} | {d['rho_60_290']:+.2f} | {('—' if 'rho_60_raw' not in d else '%+.2f' % d['rho_60_raw'])} |")
    L += ["", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k != "thresholds":
            L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    for k, v in a["c0"].items():
        L.append(f"| C0 {k} | ref {v['ref']} got {v['got']} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--which", nargs="+", default=["uea", "ucr"], choices=["uea", "ucr"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.features, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --features / --train / --rank / --analyze 之一")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.features:
        cmd_features(device=args.device, resume=args.resume, which=args.which)
    if args.train:
        cmd_train(seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, which=args.which)
    if args.rank:
        cmd_rank(seeds=args.seeds)
    if args.analyze:
        cmd_analyze()


if __name__ == "__main__":
    main()
