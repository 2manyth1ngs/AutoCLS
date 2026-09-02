"""实验 1：E 级退化诊断 + `set_fc` 常数性 + 绕过 SetTF 的 C2 复算
（results/e_degradation_diag/PROTOCOL.md；DESIGN_NOTES §3.3 / §3.5-1 / §6-1）。

三段（全部零新增 R'）：
  --train-r1        重训 C2 的 R1（SetTF，mixed，seed 0）并落盘权重 tclsc_R1_s0.pt +
                    同机消融读数 runs/settf_R1_mixed_s0.json（C2 未保存权重）
  --diag            (a) L0 TS2Vec 级 / L1 E@init / L2 E@trained 三层区分力
                    (b) set_fc(E) 常数性 ρ_c、支路相对幅度、梯度灵敏度 → diag.json
  --bypass --resume (c) set_encoder 换 MeanPoolTaskEncoder（raw / z 两臂）× R1–R5，
                    C2 同口径两档 shuffle + zero → runs/bypass_*.json
  --verdict         协议 §4 判读表 → verdict.json

用法：python scripts/30_e_degradation_diag.py --train-r1 --diag --bypass --resume --verdict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.comparator.e_diagnostics import constancy_ratio, layer_report  # noqa: E402
from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder  # noqa: E402
from autocls.task_types import is_better  # noqa: E402
from scripts.e_ablation_common import (  # noqa: E402
    WorkpointData, ablation_block, build_runs, ds_of, load_set_encoder_init, tier_of,
)

OUT = ROOT / "results" / "e_degradation_diag"
R1_WEIGHTS = OUT / "tclsc_R1_s0.pt"
BYPASS_RUN_IDS = ("R1_mixed_s0", "R2_mixed_s1", "R3_mixed_s2", "R4_hw_only", "R5_um_only")


def train_hard(data: WorkpointData, set_enc, train_tids: set[str], seed: int, device: str):
    """C2 run_one 的训练段（硬 δ 动态配对 + curriculum），set_encoder 由调用方给。"""
    torch.manual_seed(seed)
    model = TCLSC(cls_dim=data.cls_dim, set_encoder=set_enc,
                  embedding_dim=data.cfg["embedding_dim"], fz=data.cfg["fz"])
    engine = TCLSCEngine(model, data.task_windows, device=device, lr=data.cfg["lr"],
                         batch_size=data.cfg["batch_size"], seed=seed)
    t0 = time.time()
    hist = engine.train(data.records_in(train_tids), epochs=data.cfg["epochs"],
                        curriculum_epochs=data.cfg["curriculum_epochs"],
                        better_fn=lambda a, b: is_better(a, b, data.task_type),
                        delta=data.delta_by_task(train_tids))
    return engine, hist, round(time.time() - t0, 1)


def run_and_ablate(data, set_enc, spec, device, tag) -> tuple[dict, TCLSCEngine]:
    train_tids, test_tids = spec["train"], spec["test"]
    assert not (train_tids & test_tids)
    engine, hist, secs = train_hard(data, set_enc, train_tids, spec["seed"], device)
    directed, n_fwd = data.hard_test_pairs(test_tids)
    block = ablation_block(engine, directed, n_fwd, test_tids, data.all_tids)
    res = {"run_id": spec["run_id"], "arm": tag, "seed": spec["seed"],
           "train_tids": sorted(train_tids), "test_tids": sorted(test_tids),
           "n_train_records": len(data.records_in(train_tids)), "train_seconds": secs,
           "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"], "device": device, **block}
    return res, engine


def fmt(res: dict) -> str:
    w, c = res["within"], res["cross"]
    return (f"{res['train_seconds']}s | normal {res['normal']['acc']:.4f} | "
            f"within Δ̄={w['mean_delta']:+.4f} sig {w['n_sig']}/5 {'PASS' if w['pass'] else '—'} | "
            f"cross Δ̄={c['mean_delta']:+.4f} sig {c['n_sig']}/5 {'PASS' if c['pass'] else '—'} | "
            f"zero Δ={res['zero']['delta_acc']:+.4f}")


# ---------------------------------------------------------------- 段 1：R1 重训


def stage_train_r1(data: WorkpointData, device: str, resume: bool) -> None:
    out_json = OUT / "runs" / "settf_R1_mixed_s0.json"
    if resume and out_json.exists() and R1_WEIGHTS.exists():
        print("[train-r1] 已有权重与结果，跳过", flush=True)
        return
    spec = [s for s in build_runs(data.all_tids) if s["run_id"] == "R1_mixed_s0"][0]
    print(f"[train-r1] SetTF R1 mixed seed0 train={len(spec['train'])} test={len(spec['test'])} ...", flush=True)
    res, engine = run_and_ablate(data, load_set_encoder_init(data), spec, device, "settf")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(engine.model.state_dict(), R1_WEIGHTS)
    print(f"[train-r1] {fmt(res)}", flush=True)


# ---------------------------------------------------------------- 段 2：诊断 (a)(b)


def stage_diag(data: WorkpointData, device: str) -> dict:
    assert R1_WEIGHTS.exists(), "先 --train-r1 产出 tclsc_R1_s0.pt"
    tids = data.all_tids
    groups = [ds_of(t) for t in tids]
    # L0：TS2Vec 级均值池化
    L0 = np.stack([data.task_windows[t].mean(axis=(0, 1)) for t in tids])
    # L1：E@init（npz 内 E 字段 + 复算校验）
    enc_init = load_set_encoder_init(data).eval()
    with torch.no_grad():
        L1_re = enc_init.set_encode([torch.as_tensor(data.task_windows[t]) for t in tids]).numpy()
    L1_npz = np.stack([data.task_E_init[t] for t in tids])
    l1_check = float(np.max(np.abs(L1_re - L1_npz)))
    # L2：E@trained（R1 联合训练后的 SetTF）
    model = TCLSC(cls_dim=data.cls_dim, set_encoder=load_set_encoder_init(data),
                  embedding_dim=data.cfg["embedding_dim"], fz=data.cfg["fz"])
    model.load_state_dict(torch.load(R1_WEIGHTS, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    with torch.no_grad():
        L2 = model.encode_tasks([torch.as_tensor(data.task_windows[t], device=device)
                                 for t in tids]).cpu().numpy()
        # (b) set_fc 常数性
        C = model.set_fc(torch.as_tensor(L2, device=device)).cpu().numpy()          # (28,128)
        c0 = model.set_fc(torch.zeros(1, L2.shape[1], device=device)).cpu().numpy()[0]
        rho_c = constancy_ratio(C, c0)
        # 支路相对幅度：E 支路跨任务 std 范数 vs CLS 支路跨配对 std 范数（R1 测试对）
        spec = [s for s in build_runs(tids) if s["run_id"] == "R1_mixed_s0"][0]
        directed, _ = data.hard_test_pairs(spec["test"])
        a = torch.as_tensor(np.stack([p.vec_a for p in directed]), dtype=torch.float32, device=device)
        b = torch.as_tensor(np.stack([p.vec_b for p in directed]), dtype=torch.float32, device=device)
        L = torch.cat([model.cls_mlp(a), model.cls_mlp(b)], dim=1)
        cls_branch = model.cls_fc(L).cpu().numpy()                                   # (n_pairs,128)
        branch = {
            "set_fc_std_norm_across_tasks": float(np.linalg.norm(C.std(axis=0))),
            "set_fc_mean_norm": float(np.linalg.norm(C.mean(axis=0))),
            "set_fc_zero_norm": float(np.linalg.norm(c0)),
            "cls_fc_std_norm_across_pairs": float(np.linalg.norm(cls_branch.std(axis=0))),
            "cls_fc_mean_norm": float(np.linalg.norm(cls_branch.mean(axis=0))),
            "set_fc_active_dims": int((C.max(axis=0) > 0).sum()),
            "set_fc_varying_dims": int((C.std(axis=0) > 1e-6).sum()),
        }
        branch["std_ratio_set_over_cls"] = (branch["set_fc_std_norm_across_tasks"]
                                            / branch["cls_fc_std_norm_across_pairs"])
    # 梯度灵敏度（logit 对 E / cls_a 的梯度 × 输入范数）
    idx = {t: i for i, t in enumerate(tids)}
    sub = directed[: min(512, len(directed))]
    E_in = torch.tensor(np.stack([L2[idx[p.task_id]] for p in sub]), dtype=torch.float32,
                        device=device, requires_grad=True)
    a_in = a[: len(sub)].clone().requires_grad_(True)
    b_in = b[: len(sub)]
    out = model(a_in, b_in, E_in)
    logit = torch.logit(out.clamp(1e-6, 1 - 1e-6))
    gE, ga = torch.autograd.grad(logit.sum(), [E_in, a_in])
    sens = {
        "E": float((gE.norm(dim=1) * E_in.norm(dim=1)).mean()),
        "cls_a": float((ga.norm(dim=1) * a_in.norm(dim=1)).mean()),
        "E_grad_norm": float(gE.norm(dim=1).mean()), "cls_grad_norm": float(ga.norm(dim=1).mean()),
        "E_norm": float(E_in.norm(dim=1).mean()), "cls_norm": float(a_in.norm(dim=1).mean()),
    }
    sens["ratio_E_over_cls"] = sens["E"] / sens["cls_a"] if sens["cls_a"] else float("nan")
    # 前提检查（协议 §3）：L0 非退化
    layers = {"L0_ts2vec_meanpool": layer_report(L0, groups),
              "L1_E_init": layer_report(L1_re, groups),
              "L2_E_trained": layer_report(L2, groups)}
    l0 = layers["L0_ts2vec_meanpool"]
    diag = {
        "n_tasks": len(tids), "groups": {d: groups.count(d) for d in set(groups)},
        "L1_recompute_max_abs_diff_vs_npz": l1_check,
        "layers": layers,
        "set_fc": {"rho_c": rho_c, **branch},
        "sensitivity": sens,
        "L0_precheck": {"cross_cos": l0["cross_cos"], "effective_rank": l0["effective_rank"],
                        "pass": bool(l0["cross_cos"] < 0.95 and l0["effective_rank"] >= 3)},
    }
    (OUT / "diag.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in layers.items():
        print(f"[diag] {k}: within_cos {v['within_cos']:.4f} cross_cos {v['cross_cos']:.4f} "
              f"min_cos {v['min_cos']:.4f} sep {v['separation_ratio']:.2f} "
              f"eff_rank {v['effective_rank']:.2f} rel_disp {v['relative_dispersion']:.4f} "
              f"| z: sep {v['zscored']['separation_ratio']:.2f} eff_rank {v['zscored']['effective_rank']:.2f}",
              flush=True)
    print(f"[diag] L1 npz 复算 max|Δ|={l1_check:.2e} | ρ_c={rho_c:.4f} | "
          f"set_fc std/cls_fc std={branch['std_ratio_set_over_cls']:.4f} | "
          f"sens E/cls={sens['ratio_E_over_cls']:.4f} | L0 precheck {diag['L0_precheck']['pass']}", flush=True)
    return diag


# ---------------------------------------------------------------- 段 3：绕过 SetTF


def stage_bypass(data: WorkpointData, device: str, resume: bool, only=None) -> None:
    runs_dir = OUT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in build_runs(data.all_tids) if s["run_id"] in BYPASS_RUN_IDS]
    for arm in ("bypass_raw", "bypass_z"):
        for spec in specs:
            rid = f"{arm}_{spec['run_id']}"
            if only and rid not in only:
                continue
            out_path = runs_dir / f"{rid}.json"
            if resume and out_path.exists():
                print(f"[{rid}] 已有结果，跳过", flush=True)
                continue
            if arm == "bypass_raw":
                enc = MeanPoolTaskEncoder(dim_input=data.dim_input)
            else:
                mu, sd = MeanPoolTaskEncoder.fit_standardizer(data.task_windows, sorted(spec["train"]))
                enc = MeanPoolTaskEncoder(dim_input=data.dim_input, mean=mu, std=sd)
            print(f"[{rid}] train={len(spec['train'])} test={len(spec['test'])} ...", flush=True)
            res, _ = run_and_ablate(data, enc, spec, device, arm)
            out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{rid}] {fmt(res)}", flush=True)


# ---------------------------------------------------------------- 段 4：判读


def stage_verdict() -> dict:
    diag = json.loads((OUT / "diag.json").read_text(encoding="utf-8"))
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "runs").glob("*.json"))}
    L1, L2 = diag["layers"]["L1_E_init"], diag["layers"]["L2_E_trained"]
    f1_init = bool(L1["min_cos"] >= 0.98 and L1["relative_dispersion"] < 0.10)
    f1_trained = bool(L2["min_cos"] >= 0.98 and L2["relative_dispersion"] < 0.10)
    rho = diag["set_fc"]["rho_c"]
    f2 = "confirmed" if rho < 0.10 else ("grey" if rho < 0.30 else "rejected")

    def median_mixed(arm: str):
        mixed = [r for k, r in runs.items() if k.startswith(f"{arm}_R") and "mixed" in k]
        if len(mixed) < 3:
            return None
        return sorted(mixed, key=lambda r: r["within"]["mean_delta"])[1]

    f3 = {}
    for arm in ("bypass_raw", "bypass_z"):
        m = median_mixed(arm)
        if m is None:
            f3[arm] = {"status": "incomplete"}
            continue
        f3[arm] = {"median_run": m["run_id"], "normal_acc": m["normal"]["acc"],
                   "within": {k: m["within"][k] for k in ("mean_delta", "n_sig", "pass")},
                   "cross": {k: m["cross"][k] for k in ("mean_delta", "n_sig", "pass")},
                   "within_tier": tier_of(m["within"]), "cross_tier": tier_of(m["cross"]),
                   "zero_delta": m["zero"]["delta_acc"]}
        r4 = runs.get(f"{arm}_R4_hw_only")
        if r4:
            f3[arm]["r4_within_tier"] = tier_of(r4["within"])
    tiers_w = [v.get("within_tier") for v in f3.values() if "within_tier" in v]
    tiers_c = [v.get("cross_tier") for v in f3.values() if "cross_tier" in v]
    if any(v.get("within", {}).get("pass") for v in f3.values()):
        f3_code, f3_label = "feature_side_half", "任一 bypass 臂 within 达标：特征侧是半个锅"
    elif any(v.get("cross", {}).get("pass") for v in f3.values()):
        f3_code, f3_label = "domain_buried", "仅 cross 达标：域级信号原本被 SetTF 埋没"
    elif tiers_w and all(t == "zero" for t in tiers_w) and all(t == "zero" for t in tiers_c):
        f3_code, f3_label = "label_side_single_cause_upgraded", "两臂双零：label 侧单因证据升级"
    else:
        f3_code, f3_label = "ambiguous", f"档位混杂 within={tiers_w} cross={tiers_c}"
    settf = runs.get("settf_R1_mixed_s0")
    verdict = {
        "F1": {"init_degenerate": f1_init, "trained_degenerate": f1_trained,
               "L1_min_cos": L1["min_cos"], "L1_rel_disp": L1["relative_dispersion"],
               "L2_min_cos": L2["min_cos"], "L2_rel_disp": L2["relative_dispersion"],
               "L0_cross_cos": diag["layers"]["L0_ts2vec_meanpool"]["cross_cos"],
               "label": ("退化确认" if f1_init else "未退化") + ("；联合训练修不好" if f1_trained else "；训练后脱离退化带")},
        "F2": {"rho_c": rho, "status": f2},
        "F3": {"code": f3_code, "label": f3_label, "arms": f3,
               "L0_precheck": diag["L0_precheck"]},
        "settf_same_machine": (None if settf is None else
                               {"normal_acc": settf["normal"]["acc"],
                                "within_tier": tier_of(settf["within"]), "cross_tier": tier_of(settf["cross"]),
                                "zero_delta": settf["zero"]["delta_acc"]}),
    }
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-r1", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--bypass", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="仅冒烟：覆盖训练 epoch 数")
    ap.add_argument("--outdir", type=Path, default=None, help="仅冒烟：改写输出目录")
    args = ap.parse_args()
    global OUT, R1_WEIGHTS
    if args.outdir is not None:
        OUT = args.outdir
        R1_WEIGHTS = OUT / "tclsc_R1_s0.pt"
    if not any((args.train_r1, args.diag, args.bypass, args.verdict)):
        ap.error("至少指定一个阶段")
    OUT.mkdir(parents=True, exist_ok=True)
    data = None
    if args.train_r1 or args.diag or args.bypass:
        data = WorkpointData()
        if args.epochs is not None:
            data.cfg["epochs"] = args.epochs
            data.cfg["curriculum_epochs"] = min(data.cfg["curriculum_epochs"], args.epochs // 2)
        print(f"记录 {len(data.records)} / 任务 {len(data.all_tids)} | 单向硬评估对全池 {len(data.fwd_pairs)} | "
              f"δ={data.delta_map} | device={args.device}", flush=True)
    if args.train_r1:
        stage_train_r1(data, args.device, args.resume)
    if args.diag:
        stage_diag(data, args.device)
    if args.bypass:
        stage_bypass(data, args.device, args.resume, args.only)
    if args.verdict:
        stage_verdict()


if __name__ == "__main__":
    main()
