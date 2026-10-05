"""WP17-ACTS UT-ENS（results/acts/utens/PROTOCOL.md）：UT 的最后一步从“CV 选 1 条”换成“CV 前 3 条表示拼接集成”。

阶段：
  --plan        本地：UT 的搜索并集 + 3 折 CV → 每臂每靶前 3（第 1 名 = UT 选中者自检）+ 作业成本 → utens/plan.json
  --run         服务器 GPU：--worker i --workers n，剩余作业按成本 LPT 分配；作业 = （靶, 种子），三臂前 3 的并集各训一次
                （与 39.run_trajectory_cell 同一训练与截点编码），最后截点表示缓存 → 三臂各自逐块 L2 拼接 → 官方 SVM → utens/results/<靶>_s<种子>.json
  --analyze     本地读数
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTS = ROOT / "results" / "acts"
UT = ACTS / "ut"
OUT = ACTS / "utens"
K_ENS = 3
SEEDS = (0, 1, 2)
ARMS = ("ME", "UNI", "BLIND")
SINGLE_ME = 0.5826                                                  # UT 单选 ME


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def rank_cv(keys: Sequence[str], cv: Mapping[str, Sequence[float]], ens: Mapping[str, float]) -> list:
    """= 111.select_cv 的排序键（CV 均值 → 集成分数 → 键序），降序全排；无 CV 分者不进。"""
    cand = [k for k in keys if k in cv and len(cv[k])]
    return sorted(cand, key=lambda k: (round(float(np.mean(cv[k])), 12), float(ens.get(k, -np.inf)), -list(keys).index(k)), reverse=True)


def concat_l2(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """每个编码器的表示逐样本 L2 归一化后按块拼接。"""
    out = []
    for B in blocks:
        B = np.asarray(B, np.float64)
        out.append(B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12))
    return np.hstack(out)


def lpt_assign(costs: Mapping[str, float], n: int) -> list:
    """最长处理时间优先：作业按成本降序（并列按名字），逐个放到当前负载最小的进程。"""
    parts, load = [[] for _ in range(int(n))], [0.0] * int(n)
    for j in sorted(costs, key=lambda j: (-float(costs[j]), j)):
        i = int(np.argmin(load)); parts[i].append(j); load[i] += float(costs[j])
    return parts


def verdict_ens(me: float, uni: float, blind: float, single: float = SINGLE_ME, ts: Optional[float] = None, inf: Optional[float] = None) -> str:
    u = _g("118_acts_ut")
    ts = float(np.mean(list(u.TS2VEC_UT.values()))) if ts is None else ts
    inf = float(np.mean(list(u.INFOTS_S_UT.values()))) if inf is None else inf
    code = "ENS-BEATS-BOTH" if (me > inf and me > ts) else ("ENS-BEATS-TS2VEC" if me > ts else "ENS-BELOW")
    return code + ("+GAIN" if me > single else "") + ("+E" if me > uni else "") + ("+TA" if me > blind else "")


def job_list() -> list:
    return [(d, s) for d in _g("118_acts_ut").UT_TARGETS for s in SEEDS]


def _jid(d: str, s: int) -> str:
    return f"{d}_s{s}"


def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


# ================================================================ 计划（本地）
def cmd_plan(args) -> None:
    v, y, u = _g("108_acts_s3v"), _g("111_acts_s3y"), _g("118_acts_ut")
    md = Path(args.ut) / "md"; rows = v._rows(md); sel = _json(md / "select.json")
    raw = {}
    for l in (md / "runs" / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if not r.get("error") and r.get("checkpoint_results"):
            raw[(r["task_id"], r["cls_key"], int(r.get("seed", 0)))] = float(r.get("pretrain_seconds") or 0) + float(r.get("eval_seconds") or 0)
    plan, costs = {}, {}
    for d in u.UT_TARGETS:
        plan[d] = {}
        for a in ARMS:
            sr = _json(md / "search" / f"{a}_{d}.json"); keys = [e["cls_key"] for e in sr["union"]]; ens = {e["cls_key"]: e["ens"] for e in sr["union"]}
            cv = {k: [rows[(y.tcv_tid(d, i), k, 0)] for i in range(3) if (y.tcv_tid(d, i), k, 0) in rows] for k in keys}
            top = rank_cv(keys, cv, ens)[:K_ENS]
            assert top[0] == sel[a][d]["pick"], (a, d, top[0], sel[a][d]["pick"])
            byk = {e["cls_key"]: e["cls"] for e in sr["union"]}
            plan[d][a] = [{"cls_key": k, "cls": byk[k], "cv": float(np.mean(cv[k]))} for k in top]
        uniq = {e["cls_key"] for a in ARMS for e in plan[d][a]}
        secs = [raw[(f"{d}_whole_official", k, 0)] for k in uniq if (f"{d}_whole_official", k, 0) in raw]
        for s in SEEDS:
            costs[_jid(d, s)] = float(np.median(secs) if secs else 60.0) * len(uniq)
        print(f"[plan] {d}：三臂前 3 并集 {len(uniq)} 条；单格中位 {np.median(secs) if secs else float('nan'):.0f}s", flush=True)
    _write(Path(args.outdir) / "plan.json", {"k": K_ENS, "plan": plan, "costs": costs})
    print(f"[plan] 作业 {len(costs)}，估计总 {sum(costs.values()) / 3600:.1f} GPU-h", flush=True)


# ================================================================ 执行（服务器）
def capture(cell: dict, arrays: Mapping, device: str) -> tuple:
    """= 39.run_trajectory_cell 的训练与逐截点编码（同顺序，保证轨迹一致），只保留最后截点的表示；返回 (ftr, fte, 单模型结果)。"""
    g39 = _g("39_sh_deploy")
    import torch
    from autocls.space.cls_space import CLS
    g39.check_extension_consistency(cell)
    opts, seed, iters = cell["options"], int(cell["seed"]), int(cell["iters"])
    Xtr, ytr, yte = arrays["Xtr"], arrays["ytr"], arrays["yte"]
    Xtr_fit = arrays.get("Xtr_fit", Xtr); Xte = arrays["Xte_" + opts["scaler"]]; grid = opts["svm_grid"]
    box: dict = {}

    def on_ckpt(step: int, module) -> None:
        box["ftr"] = g39.encode_features(module, Xtr, device=device)
        box["fte"] = g39.encode_features(module, Xte, device=device)
        box["step"] = int(step)
        module.train()

    cls = CLS(**cell["cls"]); torch.manual_seed(seed); ov = {}
    if opts.get("input_mask", "none") != "none":
        ov["input_mask"] = opts["input_mask"]
    if float(opts.get("repr_dropout", 0.0)) > 0:
        ov["repr_dropout"] = float(opts["repr_dropout"])
    enc = g39.build_encoder(input_dims=Xtr.shape[-1], overrides=ov or None)
    g39.pretrain_encoder(enc, cls, Xtr_fit, max_iters=iters, batch_size=opts["batch_size"], lr=opts["lr"], device=device, seed=seed,
                         use_swa=opts["swa"], drop_last=opts["drop_last"], crop_mode=opts["crop_mode"],
                         checkpoints=tuple(cell["checkpoints"]), on_checkpoint=on_ckpt)
    single = g39.eval_with_timeout(lambda: g39._g35._svm_variants(box["ftr"], ytr, {opts["scaler"]: box["fte"]}, yte, (grid,), seed)[f"{opts['scaler']}/{grid}"],
                                   g39.EVAL_TIMEOUT_S)
    return np.asarray(box["ftr"], np.float32), np.asarray(box["fte"], np.float32), single, box["step"]


def cmd_run(args) -> None:
    ac, g35, g39, g83 = _g("acts_common"), _g("35_protocol_alignment"), _g("39_sh_deploy"), _g("83_acts_corpus")
    from autocls.oom_guard import is_oom_error
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); P = _json(outdir / "plan.json"); plan = P["plan"]
    remaining = {j: c for j, c in P["costs"].items() if not (outdir / "results" / f"{j}.json").exists()}
    mine = lpt_assign(remaining, args.workers)[args.worker] if remaining else []
    print(f"[run] worker {args.worker}/{args.workers}：{len(mine)} 个作业 {mine}", flush=True)
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols(); tm = _json(Path(args.ut) / "md" / "meta.json")["tasks_meta"]
    loader = g83.make_whole_loader(cfg, tm)
    from joblib import parallel_backend
    with parallel_backend("threading", n_jobs=1):                   # = 83 --run 默认（39.execute：threading + n_jobs 1）
        _run_jobs(mine, plan, outdir, loader, tm, protos, cfg, args)


def _run_jobs(mine, plan, outdir, loader, tm, protos, cfg, args) -> None:
    ac, g35, g39 = _g("acts_common"), _g("35_protocol_alignment"), _g("39_sh_deploy")
    from autocls.oom_guard import is_oom_error
    from autocls.space.cls_space import CLS
    for j in mine:
        d, s = j.rsplit("_s", 1); s = int(s); tid = f"{d}_whole_official"; t0 = time.time()
        try:
            arrays = loader(tid)
            uniq = {}
            for a in ARMS:
                for e in plan[d][a]:
                    uniq[e["cls_key"]] = e["cls"]
            reps, singles = {}, {}
            for k, c in uniq.items():
                p = outdir / "reps" / d / f"{k}_s{s}.npz"
                if p.exists():
                    with np.load(p) as z:
                        reps[k] = (z["ftr"], z["fte"]); singles[k] = float(z["acc"])
                    continue
                cell = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**c), k, s, protos, cfg, extra={"eval": "official", "whole_mode": "official"})
                ftr, fte, single, step = capture(cell, arrays, args.device)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, ftr=ftr, fte=fte, acc=single["acc"], step=step)
                reps[k] = (ftr, fte); singles[k] = float(single["acc"])
                print(f"[run] {j} {k} 单模型 acc {single['acc']:.3f}（截点 {step}）", flush=True)
            opts = ac.make_cell(tid, tm[tid], ac.ARM_RANDOM, CLS(**next(iter(uniq.values()))), next(iter(uniq)), s, protos, cfg,
                                extra={"eval": "official", "whole_mode": "official"})["options"]
            res = {"dataset": d, "seed": s, "singles": singles, "arms": {}}
            for a in ARMS:
                ks = [e["cls_key"] for e in plan[d][a]]
                Ztr = concat_l2([reps[k][0] for k in ks]); Zte = concat_l2([reps[k][1] for k in ks])
                try:
                    vv = g39.eval_with_timeout(lambda: g35._svm_variants(Ztr, arrays["ytr"], {opts["scaler"]: Zte}, arrays["yte"], (opts["svm_grid"],), s)
                                               [f"{opts['scaler']}/{opts['svm_grid']}"], g39.EVAL_TIMEOUT_S)
                    res["arms"][a] = {"keys": ks, "acc": float(vv["acc"]), "f1": float(vv["f1"]), "svm_c": vv["svm_c"]}
                except Exception as e:  # noqa: BLE001
                    res["arms"][a] = {"keys": ks, "acc": None, "error": f"{type(e).__name__}: {e}"}
                print(f"[run] {j} {a} 集成 acc {res['arms'][a].get('acc')}", flush=True)
            res["seconds"] = round(time.time() - t0, 1)
            _write(outdir / "results" / f"{j}.json", res)
        except Exception as e:  # noqa: BLE001
            if is_oom_error(e):
                import torch
                torch.cuda.empty_cache(); print(f"[run] {j} OOM → 下一轮重试：{e}", flush=True); continue
            raise


# ================================================================ 读数（本地）
def cmd_analyze(args) -> dict:
    v, u = _g("108_acts_s3v"), _g("118_acts_ut")
    outdir = Path(args.outdir); P = _json(outdir / "plan.json")["plan"]; ut = _json(Path(args.ut) / "analysis_ut.json")
    urows = v._rows(Path(args.ut) / "md"); D = u.UT_TARGETS
    R = {_jid(d, s): _json(outdir / "results" / f"{_jid(d, s)}.json") for d in D for s in SEEDS if (outdir / "results" / f"{_jid(d, s)}.json").exists()}
    o: dict = {"per": {}, "means": {}, "missing": [j for j in (_jid(d, s) for d, s in job_list()) if j not in R]}
    c0 = []
    for d in D:
        o["per"][d] = {}
        for a in ARMS:
            accs = [R[_jid(d, s)]["arms"][a]["acc"] for s in SEEDS if _jid(d, s) in R and R[_jid(d, s)]["arms"][a].get("acc") is not None]
            ks = [e["cls_key"] for e in P[d][a]]
            best_single = max(float(np.mean([R[_jid(d, s)]["singles"][k] for s in SEEDS if _jid(d, s) in R])) for k in ks) if R else np.nan
            o["per"][d][a] = {"ens": float(np.mean(accs)) if accs else np.nan, "ens_seeds": accs, "single_pick": ut["per_arm"][a][d]["test"],
                              "best_single_top3": best_single}
        for s in SEEDS:
            if _jid(d, s) in R:
                for k, acc in R[_jid(d, s)]["singles"].items():
                    if (f"{d}_whole_official", k, s) in urows:
                        c0.append(abs(acc - urows[(f"{d}_whole_official", k, s)]))
    for a in ARMS:
        for k in ("ens", "single_pick", "best_single_top3"):
            o["means"][f"{a}_{k}"] = float(np.nanmean([o["per"][d][a][k] for d in D]))
    ts, inf = float(np.mean(list(u.TS2VEC_UT.values()))), float(np.mean(list(u.INFOTS_S_UT.values())))
    m = o["means"]
    o["c0"] = {"n": len(c0), "median_abs_diff": float(np.median(c0)) if c0 else None, "max_abs_diff": float(np.max(c0)) if c0 else None}
    o["verdict"] = verdict_ens(m["ME_ens"], m["UNI_ens"], m["BLIND_ens"], SINGLE_ME, ts, inf)
    me = {d: o["per"][d]["ME"]["ens"] for d in D}
    o["wins"] = {"vs_TS2Vec": int(sum(me[d] > u.TS2VEC_UT[d] for d in D)), "vs_InfoTS_s": int(sum(me[d] > u.INFOTS_S_UT[d] for d in D)),
                 "vs_single": int(sum(me[d] > o["per"][d]["ME"]["single_pick"] for d in D))}
    o["predictions"] = {"P1": m["ME_ens"] > SINGLE_ME, "P2": m["ME_ens"] > ts, "P3": m["ME_ens"] > inf, "P4": m["ME_ens"] > m["UNI_ens"],
                        "P5": m["ME_ens"] > m["BLIND_ens"], "P6": (o["c0"]["median_abs_diff"] is not None and o["c0"]["median_abs_diff"] <= 0.01)}
    _write(outdir / "analysis.json", o)
    L = ["# UT-ENS 读数（11 个靶；官方 TEST，CV 前 3 表示拼接集成，3 种子均值）", "",
         "| 数据集 | **ENS_ME** | 单选 ME | ENS_UNI | 单选 UNI | ENS_BLIND | 单选 BLIND | TS2Vec | InfoTS_s | ME 前 3 最好单模型 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        p = o["per"][d]
        L.append(f"| {d} | {p['ME']['ens']:.3f} | {p['ME']['single_pick']:.3f} | {p['UNI']['ens']:.3f} | {p['UNI']['single_pick']:.3f} | "
                 f"{p['BLIND']['ens']:.3f} | {p['BLIND']['single_pick']:.3f} | {u.TS2VEC_UT[d]:.3f} | {u.INFOTS_S_UT[d]:.3f} | {p['ME']['best_single_top3']:.3f} |")
    L.append(f"| **均值** | **{m['ME_ens']:.4f}** | {m['ME_single_pick']:.4f} | {m['UNI_ens']:.4f} | {m['UNI_single_pick']:.4f} | {m['BLIND_ens']:.4f} | "
             f"{m['BLIND_single_pick']:.4f} | {ts:.4f} | {inf:.4f} | {m['ME_best_single_top3']:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/6", "",
          f"- ENS_ME 逐集胜：TS2Vec {o['wins']['vs_TS2Vec']}/11、InfoTS_s {o['wins']['vs_InfoTS_s']}/11、单选 ME {o['wins']['vs_single']}/11",
          f"- C0 单模型复算 vs UT 格：{o['c0']['n']} 对，中位 |差| {o['c0']['median_abs_diff']}，最大 {o['c0']['max_abs_diff']}",
          f"- 缺作业：{o['missing'] or '无'}"]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("plan", "run", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ut", default=str(UT))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("plan", cmd_plan), ("run", cmd_run), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
