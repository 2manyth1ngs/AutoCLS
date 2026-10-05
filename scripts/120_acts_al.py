"""WP17-ACTS AL（results/acts/al/PROTOCOL.md §2）：学增强（aug_learn = sup）目标侧上限探针。

UT 的 11 个未见靶、单种子 0：每靶 ME 臂 CV 前 3 条（取自 UT-ENS 计划）各开学增强训练一次 → 单模型精度 + 最后截点表示 → 逐块 L2 拼接集成；
另加 TS2Vec 等价 CLS 开 / 关各一次。关闭侧对照（前 3 条单模型 + 集成，种子 0）取自 UT-ENS 结果（同配置、同种子、同管线）。
阶段：--plan（本地）/ --run --worker i --workers n（服务器 GPU，LPT）/ --analyze（本地）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTS = ROOT / "results" / "acts"
UT = ACTS / "ut"
UTENS = ACTS / "utens"
OUT = ACTS / "al"
SEED = 0
REF_KEY = "ts2vec_equiv"


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


def verdict_al(delta_ens: float) -> str:
    return "AL-HEADROOM" if delta_ens >= 0.02 else ("AL-WEAK" if delta_ens >= 0.01 else "AL-FLAT")


def job_list() -> list:
    return list(_g("118_acts_ut").UT_TARGETS)


def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def cmd_plan(args) -> None:
    g35 = _g("35_protocol_alignment")
    P = _json(Path(args.utens) / "plan.json"); out, costs = {}, {}
    for d in job_list():
        out[d] = {"top3": P["plan"][d]["ME"], "ref": {"cls_key": REF_KEY, "cls": g35.TS2VEC_EQUIV.to_dict()}}
        costs[d] = float(P["costs"][f"{d}_s0"]) / 9 * 5 * 1.5
    _write(Path(args.outdir) / "plan.json", {"plan": out, "costs": costs, "seed": SEED})
    print(f"[plan] {len(out)} 个作业，估计 {sum(costs.values()) / 3600:.1f} GPU-h", flush=True)


def _cell(ac, g35, tid, m, c, k, protos, cfg, on: bool) -> dict:
    from autocls.space.cls_space import CLS
    cell = ac.make_cell(tid, m, ac.ARM_RANDOM, CLS(**c), k, SEED, protos, cfg, extra={"eval": "official", "whole_mode": "official"})
    if on:
        cell["options"] = dict(cell["options"], aug_learn="sup")
        cell["key"] = g35.cache_key(tid, cell["protocol"], k, SEED, cell["options"], cell["iters"], arm=cell["arm"])
    return cell


def cmd_run(args) -> None:
    ac, g35, g39, g83, e = _g("acts_common"), _g("35_protocol_alignment"), _g("39_sh_deploy"), _g("83_acts_corpus"), _g("119_acts_utens")
    from joblib import parallel_backend
    from autocls.oom_guard import is_oom_error
    outdir = Path(args.outdir); P = _json(outdir / "plan.json")
    remaining = {d: c for d, c in P["costs"].items() if not (outdir / "results" / f"{d}.json").exists()}
    mine = e.lpt_assign(remaining, args.workers)[args.worker] if remaining else []
    print(f"[run] worker {args.worker}/{args.workers}：{mine}", flush=True)
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols(); tm = _json(Path(args.ut) / "md" / "meta.json")["tasks_meta"]
    loader = g83.make_whole_loader(cfg, tm)
    with parallel_backend("threading", n_jobs=1):                   # = 83 --run
        for d in mine:
            tid = f"{d}_whole_official"; t0 = time.time(); plan = P["plan"][d]
            try:
                arrays = loader(tid); res = {"dataset": d, "seed": SEED, "singles_on": {}, "weights": {}, "meta_steps": {}}
                reps = []
                for ent in plan["top3"]:
                    info: dict = {}
                    ftr, fte, single, _step = e.capture(_cell(ac, g35, tid, tm[tid], ent["cls"], ent["cls_key"], protos, cfg, True), arrays, args.device, aug_info=info)
                    reps.append((ftr, fte)); res["singles_on"][ent["cls_key"]] = float(single["acc"])
                    res["weights"][ent["cls_key"]] = [float(w) for w in info.get("weights", [])]; res["meta_steps"][ent["cls_key"]] = int(info.get("meta_steps", 0))
                    print(f"[run] {d} {ent['cls_key']} 学增强单模型 acc {single['acc']:.3f}（元学习步 {info.get('meta_steps')}）", flush=True)
                opts = _cell(ac, g35, tid, tm[tid], plan["top3"][0]["cls"], plan["top3"][0]["cls_key"], protos, cfg, False)["options"]
                Ztr = e.concat_l2([r[0] for r in reps]); Zte = e.concat_l2([r[1] for r in reps])
                vv = g39.eval_with_timeout(lambda: g35._svm_variants(Ztr, arrays["ytr"], {opts["scaler"]: Zte}, arrays["yte"], (opts["svm_grid"],), SEED)
                                           [f"{opts['scaler']}/{opts['svm_grid']}"], g39.EVAL_TIMEOUT_S)
                res["ens_on"] = float(vv["acc"])
                for tag, on in (("ref_off", False), ("ref_on", True)):
                    info = {}
                    _f1, _f2, single, _s = e.capture(_cell(ac, g35, tid, tm[tid], plan["ref"]["cls"], REF_KEY, protos, cfg, on), arrays, args.device,
                                                     aug_info=info if on else None)
                    res[tag] = float(single["acc"])
                    if on:
                        res["weights"][REF_KEY] = [float(w) for w in info.get("weights", [])]
                print(f"[run] {d} 集成（开）{res['ens_on']:.3f}；TS2Vec 等价 关 {res['ref_off']:.3f} / 开 {res['ref_on']:.3f}", flush=True)
                res["seconds"] = round(time.time() - t0, 1)
                _write(outdir / "results" / f"{d}.json", res)
            except Exception as ex:  # noqa: BLE001
                if is_oom_error(ex):
                    import torch
                    torch.cuda.empty_cache(); print(f"[run] {d} OOM → 下一轮重试：{ex}", flush=True); continue
                raise


def cmd_analyze(args) -> dict:
    u = _g("118_acts_ut")
    outdir = Path(args.outdir); P = _json(outdir / "plan.json")["plan"]; D = job_list()
    from autocls.contrastive.learned_aug import TRANSFORM_NAMES
    rows, s_on, s_off, w_all = {}, [], [], []
    for d in D:
        r = _json(outdir / "results" / f"{d}.json"); off = _json(Path(args.utens) / "results" / f"{d}_s0.json")
        keys = [e["cls_key"] for e in P[d]["top3"]]
        on_s = [r["singles_on"][k] for k in keys]; off_s = [off["singles"][k] for k in keys]
        s_on += on_s; s_off += off_s
        w_all += [r["weights"][k] for k in keys if r["weights"].get(k)]
        rows[d] = {"ens_on": r["ens_on"], "ens_off": float(off["arms"]["ME"]["acc"]), "single_on": float(np.mean(on_s)), "single_off": float(np.mean(off_s)),
                   "ref_on": r["ref_on"], "ref_off": r["ref_off"]}
    m = {k: float(np.mean([rows[d][k] for d in D])) for k in ("ens_on", "ens_off", "single_on", "single_off", "ref_on", "ref_off")}
    ts, inf = float(np.mean([u.TS2VEC_UT[d] for d in D])), float(np.mean([u.INFOTS_S_UT[d] for d in D]))
    o = {"per": rows, "means": m, "delta_ens": m["ens_on"] - m["ens_off"], "delta_single": m["single_on"] - m["single_off"],
         "delta_ref": m["ref_on"] - m["ref_off"], "verdict": verdict_al(m["ens_on"] - m["ens_off"]),
         "single_wins_on": int(sum(a > b for a, b in zip(s_on, s_off))), "single_ties": int(sum(abs(a - b) < 1e-12 for a, b in zip(s_on, s_off))),
         "n_single": len(s_on), "weights_mean": dict(zip(TRANSFORM_NAMES, np.mean(w_all, axis=0).round(3).tolist())) if w_all else {},
         "baselines": {"TS2Vec": ts, "InfoTS_s": inf}}
    o["predictions"] = {"P1": o["delta_ens"] >= 0.02, "P2": o["delta_single"] > 0, "P3": o["delta_ref"] > 0, "P4": m["ens_on"] > inf}
    _write(outdir / "analysis.json", o)
    L = ["# AL 学增强上限探针（11 个未见靶，官方 TEST，单种子 0）", "",
         "| 数据集 | **集成 开** | 集成 关 | 单模型均值 开 | 单模型均值 关 | TS2Vec 等价 开 | TS2Vec 等价 关 | TS2Vec | InfoTS_s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in D:
        r = rows[d]
        L.append(f"| {d} | {r['ens_on']:.3f} | {r['ens_off']:.3f} | {r['single_on']:.3f} | {r['single_off']:.3f} | {r['ref_on']:.3f} | {r['ref_off']:.3f} | "
                 f"{u.TS2VEC_UT[d]:.3f} | {u.INFOTS_S_UT[d]:.3f} |")
    L.append(f"| **均值** | **{m['ens_on']:.4f}** | {m['ens_off']:.4f} | {m['single_on']:.4f} | {m['single_off']:.4f} | {m['ref_on']:.4f} | {m['ref_off']:.4f} | {ts:.4f} | {inf:.4f} |")
    L += ["", f"读数码：**{o['verdict']}**（集成 开 − 关 {o['delta_ens']:+.4f}）；预测 {sum(o['predictions'].values())}/4（P5 工程门见 PROTOCOL §6）", "",
          f"- 单模型 33 条：开 − 关 {o['delta_single']:+.4f}，开更好 {o['single_wins_on']}/{o['n_single']}（持平 {o['single_ties']}）",
          f"- TS2Vec 等价 CLS：开 − 关 {o['delta_ref']:+.4f}",
          f"- 学到的混合权重均值（{', '.join(TRANSFORM_NAMES)}）：{o['weights_mean']}"]
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
    ap.add_argument("--utens", default=str(UTENS))
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
