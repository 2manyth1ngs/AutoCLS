"""WP17-ACTS UT（results/acts/ut/PROTOCOL.md）：方法固定后在 11 个未见 UEA 数据集上部署评估（ME / UNI / BLIND + TS2Vec / InfoTS_s）。

阶段（服务器流水 scripts/pa_pod/acts_lab_run_ut.sh 依次调用；115 / 111 原样复用，只把 108.DATASETS 换成未见靶）：
  --prep        本地：11 靶 meta.json（官方 + 3 折 tcv，deploy_target = true）+ 基线对账 + 导出 merged npz
  --embed       服务器 GPU：靶 E_pt（官方 TRAIN，= 90 同法）→ ut/E/<靶>.npz（--parts / --part，--resume）
  --prep-ea     CPU：靶 E_A（官方 TRAIN）→ ut/md/ea_targets.npz
  --s0          s0src：成员 q = XS s0src 的 {F[q mod 6]}/{q} 列；Ept_target = 靶窗口表示均值 → ut/md/s0src.npz
  --models      ut/models/B_s{q}_{靶}.pt → S3-Z 模型 B_s{q}_{F[q mod 6]}.pt（符号链接）
  --fit         115 --fit 原样（21 源 192 任务）；BLIND = UNI 的专家 + γ 0
  --search      115 --search（--arm ME|UNI|BLIND --dataset 靶）
  --plan cv|final / --select   111 原样（臂 ME / UNI / BLIND）
  --analyze     本地读数
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTS = ROOT / "results" / "acts"
OUT = ACTS / "ut"
XS = ACTS / "xs"
S3Z = ACTS / "s3z"
SURVEY = ROOT / "results" / "uea_survey" / "uea_survey.csv"
UT_TARGETS = ("AtrialFibrillation", "BasicMotions", "DuckDuckGeese", "Epilepsy", "FaceDetection", "FingerMovements",
              "HandMovementDirection", "Heartbeat", "MotorImagery", "SelfRegulationSCP2", "StandWalkJump")
UEA_SOURCES = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1",
               "ArticularyWordRecognition", "ERing", "NATOPS", "PEMS-SF")
FOLDS = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1")   # = 108.DATASETS
# InfoTS 原文（arXiv 2303.11911）附录 Table 9：InfoTS_s / TS2Vec 列（6 旧靶与 108 已收录值逐值一致）
TS2VEC_UT = {"AtrialFibrillation": 0.200, "BasicMotions": 0.975, "DuckDuckGeese": 0.680, "Epilepsy": 0.964, "FaceDetection": 0.501,
             "FingerMovements": 0.480, "HandMovementDirection": 0.338, "Heartbeat": 0.683, "MotorImagery": 0.510,
             "SelfRegulationSCP2": 0.578, "StandWalkJump": 0.467}
INFOTS_S_UT = {"AtrialFibrillation": 0.267, "BasicMotions": 1.000, "DuckDuckGeese": 0.600, "Epilepsy": 0.993, "FaceDetection": 0.525,
               "FingerMovements": 0.620, "HandMovementDirection": 0.514, "Heartbeat": 0.771, "MotorImagery": 0.610,
               "SelfRegulationSCP2": 0.572, "StandWalkJump": 0.467}
ARMS_UT = ("ME", "UNI", "BLIND")
FINAL_SEEDS = (0, 1, 2)
K_TCV = 3


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def unseen_targets(rows: Mapping[str, Mapping], sources, excluded=("InsectWingbeat",)) -> list:
    """§2：UEA − 源 − 排除 − 变长 − L < 50 − L > 3000（名字序）。"""
    out = []
    for n, r in rows.items():
        L = int(r["length_max"])
        if n in set(sources) or n in set(excluded) or str(r["equal_length"]) != "True" or L < 50 or L > 3000:
            continue
        out.append(n)
    return sorted(out)


def member_fold(q: int) -> str:
    return FOLDS[int(q) % len(FOLDS)]


def verdict_ut(me: float, uni: float, blind: float, ts: Optional[float] = None, inf: Optional[float] = None) -> str:
    ts = float(np.mean(list(TS2VEC_UT.values()))) if ts is None else ts
    inf = float(np.mean(list(INFOTS_S_UT.values()))) if inf is None else inf
    code = "UT-BEATS-BOTH" if (me > inf and me > ts) else ("UT-BEATS-TS2VEC" if me > ts else "UT-BELOW")
    return code + ("+E" if me > uni else "") + ("+EXP" if uni > blind else "") + ("+TA" if me > blind else "")


# ================================================================ 上下文（115 / 111 原样复用）
def _ctx(datasets: Sequence[str]):
    v, y, g115 = _g("108_acts_s3v"), _g("111_acts_s3y"), _g("115_acts_s3md")
    v.DATASETS = tuple(datasets)
    y.ARMS_Y = {a: {} for a in ARMS_UT}
    g115.S3X = OUT / "_no_s3x"                                       # S3-X 池只有旧 6 靶：未见靶不读
    return v, y, g115


def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _survey() -> dict:
    return {r["name"]: r for r in csv.DictReader(open(SURVEY, encoding="utf-8"))}


def _targets(args) -> tuple:
    return tuple(args.only) if args.only else UT_TARGETS


# ================================================================ 阶段
def cmd_prep(args) -> None:
    ac = _g("acts_common")
    from autocls.data.uea import export_uea_merged_npz, merged_npz_path
    from autocls.labeling import run_seed_for
    rows = _survey()
    got = unseen_targets(rows, UEA_SOURCES)
    assert tuple(got) == UT_TARGETS, f"§2 规则复算 {got} ≠ 协议 {UT_TARGETS}"
    import yaml
    lit = yaml.safe_load(open(ROOT / "configs" / "uea30.yaml", encoding="utf-8"))
    lit_ts = {n: (e.get("literature") or {}).get("TS2Vec") for n, e in (lit.get("datasets") or {}).items()}
    mism = {d: (TS2VEC_UT[d], lit_ts.get(d)) for d in UT_TARGETS if lit_ts.get(d) is not None and abs(TS2VEC_UT[d] - float(lit_ts[d])) > 1e-9}
    print(f"[prep] TS2Vec 对账 configs/uea30.yaml：{'一致' if not mism else mism}（可对账 {sum(lit_ts.get(d) is not None for d in UT_TARGETS)}）", flush=True)
    md = Path(args.outdir) / "md"; (md / "runs").mkdir(parents=True, exist_ok=True)
    tm = {}
    for d in _targets(args):
        if not merged_npz_path(d).exists():
            export_uea_merged_npz(d)
        X, y = ac.load_target_full(d); n, nt = int(rows[d]["n_train"]), int(rows[d]["n_test"])
        assert n + nt == len(y), (d, n, nt, len(y))
        base = {"dataset": d, "T": int(X.shape[1]), "C": int(X.shape[2]), "n_train_official": n, "n_test_official": nt, "deploy_target": True}
        tm[f"{d}_whole_official"] = {**base, "n_train": n, "eval_mode": "official", "tag": "whole_official"}
        seed = int(run_seed_for(d, "s3y_tcv")); fid = ac.tcv_folds(np.asarray(y)[:n], K_TCV, seed)
        for i in range(K_TCV):
            tm[f"{d}_whole_tcv{i}"] = {**base, "n_train": int((fid != i).sum()), "n_eval": int((fid == i).sum()), "eval_mode": "tcv",
                                       "tcv_k": K_TCV, "tcv_fold": i, "tcv_seed": seed, "tag": f"whole_tcv{i}"}
        print(f"[prep] {d}: X={X.shape} TRAIN {n} / TEST {nt} → 3 折验证 {[int((fid == i).sum()) for i in range(K_TCV)]}", flush=True)
    _write(md / "meta.json", {"datasets": list(_targets(args)), "tasks_meta": tm})
    (md / "runs" / "runs.jsonl").touch()


def cmd_embed(args) -> None:
    import torch
    ac, g86 = _g("acts_common"), _g("86_acts_s3b")
    from autocls.task_embed import TS2VecWrapper
    outdir = Path(args.outdir); tm = _json(outdir / "md" / "meta.json")["tasks_meta"]; (outdir / "E").mkdir(parents=True, exist_ok=True)
    todo = [d for i, d in enumerate(_targets(args)) if args.parts <= 1 or i % args.parts == args.part]
    for d in todo:
        p = outdir / "E" / f"{d}.npz"
        if p.exists():
            continue
        m = tm[f"{d}_whole_official"]; X, y = ac.load_target_full(d)
        Xtr = ac.whole_arrays_official(X, y, n_train=m["n_train_official"])["Xtr"]   # = 90.cmd_embed：只用官方 TRAIN
        n, t, c = Xtr.shape; pool = Xtr.transpose(0, 2, 1).reshape(n * c, t, 1)
        iters = g86.ts2vec_iters(pool.size) if args.iters is None else int(args.iters); split_len = 0; t0 = time.time()
        try:
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            w.fit(pool, n_iters=iters)
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower():
                raise
            w = None; torch.cuda.empty_cache(); split_len = g86.SH_MAX_LEN
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            w.fit(g86.split_long(pool, g86.SH_MAX_LEN), n_iters=iters)
        w.freeze()
        r = g86.encode_windows(w, Xtr, g86.window_idx(n, f"{d}_whole_official"))
        np.savez_compressed(p, windows_repr=r, iters=iters, split_len=split_len, seconds=time.time() - t0)
        print(f"[embed] {d} Xtr={Xtr.shape} iters={iters} split={split_len} {time.time() - t0:.0f}s", flush=True)
        del w; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def cmd_prep_ea(args) -> None:
    ac = _g("acts_common")
    from autocls.data.mech_features import RESPONSE_PROFILE_COLS, response_profile
    from autocls.task_types import TASK_CLASSIFICATION
    outdir = Path(args.outdir); tm = _json(outdir / "md" / "meta.json")["tasks_meta"]; out = {}
    for d in _targets(args):
        t0 = time.time(); X, y = ac.load_target_full(d)
        wa = ac.whole_arrays_official(X, y, n_train=int(tm[f"{d}_whole_official"]["n_train_official"]))
        r = response_profile(wa["Xtr"], wa["ytr"], task_type=TASK_CLASSIFICATION)
        out[d] = np.array([float(r[k]) for k in RESPONSE_PROFILE_COLS])
        print(f"[prep-ea] {d} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(outdir / "md" / "ea_targets.npz", **out)


def cmd_s0(args) -> None:
    outdir = Path(args.outdir); md = outdir / "md"
    with np.load(XS / "md" / "s0src.npz") as z:
        xs = {k: z[k] for k in z.files}
    out = {"lens": xs["lens"]}
    for d in _targets(args):
        for q in range(10):
            out[f"{d}/{q}"] = xs[f"{member_fold(q)}/{q}"]
        with np.load(outdir / "E" / f"{d}.npz") as z:
            out[f"Ept_target/{d}"] = np.asarray(z["windows_repr"], np.float64).mean(axis=(0, 1))   # = 115.cmd_score_src
    np.savez_compressed(md / "s0src.npz", **out)
    print(f"[s0] {len(_targets(args))} 靶 × 10 成员（成员 q = 折 F[q mod 6] 种子 q），任务 {len(out['lens'])}", flush=True)


def cmd_models(args) -> None:
    src, dst = Path(args.models_src), Path(args.outdir) / "models"; dst.mkdir(parents=True, exist_ok=True)
    for d in _targets(args):
        for q in range(10):
            p, s = dst / f"B_s{q}_{d}.pt", src / f"B_s{q}_{member_fold(q)}.pt"
            if p.exists() or p.is_symlink():
                p.unlink()
            try:
                os.symlink(s.resolve(), p)
            except OSError:                                          # Windows 无符号链接权限 → 复制
                shutil.copy2(s, p)
    print(f"[models] {len(_targets(args))} 靶 × 10 → {dst}", flush=True)


def cmd_fit(args) -> None:
    m114, g116 = _g("114_acts_s3me"), _g("116_acts_xs")
    v, _y, g115 = _ctx(_targets(args))
    md = Path(args.outdir) / "md"; base = m114.load_all
    data = g116.merged_data(base, XS)
    m114.load_all = lambda: data
    try:
        g115.cmd_fit(argparse.Namespace(outdir=str(md), folds=list(_targets(args)), members=args.members))
    finally:
        m114.load_all = base
    with np.load(md / "experts.npz") as z:
        ex = {k: z[k] for k in z.files}
    for d in _targets(args):                                         # BLIND = γ 0（专家不参与；W / c / π 取 UNI 的，只为格式）
        for k in ("W", "c", "pi"):
            ex[f"BLIND/{d}/{k}"] = ex[f"UNI/{d}/{k}"]
        ex[f"BLIND/{d}/gamma"] = np.array(0.0)
    np.savez_compressed(md / "experts.npz", **ex)


def cmd_search(args) -> None:
    _v, _y, g115 = _ctx(_targets(args))
    g115.cmd_search(argparse.Namespace(outdir=str(Path(args.outdir) / "md"), arm=args.arm, dataset=args.dataset, device=args.device, resume=args.resume,
                                       members=args.members, models=str(Path(args.outdir) / "models"), ks=args.ks, gens=args.gens))


def cmd_plan(args) -> None:
    _v, y, _g115 = _ctx(_targets(args))
    y.cmd_plan(argparse.Namespace(outdir=str(Path(args.outdir) / "md"), plan=args.plan))


def cmd_select(args) -> None:
    _v, y, _g115 = _ctx(_targets(args))
    y.cmd_select(argparse.Namespace(outdir=str(Path(args.outdir) / "md")))


def cmd_analyze(args) -> dict:
    v, _y, _g115 = _ctx(_targets(args))
    md = Path(args.outdir) / "md"; rows = v._rows(md); sel = _json(md / "select.json"); fit = _json(md / "fit.json"); D = _targets(args)
    o: dict = {"per_arm": {}, "means": {}}
    for a in ARMS_UT:
        r = {}
        for d in D:
            sr = _json(md / "search" / f"{a}_{d}.json"); t = f"{d}_whole_official"
            keys = [e["cls_key"] for e in sr["union"]]; g_ = lambda k, s: rows.get((t, k, s), np.nan)  # noqa: E731  error 行（如 EvalTimeout）→ NaN
            s0 = np.array([g_(k, 0) for k in keys]); pk, co = sel[a][d]["pick"], sel[a][d]["comp_only"]
            seeds = [g_(pk, s) for s in FINAL_SEEDS]
            r[d] = {"pick": pk, "test": float(np.nanmean(seeds)), "seeds": seeds, "n_missing_pick": int(np.isnan(seeds).sum()),
                    "comp_test": float(np.nanmean([g_(co, s) for s in FINAL_SEEDS])), "U_n": len(keys), "U_missing_s0": int(np.isnan(s0).sum()),
                    "U_best_s0": float(np.nanmax(s0)), "U_mean_s0": float(np.nanmean(s0)), "sel_loss_s0": float(np.nanmax(s0) - g_(pk, 0))}
        o["per_arm"][a] = r
        for k in ("test", "comp_test", "U_best_s0", "U_mean_s0", "sel_loss_s0"):
            o["means"][f"{a}_{k}"] = float(np.nanmean([r[d][k] for d in D]))
        o["means"][f"{a}_missing"] = int(sum(r[d]["n_missing_pick"] + r[d]["U_missing_s0"] for d in D))
    ts = {d: TS2VEC_UT[d] for d in D}; inf = {d: INFOTS_S_UT[d] for d in D}; m = o["means"]
    tsm, infm = float(np.mean(list(ts.values()))), float(np.mean(list(inf.values())))
    o["baselines"] = {"TS2Vec": tsm, "InfoTS_s": infm}
    o["verdict"] = verdict_ut(m["ME_test"], m["UNI_test"], m["BLIND_test"], tsm, infm)
    me = o["per_arm"]["ME"]
    o["wins"] = {"vs_TS2Vec": int(sum(me[d]["test"] > ts[d] for d in D)), "vs_InfoTS_s": int(sum(me[d]["test"] > inf[d] for d in D)),
                 "vs_UNI": int(sum(me[d]["test"] > o["per_arm"]["UNI"][d]["test"] for d in D)),
                 "vs_BLIND": int(sum(me[d]["test"] > o["per_arm"]["BLIND"][d]["test"] for d in D))}
    o["predictions"] = {"P1": m["ME_test"] > tsm, "P2": m["ME_test"] > infm, "P3": m["ME_test"] > m["UNI_test"], "P4": m["ME_test"] > m["BLIND_test"],
                        "P5": o["wins"]["vs_TS2Vec"] >= 6, "P6": m["UNI_test"] > m["BLIND_test"]}
    # 17 集合并（旧 6 靶用 XS 读数：ME_new / UNI_new / BLIND = S3-Z NEW_B；TEST 已多次使用，只作描述）
    try:
        xs = _json(XS / "md" / "analysis.json"); z = _json(S3Z / "analysis.json"); vv = _g("108_acts_s3v")
        old = {"ME": [xs["per_arm"]["ME"][d]["test"] for d in FOLDS], "UNI": [xs["per_arm"]["UNI"][d]["test"] for d in FOLDS],
               "BLIND": [z["per_arm"]["NEW_B"][d]["test"] for d in FOLDS]}
        o["combined17"] = {a: float(np.mean(old[a] + [o["per_arm"][a][d]["test"] for d in D])) for a in ARMS_UT}
        o["combined17"].update({"TS2Vec": float(np.mean([vv.TS2VEC[d] for d in FOLDS] + list(ts.values()))),
                                "InfoTS_s": float(np.mean([vv.INFOTS_S[d] for d in FOLDS] + list(inf.values())))})
    except (FileNotFoundError, KeyError):
        o["combined17"] = None
    _write(Path(args.outdir) / "analysis_ut.json", o)
    L = ["# UT 部署读数（11 个未见 UEA 靶；官方 TEST，3 折 CV 选 1，3 种子均值）", "",
         "| 数据集 | **ME** | UNI | BLIND | TS2Vec | InfoTS_s | ME 3 种子 | ME 纯比较器 | ME 的 U 最好（种子 0） | τ / γ |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for d in D:
        a, b, c = me[d], o["per_arm"]["UNI"][d], o["per_arm"]["BLIND"][d]
        L.append(f"| {d} | {a['test']:.3f} | {b['test']:.3f} | {c['test']:.3f} | {ts[d]:.3f} | {inf[d]:.3f} | {'/'.join(f'{s:.3f}' for s in a['seeds'])} | "
                 f"{a['comp_test']:.3f} | {a['U_best_s0']:.3f} | {fit[d]['ME']['tau']:g} / {fit[d]['ME']['gamma']:g} |")
    L.append(f"| **均值** | **{m['ME_test']:.4f}** | {m['UNI_test']:.4f} | {m['BLIND_test']:.4f} | {tsm:.4f} | {infm:.4f} | | {m['ME_comp_test']:.4f} | {m['ME_U_best_s0']:.4f} | |")
    L += ["", f"读数码：**{o['verdict']}**；预测 {sum(o['predictions'].values())}/6", "",
          f"- ME 逐集胜：TS2Vec {o['wins']['vs_TS2Vec']}/{len(D)}、InfoTS_s {o['wins']['vs_InfoTS_s']}/{len(D)}、UNI {o['wins']['vs_UNI']}/{len(D)}、BLIND {o['wins']['vs_BLIND']}/{len(D)}",
          f"- 缺格（error 行，如 EvalTimeout）：ME {m['ME_missing']}、UNI {m['UNI_missing']}、BLIND {m['BLIND_missing']}（选中者 3 种子 + U 种子 0）",
          f"- CV 选择损失（种子 0）：ME {m['ME_sel_loss_s0']:+.4f}、UNI {m['UNI_sel_loss_s0']:+.4f}、BLIND {m['BLIND_sel_loss_s0']:+.4f}；"
          f"U 平均 ME {m['ME_U_mean_s0']:.4f} / UNI {m['UNI_U_mean_s0']:.4f} / BLIND {m['BLIND_U_mean_s0']:.4f}"]
    if o["combined17"]:
        c17 = o["combined17"]
        L.append(f"- 6 旧靶 + 11 未见靶合并 17 集（旧靶 TEST 已多次使用，只作描述）：ME {c17['ME']:.4f} / UNI {c17['UNI']:.4f} / BLIND {c17['BLIND']:.4f} / "
                 f"TS2Vec {c17['TS2Vec']:.4f} / InfoTS_s {c17['InfoTS_s']:.4f}")
    (Path(args.outdir) / "tables_ut.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "embed", "prep_ea", "s0", "models", "fit", "search", "select", "analyze", "resume"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--plan", choices=("cv", "final"))
    ap.add_argument("--arm", choices=ARMS_UT)
    ap.add_argument("--dataset")
    ap.add_argument("--only", nargs="*", default=None, help="smoke：只处理这些靶")
    ap.add_argument("--members", nargs="*", type=int, default=None)
    ap.add_argument("--models-src", default=str(S3Z / "models"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("prep", cmd_prep), ("embed", cmd_embed), ("prep_ea", cmd_prep_ea), ("s0", cmd_s0), ("models", cmd_models), ("fit", cmd_fit),
                     ("search", cmd_search), ("select", cmd_select), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    if args.plan:
        cmd_plan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
