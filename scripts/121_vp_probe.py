"""WP18-VP（results/acts/vp/PROTOCOL.md）：输入视图（对比在哪个域上做）上限探针。

每格 = (数据集, 评估口径, CLS, 视图, 种子)：官方 ts2vec_official 训练（bs = min(8, N)、drop_last、SWA、TRAIN 标准化）
一条轨迹跑到 4× 官方迭代数，在 1× / 2× / 4× 截点各编码一次，读出 max（官方）/ maxavg（max ⊕ mean）两种，
官方 SVM 网格评估。源集（10 个 UEA 源）用官方 TEST；11 个未见靶只用官方 TRAIN 内分层 3 折（TEST 不读）。

  --plan                 写 plan.json
  --run --part i --parts n --device cuda:k    跑分到第 i 份的格（LPT 按估计成本），追加 runs/part{i}.jsonl，按 key 续跑
  --analyze              本地读数 → analysis.json / tables.md
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "vp"
SOURCES = ("Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "SelfRegulationSCP1",
           "ArticularyWordRecognition", "ERing", "NATOPS", "PEMS-SF")
UT = ("AtrialFibrillation", "BasicMotions", "DuckDuckGeese", "Epilepsy", "FaceDetection", "FingerMovements",
      "HandMovementDirection", "Heartbeat", "MotorImagery", "SelfRegulationSCP2", "StandWalkJump")
VIEWS = ("raw", "diff", "psd", "spec")
CFGS = ("ts2v", "ggs")
MULTS = (1, 2, 4)
READOUTS = ("max", "maxavg")
K_TCV = 3


def _g(name):
    return importlib.import_module(f"scripts.{name}")


def cfg_cls(name: str):
    """ts2v = TS2Vec 在本空间的等价表示 + 官方 encoder 输入掩码；ggs = AutoCL 通用好策略（本空间合法版）。"""
    if name == "ts2v":
        return _g("35_protocol_alignment").TS2VEC_EQUIV, {"input_mask": "binomial"}
    if name == "ggs":
        return _g("m1_ggs_har").GGS, {}
    raise KeyError(name)


def survey() -> dict:
    import csv
    return {r["name"]: r for r in csv.DictReader(open(ROOT / "results" / "uea_survey" / "uea_survey.csv", encoding="utf-8"))}


def cell_key(c: dict) -> str:
    return f"{c['dataset']}|{c['mode']}|{c['cfg']}|{c['view']}|s{c['seed']}"


def build_plan(seed: int = 0) -> list[dict]:
    sv = survey()
    cells = []
    for d in SOURCES + UT:
        r = sv[d]
        modes = ("official",) if d in SOURCES else tuple(f"tcv{i}" for i in range(K_TCV))
        for m in modes:
            for cf in CFGS:
                for v in VIEWS:
                    c = {"dataset": d, "mode": m, "cfg": cf, "view": v, "seed": seed, "n_train": int(r["n_train"]),
                         "T": int(r["length_max"]), "C": int(r["n_channels"])}
                    c["key"] = cell_key(c)
                    cells.append(c)
    return cells


def est_cost(c: dict) -> float:
    size = c["n_train"] * c["T"] * c["C"]
    iters = 200 if size <= 100_000 else 600
    per_iter = 8 * c["T"] * (1.0 if c["view"] in ("raw", "diff") else 0.3)
    svm = c["n_train"] ** 1.6 * 6e-4
    return 4 * iters * per_iter * 1e-5 + svm + 5


def lpt_parts(cells: list[dict], n: int) -> list[list[dict]]:
    load = [0.0] * n
    parts: list[list[dict]] = [[] for _ in range(n)]
    for c in sorted(cells, key=est_cost, reverse=True):
        i = int(np.argmin(load))
        parts[i].append(c)
        load[i] += est_cost(c)
    return parts


# ================================================================ GPU 执行体
def arrays_for(c: dict) -> dict:
    ac = _g("acts_common")
    from autocls.labeling import run_seed_for
    X, y = ac.load_target_full(c["dataset"])
    if c["mode"] == "official":
        return ac.whole_arrays_official(X, y, n_train=c["n_train"])
    fold = int(c["mode"][3:])
    return ac.whole_arrays_tcv(X, y, n_train=c["n_train"], k=K_TCV, fold=fold, seed=int(run_seed_for(c["dataset"], "vp_tcv")))


def view_arrays(arr: dict, view: str, chunk: int = 256):
    from autocls.data.views import apply_view, standardize_train

    def tf(X):
        return np.concatenate([apply_view(X[i:i + chunk], view) for i in range(0, len(X), chunk)], axis=0)
    Xtr, Xte = tf(arr["Xtr"]), tf(arr["Xte_train"])
    return standardize_train(Xtr, Xte)


def encode_readouts(module, X: np.ndarray, device: str, bs: int = 64) -> dict:
    import torch
    module.eval()
    mx, av = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            z = module(torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).to(device))
            mx.append(z.max(dim=1).values.cpu().numpy())
            av.append(z.mean(dim=1).cpu().numpy())
    mx, av = np.concatenate(mx), np.concatenate(av)
    return {"max": mx, "maxavg": np.concatenate([mx, av], axis=1)}


def svm_acc(ftr, ytr, fte, yte) -> dict:
    from sklearn.metrics import accuracy_score
    from autocls.downstream.linear_eval import fit_svm
    clf = fit_svm(ftr, ytr, grid="ts2vec")
    pred = clf.predict(fte)
    cv = getattr(clf, "cv_score_", None)
    return {"acc": float(accuracy_score(yte, pred)), "svm_c": float(clf.C), "svm_cv": (float(cv) if cv is not None else None)}


def run_cell(c: dict, device: str) -> dict:
    import torch
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.encoders.registry import build_encoder
    t0 = time.perf_counter()
    arr = arrays_for(c)
    Xtr, Xte = view_arrays(arr, c["view"])
    ytr, yte = arr["ytr"], arr["yte"]
    base_iters = 200 if int(arr["Xtr"].size) <= 100_000 else 600            # 官方规则按原始 TRAIN 规模，各视图同预算
    ckpts = tuple(base_iters * m for m in MULTS)
    cls, ov = cfg_cls(c["cfg"])
    res: list[dict] = []

    def on_ckpt(step, module):
        te = time.perf_counter()
        ftr = encode_readouts(module, Xtr, device)
        fte = encode_readouts(module, Xte, device)
        for r in READOUTS:
            res.append({"iters": int(step), "mult": int(step // base_iters), "readout": r, **svm_acc(ftr[r], ytr, fte[r], yte),
                        "eval_s": round(time.perf_counter() - te, 1)})
        module.train()

    torch.manual_seed(int(c["seed"]))
    enc = build_encoder(input_dims=int(Xtr.shape[-1]), overrides=ov or None)
    bs = int(min(8, Xtr.shape[0]))
    try:
        pretrain_encoder(enc, cls, Xtr, max_iters=int(ckpts[-1]), batch_size=bs, lr=1e-3, device=device, seed=int(c["seed"]),
                         use_swa=True, drop_last=True, crop_mode="fixed", checkpoints=ckpts, on_checkpoint=on_ckpt)
        err = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {**c, "results": res, "error": err, "base_iters": base_iters, "shape_tr": list(Xtr.shape), "n_eval": int(len(yte)),
            "seconds": round(time.perf_counter() - t0, 1)}


def done_keys(paths) -> set:
    out = set()
    for p in paths:
        if Path(p).exists():
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("error") is None:
                    out.add(r["key"])
    return out


def cmd_run(args) -> None:
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))["cells"]
    if args.only_dataset:
        plan = [c for c in plan if c["dataset"] in set(args.only_dataset)]
    parts = lpt_parts(plan, args.parts)
    mine = parts[args.part]
    out = OUT / "runs" / f"part{args.part}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(sorted((OUT / "runs").glob("part*.jsonl")))
    todo = [c for c in mine if c["key"] not in done]
    print(f"[run] part {args.part}/{args.parts} device {args.device}: {len(todo)}/{len(mine)} 待跑", flush=True)
    for i, c in enumerate(todo):
        r = run_cell(c, args.device)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        best = max((x["acc"] for x in r["results"]), default=None)
        print(f"[run] {i + 1}/{len(todo)} {c['key']} {r['seconds']}s err={r['error']} best={best}", flush=True)


# ================================================================ 读数（本地）
def load_rows() -> list[dict]:
    rows = {}
    for p in sorted((OUT / "runs").glob("part*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") is None and r.get("results"):
                rows[r["key"]] = r
    return list(rows.values())


def acc_table(rows) -> dict:
    """(dataset, cfg, view, mult, readout) → acc（tcv 三折取均值，按折样本数加权）。"""
    acc: dict = {}
    wsum: dict = {}
    for r in rows:
        for x in r["results"]:
            k = (r["dataset"], r["cfg"], r["view"], x["mult"], x["readout"])
            acc[k] = acc.get(k, 0.0) + x["acc"] * r["n_eval"]
            wsum[k] = wsum.get(k, 0) + r["n_eval"]
    n_folds: dict = {}
    for r in rows:
        n_folds.setdefault((r["dataset"], r["cfg"], r["view"]), set()).add(r["mode"])
    return {k: acc[k] / wsum[k] for k in acc if len(n_folds[(k[0], k[1], k[2])]) == (1 if k[0] in SOURCES else K_TCV)}


def cmd_analyze(args) -> None:
    rows = load_rows()
    A = acc_table(rows)
    lines = ["# VP 探针读数", "", "源集 = 官方 TEST；未见靶 = 官方 TRAIN 内 3 折 CV（TEST 未读）。单种子 0。", ""]
    out = {"per_dataset": {}}
    for mult, ro in [(1, "max"), (4, "max"), (1, "maxavg")]:
        lines += [f"## 截点 {mult}×、读出 {ro}（两 CLS 均值；括号内 ts2v / ggs）", "",
                  "| 数据集 | " + " | ".join(VIEWS) + " | 最好非 raw − raw |", "|---|" + "---:|" * (len(VIEWS) + 1)]
        gains = []
        for d in SOURCES + UT:
            vals = {}
            for v in VIEWS:
                a = [A.get((d, cf, v, mult, ro)) for cf in CFGS]
                vals[v] = a
            if any(x is None for v in VIEWS for x in vals[v]):
                continue
            mean = {v: float(np.mean(vals[v])) for v in VIEWS}
            best_v = max((v for v in VIEWS if v != "raw"), key=lambda v: mean[v])
            g = mean[best_v] - mean["raw"]
            gains.append(g)
            out["per_dataset"].setdefault(f"{mult}x/{ro}", {})[d] = {"mean": mean, "by_cfg": vals, "best": best_v, "gain": g}
            lines.append(f"| {d}{'' if d in SOURCES else '（CV）'} | " + " | ".join(
                f"{mean[v]:.3f} ({vals[v][0]:.2f}/{vals[v][1]:.2f})" for v in VIEWS) + f" | {best_v} {g:+.3f} |")
        lines.append("")
    # 训练量与读出（raw 视图）
    lines += ["## raw 视图：训练量 × 读出（两 CLS 均值）", "", "| 数据集 | 1× max | 2× max | 4× max | 1× maxavg | 4× maxavg |", "|---|---:|---:|---:|---:|---:|"]
    for d in SOURCES + UT:
        cells = [[A.get((d, cf, "raw", m, ro)) for cf in CFGS] for m, ro in [(1, "max"), (2, "max"), (4, "max"), (1, "maxavg"), (4, "maxavg")]]
        if any(x is None for c in cells for x in c):
            continue
        lines.append(f"| {d} | " + " | ".join(f"{np.mean(c):.3f}" for c in cells) + " |")
    # 视图偏好一致性（两 CLS 同号）
    lines += ["", "## 视图偏好跨 CLS 一致性（1× max；psd / spec / diff − raw 的符号是否两 CLS 相同）", ""]
    agree = []
    for d in SOURCES + UT:
        for v in ("diff", "psd", "spec"):
            a = [A.get((d, cf, v, 1, "max")) for cf in CFGS]
            b = [A.get((d, cf, "raw", 1, "max")) for cf in CFGS]
            if None in a or None in b:
                continue
            d0, d1 = a[0] - b[0], a[1] - b[1]
            if max(abs(d0), abs(d1)) >= 0.03:
                agree.append(np.sign(d0) == np.sign(d1))
    lines.append(f"|Δ| ≥ 0.03 的 (数据集, 视图) 中两 CLS 同号比例：{np.mean(agree):.2f}（n = {len(agree)}）" if agree else "无")
    out["agree"] = float(np.mean(agree)) if agree else None
    (OUT / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-dataset", nargs="*")
    args = ap.parse_args(argv)
    if args.plan:
        cells = build_plan()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "plan.json").write_text(json.dumps({"cells": cells}, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"[plan] {len(cells)} 格")
    if args.run:
        cmd_run(args)
    if args.analyze:
        cmd_analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
