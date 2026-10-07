"""WP18 GPU 执行体：一格 = 一条 ts2vec_official 预训练轨迹（bs = min(8, N) 官方 DataLoader 口径、drop_last、SWA、lr 1e-3），
扩展轴 view（输入视图）/ mult（训练量倍数，截点）/ readout（max | maxavg）在格内生效。

两种格：
  kind = "src"：源切片 cv5（= S2 标签口径：整切片标准化 → 视图 → 整切片再标准化 → 预训练 → 5 折官方 SVM），
                一条轨迹在 mult 的每个取值处截点、每个读出各出一条标签；
  kind = "tgt"：部署靶 official / tcv{i}（TRAIN 统计量标准化；TEST / 验证折只编码），只在终点评估；
                save_reps = True 时把终点读出的 TRAIN / 评估特征落盘（集成用）。
官方迭代数按**完整官方 TRAIN** 规模（N_train·T·C ≤ 1e5 → 200，否则 600），tcv 折与全量同训练长度（修正旧流水 CV 与部署训练长度不一致）。
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
READOUTS = ("max", "maxavg")


def _g(name):
    return importlib.import_module(f"scripts.{name}")


def official_iters(size: int) -> int:
    return 200 if int(size) <= 100_000 else 600


def view_pair(Xtr: np.ndarray, Xev: Optional[np.ndarray], view: str, chunk: int = 256):
    from autocls.data.views import apply_view, standardize_train

    def tf(X):
        return np.concatenate([apply_view(X[i:i + chunk], view) for i in range(0, len(X), chunk)], axis=0)
    a = tf(Xtr)
    if Xev is None:
        return standardize_train(a)[0], None
    return standardize_train(a, tf(Xev))


def encode_readouts(module, X: np.ndarray, device: str, bs: int = 64) -> dict:
    import torch
    module.eval()
    mx, av = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            z = module(torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).to(device))
            mx.append(z.max(dim=1).values.float().cpu().numpy())
            av.append(z.mean(dim=1).float().cpu().numpy())
    mx, av = np.concatenate(mx), np.concatenate(av)
    return {"max": mx, "maxavg": np.concatenate([mx, av], axis=1)}


def svm_eval(ftr, ytr, fev, yev) -> dict:
    from sklearn.metrics import accuracy_score
    from autocls.downstream.linear_eval import fit_svm
    clf = fit_svm(ftr, ytr, grid="ts2vec")
    pred = clf.predict(fev)
    cv = getattr(clf, "cv_score_", None)
    return {"acc": float(accuracy_score(yev, pred)), "svm_c": float(clf.C), "svm_cv": (float(cv) if cv is not None else None)}


def _pretrain(cls_dict: Mapping, input_mask: str, Xtr: np.ndarray, iters_max: int, ckpts: Sequence[int], seed: int, device: str, on_ckpt):
    import torch
    from autocls.contrastive.cl_trainer import pretrain_encoder
    from autocls.encoders.registry import build_encoder
    from autocls.space.cls_space import CLS
    cls = CLS(**cls_dict)
    torch.manual_seed(int(seed))
    ov = {"input_mask": input_mask} if input_mask and input_mask != "none" else None
    enc = build_encoder(input_dims=int(Xtr.shape[-1]), overrides=ov)
    bs = int(min(8, Xtr.shape[0]))
    pretrain_encoder(enc, cls, Xtr, max_iters=int(iters_max), batch_size=bs, lr=1e-3, device=device, seed=int(seed),
                     use_swa=True, drop_last=True, crop_mode="fixed", checkpoints=tuple(int(c) for c in ckpts), on_checkpoint=on_ckpt)


# ================================================================ 源切片 cv5 标签格
def run_src_cell(cell: Mapping, arrays: Mapping, device: str) -> dict:
    """cell: {key, task_id, cls, view, mults, readouts, seed, iters(=任务官方迭代数), input_mask}。"""
    ac = _g("acts_common")
    t0 = time.perf_counter()
    X, y, folds = arrays["X"], arrays["y"], arrays["folds"]
    Xv, _ = view_pair(X, None, cell["view"])
    base = int(cell["iters"]); mults = sorted(int(m) for m in cell["mults"])
    ckpts = [base * m for m in mults]
    res = []

    def on_ckpt(step, module):
        f = encode_readouts(module, Xv, device)
        for ro in cell["readouts"]:
            v = ac.cv_eval(f[ro], y, folds, "ts2vec", int(cell["seed"]))
            res.append({"mult": int(step // base), "readout": ro, "acc": float(v["acc"]), "fold_acc": v["fold_acc"]})
        module.train()
    err = None
    try:
        _pretrain(cell["cls"], cell.get("input_mask", "none"), Xv, ckpts[-1], ckpts, cell["seed"], device, on_ckpt)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {**cell, "results": res, "error": err, "shape": list(Xv.shape), "seconds": round(time.perf_counter() - t0, 1)}


# ================================================================ 部署靶格
def target_arrays(dataset: str, mode: str, n_train: int, tcv_seed: int, k: int = 3) -> dict:
    ac = _g("acts_common")
    X, y = ac.load_target_full(dataset)
    if mode == "official":
        return ac.whole_arrays_official(X, y, n_train=n_train)
    return ac.whole_arrays_tcv(X, y, n_train=n_train, k=k, fold=int(mode[3:]), seed=tcv_seed)


def run_tgt_cell(cell: Mapping, arrays: Mapping, device: str, rep_dir: Optional[Path] = None) -> dict:
    """cell: {key, dataset, mode, cls, ext:{view, mult, readout}, seed, base_iters, input_mask, save_reps}。只在终点评估。"""
    t0 = time.perf_counter()
    ext = dict(cell.get("ext") or {})
    view, mult, ro = ext.get("view", "raw"), int(ext.get("mult", 1)), ext.get("readout", "max")
    Xtr, Xev = view_pair(arrays["Xtr"], arrays["Xte_train"], view)
    ytr, yev = arrays["ytr"], arrays["yte"]
    iters = int(cell["base_iters"]) * mult
    out: dict = {}

    def on_ckpt(step, module):
        ftr = encode_readouts(module, Xtr, device)[ro]
        fev = encode_readouts(module, Xev, device)[ro]
        out.update(svm_eval(ftr, ytr, fev, yev))
        if cell.get("save_reps") and rep_dir is not None:
            Path(rep_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(rep_dir) / f"{cell['key']}.npz", ftr=ftr.astype(np.float32), fev=fev.astype(np.float32),
                                ytr=ytr, yev=yev)
        module.train()
    err = None
    try:
        _pretrain(cell["cls"], cell.get("input_mask", "none"), Xtr, iters, (iters,), cell["seed"], device, on_ckpt)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {**cell, **out, "error": err, "n_eval": int(len(yev)), "shape": list(Xtr.shape), "iters": iters,
            "seconds": round(time.perf_counter() - t0, 1)}


# ================================================================ 通用：按 LPT 分份、续跑
def lpt_parts(cells: Sequence[Mapping], n: int, cost) -> list[list]:
    load = [0.0] * n
    parts: list[list] = [[] for _ in range(n)]
    for c in sorted(cells, key=cost, reverse=True):
        i = int(np.argmin(load))
        parts[i].append(c); load[i] += cost(c)
    return parts


def done_keys(run_dir: Path) -> set:
    out = set()
    for p in sorted(Path(run_dir).glob("*.jsonl")) if Path(run_dir).exists() else []:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("error") is None:
                out.add(r["key"])
    return out


def load_runs(run_dir: Path) -> dict:
    out = {}
    for p in sorted(Path(run_dir).glob("*.jsonl")) if Path(run_dir).exists() else []:
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("error") is None:
                out[r["key"]] = r
    return out
