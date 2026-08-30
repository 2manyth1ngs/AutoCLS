"""MX-1 可行性预测器：16 折 leave-one-dataset-out（协议 results/mx1_feasibility/PROTOCOL.md §4-§6）。

三阶段（铁律 3：逐折落盘 + --resume 幂等）：
    python scripts/16_feasibility_loo.py --fit-folds --resume   # 轻 GPU：每折 TS2Vec(600 iters，
                                                                #   只在 15 个训练数据集 fit，铁律 6 按折)
                                                                #   + 32 任务特征（E/E_mean/META）
    python scripts/16_feasibility_loo.py --train --resume       # CPU：小 MLP × 6 臂 × 5 种子（逐位确定）
    python scripts/16_feasibility_loo.py --analyze              # 判读带 §5 + 稳健性 §6 → analysis.json

折间单变量原则（协议 §4.1）：SetTF init 全局一份（set_encoder_init.pt）、
embed_seed 跨折固定；每折唯一变量 = TS2Vec fit 池。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
_b2 = importlib.import_module("scripts.2b_generate_labels_route_a")
dm = importlib.import_module("scripts.14_destruction_matrix")

from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import (  # noqa: E402
    TS2VecWrapper, TaskSetEncoder, encode_task_windows, fit_shared_ts2vec,
    task_windows,
)

OUTDIR = fl.OUTDIR
LABELS = fl.LABELS                       # ("improve", "inert", "destroy")
FOLDS_DIR = OUTDIR / "folds"
ANALYSIS_PATH = OUTDIR / "analysis.json"
SET_ENC_INIT = OUTDIR / "set_encoder_init.pt"
TS2VEC_OUTPUT_DIMS = 256                 # fit_shared_ts2vec 默认 = SetTF dim_input
N_MODEL_SEEDS = 5
N_ASSIGN_SEEDS = 5
EPOCHS, LR, WD, HIDDEN = 300, 1e-3, 1e-3, (128, 64)   # 协议 §4.2 写死
MIN_FOLD_CELLS = 10                      # 协议 §2：有效 cell <10 的折剔除
FLOOR_N_SIM = 10_000                     # 协议 §5 噪声地板模拟次数
# 协议 §6 R3 / P6：inert 三数据集（破坏率 0 ≠ 安全）
INERT_TRIO = ("HandMovementDirection", "MotorImagery", "SelfRegulationSCP1")
ARMS_FEAT = (("main", "E"), ("emean", "E_mean"), ("meta", "META"))
ALL_ARMS = ("main", "emean", "meta", "clsonly", "main_a1", "main_a2", "emean_a1")


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def fold_split(datasets: list[str]) -> list[tuple[str, list[str]]]:
    """LOO 折：[(留出数据集, 其余数据集)]，排序确定性。"""
    ds = sorted(datasets)
    return [(d, [x for x in ds if x != d]) for d in ds]


def b0_rates(train_cells: list[dict]) -> np.ndarray:
    """任务盲·全局三态率（协议 §4.3：只吃训练折 cell，留出绝不参与）。"""
    counts = np.array([sum(1 for c in train_cells if c["label"] == s)
                       for s in LABELS], dtype=float)
    return counts / counts.sum()


def b1_rates(train_cells: list[dict]) -> dict[str, np.ndarray]:
    """任务盲·per-CLS 三态率（同上，泄漏契约同 b0_rates）。"""
    by_cls: dict = defaultdict(list)
    for c in train_cells:
        by_cls[c["cls_key"]].append(c["label"])
    return {k: np.array([sum(1 for x in v if x == s) for s in LABELS],
                        dtype=float) / len(v)
            for k, v in by_cls.items()}


def observed_rates(cells: list[dict]) -> np.ndarray:
    counts = np.array([sum(1 for c in cells if c["label"] == s)
                       for s in LABELS], dtype=float)
    return counts / counts.sum()


def err_tv(p_hat: np.ndarray, p_obs: np.ndarray) -> float:
    """三态分布总变差（协议 §5 主误差）。"""
    return float(np.abs(np.asarray(p_hat) - np.asarray(p_obs)).sum() / 2.0)


def brier(p_hat: np.ndarray, label: str) -> float:
    """多类 Brier（proper；协议 §5 cell 级次判据）。"""
    onehot = np.zeros(len(LABELS))
    onehot[LABELS.index(label)] = 1.0
    return float(((np.asarray(p_hat) - onehot) ** 2).sum())


def standardize_fit(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐维 mean/σ（只喂训练折任务的特征；σ 下限 1e-6，协议 §4.2）。"""
    mu = E.mean(axis=0)
    sd = np.maximum(E.std(axis=0), 1e-6)
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def class_weights(labels: list[str]) -> np.ndarray:
    """类别权重 w=N/(3·max(count,1))，上限 10（协议 §4.2 写死）。"""
    n = len(labels)
    counts = [max(sum(1 for x in labels if x == s), 1) for s in LABELS]
    return np.minimum(np.array([n / (3 * c) for c in counts]), 10.0)


def cross_assignment_mx1(test_tids: list[str], train_tids: list[str],
                         ds_of, *, seed: int) -> dict[str, str]:
    """A1 cross-E：留出任务 → 其他数据集随机训练任务（4b cross_assignment 同语义）。"""
    rng = np.random.default_rng([seed, 20260830])
    out = {}
    for t in sorted(test_tids):
        cands = [x for x in sorted(train_tids) if ds_of(x) != ds_of(t)]
        assert cands, f"{t} 无跨数据集 donor"
        out[t] = cands[rng.integers(len(cands))]
    return out


def noise_floor(p_obs: np.ndarray, *, n_cells: int, n_sim: int,
                seed: int) -> float:
    """oracle 参照线：真分布已知时 n_cells 格多项抽样的期望 err_TV（协议 §5）。"""
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_cells, p_obs, size=n_sim) / n_cells
    return float(np.abs(draws - np.asarray(p_obs)).sum(axis=1).mean() / 2.0)


def paired_wilcoxon_less(x, y) -> float:
    """单侧配对 Wilcoxon：H1 = x 的中位差 < y。全零差返回 1.0。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.all(x - y == 0):
        return 1.0
    from scipy.stats import wilcoxon
    return float(wilcoxon(x, y, alternative="less").pvalue)


def verdict_mx1(p_d1: float, p_d2: float, p_d1_emean: float) -> str:
    """判读带（协议 §5，先于计算写死）。"""
    if p_d1 < 0.05:
        return "established" if p_d2 < 0.05 else "e_unproven"
    if p_d1 < 0.15 or p_d1_emean < 0.05:
        return "edge"
    return "not_established"


class FeasibilityMLP(torch.nn.Module):
    """协议 §4.2：z ⊕ v_cls → [128, 64] → 3 类 logits。"""

    def __init__(self, in_dim: int, hidden=HIDDEN):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, len(LABELS)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X: np.ndarray, y: np.ndarray, class_w: np.ndarray, *,
              seed: int, epochs: int = EPOCHS) -> FeasibilityMLP:
    """全批 Adam，CPU 逐位确定（协议 §4.2：无 early stopping，不窥视留出折）。"""
    torch.manual_seed(seed)
    model = FeasibilityMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_w, dtype=torch.float32))
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(model(Xt), yt)
        loss.backward()
        opt.step()
    return model.eval()


def predict_proba(model: FeasibilityMLP, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


# ---------------------------------------------------------------- 数据装配
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def load_panel() -> list[dict]:
    if not fl.LABELS_PATH.exists():
        sys.exit(f"缺 {fl.LABELS_PATH}：先跑 scripts/15_feasibility_labels.py")
    return read_jsonl(fl.LABELS_PATH)


def panel_index(rows: list[dict]):
    """→ (datasets, tids_by_ds, classes_by_tid)。"""
    datasets = sorted({r["dataset"] for r in rows})
    tids_by_ds: dict = defaultdict(set)
    classes_by_tid: dict = {}
    for r in rows:
        tids_by_ds[r["dataset"]].add(r["task_id"])
        classes_by_tid[r["task_id"]] = r["classes"]
    return datasets, {d: sorted(s) for d, s in tids_by_ds.items()}, classes_by_tid


def load_datasets_and_tasks(datasets, tids_by_ds, classes_by_tid):
    """逐数据集加载 + 注册表默认档切片，按 labels 的 classes 逐字段校验任务定义。"""
    registry = load_registry()
    targets = set(registry.names(ROLE_TARGET))
    assert not (set(datasets) & targets), \
        f"面板混入目标池 {set(datasets) & targets}（铁律 6）"
    data = {}
    for name in datasets:
        entry = registry.get(name)
        X, y = load_source_dataset(name, registry)
        _, tasks = slice_source_tasks(entry, registry, y)
        tmap = {t.task_id: t for t in tasks}
        sel = []
        for tid in tids_by_ds[name]:
            t = tmap.get(tid)
            assert t is not None, f"当前切片不含 {tid}（任务定义漂移）"
            assert list(t.classes) == list(classes_by_tid[tid]), \
                f"{tid} classes 漂移: 切片 {list(t.classes)} ≠ label {classes_by_tid[tid]}"
            sel.append(t)
        data[name] = (X, y, sel)
        print(f"[{name}] X={X.shape} → {len(sel)} 任务", flush=True)
    return data


def get_set_encoder_init(ra: dict) -> TaskSetEncoder:
    """SetTF 全局一份 init（协议 §4.1 折间单变量；2b get_set_encoder 同语义）。"""
    torch.manual_seed(ra["embed_seed"])
    enc = TaskSetEncoder(dim_input=TS2VEC_OUTPUT_DIMS, nz=ra["nz"])
    if SET_ENC_INIT.exists():
        enc.load_state_dict(torch.load(SET_ENC_INIT, map_location="cpu",
                                       weights_only=True))
    else:
        torch.save(enc.state_dict(), SET_ENC_INIT)
    return enc


def get_fold_ts2vec(fold_dir: Path, heldout: str, train_ds: list[str],
                    data: dict, ra: dict, device: str) -> TS2VecWrapper:
    """折 TS2Vec：只在 15 个训练数据集 fit（铁律 6 按折）；指纹校验防半池漂移。"""
    fit_pool = sorted(train_ds)
    assert heldout not in fit_pool, f"铁律 6 违例：{heldout} 在 fit 池内"
    path, meta_path = fold_dir / "shared_ts2vec.pt", fold_dir / "shared_ts2vec.meta.json"
    if path.exists():
        meta = (json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists() else None)
        if meta is None or meta.get("fit_datasets") != fit_pool:
            sys.exit(f"{path} fit 池 {meta and meta.get('fit_datasets')} ≠ 本折 "
                     f"{fit_pool}：口径漂移，确认后删除该折目录重跑")
        return TS2VecWrapper.load(path, device=device)
    per_ds = [(name, data[name][0], None, None) for name in fit_pool]
    pool = _b2.build_ts2vec_pool(per_ds, ra["ts2vec_max_instances_per_dataset"],
                                 ra["embed_seed"])
    t0 = time.time()
    w = fit_shared_ts2vec(pool, n_iters=ra["ts2vec_iters"], device=device,
                          seed=ra["embed_seed"])
    w.save(path)
    meta_path.write_text(json.dumps({
        "heldout": heldout, "fit_datasets": fit_pool,
        "n_instances": int(pool.shape[0]), "t_max": int(pool.shape[1]),
        "n_iters": ra["ts2vec_iters"], "embed_seed": ra["embed_seed"],
        "max_instances_per_dataset": ra["ts2vec_max_instances_per_dataset"],
        "device_name": (torch.cuda.get_device_name(0)
                        if device.startswith("cuda") else "cpu"),
        "seconds": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{heldout}] TS2Vec fit {pool.shape} {time.time() - t0:.0f}s → {path}",
          flush=True)
    return w


# ---------------------------------------------------------------- 阶段 1：fit-folds
def cmd_fit_folds(device: str, only_folds: list[str] | None) -> None:
    ra = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                        .read_text(encoding="utf-8"))["route_a"]
    rows = load_panel()
    datasets, tids_by_ds, classes_by_tid = panel_index(rows)
    data = load_datasets_and_tasks(datasets, tids_by_ds, classes_by_tid)
    set_enc = get_set_encoder_init(ra)

    for heldout, train_ds in fold_split(datasets):
        if only_folds and heldout not in only_folds:
            continue
        fold_dir = FOLDS_DIR / heldout
        feat_dir = fold_dir / "task_features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        todo = [t for name in datasets for t in data[name][2]
                if not (feat_dir / f"{t.task_id}.npz").exists()]
        if not todo and (fold_dir / "shared_ts2vec.pt").exists():
            print(f"[{heldout}] 已完成（{len(datasets) * 2} 特征齐全）", flush=True)
            continue
        shared = get_fold_ts2vec(fold_dir, heldout, train_ds, data, ra, device)
        enc_dev = set_enc.to(shared.device).eval()
        for name in datasets:
            X, y, tasks = data[name]
            for t in tasks:
                path = feat_dir / f"{t.task_id}.npz"
                if path.exists():
                    continue
                Xt, _ = materialize_task(t, X, y)
                windows = task_windows(Xt, t.task_type)   # 铁律 7 分发点
                reprs = encode_task_windows(
                    windows, shared, sample_num=ra["sample_num"],
                    seed=ra["embed_seed"])
                with torch.no_grad():
                    e = enc_dev(torch.as_tensor(reprs, dtype=torch.float32)
                                .to(shared.device)).cpu().numpy()
                meta_vec = np.array([np.log10(Xt.shape[1]), np.log10(Xt.shape[2]),
                                     float(len(t.classes)), np.log10(Xt.shape[0])],
                                    dtype=np.float32)
                np.savez_compressed(
                    path, E=e.astype(np.float32),
                    E_mean=reprs.mean(axis=(0, 1)).astype(np.float32),
                    META=meta_vec)
        n_feat = len(list(feat_dir.glob("*.npz")))
        print(f"[{heldout}] 特征 {n_feat}/{len(datasets) * 2} 齐", flush=True)
        del shared
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 收官重验（种子重放教训：勿信中途计数）
    n_ok = sum(1 for d, _ in fold_split(datasets)
               if (FOLDS_DIR / d / "shared_ts2vec.pt").exists()
               and len(list((FOLDS_DIR / d / "task_features").glob("*.npz")))
               == len(datasets) * 2)
    print(f"\nfit-folds 收官：{n_ok}/{len(datasets)} 折齐全", flush=True)


# ---------------------------------------------------------------- 阶段 2：train
def _fold_predict(heldout: str, train_cells: list[dict], test_cells: list[dict],
                  feats: dict, vcls: dict) -> dict[str, dict]:
    """单折全部臂 → {arm: {(tid, key): p 向量}}。种子按 (折, 臂) 派生。"""
    train_tids = sorted({c["task_id"] for c in train_cells})
    test_tids = sorted({c["task_id"] for c in test_cells})
    ytr = np.array([LABELS.index(c["label"]) for c in train_cells])
    w = class_weights([c["label"] for c in train_cells])
    out: dict = {}

    def build_X(cells, z):
        return np.stack([np.concatenate([z[c["task_id"]], vcls[c["cls_key"]]])
                         for c in cells])

    for arm, featkey in ARMS_FEAT:
        Etr = np.stack([feats[t][featkey] for t in train_tids])
        mu, sd = standardize_fit(Etr)     # 只喂训练折（协议 §4.2）
        z = {t: standardize_apply(feats[t][featkey], mu, sd)
             for t in list(train_tids) + list(test_tids)}
        models = [train_mlp(build_X(train_cells, z), ytr, w,
                            seed=run_seed_for(f"mx1:{heldout}:{arm}", f"m{s}"))
                  for s in range(N_MODEL_SEEDS)]

        def avg_pred(X):
            return np.mean([predict_proba(m, X) for m in models], axis=0)

        Xte = build_X(test_cells, z)
        out[arm] = {(c["task_id"], c["cls_key"]): p
                    for c, p in zip(test_cells, avg_pred(Xte))}

        if arm in ("main", "emean"):
            # A1 cross-E（评估时换 donor E；5 assignment 种子均值）
            acc = np.zeros((len(test_cells), len(LABELS)))
            for a in range(N_ASSIGN_SEEDS):
                mapping = cross_assignment_mx1(
                    test_tids, train_tids, fl.dataset_of,
                    seed=run_seed_for(f"mx1:{heldout}:{arm}", f"assign{a}"))
                za = dict(z)
                za.update({t: z[mapping[t]] for t in test_tids})
                acc += avg_pred(build_X(test_cells, za))
            out[f"{arm}_a1"] = {(c["task_id"], c["cls_key"]): p
                                for c, p in zip(test_cells, acc / N_ASSIGN_SEEDS)}
        if arm == "main":
            # A2 mean-E（标准化空间零向量）
            z0 = dict(z)
            z0.update({t: np.zeros_like(z[t]) for t in test_tids})
            out["main_a2"] = {(c["task_id"], c["cls_key"]): p
                              for c, p in zip(test_cells, avg_pred(build_X(test_cells, z0)))}

    # A3 cls-only 重训（E 通道整个去掉）
    Xtr = np.stack([vcls[c["cls_key"]] for c in train_cells])
    models = [train_mlp(Xtr, ytr, w,
                        seed=run_seed_for(f"mx1:{heldout}:clsonly", f"m{s}"))
              for s in range(N_MODEL_SEEDS)]
    Xte = np.stack([vcls[c["cls_key"]] for c in test_cells])
    pte = np.mean([predict_proba(m, Xte) for m in models], axis=0)
    out["clsonly"] = {(c["task_id"], c["cls_key"]): p
                      for c, p in zip(test_cells, pte)}
    return out


def cmd_train(resume: bool) -> None:
    rows = load_panel()
    datasets, tids_by_ds, _classes = panel_index(rows)
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space)
    vcls = {key: enc.encode(c)
            for c, key in load_probe_cls(_b2.PROBE_CATALOG, 10)[0]}

    for heldout, _train_ds in fold_split(datasets):
        pred_path = FOLDS_DIR / heldout / "predictions.jsonl"
        test_cells = [r for r in rows if r["dataset"] == heldout]
        if len(test_cells) < MIN_FOLD_CELLS:
            print(f"[{heldout}] cell {len(test_cells)} < {MIN_FOLD_CELLS}，折剔除",
                  flush=True)
            continue
        expected = len(ALL_ARMS) * len(test_cells)
        if resume and pred_path.exists() and len(read_jsonl(pred_path)) == expected:
            print(f"[{heldout}] predictions 齐全，跳过", flush=True)
            continue
        feat_dir = FOLDS_DIR / heldout / "task_features"
        feats = {}
        for ds in datasets:
            for tid in tids_by_ds[ds]:
                p = feat_dir / f"{tid}.npz"
                assert p.exists(), f"缺特征 {p}：先跑 --fit-folds"
                with np.load(p) as npz:
                    feats[tid] = {k: npz[k] for k in ("E", "E_mean", "META")}
        train_cells = [r for r in rows if r["dataset"] != heldout]
        t0 = time.time()
        arms = _fold_predict(heldout, train_cells, test_cells, feats, vcls)
        label_of = {(c["task_id"], c["cls_key"]): c["label"] for c in test_cells}
        tmp = pred_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for arm in ALL_ARMS:
                for (tid, key), p in sorted(arms[arm].items()):
                    fh.write(json.dumps({
                        "fold": heldout, "arm": arm, "task_id": tid,
                        "cls_key": key, "p": [round(float(x), 6) for x in p],
                        "label": label_of[(tid, key)],
                    }, ensure_ascii=False) + "\n")
        tmp.replace(pred_path)   # 原子替换：断点只会整折重算，不留半折
        print(f"[{heldout}] {len(ALL_ARMS)} 臂 × {len(test_cells)} cell "
              f"({time.time() - t0:.0f}s) → {pred_path}", flush=True)


# ---------------------------------------------------------------- 阶段 3：analyze
def cmd_analyze() -> None:
    rows = load_panel()
    datasets, _tids, _classes = panel_index(rows)
    global_destroy = observed_rates(rows)[LABELS.index("destroy")]

    per_fold: dict = {}
    for heldout, _train in fold_split(datasets):
        pred_path = FOLDS_DIR / heldout / "predictions.jsonl"
        test_cells = [r for r in rows if r["dataset"] == heldout]
        if len(test_cells) < MIN_FOLD_CELLS or not pred_path.exists():
            per_fold[heldout] = {"excluded": True,
                                 "reason": "insufficient_cells"
                                 if len(test_cells) < MIN_FOLD_CELLS
                                 else "missing_predictions"}
            continue
        preds = read_jsonl(pred_path)
        by_arm: dict = defaultdict(dict)
        for r in preds:
            by_arm[r["arm"]][(r["task_id"], r["cls_key"])] = np.array(r["p"])
        train_cells = [r for r in rows if r["dataset"] != heldout]
        p_obs = observed_rates(test_cells)
        b0 = b0_rates(train_cells)
        b1 = b1_rates(train_cells)
        keys = [(c["task_id"], c["cls_key"]) for c in test_cells]
        entry = {
            "excluded": False, "n_test_cells": len(test_cells),
            "obs_rates": [round(float(x), 4) for x in p_obs],
            "floor": noise_floor(p_obs, n_cells=len(test_cells),
                                 n_sim=FLOOR_N_SIM,
                                 seed=run_seed_for(heldout, "mx1_floor")),
            "err_b0": err_tv(b0, p_obs),
            "err": {}, "pred_rates": {},
        }
        for arm in ALL_ARMS:
            p_ds = np.mean([by_arm[arm][k] for k in keys], axis=0)
            entry["err"][arm] = err_tv(p_ds, p_obs)
            entry["pred_rates"][arm] = [round(float(x), 4) for x in p_ds]
        entry["brier_main"] = float(np.mean(
            [brier(by_arm["main"][k], c["label"])
             for k, c in zip(keys, test_cells)]))
        entry["brier_b1"] = float(np.mean(
            [brier(b1[c["cls_key"]], c["label"]) for c in test_cells]))
        per_fold[heldout] = entry

    active = [d for d in sorted(per_fold) if not per_fold[d]["excluded"]]
    arr = lambda key: np.array([per_fold[d][key] for d in active])  # noqa: E731
    earr = lambda arm: np.array([per_fold[d]["err"][arm] for d in active])  # noqa: E731

    tests = {
        "n_active_folds": len(active),
        "D1_p": paired_wilcoxon_less(earr("main"), arr("err_b0")),
        "D2_p": paired_wilcoxon_less(earr("main"), earr("main_a1")),
        "S1_p": paired_wilcoxon_less(arr("brier_main"), arr("brier_b1")),
        "B2_D1_p": paired_wilcoxon_less(earr("meta"), arr("err_b0")),
        "R2_D1_p": paired_wilcoxon_less(earr("emean"), arr("err_b0")),
        "R2_D2_p": paired_wilcoxon_less(earr("emean"), earr("emean_a1")),
        "A2_err_median": float(np.median(earr("main_a2"))),
        "D1_win_folds": int((earr("main") < arr("err_b0")).sum()),
        "err_median": {arm: float(np.median(earr(arm))) for arm in ALL_ARMS},
        "err_b0_median": float(np.median(arr("err_b0"))),
        "floor_median": float(np.median(arr("floor"))),
    }
    # R1：剔 k1 骨干（labels 的 r_source）后 13 折重算 D1
    k1_ds = {r["dataset"] for r in rows if r["r_source"] == "k1"}
    k3_active = [d for d in active if d not in k1_ds]
    idx = [active.index(d) for d in k3_active]
    tests["R1_folds"] = k3_active
    tests["R1_D1_p"] = paired_wilcoxon_less(earr("main")[idx], arr("err_b0")[idx])
    # P4：预测破坏率 vs 观测破坏率 Spearman
    di = LABELS.index("destroy")
    pred_destroy = [per_fold[d]["pred_rates"]["main"][di] for d in active]
    obs_destroy = [per_fold[d]["obs_rates"][di] for d in active]
    tests["P4_spearman"] = dm.spearman(pred_destroy, obs_destroy)
    # P6：inert 三折应惰性主导
    p6 = {}
    for d in INERT_TRIO:
        if d in active:
            pr = per_fold[d]["pred_rates"]["main"]
            p6[d] = bool(pr[LABELS.index("inert")] > pr[di]
                         and pr[LABELS.index("inert")] > pr[LABELS.index("improve")])
    tests["P6_inert_dominant"] = p6
    # D2 逐折功效异质性诊断（协议 §5：单折读数不作消融失败解读）
    delta = earr("main_a1") - earr("main")
    extremeness = np.abs(np.array(obs_destroy) - global_destroy)
    tests["D2_heterogeneity_spearman"] = dm.spearman(list(delta), list(extremeness))
    # clsonly 一致性自检：数据集级预测应近似常数
    cls_ds = np.array([per_fold[d]["pred_rates"]["clsonly"] for d in active])
    tests["clsonly_ds_pred_std_max"] = float(cls_ds.std(axis=0).max())

    verdict = verdict_mx1(tests["D1_p"], tests["D2_p"], tests["R2_D1_p"])
    out = {"protocol": "results/mx1_feasibility/PROTOCOL.md",
           "per_fold": per_fold, "tests": tests, "verdict": verdict}
    ANALYSIS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                        default=float), encoding="utf-8")

    # ---- 控制台报表 ----
    print(f"\n===== MX-1 LOO（{len(active)} 折）=====")
    print(f"{'留出数据集':<28}{'观测(提/惰/破)':>16}{'地板':>7}{'B0':>7}"
          f"{'模型':>7}{'A1':>7}{'meta':>7}{'clsonly':>8}")
    for d in active:
        e = per_fold[d]
        obs = "/".join(f"{v:.2f}" for v in e["obs_rates"])
        print(f"{d:<28}{obs:>16}{e['floor']:>7.3f}{e['err_b0']:>7.3f}"
              f"{e['err']['main']:>7.3f}{e['err']['main_a1']:>7.3f}"
              f"{e['err']['meta']:>7.3f}{e['err']['clsonly']:>8.3f}")
    t = tests
    print(f"\nD1(模型<B0) p={t['D1_p']:.4f}（{t['D1_win_folds']}/{len(active)} 折胜）"
          f" | D2(模型<A1) p={t['D2_p']:.4f} | S1(Brier<B1) p={t['S1_p']:.4f}")
    print(f"R1(剔 k1，{len(k3_active)} 折) D1 p={t['R1_D1_p']:.4f} | "
          f"R2(E_mean) D1 p={t['R2_D1_p']:.4f} D2 p={t['R2_D2_p']:.4f} | "
          f"B2(元数据) D1 p={t['B2_D1_p']:.4f}")
    print(f"P4 Spearman(预测破坏率,观测)={t['P4_spearman'][0]:.3f} "
          f"(p={t['P4_spearman'][1]:.3f}) | P6 inert 主导: {t['P6_inert_dominant']}")
    print(f"D2 异质性 ρ={t['D2_heterogeneity_spearman'][0]:.3f} | "
          f"clsonly 数据集级 std_max={t['clsonly_ds_pred_std_max']:.4f}")
    print(f"\n[判读] → {verdict}")
    print(f"→ {ANALYSIS_PATH}", flush=True)


# ---------------------------------------------------------------- 入口
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-folds", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--folds", nargs="+", default=None,
                    help="只处理指定留出数据集（分片/重跑）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.fit_folds:
        cmd_fit_folds(args.device, args.folds)
    if args.train:
        cmd_train(args.resume)
    if args.analyze:
        cmd_analyze()
    if not (args.fit_folds or args.train or args.analyze):
        ap.error("至少指定 --fit-folds / --train / --analyze 之一")


if __name__ == "__main__":
    main()
