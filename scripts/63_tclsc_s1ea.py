"""WP16-TCLSC 阶段 1 响应画像 E（A 臂）：不学习的机制特征 E_A 作任务嵌入，UCR 84 分组 7 折（协议 results/tclsc_s1ea/PROTOCOL.md）。

零新 R'：语料 / 折 / δ / 训练器全部复用 60 号（`results/tclsc_s1ucr/`）；E_A 由 `autocls.data.mech_features.response_profile`
在每个 UCR 数据集的**训练划分**上算（与 42 号 φ 同划分，铁律 6），落 `ea_ucr.jsonl`。任务盲 / φ12 参照臂不重训——
从 s1ucr 的 `rankings_ucr.json` 取同折同种子的排名合并进本实验的排名表重评。

子命令：--features / --train（脱离启动 + --resume）/ --rank / --analyze。
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
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import TCLSCEngine  # noqa: E402
from autocls.data.mech_features import RESPONSE_PROFILE_COLS, RESPONSE_SEEDS, rep_cols_at, response_profile  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g59 = importlib.import_module("scripts.59_tclsc_s1fp")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
ArmSpec = _m33.ArmSpec
PHI_COLS = tuple(_m33.PHI_COLS)
BUDGETS = pq.BUDGETS
DELTA_UCR = _g60.DELTA_UCR
EA_COLS: tuple[str, ...] = tuple(RESPONSE_PROFILE_COLS)
OUT = ROOT / "results" / "tclsc_s1ea"
S1UCR = ROOT / "results" / "tclsc_s1ucr"
EA_FILE = OUT / "ea_ucr.jsonl"
ET_FILE = ROOT / "results" / "tclsc_s1et" / "et_ucr.jsonl"        # T 检验：转导型 E_T（69 号 --features 产出）
REF_ARMS = ("u_blind_diff_hard", "u_phi_diff_hard")
SEEDS = (0, 1, 2)
CROSS_SEEDS = (0, 1, 2)
ARMS: dict[str, ArmSpec] = {
    s.name: s for s in (
        ArmSpec("u_ea_diff_hard", "S1", "ctx", None, None, "ea_budget", head="diff", seeds=(0, 1, 2)),
        ArmSpec("u_ea_concat_hard", "S1", "ctx", None, None, "ea_budget", head="concat", seeds=(0, 1, 2)),
        ArmSpec("u_eaphi_diff_hard", "S1", "ctx", None, None, "eaphi_budget", head="diff", seeds=(0, 1, 2)),
        # R 检验（results/tclsc_s1ea_rank/PROTOCOL.md §3）：低秩字段加性头
        ArmSpec("u_fw4_eaphi_hard", "S1", "ctx", None, None, "eaphi_budget", head="field", seeds=(0, 1, 2), head_rank=4),
        ArmSpec("u_fw_eaphi_hard", "S1", "ctx", None, None, "eaphi_budget", head="field", seeds=(0, 1, 2)),
        # T 检验（results/tclsc_s1et/PROTOCOL.md §3）：转导型 E_T
        ArmSpec("u_ft60_hard", "S1", "ctx", None, None, "et60_budget", head="field", seeds=(0, 1, 2)),
        ArmSpec("u_ft290_hard", "S1", "ctx", None, None, "et290_budget", head="field", seeds=(0, 1, 2)),
        ArmSpec("u_ftea_hard", "S1", "ctx", None, None, "etea_budget", head="field", seeds=(0, 1, 2)),
    )
}
CTX_KINDS: dict[str, tuple[str, ...]] = {"ea_budget": EA_COLS, "eaphi_budget": EA_COLS + PHI_COLS,
                                        "et60_budget": rep_cols_at(60) + PHI_COLS, "et290_budget": rep_cols_at(290) + PHI_COLS,
                                        "etea_budget": rep_cols_at(60) + EA_COLS + PHI_COLS}
# G-A4：AutoCL 随机增强字段 ↔ 响应画像列
FIELD_OPS = {"jitter": "ret_jitter", "rescale": "ret_rescale", "point_mask": "ret_point_mask", "freq_mask": "ret_freq_mask"}
FIELD_HI, FIELD_LO, FIELD_MIN_SIDE, ALIGN_BUDGET = 0.5, 0.2, 2, 290
# 门（协议 §4）
M1_MIN, ALPHA = 0.03, 0.05
PRIMARY = "acc_top10"
C0_TOL = 0.001
ALIGN_RHO, ALIGN_P = 0.3, 0.05


def ctx_cols(kind: str) -> tuple[str, ...]:
    if kind not in CTX_KINDS:
        raise KeyError(f"未知 ctx_kind {kind!r}（可选 {sorted(CTX_KINDS)}）")
    return CTX_KINDS[kind]


# ---------------------------------------------------------------- 表
def load_ea(path: Path = EA_FILE) -> dict[str, dict]:
    return {r["dataset"]: r for r in read_jsonl(path)}


def merge_tables(ea: Mapping[str, Mapping], phi: Mapping[str, Mapping]) -> dict[str, dict]:
    """{数据集: E_A 行 ⊕ φ 行}（φ 缺的数据集只带 E_A；列缺失在标准化时按 nan 处理）。"""
    return {d: {**dict(row), **dict(phi.get(d, {}))} for d, row in ea.items()}


def _val(row: Mapping, c: str) -> float:
    v = row.get(c)
    return float("nan") if v is None else float(v)


def table_standardizer(table: Mapping[str, Mapping], keys: Sequence[str], cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """逐列 nan 感知 (μ, σ)，只用给定数据集（训练折，铁律 6）；σ 下限同 φ（PHI_SD_FLOOR）。"""
    X = np.array([[_val(table[k], c) for c in cols] for k in keys], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd), sd, 0.0)
    return mu, np.maximum(sd, _m33.PHI_SD_FLOOR)


def _std_row(row: Mapping, cols: Sequence[str], mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.array([_val(row, c) for c in cols], dtype=np.float64)
    nan = ~np.isfinite(x)
    z = (np.where(nan, mu, x) - mu) / sd           # nan → 训练均值 → 标准化后 0
    return z, int(nan.sum())


def table_context(table: Mapping[str, Mapping], cols: Sequence[str], train: Sequence[str], test: Sequence[str], *,
                  ctx_kind: str, budgets: Sequence[int] = BUDGETS) -> tuple[dict[str, np.ndarray], dict]:
    """{engine tid: 标准化特征 ⊕ 预算 one-hot}（训练 + 考试数据集），μ/σ 只用训练集。"""
    mu, sd = table_standardizer(table, list(train), cols)
    ctx: dict[str, np.ndarray] = {}
    n_nan: dict[str, int] = {}
    for d in sorted(set(train) | set(test)):
        z, k = _std_row(table[d], cols, mu, sd)
        if k:
            n_nan[d] = k
        for b in budgets:
            ctx[pq.engine_task_id(f"{d}_00", int(b))] = np.concatenate([z, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    meta = {"ctx_kind": ctx_kind, "ctx_cols": list(cols), "ctx_mu": [float(v) for v in mu], "ctx_sd": [float(v) for v in sd], "n_nan": n_nan}
    return ctx, meta


def table_ctx_rows(table: Mapping[str, Mapping], cols: Sequence[str], slices: Sequence[str], mu: np.ndarray, sd: np.ndarray, *,
                   cross_map: Optional[Mapping[str, str]] = None, zero: bool = False,
                   budgets: Sequence[int] = BUDGETS) -> dict[tuple[str, int], np.ndarray]:
    """排名侧 ctx 表 {(切片, 预算): ctx}：供体（cross）/ 置零（zero）/ 自己；与 58 号 phi_ctx_rows 同形。"""
    mu, sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
    out = {}
    for s in slices:
        src = cross_map[s] if cross_map is not None else s
        z = np.zeros(len(cols), dtype=np.float64) if zero else _std_row(table[pq.dataset_of(src)], cols, mu, sd)[0]
        for b in budgets:
            out[(s, int(b))] = np.concatenate([z, pq.ctx_vector(int(b), budgets)]).astype(np.float32)
    return out


def table_for_kind(kind: str, ea: Mapping, phi: Mapping, et: Optional[Mapping] = None) -> Mapping[str, Mapping]:
    """按 ctx_kind 合成数据集表：ea / ea ⊕ φ / E_T ⊕ φ (⊕ E_A)。et 缺席时 et* 种类 fail-fast。"""
    if kind == "ea_budget":
        return ea
    if kind == "eaphi_budget":
        return merge_tables(ea, phi)
    if kind in ("et60_budget", "et290_budget", "etea_budget"):
        if et is None:
            raise ValueError(f"ctx_kind={kind!r} 需要转导型 E_T 表（{ET_FILE}，先跑 69 --features）")
        return {d: {**dict(row), **dict(et.get(d, {})), **dict(phi.get(d, {}))} for d, row in ea.items()}
    raise KeyError(f"未知 ctx_kind {kind!r}")


def load_et(path: Path = ET_FILE) -> Optional[dict[str, dict]]:
    return {r["dataset"]: r for r in read_jsonl(path)} if path.exists() else None


def _ctx_table(spec: ArmSpec, ea: Mapping, phi: Mapping, et: Optional[Mapping] = None) -> tuple[Mapping[str, Mapping], tuple[str, ...]]:
    return table_for_kind(spec.ctx_kind, ea, phi, et), ctx_cols(spec.ctx_kind)


# ---------------------------------------------------------------- 特征计算（--features）
def compute_ea_rows(tasks: Mapping[str, Mapping], *, done: set[str], seeds: Sequence[int] = RESPONSE_SEEDS):
    """逐数据集产 E_A 行（生成器）：41 号加载 → 任务物化 → 42 号同划分的训练侧 → response_profile。"""
    from autocls.data.task_slicer import materialize_task
    from autocls.labeling import run_seed_for, split_task_train_eval
    _g41 = importlib.import_module("scripts.41_ucr_rprime")
    reg = _g41.registry()
    for tid, spec in tasks.items():
        if tid in done:
            continue
        X, y = _g41.load_dataset(spec["dataset"], reg)
        t = _g41.task_from_json(spec)
        Xt, yt = materialize_task(t, X, y)
        tr, _ = split_task_train_eval(yt, t.task_type, eval_ratio=0.3, seed=run_seed_for(tid, "train_eval_split"))
        t0 = time.time()
        prof = response_profile(Xt[tr], yt[tr], task_type=t.task_type, seeds=seeds)
        yield {"task_id": tid, "dataset": spec["dataset"], "n_train": int(len(tr)), "seconds": round(time.time() - t0, 1), **prof}


def cmd_features(outdir: Path, *, resume: bool) -> None:
    tasks = json.loads((ROOT / "results" / "ucr_rprime" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    path = outdir / "ea_ucr.jsonl"
    done = {r["task_id"] for r in read_jsonl(path)} if (resume and path.exists()) else set()
    if not resume and path.exists():
        path.unlink()
    with open(path, "a", encoding="utf-8") as fh:
        for row in compute_ea_rows(tasks, done=done):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[ea] {row['dataset']}: n={row['n_train']} {row['seconds']}s " + " ".join(
                f"{c}={row[c]:+.2f}" for c in ("ret_jitter", "ret_highpass05", "ret_inst_norm", "hardness_1nn")), flush=True)


# ---------------------------------------------------------------- 训练 / 排名
def train_fold(rows: Sequence[dict], ea: Mapping, phi: Mapping, folds: Sequence[Sequence[str]], idx: int, *, arm: str, seed: int,
               outdir: Path, device: str, epochs: Optional[int] = None) -> dict:
    spec = ARMS[arm]
    test = list(folds[idx])
    train = sorted({r["dataset"] for r in rows} - set(test))
    c = dict(_g60.cfg())
    if epochs:
        c["epochs"], c["curriculum_epochs"] = int(epochs), min(int(c["curriculum_epochs"]), max(1, int(epochs) // 2))
    recs = _g60.build_records(rows, train)
    table, cols = _ctx_table(spec, ea, phi, load_et() if spec.ctx_kind.startswith("et") else None)
    ctx_map, meta_extra = table_context(table, cols, train, test, ctx_kind=spec.ctx_kind)
    cdim = int(next(iter(ctx_map.values())).shape[0])
    cls_dim = int(recs[0].vec.shape[0])
    model = _m33.build_model(spec, cls_dim=cls_dim, ctx_dim=cdim, nz=256, dim_input=256, cfg=c, seed=seed)
    train_ctx = {t: ctx_map[t] for t in sorted({r.task_id for r in recs})}
    engine = TCLSCEngine(model, {}, device=device, lr=c["lr"], batch_size=c["batch_size"], seed=seed, context_of=train_ctx)
    t0 = time.time()
    hist = engine.train(recs, epochs=c["epochs"], curriculum_epochs=c["curriculum_epochs"],
                        better_fn=lambda a, b: is_better(a, b, TASK_CLASSIFICATION), delta=DELTA_UCR)
    secs = round(time.time() - t0, 1)
    fold = f"fold{idx}"
    _m33.save_model(outdir / "models" / fold / f"{arm}_s{seed}.pt", model, spec, fold=fold, seed=seed, cls_dim=cls_dim, ctx_dim=cdim,
                    cfg=c, nz=256, settf_windows=0, settf_t_max=0, budgets=BUDGETS, ctx_encoding="onehot", encoder_mode="binned",
                    head=spec.head, ctx_kind=spec.ctx_kind, phi_stats=meta_extra)
    run = {"fold": fold, "arm": arm, "seed": seed, "train_datasets": train, "test_datasets": test, "n_train_records": len(recs),
           "delta": DELTA_UCR, "train_seconds": secs, "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"],
           "train_pairs_last": hist[-1]["n_pairs"], "settings": {"head": spec.head, "ctx_kind": spec.ctx_kind, "ctx_dim": cdim,
                                                                 "epochs": c["epochs"], "encoder_mode": "binned"},
           "ctx_meta": {k: v for k, v in meta_extra.items() if k != "ctx_cols"}}
    p = outdir / "runs" / fold / f"{arm}_s{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


def rank_fold(rows: Sequence[dict], ea: Mapping, phi: Mapping, folds: Sequence[Sequence[str]], idx: int, *, arms: Sequence[str],
              seeds: Sequence[int], outdir: Path, device: str = "cpu") -> dict:
    fold = f"fold{idx}"
    test = list(folds[idx])
    train = sorted({r["dataset"] for r in rows} - set(test))
    test_t = [f"{d}_00" for d in sorted(test)]
    train_t = [f"{d}_00" for d in train]
    cat: dict = {}
    for d in test:
        cat.update(_g60.catalog_for(rows, d))
    models = {}
    for a in arms:
        paths = [outdir / "models" / fold / f"{a}_s{s}.pt" for s in seeds]
        if all(p.exists() for p in paths):
            models[a] = [_g34.load_pq_model(p, map_location=device, expect_fold=fold, expect_budgets=BUDGETS) for p in paths]
        else:
            print(f"⚠️ [{fold}/{a}] 模型不全，跳过", flush=True)
    per: dict = {"normal": {}, "cross": {}, "zero": {}, "test": test}

    def merge(dst: dict, rk: dict) -> None:
        for b, pb in rk.items():
            for s, pa in pb.items():
                dst.setdefault(b, {}).setdefault(s, {}).update(pa)

    et = None
    for kind in CTX_KINDS:
        mm = {a: m for a, m in models.items() if ARMS[a].ctx_kind == kind}
        if not mm:
            continue
        meta0 = next(iter(mm.values()))[0][1]
        cols = tuple(meta0["ctx_cols"])
        if cols != ctx_cols(kind):
            raise ValueError(f"[{fold}/{kind}] 存档列表 {len(cols)} 列 ≠ 当前 {len(ctx_cols(kind))} 列")
        if kind.startswith("et") and et is None:
            et = load_et()
        table = table_for_kind(kind, ea, phi, et)
        mu, sd = np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"])
        rows_fn = lambda cm=None, zero=False, table=table, cols=cols, mu=mu, sd=sd: table_ctx_rows(table, cols, test_t, mu, sd, cross_map=cm, zero=zero)
        merge(per["normal"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn()))
        merge(per["zero"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(zero=True)))
        for cs in CROSS_SEEDS:
            cmap = _g58.cross_slice_map(test_t, train_t, cs)
            per["cross"].setdefault(str(cs), {"map": cmap, "rankings": {}})
            merge(per["cross"][str(cs)]["rankings"], _g56.rank_local_models(mm, cat, budgets=BUDGETS, device=device, ctx_rows=rows_fn(cm=cmap)))
    return per


def merge_ref_rankings(own: Mapping[str, Mapping], ref: Mapping[str, Mapping], *, arms: Sequence[str] = REF_ARMS) -> dict:
    """把 s1ucr 参照臂的 normal 排名（同折同种子已训模型）并进本实验的排名表；折靶必须逐位相同。"""
    out = {f: {"normal": {b: {s: dict(pa) for s, pa in pb.items()} for b, pb in per["normal"].items()}, "cross": per["cross"],
               "zero": per["zero"], "test": list(per["test"])} for f, per in own.items()}
    for f, per in out.items():
        if f not in ref:
            raise ValueError(f"参照排名缺折 {f}")
        if list(ref[f]["test"]) != list(per["test"]):
            raise ValueError(f"折 {f} 靶不一致：本实验 {per['test']} vs 参照 {ref[f]['test']}")
        for b, pb in ref[f]["normal"].items():
            for s, pa in pb.items():
                keep = {a: pa[a] for a in arms if a in pa}
                if keep:
                    per["normal"].setdefault(b, {}).setdefault(s, {}).update(keep)
    return out


# ---------------------------------------------------------------- G-A4 机制对齐
def field_preference(cat_z: Sequence[dict], field: str, *, budget: int = ALIGN_BUDGET, hi: float = FIELD_HI, lo: float = FIELD_LO,
                     min_side: int = FIELD_MIN_SIDE) -> dict[str, float]:
    """{数据集: u_f}：@budget 目录里 field ≥ hi 档候选的 z 均值 − field ≤ lo 档候选的 z 均值；两侧各 ≥ min_side 条才定义。"""
    by_ds: dict[str, tuple[list, list]] = {}
    for r in cat_z:
        if int(r["iters"]) != int(budget):
            continue
        v = float(r["cls"][field])
        hi_l, lo_l = by_ds.setdefault(r["dataset"], ([], []))
        if v >= hi:
            hi_l.append(float(r["z"]))
        elif v <= lo:
            lo_l.append(float(r["z"]))
    return {d: float(np.mean(h) - np.mean(lo)) for d, (h, lo) in sorted(by_ds.items()) if len(h) >= min_side and len(lo) >= min_side}


def field_alignment(table: Mapping[str, Mapping], cat_z: Sequence[dict], *, fields: Sequence[str] = tuple(FIELD_OPS),
                    budget: int = ALIGN_BUDGET) -> dict[str, dict]:
    out = {}
    for f in fields:
        col = FIELD_OPS[f]
        u = field_preference(cat_z, f, budget=budget)
        ds = [d for d in u if d in table and np.isfinite(_val(table[d], col))]
        if len(ds) < 3:
            out[f] = {"col": col, "n": len(ds), "rho": None, "p": None, "aligned": False}
            continue
        x, y = [_val(table[d], col) for d in ds], [u[d] for d in ds]
        r = spearmanr(x, y)
        rho, p = float(r.correlation), float(r.pvalue)
        out[f] = {"col": col, "n": len(ds), "rho": rho, "p": p, "aligned": bool(np.isfinite(rho) and abs(rho) >= ALIGN_RHO and p < ALIGN_P)}
    return out


# ---------------------------------------------------------------- 门 / 读数码
def gates_ea(deltas: Mapping, v_blind: Mapping, v_phi: Mapping) -> dict:
    """deltas[臂][指标] = {mean_delta, p}（cross 消融）；v_blind = blind → u_ea_diff_hard 的 regret@3 配对；v_phi = φ → eaphi。"""
    pos = lambda arm: _g57._positive(deltas.get(arm, {}).get(PRIMARY, {}), min_delta=M1_MIN)
    lower = lambda v: bool(v.get("p") is not None and v["p"] < ALPHA and v["mean_b"] < v["mean_a"])
    return {"GA1": {"arm": "u_ea_diff_hard", "delta": deltas.get("u_ea_diff_hard", {}).get(PRIMARY), "pass": pos("u_ea_diff_hard")},
            "GA2": {"metric": "regret3", "blind": v_blind.get("mean_a"), "ea": v_blind.get("mean_b"), "p": v_blind.get("p"), "pass": lower(v_blind)},
            "GA3": {"metric": "regret3", "phi": v_phi.get("mean_a"), "eaphi": v_phi.get("mean_b"), "p": v_phi.get("p"), "pass": lower(v_phi)},
            "GA5": {"arm": "u_ea_concat_hard", "delta": deltas.get("u_ea_concat_hard", {}).get(PRIMARY), "pass": not pos("u_ea_concat_hard")},
            "thresholds": {"metric": PRIMARY, "min_delta": M1_MIN, "alpha": ALPHA}}


def verdict_ea(g: Mapping) -> str:
    if not g["GA1"]["pass"]:
        return "EA-DEAD"
    if not g["GA2"]["pass"]:
        return "EA-USED-NO-GAIN"
    return "EA-ALIVE" if g["GA3"]["pass"] else "EA-PHI-EQUIV"


# ---------------------------------------------------------------- 子命令
def _load_all() -> tuple[list[dict], dict, dict]:
    return _g60.load_rows(), load_ea(), _g60.load_phi()


def _folds(outdir: Path) -> list[list[str]]:
    src = S1UCR / "folds.json"
    folds = json.loads(src.read_text(encoding="utf-8"))
    (outdir / "folds.json").write_text(json.dumps(folds, ensure_ascii=False, indent=1), encoding="utf-8")
    return folds


def cmd_train(outdir: Path, *, arms: Sequence[str], seeds: Sequence[int], folds_sel: Optional[Sequence[int]], device: str,
              epochs: Optional[int], resume: bool) -> None:
    rows, ea, phi = _load_all()
    folds = _folds(outdir)
    missing = sorted({r["dataset"] for r in rows} - set(ea))
    if missing:
        raise RuntimeError(f"ea_ucr.jsonl 缺 {len(missing)} 个数据集：{missing[:5]}…（先跑 --features）")
    log = outdir / "train.log"
    for idx in (folds_sel if folds_sel else range(len(folds))):
        for arm in arms:
            for seed in seeds:
                p = outdir / "models" / f"fold{idx}" / f"{arm}_s{seed}.pt"
                if resume and p.exists():
                    continue
                run = train_fold(rows, ea, phi, folds, idx, arm=arm, seed=seed, outdir=outdir, device=device, epochs=epochs)
                line = f"[fold{idx}/{arm}_s{seed}] {run['train_seconds']}s | bce {run['bce_first']:.4f}→{run['bce_last']:.4f} | pairs {run['train_pairs_last']}"
                print(line, flush=True)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")


def cmd_rank(outdir: Path, *, seeds: Sequence[int]) -> None:
    rows, ea, phi = _load_all()
    folds = json.loads((outdir / "folds.json").read_text(encoding="utf-8"))
    out = {}
    for idx in range(len(folds)):
        out[f"fold{idx}"] = rank_fold(rows, ea, phi, folds, idx, arms=list(ARMS), seeds=seeds, outdir=outdir)
        print(f"[fold{idx}] 靶 {folds[idx]}", flush=True)
    (outdir / "rankings_ea.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def _eval(rk_by_fold: Mapping, cat_z: Sequence[dict], delta: Mapping, arms: Sequence[str], which: str, *,
          cross_seed: Optional[str] = None, top_k: int) -> list[dict]:
    rows = []
    for fold, per in rk_by_fold.items():
        rk = per["cross"][cross_seed]["rankings"] if which == "cross" else per[which]
        if not rk:
            continue
        fr = _g59._eval_rows(rk, [r for r in cat_z if r["dataset"] in set(per["test"])], delta, arms, top_k)
        for r in fr:
            r["fold"] = fold
        rows += fr
    return rows


def cmd_analyze(outdir: Path) -> dict:
    rows, ea, phi = _load_all()
    rk_own = json.loads((outdir / "rankings_ea.json").read_text(encoding="utf-8"))
    rk_ref = json.loads((S1UCR / "rankings_ucr.json").read_text(encoding="utf-8"))
    rk = merge_ref_rankings(rk_own, rk_ref, arms=REF_ARMS)
    cat_z = _g56.zscore_within(rows)
    delta = {(d, int(b)): DELTA_UCR for d in {r["dataset"] for r in rows} for b in BUDGETS}
    present = [a for a in list(ARMS) + list(REF_ARMS) if any(a in pa for per in rk.values() for pb in per["normal"].values() for pa in pb.values())]
    ctx_arms = [a for a in present if a in ARMS]
    cross_seeds = sorted({cs for per in rk.values() for cs in per["cross"]})
    top_k = _g56.TOP_GOOD
    normal = _eval(rk, cat_z, delta, present, "normal", top_k=top_k)
    cross = [_eval(rk, cat_z, delta, ctx_arms, "cross", cross_seed=cs, top_k=top_k) for cs in cross_seeds]
    zero = _eval(rk, cat_z, delta, ctx_arms, "zero", top_k=top_k)
    pooled = {}
    for a in present:
        pooled[a] = {m: _g57._pooled(normal, a, m) for m in ("acc_top10", "wacc", "pair_acc", "rho")}
        pooled[a]["regret1"] = float(np.mean([r["regret"][a + "1"] for r in normal if a + "1" in r["regret"]]))
        pooled[a]["regret3"] = float(np.mean([r["regret"][a + "3"] for r in normal if a + "3" in r["regret"]]))
    deltas = {"cross": {a: {m: _g57.ablation_delta(normal, cross, a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_arms},
              "zero": {a: {m: _g57.ablation_delta(normal, [zero], a, m) for m in ("acc_top10", "wacc", "pair_acc")} for a in ctx_arms}}
    by_fold = {a: {fo: _g57.ablation_delta([r for r in normal if r["fold"] == fo], [[r for r in c if r["fold"] == fo] for c in cross], a, PRIMARY).get("mean_delta")
                   for fo in sorted({r["fold"] for r in normal})} for a in ctx_arms}
    pairs = [("u_blind_diff_hard", "u_ea_diff_hard"), ("u_phi_diff_hard", "u_eaphi_diff_hard"), ("u_phi_diff_hard", "u_ea_diff_hard"),
             ("u_blind_diff_hard", "u_eaphi_diff_hard"), ("u_blind_diff_hard", "u_ea_concat_hard")]
    value = [_g58.paired_regret(normal, a, b) for a, b in pairs if a in present and b in present]
    by_budget = {str(b): {a: _g57._pooled([r for r in normal if r["budget"] == b], a, "pair_acc") for a in present} for b in BUDGETS}
    find = lambda a, b: next((t for t in value if t["a"] == a and t["b"] == b), {"p": None, "mean_a": None, "mean_b": None})
    g = gates_ea(deltas["cross"], find("u_blind_diff_hard", "u_ea_diff_hard"), find("u_phi_diff_hard", "u_eaphi_diff_hard"))
    align = field_alignment(ea, cat_z)
    n_aligned = sum(1 for v in align.values() if v["aligned"])
    # C0：参照臂 M0 与 s1ucr analysis.json 逐位
    ref_an = json.loads((S1UCR / "analysis.json").read_text(encoding="utf-8")) if (S1UCR / "analysis.json").exists() else None
    c0 = {}
    for a in REF_ARMS:
        ref_v = None if ref_an is None else ref_an["views"]["all20"]["pooled"].get(a, {}).get("pair_acc")
        got = pooled.get(a, {}).get("pair_acc")
        c0[a] = {"ref": ref_v, "got": got, "pass": None if (ref_v is None or got is None) else bool(abs(ref_v - got) <= C0_TOL)}
    c0["pass"] = all(v["pass"] for k, v in c0.items() if k != "pass") if all(v["pass"] is not None for k, v in c0.items() if k != "pass") else None
    n_nan = {}
    for d, row in ea.items():
        k = sum(1 for c in EA_COLS if not np.isfinite(_val(row, c)))
        if k:
            n_nan[d] = k
    a = {"protocol": "results/tclsc_s1ea/PROTOCOL.md", "n_datasets": len({r["dataset"] for r in rows}), "delta": DELTA_UCR, "arms": present,
         "n_instances": len(normal), "pooled": pooled, "deltas": deltas, "by_fold_delta": by_fold, "value_tests": value,
         "by_budget_pair_acc": by_budget, "alignment": align, "n_aligned": n_aligned, "ea_nan": n_nan, "c0": c0,
         "gates": g, "verdict": verdict_ea(g), "per_instance_normal": normal}
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(f"{a['verdict']} | " + " | ".join(f"{arm} M1 {pooled[arm]['acc_top10']:.3f} M0 {pooled[arm]['pair_acc']:.3f} r3 {pooled[arm]['regret3']:.4f}"
                                            for arm in present) + f" | 对齐 {n_aligned}/4", flush=True)
    return a


def predictions(a: dict) -> dict:
    g, p = a["gates"], a["pooled"]
    m0_ea, m0_phi = p.get("u_ea_diff_hard", {}).get("pair_acc"), p.get("u_phi_diff_hard", {}).get("pair_acc")
    return {"P1": {"desc": "C0 逐位（参照臂 M0 ±0.001）", "hit": a["c0"].get("pass")},
            "P2": {"desc": "G-A1 过", "hit": g["GA1"]["pass"]}, "P3": {"desc": "G-A2 过", "hit": g["GA2"]["pass"]},
            "P4": {"desc": "G-A3 过", "hit": g["GA3"]["pass"]},
            "P5": {"desc": "G-A4 ≥ 2/4 字段对齐", "value": a["n_aligned"], "hit": bool(a["n_aligned"] >= 2)},
            "P6": {"desc": "G-A5 concat 零", "hit": g["GA5"]["pass"]},
            "P7": {"desc": "E_A 臂 M0 ≥ φ 臂 M0", "value": [m0_ea, m0_phi],
                   "hit": None if m0_ea is None or m0_phi is None else bool(m0_ea >= m0_phi)}}


def markdown_tables(a: dict) -> str:
    f = lambda b, k: ("—" if b.get(k) is None else f"{b[k]:.3f}")
    L = [f"# WP16-TCLSC 阶段 1 响应画像 E tables（UCR {a['n_datasets']} 集，7 折，δ={a['delta']}，{a['n_instances']} 实例）", "",
         f"读数码 **{a['verdict']}**", "", "## normal", "",
         "| 臂 | M1 | M2 | M0 | Spearman | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, b in a["pooled"].items():
        L.append(f"| {arm} | {f(b,'acc_top10')} | {f(b,'wacc')} | {f(b,'pair_acc')} | {f(b,'rho')} | {b['regret1']:.4f} | {b['regret3']:.4f} |")
    L += ["", "## 消融 Δ", "", "| 臂 | 消融 | 指标 | normal | 消融后 | Δ | p | 胜/负/平 |", "|---|---|---|---:|---:|---:|---:|---|"]
    for kind in ("cross", "zero"):
        for arm, dd in a["deltas"][kind].items():
            for metric, d in dd.items():
                if d.get("mean_delta") is None:
                    continue
                L.append(f"| {arm} | {kind} | {metric} | {d['normal']:.3f} | {d['cross']:.3f} | {d['mean_delta']:+.3f} | {d['p']:.3g} | {d['wins']}/{d['losses']}/{d['ties']} |")
    L += ["", "## 逐折 Δ_M1（cross）", ""]
    for arm, bf in a["by_fold_delta"].items():
        L.append(f"- {arm}: " + ", ".join(f"{fo} " + ("—" if x is None else "%+.3f" % x) for fo, x in bf.items()))
    L += ["", "## 价值（regret@3 配对，H1 b < a）", ""]
    for t in a["value_tests"]:
        L.append(f"- {t['a']} → {t['b']}: n={t['n']} {t['mean_a']:.4f} → {t['mean_b']:.4f} p={t['p']:.3g} 胜/负/平 {t.get('wins_b')}/{t.get('losses_b')}/{t.get('ties')}")
    L += ["", "## 逐预算 M0", ""]
    for b, pa in a["by_budget_pair_acc"].items():
        L.append(f"- @{b}: " + ", ".join(f"{arm} {v:.3f}" for arm, v in pa.items() if v is not None))
    L += ["", "## G-A4 机制对齐（ret_f ↔ 字段偏好 u_f @290）", "", "| 字段 | 列 | n | ρ | p | 对齐 |", "|---|---|---:|---:|---:|---|"]
    for fld, v in a["alignment"].items():
        rho = "—" if v["rho"] is None else "%+.3f" % v["rho"]
        pv = "—" if v["p"] is None else "%.3g" % v["p"]
        L.append(f"| {fld} | {v['col']} | {v['n']} | {rho} | {pv} | {v['aligned']} |")
    L += ["", f"E_A nan 数据集：{a['ea_nan']}", "", "## 门", "", "| 门 | 值 | 过 |", "|---|---|---|"]
    for k, v in a["gates"].items():
        if k == "thresholds":
            continue
        L.append(f"| {k} | {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'}, ensure_ascii=False)} | {v['pass']} |")
    for k, v in a["c0"].items():
        if k != "pass":
            L.append(f"| C0 {k} | ref {v['ref']} got {v['got']} | {v['pass']} |")
    L += ["", "## 预测对账", ""] + [f"- {k} {v['desc']}: {v['hit']}" + (f"（{v['value']}）" if "value" in v else "") for k, v in a["predictions"].items()]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not any((args.features, args.train, args.rank, args.analyze)):
        build_parser().error("至少指定 --features / --train / --rank / --analyze 之一")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.features:
        cmd_features(args.outdir, resume=args.resume)
    if args.train:
        cmd_train(args.outdir, arms=args.arms, seeds=args.seeds, folds_sel=args.folds, device=args.device, epochs=args.epochs, resume=args.resume)
    if args.rank:
        cmd_rank(args.outdir, seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
