"""行为条件化侦察：测量条件化零 GPU 选拔场（协议 results/behavior_recon/PROTOCOL.md）。

条件化输入 = 目标任务/数据集自己的 j 个实测行为格（无 E/P/G 特征）。
第 0 步 k=1 三态重测信度杀门 → 第 1 步数据集级 LOO 四臂（列先验/平移/行为
kNN/低秩补全）→ 第 2 步任务级宽面板 LOTO（行为距离核 vs 等权锚/全局先验）
→ 过闸判读。全程零 GPU、缓存只读、无新 R'。

    python scripts/28_behavior_recon.py --audit
    python scripts/28_behavior_recon.py --step0
    python scripts/28_behavior_recon.py --step1 --step2
    python scripts/28_behavior_recon.py --analyze
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

fl = importlib.import_module("scripts.15_feasibility_labels")
lo = importlib.import_module("scripts.16_feasibility_loo")
dm = importlib.import_module("scripts.14_destruction_matrix")
gp = importlib.import_module("scripts.22_gate2_pilot")

from autocls.labeling import load_probe_cls, run_seed_for  # noqa: E402

OUTDIR = ROOT / "results" / "behavior_recon"
ASSEMBLY = OUTDIR / "assembly"
ITERS = 290
STATES = fl.LABELS
N_STATES = len(STATES)

# 钉源/排除（协议 §3 写死）
PIN_CROP = {"PTB-XL", "NonInvasiveFetalECGThorax1"}
EXCLUDE_DS = {"PenDigits"}
SURVEY_BASE = ROOT / "results" / "candidate_survey" / "baseline"
SURVEY_CROP = ROOT / "results" / "candidate_survey" / "baseline_crop256"
STAGE1_TRAINED = (ROOT / "results" / "stage1_rerun_20260804" / "results"
                  / "labels_route_a" / "rprime_cache.jsonl")
K3_CACHE = (ROOT / "results" / "stage1_rerun_20260804" / "results"
            / "step0_snr_gate" / "noise_cache.jsonl")

# 宽面板任务级（协议 §6）
PRIMARY_DS = ("CharacterTrajectories", "UWaveGestureLibrary",
              "PhonemeSpectra", "Handwriting")
SUPP_DS = "UniMiB-SHAR"
N_TASKS_PANEL = 8
K_ANCHOR = 3

# 判读常数（协议 §4–§6 写死）
J_GRID = (3, 5, 7)
R_STEP1, R_STEP2 = 100, 200
K_NN, K_NN_SENS = 3, (1, 5)
RANK, RANK_SENS = 2, (1, 3)
RIDGE = 1e-3
KILL_M1, KILL_M2 = 0.35, 0.45
GREEN_M1, GREEN_M2 = 0.50, 0.60
ALPHA_STEP1, ALPHA_STEP2 = 0.025, 0.05
WIN_RATE_AMBER = 0.60


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def pin_filter(rows: list, mode: str) -> list:
    """双源钉源（协议 §3）：main 丢弃 crop 钉源数据集；crop 只保留它们。"""
    if mode == "main":
        return [r for r in rows if r.get("dataset") not in PIN_CROP]
    if mode == "crop":
        return [r for r in rows if r.get("dataset") in PIN_CROP]
    return list(rows)


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    states = sorted(set(a.tolist()) | set(b.tolist()))
    pe = sum(float((a == s).mean()) * float((b == s).mean()) for s in states)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def verdict_step0(m1: float, m2: float) -> str:
    """三档判读（协议 §4，边界含在高档一侧）。"""
    if m1 < KILL_M1 or m2 < KILL_M2:
        return "kill"
    if m1 < GREEN_M1 or m2 < GREEN_M2:
        return "amber"
    return "green"


def draw_cols(n_cols: int, j: int, *, seed: int) -> list:
    """确定性无放回均匀抽 j 列（毒哨勿复活：禁按毒性排序选格）。"""
    rng = np.random.default_rng([seed, 20260831])
    return sorted(rng.choice(n_cols, size=j, replace=False).tolist())


def rate_vector(code_rows: list) -> np.ndarray:
    """任务状态码行 → (n_cls, 3) 三态率矩阵。"""
    return np.mean([np.eye(N_STATES)[np.asarray(c)] for c in code_rows], axis=0)


def masked_brier(pred, target, cols) -> float:
    """列子集软目标 Brier：one-hot 目标 + 全列时退化为 profile_brier。"""
    cols = list(cols)
    diff = np.asarray(pred, float)[cols] - np.asarray(target, float)[cols]
    return float((diff ** 2).sum(axis=1).mean())


def behavior_distance(codes_a, codes_b, J) -> float:
    """行为距离 = 观测列上三态码不一致率（只吃观测列）。"""
    J = list(J)
    return float((np.asarray(codes_a)[J] != np.asarray(codes_b)[J]).mean())


def _normalize_rows(P: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    out = np.array(fallback, float, copy=True)
    ok = s[:, 0] > 0
    out[ok] = P[ok] / s[ok]
    return out


def arm_column_prior(train_stack: np.ndarray) -> np.ndarray:
    return train_stack.mean(axis=0)


def arm_level_shift(prior: np.ndarray, q_obs: np.ndarray, J) -> np.ndarray:
    """平移臂（协议 §5b）：逐状态常数移，只调水平不调 per-CLS 结构。"""
    lam = (np.asarray(q_obs, float) - prior[list(J)]).mean(axis=0)
    return _normalize_rows(np.clip(prior + lam, 0.0, 1.0), prior)


def arm_knn(train_named: list, q_obs: np.ndarray, J, k: int) -> np.ndarray:
    """行为 kNN（协议 §5c）：J 列均方 L2，并列按名字典序断。"""
    J = list(J)
    q_obs = np.asarray(q_obs, float)
    scored = sorted(
        (float(((Q[J] - q_obs) ** 2).sum(axis=1).mean()), name, Q)
        for name, Q in train_named)
    return np.mean([Q for _d, _n, Q in scored[:k]], axis=0)


def arm_svd_complete(train_stack: np.ndarray, q_obs: np.ndarray, J,
                     rank: int, ridge: float) -> np.ndarray:
    """低秩补全（协议 §5d）：列去均值 SVD + 观测维 ridge 回归主成分得分。"""
    n, n_cls, _ = train_stack.shape
    X = train_stack.reshape(n, -1)
    mu = X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(X - mu, full_matrices=False)
    V = vt[:rank]
    obs_idx = np.array([c * N_STATES + s for c in J for s in range(N_STATES)])
    A = V[:, obs_idx].T
    b = np.asarray(q_obs, float).reshape(-1) - mu[obs_idx]
    scores = np.linalg.solve(A.T @ A + ridge * np.eye(rank), A.T @ b)
    P = (mu + V.T @ scores).reshape(n_cls, N_STATES)
    return _normalize_rows(np.clip(P, 0.0, 1.0), mu.reshape(n_cls, N_STATES))


def predict_dataset_arms(train_named: list, q_held_full, J, *, k: int,
                         rank: int) -> dict:
    """四臂统一入口：留出行只经 J 列进入（折隔离由构造保证，单测覆盖）。"""
    q_obs = np.asarray(q_held_full, float)[list(J)]
    train_stack = np.stack([Q for _n, Q in train_named])
    prior = arm_column_prior(train_stack)
    return {"column_prior": prior,
            "level_shift": arm_level_shift(prior, q_obs, J),
            "knn": arm_knn(train_named, q_obs, J, k),
            "svd": arm_svd_complete(train_stack, q_obs, J, rank, RIDGE)}


def collect_trained_per_seed(path, shared_keys, bands, classes) -> tuple:
    """噪声缓存 per-seed 收集：查 iters/shared/classes/基线，不查 split
    （协议 §3 口径声明：跨 seed 一致性度量的是权重+划分联合噪声）。"""
    out: dict = {}
    rejects: dict = defaultdict(int)
    for r in _read_jsonl(path):
        tid, key = r.get("task_id"), r.get("cls_key")
        if tid is None or r.get("r_prime") is None:
            rejects["null"] += 1
            continue
        if r.get("iters") != ITERS:
            rejects["iters"] += 1
            continue
        if key not in shared_keys:
            rejects["not_shared"] += 1
            continue
        if tid not in bands:
            rejects["no_baseline"] += 1
            continue
        if list(r.get("classes", [])) != list(classes.get(tid, [])):
            rejects["classes"] += 1
            continue
        out[(tid, key, r.get("seed"))] = float(r["r_prime"])
    return out, dict(rejects)


# ---------------------------------------------------------------- 装配
def _read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_filtered(src: Path, dst: Path, mode: str) -> dict:
    rows = _read_jsonl(src)
    keep = pin_filter(rows, mode)
    with open(dst, "w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"src": str(src.relative_to(ROOT)), "mode": mode,
            "n_in": len(rows), "n_kept": len(keep)}


def build_assembly() -> dict:
    """钉源过滤副本（协议 §3；确定性重建，原缓存只读）。"""
    ASSEMBLY.mkdir(parents=True, exist_ok=True)
    log = [
        _write_filtered(SURVEY_BASE / "baseline_cache.jsonl",
                        ASSEMBLY / "survey_baseline_main.jsonl", "main"),
        _write_filtered(SURVEY_CROP / "baseline_cache.jsonl",
                        ASSEMBLY / "survey_baseline_crop.jsonl", "crop"),
        _write_filtered(SURVEY_BASE / "trained_fill.jsonl",
                        ASSEMBLY / "survey_trained_main.jsonl", "main"),
        _write_filtered(SURVEY_CROP / "trained_fill.jsonl",
                        ASSEMBLY / "survey_trained_crop.jsonl", "crop"),
    ]
    return {"filters": log}


def baseline_paths() -> list:
    return [ROOT / "results" / "baseline_probe" / "baseline_cache.jsonl",
            ASSEMBLY / "survey_baseline_main.jsonl",
            ASSEMBLY / "survey_baseline_crop.jsonl",
            ROOT / "results" / "wide_panel" / "baseline_cache.jsonl"]


def trained_paths() -> list:
    return [STAGE1_TRAINED,
            ROOT / "results" / "baseline_probe" / "trained_fill.jsonl",
            ASSEMBLY / "survey_trained_main.jsonl",
            ASSEMBLY / "survey_trained_crop.jsonl",
            ROOT / "results" / "wide_panel" / "rprime_cache.jsonl"]


def load_shared_keys() -> list:
    return sorted(k for _c, k in load_probe_cls(fl.PROBE_CATALOG, 10)[0])


def assemble() -> dict:
    """一次装配全部粮草：bands/classes、k1/k3、行、面板码、per-seed 网格。"""
    build_assembly()
    keys = load_shared_keys()
    shared = set(keys)
    bands, classes, rej_b = dm.collect_baselines(baseline_paths(),
                                                 exclude_plus_c=True)
    k1, rej_k1 = dm.collect_trained(trained_paths(), shared, bands, classes,
                                    lambda ds: ITERS)
    k3, rej_k3 = dm.collect_trained_k3(K3_CACHE, shared, bands, classes)
    merged, rej_m = fl.merge_trained(k3, k1)
    seed0, _ = dm.collect_trained([STAGE1_TRAINED], shared, bands, classes,
                                  lambda ds: ITERS)
    per_seed, rej_ps = collect_trained_per_seed(K3_CACHE, shared, bands, classes)
    return {"keys": keys, "bands": bands, "classes": classes,
            "k1": k1, "k3": k3, "merged": merged, "seed0": seed0,
            "per_seed": per_seed,
            "rejects": {"baseline": rej_b, "k1": rej_k1, "k3": rej_k3,
                        "merge": rej_m, "per_seed": rej_ps}}


def _flag(rp: float, tid: str, bands: dict) -> int:
    mean, band = bands[tid]
    return STATES.index(fl.tri_state(rp, mean, band))


def build_rows(asm: dict, *, granularity: str = "merged") -> tuple[dict, dict]:
    """数据集级行（协议 §3）：仅 _00/_01；13 UEA 用 k3（granularity="k1" 时
    改用 seed0 k1，敏感性 S3）。返回 (Q, source)。"""
    keys, bands = asm["keys"], asm["bands"]
    cell_src: dict = {}
    for (tid, key), info in asm["merged"].items():
        cell_src[(tid, key)] = (info["r_prime"], info["r_source"])
    if granularity == "k1":
        for (tid, key), rp in asm["k1"].items():
            cell_src[(tid, key)] = (rp, "k1")
    datasets = sorted({fl.dataset_of(tid) for tid, _k in cell_src}
                      - EXCLUDE_DS)
    Q, source = {}, {}
    for ds in datasets:
        tids = [f"{ds}_00", f"{ds}_01"]
        if not all((t, k) in cell_src and t in bands
                   for t in tids for k in keys):
            continue
        code_rows = [np.array([_flag(cell_src[(t, k)][0], t, bands)
                               for k in keys]) for t in tids]
        Q[ds] = rate_vector(code_rows)
        source[ds] = sorted({cell_src[(t, k)][1] for t in tids for k in keys})
    return Q, source


def build_panel_codes(asm: dict) -> dict:
    """任务级面板（协议 §6）：k=1 canonical 三态码，{ds: {tid: codes(10,)}}。"""
    keys, bands = asm["keys"], asm["bands"]
    panel = {}
    for ds in (*PRIMARY_DS, SUPP_DS):
        prof = {}
        for i in range(N_TASKS_PANEL):
            tid = f"{ds}_{i:02d}"
            if tid in bands and all((tid, k) in asm["k1"] for k in keys):
                prof[tid] = np.array([_flag(asm["k1"][(tid, k)], tid, bands)
                                      for k in keys])
        panel[ds] = prof
    return panel


def build_step0_grid(asm: dict) -> dict:
    """信度网格（协议 §4）：{(tid, key): [seed0, s1, s2, s3] 的 r'}。"""
    grid = {}
    for (tid, key), rp0 in asm["seed0"].items():
        obs = [rp0] + [asm["per_seed"].get((tid, key, s)) for s in (1, 2, 3)]
        if all(v is not None for v in obs):
            grid[(tid, key)] = obs
    return grid


# ---------------------------------------------------------------- 子命令
def cmd_audit(asm: dict) -> None:
    Q, source = build_rows(asm)
    panel = build_panel_codes(asm)
    grid = build_step0_grid(asm)
    all_ds = sorted({fl.dataset_of(t) for t, _k in asm["k1"]} |
                    {fl.dataset_of(t) for t, _k in asm["k3"]})
    audit = {
        "protocol": "results/behavior_recon/PROTOCOL.md", "iters": ITERS,
        "assembly": build_assembly(),
        "n_rows": len(Q),
        "rows": {ds: source[ds] for ds in sorted(Q)},
        "rows_dropped": sorted(set(all_ds) - set(Q) - EXCLUDE_DS),
        "excluded": sorted(EXCLUDE_DS),
        "panel_complete_tasks": {ds: len(p) for ds, p in panel.items()},
        "step0_cells": len(grid),
        "step0_expected": 260,
        "rejects": asm["rejects"],
    }
    (OUTDIR / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"行数 {len(Q)}（预期 25）：{', '.join(sorted(Q))}")
    print(f"落选数据集: {audit['rows_dropped'] or '无'} | 排除: {audit['excluded']}")
    print(f"面板完整任务: {audit['panel_complete_tasks']}")
    print(f"step0 网格: {len(grid)}/260 格 × 4 观测")
    if len(Q) < 20:
        print("⚠️ n<20：装配异常（协议 §3），停下排查")
    print(f"→ {OUTDIR / 'audit.json'}", flush=True)


def cmd_step0(asm: dict) -> None:
    from scipy.stats import spearmanr
    grid = build_step0_grid(asm)
    cells = sorted(grid)
    bands = asm["bands"]
    n_seeds = 4
    labels = np.array([[_flag(grid[c][s], c[0], bands) for c in cells]
                       for s in range(n_seeds)])
    pair_kappa = {f"{i}-{j}": cohen_kappa(labels[i], labels[j])
                  for i, j in combinations(range(n_seeds), 2)}
    m1 = float(np.mean(list(pair_kappa.values())))

    keys = asm["keys"]
    tids = sorted({tid for tid, _k in cells})
    rhos, n_nan = [], 0
    for tid in tids:
        prof = np.array([[grid[(tid, k)][s] for k in keys]
                         for s in range(n_seeds)])
        for i, j in combinations(range(n_seeds), 2):
            r = spearmanr(prof[i], prof[j]).statistic
            if np.isnan(r):
                n_nan += 1
            else:
                rhos.append(float(r))
    m2 = float(np.median(rhos))

    # 辅助只报不判
    per_state_agree = {}
    for s_idx, s_name in enumerate(STATES):
        present = labels == s_idx
        base = present.any(axis=0)
        per_state_agree[s_name] = float(
            (present.all(axis=0)[base]).mean()) if base.any() else None
    k3mean_codes = np.array([_flag(float(np.mean(grid[c][1:])), c[0], bands)
                             for c in cells])
    kappa_k1_vs_k3 = cohen_kappa(labels[0], k3mean_codes)

    verdict = verdict_step0(m1, m2)
    out = {"protocol": "results/behavior_recon/PROTOCOL.md",
           "n_cells": len(cells), "n_tasks": len(tids),
           "M1_kappa_mean": m1, "pair_kappa": pair_kappa,
           "M2_profile_rho_median": m2, "n_rho": len(rhos), "n_rho_nan": n_nan,
           "aux": {"per_state_agree_rate": per_state_agree,
                   "kappa_seed0_vs_k3mean": kappa_k1_vs_k3},
           "bands": {"kill": {"M1": KILL_M1, "M2": KILL_M2},
                     "green": {"M1": GREEN_M1, "M2": GREEN_M2}},
           "verdict": verdict}
    (OUTDIR / "step0_reliability.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n===== 第 0 步：k=1 三态重测信度（{len(cells)} 格 × 4 观测）=====")
    print(f"M1 格级 κ 均值 = {m1:.3f}（6 对：" +
          " ".join(f"{k}:{v:.3f}" for k, v in pair_kappa.items()) + "）")
    print(f"M2 画像 ρ 中位 = {m2:.3f}（{len(rhos)} 对，NaN {n_nan}）")
    print(f"辅助: 逐状态一致率 {per_state_agree} | seed0 vs k3均值 κ={kappa_k1_vs_k3:.3f}")
    print(f"[杀门判读] → {verdict}"
          + ("（侦察停，转 C 升级路径）" if verdict == "kill" else
             f"（主判读点 j={7 if verdict == 'amber' else 5}）"))
    print(f"→ {OUTDIR / 'step0_reliability.json'}", flush=True)


def _guard_step0() -> str:
    p = OUTDIR / "step0_reliability.json"
    if not p.exists():
        sys.exit("先跑 --step0（杀门先行，协议 §4）")
    verdict = json.loads(p.read_text(encoding="utf-8"))["verdict"]
    if verdict == "kill":
        sys.exit("第 0 步杀档：第 1/2 步不跑（协议 §4）")
    return verdict


def _run_loo(Q: dict, *, tag: str) -> dict:
    """数据集级 LOO 主循环：{arm: {j: {fold: err}}}。"""
    names = sorted(Q)
    n_cls = len(next(iter(Q.values())))
    arms = ["column_prior", "level_shift", "knn", "svd"] + \
           [f"knn_k{k}" for k in K_NN_SENS] + [f"svd_r{r}" for r in RANK_SENS]
    fold_err: dict = {a: {j: {} for j in J_GRID} for a in arms}
    for held in names:
        train_named = [(n, Q[n]) for n in names if n != held]
        train_stack = np.stack([q for _n, q in train_named])
        prior = arm_column_prior(train_stack)
        for j in J_GRID:
            acc: dict = defaultdict(list)
            for rep in range(R_STEP1):
                J = draw_cols(n_cls, j,
                              seed=run_seed_for(f"br1:{held}", f"j{j}:rep{rep}"))
                unobs = [c for c in range(n_cls) if c not in J]
                q_obs = Q[held][J]
                preds = {"column_prior": prior,
                         "level_shift": arm_level_shift(prior, q_obs, J),
                         "knn": arm_knn(train_named, q_obs, J, K_NN),
                         "svd": arm_svd_complete(train_stack, q_obs, J,
                                                 RANK, RIDGE)}
                for k in K_NN_SENS:
                    preds[f"knn_k{k}"] = arm_knn(train_named, q_obs, J, k)
                for r in RANK_SENS:
                    preds[f"svd_r{r}"] = arm_svd_complete(train_stack, q_obs,
                                                          J, r, RIDGE)
                for a, p in preds.items():
                    acc[a].append(masked_brier(p, Q[held], unobs))
            for a in arms:
                fold_err[a][j][held] = float(np.mean(acc[a]))
    return {"tag": tag, "folds": names, "fold_err": fold_err,
            "tests": _loo_tests(fold_err, names)}


def _loo_tests(fold_err: dict, names: list) -> dict:
    """行为臂 vs 两对照：p / 方向 / 折胜率 / 非零差数（协议 §5 D1r 素材）。"""
    tests: dict = {}
    behav = [a for a in fold_err if a not in ("column_prior", "level_shift")]
    for a in behav:
        tests[a] = {}
        for j in J_GRID:
            x = np.array([fold_err[a][j][n] for n in names])
            entry = {}
            for ctrl in ("column_prior", "level_shift"):
                y = np.array([fold_err[ctrl][j][n] for n in names])
                entry[f"p_vs_{ctrl}"] = lo.paired_wilcoxon_less(x, y)
                entry[f"med_diff_vs_{ctrl}"] = float(np.median(x - y))
                entry[f"win_rate_vs_{ctrl}"] = float((x < y).mean())
                entry[f"n_nonzero_vs_{ctrl}"] = int((x != y).sum())
            tests[a][f"j{j}"] = entry
    return tests


def cmd_step1(asm: dict) -> None:
    _guard_step0()
    Q, source = build_rows(asm)
    main = _run_loo(Q, tag="main")
    q_k3 = {ds: Q[ds] for ds in Q if source[ds] == ["k3"]}
    s1 = _run_loo(q_k3, tag="S1_k3_only") if len(q_k3) >= 8 else None
    q_alt, _src = build_rows(asm, granularity="k1")
    s3 = _run_loo(q_alt, tag="S3_k1_granularity")
    # S2（描述性）：5 面板数据集换 8 任务率向量
    panel = build_panel_codes(asm)
    q_s2 = dict(Q)
    for ds, prof in panel.items():
        if len(prof) == N_TASKS_PANEL and ds in q_s2:
            q_s2[ds] = rate_vector([prof[t] for t in sorted(prof)])
    s2 = _run_loo(q_s2, tag="S2_panel_8task_rows")
    out = {"protocol": "results/behavior_recon/PROTOCOL.md",
           "n_rows": len(Q), "row_source": source,
           "main": main, "S1": s1, "S2": s2, "S3": s3}
    (OUTDIR / "step1_dataset_loo.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\n===== 第 1 步：数据集级 LOO（n={len(Q)} 行 × j∈{J_GRID} × "
          f"R={R_STEP1}）=====")
    for a in ("knn", "svd"):
        for j in J_GRID:
            t = main["tests"][a][f"j{j}"]
            print(f"[{a} j={j}] vs列先验 p={t['p_vs_column_prior']:.4f} "
                  f"(win {t['win_rate_vs_column_prior']:.2f}) | vs平移 "
                  f"p={t['p_vs_level_shift']:.4f} "
                  f"(win {t['win_rate_vs_level_shift']:.2f})")
    print(f"→ {OUTDIR / 'step1_dataset_loo.json'}", flush=True)


def _run_loto(panel_ds: dict, *, seed_ns: str) -> dict:
    """单数据集 LOTO：行为核 vs 等权锚 vs 全局列先验，{arm: {j: {tid: err}}}。"""
    tids = sorted(panel_ds)
    n_cls = len(panel_ds[tids[0]])
    fold_err: dict = {a: {j: {} for j in J_GRID}
                      for a in ("kernel", "uniform", "global")}
    for held in tids:
        rest = [t for t in tids if t != held]
        glob = np.mean([np.eye(N_STATES)[panel_ds[t]] for t in rest], axis=0)
        onehot_held = np.eye(N_STATES)[panel_ds[held]]
        for j in J_GRID:
            acc: dict = defaultdict(list)
            for rep in range(R_STEP2):
                J = draw_cols(n_cls, j, seed=run_seed_for(
                    f"{seed_ns}:{held}", f"j{j}:rep{rep}"))
                unobs = [c for c in range(n_cls) if c not in J]
                anchors = gp.sample_anchors(
                    tids, held=held, k=min(K_ANCHOR, len(rest)),
                    seed=run_seed_for(f"{seed_ns}:{held}",
                                      f"anchor:j{j}:rep{rep}"))
                dists = {t: behavior_distance(panel_ds[held], panel_ds[t], J)
                         for t in rest}
                sigma = max(float(np.median(
                    [behavior_distance(panel_ds[a], panel_ds[b], J)
                     for i, a in enumerate(rest) for b in rest[i + 1:]])), 1e-6)
                wk = gp.kernel_w({a: dists[a] for a in anchors}, sigma)
                acc["kernel"].append(masked_brier(
                    gp.anchor_predict(panel_ds, anchors, wk),
                    onehot_held, unobs))
                acc["uniform"].append(masked_brier(
                    gp.anchor_predict(panel_ds, anchors,
                                      {a: 1.0 for a in anchors}),
                    onehot_held, unobs))
                acc["global"].append(masked_brier(glob, onehot_held, unobs))
            for a in fold_err:
                fold_err[a][j][held] = float(np.mean(acc[a]))
    return fold_err


def _loto_tests(fold_errs: list) -> dict:
    """pooled 折检验（协议 §6 D2r 素材）。fold_errs = 各数据集 fold_err 列表。"""
    tests = {}
    for j in J_GRID:
        k_arr = np.concatenate([[fe["kernel"][j][t] for t in sorted(fe["kernel"][j])]
                                for fe in fold_errs])
        entry = {"n_folds": len(k_arr)}
        for ctrl in ("uniform", "global"):
            c_arr = np.concatenate([[fe[ctrl][j][t] for t in sorted(fe[ctrl][j])]
                                    for fe in fold_errs])
            entry[f"p_vs_{ctrl}"] = lo.paired_wilcoxon_less(k_arr, c_arr)
            entry[f"med_diff_vs_{ctrl}"] = float(np.median(k_arr - c_arr))
            entry[f"win_rate_vs_{ctrl}"] = float((k_arr < c_arr).mean())
            entry[f"n_nonzero_vs_{ctrl}"] = int((k_arr != c_arr).sum())
        tests[f"j{j}"] = entry
    return tests


def _run_cross_ds_knn(panel: dict) -> dict:
    """探索性变体（不入闸）：40 任务池跨数据集 kNN（k=5）vs 跨池全局先验。"""
    all_tasks = {t: codes for ds, prof in panel.items()
                 for t, codes in prof.items()}
    n_cls = len(next(iter(all_tasks.values())))
    med = {}
    for j in J_GRID:
        errs_knn, errs_glob = [], []
        for held, codes in sorted(all_tasks.items()):
            ds_held = fl.dataset_of(held)
            others = {t: c for t, c in all_tasks.items()
                      if fl.dataset_of(t) != ds_held}
            glob = np.mean([np.eye(N_STATES)[c] for c in others.values()],
                           axis=0)
            onehot_held = np.eye(N_STATES)[codes]
            ek, eg = [], []
            for rep in range(R_STEP2):
                J = draw_cols(n_cls, j, seed=run_seed_for(
                    f"br2x:{held}", f"j{j}:rep{rep}"))
                unobs = [c for c in range(n_cls) if c not in J]
                scored = sorted(
                    (behavior_distance(codes, c, J), t) for t, c in others.items())
                sel = [t for _d, t in scored[:5]]
                pred = np.mean([np.eye(N_STATES)[others[t]] for t in sel], axis=0)
                ek.append(masked_brier(pred, onehot_held, unobs))
                eg.append(masked_brier(glob, onehot_held, unobs))
            errs_knn.append(float(np.mean(ek)))
            errs_glob.append(float(np.mean(eg)))
        med[f"j{j}"] = {"knn5_med": float(np.median(errs_knn)),
                        "global_med": float(np.median(errs_glob)),
                        "win_rate": float((np.array(errs_knn)
                                           < np.array(errs_glob)).mean())}
    return med


def cmd_step2(asm: dict) -> None:
    _guard_step0()
    panel = build_panel_codes(asm)
    for ds in (*PRIMARY_DS, SUPP_DS):
        if len(panel[ds]) != N_TASKS_PANEL:
            sys.exit(f"[{ds}] 完整任务 {len(panel[ds])}/{N_TASKS_PANEL}：面板缺格，先查装配")
    per_ds = {ds: _run_loto(panel[ds], seed_ns="br2") for ds in PRIMARY_DS}
    supp = _run_loto(panel[SUPP_DS], seed_ns="br2")
    out = {"protocol": "results/behavior_recon/PROTOCOL.md",
           "per_dataset": {ds: {"fold_err": fe,
                                "tests": _loto_tests([fe])}
                           for ds, fe in per_ds.items()},
           "pooled_primary": _loto_tests([per_ds[ds] for ds in PRIMARY_DS]),
           "supplementary_um": {"fold_err": supp, "tests": _loto_tests([supp])},
           "exploratory_cross_ds": _run_cross_ds_knn(panel)}
    (OUTDIR / "step2_task_loto.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\n===== 第 2 步：任务级 LOTO（{len(PRIMARY_DS)}×8=32 折主判 + UM 补充）=====")
    for j in J_GRID:
        t = out["pooled_primary"][f"j{j}"]
        print(f"[pooled j={j}] vs等权锚 p={t['p_vs_uniform']:.4f} "
              f"(win {t['win_rate_vs_uniform']:.2f}) | vs全局 "
              f"p={t['p_vs_global']:.4f} (win {t['win_rate_vs_global']:.2f})")
    print(f"→ {OUTDIR / 'step2_task_loto.json'}", flush=True)


def _judge_arm(tests: dict, arm: str, main_j: int, alpha: float,
               ctrls: tuple, amber: bool) -> dict:
    """单臂过闸判定（协议 §5/§6）：主 j 双胜 + 三档方向一致 + 琥珀加严。"""
    t_main = tests[arm][f"j{main_j}"]
    p_ok = all(t_main[f"p_vs_{c}"] < alpha for c in ctrls)
    dir_ok = all(tests[arm][f"j{j}"][f"med_diff_vs_{c}"] < 0
                 for j in J_GRID for c in ctrls)
    win_ok = (not amber) or all(t_main[f"win_rate_vs_{c}"] >= WIN_RATE_AMBER
                                for c in ctrls)
    return {"arm": arm, "p_ok": p_ok, "dir_ok": dir_ok, "win_ok": win_ok,
            "pass": p_ok and dir_ok and win_ok,
            "p": {c: t_main[f"p_vs_{c}"] for c in ctrls}}


def cmd_analyze() -> None:
    step0 = json.loads((OUTDIR / "step0_reliability.json")
                       .read_text(encoding="utf-8"))
    verdict0 = step0["verdict"]
    out = {"protocol": "results/behavior_recon/PROTOCOL.md",
           "step0": {"M1": step0["M1_kappa_mean"],
                     "M2": step0["M2_profile_rho_median"],
                     "verdict": verdict0}}
    if verdict0 == "kill":
        out["gate"] = "killed_step0"
    else:
        amber = verdict0 == "amber"
        main_j = 7 if amber else 5
        s1 = json.loads((OUTDIR / "step1_dataset_loo.json")
                        .read_text(encoding="utf-8"))
        d1r = [_judge_arm(s1["main"]["tests"], a, main_j, ALPHA_STEP1,
                          ("column_prior", "level_shift"), amber)
               for a in ("knn", "svd")]
        s2 = json.loads((OUTDIR / "step2_task_loto.json")
                        .read_text(encoding="utf-8"))
        # pooled_primary 结构与 _judge_arm 对齐：包一层 arm 名
        pooled = {"kernel": s2["pooled_primary"]}
        d2r = _judge_arm(pooled, "kernel", main_j, ALPHA_STEP2,
                         ("uniform", "global"), amber)
        out.update({
            "main_j": main_j, "amber": amber,
            "D1r": {"arms": d1r, "pass": any(a["pass"] for a in d1r)},
            "D2r": d2r,
            "gate": "pass" if (any(a["pass"] for a in d1r) or d2r["pass"])
                    else "fail"})
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print("\n===== 行为条件化侦察：过闸判读 =====")
    print(f"第 0 步 {verdict0}（M1={out['step0']['M1']:.3f} "
          f"M2={out['step0']['M2']:.3f}）")
    if verdict0 != "kill":
        for a in out["D1r"]["arms"]:
            print(f"D1r[{a['arm']}] p={a['p']} pass={a['pass']}")
        print(f"D2r[kernel] p={out['D2r']['p']} pass={out['D2r']['pass']}")
    print(f"[闸门] → {out['gate']}")
    print(f"→ {OUTDIR / 'analysis.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--step0", action="store_true")
    ap.add_argument("--step1", action="store_true")
    ap.add_argument("--step2", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not any(vars(args).values()):
        ap.error("至少指定一个子命令")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    asm = assemble() if (args.audit or args.step0 or args.step1
                         or args.step2) else None
    if args.audit:
        cmd_audit(asm)
    if args.step0:
        cmd_step0(asm)
    if args.step1:
        cmd_step1(asm)
    if args.step2:
        cmd_step2(asm)
    if args.analyze:
        cmd_analyze()


if __name__ == "__main__":
    main()
