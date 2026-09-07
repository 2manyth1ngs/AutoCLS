"""WP10-UEA30：UEA 30 靶外 23 靶的目标侧全量部署（零样本先验名单 → 逐次减半 → 冠军 3 种子）。

协议：results/uea30/PROTOCOL.md。薄驱动：importlib 复用 35（协议装载 / 工单格 / 缓存键 / SVM）、36（先验 / SH
纯函数）、39（目录工单 / 孪生 / 执行 / 轨迹装配 / finals）、40（interleave）；39 / 40 自身的输出逐位不变。
配置真源 configs/uea30.yaml（靶清单 / 分类 A–E / 尺寸 / iters / 文献数 / 预测带 / 锚点 / chance）。

每靶 22 格 = shared-10 + GGS（掩码关）+ 各自的 input_mask=binomial 孪生；不提名（提名因 HYBRID_BIAS 从未起跑，
开放空间已判死）。臂：主 SH10-4-3_prior（名单 10 只在 11 条 legacy 上按先验排）、副 SH20-3_both（名单 10 各带孪生，
两级从 50% 截点起留 3）、对照 SH6-3-3_prior / SH20-4-3_both。先验对不在源池的靶自动全 8 源拟合，SelfRegulationSCP2
（源池低权）自动 LODO；排序器不重训。

子命令（均写 --outdir，默认 results/uea30/；--resume 可续）：
    python scripts/45_uea30_deploy.py --build-candidates      # 零 GPU：candidates.json（协议锁产物）
    python scripts/45_uea30_deploy.py --check                 # 零 GPU：逐靶加载，断言尺寸 / iters / chance / EigenWorms 分段
    python scripts/45_uea30_deploy.py --dry-run               # 工单数 + 估时（yaml anchor_4090，硬件标签）
    python scripts/45_uea30_deploy.py --smoke --datasets A B  # 每靶 1 legacy + 1 孪生 × iters 8 → <outdir>/smoke/
    python scripts/45_uea30_deploy.py --catalog --resume      # S1：23 靶 × 22 格 × seed 0，三截点
    python scripts/45_uea30_deploy.py --analyze               # 零 GPU：臂 / 门 / 表 → analysis.json / picks.json / tables.md
    python scripts/45_uea30_deploy.py --finals --resume       # S2：两臂冠军 × seeds {1,2}
    python scripts/45_uea30_deploy.py --table30               # 全 30 合并表（+ HAR 外部锚）→ table30.{md,json}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.data.uea import load_uea_pair, segment_train  # noqa: E402
from autocls.labeling import cls_key  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace, twin  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g39 = importlib.import_module("scripts.39_sh_deploy")
_g40 = importlib.import_module("scripts.40_mask_field_sim")

CFG = ROOT / "configs" / "uea30.yaml"
OUT = ROOT / "results" / "uea30"
MX3 = ROOT / "results" / "mx3_pq"
WP3 = ROOT / "results" / "mx4_deploy"
WP6 = ROOT / "results" / "mx4_mask"
HAR_DIR = ROOT / "results" / "mx4_deploy_har"
CLASSES = ("A", "B", "C", "D", "E")
LIT_COLS = ("TS2Vec", "T-Loss", "TNC", "TS-TCC", "TST", "DTW")
MAIN_ARM, SECOND_ARM = "SH10-4-3_prior", "SH20-3_both"
REF_ARMS = ("SH6-3-3_prior", "SH20-4-3_both")
SH_MAIN, SH_SECOND, SH_REF6, SH_REF20 = (10, 4, 3), (20, 3), (6, 3, 3), (20, 4, 3)
TOP_LIST = 10                       # 起跑名单长度（只在 legacy 上按先验排）
N_LEGACY = 11                       # shared-10 + GGS
N_CELLS = 2 * N_LEGACY              # + 孪生
N_CKPT = 3
K_TOP = 3
FINAL_SEEDS = _g39.FINAL_SEEDS
HARDWARE = "RTX 4090"
LAUNCH_S = 10.0                     # 每格进程内启动 / 数据搬运的固定开销（估时用）
CONSERVATIVE = 1.4
FINALS_CELLS_UPPER = 4              # 两臂冠军不同时 2 冠军 × 2 种子
SMOKE_ITERS = 8
EVALUABLE_SIX = tuple(_g39.TARGETS)


# ---------------------------------------------------------------- 配置真源
def load_cfg(path: str | Path = CFG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cfg_md5(path: str | Path = CFG) -> str:
    return hashlib.md5(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_targets(cfg: Mapping) -> list[str]:
    """run: true 的靶，按 order 升序（同序按名字）。"""
    return [d for _, d in sorted((int(c["order"]), d) for d, c in cfg["datasets"].items() if c.get("run"))]


def dataset_cfg(cfg: Mapping, d: str) -> dict:
    return cfg["datasets"][d]


def sizes_from_cfg(cfg: Mapping, datasets: Optional[Sequence[str]] = None) -> dict[str, tuple[int, int, int]]:
    """{d: (n_train, L, C)}——L 是 scaler=train 下的填充长度（ragged 取 TRAIN∪TEST 最大）。"""
    ds = list(datasets) if datasets is not None else run_targets(cfg)
    return {d: (int(cfg["datasets"][d]["n_train"]), int(cfg["datasets"][d]["L"]), int(cfg["datasets"][d]["C"])) for d in ds}


def paper_reference(dataset: str, cfg: Optional[Mapping] = None) -> Optional[dict]:
    """35:TABLE10 / 39:HAR_TABLE2 优先；否则 yaml literature；都无 → None（不抛）。"""
    try:
        return _g39.paper_reference(dataset)
    except KeyError:
        pass
    lit = (((cfg or {}).get("datasets") or {}).get(dataset) or {}).get("literature")
    return dict(lit) if lit else None


def paper_ts2vec(dataset: str, cfg: Optional[Mapping] = None) -> Optional[float]:
    r = paper_reference(dataset, cfg)
    return float(r["TS2Vec"]) if r and r.get("TS2Vec") is not None else None


def default_band(klass: str, ts2vec: float, chance: float) -> Optional[tuple[float, float]]:
    """A–D 类的默认预测带（逐靶 yaml band 优先）：A 贴顶、B 短序列、D chance 型；C / E 无带。"""
    if klass == "A":
        lo, hi = ts2vec - 0.03, 1.0
    elif klass == "B":
        lo, hi = ts2vec - 0.10, ts2vec + 0.05
    elif klass == "D":
        lo, hi = chance - 0.03, ts2vec + 0.05
    else:
        return None
    return (max(0.0, float(lo)), min(1.0, float(hi)))


def band_of(dcfg: Mapping, ts2vec: Optional[float]) -> Optional[tuple[float, float]]:
    if dcfg.get("band"):
        lo, hi = dcfg["band"]
        return (float(lo), float(hi))
    if ts2vec is None or dcfg.get("chance") is None:
        return None
    return default_band(dcfg["class"], float(ts2vec), float(dcfg["chance"]))


def noise_sd(p: float, n: int) -> float:
    """计数噪声 sd = sqrt(p(1−p)/n_test)。"""
    return float(math.sqrt(max(float(p) * (1.0 - float(p)), 0.0) / int(n))) if int(n) > 0 else float("nan")


# ---------------------------------------------------------------- 数据（单槽缓存 + Xtr_fit）
_SLOT: dict = {}


def reset_slot() -> None:
    _SLOT.clear()


def current_slot_dataset() -> Optional[str]:
    return _SLOT.get("dataset")


def uea30_arrays(dataset: str, cfg: Mapping, *, cache_dir=None, load: Callable = load_uea_pair) -> dict:
    """只走 scaler=train 一路（ts2vec_official 只用 Xte_train；绕开 35.load_arrays 对 ragged / 类集的双 scaler 断言）。
    单槽缓存：换靶即释放上一靶（替代 35._ARRAYS 的无淘汰缓存）。yaml 有 max_train_length 的靶多一份 Xtr_fit。"""
    if _SLOT.get("dataset") == dataset:
        return _SLOT["arrays"]
    _SLOT.clear()
    kw = {"scaler": "train"}
    if cache_dir is not None:
        kw["cache_dir"] = cache_dir
    Xtr, ytr, Xte, yte = load(dataset, **kw)
    arrays = {"Xtr": Xtr, "ytr": ytr, "Xte_train": Xte, "Xte_split": Xte, "yte": yte}
    m = ((cfg.get("datasets") or {}).get(dataset) or {}).get("max_train_length")
    if m:
        arrays["Xtr_fit"] = segment_train(Xtr, int(m))
    _SLOT.update({"dataset": dataset, "arrays": arrays})
    fit = f" fit={arrays['Xtr_fit'].shape}" if "Xtr_fit" in arrays else ""
    print(f"[data] {dataset}: train={Xtr.shape} test={Xte.shape} classes={len(np.unique(ytr))} size={Xtr.size}{fit}",
          flush=True)
    return arrays


def make_loader(cfg: Mapping, cache_dir=None, load: Callable = load_uea_pair) -> Callable:
    """39.execute 的 loader 形态：loader(dataset, har_npz) → arrays。"""
    return lambda dataset, har_npz=None: uea30_arrays(dataset, cfg, cache_dir=cache_dir, load=load)


# ---------------------------------------------------------------- OOM 重复剔除
def drop_repeated_oom(cells: Sequence[dict], oom_log: str | Path, max_attempts: int = 2) -> list[dict]:
    """同 key 在 oom_failures.jsonl 里出现 ≥ max_attempts 次 → 剔出工单（防 --resume 死循环），打印剔除清单。"""
    p = Path(oom_log)
    if not p.exists():
        return list(cells)
    counts = Counter(r["key"] for r in read_jsonl(p) if "key" in r)
    kept, dropped = [], []
    for c in cells:
        (dropped if counts.get(c["key"], 0) >= int(max_attempts) else kept).append(c)
    if dropped:
        print(f"[oom] 剔除 {len(dropped)} 格（OOM ≥ {max_attempts} 次）："
              + ", ".join(f"{c.get('dataset')}/{c['key']}" for c in dropped), flush=True)
    return kept


# ---------------------------------------------------------------- 臂
def uea30_arms(keys_off: Sequence[str], twin_map: Mapping[str, str], traj: Mapping[str, Mapping[int, float]],
               prior_order: Sequence[str], *, budgets: Sequence[int],
               seconds: Optional[Mapping[str, Mapping[int, float]]] = None, random_seed: int = 0) -> dict:
    """一个靶的全部臂（PROTOCOL §2）：oracle22/off/on、prior1/3、random1、SH6-3-3_prior、SH10-4-3_prior（主）、
    SH20-4-3_both、SH20-3_both（副：两级从 50% 起）。名单 = 先验序（只在 legacy 上）前 10，两版 = interleave 孪生。"""
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    keys_off = [k for k in keys_off if all(b in traj.get(k, {}) for b in stages)]
    twin_map = {k: t for k, t in twin_map.items() if k in keys_off and all(b in traj.get(t, {}) for b in stages)}
    on_keys = sorted(twin_map.values())
    keys = list(keys_off) + on_keys
    truth = {k: float(traj[k][bf]) for k in keys}
    arms: dict[str, Optional[float]] = {
        "oracle22": max(truth.values()),
        "oracle_off": max(truth[k] for k in keys_off),
        "oracle_on": (max(truth[t] for t in on_keys) if on_keys else None),
    }
    cost: dict[str, int] = {}
    picks: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    secs: Optional[dict[str, float]] = {} if seconds is not None else None
    ranked_off = [k for k in prior_order if k in truth and k not in set(on_keys)]

    def zero_shot(name: str, top: int) -> None:
        chosen = ranked_off[:top]
        if not chosen:
            return
        best = sorted(chosen, key=lambda k: (-truth[k], k))[0]
        arms[name], cost[name], picks[name], lists[name] = truth[best], top * bf, best, list(chosen)
        if secs is not None:
            secs[name] = float(sum(float(seconds[k][bf]) for k in chosen))

    def sh(name: str, ranked: Sequence[str], ks: Sequence[int], bud: Sequence[int]) -> None:
        ranked = [k for k in ranked if k in truth]
        if not ranked:
            return
        path = _g36.sh_path(ranked, traj, ks, bud)
        pick = _g36.sh_pick(path[-1], traj, bf)
        arms[name], picks[name], lists[name] = truth[pick], pick, list(ranked)
        cost[name] = _g36.path_cost_iters(path, bud)
        if secs is not None:
            secs[name] = _g36.path_cost_seconds(path, bud, seconds)

    zero_shot("prior1", 1)
    zero_shot("prior3", K_TOP)
    arms["random1"] = float(np.mean([truth[k] for k in keys_off]))
    cost["random1"], cost["catalog_direct"] = bf, len(keys) * bf
    sh("SH6-3-3_prior", ranked_off[:SH_REF6[0]], SH_REF6, stages)
    sh(MAIN_ARM, ranked_off[:TOP_LIST], SH_MAIN, stages)
    mix = _g40.interleave(ranked_off[:TOP_LIST], twin_map)
    sh("SH20-4-3_both", mix, (len(mix), SH_REF20[1], SH_REF20[2]), stages)
    sh(SECOND_ARM, mix, (len(mix), SH_SECOND[1]), stages[1:])
    regret = {k: (arms["oracle22"] - v if v is not None else None) for k, v in arms.items()}
    on_set = set(on_keys)
    is_twin = {name: (picks[name] in on_set) for name in picks}
    return {"arms": arms, "regret": regret, "cost_iters": cost, "cost_seconds": secs, "picks": picks,
            "lists": lists, "is_twin": is_twin, "n_candidates": len(keys), "n_on": len(on_keys), "budgets": stages}


# ---------------------------------------------------------------- 完整性 / 3 种子
def completeness(rows_d: Sequence[dict], cand_d: Mapping) -> dict:
    """11 legacy + 11 孪生（三截点齐全、无 error）才 complete；error 行与缺格列出（名单静默变短的可见性）。"""
    expected = [e["cls_key"] for e in cand_d["shared"]] + [cand_d["ggs"]["cls_key"]]
    exp_set = set(expected)
    ok_legacy: set[str] = set()
    ok_twin: set[str] = set()
    errors: list[dict] = []
    for r in rows_d:
        if r.get("stage", "s1_catalog") != "s1_catalog" or int(r.get("seed", 0)) != 0:
            continue
        if r.get("error"):
            errors.append({"cls_key": r["cls_key"], "twin_of": r.get("twin_of"), "error": r["error"]})
            continue
        n_ck = len(r.get("checkpoints") or []) or N_CKPT
        if len(r.get("checkpoint_results") or []) != n_ck:
            continue
        if r.get("twin_of"):
            if r["twin_of"] in exp_set:
                ok_twin.add(r["twin_of"])
        elif r["cls_key"] in exp_set:
            ok_legacy.add(r["cls_key"])
    n = len(expected)
    return {"n_expected": n, "n_legacy_ok": len(ok_legacy), "n_twin_ok": len(ok_twin),
            "complete": bool(len(ok_legacy) == n and len(ok_twin) == n), "errors": errors,
            "missing_legacy": [k for k in expected if k not in ok_legacy],
            "missing_twin": [k for k in expected if k not in ok_twin]}


def three_seed_stats(k: Optional[str], d: str, acc0: Optional[float], s2_rows: Sequence[dict]) -> Optional[dict]:
    """seed 0 + finals（seeds 1/2）：与 results/mx4_mask/paper_table.py 同口径（mean / 总体 std / n）。"""
    if k is None or acc0 is None:
        return None
    extra = [float(r["acc"]) for r in s2_rows if r["dataset"] == d and r["cls_key"] == k and r.get("error") is None]
    if len(extra) < len(FINAL_SEEDS):
        return None
    v = [float(acc0)] + extra
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def analyze_target(d: str, rows_d: Sequence[dict], cand: Mapping, s2_rows: Sequence[dict], *, klass: str,
                   paper: Optional[float], band: Optional[tuple[float, float]], chance: Optional[float] = None,
                   n_test: Optional[int] = None) -> Optional[dict]:
    """一个靶：轨迹 → 臂 → 3 种子 → 完整性。rows_d 含 legacy 与孪生（twin_of）行。"""
    s1 = [r for r in rows_d if r.get("stage") == "s1_catalog" and int(r.get("seed", 0)) == 0]
    ok = [r for r in s1 if r.get("error") is None and r.get("checkpoint_results")]
    traj, secs = _g39.trajectory_from_rows(ok)
    legacy_rows = [r for r in ok if not r.get("twin_of")]
    twin_rows = [r for r in ok if r.get("twin_of")]
    budgets = sorted({b for r in legacy_rows for b in (traj.get(r["cls_key"]) or {})})
    if not budgets:
        return None
    keys_off = sorted(k for k in {r["cls_key"] for r in legacy_rows} if all(b in traj.get(k, {}) for b in budgets))
    twin_map = {r["twin_of"]: r["cls_key"] for r in twin_rows
                if r["twin_of"] in keys_off and all(b in traj.get(r["cls_key"], {}) for b in budgets)}
    prior = {k: float(v) for k, v in cand["prior"].items()}
    prior_order = _g36.rank_by(prior, keys_off)
    row = uea30_arms(keys_off, twin_map, traj, prior_order, budgets=budgets, seconds=secs)
    bf = budgets[-1]
    comp = completeness(rows_d, cand)

    def stats(arm: str) -> tuple[Optional[str], Optional[float], Optional[dict]]:
        k = row["picks"].get(arm)
        acc0 = row["arms"].get(arm)
        return k, acc0, three_seed_stats(k, d, acc0, s2_rows)

    ck, c0, cs = stats(MAIN_ARM)
    sk, s0, ss = stats(SECOND_ARM)
    f1_final = {r["cls_key"]: float(r["checkpoint_results"][-1].get("f1") or 0.0) for r in ok}
    acc_final = {k: traj[k][bf] for k in keys_off + sorted(twin_map.values())}
    delta = None
    if cs is not None and ss is not None:
        delta = ss["mean"] - cs["mean"]
    elif c0 is not None and s0 is not None:
        delta = float(s0) - float(c0)
    all_keys = list(acc_final)
    rk25 = _g36.rank_by({k: traj[k][budgets[0]] for k in all_keys}, all_keys)
    rk100 = _g36.rank_by(acc_final, all_keys)
    rho_25_100 = (float(spearmanr([rk25.index(k) for k in all_keys], [rk100.index(k) for k in all_keys]).correlation)
                  if len(all_keys) >= 3 else None)
    return {
        "class": klass, "paper_ts2vec": paper, "band": (list(band) if band else None), "chance": chance,
        "n_test": n_test, "noise_sd": (noise_sd(paper, n_test) if paper is not None and n_test else None),
        "budgets": budgets, "n_candidates": row["n_candidates"], "n_on": row["n_on"],
        "complete": comp["complete"], "completeness": comp,
        "champion_key": ck, "champion_seed0": c0,
        "champion_mean": (cs["mean"] if cs else None), "champion_std": (cs["std"] if cs else None),
        "champion_n": (cs["n"] if cs else None), "champion_is_twin": (row["is_twin"].get(MAIN_ARM) if ck else None),
        "champion_f1_seed0": (f1_final.get(ck) if ck else None),
        "secondary_key": sk, "secondary_seed0": s0,
        "secondary_mean": (ss["mean"] if ss else None), "secondary_std": (ss["std"] if ss else None),
        "secondary_n": (ss["n"] if ss else None), "secondary_is_twin": (row["is_twin"].get(SECOND_ARM) if sk else None),
        "delta_secondary_minus_main": delta, "spearman_rank25_vs_100": rho_25_100,
        "ref": {a: {"key": row["picks"].get(a), "seed0": row["arms"].get(a),
                    "is_twin": row["is_twin"].get(a),
                    "mean": (three_seed_stats(row["picks"].get(a), d, row["arms"].get(a), s2_rows) or {}).get("mean")}
                for a in REF_ARMS},
        "catalog_max": max(acc_final.values()), "catalog_max_key": max(acc_final, key=lambda k: (acc_final[k], k)),
        "arms": row["arms"], "regret": row["regret"], "cost_iters": row["cost_iters"], "cost_seconds": row["cost_seconds"],
        "picks": row["picks"], "lists": row["lists"], "is_twin": row["is_twin"],
    }


def reused_six_targets(*, wp3: Path = WP3, wp6: Path = WP6, cfg: Optional[Mapping] = None) -> dict:
    """6 个已跑靶：WP3（legacy）+ WP6（孪生 + finals）缓存上用同一套臂复算（与 paper_table.md 对账）。"""
    off = read_jsonl(wp3 / "runs.jsonl")
    on = read_jsonl(wp6 / "runs.jsonl")
    cands = _g39.load_candidates(wp6)
    s2 = [r for r in off + on if r.get("stage") == "s2_finals" and r.get("error") is None]
    per = {}
    for d in _g39.TARGETS:
        rows_d = [r for r in off + on if r.get("stage") == "s1_catalog" and r["dataset"] == d]
        dcfg = ((cfg or {}).get("datasets") or {}).get(d) or {}
        per[d] = analyze_target(d, rows_d, cands[d], s2, klass=dcfg.get("class", "E"), paper=paper_ts2vec(d, cfg),
                                band=None, chance=dcfg.get("chance"), n_test=dcfg.get("n_test"))
    return per


def har_summary(har_dir: Path = HAR_DIR) -> Optional[dict]:
    """UCI-HAR 外部锚（WP3b，legacy_full，SH6-3-3）：冠军 3 种子 mean / std。"""
    a_path = har_dir / "analysis.json"
    if not a_path.exists():
        return None
    a = json.loads(a_path.read_text(encoding="utf-8"))
    v = a["per_target"]["HAR"]
    std = None
    runs = har_dir / "runs.jsonl"
    if runs.exists() and v.get("champion_key"):
        rows = read_jsonl(runs)
        accs = [float(r["acc"]) for r in rows if r.get("error") is None and r["cls_key"] == v["champion_key"]
                and r.get("acc") is not None and (r.get("stage") == "s2_finals" or int(r.get("seed", 0)) == 0)]
        std = float(np.std(accs)) if len(accs) >= 3 else None
    return {"champion_key": v.get("champion_key"), "champion_mean": v.get("champion_mean"), "champion_std": std,
            "paper_ts2vec": v.get("paper_ts2vec"), "arm": _g39.MAIN_ARM, "protocol": "legacy_full"}


# ---------------------------------------------------------------- 门
def run_gates_uea30(per: Mapping[str, dict], cfg: Mapping) -> dict:
    """U1（E 类追平 ≥ u1_min）/ U2（A/B/D 带内率 ≥ u2_min_rate）/ U3（副臂配对 Δ）；不完整靶不入分母；C 类只报。"""
    g = cfg["gates"]
    margin = float(g["parity_margin"])
    comp = {d: v for d, v in per.items() if v.get("complete")}
    incomplete = sorted(d for d, v in per.items() if not v.get("complete"))
    E = {d: v for d, v in comp.items() if v.get("class") == "E"}
    parity = {d: bool(v.get("champion_mean") is not None and v.get("paper_ts2vec") is not None
                      and float(v["champion_mean"]) >= float(v["paper_ts2vec"]) - margin) for d, v in E.items()}
    u1 = {"n_parity": int(sum(parity.values())), "n": len(E), "parity": parity,
          "pass": bool(sum(parity.values()) >= int(g["u1_min"]))}
    ABD = {d: v for d, v in comp.items() if v.get("class") in ("A", "B", "D") and v.get("band")}
    inband = {d: bool(v.get("champion_mean") is not None
                      and float(v["band"][0]) <= float(v["champion_mean"]) <= float(v["band"][1])) for d, v in ABD.items()}
    u2 = {"n_in_band": int(sum(inband.values())), "n": len(ABD), "in_band": inband,
          "pass": bool(ABD) and (sum(inband.values()) / len(ABD) >= float(g["u2_min_rate"]))}
    deltas = {d: float(v["secondary_mean"]) - float(v["champion_mean"]) for d, v in comp.items()
              if v.get("secondary_mean") is not None and v.get("champion_mean") is not None}
    within = {d: bool(x >= -float(g["u3_tol"])) for d, x in deltas.items()}
    mean_delta = float(np.mean(list(deltas.values()))) if deltas else None
    u3 = {"n": len(deltas), "mean_delta": mean_delta, "deltas": deltas, "n_within": int(sum(within.values())),
          "pass": bool(deltas) and mean_delta >= 0.0 and (sum(within.values()) / len(deltas) >= float(g["u3_min_rate"]))}
    return {"U1": u1, "U2": u2, "U3": u3,
            "C": {"datasets": sorted(d for d, v in per.items() if v.get("class") == "C")},
            "incomplete": incomplete}


def verdict_code_uea30(g: Mapping[str, dict]) -> str:
    if not g["U1"]["pass"]:
        code = "UEA30-DEAD"
    else:
        code = "UEA30-WIN" if g["U2"]["pass"] else "UEA30-PARTIAL"
    return code + ("+TWIN" if g["U3"]["pass"] else "")


# ---------------------------------------------------------------- 估时（锚点带硬件标签）
def estimate_hours(cfg: Mapping, datasets: Sequence[str], *, launch_s: float = LAUNCH_S, n_cells: int = N_CELLS,
                   n_ckpt: int = N_CKPT, finals_cells: int = FINALS_CELLS_UPPER,
                   conservative: float = CONSERVATIVE) -> dict:
    per = {}
    for d in datasets:
        a = cfg["datasets"][d]["anchor_4090"]
        pre, ev = float(a["pretrain_s"]), float(a["eval_s"])
        per[d] = {"n_cells": int(n_cells), "pretrain_s": pre, "eval_s": ev,
                  "catalog_h": (n_cells * (pre + launch_s) + n_cells * n_ckpt * ev) / 3600.0,
                  "finals_h": finals_cells * (pre + n_ckpt * ev) / 3600.0}
    tc = float(sum(v["catalog_h"] for v in per.values()))
    tf = float(sum(v["finals_h"] for v in per.values()))
    return {"hardware": HARDWARE, "per_dataset": per, "total_catalog_h": tc, "total_finals_h": tf,
            "conservative_h": float(conservative) * (tc + tf), "conservative_factor": float(conservative)}


# ---------------------------------------------------------------- 候选目录（零 GPU，协议锁产物）
def build_candidates(cfg: Mapping, outdir: str | Path, datasets: Optional[Sequence[str]] = None, *,
                     mx3: Path = MX3) -> dict:
    """每靶 shared-10 + GGS 的先验（39.prior_with_ggs 同码）：非池靶全 8 源；池内靶（SCP2）LODO；不提名。"""
    ds = list(datasets) if datasets is not None else run_targets(cfg)
    rows = [r for r in read_jsonl(mx3 / "rprime_cache.jsonl") if r.get("error") is None]
    cat = [r for r in rows if r["origin"] != "ggs"]
    ggs_rows = [r for r in rows if r["origin"] == "ggs"]
    _picks, _key2cls, catalog = _g35.load_mx3_picks()
    pool = {r["dataset"] for r in cat}
    shared = [{"cls_key": e["cls_key"], "cls": e["cls"]} for e in catalog["shared"]]
    if isinstance(catalog.get("ggs"), dict) and "cls_key" in catalog["ggs"]:
        ggs_entry = {"cls_key": catalog["ggs"]["cls_key"], "cls": catalog["ggs"]["cls"]}
    else:
        g = _g35._m1.GGS
        ggs_entry = {"cls_key": cls_key(g), "cls": g.to_dict()}
    keys = [e["cls_key"] for e in shared] + [ggs_entry["cls_key"]]
    out = {}
    for d in ds:
        exclude_ds = d if d in pool else None
        train_cat = [r for r in cat if r["dataset"] != exclude_ds]
        train_ggs = [r for r in ggs_rows if r["dataset"] != exclude_ds]
        prior = _g39.prior_with_ggs(train_cat, train_ggs)
        out[d] = {"shared": shared, "ggs": ggs_entry, "nominated": [],
                  "prior": {k: round(v, 6) for k, v in prior.items()},
                  "prior_order": _g36.rank_by(prior, keys), "tclsc_order": [],
                  "fit_on": ("all 8 source datasets" if exclude_ds is None else f"LODO excluding {d}")}
        print(f"[candidates {d}] 拟合于 {out[d]['fit_on']}；先验前 3 {out[d]['prior_order'][:3]}", flush=True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 表
def _f(x, nd: int = 3) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def _ms(mean, std, n=None, twin=False) -> str:
    if mean is None:
        return "—"
    s = f"{float(mean):.3f}" + (f" ± {float(std):.3f}" if std is not None else "")
    if n is not None and n != 3:
        s += f"（{n} 种子）"
    return s + ("†" if twin else "")


def markdown_tables(a: Mapping) -> str:
    per = a.get("per_target") or {}
    L = ["## 靶外头条表（UEA 官方划分 test acc；主臂 SH10-4-3_prior 掩码关 / 副臂 SH20-3_both 两版都量）", "",
         "| 靶 | 类 | n_test | 论文 TS2Vec | 预测带 | oracle22 / off / on | 主臂 s0 / 3 种子 | 副臂 s0 / 3 种子 | Δ 副−主 | 完整 | 追平 | 带内 |",
         "|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---|"]
    margin = float(a.get("parity_margin", 0.03))
    for d, v in per.items():
        ar = v["arms"]
        cm = v["champion_mean"] if v["champion_mean"] is not None else v["champion_seed0"]
        par = ("—" if v.get("paper_ts2vec") is None or cm is None
               else ("✓" if cm >= v["paper_ts2vec"] - margin else "✗"))
        band = v.get("band")
        inb = ("—" if not band or cm is None else ("✓" if band[0] <= cm <= band[1] else "✗"))
        L.append(f"| {d} | {v['class']} | {v.get('n_test') or '—'} | {_f(v.get('paper_ts2vec'))} | "
                 f"{('[%.3f, %.3f]' % tuple(band)) if band else '—'} | {_f(ar['oracle22'])} / {_f(ar['oracle_off'])} / {_f(ar['oracle_on'])} | "
                 f"{_f(v['champion_seed0'])} / {_ms(v['champion_mean'], v['champion_std'], v['champion_n'], bool(v.get('champion_is_twin')))} | "
                 f"{_f(v['secondary_seed0'])} / {_ms(v['secondary_mean'], v['secondary_std'], v['secondary_n'], bool(v.get('secondary_is_twin')))} | "
                 f"{'—' if v.get('delta_secondary_minus_main') is None else '%+.3f' % v['delta_secondary_minus_main']} | "
                 f"{'✓' if v['complete'] else '✗（legacy %d/%d，孪生 %d/%d）' % (v['completeness']['n_legacy_ok'], v['completeness']['n_expected'], v['completeness']['n_twin_ok'], v['completeness']['n_expected'])} | {par} | {inb} |")
    arms = ["oracle22", "oracle_off", "oracle_on", "prior1", "prior3", "random1", "SH6-3-3_prior", MAIN_ARM,
            "SH20-4-3_both", SECOND_ARM]
    L += ["", "## 各靶臂表（seed 0，acc@100%；成本 iters 见 analysis.json）", "",
          "| 靶 | " + " | ".join(arms) + " |", "|---|" + "|".join(["---:"] * len(arms)) + "|"]
    for d, v in per.items():
        L.append(f"| {d} | " + " | ".join(_f(v["arms"].get(x)) for x in arms) + " |")
    g = a.get("gates")
    if g:
        L += ["", "## 门（PROTOCOL §3）", "",
              f"- U1：E 类追平 {g['U1']['n_parity']}/{g['U1']['n']} → {'✓' if g['U1']['pass'] else '✗'}",
              f"- U2：A/B/D 带内 {g['U2']['n_in_band']}/{g['U2']['n']} → {'✓' if g['U2']['pass'] else '✗'}",
              f"- U3：副臂 Δ 均值 {_f(g['U3']['mean_delta'])}，Δ ≥ −tol 于 {g['U3']['n_within']}/{g['U3']['n']} → {'✓' if g['U3']['pass'] else '✗'}",
              f"- C 类只报：{', '.join(g['C']['datasets']) or '—'}；不完整靶：{', '.join(g['incomplete']) or '—'}",
              "", f"读数码：**{a.get('verdict')}**（{a.get('verdict_note', '')}）"]
    return "\n".join(L) + "\n"


def table30_rows(cfg: Mapping, per_new: Mapping[str, dict], per_six: Mapping[str, dict],
                 har: Optional[Mapping]) -> list[dict]:
    """UEA 30 行（按 yaml 顺序）+ HAR 外部锚 1 行。"""
    rows = []
    for d, c in cfg["datasets"].items():
        if d == "HAR":
            continue
        klass = c.get("class")
        paper = paper_ts2vec(d, cfg)
        n_test = c.get("n_test")
        src = per_new.get(d) or per_six.get(d)
        base = {"dataset": d, "class": klass, "n_test": n_test, "paper_ts2vec": paper, "lit": paper_reference(d, cfg),
                "sd": (noise_sd(paper, n_test) if paper is not None and n_test else None),
                "band": c.get("band"), "evaluable": bool(klass == "E" or d in EVALUABLE_SIX),
                "run": bool(src is not None), "source": (c.get("reuse") if d in per_six else ("uea30" if d in per_new else None)),
                "main_mean": None, "main_std": None, "secondary_mean": None, "secondary_std": None,
                "secondary_is_twin": None, "oracle22": None, "complete": None, "note": ""}
        if src is None:
            base["note"] = "未跑" + (f"：{c['reason']}" if c.get("reason") else "")
        else:
            base.update({"main_mean": (src.get("champion_mean") if src.get("champion_mean") is not None else src.get("champion_seed0")),
                         "main_std": src.get("champion_std"), "secondary_mean": (src.get("secondary_mean") if src.get("secondary_mean") is not None else src.get("secondary_seed0")),
                         "secondary_std": src.get("secondary_std"), "secondary_is_twin": src.get("secondary_is_twin"),
                         "oracle22": (src.get("arms") or {}).get("oracle22"), "complete": src.get("complete")})
            if d in per_six:
                base["note"] = "复用 WP3/WP6 缓存（LODO 靶）"
        pm = base["main_mean"]
        base["parity_main"] = (None if pm is None or paper is None else bool(pm >= paper - 0.03))
        sm = base["secondary_mean"]
        base["parity_secondary"] = (None if sm is None or paper is None else bool(sm >= paper - 0.03))
        band = c.get("band")
        base["in_band"] = (None if not band or pm is None else bool(band[0] <= pm <= band[1]))
        rows.append(base)
    hc = (cfg["datasets"].get("HAR") or {})
    if har is not None:
        rows.append({"dataset": "HAR", "class": hc.get("class", "E"), "n_test": hc.get("n_test"),
                     "paper_ts2vec": har.get("paper_ts2vec"), "lit": paper_reference("HAR", cfg),
                     "sd": (noise_sd(har["paper_ts2vec"], hc["n_test"]) if har.get("paper_ts2vec") and hc.get("n_test") else None),
                     "band": None, "evaluable": True, "run": True, "source": hc.get("reuse", "mx4_deploy_har"),
                     "main_mean": har.get("champion_mean"), "main_std": har.get("champion_std"),
                     "secondary_mean": None, "secondary_std": None, "secondary_is_twin": None, "oracle22": None,
                     "complete": True, "note": f"外部锚，{har.get('protocol', 'legacy_full')} 口径（WP3b，{har.get('arm')}），无副臂",
                     "parity_main": (None if har.get("champion_mean") is None or har.get("paper_ts2vec") is None
                                     else bool(har["champion_mean"] >= har["paper_ts2vec"] - 0.03)),
                     "parity_secondary": None, "in_band": None})
    return rows


def table30_means(rows: Sequence[dict]) -> dict:
    """三种均值口径：全 30（已跑者）/ 剔 C / 可评集（E ∪ 6 LODO 靶 ∪ HAR）。"""
    def agg(sel):
        sel = [r for r in sel if r["run"] and r["main_mean"] is not None]
        sec = [r for r in sel if r["secondary_mean"] is not None]
        ts = [r for r in sel if r["paper_ts2vec"] is not None]
        return {"n": len(sel), "main": (float(np.mean([r["main_mean"] for r in sel])) if sel else None),
                "n_secondary": len(sec), "secondary": (float(np.mean([r["secondary_mean"] for r in sec])) if sec else None),
                "ts2vec": (float(np.mean([r["paper_ts2vec"] for r in ts])) if ts else None),
                "n_parity_main": int(sum(bool(r.get("parity_main")) for r in sel)),
                "n_parity_secondary": int(sum(bool(r.get("parity_secondary")) for r in sec))}
    uea = [r for r in rows if r["dataset"] != "HAR"]
    return {"all": agg(uea), "excl_C": agg([r for r in uea if r["class"] != "C"]),
            "evaluable": agg([r for r in rows if r.get("evaluable")])}


def table30_markdown(rows: Sequence[dict], means: Mapping) -> str:
    n_uea = sum(1 for r in rows if r["dataset"] != "HAR")
    n_run = sum(1 for r in rows if r["dataset"] != "HAR" and r["run"])
    L = [f"## 全 UEA 30 表（官方划分 test acc；已跑 {n_run}/{n_uea}；† = 冠军为掩码开的孪生；HAR 为外部锚）", "",
         "| 数据集 | 类 | n_test | 噪声 sd | " + " | ".join(LIT_COLS) + " | SH10-4-3_prior（主，3 种子） | SH20-3_both（副，3 种子） | 目录 oracle22 | 追平 主/副 | 带内 | 备注 |",
         "|---|---|---:|---:|" + "---:|" * len(LIT_COLS) + "---:|---:|---:|---|---|---|"]
    for r in rows:
        lit = r.get("lit") or {}
        L.append(f"| {r['dataset']} | {r.get('class') or '—'} | {r.get('n_test') or '—'} | {_f(r.get('sd'))} | "
                 + " | ".join(_f(lit.get(c)) for c in LIT_COLS)
                 + f" | {_ms(r['main_mean'], r['main_std'])} | {_ms(r['secondary_mean'], r['secondary_std'], twin=bool(r.get('secondary_is_twin')))} | "
                 f"{_f(r.get('oracle22'))} | {'—' if r.get('parity_main') is None else ('✓' if r['parity_main'] else '✗')}/"
                 f"{'—' if r.get('parity_secondary') is None else ('✓' if r['parity_secondary'] else '✗')} | "
                 f"{'—' if r.get('in_band') is None else ('✓' if r['in_band'] else '✗')} | {r.get('note') or ''} |")
    L += ["", "## 均值（三种口径）", "", "| 口径 | n | TS2Vec 论文 | 主臂 | 副臂（n） | 追平 主/副 |", "|---|---:|---:|---:|---:|---|"]
    names = {"all": f"全 30（已跑 {n_run}/{n_uea}）", "excl_C": "剔 C 类（n_test ≤ 50）", "evaluable": "可评集（E ∪ 6 LODO 靶 ∪ HAR）"}
    for k, m in means.items():
        L.append(f"| {names[k]} | {m['n']} | {_f(m['ts2vec'])} | {_f(m['main'])} | {_f(m['secondary'])}（{m['n_secondary']}） | "
                 f"{m['n_parity_main']}/{m['n']} / {m['n_parity_secondary']}/{m['n_secondary']} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 子命令
def _protos() -> dict:
    return _g35.load_protocols()


def _select(cfg: Mapping, args) -> list[str]:
    ds = list(args.datasets) if args.datasets else run_targets(cfg)
    bad = [d for d in ds if d not in cfg["datasets"]]
    if bad:
        raise SystemExit(f"--datasets 含 configs/uea30.yaml 之外的名字：{bad}")
    return ds


def cmd_check(cfg: Mapping, datasets: Sequence[str], *, cache_dir=None) -> dict:
    p = _protos()[cfg["protocol"]]
    out = {}
    try:
        import psutil
        proc = psutil.Process()
    except Exception:  # noqa: BLE001
        proc = None
    peak = 0.0
    for d in datasets:
        c = cfg["datasets"][d]
        a = uea30_arrays(d, cfg, cache_dir=cache_dir)
        n, L, C = a["Xtr"].shape
        assert (n, L, C) == tuple(sizes_from_cfg(cfg, [d])[d]), f"{d}: 加载形状 {(n, L, C)} ≠ yaml {sizes_from_cfg(cfg, [d])[d]}"
        it = _g35.resolve_iters_from_size(p, n * L * C)
        assert it == int(c["iters"]), f"{d}: iters {it} ≠ yaml {c['iters']}"
        assert int(len(a["yte"])) == int(c["n_test"]), f"{d}: n_test {len(a['yte'])} ≠ yaml {c['n_test']}"
        _vals, cnt = np.unique(a["yte"], return_counts=True)
        chance = float(cnt.max() / len(a["yte"]))
        assert abs(chance - float(c["chance"])) < 1e-3, f"{d}: chance {chance:.4f} ≠ yaml {c['chance']}"
        fit = a.get("Xtr_fit")
        if c.get("max_train_length"):
            k = L // int(c["max_train_length"])
            assert fit is not None and fit.shape == (n * k, math.ceil(L / k), C), f"{d}: Xtr_fit {None if fit is None else fit.shape}"
        else:
            assert fit is None
        rss = (proc.memory_info().rss / 1e9) if proc else float("nan")
        peak = max(peak, rss) if proc else peak
        out[d] = {"shape": [int(n), int(L), int(C)], "n_test": int(len(a["yte"])), "iters": int(it), "chance": chance,
                  "xtr_fit": (list(fit.shape) if fit is not None else None), "rss_gb": rss}
        print(f"[check {d}] shape={(n, L, C)} n_test={len(a['yte'])} iters={it} chance={chance:.3f}"
              + (f" Xtr_fit={fit.shape}" if fit is not None else "") + (f" rss={rss:.2f} GB" if proc else ""), flush=True)
    reset_slot()
    print(f"[check] {len(out)} 靶全部通过；RSS 峰值 {peak:.2f} GB" if proc else f"[check] {len(out)} 靶全部通过")
    return out


def cmd_dry_run(cfg: Mapping, outdir: Path, datasets: Sequence[str]) -> dict:
    cands = _g39.load_candidates(outdir)
    sizes = sizes_from_cfg(cfg, datasets)
    cells = _g39.s1_worklist(_protos(), sizes, {d: cands[d] for d in datasets}, protocol=cfg["protocol"], twins=cfg["twin"])
    est = estimate_hours(cfg, datasets)
    for d in datasets:
        n = sum(c["dataset"] == d for c in cells)
        e = est["per_dataset"][d]
        print(f"  {d:<26} {n:>3} 格  iters={cfg['datasets'][d]['iters']:<4} 锚 {e['pretrain_s']:.1f}s+{N_CKPT}×{e['eval_s']:.1f}s"
              f" → catalog {e['catalog_h']:.2f} h，finals ≤ {e['finals_h']:.2f} h")
    print(f"S1 合计 {len(cells)} 格（{len(datasets)} 靶 × {N_CELLS}）；catalog {est['total_catalog_h']:.2f} h + finals ≤ "
          f"{est['total_finals_h']:.2f} h；保守 ×{est['conservative_factor']} = {est['conservative_h']:.1f} h（{est['hardware']} 锚，"
          f"含每格 {LAUNCH_S:.0f} s 启动）")
    est["n_cells"] = len(cells)
    return est


def smoke_cells(cfg: Mapping, cands: Mapping, datasets: Sequence[str], sizes: Mapping, iters: int) -> list[dict]:
    protos = _protos()
    p = protos[cfg["protocol"]]
    space = CLSSpace.from_yaml()
    cells = []
    for d in datasets:
        e = cands[d]["shared"][0]
        c = CLS(**e["cls"])
        cell = _g35._cell("s1_catalog", d, cfg["protocol"], p, sizes, arm_names=["shared"], cls_key_=e["cls_key"],
                          cls=c, seed=0, iters=iters)
        cell["checkpoints"] = list(_g39.checkpoints_for(iters))
        t = twin(c, **cfg["twin"])
        tcell = _g35._cell("s1_catalog", d, cfg["protocol"], p, sizes, arm_names=["shared"], cls_key_=cls_key(t),
                           cls=t, seed=0, overrides=space.encoder_overrides(t) or None, iters=iters)
        tcell["checkpoints"] = list(_g39.checkpoints_for(iters))
        tcell["twin_of"] = e["cls_key"]
        cells += [cell, tcell]
    return cells


def cmd_smoke(cfg: Mapping, args, datasets: Sequence[str]) -> bool:
    outdir = Path(args.outdir) / "smoke"
    cands = _g39.load_candidates(Path(args.outdir))
    sizes = sizes_from_cfg(cfg, datasets)
    cells = smoke_cells(cfg, cands, datasets, sizes, int(args.smoke_iters))
    args.outdir = str(outdir)
    _g39.execute(cells, args, "smoke", loader=make_loader(cfg, args.cache_dir))
    rows = read_jsonl(outdir / "runs.jsonl")
    want = len(_g39.checkpoints_for(int(args.smoke_iters)))
    ok = all(len(r.get("checkpoint_results") or []) == want and r.get("error") is None for r in rows) and len(rows) >= len(cells)
    tw = [r for r in rows if r.get("twin_of")]
    ok = ok and all(r["options"]["input_mask"] == cfg["twin"]["input_mask"] for r in tw)
    print(f"smoke {'OK' if ok else 'FAILED'}：{len(rows)} 行，截点 {[len(r.get('checkpoint_results') or []) for r in rows]}，"
          f"孪生 {len(tw)} 行")
    return ok


def cmd_catalog(cfg: Mapping, args, datasets: Sequence[str]) -> None:
    outdir = Path(args.outdir)
    cands = _g39.load_candidates(outdir)
    sizes = sizes_from_cfg(cfg, datasets)
    cells = _g39.s1_worklist(_protos(), sizes, {d: cands[d] for d in datasets}, protocol=cfg["protocol"], twins=cfg["twin"])
    cells = drop_repeated_oom(cells, outdir / "oom_failures.jsonl")
    _g39.execute(cells, args, "S1", loader=make_loader(cfg, args.cache_dir))


def cmd_analyze(cfg: Mapping, outdir: Path, datasets: Sequence[str]) -> dict:
    cands = _g39.load_candidates(outdir)
    rows = read_jsonl(outdir / "runs.jsonl") if (outdir / "runs.jsonl").exists() else []
    s2 = [r for r in rows if r.get("stage") == "s2_finals" and r.get("error") is None]
    per, picks = {}, {}
    for d in datasets:
        if d not in cands:
            continue
        c = cfg["datasets"][d]
        paper = paper_ts2vec(d, cfg)
        v = analyze_target(d, [r for r in rows if r["dataset"] == d], cands[d], s2, klass=c["class"], paper=paper,
                           band=band_of(c, paper), chance=c.get("chance"), n_test=c.get("n_test"))
        if v is None:
            print(f"[analyze {d}] 无 S1 行，跳过", flush=True)
            continue
        per[d] = v
        if v["champion_key"] and v["secondary_key"]:
            picks[d] = {MAIN_ARM: v["champion_key"], SECOND_ARM: v["secondary_key"]}
        print(f"[analyze {d}] {c['class']} 候选 {v['n_candidates']}（开 {v['n_on']}）完整 {v['complete']}；oracle22 {_f(v['arms']['oracle22'])} "
              f"主 {_f(v['champion_seed0'])}/{_f(v['champion_mean'])} 副 {_f(v['secondary_seed0'])}/{_f(v['secondary_mean'])}"
              f"{'†' if v['secondary_is_twin'] else ''}（论文 TS2Vec {_f(paper)}）", flush=True)
    per_eff = {d: {**v, "champion_mean": (v["champion_mean"] if v["champion_mean"] is not None else v["champion_seed0"]),
                   "secondary_mean": (v["secondary_mean"] if v["secondary_mean"] is not None else v["secondary_seed0"])}
               for d, v in per.items()}
    gates = run_gates_uea30(per_eff, cfg) if per else None
    n_full = sum(v["champion_mean"] is not None for v in per.values())
    analysis = {"protocol": "results/uea30/PROTOCOL.md", "config_md5": cfg_md5(), "parity_margin": cfg["gates"]["parity_margin"],
                "n_targets": len(per), "n_complete": int(sum(v["complete"] for v in per.values())),
                "n_targets_with_3_seeds": n_full, "per_target": per, "gates": gates,
                "verdict": (verdict_code_uea30(gates) if gates else None),
                "verdict_note": ("主门用 3 种子均值" if n_full == len(per) else f"主门暂用 seed 0（3 种子均值只有 {n_full}/{len(per)} 靶）")}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(analysis), encoding="utf-8")
    if gates:
        print(f"verdict {analysis['verdict']}（{analysis['verdict_note']}）；U1 {gates['U1']['n_parity']}/{gates['U1']['n']} "
              f"U2 {gates['U2']['n_in_band']}/{gates['U2']['n']} U3 Δ {_f(gates['U3']['mean_delta'])} "
              f"{gates['U3']['n_within']}/{gates['U3']['n']}；不完整 {gates['incomplete']}", flush=True)
    return analysis


def cmd_finals(cfg: Mapping, args, datasets: Sequence[str]) -> None:
    outdir = Path(args.outdir)
    p = outdir / "picks.json"
    if not p.exists():
        raise SystemExit(f"缺 {p}：先跑 --analyze")
    picks = json.loads(p.read_text(encoding="utf-8"))
    picks = {d: v for d, v in picks.items() if d in set(datasets) and v and all(v.values())}
    if not picks:
        print("[finals] 无可跑的冠军", flush=True)
        return
    cands = _g39.load_candidates(outdir)
    key2cls = {e["cls_key"]: e["cls"] for d in cands for e in cands[d]["shared"] + [cands[d]["ggs"]]}
    if (outdir / "runs.jsonl").exists():
        key2cls.update({r["cls_key"]: r["cls"] for r in read_jsonl(outdir / "runs.jsonl") if r.get("cls")})
    seeds = tuple(int(s) for s in args.seeds) if args.seeds else FINAL_SEEDS
    cells = _g39.finals_worklist(_protos(), sizes_from_cfg(cfg, list(picks)), picks, key2cls, seeds=seeds,
                                 protocol=cfg["protocol"])
    _g39.execute(cells, args, "S2", loader=make_loader(cfg, args.cache_dir))


def cmd_table30(cfg: Mapping, outdir: Path) -> dict:
    a_path = outdir / "analysis.json"
    per_new = json.loads(a_path.read_text(encoding="utf-8"))["per_target"] if a_path.exists() else {}
    per_six = reused_six_targets(cfg=cfg) if (WP3 / "runs.jsonl").exists() and (WP6 / "runs.jsonl").exists() else {}
    har = har_summary()
    rows = table30_rows(cfg, per_new, per_six, har)
    means = table30_means(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "table30.json").write_text(json.dumps({"rows": rows, "means": means}, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "table30.md").write_text(table30_markdown(rows, means), encoding="utf-8")
    print(f"[table30] {len(rows)} 行；均值 全 {_f(means['all']['main'])}（n={means['all']['n']}）/ 剔 C {_f(means['excl_C']['main'])} / "
          f"可评 {_f(means['evaluable']['main'])}（n={means['evaluable']['n']}）", flush=True)
    return {"rows": rows, "means": means}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-candidates", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--finals", action="store_true")
    ap.add_argument("--table30", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None, help="默认 configs/uea30.yaml 里 run: true 的靶（按 order）")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="finals 种子（默认 1 2）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--joblib-backend", default="threading", choices=("threading", "loky"))
    ap.add_argument("--cache-dir", default=None, help="UEA 缓存目录（默认 data/uea_cache）")
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--smoke-iters", type=int, default=SMOKE_ITERS)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    outdir = Path(args.outdir)
    datasets = _select(cfg, args)
    if args.build_candidates:
        build_candidates(cfg, outdir, datasets)
    if args.check:
        cmd_check(cfg, datasets, cache_dir=args.cache_dir)
    if args.dry_run:
        cmd_dry_run(cfg, outdir, datasets)
    if args.smoke:
        cmd_smoke(cfg, args, datasets)
        return
    if args.catalog:
        cmd_catalog(cfg, args, datasets)
    if args.analyze:
        cmd_analyze(cfg, outdir, datasets)
    if args.finals:
        cmd_finals(cfg, args, datasets)
    if args.table30:
        cmd_table30(cfg, outdir)
    if not any((args.build_candidates, args.check, args.dry_run, args.smoke, args.catalog, args.analyze, args.finals,
                args.table30)):
        ap.print_help()


if __name__ == "__main__":
    main()
