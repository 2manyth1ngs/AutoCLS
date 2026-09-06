"""MX-4 WP3：全量官方协议（UEA 官方划分，ts2vec_official）上的"零样本名单 + 逐次减半"部署。

协议：results/mx4_deploy/PROTOCOL.md（截点 {25,50,100}%、目录 shared-10 + GGS + 代理提名 10、W1–W4）。
复用 scripts/35_protocol_alignment.py 的协议装载 / 工单格 / 数据装载 / SVM 评估 / 缓存键，
以及 scripts/36_prior_sh_sim.py 的先验 / 逐次减半纯函数；训练走 cl_trainer 的 SWA × 截点
（截点评估 AveragedModel 的当前平均权重 = 训到该步的 SWA 终值）。

阶段（均写 results/mx4_deploy/runs.jsonl，按 key 缓存，--resume 可续）：
    python scripts/39_sh_deploy.py --nominate            # 零 GPU：candidates.json（先验 / 提名 / T-CLSC 序），协议锁产物
    python scripts/39_sh_deploy.py --dry-run             # 工单数 + 估时
    python scripts/39_sh_deploy.py --catalog --resume    # S1：六靶 × (10+1+10) × seed 0，三截点
    python scripts/39_sh_deploy.py --analyze             # 零 GPU：淘汰赛模拟 + 门 + 表（写 picks.json）
    python scripts/39_sh_deploy.py --finals --resume     # S2：冠军 × seeds {1,2}（读 picks.json）
    python scripts/39_sh_deploy.py --analyze             # 再跑一次把 3 种子均值并进头条表
    python scripts/39_sh_deploy.py --smoke [--device cpu]  # Handwriting × 3 格 × iters 8，端到端冒烟
"""
from __future__ import annotations

import argparse
import importlib
import json
import signal
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Collection, Mapping, Optional, Sequence

import numpy as np
import torch
from joblib import parallel_backend
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.downstream.linear_eval import encode_features  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import ORIGIN_SHARED, append_jsonl, cls_key, device_label  # noqa: E402
from autocls.oom_guard import RprimeOOM, is_oom_error  # noqa: E402
from autocls.search.surrogate import FieldSurrogate, field_hamming  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
zscore_within = _g36.zscore_within
rank_by = _g36.rank_by
sh_path = _g36.sh_path
sh_pick = _g36.sh_pick
path_cost_iters = _g36.path_cost_iters
path_cost_seconds = _g36.path_cost_seconds
expected_random_sh = _g36.expected_random_sh
features = _g36.features
expected_max_of_k = _g36.expected_max_of_k

OUT = ROOT / "results" / "mx4_deploy"
MX3 = ROOT / "results" / "mx3_pq"
PA_RUNS = ROOT / "results" / "protocol_alignment" / "runs.jsonl"
TARGETS = _g35.TARGETS
PROTOCOL = "ts2vec_official"
CKPT_FRACTIONS = (0.25, 0.5, 1.0)
N_NOMINEES = 10
NOMINEE_POOL = 2000
HULL_RADIUS = 3
NOMINEE_SEED = 0
N_PARENTS = 20                       # 变异父本 = 训练侧 z 均值最高的 20 条已评配置
MAX_MUT = 3                          # 每个候选变异 1..3 个字段
HYBRID_BIAS = 10.0
K_TOP = 3
SH_MAIN = (6, 3, 3)
N_RANDOM_DRAWS = 20
FINAL_SEEDS = (1, 2)
MAIN_ARM = "SH6-3-3_prior"
PARITY_MARGIN = _g35.PARITY_MARGIN
HW_LINE = 0.485                      # protocol-alignment P1 线（TS2Vec 0.515 − 0.03）
W1_MIN = 5
W2_MIN = 4
W3_RHO = 0.5
SMOKE_ITERS = 8
# ---- WP3b：UCI-HAR 外部锚（results/mx4_deploy_har/PROTOCOL.md）----
# AutoCL Table 2（CIKM'24）UCI-HAR acc；本栈 legacy_full 口径 GGS 锚 0.9348（results/m1_rerun）
HAR_TABLE2: dict[str, dict[str, float]] = {
    "HAR": {"TS2Vec": 0.930, "AutoCL": 0.963, "GGS": 0.937, "InfoTS": 0.930, "TS-TCC": 0.904, "CPC": 0.839},
}
HAR_GGS_ANCHOR = 0.9348
HAR_MARGIN = 0.005
SIZES_EXTRA = {"HAR": (7352, 128, 9)}
ANCHORS_EXTRA = {"HAR": (166.6, 20.5)}      # (pretrain s, eval s) @ legacy_full，4060，M1 锚
HAR_NPZ_DEFAULT = ROOT / "data" / "HAR" / "har.npz"


def paper_reference(dataset: str) -> dict[str, float]:
    if dataset in _g35.TABLE10:
        return dict(_g35.TABLE10[dataset])
    if dataset in HAR_TABLE2:
        return dict(HAR_TABLE2[dataset])
    raise KeyError(f"{dataset!r} 无文献参照表")


def paper_ts2vec(dataset: str) -> float:
    return float(paper_reference(dataset)["TS2Vec"])


def paper_median(dataset: str) -> float:
    return float(statistics.median(paper_reference(dataset).values()))


def har_arrays(npz_path) -> dict:
    """har.npz（train 统计量 z-score）→ 与 35 号 load_arrays 同形状的字典；两种 scaler 同一份 test。"""
    from autocls.data.datasets import load_har
    Xtr, ytr, Xte, yte = load_har(npz_path)
    return {"Xtr": Xtr, "ytr": ytr, "Xte_split": Xte, "Xte_train": Xte, "yte": yte}


_HAR_CACHE: dict[str, dict] = {}


def load_arrays_any(dataset: str, har_npz=None) -> dict:
    if dataset == "HAR":
        key = str(har_npz or HAR_NPZ_DEFAULT)
        if key not in _HAR_CACHE:
            _HAR_CACHE[key] = har_arrays(key)
            a = _HAR_CACHE[key]
            print(f"[data] HAR: train={a['Xtr'].shape} test={a['Xte_train'].shape} "
                  f"classes={len(np.unique(a['ytr']))}", flush=True)
        return _HAR_CACHE[key]
    return _g35.load_arrays(dataset)


def run_gates_har(per: Mapping[str, dict], *, ggs_anchor: float = HAR_GGS_ANCHOR,
                  margin: float = HAR_MARGIN) -> dict:
    """WP3b：H1 冠军 3 种子 ≥ GGS 锚 − margin；H2 ≥ 论文 TS2Vec；H3 seed 0 SH ≥ prior3。"""
    v = per["HAR"]
    champ = v["champion_mean"] if v.get("champion_mean") is not None else v["champion_seed0"]
    return {"H1": {"value": champ, "line": ggs_anchor - margin, "pass": bool(champ >= ggs_anchor - margin)},
            "H2": {"value": champ, "line": v["paper_ts2vec"], "pass": bool(champ >= v["paper_ts2vec"])},
            "H3": {"sh": v["sh_seed0"], "prior3": v["prior3_seed0"], "pass": bool(v["sh_seed0"] >= v["prior3_seed0"])}}


def verdict_code_har(g: Mapping[str, dict]) -> str:
    if not g["H1"]["pass"]:
        return "HAR-DEAD"
    return "HAR-WIN" if g["H3"]["pass"] else "HAR-PARITY"
# 单个截点评估的墙钟上限：官方 SVM 网格的 C=∞ 在退化特征（破坏性 CLS，如 20be4f74 × PhonemeSpectra
# 39 类 OvO）上可达数十分钟不收敛；超时整格记 error 行跳过（这类格都在排名垫底，不进任何名单）。
EVAL_TIMEOUT_S = 480


class EvalTimeout(Exception):
    """截点评估超过 EVAL_TIMEOUT_S（Unix SIGALRM；Windows 无 SIGALRM 时不设上限）。"""


def eval_with_timeout(fn, seconds: int):
    """在 seconds 秒内运行 fn()，超时抛 EvalTimeout；顺带杀掉 loky 复用池里仍在算的 worker。"""
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        return fn()

    def _alarm(_signum, _frame):
        raise EvalTimeout(f"checkpoint eval exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(seconds))
    try:
        return fn()
    except EvalTimeout:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(kill_workers=True)
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------- 纯函数
def checkpoints_for(iters: int, fractions: Sequence[float] = CKPT_FRACTIONS) -> tuple[int, ...]:
    """官方 iters × 分位 → 截点（严格递增、末位 = iters）。"""
    fr = [float(f) for f in fractions]
    if not fr or fr != sorted(fr) or fr[-1] != 1.0 or fr[0] <= 0:
        raise ValueError(f"fractions={fractions} 必须升序、>0 且以 1.0 结尾")
    ck = tuple(int(round(int(iters) * f)) for f in fr)
    if list(ck) != sorted(set(ck)) or ck[-1] != int(iters) or ck[0] < 1:
        raise ValueError(f"iters={iters} × {fractions} → {ck} 不是严格递增的合法截点")
    return ck


def nominate(space: CLSSpace, enc: CLSEncoder, rows: Sequence[dict], *, exclude_keys,
             n: int = N_NOMINEES, pool: int = NOMINEE_POOL, radius: int = HULL_RADIUS,
             seed: int = NOMINEE_SEED, n_parents: int = N_PARENTS, max_mut: int = MAX_MUT) -> list[dict]:
    """代理提名：训练侧（LODO）拟合 hgb；候选 = 源池里 z 均值最高的 n_parents 条配置各做 1..max_mut
    个字段的局部变异（门控字段随 space.yaml 约束联动），实测字段汉明 ≤ radius 才保留，取预测 z 前 n。

    为什么不均匀采样再筛半径：19 字段空间里随机点到 461 条已评配置的最小汉明距离通常 ≥ 8，
    半径 3 内几乎筛不出东西（WP2 的"目录一半在半径内"全是 shared-10 自己）；邻域只能由构造保证。
    """
    rows = list(rows)
    if not rows:
        return []
    max_b = max(int(r["iters"]) for r in rows)
    model = FieldSurrogate("hgb", seed=seed).fit(features(rows, enc, max_b),
                                                 np.array([float(r["z"]) for r in rows]))
    seen: dict[str, CLS] = {}
    zsum: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        seen.setdefault(r["cls_key"], CLS(**r["cls"]))
        zsum[r["cls_key"]].append(float(r["z"]))
    Xtr = np.stack([enc.encode(c) for c in seen.values()])
    zmean = {k: float(np.mean(v)) for k, v in zsum.items()}
    parents = [seen[k] for k in sorted(zmean, key=lambda k: (-zmean[k], k))[:int(n_parents)]]
    rng = np.random.default_rng(seed)
    exclude = set(exclude_keys)
    fields = list(space.field_names)
    cands: dict[str, CLS] = {}
    for _ in range(int(pool)):
        parent = parents[int(rng.integers(len(parents)))]
        d = 1 + int(rng.integers(int(max_mut)))
        names = [fields[i] for i in rng.choice(len(fields), size=d, replace=False)]
        values = {name: getattr(parent, name) for name in fields}
        for nm in names:
            ch = space.choices(nm)
            values[nm] = ch[int(rng.integers(len(ch)))]
        values.update(space.inactive_overrides(values))
        c = CLS(**values)
        if not space.validate(c):
            continue
        k = cls_key(c)
        if k in exclude or k in cands or k in seen:
            continue
        cands[k] = c
    if not cands:
        return []
    keys = list(cands)
    Xc = np.stack([enc.encode(cands[k]) for k in keys])
    ham = field_hamming(Xc, Xtr, enc.field_slices)
    ok = [i for i in range(len(keys)) if ham[i] <= int(radius)]
    if not ok:
        return []
    Xf = np.hstack([Xc[ok], np.ones((len(ok), 1))])          # 预算特征 = 1.0（b_final）
    pred, _ = model.predict(Xf)
    order = sorted(range(len(ok)), key=lambda j: (-float(pred[j]), keys[ok[j]]))[:int(n)]
    return [{"cls_key": keys[ok[j]], "cls": cands[keys[ok[j]]].to_dict(),
             "pred_z": float(pred[j]), "hamming": int(ham[ok[j]])} for j in order]


def prior_with_ggs(cat_rows: Sequence[dict], ggs_rows: Sequence[dict]) -> dict[str, float]:
    """先验 = shared 的 within-(task, iters) z 均值；GGS 的 z 按同组目录 20 条的 mean/sd 算。"""
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in cat_rows:
        groups[(r["task_id"], int(r["iters"]))].append(float(r["r_prime"]))
    stats = {g: (float(np.mean(v)), float(np.std(v))) for g, v in groups.items()}
    acc: dict[str, list[float]] = defaultdict(list)
    for r in list(cat_rows) + list(ggs_rows):
        g = (r["task_id"], int(r["iters"]))
        if g not in stats or r.get("origin") not in (ORIGIN_SHARED, "ggs"):
            continue
        mu, sd = stats[g]
        acc[r["cls_key"]].append((float(r["r_prime"]) - mu) / sd if sd > 0 else 0.0)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def trajectory_from_rows(rows: Sequence[dict]):
    """runs 行 → ({key: {iters: acc}}, {key: {iters: 累计秒}})；累计秒 = 训练墙钟 + 截止该点的评估耗时。"""
    traj: dict[str, dict[int, float]] = defaultdict(dict)
    secs: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r.get("error") or not r.get("checkpoint_results"):
            continue
        ev = 0.0
        for c in sorted(r["checkpoint_results"], key=lambda c: int(c["iters"])):
            b = int(c["iters"])
            ev += float(c.get("eval_seconds") or 0.0)
            traj[r["cls_key"]][b] = float(c["acc"])
            secs[r["cls_key"]][b] = float(c.get("elapsed_pretrain_s") or 0.0) + ev
    return dict(traj), dict(secs)


def deploy_arms(keys: Sequence[str], traj: Mapping[str, Mapping[int, float]],
                rank: Mapping[str, Sequence[str]], *, budgets: Sequence[int], random_seed: int,
                seconds: Optional[Mapping[str, Mapping[int, float]]] = None) -> dict:
    """一个靶的全部臂（PROTOCOL §2）：零样本 + 三级 / 两级 / 只留 1 条 / 纯多保真 / 随机起点。"""
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    keys = [k for k in keys if all(b in traj[k] for b in stages)]
    truth = {k: float(traj[k][bf]) for k in keys}
    vals = list(truth.values())
    oracle = max(vals)
    arms: dict[str, float] = {"oracle": oracle}
    cost: dict[str, int] = {}
    secs: Optional[dict[str, float]] = {} if seconds is not None else None
    picks: dict[str, str] = {}

    def zero_shot(name: str, ranked: Sequence[str], top: int) -> None:
        chosen = [k for k in ranked if k in truth][:top]
        if not chosen:
            return
        best = sorted(chosen, key=lambda k: (-truth[k], k))[0]
        arms[name] = truth[best]
        cost[name] = top * bf
        picks[name] = best
        if secs is not None:
            secs[name] = float(sum(float(seconds[k][bf]) for k in chosen))

    for arm in ("prior", "hybrid", "tclsc"):
        if arm in rank:
            zero_shot(arm + "1", rank[arm], 1)
            zero_shot(arm + "3", rank[arm], K_TOP)
    arms["random1"] = float(np.mean(vals))
    arms["random3"] = float(expected_max_of_k(vals, min(K_TOP, len(vals))))
    arms["bottom"] = float(min(vals))
    cost["random1"], cost["random3"], cost["catalog_direct"] = bf, K_TOP * bf, len(keys) * bf
    if secs is not None:
        mean_s = float(np.mean([float(seconds[k][bf]) for k in keys]))
        secs["random1"], secs["random3"] = mean_s, K_TOP * mean_s
        secs["catalog_direct"] = float(sum(float(seconds[k][bf]) for k in keys))

    def sh(name: str, ranked: Sequence[str], ks: Sequence[int], bud: Sequence[int]) -> None:
        ranked = [k for k in ranked if k in truth]
        if not ranked:
            return
        path = sh_path(ranked, traj, ks, bud)
        pick = sh_pick(path[-1], traj, bf)
        arms[name] = truth[pick]
        picks[name] = pick
        cost[name] = path_cost_iters(path, bud)
        if secs is not None:
            secs[name] = path_cost_seconds(path, bud, seconds)

    for warm in ("prior", "hybrid", "tclsc"):
        if warm not in rank:
            continue
        sh(f"SH6-3-3_{warm}", rank[warm], SH_MAIN, stages)
        sh(f"SH6-3-1_{warm}", rank[warm], (6, 3, 1), stages)
        sh(f"SH6-3_{warm}", rank[warm], (6, 3), stages[1:])          # 两级：跳过 25%
    arms["SH6-3-3_random"] = expected_random_sh(keys, traj, SH_MAIN, stages, N_RANDOM_DRAWS, random_seed)
    cost["SH6-3-3_random"] = path_cost_iters(sh_path(keys, traj, SH_MAIN, stages), stages)
    n = len(keys)
    for top in (1, K_TOP):
        sh(f"all@{stages[0]}->top{top}", keys, (n, top, top), stages)
    regret = {k: oracle - v for k, v in arms.items()}
    return {"arms": arms, "regret": regret, "cost_iters": cost, "cost_seconds": secs,
            "picks": picks, "n_candidates": n, "budgets": stages}


_SPACE_SINGLETON: Optional[CLSSpace] = None


def _space() -> CLSSpace:
    global _SPACE_SINGLETON
    if _SPACE_SINGLETON is None:
        _SPACE_SINGLETON = CLSSpace.from_yaml()
    return _SPACE_SINGLETON


def check_extension_consistency(cell: Mapping) -> None:
    """扩展字段有两处真源（CLS 字段 → build_encoder overrides；options.<encoder_kwarg> → cache_key）。

    两者必须一致，否则孪生格会以掩码关训练却顶着掩码开的键（或反之）——在任何训练前 fail-fast。
    """
    space = _space()
    cls = CLS(**cell["cls"])
    want = space.encoder_overrides(cls)
    opts = cell.get("options") or {}
    for f in space.extension_field_names:
        spec = space.field_spec(f)
        kw, default = spec["encoder_kwarg"], spec["default"]
        got, exp = opts.get(kw, default), want.get(kw, default)
        if got != exp:
            raise ValueError(f"格 {cell.get('key')}: CLS.{f}={getattr(cls, f)!r} 推出 {kw}={exp!r}，"
                             f"但 options.{kw}={got!r}——两处真源不一致")


def s1_worklist(protos: dict, sizes: dict, cands: Mapping[str, dict], *, seed: int = 0,
                protocol: str = PROTOCOL, twins: Optional[dict] = None, twin_only: bool = False,
                skip_error_keys: Collection[tuple[str, str]] = ()) -> list[dict]:
    """S1 目录格；``twins``（MX-4 WP6）时每条目录配一格孪生（CLS 扩展字段 + options 同步），
    legacy 格在前、孪生在后；``twin_only`` 只出孪生；``skip_error_keys`` 跳过 WP3 里 error 的
    (dataset, cls_key)（如 PhonemeSpectra × 20be4f74 的 eval 超时格）。不传 twins 时输出逐位不变。"""
    from autocls.space.cls_space import twin as _twin
    p = protos[protocol]
    skip = set(skip_error_keys)
    out, tw = [], []
    for d, c in cands.items():
        entries = ([("shared", e) for e in c["shared"]] + [("ggs", c["ggs"])]
                   + [("nominated", e) for e in c.get("nominated", [])])
        for arm, e in entries:
            cell = _g35._cell("s1_catalog", d, protocol, p, sizes, arm_names=[arm],
                              cls_key_=e["cls_key"], cls=CLS(**e["cls"]), seed=seed)
            cell["checkpoints"] = list(checkpoints_for(cell["iters"]))
            out.append(cell)
            if twins and (d, e["cls_key"]) not in skip:
                t = _twin(CLS(**e["cls"]), **twins)
                tcell = _g35._cell("s1_catalog", d, protocol, p, sizes, arm_names=[arm],
                                   cls_key_=cls_key(t), cls=t, seed=seed,
                                   overrides=_space().encoder_overrides(t) or None)
                tcell["checkpoints"] = list(checkpoints_for(tcell["iters"]))
                tcell["twin_of"] = e["cls_key"]
                tw.append(tcell)
    return tw if twin_only else out + tw


def finals_worklist(protos: dict, sizes: dict, picks: Mapping[str, Mapping[str, str]],
                    key2cls: Mapping[str, dict], *, seeds: Sequence[int] = FINAL_SEEDS,
                    protocol: str = PROTOCOL) -> list[dict]:
    """S2：每靶各臂冠军 × seeds；同一 CLS 被多个臂选中时合并成一格（arm_names 多个）。
    孪生冠军（CLS 扩展字段非默认）自动带 overrides（options 与 CLS 一致）；legacy → None，逐位不变。"""
    p = protos[protocol]
    out = []
    for d, arms in picks.items():
        by_key: dict[str, list[str]] = {}
        for arm, k in arms.items():
            by_key.setdefault(k, []).append("final_" + arm)
        for k, names in by_key.items():
            cls = CLS(**key2cls[k])
            ov = _space().encoder_overrides(cls) or None
            for s in seeds:
                cell = _g35._cell("s2_finals", d, protocol, p, sizes, arm_names=names,
                                  cls_key_=k, cls=cls, seed=int(s), overrides=ov)
                cell["checkpoints"] = list(checkpoints_for(cell["iters"]))
                out.append(cell)
    return out


def error_keys_from_runs(path: Path) -> set[tuple[str, str]]:
    """WP3 runs.jsonl 里 error 非空的 s1_catalog 格 → {(dataset, cls_key)}（孪生跳过它们）。"""
    if not Path(path).exists():
        return set()
    return {(r["dataset"], r["cls_key"]) for r in read_jsonl(path)
            if r.get("stage") == "s1_catalog" and r.get("error")}


def run_gates(per: Mapping[str, dict], *, margin: float = PARITY_MARGIN, hw_line: float = HW_LINE) -> dict:
    ds = list(per)
    parity = {d: bool(per[d]["champion_mean"] is not None
                      and per[d]["champion_mean"] >= per[d]["paper_ts2vec"] - margin) for d in ds}
    hw = parity.get("Handwriting", False)
    w1 = {"n_parity": int(sum(parity.values())), "n": len(ds), "parity": parity, "hw_parity": bool(hw),
          "pass": bool(sum(parity.values()) >= W1_MIN and hw)}
    ge = {d: bool(per[d]["sh_seed0"] >= per[d]["prior3_seed0"]) for d in ds}
    diffs = [per[d]["sh_seed0"] - per[d]["prior3_seed0"] for d in ds]
    w2 = {"n_ge": int(sum(ge.values())), "n": len(ds), "ge": ge, "mean_diff": float(np.mean(diffs)),
          "pass": bool(sum(ge.values()) >= W2_MIN and float(np.mean(diffs)) >= 0.0)}
    rhos = [per[d]["spearman_slice_vs_official"] for d in ds
            if per[d].get("spearman_slice_vs_official") is not None]
    w3 = {"median_rho": (float(np.median(rhos)) if rhos else None), "rhos": rhos,
          "pass": bool(rhos and float(np.median(rhos)) >= W3_RHO), "descriptive": True}
    hw_max = per.get("Handwriting", {}).get("catalog_max")
    w4 = {"hw_catalog_max": hw_max, "line": hw_line, "pass": bool(hw_max is not None and hw_max >= hw_line)}
    return {"W1": w1, "W2": w2, "W3": w3, "W4": w4}


def verdict_code(g: Mapping[str, dict]) -> str:
    if not g["W1"]["pass"]:
        return "DEPLOY-DEAD"
    return "DEPLOY-WIN" if g["W2"]["pass"] else "DEPLOY-PARITY"


# ---------------------------------------------------------------- 提名（零 GPU，协议锁产物）
def cmd_nominate(outdir: Path, *, datasets: Sequence[str] = TARGETS, all_source: bool = False) -> dict:
    """候选目录：源池外目标（如 HAR）或 --all-source 时用全部 8 个源数据集拟合先验 / 提名；否则 LODO。"""
    rows = [r for r in read_jsonl(MX3 / "rprime_cache.jsonl") if r.get("error") is None]
    cat = [r for r in rows if r["origin"] != "ggs"]
    ggs_rows = [r for r in rows if r["origin"] == "ggs"]
    picks_by_ds, key2cls, catalog = _g35.load_mx3_picks()
    pool_datasets = {r["dataset"] for r in cat}
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    shared = [{"cls_key": e["cls_key"], "cls": e["cls"]} for e in catalog["shared"]]
    ggs_entry = {"cls_key": catalog["ggs"]["cls_key"], "cls": catalog["ggs"]["cls"]} \
        if isinstance(catalog.get("ggs"), dict) and "cls_key" in catalog["ggs"] else None
    if ggs_entry is None:
        g = _g35._m1.GGS
        ggs_entry = {"cls_key": cls_key(g), "cls": g.to_dict()}
    out = {}
    for i, d in enumerate(datasets):
        exclude_ds = None if (all_source or d not in pool_datasets) else d
        train_cat = [r for r in cat if r["dataset"] != exclude_ds]
        train_ggs = [r for r in ggs_rows if r["dataset"] != exclude_ds]
        prior = prior_with_ggs(train_cat, train_ggs)
        train_z = zscore_within(train_cat)
        exclude = {e["cls_key"] for e in shared} | {ggs_entry["cls_key"]}
        nom = nominate(space, enc, train_z, exclude_keys=exclude, seed=NOMINEE_SEED + i)
        tclsc_order = sorted(picks_by_ds[d]["borda"], key=lambda k: (-picks_by_ds[d]["borda"][k], k)) \
            if d in picks_by_ds else []
        out[d] = {"shared": shared, "ggs": ggs_entry, "nominated": nom,
                  "prior": {k: round(v, 6) for k, v in prior.items()},
                  "prior_order": rank_by(prior, [e["cls_key"] for e in shared] + [ggs_entry["cls_key"]]),
                  "tclsc_order": tclsc_order,
                  "fit_on": ("all 8 source datasets" if exclude_ds is None else f"LODO excluding {d}")}
        print(f"[nominate {d}] 拟合于 {out[d]['fit_on']}；先验前 3 {out[d]['prior_order'][:3]}；提名 {len(nom)} 条，"
              f"汉明 {[x['hamming'] for x in nom]}，预测 z {[round(x['pred_z'], 2) for x in nom]}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- GPU
def run_trajectory_cell(cell: dict, arrays: dict, device: str) -> dict:
    """一格：SWA 轨迹 + 各截点评估（scaler / SVM 网格按协议自身口径）。OOM 抛 RprimeOOM。"""
    check_extension_consistency(cell)     # 扩展字段两处真源不一致 → 训练前就断（不进 except）
    opts, seed, iters = cell["options"], int(cell["seed"]), int(cell["iters"])
    Xtr, ytr, yte = arrays["Xtr"], arrays["ytr"], arrays["yte"]
    Xte = arrays["Xte_" + opts["scaler"]]
    grid = opts["svm_grid"]
    results: list[dict] = []
    t0 = time.perf_counter()

    def on_ckpt(step: int, module) -> None:
        te = time.perf_counter()
        ftr = encode_features(module, Xtr, device=device)
        fte = encode_features(module, Xte, device=device)
        v = eval_with_timeout(
            lambda: _g35._svm_variants(ftr, ytr, {opts["scaler"]: fte}, yte, (grid,), seed)[f"{opts['scaler']}/{grid}"],
            EVAL_TIMEOUT_S)
        results.append({"iters": int(step), **v, "elapsed_pretrain_s": round(te - t0, 1),
                        "eval_seconds": round(time.perf_counter() - te, 1)})
        module.train()

    try:
        cls = CLS(**cell["cls"])
        torch.manual_seed(seed)
        ov = {}
        if opts.get("input_mask", "none") != "none":
            ov["input_mask"] = opts["input_mask"]
        if float(opts.get("repr_dropout", 0.0)) > 0:
            ov["repr_dropout"] = float(opts["repr_dropout"])
        enc = build_encoder(input_dims=Xtr.shape[-1], overrides=ov or None)
        losses = pretrain_encoder(
            enc, cls, Xtr, max_iters=iters, batch_size=opts["batch_size"], lr=opts["lr"],
            device=device, seed=seed, use_swa=opts["swa"], drop_last=opts["drop_last"],
            crop_mode=opts["crop_mode"], checkpoints=tuple(cell["checkpoints"]), on_checkpoint=on_ckpt)
    except Exception as e:  # noqa: BLE001
        if is_oom_error(e):
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            raise RprimeOOM(f"{type(e).__name__}: {e}") from e
        return {**cell, "acc": None, "f1": None, "checkpoint_results": results or None,
                "pretrain_seconds": round(time.perf_counter() - t0, 1), "device_name": device_label(device),
                "error": f"{type(e).__name__}: {e}"}
    final = results[-1]
    return {**cell, "acc": final["acc"], "f1": final["f1"], "svm_c": final["svm_c"],
            "svm_cv_score": final["svm_cv_score"], "checkpoint_results": results,
            "pretrain_seconds": results[-1]["elapsed_pretrain_s"],
            "eval_seconds": round(sum(float(c["eval_seconds"]) for c in results), 1),
            "device_name": device_label(device),
            "first_pass_loss": (float(losses[0]) if losses else None),
            "last_pass_loss": (float(losses[-1]) if losses else None),
            "n_train": int(Xtr.shape[0]), "n_test": int(len(yte)), "error": None}


def execute(cells: list[dict], args, tag: str) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs, inflight, oom_log = outdir / "runs.jsonl", outdir / "inflight.txt", outdir / "oom_failures.jsonl"
    done = _g35.load_done(runs) if args.resume else {}
    todo = _g35.pending(cells, done)
    print(f"[{tag}] 计划 {len(cells)} 格，已有 {len(cells) - len(todo)}，待跑 {len(todo)}", flush=True)
    backend_kw = {"n_jobs": 1} if args.joblib_backend == "threading" else {}
    with parallel_backend(args.joblib_backend, **backend_kw):
        for i, cell in enumerate(todo):
            arrays = load_arrays_any(cell["dataset"], getattr(args, "har_npz", None))
            who = (f"{cell['dataset']}/{cell['arm']}/{cell['cls_key'][:8]} s{cell['seed']} "
                   f"iters={cell['iters']} ckpt={cell['checkpoints']}")
            inflight.write_text(f"{cell['key']}\t{who}", encoding="utf-8")
            try:
                row = run_trajectory_cell(cell, arrays, args.device)
            except RprimeOOM as e:
                inflight.write_text("", encoding="utf-8")
                append_jsonl(oom_log, {**cell, "error": str(e), "device_name": device_label(args.device)})
                print(f"[{tag} {i + 1}/{len(todo)}] ⚠️ OOM {who} → 不写缓存：{e}", flush=True)
                continue
            append_jsonl(runs, row)
            inflight.write_text("", encoding="utf-8")
            accs = " ".join(f"{c['acc']:.3f}" for c in (row.get("checkpoint_results") or []))
            print(f"[{tag} {i + 1}/{len(todo)}] {who} acc@ckpt=[{accs}] "
                  f"({row.get('pretrain_seconds')}s+{row.get('eval_seconds')}s)"
                  + (f" ERROR {row['error']}" if row.get("error") else ""), flush=True)
    print(f"[{tag}] 收官", flush=True)


# ---------------------------------------------------------------- 装配
def load_candidates(outdir: Path) -> dict:
    path = outdir / "candidates.json"
    if not path.exists():
        raise SystemExit(f"缺 {path}：先跑 --nominate（协议锁产物）")
    return json.loads(path.read_text(encoding="utf-8"))


def sizes_for(datasets: Sequence[str], *, load: bool, har_npz=None) -> dict:
    out = {}
    for d in datasets:
        if d in SIZES_EXTRA:
            out[d] = (tuple(int(v) for v in load_arrays_any(d, har_npz)["Xtr"].shape) if load
                      else SIZES_EXTRA[d])
        else:
            out[d] = (tuple(int(v) for v in _g35.load_arrays(d)["Xtr"].shape) if load
                      else _g35.SIZES_SURVEY[d])
    return out


def _pa_targets_rows() -> list[dict]:
    if not PA_RUNS.exists():
        return []
    return [r for r in read_jsonl(PA_RUNS) if r.get("stage") == "targets" and r.get("error") is None]


def pa_reference(dataset: str) -> dict:
    """protocol-alignment 六靶表的对照臂：GGS / ts2vec_equiv / zs_top3(best-of-3) 三种子均值 + GGS seed 0。"""
    rows = [r for r in _pa_targets_rows() if r["dataset"] == dataset]
    out = {}
    for arm in ("ggs", "ts2vec_equiv", "zs_top1"):
        accs = [float(r["acc"]) for r in rows if arm in r.get("arm_names", [r.get("arm")])]
        out[arm] = float(np.mean(accs)) if accs else None
    ggs0 = [float(r["acc"]) for r in rows if "ggs" in r.get("arm_names", [r.get("arm")]) and int(r["seed"]) == 0]
    out["ggs_seed0"] = ggs0[0] if ggs0 else None
    by_key: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if "zs_top3" in r.get("arm_names", [r.get("arm")]):
            by_key[r["cls_key"]].append(float(r["acc"]))
    out["zs_top3_best"] = max((float(np.mean(v)) for v in by_key.values()), default=None)
    return out


def slice_prior_per_dataset(dataset: str, keys: Sequence[str]) -> dict[str, float]:
    """W3 用：该数据集 6 个切片 @290 的 within-task z 均值（切片协议的排名信号）。"""
    rows = [r for r in read_jsonl(MX3 / "rprime_cache.jsonl")
            if r.get("error") is None and r["origin"] != "ggs" and int(r["iters"]) == 290]
    z = zscore_within([r for r in rows if r["dataset"] == dataset])
    acc: dict[str, list[float]] = defaultdict(list)
    for r in z:
        if r["cls_key"] in keys:
            acc[r["cls_key"]].append(float(r["z"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cmd_analyze(outdir: Path, *, datasets: Sequence[str] = TARGETS) -> dict:
    cands = load_candidates(outdir)
    rows = [r for r in read_jsonl(outdir / "runs.jsonl")] if (outdir / "runs.jsonl").exists() else []
    s1 = [r for r in rows if r.get("stage") == "s1_catalog" and r.get("error") is None]
    s2 = [r for r in rows if r.get("stage") == "s2_finals" and r.get("error") is None]
    per_target, picks_out, tables = {}, {}, {}
    for d in datasets:
        if d not in cands:
            continue
        c = cands[d]
        rows_d = [r for r in s1 if r["dataset"] == d and int(r["seed"]) == 0]
        traj, secs = trajectory_from_rows(rows_d)
        if not traj:
            print(f"[analyze {d}] 无 S1 行，跳过", flush=True)
            continue
        budgets = sorted({b for v in traj.values() for b in v})
        keys = sorted(k for k in traj if all(b in traj[k] for b in budgets))
        shared_keys = [e["cls_key"] for e in c["shared"]]
        ggs_key = c["ggs"]["cls_key"]
        nom = {e["cls_key"]: float(e["pred_z"]) for e in c.get("nominated", [])}
        prior = {k: float(v) for k, v in c["prior"].items()}
        hybrid = {k: (prior[k] + HYBRID_BIAS if k in prior else nom.get(k, float("-inf"))) for k in keys}
        rank = {"prior": rank_by(prior, keys), "hybrid": rank_by(hybrid, keys),
                "tclsc": [k for k in c.get("tclsc_order", []) if k in keys]}
        row = deploy_arms(keys, traj, rank, budgets=budgets, random_seed=0, seconds=secs)
        final_b = budgets[-1]
        acc_final = {k: traj[k][final_b] for k in keys}
        # 3 种子均值：seed 0 来自 S1，seeds 1/2 来自 S2
        def three_seed(k: str) -> Optional[float]:
            extra = [float(r["acc"]) for r in s2 if r["dataset"] == d and r["cls_key"] == k]
            return float(np.mean([acc_final[k]] + extra)) if len(extra) == len(FINAL_SEEDS) else None
        champ, p3 = row["picks"].get(MAIN_ARM), row["picks"].get("prior3")
        ref = pa_reference(d)
        sp = slice_prior_per_dataset(d, shared_keys)
        common = [k for k in shared_keys if k in sp and k in acc_final]
        rho = float(spearmanr([sp[k] for k in common], [acc_final[k] for k in common]).correlation) if len(common) >= 3 else None
        rk25 = rank_by({k: traj[k][budgets[0]] for k in keys}, keys)
        rk100 = rank_by(acc_final, keys)
        rho_25_100 = float(spearmanr([rk25.index(k) for k in keys], [rk100.index(k) for k in keys]).correlation)
        per_target[d] = {
            "paper_ts2vec": paper_ts2vec(d),
            "paper_median": paper_median(d),
            "paper_reference": paper_reference(d),
            "budgets": budgets, "n_candidates": len(keys),
            "champion_key": champ, "champion_seed0": row["arms"].get(MAIN_ARM),
            "champion_mean": three_seed(champ) if champ else None,
            "champion_in_shared": bool(champ in shared_keys) if champ else None,
            "champion_is_ggs": bool(champ == ggs_key) if champ else None,
            "prior3_key": p3, "prior3_seed0": row["arms"].get("prior3"),
            "prior3_mean": three_seed(p3) if p3 else None,
            "sh_seed0": row["arms"].get(MAIN_ARM),
            "catalog_max": max(acc_final.values()),
            "catalog_max_key": max(acc_final, key=lambda k: (acc_final[k], k)),
            "ggs_seed0_here": acc_final.get(ggs_key), "pa_reference": ref,
            "ggs_consistency_abs_delta": (abs(acc_final[ggs_key] - ref["ggs_seed0"])
                                          if ggs_key in acc_final and ref.get("ggs_seed0") is not None else None),
            "spearman_slice_vs_official": rho, "spearman_rank25_vs_100": rho_25_100,
            "arms": row["arms"], "regret": row["regret"], "cost_iters": row["cost_iters"],
            "cost_seconds": row["cost_seconds"], "picks": row["picks"],
        }
        picks_out[d] = {MAIN_ARM: champ, "prior3": p3}
        # macro-F1（HAR 协议要求同时报 acc / F1）：seed 0 的冠军与 3 种子均值
        f1_final = {r["cls_key"]: float(r["checkpoint_results"][-1]["f1"]) for r in rows_d
                    if r.get("checkpoint_results")}
        per_target[d]["champion_f1_seed0"] = f1_final.get(champ) if champ else None
        f1_extra = [float(r["f1"]) for r in s2 if r["dataset"] == d and r["cls_key"] == champ]
        per_target[d]["champion_f1_mean"] = (float(np.mean([f1_final[champ]] + f1_extra))
                                             if champ and champ in f1_final and len(f1_extra) == len(FINAL_SEEDS) else None)
        per_target[d]["ggs_f1_seed0"] = f1_final.get(ggs_key)
        print(f"[analyze {d}] 候选 {len(keys)} 截点 {budgets}；oracle {row['arms']['oracle']:.3f} "
              f"SH {row['arms'].get(MAIN_ARM, float('nan')):.3f} prior3 {row['arms'].get('prior3', float('nan')):.3f} "
              f"GGS {acc_final.get(ggs_key, float('nan')):.3f}（论文 TS2Vec {paper_ts2vec(d)}）", flush=True)
    per_eff = {d: {**v, "champion_mean": (v["champion_mean"] if v["champion_mean"] is not None else v["champion_seed0"])}
               for d, v in per_target.items()}
    har_mode = bool(per_target) and set(per_target) <= set(HAR_TABLE2)
    if not per_target:
        gates, verdict = None, None
    elif har_mode:
        gates, verdict = run_gates_har(per_eff), verdict_code_har(run_gates_har(per_eff))
    else:
        gates, verdict = run_gates(per_eff), verdict_code(run_gates(per_eff))
    n_full = sum(v["champion_mean"] is not None for v in per_target.values())
    analysis = {"protocol": ("results/mx4_deploy_har/PROTOCOL.md" if har_mode else "results/mx4_deploy/PROTOCOL.md"),
                "gates_kind": ("H" if har_mode else "W"),
                "n_targets": len(per_target), "n_targets_with_3_seeds": n_full, "per_target": per_target,
                "gates": gates, "verdict": verdict,
                "verdict_note": ("主门用 3 种子均值" if n_full == len(per_target) else
                                 f"主门暂用 seed 0（3 种子均值只有 {n_full}/{len(per_target)} 靶）")}
    analysis["predictions"] = (predictions(per_target) if (per_target and not har_mode) else None)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "picks.json").write_text(json.dumps(picks_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(analysis), encoding="utf-8")
    if gates and analysis["gates_kind"] == "W":
        print(f"verdict {analysis['verdict']}（{analysis['verdict_note']}）；W1 {gates['W1']['n_parity']}/{gates['W1']['n']} "
              f"HW {gates['W1']['hw_parity']}；W2 {gates['W2']['n_ge']}/{gates['W2']['n']} Δ {gates['W2']['mean_diff']:+.3f}")
    elif gates:
        print(f"verdict {analysis['verdict']}（{analysis['verdict_note']}）；H1 {gates['H1']['value']:.4f} vs {gates['H1']['line']:.4f}；"
              f"H3 SH {gates['H3']['sh']:.4f} vs prior3 {gates['H3']['prior3']:.4f}")
    return analysis


def predictions(per: Mapping[str, dict]) -> dict:
    ds = list(per)
    r25 = [per[d]["spearman_rank25_vs_100"] for d in ds]
    d631 = [per[d]["arms"].get("SH6-3-3_prior", 0) - per[d]["arms"].get("SH6-3-1_prior", 0) for d in ds]
    hw = per.get("Handwriting")
    band = []
    for d in ("Cricket", "EthanolConcentration"):
        if d in per:
            a = per[d]["arms"]
            band.append(max(a[k] for k in a if k not in ("bottom",)) - min(a[k] for k in a if k not in ("bottom", "random1")))
    cons = [per[d]["ggs_consistency_abs_delta"] for d in ds if per[d]["ggs_consistency_abs_delta"] is not None]
    return {
        "P1": {"claim": "25% 截点排名与 100% 的 Spearman 中位 < 0.6 且 SH6-3-1 比 SH6-3-3 差 ≥ 0.02",
               "median_rho_25_100": float(np.median(r25)), "mean_gap_631": float(np.mean(d631)),
               "hit": bool(np.median(r25) < 0.6 and np.mean(d631) >= 0.02)},
        "P2": {"claim": "HW 冠军 = GGS 或提名，而非 shared-10", "champion_in_shared": (hw or {}).get("champion_in_shared"),
               "hit": bool(hw and hw.get("champion_in_shared") is False)},
        "P3": {"claim": "Cricket / Ethanol 所有臂同带（极差 < 0.03，排除 bottom/random1）", "ranges": band,
               "hit": bool(band and max(band) < 0.03)},
        "P4": {"claim": "S1 的 GGS seed 0 与 protocol-alignment 逐位或 ≤ 0.005", "deltas": cons,
               "hit": bool(cons and max(cons) <= 0.005)},
    }


def markdown_tables(a: dict) -> str:
    L = ["## 六靶头条表（官方口径 test acc；论文 = TS2Vec Table 10）", "",
         "| 靶 | 论文 TS2Vec | 文献中位 | SH6-3-3 冠军 seed0 / 3 种子 | prior3 seed0 / 3 种子 | GGS（本轮 s0 / PA 3 种子） | zs_top3(PA) | 目录 oracle | 追平 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for d, v in a["per_target"].items():
        ref = v["pa_reference"]
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        par = "✓" if (v["champion_mean"] or v["champion_seed0"] or 0) >= v["paper_ts2vec"] - PARITY_MARGIN else "✗"
        L.append(f"| {d} | {v['paper_ts2vec']:.3f} | {v['paper_median']:.3f} | {f(v['champion_seed0'])} / {f(v['champion_mean'])} | "
                 f"{f(v['prior3_seed0'])} / {f(v['prior3_mean'])} | {f(v['ggs_seed0_here'])} / {f(ref.get('ggs'))} | "
                 f"{f(ref.get('zs_top3_best'))} | {f(v['catalog_max'])} | {par} |")
    L += ["", "## 各靶臂表（seed 0，acc@100%；regret 对目录 oracle；成本 iters）", ""]
    arms = ["oracle", "prior1", "prior3", "hybrid1", "hybrid3", "tclsc1", "tclsc3", "random1", "random3",
            "SH6-3-3_prior", "SH6-3_prior", "SH6-3-1_prior", "SH6-3-3_hybrid", "SH6-3-3_tclsc", "SH6-3-3_random"]
    L.append("| 靶 | " + " | ".join(arms) + " |")
    L.append("|---|" + "|".join(["---:"] * len(arms)) + "|")
    for d, v in a["per_target"].items():
        L.append(f"| {d} | " + " | ".join(f"{v['arms'][x]:.3f}" if x in v["arms"] else "—" for x in arms) + " |")
    L.append("| 成本 iters（首靶） | " + " | ".join(
        str(next(iter(a["per_target"].values()))["cost_iters"].get(x, "—")) for x in arms) + " |")
    if a.get("gates") and a.get("gates_kind") == "H":
        g = a["gates"]
        v = a["per_target"]["HAR"]
        L += ["", "## 门（WP3b HAR）", "",
              f"- H1：冠军 acc {g['H1']['value']:.4f} vs GGS 锚 − 0.005 = {g['H1']['line']:.4f} → {'✓' if g['H1']['pass'] else '✗'}",
              f"- H2：冠军 acc vs TS2Vec {g['H2']['line']:.3f} → {'✓' if g['H2']['pass'] else '✗'}",
              f"- H3：seed 0 SH {g['H3']['sh']:.4f} vs prior3 {g['H3']['prior3']:.4f} → {'✓' if g['H3']['pass'] else '✗'}",
              f"- macro-F1：冠军 seed0 {v.get('champion_f1_seed0')} / 3 种子 {v.get('champion_f1_mean')}；GGS seed0 {v.get('ggs_f1_seed0')}",
              "", f"读数码：**{a['verdict']}**（{a['verdict_note']}）"]
    elif a.get("gates"):
        g = a["gates"]
        L += ["", "## 门", "",
              f"- W1：追平 {g['W1']['n_parity']}/{g['W1']['n']}，HW {'✓' if g['W1']['hw_parity'] else '✗'} → {'✓' if g['W1']['pass'] else '✗'}（{a['verdict_note']}）",
              f"- W2：SH ≥ prior3 于 {g['W2']['n_ge']}/{g['W2']['n']}，均值差 {g['W2']['mean_diff']:+.3f} → {'✓' if g['W2']['pass'] else '✗'}",
              f"- W3：切片排名 vs 官方 Spearman 中位 {g['W3']['median_rho']} → {'✓' if g['W3']['pass'] else '✗'}（描述性）",
              f"- W4：HW 目录最高 {g['W4']['hw_catalog_max']} vs 线 {g['W4']['line']} → {'✓' if g['W4']['pass'] else '✗'}",
              "", f"读数码：**{a['verdict']}**"]
    if a.get("predictions"):
        L += ["", "## 预测对账", "", "| 预测 | 内容 | 读数 | 命中 |", "|---|---|---|---|"]
        for k, p in a["predictions"].items():
            detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in p.items() if kk not in ("claim", "hit")}
            L.append(f"| {k} | {p['claim']} | {detail} | {'✓' if p['hit'] else '✗'} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def cmd_dry_run(protos: dict, outdir: Path, datasets: Sequence[str], *, protocol: str = PROTOCOL,
                twins: Optional[dict] = None, twin_only: bool = False,
                skip_error_keys: Collection[tuple[str, str]] = ()) -> None:
    cands = load_candidates(outdir)
    sizes = sizes_for(datasets, load=False)
    cells = s1_worklist(protos, sizes, {d: cands[d] for d in datasets if d in cands}, protocol=protocol,
                        twins=twins, twin_only=twin_only, skip_error_keys=skip_error_keys)
    pa = _pa_targets_rows()
    est = 0.0
    for d in datasets:
        rs = [r for r in pa if r["dataset"] == d]
        pre = statistics.median(r["pretrain_seconds"] for r in rs) if rs else ANCHORS_EXTRA.get(d, (30.0, 1.0))[0]
        ev = statistics.median(r["eval_seconds"] for r in rs) if rs else ANCHORS_EXTRA.get(d, (30.0, 1.0))[1]
        n = sum(c["dataset"] == d for c in cells)
        est += n * (pre + 3 * ev)
        print(f"  {d}: {n} 格 × ({pre:.1f}s + 3×{ev:.1f}s)")
    print(f"S1 合计 {len(cells)} 格，估时 {est / 3600:.2f} h（4090 锚）；S2 另加每靶 ≤ 2 CLS × 2 种子")


def cmd_smoke(protos: dict, args) -> None:
    outdir = Path(args.outdir) / "smoke"
    p = protos[PROTOCOL]
    space = CLSSpace.from_yaml()
    shared, _ = __import__("autocls.labeling", fromlist=["load_probe_cls"]).load_probe_cls(
        ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json", 2)
    cells = []
    sizes = _g35.SIZES_SURVEY
    for c, k in shared:
        cell = _g35._cell("s1_catalog", "Handwriting", PROTOCOL, p, sizes, arm_names=["shared"],
                          cls_key_=k, cls=c, seed=0, iters=SMOKE_ITERS)
        cell["checkpoints"] = list(checkpoints_for(SMOKE_ITERS))
        cells.append(cell)
    g = _g35._m1.GGS
    cell = _g35._cell("s1_catalog", "Handwriting", PROTOCOL, p, sizes, arm_names=["ggs"],
                      cls_key_=cls_key(g), cls=g, seed=0, iters=SMOKE_ITERS)
    cell["checkpoints"] = list(checkpoints_for(SMOKE_ITERS))
    cells.append(cell)
    args.outdir = str(outdir)
    execute(cells, args, "smoke")
    rows = read_jsonl(outdir / "runs.jsonl")
    ok = all(len(r.get("checkpoint_results") or []) == 3 and r.get("error") is None for r in rows)
    print(f"smoke {'OK' if ok else 'FAILED'}：{len(rows)} 行，截点 {[len(r.get('checkpoint_results') or []) for r in rows]}")
    _ = space


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nominate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--finals", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--joblib-backend", default="threading", choices=("threading", "loky"))
    ap.add_argument("--protocol", default=PROTOCOL, choices=("ts2vec_official", "legacy_full"),
                    help="训练/评估协议（configs/train.yaml: protocols）；HAR 外部锚用 legacy_full")
    ap.add_argument("--har-npz", default=str(HAR_NPZ_DEFAULT), help="UCI-HAR 缓存（data/HAR/har.npz）")
    ap.add_argument("--all-source", action="store_true", help="先验 / 提名用全部源数据集拟合（目标不在源池时自动如此）")
    # ---- 孪生（MX-4 WP6）----
    ap.add_argument("--twins", nargs="+", default=None, metavar="FIELD=VALUE",
                    help="目录每条配一格只改扩展字段的孪生（如 input_mask=binomial）；新产物目录须另开 --outdir")
    ap.add_argument("--twin-only", action="store_true", help="只跑孪生格（legacy 格在 --legacy-runs 里）")
    ap.add_argument("--legacy-runs", default=str(OUT / "runs.jsonl"),
                    help="WP3 的 runs.jsonl：error 格不出孪生（默认 results/mx4_deploy/runs.jsonl）")
    args = ap.parse_args(argv)
    datasets = tuple(args.datasets) if args.datasets else TARGETS
    protos = _g35.load_protocols()
    outdir = Path(args.outdir)
    twins, skip = None, set()
    if args.twins:
        _g32 = importlib.import_module("scripts.32_pq_generate_rprime")
        try:
            twins = _g32.parse_twins(args.twins, _space())
        except ValueError as e:
            ap.error(f"--twins：{e}")
        if outdir.resolve() == OUT.resolve():
            ap.error(f"--twins 的产物不得写进协议锁定的 {OUT}——请另开 --outdir（如 results/mx4_mask）")
        skip = error_keys_from_runs(Path(args.legacy_runs))
    if args.nominate:
        cmd_nominate(outdir, datasets=datasets, all_source=args.all_source)
    if args.dry_run:
        cmd_dry_run(protos, outdir, datasets, protocol=args.protocol,
                    twins=twins, twin_only=bool(args.twin_only), skip_error_keys=skip)
    if args.smoke:
        cmd_smoke(protos, args)
        return
    if args.catalog:
        cands = load_candidates(outdir)
        sizes = sizes_for(datasets, load=True, har_npz=args.har_npz)
        execute(s1_worklist(protos, sizes, {d: cands[d] for d in datasets if d in cands}, protocol=args.protocol,
                            twins=twins, twin_only=bool(args.twin_only), skip_error_keys=skip),
                args, "S1")
    if args.finals:
        picks_path = outdir / "picks.json"
        if not picks_path.exists():
            raise SystemExit("缺 picks.json：先 --analyze")
        picks = json.loads(picks_path.read_text(encoding="utf-8"))
        cands = load_candidates(outdir)
        key2cls = {}
        for c in cands.values():
            for e in list(c["shared"]) + [c["ggs"]] + list(c.get("nominated", [])):
                key2cls[e["cls_key"]] = e["cls"]
        # 孪生冠军的 CLS 只在本目录 runs.jsonl 里（MX-4 WP6）
        runs_path = outdir / "runs.jsonl"
        if runs_path.exists():
            for r in read_jsonl(runs_path):
                if r.get("cls"):
                    key2cls.setdefault(r["cls_key"], r["cls"])
        seeds = tuple(int(s) for s in args.seeds) if args.seeds else FINAL_SEEDS
        sizes = sizes_for(datasets, load=True, har_npz=args.har_npz)
        execute(finals_worklist(protos, sizes, {d: v for d, v in picks.items() if d in datasets and all(v.values())},
                                key2cls, seeds=seeds, protocol=args.protocol), args, "S2")
    if args.analyze:
        cmd_analyze(outdir, datasets=datasets)
    if not any((args.nominate, args.dry_run, args.catalog, args.finals, args.analyze)):
        ap.print_help()


if __name__ == "__main__":
    main()
