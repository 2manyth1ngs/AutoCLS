"""MX-4 WP8-BO：目标侧提名回灌搜索（"第三条"）——信任域提名 + 真评估回灌能否搜到目录外更好的 CLS。

场地 = WP3 官方口径 6 个 UEA 靶（`results/mx4_deploy`，ts2vec_official，截点 25/50/100%）+ WP3b UCI-HAR
（`results/mx4_deploy_har`，legacy_full，截点 580/1160/2320）。第 0 轮证据 = 目录 21 条（shared-10 + GGS + 一次性提名 10）
的完整轨迹，回灌只读 @25% 截点（部署可得），@100% 只作真值。

每轮（每靶每臂）：父本 = 靶上 @25% 前 5 的已测配置 → 1–3 字段变异（39 号 `nominate` 同式）→ 信任域 = 到源池 ∪ 靶侧
已测配置的字段汉明 ≤ 3 → `bo_ucb` 按 `pred + κ·std`（FieldSurrogate hgb bagging=10，源池 z 行 ∪ 靶侧 @25% z 行 ×10）取前 5，
`rand_nbr` 同池同半径均匀取 5 → 每条跑完整三截点轨迹（`39.run_trajectory_cell`）→ 下一轮回灌。3 轮后每臂 15 条。

部署策略全部在缓存上模拟（成本按 iters 全跑当量）：SH6-3-3_prior（WP3 数字，须逐位）/ SH10-4-3_prior / ONESHOT-SH
（先验前 6 + 一次性提名 10 → 3 → 3，6.25）/ MIX-SH（先验与提名按代理分混排前 6，3.75）/ BO-SH、RAND-SH
（先验前 6 + 15 条提名 @25% → 3 → 3，7.5）/ 池 oracle。门 B1（头顶空间）/ B2（部署）/ B3（引导有效）见 PROTOCOL.md。

子命令：--simulate（零 GPU；阶段 0 与终局共用）/ --loop（GPU；--resume 按轮续）/ --finals / --analyze / --dry-run。
不改 35/36/39/autocls；不动 SH 淘汰规则（关键决策 8）；提名只在信任域内且全部真评估（判死 (b) 的唯一出口形态）。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import cls_key  # noqa: E402
from autocls.search.surrogate import FieldSurrogate, field_hamming  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g39 = importlib.import_module("scripts.39_sh_deploy")

OUT = ROOT / "results" / "mx4_bo"
DEPLOY = _g39.OUT                                   # results/mx4_deploy
DEPLOY_HAR = ROOT / "results" / "mx4_deploy_har"
MX3 = ROOT / "results" / "mx3_pq"

TARGETS6 = tuple(_g35.TARGETS)
# 过夜顺序：便宜的先（WP3 4090 单格全长秒数：HW 5.8 / HMD 25 / UWave 21 / Cricket 45 / EC 54 / PS 94）
LOOP_ORDER = ("Handwriting", "HandMovementDirection", "UWaveGestureLibrary", "Cricket",
              "EthanolConcentration", "PhonemeSpectra", "HAR")
CELL_SECONDS_4090 = {"Handwriting": 5.8, "UWaveGestureLibrary": 21.0, "PhonemeSpectra": 94.0,
                     "EthanolConcentration": 54.0, "Cricket": 45.0, "HandMovementDirection": 25.0}
CELL_SECONDS_4060 = {"HAR": 312.0}
LOCAL_SLOWDOWN = 2.3                                # 4060 / 4090

ARMS = ("bo_ucb", "rand_nbr")
MODE_OF = {"bo_ucb": "ucb", "rand_nbr": "rand"}
ROUNDS = 3
PER_ROUND = 5
N_PARENTS_T = 5                                     # 父本 = 靶上 @25% 前 5
POOL = 2000
RADIUS = 3
MAX_MUT = 3
KAPPA = 1.0
BAGGING = 10
COPIES = 10                                         # 靶侧行复制份数（权重）
ITERS_FEATURE = 60                                  # 靶侧 @25% 行在代理特征里的预算档
MAX_B = 290
SEED_BASE = 4300
FINAL_SEEDS = tuple(_g39.FINAL_SEEDS)

POLICIES = ("SH6-3-3_prior", "SH10-4-3_prior", "ONESHOT-SH", "MIX-SH", "BO-SH", "RAND-SH")
MAIN = "BO-SH"
CONTROL = "RAND-SH"
BASELINE = "SH6-3-3_prior"
POLICY_ARM = {"BO-SH": "bo_ucb", "RAND-SH": "rand_nbr"}

B1_MARGIN = 0.01
B1_MIN = 3
B2_MIN = 4
B2_MARGIN = 0.01
HW_LINE = 0.485                                     # TS2Vec Handwriting 官方线
B3_MIN = 4
HAR_LINE = 0.935
PREDICTIONS = {"B1": 0.55, "B2": 0.40, "B3": 0.45, "HAR": 0.35}


# ================================================================ 纯函数
def mutate_pool(space: CLSSpace, parents: Sequence[CLS], *, pool: int, max_mut: int,
                rng: np.random.Generator, exclude: set) -> dict[str, CLS]:
    """父本局部变异池：每个候选改 1..max_mut 个字段（门控字段随 space.yaml 联动），validate 通过、键唯一、不在 exclude。"""
    parents = list(parents)
    if not parents:
        return {}
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
        if k in exclude or k in cands:
            continue
        cands[k] = c
    return cands


def nominate_round(space: CLSSpace, enc: CLSEncoder, src_rows: Sequence[dict], tgt_rows: Sequence[dict], *,
                   parents: Sequence[CLS], exclude, n: int = PER_ROUND, pool: int = POOL, radius: int = RADIUS,
                   mode: str = "ucb", kappa: float = KAPPA, bagging: int = BAGGING, seed: int = 0,
                   max_mut: int = MAX_MUT, max_b: int = MAX_B) -> list[dict]:
    """一轮提名。信任域 = 到（源池 ∪ 靶侧）已见配置的字段汉明 ≤ radius；ucb 取 pred + κ·std 前 n，
    rand 在同一候选池里均匀取 n（抽签在拟合代理之前、只依赖 seed，故与代理无关）。"""
    if mode not in ("ucb", "rand"):
        raise ValueError(f"mode={mode!r} 须为 ucb / rand")
    rows = list(src_rows) + list(tgt_rows)
    if not rows or not parents:
        return []
    seen: dict[str, CLS] = {}
    for r in rows:
        seen.setdefault(r["cls_key"], CLS(**r["cls"]))
    Xtr = np.stack([enc.encode(c) for c in seen.values()])
    rng = np.random.default_rng(int(seed))
    cands = mutate_pool(space, parents, pool=pool, max_mut=max_mut, rng=rng, exclude=set(exclude) | set(seen))
    if not cands:
        return []
    keys = list(cands)
    Xc = np.stack([enc.encode(cands[k]) for k in keys])
    ham = field_hamming(Xc, Xtr, enc.field_slices)
    ok = [i for i in range(len(keys)) if ham[i] <= int(radius)]
    if not ok:
        return []
    rand_order = [int(j) for j in rng.permutation(len(ok))[:int(n)]]
    model = FieldSurrogate("hgb", bagging=int(bagging), seed=int(seed)).fit(
        _g36.features(rows, enc, max_b), np.array([float(r["z"]) for r in rows]))
    Xf = np.hstack([Xc[ok], np.ones((len(ok), 1))])          # 预算特征 = 1.0（b_final）
    pred, std = model.predict(Xf)
    score = pred + float(kappa) * std
    if mode == "ucb":
        order = sorted(range(len(ok)), key=lambda j: (-float(score[j]), keys[ok[j]]))[:int(n)]
    else:
        order = rand_order
    out = []
    for rank, j in enumerate(order):
        k = keys[ok[j]]
        out.append({"cls_key": k, "cls": asdict(cands[k]), "pred_z": float(pred[j]), "std_z": float(std[j]),
                    "score": float(score[j]), "hamming": int(ham[ok[j]]), "mode": mode, "rank": rank,
                    "n_pool": len(keys), "n_in_radius": len(ok)})
    return out


def target_rows(measured: Mapping[str, dict], budget: int, *, task_id: str, copies: int = COPIES,
                iters_feature: int = ITERS_FEATURE) -> list[dict]:
    """靶侧回灌行：within-靶 @budget z（均值 0 / sd 1），复制 copies 份当权重；没有该截点的（error 格）不入。"""
    items = [(k, v) for k, v in measured.items() if int(budget) in (v.get("traj") or {})]
    if not items:
        return []
    a = np.array([float(v["traj"][int(budget)]) for _, v in items])
    mu, sd = float(a.mean()), float(a.std())
    rows = []
    for (k, v), x in zip(items, a):
        z = (float(x) - mu) / sd if sd > 0 else 0.0
        for _ in range(int(copies)):
            rows.append({"task_id": task_id, "cls_key": k, "cls": dict(v["cls"]), "iters": int(iters_feature),
                         "z": float(z), "r_prime": float(x)})
    return rows


def parents_of(measured: Mapping[str, dict], budget: int, *, k: int = N_PARENTS_T) -> list[str]:
    """父本 = 靶上 @budget 最高的 k 条已测配置（同分按 key 升序；无该截点的不入）。"""
    ok = [key for key, v in measured.items() if int(budget) in (v.get("traj") or {})]
    return sorted(ok, key=lambda key: (-float(measured[key]["traj"][int(budget)]), key))[:int(k)]


def measured_from_rows(rows: Sequence[dict]) -> dict[str, dict]:
    """runs 行 → {cls_key: {cls, traj{b: acc}, secs{b: 累计秒}, source}}；同 key 首行优先（目录行在前）；
    error 行也登记（traj 空）——它已经花过预算，后续轮次不再提名它。"""
    traj, secs = _g39.trajectory_from_rows([r for r in rows if r.get("error") is None])
    out: dict[str, dict] = {}
    for r in rows:
        k = r["cls_key"]
        if k in out:
            continue
        out[k] = {"cls": dict(r["cls"]), "traj": dict(traj.get(k, {})), "secs": dict(secs.get(k, {})),
                  "source": str(r.get("arm", ""))}
    return out


def round_cells(dataset: str, entries: Sequence[dict], *, arm: str, rnd: int, protos: dict, sizes: dict,
                protocol: str) -> list[dict]:
    """一轮提名 → 完整三截点轨迹格（seed 0；arm=f"{arm}_r{rnd}" 入缓存键，两臂同 CLS 不同键）。"""
    p = protos[protocol]
    out = []
    for e in entries:
        cell = _g35._cell("bo_round", dataset, protocol, p, sizes, arm_names=[f"{arm}_r{int(rnd)}"],
                          cls_key_=e["cls_key"], cls=CLS(**e["cls"]), seed=0)
        cell["checkpoints"] = list(_g39.checkpoints_for(cell["iters"]))
        cell["bo_arm"] = arm
        cell["round"] = int(rnd)
        out.append(cell)
    return out


def policy_table(traj: Mapping[str, Mapping[int, float]], secs: Mapping[str, Mapping[int, float]], *,
                 budgets: Sequence[int], prior: Mapping[str, float], nominated: Mapping[str, float],
                 shared_keys: Sequence[str], ggs_key: str, bo_keys: Mapping[str, Sequence[str]]) -> dict:
    """一个靶的策略表（PROTOCOL §2.3）。只用三截点齐全的配置；成本 = iters 全跑当量 + 缓存墙钟。"""
    stages = [int(b) for b in budgets]
    bf = stages[-1]

    def complete(k: str) -> bool:
        t = traj.get(k) or {}
        return all(b in t for b in stages)

    cat_all = list(shared_keys) + [ggs_key] + list(nominated)
    keys = sorted({k for k in cat_all if complete(k)})
    arm_keys = {a: sorted({k for k in ks if complete(k) and k not in keys}) for a, ks in bo_keys.items()}
    truth = {k: float(traj[k][bf]) for k in keys}
    for a in arm_keys:
        truth.update({k: float(traj[k][bf]) for k in arm_keys[a]})
    oracle_catalog = max(truth[k] for k in keys)
    oracle_catalog_key = sorted(keys, key=lambda k: (-truth[k], k))[0]
    oracle_pool, oracle_pool_key = {}, {}
    for a in arm_keys:
        pk = keys + arm_keys[a]
        oracle_pool_key[a] = sorted(pk, key=lambda k: (-truth[k], k))[0]
        oracle_pool[a] = truth[oracle_pool_key[a]]
    prior_rank = _g36.rank_by(prior, keys)
    mix_rank = _g36.rank_by({**{k: float(v) for k, v in prior.items()}, **{k: float(v) for k, v in nominated.items()}}, keys)
    nom_keys = [k for k in keys if k in nominated]

    def run(ranked: Sequence[str], ks: Sequence[int], pool_arm: Optional[str] = None) -> dict:
        path = _g36.sh_path(ranked, traj, ks, stages)
        pick = _g36.sh_pick(path[-1], traj, bf)
        cost_it = _g36.path_cost_iters(path, stages)
        try:
            cost_s = _g36.path_cost_seconds(path, stages, secs)
        except (KeyError, TypeError):
            cost_s = None
        oracle_ref = oracle_pool[pool_arm] if pool_arm else oracle_catalog
        return {"pick": pick, "acc": truth[pick], "regret_catalog": oracle_catalog - truth[pick],
                "regret_pool": oracle_ref - truth[pick], "cost_iters": int(cost_it), "cost_full": cost_it / bf,
                "cost_seconds": cost_s, "path": path, "in_catalog": pick in keys, "n_start": len(path[0])}

    pol = {
        "SH6-3-3_prior": run(prior_rank, (6, 3, 3)),
        "SH10-4-3_prior": run(prior_rank, (10, 4, 3)),
        "ONESHOT-SH": run(prior_rank[:6] + nom_keys, (6 + len(nom_keys), 3, 3)),
        "MIX-SH": run(mix_rank, (6, 3, 3)),
    }
    for name, a in POLICY_ARM.items():
        start = prior_rank[:6] + arm_keys.get(a, [])
        pol[name] = run(start, (len(start), 3, 3), pool_arm=a if a in arm_keys else None)
    nominee_max = {a: (max(truth[k] for k in ks) if ks else None) for a, ks in arm_keys.items()}
    nominee_mean = {a: (float(np.mean([truth[k] for k in ks])) if ks else None) for a, ks in arm_keys.items()}
    return {"budgets": stages, "keys": keys, "arm_keys": arm_keys, "n_catalog": len(keys),
            "oracle_catalog": oracle_catalog, "oracle_catalog_key": oracle_catalog_key,
            "oracle_pool": oracle_pool, "oracle_pool_key": oracle_pool_key,
            "nominee_max": nominee_max, "nominee_mean": nominee_mean,
            "prior_rank": prior_rank, "policies": pol}


def gates(per_target: Mapping[str, dict], *, targets: Sequence[str] = TARGETS6) -> dict:
    """B1 头顶空间 / B2 部署 / B3 引导有效（PROTOCOL §2.4）。缺靶或缺 3 种子均值 → 该门 incomplete。"""
    present = [d for d in targets if d in per_target]
    complete = len(present) == len(targets)
    b1 = {}
    for d in present:
        pt = per_target[d]
        nm = (pt.get("nominee_max") or {}).get("bo_ucb")
        b1[d] = None if nm is None else bool(nm > float(pt["oracle_catalog"]) + B1_MARGIN)
    b1_ok = complete and all(v is not None for v in b1.values())
    n1 = sum(1 for v in b1.values() if v)
    b2, diffs = {}, []
    for d in present:
        m3 = per_target[d].get("mean3") or {}
        bo, sh = m3.get(MAIN), m3.get(BASELINE)
        if bo is None or sh is None:
            b2[d] = None
            continue
        b2[d] = bool(bo >= sh)
        diffs.append(float(bo) - float(sh))
    b2_ok = complete and all(v is not None for v in b2.values())
    n2 = sum(1 for v in b2.values() if v)
    mean_diff = float(np.mean(diffs)) if diffs else None
    hw = ((per_target.get("Handwriting") or {}).get("mean3") or {}).get(MAIN)
    hw_ok = None if hw is None else bool(float(hw) >= HW_LINE)
    b3m, b3o = {}, {}
    for d in present:
        pt = per_target[d]
        mb, mr = (pt.get("nominee_mean") or {}).get("bo_ucb"), (pt.get("nominee_mean") or {}).get("rand_nbr")
        ob, orr = (pt.get("oracle_pool") or {}).get("bo_ucb"), (pt.get("oracle_pool") or {}).get("rand_nbr")
        b3m[d] = None if (mb is None or mr is None) else bool(mb > mr)
        b3o[d] = None if (ob is None or orr is None) else bool(ob >= orr)
    b3_ok = complete and all(v is not None for v in list(b3m.values()) + list(b3o.values()))
    n3m, n3o = sum(1 for v in b3m.values() if v), sum(1 for v in b3o.values() if v)
    return {
        "n_targets": len(present), "targets": list(present),
        "B1": {"complete": b1_ok, "per_target": b1, "n_pass": n1, "need": B1_MIN, "margin": B1_MARGIN,
               "pass": bool(b1_ok and n1 >= B1_MIN)},
        "B2": {"complete": b2_ok, "per_target": b2, "n_pass": n2, "need": B2_MIN, "mean_diff": mean_diff,
               "margin": B2_MARGIN, "hw_mean3": hw, "hw_line": HW_LINE, "hw_ok": hw_ok,
               "pass": bool(b2_ok and n2 >= B2_MIN and mean_diff is not None and mean_diff >= B2_MARGIN and hw_ok)},
        "B3": {"complete": b3_ok, "per_target_mean": b3m, "per_target_oracle": b3o, "n_pass_mean": n3m,
               "n_pass_oracle": n3o, "need": B3_MIN, "pass": bool(b3_ok and n3m >= B3_MIN and n3o >= B3_MIN)},
    }


def verdict_code(g: Mapping[str, dict]) -> str:
    if not (g["B1"]["complete"] and g["B2"]["complete"] and g["B3"]["complete"]):
        return "BO-PENDING"
    if not g["B1"]["pass"]:
        return "BO-DEAD"
    if g["B2"]["pass"] and g["B3"]["pass"]:
        return "BO-WIN"
    if not g["B2"]["pass"]:
        return "BO-HEADROOM"
    return "BO-LUCK"


# ================================================================ 数据装配
def deploy_dir_of(dataset: str) -> Path:
    return DEPLOY_HAR if dataset == "HAR" else DEPLOY


def protocol_of(dataset: str) -> str:
    return "legacy_full" if dataset == "HAR" else _g39.PROTOCOL


def load_catalog(dataset: str) -> tuple[dict, list[dict], list[dict]]:
    """(candidates[dataset], S1 目录行 seed 0（含 error 行）, S2 终评行)。"""
    d = deploy_dir_of(dataset)
    cands = json.loads((d / "candidates.json").read_text(encoding="utf-8"))
    if dataset not in cands:
        raise SystemExit(f"{d / 'candidates.json'} 无 {dataset}")
    rows = list(read_jsonl(d / "runs.jsonl")) if (d / "runs.jsonl").exists() else []
    s1 = [r for r in rows if r.get("stage") == "s1_catalog" and r["dataset"] == dataset and int(r["seed"]) == 0]
    s2 = [r for r in rows if r.get("stage") == "s2_finals" and r["dataset"] == dataset and r.get("error") is None]
    return cands[dataset], s1, s2


def load_bo_rows(outdir: Path, dataset: str) -> tuple[list[dict], list[dict]]:
    """(bo_round 行, bo_finals 行) of dataset。"""
    p = outdir / "runs.jsonl"
    rows = [r for r in read_jsonl(p) if r["dataset"] == dataset] if p.exists() else []
    return ([r for r in rows if r.get("stage") == "bo_round"],
            [r for r in rows if r.get("stage") == "bo_finals" and r.get("error") is None])


def arm_rows(bo_round_rows: Sequence[dict], arm: str) -> list[dict]:
    return sorted((r for r in bo_round_rows if str(r.get("arm", "")).startswith(arm + "_r")),
                  key=lambda r: (int(r.get("round", 0)), r["cls_key"]))


_SRC_CACHE: dict[Optional[str], list[dict]] = {}


def source_rows(dataset: str) -> list[dict]:
    """源池 z 行（LODO：靶在源池内则剔除它的切片；HAR 不在源池 → 全 8 源），与 39 号 cmd_nominate 同式。"""
    rows = [r for r in read_jsonl(MX3 / "rprime_cache.jsonl") if r.get("error") is None and r["origin"] != "ggs"]
    pool = {r["dataset"] for r in rows}
    ex = dataset if dataset in pool else None
    if ex not in _SRC_CACHE:
        _SRC_CACHE[ex] = _g36.zscore_within([r for r in rows if r["dataset"] != ex])
    return _SRC_CACHE[ex]


def catalog_keys_of(cands: dict) -> set[str]:
    return ({e["cls_key"] for e in cands["shared"]} | {cands["ggs"]["cls_key"]}
            | {e["cls_key"] for e in cands.get("nominated", [])})


def budgets_of(s1_rows: Sequence[dict]) -> list[int]:
    bs = sorted({int(c["iters"]) for r in s1_rows if r.get("checkpoint_results") for c in r["checkpoint_results"]})
    if len(bs) != 3:
        raise SystemExit(f"目录截点不是 3 个：{bs}")
    return bs


def three_seed(dataset: str, key: str, seed0: float, finals_rows: Sequence[dict]) -> Optional[float]:
    """3 种子均值：seed 0 = S1 / bo_round 真值；seed 1/2 = WP3 S2 或本包 bo_finals（同 cls_key 即可，臂不限）。"""
    by_seed: dict[int, float] = {}
    for r in finals_rows:
        if r["dataset"] == dataset and r["cls_key"] == key and int(r["seed"]) in FINAL_SEEDS and r.get("acc") is not None:
            by_seed.setdefault(int(r["seed"]), float(r["acc"]))
    if any(s not in by_seed for s in FINAL_SEEDS):
        return None
    return float(np.mean([seed0] + [by_seed[s] for s in FINAL_SEEDS]))


def seed_of(dataset: str, arm: str, rnd: int) -> int:
    return SEED_BASE + 100 * LOOP_ORDER.index(dataset) + 10 * ARMS.index(arm) + int(rnd)


def load_state(outdir: Path) -> dict:
    p = outdir / "bo_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(outdir: Path, state: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bo_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


# ================================================================ 模拟 / 分析
def simulate_target(dataset: str, outdir: Path) -> dict:
    cands, s1, s2 = load_catalog(dataset)
    bo_rows, bo_finals = load_bo_rows(outdir, dataset)
    budgets = budgets_of(s1)
    measured = measured_from_rows(list(s1) + list(bo_rows))
    traj = {k: v["traj"] for k, v in measured.items()}
    secs = {k: v["secs"] for k, v in measured.items()}
    shared_keys = [e["cls_key"] for e in cands["shared"]]
    ggs_key = cands["ggs"]["cls_key"]
    nominated = {e["cls_key"]: float(e["pred_z"]) for e in cands.get("nominated", [])}
    prior = {k: float(v) for k, v in cands["prior"].items()}
    per_arm_rows = {a: arm_rows(bo_rows, a) for a in ARMS}
    bo_keys = {a: [r["cls_key"] for r in rs] for a, rs in per_arm_rows.items()}
    table = policy_table(traj, secs, budgets=budgets, prior=prior, nominated=nominated, shared_keys=shared_keys,
                         ggs_key=ggs_key, bo_keys=bo_keys)
    bf = budgets[-1]
    finals_all = list(s2) + list(bo_finals)
    mean3 = {}
    for name in (BASELINE, MAIN, CONTROL):
        pk = table["policies"][name]["pick"]
        mean3[name] = three_seed(dataset, pk, table["policies"][name]["acc"], finals_all)
    # 提名明细（含 pred/std/汉明/真值）与逐轮 best-so-far
    state = load_state(outdir).get(dataset, {})
    nominee_rows: dict[str, list[dict]] = {}
    best_so_far: dict[str, list[float]] = {}
    for a in ARMS:
        meta = {}
        for rnd, st in (state.get(a) or {}).items():
            for e in st.get("proposals", []):
                meta[e["cls_key"]] = {**e, "round": int(rnd)}
        det = []
        for r in per_arm_rows[a]:
            t = traj.get(r["cls_key"], {})
            m = meta.get(r["cls_key"], {})
            det.append({"round": int(r.get("round", m.get("round", 0))), "cls_key": r["cls_key"],
                        "pred_z": m.get("pred_z"), "std_z": m.get("std_z"), "score": m.get("score"),
                        "hamming": m.get("hamming"), "acc25": t.get(budgets[0]), "acc100": t.get(bf),
                        "error": r.get("error")})
        nominee_rows[a] = det
        cur = table["oracle_catalog"]
        curve = [cur]
        for rnd in range(1, ROUNDS + 1):
            for e in det:
                if e["round"] == rnd and e["acc100"] is not None:
                    cur = max(cur, float(e["acc100"]))
            curve.append(cur)
        best_so_far[a] = curve

    def rho(pairs):
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            return None
        v = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation
        return None if v is None or np.isnan(v) else float(v)

    spear = {"oneshot": rho([(nominated[k], traj[k][bf]) for k in table["keys"] if k in nominated])}
    for a in ARMS:
        spear[a] = rho([(e["pred_z"], e["acc100"]) for e in nominee_rows[a]])
    # C0：SH6-3-3_prior 与 WP3 analysis.json 逐位
    wp3 = {}
    ap = deploy_dir_of(dataset) / "analysis.json"
    if ap.exists():
        pt = json.loads(ap.read_text(encoding="utf-8")).get("per_target", {}).get(dataset)
        if pt:
            ref = pt.get("arms", {}).get(BASELINE)
            wp3 = {"wp3_acc": ref, "wp3_pick": pt.get("picks", {}).get(BASELINE),
                   "match": (ref is not None and abs(float(ref) - table["policies"][BASELINE]["acc"]) < 1e-12
                             and pt.get("picks", {}).get(BASELINE) == table["policies"][BASELINE]["pick"])}
    n_rounds = {a: (max((int(r.get("round", 0)) for r in per_arm_rows[a]), default=0)) for a in ARMS}
    n_err = {a: sum(1 for r in per_arm_rows[a] if r.get("error")) for a in ARMS}
    return {"dataset": dataset, "protocol": protocol_of(dataset), **table, "mean3": mean3,
            "n_rounds_done": n_rounds, "n_nominee_error": n_err, "nominee_rows": nominee_rows,
            "best_so_far": best_so_far, "spearman_pred_vs_final": spear, "wp3_check": wp3,
            "paper_ts2vec": (_g39.paper_ts2vec(dataset) if dataset != "HAR" else _g39.HAR_TABLE2["HAR"]["TS2Vec"]),
            "ggs_key": ggs_key, "n_finals_rows": len(finals_all)}


def _f(x, nd=3) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def markdown_tables(per_target: Mapping[str, dict], g: Optional[dict], verdict: Optional[str]) -> str:
    out = []
    if g is not None:
        out += ["## 门与读数", "", f"读数码 **{verdict}**（n_targets={g['n_targets']}）", "",
                "| 门 | 通过 | 计数 | 备注 |", "|---|---|---|---|",
                f"| B1 头顶空间 | {g['B1']['pass']} | {g['B1']['n_pass']}/{g['n_targets']} ≥ {g['B1']['need']} | bo 提名 @100% 最大 > 目录 oracle + {g['B1']['margin']} |",
                f"| B2 部署 | {g['B2']['pass']} | {g['B2']['n_pass']}/{g['n_targets']} ≥ {g['B2']['need']} | 均值差 {_f(g['B2']['mean_diff'], 4)} ≥ {g['B2']['margin']}；HW {_f(g['B2']['hw_mean3'])} ≥ {g['B2']['hw_line']} → {g['B2']['hw_ok']} |",
                f"| B3 引导 | {g['B3']['pass']} | 均值 {g['B3']['n_pass_mean']} / oracle {g['B3']['n_pass_oracle']} ≥ {g['B3']['need']} | bo 提名均值 > rand；池 oracle(bo) ≥ 池 oracle(rand) |",
                ""]
    out += ["## 逐靶策略表（acc@100%；regret 对目录 oracle / 对本臂池 oracle；成本 = 全跑当量）", ""]
    for d, pt in per_target.items():
        out += [f"### {d}（{pt['protocol']}，截点 {pt['budgets']}，目录 {pt['n_catalog']} 条，提名 bo {len(pt['arm_keys'].get('bo_ucb', []))} / rand {len(pt['arm_keys'].get('rand_nbr', []))}；"
                f"目录 oracle {_f(pt['oracle_catalog'])} [{pt['oracle_catalog_key'][:8]}]；TS2Vec 论文 {_f(pt.get('paper_ts2vec'))}）", "",
                "| 策略 | pick | acc | regret_cat | regret_pool | 成本 | 墙钟 s | 3 种子 | 起跑 n |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in POLICIES:
            p = pt["policies"][name]
            m3 = pt["mean3"].get(name)
            out.append(f"| {name} | {p['pick'][:8]}{'' if p['in_catalog'] else ' ★'} | {_f(p['acc'])} | {_f(p['regret_catalog'])} | "
                       f"{_f(p['regret_pool'])} | {p['cost_full']:.2f} | {_f(p['cost_seconds'], 0)} | {_f(m3)} | {p['n_start']} |")
        out.append("")
        nm, nmean, op = pt["nominee_max"], pt["nominee_mean"], pt["oracle_pool"]
        out.append(f"提名 @100%：bo 最大 {_f(nm.get('bo_ucb'))} / 均值 {_f(nmean.get('bo_ucb'))}；rand 最大 {_f(nm.get('rand_nbr'))} / 均值 {_f(nmean.get('rand_nbr'))}；"
                   f"池 oracle bo {_f(op.get('bo_ucb'))} / rand {_f(op.get('rand_nbr'))}；best-so-far bo {[round(x, 3) for x in pt['best_so_far']['bo_ucb']]} "
                   f"rand {[round(x, 3) for x in pt['best_so_far']['rand_nbr']]}；Spearman(pred, @100%) 一次性 {_f(pt['spearman_pred_vs_final'].get('oneshot'), 2)} / "
                   f"bo {_f(pt['spearman_pred_vs_final'].get('bo_ucb'), 2)} / rand {_f(pt['spearman_pred_vs_final'].get('rand_nbr'), 2)}；"
                   f"WP3 逐位 {pt['wp3_check'].get('match')}；提名 error bo {pt['n_nominee_error']['bo_ucb']} / rand {pt['n_nominee_error']['rand_nbr']}")
        out.append("")
        rows = [e for a in ARMS for e in pt["nominee_rows"][a]]
        if rows:
            out += ["| 臂 | 轮 | cls | pred_z | std | 汉明 | @25% | @100% |", "|---|---:|---|---:|---:|---:|---:|---:|"]
            for a in ARMS:
                for e in pt["nominee_rows"][a]:
                    out.append(f"| {a} | {e['round']} | {e['cls_key'][:8]} | {_f(e['pred_z'], 2)} | {_f(e['std_z'], 2)} | {e['hamming'] if e['hamming'] is not None else '—'} | "
                               f"{_f(e['acc25'])} | {_f(e['acc100'])}{' ERR' if e['error'] else ''} |")
            out.append("")
    return "\n".join(out) + "\n"


def cmd_simulate(outdir: Path, datasets: Sequence[str], *, label: str) -> dict:
    per = {}
    for d in datasets:
        try:
            per[d] = simulate_target(d, outdir)
        except SystemExit as e:
            print(f"[simulate {d}] 跳过：{e}", flush=True)
            continue
        p = per[d]["policies"]
        print(f"[simulate {d}] oracle_cat {per[d]['oracle_catalog']:.3f} | " + " | ".join(
            f"{n} {p[n]['acc']:.3f}({p[n]['cost_full']:.2f})" for n in POLICIES)
              + f" | WP3 逐位 {per[d]['wp3_check'].get('match')}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{label}.json").write_text(json.dumps(per, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / f"{label}_tables.md").write_text(markdown_tables(per, None, None), encoding="utf-8")
    return per


def cmd_analyze(outdir: Path, datasets: Sequence[str]) -> dict:
    per = cmd_simulate(outdir, datasets, label="simulate")
    uea = {d: per[d] for d in TARGETS6 if d in per}
    g = gates(uea)
    verdict = verdict_code(g)
    har = per.get("HAR")
    har_read = None
    if har:
        m = har["mean3"].get(MAIN)
        har_read = {"bo_sh_mean3": m, "line": HAR_LINE, "pass": (None if m is None else bool(m >= HAR_LINE)),
                    "sh633_mean3": har["mean3"].get(BASELINE), "oracle_catalog": har["oracle_catalog"],
                    "nominee_max": har["nominee_max"]}
    preds = {"B1": {"prior": PREDICTIONS["B1"], "hit": g["B1"]["pass"] if g["B1"]["complete"] else None},
             "B2": {"prior": PREDICTIONS["B2"], "hit": g["B2"]["pass"] if g["B2"]["complete"] else None},
             "B3": {"prior": PREDICTIONS["B3"], "hit": g["B3"]["pass"] if g["B3"]["complete"] else None},
             "HAR": {"prior": PREDICTIONS["HAR"], "hit": (har_read or {}).get("pass")}}
    analysis = {"verdict": verdict, "gates": g, "har": har_read, "predictions": preds,
                "per_target": {d: {k: v for k, v in pt.items() if k != "nominee_rows"} for d, pt in per.items()},
                "nominee_rows": {d: pt["nominee_rows"] for d, pt in per.items()}}
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    md = markdown_tables(per, g, verdict)
    if har_read:
        md += f"\n## HAR（描述性）\n\nBO-SH 3 种子 {_f(har_read['bo_sh_mean3'])} vs 线 {HAR_LINE} → {har_read['pass']}；SH6-3-3 {_f(har_read['sh633_mean3'])}；目录 oracle {_f(har_read['oracle_catalog'])}；提名最大 {har_read['nominee_max']}\n"
    (outdir / "tables.md").write_text(md, encoding="utf-8")
    print(f"[analyze] 读数码 {verdict}；B1 {g['B1']['n_pass']}/{g['n_targets']} B2 {g['B2']['n_pass']} (Δ {_f(g['B2']['mean_diff'], 4)}, HW {g['B2']['hw_ok']}) "
          f"B3 {g['B3']['n_pass_mean']}/{g['B3']['n_pass_oracle']}；HAR {har_read}", flush=True)
    return analysis


# ================================================================ GPU 循环
def cmd_dry_run(datasets: Sequence[str], *, rounds: int, per_round: int) -> None:
    total = 0.0
    for d in datasets:
        n = len(ARMS) * rounds * per_round
        sec = CELL_SECONDS_4090.get(d, 0.0) * LOCAL_SLOWDOWN if d in CELL_SECONDS_4090 else CELL_SECONDS_4060.get(d, 0.0)
        est = n * sec
        total += est
        print(f"[dry-run {d}] {n} 格 × ≈{sec:.0f} s（4060 估）≈ {est / 3600:.2f} h", flush=True)
    print(f"[dry-run] 合计 ≈ {total / 3600:.2f} h（不含终评；估时锚点带硬件标签：4090 中位 × {LOCAL_SLOWDOWN}）", flush=True)


def cmd_loop(args) -> None:
    outdir = Path(args.outdir)
    protos = _g35.load_protocols()
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    datasets = [d for d in LOOP_ORDER if d in set(args.datasets)]
    arms = tuple(args.arms)
    sizes = _g39.sizes_for(datasets, load=True, har_npz=args.har_npz)
    state = load_state(outdir)
    for d in datasets:
        proto_name = protocol_of(d)
        cands, s1, _ = load_catalog(d)
        budgets = budgets_of(s1)
        b25 = budgets[0]
        src = source_rows(d)
        cat_keys = catalog_keys_of(cands)
        for rnd in range(1, int(args.rounds) + 1):
            for arm in arms:
                bo_rows, _ = load_bo_rows(outdir, d)
                measured = measured_from_rows(list(s1) + arm_rows(bo_rows, arm))
                st = state.setdefault(d, {}).setdefault(arm, {})
                if str(rnd) in st:
                    entries = st[str(rnd)]["proposals"]
                    print(f"[loop {d}/{arm}/r{rnd}] 复用已落盘提名 {len(entries)} 条", flush=True)
                else:
                    parents_keys = parents_of(measured, b25, k=N_PARENTS_T)
                    parents = [CLS(**measured[k]["cls"]) for k in parents_keys]
                    tgt = target_rows(measured, b25, task_id=f"{d}@{b25}", copies=COPIES)
                    proposed = {e["cls_key"] for s in st.values() for e in s.get("proposals", [])}
                    exclude = cat_keys | set(measured) | proposed
                    entries = nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude,
                                             n=int(args.per_round), pool=POOL, radius=RADIUS, mode=MODE_OF[arm],
                                             kappa=KAPPA, bagging=BAGGING, seed=seed_of(d, arm, rnd))
                    st[str(rnd)] = {"parents": parents_keys,
                                    "parents_acc25": [measured[k]["traj"][b25] for k in parents_keys],
                                    "feedback_budget": b25, "n_measured": len(measured),
                                    "n_target_rows": len(tgt), "n_source_rows": len(src),
                                    "seed": seed_of(d, arm, rnd), "proposals": entries}
                    save_state(outdir, state)
                    print(f"[loop {d}/{arm}/r{rnd}] 父本 {[k[:8] for k in parents_keys]} @{b25} "
                          f"{[round(measured[k]['traj'][b25], 3) for k in parents_keys]}；提名 {len(entries)} 条 "
                          f"pred {[round(e['pred_z'], 2) for e in entries]} std {[round(e['std_z'], 2) for e in entries]} "
                          f"汉明 {[e['hamming'] for e in entries]}（池 {entries[0]['n_pool'] if entries else 0} / 半径内 "
                          f"{entries[0]['n_in_radius'] if entries else 0}）", flush=True)
                if not entries:
                    print(f"[loop {d}/{arm}/r{rnd}] 无提名，跳过", flush=True)
                    continue
                cells = round_cells(d, entries, arm=arm, rnd=rnd, protos=protos, sizes=sizes, protocol=proto_name)
                _g39.execute(cells, args, f"{d}/{arm}/r{rnd}")
        print(f"[loop {d}] 完成 {args.rounds} 轮 × {arms}", flush=True)


def cmd_finals(args) -> None:
    outdir = Path(args.outdir)
    protos = _g35.load_protocols()
    datasets = [d for d in LOOP_ORDER if d in set(args.datasets)]
    sizes = _g39.sizes_for(datasets, load=True, har_npz=args.har_npz)
    cells = []
    for d in datasets:
        sim = simulate_target(d, outdir)
        cands, s1, s2 = load_catalog(d)
        bo_rows, bo_finals = load_bo_rows(outdir, d)
        measured = measured_from_rows(list(s1) + list(bo_rows))
        finals_all = list(s2) + list(bo_finals)
        by_key: dict[str, list[str]] = {}
        for name in (MAIN, CONTROL):
            by_key.setdefault(sim["policies"][name]["pick"], []).append("final_" + name)
        p = protos[protocol_of(d)]
        for k, names in by_key.items():
            have = {int(r["seed"]) for r in finals_all if r["cls_key"] == k}
            for s in FINAL_SEEDS:
                if s in have:
                    continue
                cell = _g35._cell("bo_finals", d, protocol_of(d), p, sizes, arm_names=names, cls_key_=k,
                                  cls=CLS(**measured[k]["cls"]), seed=int(s))
                cell["checkpoints"] = list(_g39.checkpoints_for(cell["iters"]))
                cells.append(cell)
        champ = ", ".join(f"{n}: {sim['policies'][n]['pick'][:8]}" for n in (MAIN, CONTROL))
        print(f"[finals {d}] 冠军 {{{champ}}}；待补种子格 {sum(1 for c in cells if c['dataset'] == d)}", flush=True)
    _g39.execute(cells, args, "finals")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="零 GPU：在缓存上模拟策略表（阶段 0 / 中途随时）")
    ap.add_argument("--label", default="simulate", help="--simulate 落盘文件名（如 stage0）")
    ap.add_argument("--loop", action="store_true", help="GPU：3 轮 × 每轮 5 条提名，逐靶逐臂")
    ap.add_argument("--finals", action="store_true", help="GPU：BO-SH / RAND-SH 冠军补 seed 1、2")
    ap.add_argument("--analyze", action="store_true", help="零 GPU：策略表 + 门 + 读数码 → analysis.json / tables.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS6))
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--per-round", type=int, default=PER_ROUND)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--joblib-backend", default="threading", choices=("threading", "loky"))
    ap.add_argument("--har-npz", default=str(_g39.HAR_NPZ_DEFAULT))
    args = ap.parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    if args.dry_run:
        cmd_dry_run(args.datasets, rounds=args.rounds, per_round=args.per_round)
    if args.simulate:
        cmd_simulate(outdir, args.datasets, label=args.label)
    if args.loop:
        cmd_loop(args)
    if args.finals:
        cmd_finals(args)
    if args.analyze:
        cmd_analyze(outdir, args.datasets)
    if not any((args.dry_run, args.simulate, args.loop, args.finals, args.analyze)):
        ap.print_help()


if __name__ == "__main__":
    main()
