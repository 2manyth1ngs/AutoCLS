"""WP9-P1 路径一：官方口径 6 靶候选池扩到 100 条，零 GPU 在缓存上模拟"学来的名单 + 淘汰赛"与标准 HPO 的同成本对照。

协议 results/mx4_p1/PROTOCOL.md（判读带 E1–E3、预测 P1–P10 先于数据写死）；真源 configs/p1.yaml。薄驱动：importlib
复用 35（工单格 / 缓存键 / 数据装载）、36（先验 / SH 纯函数）、39（截点 / 轨迹装配 / 执行 / finals 工单 / deploy_arms）、
43（源池 LODO z 行 / Windows 评估超时）、45（drop_repeated_oom）；35/36/39/43/45/autocls 零改动。

候选池（每靶 100 条，seed 0 完整 25/50/100% 轨迹）= WP3 目录 21（results/mx4_deploy）+ WP8 提名 30（results/mx4_bo）
+ 空间均匀随机 49–51（本包新跑，arm=random_space，stage=s1_pool）。三源 options / protocol / iters 逐字段断言相同后合池。
我们的臂：prior1/3、SH6-3-3_prior（3.75）、SH10-4-3_prior（5.0，头条）、hybrid16-8-3（7.5）、prior6+sur10-8-3、SH16-8-3_surrogate。
基线（N_DRAWS 固定种子抽样）：RS-n（解析）、随机起点 SH、Hyperband η=2 三支架（HB-7 / HB-5 / HB-lite / HBx2）、离散池 TPE。
BOHB 不报数（KDE 需 ≥ dim+1 观测，7.5 当量内任一保真 ≤ 5 → 退化为 HB）。

子命令（--outdir 默认 results/mx4_p1）：
    python scripts/46_p1_pool.py --build-pool                 # 零 GPU：pool.json（协议锁产物）
    python scripts/46_p1_pool.py --dry-run                    # 格数 + 估时（4090 锚）
    python scripts/46_p1_pool.py --simulate --c0              # 零 GPU：只验我们的臂与 WP3 / WP6 / WP8 逐位（不出基线列）
    python scripts/46_p1_pool.py --run --resume               # S1：297 格（GPU）
    python scripts/46_p1_pool.py --simulate                   # S2：全部臂 + 基线 → simulate.json / tables.md
    python scripts/46_p1_pool.py --analyze                    # 门 E1–E3、预测、picks.json → analysis.json
    python scripts/46_p1_pool.py --finals --resume            # S3：冠军 + 池前 3 × seed 1/2（复用三源已有终评行）
    python scripts/46_p1_pool.py --ucr --ablation --cost      # 附带零 GPU 交付
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED, _sample_unique, cls_key  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts.e_ablation_common import read_jsonl  # noqa: E402

_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g39 = importlib.import_module("scripts.39_sh_deploy")
_g43 = importlib.import_module("scripts.43_bo_nominate")
_g45 = importlib.import_module("scripts.45_uea30_deploy")
expected_max_of_k = _g36.expected_max_of_k

CFG = ROOT / "configs" / "p1.yaml"
OUT = ROOT / "results" / "mx4_p1"
MAIN_ARM = "SH10-4-3_prior"
OURS = ("prior1", "prior3", "SH6-3-3_prior", "SH10-4-3_prior", "hybrid16-8-3", "prior6+sur10-8-3", "SH16-8-3_surrogate")
SH_DEFS = {"SH6-3-3_prior": (6, 3, 3), "SH10-4-3_prior": (10, 4, 3), "hybrid16-8-3": (16, 8, 3),
           "prior6+sur10-8-3": (16, 8, 3), "SH16-8-3_surrogate": (16, 8, 3)}
FINAL_SEEDS = tuple(_g39.FINAL_SEEDS)
LAUNCH_S = 10.0
CONSERVATIVE = 1.4
SMOKE_ITERS = 8


# ---------------------------------------------------------------- 配置真源
def load_cfg(path: str | Path = CFG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cfg_md5(path: str | Path = CFG) -> str:
    return hashlib.md5(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_targets(cfg: Mapping) -> list[str]:
    return [d for _, d in sorted((int(c["order"]), d) for d, c in cfg["targets"].items())]


def sizes_from_cfg(cfg: Mapping, datasets: Optional[Sequence[str]] = None) -> dict[str, tuple[int, int, int]]:
    ds = list(datasets) if datasets is not None else run_targets(cfg)
    return {d: (int(cfg["targets"][d]["n_train"]), int(cfg["targets"][d]["L"]), int(cfg["targets"][d]["C"])) for d in ds}


def budgets_for(cfg: Mapping, d: str) -> list[int]:
    return list(_g39.checkpoints_for(int(cfg["targets"][d]["iters"])))


def _protos() -> dict:
    return _g35.load_protocols()


# ---------------------------------------------------------------- 三源合池
_OPT_FIELDS = ("protocol", "iters", "checkpoints", "options")
POOL_STAGES = ("s1_catalog", "bo_round", "s1_pool")   # 三源目录型行（终评行不入池）


def _complete(r: Mapping, n_ckpt: int = 3) -> bool:
    return r.get("error") is None and len(r.get("checkpoint_results") or []) == n_ckpt


def load_pool_rows(dataset: str, wp3_rows: Sequence[dict], wp8_rows: Sequence[dict], new_rows: Sequence[dict]) -> dict:
    """三源同靶 seed-0 目录格合并（含 error 行）：options/protocol/iters/checkpoints 逐字段断言相同；同 cls_key 多行按源序
    wp3 < wp8 < new、同源按 arm 升序取第一条，其余记 duplicates（免费同机复现读数）。"""
    def pick(rows):
        return [r for r in rows if r["dataset"] == dataset and r.get("stage") in POOL_STAGES and int(r.get("seed", 0)) == 0]
    src = [("wp3", pick(wp3_rows)), ("wp8", pick(wp8_rows)), ("new", pick(new_rows))]
    ref = None
    for name, rows in src:
        for r in rows:
            sig = {k: r.get(k) for k in _OPT_FIELDS}
            if ref is None:
                ref = sig
            elif sig != ref:
                raise SystemExit(f"{dataset}: 源 {name} 行 {r['cls_key']} 的 {_OPT_FIELDS} 与其它源不同：{sig} vs {ref}")
    by_key: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for si, (name, rows) in enumerate(src):
        for r in sorted(rows, key=lambda r: str(r.get("arm"))):
            by_key[r["cls_key"]].append((si, name, r))
    out_rows, dups, source_of, seen, err = [], [], {}, set(), set()
    for k, lst in by_key.items():
        lst.sort(key=lambda t: (t[0], str(t[2].get("arm"))))
        keep = next((t for t in lst if _complete(t[2])), lst[0])   # 优先取完整行（同键 error 行让位）
        out_rows.append(keep[2])
        source_of[k] = keep[1]
        seen.add(k)
        if keep[2].get("error") is not None or not _complete(keep[2]):
            err.add(k)
        others = [t for t in lst if t[2] is not keep[2]]
        if others:
            dups.append({"cls_key": k, "kept": keep[2].get("arm"), "dropped": [t[2].get("arm") for t in others],
                         "acc_final": [t[2].get("acc") for t in [keep] + others]})
    return {"rows": out_rows, "seen_keys": seen, "error_keys": err, "duplicates": dups, "source_of": source_of}


def read_sources(cfg: Mapping, outdir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    wp3 = read_jsonl(ROOT / cfg["pool"]["wp3_runs"]) if (ROOT / cfg["pool"]["wp3_runs"]).exists() else []
    wp8 = read_jsonl(ROOT / cfg["pool"]["wp8_runs"]) if (ROOT / cfg["pool"]["wp8_runs"]).exists() else []
    new = read_jsonl(outdir / "runs.jsonl") if (outdir / "runs.jsonl").exists() else []
    return wp3, wp8, new


# ---------------------------------------------------------------- 采样 / 先验 / 代理 / 三序
def sample_random_entries(space: CLSSpace, *, seed: int, n: int, taken) -> list[dict]:
    """空间均匀随机（sample_random 天然 input_mask=none），去重前缀稳定；taken 里的键跳过。"""
    cfgs = _sample_unique(space, np.random.default_rng(int(seed)), int(n), set(taken))
    return [{"cls_key": cls_key(c), "cls": c.to_dict(), "random_idx": i} for i, c in enumerate(cfgs)]


def pool_orders(keys: Sequence[str], prior: Mapping[str, float], pred_z: Mapping[str, float], *,
                prior_keys: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """prior_order：有先验的按先验降序，其余按 key 垫底；hybrid_order：有先验的置顶 + 其余按代理；surrogate_order：全按代理。"""
    keys = list(keys)
    pk = [k for k in keys if k in set(prior_keys)]
    rest = [k for k in keys if k not in set(prior_keys)]
    prior_order = _g36.rank_by(prior, pk) + sorted(rest)
    hybrid_order = _g36.rank_by(prior, pk) + _g36.rank_by(pred_z, rest)
    surrogate_order = _g36.rank_by(pred_z, keys)
    return prior_order, hybrid_order, surrogate_order


_SUR_CACHE: dict[str, tuple] = {}


def fit_surrogate(cfg: Mapping, dataset: str):
    """与 39.cmd_nominate 同式：LODO 源池 z 行 → FieldSurrogate("hgb", seed=NOMINEE_SEED+idx)；特征 CLSEncoder(binned) ⊕ b/290。"""
    if dataset in _SUR_CACHE:
        return _SUR_CACHE[dataset]
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    rows = _g43.source_rows(dataset)
    max_b = max(int(r["iters"]) for r in rows)
    seed = int(cfg["pool"]["surrogate"]["nominee_seed"]) + int(cfg["targets"][dataset]["nominee_idx"])
    model = FieldSurrogate(cfg["pool"]["surrogate"]["kind"], seed=seed).fit(
        _g36.features(rows, enc, max_b), np.array([float(r["z"]) for r in rows]))
    _SUR_CACHE[dataset] = (model, enc)
    return model, enc


def surrogate_predict(model, enc: CLSEncoder, cls_dicts: Sequence[Mapping]) -> np.ndarray:
    X = np.stack([enc.encode(CLS(**c)) for c in cls_dicts])
    pred, _ = model.predict(np.hstack([X, np.ones((len(cls_dicts), 1))]))
    return np.asarray(pred, dtype=float)


def build_pool(cfg: Mapping, outdir: Path, datasets: Optional[Sequence[str]] = None) -> dict:
    ds = list(datasets) if datasets else run_targets(cfg)
    wp3, wp8, new = read_sources(cfg, outdir)
    cands = json.loads((ROOT / cfg["pool"]["candidates"]).read_text(encoding="utf-8"))
    space = CLSSpace.from_yaml()
    out = {"config_md5": cfg_md5(), "n_target": int(cfg["pool"]["n_target"]), "seed_base": int(cfg["pool"]["seed_base"]),
           "arm": cfg["pool"]["arm"], "stage": cfg["pool"]["stage"], "targets": {}}
    for d in ds:
        c = cands[d]
        merged = load_pool_rows(d, wp3, wp8, new)
        cls_of = {r["cls_key"]: r["cls"] for r in merged["rows"]}
        origin_of = {e["cls_key"]: "shared" for e in c["shared"]}
        origin_of[c["ggs"]["cls_key"]] = "ggs"
        origin_of.update({e["cls_key"]: "nominated_wp3" for e in c.get("nominated", [])})
        for r in merged["rows"]:
            if r.get("stage") == "bo_round":
                origin_of[r["cls_key"]] = f"{r.get('bo_arm')}_wp8"
            elif r.get("stage") == "s1_pool":
                origin_of[r["cls_key"]] = "random_space"
        complete = [r["cls_key"] for r in merged["rows"] if _complete(r)]
        n_random = int(cfg["pool"]["n_target"]) - len(complete)
        seed = int(cfg["pool"]["seed_base"]) + int(cfg["targets"][d]["order"])
        rand = sample_random_entries(space, seed=seed, n=max(n_random, 0), taken=merged["seen_keys"]) if n_random > 0 else []
        entries = []
        order_rank = {"shared": 0, "ggs": 1, "nominated_wp3": 2, "bo_ucb_wp8": 3, "rand_nbr_wp8": 4, "random_space": 5}
        for r in sorted(merged["rows"], key=lambda r: (order_rank.get(origin_of.get(r["cls_key"], "random_space"), 9), str(r.get("arm")), r["cls_key"])):
            k = r["cls_key"]
            entries.append({"cls_key": k, "cls": cls_of[k], "origin": origin_of.get(k, "random_space"), "source": merged["source_of"][k],
                            "arm_src": r.get("arm"), "prior": (float(c["prior"][k]) if k in c["prior"] else None), "pred_z": None,
                            "random_idx": None, "error": bool(k in merged["error_keys"])})
        for e in rand:
            entries.append({**e, "origin": "random_space", "source": "new", "arm_src": cfg["pool"]["arm"], "prior": None, "pred_z": None, "error": False})
        model, enc = fit_surrogate(cfg, d)
        pz = surrogate_predict(model, enc, [e["cls"] for e in entries])
        for e, z in zip(entries, pz):
            e["pred_z"] = float(z)
        keys = [e["cls_key"] for e in entries if not e["error"]]
        prior = {e["cls_key"]: e["prior"] for e in entries if e["prior"] is not None}
        pred = {e["cls_key"]: e["pred_z"] for e in entries}
        prior_keys = [k for k in keys if k in prior]
        po, ho, so = pool_orders(keys, prior, pred, prior_keys=prior_keys)
        out["targets"][d] = {"order": int(cfg["targets"][d]["order"]), "budgets": budgets_for(cfg, d), "iters": int(cfg["targets"][d]["iters"]),
                             "fit_on": f"LODO excluding {d}", "surrogate_seed": int(cfg["pool"]["surrogate"]["nominee_seed"]) + int(cfg["targets"][d]["nominee_idx"]),
                             "seen_n": len(merged["seen_keys"]), "n_complete_seen": len(complete), "seen_error_keys": sorted(merged["error_keys"]),
                             "duplicates": merged["duplicates"], "n_random": len(rand), "random_seed": seed,
                             "prior_keys": prior_keys, "catalog_keys": [k for k in keys if origin_of.get(k) in ("shared", "ggs", "nominated_wp3")],
                             "curated_keys": [k for k in keys if origin_of.get(k) != "random_space"],
                             "random_keys": [e["cls_key"] for e in rand],
                             "entries": entries, "prior_order": po, "hybrid_order": ho, "surrogate_order": so}
        print(f"[pool {d}] 已见 {len(merged['seen_keys'])}（完整 {len(complete)}，error {len(merged['error_keys'])}，重键 {len(merged['duplicates'])}）"
              f"+ 随机 {len(rand)}（seed {seed}）→ 池 {len(keys)}；先验前 3 {po[:3]}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pool.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def read_pool(outdir: Path) -> dict:
    p = outdir / "pool.json"
    if not p.exists():
        raise SystemExit(f"缺 {p}：先跑 --build-pool（协议锁产物）")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 工单 / 执行
def p1_worklist(protos: dict, sizes: dict, pool: Mapping, *, protocol: str = "ts2vec_official", seed: int = 0,
                stage: str = "s1_pool", arm: str = "random_space") -> list[dict]:
    p = protos[protocol]
    cells = []
    for d, t in pool["targets"].items():
        for e in t["entries"]:
            if e.get("origin") != "random_space" or e.get("error") or e.get("source") != "new":
                continue
            cell = _g35._cell(stage, d, protocol, p, sizes, arm_names=[arm], cls_key_=e["cls_key"], cls=CLS(**e["cls"]), seed=seed)
            cell["checkpoints"] = list(_g39.checkpoints_for(cell["iters"]))
            cells.append(cell)
    return cells


def cmd_run(cfg: Mapping, args) -> None:
    outdir = Path(args.outdir)
    pool = read_pool(outdir)
    datasets = [d for d in run_targets(cfg) if d in pool["targets"] and (not args.datasets or d in args.datasets)]
    sub = {"targets": {d: pool["targets"][d] for d in datasets}}
    cells = p1_worklist(_protos(), sizes_from_cfg(cfg, datasets), sub, protocol=cfg["protocol"],
                        stage=cfg["pool"]["stage"], arm=cfg["pool"]["arm"])
    cells = _g45.drop_repeated_oom(cells, outdir / "oom_failures.jsonl")
    if sys.platform.startswith("win"):
        _g43.install_windows_eval_timeout()
    _g39.execute(cells, args, "S1")


def estimate_hours(cfg: Mapping, pool: Mapping, datasets: Sequence[str], *, launch_s: float = LAUNCH_S,
                   conservative: float = CONSERVATIVE) -> dict:
    per, tot = {}, 0.0
    for d in datasets:
        n = int(pool["targets"][d]["n_random"])
        a = cfg["targets"][d]["anchor_4090"]
        h = n * (float(a["cell_s"]) + launch_s) / 3600.0
        per[d] = {"n_cells": n, "cell_s": float(a["cell_s"]), "cell_max_s": float(a["cell_max_s"]), "hours": h,
                  "hours_max": n * (float(a["cell_max_s"]) + launch_s) / 3600.0}
        tot += h
    return {"hardware": "RTX 4090", "per_dataset": per, "total_h": tot, "conservative_h": conservative * tot,
            "n_cells": int(sum(v["n_cells"] for v in per.values()))}


# ---------------------------------------------------------------- 成本与基线纯函数
def sh_cost_iters(ks: Sequence[int], budgets: Sequence[int]) -> int:
    prev, cost = 0, 0
    for k, b in zip(ks, budgets):
        cost += int(k) * (int(b) - prev)
        prev = int(b)
    return int(cost)


def sh_cost_full(ks: Sequence[int], budgets: Sequence[int]) -> float:
    return sh_cost_iters(ks, budgets) / float(budgets[-1])


def random_search_expect(values: Sequence[float], k: int, beat: Optional[float] = None) -> dict:
    """不放回均匀抽 k 条各全跑取最好：均值 = expected_max_of_k；分位与 P(max > beat) 由次序统计量解析
    （max = v(i) 的组合数 C(i−1, k−1)，分位按 np.quantile 线性插值口径在加权展开序列上算，不枚举组合）。"""
    v = sorted(float(x) for x in values)
    n = len(v)
    assert 1 <= k <= n
    total = math.comb(n, k)
    counts = [math.comb(i - 1, k - 1) for i in range(1, n + 1)]      # max 恰为 v(i) 的组合数
    cum = list(itertools.accumulate(counts))                          # 展开序列里 v(i) 的末位置（1-based）

    def at(j: int) -> float:                                          # 展开序列第 j 个（0-based）
        j = min(max(int(j), 0), total - 1)
        return v[bisect.bisect_right(cum, j)]

    def quantile(q: float) -> float:
        pos = q * (total - 1)
        lo = int(math.floor(pos))
        gamma = pos - lo
        a, b = at(lo), at(lo + 1)
        return float(a + (b - a) * gamma)

    out = {"mean": expected_max_of_k(v, k), "median": quantile(0.5), "q25": quantile(0.25), "q75": quantile(0.75), "k": int(k)}
    if beat is not None:
        m = sum(1 for x in v if x <= beat)                            # P(max ≤ beat) = C(m, k) / C(n, k)
        out["p_beat"] = 1.0 - (math.comb(m, k) / total if m >= k else 0.0)
    return out


def summarize_draws(arr: np.ndarray, beat: Optional[float] = None) -> dict:
    a = np.asarray(arr, dtype=float)
    out = {"mean": float(a.mean()), "median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)),
           "q75": float(np.quantile(a, 0.75)), "n_draws": int(a.size)}
    if beat is not None:
        out["p_beat"] = float(np.mean(a > beat))
    return out


def rand_sh_draws(keys: Sequence[str], traj: Mapping, ks: Sequence[int], budgets: Sequence[int], n_draws: int, seed: int) -> np.ndarray:
    """随机起点 SH；与 36.expected_random_sh 同抽样序（均值逐位相等）。"""
    rng = np.random.default_rng(int(seed))
    keys = list(keys)
    bf = int(budgets[-1])
    out = []
    for _ in range(int(n_draws)):
        order = [keys[i] for i in rng.permutation(len(keys))]
        final, _c = _g36.successive_halving(order, traj, ks, budgets)
        out.append(float(traj[_g36.sh_pick(final, traj, bf)][bf]))
    return np.asarray(out)


def hb_bracket_plan(s: int, budgets: Sequence[int], eta: int = 2) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Hyperband（R = b_final, r_min = b0, s_max = len(budgets)−1）支架 s：n_s = ceil((s_max+1)/(s+1)·η^s)，
    第 i 级 floor(n_s/η^i)，预算 budgets[s_max−s:]。"""
    b = [int(x) for x in budgets]
    s_max = len(b) - 1
    assert 0 <= s <= s_max
    n0 = math.ceil((s_max + 1) / (s + 1) * (eta ** s))
    ks = tuple(int(n0 // (eta ** i)) for i in range(s + 1))
    return ks, tuple(b[s_max - s:])


def hb_cost_iters(budgets: Sequence[int], brackets: Sequence[int], eta: int = 2) -> int:
    return int(sum(sh_cost_iters(*hb_bracket_plan(s, budgets, eta)) for s in brackets))


def hb_n_configs(brackets: Sequence[int], budgets: Sequence[int] = (1, 2, 4), eta: int = 2) -> int:
    return int(sum(hb_bracket_plan(s, budgets, eta)[0][0] for s in brackets))


def hyperband_samples(keys: Sequence[str], brackets: Sequence[int], *, n_draws: int, seed: int, eta: int = 2) -> list[list[str]]:
    keys = list(keys)
    need = hb_n_configs(brackets, eta=eta)
    if len(keys) < need:
        raise SystemExit(f"Hyperband 支架 {list(brackets)} 需要 {need} 条不放回配置，池只有 {len(keys)}")
    rng = np.random.default_rng(int(seed))
    return [[keys[i] for i in rng.choice(len(keys), size=need, replace=False)] for _ in range(int(n_draws))]


def hyperband_draws(keys: Sequence[str], traj: Mapping, budgets: Sequence[int], brackets: Sequence[int], *,
                    n_draws: int, seed: int, eta: int = 2) -> tuple[np.ndarray, list[str]]:
    """每次 draw 不放回抽 Σn_s 条按支架切片；各支架跑 SH；终选 = 到达 b_final 者中真值最高（sh_pick）。"""
    bf = int(budgets[-1])
    accs, picks = [], []
    for sample in hyperband_samples(keys, brackets, n_draws=n_draws, seed=seed, eta=eta):
        pos, reached = 0, []
        for s in brackets:
            ks, bud = hb_bracket_plan(s, budgets, eta)
            sub = sample[pos:pos + ks[0]]
            pos += ks[0]
            path = _g36.sh_path(sub, traj, ks, bud)
            reached += list(path[-1])
        pick = _g36.sh_pick(reached, traj, bf)
        picks.append(pick)
        accs.append(float(traj[pick][bf]))
    return np.asarray(accs), picks


def tpe_gamma(n_obs: int, cap: int = 25) -> int:
    return int(min(math.ceil(0.1 * int(n_obs)), cap))


def tpe_ratio(cand: Mapping, *, good: Sequence[str], bad: Sequence[str], cls_of: Mapping[str, Mapping],
              fields: Sequence[str], choices: Mapping[str, Sequence]) -> float:
    """离散类目 Parzen：每字段 (count+1)/(n_active+m)；候选该字段为 None 或某集合无激活观测 → 该字段不入乘积。"""
    ratio = 1.0
    for f in fields:
        x = cand.get(f)
        if x is None:
            continue
        m = len(choices[f])
        g_act = [cls_of[k].get(f) for k in good if cls_of[k].get(f) is not None]
        b_act = [cls_of[k].get(f) for k in bad if cls_of[k].get(f) is not None]
        if not g_act or not b_act:
            continue
        l = (sum(1 for v in g_act if v == x) + 1) / (len(g_act) + m)
        g = (sum(1 for v in b_act if v == x) + 1) / (len(b_act) + m)
        ratio *= l / g
    return float(ratio)


def tpe_select(observed: Mapping[str, float], candidates: Sequence[str], cls_of: Mapping[str, Mapping],
               fields: Sequence[str], choices: Mapping[str, Sequence]) -> str:
    ranked = sorted(observed, key=lambda k: (-float(observed[k]), k))
    gam = tpe_gamma(len(ranked))
    good, bad = ranked[:gam], ranked[gam:]
    scores = {c: tpe_ratio(cls_of[c], good=good, bad=bad, cls_of=cls_of, fields=fields, choices=choices) for c in candidates}
    return sorted(candidates, key=lambda c: (-scores[c], c))[0]


def tpe_draws(keys: Sequence[str], cls_of: Mapping[str, Mapping], traj: Mapping, budgets: Sequence[int], n: int, *,
              fields: Sequence[str], choices: Mapping[str, Sequence], n_startup: int, n_draws: int, seed: int) -> tuple[np.ndarray, int]:
    """离散池 TPE：n_startup 条随机起点全跑 → 每步在未评估池成员里 argmax l/g → 共 n 次全跑，报最佳真值；成本 n·b_final。"""
    bf = int(budgets[-1])
    keys = list(keys)
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n_draws)):
        start = [keys[i] for i in rng.choice(len(keys), size=min(int(n_startup), len(keys)), replace=False)]
        observed = {k: float(traj[k][bf]) for k in start}
        while len(observed) < int(n):
            cands = [k for k in keys if k not in observed]
            if not cands:
                break
            pick = tpe_select(observed, cands, cls_of, fields, choices)
            observed[pick] = float(traj[pick][bf])
        out.append(max(observed.values()))
    return np.asarray(out), int(n) * bf


# ---------------------------------------------------------------- 我们的臂
def _sh_arm(ranked: Sequence[str], ks: Sequence[int], budgets: Sequence[int], traj: Mapping, secs: Optional[Mapping], oracle: float,
            oracle_cat: float) -> Optional[dict]:
    ranked = [k for k in ranked if k in traj]
    if not ranked:
        return None
    bf = int(budgets[-1])
    path = _g36.sh_path(ranked, traj, ks, budgets)
    pick = _g36.sh_pick(path[-1], traj, bf)
    acc = float(traj[pick][bf])
    ci = _g36.path_cost_iters(path, budgets)
    cs = None
    if secs is not None:
        try:
            cs = _g36.path_cost_seconds(path, budgets, secs)
        except (KeyError, TypeError):
            cs = None
    return {"pick": pick, "acc": acc, "regret_pool": oracle - acc, "regret_catalog": oracle_cat - acc, "cost_iters": int(ci),
            "cost_full": ci / bf, "cost_seconds": cs, "path": [list(s) for s in path], "n_start": len(path[0])}


def our_arms(pool_d: Mapping, traj: Mapping, secs: Optional[Mapping], budgets: Sequence[int]) -> dict:
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    keys = [k for k in traj if all(b in traj[k] for b in stages)]
    truth = {k: float(traj[k][bf]) for k in keys}

    def oracle_of(sub) -> dict:
        sub = [k for k in sub if k in truth]
        if not sub:
            return {"acc": None, "pick": None, "n": 0}
        best = sorted(sub, key=lambda k: (-truth[k], k))[0]
        return {"acc": truth[best], "pick": best, "n": len(sub)}

    oracle = oracle_of(keys)
    o_cat = oracle_of(pool_d.get("catalog_keys", []))
    out = {"oracle_pool": oracle, "oracle_catalog": o_cat, "oracle_curated": oracle_of(pool_d.get("curated_keys", [])),
           "oracle_random": oracle_of(pool_d.get("random_keys", [])), "n_candidates": len(keys), "budgets": stages}
    ocat = o_cat["acc"] if o_cat["acc"] is not None else oracle["acc"]
    prior_order = [k for k in pool_d["prior_order"] if k in truth]
    hybrid_order = [k for k in pool_d["hybrid_order"] if k in truth]
    sur_order = [k for k in pool_d["surrogate_order"] if k in truth]
    for name, top in (("prior1", 1), ("prior3", 3)):
        chosen = prior_order[:top]
        if chosen:
            best = sorted(chosen, key=lambda k: (-truth[k], k))[0]
            out[name] = {"pick": best, "acc": truth[best], "regret_pool": oracle["acc"] - truth[best], "regret_catalog": ocat - truth[best],
                         "cost_iters": top * bf, "cost_full": float(top), "path": [list(chosen)], "n_start": len(chosen),
                         "cost_seconds": (float(sum(float(secs[k][bf]) for k in chosen)) if secs is not None and all(k in secs for k in chosen) else None)}
    out["SH6-3-3_prior"] = _sh_arm(prior_order[:6], SH_DEFS["SH6-3-3_prior"], stages, traj, secs, oracle["acc"], ocat)
    out["SH10-4-3_prior"] = _sh_arm(prior_order[:10], SH_DEFS["SH10-4-3_prior"], stages, traj, secs, oracle["acc"], ocat)
    out["hybrid16-8-3"] = _sh_arm(hybrid_order[:16], SH_DEFS["hybrid16-8-3"], stages, traj, secs, oracle["acc"], ocat)
    p6 = prior_order[:6]
    p6 += [k for k in sur_order if k not in p6][:10]
    out["prior6+sur10-8-3"] = _sh_arm(p6, SH_DEFS["prior6+sur10-8-3"], stages, traj, secs, oracle["acc"], ocat)
    out["SH16-8-3_surrogate"] = _sh_arm(sur_order[:16], SH_DEFS["SH16-8-3_surrogate"], stages, traj, secs, oracle["acc"], ocat)
    n = len(keys)
    out["SH100-8-3"] = _sh_arm(sorted(keys), (n, min(8, n), min(3, n)), stages, traj, secs, oracle["acc"], ocat)
    out["random1"] = {"acc": float(np.mean(list(truth.values()))), "regret_pool": oracle["acc"] - float(np.mean(list(truth.values()))),
                      "cost_full": 1.0, "cost_iters": bf}
    return out


# ---------------------------------------------------------------- 基线（一个靶）
def baselines_for(keys: Sequence[str], cls_of: Mapping, traj: Mapping, budgets: Sequence[int], cfg: Mapping, *,
                  space_fields: Sequence[str], space_choices: Mapping, seed_offset: int = 0, beat: Optional[Mapping[str, float]] = None) -> dict:
    """RS-n（解析）/ rand-SH（抽样）/ Hyperband 变体 / TPE-n；regret 相对本子池 oracle；beat = {臂名: acc} 给出 p_beat。"""
    stages = [int(b) for b in budgets]
    bf = stages[-1]
    keys = [k for k in keys if all(b in traj.get(k, {}) for b in stages)]
    truth = [float(traj[k][bf]) for k in keys]
    oracle = max(truth)
    nd = int(cfg["sim"]["n_draws"])
    sb = cfg["sim"]["seed_base"]
    main_beat = (beat or {}).get(MAIN_ARM)
    out: dict[str, dict] = {}
    for n in cfg["arms"]["rs"]:
        if n <= len(keys):
            r = random_search_expect(truth, int(n), beat=main_beat)
            out[f"RS-{n}"] = {**r, "regret_mean": oracle - r["mean"], "cost_full": float(n), "cost_iters": int(n) * bf, "kind": "analytic"}
    for ks in cfg["arms"]["rand_sh"]:
        ks = tuple(int(x) for x in ks)
        if ks[0] <= len(keys):
            arr = rand_sh_draws(keys, traj, ks, stages, nd, int(sb["rand_sh"]) + seed_offset)
            s = summarize_draws(arr, beat=main_beat)
            name = "rand-SH" + "-".join(str(x) for x in ks)
            out[name] = {**s, "regret_mean": oracle - s["mean"], "cost_full": sh_cost_full(ks, stages), "cost_iters": sh_cost_iters(ks, stages), "kind": "draws"}
    eta = int(cfg["arms"]["hb"]["eta"])
    for name, br in cfg["arms"]["hb"]["variants"].items():
        br = [int(x) for x in br]
        if hb_n_configs(br, eta=eta) <= len(keys):
            arr, _p = hyperband_draws(keys, traj, stages, br, n_draws=nd, seed=int(sb["hb"]) + seed_offset, eta=eta)
            s = summarize_draws(arr, beat=main_beat)
            ci = hb_cost_iters(stages, br, eta)
            out[name] = {**s, "regret_mean": oracle - s["mean"], "cost_full": ci / bf, "cost_iters": ci, "kind": "draws", "brackets": br}
    t = cfg["arms"]["tpe"]
    for n in t["n"]:
        if int(n) <= len(keys):
            arr, ci = tpe_draws(keys, cls_of, traj, stages, int(n), fields=space_fields, choices=space_choices,
                                n_startup=int(t["n_startup"]), n_draws=nd, seed=int(sb["tpe"]) + seed_offset)
            s = summarize_draws(arr, beat=main_beat)
            out[f"TPE-{n}"] = {**s, "regret_mean": oracle - s["mean"], "cost_full": float(n), "cost_iters": ci, "kind": "draws"}
    out["_oracle"] = {"acc": oracle, "n": len(keys)}
    return out


# ---------------------------------------------------------------- 模拟 / 门 / 预测
def target_traj(pool_t: Mapping, rows: Sequence[dict]) -> tuple[dict, dict, dict]:
    traj, secs = _g39.trajectory_from_rows([r for r in rows if r.get("error") is None])
    cls_of = {e["cls_key"]: e["cls"] for e in pool_t["entries"]}
    return traj, secs, cls_of


def simulate_target(cfg: Mapping, d: str, pool_t: Mapping, rows: Sequence[dict], *, with_baselines: bool = True) -> dict:
    stages = [int(b) for b in pool_t["budgets"]]
    traj, secs, cls_of = target_traj(pool_t, rows)
    complete = [k for k in traj if all(b in traj[k] for b in stages)]
    ours = our_arms(pool_t, traj, secs, stages)
    res = {"budgets": stages, "n_pool": len(complete), "n_expected": len([e for e in pool_t["entries"] if not e["error"]]),
           "missing": sorted(set(e["cls_key"] for e in pool_t["entries"] if not e["error"]) - set(complete)),
           "ours": ours}
    if with_baselines:
        space = CLSSpace.from_yaml()
        fields = list(space.field_names)
        choices = {f: list(space.choices(f)) for f in fields}
        beat = {MAIN_ARM: ours[MAIN_ARM]["acc"]} if ours.get(MAIN_ARM) else None
        seed_off = int(cfg["targets"][d]["order"])
        res["baselines"] = baselines_for(complete, cls_of, traj, stages, cfg, space_fields=fields, space_choices=choices,
                                         seed_offset=seed_off, beat=beat)
        sub = {}
        for name, ks in (("random", pool_t.get("random_keys", [])), ("curated", pool_t.get("curated_keys", []))):
            ks = [k for k in ks if k in complete]
            if len(ks) >= 12:
                sub[name] = baselines_for(ks, cls_of, traj, stages, cfg, space_fields=fields, space_choices=choices,
                                          seed_offset=seed_off + 100, beat=None)
        res["subpools"] = sub
        res["curve"] = cost_curve(res)
        pool_final = {k: float(traj[k][stages[-1]]) for k in complete}
        rank_all = sorted(pool_final, key=lambda k: (-pool_final[k], k))
        rk = [rank_all.index(k) for k in pool_t.get("random_keys", []) if k in pool_final]
        res["random_stats"] = {"n": len(rk), "frac_bottom_half": (float(np.mean([r >= len(rank_all) / 2 for r in rk])) if rk else None),
                               "random_oracle": ours["oracle_random"]["acc"], "curated_oracle": ours["oracle_curated"]["acc"],
                               "random_lifts_oracle": (ours["oracle_random"]["acc"] is not None and ours["oracle_curated"]["acc"] is not None
                                                       and ours["oracle_random"]["acc"] >= ours["oracle_curated"]["acc"] + 0.01)}
    return res


def cost_curve(res: Mapping) -> dict:
    """每靶 regret 对成本（全跑当量）：我们的 5 点 + 各基线点。"""
    pts = {"ours": [], "baselines": []}
    for name in OURS:
        a = res["ours"].get(name)
        if a:
            pts["ours"].append({"arm": name, "cost_full": a["cost_full"], "regret": a["regret_pool"]})
    for name, b in res.get("baselines", {}).items():
        if name.startswith("_"):
            continue
        pts["baselines"].append({"arm": name, "cost_full": b["cost_full"], "regret": b["regret_mean"], "q25": b.get("q25"), "q75": b.get("q75")})
    return pts


def run_gates_p1(per: Mapping[str, dict], cfg: Mapping) -> dict:
    g = cfg["gates"]
    ds = list(per)

    def reg_o(d, arm):
        return float(per[d]["ours"][arm]["regret_pool"])

    def reg_b(d, name):
        return float(per[d]["baselines"][name]["regret_mean"])

    e1 = g["E1"]
    diffs = {d: reg_b(d, e1["vs"]) - reg_o(d, e1["arm"]) for d in ds}
    n_better = int(sum(v > 0 for v in diffs.values()))
    mean_diff = float(np.mean(list(diffs.values())))
    E1 = {"arm": e1["arm"], "vs": e1["vs"], "n_better": n_better, "n": len(ds), "mean_diff": mean_diff, "diffs": diffs,
          "pass": bool(n_better >= int(e1["min_targets"]) and mean_diff >= float(e1["min_mean_diff"]))}
    for a, b in e1.get("also_report", []):
        dd = {d: reg_b(d, b) - reg_o(d, a) for d in ds if b in per[d]["baselines"] and a in per[d]["ours"]}
        E1.setdefault("also", {})[f"{a} vs {b}"] = {"n_better": int(sum(v > 0 for v in dd.values())), "mean_diff": (float(np.mean(list(dd.values()))) if dd else None)}
    e2 = g["E2"]
    ours_mean = float(np.mean([reg_o(d, e2["arm"]) for d in ds]))
    base_means = {b: float(np.mean([reg_b(d, b) for d in ds])) for b in e2["vs"]}
    mb = min(base_means, key=lambda b: (base_means[b], b))
    E2 = {"arm": e2["arm"], "ours_mean": ours_mean, "baseline_means": base_means, "min_baseline": mb, "pass": bool(ours_mean <= base_means[mb])}
    e3 = g["E3"]
    ours3 = float(np.mean([reg_o(d, e3["arm"]) for d in ds]))
    base3 = {b: float(np.mean([reg_b(d, b) for d in ds])) for b in e3["vs"]}
    mb3 = min(base3, key=lambda b: (base3[b], b))
    n_ge_hb = int(sum(reg_o(d, e3["arm"]) <= reg_b(d, e3["hb_arm"]) for d in ds))
    E3 = {"arm": e3["arm"], "ours_mean": ours3, "baseline_means": base3, "min_baseline": mb3, "n_ge_hb": n_ge_hb,
          "pass": bool(ours3 <= base3[mb3] and n_ge_hb >= int(e3["hb_min_targets"]))}
    return {"E1": E1, "E2": E2, "E3": E3}


def verdict_code_p1(g: Mapping[str, dict]) -> str:
    if not g["E1"]["pass"]:
        return "P1-DEAD"
    if not g["E2"]["pass"]:
        return "P1-LIST"
    return "P1-WIN" if g["E3"]["pass"] else "P1-TIER1"


def predictions_p1(per: Mapping[str, dict], gates: Mapping, cfg: Mapping, *, ucr: Optional[Mapping] = None,
                   finals: Optional[Mapping] = None) -> dict:
    ds = list(per)
    P = cfg["predictions"]
    out = {"P1": {"claim": P["P1"]["claim"], "prior": P["P1"]["prior"], "hit": bool(gates["E1"]["pass"])},
           "P2": {"claim": P["P2"]["claim"], "prior": P["P2"]["prior"], "hit": bool(gates["E2"]["pass"])},
           "P3": {"claim": P["P3"]["claim"], "prior": P["P3"]["prior"], "hit": bool(gates["E3"]["pass"])}}
    lifts = [d for d in ds if per[d].get("random_stats", {}).get("random_lifts_oracle")]
    out["P4"] = {"claim": P["P4"]["claim"], "prior": P["P4"]["prior"], "lifts": lifts, "hit": bool(len(lifts) <= 1)}
    fb = {d: per[d].get("random_stats", {}).get("frac_bottom_half") for d in ds}
    out["P5"] = {"claim": P["P5"]["claim"], "prior": P["P5"]["prior"], "frac_bottom_half": fb,
                 "hit": bool(all(v is not None and v >= 0.5 for v in fb.values()))}
    d68 = [per[d]["baselines"]["TPE-8"]["regret_mean"] - per[d]["baselines"]["RS-8"]["regret_mean"] for d in ds
           if "TPE-8" in per[d]["baselines"] and "RS-8" in per[d]["baselines"]]
    out["P6"] = {"claim": P["P6"]["claim"], "prior": P["P6"]["prior"], "mean_diff": (float(np.mean(d68)) if d68 else None),
                 "hit": bool(d68 and abs(float(np.mean(d68))) < 0.01)}
    hb = float(np.mean([per[d]["baselines"]["HB-7"]["regret_mean"] for d in ds]))
    rsh = float(np.mean([per[d]["baselines"]["rand-SH10-4-3"]["regret_mean"] for d in ds]))
    ours = float(np.mean([per[d]["ours"][MAIN_ARM]["regret_pool"] for d in ds]))
    out["P7"] = {"claim": P["P7"]["claim"], "prior": P["P7"]["prior"], "hb": hb, "rand_sh10": rsh, "ours": ours,
                 "hit": [bool(hb < rsh), bool(hb < ours)]}
    if finals:
        drops = [d for d in ds if finals.get(d, {}).get("oracle_3seed") is not None and finals[d].get("oracle_seed0") is not None
                 and finals[d]["oracle_seed0"] - finals[d]["oracle_3seed"] >= 0.02]
        out["P8"] = {"claim": P["P8"]["claim"], "prior": P["P8"]["prior"], "drops": drops, "hit": bool(len(drops) >= 3)}
    else:
        out["P8"] = {"claim": P["P8"]["claim"], "prior": P["P8"]["prior"], "hit": None, "note": "finals 未跑"}
    if ucr:
        out["P9"] = {"claim": P["P9"]["claim"], "prior": P["P9"]["prior"], **{k: ucr.get(k) for k in ("prior_vs_rand", "prior_vs_indomain")},
                     "hit": [bool(ucr.get("prior_vs_rand", 0) >= 0.01), bool(abs(ucr.get("prior_vs_indomain", 1)) < 0.005)]}
    else:
        out["P9"] = {"claim": P["P9"]["claim"], "prior": P["P9"]["prior"], "hit": None, "note": "--ucr 未跑"}
    hw = per.get("Handwriting", {}).get("ours", {}).get("hybrid16-8-3", {})
    out["P10"] = {"claim": P["P10"]["claim"], "prior": P["P10"]["prior"], "pick": hw.get("pick"), "acc": hw.get("acc"),
                  "hit": bool(hw.get("pick", "").startswith("c92058f2"))}
    return out


# ---------------------------------------------------------------- C0
def c0_check(cfg: Mapping, outdir: Path = OUT) -> dict:
    """SH6-3-3_prior 与 WP3 analysis.json 逐位；SH10-4-3_prior 与 WP6 s1_sweep / WP8 stage0 逐位（都在目录 21 子池上）。"""
    wp3_dir = ROOT / "results" / "mx4_deploy"
    if not ((wp3_dir / "analysis.json").exists() and (wp3_dir / "runs.jsonl").exists() and (wp3_dir / "candidates.json").exists()):
        return {"_pass": None, "note": "缺 WP3 产物（analysis / runs / candidates），C0 未验"}
    wp3 = json.loads((wp3_dir / "analysis.json").read_text(encoding="utf-8"))["per_target"]
    rows = read_jsonl(wp3_dir / "runs.jsonl")
    cands = json.loads((wp3_dir / "candidates.json").read_text(encoding="utf-8"))
    stage0_p = ROOT / "results" / "mx4_bo" / "stage0.json"
    stage0 = json.loads(stage0_p.read_text(encoding="utf-8")) if stage0_p.exists() else {}
    out, ok = {}, True
    for d in run_targets(cfg):
        rd = [r for r in rows if r["dataset"] == d and r.get("stage") == "s1_catalog" and int(r.get("seed", 0)) == 0 and r.get("error") is None]
        traj, secs = _g39.trajectory_from_rows(rd)
        stages = budgets_for(cfg, d)
        keys = sorted(k for k in traj if all(b in traj[k] for b in stages))
        c = cands[d]
        po = [k for k in c["prior_order"] if k in traj] + sorted(k for k in keys if k not in set(c["prior_order"]))
        pool_d = {"prior_order": po, "hybrid_order": po, "surrogate_order": po, "catalog_keys": keys, "curated_keys": keys, "random_keys": []}
        a = our_arms(pool_d, traj, secs, stages)
        ref6 = wp3[d]["arms"]["SH6-3-3_prior"], wp3[d]["picks"]["SH6-3-3_prior"]
        m6 = bool(a["SH6-3-3_prior"]["acc"] == ref6[0] and a["SH6-3-3_prior"]["pick"] == ref6[1])
        m10 = None
        if d in stage0 and "policies" in stage0[d] and "SH10-4-3_prior" in stage0[d]["policies"]:
            p10 = stage0[d]["policies"]["SH10-4-3_prior"]
            m10 = bool(a["SH10-4-3_prior"]["acc"] == p10["acc"] and a["SH10-4-3_prior"]["pick"] == p10["pick"])
        ok = ok and m6 and (m10 is not False)
        out[d] = {"SH6-3-3_prior": {"acc": a["SH6-3-3_prior"]["acc"], "pick": a["SH6-3-3_prior"]["pick"], "match_wp3": m6},
                  "SH10-4-3_prior": {"acc": a["SH10-4-3_prior"]["acc"], "pick": a["SH10-4-3_prior"]["pick"], "match_wp8_stage0": m10}}
    out["_pass"] = bool(ok)
    return out


# ---------------------------------------------------------------- finals
def finals_reuse(cells: Sequence[dict], reuse_rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """(dataset, cls_key, seed) 已有非 error、掩码关的终评行 → 复用，不重跑。"""
    have = {}
    for r in reuse_rows:
        if r.get("error") is not None or r.get("acc") is None:
            continue
        if (r.get("options") or {}).get("input_mask", "none") != "none":
            continue
        if int(r.get("seed", 0)) not in FINAL_SEEDS:
            continue
        have.setdefault((r["dataset"], r["cls_key"], int(r["seed"])), r)
    todo = [c for c in cells if (c["dataset"], c["cls_key"], int(c["seed"])) not in have]
    used = {k: v for k, v in have.items() if k in {(c["dataset"], c["cls_key"], int(c["seed"])) for c in cells}}
    return todo, used


def cmd_finals(cfg: Mapping, args) -> None:
    outdir = Path(args.outdir)
    pool = read_pool(outdir)
    p = outdir / "picks.json"
    if not p.exists():
        raise SystemExit(f"缺 {p}：先跑 --analyze")
    picks = json.loads(p.read_text(encoding="utf-8"))
    picks = {d: v for d, v in picks.items() if (not args.datasets or d in args.datasets) and v}
    key2cls = {e["cls_key"]: e["cls"] for t in pool["targets"].values() for e in t["entries"]}
    seeds = tuple(int(s) for s in (args.seeds or cfg["finals"]["seeds"]))
    cells = _g39.finals_worklist(_protos(), sizes_from_cfg(cfg, list(picks)), picks, key2cls, seeds=seeds, protocol=cfg["protocol"])
    reuse = []
    for rel in cfg["finals"]["reuse"]:
        pth = ROOT / rel
        if pth.exists():
            reuse += [r for r in read_jsonl(pth) if r.get("stage") in ("s2_finals", "bo_finals")]
    if (outdir / "runs.jsonl").exists():
        reuse += [r for r in read_jsonl(outdir / "runs.jsonl") if r.get("stage") == "s2_finals"]
    todo, used = finals_reuse(cells, reuse)
    (outdir / "finals_reuse.json").write_text(json.dumps({"reused": [list(k) for k in used], "todo": len(todo), "planned": len(cells)},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[finals] 计划 {len(cells)} 格，复用三源 {len(used)}，待跑 {len(todo)}", flush=True)
    if sys.platform.startswith("win"):
        _g43.install_windows_eval_timeout()
    _g39.execute(todo, args, "S3")


def three_seed_table(cfg: Mapping, outdir: Path, pool: Mapping, per: Mapping) -> dict:
    """冠军与池前 3 的 3 种子均值；oracle_3seed = 池前 3 的 3 种子均值最大者。"""
    reuse = []
    for rel in cfg["finals"]["reuse"]:
        pth = ROOT / rel
        if pth.exists():
            reuse += [r for r in read_jsonl(pth) if r.get("stage") in ("s2_finals", "bo_finals")]
    if (outdir / "runs.jsonl").exists():
        reuse += [r for r in read_jsonl(outdir / "runs.jsonl") if r.get("stage") == "s2_finals"]
    reuse = [r for r in reuse if r.get("error") is None and (r.get("options") or {}).get("input_mask", "none") == "none"]
    out = {}
    for d, res in per.items():
        stages = res["budgets"]
        bf = stages[-1]
        wp3, wp8, new = read_sources(cfg, outdir)
        merged = load_pool_rows(d, wp3, wp8, new)
        traj, _s = _g39.trajectory_from_rows([r for r in merged["rows"] if _complete(r)])
        truth = {k: float(traj[k][bf]) for k in traj if bf in traj[k]}
        top3 = sorted(truth, key=lambda k: (-truth[k], k))[:int(cfg["finals"]["pool_top"])]
        rowd = {}
        for k in set(top3) | {res["ours"][a]["pick"] for a in cfg["finals"]["champion_arms"] if res["ours"].get(a)}:
            rowd[k] = _g43.three_seed(d, k, truth[k], reuse)
        o3 = [(rowd[k], k) for k in top3 if rowd.get(k) is not None]
        out[d] = {"three_seed": rowd, "pool_top3": top3, "oracle_seed0": (truth[top3[0]] if top3 else None),
                  "oracle_3seed": (max(o3)[0] if o3 else None), "oracle_3seed_key": (max(o3)[1] if o3 else None),
                  "champions": {a: {"pick": res["ours"][a]["pick"], "seed0": res["ours"][a]["acc"], "mean3": rowd.get(res["ours"][a]["pick"])}
                                for a in cfg["finals"]["champion_arms"] if res["ours"].get(a)}}
    return out


# ---------------------------------------------------------------- UCR 84
def ucr_ceiling_datasets(rows: Sequence[dict], *, budget: int, ceiling: float) -> list[str]:
    mx: dict[str, float] = defaultdict(float)
    for r in rows:
        if r.get("origin") == ORIGIN_SHARED and int(r["iters"]) == int(budget) and r.get("r_prime") is not None:
            mx[r["dataset"]] = max(mx[r["dataset"]], float(r["r_prime"]))
    return sorted(d for d, v in mx.items() if v >= float(ceiling))


def cmd_ucr(cfg: Mapping, outdir: Path) -> dict:
    u = cfg["ucr"]
    rows = [r for r in read_jsonl(ROOT / u["cache"]) if r.get("error") is None and r.get("r_prime") is not None]
    budgets = [int(b) for b in u["budgets"]]
    bf = budgets[-1]
    # UEA 全 8 源先验（UCR 不在源池）与代理
    src = [r for r in read_jsonl(ROOT / cfg["pool"]["mx3_cache"]) if r.get("error") is None and r["origin"] != "ggs"]
    src_z = _g36.zscore_within(src)
    prior = _g36.prior_scores([r for r in src_z if r["origin"] == ORIGIN_SHARED])
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    model = FieldSurrogate(cfg["pool"]["surrogate"]["kind"], seed=int(cfg["pool"]["surrogate"]["nominee_seed"])).fit(
        _g36.features(src_z, enc, max(int(r["iters"]) for r in src_z)), np.array([float(r["z"]) for r in src_z]))
    fields = list(space.field_names)
    choices = {f: list(space.choices(f)) for f in fields}
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    ceiling = set(ucr_ceiling_datasets(rows, budget=bf, ceiling=float(u["ceiling"])))
    # UCR 域内 LODO 先验（其余 83 个数据集 shared 行的 z 均值）
    ucr_z = _g36.zscore_within(rows)
    shared_z = [r for r in ucr_z if r["origin"] == ORIGIN_SHARED]
    per = {}
    nd = int(cfg["sim"]["n_draws"])
    for i, (d, rd) in enumerate(sorted(by_ds.items())):
        traj, secs = _g36.trajectories(rd)
        keys = [k for k in traj if all(b in traj[k] for b in budgets)]
        cls_of = {r["cls_key"]: r["cls"] for r in rd}
        pz = dict(zip(keys, surrogate_predict(model, enc, [cls_of[k] for k in keys])))
        pk = [k for k in keys if k in prior]
        po, ho, so = pool_orders(keys, prior, pz, prior_keys=pk)
        indom = _g36.prior_scores([r for r in shared_z if r["dataset"] != d])
        po_in = _g36.rank_by(indom, [k for k in keys if k in indom]) + sorted(k for k in keys if k not in indom)
        pool_d = {"prior_order": po, "hybrid_order": ho, "surrogate_order": so, "catalog_keys": pk, "curated_keys": pk, "random_keys": [k for k in keys if k not in pk]}
        ours = our_arms(pool_d, traj, secs, budgets)
        truth = {k: float(traj[k][bf]) for k in keys}
        oracle = max(truth.values())
        indom_arms = {"SH6-3-3_indomain": _sh_arm(po_in[:6], (6, 3, 3), budgets, traj, secs, oracle, oracle),
                      "SH10-4-3_indomain": _sh_arm(po_in[:10], (10, 4, 3), budgets, traj, secs, oracle, oracle)}
        b = {}
        vals = list(truth.values())
        b["RS-4"] = {**random_search_expect(vals, int(u["arms"]["rs"])), "cost_iters": int(u["arms"]["rs"]) * bf}
        b["RS-4"]["regret_mean"] = oracle - b["RS-4"]["mean"]
        for name, ks in (("rand-SH6-3-3", (6, 3, 3)), ("rand-SH10-4-3", (10, 4, 3))):
            arr = rand_sh_draws(keys, traj, ks, budgets, nd, int(cfg["sim"]["seed_base"]["rand_sh"]) + i)
            s = summarize_draws(arr)
            b[name] = {**s, "regret_mean": oracle - s["mean"], "cost_iters": sh_cost_iters(ks, budgets)}
        br = [int(x) for x in u["arms"]["hb"]]
        arr, _p = hyperband_draws(keys, traj, budgets, br, n_draws=nd, seed=int(cfg["sim"]["seed_base"]["hb"]) + i)
        s = summarize_draws(arr)
        b["HB"] = {**s, "regret_mean": oracle - s["mean"], "cost_iters": hb_cost_iters(budgets, br)}
        arr, ci = tpe_draws(keys, cls_of, traj, budgets, int(u["arms"]["tpe"]), fields=fields, choices=choices,
                            n_startup=int(cfg["arms"]["tpe"]["n_startup"]), n_draws=nd, seed=int(cfg["sim"]["seed_base"]["tpe"]) + i)
        s = summarize_draws(arr)
        b["TPE-4"] = {**s, "regret_mean": oracle - s["mean"], "cost_iters": ci}
        per[d] = {"n": len(keys), "oracle": oracle, "ceiling": d in ceiling, "ours": {k: ours[k] for k in OURS if ours.get(k)},
                  "indomain": indom_arms, "baselines": b}
    def mean_over(sel, get):
        v = [get(per[d]) for d in sel]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None
    all_ds, off = sorted(per), sorted(d for d in per if not per[d]["ceiling"])
    summ = {}
    for label, sel in (("all", all_ds), ("off_ceiling", off)):
        summ[label] = {"n": len(sel),
                       **{k: mean_over(sel, lambda p, k=k: p["ours"][k]["regret_pool"] if p["ours"].get(k) else None) for k in OURS},
                       **{k: mean_over(sel, lambda p, k=k: p["indomain"][k]["regret_pool"] if p["indomain"].get(k) else None) for k in ("SH6-3-3_indomain", "SH10-4-3_indomain")},
                       **{k: mean_over(sel, lambda p, k=k: p["baselines"][k]["regret_mean"]) for k in ("RS-4", "rand-SH6-3-3", "rand-SH10-4-3", "HB", "TPE-4")}}
    s_all = summ["all"]
    reading = {"prior_vs_rand": (s_all["rand-SH6-3-3"] - s_all["SH6-3-3_prior"]) if s_all.get("SH6-3-3_prior") is not None else None,
               "prior_vs_indomain": (s_all["SH6-3-3_prior"] - s_all["SH6-3-3_indomain"]) if s_all.get("SH6-3-3_indomain") is not None else None}
    out = {"n_datasets": len(per), "ceiling_datasets": sorted(ceiling), "budgets": budgets, "summary": summ, "reading": reading, "per_dataset": per}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ucr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["## UCR 84（切片口径 {60,120,290}；UEA 全 8 源先验 + 代理；regret 对 20 池 oracle @290）", "",
         "| 集合 | n | " + " | ".join(OURS) + " | SH6-3-3_indomain | SH10-4-3_indomain | RS-4 | rand-SH6-3-3 | rand-SH10-4-3 | HB(1930) | TPE-4 |",
         "|---|---:|" + "---:|" * (len(OURS) + 7)]
    for label in ("all", "off_ceiling"):
        s = summ[label]
        L.append(f"| {label} | {s['n']} | " + " | ".join(_f(s.get(k)) for k in list(OURS) + ["SH6-3-3_indomain", "SH10-4-3_indomain", "RS-4", "rand-SH6-3-3", "rand-SH10-4-3", "HB", "TPE-4"]) + " |")
    L += ["", f"读数：prior_vs_rand = {_f(reading['prior_vs_rand'])}（SH6-3-3）；prior_vs_indomain = {_f(reading['prior_vs_indomain'])}"]
    (outdir / "ucr.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ucr] {len(per)} 数据集（贴顶 {len(ceiling)}）；SH6-3-3_prior regret {_f(s_all.get('SH6-3-3_prior'))} vs rand {_f(s_all.get('rand-SH6-3-3'))} "
          f"vs indomain {_f(s_all.get('SH6-3-3_indomain'))}；HB {_f(s_all.get('HB'))}", flush=True)
    return out


# ---------------------------------------------------------------- 消融汇总 / 成本表
def ablation_rows(sources: Sequence[Mapping]) -> list[dict]:
    rows = []
    for s in sources:
        p = Path(s["path"])
        if not p.is_absolute():
            p = ROOT / p
        base = {"block": s["block"], "name": s.get("name"), "source_path": str(s["path"]), "b_final": s.get("b_final")}
        if not p.exists():
            rows.append({**base, "arm": None, "metric": s.get("key"), "value": None, "n": None, "note": "missing"})
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        key = s.get("key")
        node = a
        for part in (key.split(".") if key else []):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        n = a.get(s["n_key"]) if s.get("n_key") else None
        if isinstance(node, dict) and node:
            for arm, v in node.items():
                if isinstance(v, (int, float)):
                    rows.append({**base, "arm": arm, "metric": key, "value": float(v), "n": n})
        else:
            rows.append({**base, "arm": None, "metric": key, "value": None, "n": n, "note": "key not found"})
    return rows


def cost_table(cfg: Mapping, per: Mapping, outdir: Path) -> str:
    ac = cfg["cost"]["autocl"]
    per_full = ac["full_pretrain_variant_h"] / ac["controller_iters"]
    autocl_fpe = ac["total_h"] / per_full
    L = ["## 成本表（全跑当量 FPE；秒 = 缓存墙钟按来源硬件标签）", "",
         "| 臂 | FPE | " + " | ".join(per) + " |", "|---|---:|" + "---:|" * len(per)]
    rows = [("GGS / TS2Vec 默认", "1.0"), ("prior1", "1.0"), ("prior3", "3.0"), ("SH6-3-3_prior", "3.75"), ("SH10-4-3_prior", "5.0"),
            ("hybrid16-8-3", "7.5"), ("HB-5", "5.0"), ("HB-7", "7.0"), ("RS-5 / TPE-5", "5.0"), ("RS-8 / TPE-8", "8.0")]
    for name, fpe in rows:
        cells = []
        for d in per:
            a = per[d]["ours"].get(name) or per[d].get("baselines", {}).get(name)
            cells.append(_f(a.get("cost_seconds") if a and a.get("cost_seconds") is not None else None, 0) if a else "—")
        L.append(f"| {name} | {fpe} | " + " | ".join(cells) + " |")
    L += ["", f"AutoCL（Table 6，HAR）：总时 {ac['total_h']} h；'Full Pre-train f_E' 变体 {ac['full_pretrain_variant_h']} h = {ac['controller_iters']} 次控制器迭代各一次完整预训练 → "
          f"单次完整预训练 ≤ {per_full:.3f} h → AutoCL ≥ {autocl_fpe:.0f} 全跑当量（下界：分母含验证与阶段 2）。",
          f"离线先验一次性成本：{cfg['cost']['prior_offline']['trajectories']} 条源池轨迹 ≈ {cfg['cost']['prior_offline']['gpu_h']} GPU-h（{cfg['cost']['prior_offline']['device']}），跨全部未来目标摊销。"]
    md = "\n".join(L) + "\n"
    (outdir / "cost.md").write_text(md, encoding="utf-8")
    return md


# ---------------------------------------------------------------- 表 / 子命令
def _f(x, nd: int = 3) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}"


def markdown_tables(sim: Mapping, analysis: Optional[Mapping] = None) -> str:
    per = sim["per_target"]
    L = ["## 我们的臂（seed 0，acc@100%；regret 对池 oracle）", "",
         "| 靶 | n 池 | oracle 池/目录/策划/随机 | " + " | ".join(OURS) + " |", "|---|---:|---|" + "---:|" * len(OURS)]
    for d, r in per.items():
        o = r["ours"]
        L.append(f"| {d} | {r['n_pool']} | {_f(o['oracle_pool']['acc'])} / {_f(o['oracle_catalog']['acc'])} / {_f(o['oracle_curated']['acc'])} / {_f(o['oracle_random']['acc'])} | "
                 + " | ".join((f"{_f(o[k]['acc'])} ({_f(o[k]['regret_pool'])})" if o.get(k) else "—") for k in OURS) + " |")
    if all("baselines" in r for r in per.values()):
        names = [n for n in next(iter(per.values()))["baselines"] if not n.startswith("_")]
        L += ["", "## 基线（期望 acc；括号 regret 均值；成本 FPE）", "", "| 靶 | " + " | ".join(f"{n} ({next(iter(per.values()))['baselines'][n]['cost_full']:.2f})" for n in names) + " |",
              "|---|" + "---:|" * len(names)]
        for d, r in per.items():
            L.append(f"| {d} | " + " | ".join((f"{_f(r['baselines'][n]['mean'])} ({_f(r['baselines'][n]['regret_mean'])})" if n in r["baselines"] else "—") for n in names) + " |")
        L += ["", "## 六靶均值 regret 对成本", "", "| 臂 | FPE | 均值 regret |", "|---|---:|---:|"]
        for k in OURS:
            vals = [r["ours"][k]["regret_pool"] for r in per.values() if r["ours"].get(k)]
            if vals:
                L.append(f"| {k} | {next(r['ours'][k]['cost_full'] for r in per.values() if r['ours'].get(k)):.2f} | {np.mean(vals):.4f} |")
        for n in names:
            vals = [r["baselines"][n]["regret_mean"] for r in per.values() if n in r["baselines"]]
            if vals:
                L.append(f"| {n} | {next(iter(per.values()))['baselines'][n]['cost_full']:.2f} | {np.mean(vals):.4f} |")
    if analysis and analysis.get("gates"):
        g = analysis["gates"]
        L += ["", "## 门", "",
              f"- E1：{g['E1']['arm']} vs {g['E1']['vs']}：赢 {g['E1']['n_better']}/{g['E1']['n']}，均值差 {g['E1']['mean_diff']:+.4f} → {'✓' if g['E1']['pass'] else '✗'}",
              f"- E2：{g['E2']['arm']} 均值 regret {g['E2']['ours_mean']:.4f} vs min 基线 {g['E2']['min_baseline']} {g['E2']['baseline_means'][g['E2']['min_baseline']]:.4f} → {'✓' if g['E2']['pass'] else '✗'}",
              f"- E3：{g['E3']['arm']} 均值 regret {g['E3']['ours_mean']:.4f} vs min 基线 {g['E3']['min_baseline']} {g['E3']['baseline_means'][g['E3']['min_baseline']]:.4f}，vs HB ≥ {g['E3']['n_ge_hb']}/6 → {'✓' if g['E3']['pass'] else '✗'}",
              "", f"读数码：**{analysis['verdict']}**"]
    if analysis and analysis.get("predictions"):
        L += ["", "## 预测对账", "", "| # | 内容 | 先验 | 命中 |", "|---|---|---|---|"]
        for k, p in analysis["predictions"].items():
            L.append(f"| {k} | {p['claim']} | {p['prior']} | {p.get('hit')} |")
    return "\n".join(L) + "\n"


def cmd_simulate(cfg: Mapping, outdir: Path, datasets: Sequence[str], *, c0_only: bool = False) -> dict:
    pool = read_pool(outdir)
    wp3, wp8, new = read_sources(cfg, outdir)
    per = {}
    for d in datasets:
        if d not in pool["targets"]:
            continue
        merged = load_pool_rows(d, wp3, wp8, new)
        res = simulate_target(cfg, d, pool["targets"][d], merged["rows"], with_baselines=not c0_only)
        per[d] = res
        o = res["ours"]
        print(f"[simulate {d}] 池 {res['n_pool']}/{res['n_expected']} oracle {_f(o['oracle_pool']['acc'])}；"
              f"SH6 {_f(o['SH6-3-3_prior']['acc'])} SH10 {_f(o['SH10-4-3_prior']['acc'])} hybrid16 {_f(o['hybrid16-8-3']['acc'])}"
              + ("" if c0_only else f"；HB-7 {_f(res['baselines'].get('HB-7', {}).get('mean'))} rand-SH10 {_f(res['baselines'].get('rand-SH10-4-3', {}).get('mean'))}"), flush=True)
    c0 = c0_check(cfg, outdir)
    sim = {"config_md5": cfg_md5(), "c0": c0, "c0_only": c0_only, "per_target": per}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ("c0.json" if c0_only else "simulate.json")).write_text(json.dumps(sim, ensure_ascii=False, indent=1), encoding="utf-8")
    if not c0_only:
        (outdir / "tables.md").write_text(markdown_tables(sim), encoding="utf-8")
    print(f"[simulate] C0 {'PASS' if c0['_pass'] else 'FAIL'}；靶 {len(per)}", flush=True)
    return sim


def cmd_analyze(cfg: Mapping, outdir: Path, datasets: Sequence[str]) -> dict:
    sim_p = outdir / "simulate.json"
    if not sim_p.exists():
        raise SystemExit(f"缺 {sim_p}：先跑 --simulate")
    sim = json.loads(sim_p.read_text(encoding="utf-8"))
    per = {d: v for d, v in sim["per_target"].items() if d in set(datasets) and "baselines" in v}
    pool = read_pool(outdir)
    gates = run_gates_p1(per, cfg)
    finals = three_seed_table(cfg, outdir, pool, per)
    ucr_p = outdir / "ucr.json"
    ucr = json.loads(ucr_p.read_text(encoding="utf-8"))["reading"] if ucr_p.exists() else None
    preds = predictions_p1(per, gates, cfg, ucr=ucr, finals=finals)
    picks = {}
    for d, res in per.items():
        pk = {a: res["ours"][a]["pick"] for a in cfg["finals"]["champion_arms"] if res["ours"].get(a)}
        for i, k in enumerate(finals[d]["pool_top3"]):
            pk[f"pool_top{i + 1}"] = k
        picks[d] = pk
    analysis = {"protocol": "results/mx4_p1/PROTOCOL.md", "config_md5": cfg_md5(), "c0": sim.get("c0"), "n_targets": len(per),
                "gates": gates, "verdict": verdict_code_p1(gates), "predictions": preds, "finals": finals,
                "regret_three_columns": {d: {a: {"seed0_pool": res["ours"][a]["regret_pool"], "catalog": res["ours"][a]["regret_catalog"],
                                                 "oracle_3seed": ((finals[d]["oracle_3seed"] - (finals[d]["champions"].get(a, {}).get("mean3") or res["ours"][a]["acc"]))
                                                                  if finals[d]["oracle_3seed"] is not None else None)}
                                             for a in cfg["finals"]["champion_arms"] if res["ours"].get(a)} for d, res in per.items()}}
    (outdir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(sim, analysis), encoding="utf-8")
    print(f"verdict {analysis['verdict']}；E1 {gates['E1']['n_better']}/{gates['E1']['n']} Δ {gates['E1']['mean_diff']:+.4f}；"
          f"E2 {gates['E2']['ours_mean']:.4f} vs {gates['E2']['min_baseline']} {gates['E2']['baseline_means'][gates['E2']['min_baseline']]:.4f}；"
          f"E3 {gates['E3']['ours_mean']:.4f} vs {gates['E3']['min_baseline']} {gates['E3']['baseline_means'][gates['E3']['min_baseline']]:.4f}（HB ≥ {gates['E3']['n_ge_hb']}/6）", flush=True)
    return analysis


def cmd_dry_run(cfg: Mapping, outdir: Path, datasets: Sequence[str]) -> dict:
    pool = read_pool(outdir)
    est = estimate_hours(cfg, pool, [d for d in datasets if d in pool["targets"]])
    for d, v in est["per_dataset"].items():
        print(f"  {d:<24} {v['n_cells']:>3} 格  锚 {v['cell_s']:.1f} s（最大 {v['cell_max_s']:.1f}）→ {v['hours']:.2f} h（按最大 {v['hours_max']:.2f} h）")
    print(f"S1 合计 {est['n_cells']} 格；{est['total_h']:.2f} h；保守 ×{CONSERVATIVE} = {est['conservative_h']:.1f} h（{est['hardware']} 锚，含每格 {LAUNCH_S:.0f} s 启动）")
    return est


def cmd_smoke(cfg: Mapping, args, datasets: Sequence[str]) -> bool:
    outdir = Path(args.outdir) / "smoke"
    pool = read_pool(Path(args.outdir))
    p = _protos()[cfg["protocol"]]
    sizes = sizes_from_cfg(cfg, datasets)
    cells = []
    for d in datasets:
        ents = [e for e in pool["targets"][d]["entries"] if e["origin"] == "random_space"][:2]
        for e in ents:
            cell = _g35._cell("s1_pool", d, cfg["protocol"], p, sizes, arm_names=["random_space"], cls_key_=e["cls_key"],
                              cls=CLS(**e["cls"]), seed=0, iters=int(args.smoke_iters))
            cell["checkpoints"] = list(_g39.checkpoints_for(int(args.smoke_iters)))
            cells.append(cell)
    args.outdir = str(outdir)
    _g39.execute(cells, args, "smoke")
    rows = read_jsonl(outdir / "runs.jsonl")
    ok = all(len(r.get("checkpoint_results") or []) == 3 and r.get("error") is None for r in rows) and len(rows) >= len(cells)
    print(f"smoke {'OK' if ok else 'FAILED'}：{len(rows)} 行")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("build-pool", "dry-run", "smoke", "run", "simulate", "c0", "analyze", "finals", "ucr", "ablation", "cost"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--joblib-backend", default="threading", choices=("threading", "loky"))
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--smoke-iters", type=int, default=SMOKE_ITERS)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    outdir = Path(args.outdir)
    datasets = [d for d in run_targets(cfg) if not args.datasets or d in args.datasets]
    if args.build_pool:
        build_pool(cfg, outdir, datasets)
    if args.dry_run:
        cmd_dry_run(cfg, outdir, datasets)
    if args.smoke:
        cmd_smoke(cfg, args, datasets)
        return
    if args.run:
        cmd_run(cfg, args)
    if args.simulate:
        cmd_simulate(cfg, outdir, datasets, c0_only=bool(args.c0))
    if args.analyze:
        cmd_analyze(cfg, outdir, datasets)
    if args.finals:
        cmd_finals(cfg, args)
    if args.ucr:
        cmd_ucr(cfg, outdir)
    if args.ablation:
        rows = ablation_rows(cfg["ablation_sources"])
        (outdir / "ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[ablation] {len(rows)} 行 → ablation.json")
    if args.cost:
        sim_p = outdir / "simulate.json"
        per = json.loads(sim_p.read_text(encoding="utf-8"))["per_target"] if sim_p.exists() else {}
        cost_table(cfg, per, outdir)
        print("[cost] → cost.md")
    if not any((args.build_pool, args.dry_run, args.smoke, args.run, args.simulate, args.analyze, args.finals, args.ucr, args.ablation, args.cost)):
        ap.print_help()


if __name__ == "__main__":
    main()
