"""WP16 EV1 检验：源侧进化搜索验证——比较器当适应度、在信任域邻域（到 shared-10 + GGS 汉明 ≤ 3）里跑 AutoCTS++ Algorithm 2，提名真跑。协议 results/tclsc_ev/PROTOCOL.md。

子命令（按执行顺序）：
  --evolve --guides ev_et ev_blind [ev_es] [--random]   每切片（14 = 7 折集 × {_00, _x00}）进化 → proposals_<guide>.json；--random → proposals_rand_u.json（域内均匀 3 条）
  --rprime --guides ... [--random] / --verify           提名 + rand_u 轨迹（32 号执行体，canonical 种子；已有缓存行直接复用并标 reused_from）→ rprime_ev.jsonl
  --fit-folds                                           折共享 TS2Vec（只 fit 训练侧 7 集）+ 87 切片窗口特征 → folds/<折>/task_features（pod）
  --train-es                                            es_field_hard（SetTF + field 头）7 折 × seeds，NB1 训练标签（pod）
  --rank-es                                             NB1 考卷（12 切片 × 20 条）上 es_field_hard 排名 → rankings_es.json（pod）
  --analyze                                             门 G-V1/2/3、读数码、表
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.comparator.tclsc_engine import TCLSCEngine  # noqa: E402
from autocls.labeling import ORIGIN_RANDOM, cls_key, run_seed_for  # noqa: E402
from autocls.oom_guard import ON_OOM_SKIP  # noqa: E402
from autocls.search.evolution import evolutionary_search  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION, is_better  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_m33 = importlib.import_module("scripts.33_pq_comparator_loo")
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g34 = importlib.import_module("scripts.34_pq_search")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g62 = importlib.import_module("scripts.62_tclsc_s1e")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g73 = importlib.import_module("scripts.73_tclsc_lc")
_g74 = importlib.import_module("scripts.74_tclsc_slices")
_g75 = importlib.import_module("scripts.75_tclsc_k3")
_g76 = importlib.import_module("scripts.76_tclsc_nb")
OUT = ROOT / "results" / "tclsc_ev"
NB, K3, SOURCE, SX = _g76.OUT, _g75.OUT, pq.OUT, _g74.OUT
GUIDES = ("ev_et", "ev_es", "ev_blind")
E_GUIDES = ("ev_et", "ev_es")
GUIDE_ARM = {"ev_et": "ft60_hard", "ev_es": "es_field_hard", "ev_blind": "blind_diff_hard"}
ES_ARM, ES_TAG = "es_field_hard", "ES"
RAND_U, RAND_NB = "rand_u", "rand_nb"
SEEDS = (0, 1, 2)
FOLDS = tuple(pq.FOLD_TARGETS)
MAX_H = 3                       # 搜索域：到信任域的最小汉明 ≤ 3（19 个基础字段）
N_PROPOSE = 3                   # 提名 = 最终排名前 3 条非信任域配置（AutoCTS++ top-3）
N_RAND = 3                      # 均匀随机对照每切片 3 条
FIT_BUDGET = 290                # 适应度的 ctx 预算
EV_CFG = dict(ks=200, kp=20, generations=10, p_cross=0.8, p_mut=0.2)     # p_cross / p_mut = 论文 p1 / p2
FIXED_SEED = 7
G1_POS, G1_DELTA, G2_MIN, ALPHA = 0.50, 0.20, 0.010, 0.05
G4_OVER_MAX = 0.25
EXISTING_CACHES = (NB / "rprime_nb.jsonl", SOURCE / "rprime_cache.jsonl", SX / "rprime_cache.jsonl")


# ---------------------------------------------------------------- 搜索域
def trust_entries(space: CLSSpace) -> list[tuple[CLS, str]]:
    """信任域 = shared-10（探针目录前 10）+ GGS，所有切片相同。"""
    shared = pq.load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    ggs = pq.ggs_cls()
    return list(shared) + [(ggs, pq.cls_key(ggs))]


def min_hamming(cls: CLS, trust_cls: Sequence[CLS], fields: Sequence[str]) -> int:
    return min(_g76._hamming(cls, t, fields) for t in trust_cls)


def in_domain(space: CLSSpace, cls: CLS, trust_cls: Sequence[CLS], *, max_h: int = MAX_H) -> bool:
    return bool(space.validate(cls) and min_hamming(cls, trust_cls, space.field_names) <= max_h)


def local_mutator(space: CLSSpace) -> Callable:
    """一步局部变异：NB1 `mutate(h=1)`（比例档 ±1 / ±2、类别档任取、套约束）。"""
    return lambda c, rng: _g76.mutate(c, 1, rng, space)


def domain_sampler(space: CLSSpace, trust_cls: Sequence[CLS], rng: np.random.Generator, *, max_h: int = MAX_H) -> CLS:
    """域内随机：中心在信任域均匀取、h ∈ {1..max_h} 均匀取、NB1 mutate。"""
    ctr = trust_cls[int(rng.integers(len(trust_cls)))]
    h = int(rng.integers(1, max_h + 1))
    return _g76.mutate(ctr, h, rng, space)


def uniform_random(space: CLSSpace, slice_id: str, trust_cls: Sequence[CLS], *, taken: set[str], n: int = N_RAND, seed: int = FIXED_SEED) -> list[CLS]:
    """均匀随机对照（rand_u）：task_id 派生种子；与 taken 去重（就地扩充）。"""
    rng = np.random.default_rng([seed, run_seed_for(slice_id, "ev_rand")])
    out = []
    while len(out) < n:
        c = domain_sampler(space, trust_cls, rng)
        k = cls_key(c)
        if k in taken:
            continue
        taken.add(k)
        out.append(c)
    return out


def ev_slices() -> list[str]:
    """14 切片：7 个折数据集各 `_00` + `_x00`（固定规则）。"""
    return [s for d in FOLDS for s in (f"{d}_00", f"{d}_x00")]


# ---------------------------------------------------------------- 进化一个切片
def pick_proposals(ranking: Sequence[str], trust_keys: set[str], *, k: int = N_PROPOSE) -> list[dict]:
    """最终排名里前 k 条非信任域配置；above_trust = 位次在信任域最好一条之前。"""
    tr = [i for i, key in enumerate(ranking) if key in trust_keys]
    tb = min(tr) if tr else None
    out = []
    for i, key in enumerate(ranking):
        if key in trust_keys:
            continue
        out.append({"cls_key": key, "rank": i, "above_trust": bool(tb is None or i < tb)})
        if len(out) >= k:
            break
    return out


def evolve_slice(score_pairs: Callable, space: CLSSpace, encoder, trust: Sequence[tuple[CLS, str]], *, seed: int, cfg: Optional[Mapping] = None) -> dict:
    """信任域 + ks 域内随机 → Algorithm 2（域内变异 / 接受域）→ 整个最终种群排名 + 前 3 非信任域提名。"""
    c = dict(EV_CFG)
    c.update(cfg or {})
    tcls = [x for x, _k in trust]
    tkeys = {k for _x, k in trust}
    acc = lambda x: in_domain(space, x, tcls)
    pop = evolutionary_search(score_pairs, space, encoder, ks=int(c["ks"]), kp=int(c["kp"]), generations=int(c["generations"]),
                              p_cross=float(c["p_cross"]), p_mut=float(c["p_mut"]), top_k=int(c["kp"]), rng=np.random.default_rng([FIXED_SEED, int(seed)]),
                              sampler=lambda r: domain_sampler(space, tcls, r), mutator=local_mutator(space), accept=acc, seed_population=tcls)
    ranking = [x["cls_key"] for x in pop]
    by_key = {x["cls_key"]: x for x in pop}
    tr = [i for i, k in enumerate(ranking) if k in tkeys]
    tb = min(tr) if tr else None
    props = pick_proposals(ranking, tkeys, k=N_PROPOSE)
    for p in props:
        x = by_key[p["cls_key"]]
        p["cls"], p["wins"], p["generation_found"] = x["cls"], int(x["wins"]), int(x["generation_found"])
        p["hamming_to_trust"] = int(min_hamming(CLS(**x["cls"]), tcls, space.field_names))
    return {"ranking": ranking, "wins": [int(x["wins"]) for x in pop], "population": {x["cls_key"]: x["cls"] for x in pop},
            "proposals": props, "trust_best_rank": tb, "trust_in_top3": bool(tb is not None and tb < 3), "cfg": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in c.items()}}


# ---------------------------------------------------------------- 模型 / 上下文
def guide_model_paths(outdir: Path, guide: str, target: str, seeds: Sequence[int] = SEEDS) -> list[Path]:
    arm = GUIDE_ARM[guide]
    base = (outdir / "models" / ES_TAG) if guide == "ev_es" else (NB / "models" / _g76.TAG)
    return [base / target / f"{arm}_s{sd}.pt" for sd in seeds]


def load_guide_models(outdir: Path, guide: str, target: str, *, device: str, seeds: Sequence[int] = SEEDS) -> list[tuple]:
    return [_g34.load_pq_model(p, map_location=device, expect_fold=target, expect_budgets=pq.BUDGETS) for p in guide_model_paths(outdir, guide, target, seeds)]


def load_windows(outdir: Path, target: str, slices: Sequence[str]) -> dict[str, np.ndarray]:
    """{切片: windows_repr float32}（折特征；折 meta 的 fit 池不得含考侧，铁律 6）。"""
    fold_dir = outdir / "folds" / target
    meta = json.loads((fold_dir / "shared_ts2vec.meta.json").read_text(encoding="utf-8"))
    _m33.assert_fold_fit_pool(target, meta.get("fit_datasets", []))
    out = {}
    for s in sorted(set(slices)):
        path = fold_dir / "task_features" / f"{s}.npz"
        if not path.exists():
            raise SystemExit(f"缺特征 {path}：先跑 77 --fit-folds")
        with np.load(path) as npz:
            out[s] = npz["windows_repr"].astype(np.float32)
    return out


def ctx_row_for(models: Sequence[tuple], guide: str, slice_id: str, table: Mapping, *, budget: int = FIT_BUDGET, device: str = "cpu") -> torch.Tensor:
    meta0 = models[0][1]
    if _m33.arm_spec(GUIDE_ARM[guide]).ctx_kind == "table_budget":
        rows = _g64.slice_ctx_rows(table, tuple(meta0["ctx_cols"]), [slice_id], np.array(meta0["ctx_mu"]), np.array(meta0["ctx_sd"]))
        return torch.as_tensor(rows[(slice_id, int(budget))], dtype=torch.float32, device=device)
    return torch.as_tensor(np.asarray(pq.ctx_vector(int(budget), pq.BUDGETS)), dtype=torch.float32, device=device)


def e_rows_for(models: Sequence[tuple], guide: str, slice_id: str, windows: Optional[Mapping[str, np.ndarray]], *, device: str = "cpu") -> list:
    if guide != "ev_es":
        return [None] * len(models)
    w = torch.as_tensor(windows[slice_id], dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for m, _meta in models:
            out.append(m.encode_tasks([w])[0])
    return out


# ---------------------------------------------------------------- --evolve
def proposals_path(outdir: Path, name: str) -> Path:
    return outdir / f"proposals_{name}.json"


def load_proposals(outdir: Path, name: str) -> dict:
    p = proposals_path(outdir, name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def nb_labeled_keys(cat: Mapping, slice_id: str) -> set[str]:
    return {e["cls_key"] for e in cat["fixed"]} | {e["cls_key"] for e in cat["random"].get(slice_id, [])}


def cmd_evolve(outdir: Path, *, guides: Sequence[str], with_random: bool, device: str, smoke: bool = False, cfg: Optional[Mapping] = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    space = _g56.space()
    trust = trust_entries(space)
    tcls, tkeys = [c for c, _k in trust], {k for _c, k in trust}
    slices = ev_slices()[:1] if smoke else ev_slices()
    table = _g74.ctx_table(_g74.OUT) if any(g != "ev_es" for g in guides) else None
    cat = _g76.load_catalog(NB)
    summary: dict = {}
    for guide in guides:
        prev = load_proposals(outdir, guide)
        cache_models: dict[str, list] = {}
        cache_windows: dict[str, dict] = {}
        for s in slices:
            target = pq.dataset_of(s)
            if target not in cache_models:
                cache_models[target] = load_guide_models(outdir, guide, target, device=device)
                if guide == "ev_es":
                    cache_windows[target] = load_windows(outdir, target, [x for x in slices if pq.dataset_of(x) == target])
            models = cache_models[target]
            enc = _g56.encoder_for_meta(models[0][1])
            ctx_row = ctx_row_for(models, guide, s, table, device=device)
            e_rows = e_rows_for(models, guide, s, cache_windows.get(target), device=device)
            sp = _g34.make_score_pairs([m for m, _ in models], e_rows, ctx_row, device=device)
            t0 = time.time()
            res = evolve_slice(sp, space, enc, trust, seed=run_seed_for(s, "ev"), cfg=cfg)
            res["seconds"] = round(time.time() - t0, 1)
            res["proposals_in_nb_labeled"] = int(sum(p["cls_key"] in nb_labeled_keys(cat, s) for p in res["proposals"]))
            prev[s] = res
            print(f"[evolve {guide}] {s}: {res['seconds']}s proposals {[p['cls_key'][:8] for p in res['proposals']]} ranks {[p['rank'] for p in res['proposals']]} "
                  f"above_trust {[p['above_trust'] for p in res['proposals']]} h {[p['hamming_to_trust'] for p in res['proposals']]} trust_best_rank {res['trust_best_rank']}", flush=True)
        proposals_path(outdir, guide).write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
        summary[guide] = {"slices": len(prev), "above_trust_share": float(np.mean([p["above_trust"] for r in prev.values() for p in r["proposals"]])),
                          "trust_in_top3_share": float(np.mean([r["trust_in_top3"] for r in prev.values()]))}
    if with_random:
        prev = load_proposals(outdir, RAND_U)
        for s in slices:
            taken = set(tkeys) | nb_labeled_keys(cat, s)
            prev[s] = [{"cls_key": cls_key(c), "cls": c.to_dict(), "hamming_to_trust": int(min_hamming(c, tcls, space.field_names))} for c in uniform_random(space, s, tcls, taken=taken)]
        proposals_path(outdir, RAND_U).write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
        summary[RAND_U] = {"slices": len(prev)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


# ---------------------------------------------------------------- R'
def ev_cells(outdir: Path, *, names: Sequence[str], with_random: bool) -> list:
    """待标格：各 proposals_<guide>.json 的提名 + rand_u；同 (切片, key) 去重。"""
    seen: set[tuple[str, str]] = set()
    cells = []
    for name in list(names) + ([RAND_U] if with_random else []):
        props = load_proposals(outdir, name)
        for s, r in props.items():
            entries = r if name == RAND_U else r["proposals"]
            for p in entries:
                if (s, p["cls_key"]) in seen:
                    continue
                seen.add((s, p["cls_key"]))
                cells.append(_g32.Cell(s, ORIGIN_RANDOM, CLS(**p["cls"]), p["cls_key"]))
    cells.sort(key=lambda c: (c.slice_id, c.cls_key))
    return cells


def reuse_existing(outdir: Path, cells: Sequence, *, caches: Sequence[Path] = EXISTING_CACHES) -> int:
    """已在 NB1 / mx3 / SX 缓存里的格（同 canonical 种子）直接复制进 rprime_ev.jsonl，标 reused_from。"""
    idx_ev = _g32.cache_index(_g32.read_rows(outdir / "rprime_ev.jsonl"))
    n = 0
    with open(outdir / "rprime_ev.jsonl", "a", encoding="utf-8") as fh:
        for path in caches:
            if not path.exists():
                continue
            idx = _g32.cache_index(_g32.read_rows(path))
            for c in cells:
                keys = [(c.slice_id, c.cls_key, int(b)) for b in pq.BUDGETS]
                if any(k in idx_ev for k in keys) or not all(k in idx and idx[k].get("r_prime") is not None for k in keys):
                    continue
                for k in keys:
                    row = dict(idx[k])
                    row["reused_from"] = str(path.relative_to(ROOT)).replace("\\", "/")
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    idx_ev[k] = row
                n += 1
    return n


def cmd_rprime(outdir: Path, *, device: str, names: Sequence[str], with_random: bool, datasets: Sequence[str], smoke: bool) -> int:
    if smoke and outdir.resolve() == OUT.resolve():
        raise SystemExit("--smoke 不得写进真值目录 results/tclsc_ev")
    cells = ev_cells(outdir, names=names, with_random=with_random)
    cells = [c for c in cells if pq.dataset_of(c.slice_id) in set(datasets)]
    if smoke:
        cells = cells[:1]
    n_reused = reuse_existing(outdir, cells)
    idx = _g32.cache_index(_g32.read_rows(outdir / "rprime_ev.jsonl"))
    todo = _g32.pending_cells(cells, idx, pq.BUDGETS, resume=True, redo_partial=False)
    print(f"EV 网格: 计划 {len(cells)} 格 → 复用 {n_reused} → 待跑 {len(todo)}", flush=True)
    if not todo:
        return 0
    ds = _g75.load_k3_datasets(sorted({pq.dataset_of(c.slice_id) for c in todo}))
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    return _g32.run_jobs(_g32.make_jobs(todo, _g32.slice_index(ds)), budgets=pq.BUDGETS, cache_path=outdir / "rprime_ev.jsonl",
                         inflight=outdir / "inflight.txt", on_oom=ON_OOM_SKIP, oom_log=outdir / "oom_failures.jsonl",
                         eval_ratio=labels_cfg["eval_ratio"], batch_size=train_cfg["batch_size"], lr=train_cfg["lr"], device=device, label="ev")


def cmd_verify(outdir: Path, *, names: Sequence[str] = GUIDES, with_random: bool = True) -> dict:
    cells = ev_cells(outdir, names=[n for n in names if proposals_path(outdir, n).exists()], with_random=with_random and proposals_path(outdir, RAND_U).exists())
    rows = _g32.read_rows(outdir / "rprime_ev.jsonl")
    idx = _g32.cache_index(rows)
    keys = [(r["task_id"], r["cls_key"], int(r["iters"])) for r in rows]
    missing = [(c.slice_id, c.cls_key) for c in cells if any((c.slice_id, c.cls_key, int(b)) not in idx for b in pq.BUDGETS)]
    errs = [k for k, r in idx.items() if r.get("r_prime") is None]
    v = {"cells": len(cells), "rows": len(rows), "missing_cells": len(missing), "dup_keys": len(keys) - len(set(keys)), "errors": len(errs),
         "reused_rows": sum(1 for r in rows if r.get("reused_from")), "missing_examples": missing[:5]}
    v["ok"] = bool(not missing and v["dup_keys"] == 0 and not errs)
    print(f"[verify] cells {len(cells)} rows {len(rows)} missing {len(missing)} dup {v['dup_keys']} errors {len(errs)} reused {v['reused_rows']} → {'OK' if v['ok'] else 'NOT OK'}", flush=True)
    return v


# ---------------------------------------------------------------- SetTF 段（pod）
def cmd_fit_folds(outdir: Path, *, device: str, folds_sel: Optional[Sequence[str]] = None) -> None:
    """折共享 TS2Vec（fit 池 = 训练侧 7 集，铁律 6）+ 87 切片（原 45 + 扩展 42）窗口特征；33 号同一构造式。"""
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text("utf-8"))
    ra, pq_cfg = labels_cfg["route_a"], labels_cfg["pq"]
    data = _g75.load_k3_datasets()
    set_enc = _m33.get_set_encoder_init(outdir / "set_encoder_init.pt", nz=ra["nz"], seed=ra["embed_seed"]).to("cpu").eval()
    for target in (folds_sel or FOLDS):
        fold_dir = outdir / "folds" / target
        fit_pool = pq.fold_split(target)[0]
        shared = _m33.get_fold_ts2vec(fold_dir, target, fit_pool, data, ra, device)
        t0 = time.time()
        n = _m33._encode_all_slices(fold_dir / "task_features", data, shared, set_enc, ra=ra, sample_num=int(pq_cfg["settf_windows"]),
                                    t_max=int(pq_cfg["settf_t_max"]), names=list(pq.PQ_DATASETS))
        print(f"[fit-folds {target}] 特征 {n} 切片 {time.time() - t0:.0f}s", flush=True)
        del shared
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def train_fold_es(rows: Sequence[dict], delta: Mapping, *, target: str, seed: int, outdir: Path, device: str, epochs: Optional[int]) -> dict:
    """es_field_hard：73 号 train_fold_uea 的 SetTF 版——窗口特征进引擎、SetTF 与比较器联合训练（33 号 run_arm 同构）。"""
    spec = _m33.arm_spec(ES_ARM)
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text("utf-8"))
    ra, pq_cfg = labels_cfg["route_a"], labels_cfg["pq"]
    train_rows = [r for r in rows if r["dataset"] != target]
    slices = sorted({r["task_id"] for r in train_rows})
    recs = pq.build_records(_g60.rows_to_cache(train_rows), _g60.enc(), budgets=pq.BUDGETS, slices=slices)
    c = dict(_g60.cfg())
    if epochs:
        c["epochs"], c["curriculum_epochs"] = int(epochs), min(int(c["curriculum_epochs"]), max(1, int(epochs) // 2))
    cls_dim = int(recs[0].vec.shape[0])
    windows = load_windows(outdir, target, slices)
    wk = {pq.engine_task_id(s, int(b)): s for s in slices for b in pq.BUDGETS}
    ctx_map = {pq.engine_task_id(s, int(b)): pq.ctx_vector(int(b), pq.BUDGETS) for s in slices for b in pq.BUDGETS}
    cdim = len(pq.BUDGETS)
    model = _m33.build_model(spec, cls_dim=cls_dim, ctx_dim=cdim, nz=int(ra["nz"]), dim_input=_m33.TS2VEC_OUTPUT_DIMS, cfg=c, seed=seed,
                             init_path=outdir / "set_encoder_init.pt")
    engine = TCLSCEngine(model, windows, device=device, lr=c["lr"], batch_size=c["batch_size"], seed=seed, windows_key_of=wk, context_of=ctx_map)
    t0 = time.time()
    hist = engine.train(recs, epochs=c["epochs"], curriculum_epochs=c["curriculum_epochs"],
                        better_fn=lambda a, b: is_better(a, b, TASK_CLASSIFICATION), delta=pq.delta_map_for_fold(delta, slices, pq.BUDGETS))
    p = outdir / "models" / ES_TAG / target / f"{ES_ARM}_s{seed}.pt"
    _m33.save_model(p, model, spec, fold=target, seed=seed, cls_dim=cls_dim, ctx_dim=cdim, cfg=c, nz=int(ra["nz"]), settf_windows=int(pq_cfg["settf_windows"]),
                    settf_t_max=int(pq_cfg["settf_t_max"]), budgets=pq.BUDGETS, ctx_encoding="onehot", encoder_mode="binned", head=spec.head, ctx_kind=spec.ctx_kind)
    run = {"tag": ES_TAG, "test": target, "arm": ES_ARM, "seed": seed, "n_train_slices": len(slices), "n_train_records": len(recs),
           "train_seconds": round(time.time() - t0, 1), "bce_first": hist[0]["loss"], "bce_last": hist[-1]["loss"], "train_pairs_last": hist[-1]["n_pairs"]}
    rp = outdir / "runs" / ES_TAG / target / f"{ES_ARM}_s{seed}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    del engine, model, windows
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return run


def cmd_train_es(outdir: Path, *, seeds: Sequence[int], device: str, epochs: Optional[int], resume: bool, folds_sel: Optional[Sequence[str]] = None) -> None:
    rows, dk = _g76.load_rows(NB, "train")
    for target in (folds_sel or FOLDS):
        for seed in seeds:
            if resume and (outdir / "models" / ES_TAG / target / f"{ES_ARM}_s{seed}.pt").exists():
                continue
            run = train_fold_es(rows, dk, target=target, seed=seed, outdir=outdir, device=device, epochs=epochs)
            _g73._log(outdir, f"[{ES_TAG}/{target}/{ES_ARM}_s{seed}] {run['train_seconds']}s n_slices {run['n_train_slices']} recs {run['n_train_records']} pairs {run['train_pairs_last']}")


def cmd_rank_es(outdir: Path, *, device: str, seeds: Sequence[int] = SEEDS) -> dict:
    """NB1 考卷（12 切片 × 20 条）上 es_field_hard 的 normal / cross / zero 排名（62 号 rank_target）。"""
    cat = _g76.load_catalog(NB)
    all_sl = _g75.all_slices()
    out = {}
    for target in FOLDS:
        paths = guide_model_paths(outdir, "ev_es", target, seeds)
        if not all(p.exists() for p in paths):
            print(f"⚠️ [{target}] es 模型不全，跳过", flush=True)
            continue
        models = load_guide_models(outdir, "ev_es", target, device=device, seeds=seeds)
        test_slices = _g75.exam_slices(target)
        train_slices = [s for s in all_sl if pq.dataset_of(s) != target]
        w = load_windows(outdir, target, all_sl)
        windows = {s: torch.as_tensor(x, dtype=torch.float32, device=device) for s, x in w.items()}
        catm = {s: _g76.catalog_entries(cat, s) for s in test_slices}
        per = _g62.rank_target(models, catm, windows, train_slices=train_slices, budgets=pq.BUDGETS, cross_seeds=_g76.CROSS_SEEDS, device=device, arm=ES_ARM)
        per["test"] = test_slices
        out[target] = per
        print(f"[rank-es] {target} diag {per['diag']}", flush=True)
        del windows, w
    rk = {"".join(map(str, seeds)): out}
    (outdir / "rankings_es.json").write_text(json.dumps(rk, ensure_ascii=False, indent=1), encoding="utf-8")
    return rk


# ---------------------------------------------------------------- 分析
def subset_expectation(vals: Sequence[float], *, k: int = N_PROPOSE) -> dict:
    """C(n,k) 个子集的 max / mean 期望（rand_nb 对照）。"""
    v = [float(x) for x in vals]
    subs = list(itertools.combinations(v, k)) if len(v) >= k else [tuple(v)]
    return {"max": float(np.mean([max(s) for s in subs])), "mean": float(np.mean([np.mean(s) for s in subs])), "n": len(v), "k": k}


def slice_stats(truth: Mapping[str, float], keys: Sequence[str], *, ref: float, delta: float) -> dict:
    vals = [float(truth[k]) for k in keys if k in truth]
    if not vals:
        return {"n": 0, "best3": None, "mean3": None, "gap": None, "over_delta": None}
    best = max(vals)
    return {"n": len(vals), "best3": best, "mean3": float(np.mean(vals)), "gap": float(best - ref), "over_delta": bool(best - ref > delta)}


def paired_block(x_by: Mapping[str, float], y_by: Mapping[str, float]) -> dict:
    """配对（切片）：H1 = x > y（单侧 Wilcoxon）。"""
    common = sorted(k for k in x_by if k in y_by and x_by[k] is not None and y_by[k] is not None)
    if not common:
        return {"n": 0, "mean_x": None, "mean_y": None, "delta": None, "p": None, "wins": 0, "losses": 0}
    x, y = np.array([x_by[k] for k in common], float), np.array([y_by[k] for k in common], float)
    return {"n": len(common), "mean_x": float(x.mean()), "mean_y": float(y.mean()), "delta": float((x - y).mean()),
            "p": float(_g56.paired_wilcoxon_less(y, x)), "wins": int(np.sum(x > y)), "losses": int(np.sum(x < y))}


def gate_v1(stats: Sequence[Mapping]) -> bool:
    st = [s for s in stats if s.get("gap") is not None]
    if not st:
        return False
    return bool(np.mean([s["gap"] > 0 for s in st]) >= G1_POS and np.mean([bool(s["over_delta"]) for s in st]) >= G1_DELTA)


def gate_v2(blk: Mapping) -> bool:
    return bool(blk.get("p") is not None and blk["delta"] >= G2_MIN and blk["p"] < ALPHA)


def gate_v3(blk: Mapping) -> bool:
    return bool(blk.get("p") is not None and blk["delta"] > 0 and blk["p"] < ALPHA)


def verdict_ev(g: Mapping[str, Mapping[str, bool]]) -> str:
    e_arms = [a for a in E_GUIDES if a in g]
    if any(all(g[a].get(k) for k in ("G-V1", "G-V2", "G-V3")) for a in e_arms):
        return "EV-SEARCH-WIN"
    if any(g[a].get("G-V2") and g[a].get("G-V3") and not g[a].get("G-V1") for a in e_arms):
        return "EV-GUIDED-FLAT"
    if any(g[a].get("G-V1") for a in g) and not any(g[a].get("G-V2") for a in e_arms):
        return "EV-HEADROOM-RANDOM"
    return "EV-DEAD"


def _ggs_rows() -> dict[tuple[str, int], float]:
    out = {}
    for path in (SOURCE / "rprime_cache.jsonl", SX / "rprime_cache.jsonl"):
        for r in _g32.read_rows(path):
            if r.get("origin") == "ggs" and r.get("r_prime") is not None:
                out[(r["task_id"], int(r["iters"]))] = float(r["r_prime"])
    return out


def references(slice_id: str, budget: int, *, k3_rows: Sequence[dict], nb_truth: Sequence[dict], ggs: Mapping) -> dict:
    """参照：ref = max(shared-10 k3 均值, GGS k1)；ref_k1 = NB1 口径 shared-10 seed-0 单值。"""
    sh3 = [float(r["r_prime"]) for r in k3_rows if r["task_id"] == slice_id and int(r["iters"]) == budget]
    sh1 = [float(r["r_prime"]) for r in nb_truth if r["task_id"] == slice_id and int(r["iters"]) == budget and "nb_h" not in r]
    g = ggs.get((slice_id, budget))
    return {"shared_k3": max(sh3) if sh3 else None, "ggs_k1": g, "ref": max([v for v in (max(sh3) if sh3 else None, g) if v is not None]),
            "ref_k1": max(sh1) if sh1 else None}


def cmd_analyze(outdir: Path) -> dict:
    space = _g56.space()
    ev_rows = _g32.read_rows(outdir / "rprime_ev.jsonl")
    val = {(r["task_id"], int(r["iters"]), r["cls_key"]): float(r["r_prime"]) for r in ev_rows if r.get("r_prime") is not None}
    k3_rows = _g32.read_rows(K3 / "labels_k3.jsonl")
    nb_truth, dk = _g76.load_rows(NB, "truth")
    ggs = _ggs_rows()
    guides = [g for g in GUIDES if proposals_path(outdir, g).exists()]
    props = {g: load_proposals(outdir, g) for g in guides}
    rand_u = load_proposals(outdir, RAND_U)
    arms = guides + ([RAND_U] if rand_u else []) + [RAND_NB]
    slices = [s for s in ev_slices() if all(s in props[g] for g in guides)]
    a: dict = {"protocol": "results/tclsc_ev/PROTOCOL.md", "slices": slices, "arms": arms, "per_budget": {}, "paired": {}, "gates": {}, "over": {}, "diag": {}}
    per_slice_290: dict[str, dict] = {}
    for b in pq.BUDGETS:
        stats: dict[str, dict] = {arm: {} for arm in arms}
        refs = {}
        for s in slices:
            d = pq.dataset_of(s)
            ref = references(s, int(b), k3_rows=k3_rows, nb_truth=nb_truth, ggs=ggs)
            refs[s] = ref
            dl = float(dk[(d, int(b))])
            truth = {k: v for (t, bb, k), v in val.items() if t == s and bb == int(b)}
            for g in guides:
                keys = [p["cls_key"] for p in props[g][s]["proposals"]]
                st = slice_stats(truth, keys, ref=ref["ref"], delta=dl)
                st.update({"pred_above": float(np.mean([p["above_trust"] for p in props[g][s]["proposals"]])),
                           "realized_above": float(np.mean([truth[k] > ref["ref"] for k in keys if k in truth])) if any(k in truth for k in keys) else None,
                           "gap_k1": (st["best3"] - ref["ref_k1"]) if st["best3"] is not None and ref["ref_k1"] is not None else None,
                           "hamming": [p["hamming_to_trust"] for p in props[g][s]["proposals"]], "trust_in_top3": props[g][s]["trust_in_top3"]})
                stats[g][s] = st
            if rand_u:
                keys = [p["cls_key"] for p in rand_u[s]]
                st = slice_stats(truth, keys, ref=ref["ref"], delta=dl)
                st["realized_above"] = float(np.mean([truth[k] > ref["ref"] for k in keys if k in truth])) if any(k in truth for k in keys) else None
                stats[RAND_U][s] = st
            nbv = [float(r["r_prime"]) for r in nb_truth if r["task_id"] == s and int(r["iters"]) == int(b) and "nb_h" in r and not r.get("nb_fixed")]
            if len(nbv) >= N_PROPOSE:
                e = subset_expectation(nbv, k=N_PROPOSE)
                stats[RAND_NB][s] = {"n": e["n"], "best3": e["max"], "mean3": e["mean"], "gap": float(e["max"] - ref["ref"]), "over_delta": bool(e["max"] - ref["ref"] > dl)}
        agg = {}
        for arm in arms:
            st = [x for x in stats[arm].values() if x.get("gap") is not None]
            by_ds: dict[str, list] = {}
            for s, x in stats[arm].items():
                if x.get("gap") is not None:
                    by_ds.setdefault(pq.dataset_of(s), []).append(x["gap"])
            agg[arm] = {"n": len(st), "frac_positive": float(np.mean([x["gap"] > 0 for x in st])) if st else None,
                        "frac_over_delta": float(np.mean([bool(x["over_delta"]) for x in st])) if st else None,
                        "median_gap": float(np.median([x["gap"] for x in st])) if st else None,
                        "mean_best3": float(np.mean([x["best3"] for x in st])) if st else None, "mean_mean3": float(np.mean([x["mean3"] for x in st])) if st else None,
                        "pred_above_share": float(np.mean([x["pred_above"] for x in st])) if st and "pred_above" in st[0] else None,
                        "realized_above_share": float(np.mean([x["realized_above"] for x in st if x.get("realized_above") is not None])) if st and "realized_above" in st[0] else None,
                        "by_dataset_median_gap": {d: float(np.median(v)) for d, v in sorted(by_ds.items())},
                        "over_delta_slices": [s for s, x in stats[arm].items() if x.get("over_delta")]}
        a["per_budget"][str(b)] = {"refs": refs, "stats": stats, "agg": agg}
        if int(b) == FIT_BUDGET:
            per_slice_290 = stats
    st = per_slice_290
    m3 = lambda arm: {s: x["mean3"] for s, x in st.get(arm, {}).items()}
    b3 = lambda arm: {s: x["best3"] for s, x in st.get(arm, {}).items()}
    for g in guides:
        a["paired"][f"{g}_vs_rand_u_mean3"] = paired_block(m3(g), m3(RAND_U))
        a["paired"][f"{g}_vs_rand_u_best3"] = paired_block(b3(g), b3(RAND_U))
        a["paired"][f"{g}_vs_rand_nb_mean3"] = paired_block(m3(g), m3(RAND_NB))
    for g in E_GUIDES:
        if g in guides and "ev_blind" in guides:
            a["paired"][f"{g}_vs_blind_mean3"] = paired_block(m3(g), m3("ev_blind"))
            a["paired"][f"{g}_vs_blind_best3"] = paired_block(b3(g), b3("ev_blind"))
    if all(g in guides for g in E_GUIDES):
        a["paired"]["ev_es_vs_ev_et_mean3"] = paired_block(m3("ev_es"), m3("ev_et"))
    agg290 = a["per_budget"][str(FIT_BUDGET)]["agg"]
    for arm in arms:
        gates = {"G-V1": gate_v1(list(st.get(arm, {}).values()))}
        if arm in guides:
            gates["G-V2"] = gate_v2(a["paired"][f"{arm}_vs_rand_u_mean3"])
        if arm in E_GUIDES and arm in guides and "ev_blind" in guides:
            gates["G-V3"] = gate_v3(a["paired"][f"{arm}_vs_blind_mean3"])
            over = (agg290[arm]["pred_above_share"] or 0.0) - (agg290[arm]["realized_above_share"] or 0.0)
            a["over"][arm] = {"pred_above_share": agg290[arm]["pred_above_share"], "realized_above_share": agg290[arm]["realized_above_share"], "over": over, "G-V4": bool(over <= G4_OVER_MAX)}
        a["gates"][arm] = gates
    if "ev_blind" in guides:
        a["over"]["ev_blind"] = {"pred_above_share": agg290["ev_blind"]["pred_above_share"], "realized_above_share": agg290["ev_blind"]["realized_above_share"],
                                "over": (agg290["ev_blind"]["pred_above_share"] or 0.0) - (agg290["ev_blind"]["realized_above_share"] or 0.0)}
    a["verdict"] = verdict_ev(a["gates"])
    a["diag"] = {"reused_rows": sum(1 for r in ev_rows if r.get("reused_from")), "n_rows": len(ev_rows),
                 "proposals_in_nb_labeled": {g: int(sum(props[g][s].get("proposals_in_nb_labeled", 0) for s in slices)) for g in guides},
                 "trust_in_top3_share": {g: float(np.mean([props[g][s]["trust_in_top3"] for s in slices])) for g in guides},
                 "hamming_hist": {g: {str(h): int(sum(p["hamming_to_trust"] == h for s in slices for p in props[g][s]["proposals"])) for h in (1, 2, 3)} for g in guides},
                 "overlap_between_guides": {f"{x}&{y}": int(sum(len({p["cls_key"] for p in props[x][s]["proposals"]} & {p["cls_key"] for p in props[y][s]["proposals"]}) for s in slices))
                                            for x, y in itertools.combinations(guides, 2)}}
    a["es_exam"] = es_exam(outdir)
    a["predictions"] = predictions(a)
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], json.dumps(a["gates"], ensure_ascii=False), flush=True)
    return a


def es_exam(outdir: Path) -> Optional[dict]:
    """NB1 考卷上 es_field_hard（seeds 0/1/2 单组）vs NB1 的 ft60 / 任务盲同组：regret@3 / hit@3 / M1 / M0 + cross / zero 消融。"""
    p = outdir / "rankings_es.json"
    if not p.exists() or not (NB / "rankings.json").exists():
        return None
    rk_es = json.loads(p.read_text(encoding="utf-8"))
    rk_nb = json.loads((NB / "rankings.json").read_text(encoding="utf-8"))
    truth, dk = _g76.load_rows(NB, "truth")
    cat_z = _g56.zscore_within(truth)
    group = next(iter(rk_es))
    rows_es = _g76._eval(rk_es[group], cat_z, dk, [ES_ARM], "normal", group=group)
    rows_z = _g76._eval(rk_es[group], cat_z, dk, [ES_ARM], "zero", group=group)
    rows_c = [_g76._eval(rk_es[group], cat_z, dk, [ES_ARM], "cross", cross_seed=cs, group=group) for cs in map(str, _g76.CROSS_SEEDS)]
    out = {"group": group, "es": _g73._pooled(rows_es, ES_ARM), "ablation": {"cross": _g75._cross_ablation(rows_es, rows_c, ES_ARM), "zero": _g75._cross_ablation(rows_es, [rows_z], ES_ARM)},
           "diag": {t: per.get("diag") for t, per in rk_es[group].items()}}
    out["es"]["hit3"] = float(np.mean([r["regret"][ES_ARM + "3"] == 0 for r in rows_es if ES_ARM + "3" in r["regret"]]))
    if group in rk_nb:
        rows_nb = _g76._eval(rk_nb[group], cat_z, dk, list(_g76.ARMS), "normal", group=group)
        out["nb_same_group"] = {arm: _g73._pooled(rows_nb, arm) for arm in _g76.ARMS}
        key = lambda r: (r["fold"], r["task_id"] if "task_id" in r else r.get("slice"), r.get("budget"), r.get("iters"))
        merged = {}
        for r in rows_nb:
            merged[json.dumps([r.get(k) for k in ("fold", "task_id", "iters", "budget", "slice")])] = dict(r)
        for r in rows_es:
            k = json.dumps([r.get(k) for k in ("fold", "task_id", "iters", "budget", "slice")])
            if k in merged:
                merged[k]["regret"].update(r["regret"])
        rows_all = [r for r in merged.values() if ES_ARM + "3" in r["regret"]]
        out["paired"] = {f"es_vs_{arm}": _g58.paired_regret(rows_all, arm, ES_ARM, "3") for arm in _g76.ARMS}
        out["n_merged"] = len(rows_all)
    return out


def predictions(a: dict) -> dict:
    g = a["gates"]
    agg = a["per_budget"][str(FIT_BUDGET)]["agg"]
    over = a["over"]
    et, es = g.get("ev_et", {}), g.get("ev_es", {})
    p4 = None if not (agg.get("ev_es") and agg.get("ev_et") and agg["ev_es"]["mean_mean3"] is not None) else bool(agg["ev_es"]["mean_mean3"] <= agg["ev_et"]["mean_mean3"])
    p5 = None if not ("ev_blind" in over and "ev_et" in over) else bool(over["ev_blind"]["over"] - over["ev_et"]["over"] >= 0.10)
    ds_over = {pq.dataset_of(s) for arm in a["arms"] for s in agg.get(arm, {}).get("over_delta_slices", [])}
    p6 = None if not ds_over else bool(ds_over <= {"HandMovementDirection", "EthanolConcentration"})
    t3 = a["diag"]["trust_in_top3_share"].get("ev_et")
    return {"P1": {"desc": "G-V1 ev_et", "hit": et.get("G-V1")}, "P2": {"desc": "G-V2 ev_et", "hit": et.get("G-V2")}, "P3": {"desc": "G-V3 ev_et > blind", "hit": et.get("G-V3")},
            "P4": {"desc": "ev_es mean3 ≤ ev_et mean3", "hit": p4}, "P5": {"desc": "高估 blind − et ≥ 0.10", "hit": p5},
            "P6": {"desc": "超 δ 只在 HMD / Ethanol", "value": sorted(ds_over), "hit": p6}, "P7": {"desc": "≥ 50% 切片最终种群前 3 含信任域（ev_et）", "value": t3, "hit": None if t3 is None else bool(t3 >= 0.5)},
            "es_gates": es}


def markdown_tables(a: dict) -> str:
    f = lambda v, d=4: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    L = [f"# EV1 检验 tables：源侧进化搜索验证（{len(a['slices'])} 切片）", "", f"读数码 **{a['verdict']}**（门 {a['gates']}）", ""]
    for b in map(str, pq.BUDGETS):
        agg = a["per_budget"][b]["agg"]
        L += [f"## @{b}（参照 = max(shared-10 k3, GGS k1)）", "", "| 臂 | n | 正占比 | 超 δ 占比 | 中位 gap | mean best3 | mean mean3 | 预测超信任域占比 | 真超占比 | 超 δ 切片 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for arm, c in agg.items():
            L.append(f"| {arm} | {c['n']} | {f(c['frac_positive'], 2)} | {f(c['frac_over_delta'], 2)} | {f(c['median_gap'])} | {f(c['mean_best3'])} | {f(c['mean_mean3'])} | {f(c['pred_above_share'], 2)} | {f(c['realized_above_share'], 2)} | {[s for s in c['over_delta_slices']]} |")
        L.append("")
    L += ["## 配对（@290，H1 = 前者更高）", "", "| 对比 | n | 前者 | 后者 | Δ | p | 胜:负 |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, blk in a["paired"].items():
        L.append(f"| {name} | {blk['n']} | {f(blk['mean_x'])} | {f(blk['mean_y'])} | {f(blk['delta'])} | {f(blk['p'], 3)} | {blk['wins']}:{blk['losses']} |")
    L += ["", f"高估（预测超信任域占比 − 真超占比）：{a['over']}", "", f"诊断：{a['diag']}", ""]
    if a.get("es_exam"):
        e = a["es_exam"]
        L += ["## NB1 考卷上 es_field_hard（同组种子）", "", f"es：{e['es']}", f"NB1 同组：{e.get('nb_same_group')}", f"配对：{e.get('paired')}", f"消融：{e['ablation']}", f"E 诊断：{e['diag']}", ""]
    L += [f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for fl in ("evolve", "random", "rprime", "verify", "fit-folds", "train-es", "rank-es", "analyze", "smoke", "resume"):
        ap.add_argument(f"--{fl}", action="store_true")
    ap.add_argument("--guides", nargs="+", default=["ev_et", "ev_blind"], choices=list(GUIDES))
    ap.add_argument("--datasets", nargs="+", default=list(pq.PQ_DATASETS))
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.fit_folds:
        cmd_fit_folds(args.outdir, device=args.device, folds_sel=args.folds)
    if args.train_es:
        cmd_train_es(args.outdir, seeds=args.seeds, device=args.device, epochs=args.epochs, resume=args.resume, folds_sel=args.folds)
    if args.evolve:
        cmd_evolve(args.outdir, guides=args.guides, with_random=args.random, device=args.device, smoke=args.smoke,
                   cfg=(dict(ks=20, kp=6, generations=2) if args.smoke else None))
    if args.rprime:
        cmd_rprime(args.outdir, device=args.device, names=args.guides, with_random=args.random, datasets=args.datasets, smoke=args.smoke)
    if args.verify:
        cmd_verify(args.outdir)
    if args.rank_es:
        cmd_rank_es(args.outdir, device=args.device, seeds=args.seeds)
    if args.analyze:
        cmd_analyze(args.outdir)


if __name__ == "__main__":
    main()
