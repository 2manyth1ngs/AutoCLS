"""MX-2 端到端零样本搜索：UM↔HW 双向 leave-one-dataset-out
（协议 results/mx2_endtoend/PROTOCOL.md，判读带/先验先于数据）。

材料 = C2 语料真值网格（28 任务 × 20 CLS @工作点，k=1 canonical）；
方向别重 fit 共享 TS2Vec（铁律 6：只喂训练侧数据集，考侧仅前向编码）；
T-CLSC 3 种子多数票排名；四臂 regret（oracle/T-CLSC@1/@3/GGS/random/bottom）；
外部锚 = 全量 Handwriting 标准协议 vs TS2Vec 论文公开数字。

    python scripts/25_mx2_endtoend.py --prepare [--resume]     # TS2Vec×2 + 特征 + T-CLSC 6 跑 + 排名
    python scripts/25_mx2_endtoend.py --run-cells --resume     # GGS 28 格 + k=3 确认格
    python scripts/25_mx2_endtoend.py --anchor --resume        # 全量 HW 外部锚 6 跑
    python scripts/25_mx2_endtoend.py --analyze                # 四臂 regret + 判读（纯聚合）
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_r2b = importlib.import_module("scripts.2b_generate_labels_route_a")
_m1 = importlib.import_module("scripts.m1_ggs_har")
_zs = importlib.import_module("scripts.5_target_zero_shot")
_lo = importlib.import_module("scripts.16_feasibility_loo")

from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET, load_registry, load_source_dataset, slice_source_tasks,
)
from autocls.data.uea import load_uea_cached  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.downstream.linear_eval import evaluate_classification  # noqa: E402
from autocls.labeling import (  # noqa: E402
    append_jsonl, cls_key, comparator_records, evaluate_rprime,
    load_rprime_cache, run_seed_for,
)
from autocls.oom_guard import ON_OOM_SKIP, guarded_run  # noqa: E402
from autocls.search.rank import (  # noqa: E402
    borda_ranking, pair_vote_matrix, symmetrize,
)
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_embed import TaskSetEncoder  # noqa: E402
from autocls.task_types import is_better  # noqa: E402

OUTDIR_DEFAULT = ROOT / "results" / "mx2_endtoend"
LABELS_DIR = ROOT / "results" / "labels_workpoint"
# 协议 §1 写死：A 主（训 UM 考 HW）、B 副（训 HW 考 UM）
DIRECTIONS = {
    "A": {"train": "UniMiB-SHAR", "test": "Handwriting"},
    "B": {"train": "Handwriting", "test": "UniMiB-SHAR"},
}
REG_OF = {"Handwriting": None, "UniMiB-SHAR": "configs/candidate_pool.yaml"}
SEEDS = (0, 1, 2)             # T-CLSC 模型种子（协议 §1）
K_TOP = 3                     # top-K 真评估（协议 §2）
CONFIRM_REPS = (1, 2)         # canonical 之外的额外 weight seed（2c 同口径）
ANCHOR_SEEDS = (0, 1, 2)
TS2VEC_PAPER_HW = 0.515       # TS2Vec 论文 Handwriting 公开数字（协议 §3）
ANCHOR_MARGIN = 0.05
N_SHARED = 10


# ---------------------------------------------------------------- 纯函数（单测覆盖）
def expected_max_of_k(values, k: int) -> float:
    """不放回均匀抽 k 取最大的解析期望（次序统计量：P(max=v_(i))=C(i-1,k-1)/C(n,k)）。"""
    v = sorted(float(x) for x in values)
    n = len(v)
    assert 1 <= k <= n
    tot = math.comb(n, k)
    return sum(v[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / tot


def order_candidates(V: np.ndarray, tiebreak: np.ndarray, keys) -> list[int]:
    """多数票胜负矩阵 → 排名下标：胜场降序 → 平均得分降序 → key 升序（逐位确定）。"""
    wins = np.asarray(V).sum(axis=1)
    assert len(wins) == len(tiebreak) == len(keys)
    return sorted(range(len(keys)),
                  key=lambda i: (-int(wins[i]), -float(tiebreak[i]), keys[i]))


def task_arms(truth: dict, ranked_keys: list, ggs_r: float,
              k_top: int = K_TOP) -> dict:
    """协议 §2 四臂（+oracle/bottom）。truth: key→R'（该任务 20 候选真值网格）。"""
    vals = list(truth.values())
    return {
        "oracle": max(vals),
        "tclsc1": truth[ranked_keys[0]],
        "tclsc3": max(truth[k] for k in ranked_keys[:k_top]),
        "ggs": float(ggs_r),
        "random1": float(np.mean(vals)),
        "random3": expected_max_of_k(vals, k_top),
        "bottom": min(vals),
    }


def regret_of(arms: dict) -> dict:
    """regret = oracle − 臂（GGS 目录外可为负，协议 §2 声明）。"""
    return {k: arms["oracle"] - v for k, v in arms.items()}


def confirm_worklist(top1: dict, oracle: dict, ggs_key: str) -> list:
    """k=3 确认工单：每任务 {T-CLSC@1, oracle, GGS} 去重 → 排序 (tid, key) 列表。"""
    out = set()
    for tid in top1:
        out |= {(tid, top1[tid]), (tid, oracle[tid]), (tid, ggs_key)}
    return sorted(out)


def bottom_quartile_hit(truth: dict, pick_key: str) -> bool:
    """pick 是否落入该任务垫底四分位（n=20 → 倒数 5）。"""
    vals = sorted(truth.values())
    thresh = vals[max(1, len(vals) // 4) - 1]
    return truth[pick_key] <= thresh


def ds_of(tid: str) -> str:
    return tid.rsplit("_", 1)[0]


# ---------------------------------------------------------------- 装配
def load_cfgs():
    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml")
                                .read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml")
                               .read_text(encoding="utf-8"))
    tcfg = yaml.safe_load((ROOT / "configs" / "pretrain_tclsc.yaml")
                          .read_text(encoding="utf-8"))
    return labels_cfg, train_cfg, tcfg


def load_datasets() -> dict:
    """{name: (X, y, tasks)}，双注册表（2c SPEC 同口径）+ 铁律 6 断言。"""
    per = {}
    for name, regpath in REG_OF.items():
        reg = load_registry(ROOT / regpath) if regpath else load_registry()
        assert name not in reg.names(ROLE_TARGET), f"{name} 是目标池成员（铁律 6）"
        entry = reg.get(name)
        X, y = load_source_dataset(name, reg)
        _, tasks = slice_source_tasks(entry, reg, y)
        per[name] = (X, y, tasks)
        print(f"[{name}] X={X.shape} tasks={len(tasks)}", flush=True)
    return per


def load_catalog():
    """cls_catalog.json → (shared [(CLS,key)×10], random {tid: [(CLS,key)×10]})。"""
    cat = json.loads((LABELS_DIR / "cls_catalog.json").read_text(encoding="utf-8"))
    shared = [(CLS(**d["cls"]), d["cls_key"]) for d in cat["shared"]]
    rand = {tid: [(CLS(**d["cls"]), d["cls_key"]) for d in lst]
            for tid, lst in cat["random"].items()}
    assert len(shared) == N_SHARED
    return shared, rand


def load_truth(iters_map: dict) -> dict:
    """真值网格：tid → key → rec（含 r_prime/seconds，协议 §1 的 k=1 canonical）。"""
    cache = load_rprime_cache(LABELS_DIR / "rprime_cache.jsonl")
    truth: dict = defaultdict(dict)
    for (tid, key, it), rec in cache.items():
        if it == iters_map[ds_of(tid)] and rec.get("r_prime") is not None:
            truth[tid][key] = rec
    return truth


def truth_r(truth: dict, tid: str) -> dict:
    return {k: float(rec["r_prime"]) for k, rec in truth[tid].items()}


def candidates_for(tid: str, shared, rand) -> tuple[list, list]:
    """任务候选 = shared 10（下标 0..9）+ random 10：(CLS 列表, key 列表)。"""
    pairs = list(shared) + list(rand[tid])
    return [c for c, _k in pairs], [k for _c, k in pairs]


# ---------------------------------------------------------------- --prepare
def cmd_prepare(args) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels_cfg, _train_cfg, tcfg = load_cfgs()
    ra, wp = labels_cfg["route_a"], labels_cfg["workpoint"]
    iters_map, delta_map = wp["iters_per_dataset"], wp["delta_per_dataset"]
    task_type = labels_cfg["task_type"]
    per = load_datasets()
    shared_cat, rand_cat = load_catalog()
    truth = load_truth(iters_map)
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    assert space.validate(_m1.GGS), "GGS 不在合法空间内"

    cache = load_rprime_cache(LABELS_DIR / "rprime_cache.jsonl")
    records_by_ds = {}
    for d in per:
        rs = comparator_records(cache, encoder, iters=iters_map[d])
        assert all(ds_of(r.task_id) == d for r in rs), \
            f"iters={iters_map[d]} 捞到非 {d} 记录"
        records_by_ds[d] = rs
    n_tasks = {d: len({r.task_id for r in records_by_ds[d]}) for d in per}
    assert n_tasks == {"Handwriting": 12, "UniMiB-SHAR": 16}, n_tasks

    meta = {"seconds": {}}
    rankings = {}
    for dirn, spec in DIRECTIONS.items():
        tr_name, te_name = spec["train"], spec["test"]
        ddir = outdir / f"dir_{dirn}"
        ddir.mkdir(exist_ok=True)
        Xtr, ytr, tasks_tr = per[tr_name]
        Xte, yte, tasks_te = per[te_name]

        # ① 方向别共享 TS2Vec：只 fit 训练侧（get_shared_ts2vec 指纹断言防漂移）
        t0 = time.time()
        shared_w = _r2b.get_shared_ts2vec(
            ddir, [(tr_name, Xtr, ytr, tasks_tr)], [tr_name], ra, args.device)
        meta["seconds"][f"ts2vec_{dirn}"] = round(time.time() - t0, 1)
        set_enc0 = _r2b.get_set_encoder(LABELS_DIR / "set_encoder_init.pt",
                                        shared_w, ra)

        # ② 28 任务特征（训练侧供训练、考侧仅冻结前向编码）
        feat_dir = ddir / "task_features"
        feat_dir.mkdir(exist_ok=True)
        for nm, X, y, tasks in ((tr_name, Xtr, ytr, tasks_tr),
                                (te_name, Xte, yte, tasks_te)):
            for t in tasks:
                _r2b.ensure_task_features(t, X, y, shared_w, set_enc0, ra, feat_dir)
        task_windows = {t.task_id: np.load(feat_dir / f"{t.task_id}.npz")["windows_repr"]
                        for t in list(tasks_tr) + list(tasks_te)}

        # ③ T-CLSC 3 种子训练（4b 装配同款）+ 考侧任务成对概率
        train_records = records_by_ds[tr_name]
        delta_by_task = {r.task_id: delta_map[tr_name] for r in train_records}
        dim_input = next(iter(task_windows.values())).shape[-1]
        cls_dim = train_records[0].vec.shape[0]
        S_by_task: dict = defaultdict(list)
        for s in SEEDS:
            mpath = ddir / f"tclsc_s{s}.pt"
            torch.manual_seed(s)
            set_enc = TaskSetEncoder(dim_input=dim_input, nz=labels_cfg["nz"])
            set_enc.load_state_dict(torch.load(
                LABELS_DIR / "set_encoder_init.pt", map_location="cpu",
                weights_only=True))
            model = TCLSC(cls_dim=cls_dim, set_encoder=set_enc,
                          embedding_dim=tcfg["embedding_dim"], fz=tcfg["fz"])
            engine = TCLSCEngine(model, task_windows, device=args.device,
                                 lr=tcfg["lr"], batch_size=tcfg["batch_size"],
                                 seed=s)
            if args.resume and mpath.exists():
                model.load_state_dict(torch.load(mpath, map_location=args.device,
                                                 weights_only=True))
                print(f"[dir_{dirn} s{s}] 复用已训模型 {mpath.name}", flush=True)
            else:
                t0 = time.time()
                engine.train(train_records, epochs=tcfg["epochs"],
                             curriculum_epochs=tcfg["curriculum_epochs"],
                             better_fn=lambda a, b: is_better(a, b, task_type),
                             delta=delta_by_task)
                torch.save(model.state_dict(), mpath)
                meta["seconds"][f"tclsc_{dirn}_s{s}"] = round(time.time() - t0, 1)
            model.eval()
            for t in tasks_te:
                cands, keys = candidates_for(t.task_id, shared_cat, rand_cat)
                assert set(keys) == set(truth[t.task_id]), \
                    f"{t.task_id} 目录键 ≠ 网格键"
                vecs = torch.as_tensor(
                    np.stack([encoder.encode(c) for c in cands]),
                    dtype=torch.float32, device=args.device)
                with torch.no_grad():
                    e_row = model.encode_tasks(
                        [engine.task_windows[t.task_id]])[0]
                S_by_task[t.task_id].append(
                    symmetrize(_zs.pair_prob_matrix(model, vecs, e_row)))

        # ④ 多数票排名 + shared-10 Borda 冠军（协议 §2）
        dir_tasks = {}
        wins_sh_list, tb_sh_list = [], []
        shared_keys = [k for _c, k in shared_cat]
        for t in tasks_te:
            tid = t.task_id
            _cands, keys = candidates_for(tid, shared_cat, rand_cat)
            S_list = S_by_task[tid]
            V = pair_vote_matrix(S_list)
            mean_S = np.nanmean(np.stack(S_list), axis=0)
            tb = np.nanmean(mean_S, axis=1)
            order = order_candidates(V, tb, keys)
            ranked = [keys[i] for i in order]
            dir_tasks[tid] = {"ranked": ranked, "top1": ranked[0],
                              "top3": ranked[:K_TOP]}
            sh = np.arange(N_SHARED)      # 候选下标 0..9 恒为 shared（装配次序）
            wins_sh_list.append(V[np.ix_(sh, sh)].sum(axis=1))
            tb_sh_list.append(np.nanmean(mean_S[np.ix_(sh, sh)], axis=1))
            np.savez_compressed(ddir / f"scores_{tid}.npz",
                                S=np.stack(S_list), keys=np.array(keys))
        champ_idx = borda_ranking(wins_sh_list,
                                  np.mean(np.stack(tb_sh_list), axis=0),
                                  shared_keys)[0]
        rankings[f"dir_{dirn}"] = {"train": tr_name, "test": te_name,
                                   "champion": shared_keys[champ_idx],
                                   "tasks": dir_tasks}
        print(f"[dir_{dirn}] 冠军(shared Borda) = {shared_keys[champ_idx][:8]}",
              flush=True)

    # ⑤ 确认工单（协议 §5）：28 任务 {top1, oracle, GGS} 去重
    ggs_k = cls_key(_m1.GGS)
    top1 = {tid: e["top1"] for d in rankings.values() for tid, e in d["tasks"].items()}
    oracle = {tid: max(truth_r(truth, tid), key=lambda k: truth_r(truth, tid)[k])
              for tid in top1}
    wl = confirm_worklist(top1, oracle, ggs_k)
    (outdir / "rankings.json").write_text(
        json.dumps(rankings, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "confirm_worklist.json").write_text(
        json.dumps({"ggs_key": ggs_k, "cells": wl}, indent=2), encoding="utf-8")
    (outdir / "prepare_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"prepare 收官：排名 28 任务 / 确认工单 {len(wl)} 格 → {outdir}", flush=True)


# ---------------------------------------------------------------- --run-cells
def cmd_run_cells(args) -> None:
    outdir = Path(args.outdir)
    labels_cfg, train_cfg, _ = load_cfgs()
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    evr = labels_cfg["eval_ratio"]
    bs, lr = train_cfg["batch_size"], train_cfg["lr"]
    per = load_datasets()
    shared_cat, rand_cat = load_catalog()
    wl_doc = json.loads((outdir / "confirm_worklist.json").read_text(encoding="utf-8"))
    ggs_k = wl_doc["ggs_key"]
    key2cls = {k: c for c, k in shared_cat}
    for lst in rand_cat.values():
        key2cls.update({k: c for c, k in lst})
    key2cls[ggs_k] = _m1.GGS
    task_of = {t.task_id: (name, X, y, t)
               for name, (X, y, tasks) in per.items() for t in tasks}
    inflight, oom_log = outdir / "inflight.txt", outdir / "oom_failures.jsonl"

    # ① GGS 28 格 @工作点 canonical（rep=0）→ rprime_extra.jsonl
    extra_path = outdir / "rprime_extra.jsonl"
    done = set(load_rprime_cache(extra_path))
    todo = [tid for tid in sorted(task_of)
            if (tid, ggs_k, iters_map[ds_of(tid)]) not in done]
    print(f"GGS 格：28 计划，待跑 {len(todo)}", flush=True)
    with parallel_backend("threading", n_jobs=1):
        for i, tid in enumerate(todo):
            name, X, y, t = task_of[tid]
            it = iters_map[name]

            def mk(rec, _t=t, _n=name):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "tag": _t.tag,
                        "task_type": _t.task_type, "classes": list(_t.classes),
                        "cls_key": ggs_k, "origin": "ggs", "iters": it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            guarded_run(lambda: evaluate_rprime(
                            t, _m1.GGS, X, y,
                            weight_seed=run_seed_for(tid, ggs_k),
                            split_seed=run_seed_for(tid, "train_eval_split"),
                            iters=it, eval_ratio=evr, batch_size=bs, lr=lr,
                            device=args.device),
                        mk, cache_path=extra_path, inflight=inflight,
                        task_id=tid, cls_key=ggs_k, iters=it,
                        on_oom=ON_OOM_SKIP, oom_log=oom_log,
                        context={"dataset": name},
                        log=lambda m: print(f"[ggs {i + 1}/{len(todo)}] {m}",
                                            flush=True))

    # ② 确认格：工单 ×2 额外 weight seed → confirm_cache.jsonl（rep 入行）
    conf_path = outdir / "confirm_cache.jsonl"
    done_c = set()
    if conf_path.exists():
        for line in conf_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_c.add((r["task_id"], r["cls_key"], r["rep"]))
    work = [(tid, key, rep) for tid, key in wl_doc["cells"]
            for rep in CONFIRM_REPS if (tid, key, rep) not in done_c]
    print(f"确认格：工单 {len(wl_doc['cells'])}×{len(CONFIRM_REPS)}，"
          f"待跑 {len(work)}", flush=True)
    with parallel_backend("threading", n_jobs=1):
        for i, (tid, key, rep) in enumerate(work):
            name, X, y, t = task_of[tid]
            it = iters_map[name]

            def mk(rec, _t=t, _n=name, _k=key, _rep=rep):
                rp = rec["r_prime"]
                return {"dataset": _n, "task_id": _t.task_id, "cls_key": _k,
                        "rep": _rep, "iters": it, **rec,
                        "r_prime": None if not np.isfinite(rp) else float(rp)}

            guarded_run(lambda: evaluate_rprime(
                            t, key2cls[key], X, y,
                            weight_seed=run_seed_for(tid, key, base=rep),
                            split_seed=run_seed_for(tid, "train_eval_split"),
                            iters=it, eval_ratio=evr, batch_size=bs, lr=lr,
                            device=args.device),
                        mk, cache_path=conf_path, inflight=inflight,
                        task_id=tid, cls_key=key, iters=it,
                        on_oom=ON_OOM_SKIP, oom_log=oom_log,
                        context={"dataset": name, "rep": rep},
                        log=lambda m: print(f"[conf {i + 1}/{len(work)}] {m}",
                                            flush=True))
    print("run-cells 收官", flush=True)


# ---------------------------------------------------------------- --anchor
def cmd_anchor(args) -> None:
    outdir = Path(args.outdir)
    _labels_cfg, train_cfg, _ = load_cfgs()
    rankings = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    shared_cat, _rand = load_catalog()
    key2cls = {k: c for c, k in shared_cat}
    champ_key = rankings["dir_A"]["champion"]
    arms = {"champion": (champ_key, key2cls[champ_key]),
            "ggs": (cls_key(_m1.GGS), _m1.GGS)}
    Xtr, ytr = load_uea_cached("Handwriting", "TRAIN")
    Xte, yte = load_uea_cached("Handwriting", "TEST")
    print(f"全量 HW 官方 split: train={Xtr.shape} test={Xte.shape} | "
          f"champion={champ_key[:8]}", flush=True)
    path = outdir / "anchor_cache.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["arm"], r["seed"]))
    dev_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    with parallel_backend("threading", n_jobs=1):
        for arm, (key, cls) in arms.items():
            for s in ANCHOR_SEEDS:
                if (arm, s) in done:
                    print(f"[anchor {arm} s{s}] 已有，跳过", flush=True)
                    continue
                torch.manual_seed(s)
                enc = build_encoder(input_dims=Xtr.shape[-1])
                t0 = time.time()
                pretrain_encoder(enc, cls, Xtr,
                                 max_iters=train_cfg["full_iters"],
                                 batch_size=train_cfg["batch_size"],
                                 lr=train_cfg["lr"], seed=s)
                t1 = time.time()
                res = evaluate_classification(enc, Xtr, ytr, Xte, yte)
                append_jsonl(path, {
                    "arm": arm, "cls_key": key, "seed": s,
                    "iters": train_cfg["full_iters"], **res,
                    "pretrain_seconds": round(t1 - t0, 1),
                    "eval_seconds": round(time.time() - t1, 1),
                    "device_name": dev_name})
                print(f"[anchor {arm} s{s}] acc={res['acc']:.4f} "
                      f"({t1 - t0:.0f}s)", flush=True)
    print("anchor 收官", flush=True)


# ---------------------------------------------------------------- --analyze
def cmd_analyze(args) -> None:
    outdir = Path(args.outdir)
    labels_cfg, train_cfg, _ = load_cfgs()
    iters_map = labels_cfg["workpoint"]["iters_per_dataset"]
    truth = load_truth(iters_map)
    rankings = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    ggs_k = json.loads((outdir / "confirm_worklist.json")
                       .read_text(encoding="utf-8"))["ggs_key"]
    extra = load_rprime_cache(outdir / "rprime_extra.jsonl")

    per_task, arrays = {}, defaultdict(lambda: defaultdict(list))
    for dirn in ("A", "B"):
        dd = rankings[f"dir_{dirn}"]
        for tid, e in sorted(dd["tasks"].items()):
            tr = truth_r(truth, tid)
            grec = extra.get((tid, ggs_k, iters_map[ds_of(tid)]))
            assert grec and grec.get("r_prime") is not None, f"缺 GGS 格 {tid}"
            arms = task_arms(tr, e["ranked"], float(grec["r_prime"]))
            reg = regret_of(arms)
            per_task[tid] = {"direction": dirn, "arms": arms, "regret": reg,
                             "bottom_quartile_hit":
                                 bottom_quartile_hit(tr, e["top1"])}
            for a, v in reg.items():
                arrays[dirn][a].append(v)
                arrays["pooled"][a].append(v)

    def wilcox(scope: str) -> float:
        x = np.array(arrays[scope]["tclsc1"])
        y = np.array(arrays[scope]["random1"])
        return float(_lo.paired_wilcoxon_less(x, y))

    d1m_p = wilcox("pooled")
    p_dir = {d: wilcox(d) for d in ("A", "B")}
    mean_reg = {s: {a: float(np.mean(v)) for a, v in arrays[s].items()}
                for s in arrays}
    n_bq = sum(1 for e in per_task.values() if e["bottom_quartile_hit"])

    # 确认格误差棒（描述性）
    conf_rows = []
    cp = outdir / "confirm_cache.jsonl"
    if cp.exists():
        conf_rows = [json.loads(x) for x in
                     cp.read_text(encoding="utf-8").splitlines() if x.strip()]
    conf = defaultdict(list)
    for r in conf_rows:
        if r.get("r_prime") is not None:
            conf[(r["task_id"], r["cls_key"])].append(float(r["r_prime"]))
    conf_spread = []
    for (tid, k), v in conf.items():
        base = truth[tid].get(k) or extra.get((tid, k, iters_map[ds_of(tid)]))
        allv = list(v)
        if base and base.get("r_prime") is not None:
            allv.append(float(base["r_prime"]))
        if len(allv) >= 2:
            conf_spread.append(max(allv) - min(allv))

    # 外部锚
    anchor = None
    ap_ = outdir / "anchor_cache.jsonl"
    if ap_.exists():
        rows = [json.loads(x) for x in
                ap_.read_text(encoding="utf-8").splitlines() if x.strip()]
        anchor = {}
        for arm in ("champion", "ggs"):
            accs = [r["acc"] for r in rows if r["arm"] == arm]
            if accs:
                anchor[arm] = {"accs": accs, "mean": float(np.mean(accs)),
                               "n": len(accs)}
        anchor["paper_hw"] = TS2VEC_PAPER_HW
        anchor["line"] = TS2VEC_PAPER_HW - ANCHOR_MARGIN
        if "champion" in anchor:
            anchor["pass_line"] = bool(anchor["champion"]["mean"]
                                       >= anchor["line"])

    # 成本表（硬件标签在缓存行内；full 口径外推按 iters 比例声明）
    cost = {}
    for dirn in ("A", "B"):
        dd = rankings[f"dir_{dirn}"]
        on3, on20 = [], []
        for tid, e in dd["tasks"].items():
            secs = {k: float(rec.get("seconds", np.nan))
                    for k, rec in truth[tid].items()}
            on3.append(sum(secs[k] for k in e["top3"]))
            on20.append(sum(secs.values()))
        f = train_cfg["full_iters"] / iters_map[dd["test"]]
        cost[dirn] = {"zero_shot_top3_s": float(np.sum(on3)),
                      "catalog_direct_s": float(np.sum(on20)),
                      "full_direct_est_s": float(np.sum(on20) * f),
                      "full_iters_factor": f}

    preds = {
        "P1m": {"prior": 0.65, "hit": bool(d1m_p < 0.05), "p": d1m_p},
        "P2m": {"prior": 0.45, "hit": bool(p_dir["A"] < 0.05), "p": p_dir["A"]},
        "P3m": {"prior": 0.55, "hit": bool(p_dir["B"] < 0.05), "p": p_dir["B"]},
        "P4m": {"prior": 0.40, "hit": bool(mean_reg["pooled"]["tclsc3"]
                                           <= mean_reg["pooled"]["ggs"])},
        "P5m": {"prior": 0.55,
                "hit": (anchor or {}).get("pass_line")},
        "P6m": {"prior": 0.60, "hit": bool(n_bq == 0), "n_hits": n_bq},
    }
    out = {"protocol": "results/mx2_endtoend/PROTOCOL.md",
           "d1m": {"p": d1m_p, "pass": bool(d1m_p < 0.05)},
           "per_direction_p": p_dir, "mean_regret": mean_reg,
           "n_tasks": {d: len(arrays[d]["tclsc1"]) for d in ("A", "B", "pooled")},
           "bottom_quartile_hits": n_bq,
           "confirm_spread_max": (float(np.max(conf_spread))
                                  if conf_spread else None),
           "anchor": anchor, "cost": cost,
           "predictions": preds, "per_task": per_task}
    (outdir / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    print("\n===== MX-2 analyze =====")
    for s in ("A", "B", "pooled"):
        m = mean_reg[s]
        print(f"[{s}] n={len(arrays[s]['tclsc1'])} mean regret: "
              f"T-CLSC@1 {m['tclsc1']:.4f} | @3 {m['tclsc3']:.4f} | "
              f"GGS {m['ggs']:.4f} | rand@1 {m['random1']:.4f} | "
              f"rand@3 {m['random3']:.4f} | bottom {m['bottom']:.4f}")
    print(f"D1m（pooled T-CLSC@1 < rand@1）p={d1m_p:.4f} → "
          f"{'过' if d1m_p < 0.05 else '不过'} | 方向 p A={p_dir['A']:.4f} "
          f"B={p_dir['B']:.4f} | 垫底命中 {n_bq}/28")
    if anchor and "champion" in anchor:
        print(f"外部锚：champion {anchor['champion']['mean']:.4f} "
              f"(n={anchor['champion']['n']}) vs 论文 {TS2VEC_PAPER_HW} "
              f"− {ANCHOR_MARGIN} → {'达线' if anchor['pass_line'] else '未达线'}"
              + (f" | GGS {anchor['ggs']['mean']:.4f}" if 'ggs' in anchor else ""))
    for k, v in preds.items():
        print(f"  {k} 先验 {v['prior']:.0%} → "
              f"{'✓' if v['hit'] else ('?' if v['hit'] is None else '✗')}")
    print(f"→ {outdir / 'analysis.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run-cells", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT),
                    help="冒烟务必改到别处（铁律 3）")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if torch.get_num_threads() > 1 and (args.run_cells or args.prepare):
        print("⚠️ 建议 OMP_NUM_THREADS=1 启动（CPU-bound 教训）", flush=True)
    if args.prepare:
        cmd_prepare(args)
    if args.run_cells:
        cmd_run_cells(args)
    if args.anchor:
        cmd_anchor(args)
    if args.analyze:
        cmd_analyze(args)
    if not (args.prepare or args.run_cells or args.anchor or args.analyze):
        ap.error("至少指定 --prepare / --run-cells / --anchor / --analyze")


if __name__ == "__main__":
    main()
