"""下游评估协议探针：测"换下游评估协议"能否解决贴顶（协议锁
`results/protocol_probe/PROTOCOL.md`，判据/阈值/预测先于数据写死，commit 53461a0）。

上游结论（results/baseline_probe/REPORT.md）：贴顶是源池主流（16 个里 5 个基线
>0.90），加难类别口径已被证伪，因果链落在「这批 UEA 对『随机卷积特征 + max-pool +
RBF SVM』这个下游协议太容易」。本探针只换协议，其余口径逐项不变。

⚠️ 协议 A（RBF SVM）**就是** AutoCL 附录 B.1 的实现（原文 "the SVM classifier with
RBF kernel"）；B/B′/C 是主动偏离论文，代价见 PROTOCOL §1.2。

执行结构（先便宜后贵）：
    1. --dump --gate 1     685 条 iters=0 特征（≈3 min，不训练）
    2. --evaluate          离线跑 A/B/B′/C×3 全部变体（≈2–5 min，无 GPU）
    3. --analyze --gate 1  判 G1；不过就停，总花费 ≈10 min
    4. --calibrate         3 条 @290 iters 标定倍率 + 协议 A 复算比对阶段1 缓存
    5. --dump --gate 2a    60 条 @290 iters（AWR/NATOPS/Handwriting）
    6. --evaluate --analyze --gate 2a   判 G2/G3 → §4.4 判定表

铁律 3：特征逐条落盘 + --resume；铁律 6：目标池全程不加载，脚本内断言 fail-fast。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):   # 本机控制台默认 GBK，报表含 ≥/±/↑
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import (  # noqa: E402
    baseline_seed_specs,
    classify_tier,
    inert_cls,
)
from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET,
    TargetPoolLeakageError,
    load_registry,
    load_source_dataset,
    slice_source_tasks,
)
from autocls.downstream.protocols import PROTOCOL_NAMES, evaluate_all_protocols  # noqa: E402
from autocls.labeling import append_jsonl, cls_key, load_probe_cls, run_seed_for  # noqa: E402
from autocls.oom_guard import RprimeOOM  # noqa: E402
from autocls.protocol_probe import (  # noqa: E402
    G4_BEST_GAIN_FLOOR,
    GATE1_ITERS,
    GATE2_ITERS,
    GATE2_N_CLS,
    GATE2_N_TASKS,
    GATE2A_DATASETS,
    GATE2B_DATASETS,
    N_BASELINE_SEEDS,
    compute_task_features,
    gate1_scope,
    gate1_verdict,
    gate2_verdict,
    gate3_verdict,
    overall_verdict,
)

OUTDIR = ROOT / "results" / "protocol_probe"
FEATDIR = OUTDIR / "feats"
CACHE = OUTDIR / "protocol_cache.jsonl"
OOM_LOG = OUTDIR / "oom_failures.jsonl"
PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
STAGE1_CACHE = (ROOT / "results" / "stage1_rerun_20260804" / "results" /
                "labels_route_a" / "rprime_cache.jsonl")

GATE1, GATE2A, GATE2B = "1", "2a", "2b"


# ---------------------------------------------------------------------------
# 公共
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue   # 被杀留下的半行
    return out


def leak_check(registry, names: list[str]) -> None:
    """铁律 6：目标池绝不进探针范围，且走源加载路径必须 fail-fast。"""
    targets = sorted(registry.names(ROLE_TARGET))
    bad = set(names) & set(targets)
    assert not bad, f"目标池混入探针范围 {sorted(bad)}（铁律 6）"
    for name in targets:
        try:
            load_source_dataset(name, registry)
        except (TargetPoolLeakageError, ValueError):
            continue
        raise AssertionError(f"目标池 {name} 未被源加载路径拒绝（铁律 6 防护失效）")
    print(f"泄漏自检：本次 {len(names)} 个 ∩ 目标池 {targets} = ∅，"
          f"且目标池走源加载路径全部 fail-fast（铁律 6）", flush=True)


def sliced_tasks(registry, names: list[str], max_tasks: int | None = None):
    """与阶段1 / 基线探针同一切片入口、注册表默认档（**不用 +c 加难档**）。"""
    out = []
    for name in names:
        entry = registry.get(name)
        X, y = load_source_dataset(name, registry)
        _, tasks = slice_source_tasks(entry, registry, y)
        acts = tasks[:max_tasks] if max_tasks else tasks
        print(f"[{name}] X={X.shape} n_classes={entry.n_classes} → {len(acts)} tasks",
              flush=True)
        out.append((name, entry, X, y, acts))
    return out


def feat_path(task_id: str, key: str, iters: int, weight_seed: int) -> Path:
    return FEATDIR / f"it{iters}" / f"{task_id}__{key}__w{weight_seed}.npz"


def gate_names(gate: str, registry) -> list[str]:
    if gate == GATE1:
        return gate1_scope(registry)
    if gate == GATE2A:
        return list(GATE2A_DATASETS)
    return list(GATE2B_DATASETS)


def _work_items(gate: str, per_ds):
    """→ [(dataset, X, y, task, cls_cfg, cls_key, weight_seed, split_seed, tag)]。

    闸门1：iters=0，CLS 完全不参与（`pretrain_encoder` 的 while 不进循环），故用
      inert_cls 占位 + 5 个随机初始化种子，split 恒为 canonical（与阶段1 同划分）。
    闸门2：iters=290，前 2 任务 × 前 10 shared CLS，weight/split 种子派生与阶段1
      rprime_cache 逐字段一致 → 协议 A 复算应命中缓存值（--calibrate 校验）。
    """
    if gate == GATE1:
        c, k = inert_cls(), cls_key(inert_cls())
        return [(n, X, y, t, c, k, *baseline_seed_specs(t.task_id, s), f"s{s}")
                for n, _e, X, y, tasks in per_ds for t in tasks
                for s in range(N_BASELINE_SEEDS)]
    cls_list, _ = load_probe_cls(PROBE_CATALOG, GATE2_N_CLS)
    return [(n, X, y, t, c, k, run_seed_for(t.task_id, k),
             run_seed_for(t.task_id, "train_eval_split"), "")
            for n, _e, X, y, tasks in per_ds for t in tasks for c, k in cls_list]


# ---------------------------------------------------------------------------
# 步骤 1：dump 特征（唯一需要 GPU / 训练的一步）
# ---------------------------------------------------------------------------

def dump(gate: str, names: list[str], *, device: str, resume: bool,
         max_tasks: int | None, limit: int | None) -> None:
    if torch.get_num_threads() > 1:
        print(f"⚠️ torch.get_num_threads()={torch.get_num_threads()}>1：本负载 CPU-bound，"
              f"请用 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 启动", flush=True)
    iters = GATE1_ITERS if gate == GATE1 else GATE2_ITERS
    FEATDIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    leak_check(registry, names)

    n_tasks = max_tasks if max_tasks is not None else (
        None if gate == GATE1 else GATE2_N_TASKS)
    work = _work_items(gate, sliced_tasks(registry, names, n_tasks))
    todo = [w for w in work
            if not (resume and feat_path(w[3].task_id, w[5], iters, w[6]).exists())]
    n_cached = len(work) - len(todo)      # --limit 截断前算，否则把"被截掉"算成"已完成"
    if limit:
        todo = todo[:limit]
    print(f"\n闸门 {gate} | 设备 {device} | iters={iters}"
          f"{'（不训练，CLS 不参与）' if iters == 0 else ''} | "
          f"计划 {len(work)} 条，已 dump {n_cached}，待跑 {len(todo)}"
          + (f"（--limit 截掉 {len(work) - n_cached - len(todo)}）" if limit else "") + "\n",
          flush=True)

    # threading 后端：Windows 上 GridSearchCV 的 loky 进程池每次 spawn 8–15 s（占单条
    # ~99%），threading 逐位同值（tests/test_baseline_probe.py 有回归断言）。dump 阶段
    # 本身不跑分类器，但 encode/训练里的 BLAS 也走这条设置，保持与基线探针一致。
    t0, n_oom, n_err = time.time(), 0, 0
    with parallel_backend("threading", n_jobs=1):
        for i, (name, X, y, t, c, k, w_seed, sp_seed, tag) in enumerate(todo):
            try:
                b = compute_task_features(t, c, X, y, weight_seed=w_seed,
                                          split_seed=sp_seed, iters=iters, device=device)
            except RprimeOOM as e:
                n_oom += 1
                append_jsonl(OOM_LOG, {"dataset": name, "task_id": t.task_id,
                                       "cls_key": k, "iters": iters,
                                       "weight_seed": w_seed, "error": str(e)})
                print(f"[{i + 1}/{len(todo)}] {name}/{t.task_id} OOM，跳过（不落盘结论）",
                      flush=True)
                continue
            if b["error"] is not None:
                n_err += 1
                print(f"[{i + 1}/{len(todo)}] {name}/{t.task_id} 失败: {b['error']}",
                      flush=True)
                continue
            p = feat_path(t.task_id, k, iters, w_seed)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                p, train_feats=b["train_feats"], eval_feats=b["eval_feats"],
                y_train=b["y_train"], y_eval=b["y_eval"],
                meta=json.dumps({
                    "dataset": name, "task_id": t.task_id, "tag": t.tag,
                    "task_type": t.task_type, "classes": list(t.classes),
                    "n_classes": len(t.classes), "cls_key": k, "iters": iters,
                    "seed_tag": tag, "weight_seed": w_seed, "split_seed": sp_seed,
                    "n_train": b["n_train"], "n_eval": b["n_eval"],
                    "feat_dim": b["feat_dim"], "seconds": b["seconds"],
                    "cls": dataclasses.asdict(c)}, ensure_ascii=False))
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            print(f"[{i + 1}/{len(todo)}] {name}/{t.task_id} {k[:8]} {tag} "
                  f"({b['seconds']}s, n_tr={b['n_train']} n_ev={b['n_eval']}) "
                  f"ETA {eta / 60:.1f}min", flush=True)
    print(f"\ndump 完成 {len(todo) - n_oom - n_err}/{len(todo)} 条，"
          f"墙钟 {(time.time() - t0) / 60:.1f} min"
          + (f"，OOM {n_oom}" if n_oom else "") + (f"，失败 {n_err}" if n_err else ""),
          flush=True)


# ---------------------------------------------------------------------------
# 步骤 2：离线多协议评估（无 GPU、无训练）
# ---------------------------------------------------------------------------

def evaluate(gate: str, *, resume: bool, protocols: list[str]) -> None:
    iters = GATE1_ITERS if gate == GATE1 else GATE2_ITERS
    done = {(r["task_id"], r["cls_key"], r["weight_seed"], r["protocol"])
            for r in _read_jsonl(CACHE) if r["iters"] == iters}
    files = sorted((FEATDIR / f"it{iters}").glob("*.npz"))
    print(f"\n闸门 {gate} 离线评估：{len(files)} 份特征 × {len(protocols)} 协议 "
          f"= {len(files) * len(protocols)} 条，已缓存 {len(done)}\n", flush=True)

    t0, n = time.time(), 0
    with parallel_backend("threading", n_jobs=1):
        for i, f in enumerate(files):
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            todo = [p for p in protocols
                    if not (resume and (meta["task_id"], meta["cls_key"],
                                        meta["weight_seed"], p) in done)]
            if not todo:
                continue
            out = evaluate_all_protocols(
                z["train_feats"], z["y_train"], z["eval_feats"], z["y_eval"],
                seed=meta["weight_seed"], names=todo)
            for p, m in out.items():
                append_jsonl(CACHE, {**{kk: meta[kk] for kk in (
                    "dataset", "task_id", "tag", "classes", "n_classes", "cls_key",
                    "iters", "seed_tag", "weight_seed", "split_seed",
                    "n_train", "n_eval")}, **m})
                n += 1
            eta = (time.time() - t0) / (i + 1) * (len(files) - i - 1)
            print(f"[{i + 1}/{len(files)}] {meta['task_id']} {meta['cls_key'][:8]} "
                  + "  ".join(f"{p}={out[p]['r_prime']:.3f}" for p in todo)
                  + f"  ETA {eta / 60:.1f}min", flush=True)
    print(f"\n评估完成 +{n} 条，墙钟 {(time.time() - t0) / 60:.1f} min → {CACHE}",
          flush=True)


# ---------------------------------------------------------------------------
# 步骤 3：倍率标定 + 协议 A 复算比对（执行闸门2 前的口径校验，PROTOCOL §6.3）
# ---------------------------------------------------------------------------

def calibrate(*, device: str) -> None:
    """每个 2a 数据集跑 1 条 @290 iters：① 协议 A 复算 vs 阶段1 缓存 ② 真实倍率。

    判据：差值 ≤ 1 个 eval 样本（1/n_eval）→ 口径无错配；更大 → 停机排查。
    （5090 上 R' 有 ±1 样本抖动，跨机不保证逐位；本机 4060 内部是逐位可复现的。）
    """
    registry = load_registry()
    names = list(GATE2A_DATASETS)
    leak_check(registry, names)
    per_ds = sliced_tasks(registry, names, 1)
    cls_list, _ = load_probe_cls(PROBE_CATALOG, 1)
    stage1 = {(r["task_id"], r["cls_key"]): r for r in _read_jsonl(STAGE1_CACHE)}

    print(f"\n倍率标定 + 协议 A 复算比对（{len(per_ds)} 条 @{GATE2_ITERS} iters）\n",
          flush=True)
    rows, ok = [], True
    with parallel_backend("threading", n_jobs=1):
        for name, _e, X, y, tasks in per_ds:
            t = tasks[0]
            c, k = cls_list[0]
            w_seed = run_seed_for(t.task_id, k)
            sp_seed = run_seed_for(t.task_id, "train_eval_split")
            t0 = time.time()
            b = compute_task_features(t, c, X, y, weight_seed=w_seed,
                                      split_seed=sp_seed, iters=GATE2_ITERS,
                                      device=device)
            if b["error"]:
                print(f"  {name} 失败: {b['error']}", flush=True)
                ok = False
                continue
            got = evaluate_all_protocols(b["train_feats"], b["y_train"],
                                         b["eval_feats"], b["y_eval"],
                                         seed=w_seed, names=["A"])["A"]
            ref = stage1.get((t.task_id, k))
            quantum = 1.0 / b["n_eval"]
            d = None if ref is None else got["r_prime"] - ref["r_prime"]
            hit = d is not None and abs(d) <= quantum + 1e-12
            ok = ok and hit
            local_s = round(time.time() - t0, 1)
            rows.append({"dataset": name, "task_id": t.task_id, "cls_key": k,
                         "local_seconds": local_s,
                         "stage1_seconds": None if ref is None else ref["seconds"],
                         "protocol_a": got["r_prime"],
                         "stage1_r_prime": None if ref is None else ref["r_prime"],
                         "delta": d, "acc_quantum": quantum, "within_1_sample": hit})
            ref_txt = ("stage1 无此记录" if ref is None else
                       f"stage1={ref['r_prime']:.4f} Δ={d:+.4f} "
                       f"(量子 {quantum:.4f}) {'✅' if hit else '❌'} | "
                       f"本机 {local_s}s vs 5090 {ref['seconds']}s "
                       f"= {local_s / ref['seconds']:.2f}×")
            print(f"  {name:28s} A={got['r_prime']:.4f}  {ref_txt}", flush=True)
    ratios = [r["local_seconds"] / r["stage1_seconds"] for r in rows
              if r["stage1_seconds"]]
    out = {"rows": rows, "all_within_1_sample": ok,
           "ratio_local_over_5090": ratios,
           "ratio_median": float(np.median(ratios)) if ratios else None}
    (OUTDIR / "calibration.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n倍率中位数 {out['ratio_median']}× | 口径校验 "
          f"{'全部通过' if ok else '❌ 有超出 1 个样本的差异，停机排查'}"
          f" → {OUTDIR / 'calibration.json'}", flush=True)


# ---------------------------------------------------------------------------
# 步骤 4：分析与判定
# ---------------------------------------------------------------------------

def _by_protocol(rows: list[dict], iters: int) -> dict[str, dict[str, list[dict]]]:
    """→ {protocol: {task_id: [记录]}}，只取指定 iters 口径（铁律 2）。"""
    out: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        if r["iters"] != iters:
            continue
        out.setdefault(r["protocol"], {}).setdefault(r["task_id"], []).append(r)
    return out


def _baseline_stats(recs: list[dict]) -> dict:
    """一个 (task, protocol) 的 5 个随机初始化 → 均值 / ±2σ 零假设带。

    σ 用 ACC 量子 1/n_eval 兜底（种子同分时 σ=0 会让"超出 2σ"退化成"超出均值"）
    ——与基线探针 summarize_task 同规则，保证两份报告的诊断可比。
    """
    v = [r["r_prime"] for r in recs]
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    quantum = max((1.0 / r["n_eval"] for r in recs if r.get("n_eval")), default=0.0)
    return {"n_seeds": len(v), "mean": mean, "std": std,
            "band": 2.0 * max(std, quantum), "n_eval": recs[0].get("n_eval")}


def analyze(gate: str) -> None:
    rows = _read_jsonl(CACHE)
    base = _by_protocol(rows, GATE1_ITERS)
    trained = _by_protocol(rows, GATE2_ITERS)
    if not base:
        sys.exit("没有 iters=0 的记录——先跑 --dump --gate 1 与 --evaluate")

    ds_of_task = {r["task_id"]: r["dataset"] for r in rows}
    protocols = [p for p in PROTOCOL_NAMES if p in base]

    # ---- 闸门1：逐协议的数据集级基线 ----
    per_proto: dict[str, dict] = {}
    for p in protocols:
        task_mean: dict[str, dict] = {t: _baseline_stats(rs) for t, rs in base[p].items()}
        ds: dict[str, list[float]] = {}
        for t, st in task_mean.items():
            ds.setdefault(ds_of_task[t], []).append(st["mean"])
        per_proto[p] = {
            "tasks": task_mean,
            "datasets": {d: {"baseline_mean": float(np.mean(v)), "n_tasks": len(v),
                             "task_min": float(min(v)), "task_max": float(max(v)),
                             "tier": classify_tier(float(np.mean(v)))}
                         for d, v in sorted(ds.items())},
        }
        per_proto[p]["n_ceiling"] = sum(
            1 for d in per_proto[p]["datasets"].values() if d["tier"] == "ceiling")

    print(f"\n{'=' * 120}\n闸门1：各协议的随机初始化基线（iters=0，"
          f"{N_BASELINE_SEEDS} 种子，canonical split）\n{'=' * 120}")
    all_ds = sorted({d for p in protocols for d in per_proto[p]["datasets"]})
    print(f"{'数据集':<27}{'任务':>4}" + "".join(f"{p:>10}" for p in protocols))
    for d in sorted(all_ds, key=lambda x: -per_proto[protocols[0]]["datasets"]
                    .get(x, {}).get("baseline_mean", 0)):
        n_t = per_proto[protocols[0]]["datasets"].get(d, {}).get("n_tasks", 0)
        cells = "".join(
            f"{per_proto[p]['datasets'][d]['baseline_mean']:>10.3f}"
            if d in per_proto[p]["datasets"] else f"{'—':>10}" for p in protocols)
        print(f"{d:<27}{n_t:>4}{cells}")
    print(f"\n{'贴顶数(>0.90)':<27}{'':>4}"
          + "".join(f"{per_proto[p]['n_ceiling']:>10}" for p in protocols))

    # ---- G1 ----
    awr = "ArticularyWordRecognition"
    g1 = None
    if "B" in per_proto and awr in per_proto["B"]["datasets"]:
        g1 = gate1_verdict(per_proto["B"]["datasets"][awr]["baseline_mean"])
        print(f"\n【G1 天花板】主候选 B 下 AWR 基线 = {g1['awr_baseline']:.4f} "
              f"（阈值 <{g1['threshold']}）→ {'✅ 通过' if g1['passed'] else '❌ 不过'}")
        for p in protocols:
            if p != "B" and awr in per_proto[p]["datasets"]:
                v = per_proto[p]["datasets"][awr]["baseline_mean"]
                print(f"       （{p} 下 AWR = {v:.4f}"
                      + ("，次要读数，不参与 G1 判定" if p != "A" else "，旧协议对照") + "）")

    # ---- 闸门2：零假设诊断 + G2/G3 ----
    g2 = g3 = None
    diag: dict[str, dict] = {}
    if trained:
        print(f"\n{'—' * 120}\n闸门2：训练后 R'（iters={GATE2_ITERS}）是否跑出随机初始化 ±2σ 带")
        for p in protocols:
            if p not in trained:
                continue
            per_ds: dict[str, dict] = {}
            for t, rs in trained[p].items():
                st = per_proto[p]["tasks"].get(t)
                if st is None:
                    continue
                d = ds_of_task[t]
                acc = per_ds.setdefault(d, {"above": 0, "below": 0, "total": 0,
                                            "best": None, "realized": []})
                for r in rs:
                    acc["total"] += 1
                    acc["above"] += int(r["r_prime"] > st["mean"] + st["band"])
                    acc["below"] += int(r["r_prime"] < st["mean"] - st["band"])
                    acc["realized"].append(r["r_prime"] - st["mean"])
                gain = max(r["r_prime"] for r in rs) - st["mean"]
                acc["best"] = gain if acc["best"] is None else max(acc["best"], gain)
            for d in per_ds:
                per_ds[d]["realized_mean"] = float(np.mean(per_ds[d].pop("realized")))
            diag[p] = per_ds
            print(f"\n  协议 {p}")
            print(f"    {'数据集':<27}{'超出↑':>7}{'跌破↓':>7}{'带内':>7}{'合计':>6}"
                  f"{'最好增益':>10}{'实得增益':>10}")
            for d, a in sorted(per_ds.items()):
                print(f"    {d:<27}{a['above']:>7}{a['below']:>7}"
                      f"{a['total'] - a['above'] - a['below']:>7}{a['total']:>6}"
                      f"{a['best']:>+10.3f}{a['realized_mean']:>+10.3f}")

        if "B" in diag:
            judged = {d: (a["above"], a["below"]) for d, a in diag["B"].items()
                      if d in (awr, "NATOPS")}
            if judged:
                g2 = gate2_verdict(judged)
                print(f"\n【G2 空间真被用上】协议 B，AWR+NATOPS 均需 ↑≥{g2['min_above']}/20 "
                      f"且 ↑≥↓ → {'✅ 通过' if g2['passed'] else '❌ 不过'}"
                      + (f"（不达标：{', '.join(g2['failed'])}）" if g2["failed"] else ""))
            hw = diag["B"].get("Handwriting")
            if hw:
                g3 = gate3_verdict(hw["above"], hw["best"])
                print(f"【G3 对照不被破坏】Handwriting ↑={g3['n_above']}/20 "
                      f"（需≥{g3['min_above']}），最好增益 {g3['best_gain']:+.3f} "
                      f"（需≥{g3['min_best_gain']}）→ "
                      f"{'✅ 通过' if g3['passed'] else '❌ 不过'}")

    verdict = None
    if g1 is not None and g2 is not None and g3 is not None:
        verdict = overall_verdict(g1["passed"], g2["passed"], g3["passed"])
        label = {"protocol_works": "协议有效 → 推广到 16 个 + 重标定 δ",
                 "false_fix": "假解决：天花板降了但 headroom 用不上",
                 "no_effect": "无效：这批 UEA 对任何合理下游协议都太容易",
                 "broke_control": "改坏了：造出假空间却毁了唯一可用的源"}[verdict]
        print(f"\n{'=' * 120}\n【§4.4 判定表】G1={'✅' if g1['passed'] else '❌'} "
              f"G2={'✅' if g2['passed'] else '❌'} G3={'✅' if g3['passed'] else '❌'} "
              f"→ **{verdict}** —— {label}\n{'=' * 120}")
    elif g1 is not None and not g1["passed"]:
        verdict = overall_verdict(False, False, False)
        print(f"\n{'=' * 120}\n【§4.4 判定表】G1 不过 → **{verdict}** —— "
              f"无效：这批 UEA 对任何合理下游协议都太容易。闸门2 不必跑。\n{'=' * 120}")

    out = {"gate": gate, "protocols": protocols,
           "criterion_doc": "results/protocol_probe/PROTOCOL.md",
           "baseline_by_protocol": per_proto, "trained_diagnostic": diag,
           "G1": g1, "G2": g2, "G3": g3, "verdict": verdict,
           "best_gain_floor": G4_BEST_GAIN_FLOOR}
    (OUTDIR / "analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n分析 → {OUTDIR / 'analysis.json'}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", action="store_true", help="跑训练 + dump max-pool 特征")
    ap.add_argument("--evaluate", action="store_true", help="离线跑全部协议变体")
    ap.add_argument("--calibrate", action="store_true",
                    help="3 条 @290 iters：协议 A 复算比对阶段1 缓存 + 标定本机倍率")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gate", choices=[GATE1, GATE2A, GATE2B], default=GATE1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOL_NAMES))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="只跑指定数据集（目标池绝不可选，铁律 6）")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="试跑：只跑前 N 条")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    registry = load_registry()
    names = args.datasets or gate_names(args.gate, registry)
    bad = set(names) & set(registry.names(ROLE_TARGET))
    if bad:
        sys.exit(f"{sorted(bad)} 是目标池数据集，绝不可测（铁律 6：zero-shot 边界）")
    for p in args.protocols:
        if p not in PROTOCOL_NAMES:
            sys.exit(f"协议 {p!r} 未登记（PROTOCOL.md §3 写死 {PROTOCOL_NAMES}）")

    if args.calibrate:
        calibrate(device=args.device)
    if args.dump:
        dump(args.gate, names, device=args.device, resume=args.resume,
             max_tasks=args.max_tasks, limit=args.limit)
    if args.evaluate:
        evaluate(args.gate, resume=args.resume, protocols=args.protocols)
    if args.analyze:
        analyze(args.gate)
    if not (args.dump or args.evaluate or args.calibrate or args.analyze):
        ap.error("至少指定 --dump / --evaluate / --calibrate / --analyze 之一")


if __name__ == "__main__":
    main()
