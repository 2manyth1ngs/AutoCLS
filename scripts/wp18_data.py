"""WP18 源侧标签装配：S2（17 源 160 任务）+ XS（4 源 32 任务）旧标签 ⊕ WP18 新标签（扩展轴）→ 每任务 (键, 向量, r, n_seeds, 失败)。

旧标签：cv5 终点 acc（shared 两种子均值、random 种子 0），view = raw / mult = 1 / readout = max（= 扩展空间默认轴）。
新标签：results/acts/wp18/src_runs/*.jsonl，每行一条轨迹，截点 × 读出给出多个 (扩展键 → acc)。
失败 = r ≤ 探针地板（S2 / XS 的 probe0 均值；无探针取多数类率）。HMD / SCP2（部署靶）不进训练池。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTS = ROOT / "results" / "acts"
S2 = ACTS / "s2"
XS = ACTS / "xs"
WP18 = ACTS / "wp18"
TARGET_EXCLUDE = ("HandMovementDirection", "SelfRegulationSCP2")


def _g(name):
    return importlib.import_module(f"scripts.{name}")


def _corpus(d: Path, cfg_profile: str = "s2"):
    ac = _g("acts_common"); g83 = _g("83_acts_corpus")
    plan = json.loads((d / "plan_S.json").read_text(encoding="utf-8"))
    cat = json.loads((d / "catalog.json").read_text(encoding="utf-8"))
    rows = g83._g35.load_done(d / "runs.jsonl")
    floors = g83.floors_cv(g83.load_probe0(d), plan["tasks_meta"])
    return plan, cat, rows, floors


def pool_tasks() -> tuple[list[str], dict]:
    """训练池：S2 pool（17 源，Ethanol 降 8 片）+ XS 4 源 32 任务；返回 (任务列表, tasks_meta)。"""
    g84 = _g("84_acts_s3a")
    plan, _cat, _rows, _fl = _corpus(S2)
    an = json.loads((S2 / "analysis_S.json").read_text(encoding="utf-8"))
    meta = dict(plan["tasks_meta"])
    tasks = g84.pool_tasks(plan["probe_tasks"], meta, an["pool_after_gates"], an["reverse_demoted_to_8"])
    xplan = json.loads((XS / "plan_S.json").read_text(encoding="utf-8"))
    sel = json.loads((XS / "selected.json").read_text(encoding="utf-8"))
    xsel = sel.get("selected", sel) if isinstance(sel, dict) else sel
    for t in xplan["probe_tasks"]:
        if xplan["tasks_meta"][t]["dataset"] in set(xsel):
            tasks.append(t); meta[t] = xplan["tasks_meta"][t]
    tasks = [t for t in tasks if meta[t]["dataset"] not in TARGET_EXCLUDE]
    return tasks, meta


def old_labels(tasks, xs_space) -> dict[str, dict]:
    """{task: {key: {"cls": dict, "accs": [..], "ext": {}}}} 旧语料（默认轴）。floor 一并给出。"""
    ac = _g("acts_common")
    out: dict[str, dict] = {}
    floors: dict[str, float] = {}
    for d in (S2, XS):
        plan, cat, rows, fl = _corpus(d)
        cls_of = {s["cls_key"]: s["cls"] for s in cat["shared"]}
        for tid, lst in cat["random"].items():
            for x in lst:
                cls_of[x["cls_key"]] = x["cls"]
        for c in plan["cells"]:
            t = c["task_id"]
            if t not in tasks or c["arm"] not in ("shared", "random"):
                continue
            if c["arm"] == "random" and int(c["seed"]) != 0:
                continue
            a = ac.endpoint_of(rows.get(c["key"]))
            if a is None:
                continue
            e = out.setdefault(t, {}).setdefault(c["cls_key"], {"cls": c["cls"], "accs": [], "ext": {}})
            e["accs"].append(float(a))
        for t in tasks:
            if t in fl and t not in floors:
                pa = fl[t]["probe_accs"]
                floors[t] = float(np.mean(pa)) if pa else float(plan["tasks_meta"][t]["majority"]) if t in plan["tasks_meta"] else np.nan
    return {"labels": out, "floors": floors}


def new_labels(tasks, run_dir: Path = WP18 / "src_runs") -> dict[str, dict]:
    """WP18 新轨迹：每行 {task_id, cls, ext_base, results:[{mult, readout, acc}]} → {task: {xkey: {...}}}（xkey 由调用方的 XSpace 算）。"""
    out: dict[str, list] = {}
    for p in sorted(Path(run_dir).glob("*.jsonl")) if Path(run_dir).exists() else []:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") is None and r.get("task_id") in set(tasks):
                out.setdefault(r["task_id"], []).append(r)
    return out


def assemble(xs_space, *, include_new: bool = True, axes_used: Optional[Mapping] = None) -> dict:
    """→ {"tasks", "groups", "X": [n_t×d], "r": [n_t], "ns": [n_t], "fail": [n_t], "keys": [[...]], "floor": [...]}。
    新轨迹的每个 (mult, readout) 截点都是一条标签（轴取值不在 xs_space 里的丢弃）。"""
    from autocls.space.cls_space import CLS
    tasks, meta = pool_tasks()
    old = old_labels(tasks, xs_space)
    lab, floors = old["labels"], old["floors"]
    new = new_labels(tasks) if include_new else {}
    T, G, X, R, NS, F, K, FL = [], [], [], [], [], [], [], []
    for t in tasks:
        ent: dict[str, dict] = {}
        for k, e in lab.get(t, {}).items():
            x = xs_space.make(CLS(**e["cls"]))
            ent[xs_space.key(x)] = {"x": x, "accs": list(e["accs"])}
        for r in new.get(t, []):
            for res in r["results"]:
                ext = dict(r["ext_base"]); ext.update({k: res[k] for k in ("mult", "readout") if k in res and k in xs_space.axes})
                if any(k not in xs_space.axes for k in ext) or any(v not in xs_space.axes[k]["choices"] for k, v in ext.items()):
                    continue
                x = xs_space.make(CLS(**r["cls"]), **ext)
                ent.setdefault(xs_space.key(x), {"x": x, "accs": []})["accs"].append(float(res["acc"]))
        if len(ent) < 5:
            continue
        keys = sorted(ent)
        T.append(t); G.append(meta[t]["dataset"]); K.append(keys)
        X.append(xs_space.encode_many([ent[k]["x"] for k in keys]))
        r = np.array([np.mean(ent[k]["accs"]) for k in keys], float)
        R.append(r); NS.append(np.array([len(ent[k]["accs"]) for k in keys], int))
        fl = floors.get(t, float(meta[t]["majority"]))
        FL.append(fl); F.append(r <= fl + 1e-12)
    return {"tasks": T, "groups": np.array(G), "X": X, "r": R, "ns": NS, "fail": F, "keys": K, "floor": np.array(FL), "meta": meta}
