"""UCR-ADM 事后（零 GPU）：按 S1″ PROTOCOL §1.8 + posthoc §2 的规则算余量 H——
诚实口径：任务 oracle 按 seed 0 在 shared-30 里选、seed 1 评；任务盲底按 seed 0 在**其他**源上（LODO）选、seed 1 评。
四个底：先验@1（全局冠军）/ 前 3 全量实测（前 3 按 seed 0 终点选一条）/ SH6-3-3 / SH10-4-3（起跑名单按先验，@25% 淘汰 → @50% 淘汰 → @100% 选一条，
全部用 seed 0 的截点 acc 选、seed 1 终点评）；任务加权 + 源等权；决策阈值 = S1″（H_same = 先验@1 任务加权：≥ 0.03 GO / < 0.02 B / 中间 MID）。
池：v3′ 11（复现 S1″ posthoc）/ v3′ 11 + UCR 10 = 21（主）/ 护栏 9 + UCR 10 = 19 / UCR 10。跨域：三组（uea / external / ucr）按组 LODO；UCR ↔ 非 UCR 两向。
池大小曲线：训练侧随机抽 k 个其他源选冠军。用法：python results/acts/ucradm/headroom_H.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
S1PP = ROOT / "results" / "acts" / "s1pp"
UCR = ROOT / "results" / "acts" / "ucradm"

UEA = ["Handwriting", "PhonemeSpectra", "UWaveGestureLibrary", "EthanolConcentration", "Cricket", "SelfRegulationSCP1"]
EXTERNAL = ["UniMiB-SHAR", "PTB-XL", "SleepEDF", "PAMAP2", "Opportunity"]
UCR10 = ["NonInvasiveFetalECGThorax1", "CinCECGTorso", "SemgHandMovementCh2", "EOGHorizontalSignal", "ShapesAll",
         "MixedShapesSmallTrain", "LargeKitchenAppliances", "Adiac", "FiftyWords", "FordA"]
GUARD_REJECT_S1PP = ["PhonemeSpectra", "EthanolConcentration"]        # UCR-ADM PROTOCOL §0：护栏在 S1″ 池上拒 PS / Ethanol（HMD / SCP2 已被 v3′ 拒）
DECISION = {"go_min": 0.03, "stop_max": 0.02}                          # S1″ acts.yaml profiles.s1pp.headroom


def load_rows(path: Path, keys: list[str]) -> dict[str, dict]:
    """task → {(key, seed): (acc@25, acc@50, acc@100)}；只取 shared / cv5 / 目录 30 键 / 种子 0,1 / `<src>_aNN` 任务。"""
    out: dict[str, dict] = defaultdict(dict)
    kset = set(keys)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("arm") != "shared" or r.get("eval") != "cv5" or r.get("error") or r["cls_key"] not in kset or int(r["seed"]) not in (0, 1):
            continue
        tid = r["task_id"]
        if "_whole" in tid or "_a" not in tid:
            continue
        cps = [c["acc"] for c in r["checkpoint_results"]]
        out[tid][(r["cls_key"], int(r["seed"]))] = (float(cps[0]), float(cps[1]), float(r["acc"]))
    return out


def matrices(rows: dict[str, dict], keys: list[str], tasks: list[str]) -> dict[str, np.ndarray]:
    """三个截点 × 两种子的 (task, cfg) 矩阵；单缺 → 另一种子；双缺 → 该任务该截点最小值（S1″ _frames_c 同规则）。"""
    M = {f"{cp}_{s}": np.full((len(tasks), len(keys)), np.nan) for cp in ("q25", "q50", "end") for s in (0, 1)}
    for i, tid in enumerate(tasks):
        d = rows.get(tid, {})
        for j, k in enumerate(keys):
            for s in (0, 1):
                v = d.get((k, s))
                if v is not None:
                    M[f"q25_{s}"][i, j], M[f"q50_{s}"][i, j], M[f"end_{s}"][i, j] = v
    for cp in ("q25", "q50", "end"):
        a0, a1 = M[f"{cp}_0"], M[f"{cp}_1"]
        both = np.isnan(a0) & np.isnan(a1)
        a0 = np.where(np.isnan(a0), a1, a0); a1 = np.where(np.isnan(a1), a0, a1)
        for i in range(len(tasks)):
            if both[i].any():
                fill = np.nanmin(np.concatenate([a0[i], a1[i]]))
                a0[i, both[i]] = fill; a1[i, both[i]] = fill
        M[f"{cp}_0"], M[f"{cp}_1"] = a0, a1
    return M


def prior_order(S0_end: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """训练侧任务 seed-0 终点平均秩 → 配置序（小秩在前）。"""
    mr = np.array([rankdata(-S0_end[j]) for j in train_idx]).mean(0)
    return np.argsort(mr, kind="stable")


def sh_pick(order: np.ndarray, q25: np.ndarray, q50: np.ndarray, end: np.ndarray, k0: int, k1: int, k2: int) -> int:
    """SH k0-k1-k2：起跑前 k0 → @25% 留 k1 → @50% 留 k2 → @100% 选一条（全部按 seed 0）。"""
    c = list(order[:k0])
    c = sorted(c, key=lambda j: -q25[j])[:k1]
    c = sorted(c, key=lambda j: -q50[j])[:k2]
    return max(c, key=lambda j: end[j])


def headroom(M: dict[str, np.ndarray], groups: np.ndarray, *, train_mask_of=None) -> dict:
    """逐任务四个底的 regret（seed 1 终点）：train_mask_of(i) → 训练侧任务索引（默认 = 其他源）。"""
    S0, S1 = M["end_0"], M["end_1"]
    n = S0.shape[0]
    out = {b: np.full(n, np.nan) for b in ("prior1", "top3", "sh633", "sh1043")}
    for i in range(n):
        tr = train_mask_of(i) if train_mask_of else np.where(groups != groups[i])[0]
        if tr.size == 0:
            continue
        order = prior_order(S0, tr)
        oracle = int(np.argmax(S0[i]))
        p1 = int(order[0])
        p3 = int(order[:3][int(np.argmax(S0[i, order[:3]]))])
        s6 = sh_pick(order, M["q25_0"][i], M["q50_0"][i], S0[i], 6, 3, 3)
        s10 = sh_pick(order, M["q25_0"][i], M["q50_0"][i], S0[i], 10, 4, 3)
        for b, p in (("prior1", p1), ("top3", p3), ("sh633", s6), ("sh1043", s10)):
            out[b][i] = S1[i, oracle] - S1[i, p]
    return out


def decision(h: float) -> str:
    if not np.isfinite(h):
        return "HEADROOM-NA"
    return "HEADROOM-GO" if h >= DECISION["go_min"] else ("HEADROOM-B" if h < DECISION["stop_max"] else "HEADROOM-MID")


def summarize(reg: dict[str, np.ndarray], groups: np.ndarray) -> dict:
    res = {}
    for b, v in reg.items():
        ok = np.isfinite(v)
        per_src = {g: float(np.nanmean(v[groups == g])) for g in dict.fromkeys(groups.tolist())}
        res[b] = {"task_mean": float(np.nanmean(v)), "task_median": float(np.nanmedian(v)), "n": int(ok.sum()),
                  "source_equal_mean": float(np.mean(list(per_src.values()))), "source_equal_median": float(np.median(list(per_src.values()))),
                  "per_source": per_src}
    return res


def fmt_pool(name: str, res: dict, groups: np.ndarray) -> list[str]:
    n_src = len(set(groups.tolist()))
    md = [f"### {name}（{len(groups)} 任务，{n_src} 源）", "", "| 底 | regret 均值 | 中位 | 档 |", "|---|---:|---:|---|"]
    lab = {"prior1": "先验@1", "top3": "前3全量实测", "sh633": "SH6-3-3", "sh1043": "SH10-4-3"}
    for w, key, mkey in (("任务加权", "task_mean", "task_median"), ("源等权", "source_equal_mean", "source_equal_median")):
        for b in ("prior1", "top3", "sh633", "sh1043"):
            r = res[b]
            md.append(f"| {lab[b]} {w} | {r[key]:.3f} | {r[mkey]:.3f} | {decision(r[key])} |")
    ps = sorted(res["prior1"]["per_source"].items(), key=lambda kv: -kv[1])
    cnt = {g: int((groups == g).sum()) for g in set(groups.tolist())}
    md += ["", "逐源 regret@1（先验@1，任务加权）：" + "；".join(f"{g} {v:.3f} ({cnt[g]})" for g, v in ps), ""]
    return md


def main() -> None:
    cat = json.loads((UCR / "catalog.json").read_text(encoding="utf-8"))
    keys = [x["cls_key"] for x in cat["shared"]]
    cat_pp = json.loads((S1PP / "catalog.json").read_text(encoding="utf-8"))
    assert keys == [x["cls_key"] for x in cat_pp["shared"]], "shared-30 不一致"
    post = json.loads((S1PP / "posthoc_s1pp.json").read_text(encoding="utf-8"))
    v3 = [s for s, v in post["v3_admission"].items() if v.get("admit")]
    rows = load_rows(S1PP / "runs.jsonl", keys)
    rows.update(load_rows(UCR / "runs.jsonl", keys))
    all_tasks = sorted(rows)
    src_of = {t: t.rsplit("_a", 1)[0] for t in all_tasks}
    # S1″ 池的任务集 = posthoc 口径（试点 3 集 8 行、其余 4 行 = 准入行 1/7/3/5 + a03/a05 已含）；只取两种子都齐 ≥ 20 条的任务
    def tasks_of(srcs):
        ts = [t for t in all_tasks if src_of[t] in srcs]
        return [t for t in ts if sum(1 for k in keys if (k, 0) in rows[t] and (k, 1) in rows[t]) >= 20]
    pools = {
        "v3′ 11（复现 S1″ posthoc）": v3,
        "v3′ 11 + UCR 10 = 21（主）": v3 + UCR10,
        "护栏后 9 + UCR 10 = 19（去 PS / Ethanol）": [s for s in v3 if s not in GUARD_REJECT_S1PP] + UCR10,
        "UCR 10": UCR10,
    }
    out = {"pools": {}, "cross": {}, "curve": {}}
    md = ["# UCR-ADM 事后：余量 H（零 GPU，规则 = S1″ PROTOCOL §1.8 / posthoc §2）", "",
          "诚实口径：任务 oracle 按 seed 0 选、seed 1 评；任务盲底按 seed 0 在其他源（LODO）上选、seed 1 评；SH 的截点淘汰全按 seed 0。"
          f"决策阈值（S1″）：先验@1 任务加权 ≥ {DECISION['go_min']} GO / < {DECISION['stop_max']} 落点 B / 中间 MID。", ""]
    for name, srcs in pools.items():
        tasks = tasks_of(srcs)
        groups = np.array([src_of[t] for t in tasks])
        M = matrices(rows, keys, tasks)
        reg = headroom(M, groups)
        res = summarize(reg, groups)
        out["pools"][name] = {"sources": srcs, "n_tasks": len(tasks), "results": res, "decision_H_same": decision(res["prior1"]["task_mean"])}
        md += fmt_pool(name, res, groups)
    # 跨域
    main_tasks = tasks_of(v3 + UCR10)
    groups = np.array([src_of[t] for t in main_tasks])
    M = matrices(rows, keys, main_tasks)
    dom = np.array(["ucr" if src_of[t] in UCR10 else ("external" if src_of[t] in EXTERNAL else "uea") for t in main_tasks])
    reg3 = headroom(M, dom)
    idx_ucr = np.where(dom == "ucr")[0]; idx_non = np.where(dom != "ucr")[0]
    reg_a = headroom(M, groups, train_mask_of=lambda i: idx_non if dom[i] == "ucr" else np.array([], int))     # 非 UCR 选 → UCR 评
    reg_b = headroom(M, groups, train_mask_of=lambda i: idx_ucr if dom[i] != "ucr" else np.array([], int))     # UCR 选 → 非 UCR 评
    def m(v): return float(np.nanmean(v))
    out["cross"] = {"three_group_lodo": {b: m(v) for b, v in reg3.items()},
                    "nonucr_to_ucr": {b: m(v) for b, v in reg_a.items()}, "ucr_to_nonucr": {b: m(v) for b, v in reg_b.items()},
                    "H_cross_prior1": float(np.mean([m(reg_a["prior1"]), m(reg_b["prior1"])])),
                    "per_domain_three_group": {d: float(np.nanmean(reg3["prior1"][dom == d])) for d in ("uea", "external", "ucr")}}
    c = out["cross"]
    md += ["## 跨域（只作上界）", "",
           f"- 三组（uea / external / ucr）按组 LODO：先验@1 {c['three_group_lodo']['prior1']:.3f}（uea {c['per_domain_three_group']['uea']:.3f} / external {c['per_domain_three_group']['external']:.3f} / ucr {c['per_domain_three_group']['ucr']:.3f}）；"
           f"前 3 {c['three_group_lodo']['top3']:.3f}；SH6-3-3 {c['three_group_lodo']['sh633']:.3f}；SH10-4-3 {c['three_group_lodo']['sh1043']:.3f}",
           f"- 非 UCR 11 源选 → UCR 40 任务评：先验@1 {c['nonucr_to_ucr']['prior1']:.3f} / 前 3 {c['nonucr_to_ucr']['top3']:.3f} / SH6 {c['nonucr_to_ucr']['sh633']:.3f} / SH10 {c['nonucr_to_ucr']['sh1043']:.3f}",
           f"- UCR 10 源选 → 非 UCR 56 任务评：先验@1 {c['ucr_to_nonucr']['prior1']:.3f} / 前 3 {c['ucr_to_nonucr']['top3']:.3f} / SH6 {c['ucr_to_nonucr']['sh633']:.3f} / SH10 {c['ucr_to_nonucr']['sh1043']:.3f}",
           f"- **H_cross（两向均值，先验@1）= {c['H_cross_prior1']:.3f} → {decision(c['H_cross_prior1'])}**", ""]
    # 池大小曲线（主池；训练侧随机抽 k 个其他源）
    rng = np.random.default_rng(0)
    srcs_all = sorted(set(groups.tolist()))
    curve = {}
    for k in (2, 4, 6, 8, 10, 12, 15, 20):
        vals = {"prior1": [], "top3": [], "sh633": []}
        for _ in range(40):
            def tm(i, k=k):
                others = [s for s in srcs_all if s != groups[i]]
                pick = set(rng.choice(others, size=min(k, len(others)), replace=False).tolist())
                return np.where(np.isin(groups, list(pick)))[0]
            r = headroom(M, groups, train_mask_of=tm)
            for b in vals:
                vals[b].append(m(r[b]))
        curve[k] = {b: float(np.mean(v)) for b, v in vals.items()}
    out["curve"] = curve
    md += ["## 池大小曲线（主池 21 源 96 任务；训练侧随机抽 k 个其他源，40 次均值）", "", "| k | 先验@1 | 前3全量实测 | SH6-3-3 |", "|---:|---:|---:|---:|"]
    md += [f"| {k} | {v['prior1']:.3f} | {v['top3']:.3f} | {v['sh633']:.3f} |" for k, v in curve.items()]
    (UCR / "headroom_H.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (UCR / "headroom_H.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
