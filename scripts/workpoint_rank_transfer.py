"""工作点臂的 CLS 排名可迁移性（造 label 前的前提检验）。

背景：M2 的 ρ=0.85 在 (290, 2320)=(5ep, 40ep)@HAR 上验证；full-iters 轮又发现
ρ 主要由"排破坏"撑着（好配置半区从未显著）。现在源池工作点移到了超早期
（Handwriting 60 iters/20ep、UniMiB 30 iters/5ep），且 Handwriting@20ep 的破坏
参与只有 5%——"排破坏"这个撑 ρ 的信号在工作点几乎不存在。**工作点臂的排名
还能不能迁移到更长训练，是新工作点下造 label 的前提，从未验证过。**

做法：全部用已有缓存（零新训练）。同一批 (任务, CLS, 初始化, 划分) 在不同
iters 臂下的 R' 由 pretrain_encoder 前缀性质保证是同一条训练轨迹的不同截点
（full-iters REPORT §1.1），配对不含初始化/划分方差。对每个 (工作点臂, 更长臂)
配对算 Spearman ρ + 置换 p + bootstrap CI，并沿用 full-iters 的分层做法：
按长臂 R' 中位数二分，各半单独算 ρ（上半=好配置、下半=被训坏）。

判读线（用户预注册）：工作点臂的 ρ（尤其好配置半区）显著低于 0.5 →
early-val 在工作点不可靠，须在造 label 前重新设计。

自检：Handwriting (290, 2320) pooled raw ρ 必须复现 full-iters 轮的 0.894
（earlyval_rank_replication.py 同数据同算法）。

用法：python scripts/workpoint_rank_transfer.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import baseline_seed_specs  # noqa: E402
from scripts.m2_rho_decomposition import perm_p, spearman  # noqa: E402

# 每数据集：R' 来源文件（跨轮合并，重复键做逐位一致性校验）、基线缓存、
# 工作点臂、要对照的更长臂。290 的 Handwriting 记录来自 5090（±1 样本抖动已知，
# 协议上只作对照臂如实标注）。
SPEC = {
    "Handwriting": {
        "trained": [
            "results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl",
            "results/full_iters_probe/uea/trained_fill.jsonl",
            "results/iters_workpoint/uea/trained_fill.jsonl",
            "results/workpoint_confirm/uea/trained_fill.jsonl",
        ],
        "baseline": "results/iters_workpoint/uea/baseline_cache.jsonl",
        "workpoint": 60,
        "longer": [120, 290, 2320],
        "reference_pairs": [(290, 2320)],   # full-iters 轮的原配对，作对照锚
    },
    "UniMiB-SHAR": {
        "trained": [
            "results/candidate_survey/baseline/trained_fill.jsonl",
            "results/candidate_survey/baseline_crop256/trained_fill.jsonl",
            "results/full_iters_probe/candidates/trained_fill.jsonl",
            "results/iters_workpoint/candidates/trained_fill.jsonl",
            "results/workpoint_followup/candidates/trained_fill.jsonl",
        ],
        "baseline": "results/iters_workpoint/candidates/baseline_cache.jsonl",
        "workpoint": 30,
        "longer": [60, 120, 290],
        "reference_pairs": [],              # 无 2320 臂，最长只有 290（48.3ep）
    },
}
GOLDEN_HW_290_2320 = 0.894   # full-iters 轮 earlyval_rank_replication 的 pooled raw ρ


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_trained(name: str, spec: dict) -> dict[int, dict[tuple[str, str], float]]:
    """iters → {(task_id, cls_key): r_prime}。split_seed 必须等于 canonical
    （与 iters_workpoint_scan.collect 同一条校验）；跨文件重复键要求 r_prime
    逐位一致（种子幂等的复核，事故记忆 idempotent-seeding 的教训）。"""
    arms: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    conflicts = 0
    for rel in spec["trained"]:
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(rel)
        for r in read_jsonl(p):
            if r.get("dataset") != name or r.get("r_prime") is None:
                continue
            if r.get("split_seed") != baseline_seed_specs(r["task_id"], 0)[1]:
                continue
            key = (r["task_id"], r["cls_key"])
            it = int(r["iters"])
            v = float(r["r_prime"])
            if key in arms[it] and arms[it][key] != v:
                conflicts += 1
            arms[it][key] = v
    if conflicts:
        raise RuntimeError(f"{name}: {conflicts} 个重复键 r_prime 不一致——缓存被污染，先查再算")
    return dict(arms)


def load_baseline(spec: dict) -> dict[str, dict]:
    """task_id → {mean, band}（band = 2·max(std, ACC 量子)，与 summarize_task 同式）。"""
    import statistics
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(ROOT / spec["baseline"]):
        if r.get("r_prime") is not None:
            rows_by_task[r["task_id"]].append(r)
    out = {}
    for tid, rows in rows_by_task.items():
        vals = [float(r["r_prime"]) for r in rows]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        quantum = max((1.0 / r["n_eval"] for r in rows if r.get("n_eval")), default=0.0)
        out[tid] = {"mean": statistics.fmean(vals), "band": 2.0 * max(std, quantum)}
    return out


def boot_ci(a: list[float], b: list[float], n_boot: int = 2000,
            seed: int = 0) -> tuple[float, float]:
    """配对 bootstrap 的 ρ 95% CI（percentile）。n 小时 CI 宽，如实报。"""
    rng = random.Random(seed)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        aa, bb = [a[i] for i in idx], [b[i] for i in idx]
        r = spearman(aa, bb)
        if r == r:   # 排除退化重采样（全并列 → nan）
            stats.append(r)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def analyze_pair(arms: dict, base: dict, w: int, l: int) -> dict | None:
    """一个 (工作点臂 w, 更长臂 l) 配对的全套统计。gain = R' − 任务基线均值
    （多任务 pooled 时去掉任务难度位移；2 任务时与 raw 排名几乎同构，两口径都存）。"""
    keys = sorted(set(arms.get(w, {})) & set(arms.get(l, {})))
    if len(keys) < 10:
        return None
    tasks = sorted({t for t, _ in keys})
    aw_raw = [arms[w][k] for k in keys]
    al_raw = [arms[l][k] for k in keys]
    aw = [arms[w][k] - base[k[0]]["mean"] for k in keys]
    al = [arms[l][k] - base[k[0]]["mean"] for k in keys]

    out = {"w": w, "l": l, "n": len(keys), "tasks": tasks,
           "rho_pooled_gain": spearman(aw, al), "p_pooled_gain": perm_p(aw, al),
           "ci_pooled_gain": boot_ci(aw, al),
           "rho_pooled_raw": spearman(aw_raw, al_raw),
           "p_pooled_raw": perm_p(aw_raw, al_raw)}

    # 分层：按长臂 gain 中位数二分（full-iters/earlyval_rank_replication 同口径，
    # 只是 raw→gain 以支持 6 任务 pooled）
    med = sorted(al)[len(al) // 2]
    halves = {}
    for lab, idx in (("upper", [i for i in range(len(al)) if al[i] >= med]),
                     ("lower", [i for i in range(len(al)) if al[i] < med])):
        ai, bi = [aw[i] for i in idx], [al[i] for i in idx]
        halves[lab] = {"n": len(idx), "l_gain_range": [min(bi), max(bi)],
                       "rho": spearman(ai, bi), "p": perm_p(ai, bi),
                       "ci": boot_ci(ai, bi)}
    out["halves"] = halves

    # 每任务 ρ（n=10/任务，任务内排名才是造 label 真正用的量）
    per_task = {}
    for t in tasks:
        tk = [k for k in keys if k[0] == t]
        ai = [arms[w][k] for k in tk]
        bi = [arms[l][k] for k in tk]
        per_task[t] = {"n": len(tk), "rho": spearman(ai, bi), "p": perm_p(ai, bi)}
    out["per_task"] = per_task
    rhos = [v["rho"] for v in per_task.values() if v["rho"] == v["rho"]]
    out["per_task_mean_rho"] = sum(rhos) / len(rhos) if rhos else float("nan")

    # 破坏参与：两臂各自的带外条数（↑/↓/带内，pooled）——判读"排破坏信号还剩多少"
    for tag, vals_raw in (("w", aw_raw), ("l", al_raw)):
        up = sum(1 for k, v in zip(keys, vals_raw)
                 if v > base[k[0]]["mean"] + base[k[0]]["band"])
        dn = sum(1 for k, v in zip(keys, vals_raw)
                 if v < base[k[0]]["mean"] - base[k[0]]["band"])
        out[f"band_{tag}"] = {"above": up, "below": dn, "inband": len(keys) - up - dn}
    # 下半区里有多少条在长臂被真破坏（带外下方）——下半 ρ 的成色
    lower_idx = [i for i in range(len(al)) if al[i] < med]
    out["lower_destroyed_at_l"] = sum(
        1 for i in lower_idx
        if al_raw[i] < base[keys[i][0]]["mean"] - base[keys[i][0]]["band"])
    return out


def fmt_pair(r: dict) -> str:
    u, lo = r["halves"]["upper"], r["halves"]["lower"]
    return (f"  {r['w']:>5} vs {r['l']:<5} n={r['n']:>3} ({len(r['tasks'])}任务)  "
            f"整体 ρ={r['rho_pooled_gain']:+.3f} (p={r['p_pooled_gain']:.4f}, "
            f"CI[{r['ci_pooled_gain'][0]:+.2f},{r['ci_pooled_gain'][1]:+.2f}])  "
            f"上半 ρ={u['rho']:+.3f} (p={u['p']:.4f}, CI[{u['ci'][0]:+.2f},{u['ci'][1]:+.2f}])  "
            f"下半 ρ={lo['rho']:+.3f} (p={lo['p']:.4f})  "
            f"每任务均值 ρ={r['per_task_mean_rho']:+.3f}")


def main() -> None:
    out_dir = ROOT / "results" / "workpoint_rank_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {}

    for name, spec in SPEC.items():
        arms = load_trained(name, spec)
        base = load_baseline(spec)
        w = spec["workpoint"]
        print(f"\n{'='*100}\n{name}（工作点 {w} iters）  可用臂: "
              f"{ {it: len(v) for it, v in sorted(arms.items())} }")
        res = {"workpoint": w, "pairs": [], "reference": []}

        for l in spec["longer"]:
            r = analyze_pair(arms, base, w, l)
            if r is None:
                print(f"  {w} vs {l}: 配对不足，跳过")
                continue
            res["pairs"].append(r)
            print(fmt_pair(r))
            pt = "   ".join(f"{t.split('_')[-1]}:{v['rho']:+.2f}"
                            for t, v in r["per_task"].items())
            print(f"        每任务 ρ: {pt}")
            print(f"        带外条数 工作点臂 ↑{r['band_w']['above']}/↓{r['band_w']['below']}"
                  f"  长臂 ↑{r['band_l']['above']}/↓{r['band_l']['below']}"
                  f"  下半区在长臂被破坏 {r['lower_destroyed_at_l']}/{r['halves']['lower']['n']}")

        for (a, b) in spec["reference_pairs"]:
            r = analyze_pair(arms, base, a, b)
            if r is None:
                continue
            res["reference"].append(r)
            print(f"  [对照锚] {fmt_pair(r).strip()}")
            if (a, b) == (290, 2320):
                drift = abs(r["rho_pooled_raw"] - GOLDEN_HW_290_2320)
                ok = drift < 0.005
                print(f"        自检: pooled raw ρ={r['rho_pooled_raw']:.4f} "
                      f"vs full-iters 轮 {GOLDEN_HW_290_2320} "
                      f"{'✅' if ok else '❌ 数据或算法口径变了，先查再引用'}")
                if not ok:
                    raise RuntimeError("金值自检失败")
        payload[name] = res

    (out_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {out_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
