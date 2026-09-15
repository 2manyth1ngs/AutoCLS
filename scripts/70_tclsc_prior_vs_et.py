"""UEA 7 折 LODO 同 126 实例：任务盲先验（36 号 prior / hybrid，MX-4 头条零样本排序器）vs 任务盲 T-CLSC（blind_diff）vs E_T T-CLSC（ft60）——regret 与逐数据集 acc。
零 GPU。产物 results/tclsc_s1et/prior_vs_et.md / .json。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.labeling import ORIGIN_SHARED  # noqa: E402
from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

_g36 = importlib.import_module("scripts.36_prior_sh_sim")
_g56 = importlib.import_module("scripts.56_tclsc_l1")
_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g58 = importlib.import_module("scripts.58_tclsc_s1phi")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g69 = importlib.import_module("scripts.69_tclsc_s1et")
OUT = ROOT / "results" / "tclsc_s1et"
ARMS = ["prior", "hybrid", "blind_diff_hard", "fw_eaphi_hard", "ft60_hard"]
NAMES = {"prior": "先验(shared)", "hybrid": "先验+GBDT", "blind_diff_hard": "T-CLSC 任务盲", "fw_eaphi_hard": "T-CLSC+E_A", "ft60_hard": "T-CLSC+E_T"}


def main() -> None:
    rows, delta = _g57.load_source(pq.OUT)
    cat_z = _g56.zscore_within(rows)
    rk = _g64.merge_ref(json.loads((OUT / "uea" / "rankings_uea.json").read_text(encoding="utf-8")), _g69._refs_uea(), arms=_g69.UEA_REFS)
    enc = CLSEncoder(CLSSpace.from_yaml())                                   # 36 号：scalar 32 维
    truth = {}
    for r in rows:
        truth.setdefault((r["task_id"], int(r["iters"])), {})[r["cls_key"]] = float(r["r_prime"])
    # 先验 / hybrid 排名（36 号同式：先验只见训练侧 shared 行；GBDT 只见训练侧）
    for target in pq.FOLD_TARGETS:
        train = [r for r in cat_z if r["dataset"] != target]
        prior = _g36.prior_scores([r for r in train if r["origin"] == ORIGIN_SHARED])
        model = FieldSurrogate(kind="hgb", seed=0).fit(_g36.features(train, enc, max(pq.BUDGETS)), np.array([r["z"] for r in train]))
        for b, ps in rk[target]["normal"].items():
            for s, pa in ps.items():
                rows_sb = [r for r in cat_z if r["task_id"] == s and int(r["iters"]) == int(b)]
                keys = sorted(r["cls_key"] for r in rows_sb)
                by_key = {r["cls_key"]: r for r in rows_sb}
                g_mean, _ = model.predict(_g36.features([by_key[k] for k in keys], enc, max(pq.BUDGETS)))
                gbdt = {k: float(v) for k, v in zip(keys, g_mean)}
                pa["prior"] = _g36.rank_by(prior, keys)
                pa["hybrid"] = _g36.rank_by({k: (prior[k] + _g36.HYBRID_BIAS if k in prior else gbdt[k]) for k in keys}, keys)
    inst = _g58._eval_all(rk, cat_z, delta, ARMS, "normal")
    pooled = {a: {"regret1": float(np.mean([r["regret"][a + "1"] for r in inst])), "regret3": float(np.mean([r["regret"][a + "3"] for r in inst])),
                  "M0": _g57._pooled(inst, a, "pair_acc"), "M1": _g57._pooled(inst, a, "acc_top10")} for a in ARMS}
    tests = {f"{a}→ft60 regret@{k}": _g58.paired_regret(inst, a, "ft60_hard", key=k) for a in ("prior", "hybrid", "blind_diff_hard", "fw_eaphi_hard") for k in ("3", "1")}
    L = ["# UEA 7 折 LODO 同 126 实例：任务盲先验 vs T-CLSC（任务盲 / E_A / E_T）", "",
         "| 臂 | M1 | M0 | regret@1 | regret@3 |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        p = pooled[a]; L.append(f"| {NAMES[a]} | {p['M1']:.3f} | {p['M0']:.3f} | {p['regret1']:.4f} | {p['regret3']:.4f} |")
    L += ["", "配对 Wilcoxon（H1 ft60 更低）：", ""] + [f"- {k}: {t['mean_a']:.4f} → {t['mean_b']:.4f} p={t['p']:.3g} 胜/负/平 {t['wins_b']}/{t['losses_b']}/{t['ties']}" for k, t in tests.items()]
    out = {"pooled": pooled, "tests": tests, "per_dataset": {}}
    for label, budgets in (("@290", (290,)), ("三预算均值", tuple(pq.BUDGETS))):
        for idx, kind in ((1, "top-3"), (0, "top-1")):
            L += ["", f"## {label}，{kind} acc（每集 6 切片均值）", "", "| 数据集 | oracle | " + " | ".join(NAMES[a] for a in ARMS) + " | E_T − 先验+GBDT |",
                  "|---|---:|" + "---:|" * len(ARMS) + "---:|"]
            agg = {a: [] for a in ARMS + ["oracle"]}
            for tg in pq.FOLD_TARGETS:
                vals = {a: [] for a in ARMS}; ora = []
                for b in budgets:
                    for s, pa in rk[tg]["normal"][str(b)].items():
                        t = truth[(s, b)]; ora.append(max(t.values()))
                        for a in ARMS:
                            vals[a].append(t[pa[a][0]] if idx == 0 else max(t[k] for k in pa[a][:3]))
                o = float(np.mean(ora)); agg["oracle"].append(o)
                m = {a: float(np.mean(vals[a])) for a in ARMS}
                for a in ARMS: agg[a].append(m[a])
                out["per_dataset"][f"{label}|{kind}|{tg}"] = {"oracle": o, **m}
                L.append(f"| {tg} | {o:.3f} | " + " | ".join(f"{m[a]:.3f}" for a in ARMS) + f" | {m['ft60_hard'] - m['hybrid']:+.3f} |")
            L.append(f"| **均值** | {np.mean(agg['oracle']):.3f} | " + " | ".join(f"{np.mean(agg[a]):.3f}" for a in ARMS) + f" | {np.mean(agg['ft60_hard']) - np.mean(agg['hybrid']):+.3f} |")
    (OUT / "prior_vs_et.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "prior_vs_et.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
