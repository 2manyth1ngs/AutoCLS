"""R 检验 §9：逐数据集 acc 表（选中配置的绝对 R'）——任务盲 / fwb（无特征）/ fw（有特征）/ fw@zero_all / fw@noise，UEA 逐集 + UCR 逐集摘要。
零 GPU，只读已有排名。产物 results/tclsc_s1ea_rank/ablate/acc_tables.md + acc_tables.json。
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

from scripts import pq_common as pq  # noqa: E402

_g57 = importlib.import_module("scripts.57_tclsc_l3")
_g60 = importlib.import_module("scripts.60_tclsc_s1ucr")
_g63 = importlib.import_module("scripts.63_tclsc_s1ea")
_g64 = importlib.import_module("scripts.64_tclsc_s1ea_confirm")
_g66 = importlib.import_module("scripts.66_tclsc_s1ea_rank")
RANK = ROOT / "results" / "tclsc_s1ea_rank"
ABL = RANK / "ablate"
NOISE = ("noise0", "noise1", "noise2")


def _truth_uea():
    rows, _ = _g57.load_source(pq.OUT)
    t = {}
    for r in rows:
        t.setdefault((r["task_id"], int(r["iters"])), {})[r["cls_key"]] = float(r["r_prime"])
    return t


def _truth_ucr():
    rows = _g60.load_rows()
    t = {}
    for r in rows:
        t.setdefault((r["task_id"], int(r["iters"])), {})[r["cls_key"]] = float(r["r_prime"])
    return t


def _collect(rk_normal_by_target, rk_ablate_by_target, truth, arms, fw):
    """{(靶, 预算): {列: [(top1, top3)]}}；列 = arms + fw@zero_all + fw@noise（3 种子均值）。"""
    per = {}
    for target, pb in rk_normal_by_target.items():
        for b, ps in pb.items():
            for s, pa in ps.items():
                t = truth[(s, int(b))]
                d = per.setdefault((target, int(b)), {"oracle": []})
                d["oracle"].append(max(t.values()))
                for a in arms:
                    if a in pa:
                        d.setdefault(a, []).append((t[pa[a][0]], max(t[k] for k in pa[a][:3])))
                ab = rk_ablate_by_target.get(target, {})
                if "zero_all" in ab:
                    ranked = ab["zero_all"][b][s][fw]
                    d.setdefault(f"{fw}@zero_all", []).append((t[ranked[0]], max(t[k] for k in ranked[:3])))
                    v1, v3 = [], []
                    for ns in NOISE:
                        rn = ab[ns][b][s][fw]
                        v1.append(t[rn[0]]); v3.append(max(t[k] for k in rn[:3]))
                    d.setdefault(f"{fw}@noise", []).append((float(np.mean(v1)), float(np.mean(v3))))
    return per


def _agg(per, targets, budgets, col, idx):
    """靶 × 预算 均值（切片均值再按预算均值）。"""
    vals = []
    for tg in targets:
        for b in budgets:
            d = per.get((tg, int(b)))
            if d and col in d:
                vals.append(np.mean([x[idx] for x in d[col]]) if col != "oracle" else np.mean(d[col]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    out_md, out_json = [], {}
    # ------------------------------------------------ UEA
    ab_u = json.loads((ABL / "rankings_uea.json").read_text(encoding="utf-8"))
    rk_u = _g64.merge_ref(json.loads((RANK / "uea" / "rankings_uea.json").read_text(encoding="utf-8")),
                          json.loads((_g66.S1PHI / "binned" / "rankings_s1phi.json").read_text(encoding="utf-8")), arms=("blind_diff_hard", "phi_diff_hard"))
    normal_u = {tg: per["normal"] for tg, per in rk_u.items()}
    for tg, per in ab_u.items():                       # fwb 的 normal 并入
        for b, ps in per["normal"].items():
            for s, pa in ps.items():
                normal_u.setdefault(tg, {}).setdefault(b, {}).setdefault(s, {}).update(pa)
    cols_u = ["blind_diff_hard", "fwb_hard", "fw_eaphi_hard"]
    per_u = _collect(normal_u, {tg: per["ablate"] for tg, per in ab_u.items()}, _truth_uea(), cols_u, "fw_eaphi_hard")
    names = {"blind_diff_hard": "任务盲", "fwb_hard": "fwb（无特征）", "fw_eaphi_hard": "fw（有特征）", "fw_eaphi_hard@zero_all": "fw@全零", "fw_eaphi_hard@noise": "fw@噪声"}
    allcols = cols_u + ["fw_eaphi_hard@zero_all", "fw_eaphi_hard@noise"]
    out_md += ["# R 检验 §9 逐数据集 acc 表（选中配置的 R'；top-3 = 排名前 3 里最好的一条，top-1 = 第 1 名）", "",
               "fwb = 同一 field 头但输入只有预算（无数据集特征）；fw@全零 / fw@噪声 = 已训 fw 在推理时把 E_A ⊕ φ 换成 0 / N(0,1)。", ""]
    for label, budgets in (("@290", (290,)), ("三预算均值", tuple(pq.BUDGETS))):
        for idx, kind in ((1, "top-3"), (0, "top-1")):
            out_md += [f"## UEA {label}，{kind} acc（每集 6 切片均值）", "",
                       "| 数据集 | oracle | " + " | ".join(names[c] for c in allcols) + " | 特征贡献 fw−fwb | fw−任务盲 |",
                       "|---|---:|" + "---:|" * len(allcols) + "---:|---:|"]
            rows_json = {}
            for tg in list(pq.FOLD_TARGETS) + ["均值"]:
                tgs = list(pq.FOLD_TARGETS) if tg == "均值" else [tg]
                o = _agg(per_u, tgs, budgets, "oracle", 0)
                v = {c: _agg(per_u, tgs, budgets, c, idx) for c in allcols}
                rows_json[tg] = {"oracle": o, **v}
                line = f"| {'**均值**' if tg == '均值' else tg} | {o:.3f} | " + " | ".join(f"{v[c]:.3f}" for c in allcols)
                line += f" | {v['fw_eaphi_hard'] - v['fwb_hard']:+.3f} | {v['fw_eaphi_hard'] - v['blind_diff_hard']:+.3f} |"
                out_md.append(line)
            out_md.append("")
            out_json[f"uea_{label}_{kind}"] = rows_json
    # ------------------------------------------------ UCR
    ab_c = json.loads((ABL / "rankings_ucr.json").read_text(encoding="utf-8"))
    rk_c = _g63.merge_ref_rankings(json.loads((RANK / "ucr" / "rankings_ea.json").read_text(encoding="utf-8")),
                                   json.loads((_g66.S1UCR / "rankings_ucr.json").read_text(encoding="utf-8")), arms=("u_blind_diff_hard", "u_phi_diff_hard"))
    normal_c = {}
    for fo, per in rk_c.items():
        for b, ps in per["normal"].items():
            for s, pa in ps.items():
                normal_c.setdefault(pq.dataset_of(s), {}).setdefault(b, {}).setdefault(s, {}).update(pa)
    ablate_c = {}
    for fo, per in ab_c.items():
        for b, ps in per["normal"].items():
            for s, pa in ps.items():
                normal_c.setdefault(pq.dataset_of(s), {}).setdefault(b, {}).setdefault(s, {}).update(pa)
        for mode, pb in per["ablate"].items():
            for b, ps in pb.items():
                for s, pa in ps.items():
                    ablate_c.setdefault(pq.dataset_of(s), {}).setdefault(mode, {}).setdefault(b, {})[s] = pa
    cols_c = ["u_blind_diff_hard", "u_fwb_hard", "u_fw_eaphi_hard"]
    per_c = _collect(normal_c, ablate_c, _truth_ucr(), cols_c, "u_fw_eaphi_hard")
    dsets = sorted({tg for tg, _b in per_c})
    allc = cols_c + ["u_fw_eaphi_hard@zero_all", "u_fw_eaphi_hard@noise"]
    namec = {"u_blind_diff_hard": "任务盲", "u_fwb_hard": "fwb（无特征）", "u_fw_eaphi_hard": "fw（有特征）", "u_fw_eaphi_hard@zero_all": "fw@全零", "u_fw_eaphi_hard@noise": "fw@噪声"}
    for idx, kind in ((1, "top-3"), (0, "top-1")):
        out_md += [f"## UCR 84 集三预算均值，{kind} acc：摘要", "", "| 列 | 84 集均值 | 比 fwb 高的集数 / 低 / 平 | 比任务盲高 / 低 / 平 |", "|---|---:|---|---|"]
        summ = {}
        for c in ["oracle"] + allc:
            m = np.mean([_agg(per_c, [d], pq.BUDGETS, c, idx) for d in dsets])
            if c in ("oracle",):
                out_md.append(f"| oracle | {m:.3f} | | |")
                summ[c] = m
                continue
            vs_fwb = [np.sign(round(_agg(per_c, [d], pq.BUDGETS, c, idx) - _agg(per_c, [d], pq.BUDGETS, "u_fwb_hard", idx), 4)) for d in dsets]
            vs_bl = [np.sign(round(_agg(per_c, [d], pq.BUDGETS, c, idx) - _agg(per_c, [d], pq.BUDGETS, "u_blind_diff_hard", idx), 4)) for d in dsets]
            cnt = lambda v: f"{sum(1 for x in v if x > 0)} / {sum(1 for x in v if x < 0)} / {sum(1 for x in v if x == 0)}"
            out_md.append(f"| {namec[c]} | {m:.3f} | {cnt(vs_fwb)} | {cnt(vs_bl)} |")
            summ[c] = m
        out_json[f"ucr_{kind}"] = summ
        gains = sorted(((_agg(per_c, [d], pq.BUDGETS, "u_fw_eaphi_hard", idx) - _agg(per_c, [d], pq.BUDGETS, "u_fwb_hard", idx), d) for d in dsets), reverse=True)
        out_md += ["", f"特征贡献（fw − fwb，{kind}）最大的 5 集：" + "、".join(f"{d} {g:+.3f}" for g, d in gains[:5]) + "；最小的 5 集：" + "、".join(f"{d} {g:+.3f}" for g, d in gains[-5:]), ""]
    out_md += ["## UCR 84 集逐集（三预算均值，top-3 acc）", "", "| 数据集 | oracle | 任务盲 | fwb | fw | fw@全零 | fw@噪声 | fw−fwb |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in dsets:
        o = _agg(per_c, [d], pq.BUDGETS, "oracle", 0)
        v = {c: _agg(per_c, [d], pq.BUDGETS, c, 1) for c in allc}
        out_md.append(f"| {d} | {o:.3f} | " + " | ".join(f"{v[c]:.3f}" for c in allc) + f" | {v['u_fw_eaphi_hard'] - v['u_fwb_hard']:+.3f} |")
    (ABL / "acc_tables.md").write_text("\n".join(out_md) + "\n", encoding="utf-8")
    (ABL / "acc_tables.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out_md[:60]))


if __name__ == "__main__":
    main()
