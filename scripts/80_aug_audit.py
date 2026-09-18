"""WP16 AA1 审计：增强层净空审计——按目标选增强档位 / 开关的空间在部署口径下有多少可赢、哪一部分是数据集级、档位是否可学。协议 results/tclsc_aa/PROTOCOL.md。零 GPU、只读既有数据。

  python scripts/80_aug_audit.py --run      # Q1–Q5 → analysis.json / tables.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autocls.search.surrogate import FieldSurrogate  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

OUT = ROOT / "results" / "tclsc_aa"
WT14 = ROOT / ".claude" / "worktrees" / "wp14-hspace" / "results"
WT8 = ROOT / ".claude" / "worktrees" / "wp8-aug" / "results"
SRC = {"mask": ROOT / "results" / "mx4_mask" / "runs.jsonl", "uea30": ROOT / "results" / "uea30" / "runs.jsonl", "deploy": ROOT / "results" / "mx4_deploy" / "runs.jsonl",
       "h2": WT14 / "mx4_h2_uea" / "runs.jsonl", "wp15": WT14 / "mx4_ucr_official" / "runs.jsonl", "aug": WT8 / "mx4_aug" / "runs.jsonl",
       "sp_labels": ROOT / "results" / "tclsc_sp" / "labels.jsonl", "nb": ROOT / "results" / "tclsc_nb" / "rprime_nb.jsonl"}
AUG_FIELDS = ("resize", "rescale", "jitter", "point_mask", "freq_mask", "crop", "emb_jitter", "emb_mask")
D1_ICC, D1_GAIN, D1_FRAC, D2_ICC, D2_FRAC, D3_FRAC, D3_AGREE = 0.30, 0.01, 1 / 3, 0.30, 1 / 3, 0.5, 0.5
SEED = 0


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


# ---------------------------------------------------------------- 统计工具
def icc1(M: np.ndarray, *, n_perm: int = 2000, seed: int = SEED) -> dict:
    """单向随机效应 ICC(1)：行 = 组（数据集），列 = 评分者（配置 / 基）；nan 按行内可用值处理（不平衡时用 k 的调和近似）。置换 p：打乱每列的行标签。"""
    M = np.asarray(M, float)

    def _icc(A):
        rows = [r[~np.isnan(r)] for r in A]
        rows = [r for r in rows if len(r) >= 2]
        n = len(rows)
        if n < 2:
            return float("nan")
        k = np.mean([len(r) for r in rows])
        grand = np.mean(np.concatenate(rows))
        msb = sum(len(r) * (r.mean() - grand) ** 2 for r in rows) / (n - 1)
        msw = sum(((r - r.mean()) ** 2).sum() for r in rows) / sum(len(r) - 1 for r in rows)
        return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else float("nan")

    icc = _icc(M)
    rng = np.random.default_rng([seed, 8001])
    cnt = 0
    for _ in range(int(n_perm)):
        P = np.stack([col[rng.permutation(len(col))] for col in M.T], axis=1)
        if _icc(P) >= icc:
            cnt += 1
    return {"icc": icc, "p": (cnt + 1) / (n_perm + 1), "n_groups": int(M.shape[0]), "k": int(M.shape[1])}


def twin_matrix(rows: Sequence[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    """孪生 Δ 矩阵（靶 × 基配置）：Δ = acc(twin) − acc(base)，缺格 nan。"""
    base = {(r["dataset"], r["cls_key"]): float(r["acc"]) for r in rows if not r.get("twin_of") and r.get("acc") is not None}
    delta: dict[tuple[str, str], float] = {}
    for r in rows:
        if r.get("twin_of") and r.get("acc") is not None and (r["dataset"], r["twin_of"]) in base:
            delta[(r["dataset"], r["twin_of"])] = float(r["acc"]) - base[(r["dataset"], r["twin_of"])]
    ds = sorted({d for d, _c in delta})
    cs = sorted({c for _d, c in delta})
    M = np.full((len(ds), len(cs)), np.nan)
    for (d, c), v in delta.items():
        M[ds.index(d), cs.index(c)] = v
    return M, ds, cs


def list_rules(off: Mapping[tuple[str, str], float], on: Mapping[tuple[str, str], float]) -> dict:
    """每靶四种名单规则的最好值：always_off / always_on / by_sign（该靶 Δ 均值符号决定整靶开关）/ oracle（逐配置取大）。"""
    out = {}
    ds = sorted({d for d, _c in off})
    for d in ds:
        cs = [c for dd, c in off if dd == d and (d, c) in on]
        o = [off[(d, c)] for c in cs]; n = [on[(d, c)] for c in cs]
        if not cs:
            continue
        sign = np.mean([n[i] - o[i] for i in range(len(cs))]) > 0
        out[d] = {"always_off": max(o), "always_on": max(n), "by_sign": max(n) if sign else max(o), "oracle": max(max(o[i], n[i]) for i in range(len(cs))),
                  "mean_delta": float(np.mean([n[i] - o[i] for i in range(len(cs))])), "frac_pos": float(np.mean([n[i] > o[i] for i in range(len(cs))])), "n": len(cs)}
    return out


def pd_components(pd: Mapping[float, float], levels: Sequence[float]) -> dict:
    on = [pd[l] for l in levels if l > 0 and l in pd]
    return {"onoff": float(pd.get(0.0, np.nan) - np.mean(on)) if on else None, "which_level_range": float(max(on) - min(on)) if on else None}


def level_agreement(pds: Sequence[Mapping[float, float]], levels: Sequence[float]) -> Optional[float]:
    """跨数据集"哪档更好"的一致性：各集 PD(>0) 曲线两两 Spearman 均值。"""
    from scipy.stats import spearmanr
    on_levels = [l for l in levels if l > 0]
    curves = [np.array([p[l] for l in on_levels]) for p in pds if all(l in p for l in on_levels)]
    if len(curves) < 2:
        return None
    vals = []
    for a, b in itertools.combinations(curves, 2):
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        vals.append(spearmanr(a, b).correlation)
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- 数据
def official_noise() -> dict[str, float]:
    """每靶官方噪声 σ_d = 3 种子冠军 acc 的 sd（uea30 / mx4_mask finals；seed 0 取同 cls 的 s1 行）。"""
    sig = {}
    for key in ("uea30", "mask"):
        rows = read(SRC[key])
        s1 = {(r["dataset"], r["cls_key"]): float(r["acc"]) for r in rows if r["stage"] == "s1_catalog" and r["seed"] == 0 and r.get("acc") is not None}
        fin = defaultdict(list)
        for r in rows:
            if r["stage"] == "s2_finals" and r.get("acc") is not None and r["arm"].startswith("final_SH") and "prior" in r["arm"] and r["seed"] in (1, 2):
                fin[(r["dataset"], r["cls_key"])].append(float(r["acc"]))
        for (d, c), v in fin.items():
            if (d, c) in s1 and len(v) >= 2:
                sig.setdefault(d, []).append(float(np.std([s1[(d, c)]] + v, ddof=1)))
    return {d: float(np.mean(v)) for d, v in sig.items()}


def deploy_base_rows(*, arms: tuple = ("shared", "ggs")) -> list[dict]:
    """六靶官方目录的基行（WP3 `mx4_deploy`；WP6 / WP8-AUG 的 runs 只含孪生行）。"""
    return [r for r in read(SRC["deploy"]) if r["stage"] == "s1_catalog" and r["seed"] == 0 and r.get("acc") is not None and r["arm"] in arms and not r.get("twin_of")]


def q1_input_mask() -> dict:
    rows = deploy_base_rows()                        # 六靶基行（shared-10 + GGS）
    for key in ("mask", "uea30"):
        for r in read(SRC[key]):
            if r["stage"] != "s1_catalog" or r["seed"] != 0 or r.get("acc") is None:
                continue
            if key == "uea30":                       # uea30 无 twin_of：掩码孪生按 cls 去掉 input_mask 后匹配基
                r = dict(r)
                if r["cls"].get("input_mask") not in (None, "none"):
                    base_cls = {k: v for k, v in r["cls"].items() if k != "input_mask"}
                    r["twin_of"] = pq.cls_key(CLS(**base_cls))
                else:
                    r["twin_of"] = None
            elif r["arm"] == "nominated" or not r.get("twin_of"):     # 六靶只取 shared + GGS 的孪生行（与 uea30 同 11 条）
                continue
            rows.append(r)
    M, ds, cs = twin_matrix(rows)
    sig = official_noise()
    med = float(np.median(list(sig.values()))) if sig else 0.01
    base = {(r["dataset"], r["cls_key"]): float(r["acc"]) for r in rows if not r.get("twin_of")}
    on = {(r["dataset"], r["twin_of"]): float(r["acc"]) for r in rows if r.get("twin_of")}
    off = {k: v for k, v in base.items() if k in on}
    rules = list_rules(off, on)
    per = {}
    for i, d in enumerate(ds):
        v = M[i][~np.isnan(M[i])]
        th = 2 * sig.get(d, med)
        per[d] = {"n": int(len(v)), "mean_delta": float(v.mean()), "frac_pos": float((v > 0).mean()), "frac_over": float((v > th).mean()), "frac_under": float((v < -th).mean()), "sigma": sig.get(d, med), **rules.get(d, {})}
    gain = {k: float(np.mean([per[d][k] - per[d]["always_off"] for d in ds if k in per[d]])) for k in ("always_on", "by_sign", "oracle")}
    over = {k: float(np.mean([(per[d][k] - per[d]["always_off"]) > 2 * per[d]["sigma"] for d in ds if k in per[d]])) for k in ("always_on", "by_sign", "oracle")}
    return {"n_datasets": len(ds), "n_configs": len(cs), "icc": icc1(M), "per_dataset": per, "rule_gain_vs_off": gain, "rule_frac_over_noise": over,
            "pooled": {"mean_delta": float(np.nanmean(M)), "frac_pos": float(np.nanmean(M > 0)), "frac_over": float(np.mean([per[d]["frac_over"] for d in ds])),
                       "frac_under": float(np.mean([per[d]["frac_under"] for d in ds]))}, "sigma_median": med}


def _variant_field(r: dict) -> Optional[str]:
    v = r.get("variant")
    if not isinstance(v, dict) or v.get("how") != "on":
        return None
    return f"{v['field']}={v['value']}"


def q2_single_switch() -> dict:
    sig = official_noise()
    med = float(np.median(list(sig.values()))) if sig else 0.01
    out: dict = {"h2": {}, "wp15": {}}
    # H2：23 靶 × 干净基 + 6 单开关
    h2 = [r for r in read(SRC["h2"]) if r.get("acc") is not None]
    base = {r["dataset"]: float(r["acc"]) for r in h2 if isinstance(r.get("variant"), dict) and r["variant"].get("how") == "base"}
    per_field: dict[str, dict[str, float]] = defaultdict(dict)
    for r in h2:
        f = _variant_field(r)
        if f and r["dataset"] in base:
            per_field[f][r["dataset"]] = float(r["acc"]) - base[r["dataset"]]
    for f, dv in per_field.items():
        v = np.array(list(dv.values()))
        th = np.array([2 * sig.get(d, med) for d in dv])
        out["h2"][f] = {"n": len(v), "mean": float(v.mean()), "frac_pos": float((v > 0).mean()), "frac_over": float((v > th).mean()), "frac_under": float((v < -th).mean())}
    best = {d: max(per_field[f].get(d, -np.inf) for f in per_field) for d in base}
    out["h2"]["oracle_single_switch"] = {"mean_gain": float(np.mean([max(0.0, best[d]) for d in base])), "frac_over": float(np.mean([best[d] > 2 * sig.get(d, med) for d in base])), "n": len(base)}
    # WP15：52 集 × 两基 × 增强开关（freq_mask / affine / highpass05 / inst_norm）+ 两基 seed 1 噪声
    w = [r for r in read(SRC["wp15"]) if r.get("acc") is not None and r.get("error") is None]
    base2 = {(r["dataset"], r["base"], r["seed"]): float(r["acc"]) for r in w if isinstance(r.get("variant"), dict) and r["variant"].get("how") == "base"}
    noise = defaultdict(list)
    for (d, b, s), v in base2.items():
        if s == 1 and (d, b, 0) in base2:
            noise[d].append(abs(v - base2[(d, b, 0)]) / np.sqrt(2))
    sig15 = {d: float(np.mean(v)) for d, v in noise.items()}
    med15 = float(np.median(list(sig15.values()))) if sig15 else 0.01
    aug_fields = ("input_transform=highpass05", "input_transform=inst_norm", "freq_mask=0.5", "affine=1.0")
    delta = defaultdict(dict)            # field -> (dataset, base) -> Δ
    for r in w:
        f = _variant_field(r)
        if f in aug_fields and r["seed"] == 0 and (r["dataset"], r["base"], 0) in base2:
            delta[f][(r["dataset"], r["base"])] = float(r["acc"]) - base2[(r["dataset"], r["base"], 0)]
    for f in aug_fields:
        dv = delta[f]
        ds = sorted({d for d, _b in dv})
        M = np.array([[dv.get((d, b), np.nan) for b in ("B_T", "B_I")] for d in ds])
        v = M[~np.isnan(M)]
        th = np.array([[2 * sig15.get(d, med15)] * 2 for d in ds])[~np.isnan(M)]
        out["wp15"][f] = {"n_datasets": len(ds), "icc": icc1(M), "mean": float(v.mean()), "frac_pos": float((v > 0).mean()), "frac_over": float((v > th).mean()), "frac_under": float((v < -th).mean())}
    ds_all = sorted({d for f in aug_fields for d, _b in delta[f]})
    best15 = {d: max(np.nanmean([delta[f].get((d, b), np.nan) for b in ("B_T", "B_I")]) for f in aug_fields if any((d, b) in delta[f] for b in ("B_T", "B_I"))) for d in ds_all}
    out["wp15"]["oracle_single_switch"] = {"mean_gain": float(np.mean([max(0.0, best15[d]) for d in ds_all])), "frac_over": float(np.mean([best15[d] > 2 * sig15.get(d, med15) for d in ds_all])), "n": len(ds_all), "sigma_median": med15}
    return out


def q3_new_aug_fields() -> dict:
    space_fields = set(CLSSpace.from_yaml().field_names)
    rows = deploy_base_rows(arms=("shared", "ggs", "nominated")) + [r for r in read(SRC["aug"]) if r["stage"] == "s1_catalog" and r["seed"] == 0 and r.get("acc") is not None and r.get("twin_of")]
    base = {(r["dataset"], r["cls_key"]): float(r["acc"]) for r in rows if not r.get("twin_of")}
    by_field: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("twin_of") and (r["dataset"], r["twin_of"]) in base:
            extra = [f"{k}={v}" for k, v in r["cls"].items() if k not in space_fields and v not in (None, "none", 0, 0.0)]
            if len(extra) == 1:
                by_field[extra[0]].append(r)
    sig = official_noise()
    med = float(np.median(list(sig.values()))) if sig else 0.01
    out = {}
    for f, rs in by_field.items():
        allrows = [r for r in rows if not r.get("twin_of")] + rs
        M, ds, cs = twin_matrix(allrows)
        v = M[~np.isnan(M)]
        th = np.array([[2 * sig.get(d, med)] * len(cs) for d in ds])[~np.isnan(M)]
        per = {d: {"mean": float(np.nanmean(M[i])), "frac_pos": float(np.nanmean(M[i] > 0)), "oracle_gain": float(max(0.0, np.nanmax(M[i])))} for i, d in enumerate(ds)}
        out[f] = {"n_datasets": len(ds), "n_configs": len(cs), "icc": icc1(M), "mean": float(v.mean()), "frac_pos": float((v > 0).mean()), "frac_over": float((v > th).mean()),
                  "frac_under": float((v < -th).mean()), "per_dataset": per, "oracle_frac_over": float(np.mean([np.nanmax(M[i]) > 2 * sig.get(d, med) for i, d in enumerate(ds)]))}
    return out


def q4_levels(*, n_perm: int = 96, bagging: int = 5, seed: int = SEED) -> dict:
    """切片口径：每数据集 HGB 拟合 z，对增强字段算部分依赖，分解开关分量 / 哪档极差；置换零；跨集一致性。"""
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space, mode="binned")
    rows = read(SRC["sp_labels"])
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    res: dict = {"fields": {}, "n_per_dataset": {d: len(v) for d, v in by_ds.items()}}
    levels_of = {f: [float(x) for x in space.choices(f)] for f in AUG_FIELDS}
    models = {}
    for d, rs in by_ds.items():
        X = np.stack([enc.encode(CLS(**r["cls"])) for r in rs]).astype(float)
        z = np.array([float(r["z"]) for r in rs])
        models[d] = (X, z, FieldSurrogate("hgb", bagging=bagging, seed=seed).fit(X, z))
    rng = np.random.default_rng([seed, 8004])
    for f in AUG_FIELDS:
        sl = enc.field_slices[f]
        levels = levels_of[f]
        pds, comps, null95 = {}, {}, {}
        for d, (X, z, m) in models.items():
            def pd_of(model):
                out = {}
                for j, l in enumerate(levels):
                    Xp = X.copy(); Xp[:, sl] = 0.0; Xp[:, sl.start + j] = 1.0
                    out[l] = float(model.predict(Xp)[0].mean())
                return out
            pds[d] = pd_of(m)
            comps[d] = pd_components(pds[d], levels)
            nulls = []
            for _ in range(max(8, int(n_perm // len(by_ds)))):          # 零参照与实测同 bagging（否则单成员模型的 PD 更抖、P95 偏大）
                zp = z[rng.permutation(len(z))]
                mp = FieldSurrogate("hgb", bagging=bagging, seed=seed).fit(X, zp)
                nulls.append(pd_components(pd_of(mp), levels)["which_level_range"])
            null95[d] = float(np.percentile(nulls, 95))
        frac_over = float(np.mean([comps[d]["which_level_range"] > null95[d] for d in comps]))
        res["fields"][f] = {"levels": levels, "pd": pds, "components": comps, "null95": null95, "frac_over_null": frac_over, "agreement": level_agreement(list(pds.values()), levels),
                            "mean_onoff": float(np.mean([c["onoff"] for c in comps.values()])), "mean_which_range": float(np.mean([c["which_level_range"] for c in comps.values()]))}
    return res


def q5_aggregate(q1: dict) -> dict:
    per = q1["per_dataset"]
    rows = read(SRC["uea30"]) + read(SRC["mask"])
    champ = defaultdict(list)
    for r in rows:
        if r["stage"] == "s2_finals" and r["arm"].startswith("final_SH") and "prior" in r["arm"] and r.get("acc") is not None:
            champ[r["dataset"]].append(float(r["acc"]))
    out = {}
    for d, p in per.items():
        out[d] = {"base_oracle": p["always_off"], "aug_layer_oracle": p["oracle"], "gain": p["oracle"] - p["always_off"], "champion_3seed": float(np.mean(champ[d])) if champ.get(d) else None,
                  "over_noise": bool((p["oracle"] - p["always_off"]) > 2 * p["sigma"])}
    return {"per_dataset": out, "mean_gain": float(np.mean([v["gain"] for v in out.values()])), "frac_over_noise": float(np.mean([v["over_noise"] for v in out.values()])),
            "mean_aug_oracle_minus_champion": float(np.mean([v["aug_layer_oracle"] - v["champion_3seed"] for v in out.values() if v["champion_3seed"] is not None]))}


# ---------------------------------------------------------------- 判定
def decide_d1(x: Mapping) -> bool:
    return bool(x["icc"] >= D1_ICC and x["p"] <= 0.05 and x["by_sign_minus_off"] >= D1_GAIN and x["frac_over"] >= D1_FRAC)


def decide_d2(fields: Mapping[str, Mapping]) -> bool:
    return any(v.get("icc") is not None and v["icc"] >= D2_ICC and v.get("frac_over", 0) >= D2_FRAC for v in fields.values())


def decide_d3(fields: Mapping[str, Mapping]) -> bool:
    ok = [v["frac_over_null"] >= D3_FRAC and (v["agreement"] is None or v["agreement"] < D3_AGREE) for v in fields.values()]
    return bool(ok and np.mean(ok) >= 0.5)


def verdict_aa(d: Mapping[str, bool]) -> str:
    if not (d["D1"] or d["D2"]):
        return "AA-DEAD"
    return "AA-LEVEL-GO" if d["D3"] else "AA-SWITCH-ONLY"


def cmd_run(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    q1 = q1_input_mask(); print("Q1 done", q1["icc"], q1["rule_gain_vs_off"], flush=True)
    q2 = q2_single_switch(); print("Q2 done", flush=True)
    q3 = q3_new_aug_fields(); print("Q3 done", list(q3), flush=True)
    q4 = q4_levels(); print("Q4 done", flush=True)
    q5 = q5_aggregate(q1); print("Q5 done", flush=True)
    d1 = decide_d1({"icc": q1["icc"]["icc"], "p": q1["icc"]["p"], "by_sign_minus_off": q1["rule_gain_vs_off"]["by_sign"], "frac_over": q1["rule_frac_over_noise"]["by_sign"]})
    d2_fields = {f: {"icc": q2["wp15"][f]["icc"]["icc"], "frac_over": max(q2["wp15"][f]["frac_over"], q2["h2"].get(f, {}).get("frac_over", 0))} for f in q2["wp15"] if f != "oracle_single_switch"}
    d2 = decide_d2(d2_fields)
    d3 = decide_d3({f: {"frac_over_null": v["frac_over_null"], "agreement": v["agreement"]} for f, v in q4["fields"].items()})
    dec = {"D1": d1, "D2": d2, "D3": d3}
    a = {"protocol": "results/tclsc_aa/PROTOCOL.md", "q1": q1, "q2": q2, "q3": q3, "q4": {k: v for k, v in q4.items()}, "q5": q5, "decisions": dec, "verdict": verdict_aa(dec)}
    a["predictions"] = {"P1": {"desc": "D1", "hit": d1}, "P2": {"desc": "D2 由 freq_mask 过", "hit": bool(d2_fields.get("freq_mask=0.5", {}).get("icc", 0) >= D2_ICC and d2_fields.get("freq_mask=0.5", {}).get("frac_over", 0) >= D2_FRAC)},
                        "P3": {"desc": "D3 ✗", "hit": not d3}, "P4": {"desc": "Q5 增强层 oracle − 基础 oracle ≥ +0.02", "value": q5["mean_gain"], "hit": bool(q5["mean_gain"] >= 0.02)},
                        "P5": {"desc": "always-on < always-off", "value": q1["rule_gain_vs_off"]["always_on"], "hit": bool(q1["rule_gain_vs_off"]["always_on"] < 0)}}
    (outdir / "analysis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (outdir / "tables.md").write_text(markdown_tables(a), encoding="utf-8")
    print(a["verdict"], dec, flush=True)
    return a


def markdown_tables(a: dict) -> str:
    f = lambda v, d=3: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    q1, q2, q3, q4, q5 = a["q1"], a["q2"], a["q3"], a["q4"], a["q5"]
    L = ["# AA1 审计 tables：增强层净空", "", f"读数码 **{a['verdict']}**（判定 {a['decisions']}）", "",
         f"## Q1 input_mask（{q1['n_datasets']} 靶 × {q1['n_configs']} 配置，官方 @100%）", "",
         f"ICC(1) {q1['icc']['icc']:.3f}（p {q1['icc']['p']:.3g}）；pooled Δ 均值 {q1['pooled']['mean_delta']:+.4f}、正占比 {q1['pooled']['frac_pos']:.2f}、超阈 {q1['pooled']['frac_over']:.2f} / 低于 −阈 {q1['pooled']['frac_under']:.2f}；σ 中位 {q1['sigma_median']:.3f}",
         f"名单规则 − always-off（均值 / 超阈靶占比）：always-on {q1['rule_gain_vs_off']['always_on']:+.4f} / {q1['rule_frac_over_noise']['always_on']:.2f}；按靶符号 {q1['rule_gain_vs_off']['by_sign']:+.4f} / {q1['rule_frac_over_noise']['by_sign']:.2f}；逐配置 oracle {q1['rule_gain_vs_off']['oracle']:+.4f} / {q1['rule_frac_over_noise']['oracle']:.2f}", "",
         "| 靶 | n | Δ 均值 | 正占比 | 超阈 | 低于 −阈 | σ | off 最好 | 按符号 | oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d, p in sorted(q1["per_dataset"].items(), key=lambda kv: -kv[1]["mean_delta"]):
        L.append(f"| {d} | {p['n']} | {p['mean_delta']:+.3f} | {p['frac_pos']:.2f} | {p['frac_over']:.2f} | {p['frac_under']:.2f} | {p['sigma']:.3f} | {p.get('always_off', float('nan')):.3f} | {p.get('by_sign', float('nan')):.3f} | {p.get('oracle', float('nan')):.3f} |")
    L += ["", "## Q2 单增强开关（部署口径）", "", "H2（23 靶，干净基 + 开关）：", "", "| 开关 | n | Δ 均值 | 正占比 | 超阈 | 低于 −阈 |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in q2["h2"].items():
        if k != "oracle_single_switch":
            L.append(f"| {k} | {v['n']} | {v['mean']:+.3f} | {v['frac_pos']:.2f} | {v['frac_over']:.2f} | {v['frac_under']:.2f} |")
    L += [f"最佳单开关 oracle − 基：均值 {q2['h2']['oracle_single_switch']['mean_gain']:+.4f}，超阈靶 {q2['h2']['oracle_single_switch']['frac_over']:.2f}", "",
          "WP15（UCR 52 集 × 两基）：", "", "| 开关 | n 集 | 跨基 ICC(1) (p) | Δ 均值 | 正占比 | 超阈 | 低于 −阈 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in q2["wp15"].items():
        if k != "oracle_single_switch":
            L.append(f"| {k} | {v['n_datasets']} | {v['icc']['icc']:.3f} ({v['icc']['p']:.3g}) | {v['mean']:+.3f} | {v['frac_pos']:.2f} | {v['frac_over']:.2f} | {v['frac_under']:.2f} |")
    o = q2["wp15"]["oracle_single_switch"]
    L += [f"最佳单开关 oracle − 基：均值 {o['mean_gain']:+.4f}，超阈集 {o['frac_over']:.2f}（σ 中位 {o['sigma_median']:.3f}）", "", "## Q3 新增强字段（WP8-AUG，6 靶 × 21 配置）", "",
          "| 字段 | ICC(1) (p) | Δ 均值 | 正占比 | 超阈 | 低于 −阈 | oracle 超阈靶 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in q3.items():
        L.append(f"| {k} | {v['icc']['icc']:.3f} ({v['icc']['p']:.3g}) | {v['mean']:+.3f} | {v['frac_pos']:.2f} | {v['frac_over']:.2f} | {v['frac_under']:.2f} | {v['oracle_frac_over']:.2f} |")
    L += ["", "## Q4 档位（切片口径，每数据集 HGB 部分依赖，z 单位）", "", "| 字段 | 开关分量 PD(0) − mean PD(>0) | 哪档极差均值 | 超置换 P95 的数据集占比 | 跨集档位一致性 |", "|---|---:|---:|---:|---:|"]
    for k, v in q4["fields"].items():
        L.append(f"| {k} | {v['mean_onoff']:+.3f} | {v['mean_which_range']:.3f} | {v['frac_over_null']:.2f} | {f(v['agreement'])} |")
    L += ["", f"## Q5 聚合（29 靶）：增强层 oracle − 基础 oracle 均值 {q5['mean_gain']:+.4f}，超阈靶 {q5['frac_over_noise']:.2f}；增强层 oracle − SH10-4-3 冠军（3 种子）{q5['mean_aug_oracle_minus_champion']:+.4f}", "",
          f"预测：{a['predictions']}", ""]
    return "\n".join(L) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--outdir", type=Path, default=OUT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.run:
        cmd_run(args.outdir)


if __name__ == "__main__":
    main()
