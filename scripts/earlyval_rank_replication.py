"""在 HAR 之外复验 early-val 的排名相关性（M2 的 ρ=0.85 只在 HAR 上做过）。

M2 验证 early(290) 与 full(2320) 的 CLS 排名 Spearman ρ=0.85，**样本只有 UCI-HAR
一个数据集**，而它是目标池成员。early-validation 是整个项目成本模型的地基，却从没
在源池数据上复验过。

full-iters 判别实验恰好产出了所需数据：同一批 (任务, CLS, 初始化, 划分) 在 290 与
2320 下各一份 R'，且由 `pretrain_encoder` 的前缀性质（REPORT §1.1），两轮是同一条
训练轨迹的两个截点——比 M2 当年的两次独立训练配对得更干净。

同时复现 M2 那条已知限制在源池上的表现：把"被训坏的"剔掉之后，排名相关还剩多少。

用法：python scripts/earlyval_rank_replication.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from scripts.m2_rho_decomposition import perm_p, spearman  # noqa: E402

# 每个数据集：(early 记录来源, full 记录来源)。early=290 在历史缓存，full=2320 在本实验。
SRC = {
    "Handwriting": (
        "results/stage1_rerun_20260804/results/labels_route_a/rprime_cache.jsonl",
        "results/full_iters_probe/uea/trained_fill.jsonl"),
    "WISDM2019": (
        "results/candidate_survey/baseline/trained_fill.jsonl",
        "results/full_iters_probe/candidates/trained_fill.jsonl"),
}
EARLY, FULL = 290, 2320


def load(p: str, ds: str, iters: int) -> dict[tuple[str, str], float]:
    out = {}
    for line in (ROOT / p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dataset") == ds and r.get("iters") == iters and r.get("r_prime") is not None:
            out[(r["task_id"], r["cls_key"])] = float(r["r_prime"])
    return out


def main() -> None:
    print("early-val 排名相关性：在 HAR 之外复验（M2 只做过 HAR）")
    print(f"{'—'*92}")
    print(f"{'数据集':<16}{'n':>4}{'ρ(290 vs 2320)':>17}{'p(置换)':>10}   "
          f"{'剔掉最差4条后 ρ':>18}{'p':>9}")
    for ds, (pe, pf) in SRC.items():
        e, f = load(pe, ds, EARLY), load(pf, ds, FULL)
        keys = sorted(set(e) & set(f))
        a = [e[k] for k in keys]
        b = [f[k] for k in keys]
        r, p = spearman(a, b), perm_p(a, b)
        # 与 M2 §5 同口径：剔掉 full 最差的 4 条（即最可能"被训坏"的），看还剩多少
        order = sorted(range(len(b)), key=lambda i: b[i])[4:]
        a2, b2 = [a[i] for i in order], [b[i] for i in order]
        r2, p2 = spearman(a2, b2), perm_p(a2, b2)
        print(f"{ds:<16}{len(keys):>4}{r:>17.4f}{p:>10.4f}   {r2:>18.4f}{p2:>9.4f}")

    # 与 §5 的 M2 分解**同口径**：按 full R' 中位数二分，各半单独算 ρ
    print(f"\n{'—'*92}\n按 full R' 中位数二分（与 REPORT §5 的 M2 分解同口径）")
    print(f"{'数据集':<16}{'子集':<10}{'n':>4}{'full 范围':>18}{'ρ':>9}{'p(置换)':>10}")
    for ds, (pe, pf) in SRC.items():
        e, f = load(pe, ds, EARLY), load(pf, ds, FULL)
        keys = sorted(set(e) & set(f))
        a = [e[k] for k in keys]
        b = [f[k] for k in keys]
        med = sorted(b)[len(b) // 2]
        for lab, idx in (("上半(好)", [i for i in range(len(b)) if b[i] >= med]),
                         ("下半(坏)", [i for i in range(len(b)) if b[i] < med])):
            ai, bi = [a[i] for i in idx], [b[i] for i in idx]
            print(f"{ds:<16}{lab:<10}{len(idx):>4}"
                  f"{f'[{min(bi):.3f},{max(bi):.3f}]':>18}"
                  f"{spearman(ai, bi):>9.4f}{perm_p(ai, bi):>10.4f}")

    print(f"\n{'—'*92}")
    print("对照：M2 在 HAR 上的同一个量 ρ=0.8525（n=20，results/m2_probe/），"
          "剔掉最差 4 条后 0.7172，中位数二分后上半 0.6098(p=0.066)/下半 0.9273(p=0.0004)")

    # 各半 n=10、检出力低，单个"不显著"不等于"为 0"。用 Fisher 合并三个数据集的
    # 上半 p 值：这才是"好配置之间到底有没有排名信号"的诚实读法。
    import math

    def fisher(ps: list[float]) -> tuple[float, float]:
        stat = -2.0 * sum(math.log(p) for p in ps)
        df = 2 * len(ps)
        # 卡方上尾（df 为偶数时有闭式）
        k = df // 2
        term, s = 1.0, 1.0
        for i in range(1, k):
            term *= stat / 2 / i
            s += term
        return stat, math.exp(-stat / 2) * s

    print(f"\n{'—'*92}\nFisher 合并'上半(好配置)'的 p 值（各半 n=10，单看检出力不足）")
    for lab, ps in (("三个数据集全算", [0.0657, 0.0749, 0.8151]),
                    ("只算有学习信号的两个 (HAR + Handwriting)", [0.0657, 0.0749])):
        stat, p = fisher(ps)
        print(f"  {lab:<42} χ²={stat:6.2f} df={2*len(ps)}  p={p:.4f}")
    print("\n读法：好配置之间的排名信号**存在但弱得多**（ρ≈0.6 vs 坏半的 ≈0.9），"
          "\n      单个数据集测不显著，合并两个有信号的数据集后 p≈0.03。"
          "\n      不能说'簇内 ρ=0'，但'early-val 主要在排破坏'这个判读成立。")
    print("注：本表的 290/2320 由同一条训练轨迹的两个截点得到（REPORT §1.1 前缀性质），"
          "\n    比 M2 当年的两次独立训练配对更干净——故本表若偏低，不能归因于配对噪声。")


if __name__ == "__main__":
    main()
