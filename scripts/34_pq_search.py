"""MX-3-PQ 端到端搜索（Q6）：目录排名 regret + 进化搜索真评估。

协议：results/mx3_pq/PROTOCOL.md §1 Q6 与 §5 判读表（阈值先于数据写死）——
**D1** regret@1(A0 main) < regret@1(random@1)，126 实例配对单侧 Wilcoxon p<0.05；
**D2** regret@1(A0) < regret@1(A6 cls_b)，同法；
**D3** 进化 top-3 最佳 R'@b ≥ 目录 oracle@(切片 00, b) − 0.02 于 ≥11/21 实例。

实例定义（126 = 7 折靶 × 3 预算 × 6 切片）：每个 (目标数据集, 预算 b, 切片 s)
是一次**独立的零样本搜索**——候选 = 该切片的 20 条目录 CLS，比较器只看
(E_s, ctx=onehot(b))，不看任何 R'。真值网格来自 32 号的 rprime_cache.jsonl。

四个阶段（各自幂等、可单独跑）：
    python scripts/34_pq_search.py --rank                       # 126 实例 × 3 臂排名
    python scripts/34_pq_search.py --analyze                    # 臂/regret/D1/D2/成本
    python scripts/34_pq_search.py --evolve                     # 21 实例 × 2 臂进化 top-3
    python scripts/34_pq_search.py --evaluate-evo --resume      # 进化 pick 的真 R'
    python scripts/34_pq_search.py --smoke --outdir <临时目录>  # 合成 fixture 端到端

臂命名（与 33 号保存的模型文件名一致，PROTOCOL §3）：
`main`（A0，SetTF(E) ⊕ onehot(b)，3 个种子多数票）、`cls_b`（A6，仅 onehot(b)，
E-free 对照 = D2 的分母）、`cls_only`（A5，无任务支路，纯 CLS 先验）。

铁律落点：
- 铁律 1：候选 CLS 一律来自 cls_catalog.json / pq_common.catalog_for_slice
  （其取值集合又只来自 configs/space.yaml），本脚本零硬编码档位；
- 铁律 2：预算 b 显式进 ctx、进缓存键 (task_id, cls_key, iters)，
  真值/成本/D3 三处同口径引用同一个 b，早/全口径不混用；
- 铁律 3：进化 pick 的 R' 落 evo_cache.jsonl，按 (切片, cls_key, 预算) 恒幂等跳过；
  与主目录撞键的 pick 直接复用 rprime_cache.jsonl（canonical 种子相同 → 同一条轨迹）；
- 铁律 7/8：排名侧只吃比较器输出的成对概率（autocls/search/rank.py 纯聚合），
  "谁更优"的方向语义留在造 label 侧；本脚本只从 task_types 取 **task_type 取值
  常量**（铁律 7 的唯一真源）来断言真值行确实是分类，不引 is_better——
  regret/oracle 的方向是分类专用的、写死在 instance_arms 里，换任务类型要重写那段。

不修改 autocls/**、不修改其它脚本：`scripts/5_target_zero_shot.pair_prob_matrix`
不接 ctx，故此处实现本地变体 `pair_prob_matrix_ctx`（brief 明确要求）。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import yaml
from joblib import parallel_backend

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 模块名以数字开头 → importlib（25 号的臂/regret 机器与 16 号的单侧 Wilcoxon
# 是既有唯一真源，绝不在此 fork 第二份定义）
_mx2 = importlib.import_module("scripts.25_mx2_endtoend")
_lo = importlib.import_module("scripts.16_feasibility_loo")
# 32/33 的既有实现直接复用（只读，绝不改它们）：四态 resume、行合成、铁律 6 断言
_g32 = importlib.import_module("scripts.32_pq_generate_rprime")
_g33 = importlib.import_module("scripts.33_pq_comparator_loo")

from autocls.comparator import TCLSC  # noqa: E402
from autocls.comparator.mean_pool_encoder import MeanPoolTaskEncoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    ORIGIN_RANDOM, ORIGIN_SHARED, cls_key, load_probe_cls, load_rprime_cache,
    evaluate_rprime_trajectory,
)
from autocls.oom_guard import ON_OOM_RECORD, ON_OOM_SKIP, guarded_run_multi  # noqa: E402
from autocls.search.evolution import evolutionary_search  # noqa: E402
from autocls.search.rank import pair_vote_matrix, symmetrize  # noqa: E402
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402
from autocls.task_embed import TaskSetEncoder  # noqa: E402
from autocls.task_types import TASK_CLASSIFICATION  # noqa: E402
from scripts import pq_common as pq  # noqa: E402

# 25 号的定义原样复用（同一套 regret 语义，防漂移）
expected_max_of_k = _mx2.expected_max_of_k
order_candidates = _mx2.order_candidates
regret_of = _mx2.regret_of
bottom_quartile_hit = _mx2.bottom_quartile_hit
paired_wilcoxon_less = _lo.paired_wilcoxon_less
# 32 号的四态 resume 与行合成、33 号的铁律 6 断言（同一份定义，禁止 fork）
trajectory_state = _g32.trajectory_state
make_row_for = _g32.make_row_for
assert_fold_fit_pool = _g33.assert_fold_fit_pool
STATE_DONE, STATE_TERMINAL = _g32.DONE, _g32.TERMINAL
NON_FOLD_MODELS = (_g33.S2_FOLD, _g33.ALL8_FOLD)   # 非 LOO 折的存档（S2 上界臂）
S2_FOLD, ALL8_FOLD = _g33.S2_FOLD, _g33.ALL8_FOLD  # "S2"（折名）/"all8"（特征目录）

# ---------------------------------------------------------------- 协议常量
RANK_ARMS = ("main", "cls_b", "cls_only")
# 33 号的种子覆盖（PROTOCOL §3 表）：A0 三种子多数票，A5/A6 单种子
ARM_SEEDS: dict[str, tuple[int, ...]] = {"main": (0, 1, 2), "cls_b": (0,),
                                         "cls_only": (0,)}
# 排名臂 → 报表臂名（@1 / @3；cls_only 协议只报 @1）
ARM_REPORT: dict[str, tuple[str, Optional[str]]] = {
    "main": ("tclsc1", "tclsc3"),
    "cls_b": ("clsb1", "clsb3"),
    "cls_only": ("clsonly1", None),
}
PICK_ARMS = tuple(ARM_REPORT[a][0] for a in RANK_ARMS)   # 垫底四分位统计的对象
EVO_ARMS = ("main", "cls_b")
EVO_ORIGIN = {"main": "evo_main", "cls_b": "evo_cls_b"}  # brief「Task 7」写死

K_TOP = 3                # top-K 真评估（协议 §5 D3 的 top-3）
D3_MARGIN = 0.02         # D3 容差（协议 §5）
D3_MIN_HITS = 11         # D3 门：≥11/21 实例
ALPHA = 0.05             # D1/D2 单侧 Wilcoxon 显著性
EVO_SEED = 0             # 进化 rng 的基种子（rng = [EVO_SEED, 折下标, 预算]）
DIM_INPUT = 256          # 折特征 windows_repr 的最后一维（interfaces.md 钉死）

# smoke 口径（与 32 号 smoke 的 --checkpoints 4 8 12 对齐）
SMOKE_TARGETS = ("Handwriting", "UniMiB-SHAR")
SMOKE_BUDGETS = (4, 8, 12)
SMOKE_SLICES = 2
SMOKE_EVOLUTION = {"ks": 20, "kp": 6, "generations": 1, "p_cross": 0.8,
                   "p_mut": 0.2, "top_k": K_TOP}
FIXTURE_MARKER = "SMOKE_FIXTURE.json"

# 真评估入口做成模块级名字：--smoke / 单测用假评估器替换它（零 GPU），
# 生产路径永远指向 autocls.labeling.evaluate_rprime_trajectory。
EVALUATE_TRAJECTORY: Callable[..., dict] = evaluate_rprime_trajectory


# ================================================================ 纯函数
def feature_fold_of(meta: dict, default: Optional[str] = None) -> Optional[str]:
    """该存档实际使用的**特征目录名**（Task 6 评审后 meta 新增 `feature_fold`）。

    取值：S1 折 = `<target>`、S2 上界臂 = `"all8"`、不吃 E 的臂（cls_b/cls_only）
    = null。契约（协调方增补）：**键在场就以它为准**，仅当键完全缺席（老存档）
    才回落到 `meta["fold"]`——两者语义不同（`fold` 是"这个模型属于哪一折"，
    `feature_fold` 是"E 从哪个目录来"），S2 臂上两者恰好不等。
    `default` 只在两个键都拿不到时兜底。
    """
    if "feature_fold" in meta:
        ff = meta["feature_fold"]
        return None if ff is None else str(ff)
    fold = meta.get("fold")
    return default if fold is None else str(fold)


def load_pq_model(path: str | Path, *, map_location: str = "cpu",
                  expect_fold: Optional[str] = None,
                  expect_budgets: Optional[Sequence[int]] = None
                  ) -> tuple[TCLSC, dict]:
    """33 号保存的 `models/<target>/<arm>_s<seed>.pt` → (TCLSC, meta)。

    格式是 interfaces.md「Task 6 → Task 7 saved-model format」的绑定契约：
    `{"state_dict": ..., "meta": {arm, seed, fold, task_branch, ctx_dim, cls_dim,
    embedding_dim, fz, nz, set_encoder, e_source, settf_windows, settf_t_max,
    budgets, ctx_encoding}}`。本函数**独立实现**（不 import 33 号）——两边各持一份
    同格式 loader 是 brief 的明确要求，任何一边改格式都会在这里 load_state_dict 炸。

    `dim_input`（SetTF 第一层输入维）不在绑定 meta 键里：折特征 windows_repr 的
    最后一维被 interfaces.md 钉死为 256，故缺省取 DIM_INPUT；meta 若显式给了
    `dim_input`（smoke fixture 用小维度）则以 meta 为准。

    两道使用点自检（都是"静默错答案"级别的事故，宁可开跑前炸）：
    - `expect_fold`：存档的 `meta["fold"]` 必须等于它被当作哪一折在用。
      `models/Handwriting/main_s0.pt` 里躺着别折的权重 = 考侧数据集参与过训练
      （铁律 6 违例），而排名照样能算出来，只是全错。S2 存档（`fold == "S2"`，
      特征目录 `all8`）不是 LOO 折，**只有** `expect_fold="S2"` 时才放行。
    - `meta["feature_fold"]`（Task 6 评审后新增）= 该臂真正用的特征目录名：
      S1 折是 `<target>`、S2 臂是 `"all8"`、无折特征的臂（cls_b/cls_only）是
      null。特征目录一律从它解析，**仅在该键缺席时**才回落到 `fold`（老存档）。
      `all8` 是含全部 8 个数据集的 fit 池（非 zero-shot），出现在 S1 折上就是
      铁律 6 违例，故在此拦死。
    - `expect_budgets`：ctx 是**预算 one-hot**，它的分量顺序由训练时的 budgets
      列表定义。`--budgets 290 60 120` 这样重排后，同一个 one-hot 会被读成另一个
      预算——模型不会报错，regret 表却整体错位。故逐位比对 `meta["budgets"]`，
      并核对 `ctx_encoding` 仍是 onehot（换编码 = 换语义）。
    """
    try:
        blob = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as e:      # meta 里混进非原语（不符合 interfaces 契约）时的兜底
        print(f"⚠️ {path} 的 weights_only 读取失败（{e}）——meta 应只含 JSON 原语，"
              "回落到非安全反序列化（仅限本仓自产文件）", flush=True)
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not (isinstance(blob, dict) and "state_dict" in blob and "meta" in blob):
        raise ValueError(
            f"{path} 不是 Task 6 约定的 {{state_dict, meta}} 存档（interfaces.md）")
    meta = dict(blob["meta"])
    kind = meta.get("set_encoder")
    nz = meta.get("nz")
    dim_input = int(meta.get("dim_input") or DIM_INPUT)
    if kind == "settf":
        set_enc: Optional[torch.nn.Module] = TaskSetEncoder(
            dim_input=dim_input, nz=int(nz))
    elif kind == "meanpool":
        set_enc = MeanPoolTaskEncoder(dim_input=dim_input)
    elif kind in (None, "null", ""):
        set_enc = None
    else:
        raise ValueError(f"未知 set_encoder={kind!r}（可选 settf/meanpool/null）")
    model = TCLSC(cls_dim=int(meta["cls_dim"]), set_encoder=set_enc,
                  embedding_dim=int(meta["embedding_dim"]), fz=int(meta["fz"]),
                  ctx_dim=int(meta["ctx_dim"]),
                  task_branch=str(meta["task_branch"]),
                  head=str(meta.get("head", "concat")),      # WP16-L3：老存档无 head 键 = concat
                  cls_encoder=("graph" if meta.get("encoder_mode") == "graph" else "mlp"),   # WP16-L2
                  graph_opts=meta.get("graph_opts"))
    if expect_fold is not None and str(meta.get("fold")) != str(expect_fold):
        allowed = (f"（非 LOO 折存档 {NON_FOLD_MODELS} 只在显式请求该折时放行）"
                   if str(meta.get("fold")) in NON_FOLD_MODELS else "")
        raise ValueError(
            f"{path} 的 meta['fold']={meta.get('fold')!r} ≠ 使用处的折 "
            f"{expect_fold!r}{allowed}——折与权重错配等于考侧数据集参与过训练"
            "（铁律 6），排名会静默算出全错的结果")
    ff = feature_fold_of(meta)
    if (ff == ALL8_FOLD and expect_fold is not None
            and str(expect_fold) != S2_FOLD):
        raise ValueError(
            f"{path} 的 feature_fold='{ALL8_FOLD}'（fit 池含全部 8 个数据集，"
            f"非 zero-shot）却被当作 S1 折 {expect_fold!r} 使用——铁律 6 违例；"
            f"S2 上界臂只能以 expect_fold='{S2_FOLD}' 显式请求")
    if expect_budgets is not None:
        got = tuple(int(b) for b in meta.get("budgets", ()))
        want = tuple(int(b) for b in expect_budgets)
        if got != want:
            raise ValueError(
                f"{path} 的 meta['budgets']={list(got)} ≠ 本次 budgets={list(want)}"
                "——ctx one-hot 的分量顺序由训练时的 budgets 定义，重排/取子集会让"
                "同一个 one-hot 被读成另一个预算（铁律 2）")
        if meta.get("ctx_encoding") != "onehot":
            raise ValueError(
                f"{path} 的 ctx_encoding={meta.get('ctx_encoding')!r} ≠ 'onehot'"
                "——本脚本的 ctx 一律走 pq_common.ctx_vector 的 one-hot")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, meta


def uses_E(model) -> bool:
    """模型是否吃任务嵌入 E（唯一判定点——曾在三处各写一份 getattr 默认值）。

    默认 True：没有该属性的对象按"吃 E"处理，与 5 号 `pair_prob_matrix` 的原语义
    一致（它无条件传 E）。TCLSC 恒有该属性，故只影响玩具/替身模型。
    """
    return bool(getattr(model, "uses_E", True))


def model_task_inputs(model, e_row, ctx_row):
    """按模型的任务支路裁剪 (E, ctx)：不用的那路一律传 None（模型会 fail-fast）。

    A5 `cls_only`（task_branch="none", ctx_dim=0）两路都不吃；A6 `cls_b`
    （task_branch="ctx"）只吃 ctx。喂多了会 ValueError，喂少了会静默降级——
    所以在这一处统一裁，调用方不各自判断。
    """
    E = e_row if uses_E(model) else None
    ctx = ctx_row if int(getattr(model, "ctx_dim", 0)) > 0 else None
    return E, ctx


def pair_prob_matrix_ctx(model, vecs: torch.Tensor, e_row=None, ctx_row=None,
                         *, batch_size: int = 4096) -> np.ndarray:
    """全部有序对一批前向：P[i,j] = P(候选 i 优于 j | E, ctx)，对角 NaN。

    `scripts/5_target_zero_shot.pair_prob_matrix` 的 ctx 版本本地变体（brief 约束：
    不改 5 号）。差异只有两条：多传一路 ctx、按 `batch_size` 切批。

    分批只是防御性的：本函数的实际调用点是目录排名（20 条候选 → 380 对，一批装得
    下）。进化搜索**不走这里**——它用 `make_score_pairs` 闭包，切批由
    `autocls/search/evolution.py::_pairwise_wins` 自己负责（默认 4096 对/批）。
    """
    n = int(vecs.shape[0])
    ii, jj = np.where(~np.eye(n, dtype=bool))
    P = np.full((n, n), np.nan)
    with torch.no_grad():
        for s in range(0, ii.size, batch_size):
            a_idx, b_idx = ii[s:s + batch_size], jj[s:s + batch_size]
            m = len(a_idx)
            E = None if e_row is None else e_row.expand(m, -1)
            ctx = None if ctx_row is None else ctx_row.expand(m, -1)
            out = model(vecs[a_idx], vecs[b_idx], E, ctx).cpu().numpy()
            P[a_idx, b_idx] = out
    return P


def rank_from_scores(S_list: Sequence[np.ndarray], keys: Sequence[str]) -> list[str]:
    """多种子对称化得分 → 排名键表（多数票胜场 → 平均得分 → key 升序，逐位确定）。"""
    V = pair_vote_matrix(list(S_list))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 对角全 NaN 的空切片均值
        mean_S = np.nanmean(np.stack([np.asarray(S, float) for S in S_list]), axis=0)
        tb = np.nanmean(mean_S, axis=1)
    order = order_candidates(V, tb, list(keys))
    return [keys[i] for i in order]


def rank_candidates(models, e_rows, vecs: torch.Tensor, keys: Sequence[str],
                    ctx_row=None) -> tuple[list[str], list[np.ndarray]]:
    """一个 (切片, 预算) 实例的 zero-shot 排名：每种子一张对称化得分矩阵 → 多数票。

    models / e_rows 等长（e_rows[i] 是第 i 个种子自己的 SetTF 产出的 E——SetTF 与
    比较器联合训练，三个种子的 E **不是**同一个向量，不能只算一次共用）。
    """
    assert len(models) == len(e_rows), "models 与 e_rows 必须一一对应"
    S_list = []
    for model, e_row in zip(models, e_rows):
        E, ctx = model_task_inputs(model, e_row, ctx_row)
        S_list.append(symmetrize(pair_prob_matrix_ctx(model, vecs, E, ctx)))
    return rank_from_scores(S_list, keys), S_list


def instance_arms(truth: dict[str, float], ranked: dict[str, Sequence[str]],
                  ggs_r: Optional[float], k_top: int = K_TOP) -> dict[str, float]:
    """一个搜索实例的全部臂（协议 §5 Q6 行 + 25 号 task_arms 的 PQ 扩展）。

    truth: 该 (切片, 预算) 的 20 条目录 CLS → R'；ranked: {排名臂: 键表}。
    `main` 是 D1/D2 的分子，必须在场；`cls_b`/`cls_only` 缺席时**不造假**——
    对应臂直接不出现在返回里（下游 gate 会报 reason 而不是拿别的臂顶包）。
    random1/random3 是解析期望（不放回抽 k 取最大），非某次抽样的实现值。
    """
    if "main" not in ranked:
        raise KeyError("instance_arms 需要 main 臂排名（D1/D2 的分子）")
    vals = [float(v) for v in truth.values()]
    arms: dict[str, float] = {"oracle": max(vals)}
    for arm, (name1, name3) in ARM_REPORT.items():
        keys = ranked.get(arm)
        if not keys:
            continue
        arms[name1] = float(truth[keys[0]])
        if name3 is not None:
            arms[name3] = max(float(truth[k]) for k in keys[:k_top])
    if ggs_r is not None:
        arms["ggs"] = float(ggs_r)
    arms["random1"] = float(np.mean(vals))
    arms["random3"] = expected_max_of_k(vals, min(k_top, len(vals)))
    arms["bottom"] = min(vals)
    return arms


def gate(rows: Sequence[dict], arm_a: str, arm_b: str, *,
         alpha: float = ALPHA) -> dict:
    """配对单侧 Wilcoxon 门：H1 = regret[arm_a] 的中位差 < regret[arm_b]。

    D1 = gate(..., "tclsc1", "random1")、D2 = gate(..., "tclsc1", "clsb1")。
    任何一行缺臂 → 不做"有多少算多少"的静默子集检验（实例集会变，p 值不可比），
    直接返回 p=None + reason。
    """
    missing = [r for r in rows
               if arm_a not in r["regret"] or arm_b not in r["regret"]]
    if not rows or missing:
        return {"arm_a": arm_a, "arm_b": arm_b, "n": len(rows), "p": None,
                "pass": None, "mean_a": None, "mean_b": None,
                "reason": (f"{len(missing)}/{len(rows)} 个实例缺 {arm_a}/{arm_b} 臂"
                           if rows else "无实例")}
    x = np.array([float(r["regret"][arm_a]) for r in rows])
    y = np.array([float(r["regret"][arm_b]) for r in rows])
    p = float(paired_wilcoxon_less(x, y))
    return {"arm_a": arm_a, "arm_b": arm_b, "n": len(rows), "p": p,
            "pass": bool(p < alpha), "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "median_delta": float(np.median(x - y)),
            "n_better": int(np.sum(x < y)), "reason": None}


def expected_instances(budgets: Sequence[int]) -> int:
    """满格搜索实例数 = Σ_折靶 该折切片数 × 预算数（协议 §5 的 126 = 7×6×3）。

    切片数从注册表读（`pq.pq_slice_ids`）而不是写死 6：SCP2 只有 3 个切片，将来
    若有折靶切片数不同，这个式子仍是对的。
    """
    return sum(len(pq.pq_slice_ids(t)) for t in pq.FOLD_TARGETS) * len(tuple(budgets))


def mark_incomplete(g: dict, *, n_expected: int, allow_incomplete: bool) -> dict:
    """不满格时给门打 `incomplete` 并**收回 pass/fail 结论**（除非显式放行）。

    D1/D2 是先于数据写死的门（协议 §5），分母写死 126。在子集上算出的 p 值不是
    那个门——把它印成"过/不过"就是拿一场小考的成绩去宣布大考结果。故：不满格
    时 `pass=None` + reason，p 值照常保留（描述性可读），`--allow-incomplete`
    才恢复判定（报表须自行声明 n）。D3 已有同款处理。
    """
    n = int(g.get("n") or 0)
    g["n_expected"] = int(n_expected)
    g["incomplete"] = bool(n != n_expected)
    if g["incomplete"] and g.get("reason") is None:
        msg = (f"只有 {n}/{n_expected} 个搜索实例（协议 §5 的门按满格写死）")
        if allow_incomplete:
            g["reason"] = msg + "；--allow-incomplete 已放行，报表须声明 n"
        else:
            g["pass"] = None
            g["reason"] = msg + "；未加 --allow-incomplete，不作过/不过判定"
    return g


def mean_regret(rows: Sequence[dict]) -> dict[str, float]:
    """逐臂平均 regret（只对在场的臂求均值，缺席臂不出现）。"""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for a, v in r["regret"].items():
            acc[a].append(float(v))
    return {a: float(np.mean(v)) for a, v in sorted(acc.items())}


def _stratum_sort_key(item):
    """分层键排序：数字层按数值升序（"60" < "120" < "290"），其余按字典序、pooled 殿后。"""
    k = item[0]
    return (1, 0.0, k) if not str(k).lstrip("-").isdigit() else (0, float(k), "")


def strata_table(rows: Sequence[dict], field: str) -> dict[str, dict]:
    """按 `budget` / `dataset` 分层的 regret 表（含层内 D1/D2 的描述性 p）。

    层内 p 是**描述性读数**——协议只把 pooled 126 实例的 D1/D2 写成门，分层表
    用来看"赢面集中在哪个预算/哪个数据集"，不另设显著性阈值。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    out = {}
    for k, rs in sorted(groups.items(), key=_stratum_sort_key):
        out[k] = {"n": len(rs), "mean_regret": mean_regret(rs),
                  "D1": gate(rs, "tclsc1", "random1"),
                  "D2": gate(rs, "tclsc1", "clsb1")}
    return out


def cost_table(rows: Sequence[dict], *, full_iters: int) -> dict[str, dict]:
    """成本表：零样本 top-3 真评估 vs 直接搜完 20 条目录（逐预算 + pooled）。

    口径（25 号 cmd_analyze 同款，且必须带 @iters 标签——CLAUDE.md「净空/增益是
    口径量」）：两个臂都取缓存行的 `seconds`（该 checkpoint 为止的累计墙钟，含本
    次评估），故同 b 同硬件可比；`catalog_direct_full_est_s` 是把**直接搜索臂**
    外推到 full_iters=2320 的估计（因子 2320/b），零样本臂仍记 @b 的实付。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["budget"])].append(r)
    groups["pooled"] = list(rows)

    def cell(rs: Sequence[dict], b: Optional[int]) -> dict:
        top3 = float(np.nansum([r["seconds_top3"] for r in rs]))
        cat = float(np.nansum([r["seconds_catalog"] for r in rs]))
        # pooled 行的 iters 口径不唯一 → 因子按各实例自己的 b 加权外推
        est = (cat * (full_iters / b) if b else
               float(np.nansum([r["seconds_catalog"] * (full_iters / r["budget"])
                                for r in rs])))
        out = {"n": len(rs), "zero_shot_top3_s": top3, "catalog_direct_s": cat,
               "full_iters_factor": (full_iters / b) if b else None,
               "catalog_direct_full_est_s": est}
        out["speedup_vs_direct"] = (cat / top3) if top3 > 0 else None
        out["speedup_vs_direct_full"] = (est / top3) if top3 > 0 else None
        out["log10_speedup_full"] = (float(np.log10(est / top3))
                                     if top3 > 0 and est > 0 else None)
        return out

    return {k: cell(rs, None if k == "pooled" else int(k))
            for k, rs in sorted(groups.items(), key=_stratum_sort_key)}


def evo_rng(target: str, budget: int, *, base_seed: int = EVO_SEED
            ) -> np.random.Generator:
    """进化搜索的 rng：`default_rng([base_seed, 折下标, 预算])`（brief 写死）。

    折下标取 FOLD_TARGETS 的位置（而非哈希）——21 个实例各有独立随机流，
    重跑逐位复现，且 `--folds` 子集跑出的结果与全量跑逐位相同。
    """
    return np.random.default_rng([int(base_seed), pq.FOLD_TARGETS.index(target),
                                  int(budget)])


def evo_worklist(picks: dict, *, budgets: Sequence[int] = pq.BUDGETS) -> list[dict]:
    """evo_picks.json → 去重后的真评估工单（每条 = 一条要跑的轨迹）。

    去重理由（铁律 3）：一条轨迹由 (切片, cls_key) 唯一确定——canonical 种子
    `canonical_seeds(slice, cls_key)` 不含臂与预算，所以 main@60 与 cls_b@120 选中
    同一条 CLS 时跑两遍会得到**逐位相同**的 R'，却在缓存里写出重复键。故按
    (fold, cls_key) 合并：`origin`/`budget_searched` 取先到者（折序 → 预算升序 →
    臂序 → 名次），全部认领方在 `claims` 里逐条留痕，D3 判读按 claims 归位。
    """
    seen: dict[tuple[str, str], dict] = {}
    for target in sorted(picks, key=lambda t: (pq.FOLD_TARGETS.index(t)
                                               if t in pq.FOLD_TARGETS else 99, t)):
        slice00 = f"{target}_00"
        per_b = picks[target]
        for b in sorted(int(x) for x in per_b):
            if int(b) not in {int(x) for x in budgets}:
                continue
            per_arm = per_b.get(str(b), per_b.get(b, {}))
            for arm in EVO_ARMS:
                for rank, entry in enumerate(per_arm.get(arm, [])):
                    k = entry["cls_key"]
                    claim = {"arm": arm, "budget": int(b), "rank": rank}
                    if (target, k) in seen:
                        seen[(target, k)]["claims"].append(claim)
                        continue
                    seen[(target, k)] = {
                        "fold": target, "slice": slice00, "cls_key": k,
                        "cls": dict(entry["cls"]), "origin": EVO_ORIGIN[arm],
                        "budget_searched": int(b), "claims": [claim]}
    return [seen[k] for k in sorted(seen)]


def d3_report(picks: dict, r_of: Callable[[str, int, str], Optional[float]],
              oracle_of: Callable[[str, int], Optional[float]], *,
              margin: float = D3_MARGIN, min_hits: int = D3_MIN_HITS) -> dict:
    """D3（协议 §5）：进化 top-3 最佳 R'@b ≥ 目录 oracle@(切片 00, b) − margin。

    `r_of(fold, budget, cls_key)` 给该 pick 在 @budget 的 R'（未评估 → None），
    `oracle_of(fold, budget)` 给目录 oracle。门只对 `main` 生效；`cls_b` 同法报，
    外加两臂命中差——"进化比目录强"若在 E-free 臂上同样成立，则该增益与 E 无关。
    缺 R' 的 pick 一律**记为不命中**并逐个列进 `missing`（不当作"暂缺"跳过：
    跳过会让分母缩水，把没跑完的实验读成过门）。
    """
    per_arm: dict[str, dict] = {}
    n_instances = 0
    for arm in EVO_ARMS:
        insts, hits, evaluated = [], 0, 0
        for target in sorted(picks, key=lambda t: (pq.FOLD_TARGETS.index(t)
                                                   if t in pq.FOLD_TARGETS else 99, t)):
            per_b = picks[target]
            for b_key in sorted(per_b, key=lambda x: int(x)):
                b = int(b_key)
                entries = per_b[b_key].get(arm, [])
                vals, missing = [], []
                for e in entries:
                    v = r_of(target, b, e["cls_key"])
                    if v is None or not np.isfinite(v):
                        missing.append(e["cls_key"])
                    else:
                        vals.append(float(v))
                oracle = oracle_of(target, b)
                best = max(vals) if vals else None
                hit = bool(vals and oracle is not None
                           and best >= float(oracle) - margin)
                hits += int(hit)
                evaluated += int(bool(vals))
                insts.append({"fold": target, "budget": b, "evo_best": best,
                              "catalog_oracle": (None if oracle is None
                                                 else float(oracle)),
                              "hit": hit, "n_picks": len(entries),
                              "missing": missing})
        per_arm[arm] = {"hits": hits, "n": len(insts), "n_evaluated": evaluated,
                        "pass": bool(hits >= min_hits), "instances": insts}
        n_instances = max(n_instances, len(insts))
    return {"margin": margin, "min_hits": min_hits, "n_instances": n_instances,
            "per_arm": per_arm,
            "difference": (per_arm["main"]["hits"] - per_arm["cls_b"]["hits"]
                           if {"main", "cls_b"} <= set(per_arm) else None)}


def make_evo_row(entry: dict, task, rec: dict, *, iters: int,
                 weight_seed: int) -> dict:
    """进化 pick 的一行缓存 = 32 号的 `make_row_for` + 三个进化专属字段。

    行体**不自己拼**：`_g32.make_row_for` 是 rprime_cache 行 schema 的唯一实现
    （dataset/task_id/tag/task_type/classes/cls_key/origin/seed/iters/
    trajectory_id/`**rec`/r_prime/cls），照抄一份就会在对方改字段时静默分叉。
    `rec` 自带的 `checkpoint_iters` 由它原样带出（guarded_run_multi 只核对不
    改写）。本函数只追加 `fold` / `budget_searched` / `claims`——它们是"这条
    轨迹是被谁、在哪个预算上搜出来的"的溯源，主缓存里没有对应概念。
    """
    row = make_row_for(int(iters), rec,
                       dataset=pq.dataset_of(entry["slice"]), task=task,
                       cls_key=entry["cls_key"], origin=entry["origin"],
                       cls_cfg=CLS(**entry["cls"]), weight_seed=int(weight_seed))
    return {**row, "fold": entry["fold"],
            "budget_searched": int(entry["budget_searched"]),
            "claims": list(entry["claims"])}


# ================================================================ 装载
def _load_datasets(names: Sequence[str]) -> dict:
    """pq_common 的数据装载入口（单测 monkeypatch 此处即可零数据跑通评估路径）。"""
    return pq.load_pq_datasets(tuple(names))


def _cls_entry(d) -> tuple[CLS, str]:
    """{"cls": {...}, "cls_key": ...} / 裸 CLS 字典 → (CLS, key)。"""
    if isinstance(d, dict) and "cls" in d:
        c = CLS(**d["cls"])
        return c, d.get("cls_key") or cls_key(c)
    c = CLS(**d)
    return c, cls_key(c)


def load_catalog(outdir: Path, space: CLSSpace, *,
                 slices: Optional[Sequence[str]] = None
                 ) -> tuple[dict[str, list[tuple[str, CLS, str]]], CLS, str]:
    """候选目录 → ({切片: [(origin, CLS, key)]}, GGS CLS, GGS key)。

    首选 32 号落盘的 `cls_catalog.json`（2c schema：shared 列表 + random 逐切片）；
    缺席时回落到 `pq_common.catalog_for_slice`——它才是目录的构造式真源（10 shared
    探针 + 10 按 task_id 派生的 random，seed 42），两条路给出的键集合逐位相同。
    """
    path = Path(outdir) / "cls_catalog.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if doc is not None and not ({"shared", "random"} <= set(doc)):
        if slices is None:
            raise ValueError(
                f"{path} 不是 2c schema（需 shared 列表 + random 逐切片，"
                f"实得键 {sorted(doc)}），且无 slices 可回落到构造式")
        print(f"⚠️ {path} 不是 2c schema（键 {sorted(doc)}）——回落到 "
              "pq_common.catalog_for_slice 构造式", flush=True)
        doc = None
    if doc is not None:
        shared = [_cls_entry(d) for d in doc["shared"]]
        # 切片全集 = 调用方给的 slices ∪ random 表的键：shared 对全部切片共用，
        # 某切片的 random 名单为空时它仍是个合法的候选集（只有 shared 候选），
        # 只按 random 的键建表会让这种切片整个消失（静默少实例）。
        tids = sorted(set(doc["random"]) | set(slices or ()))
        cat = {}
        for tid in tids:
            rnd = [_cls_entry(d) for d in doc["random"].get(tid, [])]
            cat[tid] = ([(ORIGIN_SHARED, c, k) for c, k in shared]
                        + [(ORIGIN_RANDOM, c, k) for c, k in rnd])
        if "ggs" in doc:
            ggs, ggs_key = _cls_entry(doc["ggs"])
        else:
            ggs = pq.ggs_cls()
            ggs_key = doc.get("ggs_key") or cls_key(ggs)
        return cat, ggs, ggs_key
    if slices is None:
        raise FileNotFoundError(
            f"{path} 不存在，且未给 slices 无法回落到 catalog_for_slice"
            "（先跑 `scripts/32_pq_generate_rprime.py --tasks`）")
    shared_list = load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    cat = {s: pq.catalog_for_slice(space, s, shared_list) for s in slices}
    ggs = pq.ggs_cls()
    return cat, ggs, cls_key(ggs)


def fold_model_paths(outdir: Path, target: str, arm: str) -> list[Path]:
    return [Path(outdir) / "models" / target / f"{arm}_s{s}.pt"
            for s in ARM_SEEDS[arm]]


def load_fold_models(outdir: Path, target: str, *, arms: Sequence[str] = RANK_ARMS,
                     device: str = "cpu", budgets: Optional[Sequence[int]] = None
                     ) -> dict[str, list[tuple[TCLSC, dict]]]:
    """{臂: [(模型, meta)]}——某臂的模型文件缺席则整臂缺席（不用别的种子顶包）。

    每个存档都过 `load_pq_model` 的两道使用点自检：`meta["fold"]` 必须等于
    `target`（铁律 6），`meta["budgets"]` 必须与本次 budgets 逐位相同（ctx one-hot
    的分量顺序，铁律 2）。`budgets=None` 时跳过后者（纯装载场景）。
    """
    out: dict[str, list[tuple[TCLSC, dict]]] = {}
    for arm in arms:
        paths = fold_model_paths(outdir, target, arm)
        have = [p for p in paths if p.exists()]
        if not have:
            continue
        if len(have) != len(paths):
            print(f"⚠️ [{target}/{arm}] 只有 {len(have)}/{len(paths)} 个种子模型："
                  f"缺 {[p.name for p in paths if not p.exists()]}——本臂按现有种子"
                  "多数票（协议要求 3 种子，报表须声明）", flush=True)
        models = []
        for p in have:
            m, meta = load_pq_model(p, map_location=device, expect_fold=target,
                                    expect_budgets=budgets)
            models.append((m.to(device), meta))
        out[arm] = models
    return out


def set_encoder_dim_input(model, meta: Optional[dict] = None) -> Optional[int]:
    """该模型的 SetTF/均值池化编码器**期望的窗口特征维** D。

    取值优先级（都取"输入维的事实"，绝不拿输出维凑数）：
    1. TaskSetEncoder：`intra_setpool.enc[0].mab.fc_q.in_features`
       （SetPool → SAB → MAB.fc_q 是第一个吃 dim_input 的线性层）；
    2. MeanPoolTaskEncoder：标准化缓冲区 `mu` 的长度**就是** dim_input；
    3. `meta["dim_input"]`（本脚本 smoke fixture 会写；33 号不写该键）；
    4. 兜底 `enc.nz`——**那是输出维**，只在前三条都拿不到时才用，并且告警。
       对 TaskSetEncoder 而言 nz 与 dim_input 可以不等（如 dim_input=16, nz=8），
       静默拿它当输入维会让维度闸放行本该拦下的错配。
    不吃 E 的臂返回 None。
    """
    enc = getattr(model, "set_encoder", None)
    if enc is None:
        return None
    try:
        return int(enc.intra_setpool.enc[0].mab.fc_q.in_features)
    except (AttributeError, IndexError, TypeError):
        pass
    mu = getattr(enc, "mu", None)          # MeanPoolTaskEncoder：逐维标准化缓冲
    if mu is not None and getattr(mu, "shape", None):
        return int(mu.shape[0])
    if meta is not None and meta.get("dim_input") is not None:
        return int(meta["dim_input"])
    nz = getattr(enc, "nz", None)
    if nz is not None:
        print(f"⚠️ 无法从 {type(enc).__name__} 读出 dim_input，退而用 nz={nz}"
              "（那是**输出**维）作维度闸的期望值；meta 里补 dim_input 可消除本告警",
              flush=True)
        return int(nz)
    return None


def check_windows_dim(w, model, *, meta=None, where: str = "") -> None:
    """喂进 SetTF 之前核对 `windows_repr` 的最后一维 == 编码器的 dim_input。

    不核对的后果就是评审轮 2 的现场：16 维合成窗口撞上 33 号真模型的 256 维
    SetTF，报 `mat1 and mat2 shapes cannot be multiplied (24x16 and 256x256)`
    ——错误发生在注意力的矩阵乘法里，堆栈完全看不出"特征和模型不是一套"。
    """
    want = set_encoder_dim_input(model, meta)
    if want is None:
        return
    got = int(w.shape[-1])
    if got != want:
        raise ValueError(
            f"特征与模型不是一套：{where} 的 windows_repr 最后一维 = {got}，"
            f"而该模型的 set_encoder 期望 dim_input = {want}。"
            "常见原因是同一个目录里混了 33 号的真产物与 34 号 --smoke 的合成"
            "替身——用 `--smoke --fresh` 清掉合成部分重来，或先跑完 "
            "`33_pq_comparator_loo.py --smoke` 让全目录同源")


def check_windows_count(w, meta: Optional[dict], *,
                        n_instances: Optional[int] = None,
                        where: str = "") -> None:
    """核对 `windows_repr` 的**条数** == 被打分模型 meta 的 `settf_windows`。

    `settf_windows` 是协议 §3 的预注册旋钮（100 → 32）。33 号改了旋钮却没重跑
    `--fit-folds` 时，34 号会拿旧 npz 的 100 个窗口去喂一个"记着 32"的模型：
    形状不会报错（SetTF 本就吃变长集合），但 E 与训练时不同源，排名与 regret
    全部失真且**无任何征兆**——比维度不符更危险，因为它不会崩。

    期望值按切片实例数封顶（`pq.expected_window_count`）：源池里 Cricket_02 只有
    90 个实例，`encode_task_windows` 的 `m = min(sample_num, N)` 让它在
    settf_windows=100 下只可能有 90 行；写死 `== 100` 会把这一格永远判红。
    """
    want = None if not meta else meta.get("settf_windows")
    if want is None:
        return
    exp = pq.expected_window_count(int(want), n_instances)
    got = int(w.shape[0])
    if got != exp:
        raise ValueError(
            f"特征与模型的 settf_windows 不是一套：{where} 有 {got} 个窗口，"
            f"而该模型 meta 记的 settf_windows = {int(want)}"
            + (f"（按切片实例数 {n_instances} 封顶后应为 {exp}）"
               if exp != int(want) else "")
            + "。旋钮改过却没重跑特征：先移走该折的 task_features/，再跑 "
              f"`python scripts/33_pq_comparator_loo.py --fit-folds "
              f"--settf-windows {int(want)}`")


def slice_n_instances(outdir: Path, feature_fold: str,
                      slice_id: str) -> Optional[int]:
    """折特征 npz 的 META[3] → 该切片实例数（缺文件/缺 META → None）。"""
    p = Path(outdir) / "folds" / feature_fold / "task_features" / f"{slice_id}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return pq.n_instances_from_meta(z.get("META"))


def load_slice_windows(outdir: Path, feature_fold: str, slice_id: str, *,
                       device: str = "cpu",
                       meta: Optional[dict] = None) -> Optional[torch.Tensor]:
    """`folds/<feature_fold>/task_features/<slice>.npz` → windows_repr (m,T,D)。

    目录名是**特征折**（`meta["feature_fold"]`）而非模型折：S2 上界臂的权重记在
    `fold="S2"`，它的 E 却来自 `folds/all8/`。两者混用会静默读到别的折的 E。

    传 `meta`（被打分模型的 meta）时顺带核对窗口条数（见 check_windows_count）。
    """
    p = Path(outdir) / "folds" / feature_fold / "task_features" / f"{slice_id}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        w = np.asarray(z["windows_repr"], dtype=np.float32)
        n_inst = pq.n_instances_from_meta(z.get("META"))
    t = torch.as_tensor(w, device=device)
    check_windows_count(t, meta, n_instances=n_inst, where=str(p))
    return t


def target_slices(outdir: Path, target: str, catalog: dict, *,
                  feature_fold: Optional[str] = None) -> list[str]:
    """该折靶参与搜索的切片：注册表口径的 PQ 切片 ∩ 有目录 ∩ 有特征。

    `feature_fold` 缺省 = `target`（S1 折的常规情形）；S2 臂传 "all8"。
    """
    feat = Path(outdir) / "folds" / (feature_fold or target) / "task_features"
    return [s for s in pq.pq_slice_ids(target)
            if s in catalog and (feat / f"{s}.npz").exists()]


def resolve_budgets(outdir: Path, cli: Optional[Sequence[int]]) -> tuple[int, ...]:
    """预算口径：命令行 > fixture 标记（smoke 目录自描述）> pq_common.BUDGETS。

    生产目录里出现 fixture 标记 = 合成产物已经污染了真值目录：这时**不能**默默
    按标记里的假预算跑下去（那会把 4/8/12 的合成口径套到真 R' 上），直接 fail-fast
    并要求人工清理。`write_smoke_fixture` 有对称的写入端拦截。
    """
    outdir = Path(outdir)
    marker = outdir / FIXTURE_MARKER
    if marker.exists() and outdir.resolve() == pq.OUT.resolve():
        sys.exit(f"生产目录 {pq.OUT} 里出现 {FIXTURE_MARKER}（合成 fixture 标记）"
                 "——真值缓存可能已被污染，请人工核查并清理后再跑")
    if cli:
        return tuple(int(b) for b in cli)
    if marker.exists():
        doc = json.loads(marker.read_text(encoding="utf-8"))
        return tuple(int(b) for b in doc["budgets"])
    return tuple(pq.BUDGETS)


def _as_float(v) -> float:
    """JSON 数值 → float；None / 非有限一律 nan（下游用 nansum 聚合）。"""
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _read_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def assert_not_production(outdir: Path, what: str) -> None:
    """合成产物绝不许落进 results/mx3_pq（铁律 3：真值缓存一旦掺假就再也分不出来）。"""
    if Path(outdir).resolve() == pq.OUT.resolve():
        sys.exit(f"{what} 禁止写生产目录 {pq.OUT}——请指定别的 --outdir")


def assert_fold_features_compliant(outdir: Path, target: str) -> Optional[dict]:
    """使用折特征**之前**核对铁律 6：该折的 TS2Vec fit 池不得含考侧数据集。

    断言体沿用 33 号的 `assert_fold_fit_pool`（同一份定义，all8 折显式豁免）。
    meta 缺席时只告警不拦——`--smoke` 的合成特征与"33 还没跑完 --fit-folds"都是
    合法中间态；有 meta 却违例则必须炸（33 号 `windows_for` 同语义）。
    """
    meta_path = Path(outdir) / "folds" / target / "shared_ts2vec.meta.json"
    if not meta_path.exists():
        print(f"⚠️ [{target}] 缺 {meta_path.name}，无法在使用点核对铁律 6 的 fit 池"
              "（33 号 --fit-folds 会写它）——本折特征按未验证使用", flush=True)
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert_fold_fit_pool(target, meta.get("fit_datasets", []))
    return meta


def deep_merge(prev, new):
    """逐键深合并（字典递归、其余整体替换）——`--folds`/`--arms` 子集跑的落盘语义。

    子集跑直接覆写会把别的折/臂**静默抹掉**（32 号 `_write_merged` 同款教训：
    pod 上每数据集一个进程共用一份 json）。这里对 rankings.json（折/预算/切片/臂
    四层）与 evo_picks.json（折/预算/臂三层）通用：叶子是列表，整体替换即"本次
    重算了这一格"。
    """
    if not (isinstance(prev, dict) and isinstance(new, dict)):
        return new
    out = dict(prev)
    for k, v in new.items():
        out[k] = deep_merge(prev[k], v) if k in prev else v
    return out


def write_json_merged(path: Path, payload: dict) -> dict:
    """深合并进既有文件后**原子**落盘（tmp + os.replace）。

    原子性理由同 32 号的多行原子写：写到一半被杀会留下半个 JSON，下次
    `json.loads` 直接炸，而这时既有的 126 格排名已经没了。
    """
    path = Path(path)
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"⚠️ {path} 不是合法 JSON（半截文件？），本次整体重写", flush=True)
    merged = deep_merge(prev, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
    return merged


# ================================================================ --rank
def cmd_rank(outdir: Path, *, budgets: Sequence[int], folds: Sequence[str],
             arms: Sequence[str] = RANK_ARMS, device: str = "cpu") -> dict:
    """126 实例 × 臂的 zero-shot 排名 → rankings.json。

    条件化输入严格分工（铁律 7）：E 来自**折内**特征（目标数据集从 TS2Vec fit /
    label / δ 里整体抠除，考试时才前向编码）、ctx = onehot(b)。比较器一次都看不到
    R'——真值网格只在 --analyze 里作裁判。
    """
    outdir = Path(outdir)
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    all_slices = [s for t in folds for s in pq.pq_slice_ids(t)]
    catalog, _ggs, _ggs_key = load_catalog(outdir, space, slices=all_slices)

    rankings: dict[str, dict] = {}
    for target in folds:
        models = load_fold_models(outdir, target, arms=arms, device=device,
                                  budgets=budgets)
        if "main" not in models:
            print(f"⚠️ [{target}] 无 main 臂模型，整折跳过（先跑 33 号 --train）",
                  flush=True)
            continue
        main_ff = feature_fold_of(models["main"][0][1], target) or target
        slices = target_slices(outdir, target, catalog, feature_fold=main_ff)
        if not slices:
            print(f"⚠️ [{target}] 无可用切片（缺目录或折特征），跳过", flush=True)
            continue
        # E 逐 (臂, 种子, 切片) 只算一次：SetTF 与比较器联合训练，种子间 E 不同；
        # 窗口表征逐 (特征折, 切片) 只读一次（(100,152,256) float16 ≈ 8 MB/切片）
        w_cache: dict[tuple[str, str], torch.Tensor] = {}
        n_cache: dict[tuple[str, str], Optional[int]] = {}
        e_cache: dict[tuple[str, int, str], Optional[torch.Tensor]] = {}
        checked_ff: set[str] = set()
        for arm, mm in models.items():
            for i, (model, meta) in enumerate(mm):
                ff = feature_fold_of(meta, target)
                for s in slices:
                    if not uses_E(model):
                        e_cache[(arm, i, s)] = None
                        continue
                    assert ff, f"[{target}/{arm}] 吃 E 的臂却没有 feature_fold"
                    if ff not in checked_ff:     # 铁律 6：用特征前核对该特征折
                        assert_fold_features_compliant(outdir, ff)
                        checked_ff.add(ff)
                    if (ff, s) not in w_cache:
                        w = load_slice_windows(outdir, ff, s, device=device)
                        assert w is not None, f"folds/{ff}/{s} 特征缺失"
                        w_cache[(ff, s)] = w
                        n_cache[(ff, s)] = slice_n_instances(outdir, ff, s)
                    where = (f"folds/{ff}/task_features/{s}.npz"
                             f"（{target}/{arm}_s{i}）")
                    # 两道闸都**逐模型**核对（缓存按 (特征折, 切片) 复用，而 meta
                    # 逐臂/逐种子不同）：维度不符会崩在 SetTF 的矩阵乘里，条数不符
                    # 则完全不崩——后者只能靠这里拦。
                    check_windows_dim(w_cache[(ff, s)], model, meta=meta,
                                      where=where)
                    check_windows_count(w_cache[(ff, s)], meta,
                                        n_instances=n_cache[(ff, s)], where=where)
                    with torch.no_grad():
                        e_cache[(arm, i, s)] = model.encode_tasks(
                            [w_cache[(ff, s)]])[0]
        vec_cache: dict[str, torch.Tensor] = {}
        for s in slices:
            cands = catalog[s]
            vec_cache[s] = torch.as_tensor(
                np.stack([encoder.encode(c) for _o, c, _k in cands]),
                dtype=torch.float32, device=device)

        per_target: dict[str, dict] = {}
        for b in budgets:
            ctx_row = torch.as_tensor(np.asarray(pq.ctx_vector(b, budgets)),
                                      dtype=torch.float32, device=device)
            per_b: dict[str, dict] = {}
            for s in slices:
                keys = [k for _o, _c, k in catalog[s]]
                per_arm: dict[str, list[str]] = {}
                for arm, mm in models.items():
                    ranked, _S = rank_candidates(
                        [m for m, _ in mm],
                        [e_cache[(arm, i, s)] for i in range(len(mm))],
                        vec_cache[s], keys, ctx_row=ctx_row)
                    per_arm[arm] = ranked
                per_b[s] = per_arm
            per_target[str(b)] = per_b
        rankings[target] = per_target
        print(f"[{target}] 排名 {len(slices)} 切片 × {len(budgets)} 预算 × "
              f"{len(models)} 臂", flush=True)

    # 子集跑（--folds/--arms）必须并入既有文件：直接覆写会静默抹掉别的折/臂
    merged = write_json_merged(outdir / "rankings.json", rankings)
    n = sum(len(pb) for pt in rankings.values() for pb in pt.values())
    n_all = sum(len(pb) for pt in merged.values() for pb in pt.values())
    print(f"--rank 收官：本次 {n} 个搜索实例，文件累计 {n_all} → "
          f"{outdir / 'rankings.json'}", flush=True)
    return merged


# ================================================================ --analyze
def assert_truth_task_type(cache: dict,
                           expect: str = TASK_CLASSIFICATION) -> None:
    """真值行的 `task_type` 必须全是 classification（铁律 8 的方向承诺）。

    regret / oracle / bottom-quartile 全部按 "R' 越大越优" 算死在本脚本里
    （`instance_arms` 直接取 max）。混进一条 forecasting 行（MSE 越小越优），
    那一格的"最优"就是最差——而数字看着完全正常。方向要改，得先经
    `autocls.task_types.is_better` 重写这一整段，不是放宽这道断言。
    """
    seen = {str(r.get("task_type")) for r in cache.values()}
    bad = sorted(t for t in seen if t != expect)
    if bad:
        raise ValueError(
            f"真值缓存里出现非 {expect} 的 task_type {bad}——本脚本的 regret/"
            "oracle 方向按分类（R' 越大越优）写死（铁律 8）；换任务类型须先把"
            " instance_arms/regret_of 改成经 task_types.is_better 判优劣")


def build_instances(rankings: dict, catalog: dict, ggs_key: str, cache: dict, *,
                    allow_incomplete: bool = False
                    ) -> tuple[list[dict], list[dict]]:
    """rankings + 真值缓存 → per-instance 臂/regret 行（+ 被跳过的实例清单）。"""
    rows, skipped = [], []
    for target in sorted(rankings):
        for b_key in sorted(rankings[target], key=lambda x: int(x)):
            b = int(b_key)
            for s in sorted(rankings[target][b_key]):
                ranked = rankings[target][b_key][s]
                keys = [k for _o, _c, k in catalog[s]]
                truth, missing = {}, []
                for k in keys:
                    rec = cache.get((s, k, b))
                    rp = None if rec is None else rec.get("r_prime")
                    if rp is None or not np.isfinite(rp):
                        missing.append(k)
                    else:
                        truth[k] = float(rp)
                if missing:
                    skipped.append({"fold": target, "slice": s, "budget": b,
                                    "missing_keys": missing})
                    continue
                grec = cache.get((s, ggs_key, b))
                ggs_r = None if grec is None else grec.get("r_prime")
                ggs_r = (float(ggs_r) if ggs_r is not None
                         and np.isfinite(ggs_r) else None)
                arms = instance_arms(truth, ranked, ggs_r)
                # seconds 可能是 JSON null（OOM 标记行 / 老缓存）：.get 的默认值
                # 对"键在但值是 None"无效，必须显式判空，否则 float(None) 直接炸
                secs = {k: _as_float(cache[(s, k, b)].get("seconds"))
                        for k in keys}
                top3 = ranked["main"][:K_TOP]
                rows.append({
                    "fold": target, "dataset": pq.dataset_of(s), "slice": s,
                    "budget": b, "arms": arms, "regret": regret_of(arms),
                    "picks": {ARM_REPORT[a][0]: ranked[a][0]
                              for a in ranked if a in ARM_REPORT},
                    "top3": {a: list(ranked[a][:K_TOP]) for a in ranked},
                    "bottom_quartile_hit": {
                        ARM_REPORT[a][0]: bool(bottom_quartile_hit(truth, ranked[a][0]))
                        for a in ranked if a in ARM_REPORT},
                    "seconds_top3": float(np.nansum([secs[k] for k in top3])),
                    "seconds_catalog": float(np.nansum(list(secs.values()))),
                })
    if skipped and not allow_incomplete:
        head = "; ".join(f"{e['fold']}/{e['slice']}@{e['budget']} 缺 "
                         f"{len(e['missing_keys'])} 键" for e in skipped[:5])
        raise ValueError(
            f"{len(skipped)} 个搜索实例真值不全（先跑完 32 号 --fill）：{head}"
            "…… 确要按现有实例出报表请加 --allow-incomplete（实例集会变，"
            "D1/D2 的 n 必须在报表里声明）")
    return rows, skipped


def cmd_analyze(outdir: Path, *, budgets: Sequence[int],
                allow_incomplete: bool = False) -> dict:
    """臂/regret/D1/D2/D3/分层表/成本表 → search_analysis.json（纯聚合，零 GPU）。"""
    outdir = Path(outdir)
    space = CLSSpace.from_yaml()
    rankings = json.loads((outdir / "rankings.json").read_text(encoding="utf-8"))
    all_slices = [s for t in rankings for s in pq.pq_slice_ids(t)]
    catalog, _ggs, ggs_key = load_catalog(outdir, space, slices=all_slices)
    cache = load_rprime_cache(outdir / "rprime_cache.jsonl")
    assert_truth_task_type(cache)
    rows, skipped = build_instances(rankings, catalog, ggs_key, cache,
                                    allow_incomplete=allow_incomplete)
    if not rows:
        raise ValueError("没有可用搜索实例——先跑 --rank 与 32 号 --fill")

    full_iters = int(_read_yaml("configs/train.yaml")["full_iters"])
    n_expected = expected_instances(budgets)
    gates = {"D1": gate(rows, "tclsc1", "random1"),
             "D2": gate(rows, "tclsc1", "clsb1"),
             "aux_tclsc3_vs_random3": gate(rows, "tclsc3", "random3"),
             "aux_tclsc1_vs_ggs": gate(rows, "tclsc1", "ggs")}
    # 满格自检（协议 §5 的 D1/D2 分母写死 126）：不满格收回过/不过结论
    for code in ("D1", "D2"):
        mark_incomplete(gates[code], n_expected=n_expected,
                        allow_incomplete=allow_incomplete)
    if len(rows) != n_expected:
        print(f"⚠️ 搜索实例 {len(rows)}/{n_expected} 不满格（协议 §5 的 126 = 7 折 × "
              f"6 切片 × 3 预算）——D1/D2 "
              + ("已按 --allow-incomplete 照常判定，报表须声明 n"
                 if allow_incomplete else "不作过/不过判定"), flush=True)
    avoid_worst = {a: int(sum(1 for r in rows
                              if r["bottom_quartile_hit"].get(a)))
                   for a in PICK_ARMS}

    # D3（进化臂）：evo_picks.json 在场才算；R' 先查 evo_cache 再回落主缓存
    d3 = None
    picks_path = outdir / "evo_picks.json"
    if picks_path.exists():
        picks = json.loads(picks_path.read_text(encoding="utf-8"))
        evo_cache = load_rprime_cache(outdir / "evo_cache.jsonl")

        def r_of(fold: str, b: int, key: str):
            s = f"{fold}_00"
            rec = evo_cache.get((s, key, b)) or cache.get((s, key, b))
            rp = None if rec is None else rec.get("r_prime")
            return None if rp is None or not np.isfinite(rp) else float(rp)

        def oracle_of(fold: str, b: int):
            s = f"{fold}_00"
            vals = [cache[(s, k, b)]["r_prime"] for _o, _c, k in catalog.get(s, [])
                    if (s, k, b) in cache
                    and cache[(s, k, b)].get("r_prime") is not None]
            return max(float(v) for v in vals) if vals else None

        d3 = d3_report(picks, r_of, oracle_of)
        n_full = len(pq.FOLD_TARGETS) * len(budgets)
        d3["incomplete"] = bool(d3["n_instances"] != n_full)
        if d3["incomplete"]:
            print(f"⚠️ D3 只有 {d3['n_instances']}/{n_full} 个实例，而门限 "
                  f"{D3_MIN_HITS} 是按满格 {n_full} 写死的（协议 §5，跑完前不可读作过/不过）",
                  flush=True)
            if not allow_incomplete:
                for a in d3["per_arm"].values():
                    a["pass"] = None
        gates["D3"] = {"pass": d3["per_arm"]["main"]["pass"],
                       "hits": d3["per_arm"]["main"]["hits"],
                       "n": d3["per_arm"]["main"]["n"], "n_expected": n_full,
                       "incomplete": d3["incomplete"],
                       "min_hits": d3["min_hits"], "margin": d3["margin"]}

    out = {
        "protocol": "results/mx3_pq/PROTOCOL.md",
        "budgets": [int(b) for b in budgets],
        "n_instances": len(rows),
        "n_skipped": len(skipped),
        "skipped_instances": skipped,
        "gates": gates,
        "mean_regret": {"pooled": mean_regret(rows)},
        "by_budget": strata_table(rows, "budget"),
        "by_dataset": strata_table(rows, "dataset"),
        "avoid_worst": avoid_worst,
        "cost": cost_table(rows, full_iters=full_iters),
        "d3": d3,
        "per_instance": rows,
    }
    out["predictions"] = {
        "P7_D1": {"prior": 0.85, "hit": gates["D1"]["pass"], "p": gates["D1"]["p"]},
        "P7_D2": {"prior": 0.20, "hit": gates["D2"]["pass"], "p": gates["D2"]["p"]},
        "P8_D3": {"prior": 0.35,
                  "hit": (None if d3 is None else d3["per_arm"]["main"]["pass"])},
    }
    (outdir / "search_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    m = out["mean_regret"]["pooled"]
    print("\n===== MX-3-PQ 搜索判读 =====")
    print(f"实例 {len(rows)}（跳过 {len(skipped)}）｜平均 regret："
          + " | ".join(f"{a} {m[a]:.4f}" for a in
                       ("tclsc1", "tclsc3", "clsb1", "clsonly1", "ggs",
                        "random1", "random3", "bottom") if a in m))
    for code in ("D1", "D2", "D3"):
        g = gates.get(code)
        if g is None:
            print(f"  {code}：未计算（缺 evo_picks.json）")
        elif code == "D3":
            verdict = ("未判定" if g["pass"] is None
                       else ("过" if g["pass"] else "不过"))
            print(f"  D3：{g['hits']}/{g['n']}（满格 {g['n_expected']}）≥ "
                  f"{g['min_hits']} → {verdict}")
        elif g.get("reason"):
            pv = "" if g.get("p") is None else f"（描述性 p={g['p']:.4g}）"
            print(f"  {code}：无判定{pv}——{g['reason']}")
        else:
            print(f"  {code}（{g['arm_a']} < {g['arm_b']}）p={g['p']:.4g} "
                  f"n={g['n']} → {'过' if g['pass'] else '不过'}")
    print(f"  avoid-worst（垫底四分位命中）：" +
          " ".join(f"{a} {n}/{len(rows)}" for a, n in avoid_worst.items()))
    for b_key, c in out["cost"].items():
        sp = c["speedup_vs_direct_full"]
        print(f"  成本@{b_key}: top3 {c['zero_shot_top3_s']:.0f}s vs 目录直搜 "
              f"{c['catalog_direct_s']:.0f}s（外推 full {c['catalog_direct_full_est_s']:.0f}s"
              + (f"，{sp:.1f}×）" if sp else "）"))
    print(f"→ {outdir / 'search_analysis.json'}", flush=True)
    return out


# ================================================================ --evolve
def make_score_pairs(models, e_rows, ctx_row, *, device: str = "cpu"):
    """闭包出 evolutionary_search 要的 `score_pairs(vecs_a, vecs_b) -> P(a≻b)`。

    多种子聚合取**平均 P**（brief 二选一，此处选平均并记录理由）：进化搜索里每代
    要对上万对打分，round(P) 的多数票在 3 个种子下频繁出现 1:1:1 之外的平票，需要
    再引入一层破平规则；平均 P 是连续量、无平票、且与 `symmetrize` 的双向均值同
    精神。目录排名侧仍用多数票（那里只有 20 条候选、且与 25 号口径对齐）。
    """
    def score_pairs(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        A = torch.as_tensor(np.asarray(vecs_a), dtype=torch.float32, device=device)
        B = torch.as_tensor(np.asarray(vecs_b), dtype=torch.float32, device=device)
        n = int(A.shape[0])
        acc = np.zeros(n, dtype=float)
        with torch.no_grad():
            for model, e_row in zip(models, e_rows):
                E, ctx = model_task_inputs(model, e_row, ctx_row)
                E = None if E is None else E.expand(n, -1)
                ctx = None if ctx is None else ctx.expand(n, -1)
                acc += model(A, B, E, ctx).cpu().numpy()
        return acc / len(models)

    return score_pairs


def cmd_evolve(outdir: Path, *, budgets: Sequence[int], folds: Sequence[str],
               arms: Sequence[str] = EVO_ARMS, device: str = "cpu",
               evolution: Optional[dict] = None) -> dict:
    """21 实例（7 折 × 3 预算）× 2 臂的进化搜索 top-3 → evo_picks.json。

    条件化在目标的切片 `_00`（协议 §5 D3 的判读锚点就是该切片的目录 oracle）。
    """
    outdir = Path(outdir)
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    cfg = dict(evolution or pq.load_pq_config()["evolution"])
    picks: dict[str, dict] = {}
    for target in folds:
        models = load_fold_models(outdir, target, arms=arms, device=device,
                                  budgets=budgets)
        if not models:
            print(f"⚠️ [{target}] 无进化臂模型，跳过", flush=True)
            continue
        checked_ff: set[str] = set()
        slice00 = f"{target}_00"
        per_b: dict[str, dict] = {}
        for b in budgets:
            ctx_row = torch.as_tensor(np.asarray(pq.ctx_vector(b, budgets)),
                                      dtype=torch.float32, device=device)
            per_arm: dict[str, list] = {}
            for arm in arms:
                mm = models.get(arm)
                if not mm:
                    continue
                e_rows = []
                for model, meta in mm:
                    if not uses_E(model):
                        e_rows.append(None)
                        continue
                    ff = feature_fold_of(meta, target) or target
                    if ff not in checked_ff:     # 铁律 6：用特征前核对该特征折
                        assert_fold_features_compliant(outdir, ff)
                        checked_ff.add(ff)
                    w = load_slice_windows(outdir, ff, slice00, device=device,
                                           meta=meta)
                    assert w is not None, \
                        f"folds/{ff}/{slice00} 特征缺失（33 号 --fit-folds）"
                    check_windows_dim(
                        w, model, meta=meta,
                        where=f"folds/{ff}/task_features/{slice00}.npz"
                              f"（{target}/{arm}）")
                    with torch.no_grad():
                        e_rows.append(model.encode_tasks([w])[0])
                t0 = time.time()
                per_arm[arm] = evolutionary_search(
                    make_score_pairs([m for m, _ in mm], e_rows, ctx_row,
                                     device=device),
                    space, encoder, **cfg, rng=evo_rng(target, b))
                print(f"[{target}@{b} {arm}] top-{len(per_arm[arm])} "
                      f"({time.time() - t0:.1f}s) "
                      f"best={per_arm[arm][0]['cls_key'][:8]}", flush=True)
            per_b[str(b)] = per_arm
        picks[target] = per_b
    merged = write_json_merged(outdir / "evo_picks.json", picks)
    n = sum(1 for pt in picks.values() for pb in pt.values() for _ in pb)
    n_all = sum(1 for pt in merged.values() for pb in pt.values() for _ in pb)
    print(f"--evolve 收官：本次 {n} 个（实例 × 臂）格，文件累计 {n_all} → "
          f"{outdir / 'evo_picks.json'}", flush=True)
    return merged


# ================================================================ --evaluate-evo
def cmd_evaluate_evo(outdir: Path, *, budgets: Sequence[int],
                     device: Optional[str] = None, on_oom: str = ON_OOM_SKIP,
                     inflight: Optional[Path] = None,
                     oom_log: Optional[Path] = None) -> int:
    """进化 pick 的真 R'（一条轨迹三截点，原子落盘）→ evo_cache.jsonl。

    与 32 号 --fill 同口径：canonical 种子、`checkpoints=budgets`、
    `guarded_run_multi` 多行原子写入 + inflight 打点。

    跳过键 = (切片, cls_key, 预算)，在 evo_cache.jsonl **与** 主缓存
    rprime_cache.jsonl 上都查（进化恰好选中目录里的 CLS 时，canonical 种子相同 →
    同一条轨迹，重跑只会得到逐位相同的 R' 并写出重复缓存键）。因此本阶段
    **恒幂等**：跳过不受开关控制，`--resume` 只为与 32 号的 CLI 对称而保留。
    """
    outdir = Path(outdir)
    picks_path = outdir / "evo_picks.json"
    if not picks_path.exists():
        raise FileNotFoundError(f"{picks_path} 不存在——先跑 --evolve")
    picks = json.loads(picks_path.read_text(encoding="utf-8"))
    budgets = tuple(int(b) for b in budgets)
    worklist = evo_worklist(picks, budgets=budgets)
    evo_path = outdir / "evo_cache.jsonl"
    inflight = Path(inflight) if inflight else outdir / "inflight.txt"
    oom_log = Path(oom_log) if oom_log else outdir / "oom_failures.jsonl"

    # 四态 resume（32 号 trajectory_state 同一份定义）：done/terminal 都不再跑。
    # **terminal 是本轮修复的重点**：看门狗隔离的轨迹只留一行 max-budget 的
    # `r_prime=null + error`，按"三个预算齐了才算完"判会永远判成 partial →
    # 每次 resume 都重跑，且每次都再追加一行 max-budget，缓存越滚越脏。
    index = {**load_rprime_cache(outdir / "rprime_cache.jsonl"),
             **load_rprime_cache(evo_path)}
    states = {(e["slice"], e["cls_key"]):
              trajectory_state(index, e["slice"], e["cls_key"], budgets)
              for e in worklist}
    todo = [e for e in worklist
            if states[(e["slice"], e["cls_key"])] not in (STATE_DONE, STATE_TERMINAL)]
    tally = defaultdict(int)
    for st in states.values():
        tally[st] += 1
    print(f"进化真评估：工单 {len(worklist)} 条轨迹（去重后），状态 "
          + " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
          + f"，待跑 {len(todo)}", flush=True)
    if tally.get(STATE_TERMINAL):
        print(f"  ↳ {tally[STATE_TERMINAL]} 条 terminal（r_prime=null + error："
              "训练发散或看门狗隔离）按 interfaces.md 视作整条轨迹终结，不重试",
              flush=True)
    if not todo:
        return 0

    labels_cfg = _read_yaml("configs/labels.yaml")
    train_cfg = _read_yaml("configs/train.yaml")
    datasets = _load_datasets(sorted({e["fold"] for e in todo}))
    n_ok = 0
    # Windows 教训：loky spawn 吃掉 R' 单条 99% 耗时 → 下游评估强制线程后端（勿删）
    with parallel_backend("threading", n_jobs=1):
        for i, entry in enumerate(todo):
            ds = datasets[entry["fold"]]
            task = next((t for t in ds.slices if t.task_id == entry["slice"]), None)
            assert task is not None, f"{entry['fold']} 无切片 {entry['slice']}"
            ws, ss = pq.canonical_seeds(entry["slice"], entry["cls_key"])

            def mk(iters: int, rec: dict, _e=entry, _t=task, _ws=ws) -> dict:
                return make_evo_row(_e, _t, rec, iters=iters, weight_seed=_ws)

            rows = guarded_run_multi(
                EVALUATE_TRAJECTORY, mk, cache_path=evo_path,
                inflight_path=inflight, task_id=entry["slice"],
                cls_key=entry["cls_key"], iters_list=budgets, on_oom=on_oom,
                oom_log_path=oom_log,
                context={"dataset": entry["fold"], "origin": entry["origin"],
                         "budget_searched": entry["budget_searched"]},
                log=lambda m, _i=i: print(f"[evo {_i + 1}/{len(todo)}] {m}",
                                          flush=True),
                task=task, cls_cfg=CLS(**entry["cls"]), X=ds.X, y=ds.y,
                weight_seed=ws, split_seed=ss, checkpoints=budgets,
                eval_ratio=labels_cfg["eval_ratio"],
                batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
                device=device)
            if rows:
                n_ok += 1
                print(f"[evo {i + 1}/{len(todo)}] {entry['slice']}/"
                      f"{entry['cls_key'][:8]} {entry['origin']} → "
                      + " ".join(f"@{r['iters']}={r['r_prime']}" for r in rows),
                      flush=True)
    print(f"--evaluate-evo 收官：{n_ok}/{len(todo)} 条轨迹落盘 → {evo_path}",
          flush=True)
    return n_ok


# ================================================================ smoke fixture
# --smoke 会写的 34 相关产物（--fresh 清理范围；**永不含** R' 缓存与 32 号产物）
# --fresh 绝不删的东西。判据不是"看着像派生产物"，而是**这些文件可能装着真跑
# 的结果**：R' 缓存与噪声探针要真 GPU；`evo_cache.jsonl` 装的是 --evaluate-evo
# 的真 R' 行（轮 3 评审：它原先被当成"派生产物"删掉了，那是真的数据损失）；
# rankings/evo_picks/search_analysis 可能来自真产物目录上的真跑。
# fixture **从不写**这几个里的任何一个（除 rprime_cache/cls_catalog 在 absent
# 档合成时会写，那两个会进 written 清单，按清单删仍是"只删自己写过的"）。
SMOKE_KEEP = ("rprime_cache.jsonl", "noise_probe.jsonl", "cls_catalog.json",
              "tasks.json", "delta_calibration.json",
              "evo_cache.jsonl", "evo_picks.json", "rankings.json",
              "search_analysis.json")


def fixture_manifest(outdir: Path) -> Optional[set[str]]:
    """本脚本的 fixture 在该目录写过哪些文件（`SMOKE_FIXTURE.json` 的 `written`）。

    返回 None = **没有清单**：要么目录不是我们造的（33 号的真产物），要么是轮 2
    之前的老标记。"没有清单"必须一路传导到 `--fresh`（拒绝删任何东西）与
    `mixed_state_message`（不推荐 `--fresh`）——删自己没写过的文件是轮 3 评审
    点名的事故。
    """
    marker = Path(outdir) / FIXTURE_MARKER
    if not marker.exists():
        return None
    try:
        written = json.loads(marker.read_text(encoding="utf-8")).get("written")
    except json.JSONDecodeError:
        return None
    return set(written) if written else None


def budgets_from_models(outdir: Path, targets: Sequence[str]
                        ) -> Optional[tuple[int, ...]]:
    """从既有 main 臂存档的 meta 读预算口径（复用 33 号产物时以文件为准）。

    33 号的 smoke 口径将来若从 4/8/12 改成别的，34 号不该因为自己写死
    SMOKE_BUDGETS 就把整目录判成"真假混用"——预算是**读**出来的，不是猜的。
    """
    mdir = Path(outdir) / "models"
    for t in targets:
        for p in sorted(mdir.glob(f"{t}/main_s*.pt")):
            try:
                _m, meta = load_pq_model(p)
            except Exception:                      # noqa: BLE001（体检不抛）
                continue
            got = tuple(int(b) for b in meta.get("budgets", ()))
            if got:
                return got
    return None


def mixed_state_message(outdir: Path, state: dict) -> str:
    """`mixed` 档的报错正文：逐条点名哪儿对不上 + 与该目录**实情相符**的出路。

    出路② `--fresh` **只在该目录确有 fixture 清单时**才提（轮 3 评审）：清单不在
    就说明这儿的 models/folds 不是我们写的，推荐 --fresh 等于教人删别人的 GPU
    产物——而且 --fresh 自己也会拒绝执行。
    """
    lines = [f"{outdir} 处于**真假混用**状态（模型 {state['n_models']} / 特征 "
             f"{state['n_features']}），34 号不能在这上面整套复用。逐条："]
    lines += [f"  - {r}" for r in state["reasons"]]
    lines.append("出路：")
    lines.append("  ① 先把 33 号跑完：`python scripts/33_pq_comparator_loo.py "
                 f"--smoke --outdir {outdir}`（让全目录同源）；")
    if state.get("has_manifest"):
        lines.append("  ② 或清掉本脚本 fixture 写过的合成件重来："
                     "`python scripts/34_pq_search.py --smoke --fresh "
                     f"--outdir {outdir}`（**只删清单内的 fixture 文件**，"
                     "33 号的 models/folds 与各类缓存一律不动）。")
    else:
        lines.append(f"  ② 或换一个空目录跑合成 smoke：`--smoke --outdir <新目录>`。"
                     "（本目录没有 fixture 清单 = 这些产物不是 34 号写的，"
                     "`--fresh` 会拒绝执行，也**不要**手删 models/folds——"
                     "那是 33 号的 GPU 产物。）")
    lines.append("34 号绝不逐文件补洞——16 维合成窗口撞 256 维真 SetTF 正是这么来的。")
    return "\n".join(lines)


def smoke_dir_state(outdir: Path, *, targets: Sequence[str],
                    budgets: Sequence[int]) -> dict:
    """`--smoke` 前的目录体检：`absent` / `reusable` / `mixed`。

    评审轮 2 的事故根因：原来的 fixture 逐文件"存在就跳过"，于是 33 号只产出
    `main_s0` + 切片 `_00` 特征时，34 号把缺的 `main_s1/s2` 与切片 `_01` 特征
    用 16 维合成件补齐——同一个目录里 256 维真模型与 16 维假特征并存，
    一路跑到 SetTF 的矩阵乘法才炸。**修法是按目录整体决策，不再逐文件补洞。**

    - `absent`：没有任何折靶可用 → 整套合成；
    - `reusable`：至少一个折靶自洽（有 main 臂模型，且它在**自己的特征折目录**
      里至少有一个切片特征、维度与该模型 set_encoder 一致）→ **一件不合成**，
      只跑这些折靶；
    - `mixed`：某折靶**真假混着**——维度不符 / 预算口径不同 / 模型读不出来 /
      fixture 清单里的文件与非清单文件在同一折里共存。

    两条轮 3 评审钉死的口径：
    1. **特征归属按折目录限定**。33 号把**每个数据集的全部切片**都编进**每个**
       折目录（`folds/Handwriting/task_features/UniMiB-SHAR_00.npz` 是正常产物），
       所以"按切片名跨全部折目录找特征"会把 UniMiB 的特征算到 UniMiB 头上，
       于是"只训了 Handwriting"这种**合法的部分 33 跑**被误判成 mixed，报错再把
       操作者引向 `--fresh`——那会删掉 GPU 产的折特征。只在
       `folds/<该模型的 feature_fold>/task_features/` 下找该折靶自己的切片。
    2. **该折靶没有 main 模型 = `skipped`**（不管有多少特征）。33 号只训了部分
       折是常态；没有模型就没法排名，跳过即可，不是"混用"。

    返回 {state, reasons, usable, skipped, skipped_why, slices,
          n_models, n_features, has_manifest}。
    """
    outdir = Path(outdir)
    mdir, fdir = outdir / "models", outdir / "folds"
    have_models = sorted(mdir.glob("*/*.pt")) if mdir.exists() else []
    have_feats = sorted(fdir.glob("*/task_features/*.npz")) if fdir.exists() else []
    manifest = fixture_manifest(outdir)
    base = {"n_models": len(have_models), "n_features": len(have_feats),
            "has_manifest": manifest is not None}
    if not have_models and not have_feats:
        return {"state": "absent", "reasons": [], "slices": {}, "usable": [],
                "skipped": list(targets), "skipped_why": {}, **base}

    def _is_fixture(p: Path) -> Optional[bool]:
        """该文件是不是本脚本 fixture 写的（无清单 → None = 无从判断）。"""
        if manifest is None:
            return None
        return p.relative_to(outdir).as_posix() in manifest

    reasons: list[str] = []
    resolved: dict[str, list[str]] = {}
    skipped: list[str] = []
    skipped_why: dict[str, str] = {}
    for t in targets:
        main_paths = [p for p in have_models
                      if p.parent.name == t and p.name.startswith("main_s")]
        t_models = [p for p in have_models if p.parent.name == t]
        if not main_paths:
            # 口径 2：没有 main 模型就是"这折没训"，与它有多少特征无关。
            skipped.append(t)
            skipped_why[t] = (
                f"缺 models/{t}/main_s*.pt"
                + (f"（该折另有 {len(t_models)} 个非 main 臂模型）" if t_models
                   else "（该折一件模型都没有）"))
            continue
        try:
            model, meta = load_pq_model(main_paths[0])
        except Exception as e:                        # noqa: BLE001（体检要吞异常）
            reasons.append(f"{t}：models/{t}/{main_paths[0].name} 读不出来（{e}）")
            continue
        got_budgets = tuple(int(b) for b in meta.get("budgets", ()))
        if got_budgets and got_budgets != tuple(int(b) for b in budgets):
            reasons.append(f"{t}：模型 budgets={list(got_budgets)} ≠ 本次 "
                           f"{list(budgets)}（ctx one-hot 口径不一致）")
            continue
        # 口径 1：只在该模型自己的特征折目录下找**该折靶自己的**切片
        ff = feature_fold_of(meta, t) or t
        want_dim = set_encoder_dim_input(model, meta)
        ok_slices: list[str] = []
        used: list[Path] = list(main_paths)
        for sid in pq.pq_slice_ids(t):
            p = fdir / ff / "task_features" / f"{sid}.npz"
            if not p.exists():
                continue
            used.append(p)
            with np.load(p) as z:
                d = int(z["windows_repr"].shape[-1])
            if want_dim is not None and d != want_dim:
                reasons.append(f"{t}：{p.relative_to(outdir).as_posix()} 的特征维 "
                               f"{d} ≠ main 臂 set_encoder 的 {want_dim}"
                               "（真假产物混用）")
            else:
                ok_slices.append(sid)
        # 清单判混用：同一折里 fixture 写的与非 fixture 的文件共存
        flags = {_is_fixture(p) for p in used}
        if {True, False} <= flags:
            synth = [p.relative_to(outdir).as_posix() for p in used if _is_fixture(p)]
            real = [p.relative_to(outdir).as_posix()
                    for p in used if _is_fixture(p) is False]
            reasons.append(
                f"{t}：同一折里 fixture 合成件与非 fixture 产物共存——"
                f"合成 {synth[:3]}{'…' if len(synth) > 3 else ''} vs "
                f"既有 {real[:3]}{'…' if len(real) > 3 else ''}")
        if not ok_slices and t not in {r.split("：")[0] for r in reasons}:
            skipped.append(t)
            skipped_why[t] = f"folds/{ff}/task_features/ 下没有它自己的切片特征"
            continue
        resolved[t] = ok_slices
    usable = sorted(k for k, v in resolved.items() if v)
    if reasons:
        state = "mixed"
    elif usable:
        state = "reusable"
    else:
        state = "absent"           # 目录里的东西与本次折靶无关
    return {"state": state, "reasons": reasons, "slices": resolved,
            "usable": usable, "skipped": skipped, "skipped_why": skipped_why,
            **base}


def fresh_smoke_dir(outdir: Path) -> list[str]:
    """`--fresh`：**只删本脚本 fixture 自己写过的文件**，一件多的都不删。

    轮 3 评审的事故：上一版在"没有清单"时退回 `rmtree(models/)` 与
    `rmtree(folds/)`，而那两处装的是 33 号用 GPU 产的折 TS2Vec 与特征；
    `evo_cache.jsonl`（--evaluate-evo 的真 R' 行）也被当成派生产物删掉。
    更糟的是上一版 `mixed` 误判会把操作者**主动引向** `--fresh`。

    现在的铁规矩：
    - 没有 `SMOKE_FIXTURE.json` 的 `written` 清单 → **拒绝执行**（退出并说明），
      绝不猜哪些文件"看起来像合成的"；
    - 有清单 → 逐条删清单里的文件（外加 SMOKE_KEEP 双保险），最后删标记本身；
    - 目录里其余一切（33 号的 models/folds、32 号的缓存、真跑出的
      rankings/evo_cache/…）**原样不动**。清单外的东西不归 --fresh 管。
    """
    outdir = Path(outdir)
    assert_not_production(outdir, "--fresh（清理合成 fixture）")
    manifest = fixture_manifest(outdir)
    if manifest is None:
        sys.exit(
            f"--fresh 拒绝执行：{outdir} 里没有 {FIXTURE_MARKER} 的 `written` "
            "清单，也就是说本脚本的 fixture 没在这儿写过东西——目录里的 "
            "models/ 与 folds/ 是别人（33 号，GPU 产物）的，"
            "`rprime_cache.jsonl` / `evo_cache.jsonl` 装的是真 R' 行。"
            "--fresh 只删自己写过的文件，绝不按「看着像合成的」去猜。"
            "要重造合成 fixture 请换一个空目录 --outdir；"
            "要清理这个目录请自行确认后手动删除。")
    removed: list[str] = []
    for rel in sorted(manifest):
        p = outdir / rel
        if p.name in SMOKE_KEEP:
            continue              # 双保险：清单里若混进真跑产物一律不删
        if p.exists() and p.is_file():
            p.unlink()
            removed.append(rel)
    marker = outdir / FIXTURE_MARKER
    if marker.exists():
        marker.unlink()
        removed.append(FIXTURE_MARKER)
    kept = sorted(n for n in SMOKE_KEEP if (outdir / n).exists())
    print(f"--fresh：按清单删掉 {len(removed)} 个 fixture 文件；"
          f"保留 {kept}（真跑产物 / 造一次要真 GPU），清单外一切未动", flush=True)
    return removed


def _fake_rprime(slice_id: str, key: str, b: int) -> float:
    """合成 R'（仅 fixture 用）：(切片, cls, 预算) 的确定性伪随机，非任何真实结果。"""
    rng = np.random.default_rng(
        [abs(hash(slice_id)) % (2 ** 31), int(key[:6], 16), int(b)])
    return float(np.round(0.30 + 0.55 * rng.random(), 4))


def write_smoke_fixture(outdir: Path, *, targets: Sequence[str] = SMOKE_TARGETS,
                        n_slices: int = SMOKE_SLICES,
                        budgets: Sequence[int] = SMOKE_BUDGETS,
                        dim_input: int = 16, nz: int = 16, n_windows: int = 4,
                        t_max: int = 6, seed: int = 0) -> dict:
    """在 outdir 里合成 Task 5/6 的产物替身（目录 / 真值 / 折特征 / 三臂模型）。

    **为什么脚本里带 fixture 而不是只放测试里**：brief 要求 `--smoke` 在 Task 5/6
    的产物缺席时也能端到端跑通 34 号自己的四个阶段；把替身写在这里，`--smoke`
    与单测共用同一份构造，不会出现"测试过了但 --smoke 走另一条路"的漂移。
    产物**全是合成数字**（`_fake_rprime`），落在 --outdir 指定的临时目录，
    绝不允许写进 results/mx3_pq（写入端首行硬闸）；任何报表都不得引用。

    **整体决策，不逐文件补洞**（评审轮 2）：进来先过 `smoke_dir_state`——目录里
    已有 33 号产物就一件不合成（`reusable`，由 cmd_smoke 走复用分支），真假混用
    一律报错。本函数只负责 `absent` 那一档：**要么整套写，要么不写**。32 号的
    产物（目录 / R' 缓存）单独判断：它们与模型维度无关，缺了就补。
    写出的每个文件都记进标记的 `written` 清单，`--fresh` 据此精确回滚。
    """
    outdir = Path(outdir)
    assert_not_production(outdir, "write_smoke_fixture（合成 fixture）")
    written: list[str] = []

    def _note(p):
        written.append(Path(p).relative_to(outdir).as_posix())
        return p

    state = smoke_dir_state(outdir, targets=targets, budgets=budgets)
    if state["state"] == "mixed":
        raise ValueError(mixed_state_message(outdir, state))
    if state["state"] == "reusable":
        raise ValueError(
            f"{outdir} 已有 33 号的真产物（模型 {state['n_models']} / 特征 "
            f"{state['n_features']} / 可用折靶 {state['usable']}）——不该再合成"
            "替身（真假混用正是评审轮 2 的事故）。cmd_smoke 会自动走复用分支；"
            "要强制重造请先 `--smoke --fresh`")
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    space = CLSSpace.from_yaml()
    encoder = CLSEncoder(space)
    cls_dim = encoder.dim
    budgets = tuple(int(b) for b in budgets)
    slices = {t: pq.pq_slice_ids(t)[:n_slices] for t in targets}
    ggs = pq.ggs_cls()
    ggs_key = cls_key(ggs)

    # ① 目录（10 shared + 10 random，构造式与 pq_common 同源）
    cat_path = outdir / "cls_catalog.json"
    shared = load_probe_cls(pq.SHARED_CATALOG, pq.N_SHARED)[0]
    catalog = {s: pq.catalog_for_slice(space, s, shared)
               for t in targets for s in slices[t]}
    if not cat_path.exists():
        _note(cat_path).write_text(json.dumps({
            "shared": [{"cls_key": k, "cls": dataclasses.asdict(c)}
                       for c, k in shared],
            "random": {s: [{"cls_key": k, "cls": dataclasses.asdict(c)}
                           for o, c, k in cands if o == ORIGIN_RANDOM]
                       for s, cands in catalog.items()},
            "ggs": {"cls_key": ggs_key, "cls": dataclasses.asdict(ggs)},
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ② 合成真值网格（20 条目录 CLS + GGS）× 预算
    cache_path = outdir / "rprime_cache.jsonl"
    if not cache_path.exists():
        with open(_note(cache_path), "w", encoding="utf-8", newline="\n") as fh:
            for t in targets:
                for s in slices[t]:
                    entries = list(catalog[s]) + [("ggs", ggs, ggs_key)]
                    for origin, c, k in entries:
                        for i, b in enumerate(budgets):
                            fh.write(json.dumps({
                                "dataset": t, "task_id": s, "tag": "smoke",
                                "task_type": "classification", "classes": [0, 1],
                                "cls_key": k, "origin": origin, "seed": 0,
                                "iters": int(b),
                                "checkpoint_iters": [int(x) for x in budgets],
                                "trajectory_id": f"{s}:{k}:0",
                                "r_prime": _fake_rprime(s, k, b), "acc": 0.5,
                                "f1": 0.5, "svm_c": 1.0, "n_train": 60,
                                "n_eval": 20, "weight_seed": 0, "split_seed": 0,
                                "device_name": "smoke", "swa": False,
                                "seconds": 1.0 + i, "trajectory_seconds": 3.0,
                                "error": None,
                                "cls": dataclasses.asdict(c)}) + "\n")

    # ③ 折特征（合成 windows_repr）+ ④ 三臂模型（随机初始化）
    ctx_d = pq.ctx_dim(budgets)
    for ti, t in enumerate(targets):
        feat_dir = outdir / "folds" / t / "task_features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        # 折 TS2Vec 的 meta（合成）：fit 池按铁律 6 剔掉考侧，使用点自检因此在
        # smoke 上也真的执行一遍，而不是走"meta 缺席只告警"的旁路
        meta_path = outdir / "folds" / t / "shared_ts2vec.meta.json"
        if not meta_path.exists():
            _note(meta_path).write_text(json.dumps({
                "fold": t, "fit_datasets": sorted(pq.fold_split(t)[0]),
                "synthetic": True,
                "note": "合成 fixture 的折 meta——无对应权重文件，禁止引用",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        for si, s in enumerate(slices[t]):
            p = feat_dir / f"{s}.npz"
            if p.exists():
                continue
            rng = np.random.default_rng([seed, ti, si])
            np.savez_compressed(
                _note(p),
                windows_repr=rng.standard_normal(
                    (n_windows, t_max, dim_input)).astype(np.float16),
                E_init=rng.standard_normal(nz).astype(np.float32),
                META=np.array([np.log10(t_max), 0.0, 2.0, 1.0], dtype=np.float32))
        mdir = outdir / "models" / t
        mdir.mkdir(parents=True, exist_ok=True)
        for arm in RANK_ARMS:
            branch = {"main": "E+ctx", "cls_b": "ctx", "cls_only": "none"}[arm]
            for s_ in ARM_SEEDS[arm]:
                path = mdir / f"{arm}_s{s_}.pt"
                if path.exists():
                    continue
                torch.manual_seed(1000 * ti + 10 * s_ + RANK_ARMS.index(arm))
                set_enc = (TaskSetEncoder(dim_input=dim_input, nz=nz)
                           if branch == "E+ctx" else None)
                model = TCLSC(cls_dim=cls_dim, set_encoder=set_enc,
                              embedding_dim=8, fz=8,
                              ctx_dim=0 if branch == "none" else ctx_d,
                              task_branch=branch)
                torch.save({"state_dict": model.state_dict(), "meta": {
                    "arm": arm, "seed": s_, "fold": t, "task_branch": branch,
                    "ctx_dim": 0 if branch == "none" else ctx_d,
                    "cls_dim": cls_dim, "embedding_dim": 8, "fz": 8,
                    "nz": nz if branch == "E+ctx" else None,
                    "set_encoder": "settf" if branch == "E+ctx" else None,
                    "e_source": "fold" if branch == "E+ctx" else None,
                    "feature_fold": t if branch == "E+ctx" else None,
                    "settf_windows": n_windows, "settf_t_max": t_max,
                    "budgets": [int(b) for b in budgets],
                    "ctx_encoding": "onehot", "dim_input": dim_input,
                    "synthetic": True}}, _note(path))

    marker = {"synthetic": True, "source": "34_pq_search.write_smoke_fixture",
              "targets": list(targets),
              "slices": {t: slices[t] for t in targets},
              "budgets": [int(b) for b in budgets],
              "dim_input": int(dim_input), "nz": int(nz),
              "written": sorted(written),
              "note": "合成 fixture（Task 5/6 产物替身）——数字全是假的，禁止引用"}
    (outdir / FIXTURE_MARKER).write_text(
        json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
    return marker


def cmd_smoke(outdir: Path, *, device: str = "cpu", fresh: bool = False,
              budgets: Optional[Sequence[int]] = None) -> None:
    """fixture（合成或复用 33 号真产物）→ --rank → --evolve → --analyze。

    **目录整体决策**（评审轮 2 的修法）：
    - `absent`：整套合成，走 SMOKE_BUDGETS（4/8/12）小口径；
    - `reusable`：33 号产物齐备 → **一件不合成**，预算改从模型 meta 读
      （33 号的 smoke 也是 4/8/12，但口径以文件为准，不由本函数假设）；
    - `mixed`：报清单并退出（详见 mixed_state_message）。
    `--fresh` 先清掉 34 号自己的合成/派生产物（永不动 R' 缓存），再按上面判定。
    """
    outdir = Path(outdir)
    assert_not_production(outdir, "--smoke（合成数字会污染真值缓存，铁律 3）")
    if fresh:
        fresh_smoke_dir(outdir)
    # 预算口径：命令行 > 既有 main 存档的 meta（复用 33 号产物时以文件为准）
    #  > SMOKE_BUDGETS（自己合成时的小口径）
    bud = (tuple(int(b) for b in budgets) if budgets
           else budgets_from_models(outdir, SMOKE_TARGETS) or SMOKE_BUDGETS)
    state = smoke_dir_state(outdir, targets=SMOKE_TARGETS, budgets=bud)
    if state["state"] == "mixed":
        sys.exit(mixed_state_message(outdir, state))
    folds = SMOKE_TARGETS
    # 目录里的东西是谁造的：有本脚本的标记 = 上一轮自己合成的，否则当 33 号真产物
    own = (outdir / FIXTURE_MARKER).exists()
    whose = "本目录既有的合成 fixture" if own else " 33 号既有产物"
    if state["state"] == "reusable":
        folds = tuple(state["usable"])
        n_sl = sum(len(v) for v in state["slices"].values())
        print(f"[smoke] 复用{whose}（模型 {state['n_models']} / 特征 "
              f"{state['n_features']} / 折靶 {list(folds)} / 可用切片 {n_sl}），"
              "**不写任何合成件**", flush=True)
        if state["skipped"]:
            print(f"[smoke] 折靶 {state['skipped']} 在该目录里没有任何产物，"
                  "本次不参与（33 号只跑了一部分折时的正常情形）", flush=True)
    else:
        write_smoke_fixture(outdir, budgets=bud)
    cmd_rank(outdir, budgets=bud, folds=folds, device=device)
    cmd_evolve(outdir, budgets=bud, folds=folds, device=device,
               evolution=dict(SMOKE_EVOLUTION))
    cmd_analyze(outdir, budgets=bud)
    tag = (f"复用{whose}" + ("，禁止引用" if own else "")
           if state["state"] == "reusable"
           else "全部数字为合成 fixture，禁止引用")
    print(f"--smoke 收官（{tag}）→ {outdir}", flush=True)


# ================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rank", action="store_true", help="126 实例 × 臂的排名")
    ap.add_argument("--analyze", action="store_true", help="臂/regret/D1/D2/D3")
    ap.add_argument("--evolve", action="store_true", help="21 实例 × 2 臂进化 top-3")
    ap.add_argument("--evaluate-evo", action="store_true",
                    help="进化 pick 的真 R'（GPU）")
    ap.add_argument("--smoke", action="store_true", help="合成 fixture 端到端")
    ap.add_argument("--fresh", action="store_true",
                    help="--smoke 前清掉 34 号的合成/派生产物（models/folds/"
                         "rankings/evo_*/analysis），**不动 R' 缓存**")
    ap.add_argument("--resume", action="store_true",
                    help="（--evaluate-evo 恒按缓存键跳过；本开关只为与 32 号 CLI 对称）")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="--analyze：真值不全的实例跳过而非 fail-fast")
    ap.add_argument("--outdir", default=str(pq.OUT),
                    help="冒烟务必改到别处（铁律 3）")
    ap.add_argument("--folds", nargs="+", default=None,
                    help=f"折靶子集（默认全部 {len(pq.FOLD_TARGETS)} 个）")
    ap.add_argument("--arms", nargs="+", default=None,
                    help=f"排名/进化臂子集（默认 {list(RANK_ARMS)} / {list(EVO_ARMS)}）")
    ap.add_argument("--budgets", nargs="+", type=int, default=None,
                    help="预算档覆盖（默认读 fixture 标记，再回落 pq_common.BUDGETS）")
    ap.add_argument("--evo-ks", type=int, default=None)
    ap.add_argument("--evo-kp", type=int, default=None)
    ap.add_argument("--evo-generations", type=int, default=None)
    ap.add_argument("--on-oom", choices=[ON_OOM_SKIP, ON_OOM_RECORD],
                    default=ON_OOM_SKIP)
    ap.add_argument("--inflight", default=None)
    ap.add_argument("--oom-log", default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    if not (args.rank or args.analyze or args.evolve or args.evaluate_evo
            or args.smoke):
        ap.error("至少指定 --rank / --analyze / --evolve / --evaluate-evo / --smoke")
    if args.smoke:
        cmd_smoke(outdir, device=args.device, fresh=args.fresh,
                  budgets=args.budgets)
        return
    if args.fresh:
        ap.error("--fresh 只在 --smoke 下有意义（它清的是 smoke 目录的合成产物）")

    budgets = resolve_budgets(outdir, args.budgets)
    folds = tuple(args.folds) if args.folds else pq.FOLD_TARGETS
    unknown = [f for f in folds if f not in pq.FOLD_TARGETS]
    if unknown:
        ap.error(f"未知折靶 {unknown}（合法：{list(pq.FOLD_TARGETS)}）")

    if args.arms:
        legal = set(RANK_ARMS) | set(EVO_ARMS)
        bad = [a for a in args.arms if a not in legal]
        if bad:
            ap.error(f"未知臂 {bad}——合法臂：--rank {list(RANK_ARMS)} / "
                     f"--evolve {list(EVO_ARMS)}")
    if args.rank:
        arms = tuple(args.arms) if args.arms else RANK_ARMS
        bad = [a for a in arms if a not in RANK_ARMS]
        if bad:
            ap.error(f"--rank 不支持臂 {bad}（合法：{list(RANK_ARMS)}）")
        cmd_rank(outdir, budgets=budgets, folds=folds, arms=arms,
                 device=args.device)
    if args.evolve:
        arms = tuple(args.arms) if args.arms else EVO_ARMS
        bad = [a for a in arms if a not in EVO_ARMS]
        if bad:
            ap.error(f"--evolve 不支持臂 {bad}（合法：{list(EVO_ARMS)}）")
        cfg = dict(pq.load_pq_config()["evolution"])
        for k, v in (("ks", args.evo_ks), ("kp", args.evo_kp),
                     ("generations", args.evo_generations)):
            if v is not None:
                cfg[k] = int(v)
        cmd_evolve(outdir, budgets=budgets, folds=folds, arms=arms,
                   device=args.device, evolution=cfg)
    if args.evaluate_evo:
        if torch.get_num_threads() > 1:
            print("⚠️ 建议 OMP_NUM_THREADS=1 启动（造 R' 是 CPU-bound）", flush=True)
        cmd_evaluate_evo(outdir, budgets=budgets, device=args.device,
                         on_oom=args.on_oom, inflight=args.inflight,
                         oom_log=args.oom_log)
    if args.analyze:
        cmd_analyze(outdir, budgets=budgets,
                    allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
