"""任务嵌入 E = SetTransformer(共享冻结 TS2Vec(样本序列集))（TECHNICAL_DESIGN.md 模块 5）。

流程对应 AutoCTS++ 的 exps/generate_task_feature.py（TS2Vec 训练+编码+采样）与
NAS_Net/AHC/ahc_engine.py::set_encode（两级 Set-Transformer 聚合），分类任务
语义差异见各模块 docstring。

【定稿语义】单变量共享冻结 TS2Vec（AutoCTS++ generate_task_feature.py 原味）：
- fit_shared_ts2vec(source_data)：在源池上无监督训练一份 input_dims=1 的
  TS2Vec（通道展开成独立实例），训完即冻结；
- embed_task(X, shared_ts2vec)：所有任务共用这份编码器，通道当独立实例编码、
  表征跨通道取均值（← generate_task_feature.py L137-140 对 N 个传感器取 mean），
  天然支持任意通道数（HAR 9ch / Epilepsy 1ch / 通道子集），不做零填充。

为什么不是"每任务重训 TS2Vec"（旧默认，已移除）：诊断
（results/task_embed_sanity/ + scripts/diag_task_embed_shared.py）表明每任务
独立重训时 E 被训练随机性主导（对比表征的基任意性：两次独立训练学到的表征
空间相差任意旋转），同数据不同种子的 E 距离 ≥ 不同任务间距离，区分力≈0；
共享冻结消除基噪声后 TS2Vec 级特征区分力清晰（同构任务 cos≈0.99、类别子集
间 ≈0.65、通道子集间 ≈0.92）。

【铁律 6，泄漏防护】共享 TS2Vec 仅在源池训练、冻结后编码所有任务；目标池/
未见数据集绝不参与 TS2Vec 训练。API 层面强制："用哪批数据训练"必须经
fit_shared_ts2vec(source_data) 显式传入，embed_task 只接受已冻结的单变量
编码器，不能隐式复用全量数据。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..task_types import TASK_CLASSIFICATION, validate_task_type
from .set_transformer import TaskSetEncoder
from .ts2vec_wrapper import TS2VecWrapper

__all__ = [
    "TS2VecWrapper",
    "TaskSetEncoder",
    "fit_shared_ts2vec",
    "encode_task_windows",
    "embed_task",
    "task_windows",
    "pool_time_axis",
]


def task_windows(X: np.ndarray, task_type: str) -> np.ndarray:
    """按 task_type 分发任务窗口构造（铁律 7 分发点：task_embed 窗口逻辑）。

    classification: 窗口 = 样本序列本身（分类样本天然是定长分段，无 P+Q 滑窗
    语义），原样返回；forecasting: 应做 S=P+Q 滑窗（AutoCTS++
    generate_task_feature.py 的 sample_split 语义）——未实现，接入时在此补分支。
    调用方（scripts/2_generate_labels.py 等）必须经本函数取窗口再喂
    encode_task_windows/embed_task，禁止隐式假设"就是分类"。
    """
    validate_task_type(task_type)
    if task_type == TASK_CLASSIFICATION:
        return X
    raise NotImplementedError(
        f"task_type={task_type!r} 的窗口逻辑未实现——forecasting 为 S=P+Q 滑窗"
        "（generate_task_feature.py sample_split 语义），接入时在此补分支"
    )


def pool_time_axis(reprs: np.ndarray, t_max: int) -> np.ndarray:
    """时间轴封顶：(m, T, D) → (m, min(T, t_max), D)（MX-3-PQ 协议 §3）。

    动机：Set-Transformer（IntraSetPool/InterSetPool，set_transformer.py）的注意力
    代价 ∝ T²——每个窗口表征矩阵是 400·T² 个 float（batch 400 的经验上限口径），
    T 一旦过大（长序列端如 EthanolConcentration L=1024）显存/时间双爆。封顶阈值
    由 configs/labels.yaml: pq.settf_t_max（=152，覆盖本轮源池 8 个数据集的窗口长度）
    唯一给定，本函数不硬编码该值。

    T <= t_max 时原样返回（identity，不额外分配/拷贝语义上等价于恒等映射）；
    否则沿时间轴做自适应平均池化到恰好 t_max 个 bin（torch.nn.functional.
    adaptive_avg_pool1d 语义：T 整除 t_max 时为非重叠等长 bin 的均值，不整除时
    bin 边界按 adaptive pooling 的标准取整规则）。计算在 torch（float32）里做，
    结果转回 numpy 并转换回输入的原始 dtype（半精度 windows_repr 落盘产物同样
    适用，不因内部升精度而改变对外 dtype 契约）。
    """
    reprs = np.asarray(reprs)
    assert reprs.ndim == 3, f"reprs 应为 (m, T, D)，得到 {reprs.shape}"
    _, t, _ = reprs.shape
    if t <= t_max:
        return reprs

    dtype = reprs.dtype
    x = torch.from_numpy(reprs.astype(np.float32, copy=False)).permute(0, 2, 1)  # (m, D, T)
    pooled = F.adaptive_avg_pool1d(x, t_max).permute(0, 2, 1).contiguous()  # (m, t_max, D)
    return pooled.numpy().astype(dtype, copy=False)


def fit_shared_ts2vec(
    source_data: np.ndarray,
    *,
    n_iters: int,
    output_dims: int = 256,   # 与 AHC set_encoder dim_input=256 对齐（ahc_engine.py）
    hidden_dims: int = 64,
    depth: int = 10,
    batch_size: int = 128,    # STD_BATCH_SIZE（configs/train.yaml）
    device: Optional[str] = None,
    seed: int = 0,
) -> TS2VecWrapper:
    """在【源池】上训练共享单变量 TS2Vec，训完冻结返回（铁律 6 的唯一训练入口）。

    source_data: (N, T, C) 源池样本。通道展开成 (N*C, T, 1) 单变量实例池训练
    （AutoCTS++ generate_task_feature.py 对每个传感器独立编码的原味语义）。
    n_iters: 参数更新次数，必须显式传参（同铁律 2 的 iters 口径）。

    泄漏防护（铁律 6）：source_data 只能是源池数据——路线 B（HAR 内 leave-out）
    为"用于造 label 的切片"，留出测试切片不得混入；路线 A（UEA 多数据集）为
    源池数据集，目标池（含 HAR/Epilepsy）零参与。返回的编码器已冻结，再 fit
    会直接报错。
    """
    assert source_data.ndim == 3, f"source_data 应为 (N, T, C)，得到 {source_data.shape}"
    n, t, c = source_data.shape
    pool = source_data.transpose(0, 2, 1).reshape(n * c, t, 1)
    wrapper = TS2VecWrapper(
        input_dims=1, output_dims=output_dims, hidden_dims=hidden_dims,
        depth=depth, device=device, batch_size=batch_size, seed=seed,
    )
    wrapper.fit(pool, n_iters=n_iters)
    wrapper.freeze()
    return wrapper


def encode_task_windows(
    X: np.ndarray,
    shared_ts2vec: TS2VecWrapper,
    *,
    sample_num: int = 100,    # ← generate_task_feature.py --sample_num 默认 100
    seed: int = 0,
    encoding_window=None,
    allow_unfit: bool = False,
) -> np.ndarray:
    """采样任务窗口并用共享冻结 TS2Vec 编码：通道当独立实例、表征跨通道均值。

    X: (N, T, C)，任意通道数。采样 min(sample_num, N) 个窗口（分类语义：
    窗口=样本序列本身，无 P+Q 滑窗），每窗口按通道展开为 C 个单变量序列编码，
    再对表征取通道均值（← generate_task_feature.py L137-140）。
    返回 (m, T, D)；encoding_window='full_series' 时返回 (m, D)——后者是
    sanity check 的"TS2Vec 级基线"（不过 SetTF 验证任务分布信号）。

    allow_unfit=True：放宽"必须先 fit_shared_ts2vec 训练过"的门槛，只保留
    frozen（仍不放宽 input_dims==1）——供 MX-3-PQ A2「w/o TS2Vec」消融臂使用：
    该臂用随机初始化（未训练）的单变量 TS2Vec 权重直接冻结后当特征提取器，
    衡量"TS2Vec 预训练本身"对任务嵌入的贡献。仍要求显式 freeze()（不能隐式
    绕过铁律 6 的"训练数据来源必须可追溯"约束——未训练也必须显式声明"不训练"
    意图，而非漏调 freeze）。
    """
    assert X.ndim == 3, f"X 应为 (N, T, C)，得到 {X.shape}"
    assert shared_ts2vec.input_dims == 1, \
        "只接受单变量共享 TS2Vec（fit_shared_ts2vec 产出）；多变量+零填充方案已被诊断否决"
    if allow_unfit:
        assert shared_ts2vec.frozen, \
            "allow_unfit=True 仍要求已冻结（w/o TS2Vec 消融臂用随机初始化编码器，" \
            "但必须显式 freeze() 声明「不训练」意图）"
    else:
        assert shared_ts2vec.frozen and shared_ts2vec.n_iters > 0, \
            "共享 TS2Vec 必须先经 fit_shared_ts2vec 在源池上训练并冻结（铁律 6）"

    rng = np.random.default_rng(seed)
    m = min(sample_num, X.shape[0])
    idx = np.sort(rng.choice(X.shape[0], size=m, replace=False))
    s = X[idx]  # (m, T, C)

    m_, t, c = s.shape
    uni = s.transpose(0, 2, 1).reshape(m_ * c, t, 1)
    reprs = shared_ts2vec.encode(uni, encoding_window=encoding_window)
    return reprs.reshape(m_, c, *reprs.shape[1:]).mean(axis=1)  # 跨通道均值


def embed_task(
    X: np.ndarray,
    shared_ts2vec: TS2VecWrapper,
    *,
    sample_num: int = 100,
    nz: int = 256,             # 任务嵌入维（ahc_engine.py self.nz）
    set_encoder: Optional[TaskSetEncoder] = None,
    seed: int = 0,
) -> np.ndarray:
    """把一个（切片）任务的无标签样本集 X (N, T, C) 编码为定长任务向量 E (nz,)。

    流程（← generate_task_feature.py 主流程 + ahc_engine.py::set_encode）：
    1. encode_task_windows：采样窗口 → 共享冻结单变量 TS2Vec 编码 → 跨通道均值
       得 (m, T, D)。不接触任何标签——E 必须是"数据分布特征"，否则破坏
       zero-shot 迁移前提。
    2. TaskSetEncoder 两级聚合（Eq.10-12）→ E (nz,)。

    shared_ts2vec 必须是 fit_shared_ts2vec 在源池上训练冻结的编码器（铁律 6）。
    set_encoder 为 None 时在 torch.manual_seed(seed) 下新建：同 seed 多次调用得到
    同一份初始化权重，保证跨任务 E 可比（sanity check 用法）。正式流程（M3 起）
    应显式传入同一个实例——该编码器最终与 T-CLSC 联合训练。
    """
    device = shared_ts2vec.device  # 与共享编码器同设备，避免张量搬运不一致

    if set_encoder is None:
        torch.manual_seed(seed)
        set_encoder = TaskSetEncoder(dim_input=shared_ts2vec.output_dims, nz=nz)
    set_encoder = set_encoder.to(device).eval()

    reprs = encode_task_windows(X, shared_ts2vec, sample_num=sample_num, seed=seed)

    with torch.no_grad():
        e = set_encoder(torch.from_numpy(reprs).to(torch.float32).to(device))
    return e.cpu().numpy()
