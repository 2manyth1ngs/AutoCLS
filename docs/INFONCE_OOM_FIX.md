# InfoNCE / 跨尺度 / triplet 的"从表征逐块造相似度"显存修复（2026-09-03）

姊妹文档：`docs/TRIPLET_OOM_FIX.md`（只分块了 triplet 的 hinge，仍物化整块 sim）。

## 触发
MX-3-PQ 上云（RTX 5090，31.4 GiB）900 条轨迹里 12 条 OOM，全部 `temporal=True`：
Ethanol（L=1024）10 条 + SCP2（L=1152）2 条，`resize` 把编码长度 n 抬到 2T 以上
（SCP2 最坏 n=4378）。temporal InfoNCE 物化 `(B=128, n, n)` 的 sim / logits / log-softmax /
梯度各一份（n=2304 时每份 2.7 GB）。`expandable_segments` 重试同样失败——是真容量缺口。

## 两轮修法（`autocls/contrastive/losses.py`、`pair_construction.py`）
1. **第一轮（sim 已物化时的分块）**：`_masked_info_nce` 在 `sim.numel() > INFONCE_CHUNK_ELEMS`
   （默认 2^26，env `AUTOCLS_INFONCE_CHUNK_ELEMS`）时按前导维分块，手写反向
   `∂/∂logits = pos − k_i·softmax`，全局正样本计数做分母。**只救回 6/12**：sim 与其梯度缓冲
   仍是全尺寸（2×n²）。
2. **第二轮（从表征逐块造 sim）**：`masked_info_nce_from_reprs` / `masked_triplet_from_reprs`
   在自定义 autograd.Function 内按 batch 块 `step = INFONCE_CHUNK_ELEMS // n²` 现算
   `pairwise_sim(z[b0:b1])`，反向在局部 autograd 里重算该块并 `autograd.grad` 回 z；
   `pair_construction._maybe_from_reprs` 在 temporal 的每个层级与 cross-scale 建 sim 之前
   分派。**峰值不再含 n² 项**（实测 峰值 ≈ 7 × 块预算 + O(B·n·C)）；外推 B=128、n=4378：
   19.7 → 4.2 GiB。triplet 复用既有 `_TripletHingeSum` 于块内，hinge 语义与 85 条旧测试不动。

## 数值保证（`tests/test_infonce_chunking.py`，208 用例）
- 阈值以下四条路径与改动前实现 `torch.equal`（值与梯度）；40 iters 真实轨迹末权重逐位相同。
- 阈值以上 396 组（temporal/cross-scale × info_nce/triplet × dot/cosine/euclidean × fp32/fp64
  × 4 种块大小）值与两视图梯度 vs 冻结 oracle：fp32 ≤ 1.87 eps、fp64 ≤ 2.61 eps；fp64 gradcheck 通过。
- CUDA 用例断言整调用峰值 < 0.5 × 全尺寸 sim 字节数（证明从未建过 (B,n,n)）。

## 口径说明
- 只有 `B·n·n > 2^26` 的格才走新路径；已缓存的格永不重算，故缓存 R' 不受影响。
  MX-3-PQ 的 12 个补格用新路径算出，与旧路径差 ~2 eps/步，远低于 GPU 非确定性（见
  `results/mx3_pq/reproduction_diagnosis.md`）。
- 收窄块预算（如 2^24）是纯显存旋钮；`configs/runtime.yaml: env` 与 `scripts/pq_pod/*.sh` 同步。
