# 复现闸（PROTOCOL §8）失守诊断 —— 2026-09-03 01:30，pod hlp3nsvl4fpg5o（RTX 5090，driver 580.126.20）

`32 --reproduce-check` 结果（`reproduction_check.json`）：

| 参照 | n | 命中（≤tol） | 门 | 中位 \|Δ\| | 符号均值 | 最大 \|Δ\| |
|---|---:|---:|---|---:|---:|---:|
| source_pool_v2（同 5090，8 月另一台 pod） | 175 | **0.760** | ≥0.90 @0.02 | 0.0000 | −0.0003 | 0.292 |
| wide_panel / 5090 子集 | 107 | **0.897** | ≥0.90 @0.02 | 0.0000 | −0.0003 | 0.137 |
| wide_panel / 4060 子集（混机口径） | 73 | 0.945 ✅ | ≥0.85 @0.03 | 0.0000 | +0.0028 | 0.072 |

## 诊断（零 GPU，`results/mx3_pq/pod_pull/` 上复算）

1. **62% 的格逐位相同**（source_pool_v2 子集 exact0 = 0.623），中位 \|Δ\| = 0，符号均值 ≈ 0：没有系统性偏移。口径错误（种子/切分/iters 不同）会让 exact0 ≈ 0 并整体偏移——排除。
2. **失配集中在已知的环境敏感 CLS**：20be4f74（命中 0.31）、8b87d66f（0.31）、65627517（0.54）、cffb5503（0.62）；稳定 CLS（4f7d4135 / f48e5da1 / e490b88e）命中 1.00、max \|Δ\| ≤ 0.013。这四个键与 `eval_size_probe` / `swa_probe` 记录的"上下文敏感 CLS"重合（memory `eval-size-probe-sqrtn-confirmed`）。
3. **按数据集**：短序列（PS/UWave/UM）1.00 且逐位相同；长序列 Cricket 0.65 / Ethanol 0.74 / HMD 0.78。长序列上 cuDNN 卷积算法选择随 driver/上下文变化，混沌轨迹一旦分叉即等价于换种子。
4. **失配幅度落在本机重复噪声尺度内**：本次 k=3 探针（同机不同 weight seed）的 \|ΔR'\| p90：Cricket 0.144 / HMD 0.104 / Handwriting 0.138 / Ethanol 0.051；复现失配的 p90 = 0.042（source_pool_v2）。即"另一台 pod 上同种子重跑"对敏感 CLS 的效果 ≤ "同机换种子"。
5. 跨设备（4060 vs 5090）子集反而 0.945 命中——说明决定因素不是硬件型号而是敏感 CLS 的轨迹混沌（与 `rprime-nondeterminism-on-5090` 记忆一致）。

## 裁决

- §8 的 0.90@0.02 门是按 2026-08-01 闸门（短数据集、30 iters、88/90 逐位）外推的，未考虑 290 iters 长序列上敏感 CLS 的分叉。**失守原因已定位为跨 pod 轨迹分叉，不是口径错误**；R' 协议（种子/切分/iters/评估器）本身经 62% 逐位一致与稳定 CLS 100% 命中验证无误。
- 本轮 δ 由本 pod 自身的 k=3 探针标定（`delta_calibration.json`），label 在本轮内部自洽；敏感 CLS 的跨 pod 不可复现性写入 limitation。
- 因此以 `--allow-reproduction-breach` 放行进入比较器阶段，本文件与 `reproduction_check.json` 一并归档；REPORT 须如实报"§8 复现闸失守（0.760/0.897），诊断见本文"。
