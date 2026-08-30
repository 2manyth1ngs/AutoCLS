# MX-2 预埋备忘：端到端展示的已拍板决定（2026-08-30，届时协议必含）

出处：用户 2026-08-30 拍板（出路 X 路线图，与 memory `outlet-x-confirmed-roadmap`
同款记录）。**本备忘只预埋决定，不执行**；MX-2 立项时其 PROTOCOL.md 必须逐条
包含以下内容，缺一即协议不完整。

## 1. 扩源对照臂（用户质疑单源样本量后新增）

leave-HW-out 方向的 comparator 源池做**两臂对比**，同一 HW 测试集：

- **臂 A（UM-only）**：现状基线，R7 跨数据集准确率 0.6929
  （`results/decisive_e_ablation`，UM→HW）；
- **臂 B（UM + 15 UEA）**：源数据集 1→15、任务 16→约 46，用 @290 既有缓存
  重造配对 label（**零 GPU**——R' 全部现成，只重跑 make_pairs + 训练）。

预期（先验，届时协议再细化）：强化 avoid-worst（破坏样本更多样）；好半区
分辨力未必涨（任务信号地板结论不因扩源改变）。

## 2. 端到端指标形状

按**任务**报 28 个搜索实例（HW 12 + UM 16）的 regret±std + 成本表
（vs 直接搜索上界 / GGS / 随机 / 垫底；成本差 1–2 数量级）。

## 3. 外部锚

零样本选中的 CLS 跑**全量 Handwriting 标准协议**，与 TS2Vec 论文 UEA 公开数字
同表（T-Loss / TS-TCC / TNC 一并列出）；可选加全量 UCI-HAR vs AutoCL。
区别于切片舞台：这是标准协议全量口径（CLAUDE.md 关键决策 5 的外部锚定位）。

## 4. 同机铁规

同一张对照表（top-K / 上界 / GGS / 随机 / 垫底）的**全部 R' 必须同一硬件跑**
——GPU 非确定性会污染 regret（memory `rprime-nondeterminism-on-5090` /
`gpu-desktop-load-breaks-determinism`）。验收：记录逐条 `device_name`
（evaluate_rprime 2026-08-10 起自带），混机即重跑。

重活（直接搜索上界 + 外部锚，约一天 GPU）计划租云（RunPod），遵守
memory `runpod-pod-reprovision-checklist` 与 OMP=1 / loky 孤儿进程教训
（`route-a-throughput-omp1-and-worker-scaling` / `loky-orphans-after-sigkill`）。

## 5. 三个诚实折扣（写进论文，勿隐去）

1. 双向 LOO 的 n=2（UM↔HW 两方向）；
2. "目标可评"是事后已知（由 MX-1 可行性预测层在推理侧接住，写清分工）；
3. 方法层知识渗漏（协议/超参在两数据集上共同演化过）。
