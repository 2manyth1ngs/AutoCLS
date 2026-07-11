# CLAUDE.md — AutoCLS

Zero-Shot NAS for Contrastive Learning Strategy Search。把 **AutoCL 的对比学习策略空间（CLS）** 作为搜索对象，用 **AutoCTS++ 的零样本比较器机制** 来搜，encoder 固定。目标：给未见时序数据集/下游任务，零样本排出最优 CLS。

完整设计见 `TECHNICAL_DESIGN.md`。

本机python环境：Anaconda 下的名为pytorch的虚拟环境

## 核心概念
- **CLS**：一个对比学习策略 = 19 个离散字段的配置向量（AutoCL Table 1）。这是"要搜的架构"。
- **T-CLSC**：比较器，输入 `(任务嵌入 E, cls_a, cls_b)` 输出二值 y（哪个 CLS 在该任务上更好）。核心创新点。
- **early-validation**：造训练 label 时只对比预训练 575 iters（≈5 epoch 等量）就评估，用来快速排名。**只有最终 top-K 才做完整预训练。**
- **固定 encoder**：架构固定、每个 CLS 各自重新预训练一份权重（不是权重冻结）。

## 关键决策（已定，勿改）
1. CLS 编码用 **向量 + MLP**（方案 A），不用图/GIN。
2. encoder 两者可切换：`dilated_cnn`（默认，TS2Vec式）/ `transformer`，由 `configs/encoder.yaml` 控制。
3. 第一阶段只做**分类**（HAR / Epilepsy）。

## 铁律
1. `configs/space.yaml` 是策略空间**唯一真源**，代码不得硬编码取值集合。
2. `cl_trainer` 的 `max_iters`（参数更新次数）必须显式传参，early-val 与最终评估禁止混用；跨任务训练量必须用 iters 而非 epoch 保证一致（early=290, full=2320，STD_BATCH_SIZE=128，固化在 `configs/train.yaml`。基准为 M1 实测 bs=128（40 epoch=2320 iters，ACC 0.9372），非早期 bs=64 误算；early:full 保持 M2 验证的 1:8 比例）。
3. 所有 `(task, cls) → R'` 结果必须落盘缓存，脚本支持 `--resume`。这是最贵的一步。
4. `task_embed/` 相关代码必须先读 AutoCTS++ 真实源码再适配，注释标注对应原文件+公式号。
5. 先写 `tests/` 再写实现。与论文有出入的补全须注释标注"论文未给全，按 X 语义补全"（如附录 A 的 resize/rescale 公式被截断）。

## 复用 AutoCTS++（repo: decisionintelligence/AutoCTS_plusplus）
- `Ts2vec_model/` → `autocls/task_embed/ts2vec_wrapper.py`
- `AHC/set_encoder/` → `autocls/task_embed/set_transformer.py`
- `AHC/ahc_engine.py` → `autocls/comparator/tclsc_engine.py`（GIN 支路换成 MLP）
- `exps/random_search.py` → `scripts/3_pretrain_tclsc.py` + `4_zero_shot_search.py`

## 数据流
```
离线: 源数据→切片成任务 t_i →(TS2Vec+SetTF 得 E_i);(采样CLS)
     →对(t_i,cls)跑 k=5 预训练+下游评估得 R' →配对 (E_i,cls_a,cls_b,y) →BCE训T-CLSC
在线: 未见D→E →进化算法+T-CLSC两两比较排名→top-K完整预训练评估→选 cls*
```

## 里程碑（按序，M1/M2 是关键闸口）
- M0 骨架+configs+空间/编码器+单测
- **M1** 单 CLS 在 HAR 上能预训练+出 acc（复现 AutoCL 量级 ≈0.9+）
- **M2** early-val(k=5) 与完整训练的 acc 排名 Spearman 相关性 ← **风险最高，优先跑**
- M3 T-CLSC 留出任务配对准确率 >0.5
- M4 零样本搜索接近直接搜索上界、耗时低 1-2 个数量级
- M5 消融（w/o TS2Vec / SetTF / shared samples）+ 耗时表

## 目录
`autocls/{space,contrastive,encoders,task_embed,comparator,downstream,search,data}` + `scripts/1-5_*.py` + `configs/*.yaml` + `tests/`
