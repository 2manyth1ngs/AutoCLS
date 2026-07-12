# M3（路线 B 收尾）：T-CLSC 机制验证 + E 消融

**日期** 2026-07-12 | **代码** `autocls/comparator/` + `scripts/3_pretrain_tclsc.py` | **产物** `results/tclsc/{report,history}.json, tclsc.pt`（不入库，可由脚本复现）

## 定位（先读这个）

路线 B 是**机制工程验证，不是 zero-shot 证据**：留出任务与源任务同分布（同一
HAR、同一切片协议），只能验证"泛化到新任务切片"，不能验证"泛化到新数据集"。
真 zero-shot 结果由路线 A（多 UEA 数据集、源/目标池分离，HAR/Epilepsy 作目标池）
产出。本实验同时是对"16 个同源任务多样性是否足以支撑 task-aware"的直接检验。

## 数据

- 源池 16 任务 × 6 CLS（3 shared + 3 random），96 条 R'，0 失败 → 145 静态配对
- 留出池 8 任务 × 6 CLS（shared 与源池同一批），48 条 R' → 71 配对（评估双向=142）
- δ=0.02 弃样约 40%（两池一致；高分簇密集，与 M2 结论相符）
- 训练/测试任务零重叠（铁律 6）；早停协议 early_iters=290（铁律 2）
- 缓存：`results/labels/rprime_cache{,_holdout}.jsonl`（铁律 3，--resume 可续）

## 训练

结构对齐 AutoCTS++ `ahc_engine.py::AHC`（GIN→共享 MLP 为唯一实质改动）；
动态配对（`AHC_DataLoader_linear` 语义）+ curriculum（前 40 epoch 仅 shared）+
SetTF 联合训练（`set_encoder_init.pt` 初始化）。超参 `configs/pretrain_tclsc.yaml`
（lr 1e-4、bs 64、200 epoch），训练 123s（RTX 4060 Laptop）。

## 验收 1：机制能训起来 ✓

| 指标 | 值 |
|---|---|
| BCE（epoch 0 → 199） | 0.6927 → 0.2061 |
| 训练集配对准确率（n=290 双向） | 0.9034 |
| **留出任务配对准确率（n=142 双向）** | **0.8662** |
| 显著性（单向 n=71，H0=0.5，单侧二项） | p = 2.3e-10 |

## 验收 2：E 消融（task-aware 检验）✗

不重训、只改测试输入：

| 档 | 留出准确率 | Δ vs 基线 |
|---|---|---|
| 基线（各任务用自己的 E） | 0.8662 | — |
| shuffle E（任务错位 derangement，seed 0–4 共 5 个排列） | 0.8662（五个种子全同） | **+0.0000** |
| zero E（E 置零） | 0.8451 | +0.0211 |

shuffle 档 loss 0.3123→0.3125 有微小变化，证明消融确实扰动了输入、非实现假象；
但预测几乎不动。

## 判读

**T-CLSC 没有在用任务信息 E**：它学成了"通用 CLS 排序器"——在同分布留出任务
上 0.866 的准确率全部来自"哪些 CLS 平均更好"的任务无关知识，换 E、去 E 都不掉。
这不是实现 bug（模型级测试证明 E 参与前向且梯度可回传 SetTF；zero-E 使 loss
明显上移 0.31→0.38），而是 16 个同源 HAR 切片任务之间"最优 CLS 排名差异"太小，
BCE 上没有值得为 E 付出的增益——**正好印证任务多样性不足、必须转路线 A** 的
判断。机制层面（模型/引擎/数据管线/训练收敛/泛化到新切片）全部成立，可整体
迁移到路线 A 的多数据集设定复用。

## 复现

```
python scripts/2_generate_labels.py --resume --max-shared 3 --max-random 3
python scripts/2_generate_labels.py --resume --pool holdout --max-shared 3 --max-random 3
python scripts/3_pretrain_tclsc.py
```
（GPU 非确定性：准确率重跑 ±0.01 内不算差异。）
