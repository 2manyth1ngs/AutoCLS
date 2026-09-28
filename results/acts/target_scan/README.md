# 池外目标筛选（探索，2026-09-28）

用途：为"部署口径下对手工模型过门"的实验挑目标数据集——池外、统计特征上与 17 个源相近。本目录只含数据统计与文献数字，**不含任何模型读数**。

- `sibling_scan.py`：17 个源（ACTS 整集）与池外候选（UEA 官方 TRAIN、UCR 官方 train，aeon 下载）的 11 维统计特征（log T / log C / log k + 6 维信号统计 + 趋势 / 季节强度；不含需训练的 base_*）；`sibling_feats.json`。
- `sibling_rank.py`：按源的均值 / 标准差标准化，候选到最近源的欧氏距离；参照线 = 源与源最近距离中位 2.17；`sibling_rank.json`（near_dup = 与池内源同一批录音）。
- 文献数字（PyMuPDF 从 arXiv PDF 原文解析）：
  - `ts2vec_ucr.json`：TS2Vec（arXiv 2106.10466）UCR 表 122 / 128 行，列 = B=4 / B=8 / B=16 / T-Loss / TNC / TS-TCC / TST / DTW（论文主结果 = B=8）；
  - `infots.json`：InfoTS（arXiv 2303.11911）UCR 127 行、UEA 30 行，列 = InfoTS_s（半监督）/ InfoTS / TS2Vec / T-Loss / TNC / TS-TCC / TST / DTW；
  - `autotcl_uea.json`：AutoTCL（arXiv 2402.10434，ICLR 2024）Table 12，UEA 30 行，列 = AutoTCL / TS2Vec / T-Loss / TNC / TS-TCC / TST / DTW / TF-C / InfoTS（= InfoTS_s）。AutoTCL 只报 UEA，无 UCR。
- 注意：WebFetch 的网页摘要曾编造过 AutoTCL 表，以上数字一律取自 PDF 原文解析。
