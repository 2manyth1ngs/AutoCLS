# WP16 GEO-E 检验 PROTOCOL：E_T 能否预测目标的损失几何家族——几何子空间预测器的零 GPU 判死门

**会话定位**：四轮实验（EV1 / SP1 / LC2 / AA1）后，空间里唯一数据集级、部署口径有净空的对象是**损失几何家族**（loss_type × sim_func × norm；WP15 两基 ICC 0.72；UEA 28 官方靶按靶选家族 − 固定最好家族 +0.022、11/28 靶 > 0.02）。用户 2026-09-18 拍板：先验证 E 能否预测家族，能则把几何子空间（18 家族 × 5 温度）补齐标签、做"分层搜索"（E 定家族、任务盲层定家族内旋钮、SH 验证）。
GEOM（表示空间几何，WP15 §9）与原始特征池（WP15 A3）在 UCR 上都输给"永远 B_T"；E_T（GGS 探针的响应画像）没在这个对象上试过——本协议只回答这一问。产物 `results/tclsc_geo/`；脚本 `scripts/81_tclsc_geo.py`；单测 `tests/test_tclsc_geo.py`。零 GPU。

## §0 已看过的读数

- 28 个 UEA 官方靶（WP3 六靶 + WP10 二十二靶，11 条 shared-10 + GGS，掩码关，seed 0）：8 个家族在目录里；oracle11 − 先验 top-1 +0.051；家族 oracle − 固定最好家族（info_nce/euclidean/none）+0.022；冠军家族 info_nce/euclidean/none 11、triplet/dot/none 10、其余 7；家族增益 > 0.02 的靶 11/28（SCP1 / PEMS-SF / HMD / SCP2 / FaceDetection / Libras / MotorImagery / ERing …）。
- UCR 52（WP15）：B_I 更好 23%、oracle − 永远 B_T 只 +0.007；GEOM-DEAD、A3 特征池不过（U 56% / R 65% vs 猜多数 75%）。
- E_T 官方靶表：`tclsc_s1et_deploy/et_official.jsonl`（6）+ `tclsc_s1et_uea30/et_uea30.jsonl`（23）：rep@60 22 维、rep@290 22 维、φ12（六靶缺 φ）。
- **没看过**：E_T 与家族标签的任何关系。

## §1 设计

- **标签**：每靶每家族 u_d(f) = 该家族最好配置的 acc − 该靶 11 条最好 acc（≤ 0，家族选择的 regret）；家族 = (loss_type, sim_func, norm)；目录里有的 8 个家族。冠军家族 = argmax。噪声 σ_d = 冠军 3 种子 sd。
- **特征臂**：主臂 `et60`（rep@60 22 维）；副臂 `et290`、`phi`（23 靶）、`et60+phi`；全部折内标准化（只用训练靶）。
- **预测器**：多输出岭回归 ẑ(f | D) ≈ u_d(f)，α ∈ {0.1, 1, 10} 由训练折内层 LOO 选；选择 = argmax_f；副读数 = 两大家族二分类（logistic）。
- **对照（LODO 诚实）**：固定最好家族（训练折上均值 u 最高的家族）；先验 top-1 家族（triplet/dot/none）；oracle。
- **指标**（LODO 28 靶）：mean regret u_d(选)；命中率（选 == 冠军家族）；增益 = mean(u_选 − u_固定最好)；置换 p（靶 ↔ 特征行打乱 2000 次）。
- 可兑现翻译（只报）：预测家族内的配置（1–3 条）取实测最好 vs SH10-4-3 冠军（3 种子）vs 固定最好家族内最好。

## §2 判读（先于数据）

| 门 | 定义 |
|---|---|
| **G-G1 有价值** | `et60` 增益（vs 固定最好家族）≥ +0.005 且置换 p < 0.05 |
| **G-G2 学得到** | `et60` 命中率 ≥ 猜多数 + 0.10 且置换 p < 0.05 |

**读数码**：G-G1 ✓ → **GEO-E-GO**（写几何子空间补标 + 分层搜索协议）；G-G1 ✗ ∧ G-G2 ✓ → **GEO-E-WEAK**（能分家族但值不出净空：先补 18 家族标签再判）；其余 → **GEO-E-DEAD**（E_T 也读不出家族偏好，几何路线与 GEOM 同判：只能测量）。
只报：四个特征臂同表；逐靶选择与 regret；二分类副读数；UCR 52 上用 WP15 特征池（无 E_T）的同式对照。

## §3 预测（先验）

| # | 预测 | 先验 |
|---|---|---|
| P1 | G-G1 | 35% |
| P2 | G-G2 | 45% |
| P3 | `et60` 优于 `phi`（增益） | 55% |
| P4 | 增益集中在 EEG / 传感器类靶（SCP1 / SCP2 / HMD / MotorImagery / FaceDetection / PEMS-SF） | 60% |

## §4 执行

1. 本文件 + `tests/test_tclsc_geo.py`（先红）→ 协议锁 commit；2. `scripts/81_tclsc_geo.py --run`（零 GPU）→ REPORT → commit → memory。

## §5 执行台账（跑后填写）

- 2026-09-18 本地零 GPU：29 靶（PhonemeSpectra 10 配置按 ≥ 10 纳入）；干跑 20 次置换 10 s 后正式 1000 次置换 ≈ 15 min；φ 去掉两个基线列（官方靶无 iters0）；6 例单测绿。**GEO-E-DEAD**（E_T@60 命中 0.14 vs 猜多数 0.38、增益 vs 固定最好家族 −0.026 p 0.78；四臂全负；预测 1/4）。REPORT.md。
