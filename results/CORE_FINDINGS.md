# 核心发现 / Core Findings

STGCN on PEMS-BAY, **81 trained configurations**: 9 variants × 3 rolling folds × 3 seeds.
Variants = 5 ablation rungs, 2 with a congestion-weighted loss, 2 with a directed
diffusion graph operator. All numbers: MAE in mph at the 45-minute horizon,
averaged over seeds and folds. Folds 00–02 only — folds 3–5 contain zero rain
episodes and zero NHL fixtures.

---

## ① 五个指标，不同的赢家
## ① Different metrics, different winners

| model | 汇总MAE ↓ | 雨×通勤MAE ↓ | 传播保真 ↑ | 起始召回 ↑ | 误报 ↓ |
|---|---|---|---|---|---|
| `5_event_geo_att` | **2.070 ①** | **4.551 ①** | 1.424 ⑦ | 0.752 ③ | 1,823 ⑤ |
| `6_all` + diffusion | 2.108 ② | 4.946 ⑦ | 1.362 ⑧ | 0.740 ⑤ | 1,784 ④ |
| `1_traffic` | 2.116 ③ | 4.622 ② | **1.522 ①** | 0.726 ⑥ | 1,770 ③ |
| `6_all` + weighted | 2.118 ④ | 4.765 ④ | 1.435 ⑥ | **0.780 ①** | 2,445 ⑨ |
| `6_all` | 2.124 ⑤ | 4.797 ⑥ | 1.452 ⑤ | 0.738 ⑦ | 1,775 ③ |
| `3_weather` | 2.206 ⑥ | 4.954 ⑧ | 1.516 ② | 0.688 ⑨ | **1,381 ①** |
| `0_speed` + diffusion | 2.213 ⑦ | 4.782 ⑤ | 1.348 ⑨ | 0.701 ⑧ | 1,380 ① |
| `0_speed` (published STGCN) | 2.214 ⑧ | 4.739 ③ | 1.466 ④ | 0.702 ⑧ | 1,433 ② |
| `0_speed` + weighted | 2.280 ⑨ | 4.997 ⑨ | 1.401 ⑦ | 0.747 ④ | 2,354 ⑧ |

**Spearman(汇总MAE, 传播保真) = −0.03 (n=9, p=0.93)** — 完全无关。

**中文：** 汇总 MAE 选出的最优模型（`5_event_geo_att`），在"是否学到激波物理"上排倒数第三；
传播保真最好的 `1_traffic` 在汇总 MAE 上只排第三，在召回上排第六；召回最高的
`6_all`+weighted 在汇总 MAE 上排第四、误报最多。**汇总指标不是低估了差异，而是选错了模型。**

**EN:** The model that wins on aggregate MAE (`5_event_geo_att`) ranks 7th of 9 on
reproducing shockwave physics. The model that best reproduces the physics
(`1_traffic`) ranks 3rd on MAE and 6th on recall. The model that catches the most
onsets (`6_all` + weighted) is 4th on MAE and has the most false alarms. Spearman
rank correlation between aggregate MAE and propagation fidelity is **−0.03** —
none. An aggregate metric does not merely under-state the differences that
matter; it **selects a different model**.

---

## ② 天气通道让模型变差，不是没帮助
## ② Weather channels make the model worse, not merely no better

| | 汇总 MAE | 雨×通勤 MAE | 起始召回 | 误报 |
|---|---|---|---|---|
| `0_speed` | 2.214 | 4.739 | 0.702 | 1,433 |
| `3_weather` (+ 温度/降水/风速) | 2.206 | **4.954** (+4.5%) | **0.688** | 1,381 |
| `1_traffic` (+ 占有率) | **2.116** | **4.622** | 0.726 | 1,770 |

**中文：** `3_weather` 在两个天气窗口里连纯速度基线都不如，召回也更低。训练前的数据分析已经
预示了原因：控制时段后，下雨时激波发生率 **91.8%**，晴天 **93.1%**；卷入传感器数 6.25 对
6.46 —— 雨不制造崩溃，只是全网均匀降速 **−3.04 mph**，而均匀降速是速度历史里最容易读出的信号。

**EN:** `3_weather` is beaten by the speed-only baseline inside both weather windows
and on onset recall. Consistent with what the data said before any training:
controlling for time of day, rain does not change breakdown frequency (**91.8%**
of commute-peak timesteps carry a shockwave in rain against **93.1%** dry) or
size (6.25 sensors involved against 6.46). Rain is a uniform capacity reduction
of **−3.04 mph**, and a uniform slowdown is already legible in the speed history.

---

## ③ 决定性能的是空间结构，不是通道数量
## ③ Spatial structure, not channel count, decides performance

| | 加的 3 个通道 | 空间结构 | 参数量 | 雨×通勤 MAE |
|---|---|---|---|---|
| `3_weather` | 温度、降水、风速 | ❌ 一个值广播到 325 节点 | 118,732 | **4.954** |
| `5_event_geo_att` | 场馆距离、活动、人数 | ✅ 逐节点，路网距离衰减 | 118,732 | **4.551** |

**中文：** 参数量完全相同、通道数完全相同（c_in = 5），唯一差别是通道里有没有空间变化。
结果在每个窗口、每一折上都相反。**在图模型上，没有空间变化的通道不是弱特征，而是有害特征** ——
广播标量经过图卷积后近似一个全局偏置，白白占掉三个维度。

**EN:** Identical parameter count (118,732) and identical channel count (c_in = 5).
The only difference is whether the added channels vary across nodes. The outcome
reverses in every window and on every fold. **On a graph model, a channel with no
spatial variation is not a weak feature — it is a harmful one:** passed through a
graph convolution a broadcast scalar is close to a global bias, spending three
dimensions of capacity saying nothing about which node is which.

---

## ④ 加权损失：汇总指标不动，召回上升，误报上升
## ④ The weighted loss: aggregate unchanged, recall up, false alarms up

| | 汇总 MAE | 起始召回 | 误报 |
|---|---|---|---|
| `0_speed` | 2.214 | 0.702 | 1,433 |
| `0_speed` + weighted | 2.280 (差 3.0%) | **0.747** (+4.5pt) | 2,354 (+64%) |
| `6_all` | 2.124 | 0.738 | 1,775 |
| `6_all` + weighted | **2.118 (不变)** | **0.780** (+4.2pt) | 2,445 (+38%) |

**中文：** 把 L1 损失按低速加权后，`6_all` 的汇总 MAE 完全不变（2.124 → 2.118，seed 标准差
±0.14），起始召回从 0.738 升到 **0.780**，代价是误报增加 38%。在 `0_speed` 上代价是汇总 MAE
差 3%。**如果你认为汇总 MAE 掩盖了重要时段，那你的损失函数也不该被它主导。**
这笔交易值不值取决于成本比，而两类用户正好在两端 —— 见 `results/decision__fold*.json`
的 1/2/5/10/20/50 扫描。

**EN:** Weighting the L1 loss towards low speeds leaves `6_all`'s aggregate MAE
unchanged (2.124 → 2.118, seed sd ±0.14) while raising onset recall from 0.738
to **0.780**, at 38% more false alarms; on `0_speed` it costs 3% of aggregate MAE
for the same 4-point recall gain. **If aggregate MAE hides the hours that matter,
the loss function should not be governed by it either.** Whether the trade is
worth taking depends on the cost ratio, and the two users of this forecast sit at
opposite ends of it.

---

## ⑤ 模型确实学到了逆流传播，但都不完整
## ⑤ The models do learn upstream propagation — none of them fully

观测真值 / observed truth: **1.683** (per fold 1.574 / 1.639 / 1.835; upstream/downstream
co-occurrence of shockwave onsets at a 15-min lag, over the directed adjacency)

```
1_traffic              1.522   (90% of truth)
3_weather              1.516   (90%)
0_speed                1.466   (87%)
6_all                  1.452   (86%)
6_all   + weighted     1.435   (85%)
5_event_geo_att        1.424   (85%)
0_speed + weighted     1.401   (83%)
6_all   + diffusion    1.362   (81%)   <- see ⑥
0_speed + diffusion    1.348   (80%)   <- see ⑥
```

**中文：** 没有一个模型把比值压平到 1.0，说明学到的是**传播的波**而不是平滑的条件均值；
但也没有一个达到真值。这是"shockwave"这个标题在建模层面唯一的直接检验。

**EN:** No model collapses the ratio to 1.0, so all are learning a **travelling
wave** rather than a smooth conditional mean; none reaches the observed value
either. This is the only direct model-level test of the word "shockwave" in the
title.

---

## ⑥ 有向图算子让传播保真变差 —— 一个负结果
## ⑥ A directed graph operator makes propagation *worse* — a negative result

**动机（训练前写下的）：** Chebyshev 卷积需要对称拉普拉斯，所以模型分不清上游邻居和下游邻居；
而现象本身逆流传播的频率是顺流的 1.68 倍。给它一个有方向的算子（DCRNN/Graph WaveNet 的
双向随机游走，`--graph-conv diffusion`）应该能学到更真实的波。

**结果：方向反了。** 3 折 × 3 seed，45 分钟视野，15 分钟滞后：

| | fold00 | fold01 | fold02 | **均值 (n=9)** | sd |
|---|---|---|---|---|---|
| 观测真值 | 1.574 | 1.639 | 1.835 | **1.683** | — |
| `0_speed` | 1.510 | 1.453 | 1.435 | **1.466** | 0.042 |
| `0_speed` + diffusion | 1.379 | 1.310 | 1.355 | **1.348** | 0.053 |
| `6_all` | 1.493 | 1.432 | 1.430 | **1.452** | 0.065 |
| `6_all` + diffusion | 1.365 | 1.363 | 1.357 | **1.362** | 0.070 |

按 (fold, seed) 配对：

```
0_speed   diffusion − Chebyshev:  −0.118 ± 0.053   t = −6.64   9/9 对全部变差
6_all     diffusion − Chebyshev:  −0.090 ± 0.090   t = −3.00   6/9 对变差
```

同时**汇总 MAE 几乎不动**：`6_all` 2.124 → 2.108，`0_speed` 2.214 → 2.213，两者都在
seed 标准差 (±0.17) 之内。参数量 121k → 138k、117k → 133k。

**中文：** 这是 ① 的第三个独立例证，也是最干净的一个 —— diffusion 唯一改动的就是空间算子，
通道、损失、数据、划分全部不变。**汇总 MAE 会把这次改动报告成"略有改善"，而它实际上让模型
离激波物理更远了。** 它还给 ③ 加了一层：起作用的是"节点之间有区别"这个信息本身
（`dist_to_venue` 那类通道），不是图上的方向性传播机制。

**EN:** The change was motivated *precisely* by the observed 1.68× upstream
asymmetry and is the cleanest single-variable experiment in the set: only the
spatial operator changes, with channels, loss, data and splits held fixed. It
leaves aggregate MAE unchanged (2.124 → 2.108, within seed sd 0.17) while
*reducing* propagation fidelity from 1.452 to 1.362, worse on 9 of 9 paired runs
for `0_speed` and 6 of 9 for `6_all`. **Aggregate MAE would have reported this as
a marginal improvement.** It also refines ③: what helps is a channel that tells
one node from another, not a mechanism for directional propagation on the graph.

---

## ⚠️ 一个不能用的指标：提前量 / One metric that does not discriminate: lead time

`metrics.onset_lead_time`（模型预测崩溃的时刻 vs 反应式规则确认的时刻，<45 mph 持续 15 分钟，
场馆 2 km 内 28 个传感器）在这批结果上，**19 个模型的中位提前量全部是 0.0 分钟**，均值在
0.00–0.32 分钟之间，无法区分任何两个模型。

原因是结构性的：预测视野只有 60 分钟，而反应式规则本身有 15 分钟的确认延迟，所以可分辨的
提前量上限很窄。**不要在正文里报告提前量。** 起始召回和误报数（① ④ 两表）测的是同一件事，
且在这份数据上是可分辨的。

An earlier draft of this file reported lead times of 15.6 / 25.6 / 40.0 min.
Those numbers do not reproduce from the prediction files and have been removed.

---

## 复现 / Reproducing these numbers

所有数字来自 `results_bundle_v2_diffusion.tgz`：

| 数字 | 来源 |
|---|---|
| 汇总 MAE、每视野 MAE | `checkpoints/*.json` → `test.45min.mae` |
| 窗口内 MAE（雨×通勤等） | `results/report__fold0*.json` → `<window>.inside.45min.mean` |
| 起始召回、误报 | `results/decision__fold0*.json` → `models.<v>.all.{recall,fp}` |
| 传播保真 | `metrics.upstream_propagation(pred[:, 8, :], adj_mx.npy, lags=(3,))`，预测按 `timesteps` 排序，`adj_mx.npy` 必须是**有向**的原始邻接（1,789 条单向边） |
