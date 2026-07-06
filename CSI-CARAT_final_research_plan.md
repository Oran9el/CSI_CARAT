# CSI-CARAT：面向跨域 WiFi CSI 感知的因果自适应增强与风险感知测试时校准方法

## 1. 方法名称

**CSI-CARAT: Causal Adaptive Risk-aware Augmentation and Test-time Calibration for Cross-Domain WiFi CSI Sensing**

中文名：

**面向跨域 WiFi CSI 感知的因果自适应增强与风险感知测试时校准方法**

建议研究任务：

> WiFi CSI-based Human Activity Recognition / Gesture Recognition 的跨域泛化

目标指标：

> 在未见用户、未见位置、未见朝向、未见环境、LOS/NLOS 或设备变化下，使 **average accuracy ≥ 75%**，并尽量使 **worst-domain macro-F1 ≥ 75%**。

该目标不是凭空假设。已有 WiFi CSI 跨域泛化研究中，GenFi 在复杂 unseen scenario 设置下报告了超过 75% 的结果；Wi-ViTAL 也在 unseen-domain 任务中报告超过 76%；DATTA 证明了 domain-adversarial training 与 test-time adaptation 可用于 WiFi CSI 跨域 HAR。

---

## 2. 方案依据

本方案不是直接复刻已有 WiFi CSI 方法，而是基于以下已有研究进行组合和改造。

| 来源 | 已有结论 | 在 CSI-CARAT 中的作用 |
|---|---|---|
| **GenFi, IEEE ICC 2025** | WiFi CSI 跨域泛化不应只提取 domain-invariant feature；scenario-specific feature 也可能包含有助于分类的信息。GenFi 使用 scenario-invariant / scenario-specific feature disentanglement 与 meta-learning 提升 unseen scenario 下的 HAR 表现。 | 保留“稳定动作因子 + 场景相关因子”的思想，但不复刻其元学习融合，而改成因果约束、风险优化和测试时轻量校准。 |
| **Wi-ViTAL, IEEE TMC 2025/2026** | 使用 multi-scale linear attention ViT 和 adversarial learning 进行 wireless HAR domain generalization，并在 unseen-domain 任务中报告超过 76% 的结果。 | 采用多尺度时频注意力 backbone，并引入域对抗对齐。 |
| **DATTA, WACV 2026** | 将 domain-adversarial training、test-time adaptation 和 random weight resetting 用于 cross-domain WiFi HAR。 | 采用测试时校准思想，但只更新 norm、adapter、gate、temperature 等轻量模块，避免全模型漂移。 |
| **AutoTCL, ICLR 2024** | 针对时间序列，提出可学习的参数化增强，而不是完全依赖人工指定增强。 | 设计 CSI-aware adaptive augmentation，使增强作用于 time、subcarrier、antenna、phase、Doppler 等 CSI 结构。 |
| **CauDiTS, ICML 2024** | 多变量时间序列跨域可通过 causal disentanglement 分离 domain-common causal rationales 与 domain-specific correlations。 | 将动作引起的稳定动态扰动建模为 causal rationale，将房间、设备、位置、LOS/NLOS 造成的多径差异建模为 spurious correlation。 |
| **IST, CVPR 2024** | 测试时自适应中伪标签可能不可靠，使用多增强视图、特征图伪标签修正和 EMA 可提高稳定性。 | 测试时对 CSI window 做多视图预测、伪标签修正和 EMA 稳定更新。 |
| **DeYO / ViDA / ZERO, ICLR/NeurIPS 2024** | DeYO 关注 TTA 中 entropy-only 选择不可靠；ViDA 使用 adapter 做 continual TTA；ZERO 通过多增强视图聚合实现低成本测试时适应。 | 设计 motion-preserving / motion-destroying CSI 变换进行可靠样本选择，并采用低成本 adapter 或多视图聚合。 |

---

## 3. 核心研究假设

WiFi CSI 跨域失败主要来自：

1. **静态多径环境差异**：房间结构、家具、墙体、LOS/NLOS 改变 CSI 分布；
2. **设备与天线差异**：网卡、天线顺序、采样率、子载波质量不同；
3. **用户行为差异**：同一动作在不同用户、朝向、位置下的动态模式不同；
4. **模型学习了伪相关**：模型可能依赖某个环境中的静态频谱纹理，而不是动作本身。

因此，CSI-CARAT 的核心假设是：

> 跨域泛化的关键不是完全消除域信息，而是让模型主要依赖动作引起的稳定动态因子，同时通过风险约束和测试时校准控制域相关因子的使用。

---

## 4. 方法总体框架

CSI-CARAT 包含四个核心模块：

```text
CSI Input
   │
   ├── 1. CSI-aware Adaptive Augmentation
   │
   ├── 2. Multi-scale CSI Transformer Encoder
   │
   ├── 3. Causal / Spurious Feature Disentanglement
   │
   └── 4. Lightweight Test-time Calibration
```

最终预测形式为：

$$
\hat{y}=C(z_c + g(x)\cdot A(z_s))
$$
其中：

- $z_c$：动作相关的 causal feature；
- $z_s$：域相关或场景相关 feature；
- $A(·)$：轻量 adapter；
- $g(x)$：风险感知 gate，控制当前样本是否允许使用域相关信息；
- $C(·)$：分类器。

---

## 5. 模块设计

### 5.1 输入与预处理

建议使用三类 CSI 表示：

| 输入分支 | 说明 |
|---|---|
| **Amplitude branch** | CSI amplitude 或 normalized amplitude |
| **Phase-difference branch** | 天线间相位差、子载波相位差，减少硬件相位偏移影响 |
| **Doppler / spectrogram branch** | 强化动作动态特征，弱化静态环境纹理 |

预处理建议：

- Hampel filter / median filter 去异常点；
- packet-level 插值与重采样；
- instance normalization；
- temporal difference：$Δx_t = x_t - x_{t-1}$；
- 子载波质量筛选；
- 滑窗切分，例如 1–3 秒 window，50% overlap。

---

### 5.2 CSI-aware Adaptive Augmentation

不要只使用固定增强。CSI-CARAT 使用增强策略网络 $P_φ$，根据输入样本自动选择增强类型和强度：
$$
\tilde{x}=Aug(x; P_\phi(x))
$$
增强空间包括：

| 增强 | 物理含义 |
|---|---|
| time masking | 动作起止时间偏移、局部丢包 |
| subcarrier masking | 子载波衰落、设备差异 |
| antenna shuffling | 天线顺序变化 |
| amplitude scaling | 路径损耗变化 |
| phase drift | 硬件相位偏移 |
| Doppler perturbation | 速度扰动 |
| random erasing | packet loss / 局部 CSI 缺失 |
| mixup / manifold mixup | 平滑跨域决策边界 |

AutoTCL 的依据是：时间序列增强难以完全依靠人工经验选择，因此它提出 parametric augmentation，用可学习增强生成适合时间序列的 positive views。

CSI-CARAT 的创新点在于：将 AutoTCL 的“参数化时间序列增强”改造成 **CSI 物理感知增强**，增强维度不是图像 patch，而是：

```text
time × subcarrier × antenna × phase / Doppler
```

---

### 5.3 Multi-scale CSI Transformer Encoder

Backbone 建议采用：

> CNN / TCN front-end + Multi-scale Linear Attention Transformer

结构：

```text
CSI amplitude / phase / Doppler
        │
Local CSI convolution block
        │
Temporal-subcarrier tokenization
        │
Multi-scale linear attention transformer
        │
Feature tokens
```

理由：

- CNN/TCN 适合局部时间-子载波模式；
- Transformer 适合长时间依赖；
- linear attention 可降低计算复杂度；
- Wi-ViTAL 已经证明 linear attention ViT + adversarial learning 可用于无线 HAR 域泛化。

---

### 5.4 因果-伪相关特征解耦

模型输出两个表征：

$$
z_c = E_c(x), \quad z_s = E_s(x)
$$
其中：

- $z_c$：跨域稳定动作因子；
- $z_s$：环境、设备、位置、用户相关因子。

训练时包含以下损失。

#### 1. 分类损失

$$
L_{ce}=CE(C(z_c + g(x)A(z_s)), y)
$$



#### 2. 监督对比损失

同类动作跨域拉近，不同动作拉远：

$$
L_{supcon}
$$


#### 3. 域对抗损失

让 $z_c$ 难以预测 domain label：

$$
L_{adv}=CE(D(GRL(z_c)), d)
$$


#### 4. 解耦损失

降低 $z_c$ 与 $z_s$ 的相关性：

$$
L_{dis}=HSIC(z_c,z_s)
$$
或：

$$
L_{dis}=||Cov(z_c,z_s)||_F^2
$$


#### 5. 增强一致性损失

同一样本不同增强视图预测一致：

$$
L_{cons}=KL(p(y|x) || p(y|\tilde{x}))
$$


#### 6. 风险感知损失

重点优化最差源域，而不是只优化平均准确率：

$$
L_{risk}=\log \sum_{d \in D_s} \exp(\eta L_d)
$$
最终训练目标：

$$
L = L_{ce}
+ \lambda_1 L_{supcon}
+ \lambda_2 L_{adv}
+ \lambda_3 L_{dis}
+ \lambda_4 L_{cons}
+ \lambda_5 L_{risk}
$$


---

### 5.5 风险感知 Gate

GenFi 已经做了 invariant / specific feature fusion，所以 CSI-CARAT 不应简单声称“提出融合”。区别应放在 **动态 gate + 风险约束** 上。

定义：

$$
g(x)=\sigma(MLP([u(x), h_d(x), conf(x)]))
$$
其中：

- `u(x)`：预测不确定性；
- `h_d(x)`：domain-shift score；
- `conf(x)`：多视图一致性置信度。

如果样本高度跨域且不确定性高，模型减少 `z_s` 的权重：

$$
g(x) \downarrow
$$
如果样本与源域结构一致，且 `z_s` 对分类有帮助，则允许适度使用：

$$
g(x) \uparrow
$$
这样可以避免两种极端：

- 只用 $z_c$：可能丢失任务相关细节；
- 大量使用 $z_s$：容易过拟合环境/设备。

---

## 6. 测试时校准

CSI-CARAT 的测试时校准不更新整个 backbone，只更新：

- normalization statistics；
- lightweight adapter $A(·)$；
- gate $g(x)$；
- classifier temperature；
- class prototypes。

测试时对每个 CSI window 生成 $K$ 个增强视图：

$$
\{A_1(x), A_2(x), ..., A_K(x)\}
$$
预测聚合：

$$
\hat{y}=Agg\{C(E(A_i(x)))\}_{i=1}^{K}
$$
校准损失：

$$
L_{tta}=L_{entropy}+L_{pseudo}+L_{view-cons}+L_{proto}
$$
其中：

- $L_entropy$：低熵置信预测；
- $L_pseudo$：高置信伪标签监督；
- $L_view-cons$：多视图一致性；
- $L_proto$：目标样本向源域类别原型靠近。

为了提高鲁棒性，引入 DeYO 式样本选择思想：

| 变换 | 作用 |
|---|---|
| motion-preserving transform | 保留动作动态，只改变幅值、子载波、天线顺序 |
| motion-destroying transform | 打乱时间顺序或破坏 Doppler 连续性 |

如果一个样本在 motion-preserving transform 下预测稳定，但在 motion-destroying transform 下预测明显变化，说明模型更可能依赖动作动态，而不是静态环境纹理。该样本才用于 TTA 更新。

---

## 7. 数据集设计

### 7.1 主数据集：Widar3.0

Widar3.0 适合作为主数据集，因为它本身面向 WiFi cross-domain gesture recognition，包含位置、朝向、环境等域因素。

建议实验设置：

| 设置 | 训练域 | 测试域 |
|---|---|---|
| Cross-location | 部分 locations | unseen location |
| Cross-orientation | 部分 orientations | unseen orientation |
| Cross-environment | 部分 environments | unseen environment |
| Cross-user | 部分 subjects | unseen subject |
| Mixed unseen | 多源域 | 未见用户 + 未见位置 + 未见朝向 |

---

### 7.2 第二数据集：MM-Fi

MM-Fi 是 NeurIPS 2023 Datasets and Benchmarks 数据集，包含 RGB、depth、LiDAR、mmWave radar 和 WiFi CSI 等多模态数据，包含多个 subject、action category、2D/3D pose 和 3D position 标注。

在本研究中，MM-Fi 建议只使用 **WiFi CSI modality** 做主实验，并可额外使用 pose / RGB 作为训练时辅助或上界分析，但最终推理必须是 WiFi-only。

建议设置：

| 设置 | 说明 |
|---|---|
| Cross-subject | 训练 subjects 与测试 subjects 不重叠 |
| Cross-scene | 训练场景与测试场景不重叠 |
| WiFi-only | 只输入 WiFi CSI |
| Optional teacher | 可用 RGB/pose 做训练时辅助蒸馏，但推理时不能使用 |

---

### 7.3 第三数据集：XRF55

XRF55 是大规模 RF action dataset，动作类别更多，适合作为泛化压力测试。

建议设置：

| 设置 | 说明 |
|---|---|
| Cross-subject | 未见 subject |
| Cross-day | 未见采集日期 |
| Cross-environment | 未见环境 |
| Long-tail actions | 测试类别不均衡下的 macro-F1 |

---

### 7.4 自采数据集

如果有实验条件，建议补充一个小规模自采数据集，用于验证真实部署价值。

最少配置：

| 项目 | 建议 |
|---|---|
| 场景 | 3 个房间：实验室、宿舍、会议室 |
| 用户 | 8–12 人 |
| 动作 | 6–8 类 |
| 设备 | 至少 2 种 WiFi NIC 或 2 种天线配置 |
| 链路 | LOS + NLOS |
| 每类样本 | 每用户每场景每动作 ≥ 30 次 |

自采数据的价值不是追求大，而是验证：

- 换房间是否仍能超过 75%；
- 换用户是否仍能超过 75%；
- 换设备/天线是否仍能超过 75%；
- TTA 是否真的能提高 worst-domain performance。

---

## 8. Baseline 设计

### 8.1 基础深度学习 Baseline

| Baseline | 作用 |
|---|---|
| CNN | 最基础 CSI spectrogram 分类器 |
| TCN | 时间建模 baseline |
| GRU / BiLSTM | 序列建模 baseline |
| ResNet / EfficientNetV2 | 时频图像化 CSI baseline |
| Transformer | 长时依赖 baseline |
| WiFi CSI Benchmark / SenseFi-style models | 传统 WiFi sensing 网络对比 |

---

### 8.2 域泛化 Baseline

| Baseline | 说明 |
|---|---|
| ERM | 所有源域混合训练 |
| Mixup | 输入或特征层 mixup |
| CORAL | 对齐源域协方差 |
| MMD | 最大均值差异对齐 |
| DANN | 梯度反转域对抗 |
| GroupDRO | 优化最差源域风险 |
| IRM / VREx | 学习跨域稳定预测规律 |
| SupCon-DG | 跨域监督对比学习 |

---

### 8.3 WiFi CSI 专用 Baseline

| Baseline | 必要性 |
|---|---|
| AirFi | 经典 WiFi gesture domain generalization 方法 |
| Wi-AM | 近年 cross-domain gesture recognition 方法 |
| GenFi | 最重要 baseline；它已做 feature disentanglement + meta-learning |
| Wi-ViTAL | Transformer + adversarial domain generalization baseline |
| DATTA | WiFi CSI test-time adaptation baseline |
| MDTA | 如果能获取代码或复现细节，建议加入，因为它做了 multi-domain factor disentanglement + TTA |

---

### 8.4 顶会迁移 Baseline

| Baseline | 迁移方式 |
|---|---|
| AutoTCL-CSI | 只迁移 AutoTCL 式自适应时间序列增强 |
| CauDiTS-CSI | 只迁移 causal disentangled MTS domain adaptation |
| IST-CSI | 只迁移 CVPR 2024 IST 测试时自训练 |
| DeYO-CSI | 用 motion-destroying CSI transform 替代 object-destroying transform |
| ViDA-CSI | 在 Transformer 中插入高/低秩 adapter |
| ZERO-CSI | 多 CSI 增强视图投票，不反向传播 |

---

## 9. 评价指标

主指标：

| 指标 | 说明 |
|---|---|
| Accuracy | 总体分类准确率 |
| Macro-F1 | 类别不均衡时更可靠 |
| Worst-domain Accuracy | 所有目标域中最差一个域的准确率 |
| Worst-domain Macro-F1 | 最关键指标 |
| Domain Std | 不同目标域性能标准差 |
| Per-class Accuracy | 分析哪些动作跨域最差 |

辅助指标：

| 指标 | 说明 |
|---|---|
| Params | 参数量 |
| FLOPs | 计算复杂度 |
| Inference latency | 单窗口推理时间 |
| TTA update time | 测试时校准开销 |
| Memory usage | 部署成本 |
| Confidence calibration | ECE / NLL，可选 |

核心结果表建议采用：

```text
Average ACC / Macro-F1 / Worst-domain ACC / Worst-domain Macro-F1 / Std
```

成功标准建议写成：

$$
AvgAcc \ge 75\%, \quad WorstDomainMacroF1 \ge 75\%
$$
如果 XRF55 太难，可以把 XRF55 的目标设为：

$$
AvgAcc \ge 75\%, \quad WorstDomainMacroF1 尽量接近或超过 70\%
$$

---

## 10. 实验协议

### 10.1 Leave-One-Domain-Out

对每个域因素分别做 LODO：

```text
Train: domains except d_i
Test:  d_i
```

域因素包括：

- user；
- location；
- orientation；
- environment；
- day / time；
- device / antenna；
- LOS/NLOS。

---

### 10.2 Multi-source to Unseen-domain

训练多个源域，测试一个完全未见域：

```text
Train: domain A + B + C
Test:  domain D
```

---

### 10.3 Combined Unseen Setting

最难设置：

```text
Train: seen users + seen rooms + seen orientations
Test:  unseen user + unseen room + unseen orientation
```

这个设置对应真实部署，也最能体现方法价值。

---

### 10.4 Pure DG 与 DG + TTA 分开报告

必须分两种：

| 设置 | 是否使用目标域无标签数据 |
|---|---|
| Pure DG | 不使用 |
| DG + TTA | 只使用测试流无标签数据 |

这样可以避免审稿人质疑“其实做的是 domain adaptation，不是 domain generalization”。

---

## 11. 消融实验

| 消融项 | 目的 |
|---|---|
| w/o adaptive augmentation | 验证 CSI-aware augmentation 贡献 |
| fixed augmentation only | 证明自适应增强优于人工固定增强 |
| w/o causal disentanglement | 验证因果/伪相关分离 |
| w/o risk loss | 验证 worst-domain 优化 |
| w/o gate | 验证动态融合必要性 |
| w/o TTA | 区分纯 DG 与 TTA 增益 |
| full TTA update | 证明只更新 adapter/gate 更稳定 |
| entropy-only TTA | 证明 DeYO/IST-style 选择比纯 entropy 更稳 |
| ZERO-style no-backprop | 验证无反传聚合的实际收益 |

---

## 12. 预期贡献写法

可以将论文贡献写成：

1. **提出 CSI-CARAT，一种面向 WiFi CSI 跨域泛化的因果自适应增强与测试时校准框架。**  
   它不是只追求域不变特征，而是显式区分动作因果因子与域相关伪相关。

2. **提出 CSI-aware adaptive augmentation。**  
   将 AutoTCL 的参数化时间序列增强思想迁移到 CSI 的 time-subcarrier-antenna-phase 结构中。

3. **提出 risk-aware gated fusion。**  
   通过最差域风险优化和动态 gate 控制 domain-specific feature 的使用，目标是提高 worst-domain performance，而不是只提高平均准确率。

4. **提出 lightweight CSI test-time calibration。**  
   测试时只更新 norm / adapter / gate / temperature，并结合多视图一致性和 motion-destroying sample selection，降低伪标签错误和灾难性遗忘。

5. **在 Widar3.0、MM-Fi、XRF55 和自采数据上系统验证。**  
   报告 average accuracy、macro-F1、worst-domain accuracy、worst-domain macro-F1、FLOPs 和 latency。

---

## 13. 最终推荐结论

最终建议采用：

> **CSI-CARAT: Causal Adaptive Risk-aware Augmentation and Test-time Calibration for Cross-Domain WiFi CSI Sensing**

不要把论文主线写成“特征解耦 + 元学习 + TTA”，因为这容易撞 GenFi、DATTA、MDTA。

更稳的创新集中在：

- **CSI 物理感知自适应增强**；
- **动作因果因子与域伪相关的分离**；
- **worst-domain 风险优化**；
- **只更新轻量 adapter/gate 的测试时校准**。

这条路线有明确依据：WiFi CSI 领域已有 GenFi、Wi-ViTAL、DATTA 证明了解耦、对抗、TTA 的有效性；顶会方法 AutoTCL、CauDiTS、IST、DeYO、ViDA、ZERO 提供了可迁移的时间序列增强、因果解耦和测试时校准机制。

目标设为 **average accuracy ≥ 75%** 是现实的；更有论文价值的目标应是：

$$
Worst-domain macro-F1 ≥ 75%
$$

---

## 参考文献与链接

1. GenFi: Enhancing WiFi-based Human Activity Recognition to Diverse Unseen Scenarios, IEEE ICC 2025.  
   https://www.ece.nus.edu.sg/stfpage/bsikdar/papers/icc_han_25.pdf

2. Wi-ViTAL: Domain Generalization of Wireless Human Activity Recognition Using Linear Attention Vision Transformer With Adversarial Learning, IEEE Transactions on Mobile Computing.  
   https://research.nottingham.edu.cn/en/publications/wi-vital-domain-generalization-of-wireless-human-activity-recogni/

3. DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition, WACV 2026.  
   https://openaccess.thecvf.com/content/WACV2026/html/Strohmayer_DATTA_Domain-Adversarial_Test-Time_Adaptation_for_Cross-Domain_WiFi-Based_Human_Activity_Recognition_WACV_2026_paper.html

4. AutoTCL: Automated Time Series Contrastive Learning, ICLR 2024.  
   https://proceedings.iclr.cc/paper_files/paper/2024/file/ccf6d8b4a1fe9d9c8192f00c713872ea-Paper-Conference.pdf

5. CauDiTS: Causal Disentanglement for Multivariate Time-Series Domain Adaptation, ICML 2024.  
   https://proceedings.mlr.press/v235/lu24i.html

6. Improved Self-Training for Test-Time Adaptation, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Improved_Self-Training_for_Test-Time_Adaptation_CVPR_2024_paper.html

7. DeYO: Detecting and Yielding Out-of-distribution Samples for Test-Time Adaptation, ICLR 2024.  
   https://openreview.net/forum?id=9w3iw8wDuE

8. MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing, NeurIPS 2023 Datasets and Benchmarks.  
   https://proceedings.neurips.cc/paper_files/paper/2023/file/3baf7a39d07e9f4f1e258a412df94521-Paper-Datasets_and_Benchmarks.pdf

9. XRF55 dataset.  
   https://aiotgroup.github.io/XRF55/
