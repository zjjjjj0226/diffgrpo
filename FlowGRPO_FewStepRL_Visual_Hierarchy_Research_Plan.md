# Flow-GRPO 与 FewStepRL-SD3.5：视觉层级演化、RL 收敛及组内多样性分析实验方案

> **用途**：研究方案 / 复现实验规范 / 可提供给 AI 编程助手开发分析脚本的需求文档  
> **版本**：v1.0（2026-11-25）  
> **定位**：以诊断与可视化为主，优先不改动原有 RL 训练目标；后续再研究自适应采样与计算预算分配。

## 0. 研究目标与核心问题

本项目包含两项相互关联、但必须分别开展的实验：

**研究 I：Flow-GRPO 视觉层级演化。** 在原始 Flow-GRPO 框架中研究 RL 训练过程中，以及一次采样轨迹的不同去噪/流时间步中，生成图像的六种信息层级如何形成和改善：

1. 全局语义（Global Semantics）
2. 物体布局（Object Layout / Spatial Composition）
3. 前景（Foreground）
4. 背景（Background）
5. 局部纹理（Local Texture）
6. 像素细节（Pixel-level Details）

**研究 II：FewStepRL 少步模型的组内多样性与视觉层级演化。** 在 FewStepRL-SD3.5 中比较少步模型与普通/多步模型在 GRPO rollout 内的多样性，检验“少步模型组内多样性更小”这一**待检验假设**，并记录少步模型每个采样步的六种视觉信息变化及其在 RL 训练前后的改善。

**核心区别：**
- **采样步内演化（within-trajectory formation）**：同一 Checkpoint 的一次生成轨迹，在第 1、2、…、K 个采样步时，视觉信息如何逐渐显现。
- **RL 训练提升（across-training improvement）**：同一采样步的位置，在训练 Step 0、50、100、200、400 的独立 Checkpoint 之间，质量发生了怎样的变化。
- **组内多样性（within-prompt diversity）**：同一个 Prompt 下 G 个不同 rollout 的最终图像及其中间预测在语义、布局、视觉特征等层面的差异。不能将其等同于 Prompt 之间的总体多样性。

三者需要独立记录，并提供联合图表。不能仅凭中间图像越来越清晰就声称“RL 提升了质量”，也不能仅凭 GRPO 的 group reward 方差变小就声称“图像多样性下降”。

---

## 1. 仓库、算法与核验入口

### 1.1 原始 Flow-GRPO

- 官方实现：<https://github.com/yifan123/flow_grpo>
- 论文：<https://arxiv.org/abs/2505.05470>
- 关键位置（在本地仓库再次确认）：
  - `scripts/train_sd3.py`：SD3/SD3.5 训练入口
  - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`：生成轨迹与 log-prob 流程
  - `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`：SDE 采样路径
  - `config/grpo.py`：配置；实际支持的模型/奖励/脚本以当前 Git Commit 为准
- 方法特征：将 ODE 转换为具有随机性的 SDE 以支持 GRPO rollout，部分版本还提供 Flow-GRPO-Fast 等训练采样策略。**务必区分训练采用的 SDE/ODE 路径与评测时的固定推理路径。**

### 1.2 FewStepRL-SD3.5

- 官方仓库：<https://github.com/EDAPINENUT/fewsteprl-sd35>
- 论文：<https://arxiv.org/abs/2611.08240>
- 关键位置（以当前检出的代码为准）：
  - `fewsteprl/trainers/`：包括 MeanFlowGRPO / FlowGRPO 等训练方法
  - `fewsteprl/rewards/`：MixReward 与各 Scorer
  - `scripts/train/configs/flowgrpo_mix.yaml`：FlowGRPO Mix 配置
  - `scripts/train/configs/fewstepgrpo_mix.yaml`：少步 GRPO 相关配置（若当前代码存在）
  - `inference.py`、`scripts/eval/`：推理与评测
- 方法特征：使用 SD3.5-Medium 及 AnyFlow Stage 1/2 初始化形成的少步生成器；Stage 3 执行 RL。具体采样随机性、Noise/ODE 设置、`flowgrpo_mix` 和 `fewstepgrpo_mix` 目标差异，**以代码检查结果为准**。

### 1.3 复现原则

1. 分别克隆并记录两个仓库的 Git Commit：`git rev-parse HEAD`；尽量使用独立环境。
2. 不预设两个仓库里的 `step`、`velocity`、`sigma`、`scheduler` 语义相同；统一记录真实噪声水平与实际时间序列。
3. 首次实验只增加日志、采样导出和离线评测，不修改更新目标。
4. 统一评测 Prompt、图像尺寸、随机种子、基础模型版本及指标预处理。
5. 每次输出都附完整配置快照、模型检查点、代码 Commit、推理精度及采样器信息。

---

## 2. 六个视觉层级的可操作定义与指标

> 这些层级不是绝对独立的神经特征。需要以**可验证的任务和评价器**来定义每一个层级，尽量避免对同一现象重复打分。

| 层级 | 关键问题 | 建议首选指标/模型 | 可选指标 | 注意事项 |
|---|---|---|---|---|
| L1 全局语义 | Prompt 指定对象、属性、动作是否正确 | VQAScore；TIFA 分项正确率 | CLIPScore、BLIP-ITM | CLIPScore 是整体图文对齐代理，不足以独立检验复杂关系 |
| L2 物体布局 | 物体数目、位置、左右上下、距离、遮挡是否正确 | Grounding DINO 检测 + 规则验证；SpatialScore（若可用） | T2I-CompBench++ 的空间/数量分项、场景图一致性 | 检测失败应单独记录；布局奖励不可只在“成功检测”的样本上计算 |
| L3 前景 | 核心主体是否完整、属性正确、主体特征是否稳定 | Grounding DINO / SAM2 前景掩码 + Region-CLIP / 前景 VQA | DINOv2 区域特征、主体检测置信度 | 遮挡与裁剪对分数有影响；记录掩码质量与主体面积 |
| L4 背景 | 场景类别、背景物体、环境属性是否符合 Prompt | Background-VQA（自建） + 场景文本匹配 | 场景图、背景区域 CLIP 相似度 | 只对明确含背景描述的 Prompt 评分；背景单独裁剪不一定保留上下文 |
| L5 局部纹理 | 局部自然度、伪影、纹理清晰度 | TOPIQ-NR / MUSIQ 的区域或较大 Patch 评价 | MANIQA、LIQE、局部质量异常率 | 常规 IQA 对小 Patch 可能存在分布偏移，需人工验证 |
| L6 像素细节 | 高频、噪声、过锐化、压缩/模糊与参考保真 | NIQE/BRISQUE + 频谱统计（诊断） | LPIPS、DISTS、SSIM（有配对参考时） | 高频能量大不等于质量高；无配对 GT 时不能把 PSNR/SSIM 当作绝对文生图质量 |

**建议分为两套指标：**

- **Core（首轮跑通）**：CLIPScore/VQAScore、Grounding-based Layout、Foreground Object Score、Background-VQA、TOPIQ-NR、NIQE + 高频能量诊断。
- **Extended（论文级）**：TIFA/DSG、SpatialReward/SpatialScore、DINOv2 Region Feature、MUSIQ/LIQE、区域级人工排序及跨模型验证。

对每个指标记录：原始分数、定义方向（↑/↓）、缺失原因、计算耗时、版本和归一化方法。**对六层的改善量以各自的独立评价尺度呈现，不强行将不同量纲的原始分数相加。**

### 2.1 各层评测所需 Prompt 设计

构建固定评价集，建议首轮 `N=120` 个 Prompt，六类情境各 20 个：

1. 单主体明确、属性简单（前景语义）
2. 多主体相对关系（左右、前后、上下）
3. 复杂布局和多对象数量（空间构图）
4. 强背景场景描述（自然环境、室内、城市）
5. 显著材质/纹理描述（玻璃、毛发、木材、金属）
6. 强低层细节需求（纹理锐度、细线、规则重复图案）

每个 Prompt 配有结构化标注 `objects`、`attributes`、`relations`、`foreground`、`background`、`texture_targets`、`low_level_targets`。对于不能可靠评估的层级采用 `not_applicable`，不可记为零。

提供一套训练集外 held-out Prompt，避免只评价训练数据上可能被过拟合的指标。

---

## 3. 中间采样步的图像：必须正确重建与评价

**不能直接对任意带噪 latent `z_t` 做 VAE Decode，然后把噪声图交给 CLIP/IQA 并当作该步的完成质量。**

每条生成轨迹记录：

```text
prompt_id, sample_id, checkpoint_step, sampler, num_inference_steps,
step_index, t_or_sigma, z_t, predicted_velocity,
predicted_x0, intermediate_image, final_image, seed
```

对 Flow Matching 使用项目实际定义的时间方向和速度符号，构造与代码一致的 **clean-image prediction / `predicted_x0`**；如果当前实现使用可以验证的 `x0 = x_t - t v_theta` 约定，可以使用该公式，但不得直接套到所有实现。FewStepRL 的平均速度 `u_theta(z_t,t,r)` 与原始 Flow-GRPO 的瞬时速度/ODE 更新定义需要分别核验。

建议同时记录两类中间表示：

- **Predicted-clean view**：在真实时间步处估计的 `x0`，用 VAE 解码后进行层级打分；标记为估计图像而非完成图像。
- **Prefix + continuation view**：保存当前 `z_t`，固定后续采样规则、噪声及条件，完成剩余轨迹后评分；适合检验某一时间点已经确定了多少最终信息。

首步预测往往不可靠：记录异常/过饱和/失真率。对于 SDE 中间状态，固定后续随机数流并重复采样（至少若干次）估计 continuation 的随机波动。

统一使用**归一化采样进度** `p=k/K` 展示不同步数结果，同时保留真实时间 `t/sigma`；不同 Scheduler 在相同进度的噪声水平不一定相同，应另外按 sigma 匹配或插值对比。少步四个时间点不可直接与普通模型二十个点逐个编号硬匹配。

---

## 4. 研究 I：原始 Flow-GRPO 的视觉层级演化

### 4.1 目的

观察 RL 过程中六层视觉能力的变化，同时研究同一采样轨迹上，六层信息形成的先后、稳定时间与分布。

### 4.2 固定设置

- 仓库：`yifan123/flow_grpo`
- 主干：优先 SD3.5-Medium，确保与 FewStepRL 共用同一基础家族（如果原始配置不可直接跑，记录适配修改）。
- 模型状态：Base / RL Step 50 / 100 / 200 / 400（不强制，依据实际保存位置）。
- 采样步数：先按该仓库对应任务的原始推荐配置（例如 10/20），另设 `K=4,8,16,20` 用于**推理诊断**，不要把推理步数减少与训练步数减少混为一谈。
- Prompt：固定 held-out 集 120 条；每个 Prompt 至少 4 个 seed，用于曲线估计；组内多样性对比另用固定 G=16/24。
- RL 训练主 Reward：首轮**原样保留原始配置**；所有 L1–L6 指标均为旁路/离线评价，不要通过新指标改变训练轨迹。

### 4.3 实验 I-A：六层指标随 RL Step 的变化

对每个 checkpoint `s` 运行相同评测协议，输出：

\[
Q_l(s)=\frac{1}{NP}\sum_{i=1}^{N}\sum_{a=1}^{P}m_l(x_{i,a}^{(s)},c_i),\quad l\in\{1,\dots,6\}.
\]

记录原始 `Q_l(s)` 与相对增益 `Q_l(s)-Q_l(0)`，配置信赖区间/多随机种子波动。若某层指标越小越好，先明确方向再计算提升。

**产出**：六层质量—RL Step 曲线、六层质量—GPU Hours 曲线；不同指标可分别作图或在图中使用相对初始改善，不用未校准的原始值放同轴。

### 4.4 实验 I-B：六层信息随采样步的形成

对固定 checkpoint `s`，记录 `Q_l(s,k)`：当前第 k 个采样位置的 predicted-clean/continuation 图像的质量指标。采用相同 Prompt/seed，绘制：

- 采样进度 `p` × 视觉层级 `l` 的热力图。
- L1–L6 的分层采样演化折线图。
- 案例图：Prompt + `predicted_x0` 拼图 + 最终图 + 各层分数。
- `t/sigma` 作为辅助横轴或单独呈现。

### 4.5 实验 I-C：RL 训练 × 采样时间的二维增益

计算：

\[
\Delta Q_l(s,k)=Q_l(s,k)-Q_l(0,k)
\]

在**相同采样协议与可比噪声水平**下比较同一步的训练前后质量，得到：

`visual_level × sample_progress × training_step`

这张三维张量是研究 I 的核心交付：可以验证 RL 训练是否首先改善语义、布局，或先改善局部细节；不要预先假定某个顺序。

### 4.6 实验 I-D：稳定时间与因果检验（扩展）

定义某层相对最终结果的稳定度，例如 `agreement_l(k, final)`：

- 语义：中间预测与最终图像的问答一致率。
- 布局：物体框匹配、空间关系一致率。
- 前景：分割目标的区域特征相似度。
- 背景：场景内容/属性一致率。
- 纹理：同区域感知特征的相似度。
- 像素：参考为同一条轨迹最终图的 LPIPS/SSIM（**只表示预测与最终图的接近程度，不代表绝对图像质量**）。

定义稳定点 `k*_l`：首次达到最终一致性阈值，并在之后保持。不能只看单个噪声种子，尤其对于随机 SDE 采样。

进一步对一个时间点的速度做轻量扰动并继续采样，观察最终不同层级评分变化，作为归因验证；这属于扩展实验，不能将梯度范数直接解释为因果贡献。

---

## 5. 研究 II：FewStepRL 少步 vs 普通模型的 GRPO 组内多样性

### 5.1 待检验假设与混杂因素

**H1（待检验）**：少步生成器在相同 GRPO 组内采样预算、Prompt 和采样器条件下，其输出多样性低于普通/多步生成器。

必须显式区分影响多样性的三种因素：

1. **模型因素**：原始 SD3.5-Medium vs AnyFlow 少步蒸馏后模型 vs RL 训练后的少步模型。
2. **采样步数因素**：4、8、16、20 等推理步数。
3. **探索因素**：ODE（固定条件下确定性）与 SDE（噪声注入）、噪声幅度、随机分支位置、初始噪声是否共享。

如果少步模型使用 ODE，而普通模型使用 SDE，观察到的差异不能归因于模型是否少步。反过来，如果一个 Prompt 的 G 张图像都共用完全相同的初始 latent、且使用确定性 ODE，生成完全一样可能只是采样设计所致。

### 5.2 主对照矩阵

| 组别 | 模型初始化 | RL 状态 | 推理步数 | 采样器/随机性 | 用途 |
|---|---|---|---|---|---|
| B0 | 原始 SD3.5-Medium | 无 RL | 20（正常配置） | 固定 ODE，多独立初始噪声 | 普通生成器参考 |
| B1 | 原始 SD3.5-Medium | 无 RL | 4 | 与 B0 相同策略 | 仅改变推理步数（可能质量下降） |
| B2 | AnyFlow Stage 1+2 | 无 RL | 4 | ODE，多独立初始噪声 | 少步蒸馏模型参考 |
| B3 | FewStepRL `fewstepgrpo_mix` | RL 之后 | 4 | 与 B2 相同 | 少步 RL 影响 |
| B4 | FewStepRL `flowgrpo_mix` | RL 之后 | 4 | 与 B2 相同 | 少步模型上的 FlowGRPO 对照 |
| B5 | 普通 Flow-GRPO 模型 | RL 之后 | 20 或官方推荐步数 | 尽量匹配随机性 | 普通 RL 对照 |
| B6（扩展） | 普通/少步各自模型 | 固定 | 4/8/16/20 | 匹配不同 SDE 强度 | 步数 × 模型 × 采样器析因分析 |

> **注意：B5 与 B3/B4 不能单独用于推断“步数导致多样性变化”，因为训练初始化、采样器、RL 算法等因素可能同时变化。** 主论断至少依赖 B0/B1/B2 的控制比较，并辅以匹配的采样器实验。`fewstepgrpo_mix` 与 `flowgrpo_mix` 必须先在本地源码确认实际区别，不得只根据脚本名称推断目标函数相同。

### 5.3 GRPO Group 构造

- Prompt：建议固定 `N=120`，训练集外；按复杂语义/布局/前景/背景/纹理六类分层。
- 每个 Prompt 的候选样本数：首轮 `G=16`，然后 `G=4,8,16,24` 做敏感性分析。
- 固定同一组 `initial_seed` 列表；如果要比较 SDE，额外固定并记录每步噪声序列/随机分支策略。
- 与训练 GRPO 组中使用的采样策略分别统计：
  1. **Matched evaluation groups**：用于公平比较的固定离线采样。
  2. **Native training rollout groups**：记录实际训练时的真实多样性，允许各算法不同，但仅作为机制观察。
- 如果训练组内有共享初始噪声、扩展复制或 branching，需要记录 `shared_latent`、branching 时点、SDE window、SDE noise scale。

### 5.4 多样性不止一种：至少四类指标

对于同一 Prompt 的 G 张图像 `x_1,...,x_G`：

**(A) 感知/视觉多样性（primary）**

\[
D_{\mathrm{LPIPS}}(g)=\frac{2}{G(G-1)}\sum_{i<j}\operatorname{LPIPS}(x_i,x_j)
\]

LPIPS 在此衡量同一 Prompt 不同输出的感知差异，并非图像质量。可再记录 DINOv2 特征平均余弦距离与 CLIP image embedding 多样性。

**(B) 语义多样性**

对每张图提取对象列表、属性和 VQA 答案，在同 Prompt 组内统计类别/答案/属性组合的不同种类与熵；记录多样性的同时控制 prompt 忠实度，避免“多出错误物体”被当作好事。

**(C) 布局多样性**

用固定检测器检测 Bounding Boxes、相对位置和场景图，计算匹配后布局位置分散度、不同空间关系比例、场景图编辑距离。对检测失败提供失败率并单独报告；多对象匹配必须考虑类别及实例分配。

**(D) Reward/GRPO 有效多样性**

每组记录：

- `reward_std`、`reward_range`、`reward_iqr`
- `advantage_std` 与有效非零 Advantage 比例
- 所有样本分数几乎相同的 `degenerate_group_rate`
- 图像嵌入的有效秩 / 平均成对相似度
- 组内高质量且相互差异足够大的候选数（`quality-conditioned diversity`）

**关键解释：** 组内 Reward 方差低不等于图像差异小，可能是 Reward Model 饱和或对某类视觉差异不敏感。因此应将视觉特征多样性与 Reward 差异并列报告。

### 5.5 多样性 × 质量联合分析

不能为了追求多样性让图像完全偏离 Prompt。建议画 **Diversity–Quality Pareto**：

- 横轴：组内 LPIPS 或 DINO 特征距离。
- 纵轴：独立评价的语义/布局/偏好质量。
- 点：不同模型与采样步数。

对图像质量匹配的模型分层比较多样性（例如仅比较达到某个质量阈值的组），避免普通模型的明显劣质 4-step 输出因不稳定而看起来“更有多样性”。

统计检验以 Prompt 为单位进行 paired bootstrap / 分层 Bootstrap；不要把同一组内的所有图像对视为独立样本，否则置信区间过窄。

### 5.6 少步模型的采样轨迹多样性

对 B2、B3、B4 的 4-step 生成以及 B0/B5 的多步生成，记录中间 predicted-clean/continuation 图像：

\[
D_l(s,k)=\operatorname{Diversity}_l\left(\hat x_{0,1}^{(s,k)},...,\hat x_{0,G}^{(s,k)}\right).
\]

分别绘制：

- `group diversity × normalized sampling progress`
- `group diversity × RL training step`
- `quality × diversity × sample progress` 的联合关系
- 不同 `noise scale / SDE window` 对组内多样性的影响

这可以判断“多样性差异是在早期形成、后期收缩，还是 RL 训练之后逐渐收缩”。但对于不可靠的早期 `x0` 估计，应以 continuation-based 多样性作为补充验证。

---

## 6. 研究 II：FewStepRL 少步采样的六层视觉信息提升

### 6.1 固定 4-step 路径

使用 B2（RL 前 AnyFlow Stage 1+2）、B3（FewStepGRPO Mix）、B4（FlowGRPO Mix）的同一 Prompt、初始噪声和推理设置。

对四个采样时间位置分别输出：

- `predicted_x0` / VAE 解码图
- `t_or_sigma`
- L1–L6 的各自评价分数
- 当前状态继续采样得到的 final-image 分数（可选）
- 组内多样性（同 Prompt 的 G 个 rollout）

### 6.2 两种“提升”分别统计

**A. 单条轨迹中相邻采样步的增益**

\[
\delta Q_l(s,k)=Q_l(s,k)-Q_l(s,k-1)
\]

这反映同一次推理过程各步的视觉变化；对于无法稳定打分的早期预测，应标记无效值。

**B. RL 训练后的同一步增益**

\[
\Delta Q_l(s,k)=Q_l(s,k)-Q_l(0,k).
\]

这反映 RL 对某个采样位置视觉质量的改善。需要对比同一采样路径与噪声条件。采样轨迹可能随训练改变，因此要同时报告“固定初始噪声自由生成”和“固定 latent 的 teacher-forced 预测”两类分析。

### 6.3 主要待检验现象

- 六层信息在四步内是否具有不同的出现/稳定顺序？
- RL 是否对某些层级的提升更早发生、对另一些提升更晚？
- 少步模型是否出现“语义已确定，但纹理长期欠佳”之类的瓶颈？
- 少步模型组内视觉差异是否在某一步快速收缩？这与该步的质量改善有关还是与采样随机性减弱有关？
- `fewstepgrpo_mix` 与 `flowgrpo_mix` 的时间/区域提升模式是否不同？

均作为假设，不预设结果。

---

## 7. 实验设计总矩阵

| 实验 ID | 仓库 | 比较维度 | 核心输出 | 重要性 |
|---|---|---|---|---|
| E1 | Flow-GRPO | 训练 Step × 六层质量 | 六层收敛曲线 | P0 |
| E2 | Flow-GRPO | 采样 Step × 六层质量 | 采样轨迹热力图 | P0 |
| E3 | Flow-GRPO | 训练 Step × 采样 Step × 六层 | RL 增益三维张量 | P0 |
| E4 | FewStepRL + 普通模型 | 模型类型 × 采样步数 | 同 Prompt 组内多样性 | P0 |
| E5 | FewStepRL | 训练前后 × 4-step × 六层 | 少步视觉层级演化 | P0 |
| E6 | 双仓库 | ODE/SDE × 初始噪声 × 分支策略 | 组内多样性机制拆分 | P1 |
| E7 | 双仓库 | 质量约束 × 多样性 | Diversity–Quality Pareto | P1 |
| E8 | 双仓库 | 贡献 / 速度扰动 × 最终六层分数 | 时间步因果归因 | P2 |

### 7.1 最小可运行版本（MVP）

先做 **E1/E2/E4/E5**：

- 一个 Flow-GRPO 基线及 1–2 个 RL Checkpoint。
- FewStepRL Stage 1+2、一个少步 GRPO Checkpoint、原始 SD3.5-Medium。
- 32 个 held-out Prompt，每个 Prompt 8 或 16 个独立采样。
- Flow-GRPO 取 4–6 个关键采样点；FewStepRL 取全部 4 个采样点。
- 指标先使用 CLIPScore、布局规则、Foreground/Background-VQA、TOPIQ-NR、DINOv2/LPIPS 组内多样性。

再扩展至 120 Prompt、3 次训练随机种子、完整 0/50/100/200/400 Checkpoint。

---

## 8. 数据格式与日志接口（建议）

统一建一个与原仓库解耦的 `visual_analysis/` 目录，可在两个仓库通过少量 Hook 调用。

```text
visual_analysis/
├── configs/
│   ├── flowgrpo_visual.yaml
│   └── fewsteprl_visual.yaml
├── hooks/
│   ├── trajectory_collector.py
│   └── checkpoint_collector.py
├── metrics/
│   ├── semantics.py
│   ├── layout.py
│   ├── foreground_background.py
│   ├── texture.py
│   ├── lowlevel.py
│   └── diversity.py
├── scripts/
│   ├── export_trajectories.py
│   ├── evaluate_hierarchy.py
│   ├── evaluate_diversity.py
│   ├── analyze_convergence.py
│   └── plot_all.py
├── prompts/
│   └── heldout_prompts.jsonl
└── README.md
```

**`trajectories.jsonl` 最低字段：**

```json
{"run_id":"run001","repo":"flow_grpo","commit":"<git_sha>","checkpoint_step":100,"prompt_id":"p001","sample_id":3,"seed":42,"sampler":"sde","num_steps":20,"step_index":5,"sigma":0.75,"latent_path":"...","pred_x0_path":"...","final_path":"...","is_pred_x0_valid":true}
```

**`scores.csv`：**

```text
run_id,repo,checkpoint_step,prompt_id,sample_id,step_index,sigma,view_type,
metric,visual_level,raw_score,score_direction,valid,invalid_reason,score_time_ms
```

**`group_diversity.csv`：**

```text
run_id,repo,model_variant,checkpoint_step,prompt_id,group_size,num_steps,
sampler,noise_scale,shared_initial_latent,step_index,sigma,
metric,diversity_value,reward_std,degenerate_group,valid
```

**`run_metadata.yaml`：**包含 GPU、torch/diffusers 版本、seed、Prompt 列表哈希、模型权重标识、sampler、schedule、SDE 参数、CFG、图像分辨率、耗时、git commit。

大量 latent、预测速度与中间图片使用按需落盘策略；优先只存压缩图、指标与少量核验 latent，避免每个 rollout 都保存完整高维轨迹导致存储爆炸。图片原始分数与 `raw` 统计必须保留，不只保存经过标准化的结果。

---

## 9. 必须生成的论文级图表

| 图号 | 英文图题建议 | 横/纵轴与内容 |
|---|---|---|
| Fig. 1 | Visual-Level Quality Evolution during Flow-GRPO Training | RL Step × 各视觉层级评分/改进量 |
| Fig. 2 | Hierarchical Information Emergence along Sampling Trajectories | 采样进度 × 六层指标 |
| Fig. 3 | Reward Training Gain across Sampling Time and Visual Levels | 采样进度 × 层级，颜色为相对初始 Checkpoint 的提升 |
| Fig. 4 | Within-Prompt Diversity: Few-Step vs Multi-Step Models | 模型/采样设置 × LPIPS、DINO、语义与布局多样性 |
| Fig. 5 | Diversity Dynamics along Flow Sampling | 归一化采样进度 × 组内多样性；多模型/多噪声设置 |
| Fig. 6 | Diversity–Quality Trade-off under Matched Prompts | 横轴多样性，纵轴质量；不同模型散点与置信区间 |
| Fig. 7 | Four-Step Visual Hierarchy Evolution before and after RL | FewStepRL 四步图像网格和六层评分 |
| Fig. 8 | Cross-Checkpoint Visual Gains in FewStepRL | 训练 Step × 4-step × 视觉层级 |
| Fig. 9 | Sampling Stochasticity Ablation | ODE/SDE、noise scale、shared latent 对多样性的影响 |
| Fig. 10 | Qualitative Trajectory and Group Gallery | 每个 Prompt 同组多个样本及其每一步 predicted-clean 图像 |

建议主图构成：**(a) 实验总览；(b) 普通 Flow-GRPO 的六层演化；(c) 少步 vs 多步组内多样性；(d) 少步模型四步视觉层级演化；(e) diversity–quality tradeoff。**

每张图使用实测数据；缺失指标不要插值伪造。若某层指标无法在最早几个步骤可靠计算，图中应明确显示无效/遮罩区域。

---

## 10. 主要实验陷阱与控制变量

1. **不要把 Group Reward Std 当作图像多样性**：Reward 可能饱和；必须至少同时使用 LPIPS/DINO 特征分散度。
2. **不把蒸馏效应与采样步数效应混同**：原始 SD3.5 4-step vs AnyFlow 4-step 的比较也改变了模型；与原始 SD3.5 20-step 联合解读。
3. **ODE/SDE 随机性必须控制**：同一初始 latent 的 ODE 轨迹天然确定；不同 seed 对应不同初始噪声。
4. **不将 early-step 带噪图直接交给正常图像评价器**：使用经过核验的 `predicted_x0` 或 continuation 生成图。
5. **时钟不能只用 step 数**：少步与多步模型的训练/推理成本不同，记录图像数、实际步数、墙钟时间和 GPU Hours。
6. **评价模型与训练 Reward 分离**：评测视觉层级的独立指标不能全部和训练 Reward 同源，否则只能证明优化了代理评分。
7. **保证组内独立性**：统计置信区间以 Prompt（以及训练 seed）为独立单位；样本对不独立。
8. **空间分割失败需要作为结果记录**：只报告检测成功图像容易产生选择偏差。
9. **保证采样时间对齐**：除了归一化进度，按 sigma/噪声水平对齐不同采样器。
10. **避免方向性错觉**：NIQE/LPIPS/错误率与 CLIPScore 方向相反；每项明确 ↑/↓。
11. **注意评测开销**：VQA、检测/分割、IQA 可能显著慢于采样；应缓存、批处理与离线复用中间结果。
12. **检查 Reward Hacking**：如出现多样性变大但语义正确率下降，不能称为改进。
13. **比较的 GRPO 应尽量共享关键设置**：同数据、组大小、训练预算、Reward、LoRA rank、学习率、CFG；无法统一的项如实报告。
14. **仅用四个采样点不足以证明连续时间因果关系**：可使用辅助密集轨迹或扰动实验验证。

---

## 11. 实施步骤及验收条件

### Phase 0：代码审计与小规模验证

- [ ] 克隆两个仓库并保存 Commit。
- [ ] 确认各自训练入口、sampler、velocity 符号、时间网格和 RL loss。
- [ ] 确认 FewStepRL `fewstepgrpo_mix`/`flowgrpo_mix` 名称与实际函数。
- [ ] 验证固定 Prompt 和初始种子生成可复现结果。
- [ ] 验证 predicted-clean 图像的合法性，以及终点与最终图的一致性。
- [ ] 验证多卡样本 id 唯一，不重复/遗漏。

**验收**：每个仓库均可导出一条带逐步状态的有效生成轨迹。

### Phase 1：六层评测框架

- [ ] 创建/标注 held-out Prompt。
- [ ] 接入六层指标并记录耗时。
- [ ] 用人工检查 10–20 组图像，确认指标方向及明显错误是否合理。
- [ ] 支持缺失项、Mask 失败和质量异常的显式标记。

**验收**：输入一组生成图可导出完整 `scores.csv`，并正确记录不适用项。

### Phase 2：Flow-GRPO 演化实验

- [ ] 收集多个训练 Checkpoint。
- [ ] 每个 Checkpoint 固定 Prompt/seed 生成逐步 predicted-clean。
- [ ] 绘制 `Q_l(s)`、`Q_l(s,k)` 和 `ΔQ_l(s,k)`。

**验收**：完成 Fig. 1–3 与可核对的原始数据。

### Phase 3：FewStepRL 多样性实验

- [ ] 准备 B0–B5，对齐 Prompt/seed/随机性。
- [ ] 生成同 Prompt 的 G=16 个候选。
- [ ] 同时计算视觉组内多样性与 Reward Std。
- [ ] 做 ODE/SDE、G、num_steps 的消融。
- [ ] 报告质量约束下的多样性。

**验收**：能够回答“少步模型在何种控制条件下更少/不更少多样性”，并提供误差条与显著性分析。

### Phase 4：FewStepRL 四步视觉演化

- [ ] 对 B2/B3/B4 记录所有 4-step 状态。
- [ ] 比较逐步六层质量、同 Prompt 组内多样性与训练前后提升。
- [ ] 绘制时间 × 区域 × 层级的可解释分析图。

**验收**：完成 Fig. 5、7、8、10，明确哪些仅是相关性，不把其解释为因果贡献。

### Phase 5：可选的方法设计

如果实验显示某些时间段/层级存在稳定差异，再设计：

- Reward-aware timestep sampling
- Layer/Region-aware reward scheduling
- Group diversity-aware adaptive rollout
- Cost-aware reward evaluation

所有加速方法需要用**同等图像质量下的 GPU Hours**检验，避免仅用更新次数作为效率结论。

---

## 12. 给 AI 编程助手的执行指令（可直接复制）

> 你是熟悉 PyTorch、Diffusers、Flow Matching、GRPO 与 SD3.5 的研究工程师。请按本 Markdown 文档，为 `yifan123/flow_grpo` 和 `EDAPINENUT/fewsteprl-sd35` 分别开发**旁路分析工具**，不要擅自修改原始 RL loss。先检查本地实际源代码，记录 Git Commit、可用函数与中间状态张量的形状，核验 predicted-x0 的符号和调度器时间方向。第一阶段只实现：固定 Prompt/seed 采样轨迹导出；六层视觉评价的统一接口；同 Prompt 的 LPIPS/DINO/语义/布局多样性；三类必需图（Flow-GRPO 训练×采样时间演化、少步vs多步组内多样性、FewStepRL 四步层级演化）。先用 4 个 Prompt × 4 个样本完成真实 GPU smoke test，再按本文对照矩阵扩展。必须提供可运行脚本、CLI、配置文件、单元测试、日志 schema、原始 CSV、图像与 README。不得虚构指标，不得把带噪 latent 解码误当作完成图像，不得把 Reward Std 当作视觉多样性。任何无法从源码核实的细节，请在实施报告中列为待确认项。

---

## 13. 参考链接

1. Flow-GRPO 官方仓库：<https://github.com/yifan123/flow_grpo>
2. Flow-GRPO 论文：<https://arxiv.org/abs/2505.05470>
3. FewStepRL-SD3.5 官方仓库：<https://github.com/EDAPINENUT/fewsteprl-sd35>
4. FewStepRL 论文：<https://arxiv.org/abs/2611.08240>
5. FewStepRL FlowGRPO Mix 配置：<https://github.com/EDAPINENUT/fewsteprl-sd35/blob/main/scripts/train/configs/flowgrpo_mix.yaml>
6. TIFA：<https://tifa-benchmark.github.io/>
7. VQAScore / GenAI-Bench：<https://linzhiqiu.github.io/papers/genai_bench/>
8. Grounding DINO：<https://github.com/IDEA-Research/GroundingDINO>
9. SAM 2：<https://github.com/facebookresearch/sam2>
10. DINOv2：<https://github.com/facebookresearch/dinov2>
11. IQA-PyTorch（IQA 模型统一接口）：<https://github.com/chaofengc/IQA-PyTorch>

> **研究结论预设：无。** 文档中“少步多样性更小”“语义先于纹理”等均是待验证假设，不是本项目已有实测结论。
