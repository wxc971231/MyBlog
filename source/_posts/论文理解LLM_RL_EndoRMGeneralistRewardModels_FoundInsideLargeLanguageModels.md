---
title: 论文理解 【LLM-RL】——【EndoRM】Generalist Reward Models-Found Inside Large Language Models
date: 2025-09-08 11:05:24
index_img: /img/论文理解LLM_RL_EndoRMGeneralistRewardModels_FoundInsideLargeLanguageModels/index.png
tags:
  - LLM-RL
  - LLM
  - RL
  - EndoRM
categories:
  - 机器学习
  - 论文理解
description: 本文发现任何通过标准next token prediction训练的LLM内部已隐含通用奖励模型，并基于逆强化学习提出一种从预训练模型中直接提取内生奖励函数的方法，为LLM对齐提供了新的理论基础和实践路径。
---

- 首发链接：[论文理解 【LLM-RL】——【EndoRM】Generalist Reward Models-Found Inside Large Language Models](https://blog.csdn.net/wxc971231/article/details/151181272)
- 文章链接：[Generalist Reward Models: Found Inside Large Language Models](https://arxiv.org/abs/2506.23235)
- 发表：2025年6月（arXiv 预印本）
- 领域：LLM-RL
- 一句话总结：在 LLM 对齐过程中，当前主流范式 RLHF 使用昂贵且难以扩展的人类偏好数据来训练奖励模型，基于 AI 反馈的替代方案 RLAIF 虽然降低了成本但缺乏严格的理论基础。本文指出，**任何通过标准 “next token prediction” 训练得到的 LLM 内部已经隐含了一个通用奖励模型，其本质上等价于离线逆强化学习所得的奖励函数**，LLM 的 logits 可以直接解释为 Soft Q 函数，无需额外训练即可从中恢复 “内生奖励函数”，并进一步证明**用该奖励进行 LLM-RL 可以避免模仿学习中的误差累积，实现更优的理论性能**。实验表明，该方法在多项基准上超越了现有的 “LLM-as-a-judge” 及显式训练的奖励模型，揭示了对齐范式可由 “外部奖励构造” 转向 “内在奖励挖掘” 的可能
-----
- 摘要：大型语言模型（LLMs）的对齐严重依赖于在昂贵的人类偏好数据上训练的奖励模型。尽管最近的研究探索了通过人工智能反馈来绕过这一成本，但这些方法通常缺乏严格的理论基础。在本文中，我们发现，**通过标准的下一个标记预测训练的任何大型语言模型中，已经隐含地存在一个强大的通用奖励模型**。我们证明，这种内生奖励并非一种启发式方法，而是**理论上等同于通过离线逆强化学习（IRL）学习到的奖励函数**。这种联系使我们能够直接从基础（预训练或监督微调）模型中提取高质量的奖励信号，而无需进一步训练。至关重要的是，我们还证明，使用这种内生奖励进行后续强化学习，会导致一个**具有可证明的更优误差界限的策略**，与基础模型相比。据我们所知，这是**第一个关于强化学习对大型语言模型有效的理论证明**。我们的实验验证了这一理论，表明我们的方法不仅优于现有的基于 LLM 的评判方法，甚至可以超越明确训练的奖励模型。这些发现表明，奖励建模阶段可以被一种原则化的方法所取代，即从预训练中已经捕获的知识中提取知识，预示着一种更高效、更强大、更可扩展的大型语言模型对齐范式，以及多模态模型的对齐

# 1. 背景
## 1.1 LLM 的 MDP 表述
- 从强化学习的角度分析，LLM 本身可以视作文本生成策略 $\pi(\cdot|s)$，生成文本的过程可以被自然地建模为具有确定性环境转移的 [MDP](https://blog.csdn.net/wxc971231/article/details/119381364) $<\mathcal{S,V},r,P,\rho,H>$，其中：
    1. 状态空间 $\mathcal{S}$：有限长度 token 序列的集合
    2. 动作空间 $\mathcal{V}$： 词表
    3. 初始状态分布 $\rho$：从中采样得到作为初始状态的提示词 $s_{1}=\left(x_{1}, x_{2}, \cdots, x_{m}\right), \forall x_i\in \mathcal{V}$
    4. 生成序列长度 $H$：为分析清晰起见，作者假设模型响应长度均匀且恰好为 $H$（需要在 \<EOS\> token 后进行适当的 Padding）
    5. 奖励函数 $r$：每个 timestep $h\in [H]$ 的即时奖励 $r_h\in [0,1]$
    6. 环境转移 $P$：每个 timestep $h\in [H]$，LLM 策略生成动作 $a_h\sim \pi(\cdot|s_h)$，并确定性地转移到 $s_{h+1}=s_h \oplus a_h$
- 基于以上 MDP 设计，LLM 生成的文本可以看作 rollout 轨迹 $\tau$，定义价值函数 $V^\pi$ 作为策略 $\pi$ 的质量度量
    $$
    V^\pi = \mathbb{E}_{\tau\sim \pi}\left[\sum_{h=1}^H r(s_h, a_h)\right]\tag{1}
    $$

## 1.2 Next Token Prediction
- 当前 LLM 遵循 Next Token Prediction 范式进行**预训练**，即最大化模型在前驱 token 序列的条件下预测下一个 token 的概率。形式化地讲，给定数据集 $\mathcal{D}=\left\{\left(s_{1}^{i}, a_{1:H}^{i}\right)\right\}_{i=1}^{n}$，Next Token Prediction 解决以下优化问题
    $$
    \max _{\pi} \sum_{i=1}^{n} \sum_{h=1}^{H} \log \left(\pi\left(a_{h}^{i} \mid s_{h}^{i}\right)\right)\tag{2}
    $$
    值得注意的是，**这可以被看作是一种行为克隆方法（BC），属于模仿学习**
## 1.3 基于人类反馈的强化学习（RLHF）
- Next Token Prediction 是有效的模仿学习方案，但使 LLM 的行为对齐人类价值观需要更直接的反馈形式。人类反馈强化学习（RLHF）是这个任务的标准范式，其可分为两个步骤
    1. **训练奖励模型**$R_\phi$：奖励模型 RM 的训练目标是**给予人类偏好回答更高的标量分数**。给定偏好数据集 $\mathcal{D}_{\text{pref}}=\{(x, y_w, y_l)_i\}$，其中 $x$ 表示给定的提示（问题），$y_w, y_l$ 分别表示人类标记者偏好和不偏好的回答。遵循 Bradley-Terry 模型，$y_w$ 比 $y_l$ 更被偏好的概率被建模为
        $$
        \begin{aligned}
        P\left(y_{w} \succ y_{l} \mid x\right)
        &=\sigma\left(R_{\phi}\left(x, y_{w}\right)-R_{\phi}\left(x, y_{l}\right)\right) \\
        &=\frac{\exp R_{\phi}\left(x, y_{w}\right)}{\exp R_{\phi}\left(x, y_{w}\right)+\exp R_{\phi}\left(x, y_{l}\right)}
        \end{aligned}
        \tag{3}
        $$
        其中 $\sigma$ 是 Sigmoid 函数。通过最小化偏好的负对数似然来训练 RM，即
        $$
        \mathcal{L}_{R M}(\phi)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}_{\text {pref }}}\left[\log \sigma\left(R_{\phi}\left(x, y_{w}\right)-R_{\phi}\left(x, y_{l}\right)\right)\right]
        \tag{4}
        $$
    2. **强化学习微调**：最大化 RM 反馈的同时约束策略偏移以避免 reward hacking，如下
        $$
        \max _{\theta} \mathbb{E}_{x \sim \mathcal{\rho}, y \sim \pi_{\theta}(\cdot \mid x)}\left[R_{\phi}(x, y)-\beta \cdot D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot \mid x) \| \pi_{\mathrm{SFT}}(\cdot \mid x)\right)\right]\tag{5}
        $$
- RLHF 最早被用于解决 RL 领域常见的控制问题，见 [论文理解【IL - IRL】 —— Deep Reinforcement Learning from Human Preferences](https://blog.csdn.net/wxc971231/article/details/121785301)
## 1.4 逆强化学习（IRL）
- IRL 是 BC 以外的另一类模仿学习方法，它的目标是**从给定的专家演示数据集**$\mathcal{D}$**中恢复奖励模型**
    > 模仿学习假设专家策略是最优的，IRL 学到的奖励函数要让 $\mathcal{D}$ 看起来是最优策略生成的
- 一种突出且有理论支撑的 IRL 方法是最大熵逆强化学习（[MaxEnt](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)），这种方法恢复的奖励可以在**解释专家演示的同时对数据中发现的行为保持最大的不确定性**，这一原则导致了一个极小极大优化问题
    $$
    \max _{r} \min _{\pi}\left(\mathbb{E}_{\tau \sim \pi^E}\left[\sum_{h=1}^{H} r\left(s_{h}, a_{h}\right)\right]-\mathbb{E}_{\tau \sim \pi}\left[\sum_{h=1}^{H} r\left(s_{h}, a_{h}\right)+\alpha H\left(\pi\left(\cdot \mid s_{h}\right)\right)\right]\right)
    \tag{6}
    $$
    其中 $\mathbb{E}_{\tau \sim \pi^E}$ 是对专家策略 $\pi^E$ 诱导的轨迹取期望（通过数据集 $\mathcal{D}$ 近似），$\mathbb{E}_{\tau \sim \pi}$ 是对学到的策略 $\pi$ 诱导的轨迹取期望，熵项 $H\left(\pi\left(\cdot \mid s_{h}\right)\right)=\mathbb{E}_{a_{h} \sim \pi\left(\cdot \mid s_{h}\right)}\left[-\log \pi\left(a_{h} \mid s_{h}\right)\right]$
    > 上式可以理解为：
    > 1. 外层 $\max_r$ 希望找到一个奖励函数 $𝑟$，使得它能最大化专家策略 $\pi^E$ 与当前学习策略 $\pi$ 之间的差距，这个差距越大越能体现专家策略的优势，因此这是要**找到一个解释专家行为的奖励模型**
    > 2. 内层 $\min_\pi$ 只对第二项有作用，它表示对于给定的奖励函数 $r$，找到最大化累计奖励并保持高探索性的策略 $\pi$（$r$ 下的最优最优策略）。**这保证了外层学到奖励函数 $r$ 的解释强度，即该解释下专家策略即使面对很好的策略仍具有优势**
- 将 EaxEnt IRL 和 RLHF 对比分析，显然 **RLHF 可以视作 IRL 范式的一个实例**，它没有直接进行上式内层的 $\min_\pi$ 最优策略求解，而是直接最大化偏好数据集中“胜者”与“败者”之间的奖励差距
    1. RLHF 中的奖励模型是通过 **“样本对” 级别的比较**学到的
    2. EaxEnt IRL 中的奖励模型是通过 **“样本分布” 级别的比较**学到的 
# 2. 本文方法
## 2.1 利用 IRL 恢复奖励函数
- 本文的核心思想是**通过 IRL 框架直接恢复能解释专家数据集**$\mathcal{D}$**的最优奖励函数**。但直接应用公式 (6) 是困难的，因为通常需要高成本的 online rollout 来计算 $\mathbb{E}_{\tau \sim \pi}$。为避免交互成本，作者使用了 offline IRL 方法 Inverse Soft Q-learning
    > - Inverse Soft Q-learning 和 Soft Q-Learning 是一对互为逆问题的方法，二者都基于最大熵思想，后者是经典 model-free RL 算法 SAC 的前身，详见 [论文理解【RL经典】—— 【SQL】Reinforcement Learning with Deep Energy-Based Policies](https://blog.csdn.net/wxc971231/article/details/127260196)
    > - 最大熵 RL 里，对于任意策略 $\pi$ 定义 Soft Bellman operator
        > $$
        \begin{aligned}
        Q^{\pi}(s, a)&=r(s, a)+\gamma \mathbb{E}_{s^{\prime} \sim P(\cdot \mid s, a)}\left[V^{\pi}\left(s^{\prime}\right)\right]\\
        V^{\pi}(s)&=\mathbb{E}_{a \sim \pi}Q^{\pi}(s, a)+\alpha H(\pi(\cdot \mid s)) \\
        &=\mathbb{E}_{a \sim \pi}\left[Q^{\pi}(s, a)-\alpha \log \pi(a \mid s)\right] .
        \end{aligned}
        $$
    > - 基于[能量模型](https://blog.csdn.net/wxc971231/article/details/126918635)定义**最优**策略如下 $\pi^{\star}(a \mid s)=\frac{\exp (Q^*(s, a) / \alpha)}{\sum_{a^{\prime}} \exp \left(Q^*\left(s, a^{\prime}\right) / \alpha\right)}$ 回代得到**最优** Soft Q-value 和**最优** Soft V-value 的闭式解：
        > $$
        \begin{aligned}
        Q^*(s, a)&=r(s, a)+\gamma \mathbb{E}_{s^{\prime} \sim P(\cdot \mid s, a)}\left[V^*\left(s^{\prime}\right)\right]\\
        V^*(s)&=\alpha \log \sum_{a^{\prime}} \exp \left(Q^*\left(s, a^{\prime}\right) / \alpha\right)
        \end{aligned}
        $$
- Inverse Soft Q-learning 旨在找到能解释 $\mathcal{D}=\left\{\left(s_{1}^{i}, a_{1:H}^{i}\right)\right\}_{i=1}^{n}$ 的 $Q^*$ 函数，这需要求解
    $$
    \max_{Q^*} \frac{1}{n} \sum_{i=1}^{n} \sum_{h=1}^{H}\left[Q^*\left(s_{h}^{i}, a_{h}^{i}\right)-\alpha \log \left(\sum_{a_{h} \in \mathcal{V}} \exp \left(Q^*\left(s_{h}^{i}, a_{h}\right)\right) / \alpha\right)\right]
    \tag{7}
    $$
    这是把公式 (6) 和最大策略熵思想结合得到的
    > 具体地，把专家数据 $\mathcal{D}$ 看作由最优策略 $\pi^*$ 生成，最大化其对数似然
        >$$
        \max _{Q^{*}} \sum_{i=1}^{n} \sum_{h=1}^{H} \log \pi^{*}\left(a_{h}^{i} \mid s_{h}^{i}\right)
        $$
    > 带入 $\pi^{\star}(a \mid s)=\frac{\exp (Q^*(s, a) / \alpha)}{\sum_{a^{\prime}} \exp \left(Q^*\left(s, a^{\prime}\right) / \alpha\right)}$，得到
        >$$
        \max _{Q^{*}}  \sum_{i,h} \left[\frac{1}{\alpha}Q^*(s_h^i, a_h^i) -\log  \sum_{a^{\prime}} \exp \left(Q^*\left(s_h^i, a^{\prime}\right) / \alpha\right) \right]
        $$
    改成经验平均就得到式(7)了
- 一旦解得 $Q^*$，可以如下恢复奖励函数（令 $\gamma=1$）
    $$
    \begin{aligned}
    r^{\star}\left(s_{h}, a_{h}\right)
    &=Q^{\star}\left(s_{h}, a_{h}\right)-\gamma \mathbb{E}_{s_{h+1}}\left[V^*\left(s_{h+1}\right)\right]\\
    &=Q^{\star}\left(s_{h}, a_{h}\right)-\alpha \log \left(\sum_{a_{h+1} \in \mathcal{V}} \exp \left(Q^{\star}\left(s_{h+1}, a_{h+1}\right)\right) / \alpha\right)
    \end{aligned}
    \tag{8}
    $$
## 2.2 内生奖励
- 作者注意到，优化公式 (7) 得到的 $Q^*$ 实际无需重新训练，它已经蕴含在 Next-Token Prediction 训练的结果中。在 2.1 节的推理过程中，我们是从对 $\mathcal{D}$ 做最大似然估计找到 $\pi^*$ 进而推出 $Q^*$ 的，其中 $\pi^*$ 又是一种源自能量模型的 softmax 策略
    $$
    \pi^*(\cdot \mid s_{h})=\operatorname{softmax}\left(Q^*\left(s_{h}, \cdot\right) ; \alpha\right)\tag{9}
    $$
- **这恰好是 Next-Token Prediction 的训练目标**，如式(2)所示，LLM 对训练语料做最大似然估计，得到的生成策略是在 logits 上做 softmax，设状态 $s_h$ 的 logits 由 $\hat{f}$ 得到，LLM 可以表示为
    $$
    \widehat{\pi}\left(\cdot \mid s_{h}\right)=\operatorname{softmax}\left(\widehat{f}\left(s_{h}, \cdot\right) ; \alpha\right)\tag{10}
    $$
    这说明 **LLM 的 logits 不仅仅是任意的分数，而是一种原则性的**$Q$**函数，隐含地代表了模型训练数据 $\mathcal{D}$ 的最优奖励函数**
- 基于以上观察，给定任何通过 Next-Token Prediction 训练的语言模型，无论在预训练还是微调阶段，都可以提取模型输出的 logits 作为 Soft Q 函数 $\hat{Q}=\hat{f}$，并从中恢复奖励值
    $$
    \widehat{r}\left(s_{h}, a_{h}\right):=\widehat{Q}\left(s_{h}, a_{h}\right)-\alpha \log \left(\sum_{a_{h+1} \in \mathcal{V}} \exp \left(\widehat{Q}\left(s_{h+1}, a_{h+1}\right)\right) / \alpha\right)
    \tag{11}
    $$
    进一步定义 Soft V 函数 $V_{\widehat{Q}}\left(s_{h}\right):=\alpha \log \left(\sum_{a_{h} \in \mathcal{V}} \exp \left(\widehat{Q}\left(s_{h}, a_{h}\right) / \alpha\right)\right)$，得到 LLM 的内生奖励为
    $$
    \widehat{r}\left(s_{h}, a_{h}\right)=\alpha \log \left(\widehat{\pi}\left(a_{h} \mid s_{h}\right)\right)+V_{\hat{Q}}\left(s_{h}\right)-V_{\hat{Q}}\left(s_{h+1}\right)
    \tag{12}
    $$
- **`内生奖励(Endogenous Reward)`** 可以从三个方面理解
    1. **内生奖励**$\hat{r}$**可以和 $\log(\hat{\pi}(a_h|s_h))$ 在收敛位置上等价**。根据奖励塑形理论，$\tilde{r}(s_h,a_h)$ 可以被看作 $\log(\hat{\pi}(a_h|s_h))$ 基础上的塑形奖励，引入的 $V$ 只是让学习更稳定，不影响收敛位置
    2. **内生奖励是 “统计驱动” 的，常见合理的序列**$\hat{\pi}(\tau|s_1)$**更大，会被赋予更高的奖励**。对整个轨迹的内生奖励求和可见
        $$
        \begin{aligned}
        \hat{r}(\tau)
        &= \sum_{h=1}^H \hat{r}(s_h, a_h) = \alpha \sum_{h=1}^H \log(\hat{\pi}(a_h|s_h)) + V_{\hat{Q}}(s_1)\\
        &= \alpha \log(\hat{\pi}(\tau|s_1)) + V_{\hat{Q}}(s_1)
        \end{aligned}
        $$
    3. **现有的生成式奖励模型（GenRM）是内生奖励的一种特例**。以往的生成式奖励模型其实都是在用 LLM 内部的概率分布构造奖励，只是场景受限，内生奖励是一种更一般的形式
        > 问答场景中，某些奖励模型用 LLM 去回答一个 “Verifier Prompt”（如：“Is the answer correct?”），然后取“是”的概率 $\pi(\text{Yes}|x, y, P)$ 作为奖励，如果把初始状态设为 $s_1=(x, y, P)$，动作为 $a_1=\text{Yes}$，那么奖励 $\hat{r}(\tau) = \alpha \log \hat{\pi}(\text{Yes}|x,y,P)$ 和内生奖励定义完全相同

# 3. 内生奖励的理论论证
## 3.1 内生奖励的误差分析
- 现在考虑对内生奖励 $\hat{r}$ 的准确性进行分析，设 $\pi^E$ 是关于未知真实奖励 $r^*$ 的熵正则化最优策略，即满足
    $$
    r^*\left(s_{h}, a_{h}\right)=\alpha \log \left(\pi_{E}\left(a_{h} \mid s_{h}\right)\right)+V_{Q^*_{r^*}}\left(s_{h}\right)-V_{Q^*_{r^*}}\left(s_{h+1}\right)
    $$
    其中 $Q^*_{r^*}$ 是基于 $r^*$ 的熵正则化最优动作状态价值函数。
- 由于奖励存在不唯一性（即不同的奖励函数可能导出相同的专家策略），无法对 $|\hat{r}(s_h, a_h) − r^*(s_h, a_h)|$ 进行任何评估。然而，建立奖励模型的主要目标之一是执行成对比较，因此作者转而**分析利用奖励模型比较两个响应时的表现**。形式化地说，利用奖励模型评估一对响应 $(\tau, \tau')$ 的偏好，其中 $\tau = (s_1, a_{1:H})，\tau' = (s_1, a'_{1:H})$，根据 1.3 节的 Bradley-Terry 模型，奖励 $r$ 诱导的偏好分布可以写为 
    $$
    \mathbb{P}_{r}\left(\tau \succ \tau^{\prime} \mid \tau, \tau^{\prime}\right)=\sigma\left(r(\tau)-r\left(\tau^{\prime}\right)\right)
    \tag{13}
    $$
    其中 $\sigma$ 是 sigmoid 函数。误差分析的目标是分析真实偏好分布 $\mathbb{P}_{r^*}\left(\tau \succ \tau^{\prime} \mid \tau, \tau^{\prime}\right)$ 与推断分布 $\mathbb{P}_{\hat{r}}\left(\tau \succ \tau^{\prime} \mid \tau, \tau^{\prime}\right)$ 之间的差异
- 作者通过定理1证明：**如果用于提取奖励的 LLM $\hat{\pi}$ 在响应上的对数概率接近于潜在专家 $\pi^E$，那么 $\hat{r}$ 诱导的偏好分布也会接近于**$r^*$**诱导的分布。这说明从 LLM 中提取的内生奖励可以在理论上继承其表现**
    > 定理1：在 token MDP 中，设未知真实奖励为 $r^⋆$，$π^E$ 是熵正则化的最优策略。考虑 $\hat{\pi}$是通过 next token prediction 训练得到的策略，$\hat{r}$ 是在式(12) 中定义的内生奖励，对于任意相应二元组 $(\tau, \tau')$，有
        > $$
        D_{\mathrm{TV}}\left(\mathbb{P}_{r^{\star}}\left(\cdot \mid \tau, \tau^{\prime}\right), \mathbb{P}_{\widehat{r}}\left(\cdot \mid \tau, \tau^{\prime}\right)\right) \leq \frac{\alpha H}{2} \varepsilon_{\pi}
        $$
    > 其中 $D_{\mathrm{TV}}(p, q)=(1 / 2) \sum_{x \in \mathcal{X}}|p(x)-q(x)|$ 是分布 $p,q\in \triangle(\mathcal{X})$ 之间的全变差距离，偏差值 $\varepsilon_{\pi}:=\max _{h \in[H]} \max _{\left(s_{h}, a_{h}\right)}\left|\log \left(\pi^{\mathrm{E}}\left(a_{h} \mid s_{h}\right)\right)-\log \left(\widehat{\pi}\left(a_{h} \mid s_{h}\right)\right)\right|$。该定理可以理解为：
## 3.2 由内生奖励微调的 LLM 的误差分析
- 提取奖励函数的最终目标是通过 RL 训练更优的策略。本节分析新学到的策略的表现，对比以下策略在 $r^*$ 下的次优程度：
    1. 直接在 $\mathcal{D}$ 上应用 Next-Token Prediction 得到基线策略 $\hat{\pi}$
    2. 先根据式(12) 构建内生奖励 $\hat{r}$，然后强化学习得到新策略 $\pi^{\mathrm{RL}}=\operatorname{argmax}_{\pi} V_{\widehat{r}}^{\pi}$，忽略求解优化误差
- 作者通过定理2证明：****$\pi^{RL}$**在次优边界（sub-optimality bound）中对响应长度 $H$ 呈线性依赖，而 $\hat{\pi}$ 则呈二次依赖，这反映了模仿学习的累积误差问题。而逆强化学习的根本优势在于它直接模仿专家动作，而是恢复底层奖励函数并在此奖励下执行 RL，从而消除了累积误差问题**
    > 定理2：在 token MDP 中，设未知真实奖励为 $r^⋆$，$π^E$ 是熵正则化的最优策略。考虑 $\hat{\pi}$ 是通过 next token prediction 训练得到的策略。$\hat{r}$ 是在式(12) 中定义的内生奖励，基于其强化学习得到新策略 $\pi^{\mathrm{RL}}=\operatorname{argmax}_{\pi} V_{\widehat{r}}^{\pi}$，有：
        > $$
        V_{\pi E}^{r \star}-V_{\hat{\pi}}^{r \star} \preceq H^{2} \varepsilon_{\pi}, \quad V_{\pi E}^{r \star}-V_{\pi R L}^{r \star} \preceq H \varepsilon_{\pi}
        $$
    > 其中 $V_r^\pi$ 表示策略 $\pi$ 在奖励 $r$ 下的价值，偏差值 $\varepsilon_{\pi}:=\max _{h \in[H]} \max _{\left(s_{h}, a_{h}\right)}\left|\log \left(\pi^{\mathrm{E}}\left(a_{h} \mid s_{h}\right)\right)-\log \left(\widehat{\pi}\left(a_{h} \mid s_{h}\right)\right)\right|$，该定理可以扩展到无限时域情形
## 3.3 迭代改进的无效性
- 我们基于预训练 LLM 得到内生奖励 $\hat{r}$，再利用内生奖励训练更好的 LLM $\pi^{RL}$，这种自我改进过程是不可重复的。这是因为 $\pi^{RL}$ 已经是奖励 $\hat{r}$ 对应的最优策略，从 $\pi^{RL}$ 提取的内生奖励依然是 $\hat{r}$

# 4. 实验
- 作者的实验主要考虑三个问题
    1. 与启发式基线和明确训练的 SOTA 奖励模型相比，免训练内生奖励模型（EndoRM）在基准测试上的表现如何？
    2. EndoRM 是否具有强大的指令遵循能力，作为一个可以提示的通用奖励模型？
    3. 使用 EndoRM 进行 RL 能否产生更好的策略，实现自我改进？
## 4.1 多样任务轨迹对的奖励准确性（Q1）
- 在 RM-Bench 上评估奖励模型面对多样化任务响应二元组时预测偏好准确率（计算每个响应的 token 奖励总和，分数更高的被视为“chosen”）
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_RL_EndoRMGeneralistRewardModels_FoundInsideLargeLanguageModels/img_001.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

    上面四个方法是显示训练的奖励模型，下面四个是免训练方法。结果显示，**EndoRM 不仅显著优于所有免训练基线，还在平均准确率上超过了 SOTA 的显式训练奖励模型**
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_RL_EndoRMGeneralistRewardModels_FoundInsideLargeLanguageModels/img_002.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

## 4.2 奖励的指令遵循能力（Q2）
- 使用 Multifaceted-Bench（涵盖多样化用户偏好）和领域特定偏好数据集 DSP（包含不同专业领域的偏好）测试内生奖励是否能根据提示进行 “定制”
- 将四个领域的系统提示作为输入，得到四个领域专属的 EndoRM，测试它们在四个领域测试集上的表现。
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_RL_EndoRMGeneralistRewardModels_FoundInsideLargeLanguageModels/img_003.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

    注意到出现了明显的 **“对角线” 模式**：每个 EndoRM 在其所属领域的准确率最高，说明 **EndoRM 并不是固定评估器，而是一个 可提示的动态裁判，继承了 LLM 的指令遵循能力**

## 4.3 通过强化学习实现自我改进（Q3）
- 使用 MATH-lighteval 验证定理2的核心结论：RL + EndoRM 能改进基线策略
    1. 使用 Qwen2.5-Math-7B 作为基线模型和 EndoRM
    2. 在 MATH-lighteval 上进行 RL 微调，EndoRM 的参数保持固定，仅用于奖励
    3. 设置：PPO 算法，最大长度 1024，KL 系数 0.01
- 结果如下所示，**RL + EndoRM 微调后在所有五个数学基准上全面优于基线模型**。在附录中示例显示：基线模型在长推理时容易跑偏甚至输出 Python 代码，而 RL 优化后的模型则给出简洁清晰的解答
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_RL_EndoRMGeneralistRewardModels_FoundInsideLargeLanguageModels/img_004.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

# 5. 讨论
- 内生奖励的概念对 LLM 对齐与发展有深远意义，作者进行了详细讨论，总结如下
    1. **对齐范式转变**：
        - 传统 **三阶段对齐流程（SFT → 奖励建模 → RL 微调）被大幅简化**：奖励建模阶段不再需要
        -  **偏好数据可以直接融入预训练或 SFT**，增强模型的内生奖励能力
        - 这种方式与 “强化预训练（Reinforcement Pre-training）” 的理念类似，但更有理论依据
        - 好处：降低工程复杂度、计算成本和部署时间
    2. **个性化与可控性增强**
        - 外部奖励模型通常是“一刀切”，很难为特定用户或企业定制
        - **内生奖励是动态的、可提示的**：评价标准可以在推理时通过自然语言指令来改变
        - 用户只需调整提示，即可引导 RL 训练更贴合自身价值观
        - 这使得对齐过程从 “静态后处理” 变为 “动态交互式对话”
    3. **强化学习知识蒸馏**
        - 传统知识蒸馏：学生模型模仿教师的输出
        - RLAIF：教师生成偏好标签，间接指导学生
        - EndoRM：直接从教师模型的 **logits 提取内生奖励**，用 RL 优化学生策略。**学生学到的是教师的“底层判断原则”，而不仅仅是表层输出，这意味着更强的蒸馏能力，小模型可能更有效继承大模型的能力**
    4. 将强化学习扩展到文本之外
        - RLHF 依赖人工偏好标签，在图像、视频、音频等模态更难收集
        -  **内生奖励机制不限于语言，只要模型是自回归生成式，就能适用，因此可以扩展到图像生成、视频合成、音乐创作** 等领域
        - 为多模态模型对齐提供了可扩展的解决方案
    5. 局限性与未来工作
        - **潜在风险**：内生奖励完全依赖模型的内部世界观，如果模型带有偏见或错误，它可能会自我强化这些问题。
        - **可能后果**：模型可能奖励自身生成的错误或有害输出。
        - **未来方向**：
            1. 融合混合方法：以内生奖励为主要稠密信号，同时用少量高质量人工反馈或规则进行矫正
            2. 研究更安全的 **提示工程** 技术，用于稳健地引导奖励信号。




