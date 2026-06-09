---
title: 论文理解【LLM-Agent】—— 【ReAct】Synergizing Reasoning and Acting in Language Models
date: 2026-06-09 14:43:15
index_img: /img/论文理解LLM_Agent_ReActSynergizingReasoningandActinginLanguageModels/index.png
tags:
  - LLM-Agent
  - LLM
categories:
  - 机器学习
  - 论文理解
  - LLM-Agent
description: 本文介绍LLM-agent领域的经典方法ReAct，它将LLM的自然语言推理能力和动作生成能力结合，使其同时适用于各类NLP和控制任务，并起到1+1>2的效果
---

- 首发链接：[论文理解【LLM-Agent】—— 【ReAct】Synergizing Reasoning and Acting in Language Models](https://blog.csdn.net/wxc971231/article/details/141727455)
- 文章链接：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- 发表：ICLR 2023
- 领域：LLM agent
- 一句话总结：作者将基于 LLM 的推理能力和环境交互能力结合起来，提出一种新的**通用范式 ReAct，使语言模型能够解决各种语言推理和决策任务**。这种新范式在 prompt 和 fine-tune 两种 learning setting 下均系统地优于仅推理（reasoning-only）和仅行动（acting-only）范式
------
- 摘要：尽管大型语言模型（LLMs）在语言理解和交互式决策任务中展示了令人印象深刻的能力，但它们的**推理（例如思维链提示）和行动（例如行动计划生成）能力主要被作为独立主题研究**。在本文中，我们探讨了使用LLMs以交错方式生成推理轨迹和特定任务行动的方法，允许两者之间更大的协同作用：**推理轨迹帮助模型引导、跟踪和更新行动计划以及处理异常情况，而行动使其能够与外部来源（如知识库或环境）接口，收集额外信息**。我们将这种方法命名为ReAct，并将其应用于多样化的语言和决策任务，并展示了其相对于最先进基线的有效性，以及相较于没有推理或行动组件的方法在人类可解释性和可信度方面的改进。具体来说，在问答（HotpotQA）和事实验证（Fever）任务上，ReAct通过与简单的维基百科API互动，克服了思维链推理中普遍存在的幻觉和错误传播问题，并生成了比没有推理轨迹的基线更易于解释的人类般的问题解决轨迹。在两个交互式决策基准测试（ALFWorld和WebShop）上，ReAct仅通过一到两个上下文示例提示，就分别以34%和10%的绝对成功率超越了模仿和强化学习方法


@[toc]
# 1. 方法

- 本文研究开展于 2022 年，当时 GPT3 已经发布了一段时间，研究人员注意到 LLM 在 CoT 技巧加持下展现出良好的自回归推理能力（称之为`仅推理reasoning-only范式`）；同时，一些用预训练模型作为 agent 的初步研究也验证了 LLM 在各种互动环境中进行规划和行动的能力（`仅行动acting-only范式`）。但是，当时具有良好推理能力的标准 LLM 无法和环境交互来获取外部信息；能够和环境交互的 LLM agent 都是直接将交互上下文映射到动作，缺乏对高级目标的抽象推理，也没有保持工作记忆以支持长时间跨度的交互动作，因此
  1. **对于 reasoning-only LLM**：无法和外部环境交互导致其无法利用外部数据库或工具，限制了其 NLP 任务的能力
  2. **对于 acting-only LLM**：缺乏推理步骤的情况下模型只是在利用 “直觉” 行动，它直接将内部状态映射到动作，但并不理解这种映射关系的原因，尤其在 finetune 模型时属于 IL 中的 BC 方法。这导致其只能得到一个非常窄的策略分布，策略泛化能力差
- 为此，作者提出将两种范式结合得到 `ReAct` 范式，**使 LLM 能够以交替的方式生成口头推理轨迹和文本动作**
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReActSynergizingReasoningandActinginLanguageModels/img_001.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

  1. **通过执行动作和环境交互，模型可以得到外部观测反馈**
  2. **通过生成自回归推理轨迹，模型将有用的信息整合到上下文中，进而影响其内部状态以支持未来的推理和行动**
- 作者将新范式应用在 prompt 和 finetuning 两种 learning 设定下
  1. **`ReAct prompting`**：使用 frozen 的大规模预训练模型（[PaLM-540B](https://arxiv.org/pdf/2204.02311.pdf)），通过 few shot in-context learning 控制模型输出任务领域动作或开放式推理轨迹
     > - `domain-specific actions`：如 “search” in question answering task；“go to” in room navigation task
     > - `free-form language reasoning traces`：如 “Now I need to find a cup, and put it on the table”

     **在对于推理至关重要的任务中**，作者交替产生推理轨迹和动作，使得任务解决轨迹由多个推理-动作-观察步骤组成；**对于涉及大量动作的决策任务**，作者通过稀疏推理 prompt 让语言模型自己决定推理轨迹和动作的异步发生时机，推理轨迹只需要稀疏地出现在轨迹中的最相关位置即可
  2. **`ReAct finetuning`**：使用 prompting 设定构造 ReAct 形式数据，然后用来微调较小规模的语言模型 (PaLM-8/62B)，直接将 ReAct 范式注入为模型的内部知识
# 2. 实验
- 作者注意到推理轨迹可以在大量任务中发挥作用，比如
  - 分解任务目标以创建行动计划
  - 注入与任务解决相关的常识知识
  - 从环境交互观测中提取重要部分
  - 在保持计划执行的同时跟踪任务进度
  - 通过调整行动计划来处理异常
- 推理和行动之间的协同带来诸多好处
  - 对于侧重生成动作的 Agent 类任务，**ReAct 允许模型执行动态推理以创建、维护和调整行动的高级计划 (**$\text{reason} \to \text{act}$**)**

  - 对于侧重生成回答的 NLP 类任务，**ReAct 允许模型与外部环境 (如维基百科) 交互，并将获取的附加信息合并到推理中 (**$\text{act} \to \text{reason}$**)**
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReActSynergizingReasoningandActinginLanguageModels/img_002.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

- 作者在 “问答(HotPotQA)”、“事实验证(Fever)”、“基于文本游戏(ALFWorld)” 和 “网页导航(WebShop)” 四个基准上测试 ReAct 方法的性能，发现

  1. NLP 类任务 HotPotQA 和 Fever 上，作者使用 Wikipedia API 作为外部环境，发现将 ReAct 和 CoT 组合可以实现最优性能，**这种设定允许模型在推理过程中同时使用内部知识和外部获得的信息**
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReActSynergizingReasoningandActinginLanguageModels/img_003.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>

  2. 控制类任务 ALFWorld 和 WebShop 上，few-shot prompt ReAct 的性能超过了使用大量样本训练的 RL 和 IL agent。这里尤其可以把 ReAct 和 IL 对比，**IL 中 Agent 只是模仿给定观测下的动作，但并不知道这样动作的原因，这种知识的环境泛化性就很差。相比而言，ReAct 在模仿动作以外还会模仿给定观测下的推理、分析、规划过程，后者的跨环境泛化能力要强很多，这赋予模型传统方法不具备的 few-shot prompt 能力**
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReActSynergizingReasoningandActinginLanguageModels/img_004.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>

  3. 由于 LLM 的一切交互都是以自然语言作为媒介的，其天然具有很好的解释性，因此**人类检查员可以通过编辑 ReAct 的推理轨迹来实现 human-in-the-loop 交互，这种的的人机协作形式可以有效纠正 LLM 的幻觉问题**

# 3. 总结
- ReAct 是一个简单但非常有效的范式，核心思想是 **`通过推理指导行动，再通过行动支持推理`**。ReAct 基于 LM 实现了对思想、动作和环境反馈的联合建模，使其成为一个高性能 multi-task agent。自从 2022 年起，出现了大量 follow 此文章的工作，包括著名的 AutoGPT 等，体现了此范式的有效性。该范式带来的好处包括
  1. **灵活性强**：ReAct 可以直接适用于广泛的推理和交互任务
  2. **泛化性强**：无论 few-shot in context prompt 还是 fine-tune，ReAct 都表现出很强的性能。尤其是在控制决策任务上，**推理过程其实代表了某种跨环境的通用知识，其引入大幅改善了标准 RL/IL 的环境泛化问题**
     > 但也要注意到，这种能力是建立在大规模的 LLM 基础上的，直接和 RL 对比可能不太公平
  3. **和人类对齐**：ReAct 范式下的问题解决过程，相比过去的方法更像人类
- 从两个角度理解 ReAct
  - 从简单的视角看，ReAct 可以看作扩展了环境交互能力的 LM；也可以看作具有内省推理能力的 RL/IL agent
  - 从复杂的视角看，ReAct 可以看作**给 RL/IL agent 扩展了工作记忆，记忆本身存储在上下文中，agent 可以从环境中获取信息，也可以从工作记忆中获取信息。但是这种记忆还是短期的，且无法编辑（append-only）**。这启发了后续工作，即给 LM agent 添加更长、更可编辑的记忆
