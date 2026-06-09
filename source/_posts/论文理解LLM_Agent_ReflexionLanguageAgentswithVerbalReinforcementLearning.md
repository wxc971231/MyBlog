---
title: 论文理解【LLM-Agent】—— 【Reflexion】Language Agents with Verbal Reinforcement Learning
date: 2026-06-09 14:13:22
index_img: /img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/index.png
tags:
  - LLM-Agent
  - LLM
categories:
  - 机器学习
  - 论文理解
  - LLM-Agent
description: 本文介绍 LLM agent 领域的经典方法 Reflexion，它通过引入 episodic memory 为 agent 提供长期记忆，在不微调模型的参数的同时实现了基于上下文的试错学习，性能提升显著
---

- 首发链接：[论文理解【LLM-Agent】—— 【Reflexion】Language Agents with Verbal Reinforcement Learning](https://blog.csdn.net/wxc971231/article/details/141812277)

- 文章链接：[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- 代码：[GitHub - noahshinn/reflexion](https://github.com/noahshinn/reflexion)
- 发表：NIPS 2023
- 领域：LLM agent
- 一句话总结：传统强化学习 Agent 通过和环境交互进行试错学习，但对于 LLM agent 来说，微调其权重的成本很高。**本文提出的 Reflexion 框架将基于权重调整的试错学习转化为针对 LLM agent 的基于自然语言反馈的试错学习**。和强化学习中仅依靠 reward 数值反馈指导优化不同，Reflexion agent **通过维护一个 episodic memory 作为决策的依据**，得益于自然语言媒介的灵活性，agent 可以利用各种数据类型和不同来源（外部观测或内部反思）的反馈信号，在多种任务（顺序决策、编码、语言推理）中获得了显著改进。
    > 本文是 [ReAct](https://blog.csdn.net/wxc971231/article/details/141727455) 论文的一个扩展工作，两文作者有重合。ReAct 方法通过令 LLM 交替生成推理轨迹和任务动作，同时在 NLP 推理任务和控制任务上达成了良好的效果。从智能决策角度看，ReAct 可以看作**给 RL/IL agent 扩展了工作记忆，记忆本身存储在上下文中，agent 既可以从环境中获取信息，也可以从工作记忆中获取信息。但是这种记忆还是短期的，且无法编辑（append-only）**。相比而言，本文通过维护记忆缓冲实现了可编辑的长期记忆，进一步提升了性能
- --------
- 摘要：LLM 已经越来越多地用作 agent 和外部环境（如游戏、编译器、api）进行交互。然而，**由于传统的强化学习方法需要大量的训练样本和昂贵的模型微调，让语言模型快速有效地进行试错学习仍然具有挑战性**。我们提出了 Reflexion 框架，通过语言反馈而非权重更新，来强化 LM agent。**具体地说，Reflexion agent 对任务反馈信号进行自然语言反思，然后在情景记忆缓冲区（episodic memory）中保持它们自己的反思文本，以便在随后的 rollout 中做出更好的决策**。这种框架足够灵活，可以合并各种数据类型（标量值或任意形式的语言）和不同来源（外部观测或内部反思）的反馈信号。在多种任务（顺序决策、编码、语言推理）中，我们的方法相比 strong baseline 获得了显著改进。例如在 HumanEval benchmark 上达到了 91% 的 pass@1 精度，超过了之前最先进的 GPT-4（达成80%）。我们还使用不同的反馈信号、反馈合并方法和 agent 类型进行消融和分析研究，并提供关于它们如何影响性能的见解。

# 1. 方法
## 1.1 基本思想
- 本文之前，ReAct、Toolformer、HuggingGPT 等工作已经实现了结合环境交互和内部思考推理的 LLM agent，但是这些方法都还无法有效地通过试错进行学习。这本质是因为**这些方法不包含 “学习” 过程，模型的 “工作记忆” 仅存在于其上下文内，而每次 rollout 都会清空上下文，因此`以轨迹为单位的`交互过程没有对 agent 产生任何影响**，某种程度上看就是带有环境交互的一种特殊 LLM 评估任务而已，没有试错学习的能力。但是，直接把 RL 的试错学习框架套用到 LLM agent 上是不合适的，原因有三
  1. LLM 参数规模大，在线训练成本太高，且 batch size 太小不够稳定
  2. 对于复杂任务，MDP 交互框架中的标量 reward 反馈信息量太少，导致信用分配困难
  3. 没有充分利用 LLM 的规划、分析能力
- 为此，作者提出**将环境的标量反馈转化为文本摘要形式的自然语言反馈，然后在下一回合中将其作为额外的上下文输入 LLM agent 中。这种自我反思式的反馈充当了一种 “语义” 梯度信号，通过为 agent 提供一个具体的改进方向帮助它从先前的错误中学习**，从而在任务中表现得更好
  - 这种 Reflexion 学习过程中虽然没有调整 LLM 的参数权重，但通过维护情景记忆（episodic memory）保留了交互过程的影响，并将其通过类似 In-context learning 的方式作用到 LLM 上
    > 注：对于 Decoder-Only Transformer 模型来说，prompt 信息可以直接地作用于 CD 模型的每一层的参数，可以视为一种隐式微调。详见 [Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers](https://arxiv.org/abs/2212.10559)
  - 相比经典 RL，这种通过 Reflexion 交互多次迭代地学习完成复杂任务的过程更加类似人类。因为在**每一次 rollout 结束后都存在一个反思之前的失败原因并构造下次尝试改进计划的阶段**，目前只有 LLM 有能力进行这种反思
- 下图显示了 Reflexion 框架在决策、编程和推理任务上的示例    
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_001.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    1. 从任务描述开始
    2. 使用类似 ReAct 的方式和环境交互，其中交替执行推理分析和动作生成，得到完整轨迹
    3. 对轨迹进行评估，这一步有多种设计，包括
        - 简单的 binary 环境反馈，描述任务成功或失败
        - 针对常见失败案例的预定义启发式方法，有点类似 RL 中的奖励函数设计
        - 自我评估方法（如使用 LLM 判断决策问题的成败结果，或在编程任务中编写单元测试
    4. 在 reflecxion 过程中，用自然语言将评估信号表述为经验摘要，并存储在 episodic mem 中
    5. 在新一轮 rollout 中，agent 同时利用交互轨迹上下文和 episodic mem 的经验进行决策
- 虽然 Reflexion 依赖于 LLM 的自我评估和解释能力（启发式方法），缺乏成功控制的理论保证，但相比传统 RL 方法，Reflexion 有以下优点
    1. 轻量，不需要对 LLM 进行微调
    2. 允许基于自然语言的，更精确和细致的反馈形式，而非简单的标量或向量奖励
    3. 允许对先前经验的情节记忆进行更明确和可解释的表达
    4. 通过调整 prompt，为未来 rollout 中的行动提供了更直接明确的提示
## 1.2 Reflexion：通过口语式反思进行强化学习
- 本节详细说明 Reflexion 框架的内部结构，如下图所示
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_002.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

- 注意到整体是一个 Multi-agent 系统，有三个 LM 分工合作完成任务，主要部件包括
    1. **Actor**$M_a$**：生成动作和文本（推理、分析、反思）的模型**。类似传统强化学习设置，它在 $t$ 时刻接收到环境观测 $o_t$，然后通过特定的提示生成推理分析文本，或基于当前策略 $\pi$ 采样/生成一个动作 $a_t$。生成动作时，使用 episodic memory 提供额外的上下文
        > - 作者称 Memory 的引入受到了《[Large Language Models can Implement Policy Iteration](https://arxiv.org/abs/2210.03821)》 的启发，其建议使用上下文学习的策略迭代方法
        > - 在 RL 领域，曾经 episodic 也小小地流行过一段时间，基本思想是记住并重现过去表现好的动作，参考 [论文理解【RL - Episodic Control】 ——【MFEC】Model Free Episodic Control](https://blog.csdn.net/wxc971231/article/details/121565666)
    2. **Evaluator**$M_e$**：对 $M_a$ 的输出进行评分的模型**。它接受生成的轨迹作为输入，并计算一个反映其在给定任务环境中表现的奖励分数。定义适用于语义空间的有效的价值和奖励函数是困难的
        - 对于推理任务（reasoning），作者使用了基于精确匹配（exact match, EM）评分的奖励函数，确保生成的输出与预期解决方案紧密对齐
        - 对于决策任务，作者采用针对特定评估标手工设计的启发式函数
        - 此外，作者还尝试使用 LLM 本身的不同实例对象为决策和编程任务生成奖励分数
    3. **Self-Reflexion**$M_{sr}$**：生成口语式强化提示，以帮助 Actor 进行策略提升的 Self-Reflection 模型**。它基于一个稀疏的奖励信号（比如 binary 成败信号）、当前的轨迹和 mem 信息生成自然语言形式的，细致且具体的反馈。这种比标量奖励（scalar rewards）更具信息量的反馈随后存储在 mem 中
        > 例如，决策任务中某次 rollout 以失败结束，$M_{sr}$ 可以根据相关信息推断出特定动作 $a_i$ 导致了后续的错误动作 $a_{i+1}, a_{i+2}$，随后它以自然语言表达应该采取不同的动作 $a_i'$，这将导致 $a'_{i+1}, a'_{i+2}$，并将这一经验存储在其记忆中。下次 rollout 时，agent 可以利用该经验在 $t$ 时刻选择 $a_i'$ 来避免失败，从而实现试错学习
    4. **Memory：情景记忆（episodic memory）赋予 agent 长期记忆。这允许 Actor 在推理时同时基于短期记忆（历史轨迹）和长期记忆来合成 LM 上下文并指导决策**，类似于人类记忆最近的细节的同时也回顾起长期记忆中提炼出的重要经验。相对于其他 LLM agent 相关工作，通过结合长短期记忆，合成同时受到多次试验中学到的经验教训影响的上下文，是 Reflexion agent 的一个关键优势
    5. **Reflexion Process**：如图2中的迭代优化过程所示。在第一次 rollout 时，Actor 提供完整的交互轨迹$\tau_0$，Evaluator 生成打分 $r_0=M_e(\tau_0)$，**这里 $r_t$ 只是一个稀疏的标量奖励**。接下来 Self-Reflection 模型 $M_{sr}$ 分析 $\{\tau_0, r_0\}$，生成自然语言摘要 $sr_0$，并将其存入 memory。Actor、Evaluator 和 Self-Reflection 模型通过一系列 rollout 试错学习，直到 Evaluator 认为 $\tau_t$ 是成功的为止
        > 作者对 memory 设置了最大存储经验数（通常设置为1-3）以符合 LLM 的上下文长度限制
- Reflexion 和 Beam Search 机制有些类似，但是后者缺少 memory 组件，对比如下
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_003.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>


# 2. 实验
- 在文字游戏任务 ALFWorld 上，Reflexion 机制使性能提升了 22%
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_004.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

- 推理问答任务 HotPotQA 上，作者实现了一个 Reflexion + CoT 的 agent，它可以使用维基百科 API 检索上下文，并使用逐步显式思考推断答案。加入 Reflexion 使性能提升了 20%    
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_005.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

- 编程任务上，由于涉及到编译器反馈，对比方法也或多或少地和使用了和 Reflexion 框架类似的组件或机制，比如生成单元测试，分析报错信息等，但是都没有 Reflexion 做得这么全面，且都没有核心的 Reflexion 机制
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_006.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

    最终性能也是 Reflexion 显著更好
    <div align="center">
        <img src="/MyBlog/img/论文理解LLM_Agent_ReflexionLanguageAgentswithVerbalReinforcementLearning/img_007.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

# 3. 总结
- 本文提出的 Reflexion 是一种基于自然语言反馈的试错学习框架，它引入了 episodic memory 形式的长期记忆缓存，通过构造包含短期记忆和长期记忆的上下文，在不调整模型参数的情况下，允许 LLM agent 从历史错误经验中学习和强化，实现了显著的性能提升。另外，这种 LLM agent 相比 RL agent 更具可解释性
- 本文的局限性包括
    1. Memory 设计比较简单，只是一个最大容量的滑动窗口。可以考虑进一步扩展为向量嵌入数据库或传统的SQL数据库
    2. Memory 只能添加和移除，可以考虑引入编辑
    3. 可以考虑将 Memory 和 LoRA 等微调手段结合使用，同时从上下文和参数两方面进行策略提升
    4. 问题设置比较简单，所有问题都只有 binary 成败反馈，可以考虑扩展到具有连续值反馈的任务上






