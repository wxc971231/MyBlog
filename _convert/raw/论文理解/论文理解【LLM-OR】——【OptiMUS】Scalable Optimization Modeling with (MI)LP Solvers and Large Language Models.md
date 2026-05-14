- 文章链接：[OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/abs/2402.10172)
- 发表：ICML 2024
- 代码：[OptiMUS/tree/optimus-v0.2](https://github.com/teshnizi/OptiMUS/tree/optimus-v0.2)
- 领域：LLM OR
- 一句话总结：使用 LLM 对运筹优化问题（OR Problem）进行建模和求解代码生成时，直接使用 CoT 等朴素方法受限于上下文增长导致性能下降，本文提出的 OptiMUS 通过 **“预处理结构化问题 + 多智能体协作 + 连接图检索相关上下文” 的模块化流程，实现从自然语言到（MI）LP 建模、代码生成与调试评估的端到端自动化**，并在 NL4OPT、ComplexOR 与新数据集 NLP4LP 上取得显著更高的建模准确率
----
- 摘要：优化问题在制造业、物流配送到医疗健康等各个领域都无处不在。然而，由于建模求解所需的专业知识门槛较高，这类问题大多仍依赖人工经验式方法解决，而非通过先进求解器实现最优解。本文介绍的 OptiMUS 是一款 **LLM Agent**，能够**根据自然语言描述自动构建并求解（混合整数）线性规划问题**。该系统具备数学建模能力，可编写调试求解器代码，评估生成解，并根据评估结果持续优化模型与代码。其采用模块化架构处理问题，无需冗长提示即可应对长描述和复杂数据。实验表明，OptiMUS在简单数据集上较现有最先进方法提升超20%，在包含长篇复杂问题的NLP4LP等新数据集上表现更优，提升幅度超过30%

@[toc]
# 1. 背景
- 本文研究**优化问题**的自动建模与编程，以减轻对人类专家的严重依赖。具体而言，这类问题要求**输入一段自然语言描述的问题**（如配送货、生产规划等问题），要求模型或系统完成**运筹学建模**，并**生成问题求解代码**
- 针对该任务，当前主要存在基于提示和基于微调的两类方法：
	1. **基于提示的建模prompt-based modeling**：通过为 GPT-4o 等大规模预训练 LLM 精心设计建模 Prompt 来工作，相关方法包括 [OptiTree](https://blog.csdn.net/wxc971231/article/details/156361583)、[PaMOP](https://blog.csdn.net/wxc971231/article/details/157328214) 等。这类方法的**重点在于通过引入树、图、多智能体等设计，将 “复杂问题描述上下文 -> 严格式要求代码” 的端到端生成过程拆分为多个子过程**，从而降低各环节难度，并使各环节的 prompt 更具指向性
	2. **基于微调的建模fine-tuned LLM modeling agents**：通过构造大规模运筹学及建模知识对 LLM 进行微调，形成专用的建模语言模型，如 [ORLM](https://blog.csdn.net/wxc971231/article/details/141610882)、[Step-Opt](https://blog.csdn.net/wxc971231/article/details/157399452)、OptMATH 等。这类方法的**重点在于设计数据构方法和错误过滤方法**，实现多样、正确、难度可控的高质量数据集
- 本文是 LLM-OR 研究早期的工作，和 ORLM 同期，这个阶段还没有针对 OR 问题建模求解的 LLM-based 工作，只有 CoT 和 [Reflexion](https://blog.csdn.net/wxc971231/article/details/141812277) 等 NLP 任务通用方法
	> 本文主要关注混合整数线性规划问题（MILP），形式化描述如下
		$$
		\begin{array}{cl}
		\underset{\left\{x_{j}\right\}}{\operatorname{minimize}} & \sum_{j=1}^{n} c_{j} x_{j} \\
		\text { subject to } & \sum_{j=1}^{n} a_{i j} x_{j}(\leq,=, \geq) b_{i}, i=1, \ldots m \\
		& l_{j} \leq x_{j} \leq u_{j}, j=1, \ldots, n \\
		& x_{j} \in \mathbb{Z}, j \in \mathcal{I}
		\end{array}
		$$ 这类优化问题的目标函数是线性的，约束条件也是线性的，部分决策变量（$j \in \mathcal{I}$）被要求取整数。其常用于建模 “既有连续决策（比如产量、流量），又有离散/是否选择（比如开不开厂、选不选路线、用几台机器必须是整数）” 的现实优化问题
- 当时，使用 LLM 解决 MILP 问题的主要挑战包括以下四项：
	1. **歧义术语**：一个优化问题可以用多种方式描述，难以定义合适的优化变量
	2. **问题描述长**：真实世界问题描述可能长且复杂，而 LLM 上下文窗口有限
	3. **问题数据量大**：优化问题的规格说明常常涉及大量数据，例如客户属性或商品销量，难以直接作为上下文输入
	4. **输出不可靠**：LLM 给出的解并不总是可靠，生成的代码可能无法执行，能执行的情况下也难以验证结果是否正确
- 本文提出的 OptiMUS 就是针对以上问题设计的 LLM-based agent 
	1. 针对当时自然语言优化建模的数据集过于简单，无法体现 “长问题描述” 和 “大规模数据” 的问题，提出 NLP4LP 数据集，包含 67 个复杂优化问题
	2. 引入一种新的连接图（connection graph）机制，将约束与目标相互独立地处理，缩短复杂问题提示词
	3. 将数据与问题描述分离，缩短大规模数据问题提示词

# 2. 本文方法
- OptiMUS 的工作流程如下图所示
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5516a70c243f425c8544daa463097244.png#pic_center =95%x)
	1. 对问题的自然语言描述进行预处理，提取包含参数、约束、目标函数以及背景的结构化信息
	2. 一组智能体通过迭代方式为该结构化问题添加连接图、各子句的数学公式及对应代码。智能体持续工作直至问题解决
## 2.1 问题预处理
- OptiMUS 的预处理器将问题的自然语言描述转换为一个结构化问题，包含以下组件
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c9a614b073d94ddbbfd6b82ce3cfddd0.png#pic_center =95%x)
	1. **参数Parameters**：优化问题的一组参数。每个参数包含 `符号symbol`、`形状shape`、`文本定义text definition` 三部分
		- 若问题陈述中已经进行参数定义，则使用陈述定义参数；若问题陈述中没有明确给出，OptiMUS 可以选择符号、推断形状并定义这些参数
		- 问题陈述中可能包含的**数值数据会从参数中省略并存储起来以供后续使用**，通过**将数据和参数分离**，缩短后续构建的上下文长度
	2. **子句Clauses**：优化问题的子句列表，包括**目标与约束**
		- 预处理阶段，这些子句被初始化为响应的目标/约束自然语言描述
		- 之后在多智能体讨论阶段，这些子句将被补充加入 LaTeX 形式的公式以及代码
	3. **Background（背景）**：一段简短字符串，用于解释问题的真实世界背景。该字符串会被包含在每一次提示中，以提升常识推理能力
- 以上核心的参数和子句是分别使用三组 prompt 进行提取的，且包含一定的自我迭代纠正动作
	1. **抽取参数**：
		- [parameters.py](https://github.com/teshnizi/OptiMUS/blob/main/parameters.py) 中的 `prompt_params` 用于提取参数；
		- [parameters.py](https://github.com/teshnizi/OptiMUS/blob/main/parameters.py) 中的 `prompt_params_q` 用于评估提取参数的质量，分数太低则触发重新提取
	2. **将问题切分为目标与约束**：
		- [objective.py](https://github.com/teshnizi/OptiMUS/blob/main/objective.py) 中的 `prompt_objective` 用于提取目标描述，输出 python 字符串
		- [constraint.py](https://github.com/teshnizi/OptiMUS/blob/main/constraint.py) 中的 `prompt_constraints` 用于提取约束描述（且需包含非负等隐含约束），输出 python  字符串列表
	3. **消除无效约束**：
		- [constraint.py](https://github.com/teshnizi/OptiMUS/blob/main/constraint.py) 中的 `prompt_constraints_redundant` 用于去冗余/合并；
		- [constraint.py](https://github.com/teshnizi/OptiMUS/blob/main/constraint.py) 中的 `prompt_constraints_q` + `qs` + `prompt_constraint_feedback` 用于筛掉不必要、可能错误的约束。其中前两个对每条候选约束问 “它到底是不是约束、是否应显式建模、给出置信度评分”，分数低时用最后一个进行修正
## 2.2 多智能体系统
- 经过预处理之后，原本完整的问题描述被打散成包含目标、约束的自然语言片段，参数也被提取为标准格式。OptiMUS 在此基础上，使用一个多智能体系统进行问题变量定义，并对子句进行建模与编码。**初始时变量列表、LaTeX 公式与代码在为空；当所有子句（目标、约束）都完成建模、编程并通过验证后，流程结束**
- 为确保建模结果的一致性，OptiMUS **构建并维护一个`连接图connection graph`，用于记录每个约束中出现了哪些变量与参数**。该连接图是 OptiMUS 性能与可扩展性的关键，因为它使 LLM 在每次提示中**只需关注相关上下文**，从而生成更稳定的结果
- 多智能体系统的构成如下
	1. **Manager：用于协调迭代的建模、编程与评估工作**。在每一步，manager 查看当前为止的对话内容，选择下一个 agent，为其生成并分配任务（例如 “审阅并修复目标函数的数学建模"）
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e917ed9f89d94a64bedcbe75af17f12b.png#pic_center =50%x)
	3. **Formulator：负责为变量/子句编写/纠正数学建模公式**，这涉及新变量和辅助约束的定义、以及更新连接图中的链接
		- **新子句建模**：Formulator 遍历尚未被建模的子句并为其生成新建模公式，此过程中，它会在必要时定义辅助约束与新变量、判断哪些参数与变量与该子句相关，并利用这些信息更新连接图
			> 下图显示了为某个新约束条件进行建模的过程，Formulator 会识别所有约束相关的参数和变量，根据需要定义新变量，更新连接图，并用LaTeX公式对约束条件进行标注（虚线表示新增连接与变量）
			> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78352fb92821414dbcbc5ddc9c47cfd6.png)
		- **修复子句建模**：Formulator 遍历被标记为不正确的子句，修复其建模公式，并更新连接图
		- Formulator 还具有一个额外的建模层，用于捕获特殊的模型结构（例如 special-ordered-set 与 indicator variables）
			> 现代优化求解器在求解 MILP 时会利用问题的特定结构来提升性能，并且通常为这些特殊结构提供定制化接口。为了利用这些接口，Formulator 会在一系列速查表提示词 cheatsheet prompt 之间迭代，考察能否使用某些高级建模技巧进行建模。每个提示词会向 LLM 提供该结构的描述，通过一个示例说明应如何利用该结构，然后要求 LLM 判断该结构是否可以应用到现有的建模公式中。一旦识别出合适且可用的结构，就调整建模公式以使用定制化的求解器接口。如下图所示
			> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83b74e7a7c7849268e5d2c8d0855c17b.png#pic_center =99%x)
	4. **Programmer：负责编写并调试求解器代码**，作者的实验中使用 Python 作为编程语言，使用 Gurobi 作为求解器。需要注意的是，Programmer 的编程粒度控制在子句级别，每次只写一段优化目标或约束条件，不会一次写完全部代码
		- **新子句编程**：Programmer 遍历未被编码的子句，并根据其建模公式生成代码
		- **修复错误建模**：Programmer 遍历被标记为 bogus 的子句并修复其代码
	5. **Evaluator：负责在数据上执行生成的代码，并识别执行过程中出现的任何错误**。如果 evaluator 遇到运行时错误，它会将具有 bogus 代码的变量或子句标记出来，并向 manager 返回对错误的适当解释。这些信息随后会被其他 agent 用来修复公式并调试代码

## 2.3 连接图
- OptiMUS 在约束、目标、参数与变量之上维护一个连接图，并使用该图为每一次 prompt 检索相关上下文，从而控制上下文长度。该图也用于生成/调试代码，和错误公式纠正。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11fd7413dc924865b9ea67a52df60f30.png)
	如图所示，Programmer 修正代码错误时，首先根据 Evaluator 标记找到相关的约束条件子句，然后就能利用连接图检索到相关参数、变量和代码片段，从而控制 prompt 中的上下文构成，避免输入完整程序

# 3. 实验
## 3.1 实验设定
- **数据集**：
	1. **NL4OPT**：1101 个简单 LP，自然语言描述 + 带标注的中间表示（参数/变量/子句）
	2. **ComplexOR**：原始 37 个复杂 OR 问题；公开版本不完整，作者收集 21 个并对缺数据问题用合成数据增强（数据在补充材料）
	3. **NLP4LP**：67 个实例（54 LP + 13 MILP），来自教材/讲义，覆盖选址、网络流、调度、投资组合、能源等；提供描述、示例数据文件与最优值
- **对比基线**：Standard prompting、Reflexion、Chain-of-Experts（CoE，文中称为该任务 SOTA）
- **评价指标**：文献常用 accuracy、CE、RE，但作者认为“短且无关/删代码也能跑”会误导，因此**只比较 accuracy**

## 3.2 实验结果
### 3.2.1 总体性能
- OptiMUS 在所有数据集上都以很大的优势超过了其他所有方法，体现了模块化和结构化的重要性
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/626c2128b8ca49cc812cf13d9e682d07.png#pic_center =60%x)
### 3.2.2 消融实验
- **使用小模型作为基础时，性能显著下降**
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df07ed421949415d982d8e515908dc5e.png#pic_center =60%x)
	作者任务小模型表现不佳的主要原因是
	1. 小模型的上下文窗口较小
	2. OptiMUS 的提示词具有**新颖且模块化的结构**，这些提示很可能在基础模型训练时罕见，**小模型泛化能力较差，导致无法良好地响应提示**
	3. 作者评估了一个仅 manager 使用 GPT-3.5，而其他 agent 使用 GPT-4 的版本，其在 NL4OPT 上性能差异很小，原因是 NL4OPT 的大多数实例都可以通过一个简单的“公式化—编程—评估”链条解决。然而，**在 ComplexOR 与 NLP4LP 中需要更复杂的 agent 交互，此时 manager 的重要性更加明显**
	4. 禁用 programmer agent 调试功能的实验展示了与 manager 类似的结论，我们看到**调试在更复杂的数据集上更重要**
- **随着 Agent 调用次数增加，困难问题的解决概率也增加**，如下图所示
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a354e1203d164bdf9a498865acd06dc4.png#pic_center =50%x)
- 所有 Agent 中，programmer 与 evaluator 被选择的次数比 formulator 更多
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3f637c51901c4722b4e50e77b1be2dea.png#pic_center =50%x)
	1. 编码错误更常见。LLM 往往生成带有易于修复的琐碎 bug 的代码，需要更多地调用 Programmer 进行修复
	2. 相比修复编码错误，识别公式化中的 bug 需要更深层推理，也更难。因此，OptiMUS 的 manager 被提示优先修复代码，再考虑公式化中的错误。**只有当 programmer 声称代码是正确的时，formulator 才会被选择用于调试**
- 随着问题难度增加，CoE 需要越来越长的 Prompt，**OptiMUS 基本保持 prompt 长度不变**。模块化方法使 OptiMUS 能够在每次 LLM 调用时仅提取并处理相关上下文，允许其扩展到更大、更长的问题
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5d5aa68cf6c84984b984dfe8bb8fc9c9.png#pic_center =50%x)
### 3.2.3 失败情况
- 作者把失败原因总结为三类
	1. **约束缺失或错误**：在预处理步骤中生成了错误约束（例如 price ≥ 0，但 price 是一个参数），或者未能从描述中抽取出所有约束
	2. **建模错误**：用不正确的数学模型来处理问题（例如在 TSP 中把“访问城市”定义为二元变量，而不是把“链接”定义为二元变量）
	3. **编码错误**：即使经过调试，仍无法生成 bug-free 代码。编码错误常在 LLM 被语言表述弄混时出现（例如在 ComplexOR 的 “prod” 问题中，描述显式提到了 “parameters” 与 “variables”）
- 下表整理了 OptiMUS 的失败比例，**失败的主要原因是建模错误，作者认为这一点可以通过微调进行弥补**
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/532ec4f4c8274b71831e8bdcc7a51a47.png#pic_center =50%x)

# 4. 总结
- OptiMUS 是一个模块化的、基于 LLM 的智能体，用于从自然语言描述中对优化问题进行建模与求解，展示了通过将 LLM 与传统求解器相结合，可以自动化优化流程中各个阶段的潜力。本文的核心思想在于精确构造提示词，控制上下文长度的同时尽量发挥 base LLM 的能力，具体通过两个手段实现
	1. 将复杂的端到端任务**划分为多个阶段**（多 Agent 交互），每个阶段（Agent）使用针对性的提示词
	2. **使用连接图提取原问题中的子问题作为编辑单位**，控制上下文长度，增加提示精度并降低复杂度。这个思路在后续的 [PaMOP](https://blog.csdn.net/wxc971231/article/details/157328214) 工作中得到进一步发展，通过不断拆分子问题精细化提示词
- 本文的另一个贡献是 NLP4LP 数据集，这是一个包含长文本的有一定难度的数据集，在后续 OptiTree / SIRL / Step-Opt 等多篇文章中作为测试数据集使用
- **未来方向**
	1. 更小的 LLM 更快更便宜，但能力更弱，可以考虑在不同阶段使用不同的 LLM 降低成本
	2. 将用户反馈整合到流程中可以提升智能体在自然语言优化建模上的表现；研究这类智能体与用户之间的交互是一个令人兴奋的方向
	3. 用强化学习增强本文提出的模块化 LLM 结构，从而教会 manager 如何选择下一个 agent