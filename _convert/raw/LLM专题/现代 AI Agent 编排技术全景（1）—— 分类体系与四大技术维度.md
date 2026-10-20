> 本文主要讨论**以 LLM 为核心、面向复杂任务执行的 Agent 编排**，覆盖单 Agent、多 Agent、业务 Workflow 和长期运行 Agent。文中 **D1–D4 / S1–S3 是基于官方公开资料提出的分析框架**，不是某家厂商的正式标准

@[toc]
# 0. 引言：模型越来越强，为什么仍然需要编排？
- 给模型配置工具，让它根据任务选择调用、观察返回结果，再决定下一步，就得到了最基本的 **Agent Loop**。对于路径未知的检索、编程与环境操作，这种循环很有用。然而，任务一旦变成批量审查几百个文件、跨多套软件串联执行，或持续数周等待事件和人工审批，单靠一个模型逐轮记住并管理全部工作，就可能遇到上下文拥塞、遗漏步骤、协调冲突、不可恢复以及验收不清的问题
- 当前，业界正在探索**把计划、执行、协作、状态与验收拆开，交给性质不同的机制负责**，比如
	- 把流程边界交还给代码
	- 让模型生成执行程序
	- 建立专门的任务调度和通信系统
	- 训练模型更好地决定何时分工并行
- Graph、Dynamic Workflow、Agent Team、Swarm 等概念看似代表不同的 Agent 架构流派，实际上往往关注不同层面的技术问题。本文旨在建立一套**多维度、可组合的 Agent 编排技术分类框架**，以此系统性地分析和比较主流 Agent 产品技术架构，并为新型 Agent 编排框架的设计与研发提供指导
# 1. Agent 基础概念与经典执行范式
## 1.1 Agent 编排的核心概念
- 在深入讨论 Agent 编排的技术分类与实现方式之前，首先需要厘清几个密切相关但处于不同抽象层次的基础概念，明确模型能力、执行机制、运行环境与编排逻辑之间的关系。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c44a4c8440ca4637b99b8caac6105179.png#pic_center =85%x)
	- **Augmented LLM**：具备工具、检索、记忆等增强能力的模型调用，是 Agent 系统的能力构件
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6815b7b13dd54ee28a6702553525e4e0.png)
	- **Agent Loop**：模型基于目标与观察选择行动，调用工具或与环境交互，然后依据新结果再次决策，直到满足停止条件或被外部终止，**强调模型自主决策**
		> Agent Loop 既可以独立构成单 Agent 系统，也可以作为多 Agent 系统中各个自主 Agent 的内部执行机制。多个 Agent Loop 之间的任务分配、通信与协作则由更高层的编排机制组织
	- **Workflow**：狭义的 Workflow 指通过**预定义的**多步控制逻辑来协调 LLM 和工具的系统，和 Agent Loop 并列。执行路径可以包含顺序、分支、并行、循环等，**强调多步骤工作的组织和控制**
		> 现代工程框架中，Agent Loop 和 Workflow 之间的界限逐渐模糊，二者**可以相互嵌套，控制逻辑也可以在运行时动态构建**
	- **Harness**：支撑 Agent 执行的技术环境，通常包含模型调用、工具执行、上下文管理、沙箱与必要的任务控制，它可能包含若干编排功能，但不等同于多 Agent 组织结构；
	- **Orchestration（编排）**：指决定工作怎样被计划、推进、委派、协作及验收，它和 Harness 高度相关
		> **Agent Orchestration 不等于 Multi-Agent，也不等于完全由 LLM 自主决策**。单个 Agent 也需要行动选择、循环控制和结果验证；而在复杂 Agent 系统中，编排既可以由模型动态决策，也可以通过确定性 Workflow 实现，或采用两者结合的混合方式
## 1.2 经典 Agent 与 Workflow 执行范式
- 在上述概念的基础上，在经典 Agent 架构研究中，[Anthropic《Building Effective Agents》](https://www.anthropic.com/engineering/building-effective-agents) 进一步归纳了两类经典执行范式：模型动态决策的 Agent Loop 与 预定义控制路径的 Workflow，并总结了五种典型 Workflow 模式，为理解现代 Agent 的任务执行策略提供了基础框架（注：本节图中 LLM call 均为 Augmented LLM call）
	1. **`Agent Loop`：指 LLM 动态地指导自身流程和工具使用，从而保持对完成任务方式的控制的系统**。现代 LLM 具备理解复杂输入、进行推理和规划、可靠地使用工具以及从错误中恢复的能力，为 LLM 接入 Harness 系统后，其就具备了与生产环境交互，观测必要信息，规划下一步动作，并通过多步交互完成目标的能力。这样系统形式就从 LLM 升级为 Agent，并通过 Agent Loop 完成工作。**适用于难以预测或无法预测所需步骤数，且无法预先设定固定路径的开放式问题**
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5345ef889ec84f14b2c157346d402696.png)
	智能体最典型的应用是编码任务，其工作通常始于人类用户的指令或与其进行的交互式讨论。一旦任务明确，智能体便会独立进行规划和运行，并可能返回给人类用户以获取更多信息或判断。执行过程中，智能体需要在每个步骤中从环境中获取 “真实观测数据”（例如工具调用结果或代码执行情况）以评估其进度。智能体可以在检查点或遇到障碍时暂停，等待人类用户的反馈。任务通常会在完成后终止，但为了保持控制，通常也会设置停止条件（例如最大迭代次数）
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9808dbb71eda41f8ae0c14a6199b1009.png)
	值得注意的是，虽然智能体可以处理复杂的任务，但它们的实现通常很简单，只是基于环境反馈循环使用工具的 LLM，因此**清晰周全地设计工具集及其文档至关重要**；另外，由于 LLM 可能需要运行很多回合，**必须对其决策有一定的信任度**
  2. **`Workflow`：通过预定义的代码路径来协调 LLM 和工具的系统**，五种典型工作流如下：
	  - **Prompt Chaining（提示链）：将多个任务步骤顺序串联执行**。适合任务可以轻松且清晰地分解为固定子任务的情况。其主要目标是通过简化每次 LLM 调用，以牺牲延迟为代价来换取更高的准确率
	  	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/398e23243db14d4c81b4cad1775a608d.png)
	  - **Routing（路由）：根据输入特征选择不同执行路径，通过分离关注点构建更具针对性的提示**。适用于有不同处理方式的复杂任务，并且可以通过 LLM 或更传统的分类模型/算法准确地进行分类，不会在针对一种输入进行优化时影响对其他输入的性能
	  	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e32db6abce404fe4ba5a7ff2ca506131.png)  	
	  - **Parallelization（并行）：并行处理相对独立的子任务，或对同一任务并行运行多次后投票**。当拆分后的子任务可以并行处理以提高速度，或者需要从多个角度或多次尝试以获得更高置信度的结果时，并行化非常有效。对于涉及多个方面的复杂任务，如果每个方面都由单独的 LLM 调用来处理通常表现更佳，这样可以集中精力关注每个特定方面
	  	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/62906c319aa7407e915f4e199d706d1a.png)
	  - **Orchestrator-Workers（协调者-工作者）：协调者LLM动态拆分任务，分派给工作者LLM执行并汇总结果**。适用于难以预测所需子任务的复杂任务（例如，在编码过程中，需要修改的文件数量以及每个文件的修改性质可能取决于具体任务）。虽然其拓扑结构与并行化类似，但关键区别在于其灵活性——**子任务并非预先定义，而是由协调器根据具体输入确定**
	  	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f0a9c1aa6da420c82a0e2a45dc0399d.png)
	  - **Evaluator-Optimizer（评估器-优化器）：一个LLM调用生成响应，另一个LLM调用提供评估和反馈，迭代改进执行结果**，适用于评估标准明确且迭代改进能够带来可衡量价值的情况
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/17741cae562044678026bea0e365b86f.png)
- 以上分类法源自 [Anthropic《Building Effective Agents》](https://www.anthropic.com/engineering/building-effective-agents) ，其成文于 2024 年底，已经有一点过时。**现代 Agent Loop 与 Workflow 模式并非互斥**，一个 Workflow 可以将 Agent Loop 作为执行节点，Agent 也可以在自主执行过程中调用完整 Workflow。上述五种模式主要刻画典型任务处理结构，也未穷尽现代 Agent 系统的执行策略。更完整的编排技术分类及其组合关系将在第 3 节展开

# 2. 统一技术分类
- 不同 Agent 系统通常由多种技术机制组合构成，为统一描述和比较不同系统，本文提出一种**多维技术分类框架**，从决策与控制、编排表达与执行、多 Agent 组织与协作、任务执行模式四个纵向设计维度，以及上下文与状态管理、协同调度与 Runtime、验证与评估三类横向支撑能力，对 Agent 编排系统进行分析，如图所示：
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/46d8704c82654eb2836099d5ea05dee9.png)
	需要注意，编排表达与执行（D2）、多Agent组织协作（D3）、任务执行模式（D4/S2）都可以用图结构表示，但其节点与边具有不同语义：因此一个 Agent 编排系统至少可以画出三张交互关系图，分别表示任务执行路径、Agent 通信关系和任务依赖关系。如下：
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b4dcddafd48466bb52cdc3854e8847b.png)
- **本文重点关注四个技术维度**，其中每个维度可进一步拆解为若干技术选型问题，并通过相应技术选项描述系统的具体实现，如下表所示。这**些选项并非互斥关系，具体系统可以单选、多选或在不适用时不选**，由此形成能够刻画不同 Agent 产品架构特征的技术组合。后文将按照各维度依次展开分析。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c307af821527489291b67bfb9763a37a.png)
# 3. Agent 编排的四个技术维度
- Agent 编排涉及从高层决策到具体任务执行的多个技术层面。为了避免将控制策略、执行机制、多 Agent 协作结构和任务处理模式混为一谈，本文从决策与控制（D1）、编排表达与执行（D2）、多 Agent 组织与协作（D3）、任务执行模式（D4）四个相互关联的维度构建技术分类体系。这四个维度分别回答**谁来决策**、**如何表达与执行**、**如何协作**，以及**如何组织和推进任务**，共同刻画现代 Agent 编排架构的主要设计空间
## 3.1 决策与控制
- Agent 编排首先要解决的是**执行控制权如何分配**。[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/multi_agent/) 提出了 Code-driven 和 LLM-driven 两种主要编排方式，本文分别将其称为 **Code-driven** 和 **Model-driven**：前者由代码控制执行流程，后者由模型根据任务目标与上下文自主决定后续行动，两者也可以混合使用。在此基础上，我们进一步从**控制权归属、计划的形成与演化、编排策略的构建与优化**三个相互关联的角度分析编排决策机制，形成如下技术分类表：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/683b010bc8eb42af9381a0af5258453d.png)
- 需要注意，**这些技术选项并非互斥的架构类型，而是具有不同分类关系的设计机制**：控制权可以在代码与模型之间分配，并叠加人工干预；计划的形成方式与演化机制相互正交，可以在不同执行阶段组合；编排策略则可以通过人工编写、模型合成、Prompt 引导或专门训练等方式构建与优化。**三个技术选型问题共同刻画 Agent 系统的编排决策架构**，下面分别展开分析。
### 3.1.1 谁拥有执行控制权？
- 控制权分配的核心是**下一步行动由代码规则还是模型推理决定，而非执行路径是否预先固定**。存在两个基本方向，两者可以根据不同任务或执行节点**组合使用**
	- **Code-driven**：由显式程序规则决定执行流程，即使存在动态分支和循环，决策仍由代码确定
	- **Model-driven**：由模型根据任务目标与上下文推理决定下一步，即使只能选择预定义的候选路径，决策权仍属于模型
- 同时，**Human-in-the-loop** 是可以叠加在任意控制机制上的人工干预方式，由人类在关键节点审批、修正或接管；**Hybrid** 则表示代码、模型及必要的人工控制机制组合使用。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d976fb37af594e56bc0ac688b2925577.png)
	> 以 [Google ADK 2.0](https://developers.googleblog.com/en/why-we-built-adk-20/) 为例，其采用 **Code-driven + Model-driven（Hybrid）** 的混合控制方式：通过确定性执行图控制整体流程，同时将需要语义理解和专业判断的步骤交给 LLM Agent，实现代码规则与模型推理之间的控制权分配。
### 3.1.2 计划如何形成与演化？
- 与控制权归属不同，计划形成方式关注的是**哪些编排决策预先确定，哪些留待运行时形成，以及计划如何进一步细化和调整**。可以从两个相互关联的角度分析：
  - **计划如何形成**：根据编排决策的形成时机分为 **Design-time** 和 **Runtime** 两种主要方式：
    - **Design-time**：在设计阶段预先定义计划、流程或控制规则。
    - **Runtime**：根据具体任务和执行上下文，在运行过程中形成编排决策。按照决策粒度，又可以分为三种典型方式：
      - One-shot Runtime：运行时一次性生成整体计划，再按计划执行。
      - Phased Runtime：运行过程中按任务阶段逐步生成后续局部计划，不要求一次性确定完整计划。
      - Stepwise：不必预先生成完整计划，而是根据每轮执行观察逐步决定下一步行动。
  - **计划如何演化**：描述已有计划如何随着任务推进进一步细化或调整，主要包括两种正交机制：
    - **Replanning**：根据执行反馈或环境变化修订已有计划
    - **Recursive Planning**：将高层计划逐步细化为更具体的子计划，直到形成可执行任务步骤

- 因此，**Design-time 与 Runtime 描述计划的形成时机，其中 Runtime 又可细分为 One-shot / Phased / Stepwise 等不同粒度，而 Replanning 与 Recursive Planning 描述计划的修订与细化方式**。这些机制并非互斥，同一系统可以预定义外层流程、运行时生成局部计划，并根据执行反馈进一步调整或细化
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/799340aea1b9466796556b5fd8fd2030.png)
  > 以 Claude Code Dynamic Workflows 为例，其可以在不同执行层级组合 **One-shot Runtime + Stepwise**：外层模型一次性生成 JavaScript 编排脚本，由脚本控制整体执行流程；内部 Subagent 则根据执行观察逐轮自主决策，实现整体规划与局部动态决策的结合。
### 3.1.3 编排策略如何获得？
- 进一步的问题是：**编排决策所依赖的控制逻辑如何构建，模型的编排决策能力又如何引导和优化？** 可以从两个相互关联的方面分析：
	- **显式控制逻辑的构建**：通过规则、流程或程序表达控制逻辑，既可以由开发者手工编写（Handcrafted），也可以由模型合成程序或其他可执行控制结构（Program Synthesis）。
	- **模型决策行为的引导与优化**：可以通过 Prompt、Skill 等方式在推理时引导模型决策（Prompt-based），也可以通过专门训练优化其编排能力（Learned Policy）。
- **这些选项不是互斥分类，而是可以组合的策略构建方式**，例如 Prompt 可以指导模型生成程序，经过训练的模型也仍然可以通过 Prompt 接收任务和约束，而开发者也可以在手工编写的控制框架中嵌入模型合成的局部流程
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd0bdf62742d41e48c202ffd90306276.png)
	> 这两条路线在实际系统中都有代表性案例。[Claude Code Dynamic Workflows](https://code.claude.com/docs/en/workflows) 采用 **Prompt-based + Program Synthesis**，通过 Prompt 引导模型生成 JavaScript 编排程序，再由 Runtime 执行；[Kimi K2.5](https://arxiv.org/html/2602.02276v1) 则以 **Learned Policy** 为核心，通过 PARL 强化学习训练 Orchestrator 的任务拆分与并行调度能力，体现了通过训练获得编排策略的技术路线。
## 3.2 编排表达与执行
- 在 3.1 节明确编排决策的控制机制之后，编排表达与执行需要回答：**这些决策如何被表达出来，以及如何转化为实际执行过程？** [Google ADK 2.0](https://developers.googleblog.com/en/why-we-built-adk-20/) 同时支持图结构工作流、程序控制的动态工作流，以及嵌入其中的 LLM Agent，说明**控制逻辑的表达形式、执行机制和组合方式既有区别，也可以灵活结合**。在此基础上，我们从**控制逻辑的表达形式、执行过程的驱动机制、不同执行机制的组合方式**三个相互关联的角度分析编排实现架构，形成如下技术分类表：
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f6ba494e0f494b60b1c54f5252ba4205.png)
- 需要注意，**编排表达与执行并不是 Graph、Program 和 Agent Loop 等技术方案之间的简单选择**。控制逻辑可以通过图或程序显式表达，也可以由模型根据上下文推理决定；Workflow 与 Agent Loop 可以采用不同 Runtime 实现，并通过嵌套和调用形成混合执行架构。**三个技术选型问题分别刻画控制逻辑的表示、执行和组合方式**，下面分别展开分析。
### 3.2.1 控制逻辑采用什么形式表达？
- 控制逻辑的表达首先可以分为**显式控制表示**与**模型内隐决策**两种基本方式，前者通过外部结构明确描述执行规则与控制关系，后者由模型根据任务目标和上下文推理决定后续行动：
	- **显式控制表示**：主要包括两种技术形式：
	    - **Graph**：通过节点、边和路由条件描述执行步骤及控制流转关系，强调结构化表达执行路径
	    - **Program**：通过程序语言中的条件、循环和函数调用组织控制逻辑，强调程序化的执行控制
	- **模型内隐决策（Model Policy）**：由模型根据任务目标、上下文和执行反馈推理决定后续行动，不要求预先将完整的行动策略表达为外部流程。
- **显式控制表示通常由多种基础技术共同支撑**，主要包括：
  - **控制流结构（Control Flow）**：通过顺序、分支、循环和并行等结构组织执行路径。
  - **规则与条件表达式（Rules & Conditions）**：通过显式规则、条件判断和约束控制执行行为。
  - **状态迁移模型（State Machine）**：通过状态、事件及迁移规则描述和约束执行过程。
  - **领域专用语言（DSL）**：通过专用语法描述任务流程、执行规则与约束。
- **这些基础技术可以灵活组合，共同支撑 Graph 或 Program 的具体实现，而不必视为独立的顶层编排类型**。显式控制与模型内隐决策也可以结合使用，例如通过程序构建执行图，并将局部行动选择交由模型决策。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96f4814d1800414fad127e3b99c26290.png)
	> 以 [Google ADK 2.0](https://adk.dev/graphs/) 为例，其支持 **Graph + Program + Model Policy** 的组合：开发者可以通过执行图定义节点与路由，也可以使用 Python 等程序语言组织复杂控制逻辑，并将需要语义判断的步骤交由 LLM Agent 决策。
### 3.2.2 由什么机制驱动执行？
- 表达出来的控制逻辑还需要相应的运行机制来推进。根据**执行如何组织、如何实现以及如何触发**，可以从三个相互关联的层次分析：
  - **执行组织方式**：主要包括 Workflow 与 Agent Loop，分别强调多步骤工作的流程组织和模型自主决策循环。
    - **Workflow**：通过顺序、分支、并行或循环等控制结构组织多步骤工作，控制逻辑可以预先定义，也可以在运行时生成。
    - **Agent Loop**：由模型根据任务目标与执行反馈逐轮决定下一步行动，持续执行、观察和决策，直到满足停止条件。
  - **运行实现机制**：负责将控制逻辑转化为具体执行过程，主要包括两种方式：
    - **Graph Runtime**：根据执行图中的节点、边和路由规则调度执行。
    - **Script Runtime**：运行控制程序，通过条件、循环和函数调用推进任务。
  - **横向驱动机制**：**Event-driven Runtime** 通过消息、计时器、回调或其他事件触发、恢复和推进任务，可以与上述机制结合。
- 因此，**Workflow 与 Agent Loop 描述执行如何组织，Graph Runtime 与 Script Runtime 描述执行如何实现，Event-driven Runtime 则描述执行如何受到事件驱动**。这些机制不是互斥的执行引擎：Workflow 可以由 Graph Runtime 或 Script Runtime 承载，也可以嵌入 Agent Loop；Agent Loop 则可以独立运行，或在工作流节点中被调用。具体的嵌套与执行能力取决于相应 Runtime 的实现。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63d4e244ffe047e9bd700b80aae5e42d.png)
  > 以 [Claude Code Dynamic Workflows](https://code.claude.com/docs/en/workflows) 为例，其采用 **Workflow + Script Runtime + Agent Loop** 的组合：JavaScript 脚本负责组织多步骤工作、控制分支与循环，并调用能够自主决策的子 Agent。另一种代表是 [UiPath Maestro](https://docs.uipath.com/maestro/automation-cloud/latest/user-guide/events)，其采用 **BPMN Workflow + Event-driven Execution** 的组合：通过 BPMN 描述任务、网关与控制流，由流程引擎执行，并利用消息、计时器等事件触发、等待或恢复流程。在本文的分类框架中，BPMN 流程执行可归入广义的图驱动工作流执行机制。
### 3.2.3 不同执行机制如何组合？
- 在真实 Agent 系统中，不同执行机制往往通过**分层嵌套、节点封装与子流程调用**等方式组织起来，而不是完全独立运行。两种典型的组合手段是：
  - **Agent Node**：将 LLM Agent 封装为上层流程中的执行节点，由外层控制流负责调度；节点内部可以执行单次模型推理，或在 Runtime 支持的情况下运行完整的 Agent Loop。
  - **Nested Workflow**：将子 Workflow 封装并嵌入更大的执行流程，使复杂任务能够分层组织、独立复用；子流程内部也可以继续包含 Agent 节点或其他子工作流。
- **Agent Node 与 Nested Workflow 是两种具体的嵌套与复用手段，而 Hybrid Execution 描述不同类型执行机制协同运行所形成的混合架构**。例如，确定性 Workflow 与自主 Agent Node 结合，或者图驱动工作流与程序驱动的子工作流协同执行，都可以形成 Hybrid Execution。注意，**两种嵌套手段可以单独或组合使用，且嵌套本身不必然构成混合执行，关键在于不同执行机制是否实际协同工作**。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4cec24cc5a6a4b2d80cbc539ad1b61ac.png)
  > 以 [Google ADK 2.0](https://adk.dev/graphs/) 为例，其支持 **Agent Node + Nested Workflow** 等嵌套机制：外层 Graph Runtime 控制整体执行路径，可以将专业 LLM Agent 或复杂子 Workflow 封装为执行节点。当确定性控制流程与具备自主决策能力的 Agent 协同执行时，即形成 **Hybrid Execution**，实现工作流控制与模型自主执行的分层组合。
## 3.3 多 Agent 组织与协作
- 当任务需要多个 Agent 共同完成时，除了决定执行流程如何组织，还需要解决**多个执行主体如何分工、交互与协调**的问题。[Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system)、[Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams) 和 [Cursor](https://cursor.com/blog/self-driving-codebases) 分别采用中心协调、成员直接通信和递归委派等不同机制，说明多 Agent 协作并不存在唯一的组织形式。在此基础上，我们从**组织拓扑、任务委派与信息交互、角色能力与协作关系、任务分配与成员生命周期**四个相互关联的角度分析多 Agent 协作架构，形成如下技术分类表：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c2a85dacbc5e4c399b5d777cdcc87022.png)
- 需要注意，**多 Agent 组织不是对 Star、Tree、Network 等固定架构的简单选择**。不同组织拓扑可以分层组合，任务委派与信息交换是两个正交方面，角色能力差异与协作关系也可以独立配置；任务分配、成员组成与创建、成员存续则是三个相互关联的管理机制。**四个技术选型问题共同刻画多个 Agent 如何建立组织关系、分配工作并持续协作**，下面分别展开分析
### 3.3.1 多个执行主体如何组织？
- 多 Agent 的组织结构首先体现**协作关系的集中程度、层级结构和任务交接方式**。按照组织关系，可以区分三种典型的协调拓扑：
  - **Star**：由中心 Agent 统一协调任务分配、执行进度与结果汇总
  - **Tree**：通过父子 Agent 建立多层级委派关系，支持复杂任务的层级协调与责任划分
  - **Network**：允许 Agent 之间直接建立多对多交互关系，形成分布式协作网络
- 此外，**Pipeline** 描述一种**阶段式协作结构，由不同 Agent 按阶段串联执行**，将前一阶段的结果交给后一阶段。它强调任务在成员之间的交接顺序，与前三种协调拓扑并非同一分类层级。上述组织方式也可以按层级、阶段或局部结构组合，形成 **Hybrid** 架构
- 因此，**Star、Tree、Network 主要描述协调拓扑，Pipeline 描述阶段式协作结构，Hybrid 则表示不同组织结构的组合形态**。例如 Tree 的局部层级可以呈现 Star 结构，Network 内部也可以包含 Pipeline。需要注意，这里的组织拓扑描述的是**Agent 之间的组织与协作关系**，不等于 D2 的执行控制图或 D4 的任务依赖图
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea6c6936f64247408df9b2a222b019a7.png)
 	> 以 [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system) 为例，其采用以 **Star** 为主的中心协调架构：Lead Researcher 负责规划任务、创建多个并行工作的 Subagent，并汇总研究结果。另一种代表是 [Cursor](https://cursor.com/blog/self-driving-codebases)，其采用 **Tree** 式递归委派：Root Planner 可以创建 Subplanner，再由不同层级的 Planner 分配任务给 Worker，形成层级化的任务组织与责任管理。
### 3.3.2 Agent 之间如何委派和交互？
- Agent 之间的协作可以从两个正交方面分析：**任务及执行控制权如何委派，以及 Agent 之间如何交换信息**。在任务委派方面，根据当前执行控制权是否移交、子任务是否具有独立执行责任，可以区分三种典型机制：
  - **Agent-as-Tool**：将专业 Agent 封装为可调用工具，由主 Agent 发起调用并接收结果，保留整体任务的主导控制权
  - **Independent Task Delegation**：将相对独立的子任务委派给其他 Agent 自主推进，原 Agent 保留整体任务控制权，多个执行者可以并行工作
  - **Handoff**：将当前任务或会话的后续执行控制权移交给其他 Agent，使接收方成为当前任务的主要执行者
- 除了任务委派，Agent 还可以通过不同方式**交换与共享信息**：
  - **Messaging**：通过显式消息实现 Agent 之间的定向通信或消息广播
  - **Group Chat**：多个 Agent 通过共享对话交换观点、讨论问题和协调行动
  - **Blackboard**：通过共享黑板、任务板或状态空间发布和获取信息，实现间接协作
- 因此，**委派机制描述任务责任与执行控制权如何分配，信息交换机制描述 Agent 如何共享上下文和协调行动**。两类机制可以独立组合。其中，Agent-as-Tool 强调保留主控的能力调用，Independent Task Delegation 强调独立子任务的责任委派，Handoff 则强调当前执行控制权的转移；而 Group Chat 可以基于 Messaging 实现，Blackboard 则提供基于共享状态的间接协作方式。需要注意，广义的任务委派也可能涵盖 Agent-as-Tool，这里的 Independent Task Delegation 特指独立子任务委派。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b60a7d38d0c4eaeb902d40856760fbd.png)
  > 以 [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/handoff/) 为例，其采用 **Handoff + Messaging** 的组合：Agent 可以根据任务上下文将后续执行控制权移交给其他专业 Agent，同时通过消息同步机制维护参与者的对话上下文。另一种代表是 [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams)，其采用 **Independent Task Delegation + Messaging + Blackboard** 的组合：Lead 向独立运行的 Teammate 分派任务，成员通过 Mailbox 直接通信，并利用共享任务列表协调任务状态与认领工作。其中 Blackboard 是本文对共享任务列表机制的技术归类，并非 Claude 官方命名
### 3.3.3 角色能力及协作关系如何配置？
- 多 Agent 的角色配置可以从**能力与职责差异**以及**成员之间的协作关系**两个正交方面分析。首先，根据成员的实际角色和能力配置是否相似，可以分为：
  - **Homogeneous**：多个 Agent 具有相似的角色、工具和能力配置，适合并行探索、独立采样或处理同类重复任务。
  - **Heterogeneous**：不同 Agent 配置不同角色、工具或专业能力，按照任务需求进行专业化分工。
- 另一方面，根据成员如何共同推进任务，可以区分两种典型协作关系：
  - **Cooperative**：多个 Agent 围绕共同目标合作，通过信息交换、任务分担和结果汇总完成工作
  - **Adversarial**：在局部引入质疑、独立评审或对抗式验证，通过相互检查和反馈发现问题、提高结果质量
- **能力同质性与协作关系是两个正交维度**：同质 Agent 可以合作，也可以互相评审；异质 Agent 同样可以采用合作或对抗式验证。Cooperative 与 Adversarial 也不严格互斥，团队可以在总体目标上合作，同时通过局部对抗机制提升交付质量。这里的 Adversarial 强调**协作过程中的批判与验证关系**，并不要求不同 Agent 具有相互对立的最终任务目标
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2786be7ffb654b248dec4df5db380ddd.png)
	> 以 [MiniMax Agent Team](https://www.minimax.io/blog/minimax-agent-team-long-running-1779893953) 为例，其采用 **Heterogeneous + Cooperative + Adversarial** 的组合：Leader 负责规划和协调，Worker 负责执行子任务，Verifier 则独立检查结果并提出修正要求。团队整体围绕共同目标协作，同时在 Worker 与 Verifier 之间引入对抗式验证，通过独立验收与反馈迭代提高结果质量
### 3.3.4 任务分配与成员生命周期如何管理？
- 当多个 Agent 并行工作时，还需要解决**任务如何分配给执行者，以及 Agent 成员如何创建和持续运行**的问题。这可以从三个相互关联的方面分析。
	1. 在**任务分配机制**上，主要有**可以组合**使用的两种方式，均用于建立和维护任务责任归属（Task Ownership）：
		- **Explicit Assignment**：由协调者或调度器显式指定任务执行者，将任务分配给特定 Agent
		- **Task Claiming**：允许 Agent 从共享任务池中主动认领任务，根据自身能力与任务状态动态确定执行者
	2. 在**成员组成与创建机制**上，根据**团队成员是否预先确定、运行过程中是否允许动态扩展**，区分：
		- **Fixed Roster**：预先确定参与协作的 Agent 集合及角色配置，运行时在既定成员范围内进行任务调度。
		- **Dynamic Spawning**：根据任务规模、分工或执行需要，在运行过程中动态创建新的 Agent，支持团队按需扩展。
	3. 在**成员存续机制**上，根据 Agent **完成当前工作后是否继续保持可用**，可以区分：
		- **Ephemeral Agent**：围绕特定任务或有限会话运行，在工作完成后退出，不要求原实例继续接收新任务
		- **Persistent Agent**：在较长时间内保持可寻址的身份与可用性，能够持续接收任务或事件，并在运行环境支持的情况下恢复执行
- **成员创建方式与存续方式是两个正交的技术问题**：例如，动态创建的 Agent 可以是一次性执行任务的 Ephemeral Agent，也可以在后续保留为 Persistent Agent；预先配置的成员同样可以采用不同存续机制。需要注意，**Persistent Agent 不要求底层进程始终运行，也不等同于任务状态持久化**，其持续可用与跨会话恢复能力仍取决于具体 Runtime 的支持
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/94535df0157d4fdeb296f62af4fce9df.png)
	> 以 [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams) 为例，其结合 **Explicit Assignment + Task Claiming + Dynamic Spawning + Ephemeral Agent**：Lead 可以显式指派任务，Teammate 也可以自主认领；团队成员按需创建，并可在团队会话结束时退出，不保证跨会话持续运行。对于 **Fixed Roster**，[Google ADK](https://adk.dev/agents/custom-agents/) 支持通过 `sub_agents` 预先定义协作成员，在固定成员集合内组织任务委派。另一种代表是 [Cloudflare Agents](https://developers.cloudflare.com/agents/concepts/agentic-patterns/long-running-agents/)，其支持 **Dynamic Spawning + Persistent Agent**：协调 Agent 可以按需创建具有独立身份的子 Agent，后者能够保留持久状态，在休眠后通过新事件重新唤醒，实现跨任务的持续协作
## 3.4 任务执行模式
- 前文 3.2.2 已经区分了 Workflow 与 Agent Loop 两种执行组织机制，它们主要回答**执行过程由什么机制驱动**，而本节进一步讨论**任务如何组织、推进与优化**。同一种任务执行模式可以通过 Workflow、Agent Loop 或两者的组合实现，因此 “任务执行模式” 并非 “编排表达与执行” 的下级分类，而是关注任务处理策略的另一技术维度。
- 第 1 节介绍的 [Anthropic《Building Effective Agents》](https://www.anthropic.com/engineering/building-effective-agents) 总结了 **Prompt Chaining、Routing、Parallelization、Orchestrator-Workers 和 Evaluator-Optimizer** 五种经典 Workflow 模式，为理解任务执行策略提供了基础。但现代 Agent 系统还广泛采用递归任务分解、反馈重规划、候选方案搜索和多种验证改进机制，任务执行方式已经超出了狭义 Workflow 的范围。在此基础上，我们从**任务串联与分支选择、并行执行与结果汇总、动态任务分解与重规划、候选方案搜索与筛选、评价反馈与迭代改进**五个相互关联的角度分析任务执行策略，形成如下技术分类表：
	![任务执行模式技术分类表](https://i-blog.csdnimg.cn/direct/a057332b69f245e0a66cbc66c8be142c.png)
- 需要注意，**五个技术选型问题并非互斥的执行模式，也不是必须依次进行的五个阶段**。一个复杂 Agent 系统可以动态分解任务、并行推进子任务、搜索多个候选方案，再根据评价反馈持续改进。例如，Orchestrator-Workers 可以结合 Parallelization，Tree Search 可以在候选扩展过程中引入 Evaluator-Optimizer。**五个技术选型问题共同刻画任务如何展开、探索与收敛**，下面分别展开分析
### 3.4.1 任务如何串联或选择分支？
- 任务执行最基本的两种组织方式是**顺序串联与条件分支**，分别对应 Prompt Chaining 和 Routing：
  - **Prompt Chaining**：将任务拆分为多个前后依赖的步骤，上一阶段的输出作为下一阶段的输入，形成顺序执行链
  - **Routing**：根据任务类型、输入特征或运行状态，选择不同的后续执行路径。Classify-and-Act 是一种典型实现，它先对任务分类、再执行对应动作
- 两者分别关注不同的控制关系：**Prompt Chaining 描述任务步骤之间的顺序依赖，Routing 描述后续执行路径的条件选择**，两者可以嵌套，例如先识别任务类型，再进入对应的多步骤处理流程；也可以在顺序执行链中设置条件分支
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ad74635ba9da4bf2b16a448372e8be9b.png)
  > 以 [Anthropic《Building Effective Agents》](https://www.anthropic.com/engineering/building-effective-agents) 中的客服处理场景为例，可以设计 **Routing + Prompt Chaining** 的组合：先将请求路由到一般咨询、退款处理或技术支持等不同流程，再按顺序执行信息查询、问题处理与结果检查
### 3.4.2 子任务如何并行执行和汇总？
- 当多个子任务之间不存在严格的先后依赖时，可以通过**并行执行（Parallelization）** 提升处理效率，或通过多次独立尝试增加结果可靠性。[Anthropic](https://www.anthropic.com/engineering/building-effective-agents) 将 Parallelization 进一步划分为两种典型模式：
	- **Sectioning**：将任务拆分为不同的独立子任务，并行处理后汇总结果，主要用于提高吞吐量或分别处理不同方面
	- **Voting**：对同一任务进行多次独立求解或判断，通过结果聚合、投票或阈值规则形成最终结论，主要用于提高可靠性
- **Sectioning 强调不同子任务的分工处理，Voting 强调同一任务的多次独立尝试与结果聚合**。两者都需要汇总并行执行的结果，但聚合目标不同：前者通常整合不同子任务的成果，后者则通过多个独立判断形成综合结论。需要注意，Voting 与 3.4.4 中的 Best-of-N 不同：前者侧重结果聚合，后者侧重从多个候选方案中选择较优结果。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f66079609c094312b8d6be8f8efda5d6.png)
  > 以 [Claude Code Dynamic Workflows](https://code.claude.com/docs/en/workflows) 为例，其可以采用 **Parallelization 中的 Sectioning模式**：通过 JavaScript 编排多个 Subagent 并行审查不同文件，再汇总审查发现、排序和去重，实现独立任务的并行处理与结果整合。
### 3.4.3 任务如何动态拆分和重新规划？
- 前述串联和并行模式可以基于预定义任务执行，但复杂任务的子任务数量、依赖关系和执行顺序往往难以提前确定，因此需要**动态任务分解与反馈重规划**。相关技术可以从任务分工、层级分解和计划调整三个角度分析：
  - **Orchestrator-Workers**：由协调者根据当前任务动态拆解工作，分派给多个 Worker 执行并汇总结果
  - **Recursive Decomposition**：将复杂任务递归分解为更小的子任务，直到形成可以直接执行的工作单元
  - **Plan-Execute-Replan**：先制定计划并执行，再根据执行结果和环境变化调整后续计划
- **三者并非互斥的分解算法，而是分别描述任务分工、分解层级和反馈调整机制**。例如，Orchestrator 可以通过 Recursive Decomposition 逐级拆分任务，Worker 返回结果后，再通过 Plan-Execute-Replan 调整后续工作。其中，Orchestrator-Workers 与预定义 Parallelization 的主要区别在于**具体子任务可以运行时动态确定**；Recursive Decomposition 描述任务的层级分解，不等于 3.3 的 Agent Tree 组织拓扑；Plan-Execute-Replan 则是 3.1 中 Replanning 在任务执行层面的具体应用
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f976b217457484fb534ae74319bf91b.png)
  > 以 [Cursor Self-driving Codebases](https://cursor.com/blog/self-driving-codebases) 为例，其采用 **Orchestrator-Workers + Recursive Decomposition** 的组合：Root Planner 根据任务复杂度创建 Subplanner，各级 Planner 继续拆分工作，Worker 负责具体实现；Planner 接收任务结果与执行反馈后，可以继续调整后续规划，体现出 **Plan-Execute-Replan** 的反馈重规划特征。另一种代表是 [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system)，其采用 **Orchestrator-Workers + Parallelization**，由 Lead Researcher 动态分解研究任务并协调多个并行工作的 Subagent
### 3.4.4 候选方案如何搜索和筛选？
- 动态任务分解解决的是**如何组织和推进工作**，但对于复杂推理、编程和方案设计，系统还可能需要**探索多个候选解，并从中寻找更优方案**。按照候选空间的组织与探索方式，可以区分三种典型搜索策略：
  - **Best-of-N**：针对同一问题生成多个候选方案，通过评价或比较选择较优结果，主要体现单轮多候选选优。
  - **Tree Search**：将中间决策组织为搜索树，探索不同的后续分支，支持前瞻、剪枝与回溯。
  - **Evolutionary Search**：维护候选方案群体，通过选择、变异、重组与迭代评价持续改进方案。
- **三者分别体现单轮候选选优、多步路径探索与群体迭代优化**，但并不是严格互斥的搜索策略。例如，Tree Search 可以在节点扩展时生成多个候选并择优探索，Evolutionary Search 也可以在每轮迭代中对多个候选进行评价与筛选。需要注意，**Tree Search 与 Recursive Decomposition 不同**：前者探索备选决策路径，后者分解需要完成的工作任务。此外，Best-of-N 与 3.4.2 中的 Voting 都涉及多次尝试，但前者侧重根据评价选择较优候选，后者侧重通过聚合多个独立判断形成结论。
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8b0472e8ad14f1780e527c7af7c7bec.png)
  > 以 [Google DeepMind AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 为例，其采用 **Evolutionary Search**：模型生成候选算法程序，由自动评估器验证和评分，再通过演化机制持续选择和改进候选方案。**Tree Search** 的代表性研究则是 [Tree of Thoughts](https://arxiv.org/abs/2305.10601)，其将推理过程组织为多条可探索的中间路径，通过评价、搜索与回溯寻找更优解。
### 3.4.5 结果如何评价与迭代改进？
- 无论任务采用顺序、并行还是搜索方式执行，都需要解决**如何判断当前结果是否满足要求，以及如何利用反馈持续改进**的问题。**Evaluator-Optimizer** 是一种典型的反馈优化模式：由执行者生成结果，评估者检查结果并提供反馈，再根据反馈进行修正与重新评价，形成迭代优化闭环。根据**评价反馈的来源和形成机制**，可以进一步区分三种典型方式：
  - **Self-Reflection**：由 Agent 对自身生成的结果进行检查、反思并提出修正建议，实现自我反馈与改进。
  - **Adversarial Verification**：由独立验证者通过质疑、检查、反例或测试发现结果中的错误与缺陷，为后续修正提供依据。
  - **Debate**：通过多个 Agent 的讨论或辩论比较不同观点、证据和候选结果，形成评价或改进建议。
- **Evaluator-Optimizer 描述生成、评价与修正的反馈优化闭环**，Self-Reflection、Adversarial Verification 与 Debate 则是评价环节可以采用的**三种典型反馈机制**。它们可以分别嵌入优化闭环，也可以组合产生反馈，但单独使用时不必然形成完整的迭代优化过程。例如，同一 Agent 可以通过 Self-Reflection 反复修正结果，也可以由独立 Verifier 进行对抗式检查，或通过多个 Agent 的 Debate 形成改进建议。。反馈优化通常还需要设置**迭代终止条件**，例如结果通过验收、达到最大迭代次数或资源预算上限。需要注意，这里的反馈优化主要关注**执行结果的质量改进**，不同于 3.4.3 中面向任务组织与执行计划的反馈重规划；而 Adversarial Verification 和 Debate 描述的是具体评价过程
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e72ce45229f34f17b0d90335452f04a5.png)
  > 以 [Self-Refine](https://arxiv.org/abs/2303.17651) 为例，其采用 **Self-Reflection + 迭代优化**，通过同一模型生成初始结果、自我评价并反复修正。[MiniMax Agent Team](https://www.minimax.io/blog/minimax-agent-team-long-running-1779893953) 则体现了 **Evaluator-Optimizer + Adversarial Verification** 的组合：Worker 生成结果，Verifier 独立检查并提出修正要求，验证失败后重新唤醒执行节点进行改进。另一种代表是 [Google DeepMind Co-Scientist](https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/)，其结合 **Debate + 迭代优化**，通过多 Agent 评审、观点辩论和候选演化持续改进研究假设
# 4. 典型 Agent 架构与技术选型
- 前文从**决策与控制（D1）、编排表达与执行（D2）、多 Agent 组织与协作（D3）、任务执行模式（D4）** 四个维度梳理了现代 Agent 编排的主要技术选项。在真实系统中，这些机制并不是独立使用的，而是根据任务需求形成不同的架构组合。以 Codex、Google ADK、Anthropic Multi-Agent Research 和 Cursor 等代表性系统为例，可以进一步分析不同编排架构的设计思路与适用场景。
## 4.1 典型架构的技术组合
- 按照主要控制机制和协作结构，可以归纳出四种具有代表性的架构形态：
	| 典型架构及代表系统 | D1 决策与控制 | D2 编排表达与执行 | D3 多 Agent 组织与协作 | D4 任务执行模式 |
	|---|---|---|---|---|
	| **自主 Agent Loop**（Codex、Claude Code） | Model-driven 为主，代码约束工具执行 | Agent Loop + Harness | 单 Agent 为核心，可扩展 Subagent | 动态行动选择、反馈修正 |
	| **Workflow-Agent 混合架构**（Google ADK 2.0） | Code-driven + Model-driven | Graph / Program + Agent Node + Workflow | 可配置专业 Agent 与协作关系 | Prompt Chaining、Routing、Evaluator-Optimizer 等 |
	| **中心协调式多 Agent**（Anthropic Multi-Agent Research） | Lead Agent 动态决策 | 多个 Agent Loop 协同执行 | Star + Independent Task Delegation | Orchestrator-Workers + Parallelization |
	| **递归分工式多 Agent**（Cursor Self-driving Codebases） | 多层 Planner 动态决策 | 多 Agent 自主执行与协同调度 | Tree + Task Claiming + Dynamic Spawning | Recursive Decomposition + 反馈重规划 |
	- **自主 Agent Loop 架构**以 [OpenAI Codex](https://openai.com/index/unrolling-the-codex-agent-loop/) 和 [Claude Code](https://code.claude.com/docs/en/how-claude-code-works) 为代表，强调模型根据执行反馈逐步决定后续行动，Harness 负责工具调用、上下文组织与执行约束。它不要求预先定义完整的任务流程，适合代码修改、问题排查等执行步骤难以提前确定的任务。
	- **Workflow-Agent 混合架构**以 [Google ADK 2.0](https://adk.dev/graphs/) 为代表，通过显式控制流程组织整体任务，将需要模型推理的步骤交给 Agent 执行。这种架构兼顾确定性流程控制与局部自主决策，适合具有明确业务规则、同时需要语义理解和动态处理能力的场景。[Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/) 也提供 Sequential、Concurrent、Handoff 和 Group Chat 等多种编排方式
	- **中心协调式多 Agent 架构**以 [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system) 为代表，由 Lead Agent 统一规划、委派和汇总任务，多个 Subagent 分别探索不同方面。其优势是能够利用任务分解与并行执行提高复杂任务的处理效率，但也会引入协调开销以及上下文和结果汇总成本
	- **递归分工式多 Agent 架构**以 [Cursor Self-driving Codebases](https://cursor.com/blog/self-driving-codebases) 为代表，通过 Root Planner、Subplanner 和 Worker 建立多层级任务分解与责任管理机制。相较于单层协调结构，它能够进一步细化大规模任务，但也对任务边界、责任归属、执行隔离与结果整合提出更高要求
- 需要注意，**以上四种架构是基于本文技术体系归纳的代表性组合，而不是严格互斥的产品类别**。自主 Agent 可以调用 Workflow，中心协调架构可以采用递归分解，递归多 Agent 系统内部也可以嵌入确定性执行流程。因此，架构设计的关键不是选择某一个固定模板，而是根据任务特点组合适当的技术机制。
## 4.2 编排架构的技术选型原则
- 实际选型应当从**任务确定性、动态决策需求、并行收益与可靠性要求**出发，在编排灵活性、执行效率、系统复杂度和运行成本之间权衡：
  1. **优先采用简单、可控的执行机制**：对于执行步骤明确、业务规则稳定的任务，可以优先使用 Code-driven Workflow，通过 Prompt Chaining、Routing 等模式组织工作，而不必引入自主 Agent 或复杂多 Agent 架构。
  2. **根据任务不确定性引入模型自主决策**：当任务步骤难以预先定义，需要根据环境反馈不断选择行动时，可以采用 Model-driven Agent Loop，并通过工具权限、执行边界和终止条件约束其行为。
  3. **根据分工收益决定是否扩展多 Agent**：当任务可以拆分为相对独立的工作单元，且并行收益能够覆盖协调成本时，可以采用 Orchestrator-Workers 与 Star 结构；只有在任务规模和层级复杂度确实需要时，才进一步引入 Tree 与 Recursive Decomposition。
  4. **根据质量要求配置反馈与验证机制**：对于可靠性要求较高的任务，可以结合 Evaluator-Optimizer、独立验证、自动测试和必要的人工审批，通过明确的验收标准与反馈闭环提升交付质量。
- 总体而言，**编排架构应当随任务复杂度逐步扩展，而不是默认采用能力最全面的多 Agent 系统**。增加 Agent 数量或控制层级并不必然提高效果，反而可能带来更多通信、状态同步、错误传播和资源消耗。因此，合理的架构通常是在满足任务要求的前提下，使用尽可能简单的控制机制，并将自主决策、并行协作与反馈优化能力引入真正需要的位置。
# 5. 总结与展望
- 本文从**决策与控制、编排表达与执行、多 Agent 组织与协作、任务执行模式**四个维度，系统梳理了现代 AI Agent 编排的核心技术及其分类关系。可以看到，**Agent 编排并不是某一种特定框架或算法，而是多个不同层级、相互关联的技术机制共同构成的系统架构**。Workflow 与 Agent Loop 可以嵌套，代码控制与模型决策可以结合，多 Agent 协作结构和任务执行策略也可以根据需求灵活配置。
- 随着 Agent 能力增强，系统设计的重点正在从单纯提高模型自主性，转向**自主决策与确定性控制的合理结合，以及复杂任务的可靠执行**。面对不同应用需求，并不存在统一的最优编排架构，关键在于根据任务的不确定性、可分解性、质量要求和资源约束选择适当的技术组合。
- 但**编排策略只是 Agent 系统可靠运行的一部分**。当任务规模扩大、执行时间延长、协作关系复杂化时，还需要上下文组织、任务调度、状态恢复和质量验证等工程能力提供支撑。因此，下一篇将进一步从**上下文与状态管理（S1）、协同调度与 Runtime（S2）、验证与评估（S3）** 三个横向维度，分析现代 Agent 系统的工程支撑机制及其技术选型。
	
	
