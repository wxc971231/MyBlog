>因为想申请 CSDN 博客认证需要一定的粉丝量，而我写了五年博客才 700 多粉丝，本文开启关注才可阅读全文，很抱歉影响您的阅读体验
- 重要度采样比的概念在 RL 中似乎很简单，我以前也没太关注过，最近看 PER 论文突然想到一个问题，为何基于 DQN 的 PER 需要重要度采样比，而基于 Q-learning 的优先级 Dyna-Q 则不用，由此引发的一些思考整理成此文
- 开篇先说一下我的看法，针对  **off-policy** 的 **value based control** 方法 
	>1. MC Control 类方法**使用行为策略 $b$ 收集的数据估计目标策略 $\pi$ 对应的价值 $Q_\pi$，必须用重要性采样比调整期望**
	>2. TD Control 方法基于 Bellman Optimal Equation，使用行为策略 $b$ 收集的数据直接估计最优策略 $\pi_*$ 对应的价值 $Q_*$ 得到 $Q$，进而导出行为策略 $\pi$。这时**只要考虑 $Q_*$ 的估计情况**，由公式可知 **$Q_*$ 的对应的期望和策略无关**，因此我们可以使用任何行为策略收集数据，自然也不需要重要度采样比，这**对于表格型 TD control 方法都成立**（如 Q-learning、优先级 Dyan-Q 等）
	>3. 对于使用的函数近似的 TD control 方法而言，由于函数近似方法本身的性质，它们得到的价值估计总会存在误差（注意并不是 “错误”），**行为策略 $b$ 越少访问的那些 $(s,a)$，价值估计误差越大**。这时我们应该**避免使用和目标策略 $\pi$ 相差太大的行为策略 $b$，以保证训练过程中遇到的 $q(s,a)$ 总是有较小的估计误差**。DQN 中行为 $b$ 和 $\pi$ 相差不大因此不需要重要度采样比；引入 PER 后这个差距无法控制，因此需要使用重要度采样比
- -----
@[toc]
- 重要度采样比主要用于 **off-policy** 的 **value based control** 方法，这类方法特点为
	1. value based 意味着 agent 首先估计价值函数，再从中导出策略
	2. off-policy 意味着 agent 学习的 target policy $\pi$ 和与环境交互使用的 behavior policy $b$ 不同

	 解 control 问题时，此类方法通常基于 policy iteration 思想，不停循环 “估计目标策略价值” 和 “优化目标策略” 两步直到收敛，示意图如下
		![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/92f9cafb29c8ae6f80efbd8570f5d09b.png#pic_center =70%x)



# 1. 什么是重要度采样比
- 有时我们想估计目标分布下某一统计量的期望值，但是只有另一个分布的样本，这种**利用来自其他分布的样本估计目标分布期望值的通用方法称为 `重要度采样`**
- 观察前面的算法交互图，我们的**目标是估计出目标策略 $\pi$ 对应的价值** $\left\{
\begin{aligned}
&V_\pi(S) = \mathbb{E}_\pi[G|S] \\
&Q_\pi(S,A) = \mathbb{E}_\pi[G|S,A] \\
\end{aligned}
\right.$，但是估计时利用的样本全部来自行为策略 $b$，所以需要做重要度采样，我们可以**根据每个 episode 或 transition 在目标策略和行动策略中出现的概率比例（相对概率）对其 return $G$ 进行加权调整**，从而获得目标策略 $\pi$ 的 return 期望。**这个用于调整的相对概率就是`重要度采样比 ρ`** 
	$$
	\begin{aligned}
	&对于一个 \text{transition} &&\rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}\\
	&对于一段 \text{episode}&&\rho_{t:T} = \prod_{k=t}^T\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
	\end{aligned}
	$$
- 简而言之，当价值估计方法**对行为策略 $b$ 敏感**，价值估计 $V_b(S),Q_b(S,A)$ 和目标价值 $V_\pi(S),Q_\pi(S,A)$ **不相等**时，可以使用重要度采样比调整期望，使 $\left\{
\begin{aligned}
&\mathbb{E}[V_\pi(S)] = \mathbb{E}[V_b(S)] \\
&\mathbb{E}[Q_\pi(S,A)] = \mathbb{E}[Q_b(S,A)] \\
\end{aligned}
\right.$，**目标价值估计更准确**
# 2. 使用重要度采样比的场景
## 2.1 MC Control 方法（使用）
- MC Control 使用蒙特卡洛方法做 prediction，其思想是**直接使用经验期望估计真实期望**，公式为（$\leftarrow$ 表示估计）
	$$
	\begin{aligned}
	&v_\pi(s) \leftarrow v(s) =\mathbb{E}_b[G_t|S_t=s] \\
	&q_\pi(s,a)  \leftarrow q(s,a) = \mathbb{E}_b[G_t|S_t=s,A_t=a] \\
	\end{aligned}
	$$ 显然经验期望是受到行为策略影响的，因此需要做重要度采样
- 对于这种 MC prediction，有两种重要度采样方法，定义 $\tau(s)$ 为所有**访问过状态 $s$ 的时刻**的集合
	1. 使用`普通重要度采样比`：$v(s) = \frac{\sum_{t \in \tau(s)}\rho_{t:T(t)-1}}{|\tau(s)|}G_t$，得到无偏估计 $\mathbb{E}[v(s)] =  \mathbb{E}_\pi[G_t|S_t=s]$，但方差无界
	2. 使用`加权重要度采样比`：$v(s) = \frac{\sum_{t \in \tau(s)}\rho_{t:T(t)-1}}{\sum_{t \in \tau(s)}\rho_{t:T(t)-1}}G_t$，得到有偏估计 $\mathbb{E}[v(s)] =  \mathbb{E}_b[G_t|S_t=s]$，但可收敛到0

- 估计 $Q(S,A)$ 的情况完全类似，这里不再展开，示意图如下
	![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a8473d6fe8e5c66eaf29212e4927b1e7.png#pic_center =60%x)
	
## 2.2 TD Control 方法（必要时使用）
- 所有 TD 方法的核心都是 Bellman equation 或 Bellman optimal equation，分别对应 on-policy 方法和 off-policy 方法
- 这两个等式有很好的性质，比如他们对应的算子都是压缩映射（可以依此证明价值估计一定收敛），又比如 Off-policy 情况下 TD target 是和行为策略无关的，请看下文分析
### 2.2.1 Q-learning（不使用）
- Q-learning 算法直接对 Bellman optimal equation 不停迭代来估计价值，得到的估计期望如下
	$$
	\begin{aligned}
	q_*(s,a) &= \max\mathbb{E}_{\pi_*}[G_t|S_t=s,A_t=a] \\
				&= \max \mathbb{E}_{\pi_*}[R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a] \\
				&= \mathbb{E}[R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a')|S_t=s,A_t=a] \\
				&= \sum_{s',r}p(s',r|s,a)[r+\gamma \max_{a'}q_*(s',a')]
	\end{aligned}
	$$ 注意到**等式右侧的期望只与状态转移分布 $p$ 有关而与策略无关，不论训练 transition 来自于哪个策略，按照 Q-Learning 的更新式更新都能使 $Q$ 接近 $Q^*$，又因为 Q-learning 是一个表格型方法没有泛化性，对一个 $(s,a)$ 的价值更新完全不会影响到其他 $(s,a)$ 的价值估计，因而无论使用什么行为策略去收集 transition，都不会引入偏差，也就不需要重要性采样**
- 再仔细分析一下，得益于使用 Bellman optimal equation，这里有了一些 value iteration 的意味，我们**估计的对象从目标策略的价值 $Q_\pi(S,A)$ 变成了最优策略价值 $Q_*(S,A)$**，这样我们就不用再考虑对 $Q_\pi(S,A)$ 的估计有没有偏差了，具体地讲
	1. 原来我们利用 $b$ 交互的样本估计 $Q_\pi(S,A)$，再根据结果提升 $\pi$ 和 $b$，直到 $\pi \to \pi_*$，引入重要度采样是为了更好地估计 $Q_\pi(S,A)$
	2. 现在我们利用 $b$ 交互的样本直接估计 $\pi_*$ 对应的 $Q_*(S,A)$，再从中直接导出更好的 $\pi$ 和 $b$。注意到这里**估计 $Q_*(S,A)$ 没有偏差，因此不需要引入重要度采样比**
	
- 需要注意的是，**不同行为策略 $b$ 导致的价值收敛过程还是不一样的**，行为策略更频繁访问 transition 的价值会更快收敛，当计算 transition 数量趋向无穷时，所有能无限采所有 $(s,a)$ 的行为策略 $b$ 最终会学到一样的价值函数
		
	> 这就好像是用茶壶往一堆杯子里倒水，向一个杯子倒水不会干扰其他杯子。行为策略指导你向哪些杯子倒，其频繁访问的杯子更快装满水（价值收敛），如果所有杯子被访问的概率都大于 0，则时间趋于无穷时总能使所有杯子装满水（价值全部收敛到相同位置）
### 2.2.2 Dyna-Q（不使用）
- Dyna-Q 是 Sutton 在 RL 圣经中第八章介绍的方法，其实就是 Q-learning 加上了经验重放机制，同一章中还介绍了一种 “优先级遍历算法”，其实就是 Dyna-Q 加上了简化版本的经验优先重放（PER）机制
- 无论是均匀的经验重放，还是不均匀的经验重放，都可以把他们看做行为策略 $b$ 的一部分，根据 2.2.1 节中的分析，这些**不会改变价值估计结果，只是在经验重放中被强调过的那些 $(s,a)$ 的价值估计会较快收敛而已**

### 2.2.3 DQN（不使用）
- DQN 基本可以看作将 Dyna-Q 直接扩展到深度学习得到的方法，它通过优化 TD error 的 L2 损失使价值估计不断靠近 TD target，最终使所有 $s$ 或 $s,a$ 的价值估计符合 Bellman optimal equation 等式（收敛），并且引入了经验重放机制
- 具体来说，梯度下降第 $i$ 轮迭代时损失函数为
		$$
		\mathcal{L_i} = \mathbb{E}_{s,a\sim\rho(·)}\big[(y_i-Q(s,a,\theta_i)^2)\big]
		$$ **其中 $\rho$ 是行为策略 $b$ 诱导的 $(s,a)$ 分布**，$y_i$ 是 TD target $$y_i = \mathbb{E}_{s'\sim \text{env}}\big[r+\gamma\max_{a'}Q(s',a',\theta_{i-1})|s,a\big]$$ 损失函数梯度为
		$$
		\triangledown_{\theta_i}\mathcal{L}_i = \mathbb{E}_{s,a\sim\rho(·),s'\sim \text{env}}\big[\big(r+\gamma\max_{a'}Q(s',a',\theta_{i-1})-Q(s,a,\theta_i)\big)\triangledown_{\theta_i} Q(s,a,\theta_i)\big]
		$$ 和 Q-leaning 非常相似，这里也是基于 Bellman optimal equation 做更新，利用行为策略收集的样本直接估计价值 $Q_*(S,A)$，并从中导出更好的 $\pi$ 和 $b$。**区别在于，这里梯度期望中 $\rho$ 是由行为策略 $b$ 决定的**，注意这在 Q-learning 中也存在（行为策略总会偏向某些 $(s,a)$），但是
	1. 表格型方法没有泛化性，行为策略访问更多的 $(s,a)$ 的价值估计虽然收敛快，但不会影响其他 $(s,a)$ 的价值估计       
	2. 函数近似方法有泛化性，行为策略 $b$ 访问更多的 $(s,a)$ 占据了梯度下降时 mini-batch 中的大部分，决定了价值网络更新时的梯度方向，这会**使 $b$ 偏好的那部分 $(s,a)$ 的价值估计更准，同时使其他 $(s,a)$ 的价值估计更差**，有点像监督学习中的过拟合
	
		> 这就好像是用水桶往一堆杯子泼水，往一个地方泼水，所有杯子都会溅到，水还可能洒出来。行为策略指导你向哪些泼水，这时行为策略的一点点区别，都会导致各个杯子最终的水量不同  
- 虽然听起来好像问题很严重，甚至让人怀疑 DQN 这类方法到底能不能接受任意行为策略 $b$ 产生数据，但是这个 “不能算 bug，只能算 feature”，函数近似方法假设**要估计价值的状态维度远远多于权重维度**，因此**一个状态的价值更新会影响到许多状态，某些状态估计价值越准确，意味着其他状态的价值估计越不准确**，这是通过参数化模型减少待估参数数量时的自然性质。具体地讲
	1. DQN 作为 off-policy 方法是没错的，因为借助 bellman Optimal equation，我们确实可以利用任意行为策略 $b$ 产生的数据来估计 $Q_*(S,A)$ 并进一步得到 $\pi_*$
	2. DQN 不使用重要度采样比是没错的，可以这样想，假设我们用了一个参数量非常多的价值网络，多到参数量和 $(s,a)$ 数量一样了（这时其实你可以把它看作 Q-learning 中的 Q 表），那么 2.2.1 节中的等式仍然对所有 $(s,a)$ 成立，无论使用什么行为策略 $b$ 都不会引入偏差


	虽然没有错，但是我们还是**应该尽量避免使用和当前目标策略 $\pi$ 相差太多的行为策略 $b$ 收集数据**，因为 $\pi$ 是从当前估计的 $Q$ 中得到的，如果 $b$ 和 $\pi$ 相差很大，那么 $b$ 收集的数据就都是来自**当前 $Q$ 估计不好的那些区域**，由于函数近似法的特性，我们**不可能把所有地方的价值都估计好**，如果一直用和 $\pi$ 相差很多的 $b$，就好像用小一号的床单铺床一样 “床头盖上了床尾漏出来，又去盖床尾床头又漏出来”，$Q$ 估计值不断震荡难以收敛， 探索也没有方向性
- DQN 原文中使用了均匀的经验重放机制，行为策略可以看做近一段时间目标策略 $\pi$ 的均匀混合，因此最终是能 work 的。DQN 论文讲解请参考：[论文理解【RL经典】 —— 【DQN】Human-level control through deep reinforcement learning](https://blog.csdn.net/wxc971231/article/details/124110973)

### 2.2.4 PER（使用）
- PER 是适用于 DQN 系列方法的一个 trick，它在经验回放的过程中以更高的概率重放那些 TD error 更大的 transition，这相当于给原始 DQN 中的行为策略 $b$ 加入了一个无法预测的偏移，这时 $b$ 就可能和 $\pi$ 相差很多了，根据 2.2.3 节的分析，这时需要使用重要度采样比，把 $b$ 对应的价值期望拉回到 $\pi$ 的水平
- PER 论文讲解请参考：[论文理解【RL - Exp Replay】 —— 【PER】Prioritized Experience Replay](https://blog.csdn.net/wxc971231/article/details/123415486)

## 2.3 其他
- RL 中还有一些重要性采样比的高级用法，比如 “折扣敏感的重要度采样比”（见 Sutton 圣经书 5.8 节），但和本文主要讨论的问题无关，不再展开