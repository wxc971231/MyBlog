- NIPS 2025 Best paper [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://blog.csdn.net/wxc971231/article/details/155713258) 观察到当前对 LLM 进行 RL 训练的局限性：**RLVR 只能在基座模型已有推理路径上提高采样效率，不仅无法创造新的推理能力，还常导致推理覆盖范围缩窄**
- 本文参考俞扬老师的文章 “[从一些基础的优化说一下RL在LLM中的探索问题](https://zhuanlan.zhihu.com/p/1924231348333314311)”，分析 RL 等无梯度优化方法在平衡探索和利用时的先天特性，并说明 **LLM-RL 的困境本质：探索不足导致陷入局部最优**
------ 
@[toc]
# 1 可微代理优化
- **`可微代理优化 (Differentiable Surrogate Optimization)`**：考虑原始优化问题 $\max_x f(x)$，当目标函数 $f(x)$ 对决策变量 $x$ **不可微、不可得解析梯度或梯度估计不稳定**时，难以直接基于 $\nabla_x f(x)$ 进行优化。此时引入一个由参数 $\phi$ 控制、且对 $\phi$ **可微**的决策变量代理分布 $p_\phi(x)$，构造代理目标 $$J(\phi)=\mathbb{E}_{x\sim p_\phi}[f(x)]$$ 假设 $f$ 不显式依赖 $\phi$（$\phi$ 仅通过采样分布影响 $x$），则即使 $f(x)$ 对 $x$ 不可微，仍可通过对数导数技巧得到对 $\phi$ 的梯度估计
	$$
	\begin{aligned}
	\nabla_{\phi} J(\phi)
	&=\nabla_{\phi} \int f(x) p_{\phi}(x) d x \\
	&=\int f(x) p_{\phi}(x) \nabla_{\phi} \log p_{\phi}(x) d x \\
	&=\mathbb{E}_{x \sim p_{\phi}}\left[f(x) \nabla_{\phi} \log p_{\phi}(x)\right]
	\end{aligned}
	$$ 再加 baseline $b$ 降低方差，并利用 $\mathbb{E}_{x\sim p\phi}[\nabla_\phi \log p_\phi(x)]=0$ 保持期望不变，得到核心结构
	$$\nabla_{\phi} J(\phi)=\mathbb{E}_{x \sim p_{\phi}}\left[(f(x)-b) \nabla_{\phi} \log p_{\phi}(x)\right]$$ **如此通过梯度更新 $\phi$ 改变代理对象 $p_\phi$**，就可以把搜索/采样的概率质量推向高 $f(x)$ 的区域，从而间接提升原始目标 $f$ 的性能
- 注意到 “可微代理优化” 包含两个核心要素：
	1. **代理（surrogate）**：不直接在 “决策本体” $f$ 上做离散的 $\argmax$ 搜索，而是选一个可控的概率分布族 $p_\phi$ 来 “代表你当前在搜索什么”，通过调整分布参数 $\phi$ 来把概率质量推向高回报区域
	2. **可微（differentiable）**：选出 的 $p_\phi$ 对参数 $\phi$ 是可微的，这保证我们可以用对数导数技巧得到一个可用的梯度估计
# 2 Policy-Gradient RL 是一类可微代理优化方法
- 从可微代理优化的角度看，**RL 里我们想最大化交互轨迹的回报，但回报来自不可微的环境反馈没法直接优化，为此我们引入诱导了轨迹分布的可微策略作为代理分布，通过策略梯度优化策略从而间接提升回报**
- 形式化地讲，设有 $\phi$ 参数化的策略 $\pi_\phi(a|s)$，其和环境 $p(s'|s,a)$ 交互诱导出的轨迹分布为 $$p_{\phi}(\tau)=p\left(s_{0}\right) \prod_{t} \pi_{\phi}\left(a_{t} \mid s_{t}\right) p\left(s_{t+1} \mid s_{t}, a_{t}\right)$$ 优化目标是最大化期望回报 $$J(\phi) = \mathbb{E}_{\tau\sim p_\phi(\tau)}[R(\tau)]$$ 此时轨迹 $\tau$ 是决策变量 $x$，累积（折扣）奖励 $R$ 就是目标函数 $f$，策略 $\pi_\phi$ 是代理分布。由于初始状态分布 $p(s_0)$ 和环境转移 $p(s_{t+1} \mid s_{t}, a_{t})$ 都和 $\phi$ 无关，得到梯度公式 
	$$
	\begin{aligned}
	\nabla_{\phi} J(\phi)
	&=\mathbb{E}_{\tau\sim p_{\phi}}\left[R(\tau) \nabla_{\phi} \log p_{\phi}(\tau)\right] \\
	&=\mathbb{E}_{\tau\sim p_{\phi}}\left[R(\tau) \left(\nabla_{\phi} \log p\left(s_{0}\right)+ \nabla_{\phi}\sum_{t} \log \pi_{\phi}\left(a_{t} \mid s_{t}\right)+\nabla_{\phi} \sum_{t} \log p\left(s_{t+1} \mid s_{t}, a_{t}\right)\right) \right]\\
	&=\mathbb{E}_{\tau\sim p_{\phi}}\left[R(\tau)\sum_{t} \nabla_{\phi}\log \pi_{\phi}\left(a_{t} \mid s_{t}\right) \right]
	\end{aligned}
	$$ 注意到其中 $\nabla_{\phi}\log \pi_{\phi}\left(a_{t} \mid s_{t}\right)$ 只和当前的 $(s_t,a_t)$ 有关，我们从 $t$ 时刻把轨迹切分成前后两部分，引入 $R_{<t}=\sum_{k=0}^{t-1}\gamma^k r_k$ 和 RTG $G_t=\sum_{k=t}^{\infin}\gamma^k r_k$，使 $R(\tau)=R_{<t} +G_t$，继续推导
	$$
	\begin{aligned}
	\nabla_{\phi} J(\phi)
	&=\mathbb{E}_{\tau\sim p_{\phi}}\left[R(\tau)\sum_{t} \nabla_{\phi}\log \pi_{\phi}\left(a_{t} \mid s_{t}\right) \right] \\
	&= \mathbb{E}_{\tau\sim p_{\phi}}\left[\sum_{t} \nabla_{\phi}\log \pi_{\phi}\left(a_{t} \mid s_{t}\right) R(\tau) \right] \\
	&= \mathbb{E}_{\tau}\left[\sum_{t} \nabla_{\phi} \log \pi_\phi\left(a_{t} \mid s_{t}\right) R_{<t}\right]  + \mathbb{E}_{\tau}\left[\sum_{t}\nabla_{\phi} \log \pi_\phi\left(a_{t} \mid s_{t}\right) G_{t} \right] \\
	&=\mathbb{E}_{\tau}\left[\sum_{t}\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, G_{t}\right].
	\end{aligned}
	$$
	> 下面证明第三行第一项 $\mathbb{E}_{\tau}\left[\sum_{t} \nabla_{\phi} \log \pi_\phi\left(a_{t} \mid s_{t}\right) R_{<t}\right]$ 期望为 0。考虑时刻 $t$，引入历史轨迹 $$H_t\doteq (s_0,a_0,r_0,\dots,s_{t-1},a_{t-1},r_{t-1},s_t)$$ 给定 $H_t$ 后 $R_{<t}$ 是常数，用塔式法则展开 $$\mathbb{E}_{\tau}\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, R_{<t}\right] =\mathbb{E}_{H_t}\left[
	R_{<t}\ \mathbb{E}_\tau\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ \big|\ H_t\right]\right]. $$ 给定 $s_t$ 后，内层唯一还在随机的量只有 $a_t$，因此有 
	$$\begin{aligned}
	\mathbb{E}_\tau\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ \big|\ H_t\right]
	&=\mathbb{E}_{a_t\sim \pi_\phi(\cdot|s_t)}\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ \big|\ H_t\right] \\
	&=\sum_{a}\pi_\phi(a\mid s_t)\nabla_\phi \log \pi_\phi(a\mid s_t) \\
	&=\sum_{a}\nabla_\phi \pi_\phi(a\mid s_t) =\nabla_\phi \sum_{a}\pi_\phi(a\mid s_t) =\nabla_\phi 1 =0
	\end{aligned}.
	$$ 直观理解，依 $\nabla_{\phi} J(\phi)$ 更新策略时，本质是在按加权的 $\nabla_{\phi}\log \pi_{\phi}\left(a_{t} \mid s_{t}\right)$ 方向调整策略，提高好动作的出现概率，降低坏动作的出现概率。此时乘上的系数决定了调整的力度，是 credit assignment 的体现。注意到权重 $R(\tau)$ 中包含的 **$R_{<t}$ 部分和 $\pi_{\phi}\left(a_{t} \mid s_{t}\right)$ 无关，因为它在选择 $a_t$ 之前就已经确定了，故其只会把调整力度随机放大/缩小，但不会稳定地偏向某个方向——平均下来贡献就是 0**，因此 $t$ 时刻的 $\mathbb{E}_\tau\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ \big|\ H_t\right]$ 就是一个不影响梯度期望，只增大方差的 baseline 项，应当将其减去
- 至此我们得到 $\nabla_{\phi} J(\phi)=\mathbb{E}_{\tau}\left[\sum_{t}\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, G_{t}\right]$，其实它就是 Policy-Gradient RL 理论分析中常见的策略梯度定理 $\nabla_{\phi} J(\phi) \propto \mathbb{E}_{\pi_{\phi}}\left[Q_{\pi_{\phi}}(s, a) \nabla_{\phi} \log \pi_{\phi}(a \mid s)\right]$ 的另一种写法。下面我们完成这个形式转换，核心就是两步：
	1. **把轨迹 RTG $G_t$ 换成 $(s_t,a_t)$ 条件化后的 $Q_{\pi_\phi}(s_t, a_t)$**：依定义有
		$$
		Q_{\pi_\phi}(s_t,a_t)\doteq \mathbb{E}\!\left[G_t\mid s_t,a_t;\pi_\phi\right].
		$$ 在任意时刻 $t$ 给定 $(s_t, a_t)$ 后，$\nabla_{\phi} \log \pi_{\phi}(a_t \mid s_t)$ 看作常数，有
		$$
		\begin{aligned}
		\mathbb{E}_{\tau\sim p_{\phi}}\!\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, G_{t}\right]
		&=\mathbb{E}_{(s_t,a_t)\sim p_{\phi}^t}\!\left[
		\mathbb{E}\!\left[\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, G_{t}\ \big|\ s_t,a_t;\pi_\phi\right]
		\right]\\
		&=\mathbb{E}_{(s_t,a_t)\sim p_{\phi}^t}\!\left[
		\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ \mathbb{E}\!\left[G_{t}\ \big|\ s_t,a_t;\pi_\phi\right]
		\right]\\
		&=\mathbb{E}_{(s_t,a_t)\sim p_{\phi}^t}\!\left[
		\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\ Q_{\pi_\phi}(s_t,a_t)
		\right].
		\end{aligned}
		$$ 将上式对 $t=0,1,2,...$ 求和，并利用期望的线性性，得到
		$$
		\begin{aligned}
		\nabla_{\phi} J(\phi)
		&=\mathbb{E}_{\tau\sim p_{\phi}}\!\left[\sum_t\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, G_{t}\right]\\
		&=\mathbb{E}_{\tau\sim p_{\phi}}\!\left[\sum_t\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, Q_{\pi_\phi}(s_t,a_t)\right].
		\end{aligned}
		$$
	2. **把“沿时间求和的轨迹期望”改写为“在状态-动作访问分布下的期望”**：注意到上式右端是对轨迹上各时刻 $(s_t,a_t)$ 的求和。为将其写成对 $(s,a)$ 的一次期望，定义策略 $\pi_\phi$ 的折扣状态-动作占用分布
		$$
		d_{\pi_\phi}(s,a)\doteq (1-\gamma)\sum_{t}\gamma^t\,p_{\phi}^t(s_t=s,a_t=a),
		$$ 其中 $\gamma\in(0,1)$ 为折扣因子。现在我们可以**换一种写法**，把 “沿时间的加权求和期望” 改写成 “在加权访问分布下的期望”，对任意可积函数 $f(s,a)$ 恒有
		$$
		\begin{aligned}
	\mathbb{E}_{\tau\sim p_{\phi}}\!\left[\sum_{t=0}^{\infty}\gamma^t f(s_t,a_t)\right]
	&=\sum_{t=0}^{\infty} \gamma^{t}\, \mathbb{E}_{(s_t,a_t)\sim p_{\phi}^t}\!\left[f(s_t,a_t)\right] \\
	&=\sum_{t=0}^{\infty} \gamma^{t}\, \mathbb{E}_{(s,a)\sim p_{\phi}^t}\!\left[f(s,a)\right] \\
	&=\frac{1}{1-\gamma}\ \mathbb{E}_{(s,a)\sim d_{\pi_\phi}}\!\left[f(s,a)\right],
	\end{aligned}
		$$ 令 $f(s,a)\doteq \nabla_{\phi}\log \pi_{\phi}(a\mid s)\,Q_{\pi_\phi}(s,a)$ 代入上式，可得
		$$
		\begin{aligned}
		\nabla_{\phi} J(\phi)
		&=\mathbb{E}_{\tau\sim p_{\phi}}\!\left[\sum_{t}\gamma^t\nabla_{\phi}\log \pi_{\phi}(a_t \mid s_t)\, Q_{\pi_\phi}(s_t,a_t)\right]\\
		&=\frac{1}{1-\gamma}\ \mathbb{E}_{(s,a)\sim d_{\pi_\phi}}\!\left[Q_{\pi_\phi}(s,a)\nabla_{\phi}\log \pi_{\phi}(a \mid s)\right].
		\end{aligned}
		$$ 忽略与 $\phi$ 无关的常数因子 $1/(1-\gamma)$，得到常见的策略梯度定理写法
		$$
		\nabla_{\phi} J(\phi)\propto \mathbb{E}_{(s,a)\sim d_{\pi_\phi}}\!\left[Q_{\pi_\phi}(s,a)\nabla_{\phi}\log \pi_{\phi}(a \mid s)\right].
		$$
- 关于策略梯度定理的详细证明可以参考 [RL 实践（5）—— 二维滚球环境【REINFORCE & Actor-Critic】](https://blog.csdn.net/wxc971231/article/details/131882224)
# 3 可微代理优化的内在机制
- 为了简化分析，设被优化参数为 $\theta$，用多维高斯分布 $p_\phi(\theta)=\mathcal{N}(\theta\mid \mu,\sigma^2 I)$ 作为代理分布，此时 $\phi = \{\mu,\sigma\}$，其中
	- **均值 $\mu$ 代表了当前搜索的“重心”**，算法会努力将 $\mu$ 移动到回报更高的区域
	- **标准差 $\sigma$ 是探索的体现**，$\sigma$ 越大采样的策略越多样化，$\sigma$ 越小策略越集中

- 首先计算代理分布关于 $\sigma$ 的偏导数。设参数维度为 $d$，有 $$p(\theta)=\left(2 \pi \sigma^{2}\right)^{-d / 2} \exp \left(-\frac{\|\theta-\mu\|^{2}}{2 \sigma^{2}}\right) \\ \log p(\theta)=-\frac{d}{2} \log (2 \pi)-d \log \sigma-\frac{\|\theta-\mu\|^{2}}{2 \sigma^{2}}$$ 对标量 $\sigma$ 求导，得到 $$\frac{\partial}{\partial \sigma} \log p(\theta)=-\frac{d}{\sigma}+\frac{\|\theta-\mu\|^{2}}{\sigma^{3}}$$ 通常使用**重参数化方法**对各向同性高斯分布进行采样，即 $\theta=\mu+\sigma \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$，用 $\|\theta-\mu\|^{2}=\sigma^{2}\|\epsilon\|^{2}$ 带入，有	$$\frac{\partial}{\partial \sigma} \log p(\theta)=-\frac{d}{\sigma}+\frac{\sigma^{2}\|\epsilon\|^{2}}{\sigma^{3}}=\frac{\|\epsilon\|^{2}-d}{\sigma}$$
- 使用可微代理优化方法，得到的关于 $\sigma$ 的梯度估计为
	$$
	\begin{aligned}
	\nabla_{\sigma} J(\phi)
	&=\mathbb{E}_{\theta \sim p_{\phi}}\left[(J(\theta)-b) \nabla_{\sigma} \log p_{\phi}(\theta)\right] \\
	&\approx \frac{1}{N} \sum_{i=1}^{N} \frac{\left\|\epsilon_{i}\right\|^{2}-d}{\sigma} \left(J_{i}-b\right)
	\end{aligned}
	$$ 其中 $J_i$ 是第 $i$ 次采样策略的回报，$b$ 是回报的基线（通常是平均回报），$\epsilon_i$ 是重参数化采样时引入的噪声。这个公式揭示了无梯度优化的内在机制：**该类算法天生就倾向于在无法探索（落入局部最优）时缩小探索范围。**
	- **当算法在局部最优附近获得正反馈时，算法会认为“当前重心附近已经能稳定获得较高回报”，于是倾向于收缩搜索分布、减少无谓的探索，在该区域进行更精细的局部搜索**：通常靠近均值 $\mu$ 的样本（即扰动幅度较小 $|\epsilon_i|^2<d$）更容易保持可行性并获得更高回报，因此其优势项 $(J_i-b)$ 往往为正。此时梯度中的 “距离信号项” $\frac{\left\|\epsilon_{i}\right\|^{2}-d}{\sigma}$ 为负，因此单个样本对 $\nabla_\sigma J(\phi)$ 的贡献 $\frac{|\epsilon_i|^2-d}{\sigma}(J_i-b)$ 为负，整体梯度倾向于推动 $\sigma$ 减小
	- **当算法因过度探索受到惩罚时，算法将外部区域视为“高风险、低收益”的区域，因而通过收缩 $\sigma$ 来降低采样到这些失败样本的概率，进一步固守当前相对安全的搜索半径**：通常远离均值的样本（即扰动幅度较大 $|\epsilon_i|^2>d$）更容易落入低回报区域，其回报低于基线使得 $(J_i-b)$ 通常为负。此时距离信号项 $\frac{|\epsilon_i|^2-d}{\sigma}$ 为正，于是乘积 $\frac{|\epsilon_i|^2-d}{\sigma}(J_i-b)$ 仍为负，梯度依然倾向于推动 $\sigma$ 减小
- 综上所述，LLM+RL中出现的 “方差坍塌” 等熵相关现象，并非RL算法失灵，而是经典的 **“探索不足导致陷入局部最优”**。具体而言，策略梯度 RL 作为一类可微代理优化方法，其天然性质会导致以下现象：
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6372ab7ba11f4dfda68819467f0391a9.png#pic_center =90%x)

	1. **算法进入局部最优时出现方差坍塌**：当 Agent 在策略空间中探索受困时，RL 会系统性地降低探索标准差，使策略分布锁定在当前模式上，体现为 LLM 输出熵崩溃，多样性下降
	2. **pass@1 提升，但 pass@k 下降**：当 Agent 在策略空间中探索受困时，RL 锁定在局部最优模式，使策略均值 $\mu$ 精细地调整到这个 “小山丘” 的顶峰并缩小标准差 $\sigma$，采样到 “峰顶” 附近的概率增大，从而提升了 pass@1 指标。然而，这也导致模型更难采样到远离这个局部最优的更优解，导致 pass@k 下降
	3. **优化后的 pass@1 未超过基础模型的 pass@k**：这揭示了局部最优解与全局最优解之间的鸿沟。RL 会系统性地推动模型深陷局部最优区域
- 以上分析都是建立在多维高斯代理分布上的，实际的 LLM 作为代理分布会复杂很多，比如其分布往往是多峰而非单峰的，且 LLM 本质在进行条件生成，其探索强度也无法用类似方差的一个全局标量进行描述。尽管如此，**以上分析的 “定性机制” 在很大范围内仍然成立**。回顾梯度公式 $$\nabla_{\phi} J=\mathbb{E}_{x \sim p_{\phi}}\left[(f(x)-b) \nabla_{\phi} \log p_{\phi}(x)\right]$$ 它表达的核心就是：**高回报样本的 log-prob 被增大，低回报样本的 log-prob 被减小**，只要策略更新本质上在做这件事，就会天然有一种 “把概率质量往少数高回报区域挤” 的趋势。REINFORCE、PPO 及其近似（GRPO 等）都是这个味道