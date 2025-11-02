- 参考：[Optimal Transport and Wasserstein Distance](https://www.stat.cmu.edu/~larry/=sml/Opt.pdf)
- ----
- 机器学习中我们常常要度量两个分布间的距离，常用的度量包括 KL 散度、JS 散度、总变差距离等。Wasserstein 距离又称推土机距离，是一种**基于最优传输思想的度量，其从几何角度衡量一个分布“变形”为另一个分布所需的最小代价**

@[toc]
# 1. Wasserstein 距离的定义
- `Wasserstein 距离`：记紧空间 $\Omega$ 上的所有概率分布组成的空间为 $\mathcal{P}(\Omega)$，设分布 $P,Q\in \mathcal{P}(\Omega)$，随机变量 $X,Y$ 分布服从分布 $P,Q$，设 $\mathcal{J}(P,Q)$ 表示随机向量 $(X,Y)$ 的所有联合分布，即任意联合分布 $J\in \mathcal{J}(P,Q)$ 的边缘分布为 $P,Q$，Wasserstein 距离定义为
	$$
	\begin{aligned}
	W_{p}(P, Q)
	&=\left(\inf _{J \in \mathcal{J}(P, Q)} \int_{\Omega \times \Omega}\|x-y\|^{p} d J(x, y)\right)^{\frac{1}{p}} \\
	&=\left(\inf _{J \in \mathcal{J}(P, Q)} \mathbb{E}_{(X,Y)\sim J}\| X-Y\|^p \right)^{\frac{1}{p}} \\
	\end{aligned}
	\tag{1}
	$$ 
- 直观来看，**Wasserstein 距离就是在所有把分布 $P$  变为 $Q$ 的联合分布（搬运方案）中，期望搬运代价的最小值**
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/253dffa09ca243f79bb94ac7b5738e56.png#pic_center =40%x)
	1. 如图所示，任意联合分布 $J$ 可以看作把分布 $P$ 转换为分布 $Q$ 的一种方式。**联合分布 $J$ 上任意一点 $J(x,y)$ 可以看作将来自 $P$ 的概率质量的一部分从位置 $x$ 搬运到 $Q$ 的位置 $y$ 的方式**，即图中红色线
	2. **积分 $\int_{\Omega \times \Omega}\|x-y\|^{p} d J(x, y)$ 可以理解为 “按照分配方案 $J$ 把分布 $P$ 搬运成分布 $Q$ 所需的平均搬运成本**，其中从 $x$ 到 $y$ 的 “搬运” 代价为 $\|x-y\|^{p}$，被搬运的概率质量为 $J(x,y)$
	3. 下确界符号 $\inf _{J \in \mathcal{J}(P, Q)}$ 表示**在所有可能的搬运方案 $\mathcal{J}$ 里选出那个搬运成本最小的**
	4. $p$ 是距离范数的指数，计算平均后还要开 $p$ 次方，保证距离的量纲和原变量一致
- 下面给出一些例子
	> 1. 一维离散分布例子：假设相邻两列的距离（搬运代价）为1，图中给出的两种搬运方案代价都最小，加权移动总量为 8，除以格子数量 14（把 P，Q转换为概率分布），Wasserstein 距离为 4/7
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f9af4b4d30046a98f1aa9aff96cde19.png#pic_center =80%x)
	> 2. 二维连续分布例子：按中间的灰色箭头移动每个点对应的概率密度，可以将蓝色分布转换为红色分布，Wasserstein 距离可以理解为最小化这些箭头的平均平方长度
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b9089d0535d149fdbba9d7a9f3e3e380.png)
# 2. Wasserstein 距离的优点
## 2.1 可处理支撑集不重叠的情况
- KL 散度、JS 散度、总变差距离等分布间距离度量大都需要分布间具有重叠的支撑集，**当两个分布的概率非零区域完全不重叠时，这些度量会变为固定值或无穷值，导致度量失效**，难以反映实际差异
	> ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c5480009962548339cb70efab5f561ca.png#pic_center =80%x)
	如图例所示，以上三个分布支撑集都不重叠，大部分度量会认为它们两两之间距离相同，但直观来看 $p_1$ 距离 $p_2$ 比 $p_1$ 距离 $p_3$ 更近
- Wasserstein 距离**从几何角度衡量**一个分布“变形”为另一个分布所需的最小代价，因而可以有效处理支撑集（概率非0区域）不重叠的情况
## 2.2 Wasserstein 平均维持了原始分布的形态特征
- 给定一组分布 $P_1, ..., P_N$，有时我们想找到一个平均分布 $\bar{P}$ 来代表它们。下图给出了 50 个圆形二维分布，它们的定义域都是 $(0,0)$ 和 $(1,1)$ 围成的单位方格内，概率在圆形支撑集上均匀分布
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07b0c55e40994fcab1171cc84b6a4f0c.png#pic_center =60%x)
	有两种方式可以计算代表这 50 个分布的平均分布，Wasserstein 平均更好地维持了原始分布的形态特征
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a17b0a253584d4589040e8727c73c03.png#pic_center =60%x)
	1. 欧式平均（左）：直接把概率平均分配到所有分布的支撑集上，对密度值取算术平均
		$$
		\bar{p}(x) = \frac{1}{N}\sum_{j=1}^Np_j(x)
		$$
	2. Wasserstein 平均（右）：找一个分布，使得它到各个样本分布的 “最优搬运代价” 之和最小
		$$
		\bar{P} = \argmin_{P} \sum_{j=1}^N W(P,P_j)
		$$
	
- 从某种程度上来说，**Wasserstein 平均实际上是在进行分位数的转换和平均**，以下显示了更多例子
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2cdbf4be2dd842c9bfe0a2f880c4a38c.png#pic_center =80%x)


## 2.3 Wasserstein 距离反映了分布的转化过程
- 使用 KL 散度等方式度量分布 $P,Q$ 间差距时，我们仅得到一个数字。**使用 Wasserstein距离度量时，我们不但得到数字，还得到了一张图，展示了如何移动 $P$ 的概率质量来将其变形为 $Q$**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec1fdcbf3b444d93badf55afbeac6614.png#pic_center =40%x)
- 第一节中的二维连续例子也体现了这种能力
## 2.4 基于 Wasserstein 距离定义测地线可在变换过程中保存分布的形态结构
- 有时我们想看清从分布 $P_0$ 变换为 $P_1$ 的过程，也就是说在两个分布之间构造一个**某种距离度量下的最短路径**，沿着它走就能从原始分布变化为目标分布，这种路径需通过 “测地线” 进行构造
- **`测地线Geodesics` 本质是一个从 $[0,1]$ 区间到中间分布集合的映射 $c$**，满足 $c(0)=P_0, c(1)=P_1$，形成过程分布集合 $\{P_t\}_{t=0}^1$。给定分布间距度量 $d$，任意两个相邻分布间距离为 $d(c(t-1),c(t))$，测地线的总长度定义为路径无限细分时的相邻分布间距之和
	$$
	L(c)=\sup _{m, 0=t_{0}<t_{1}<\cdots<t_{m}=1} \sum_{i=1}^{m} W_{p}\left(c\left(t_{i-1}\right), c\left(t_{i}\right)\right)
	$$ 式子中上确界 $\sup$ 表示无限细分，**测地线是能够使 $L(c)$ 最小化的映射 $c$**
- 测地线是分布空间中的 “直线”（最短），满足线性关系，在欧式空间中直接插值就能得到测地线，即令 $$P_t=(1-t)P_0+tP_1$$ 它对应的距离度量是相应欧式空间中的距离度量，比如 $L_1$ 或 $L_2$ 范数。这种测地线构造方式和 2.2 节具有相同问题，不同位置的概率质量直接叠加，导致“混合”、“模糊”、“重影”，无法保持分布的形态结构和空间连续性，如下所示
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2dd7c346414648eda8dbecd56e49774e.png#pic_center =85%x)
- 在 Wasserstein 框架下，我们有最优传输映射 $$T^{*}: \Omega \rightarrow \Omega, \quad \text { s.t. } \quad T^{*} \# P_{0}=P_{1}$$ 所以**以 Wasserstein 度量作为距离度量的测地线，可以理解为在最优传输过程中进行插值**，即 $$T_{t}(x)=(1-t) x+t T^{*}(x), \quad P_{t}=T_{t} \# P_{0} $$ 由于它不是在密度函数上插值，而是在 “概率质量的流动” 层面定义路径，**每个概率质量微元沿最优方向运动，所以形状不会被混合或模糊化**，如下所示
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1f03385934c24a0583282ad554767442.png#pic_center =85%x)
- 下面给出一个更直观的例子，把字母 J 的图像变成 V 的图像，**基于 Wasserstein 距离定义测地线可在连续的变化过程中保存结构**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5332ddfccd504eb69732d6036b694f2d.png#pic_center =85%x)
# 3. 一维 Wasserstein 距离
## 3.1 一维 Wasserstein 距离可通过累积分布函数（CDF）计算
- 根据定义，Wasserstein 距离涉及一个下确界的优化问题
	$$
	W_{p}(P, Q)=\left(\inf _{J \in \mathcal{J}(P, Q)} \int_{\Omega \times \Omega}\|x-y\|^{p} d J(x, y)\right)^{\frac{1}{p}}
	$$ 这个优化在高维很复杂，但**在一维情况下有一个重大简化：最优传输映射是单调递增的**，这意味着从 $P$ 到 $Q$ 的最优传输可以如下计算 
	$$
	T^{*}(x)=F_{Q}^{-1}\left(F_{P}(x)\right)
	$$ 其中 $F$ 表示累积分布函数（CDF），$F^{-1}$ 表示分位数函数（quantile function），一维 Wasserstein 距离可以如下计算
	$$
	W_{p}(P, Q)=\left(\int_{0}^{1}\left|F_{P}^{-1}(u)-F_{Q}^{-1}(u)\right|^{p} d u\right)^{1 / p}
	\tag{2}
	$$ 
	1. 每一个分位数点 $u$ 对应一小块概率质量 $d_u$，$P,Q$ 分布的 $u$ 分位点 $x_P=F^{-1}_P(u)$ 和 $x_Q=F^{-1}_Q(u)$ 给出了这小块质量在两个分布间的位置
	2. 这两个分位点的距离 $|x_P-x_Q|^p$ 代表把小块概率质量 $du$ 从 $P$ 搬运到 $Q$ 的代价
	3. 对 $u$ 积分就得到总代价
- 因此，**只要知道两个一维分布的分位数函数，就能通过积分或数值求和计算出 Wasserstein 距离**。直观上看，就是**从左到右比较两个分布的累积概率质量，把两个分布的质量“从左到右”单调地匹配起来**
	> 用第 1 节的一维离散例子进行说明，首先给出原始分布和目标分布
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9fdc83d1f7ed434d8e4cf1f232e08dcf.png#pic_center =60%x)
	> 计算两个分布的 CDF
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bca3043c65d4bc8a92269caa967e9d3.png#pic_center =60%x)
	> 计算 Wasserstein 距离（除以总方格数 14 变为概率）
		$$
		\begin{aligned}
		W_1(Q,P)&=\frac{\sum_{i=0}^{4}\left|C D F\left(Q_{i}\right)-C D F\left(P_{i}\right)\right|}{14} \\
		&=\frac{|2-3|+|5-8|+|9-10|+|14-11|+|14-14|}{14} \\
		&=\frac{1+3+1+3}{14}=\frac{8}{14}=0.5714
		\end{aligned}
		$$
## 3.2 一维 Wasserstein 距离和 CRPS 损失的等价性
### 3.2.1 学习直方图分布
- 考虑回归问题，有时我们希望模型不仅能输出一个点回归值，还能给出分布形状。为此可以先将数值归一化到指定区间上，再将其划分为 $N$ 个 bin，假设模型输出一个在区间上离散化的概率分布（直方图分布）
	$$
	\hat{p}_k = \text{Pr}(y\in \text{bin}_k)
	$$ 真实标签是一个确定性数值 $y^*$，它会落在某个 bin 中
- 解决这个问题的一个直观思路是将其看作分类问题，为 $y^*$ 落入的 bin 构造 one-hot 标签，再用交叉熵损失优化。这种方式虽然能进行学习，但 CE loss 的问题在于缺少数值敏感性，预测偏一格和偏十格损失一样大，最后学出的分布形状往往不平滑
### 3.2.2 CRPS 损失
- 使用 `CRPS(Continuous Ranked Probability Score)` 损失训练直方图分布使用了回归思想，该损失可以**度量预测分布与真实 CDF 的整体差距**，设模型预测分布为 $P$，CDF 为 $F_P$，真实标签为 $y^*$，定义如下
	$$
	\text{CRPS}(F_P,y^*) = \int_{-\infin}^{+\infin} (F_P(y) - \mathbb{1}\{y > y^*\})^2 dy
	$$
	1. $F_P(y)$ 是模型预测真实标签 $\leq y$ 的概率
	2. $\mathbb{1}\{y > y^*\}$ 是样本对应的真实 CDF（阶跃函数）
	3. 用二者之差的平方在全域上的积分衡量预测分布与真实点分布的差距
-  CRPS 的离散版本直接适用于优化直方图分布，这也是实现中最常用的版本。这里我们假设数值区间离散为 $N$ 个有序的 bin：$b_1<b_2<\dots < b_N$，模型预测 CDF 记为 $F_P^i=\sum_{k=1}^i\hat{p}_k$。设真实值 $y^*$ 落在第 $k^*$ 个 bin 中，真实 CDF 为 $H_k=\mathbb{1}\{k\geq k^*\}$，离散 CRPS 定义为
	$$
	\text{CRPS}(F_P,y^*) = \sum_{k=1}^N(F_P^k-\mathbb{1}\{k\geq k^*\})^2
	$$
### 3.2.3 CRPS 和 Wasserstein 距离
- 从 Wasserstein 距离角度分析训练任务，对于每个训练样本，模型预测分布 $P$，对应CDF 和对应的分位数函数为 $F_P(y) 和 F_P^{-1}(u)$，真实分布 $Q(y)=\mathbb{1}\{y\geq y^*\}$ 是由样本决定的在 $y^*$ 处取概率为 1 的狄利克雷函数，其 CDF 和对应的分位数函数为
	$$
	\begin{aligned}
	F_Q(y) &= 
	\left\{\begin{array}{ll}
	0, & y<y^{*} \\
	1, & y \geq y^{*}
	\end{array}\right. \\
	F_Q^{-1}(u) &= y^*, \quad \forall u\in (0,1)
	\end{aligned}
	$$ 带入 Wasserstein 距离定义有（$p=1$）
	$$
	\begin{aligned}
	W_1(P,Q)&= \int_{0}^{1}\left|F_{P}^{-1}(u)-F_{Q}^{-1}(u)\right| d u \\
	&=\int_{0}^{1}\left|F^{-1}_P(u)-y^{*}\right| d u \\
	&=\mathbb{E}_{Y \sim P}\left|Y-y^{*}\right|
	\end{aligned}
	\tag{3}
	$$ 从直觉上看，真实分布 $Q$ 的所有概率质量都堆在一点上，**要把预测分布的质量 “搬运” 到这一点，成本就是预测值到 $y^*$ 的绝对距离的期望**
- 为分析 CRPS loss，要引用一个关于CDF差平方的积分恒等式 [Gneiting, T., & Raftery, A. E. (2007).](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)
	$$
	\int_{-\infty}^{+\infty}\left(F_{P}(y)-F_{Q}(y)\right)^{2} d y=\frac{1}{2} \mathbb{E}_{Y, Y^{\prime} \sim P}\left|Y-Y^{\prime}\right|+\frac{1}{2} \mathbb{E}_{Z, Z^{\prime} \sim Q}\left|Z-Z^{\prime}\right|-\mathbb{E}_{Y \sim P, Z \sim Q}|Y-Z|
	$$ 把 $Q(y)=\mathbb{1}\{y\geq y^*\}$ 带入，
	$$
	\begin{aligned}
	\text{CRPS}(F_P,y^*) 
	&= \int_{-\infin}^{+\infin} (F_P(y) - F_{Q})^2 dy \\
	&= \mathbb{E}_{Y \sim P}\left|Y-y^{*}\right|-\frac{1}{2} \mathbb{E}_{Y, Y^{\prime} \sim P}\left|Y-Y^{\prime}\right| \\
	&= W_1(P,Q) - \frac{1}{2} \mathbb{E}_{Y, Y^{\prime} \sim P}\left|Y-Y^{\prime}\right|
	\end{aligned}
	$$
 注意到 CRPS loss 和 $W_1$ 距离只差了一个 $\frac{1}{2} \mathbb{E}_{Y, Y^{\prime} \sim P}\left|Y-Y^{\prime}\right|$，这一项可以看作**对模型分布自身的内部散度的度量**，直觉上
 	1. $W_1(P,Q)$ 衡量模型分布质量与目标单点分布的**总体偏移量**
 	2. CRPS 不希望模型只通过 “拉宽分布” 来掩盖误差（这会增加不确定性但降低平均偏移），减去 $\frac{1}{2} \mathbb{E}_{Y, Y^{\prime} \sim P}\left|Y-Y^{\prime}\right|$ 在鼓励模型接近 $y^*$ 分布的同时保持集中（sharpness）
- 综上，当**真实分布 $P$ 非常锐利或为点分布时（比如对数学计算结果进行回归），CRPS 和 Wasserstein 距离等价；当真实分布 $P$ 有方差时，CRPS 比 Wasserstein 更 “保守”**
### 3.2.4 小结
- 针对直方图分布学习，对比 CE loss 和 CRPS loss 
	| 性质        | CE      | CRPS             |
	| --------- | --------------------- | --------------------------------------------- |
	| 基本思想      | 最大化真 bin 的概率（分类思维）    | 度量预测分布与真实 CDF 的整体差距（回归思维）                     |
	| 标签形式        | one-hot 向量    | 阶跃 CDF                                     |
	| 损失形态      | $-\log \hat{p}_{y^*}$ | $\sum_i (F_i - \mathbf{1}\{i \ge y^*\})^2$      |
	| 距离度量      | KL散度（非对称）             | 一维Wasserstein距离的变体（对称且连续）                     |
	| 数值敏感性 | 不敏感，预测偏一格和偏十格损失一样大    | 敏感，预测离得越远惩罚越大                                 |
	| 输出解释      | “分类概率”                | “概率预测的累积分布”                                   |
- 注意到 CRPS 的一个重大优势是其具备数值敏感性，因此在 LLM 数值回归论文 [NTL](https://blog.csdn.net/wxc971231/article/details/149466344) 中得到应用