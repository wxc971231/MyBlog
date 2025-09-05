---
title: 经典机器学习方法(2)——Softmax 回归
date: 2025-09-05 19:35:20
index_img: /img/经典机器学习方法(2)_Softmax回归/index_img.png
tags:
  - softmax回归
  - pytorch
  - 机器学习
categories:
  - 实践
description: 本文详细介绍了softmax回归模型,包括其原理、计算表达式、交叉熵损失函数和模型训练。通过实例展示了如何在PyTorch中实现softmax回归,以及训练过程和模型评估。此外,还探讨了为何在分类问题中使用交叉熵损失而非MSE损失的原因。
---

- 参考：动手学深度学习
- 注：本文是 jupyter notebook 文档转换而来，部分代码可能无法直接复制运行！
- -----
- 前文介绍的 [经典机器学习方法（1）—— 线性回归](https://blog.csdn.net/wxc971231/article/details/122869916) 适用于连续值预测问题（回归问题），本文介绍适用于离散值预测（分类问题）的 softmax 回归模型，这是一种基于神经网络的经典**分类模型**
- softmax 回归和线性回归**内部一样是线性模型**，区别在于
    1. softmax 回归的输出从一个变成了多个
    2. softmax 回归中引入了 softmax 运算，使其更适合于离散值的预测和训练


# 1. softmax 回归原理
## 1.1 分类问题 
- 考虑以下简单的分类问题
    1. 输入：$2\times 2$ 尺寸的图像 $\pmb{x}$，四个像素记为 $x_1,x_2,x_3,x_4$
    2. 输出：预测标记 $y \in\mathcal{Y}$，其中 $\mathcal{Y} = \{y_1,y_2,y_3\}$ 是大小为 3 的输出空间。我们习惯使用离散的数值来表示类别，比如将其处理为 $y_1=1，y_2=2,y_3=3$，这样需要输出 1，2，3 这三个数字中的一个
- 如果向上面那样简单地使用数值化的标记，仍然可以使用回归模型来处理，将预测值就近离散化到 1、2、3 这三个值即可。但是有两个问题
	1. 数值化标记有**距离关系**，1 和 2 比较接近，1 和 3 比较远，但事实上没有这种关系
	2. **连续值到离散值的转化通常会影响分类质量**

	所以一般采用专门针对离散值输出的分类模型来解决分类问题，做两个变化
	1. 样本标记使用 one-hot 向量形式
	2. 模型输出加一个 softmax 函数，得到概率性的综合 one-hot 预测值 
 
 
## 1.2 softmax 回归模型
- softmax 回归模型内部和线性回归模型几乎一致，也是一个简单的**单层全连接神经网络**，只是在**输出层增加了节点，以获得 $|\mathcal{Y}|$ 个输出**（构成 one-hot 向量），以 1.1 节的 4 维输入（特征维度为4） 3 维输出（类别总数为3）为例
	![在这里插入图片描述](/MyBlog/img/经典机器学习方法(2)_Softmax回归/img_001.png)

	其中每个输出层节点都是输入的线性组合，即
    $$
    \begin{aligned}
    &o_1 = x_1w_{11}+x_2w_{21}+x_3w_{31}+x_4w_{41} + b_1\\
    &o_2 = x_1w_{12}+x_2w_{22}+x_3w_{32}+x_4w_{42} + b_2\\
    &o_3 = x_1w_{13}+x_2w_{23}+x_3w_{33}+x_4w_{43} + b_3\\
    \end{aligned}
	$$
- 为了得到离散的 one-hot 形式的预测输出，把输出值 $o_i$ 看作置信度，输出越大的节点，对应的标记越可能是真实标签。**使用 softmax 运算将其输出值转换为正且和为 1 的概率分布**，即
    $$
    \hat{y}_1,\hat{y}_2,\hat{y}_3 = \text{softmax}(o_1,o_2,o_3)
    $$
    其中 $\hat{y}_i$ 是如下计算的
    $$
    \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
    $$
    考察 softmax 操作的性质
   	1. $\arg\max_i o_i = \arg\max_i \hat{y}_i$，因此 **softmax 运算不改变预测类别输出**
   	2. 由于进行了 $\text{exp}$ 变换，softmax 会使**数值之间的相对差距**放大 
   	3. softmax 操作**不改变向量尺寸**，输出的 $[\hat{y}_1,\hat{y}_2,\hat{y}_3]^\top$ 即为预测 one-hot 向量
- softmax 模块示意图如下
	![在这里插入图片描述](/MyBlog/img/经典机器学习方法(2)_Softmax回归/img_002.png)

- 考虑二分类问题的特殊情况，这时输出只有两个，设两组参数为 $\pmb{w_1},b_1$ 和 $\pmb{w_2},b_2$，给定样本 $\pmb{x}$，输出 $o_1$ 如下，可见变成了 sigmoid 函数的形式，也就是说：**对于二分类问题，softmax 回归等价于 logistic 回归（逻辑回归/对数几率回归）**
    $$
    \begin{aligned}
    o_1
    &= \frac{\text{exp}(\pmb{w}_1^\top\pmb{x}+b_1)}{\text{exp}(\pmb{w}_1^\top\pmb{x}+b_1)+\text{exp}(\pmb{w}_2^\top\pmb{x}+b_2)} \\
    &= \frac{1}{1+\text{exp}((\pmb{w}_2-\pmb{w}_1)^\top\pmb{x}+(b_2-b_1))}
    \end{aligned}
    $$

### 1.2.1 单样本分类的矢量计算表达式

- 为了提升运算效率，将上述运算都改成矩阵形式，
  1. softmax 回归的权重和偏置参数为
        $$
        \pmb{W} =
        \begin{bmatrix}
        w_{11} & w_{12}   & w_{13} \\
        w_{21} & w_{22}   & w_{23} \\
        w_{31}  &w_{32}   & w_{33} \\
        w_{41}  &w_{42}   & w_{43} \\
        \end{bmatrix}\space\space\space
        \pmb{b} = [b_1,b_2,b_3]
        $$
  2. 第 $i$ 个样本特征为
        $$
        \pmb{x}^{(i)}_{1\times 4} = [x_1^{(i)},x_2^{(i)},x_3^{(i)},x_4^{(i)}]
        $$
  3. 输出层输出为
        $$
        \pmb{o}^{(i)}_{1\times 3} = [o_1^{(i)},o_2^{(i)},o_3^{(i)}]
        $$
  4. 预测概率分布为
    $$
    \pmb{\hat{y}}^{(i)}_{1\times 3} = [\pmb{\hat{y}}_1^{(i)},\pmb{\hat{y}}_2^{(i)},\pmb{\hat{y}}_3^{(i)}]
    $$
  5. 通常把预测概率最大的类别作为预测类别
    $$
    \hat{y} = \arg\max_{j}\pmb{\hat{y}}_j^{(i)}
    $$
    注意我们习惯使用离散的数值来表示类别，比如将其处理为 $y_1=1，y_2=2,y_3=3$，这样需要输出 1，2，3 这三个数字中的一个
  6. softmax 回归对样本 $\pmb{x}^{(i)}$ 进行的运算为
    $$
    \begin{aligned}
    &\pmb{o}^{(i)}_{1\times 3}  = \pmb{x}^{(i)}_{1\times 4}\pmb{W}_{4\times 3} +\pmb{b}_{1\times 3} \\
    &\pmb{\hat{y}}^{(i)}_{1\times 3} = \text{softmax}(\pmb{o}^{(i)}_{1\times 3})
    \end{aligned}
    $$


### 1.2.2 mini-batch 样本分类的矢量计算表达式
- 为了进一步提升计算效率，结合常用的 mini-batch 梯度下降优化算法，我们常常对小批量数据做矢量运算。设一个小批量样本批量大小为 $n$，输入特征个数为 $d$，输出个数（类别数为）$q$，则
    1. 批量样本特征为 $\pmb{X}\in\mathbb{R}^{n\times d}$
    2. 权重参数为 $\pmb{W} \in\mathbb{R}^{d\times q}$
    3. 偏置参数为 $\pmb{b}\in\mathbb{R}^{1\times q}$
- 矢量计算表达式为
    $$
    \begin{aligned}
    &\pmb{O}_{n\times q} = \pmb{X}_{n\times d}\pmb{W}_{d\times q}+\pmb{b}_{1\times q}\\
    &\pmb{\hat{Y}}_{n\times q} = \text{softmax}(\pmb{O}_{n\times q})
    \end{aligned}
    $$
    其中**加法使用了广播机制**，$\pmb{O},\hat{\pmb{Y}}\in\mathbb{R}^{n\times q}$ 且其中第 $i$ 行分别为样本 $i$ 的输出 $\pmb{o}^{(i)}$ 和概率分布 $\pmb{\hat{y}}^{(i)}$
  
## 1.3 交叉熵损失函数
- 对于某个样本 $\pmb{x}_i$，上面我们利用 softmax 运算得到看其预测标记分布 $\pmb{\hat{y}}^{(i)}$。另一方面，此样本的真实标记也可以用一个输出空间上的分布 $\pmb{y}^{(i)}$ 来表示
    > 比如样本只有一个标记时，可以构造一个 one-hot 向量 $\pmb{y}^{(i)}\in\mathbb{R}^q$，使其真实标记对应的向量元素设为 1，其他设为 0，从而将真实标记转换为一个输出空间上的分布
  
  这样我们的训练目标可以设为**使预测概率分布 $\pmb{\hat{y}}^{(i)}$ 尽量接近真实概率分布 $\pmb{y}^{(i)}$**
- 这里不适合使用线性回归的平方损失（MSE 损失） $||\hat{\pmb{y}}^{(i)}-\pmb{y}^{(i)}||^2/2$，因为想得到正确的预测分类结果，只要保证真实类别的预测概率最大即可，**平方损失函数要求所有可能类别的预测概率和真实概率都相等，这过于严格**
    > 假设真实标记是 $\pmb{\hat{y}}^{(i)}_3$，当 $\pmb{\hat{y}}^{(i)}_3$ 预测值为 0.6 时即可以保证一定预测正确，如果用平方损失，这时 $\pmb{\hat{y}}^{(i)}_1=\pmb{\hat{y}}^{(i)}_2=0.2$ 比 $\pmb{\hat{y}}^{(i)}_1=0$,$\pmb{\hat{y}}^{(i)}_2=0.4$ 的损失小很多，尽管二者有同样的分类结果
   
   下面引用一个李宏毅机器学习课程中的例子，左图和右图分别显示使用 MSE 损失和 Cross-entropy 损失导致的 error surface，**可见使用交叉熵时梯度比较陡峭，利于做优化；而 MSE 导致的梯度有很大的平坦区域，优化过程很可能卡住**（可能必须要用 Adam 等高级的优化方案）
![在这里插入图片描述](/MyBlog/img/经典机器学习方法(2)_Softmax回归/img_003.png)
   
   关于这个问题其实还有不少可讲的，请参考：[分类问题为什么用交叉熵损失不用 MSE 损失](https://blog.csdn.net/wxc971231/article/details/123866413)
- 我们可以使用**衡量两个分布间差异的测量函数**作为损失，交叉熵（cross entropy）是一个常用的选择，它将分布 $\pmb{y}^{(i)}$ 和 $\hat{\pmb{y}}^{(i)}$ 的差距表示为
    $$
    H(\pmb{y}^{(i)},\hat{\pmb{y}}^{(i)}) = -\sum_{j=1}^q y_j^{(i)}\log\hat{y}_j^{(i)}
    $$
    注意其中 $y_j^{(i)}$ 是真实标记分布 $\pmb{y}^{(i)}$ 中非零即一的元素，样本真实标记为 $y^{(i)}$，因此 $\pmb{y}^{(i)}$ 中只有   $y^{(i)}_{y^{(i)}}=1$，其他全为 0，因此上述交叉熵可以化简为
 
     $$
     H(\pmb{y}^{(i)},\hat{\pmb{y}}^{(i)}) =  -\log\hat{y}_{y^{(i)}}^{(i)}
     $$
     可见**最小化交叉熵损失等价于最大化对正确类别的预测概率，它只关心对正确类别的预测概率**，这是合理的，因为只要其值足够大就能保证分类正确
     > 当遇到一个多标签样本时，例如图像中含有不止一个物体时，不能做这样的简化，但是这种情况下交叉熵损失也仅仅关心图像中出现的物体类别的预测概率
 
 
- 假设训练数据样本量为 $n$，交叉熵损失函数定义为
    $$
    \mathscr{l}(\Theta) = \frac{1}{n}\sum_{i=1}^n  H(\pmb{y}^{(i)},\hat{\pmb{y}}^{(i)})
    $$
    其中 $\Theta$ 是模型参数，如果每个样本只有一个标签，则上述损失可以化简为
    $$
    \mathscr{l}(\Theta) = \frac{1}{n}\sum_{i=1}^n  -\log\hat{y}_{y^{(i)}}^{(i)}
    $$
    最小化 $\mathscr{l}(\Theta)$ 等价于最大化 $\exp(-n\mathscr{l}(\Theta)) = \prod_{i=1}^n \hat{y}_{y^{(i)}}^{(i)}$，**即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率**
    
## 1.4 模型预测与评价
- 训练好 softmax 回归模型后，给定任一样本特征，就可以预测每个输出类别的概率
- 通常把预测概率最大的类别作为输出类别，如果它与真实类别（标签）一致，说明这次预测是正确
- 对于分类问题，可以使用 `准确率accuracy` 来评价模型的表现，它等于正确预测数量与总预测数量之比

# 2. 实现 softmax 回归

## 2.1 数据准备
- 使用 Fashion-MNIST 图像分类数据集进行试验，该数据集可以使用 `torchvision.datasets` 方便地获取和使用，具体请参考：[在 pytorch 中加载和使用图像分类数据集 Fashion-MNIST](https://blog.csdn.net/wxc971231/article/details/124347267)

- 先定义好读取小批量数据的方法，构造数据读取迭代器
	```python
	import torch
	import torchvision
	import torchvision.transforms as transforms
	import numpy as np
	
	def load_data_fashion_mnist(batch_size, num_workers=0, root='./Datasets/FashionMNIST'):
	    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,transform=transforms.ToTensor())
	    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,transform=transforms.ToTensor())
	
	    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	
	    return train_iter, test_iter
	
	# 数据读取迭代器
	batch_size = 256
	train_iter, test_iter = load_data_fashion_mnist(batch_size, 4)
	```
## 2.2 模型设计
- **模型参数初始化**：输入图像样本尺寸均为 28x28，拉平后输入向量长度为 $28\times 28 = 784$；由于图像有 10 个类别，输出层输出向量尺寸为 10，因此权重参数尺寸为 $\pmb{W}_{728\times 10}$，偏置参数尺寸为 $\pmb{b}_{1\times 10}$，如下初始化
    1. $w_{ij}\sim N(0,0.01^2),\space i=1,2,...,728;j=1,2,...,10$ 
    2. $b_i=0,\space i=1,2,...,10$
     
  注意设置属性 `requires_grad = True`，这样在后续训练过程中才能对这些参数求梯度并迭代更新参数值

	```python
	# 初始化模型参数
	num_inputs = 28*28   # 图像尺寸28x28，拉平后向量长度为 28*28
	num_outputs = 10     # 10个类别
	
	W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True) 
	b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
	```
- **实现 softmax 运算**：下面实参 O 是行数为样本数，列数为输出个数的矩阵，即 $\pmb{O}_{n\times 10} = \pmb{X}_{n\times 728}\pmb{W}_{728\times 10}+\pmb{b}_{1\times 10}$ （最后加向量 $\pmb{b}$ 使用了广播机制）。首先调用 `.exp()` 对矩阵中所有元素求 exp 函数值，然后按列求和得到 $728\times 1$ 的中间向量，最后利用广播机制将每一个元素除以其所在行的中间元素。**这样处理后得到的矩阵每行元素和为1且非负，成为合法的概率分布，代表各个样本在各个输出类别上的预测概率**

	```python
	def softmax(O):
	    O_exp = O.exp()                             # 所有元素求 exp
	    partition = O_exp.sum(dim=1, keepdim=True)  # 对列求和
	    return O_exp / partition                    # 这里应用了广播机制
	```
	测试一下，假设类别数为 5，样本数为 2

	```python
	# 对于任意网络输出，softmax 将每个元素变成了非负数，且每一行和为1，这样就能看做将样本预测为各个类别的概率
	output = torch.rand((2, 5))      # 随机生成网络输出层各结点值
	y_hat = softmax(output)          # 用 softmax 转换为预测概率分布
	print(y_hat, y_hat.sum(dim=1))   
	
	'''
	tensor([[0.3455, 0.1642, 0.1698, 0.1335, 0.1870],
	        [0.1873, 0.2624, 0.2497, 0.1375, 0.1631]]) tensor([1., 1.])
	'''
	
	# 根据样本真实标签获取预测概率时，可以使用 tensor.gather(dim, indexs) 方法
	# 该方法在dim维度上按indexs索引一个和indexs维度相同大小的tensor
	y = torch.tensor([0,2])                 # 假设有两个样本真实标签为 0 和 2
	print(y_hat.gather(1, y.view(-1, 1)))   # 获取这两个样本预测为相应的真实标签的概率
	
	'''
	tensor([[0.3455],
	        [0.2497]])
	'''
	```
- 定义模型 $\pmb{\hat{Y}} = \text{softmax}(\pmb{X}\pmb{W}+\pmb{b})$

	```python
	def net(X):
	    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
	```
- 定义损失函数 $H(\pmb{y}^{(i)},\hat{\pmb{y}}^{(i)}) =  -\log\hat{y}_{y^{(i)}}^{(i)}$

	```python
	def cross_entropy(y_hat, y):
	    return - torch.log(y_hat.gather(1, y.view(-1, 1))) # 这里返回 n 个样本各自的损失，是 nx1 向量
	```
## 2.3 模型评价
- 使用**分类准确率**评价分类模型的性能，假设有 q 个类别 n 个预测样本，和 2.2 节中一样记真实标记向量为 `y `（尺寸 `torch.Size([n])`），模型输出为 `y_hat` （尺寸 `torch.Size([n, q])`）

	1. 计算**一批样本**的预测准确率：
	    1. `y_hat.argmax(dim=1)` 获取所有样本的预测标签，尺寸为 `torch.Size([n])`
	    2. `y_hat.argmax(dim=1) == y` 和样本真实标签比较，得到 bool 型 tensor，尺寸为 `torch.Size([n])` 
	    3. `(y_hat.argmax(dim=1) == y).float()` 把 bool 型 tensor 转为取值 0 或 1 的浮点型 tensor，尺寸为 `torch.Size([n])` 
	    4. `(y_hat.argmax(dim=1) == y).float().mean()` 计算均值得到准确率，返回尺寸为 `torch.Size([])` 的浮点型 tensor
	    5. `(y_hat.argmax(dim=1) == y).float().mean().item()` 将上面这种只有一个元素的 tensor 转为 python 标量
	   
		```python
		# 计算一批样本的预测准确率
		def accuracy(y_hat, y):
		    return (y_hat.argmax(dim=1) == y).float().mean().item()
		```
	2.  计算**整个训练集/测试集**上的分类准确率：利用前面定义的数据获取迭代器遍历数据集，计算所有样本准确率的均值，从而评估模型 `net` 在整个数据集上的准确率，如下
		```python
		def evaluate_accuracy(data_iter, net):
		    acc_sum = 0.0  # 所有样本总准确率
		    n =  0         # 总样本数量
		    for X, y in data_iter:
		        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() # 注意这里中间的 mean() 改成 sum()
		        n += y.shape[0]
		    return acc_sum / n
		```
	编程实践中，通常直接使用第二种方式

## 2.4 模型训练
### 2.4.1 优化算法
- 使用小批量随机梯度下降来优化参数
	```python
	# 小批量随机梯度下降
	def sgd(params, lr, batch_size):
	    for param in params:
	        param.data -= lr * param.grad / batch_size # 注意这里更改 param 时用的param.data，这样不会影响梯度计算
	```
### 2.4.2 训练流程
- 训练流程和线性回归类似
	1. 设定超参数 `num_epochs`（迭代次数）和 `lr`（学习率）
	2. 在每轮迭代中逐小批次地遍历训练集，计算损失 -> 对参数求梯度 -> 做小批量随机梯度下降优化参数
- 训练程序如下
	```python
	# 超参数
	num_epochs, lr = 5, 0.1
	
	def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None):
	    # 训练执行 num_epochs 轮
	    for epoch in range(num_epochs):
	        train_l_sum = 0.0    # 本 epoch 总损失
	        train_acc_sum = 0.0  # 本 epoch 总准确率
	        n = 0                # 本 epoch 总样本数
	        
	        # 逐小批次地遍历训练数据
	        for X, y in train_iter:
	            
	            # 计算小批量损失
	            y_hat = net(X)
	            l = loss(y_hat, y).sum()  
	
	            # 梯度清零
	            if params is not None and params[0].grad is not None:
	                for param in params:
	                    param.grad.data.zero_()
	        
	            # 小批量的损失对模型参数求梯度
	            l.backward()
	            
	            # 做小批量随机梯度下降进行优化
	            sgd(params, lr, batch_size)   # 手动实现优化算法
	 
	            # 记录训练数据
	            train_l_sum += l.item()
	            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
	            n += y.shape[0]
	        
	        # 训练完成一个 epoch 后，评估测试集上的准确率
	        test_acc = evaluate_accuracy(test_iter, net)
	        
	        # 打印提示信息
	        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
	              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
	
	        
	# 进行训练
	train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
	
	'''
	epoch 1, loss 0.4741, train acc 0.840, test acc 0.827
	epoch 2, loss 0.4649, train acc 0.842, test acc 0.832
	epoch 3, loss 0.4579, train acc 0.845, test acc 0.833
	epoch 4, loss 0.4520, train acc 0.847, test acc 0.835
	epoch 5, loss 0.4463, train acc 0.849, test acc 0.830
	'''
	```
## 2.5 使用模型进行预测
- 训练完成后就可以用模型对测试图像进行分类了，先定义一些显示结果使用的工具函数
	```python
	from IPython import display
	import matplotlib.pyplot as plt
	
	def get_fashion_mnist_labels(labels):
	    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
	                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	    return [text_labels[int(i)] for i in labels]
	
	def show_fashion_mnist(images, labels):
	    display.set_matplotlib_formats('svg')
	    
	    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
	    for f, img, lbl in zip(figs, images, labels):
	        f.imshow(img.view((28, 28)).numpy())
	        f.set_title(lbl)
	        f.axes.get_xaxis().set_visible(False)
	        f.axes.get_yaxis().set_visible(False)
	```
- 下面给定一系列图像，真实标签和模型预测结果分别显示在第一和第二行

	```python
	X, y = iter(test_iter).next()
	
	true_labels = get_fashion_mnist_labels(y.numpy())
	pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
	titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
	
	show_fashion_mnist(X[0:9], titles[0:9])
	
	```
	预测结果如下
![在这里插入图片描述](/MyBlog/img/经典机器学习方法(2)_Softmax回归/img_004.png)
## 2.6 完整代码
- 整合上述过程，给出完整代码，可以直接粘贴进 vscode 运行
	```python
	import torch
	import torchvision
	import torchvision.transforms as transforms
	import numpy as np
	from IPython import display
	import matplotlib.pyplot as plt
	
	# 数据集相关 --------------------------------------------------------------------------------------------------
	# 加载数据集
	def load_data_fashion_mnist(batch_size, num_workers=0, root='./Datasets/FashionMNIST'):
	    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,transform=transforms.ToTensor())
	    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,transform=transforms.ToTensor())
	
	    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	
	    return train_iter, test_iter
	
	# 数据集标签转换
	def get_fashion_mnist_labels(labels):
	    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
	                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	    return [text_labels[int(i)] for i in labels]
	
	# 显示数据图片
	def show_fashion_mnist(images, labels):
	    display.set_matplotlib_formats('svg')
	    
	    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
	    for f, img, lbl in zip(figs, images, labels):
	        f.imshow(img.view((28, 28)).numpy())
	        f.set_title(lbl)
	        f.axes.get_xaxis().set_visible(False)
	        f.axes.get_yaxis().set_visible(False)
	
	
	# 模型定义 --------------------------------------------------------------------------------------------------------
	def softmax(O):
	    O_exp = O.exp()                             # 所有元素求 exp
	    partition = O_exp.sum(dim=1, keepdim=True)  # 对列求和
	    return O_exp / partition                    # 这里应用了广播机制
	
	# 模型定义
	def net(X):
	    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
	
	# 交叉熵损失
	def cross_entropy(y_hat, y):
	    return - torch.log(y_hat.gather(1, y.view(-1, 1)))  # 这里返回 n 个样本各自的损失，是 nx1 向量
	
	# 优化方法：小批量随机梯度下降
	def sgd(params, lr, batch_size):
	    for param in params:
	        param.data -= lr * param.grad / batch_size      # 注意这里更改 param 时用的param.data，这样不会影响梯度计算
	
	# 准确率评估
	def evaluate_accuracy(data_iter, net):
	    acc_sum = 0.0  # 所有样本总准确率
	    n =  0         # 总样本数量
	    for X, y in data_iter:
	        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
	        n += y.shape[0]
	    return acc_sum / n
	
	# 模型训练 --------------------------------------------------------------------------------------------------------
	def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None):
	    # 训练执行 num_epochs 轮
	    for epoch in range(num_epochs):
	        train_l_sum = 0.0    # 本 epoch 总损失
	        train_acc_sum = 0.0  # 本 epoch 总准确率
	        n = 0                # 本 epoch 总样本数
	        
	        # 逐小批次地遍历训练数据
	        for X, y in train_iter:
	            
	            # 计算小批量损失
	            y_hat = net(X)
	            l = loss(y_hat, y).sum()  
	
	            # 梯度清零
	            if params is not None and params[0].grad is not None:
	                for param in params:
	                    param.grad.data.zero_()
	        
	            # 小批量的损失对模型参数求梯度
	            l.backward()
	            
	            # 做小批量随机梯度下降进行优化
	            sgd(params, lr, batch_size)   # 手动实现优化算法
	 
	            # 记录训练数据
	            train_l_sum += l.item()
	            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
	            n += y.shape[0]
	        
	        # 训练完成一个 epoch 后，评估测试集上的准确率
	        test_acc = evaluate_accuracy(test_iter, net)
	        
	        # 打印提示信息
	        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
	              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
	
	
	if __name__ == '__main__':
	    # 输入输出维度
	    num_inputs,num_outputs = 28*28,10   # 图像尺寸28x28，拉平后向量长度为 28*28；类别空间为 10
	
	    # 初始化模型参数 & 设定超参数
	    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True) 
	    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
	    num_epochs, lr = 5, 0.1             # 超参数
	
	    # 获取数据读取迭代器
	    batch_size = 256
	    train_iter, test_iter = load_data_fashion_mnist(batch_size, 4)
	
	    # 进行训练
	    train(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
	
	    # 使用得到模型预测 10 张图
	    X, y = iter(test_iter).next()
	
	    true_labels = get_fashion_mnist_labels(y.numpy())
	    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
	    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
	
	    # 显示预测结果
	    show_fashion_mnist(X[0:9], titles[0:9])
	    plt.show()
	```

# 3. 利用 Pytorch 简洁地实现 softmax 回归
- pytorch 中提供了大量预定义的神经网络层，常用损失函数及优化器，可以大大简化 softmax  回归模型的实现
- 数据准备、模型评价、使用模型进行预测三部分和第 2 节实现相同，本节不再重复

## 3.1 模型设计
### 3.1.2 模型定义
- 如 1.2 节所示，softmax 回归的输出是一个全连接层，可以使用 `torch.nn.Linear` 方法定义，如下
	```python
	num_inputs = 28*28
	num_outputs = 10
	
	class LinearNet(nn.Module):
	    def __init__(self, num_inputs, num_outputs):
	        super(LinearNet, self).__init__()
	        self.linear = nn.Linear(num_inputs, num_outputs)
	        
	    def forward(self, x):                        # x shape: (batch, 1, 28, 28)
	        y = self.linear(x.view(x.shape[0], -1))  # 拉平样本数据为 (batch, 1x28x28)
	        return y
	
	net = LinearNet(num_inputs, num_outputs)
	```
	注意到原始图像尺寸为 `torch.Size([1, 28, 28])`，数据迭代器返回的 batch `x` 尺寸为 (batch_size, 1, 28, 28)，**在做前向传播时，必须要把样本都拉平，即把 `x` 的形状转换为 (batch_size, 1x28x28) 才能送入全连接层**

- 按照深度学习的习惯，可以**把数据拉平这件事定义成神经网络的一个层**，如下
	```python
	class FlattenLayer(nn.Module):
	    def __init__(self):
	        super(FlattenLayer, self).__init__()
	        
	    def forward(self, x): # x shape: (batch, *, *, ...)
	        return x.view(x.shape[0], -1)
	```
	这样就能**更符合习惯地，利用 `Sequential` 容器搭建网络模型**
	```python
	from collections import OrderedDict
	
	net = nn.Sequential(
	    OrderedDict([
	        ('flatten', FlattenLayer()),
	        ('linear', nn.Linear(num_inputs, num_outputs))
	    ])
	)
	```
### 3.1.2 模型初始化
- 利用 `torch.nn.init` 包提供的方法进行初始化，初始化值同 2.2 节
	```python
	nn.init.normal_(net.linear.weight, mean=0, std=0.01)
	nn.init.constant_(net.linear.bias, val=0) 
	
	'''
	Parameter containing:
	tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
	'''
	```
### 3.1.3 softmax 和交叉熵损失函数
- 第 2 节中我们**按照数学定义分别定义了 softmax 函数和 CrossEntropy 损失，这样做可能导致数据不稳定**。softmax 函数定义为
    $$
    \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
    $$
    注意这里都是 exp 运算，一旦**网络初始化不当，或输入数值有较大噪音，很可能导致数值溢出（$o$ 是很大的正整数会导致上溢，$o$ 是很大的负整数会导致下溢）**，这时可以用 **Log-Sum-Exp Trick（logSoftmax）**处理，它在数学上就是在普通 softmax 外面套了一个 log 函数，这不会影响概率排序，但是**通过有技巧地实现可以有效解决溢出问题**。具体参考 [深入理解softmax](https://blog.csdn.net/qq_34554039/article/details/122087189)
- 总之，我们**可以用 logSoftmax + NLLLoss 避免数据溢出，保证数据稳定性，并且得到等价的交叉熵损失，pytorch 中直接把这两个放在一起封装了一个 `nn.CrossEntropyLoss` 方法**，对于一组小批量数据，假设模型输出和真实标签分别为 $output$ 和 $truth$， `nn.CrossEntropyLoss` 如下计算小批量损失
    $$
    \text{CrossEntropyLoss}(output,truth) = \text{NLLLoss}(\text{logSoftmax}(output),truth)
    $$
	在这里**我们直接使用它来替代前面自己定义的 softmax 函数和 CrossEntropy 损失**。可参考 [Pytorch中Softmax、Log_Softmax、NLLLoss以及CrossEntropyLoss的关系与区别详解](https://blog.csdn.net/qq_28418387/article/details/95918829)
	```python
	loss = nn.CrossEntropyLoss()
	```
## 3.2 模型训练
### 3.2.1 优化器
- 直接使用 pytorch 提供的小批量随机梯度优化器 `torch.optim.SGD` 

	```python
	optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # 学习率 0.1
	```
- 基本用法
	1. 梯度清零：`optimizer.zero_grad()`
	2. 执行一次优化：`optimizer.step() `
### 3.2.2 训练流程
- 和 2.4.2 节完全类似，训练程序只须改一下优化器部分的处理即可
	```python
	num_epochs = 5
	
	def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
	    # 训练执行 num_epochs 轮
	    for epoch in range(num_epochs):
	        train_l_sum = 0.0    # 本 epoch 总损失
	        train_acc_sum = 0.0  # 本 epoch 总准确率
	        n = 0                # 本 epoch 总样本数
	        
	        # 逐小批次地遍历训练数据
	        for X, y in train_iter:
	            
	            # 计算小批量损失
	            y_hat = net(X)
	            l = loss(y_hat, y).sum()  
	
	            # 梯度清零
	            optimizer.zero_grad()
	
	            # 小批量的损失对模型参数求梯度
	            l.backward()
	            
	            # 做小批量随机梯度下降进行优化
	            optimizer.step()              
	
	            # 记录训练数据
	            train_l_sum += l.item()
	            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
	            n += y.shape[0]
	        
	        # 训练完成一个 epoch 后，评估测试集上的准确率
	        test_acc = evaluate_accuracy(test_iter, net)
	        
	        # 打印提示信息
	        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
	              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
	
	train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
	'''
	epoch 1, loss 0.0021, train acc 0.826, test acc 0.816
	epoch 2, loss 0.0020, train acc 0.833, test acc 0.822
	epoch 3, loss 0.0019, train acc 0.836, test acc 0.823
	epoch 4, loss 0.0019, train acc 0.840, test acc 0.827
	epoch 5, loss 0.0018, train acc 0.843, test acc 0.829
	'''
	```
## 3.3 完整代码
- 整合上述过程，给出完整代码，可以直接粘贴进 vscode 运行

	```python
	import torch
	from torch import nn
	import torchvision
	import torchvision.transforms as transforms
	import numpy as np
	from IPython import display
	from collections import OrderedDict
	import matplotlib.pyplot as plt
	
	
	# 数据集相关 --------------------------------------------------------------------------------------------------
	# 加载数据集
	def load_data_fashion_mnist(batch_size, num_workers=0, root='./Datasets/FashionMNIST'):
	    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,transform=transforms.ToTensor())
	    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,transform=transforms.ToTensor())
	
	    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	
	    return train_iter, test_iter
	
	# 数据集标签转换
	def get_fashion_mnist_labels(labels):
	    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
	                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	    return [text_labels[int(i)] for i in labels]
	
	# 显示数据图片
	def show_fashion_mnist(images, labels):
	    display.set_matplotlib_formats('svg')
	    
	    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
	    for f, img, lbl in zip(figs, images, labels):
	        f.imshow(img.view((28, 28)).numpy())
	        f.set_title(lbl)
	        f.axes.get_xaxis().set_visible(False)
	        f.axes.get_yaxis().set_visible(False)
	
	
	# 模型定义 --------------------------------------------------------------------------------------------------------
	class FlattenLayer(nn.Module):
	    def __init__(self):
	        super(FlattenLayer, self).__init__()
	        
	    def forward(self, x): # x shape: (batch, *, *, ...)
	        return x.view(x.shape[0], -1)
	
	# 准确率评估
	def evaluate_accuracy(data_iter, net):
	    acc_sum = 0.0  # 所有样本总准确率
	    n =  0         # 总样本数量
	    for X, y in data_iter:
	        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
	        n += y.shape[0]
	    return acc_sum / n
	
	def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
	    # 训练执行 num_epochs 轮
	    for epoch in range(num_epochs):
	        train_l_sum = 0.0    # 本 epoch 总损失
	        train_acc_sum = 0.0  # 本 epoch 总准确率
	        n = 0                # 本 epoch 总样本数
	        
	        # 逐小批次地遍历训练数据
	        for X, y in train_iter:
	            
	            # 计算小批量损失
	            y_hat = net(X)
	            l = loss(y_hat, y).sum()  
	
	            # 梯度清零
	            optimizer.zero_grad()
	
	            # 小批量的损失对模型参数求梯度
	            l.backward()
	            
	            # 做小批量随机梯度下降进行优化
	            optimizer.step()              
	
	            # 记录训练数据
	            train_l_sum += l.item()
	            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
	            n += y.shape[0]
	        
	        # 训练完成一个 epoch 后，评估测试集上的准确率
	        test_acc = evaluate_accuracy(test_iter, net)
	        
	        # 打印提示信息
	        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
	              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
	
	if __name__ == '__main__':
	    # 输入输出维度
	    num_inputs,num_outputs = 28*28,10
	    
	    # 超参数
	    num_epochs,lr = 5,0.1
	
	    # 获取数据读取迭代器
	    batch_size = 256  
	    train_iter, test_iter = load_data_fashion_mnist(batch_size, 4)
	
	    # 定义模型网络结构
	    net = nn.Sequential(
	    OrderedDict([
	        ('flatten', FlattenLayer()),
	        ('linear', nn.Linear(num_inputs, num_outputs))
	        ])
	    )
	
	    # 初始化模型参数
	    nn.init.normal_(net.linear.weight, mean=0, std=0.01)
	    nn.init.constant_(net.linear.bias, val=0) 
	
	    # 损失 & 优化器
	    loss = nn.CrossEntropyLoss()
	    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 学习率 0.1
	
	    # 进行训练
	    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
	
	    # 使用得到模型预测 10 张图
	    X, y = iter(test_iter).next()
	
	    true_labels = get_fashion_mnist_labels(y.numpy())
	    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
	    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
	
	    # 显示预测结果
	    show_fashion_mnist(X[0:9], titles[0:9])
	    plt.show()
	```
