---
title: 论文理解【Vision Transformer】——【VIT】An Image is Worth 16x16 Words-Transformers for Image Recognition at Scale
date: 2025-09-10 17:31:46
index_img: img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_001.png
tags:
  - Transformer-Based
  - Vision Transformer
  - CV backbone
  - CV
categories:
  - 论文理解
description: VIT将图像切分成16x16的patch块，通过标准Transformer进行图像分类，在引入尽量少图像归纳偏置的情况下，验证了纯Transformer在图像分类任务中的有效性，为CV和NLP的模型统一奠定基础。
---

- 首发链接：[论文理解【CV】——【VIT】An Image is Worth 16x16 Words-Transformers for Image Recognition at Scale](https://blog.csdn.net/wxc971231/article/details/141721062)
- 文章链接：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- 代码：[GitHub - google-research/vision_transformer](https://github.com/google-research/vision_transformer)
- 发表：ICLR 2021
- 领域：Transformer-based CV
- 一句话总结：VIT 将完整图像切分成多个 16x16 的 patch 块，每个块拉平后经过线性变换看作 token embedding 输入**标准 Bert 模型**提取特征，用**有监督方法**训练图像分类模型。VIT 使用了尽量少的图像归纳偏置，因此在小数据量的情况下不如 CNN 类模型，但得益于 Transformer 更好的 scaling 能力，其在大规模数据集上表现更好。**VIT 验证了标准 Transformer 模型也可以良好地解决 CV 问题，并且仍然保持 NLP 中的 Scaling 能力**。VIT 使得 CV 和 NLP 任务在模型上统一起来，启发了大量后续研究，并揭示了多模态大模型的可行性
    > 关于归纳偏置，请参考：[从模型容量的视角看监督学习](https://blog.csdn.net/wxc971231/article/details/128107548)
- -------
- 摘要：虽然 Transformer 已经成为了 NLP 任务中的标准结构，它在 CV 领域的应用仍很有限。目前，在 CV 任务中注意力机制要么与卷积网络结合应用，要么用于在保持网络整体结构的同时替换卷积网络的某些组成部分。**我们证明了这种对 CNN 的依赖是不必要的，一个直接应用于图像 patch 序列的纯 Transformer 就可以很好地执行图像分类（目标识别）任务**。当对大量数据进行预训练，并转移到多个中型或小型图像识别基准测试（ImageNet、CIFAR-100、VTAB等）时，与最先进的卷积网络相比，我们提出的 Vision Transformer（ViT）获得了良好的结果，且需要的计算资源更少

@[toc]
# 1. 方法
- 自从 2016 年 self attention 机制出现后，已经有大量研究将其用到 CV 领域中，特别是尝试将其和 Transformer 类模型结合以利用其 Scaling 优势。这里的主要困难是当时 Transformer 类模型输入长度无法做得很大，也缺乏序列长度泛化能力，**直接简单地把图像各个像素值拉平作为 token 进行训练会导致序列长度太长而无法训练**。针对该问题，早期研究包括
    1. 先进行多层卷积和池化，得到较小的特征图（如14x14）后拉平作为 token 训练 Transformer 模型
    2. 类似卷积核，使用滑动窗口扫描图像，仅在局部进行 attention 计算，并以此代替卷积运算
    3. 使用稀疏注意力或轴注意力，降低标准自注意力的计算量。但这些做法大都无法进行并行加速训练
- 以上问题本质上就是一个**图像的 `tokenize & embed` 问题**，核心要素有两个
    1. 能够**把较大尺寸的图像（如 224x224x3）变化成可接受长度的（如**$\leq$**1000）的 token embedding 序列**，从而适配于标准的 Transformer 模型
    2. 为了使模型可以尽可能地在 CV 和 NLP 任务上通用，**该方法应当尽量少地使用图像数据的归纳偏置**
- 本文作者设计了一种基于 patch 切分的 tokenize & embedding 方案，提出的 VIT 方法如下
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_001.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

    如图可见，作者将原始尺寸为 224x224x3 的图像切分成 196 个 16x16x3 的 patch 块，每个块拉平后得到尺寸 768 维的张量（地位等同于 NLP token），再用 768x768 的线性层投影到嵌入空间（地位等同于 NLP embedding）。由于本文考虑的是图像分类任务（目标识别任务），作者使用了作为自编码器的 BERT 骨干模型，使用可学习的 1D 位置编码，并引入 [CLS] 分类 token 构成完整的 VIT 模型。**除了 tokenize & embedding 方法不同外，其他设计和 BERT 分类模型完全一致**，形式化描述如下
    $$
    \begin{array}{l}
    \begin{aligned}
    \mathbf{z}_{0} & =\left[\mathbf{x}_{\text {class }} ; \mathbf{x}_{p}^{1} \mathbf{E} ; \mathbf{x}_{p}^{2} \mathbf{E} ; \cdots ; \mathbf{x}_{p}^{N} \mathbf{E}\right]+\mathbf{E}_{p o s}, & & \mathbf{E} \in \mathbb{R}^{\left(P^{2} \cdot C\right) \times D}, \mathbf{E}_{p o s} \in \mathbb{R}^{(N+1) \times D} \\
    \mathbf{z}_{\ell}^{\prime} & =\operatorname{MSA}\left(\operatorname{LN}\left(\mathbf{z}_{\ell-1}\right)\right)+\mathbf{z}_{\ell-1}, & & \ell=1 \ldots L \\
    \mathbf{z}_{\ell} & =\operatorname{MLP}\left(\operatorname{LN}\left(\mathbf{z}_{\ell}^{\prime}\right)\right)+\mathbf{z}_{\ell}^{\prime}, & & \ell=1 \ldots L \\
    \mathbf{y} & =\operatorname{LN}\left(\mathbf{z}_{L}^{0}\right) & &
    \end{aligned}\\
    \end{array}
    $$
    其中 $\mathbf{x}_{\text {class}}$ 即为随机初始化的 [CLS] token，$\mathbf{x}_{p}^{1},...,\mathbf{x}_{p}^{N}$ 为拉平的 patch token，$\mathbf{E}$ 为线性嵌入层，MSA 为多头自注意力层
    > 事实上，ICLR 2020 论文 On the relationship between self-attention and convolutional layers 其实已经提出和本文 VIT 技术上完全一致的方法，但它使用的数据集 CIFAR 10 的图像尺寸仅 32x32，其切分的 patch 尺寸为 2x2，没有验证扩展能力。VIT 基本就是把这个设计用到更大规模的数据集和模型上
- ViT 可以应用到 CV 领域常见的预训练 + 微调框架中，即首先大数据集上预训练，然后使较小的下游任务数据集进行微调。微调的具体做法为
    1. 移除预训练好的预测头，重新加上一个全零初始化的 $D\times K$ 的前馈神经网络作为新的预测头（$K$是下游任务的分类数）
    2. 使用更高分辨率的图像进行微调通常更有益（比如在 224x224 的图像上预训练，在 600x600 的任务上微调）。但问题是，当保持相同大小的 patch 块时，更高分辨率图像对应的 token 序列长度会变长，这会导致预训练的位置编码失效。作者通过在预训练位置编码之间插值来解决这个问题
- **VIT 针对图像归纳偏置的设计远远少于 CNN**
    1. CNN 中每一层卷积核都用到了图像的局部性、二维结构和平移不变性，归纳偏置的针对性设计贯穿了整个模型
    2. ViT 中只有 MLP 层利用到图像的局部性和平移不变性，自注意力层完全没有用（都是全局信息）。图像二维结构的使用也很谨慎，仅在切分 patch 和微调时调整不同图像分辨率位置信息时用到。位置编码在初始化时也没有携带任何的二维图像块信息，所有位置信息的空间关系都需要从头开始学习

# 2. 实验
- 作者设置了多种规模，使用多种 patch 尺寸的 ViT 模型，使用两种尺寸的 ImageNet 数据集和 JFT 数据集进行预训练。作为对比，CNN based 模型 MobileNet-v2 参数规模为13M，ResNet34 模型参数规模 85M
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_002.png" alt="在这里插入图片描述" style="width: 75%;">
    </div>

    文章的主实验如下，使用不同规模数据集预训练，比较其与 CNN 模型的性能
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_003.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    可见，**在数据量较小时，无论是在ImageNet还是JFT数据集，BiT（以ResNet为骨干的CNN模型）准确率相对更高**，但是**当数据集量增大到一定程度时，ViT模型略优于CNN模型**。所以，ViT 模型更需要大数据集进行预训练，以提高模型的表征。这种结果是可预期的
    1. **ViT 使用了更少的归纳偏置，相当于人为引入的领域知识更少，这会导致模型的数据利用率较低**，在小数据集上性能会较差
    2. 同样因为 **ViT 使用的归纳偏置少，模型的假设空间更大，使用更多数据有效训练后，其性能上限会比 CNN 类方法更好**，这也可以理解为人类设计的领域知识不一定完全适配于数据
- 预训练后，**ViT 微调到下游任务的效率更高**，如图所示，微调训练 7 个epoch时，ViT 类的模型相较于 CNN 模型效果更好。这说明 ViT 学习到的特征可能比 CNN 更通用
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_004.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    > 参考 [ViT论文及代码解读-ICLR2021：Transformer用于视觉分类也有很好的性能](https://zhuanlan.zhihu.com/p/510218124) ：无论是使用小模型和轻量化模型AlexNet、MobileNetv2，还是使用大模型ResNet50，要达到较好预测，都要训练30-50epoch甚至更高。而使用ViT模型仅需要2-3个epoch便可达到更优秀的性能
- 将 ViT 与基于 CNN 的 SOTA 模型对比，发现**各个任务下，ViT 模型都轻微超越了 SOTA 性能，同时其训练使用的计算资源要低很多**
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_005.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    > 这里有一点值得注意的，原始 ViT-L/16 这个模型由于参数规模太大，容易过拟合，作者使用了很大规模的  I21K 数据集才能稳定训练。MAE 论文中直接用小 21 倍的 IN1k 数据集训练该模型得到的性能只有 76.5，但是如果给他更强的正则化约束，则也可以稳定收敛，且性能提升到 82.5
- ViT 模型内部可视化分析
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_006.png" alt="在这里插入图片描述" style="width: 90%;">
    </div>

    1. 左图可视化了**线性嵌入层**$E$**学到的 28 个主成分，看起来它们可以作为每个 patch 二维结构特征的合理基函数**。对 CNN 卷积核的可视化中也能看到类似的成分
    2. 中图可视化了训练完毕的 1D 可学习位置编码得到的，各个位置嵌入之间的相关性。一方面注意到二维空间上越接近的位置相关性越高，另一方面注意到 patch 所在行列的相关性较高，说明 **1D 位置编码也能够学到良好的 2D 空间位置信息**
    3. 右图可视化了 ViT 模型每一层的所有注意力头的`平均注意距离`，它描述了图像空间中被整合信息的平均距离，类似 CNN 中的接受野大小。**如图可见，ViT 中最底层就存在一些注意力头可以整合全局信息，而 CNN 每个注意力头都只能整合局部信息**。同时，注意到较高层中注意力头整合信息的距离都比较远，这说明其已经学习到语义概念，**高层的注意力头主要关注了语义相关的图像区域**
        > 对于每个注意力头，图像上任意两个patch之间存在一个注意力权重，这二者间的注意力距离就是二者间欧式距离乘以注意力权重，所有patch二元组之间注意力距离的平均值被定义为平均注意距离
        
        对 ViT 最后一层（输出层）注意力头的注意范围进行可视化，可以发现其确实关注位置和类别语义信息是一致的
        <div align="center">
            <img src="/MyBlog/img/论文理解VisionTransformer_VITAnImageisWorth16x16Words_TransformersforImageRecognitionatScale/img_007.png" alt="在这里插入图片描述" style="width: 90%;">
        </div>

# 3. 总结
- **ViT = NLP 序列模型（Transformer） + CV 训练范式（有监督预训练+微调）**
- 本文验证了 NLP 中的标准 Transformer 模型也能有效应用到 CV 任务中，属于一个破局挖坑的工作。以前 CV 领域的研究主要还是集中在 CNN 的各种变形上，但本文说明完全不用 CNN 也能解决复杂的 CV 问题，而且有潜力比 CNN 做得更好，这就带来了很多新的思路，比如
    1. 新的图像 Tokenize & Embedding 方法
    2. 本文只做了目标识别，还可以将 ViT 用于目标检测、分割等其他 CV 任务
    3. 修改中间的 Transformer Block 结构，后来有人把 attention 换成 MLP 甚至不需学习的 Pooling 层，发现效果也还不错
    4. 目标函数上可以继续走 CV 的有监督，也可以走 NLP 中的自监督，而且后者意义更大，因为无需构造标记数据了。后来 [MAE](https://blog.csdn.net/wxc971231/article/details/142708130?spm=1001.2014.3001.5502) 方法补足了这块
    5. 本文验证了 Transformer 这个骨干可以同时有效提取一维结构化数据（文本）和二维结构化数据（图像）的特征，这暗示了其处理多模态数据的能力，这一点近年来也被深度发掘了