---
title: 论文理解【Vision Transformer】—— 【MAE】Masked Autoencoders Are Scalable Vision Learners
date: 2025-09-10 17:41:42
index_img: img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_002.png
tags:
  - Transformer-Based
  - Vision Transformer
  - CV
categories:
  - 论文理解
description: MAE 是一种 Transformer-Based CV backbone，其核心在于使用了类似 Bert 模型的训练机制，通过高比例随机 mask 图像 patch，使用非对称 Encoder-Decoder 架构进行自监督训练重建图像，实现了高效的视觉特征学习。
---

- 首发链接：[论文理解【Vision Transformer】—— 【MAE】Masked Autoencoders Are Scalable Vision Learners](https://blog.csdn.net/wxc971231/article/details/142708130)
- 文章链接：[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- 代码：[GitHub - facebookresearch/mae](https://github.com/facebookresearch/mae)
- 发表：CVPR 2022
- 领域：Transformer-based CV
- 一句话总结：本文提出的 MAE 是一种将 Transformer 模型用作 CV backbone 的方法，核心在于其使用了类似 Bert 模型的训练机制。具体而言，MAE 首先对图像 patch 块进行**高比例**的随机 mask，再用一个非对称的 **Encoder-Decoder 架构**进行**自监督**训练来重建图像，训练结束后的 Encoder 提取到高质量的图像特征，可通过微调适用于下游任务。**相比 ViT，MAE 的自监督预训练过程无需图像标签，训练效率更高，在加速训练的同时提高了准确性，并且表现出有潜力的 Scaling 能力**
    > 1. ViT = NLP 序列模型（Transformer Encoder） + CV 训练范式（有监督预训练+微调）
    > 2. **MAE = NLP 序列模型（Transformer Encoder-Decoder） + NLP 训练范式（MLM自监督预训练+微调）**
-------
- 摘要: 本文证明了掩码自编码器（MAE）是一种可扩展的 CV 自监督学习器。我们的MAE方法很简单：先随机 mask 掉输入图像的部分 patch 块，再重建缺失的像素。MAE 包含两个核心设计：**首先，我们开发了一个非对称的 Encoder-Decoder 架构，其中 Encoder 只作用于没有被 mask 的可见 patch 子集，轻量级 Decoder 则从 latent representation 和 mask tokens 重构原始图像。其次，我们发现掩盖高比例的输入图像，例如75%，可以构造一个普适而有意义的自监督任务**。耦合这两种设计使我们能够快速有效地训练大规模模型并提高准确性。我们的可扩展方法能够学到泛化能力很强的高容量模型...

# 1. 方法
- 作者注意到 NLP 领域中，以 BERT 为代表的 MLM 自监督训练范式取得了很大的成功，为了把 MLM 训练扩展到 CV 领域中，作者先分析了 CV 和 NLP 任务的差异
    1. **架构差异**：CV 领域长期以来被 CNN 架构所主导，由于 CNN 使用卷积核汇聚局部特征，将 Mask 遮盖集成到 CNN 中是困难的。不过这个困难已经被近期提出的 ViT 模型通过 **patch 形式的 tokenize 方法**解决了
    2. **信息密度差异**：语言信号是高度语义化的，信息密度高，而图像信号具有很高的空间冗余性。和 ViT 的 patch 图块相比，自然语言中单个 token 的语义性更强，信息量更高。因此应该**遮盖更大比例的 patch 图块，使模型不能仅依靠低级统计信息完成重建任务，从而迫使模型学会对图像的整体理解**
        <div align="center">
            <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_001.png" alt="在这里插入图片描述" style="width: 100%;">
        </div>

    3. **解码器扮演角色的差异**：NLP 中解码器只用来还原一个词表中的 token 索引，抽象程度高数据复杂度低，因此用 MLP 解码器就足够了；**CV 任务中解码器需要重建一个图块 patch，抽象程度低数据复杂度高，为此需要更复杂的解码器结构**

- 针对以上差异，作者设计的 MAE 模型结构如下所示
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_002.png" alt="在这里插入图片描述" style="width: 60%;">
    </div>

    如图所示，MAE 模型使用**非对称的 encoder-decoder 架构**，它输入被 mask 遮盖的图像，由较大规模的 Transformer Encoder 将所有**无遮盖的** patch token 投影到 embedding 空间中，然后**按顺序插入统一的 learnable mask token embedding** 并**加上位置编码**，最后使用一个轻量的 Transformer Decoder 将 embedding 序列重建为图像。具体而言
    1. **`Masking`**：使用和 ViT 相同的方法将输入图像切分成不重叠的 patch 块，然后从中**均匀随机采样高比例（如**$75\%$**）的子集 mask 掉**，一方面消除冗余信息以提高 MLM 任务的难度；另一方面为 Encoder 提供稀疏的输入以降低计算成本
    2. **`MAE Encoder`**：直接使用一个 **ViT 模型作为 Encoder**，它通过线性投影来嵌入 patch 块，添加位置编码后通过一系列的Transformer block 进行处理。Encoder **仅应用于可见的，未被 mask 的 patch**，其比例很低，因此计算成本并不高
    3. **`MAE Decoder`**：Decoder 的**输入是完整的 patch embedding 序列，其中所有 masked token 共享一个可学习 embedding 向量**。因此，需要按 masked patch 的原始拉平顺序 masked embedding 的多个副本插入到 Encoder 输出中组成完整序列，添加完整的位置编码后输入 Transformer Decoder 进行图像重建（预测对应 patch 的图像像素点值）
       > 注意此 Decoder 不同于常说的 Transformer Decoder，它还是用的**双向注意力，本质是 BERT 类的模型**
    4. **Reconstruction target**: MAE 通过预测每个 masked patch 的像素值来重建图像。Decoder 的最后一层是一个线性投影层，其输出尺寸和一个 patch 的像素数量相同，因此输出向量可以被 reshape 重建成 patch 图像。**损失设计为重建图像和原始图像的像素值 mse 误差，类似 BERT，仅在 masked patch 位置计算损失**。作者还研究了一个变体，其重建目标是每个 masked patch 的归一化像素值，实验表明这个重建目标提高了表示质量
- MAE pipeline 的一个简单实现如下
    1. 将原始图像切分为不重叠的 patch 图块
    2. 使用带有位置嵌入的线性层将每个 patch 转换为 token embedding，拉平得到标准顺序序列
    3. 随机打乱 token embedding 序列（shuffling），并根据 masked 比例删除列表的最后一部分，将保留的前驱部分输入 Encoder
    4. 获取 Encoder 输出，在其后拼接 masked embedding 序列以恢复序列长度，重新顺序为标准顺序（unshuffling），使所有自监督标签和序列中的 embedding 对齐
    5. 向完整序列添加位置编码后输入 Decoder 进行重建，计算并优化损失
- 值得注意的是，在 MAE 之前，已经有一些探索如何把 Transformer 模型用于 CV 任务的工作尝试了 MLM 式的自监督训练方法（比如 ViT 就做过），但是效果不好没有流行起来。主要原因有以下几点
    1. **没有注意到图像信息密度低的归纳偏置，mask 比例不够**
    2. **没有直接重建原始图像 pixel ，而是重建一些低维表示**（比如马赛克或者 mean color）
    3. **没有设置独立的 Decoder 模型，而是直接在 Encoder 输出上进行重建**。但是 Encoder 的编码结果应该是一个更全局、高维的表示，直接在其基础上重建 pixel 会让 Encoder 后几层太关注局部细节，导致提取特征的抽象程度不足
# 2. 实验
## 2.1 主实验
- 作者在 IN1k 数据集上比较了 ViT-L/16 有监督从头训练和 MAE finetune 的结果，可以看到 MAE finetune 超越了有监督训练的效果
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_003.png" alt="在这里插入图片描述" style="width: 60%;">
    </div>

    > 这里特别注意，ViT 原始论文的实现正则化强度太低了，ViT-L/16 这个规模的模型很容易过拟合，因此在 IN1k 这个规模较小的数据集上只达成了 76.5 的性能，且不太稳定。MAE 作者调整了实现细节后性能提升到 82.5，但是仍不如 MAE finetune 的 84.9

    和其他经典方法的对比体现了 MAE 的有效性
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_004.png" alt="在这里插入图片描述" style="width: 60%;">
    </div>

- 作者在 COCO 数据集的目标检测任务和 ADE20k 数据集上的分割任务验证了 MAE 的**迁移性能**，同样是最好的
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_005.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

## 2.2 消融实验
- 作者考察了 masking ratio 的影响。作者评估了两种微调方法，一是`全参数微调fine-tuning`，二是只`微调Encoder最后的线性层linear probing`，注意到**最佳 masking ratio 是相当高的，达到 75% 左右**
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_006.png" alt="在这里插入图片描述" style="width: 65%;">
    </div>

    注意到全量 fine-tuning 比较稳健，而 linear probing 关于 mask ratio 更敏感。作者进一步考察了介于二者之间的部分微调方案
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_007.png" alt="在这里插入图片描述" style="width: 65%;">
    </div>

    可见微调一半的 Transformer Block 基本就足够了
- 作者进一步对解码器的结构进行对比，其中 ft 表示全量微调，lin 表示只调线性层
    <div align="center">
        <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_008.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    - \(C\) 说明在 **Encoder 中不加入 Mask token 的话性能更好，而且计算量更少**
    - (d) 对比数据增强方法，发现**简单随机裁剪**就足够好了
    - (e) 对比 mask 策略，发现**均匀随机 mask 效果最好**
        <div align="center">
            <img src="/MyBlog/img/论文理解VisionTransformer_MAEMaskedAutoencodersAreScalableVisionLearners/img_009.png" alt="在这里插入图片描述" style="width: 65%;">
        </div>


# 3 总结 & 感受
- MAE 进一步发展了 ViT 将 NLP 方法和训练范式运用到 CV 任务上的思路，通过以下三个创新在 Image 1K 上取得了很好的结果
  1. 由于图像的信息密度较低，CV 任务上应该使用更高的 mask ratio
  2. 使用 Transformer decoder 重建图像
  3. 加上了 ViT 之后的各种 trick，使训练更鲁棒
- 这篇文章的思路不难，但是故事讲得足够好，从将 NLP 范式直接用到 CV 任务的问题存在的问题开始讲，这个写法是可以帮助读者理解的。另外实验做得很详细，具有很好的借鉴意义
- 参考：[如何看待何恺明最新一作论文Masked Autoencoders？ - 李rumor的回答 - 知乎](https://www.zhihu.com/question/498364155/answer/2240224120)
