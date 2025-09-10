---
title: 论文理解【CV】——【Swin Transformer】Hierarchical Vision Transformer using Shifted Windows
date: 2025-09-10 16:21:43
index_img: img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_001.png
tags:
  - Transformer-Based
  - Vision Transformer
  - CV
categories:
  - 论文理解
description: Swin Transformer 是一种 Transformer-Based 通用 CV 骨干网络。该模型借鉴 CNN 中的卷积和池化操作，设计了分层金字塔结构、滑动窗口机制和 Patch Merging 操作，通过将图像划分为可重叠的局部窗口，令注意力计算约束在窗口内部，使计算复杂度与图像大小呈线性关系，在保持 Transformer 强大建模能力的同时引入视觉归纳偏置，是首个在通用视觉任务上全面超越 CNN 的 Transformer 架构
---

- 首发链接：[论文理解【CV】——【Swin Transformer】Hierarchical Vision Transformer using Shifted Windows](https://blog.csdn.net/wxc971231/article/details/148057310)
- 文章链接：[Swin transformer: Hierarchical vision transformer using shifted windows](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper)
- 代码：[microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- 发表：ICCV 2021
- 领域：Transformer-based CV
- 一句话总结：Swin Transformer 是一种**基于 Transformer 的通用 CV 骨干网络**。该模型借鉴 CNN 中的卷积和池化操作，设计了分层金字塔结构、滑动窗口机制和 Patch Merging 操作，通过将图像划分为可重叠的局部窗口，令**注意力计算约束在窗口内部，使计算复杂度与图像大小呈线性关系**，突破了 VIT 等传统 Transformer-based CV 模型的二次方复杂度限制，在保持Transformer强大建模能力的同时更适配视觉任务特性，成为**首个在通用视觉任务上全面超越CNN的Transformer架构，推动了视觉Transformer的实用化进程**
----------
- 摘要：本文提出了一种新型视觉 Transformer——Swin Transformer，可以作为**CV领域的通用主干网络**使用。将Transformer从语言领域迁移到视觉领域面临诸多挑战，主要源于两类数据的差异：视觉实体尺度变化大，且图像像素分辨率远高于文本单词。为此，我们提出采用移位窗口（Shifted Windows）计算表征的分层Transformer结构。**该移位窗口方案通过将自注意力计算限制在非重叠的局部窗口内提升效率，同时允许跨窗口连接。这种分层架构能灵活建模多尺度特征，且计算复杂度与图像大小呈线性关系**。Swin Transformer广泛适用于各类视觉任务：图像分类（ImageNet-1K达到87.3% top-1准确率）、密集预测任务如目标检测（COCO test-dev上58.7 box AP和51.1 mask AP）和语义分割（ADE20K val上53.5 mIoU）。其性能显著超越此前最佳水平：COCO检测任务提升+2.7 box AP和+2.6 mask AP，ADE20K分割任务提升+3.2 mIoU，证明了Transformer作为视觉骨干网络的潜力。这种分层设计和移位窗口方法对全MLP架构也具借鉴意义。

# 1. 方法
- CV 领域的通用视觉骨干网络长期以来由卷积网络 CNNs 主导，虽然 Transformer 在 NLP 领域取得了巨大成功，但将其应用于 CV 任务时仍面临挑战
- [VIT](https://blog.csdn.net/wxc971231/article/details/141721062) 是首篇成功地使用 Transformer（Encoder）模型解决图像分类任务的文章，该方法针对图像信息密度低于文字的特点，提出了基于 Patch 分块的图像 tokenize & embedding 方案，在 `图像分类` 任务上取得了比 CNN 骨干更高的性能上限，但其仍有局限性：
    1. VIT **缺少 CNNs 中的池化机制，无法逐层扩展感受野，无法提取多尺度特征**，针对 224x224x3 的输入图像，其每层特征图都直接切分成 16x16 个 16x16x3 的 patch 块，细分粒度固定且较粗
    2. VIT 没有利用图像的局部性归纳偏置，每一层都是在全图切分的 196 个 patch 上计算 self attention，总是进行全局建模，**计算复杂度（patch数量）随输入图像尺寸增加以平方速度扩展，导致计算量快速膨胀**，难以用于大尺寸图像
        > VIT 致力于设计 CV & NLP 通用模型，**仅最小程度地利用图像数据的归纳偏置，因此效率训练效率低**
    3. VIT 只针对基础的图像分类任务，**由于无法提取多尺度特征，难以应用于目标检测/图像分割等密集预测型任务，无法作为通用视觉骨干网络**
- 针对以上问题，Swin Transformer 充分利用了图像的归纳偏置，通过借鉴 CNNs 中的池化设计，控制计算复杂度随输入图像尺寸增长而线性增长，保持了和 CNNs 相同数量级的计算效率。同时，Swin Transformer 实现了多尺度特征提取，可作为通用视觉骨干模型使用
    > - 无多尺度特征 = 图像分类；
    > - 多尺度特征 + FPN = 目标检测；
    > - 多尺度特征 + U-Net = 图像分割
## 1.1 分层特征映射
- 分层特征映射是 Swin Transformer 的最主要设计，其**借鉴了 CNNs 的池化操作**，是允许模型进行**多尺度特征提取**和实现**计算复杂度随输入尺寸线性增长**的核心
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_001.png" alt="在这里插入图片描述" style="width: 60%;">
    </div>

    其中 **self-attention 永远在红框**中计算，所有红框 patch数量（Transformer计算量）相同，如图所示
    1. VIT 不进行下采样，每层特征图尺寸相同，都切分为相同数量的 patch 块，在全图计算 attention
    2. Swin Transformer 中引入了类似池化的 Patch Merging 操作，图示中由下至上进行了两次 2 倍下采样（Patch Merging），每次操作**使得特征图尺寸减小2x2=4倍，而通道数（nx）增加2倍，红框内 self attention 部分计算量不变但感受野增大**
        > 默认设定下，Swin Transformer 每个 self attention 窗口内有 7x7=49 个 patch，每个 patch 尺寸 4x4x3
- 具体而言，`Patch Merging` 操作如下：设原图尺寸 $H\times W\times C$，首先对原来的特征图进行一次 2 倍下采样，变成 4 张分辨率更低的特征图，然后扩展通道，得到特征图尺寸 $\frac{H}{2}\times \frac{W}{2}\times 4C$，经过 LayerNorm 后通过 1x1 卷积减半通道数，最终特征图尺寸 $\frac{H}{2}\times \frac{W}{2}\times 2C$
    > 注：降低通道数是为使处理后通道数翻倍而非翻四倍，从而和 CNNs 中的池化操作特点保持一致
    
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_002.png" alt="在这里插入图片描述" style="width: 75%;">
    </div>

## 1.2 基于移位窗口的自注意力
- Transformer 中全局自注意力计算每个 token 与所有其他 token 之间的关系，导致与计算复杂度关于 token 数量呈二次方关系，使其不适合需要大量 token 进行密集预测或输入高分辨率图像的视觉问题，**窗口自注意机制可以使计算复杂度关于 token 数量呈线性增长**。具体地，设输入图像通道为 $C$，切分为 $h\times w$ 个不重叠的自注意力窗口，每个窗口由 $M\times M$ 个不重叠的 patch 组成，永远在窗口内部计算自注意力（序列长度永远为 $M^2$）。这种情况下，传统的多头自注意力MSA和窗口多头自注意力W-MSA的计算量如下
    $$
    \begin{aligned}
    &\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C \\
    &\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC \\
    \end{aligned}
    $$
    把图像看作长度 $h\times w$ 的序列，**可见**$\Omega(\text{MSA})$**关于 hw 二次方增长，而 $\Omega(\text{W-MSA})$ 关于 hw 线性增长**
    > - 按默认设定，Swin Transformer 输入图像尺寸 224x224x3，第一层中将其切分为 56x56 个尺寸 4x4x3 的 patch，每 7x7 个 patch 组成一个注意力窗口。这里 $h=w=224, C=3, M=7$
    > - 以上计算复杂度只考虑浮点数乘法次数，MSA 输入等价于长度 hw 的 C 维 embedding 序列，计算方式参考 [序列模型（3）—— LLM的参数量和计算量](https://blog.csdn.net/wxc971231/article/details/135434478) 第 2.2 节；W-MSA 相当于计算 $\frac{h}{M}x\frac{w}{M}$ 次长度 $M$ 的 $C$ 维度 embedding 序列
- 窗口自注意 W-MSA 虽然有效降低了计算复杂度，但使得窗口间无法交换信息，为此，作者进一步提出了**移位窗口机制**，通过**向右下移动窗口后再计算一次 W-MAS 实现上下文信息聚合（称为 SW-MAS）**，如下图所示
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_003.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>

    -  **W-MSA 和 SW-MSA 是绑定的，必须在连续的两层中执行，组成一个基本计算单元**，因此 Swin Transformer 的总 Transformer Block 层数一定是偶数。连续两层计算过程的数学表达为
        $$
        \begin{array}{l}
        \hat{\mathbf{z}}^{l}=\text { W-MSA }\left(\mathrm{LN}\left(\mathbf{z}^{l-1}\right)\right)+\mathbf{z}^{l-1}, \\
        \mathbf{z}^{l}=\mathrm{MLP}\left(\mathrm{LN}\left(\hat{\mathbf{z}}^{l}\right)\right)+\hat{\mathbf{z}}^{l}, \\
        \hat{\mathbf{z}}^{l+1}=\mathrm{SW}-\mathrm{MSA}\left(\mathrm{LN}\left(\mathbf{z}^{l}\right)\right)+\mathbf{z}^{l}, \\
        \mathbf{z}^{l+1}=\mathrm{MLP}\left(\mathrm{LN}\left(\hat{\mathbf{z}}^{l+1}\right)\right)+\hat{\mathbf{z}}^{l+1},
        \end{array}
        $$
    - SW-MSA 中自注意力还是在红框中计算，一共有 9 个区域要计算。为了提升计算效率，作者在此设计了一个巧妙的移位+掩码计算方法，使得两次自注意力计算的 patch 数和 patch 尺寸都一致。基本思路如下图所示：
        <div align="center">
            <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_004.png" alt="在这里插入图片描述" style="width: 80%;">
        </div>

        首先通过 cyclic shift 将 A,B,C 块移动下去，使其和原先一样切分成 4 个区域。然后注意到右上、右下、左下三块中移过来的部分和原先的部分不是相邻图像，因此不应该交互信息，故需通过设置 mask 控制注意力计算的范围，实际还是在以下 9 个区域内部计算自注意力
        <div align="center">
            <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_005.png" alt="在这里插入图片描述" style="width: 20%;">
        </div>

## 1.3 相对位置编码
- 不同于 VIT，Swin Transformer 使用相对位置编码，其基本思想是：**如果两个 Patch 的相对位置一致，那么相对位置编码也应该一致**。在模型中，它体现在**为每个 Attention head 计算 Q K 相似度时引入相对位置偏置** $B\in \mathbb{R}^{M^2\times M^2}$，即：
    $$
    \text { Attention }(Q, K, V)=\operatorname{SoftMax}\left(Q K^{T} / \sqrt{d}+B\right) V
    $$
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_006.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>

    注意到相对位置编码总是以自己为（0, 0）计算其他 patch 的相对位置，分别把4个相对位置拉平即得到4x4的矩阵。接下来要做的事就是**把每个框中的**$(x_{r},y_r)$**转换为一维数字 $k_{xy}$，并且保证相同的 $(x_{r},y_r)$ 对应的 $k_{xy}$ 一致，从而可以用一个可学习的参数表示它们**。具体地，作者通过以下三个操作实现
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_007.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>

    这里需要注意，$B\in \mathbb{R}^{M^2\times M^2}$ 代表任意两个 patch 间的相对位置关系，但实际上 patch 在每个轴上的相对位置关系都在区间 $[-M+1, M-1]$ 内，区间长度 $2M-1$，$k_{xy}$ 最多有 $(2M-1)^2$ 个取值，**故只需要使用一个可学习的 `nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))` 学习较小的 $\hat{B}\in \mathbb{R}^{(2M-1)\times (2M-1)}$，即可从中恢复 $B$**，如下所示
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_008.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>





## 1.4 总体结构

- Swin Transformer 结构图如下
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_009.png" alt="在这里插入图片描述" style="width: 100%;">
    </div>

    1. 输入图像尺寸 $H\times W\times 3$
    2. 切分 patch，Swin Transformer 默认每个 patch 为 4x4x3 的图块，处理后图像尺寸 $\frac{H}{4}\times \frac{W}{4}\times 48$，其中 48 是 patch 拉平得到的
    3. 在 Stage 1 中，先做线性变换调整通道数量为指定超参数 $C$，图像尺寸变成 $\frac{H}{4}\times \frac{W}{4}\times C$ 
        > VIT 至此都一样，之后它直接把特征图看作长度 $\frac{H}{4}\times \frac{W}{4}$ 的 $C$ 维 embedding 序列计算 self-attention
    4. 在 Stage 1 中，经 Swin Transformer Block 输出尺寸不变，维持 $\frac{H}{4}\times \frac{W}{4}\times C$ 
    5. 在 Stage 2 中，经 Patch Merging，尺寸变为 $\frac{H}{8}\times \frac{W}{8}\times 2C$，经过 Swin Transformer Block 维持不变
    6. 在 Stage 3、Stage 4 中，操作和 Stage 2 完全相同，最终输出尺寸 $\frac{H}{32}\times \frac{W}{32}\times 8C$，直接用于各类视觉下游任务（比如图像分类任务就是用 Global average pooling 把 7x7 取平均拉平，变成 1x1x8C，再用线性层调整维度为类别数量，接softmax分类头）
- 注意到 **Stage 1 的 patch 对应原图尺寸（感受野）最小，随着网络的加深，Stage 2、Stage 3、Stage 4 通过 patch merging 层将相邻的图像块合并，从而降低分辨率并增大感受野**。这种逐步降低分辨率的过程使得 Swin Transformer 能够在不同的阶段输出不同尺寸的特征图，从而实现 **`多尺度特征表示`**
# 2. 实验
- 在ImageNet22K数据集上，准确率能达到惊人的86.4%。另外在检测，分割等任务上表现也很优异，感兴趣的可以翻看论文最后的实验部分
    <div align="center">
        <img src="/MyBlog/img/论文理解CV_SwinTransformerHierarchicalVisionTransformerusingShiftedWindows/img_010.png" alt="在这里插入图片描述" style="width: 80%;">
    </div>


# 3. 总结 & 感受
- 本文提出的 Swin Transformer 是一种新型视觉 Transformer 架构，通过引入分层特征表示和移位窗口机制，实现了线性计算复杂度，并能够有效地建模多尺度特征。与传统的 Vision Transformer（ViT）相比，Swin Transformer在多个方面展现了显著的优势。
    1. 相比 ViT 只能生成单一低分辨率特征图，**Swin Transformer 的分层特征映射机制使其能够生成多尺度的特征图，使其能更好地适应目标检测和语义分割等密集预测任务**
    2. 相比 ViT 使用全局自注意力机制，计算复杂度呈二次增长，难以扩展到大规模图像处理任务，S**win Transformer 的移位窗口自注意机制通过在不同层之间交替使用规则和移位窗口划分，增强了模型捕捉跨窗口信息的能力，并实现了计算复杂度的线性增长**
    3. 实验表明，Swin Transformer 在 ImageNet 图像分类、COCO 目标检测和 ADE20K 语义分割等任务上均表现出色，显著超越了ViT及其变体
- 总的来说，**Swin Transformer 有效地将图像数据的归纳偏置和 Transformer 骨干相结合，相比朴素的 VIT 方法大幅提升了计算效率和性能表现**。但是**图像归纳偏置的引入也限制了 Swin Transformer 的通用性**，因为 Shift Window 的机制用到 NLP 领域的合理性不是很强，相比而言 ViT 可能更适合作为 CV/NLP 的大一统模型骨干
