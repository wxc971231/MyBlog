---
title: Transformers库（1）—— 基础介绍
date: 2025-10-01 19:47:57
index_img: img/Transformers库/Transformers.png
tags:
  - Transformers库
categories:
  - 常用库 
  - Transformers库
description: 介绍 Transformers 库的基本概念
---

- 首发链接：[Transformers库（1）—— 基础介绍](https://blog.csdn.net/wxc971231/article/details/140231114)
- Hugging Face 是一家在 AI 领域具有重要影响力的科技公司，他们的开源工具和社区建设为NLP研究和开发提供了强大的支持。它们拥有当前最活跃、最受关注、影响力最大的 [NLP 社区](https://huggingface.co/)，最新最强的机器学习模型大多在这里发布和开源。该社区也提供了丰富的教程、文档和示例代码，帮助用户快速上手并深入理解各类 Transformer 模型和 NLP 技术
- Transformers 库是 Hugging Face 最著名的贡献之一，它最初是 Transformer 模型的 pytorch 复现库，随着不断建设，至今已经成为 NLP 领域最重要，影响最大的基础设施之一。该库提供了大量预训练的模型，涵盖了多种语言和任务，成为当今大模型工程实现的主流标准，换句话说，**如果你正在开发一个大模型，那么按 Transformer 库的代码格式进行工程实现、将 check point 打包成 hugging face 格式开源到社区，对于推广你的工作有很大的助力作用**。本系列文章将介绍 Transformers 库的基本使用方法
- 官方文档链接：[Transformers](https://huggingface.co/docs/transformers/main/en/index)
-----

# 1. 常见自然语言处理任务
- 目前常见的 NLP 任务主要可以归纳为
    || 任务 | 描述 |
    |:--|:--|:--|
    |1| 情感分析 (sentiment-analysis) | 对给定的文本分析其情感极性 |
    |2|文本生成 (text-generation)|根据给定的文本进行生成|
    |3|命名实体识别 (ner)|标记句子中的实体|
    |4|阅读理解 (question-answering)|给定上下文与问题，从上下文中抽取答案|
    |5|掩码填充 (fill-mask)|填充给定文本中的掩码词|
    |6|文本摘要 (summarization)|生成一段长文本的摘要|
    |7|机器翻译 (translation)|将文本翻译成另一种语言|
    |8|特征提取 (feature-extraction)|生成给定文本的张量表示|
    |9|对话机器人 (conversional)|根据用户输入文本，产生回应，与用户对话|
- 稍早时（17年 Transformer 发表到 20 年 GPT 3 发布），研究人员大多使用源自 CV 的 “特征预训练 + 下游任务微调” 模式，这时不同任务有各自适合的模型
    - 7 是原生 Transformer 模型（Encoder-Decoder结构）的目标任务
    - 5 是 BERT 模型（Transformer Encoder）的预训练任务之一
    - 8 过去经常用 BERT 模型完成
    - 1 过去经常用 BERT 模型完成，基本可以看作 8 之后接一个分类或者回归头
    - 2、3、4、6、9 都可以用 GPT 模型完成（Transformer Decoder）
- 自从 20 年 GPT3 验证了增大参数量带来的规模效应（Scaling Law）后，GPT 类模型越来越受到重视，至今已逐渐实现了各种任务的大一统
# 2. 自然语言处理的几个发展阶段
1. **第一阶段（传统机器学习思路）**：统计模型＋数据（特征工程)
    - 特征提取：TF-IDF、BOW...
      - 使用经典统计模型：决策树、SVM、HMM、CRF...
2. **第二阶段（深度学习思路）**：神经网络＋数据
    - 特征提取：Word2vec、Glove...
      - 使用深度学习模型：MLP、CNN、RNN/LSTM/GRU、Transformer...
3. **第三阶段（预训练微调思路）**：预训练＋(少量)数据微调思路
    - 特征提取：BERT 类 Transformer Encoder 模型（同时有很多工作直接端到端）
    - 使用 Transformer 类序列模型：GPT、BERT/RoBERTa/ALBERT、BART/T5
4. **第四阶段（大模型思路）**：神经网络＋更大的预训练模型＋Prompt
    - 基本没有明确的特征提取阶段，或者说用 GPT 提取前驱序列特征
    - 基于 GPT 的各种序列生成模型：ChatGPT、Bloom、LLaMA、Alpaca、Vicuna、MOSS...
# 3. Transformers简单介绍
- Transformers 是**文本**、**计算机视觉**、**音频**、**视频**和**多模态**模型中 SOTA 模型的**模型定义框架，用于推理和训练**，**其模型定义得到了整个生态系统的支持**，与大多数训练框架（Axolotl、Unsloth、DeepSpeed、FSDP、PyTorch-Lightning 等）、推理引擎（vLLM、SGLang、TGI 等）和相关模型库（llama.cpp、mlx 等）兼容
- [Hugging Face Hub](https://huggingface.com/models)上有超过 100 万个 Transformer-based 模型 checkpoint 可以使用，且支持用户自行上传模型、数据集等组件，社区完善，文档全面，三两行代码便可快速实现模型训练推理，上手简单
## 3.1 特性
- Transformers 库的主要特性包括：
    1. [Pipeline](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial)：针对许多机器学习任务（如文本生成、图像分割、自动语音识别、文档问答等）的简单且优化的推理类
    2. [Trainer](https://huggingface.co/docs/transformers/main/en/trainer)：一个全面的训练器，支持混合精度、torch.compile 和 FlashAttention 等功能，用于 PyTorch 模型的训练和分布式训练
    3. [generate](https://huggingface.co/docs/transformers/main/en/llm_tutorial)：使用大型语言模型（LLM）和视觉语言模型（VLM）快速生成文本，包括对流和多种解码策略的支持

## 3.2 设计哲学
- Transformers 是一个专为以下用途而构建的库：
    1. 寻求使用、研究或扩展大规模 Transformers 模型的机器学习研究人员和教育工作者
    2. 想要微调这些模型或在生产中使用这些模型，或两者兼而有之的实践者
    3. 只想下载预训练模型并使用它来解决给定的机器学习任务的工程师
- Transformers 库的设计有两个重要的目标：
    1. **尽可能简单、快速地使用**：
        - 我们**严格限制了需要学习的面向用户的抽象的数量**，使用每个模型只需要三个标准类：[configuration](https://huggingface.co/docs/transformers/main/en/main_classes/configuration)、[models](https://huggingface.co/docs/transformers/main/en/main_classes/model) 和预处理类（用于 NLP 的 [tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer)、用于 CV 的 [image processor](https://huggingface.co/docs/transformers/main/en/main_classes/image_processor)、用于音频的 [feature extractor](https://huggingface.co/docs/transformers/main/en/main_classes/feature_extractor)  和用于多模态输入的 [processor](https://huggingface.co/docs/transformers/main/en/main_classes/processors)）
        - 所有这些类都可以**使用通用方法 `from_pretrained()` 从预训练实例中以简单统一的方式进行初始化**， 该方法从Hugging Face Hub 上提供的预训练检查点或您自己保存的 ckpt 下载、缓存和加载相关类实例和相关数据（配置超参数、tokenizer 词表和模型权重等）
        - 除了这三个基类之外，该库还提供了两个 API：[pipeline()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) 用于快速使用模型对给定任务进行推理，[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) 用于快速训练或微调 PyTorch 模型。
        - 因此，此库并非一个模块化的神经网络构建工具箱。如果想扩展或构建此库，只需使用常规 Python 或 PyTorch 并继承库的基类即可重用模型加载和保存等功能
    2. **提供性能尽可能接近原始模型的 SOTA 模型**：
        - 我们为每个架构提供至少一个示例，以重现该架构的官方作者提供的结果
        - 代码通常尽可能接近原始代码库，这意味着一些 PyTorch 代码可能不像从其他深度学习框架转换而来的那样具有*PyTorchic* 性。
- 其他一些目标：
    1. **尽可能一致地暴露模型的内部结构**：
        - 我们使用单个 API 来访问完整的隐藏状态和注意力权重。
        - 预处理类和基础模型 API 是标准化的，以便在模型之间轻松切换。
    2. **结合有前景的工具来微调和调查这些模型**：
        - 提供了一种简单而一致的方法，可以向词汇表和嵌入中添加新的标记以进行微调
        - 提供了 mask 和 prune Transformer head 的简单方法
- 主要概念
    - 该库中每个模型都围绕三个类进行构建：
        1. **Model classes**：PyTorch 模型（`torch.nn.Module`）
        2. **Configuration classes**：存储构建模型所需的超参数（如层数和隐藏层大小）。您并不总是需要亲自实例化这些参数。特别是，如果您使用的是未经任何修改的预训练模型，则创建模型时会自动实例化配置（它是模型的一部分）
        3. **Preprocessing classes**：将原始数据转换为模型可接受的格式。`tokenizer` 存储每个模型的词汇表，并提供用于对输入到模型的分词嵌入索引列表中的字符串进行编码和解码的方法；`Image processors` 预处理视觉输入， `feature extractors`  预处理音频输入，还有一个 `processor` 处理多模态输入。
    - 所有这些类都可以从预训练的实例中实例化、本地保存，并通过三种方法在 Hub 上共享：
        1. `from_pretrained()` 允许您从库本身提供的预训练版本（可以在模型中心找到支持的模型）或用户在本地（或服务器上）存储的预训练版本实例化模型、配置和预处理类
        2. `save_pretrained()` 允许您在本地保存模型、配置和预处理类，以便可以使用重新加载 from_pretrained()
        3. `push_to_hub()` 让您与 Hub 共享模型、配置和预处理类，以便每个人都可以轻松访问
# 4. Transformers及相关库
- Transformers 库包含以下核心组件

    | 组件 | 描述 |
    |:--|:--|
    | Transformers | 核心库，模型加载、模型训练、流水线等 |
    |Tokenizer|分词器，对数据进行预处理，文本到 token 序列的互相转换|
    |Datasets|数据集库，提供了数据集的加载、处理等方法|
    |Evaluate|评估函数，提供各种评价指标的计算函数|
    |PEFT|高效微调模型的库，提供了几种高效微调的方法，小参数量撬动大模型|
    |Accelerate|分布式训练，提供了分布式训练解决方案，包括大模型的加载与推理解决方案|
    |Optimum|优化加速库，支持多种后端，如Onnxruntime、OpenVino等|
    |Gradio|可视化部署库，几行代码快速实现基于Web交互的算法演示系统|
- 安装方法：目前（2024.7.6）最新版本需要 python 3.8+ 和 PyTorch 1.11+，如下使用 pip 或 conda 安装
    ```shell
    pip install transformers
    conda install conda-forge::transformers
    ```
    如果你想要测试用例或者想在正式发布前使用最新的开发中代码，你得[从源代码安装](https://huggingface.co/docs/transformers/installation#installing-from-source)
- 部分官方文档
    | 章节 | 描述 |
    |:--|:--|
    | [文档](https://huggingface.co/docs/transformers/) | 完整的 API 文档和教程 |
    | [任务总结](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers 支持的任务 |
    | [预处理教程](https://huggingface.co/docs/transformers/preprocessing) | 使用 `Tokenizer` 来为模型准备数据 |
    | [训练和微调](https://huggingface.co/docs/transformers/training) | 在 PyTorch/TensorFlow 的训练循环或 `Trainer` API 中使用 🤗 Transformers 提供的模型 |
    | [快速上手：微调和用例脚本](https://github.com/huggingface/transformers/tree/main/examples) | 为各种任务提供的用例脚本 |
    | [模型分享和上传](https://huggingface.co/docs/transformers/model_sharing) | 和社区上传和分享你微调的模型 |
    | [迁移](https://huggingface.co/docs/transformers/migration) | 从 `pytorch-transformers` 或 `pytorch-pretrained-bert` 迁移到 🤗 Transformers |
    | [教程](https://huggingface.co/learn) | 包含 LLM、MCP、Agents、RL、CV、Audio 等各种方向的机器学习教程 |