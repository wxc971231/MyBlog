- Hugging Face 是一家在 NLP 和 AI 领域具有重要影响力的科技公司，他们的开源工具和社区建设为NLP研究和开发提供了强大的支持。它们拥有当前最活跃、最受关注、影响力最大的 [NLP 社区](https://huggingface.co/)，最新最强的 NLP 模型大多在这里发布和开源。该社区也提供了丰富的教程、文档和示例代码，帮助用户快速上手并深入理解各类 Transformer 模型和 NLP 技术
- Transformers 库是 Hugging Face 最著名的贡献之一，它最初是 Transformer 模型的 pytorch 复现库，随着不断建设，至今已经成为 NLP 领域最重要，影响最大的基础设施之一。该库提供了大量预训练的模型，涵盖了多种语言和任务，成为当今大模型工程实现的主流标准，换句话说，**如果你正在开发一个大模型，那么按 Transformer 库的代码格式进行工程实现、将 check point 打包成 hugging face 格式开源到社区，对于推广你的工作有很大的助力作用**。本系列文章将介绍 [Transformers库](https://github.com/huggingface/transformers) 的基本使用方法
- 前文：[Hugging face Transformers（1）—— 基础知识](https://blog.csdn.net/wxc971231/article/details/140231114)
-----
@[toc]
# 1. 什么是 Pipeline
- Pipeline 是 Transformers 库的一个高层次封装类，它可以将**数据预处理**、**模型调用**、**结果后处理**三部分组装成流水线，**为用户忽略复杂的中间过程，仅保留输入输出接口**
- 利用 Pipeline，用户可以方便地加载各种模型检查点，直接输入文本来获取最终的结果，而无需关注中间细节
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/055cdf2eed0d4fcabfb84f84d8a47157.png#pic_center)
	上图显示了 Pipeline 的处理过程
	1. **输入预处理**：原始输入字符串被 Tokenizer 组件处理为目标模型支持格式的 token 序列（由词表索引组成的列表）
	2. **模型预测**：Token 序列被输入目标模型，通过前向过程得到模型输出，基于目标任务不同，输出形式会有所区别
		> 图示为情感二分类任务，故输出的 logits 只有两个维度
	3. **结果后处理**：基于目标任务类型，对模型输出进行后处理，生成结果
		> 对于图示的情感分类任务，后处理是过 softmax 后概率采样或贪心选取结果
# 2. Pipeline 支持的任务类型
- 前文 [Hugging face Transformers（1）—— 基础知识](https://blog.csdn.net/wxc971231/article/details/140231114) 提到了目前主要的九类 NLP 任务，除了这些经典任务外，Transformers 库还支持关于图像、音频等其他模态的任务。可以用如下代码检查所有任务类型
	```python
	from transformers.pipelines import SUPPORTED_TASKS
	
	tasks = []
	for k, v in SUPPORTED_TASKS.items():
	    tasks.append(f"{v['type']:15} {k}")
	for t in sorted(tasks):
	    print(t)
	```
	
	```shell
	audio           audio-classification
	image           depth-estimation
	image           image-classification
	image           image-feature-extraction
	image           image-to-image
	multimodal      automatic-speech-recognition
	multimodal      document-question-answering
	multimodal      feature-extraction
	multimodal      image-segmentation
	multimodal      image-to-text
	multimodal      mask-generation
	multimodal      object-detection
	multimodal      visual-question-answering
	multimodal      zero-shot-audio-classification
	multimodal      zero-shot-image-classification
	multimodal      zero-shot-object-detection
	text            conversational
	text            fill-mask
	text            question-answering
	text            summarization
	text            table-question-answering
	text            text-classification
	text            text-generation
	text            text-to-audio
	text            text2text-generation
	text            token-classification
	text            translation
	text            zero-shot-classification
	video           video-classification
	```
-  官方提供的任务表格如下
	![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/38cc4e2bdbac4a0bb53478031c5be6f4.png#pic_center)



# 3. 创建和使用 Pipeline
## 3.1 基本操作
- 创建 Pipeline 对象，在传入参数中指定任务类型、模型和 tokenizer 等
	1. 任务类型如第 2 节表格所示
	2. 模型地址可以在[模型列表](https://huggingface.co/models)找到
	3. 如果不指定模型，将下载目标任务的默认模型和配套 tokenizer
		```python
		from transformers import *
		
		# 1. 根据任务类型直接创建 Pipeline，这时会使用默认的某个英文模型
		pipe = pipeline("text-classification")	# 文本分类任务
		pipe('very good!')	# [{'label': 'POSITIVE', 'score': 0.9998525381088257}]
		
		# 2. 同时指定任务类型和模型
		# model 可以在 https://huggingface.co/models 找到
		pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")
		pipe("我觉得不太行")	# [{'label': 'negative (stars 1, 2 and 3)', 'score': 0.9743660688400269}]
		```
- 以上调用发起时，pipeline 对象会如下创建（下文第 5 节会进一步讨论等效实现）
	1. 会自动下载 model snapshot、配置文件、词表等必须内容，默认保存地址为 C:\Users\username\.cache\huggingface\hub（已下载过则从此加载）
	2. 根据 Transformers 库内置源码创建，并使用下载文件初始化 model 对象和 Tokenizer 对象
	3. 根据任务类型连接后处理方法
- 可以先创建 model 对象，再将其作为参数创建 Pipeline，这种情况下必须同时指定传入 model 对象和 tokenizer 对象
	```python
	from transformers import *
	
	# 3. 先加载模型，再创建Pipeline（必须同时指定 model 和 tokenizer）
	model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
	pipe("我觉得不太行") 	# [{'label': 'negative (stars 1, 2 and 3)', 'score': 0.9743660688400269}]
	```
- 使用 Pipeline 时，可以传入原始字符串或字符串列表，当原始数据量特别多时，最好传入测试数据生成器
	```python
	# 使用默认的情感分类模型
	pipe = pipeline("text-classification")
	
	# 直接传入原始字符串
	pipe('very good!')  
	'''
	[{'label': 'POSITIVE', 'score': 0.9998525381088257}]
	'''
	
	# 传入原始字符串列表
	pipe(['very good!', 'very bad!'])
	'''
	Disabling tokenizer parallelism, we're using DataLoader multithreading already
	[{'label': 'POSITIVE', 'score': 0.9998525381088257},
	 {'label': 'NEGATIVE', 'score': 0.9997695088386536}]
	'''
	
	# 传入原始字符串生成器
	def list_to_generator(lst):  
	    for item in lst:  
	        yield item  
	sentence_generator = list_to_generator(['very good!', 'very bad!'])  
	
	for res in pipe(sentence_generator):
	    print(res)
	'''
	{'label': 'POSITIVE', 'score': 0.9998525381088257}
	{'label': 'NEGATIVE', 'score': 0.9997695088386536}
	'''
	```
## 3.2 Auto 类型
- 在以上情感分类示例的幕后，Pipeline 使用了 `AutoModelForSequenceClassification` 和 `AutoTokenizer` 构造上面的 pipe 对象。**[AutoClass](https://huggingface.co/docs/transformers/model_doc/auto) 是一种快捷方式，它可以从名称或路径中自动检索预训练模型的体系结构**，例如上面出现过的
	```python
	model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	```
	可见，只需要用模型路径调用 AutoClass 的 `.from_pretrained()` 方法，大多数情况下都可以自动检索相关的预训练 model 的 weights/config 或 tokenizer 的 vocab table

- 注意到模型的自动类 `AutoModelForSequenceClassification` 包含 `ForSequenceClassification` 后缀，这是因为**第 2 节所述的各类任务重，很多是可以用相同的 model 骨干完成的**。比如对于 “句子情感分类” 和 “句子自回归生成” 两个任务，前者可以看作是基于前驱序列特征做二分类任务（正面情感/负面情感），后者可以看作是基于前驱序列特征做多分类任务（从词表中选择一个token索引），两个任务中 “前驱序列特征” 都是可以用 GPT 模型提取的，也就是说**相同的 model，接入不同的 post processing 模块，就可以用于不同的任务**。因此，在 Transformers 库的设计上，一个相同的模型骨干可以对应多个不同的任务，它们使用后缀进行区分，详见[源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/modeling_auto.py#L1438)
# 4. 使用 GPU 进行推理
- 定义的 pipeline 默认在 CPU 执行，速度慢

	```python
	# 默认的执行设置是 CPU
	pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")
	print(pipe.model.device)	# cpu
	
	# 在 CPU 运行会很慢
	import torch
	import time
	import numpy as np
	times = []
	for i in range(100):
	    start = time.time()
	    pipe("我觉得不太行")
	    torch.cuda.synchronize()# 阻塞CPU线程，直到所有在当前设备上排队的CUDA核心完成执行为止
	    end = time.time()
	    times.append(end-start)
	print(np.mean(times))		# 0.07967857599258422
	```
- 通过 device 参数，在定义 pipeline 时指定到 GPU 设备运行，可以有效提高推理速度
	```python
	# 通过 device 参数指定执行设备
	pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese", device=0)
	
	# 查看模型的运行设备
	print(pipe.model.device)		# cuda:0
	
	# 在 GPU 运行会更快
	import torch
	import time
	import numpy as np
	times = []
	for i in range(100):
	    start = time.time()
	    pipe("我觉得不太行")
	    torch.cuda.synchronize()  	# 阻塞CPU线程，直到所有在当前设备上排队的CUDA核心完成执行为止
	    end = time.time()
	    times.append(end-start)
	print(np.mean(times))			# 0.022842261791229248
	```
# 5. Pipeline 背后的实现
- 下面我们分别实现以上情感分类 pipeline 内部的组件，并手动执行数据流，看清其执行过程
	```python
	# 1. 初始化 tokenizer
	tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	
	# 2. 初始化 model
	model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	
	# 3. 输入预处理（Tokenize）
	input_text = "我觉得不太行"
	inputs = tokenizer(input_text, return_tensors='pt')
	print(inputs)  # {'input_ids': tensor([[ 101, 2769, 6230, 2533,  679, 1922, 6121,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
	
	# 4. 模型预测
	res = model(**inputs)
	print(res)     # SequenceClassifierOutput(loss=None, logits=tensor([[ 1.7459, -1.8919]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
	
	# 5. 结果后处理
	logits = res.logits
	logits = torch.softmax(logits, dim=-1)
	print(logits)  # tensor([[0.9744, 0.0256]], grad_fn=<SoftmaxBackward0>) 正面情感/负面情感
	pred = torch.argmax(logits).item()        # 0
	result = model.config.id2label.get(pred)  
	print(result)  # negative (stars 1, 2 and 3)
	```
