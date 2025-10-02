- Hugging Face 是一家在 AI 领域具有重要影响力的科技公司，他们的开源工具和社区建设为NLP研究和开发提供了强大的支持。它们拥有当前最活跃、最受关注、影响力最大的 [NLP 社区](https://huggingface.co/)，最新最强的 NLP 模型大多在这里发布和开源。该社区也提供了丰富的教程、文档和示例代码，帮助用户快速上手并深入理解各类 Transformer 模型和 NLP 技术
- Transformers 库是 Hugging Face 最著名的贡献之一，它最初是 Transformer 模型的 pytorch 复现库，随着不断建设，至今已经成为 NLP 领域最重要，影响最大的基础设施之一。该库提供了大量预训练的模型，涵盖了多种语言和任务，成为当今大模型工程实现的主流标准，换句话说，**如果你正在开发一个大模型，那么按 Transformer 库的代码格式进行工程实现、将 check point 打包成 hugging face 格式开源到社区，对于推广你的工作有很大的助力作用**。本系列文章将介绍 [Transformers库](https://github.com/huggingface/transformers) 的基本使用方法
- 前文：[Hugging face Transformers（2）—— Pipeline](https://blog.csdn.net/wxc971231/article/details/140233173)
-----
@[toc]
# 1. Tokenizer 及其基本使用
- **Tokenizer 是将原始字符串转换为模型可以计算的数值形式（通常是 token IDs）的工具**。不同的模型可能需要不同的 tokenizer，因为不同的预训练任务和数据集可能会导致不同的词汇表（vocabulary）和 tokenization 策略。
- Tokenizer 用于数据预处理，其作用包括
	1. **分词**：使用分词器对文本数据进行分词 (字、字词)
	2. **构建词典**：根据数据集分词的结果，构建词典映射 (这步并不绝对，如果采用预训练词向量，词典映射要根据词向量文件进行处理)
	3. **数据转换**：根据构建好的词典，将分词处理后的数据做映射，将文本序列转换为数字序列。其中可能涉及添加特殊标记（如 `[CLS]`、`[SEP]`、`[MASK]` 等），以便模型能够识别文本的不同部分或执行特定的任务（如分类、问答等）
	4. **数据填充与截断**：在以batch输入到模型的方式中，需要对过短的数据进行填充，过长的数据进行截断，保证数据长度符合模型能接受的范围，同时batch内的数据维度大小一致
## 1.1 保存与加载
- 如前文 [Hugging face Transformers（2）—— Pipeline](https://blog.csdn.net/wxc971231/article/details/140233173) 3.2 节所述，可以用 `AutoTokenizer` 自动类，从模型地址直接识别、创建并初始化所需的 tokenizer 对象。这里我们还是使用前文的中文情感分类模型的 tokenizer
	```python
	# AutoTokenizer 包可以根据传入的参数（如模型名）自动判断所需的 tokenizer
	from transformers import AutoTokenizer
	
	# 样例字符串
	sen = "这是一段测试文本"
	
	# 从 hugging face 加载，输入模型名称即可加载对应的分词器
	tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	tokenizer
	```
	```shell
	BertTokenizerFast(name_or_path='uer/roberta-base-finetuned-dianping-chinese', vocab_size=21128, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
		0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	}
	```
	从打印信息可见，这是一个 BERT 模型的 Tokenizer，其中有五个特殊符号，在右侧进行填充或截断...
- 第一次创建 Tokenizer 对象时，词表等相关配置文件会下载并保存至默认路径 C:\Users\username\\.cache\huggingface\hub，之后会默认从此处重新加载。可以将构造的 tokenizer 对象手动保存到指定路径，并从指定路径加载
	```python
	# 自动下载的 model 和 tokenizer 等组件位于 C:\Users\username\.cache\huggingface\hub 中
	# 可以把 tokenizer 单独保存到指定路径
	tokenizer.save_pretrained("./roberta_tokenizer")
	
	# 可以从本地加载保存的 tokenizer
	tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer")
	```
## 1.2 句子分词
- Tokenizer 工作的第一步是文本分割，即**将原始输入字符串拆开成一系列字符、词、字节码或短句（称之为`token`）**。在中文自然语言处理中分词尤为重要，因为中文的词与词之间没有空格这样明显的分隔符。分词方法的设计是开放的，相同句子可以有多种不同的分词方案，常见的包括
	1. **Word-based tokenize (基于词的分词)：将原始文本拆分为单词**，如下所示这类方法有多种变体，通常需要设置 “unknown” token（“[UNK]” 或 “\<unk\>”），制作词表时的一个目标是将尽可能少的单词标记为 “[UNK]”，因此通常会构成非常大的词表
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c42833d6afd843d595b7733fa690fafa.png#pic_center =80%x)
	2. **Character-based tokenize (基于字符的分词)：将文本拆分为字符**，这种方式构成的词表小得多，且能有效减少 “unknown” token，但这种方法也不完美。和单词不同，直觉上拉丁语言中每个字符本身并没有多大意义，而且这样做会导致模型需要处理大量的 tokens，导致计算复杂度上升
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71b3c6e3156c4afbb7a0fd0fdf728815.png#pic_center =80%x)
	3. **Subword-based tokenize (基于子词的分词)：这种方法基于一个原则，常用词不应被分解为更小的子词，但罕见词应被分解为有意义的子词**。下例中，“tokenization” 被分割成 “token” 和 “ization”，这两个 tokens 在保持空间效率的同时具有语义意义，这让我们能够在词汇量小的情况下获得相对良好的覆盖率，并且几乎没有未知的 token
		![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11e4224c6de2441cbfa3cbd10467a507.png#pic_center =80%x)
	这种方法在土耳其语等粘着型语言（agglutinative languages）中特别有用，可以通过将子词串在一起来形成（几乎）任意长的复杂词
	4. **其他先进技术**：分词技术在 LLM 流行之前就在 NLP 领域得到大量研究，其他先进方案包括 GPT-2 使用的 Byte-level BPE、BERT 使用的 WordPiece、多个语言模型使用的 SentencePiece or Unigram 等

- 根据分词方法不同，对应的词表也会有所区别。一般而言，较大的词表可以包含更多的词汇，有助于模型更好地理解和表达文本，提高模型性能，增强泛化能力。然而，随着词表尺寸的增加，模型的计算复杂度和内存需求也会相应增加。可以通过 Tokenizer 对象的 `.vocab` 属性查看词表
	```python
	tokenizer.vocab
	```
	
	```shell
	{'##净': 14169,
	 'ま': 567,
	 '##copyright': 13291,
	 '疡': 4550,
	 '抢': 2843,
	 '枇': 3355,
	 '##尘': 15269,
	 '贺': 6590,
	 'ne': 10564,
	 '庸': 2435,
	 '##馬': 20736,
	 '臾': 5640,
	 '勖': 1241,
	 '##粱': 18175,
	 '##⒋': 13574,
	 '褥': 6191,
	 'doc': 9656,
	 '釁': 7022,
	 'alex': 10179,
	 '##フト': 10868,
	 '屹': 2256,
	 'yumi': 11697,
	 '##nne': 12866,
	 '莫': 5811,
	 '816': 10937,
	...
	 '##躍': 19770,
	 '皺': 4653,
	 '##ろ': 10460,
	 '##孪': 15169,
	 ...}
	```

- Transformers 库的 tokenizer 支持传入原始字符串或原始字符串列表，如下所示
	```python
	tokens = tokenizer.tokenize(sen)
	print(tokens)   # ['这', '是', '一', '段', '测', '试', '文', '本']
	
	tokens = tokenizer.tokenize([sen, sen])
	print(tokens)   # ['这', '是', '一', '段', '测', '试', '文', '本', '这', '是', '一', '段', '测', '试', '文', '本']
	```
## 1.3 索引转换
- **只进行分词，得到的还是一些字符串和字符对象，还需要进行一步索引转换才能变成可计算的数值数据**。所谓索引转换，其实就是把分词结果一一替换为词表中的索引（称之为 token id），之后在做 embedding 的时候，这些 id 会先转换为 one-hot 向量，再通过线性层投影到嵌入空间（也称为 lookup table 操作），此后就可以在隐空间向量上进行注意力计算了
- 结合 1.2 节的分词和索引转换，完整的 tokenize 过程如下
	```python
	# Tokenize流程：原始字符串 -> token 序列 -> id 序列
	tokens = tokenizer.tokenize(sen)
	ids = tokenizer.convert_tokens_to_ids(tokens)
	print(ids)		# [6821, 3221, 671, 3667, 3844, 6407, 3152, 3315]
	
	# 也可以逆向操作：id 序列-> token 序列
	tokens = tokenizer.convert_ids_to_tokens(ids)
	print(tokens)	# ['这', '是', '一', '段', '测', '试', '文', '本']
	
	# 也可以逆向操作：token 序列 -> 字符串
	str_sen = tokenizer.convert_tokens_to_string(tokens)
	print(str_sen)	# 这 是 一 段 测 试 文 本
	```
- Transformers 库还提供称为 “编码” 和 “解法” 的简便方法，实现从原始字符串到 id 序列相互转换的一步操作
	```python
	# “编码”: 原始字符串 -> id 序列
	ids = tokenizer.encode(sen, add_special_tokens=True)        # add_special_tokens 在 tokenize 时序列设置特殊 token
	print(ids)                                                  # 注意到首尾多了特殊 token [CLS](101) 和 [SEP](102)
	# “解码”：id 序列 -> 原始字符串
	str_sen = tokenizer.decode(ids, skip_special_tokens=False)  # skip_special_tokens 可以跳过可能存在的特殊 token
	print(str_sen)
	str_sen = tokenizer.decode(ids, skip_special_tokens=True)
	print(str_sen)
	
	'''
	[101, 6821, 3221, 671, 3667, 3844, 6407, 3152, 3315, 102]
	[CLS] 这 是 一 段 测 试 文 本 [SEP]
	这 是 一 段 测 试 文 本
	'''
	```
	注意，在 `encode` 方法传入 `add_special_tokens` 参数；在 `decode` 方法传入 `skip_special_tokens` 参数，可以控制特殊 token 的引入和跳过
## 1.4 截断和填充
- 通常使用 batch 形式训练 Transformer 类模型，这要求我们把序列数据长度全部处理成和模型输入一致的状态。为此，需要进行截断或填充操作
	```python
	# 填充
	ids = tokenizer.encode(sen, padding="max_length", max_length=15)
	print(ids)  # [101, 6821, 3221, 671, 3667, 3844, 6407, 3152, 3315, 102, 0, 0, 0, 0, 0]
	
	# 截断
	ids = tokenizer.encode(sen, max_length=5, truncation=True)
	print(ids)  # [101, 6821, 3221, 671, 102]
	ids = tokenizer.encode(sen, max_length=5, truncation=False)	# 禁止截断则正常做 tokenize
	print(ids)  # [101, 6821, 3221, 671, 3667, 3844, 6407, 3152, 3315, 102]
	```
- 如上所示，通过在 `encode` 方法传入 `max_length` 参数控制最终序列长度，通过 `padding` 参数控制填充类型。注意到该 tokenizer 是在右侧进行 zero-padding 的，该设置可以在 1.1 节的 tokenizer 信息中观察到。另外可以通过 `truncation` 参数控制是否截断
- 需要注意的是，对于 BERT 类双向注意力模型来说，引入的 Padding token 会影响句子内容 logits 的计算，因此我们需要告诉模型的 attention layer 忽略 padding token。这是通过使用注意力掩码（attention mask）层来实现的，举例来说
	```python
	model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
	
	# batch 内 padding 到相同长度，logits 值被 padding token 影响
	sequence1_ids = [[200, 200, 200]]
	sequence2_ids = [[200, 200]]
	batched_ids = [
	    [200, 200, 200],
	    [200, 200, tokenizer.pad_token_id],
	]
	print(model(torch.tensor(sequence1_ids)).logits)
	print(model(torch.tensor(sequence2_ids)).logits)
	print(model(torch.tensor(batched_ids)).logits)
	
	# 引入注意力掩码，消除 padding token 影响
	attention_mask = [
	    [1, 1, 1],
	    [1, 1, 0],
	]
	outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
	print(outputs.logits)
	
	'''
	tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
	tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
	tensor([[ 1.5694, -1.3895],
	        [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)	# 不使用 attention mask，logits 值被影响
	tensor([[ 1.5694, -1.3895],
	        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)	# 使用 attention mask，logits 值不被影响
	'''
	```
## 1.5 高级封装
- 以上 1.2 到 1.5 节中，我们探索了 tokenizer 的运行机制，并且了解了分词、转换为 inputs ID、填充、截断和注意力掩码的处理方式
- Transformers API 提供了一个高级函数为我们处理所有这些工作，直接以 `tokenizer()` 形式调用即可
### 1.5.1 附加信息
- 除 token id 以外，Transformer 类模型的前向过程通常还需要一些附加信息，比如在 BERT 的上下句预训练任务中，不仅需要 attention_mask 遮盖 zero padding 的部分，还需要明确各个 token 所属的上下句信息。这些信息我们可以像 1.4 节一样手动构造
	```python
	ids = tokenizer.encode(sen, padding="max_length", max_length=15)
	
	# 除 token 外，Transformer 类模型的输入往往还有一些附加信息
	attention_mask = [1 if idx != 0 else 0 for idx in ids]  # attention_mask 用于遮盖 zero padding 部分
	token_type_ids = [0] * len(ids)                         # bert 有一个判断上下句任务，模型预训练时需要 token 所属句子 id 信息
	ids, attention_mask, token_type_ids
	```
	```shell
	([101, 6821, 3221, 671, 3667, 3844, 6407, 3152, 3315, 102, 0, 0, 0, 0, 0],
	 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	```
	 这些信息的生成方法已经被集成到 tokenizer 高级封装中
	```python	
	inputs = tokenizer(sen, padding="max_length", max_length=15)
	print(inputs)	# {'input_ids': [101, 6821, 3221, 671, 3667, 3844, 6407, 3152, 3315, 102, 0, 0, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}
	```
### 1.5.2 处理 batch 数据
- 前文 1.2 节提到过，tokenizer 支持字符串和字符串列表形式的输入，其中后者是为了 batch 数据而专门设计的，可以有效提高 tokenize 效率。基本使用如下
	```python
	sens = [
	    "AABBCCDDEEFF",
	    "哈哈哈哈哈哈哈哈哈哈哈",
	    "你好你好你好你好"
	]
	res = tokenizer(sens)	# batch tokenize 不要求各原始字符串长度一致
	res
	```
	```shell
	{'input_ids': [[101, 9563, 10214, 8860, 9879, 8854, 9049, 102], [101, 1506, 1506, 1506, 1506, 1506, 1506, 1506, 1506, 1506, 1506, 1506, 102], [101, 872, 1962, 872, 1962, 872, 1962, 872, 1962, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
	```
- 对比单条操作+循环和成批量操作的时间消耗
	```python
	%%time
	# 单条循环处理，慢
	for i in range(1000):
	    tokenizer(sen)
	
	'''
	CPU times: total: 172 ms
	Wall time: 242 ms
	'''
	```
	```python
	%%time
	# 成 batch 批量计算，快
	tokenizer([sen] * 1000)
	
	'''
	CPU times: total: 78.1 ms
	Wall time: 27.9 ms
	'''
	```
### 1.5.3 填充、截断和张量类型转换
- 高级封装提供以上 1.2-1.4 描述的所有填充、截断功能，且能以指定类型返回数据
	```python
	from transformers import AutoTokenizer
	
	checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	
	sequences = [
	    "I've been waiting for a HuggingFace course my whole life.", 
	    "So have I!"
	]
	
	# 使用多种方式进行填充
	model_inputs = tokenizer(sequences, padding="longest")      # 将句子序列填充到最长句子的长度
	print(len(model_inputs["input_ids"][0]))                    # 16
	model_inputs = tokenizer(sequences, padding="max_length")   # 将句子序列填充到模型的最大长度 (512 for BERT or DistilBERT)
	print(len(model_inputs["input_ids"][0]))                    # 512
	model_inputs = tokenizer(sequences, padding="max_length", max_length=8) # 将句子序列填充到指定的最大长度
	print(len(model_inputs["input_ids"][0]))                                # 16
	
	# 使用多种方式进行截断
	model_inputs = tokenizer(sequences, truncation=True)                # 截断比模型最大长度长的句子序列 (512 for BERT or DistilBERT)
	model_inputs = tokenizer(sequences, max_length=8, truncation=True)  # 将截断长于指定最大长度的句子序列
	
	# 处理指定框架张量的转换
	model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")  # 返回 PyTorch tensors
	model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")  # 返回 TensorFlow tensors
	model_inputs = tokenizer(sequences, padding=True, return_tensors="np")  # 返回 NumPy arrays
	```

# 2. Fast/Slow Tokenizer
- Transformer 库提供了两种 tokenizer
   1. `FastTokenizer`: 基于 Rust 实现，速度快，可以提供更多附加信息，类型名有后缀 Fast
   2. `SlowTokenizer`: 基于 python 实现，速度慢
- 直接创建的 Tokenizer，**如果存在 Fast 类型，则默认都是 Fast 类型**
	```python
	sen = "快慢Tokenizer测试"
	fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
	fast_tokenizer # 类型名有后缀 Fast
	```
	
	```python
	BertTokenizerFast(name_or_path='uer/roberta-base-finetuned-dianping-chinese', vocab_size=21128, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
		0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	}
	```
- 构造 Tokenizer 时，可以通过传入 `use_fast=False` 强制构造 Slow Tokenizer
	```python
	# 设置 use_fast=False 来构造 SlowTokenizer
	slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)
	slow_tokenizer # 类型名无后缀 Fast
	```
	```shell
	BertTokenizer(name_or_path='uer/roberta-base-finetuned-dianping-chinese', vocab_size=21128, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
		0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
		103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	}
	```
	注意到 Tokenizer 对象类型名没有 Fast 后缀了。两种 Tokenizer 具有明显的速度差距
	```python
	%%time
	fast_tokenizer([sen] * 10000)
	
	'''
	CPU times: total: 1.02 s
	Wall time: 349 ms
	'''
	```
	
	```python
	%%time
	slow_tokenizer([sen] * 10000)
	
	'''
	CPU times: total: 2.89 s
	Wall time: 3.05 s
	'''
	```
- Fast Tokenizer 有时会返回一些额外信息，例如有时候原始输入字符串中的英文不会按字母分词，而是按词根词缀分词，这时相应的 token 会对应到原始字符串中的一个索引区域，Fast Tokenizer 可以通过设置 `return_offsets_mapping=True` 获取 token 和原始索引区域的对应信息
	```python
	sen = "快慢Tokenizer测试"
	inputs = fast_tokenizer(sen, return_offsets_mapping=True) # (只有 FastTokenizer 可以设置 return_offsets_mapping=True)
	print(sen)                       # 打印原始字符串
	print(inputs.word_ids())         # 打印各个 token 对应到原始字符串的 “词索引”，注意到原始字符串中 ”Tokenizer“ 这个词被拆成了4个token (只有 FastTokenizer 可以调用这个)
	print(inputs['offset_mapping'])  # offset_mapping 指示了各个 token 对应的原始字符串索引区域
	```
	```python
	快慢Tokenizer测试
	[None, 0, 1, 2, 2, 2, 2, 3, 4, None]
	[(0, 0), (0, 1), (1, 2), (2, 4), (4, 7), (7, 10), (10, 11), (11, 12), (12, 13), (0, 0)]
	```

# 3. 加载特殊 Tokenizer
- 有些开源模型的 Tokenizer 没有嵌入到 Transformers 库中，而是由作者在开源时于其远程仓库中提供，这种情况下 Tokenizer 的行为可能和 Transformers 库中其他 Tokenizer 的一般行为有所不同，直接加载这些模型会报错
	```python
	tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-13B-base", trust_remote_code=False)
	# ValueError: Loading Skywork/Skywork-13B-base requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.
	```
	这时，**需要在 `.from_pretrained` 方法中传入 `trust_remote_code=True` 对远程代码添加信任**，才能正常下载目标 tokenizer
	
	```python
	tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-13B-base", trust_remote_code=True)
	tokenizer
	```	
	```shell
	You are using the legacy behaviour of the <class 'transformers_modules.Skywork.Skywork-13B-base.bc35915066fbbf15b77a1a4a74e9b574ab167816.tokenization_skywork.SkyworkTokenizer'>. This means that tokens that come after special tokens will not be properly handled. 
	SkyworkTokenizer(name_or_path='Skywork/Skywork-13B-base', vocab_size=65519, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
		0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
		1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
		2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	}
	```
- 下载之后，可以用前文 1.1 节方法将其保存到本地
	