- Hugging Face 是一家在 AI 领域具有重要影响力的科技公司，他们的开源工具和社区建设为NLP研究和开发提供了强大的支持。它们拥有当前最活跃、最受关注、影响力最大的 [NLP 社区](https://huggingface.co/)，最新最强的 NLP 模型大多在这里发布和开源。该社区也提供了丰富的教程、文档和示例代码，帮助用户快速上手并深入理解各类 Transformer 模型和 NLP 技术
- Transformers 库是 Hugging Face 最著名的贡献之一，它最初是 Transformer 模型的 pytorch 复现库，随着不断建设，至今已经成为 NLP 领域最重要，影响最大的基础设施之一。该库提供了大量预训练的模型，涵盖了多种语言和任务，成为当今大模型工程实现的主流标准，换句话说，**如果你正在开发一个大模型，那么按 Transformer 库的代码格式进行工程实现、将 check point 打包成 hugging face 格式开源到社区，对于推广你的工作有很大的助力作用**。本系列文章将介绍 Transformers 库的基本使用方法
- 参考：
	- [官方教程](https://huggingface.co/docs/transformers/index)
	- [手把手带你实战HuggingFace Transformers](https://www.bilibili.com/video/BV1KM4y1q7Js)
------
- datasets 是一个简单易用的数据集加载库，可方便地从本地或 hugging face hub 加载数据集
  - 开源数据集列表：https://huggingface.co/datasets
  - 文档地址：https://huggingface.co/docs/datasets/index
- 无论自定义还是从 Hugging Face Hud 下载，Transformers 库中的数据集 (Dataset) 是一个包含以下内容的目录：
	1. **一些通用格式数据文件**（如 JSON、CSV、Parquet、文本文件等）
	2. **一个数据加载脚本**，它定义一个 `datasets.GeneratorBasedBuilder`，用于从数据文件构造最终程序使用的 `datasets.arrow_dataset.Dataset` 对象。Transformers 库默认调研各类型文件的通用数据加载脚本，遇到以下复杂情况时则需自定义
		| 情况 | 说明  |
		|--|--|
		|复杂的数据结构 | 如嵌套的 JSON、特殊格式 |
		|多文件组合 | 需要从多个文件中组合数据 |
		|特殊预处理 | 需要在加载时进行数据清洗或转换 |
		|自定义字段映射 | 原始数据字段与期望格式不匹配 |

@[toc]
# 1. Datasets 的基本使用
## 1.1 加载在线数据集
- 使用 `datasets.load.load_dataset` 方法，可直接从 HF Hub 下载 `path` 形参指定的在线开源数据集
	```python
	from datasets import *
	datasets = load_dataset(path="madao33/new-title-chinese")
	datasets
	```
	
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['title', 'content'],
	        num_rows: 5850
	    })
	    validation: Dataset({
	        features: ['title', 'content'],
	        num_rows: 1679
	    })
	})
	```
	注意到在线数据集通常已进行划分并以字典形式呈现。可以传入 `split` 参数直接加载指定划分，且能通过切片方式加载指定数据
	```python
	# 只加载训练集
	dataset = load_dataset("madao33/new-title-chinese", split="train")
	
	# 用切片方式，只加载训练集的前100条数据
	dataset = load_dataset("madao33/new-title-chinese", split="train[:100]")  
	
	# 以列表形式加载多个数据集
	dataset = load_dataset("madao33/new-title-chinese", split=["train[50%:]", "train[:50%]", "validation[10:20]"])  
	```
- 有些数据集是多任务数据集，它们包含多个子任务，需要通过 `name` 形参指定加载哪个任务的数据。例如
	```python
	# 错误的用法
	super_glue_datasets = load_dataset(path="super_glue")  # ❌ 会报错
	'''
	ValueError: Config name is missing.
	Please pick one among the available configs: ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed']
	'''
	
	# 正确的用法 - 指定具体任务
	boolq_dataset = load_dataset("super_glue", name="boolq", trust_remote_code=True)	# 布尔问答
	copa_dataset = load_dataset("super_glue", name="copa", trust_remote_code=True)		# 因果推理
	```
	其中 `trust_remote_code=True` 代表使用 HF Hub 开源数据集自定义的数据集脚本
- 数据集、模型等默认下载到 `HF_HOME` 和 `HUGGINGFACE_HUB_CACHE` 等全局变量指定位置，可以调整
	```python
	# 全局设置存储目录
	import os
	os.environ['HF_HOME'] = r'D:\Programmer\HuggingFace'
	os.environ['HUGGINGFACE_HUB_CACHE'] = r'D:\Programmer\HuggingFace\Hub'
	
	# 在下载时指定存储目录
	datasets = load_dataset("madao33/new-title-chinese", cache_dir="D:/MyDatasets/cache")
	
	# 详细配置下载行为
	from datasets import DownloadConfig
	download_config = DownloadConfig(	# 创建下载配置
	    cache_dir="D:/MyDatasets/cache",
	    force_download=False,  			# 是否强制重新下载
	    resume_download=True,  			# 是否支持断点续传
	)
	datasets = load_dataset("madao33/new-title-chinese", download_config=download_config)
	```
## 1.2 查看数据集
- 数据通常以 `Dict[str, list]` 的字典形式保存，支持通过切片形式访问
	```python
	# 加载数据
	from datasets import *
	datasets = load_dataset("madao33/new-title-chinese")
	
	# 支持切片形式访问，字典形式（元素为列表）返回
	datasets['train'][:2]   
	'''
	{
		'title': ['望海楼美国打“台湾牌”是危险的赌博', '大力推进高校治理能力建设'],
		'content': ['近期，美国国会众院通过法案...', '在推进“双一流”高校建设进程中...']
	}
	'''
	
	# 按字段访问，便于做 batch tokenize
	datasets['train']['title'][:2]  
	'''
	['望海楼美国打“台湾牌”是危险的赌博', '大力推进高校治理能力建设']
	'''
	
	# 获取字段名和datasets.features.features.Features 对象
	print(datasets['train'].column_names)   # ['title', 'content']
	print(datasets['train'].features)       # {'title': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}
	```
## 1.3 划分数据集
- 有些数据集未进行原始划分或划分不满足要求，这时可以使用 `dataset.train_test_split` 方法进行划分。**对任意 `Dataset` 对象调用此方法，会返回一个由 train 和 test 构成的 `DatasetDict`**
	```python
	datasets = load_dataset("madao33/new-title-chinese")
	dataset = datasets['train']
	
	# 将原本的 'train' 数据集再次按比例划分，10%做测试集，90%做训练集
	final_datasets = dataset.train_test_split(test_size=0.1) 
	final_datasets
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['title', 'content'],
	        num_rows: 5265
	    })
	    test: Dataset({
	        features: ['title', 'content'],
	        num_rows: 585
	    })
	})
	```
- 下例演示如何将 "madao33/new-title-chinese" 的 train 数据集重新划分为 train、test、valid 三部分
	```python
	datasets = load_dataset("madao33/new-title-chinese")
	dataset = datasets['train']
	
	# 先划分出训练集，train 占 80%
	train_test = dataset.train_test_split(test_size=0.2)
	# 把占 20% 的 test 对半分，作为 test 和 valid
	test_val = train_test['test'].train_test_split(test_size=0.5)
	
	# 重新组织数据集
	final_datasets = DatasetDict({
	    'train': train_test['train'],      # 80%
	    'test': test_val['train'],         # 10%
	    'validation': test_val['test']     # 10%
	})
	final_datasets
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['title', 'content'],
	        num_rows: 4680
	    })
	    test: Dataset({
	        features: ['title', 'content'],
	        num_rows: 585
	    })
	    validation: Dataset({
	        features: ['title', 'content'],
	        num_rows: 585
	    })
	})
	```
- 如果只是为了划分数据，用 1.1 节介绍的切分加载形式也能做到， `dataset.train_test_split` 方法的意义在于其可以进行更精细的控制。例如**对二分类任务数据集 BoolQ 来说，我们希望划分后的 train 和 test 都具有相同的正负样本比例，这就需要设置 `stratify_by_column` 对指定列（标签）进行分层采样**
	```python
	# super_glue 是一个多任务数据集合，只加载其中 boolq 任务的数据
	boolq_dataset = load_dataset("super_glue", "boolq", trust_remote_code=True) 
	dataset = boolq_dataset['train']
	
	# 按比例划分，同时确保给定字段 'label' 的取值在数据集中是均衡的
	final_datasets = dataset.train_test_split(test_size=0.1, stratify_by_column='label') 
	final_datasets
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['title', 'content'],
	        num_rows: 4680
	    })
	    test: Dataset({
	        features: ['title', 'content'],
	        num_rows: 585
	    })
	    validation: Dataset({
	        features: ['title', 'content'],
	        num_rows: 585
	    })
	})
	```
## 1.4 数据选取、过滤和打乱
- 前文 1.2 节说明了可以通过切片方式查看数据，注意其会返回普通 python 字典或列表
	```python
	from datasets import *
	datasets = load_dataset("madao33/new-title-chinese")
	
	# 支持切片形式访问，字典形式（元素为列表）返回
	datasets['train'][:2]   
	'''
	{
		'title': ['望海楼美国打“台湾牌”是危险的赌博', '大力推进高校治理能力建设'],
		'content': ['近期，美国国会众院通过法案...', '在推进“双一流”高校建设进程中...']
	}
	'''
	```
	以上方式适合快速查看数据，但有时我们想获取一个子数据集对象，类似 1.1 节切片加载的效果。这时可以**使用 `dataset.select()` 方法，这会通过索引引用方式创建新的 Dataset 对象，从而保持所有 Dataset 的方法和属性**
	```python
	dataset = load_dataset("madao33/new-title-chinese", split="train")
	
	# 取给定索引位置的样本，创建新 datset 对象（注意和 datasets['train'][:2] 这种查看方式不同，后者不创建 dataset 对象）
	dataset.select([0,1])   
	```
	```shell
	Dataset({
	    features: ['title', 'content'],
	    num_rows: 2
	})
	```
- 在数据预处理时，通常需要从大数据集中筛选出符合特定条件的高质量数据，这时**可用 `dataset.filter()` 方法返回符合条件数据构成的数据集对象**
	```python
	dataset = load_dataset("madao33/new-title-chinese", split="train")
		
	# 使用 lambda 函数作为条件，过滤数据集（这个不是in-place的）
	filter_dataset = dataset.filter(lambda example: "中国" in example['title'])  
	filter_dataset['title'][:5]
	```
	```shell
	['聚焦两会，世界探寻中国成功秘诀',
	 '望海楼中国经济的信心来自哪里',
	 '“中国奇迹”助力世界减贫跑出加速度',
	 '和音瞩目历史交汇点上的中国',
	 '中国风采感染世界']
	```
	可通过批处理或多进程方式提高性能
	```python
	dataset = load_dataset("madao33/new-title-chinese", split="train")
	
	# 对于大数据集，可以使用批处理模式提高效率
	def batch_filter_function(examples):
	    return ["中国" in title for title in examples['title']]
	filtered_dataset = dataset.filter(batch_filter_function, batched=True)
	
	# 即使处理方法不支持 batch 计算，还可以用多进程加速
	filtered_dataset = dataset.filter(
	    lambda x: "中国" in x['title'],
	    num_proc=4  # 使用4个进程
	)
	```
- **使用 `Dataset.shuffle()` 可以简单地打乱数据集**，通过链接 Dataset.shuffle() 和 Dataset.select() 函数可以快速创建一个随机的数据子集
	```python
	dataset.shuffle(seed=42).select(range(1000))
	```
## 1.5 数据映射
- `dataset.map()` 是 Hugging Face Datasets 库中最重要的 数据转换方法 ，用于**对数据集中的每个样本应用自定义的处理函数**
	```python
	dataset = load_dataset("madao33/new-title-chinese", split="train")
	
	# 数据映射.map方法支持我们定义一个样本处理函数，使用它处理数据集中的每一个样本
	def add_prefix(example):
	    example['title'] = "Prefix: " + example['title']
	    return example
	    
	prefix_datset = dataset.map(add_prefix)
	prefix_datset[:2]['title']
	```
	```shell
	['Prefix: 望海楼美国打“台湾牌”是危险的赌博',
	'Prefix: 大力推进高校治理能力建设',]
	```
- **`dataset.map()` 主要用于配合 tokenizer 完成高效的数据预处理**
	```python
	# 数据映射功能主要是结合 tokenizer 使用的，方便进行数据预处理
	from transformers import AutoTokenizer
	from datasets import *
	tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
	datasets = load_dataset("madao33/new-title-chinese")	
	
	# 利用 map 方法处理 DatasetDict 中各数据集的所有数据
	def preprocess_function(example, tokenizer):
	    model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
	    labels = tokenizer(example["title"], max_length=32, truncation=True)
	    model_inputs["labels"] = labels["input_ids"]    # 摘要任务，title 的编码结果作为 label
	    return model_inputs

	processed_dataset = datasets.map(lambda example: preprocess_function(example, tokenizer))
	processed_dataset   # 处理后的数据集中增加了 'input_ids', 'token_type_ids', 'attention_mask', 'labels' 等字段
	```
	```python
	DatasetDict({
	    train: Dataset({
	        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
	        num_rows: 5850
	    })
	    validation: Dataset({
	        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
	        num_rows: 1679
	    })
	})
	```
- 可通过批处理或多进程方式提升效率
	```python
	# 调用 .map() 方法时设置 batched=True 启动批量处理
	# 以上 preprocess_function 只做了 tokenize，当 Tokenizer 有 fast 实现时，以 batch 形式进行处理会更快
	processed_dataset = datasets.map(
	    lambda examples: preprocess_function(examples, tokenizer),
	    batched=True
	)
	
	# 调用 .map() 方法时设置 num_proc=n 启动多进程处理
	# 当处理方法 preprocess_function 内含有不支持 batch 的方法时，还可以用多进程加速
	# 这里需注意使用 partial，否则子进程无法访问在主进程中定义的 preprocess_function 函数，除非 preprocess_function 定义为全局函数，无需传入 example 以外的参数
	from functools import partial
	processed_dataset = datasets.map(
	    partial(preprocess_function, tokenizer=tokenizer),
	    num_proc=4
	)
	```
- 调用 `dataset.map()` 方法时，可**在 `remove_columns` 参数中设置要去除的字段列表**
	```python
	# 调用 .map() 方法时，在 remove_columns 参数中设置要去除的字段列表
	# 常用此方式去除数据的原始字段
	processed_datasets = datasets.map(
	    lambda example: preprocess_function(example, tokenizer), 
	    batched=True, 
	    remove_columns=datasets["train"].column_names
	)
	processed_datasets
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
	        num_rows: 5850
	    })
	    validation: Dataset({
	        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
	        num_rows: 1679
	    })
	})
	```
## 1.6 数据集的本地保存和加载
- 1.5 节中数据处理往往很耗时，可以用 `dataset.save_to_disk` 和 `dataset.load_from_disk` 直接把处理好的 `DatasetDict` 序列化保存到本地，或从本地加载使用
	```python
	# 保存到指定路径
	processed_datasets.save_to_disk('./saved/processed_datasets')
	# 从本地加载
	processed_datasets = load_from_disk("./saved/processed_datasets")
	```
	这会在指定路径创建具有以下结构的目录：
	```python
	processed_datasets/
	├── dataset_dict.json
	├── train
	│   ├── data-00000-of-00001.arrow
	│   ├── dataset_info.json
	│   └── state.json
	└── validation
	    ├── data-00000-of-00001.arrow
	    ├── dataset_info.json
	    └── state.json
	```
	注意到每个部分都有 `.arrow` 表以及保存元数据的 `dataset_info.json` 和 `state.json` 。**可以将 Arrow 格式视为一个优化过的列和行的精美表格，它针对构建处理和传输大型数据集的高性能应用程序进行了优化**
- 此外，还可以将数据保存为 csv 或 json 格式。这时我们必须将每个部分存储为单独的文件，一种方法是遍历 DatasetDict 中的键和值
	```python
	# 这将把每个部分保存为 JSON Lines格式 ，其中数据集中的每一行都存储为一行 JSON
	for split, dataset in processed_dataset.items():
	    dataset.to_json(f"./saved/processed_json/{split}.jsonl")
	```
	使用下文 2.1 节所述方法直接加载多个 json 文件
	
	```python
	data_files = {
	    "train": "./saved/processed_json/train.jsonl",
	    "validation": "./saved/processed_json/validation.jsonl",
	}
	processed_dataset = load_dataset("json", data_files=data_files)
	```

# 2. 加载本地数据集
- Datasets 提供了加载本地数据集的方法。它支持几种常见的数据格式。对于每种数据格式，我们只需要在 `load_dataset()` 函数中指定数据的类型，并使用 data_files 指定一个或多个文件的路径的参数
	| 数据格式 | 类型参数 | 加载的指令  |
	|--|--|--|
	|CSV & TSV	|csv|	load_dataset("csv", data_files="my_file.csv")|
	|Text files	|text|	load_dataset("text", data_files="my_file.txt")| 
	|JSON & JSON Lines	|json|	load_dataset("json", data_files="my_file.jsonl")|
	|Pickled DataFrames	|pandas|	load_dataset("pandas", data_files="my_dataframe.pkl")|

## 2.1 加载 csv 文件
- 首先准备 csv 文件 `ChnSentiCorp_htl_all.csv`，这是一个酒店评分数据，包含 label 和 review 两个字段
	```cvs
	label,review
	1,"距离川沙公路较近,但是公交指示不对,如果是""蔡陆线""的话,会非常麻烦.建议用别的路线.房间较为简单."
	1,商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!
	...
- 利用多态实现，前文用于加载在线开源数据集的 `load_dataset` 方法也可以直接加载本地文件
	```python
	from datasets import *
	
	# 直接加载默认得到一个 DatasetDict 对象，加载的数据集为其中的 'train' 数据集
	dataset = load_dataset(path='csv', data_files=f'./ChnSentiCorp_htl_all.csv')
	dataset 
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['label', 'review'],
	        num_rows: 7766
	    })
	})
	```

	```python
	# 指定 split="train" 加载，得到 Dataset 对象
	dataset = load_dataset('csv', data_files=f'./ChnSentiCorp_htl_all.csv', split='train')
	dataset 
	```
	```shell
	Dataset({
	    features: ['label', 'review'],
	    num_rows: 7766
	})
	```
- 可通过列表方式合并加载多个 .csv 文件
	```python
	def get_filepath_in_floder(floder_path):
	    filepathes = os.listdir(floder_path)  # 获取所有文件名称
	    filepathes = [os.path.join(floder_path, file) for file in filepathes]
	    return filepathes
	
	# 以列表形式加载多个 csv，会得到合并数据对应的 dataset
	dataset = load_dataset("csv", data_files=get_filepath_in_floder(f"./all_data"), split='train')
	dataset 
	```
	```shell
	Dataset({
	    features: ['label', 'review'],
	    num_rows: 23298
	})
	```
- **可用类方法 `Dataset.from_csv()`  加载 csv 文件**，效果和 `load_dataset` 等价
	```python
	dataset = Dataset.from_csv(f"./ChnSentiCorp_htl_all.csv")
	```
## 2.2 加载 pandas 对象
- **可用类方法 `Dataset.from_pandas()`  把 `pandas.core.frame.DataFrame` 直接转为 Dataset**
	```python
	import pandas as pd
	
	data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
	dataset = Dataset.from_pandas(data)
	```
## 2.3 加载 python list 对象
-  **可用类方法 `Dataset.from_list()`  把 python list 直接转为 Dataset**
	```python
	# List格式的数据需要内嵌{}，明确数据字段
	data = [{"text": "abc", "label": 1}, {"text": "def", "label": 2}]
	Dataset.from_list(data)
	```
	
	```shell
	Dataset({
	    features: ['text', 'label'],
	    num_rows: 2
	})
	```
## 2.4 使用自定义数据加载脚本
- 本节我们处理一个具有复杂嵌套结构的 json 数据集 `cmrc2018_trial.json`，其由一系列问答段落组成
	```json
	{
	  "version": "v1.0", 
	  "data": [
	    {
	      "paragraphs": [
	        {
	          "id": "TRIAL_800", 
	          "context": "基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师) ，可选择经典、热血、狙击等模式进行游戏。若游戏中离，则4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。", 
	          "qas": [
	            {
	              "question": "生命数耗完即算为什么？", 
	              "id": "TRIAL_800_QUERY_0", 
	              "answers": [
	                {
	                  "text": "踢爆", 
	                  "answer_start": 127
	                }
	              ]
	            }, 
	            {
	              "question": "若游戏中离，则多少分钟内不得进行配对？", 
	              "id": "TRIAL_800_QUERY_1", 
	              "answers": [
	                {
	                  "text": "4分钟", 
	                  "answer_start": 301
	                }
	              ]
	            }, 
	            {
	              "question": "玩家用什么泡封敌人？", 
	              "id": "TRIAL_800_QUERY_2", 
	              "answers": [
	                {
	                  "text": "玩家以水枪、小枪、锤子或是水炸弹泡封敌人", 
	                  "answer_start": 85
	                }
	              ]
	            }, 
	            {
	              "question": "游戏的模式有哪些？", 
	              "id": "TRIAL_800_QUERY_3", 
	              "answers": [
	                {
	                  "text": "可选择经典、热血、狙击等模式进行游戏。", 
	                  "answer_start": 275
	                }
	              ]
	            }
	          ]
	        }
	      ], 
	      "id": "TRIAL_800", 
	      "title": "泡泡战士"
	    }, 
	    ...
	```
	 这个 json 可\直接 `load_dataset` 加载，通过 `filed` 形参指定要加载的字段
	```python
	# field 用于指定 json 文件中包含数据集的字段名
	load_dataset("json", data_files="./cmrc2018_trial.json", field="data")
	```
	```shell
	DatasetDict({
	    train: Dataset({
	        features: ['paragraphs', 'id', 'title'],
	        num_rows: 256
	    })
	})
	```
- 默认行为不符合预期，通过继承 `GeneratorBasedBuilder` 自定义 dataset builder，把段落中的问答对作为数据样本
	```python
	import json
	import datasets
	from datasets import DownloadManager, DatasetInfo
	
	class CMRC2018TRIAL(datasets.GeneratorBasedBuilder):
	    def _info(self) -> DatasetInfo:
	        """
	            info方法, 定义数据集的信息,这里要对数据的字段进行定义
	        :return:
	        """
	        return datasets.DatasetInfo(
	            description="CMRC2018 trial",
	            features=datasets.Features({
	                    "id": datasets.Value("string"),
	                    "context": datasets.Value("string"),
	                    "question": datasets.Value("string"),
	                    "answers": datasets.features.Sequence(
	                        {
	                            "text": datasets.Value("string"),
	                            "answer_start": datasets.Value("int32"),
	                        }
	                    )
	                })
	        )
	
	    def _split_generators(self, dl_manager: DownloadManager):
	        """
	            返回datasets.SplitGenerator
	            涉及两个参数: name和gen_kwargs
	            name: 指定数据集的划分
	            gen_kwargs: 指定要读取的文件的路径, 与_generate_examples的入参数一致
	        :param dl_manager:
	        :return: [ datasets.SplitGenerator ]
	        """
	        return [datasets.SplitGenerator(
	            name=datasets.Split.TRAIN, 
	            gen_kwargs={"filepath": "./cmrc2018_trial.json"})
	        ]
	
	    def _generate_examples(self, filepath):
	        """
	            生成具体的样本, 使用yield
	            需要额外指定key, id从0开始自增就可以
	        :param filepath:
	        :return:
	        """
	        # Yields (key, example) tuples from the dataset
	        with open(filepath, encoding="utf-8") as f:
	            data = json.load(f)
	            for example in data["data"]:
	                for paragraph in example["paragraphs"]:
	                    context = paragraph["context"].strip()
	                    for qa in paragraph["qas"]:
	                        question = qa["question"].strip()
	                        id_ = qa["id"]
	
	                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
	                        answers = [answer["text"].strip() for answer in qa["answers"]]
	
	                        yield id_, {
	                            "context": context,
	                            "question": question,
	                            "id": id_,
	                            "answers": {
	                                "answer_start": answer_starts,
	                                "text": answers,
	                            },
	                        }
	```
	使用它加载数据集，需要在 `load_dataset` 的 `path` 形参传入加载脚本路径，并 `trust_remote_code`
	```python
	dataset = load_dataset("./load_script.py", split="train",  trust_remote_code=True)
	dataset
	```
	```shell
	Dataset({
	    features: ['id', 'context', 'question', 'answers'],
	    num_rows: 1002
	})
	```
# 3. DataCollector
- Transformers 库中，**`DataCollator` 是一个用于处理模型输入数据的实用工具。它通常用于将数据进行批处理、填充、截断或者任何其他处理**。利用DataCollator，可以更方便地构造torch.utils.data.Dataloader，以便在训练模型时有效地处理输入数据。具体而言，DataCollator 的作用如下
    1. **批处理处理**: 将输入数据按照模型的要求组合成 batch，以便进行训练
    2. **填充与截断**: 通过填充和截断处理不同长度的输入数据。这对于需要固定长度的模型输入非常重要
    3. **特殊处理**: 在某些情况下，为了满足模型输入的特殊要求，DataCollator 可能需要进行特殊处理，比如添加特殊的标记或者标签
- **DataCollator 只能自动处理 hf 格式的数据，限于以下字段，当自定义数据很复杂不止这些字段时，不建议使用** 
   - `'input_ids'`
   - `'token_type_ids'`
   - `'attention_mask'`
   - `'labels'`
  
 - 不使用 Dataset 库时，我们使用 Pytorch 原生的 Dataset 和 Dataloader 进行数据集构造。**需定义 `collate_func()` 对 dataloader 得到的 batch data 进行后处理** 
	```python
	# 不用 DataCollator 时，从 ChnSentiCorp_htl_all 数据构造 Dataloader 的方法如下
	import torch
	import os
	from torch.utils.data import Dataset, DataLoader, random_split
	
	class MyDataset(Dataset):
	    def __init__(self) -> None:
	        super().__init__()
	        self.data = pd.read_csv(f"{os.getcwd()}/ChnSentiCorp_htl_all.csv")    # 加载原始数据
	        self.data = self.data.dropna()                                        # 去掉脏数据, 去掉 nan 值
	
	    def __getitem__(self, index):
	        text:str = self.data.iloc[index]["review"]
	        label:int = self.data.iloc[index]["label"]
	        return text, label
	    
	    def __len__(self):
	        return len(self.data)
	    
	def collate_func(batch):
	    # 对 dataloader 得到的 batch data 进行后处理
	    # batch data 是一个 list，其中每个元素是 (sample, label) 形式的元组
	    texts, labels = [], []
	    for item in batch:
	        texts.append(item[0])
	        labels.append(item[1])
	    
	    # 对原始 texts 列表进行批量 tokenize，通过填充或截断保持 token 长度为 128，要求返回的每个字段都是 pytorch tensor
	    global tokenizer
	    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
	
	    # 增加 label 字段
	    inputs["labels"] = torch.tensor(labels)
	    return inputs
	
	
	# 构造 torch.utils.data.Dataset，划分训练和测试集
	dataset = MyDataset()
	train_size = int(0.9*len(dataset))
	vaild_size = len(dataset) - train_size
	trainset, validset = random_split(dataset, lengths=[train_size, vaild_size])
	
	# 构造 torch.utils.data.Dataloader，在 collate_func 中批量后处理（tokenize、truncation、padding）
	tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
	trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
	validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)
	```
- 使用 Dataset 库的 Dataset 和 DataCollector 可以更简洁地实现
	```python
	# 使用 DataCollator 完成同样的任务
	from datasets import *
	from transformers import DataCollatorWithPadding   # 该 Collator 会动态地对输入进行 padding 操作
	
	# 数据加载 & 清洗
	dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split='train')
	dataset = dataset.filter(lambda x: x["review"] is not None)
	
	# 批量预处理（tokenize & truncation，不做 padding）得到 datasets.arrow_dataset.Dataset
	def process_function(examples, tokenizer):
	    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
	    tokenized_examples["labels"] = examples["label"]
	    return tokenized_examples
	tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
	tokenized_dataset = dataset.map(lambda examples: process_function(examples, tokenizer), batched=True, remove_columns=dataset.column_names)
	
	# 划分训练/测试集
	datasets = tokenized_dataset.train_test_split(test_size=0.1)
	trainset, validset = datasets['train'], datasets['test']
	
	# 定义 collator 对象，将其作为 collate_fn 定义 torch.utils.data.Dataloader
	collator = DataCollatorWithPadding(tokenizer=tokenizer)
	trainloader = DataLoader(trainset, batch_size=4, shuffle=True, collate_fn=collator)
	validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collator)
	```
	注意这里我们设置 `tokenizer(examples["review"], max_length=128, truncation=True)` 并使用了 `DataCollatorWithPadding`，这会自动把 batch 中 `input_ids` 右侧 zero-padding 到 batch 内最大长度，并将所有长度超过 128 的序列截断到 128
	```python
	# 取 10 个 batch，观察 padding 长度
	for i, batch in enumerate(trainloader):
	    print(batch["input_ids"].size())
	    if i > 10:
	        break
	
	‘’‘
	torch.Size([4, 74])
	torch.Size([4, 87])
	torch.Size([4, 128])
	torch.Size([4, 128])
	torch.Size([4, 128])
	torch.Size([4, 128])
	torch.Size([4, 127])
	torch.Size([4, 128])
	torch.Size([4, 128])
	torch.Size([4, 128])
	torch.Size([4, 128])
	‘’’
	```
# 4. 最佳实践
- 使用 IMDb 小型评论数据集的子集进行全流程实践，包括：
    1. 用 `load_dataset()` 加载数据，使用 `dataset.shuffle()` 打乱并用  `dataset.select()` 提取子集
    2. 用 `dataset.map()` 和 `dataset.filter()` 清洗数据
    3. 用 `dataset.map()` 补充特征
    4. 创建 `AutoTokenizer`，用 `dataset.map(tokenize_function, batched=True)` 进行高效批量分词 
    5. 用 `DataCollatorWithPadding` 动态填充
    6. 构造可直接用于训练的 `DataLoader`
- 示例代码如下
	```python
	from datasets import load_dataset, DatasetDict
	from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
	from torch.utils.data import DataLoader
	import torch
	import html
	
	# =============================================
	# 1. 加载数据（使用 Hugging Face 官方 IMDb 小样本）
	# =============================================
	raw_datasets = load_dataset("imdb")
	
	# 为了演示，随机抽样 200 条训练 + 100 条测试
	small_train = raw_datasets["train"].shuffle(seed=42).select(range(200))
	small_test = raw_datasets["test"].shuffle(seed=42).select(range(100))
	
	print("✅ 原始样本示例：")
	print(small_train[0], "\n")
	
	# =============================================
	# 2. 清洗文本
	# =============================================
	def clean_text(example):
	    # HTML 转义解码 + 小写化
	    text = html.unescape(example["text"]).replace("<br />", " ").lower()
	    return {"text": text}
	
	small_train = small_train.map(clean_text).filter(lambda x: len(x["text"]) > 30)
	small_test = small_test.map(clean_text).filter(lambda x: len(x["text"]) > 30)
	
	# =============================================
	# 3. 增加特征列（Feature Engineering）
	#   - 字符长度
	#   - 词数
	#   - 平均词长
	# =============================================
	def add_features(example):
	    text = example["text"]
	    words = text.split()
	    char_len = len(text)
	    word_len = len(words)
	    avg_word_len = sum(len(w) for w in words) / (word_len + 1e-8)
	    return {
	        "char_len": char_len,
	        "word_len": word_len,
	        "avg_word_len": avg_word_len,
	    }
	
	small_train = small_train.map(add_features)
	small_test = small_test.map(add_features)
	print("✅ 数据列：", small_train.column_names)	# ['text', 'label', 'char_len', 'word_len', 'avg_word_len']
	
	# =============================================
	# 4. Tokenization（不做 padding）
	# =============================================
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	
	def tokenize_function(examples):
	    tokenized = tokenizer(examples["text"], truncation=True, max_length=128)
	    tokenized["labels"] = examples["label"]
	    # 保留数值特征列
	    tokenized["char_len"] = examples["char_len"]
	    tokenized["word_len"] = examples["word_len"]
	    tokenized["avg_word_len"] = examples["avg_word_len"]
	    return tokenized
	
	tokenized_train = small_train.map(tokenize_function, batched=True, remove_columns=["text"])
	tokenized_test = small_test.map(tokenize_function, batched=True, remove_columns=["text"])
	
	print("✅ Tokenized 示例：")
	print(tokenized_train[0].keys(), "\n")
	
	# =============================================
	# 5. 构造 DataCollator & Dataloader
	# =============================================
	collator = DataCollatorWithPadding(tokenizer=tokenizer)
	
	train_loader = DataLoader(tokenized_train, batch_size=8, shuffle=True, collate_fn=collator)
	valid_loader = DataLoader(tokenized_test, batch_size=8, shuffle=False, collate_fn=collator)
	
	# 查看 batch 样例
	batch = next(iter(train_loader))
	print("✅ 一个 batch 的字段：")
	for k, v in batch.items():
	    if isinstance(v, torch.Tensor):
	        print(f"{k:<15} -> shape {tuple(v.shape)}")
	    else:
	        print(f"{k:<15} -> type {type(v)}")
	
	
	```
	
	```shell
	✅ 原始样本示例：
	{'text': 'There is no relation at all between Fortier and Profiler but the fact that both are police series about violent crimes. Profiler looks crispy, Fortier looks classic. Profiler plots are quite simple. Fortier\'s plot are far more complicated... Fortier looks more like Prime Suspect, if we have to spot similarities... The main character is weak and weirdo, but have "clairvoyance". People like to compare, to judge, to evaluate. How about just enjoying? Funny thing too, people writing Fortier looks American but, on the other hand, arguing they prefer American series (!!!). Maybe it\'s the language, or the spirit, but I think this series is more English than American. By the way, the actors are really good and funny. The acting is not superficial at all...', 'label': 1} 
	
	✅ 数据列： ['text', 'label', 'char_len', 'word_len', 'avg_word_len']
	
	✅ Tokenized 示例：
	dict_keys(['label', 'char_len', 'word_len', 'avg_word_len', 'input_ids', 'token_type_ids', 'attention_mask', 'labels']) 
	
	✅ 一个 batch 的字段：
	char_len        -> shape (8,)
	word_len        -> shape (8,)
	avg_word_len    -> shape (8,)
	input_ids       -> shape (8, 128)
	token_type_ids  -> shape (8, 128)
	attention_mask  -> shape (8, 128)
	labels          -> shape (8,)
	```
