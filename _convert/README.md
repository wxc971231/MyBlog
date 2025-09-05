# CSDN博客转换器

这是一个用于将CSDN博客文章转换为Hexo格式Markdown文件的自动化工具。

## 功能特性

- ✅ **文章内容爬取**：自动获取CSDN博客文章的完整内容
- ✅ **图片本地化**：下载文章中的图片到本地，避免外链失效
- ✅ **元数据提取**：自动提取标题、发布时间、作者、分类、标签等信息
- ✅ **Front Matter生成**：生成符合Hexo规范的Front Matter
- ✅ **代码块处理**：智能识别代码语言，保持代码格式
- ✅ **批量转换**：支持批量处理多篇文章
- ✅ **格式优化**：将HTML内容转换为标准Markdown格式
- ✅ **配置灵活**：支持自定义配置文件

## 文件结构

```
_convert/
├── csdn_converter.py    # 主转换脚本
├── batch_convert.py     # 批量转换脚本
├── config.yaml          # 配置文件
├── requirements.txt     # Python依赖包
├── README.md           # 使用说明
├── urls.txt            # URL列表文件（运行后自动生成）
└── conversion.log      # 转换日志（运行后生成）
```

## 安装依赖

在使用转换器之前，需要安装Python依赖包：

```bash
# 进入_convert目录
cd _convert

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 方法一：转换单篇文章

直接运行主脚本，它会转换示例文章：

```bash
python csdn_converter.py
```

或者修改脚本中的`test_url`变量为你要转换的文章URL。

### 方法二：批量转换多篇文章

1. 运行批量转换脚本（首次运行会创建`urls.txt`文件）：
   ```bash
   python batch_convert.py
   ```

2. 编辑`urls.txt`文件，添加要转换的CSDN文章URL：
   ```
   # CSDN文章URL列表
   # 每行一个URL，以#开头的行为注释
   https://blog.csdn.net/wxc971231/article/details/122869916
   https://blog.csdn.net/username/article/details/123456789
   https://blog.csdn.net/username/article/details/987654321
   ```

3. 再次运行批量转换脚本：
   ```bash
   python batch_convert.py
   ```

### 方法三：在Python代码中使用

```python
from csdn_converter import CSDNConverter

# 创建转换器实例
converter = CSDNConverter()

# 转换单篇文章
url = "https://blog.csdn.net/wxc971231/article/details/122869916"
converter.convert_article(url)

# 批量转换
urls = [
    "https://blog.csdn.net/username/article/details/123456789",
    "https://blog.csdn.net/username/article/details/987654321"
]
converter.convert_multiple_articles(urls)
```

## 配置说明

转换器支持通过`config.yaml`文件进行配置。主要配置项包括：

### 输出设置
- `posts_dir`: Markdown文件输出目录
- `images_dir`: 图片保存目录

### 网络设置
- `timeout`: 请求超时时间
- `delay`: 请求间隔时间
- `retry_count`: 重试次数

### 图片处理
- `download_images`: 是否下载图片到本地
- `max_size_mb`: 图片最大大小限制

### 元数据设置
- `auto_generate_tags`: 是否自动生成标签
- `tag_keywords`: 标签关键词映射

## 输出格式

转换后的Markdown文件包含完整的Front Matter：

```yaml
---
title: 经典机器学习方法（1）—— 线性回归
date: 2022-01-20 15:30:00
categories:
  - 机器学习
tags:
  - Python
  - 机器学习
  - 线性回归
description: 参考：动手学深度学习注：本文是 jupyter notebook 文档转换而来，部分代码可能无法直接复制运行！...
author: wxc971231
original_url: https://blog.csdn.net/wxc971231/article/details/122869916
---

# 线性回归

## 基础概念

线性回归，顾名思义，就是用线性模型来处理回归问题...
```

## 转换后的文件位置

- **Markdown文件**：保存在`../source/_posts/`目录下
- **图片文件**：保存在`../source/img/`目录下
- **文件命名**：格式为`YYYY-MM-DD-文章标题.md`

## 生成Hexo站点

转换完成后，返回Hexo根目录并生成站点：

```bash
# 返回Hexo根目录
cd ..

# 清理缓存
hexo clean

# 生成静态文件
hexo generate

# 启动本地服务器预览
hexo server
```

## 注意事项

1. **网络连接**：确保网络连接正常，能够访问CSDN网站
2. **请求频率**：脚本会自动添加延时，避免请求过于频繁
3. **图片下载**：大图片可能需要较长时间下载
4. **文件覆盖**：默认会跳过已存在的文件，可在配置中修改
5. **编码问题**：确保系统支持UTF-8编码

## 常见问题

### Q: 转换失败怎么办？
A: 检查以下几点：
- URL是否正确且可访问
- 网络连接是否正常
- 是否安装了所有依赖包
- 查看`conversion.log`日志文件获取详细错误信息

### Q: 图片无法下载？
A: 可能的原因：
- 图片链接失效
- 网络连接问题
- 图片过大（超过配置的大小限制）
- 图片格式不支持

### Q: 代码块格式不正确？
A: 脚本会尝试自动检测代码语言，如果检测不准确，可以：
- 手动修改生成的Markdown文件
- 在配置文件中调整代码检测规则

### Q: 如何自定义分类和标签？
A: 可以通过以下方式：
- 修改`config.yaml`中的`tag_keywords`映射
- 转换后手动编辑Markdown文件的Front Matter

## 技术实现

- **网络请求**：使用`requests`库发送HTTP请求
- **HTML解析**：使用`BeautifulSoup`解析HTML内容
- **内容转换**：自定义HTML到Markdown的转换逻辑
- **图片处理**：自动下载并重命名图片文件
- **元数据提取**：通过CSS选择器提取页面信息

## 扩展功能

如需添加新功能，可以修改`csdn_converter.py`文件：

1. **支持其他博客平台**：修改URL匹配和页面解析逻辑
2. **添加新的内容处理**：在`html_to_markdown`方法中添加新的HTML标签处理
3. **自定义文件名格式**：修改`generate_filename`方法
4. **添加新的元数据字段**：在`extract_metadata`方法中添加提取逻辑

## 许可证

本项目仅供学习和个人使用，请遵守CSDN网站的使用条款和robots.txt规定。

## 更新日志

- **v1.0.0** (2025-01): 初始版本，支持基本的CSDN文章转换功能