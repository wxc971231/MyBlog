#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSdn文章转换器 - 将CSDN文章转换为Hexo博客格式

功能:
1. 读取CSDN Markdown源码
2. 下载文章中的图片到本地
3. 从文章链接提取tags和categories
4. 生成Hexo Front Matter
5. 输出转换后的文章到source/_posts目录
"""

import os
import re
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
from datetime import datetime
import hashlib

class CSDNConverter:
    def __init__(self, base_dir="d:\\Programmer\\Hexo"):
        self.base_dir = base_dir
        self.source_dir = os.path.join(base_dir, "source")
        self.posts_dir = os.path.join(self.source_dir, "_posts")
        self.img_dir = os.path.join(self.source_dir, "img")
        
        # 确保目录存在
        os.makedirs(self.posts_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
    
    def generate_article_name(self, title):
        """生成文章名称：去掉空格，破折号变下划线"""
        # 去掉所有空格
        name = title.replace(" ", "")
        # 破折号变成下划线（处理中文破折号和英文破折号）
        name = name.replace("——", "_").replace("—", "_").replace("-", "_")
        # 去掉其他特殊字符
        name = re.sub(r'[^\w\u4e00-\u9fff_()]', '', name)
        return name
    
    def extract_title_from_file_path(self, file_path):
        """从文件路径中提取标题"""
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return filename
    
    def extract_title_from_markdown(self, markdown_content):
        """从markdown内容中提取标题（备用方法）"""
        lines = markdown_content.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                # 提取第一个一级标题
                title = line.strip()[2:].strip()
                # 去掉数字编号
                title = re.sub(r'^\d+\.\s*', '', title)
                return title
        return "未知标题"
    
    def download_image(self, img_url, article_name, img_index):
        """下载图片到本地"""
        try:
            # 创建文章图片目录
            article_img_dir = os.path.join(self.img_dir, article_name)
            os.makedirs(article_img_dir, exist_ok=True)
            
            # 获取图片扩展名
            parsed_url = urlparse(img_url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = '.png'  # 默认扩展名
            
            # 生成本地文件名
            local_filename = f"img_{img_index:03d}{ext}"
            local_path = os.path.join(article_img_dir, local_filename)
            
            # 下载图片
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(img_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            # 返回相对路径
            relative_path = f"/MyBlog/img/{article_name}/{local_filename}"
            print(f"下载图片成功: {img_url} -> {relative_path}")
            return relative_path
            
        except Exception as e:
            print(f"下载图片失败: {img_url}, 错误: {e}")
            return img_url  # 返回原URL
    
    def process_images(self, markdown_content, article_name):
        """处理markdown中的图片和内容缩进"""
        lines = markdown_content.split('\n')
        img_index = 1
        indent_level = 0  # 当前缩进级别计数器
        
        for i, line in enumerate(lines):
            # 更新缩进级别计数器
            if line.strip():
                # 计算当前行的缩进级别（每4个空格或1个tab为一级）
                current_indent = 0
                for char in line:
                    if char == ' ':
                        current_indent += 1
                    elif char == '\t':
                        current_indent += 4
                    else:
                        break
                current_level = current_indent // 4
                
                # 如果是列表项，更新缩进级别
                stripped_line = line.strip()
                if (stripped_line.startswith(('-', '*', '+')) or 
                    re.match(r'^\d+\.\s', stripped_line)):  # 处理有序列表和无序列表
                    # 列表项内容的缩进级别应该是当前列表项缩进级别 + 1
                    indent_level = current_level + 1
                elif line.strip().startswith('#'):  # 标题重置缩进级别
                    indent_level = 0
                elif current_level == 0:  # 顶级内容重置缩进级别
                    indent_level = 0
                else:
                    # 对于非列表项的普通内容，如果当前缩进不足，需要调整
                    # 但要排除列表项，避免错误处理有序列表和无序列表
                    if (current_level < indent_level and 
                        not stripped_line.startswith(('<', '>', '```')) and
                        not stripped_line.startswith(('-', '*', '+')) and
                        not re.match(r'^\d+\.\s', stripped_line)):
                        # 调整缩进到正确级别
                        base_indent = '    ' * indent_level
                        lines[i] = base_indent + stripped_line
            
            # 匹配图片链接的正则表达式，包括可能的缩放参数
            img_match = re.search(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
            if img_match:
                alt_text = img_match.group(1)
                img_url = img_match.group(2)
                
                # 只处理网络图片
                if img_url.startswith('http'):
                    # 提取缩放比例
                    scale_match = re.search(r'#pic_center\s*=\s*(\d+)%x', img_url)
                    if scale_match:
                        width_percent = int(scale_match.group(1))
                    else:
                        width_percent = 100  # 默认值
                    
                    # 去掉URL中的参数和锚点
                    clean_url = img_url.split('#')[0].split('?')[0]
                    local_path = self.download_image(clean_url, article_name, img_index)
                    img_index += 1
                    
                    # 使用HTML格式，居中对齐，根据缩进级别应用正确的缩进
                    if not alt_text:
                        alt_text = "图片"
                    
                    # 检查图片前面是否有代码块结束标记，如果有则使用代码块的缩进
                    img_indent_level = indent_level
                    if i > 0:
                        # 向前查找最近的代码块结束标记
                        for j in range(i-1, max(0, i-5), -1):
                            prev_line = lines[j].strip()
                            if prev_line == '```':
                                # 找到代码块结束，获取其缩进
                                code_block_indent = re.match(r'(\s*)', lines[j]).group(1)
                                img_indent_level = len(code_block_indent.replace('\t', '    ')) // 4
                                break
                            elif prev_line and not prev_line.startswith('#'):
                                # 遇到其他非空内容，停止查找
                                break
                    
                    # 根据缩进级别生成缩进字符串
                    base_indent = '    ' * img_indent_level
                    div_content = f'{base_indent}<div align="center">\n{base_indent}    <img src="{local_path}" alt="{alt_text}" style="width: {width_percent}%;">\n{base_indent}</div>\n'  # </div> 后要加空格，避免内联公式问题
                    lines[i] = div_content
                    
                    # 处理图片后续的文本行，确保它们有正确的缩进
                    for k in range(i+1, len(lines)):
                        next_line = lines[k]
                        if not next_line.strip():  # 跳过空行
                            continue
                        # 如果是新的列表项、标题或顶级内容，停止处理
                        stripped_next = next_line.strip()
                        if (stripped_next.startswith(('-', '*', '+', '#')) or
                            re.match(r'^\d+\.\s', stripped_next) or  # 处理有序列表
                            not next_line.startswith(' ') and not next_line.startswith('\t')):
                            break
                        # 如果行有内容但缩进不足，调整缩进
                        if (next_line.strip() and 
                            not next_line.strip().startswith('<') and  # 跳过HTML标签
                            not next_line.strip().startswith('>')): # 跳过引用块
                            # 计算当前行的缩进级别
                            current_indent = 0
                            for char in next_line:
                                if char == ' ':
                                    current_indent += 1
                                elif char == '\t':
                                    current_indent += 4
                                else:
                                    break
                            current_line_level = current_indent // 4
                            
                            # 如果缩进级别不足，调整到正确的级别
                            # 但要排除列表项，避免错误处理有序列表和无序列表
                            if (current_line_level < indent_level and
                                not next_line.strip().startswith(('-', '*', '+')) and
                                not re.match(r'^\d+\.\s', next_line.strip())):
                                content = next_line.strip()
                                lines[k] = base_indent + content
                        else:
                            break
        
        return '\n'.join(lines)
    
    def process_math_formulas(self, markdown_content):
        """处理数学公式，确保所有双$$符号都单独占一行，并保持缩进对齐"""
        lines = markdown_content.split('\n')
        processed_lines = []
        in_math_block = False
        math_block_indent = ''
        
        for line in lines:
            # 检查是否包含$$
            if '$$' in line:
                # 计算$$出现的次数
                dollar_count = line.count('$$')
                
                if not in_math_block:
                    # 不在数学块中，这是开始
                    # 提取$$前的缩进，将TAB替换为4个空格
                    match = re.match(r'(\s*).*?\$\$', line)
                    if match:
                        math_block_indent = match.group(1).replace('\t', '    ')
                    
                    if dollar_count == 1:
                        # 只有一个$$，这是多行数学公式的开始
                        if line.strip() == '$$':
                            # 整行只有$$
                            processed_lines.append(line.replace('\t', '    '))
                        else:
                            # $$后面还有内容，需要分行
                            before_dollars = line[:line.find('$$')].strip()
                            after_dollars = line[line.find('$$') + 2:].strip()
                            
                            if before_dollars:
                                processed_lines.append(math_block_indent + before_dollars)
                            processed_lines.append(math_block_indent + '$$')
                            if after_dollars:
                                processed_lines.append(math_block_indent + after_dollars)
                        in_math_block = True
                    
                    elif dollar_count >= 2:
                        # 两个或更多$$，需要逐个处理
                        current_line = line
                        line_indent = re.match(r'(\s*)', line).group(1).replace('\t', '    ')
                        
                        # 记录是否是列表项，用于后续缩进处理
                        is_list_item = (line.strip().startswith(('-', '*', '+')) or 
                                       re.match(r'^\s*\d+\.\s', line))
                        
                        # 逐个处理$$对
                        while current_line.count('$$') >= 2:
                            start_pos = current_line.find('$$')
                            end_pos = current_line.find('$$', start_pos + 2)
                            
                            if end_pos == -1:
                                break
                            
                            # 处理一对$$
                            before_formula = current_line[:start_pos]
                            formula_content = current_line[start_pos + 2:end_pos].strip()
                            remaining_text = current_line[end_pos + 2:]
                            
                            # 添加公式前的文本
                            if before_formula.strip():
                                processed_lines.append(line_indent + before_formula.strip())
                            
                            # 添加数学公式
                            if is_list_item:
                                formula_indent = line_indent + '    '  # 列表项内容缩进
                            else:
                                formula_indent = line_indent
                            
                            processed_lines.append(formula_indent + '$$')
                            if formula_content:
                                processed_lines.append(formula_indent + formula_content)
                            processed_lines.append(formula_indent + '$$')
                            
                            # 继续处理剩余部分
                            current_line = remaining_text
                            if is_list_item:
                                line_indent = formula_indent  # 后续内容保持列表项缩进
                        
                        # 处理最后剩余的文本
                        if current_line.strip():
                            if current_line.count('$$') == 1:
                                # 剩余一个$$，开始数学块
                                in_math_block = True
                                math_block_indent = line_indent
                                
                                start_pos = current_line.find('$$')
                                before_dollars = current_line[:start_pos].strip()
                                after_dollars = current_line[start_pos + 2:].strip()
                                
                                if before_dollars:
                                    processed_lines.append(line_indent + before_dollars)
                                processed_lines.append(line_indent + '$$')
                                if after_dollars:
                                    processed_lines.append(line_indent + after_dollars)
                            else:
                                # 没有$$或处理完毕，添加剩余文本
                                processed_lines.append(line_indent + current_line.strip())
                
                else:
                    # 在数学块中，这是结束
                    if line.strip() == '$$':
                        # 整行只有$$
                        processed_lines.append(math_block_indent + '$$')
                    else:
                        # $$前面还有内容，需要分行
                        before_dollars = line[:line.find('$$')].strip()
                        after_dollars = line[line.find('$$') + 2:].strip()
                        
                        if before_dollars:
                            processed_lines.append(math_block_indent + before_dollars)
                        processed_lines.append(math_block_indent + '$$')
                        if after_dollars:
                            processed_lines.append(math_block_indent + after_dollars)
                    
                    in_math_block = False
                    math_block_indent = ''
            
            else:
                # 不包含$$的行
                if in_math_block:
                    # 在数学块中，确保内容与开始$$对齐，将TAB替换为4个空格
                    content = line.strip()
                    if content:
                        processed_lines.append(math_block_indent + content)
                    else:
                        processed_lines.append('')  # 保持空行
                else:
                    # 普通行，将TAB替换为4个空格
                    processed_lines.append(line.replace('\t', '    '))
        
        return '\n'.join(processed_lines)
    
    def process_markdown_formatting(self, markdown_content):
        """处理引用块和加粗文本格式问题，以及代码块后图片的缩进问题"""
        lines = markdown_content.split('\n')
        processed_lines = []
        in_code_block = False
        code_block_indent = ''
        
        for i, line in enumerate(lines):
            # 检测代码块的开始和结束
            if line.strip().startswith('```'):
                if not in_code_block:
                    # 代码块开始，记录缩进
                    in_code_block = True
                    code_block_indent = re.match(r'(\s*)', line).group(1)
                else:
                    # 代码块结束
                    in_code_block = False
                processed_lines.append(line)
                continue
            
            # 如果刚结束代码块，检查下一行是否是图片div
            if (not in_code_block and 
                i > 0 and 
                lines[i-1].strip() == '```' and 
                line.strip().startswith('<div align="center">')):
                # 为代码块后的图片div添加相同的缩进
                processed_lines.append(code_block_indent + line.strip())
                continue
            
            # 如果是图片div内的img标签或结束标签，也要保持相同缩进
            if (not in_code_block and 
                code_block_indent and 
                (line.strip().startswith('<img ') or line.strip() == '</div>') and 
                i > 0 and 
                any(processed_lines[j].strip().startswith('<div align="center">') 
                    for j in range(max(0, len(processed_lines)-3), len(processed_lines)))):
                processed_lines.append(code_block_indent + line.strip())
                # 如果是结束标签，清除缩进记录
                if line.strip() == '</div>':
                    code_block_indent = ''
                continue
            
            # 处理引用块缩进问题
            if line.strip().startswith('>'):
                # 确保引用块前有正确的缩进
                line_indent = ''
                for char in line:
                    if char in [' ', '\t']:
                        line_indent += '    ' if char == '\t' else char
                    else:
                        break
                
                # 提取引用内容
                quote_content = line.strip()[1:].strip()  # 去掉 > 符号
                processed_lines.append(f'{line_indent}> {quote_content}')
            
            # 处理加粗文本内的数学公式
            elif '**' in line and '$' in line:
                # 查找加粗文本内的数学公式
                # 匹配 **文本 $公式$ 文本** 的模式
                bold_math_pattern = r'\*\*([^*]*?)\$([^$]+?)\$([^*]*?)\*\*'
                
                def replace_bold_math(match):
                    before_math = match.group(1).strip()
                    math_content = match.group(2)
                    after_math = match.group(3).strip()
                    # 将加粗文本内的数学公式分离出来，避免空段导致连续 '****'
                    parts = []
                    if before_math:
                        parts.append(f'**{before_math}**')
                    parts.append(f'${math_content}$')
                    if after_math:
                        parts.append(f'**{after_math}**')
                    return ''.join(parts)
                
                processed_line = re.sub(bold_math_pattern, replace_bold_math, line)
                processed_lines.append(processed_line)
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def extract_metadata_from_url(self, article_url):
        """从CSDN文章URL提取tags和categories（已禁用，返回空列表）"""
        # 不再自动提取，返回空列表让用户手动填写
        print("已禁用自动提取tags和categories，请手动填写")
        return [], []
    
    def generate_front_matter(self, title, tags, categories):
        """生成Hexo Front Matter"""
        # 生成当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建Front Matter，留空tags、categories、description供用户手动填写
        front_matter = f"""---
title: {title}
date: {current_time}
tags:
  - 
categories:
  - 
description: 
---\n\n"""
        
        return front_matter
    
    def convert_article(self, markdown_file_path, article_url):
        """转换文章"""
        print(f"开始转换文章: {markdown_file_path}")
        print(f"文章URL: {article_url}")
        
        # 读取markdown文件
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # 从文件路径提取完整标题
        title = self.extract_title_from_file_path(markdown_file_path)
        print(f"文章标题: {title}")
        
        # 生成文章名称
        article_name = self.generate_article_name(title)
        print(f"文章名称: {article_name}")
        
        # 提取元数据
        tags, categories = self.extract_metadata_from_url(article_url)
        
        # 如果没有从URL提取到categories，使用默认分类
        if not categories:
            categories = ["机器学习", "实践"]
        
        # 处理图片
        print("开始处理图片...")
        processed_content = self.process_images(markdown_content, article_name)
        
        # 处理数学公式
        print("开始处理数学公式...")
        processed_content = self.process_math_formulas(processed_content)
        
        # 处理引用块和加粗文本格式
        print("开始处理引用块和加粗文本格式...")
        processed_content = self.process_markdown_formatting(processed_content)
        
        # 生成Front Matter
        front_matter = self.generate_front_matter(title, tags, categories)
        
        # 添加首发链接
        first_publish_link = f"- 首发链接：[{title}]({article_url})\n"
        
        # 组合最终内容
        final_content = front_matter + first_publish_link + processed_content
        
        # 保存到_posts目录
        output_filename = f"{article_name}.md"
        output_path = os.path.join(self.posts_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"转换完成! 输出文件: {output_path}")
        return output_path

def main():
    # 配置文件路径和URL
    markdown_file = r"D:\Programmer\Hexo\_convert\raw\数学杂烩\小目标检测的尺寸极限.md"
    article_url = "https://blog.csdn.net/wxc971231/article/details/151802535"
    
    # 创建转换器
    converter = CSDNConverter()
    
    # 执行转换
    try:
        output_path = converter.convert_article(markdown_file, article_url)
        print(f"\n转换成功完成!")
        print(f"输出文件: {output_path}")
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()