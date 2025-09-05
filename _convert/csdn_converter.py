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
            relative_path = f"MyBlog/img/{article_name}/{local_filename}"
            print(f"下载图片成功: {img_url} -> {relative_path}")
            return relative_path
            
        except Exception as e:
            print(f"下载图片失败: {img_url}, 错误: {e}")
            return img_url  # 返回原URL
    
    def process_images(self, markdown_content, article_name):
        """处理markdown中的图片"""
        # 匹配图片链接的正则表达式
        img_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        
        img_index = 1
        def replace_img(match):
            nonlocal img_index
            alt_text = match.group(1)
            img_url = match.group(2)
            
            # 只处理网络图片
            if img_url.startswith('http'):
                # 去掉URL中的参数和锚点
                clean_url = img_url.split('#')[0].split('?')[0]
                local_path = self.download_image(clean_url, article_name, img_index)
                img_index += 1
                return f"![{alt_text}]({local_path})"
            else:
                return match.group(0)  # 保持原样
        
        # 替换所有图片链接
        processed_content = re.sub(img_pattern, replace_img, markdown_content)
        return processed_content
    
    def extract_metadata_from_url(self, article_url):
        """从CSDN文章URL提取tags和categories"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(article_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取tags (带#前缀)
            tags = []
            tag_elements = soup.find_all('a', class_='tag-link')
            for tag_elem in tag_elements:
                tag_text = tag_elem.get_text().strip()
                if tag_text.startswith('#'):
                    tags.append(tag_text[1:])  # 去掉#前缀
            
            # 如果没找到tag-link类，尝试其他可能的选择器
            if not tags:
                # 尝试查找包含#的链接
                all_links = soup.find_all('a')
                for link in all_links:
                    text = link.get_text().strip()
                    if text.startswith('#') and len(text) > 1:
                        tags.append(text[1:])
            
            # 提取categories (包含"篇文章"字样)
            categories = []
            category_elements = soup.find_all('a')
            for cat_elem in category_elements:
                cat_text = cat_elem.get_text().strip()
                if '篇文章' in cat_text:
                    # 提取"xx 篇文章"前面的部分
                    category = re.sub(r'\s*\d+\s*篇文章.*', '', cat_text).strip()
                    if category:
                        categories.append(category)
            
            # 去重
            tags = list(set(tags))
            categories = list(set(categories))
            
            print(f"提取到的tags: {tags}")
            print(f"提取到的categories: {categories}")
            
            return tags, categories
            
        except Exception as e:
            print(f"提取元数据失败: {e}")
            return [], []
    
    def generate_front_matter(self, title, tags, categories):
        """生成Hexo Front Matter"""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d %H:%M:%S')
        
        front_matter = f"""---
title: {title}
date: {date_str}
"""
        
        if tags:
            front_matter += "tags:\n"
            for tag in tags:
                front_matter += f"  - {tag}\n"
        
        if categories:
            front_matter += "categories:\n"
            for category in categories:
                front_matter += f"  - {category}\n"
        
        front_matter += "---\n\n"
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
        
        # 生成Front Matter
        front_matter = self.generate_front_matter(title, tags, categories)
        
        # 组合最终内容
        final_content = front_matter + processed_content
        
        # 保存到_posts目录
        output_filename = f"{article_name}.md"
        output_path = os.path.join(self.posts_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"转换完成! 输出文件: {output_path}")
        return output_path

def main():
    """主函数"""
    # 配置
    markdown_file = "d:\\Programmer\\Hexo\\_convert\\raw\\实践\\经典机器学习方法(1)——线性回归.md"
    article_url = "https://blog.csdn.net/wxc971231/article/details/122869916"
    
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