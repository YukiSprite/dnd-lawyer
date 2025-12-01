# -*- coding: utf-8 -*-

"""
阶段 2: 配置化打包 (package_json_final.py)

- 目标:
  1. 读取 'config.json' (人工审查后的版本)。
  2. 严格按照 config.json 中的规则 (action, split_by, selector) 递归处理。
  3. 从 .html 文件中提取内容。
  4. 使用 html2text 将 HTML 转换为保留结构 (表格、列表) 的 Markdown。
  5. 按照 "最终数据规范" 生成 'data/rules_data.json'。
- 设计哲学: "愚蠢的执行者" (Strict Executor)

- V3 改进:
  - 彻底抛弃基于HTML节点遍历的分割 (get_content_between_tags),
    因为它对HTML结构有“理想化”的假设。
  - 采用基于正则表达式 (Regex) 的分割方法, 直接在HTML字符串上操作。
  - 此方法极其健壮, 能处理 <split_by> 标签被任意嵌套的情况 (e.g., <div><h4>...</h4></div>)。
  - 保留所有内容, 包括第一个 <split_by> 标签前 (Chunk 0) 和
    最后一个 <split_by> 标签后的内容。

- V5 改进 (基于 V3):
  - 增加 clean_node_name 函数, 智能清理 config.json 中的键 (key)。
  - 解决因键名包含 "::" 或 "path/..." 导致的 full_path 重复问题。
  - 自动移除键名中的 .html / .htm 后缀。

- 2025-12-01 改进 (基于 V5):
  - 新增法术和专长文件的自动识别功能（仅 PF 版本）。
  - 法术文件自动按 <B><SPAN style="COLOR: #cc00cc">法术名</SPAN></B> 模式切分。
  - 专长文件自动按 <SPAN class=bbc_size style="FONT-SIZE: 12pt"><STRONG>专长名</STRONG></SPAN> 
    或 <SPAN class=bbc_color style="COLOR: brown"><STRONG>专长名</STRONG></SPAN> 模式切分。
  - 对于无法识别的专长文件，尝试按 <HR> 标签切分。
  - 每个法术/专长将被独立提取为一个文档条目，便于精准检索。
  - [修复] PF 版本始终使用 body 获取完整 HTML 内容，避免 select_one 只返回第一个匹配元素。
    之前的实现使用 select_one("p.MsoNormal") 只返回第一个 <p> 元素，导致大量内容丢失。
  - [新增] 蓝色小标题切分模式：自动检测 <B><SPAN style="COLOR: #0033cc">标题名</SPAN></B> 格式的小标题。
    适用于炫技、神话能力、变体能力等带有蓝色小标题的页面。
    当页面包含 2 个以上蓝色小标题且内容超过 5KB 时自动启用切分。
  - DND 版本保持原有逻辑不变。
"""

import os
import sys
import json
from pathlib import Path
from bs4 import BeautifulSoup, Tag
import html2text
import chardet  # 用于编码检测
import logging
import re  # 导入正则表达式
from typing import List, Dict, Any, Optional
from urllib.parse import unquote  # 用于 URL 解码

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入版本设置
from config.settings import get_current_version, is_pf, is_dnd

# --- 全局配置 ---

# 数据处理配置文件（由 analyze_chm.py 自动生成）
CONFIG_FILE = Path("./config/data_processing.json")

# 单个文档的最大字符数阈值（超过此值将尝试自动切分）
# 降低到 20K 以避免上下文超限（4个文档 * 20K ≈ 50K tokens，远低于128K限制）
MAX_DOC_SIZE = 20000  # 20KB，约 2 万字符（≈13K tokens）

# 单个切分块的最小字符数（低于此值认为切分过细）
MIN_CHUNK_SIZE = 100  # 100 字符，约 60 tokens

# [2025-12-02 新增] 切分块的硬性约束
# 如果切分后任何块小于 MIN 或大于 MAX，则回退到按固定大小切分
CHUNK_SIZE_MIN = 100    # 最小 100 字符
CHUNK_SIZE_MAX = 10000  # 最大 10000 字符（10KB）

# HTML 标题标签优先级（从高到低）
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# --- 法术和专长的标题识别模式 ---
# 法术标题模式: <B><SPAN style="...COLOR: #cc00cc...">法术名 (English Name)</SPAN></B>
SPELL_TITLE_PATTERN = re.compile(
    r'(<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#cc00cc[^>]*>.*?</SPAN>\s*</B>)',
    re.IGNORECASE | re.DOTALL
)

# 专长标题模式1: <SPAN class=bbc_size style="FONT-SIZE: 12pt"><STRONG>专长名</STRONG>
FEAT_TITLE_PATTERN_1 = re.compile(
    r'(<SPAN\s+class=bbc_size\s+style="FONT-SIZE:\s*12pt">\s*<STRONG>.*?</STRONG>.*?</SPAN>)',
    re.IGNORECASE | re.DOTALL
)

# 专长标题模式2: <SPAN class=bbc_color style="COLOR: brown"><STRONG>
FEAT_TITLE_PATTERN_2 = re.compile(
    r'(<SPAN\s+class=bbc_color\s+style="COLOR:\s*brown">\s*<STRONG>.*?</STRONG>.*?</SPAN>)',
    re.IGNORECASE | re.DOTALL
)

# [2025-12-01 新增] 通用小标题模式：蓝色标题 <B><SPAN style="...COLOR: #0033cc...">标题名</SPAN></B>
# 用于处理炫技、神话能力等带有蓝色小标题的页面
# 例如：<B><SPAN style="FONT-SIZE: 12pt; COLOR: #0033cc; ...">大胆特技(Ex)</SPAN></B>
# [2025-12-02 修复] 匹配一个或多个连续的蓝色标题标签作为一个整体
# 避免"大胆特技（"、"Ex"、"）："被分割成三个独立的切分点
BLUE_TITLE_PATTERN = re.compile(
    # 匹配一个或多个连续的蓝色标题块，它们之间可能有空白
    r'((?:<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#0033cc[^>]*>.*?</SPAN>\s*</B>\s*)+)',
    re.IGNORECASE | re.DOTALL
)

# 绿色大标题模式：<B><SPAN style="...COLOR: #009900...">页面标题</SPAN></B>
# 用于识别页面主标题，不作为切分点
GREEN_TITLE_PATTERN = re.compile(
    r'<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#009900[^>]*>(.*?)</SPAN>\s*</B>',
    re.IGNORECASE | re.DOTALL
)

# ============================================================================
# [2025-12-01 重构] 通用标题检测系统
# ============================================================================
# 定义所有可识别的标题样式，按优先级排序
# 每个条目包含：名称、正则模式、最小匹配数、最小内容大小、文件名过滤器
TITLE_PATTERNS = [
    # --- 法术文件专用模式 (仅匹配 Spell 文件) ---
    {
        'name': 'spell_purple',
        'description': '法术标题 (紫色 #cc00cc)',
        'pattern': re.compile(
            r'(<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#cc00cc[^>]*>.*?</SPAN>\s*</B>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 2,
        'min_content_size': 1000,
        'file_filter': lambda f: 'spell' in f.lower(),
    },
    {
        'name': 'spell_maroon',
        'description': '法术标题 (栗色 maroon, ISG)',
        'pattern': re.compile(
            # ISG 格式: <SPAN class=bbc_color style="COLOR: maroon">法术名</SPAN>
            r'(<SPAN\s+class=bbc_color\s+style="COLOR:\s*maroon"[^>]*>.*?</SPAN>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 3,
        'min_content_size': 1000,
        'file_filter': lambda f: 'spell' in f.lower() and ('isg' in f.lower() or '内海诸神' in f),
    },
    {
        'name': 'spell_navy',
        'description': '法术标题 (navy 色, OA/MA/UW等)',
        'pattern': re.compile(
            # 匹配两种 navy 格式:
            # 1. <B><SPAN style="COLOR: navy">法术名</SPAN></B> (OA)
            # 2. <SPAN class=bbc_color style="COLOR: navy"><STRONG>法术名</STRONG></SPAN> (MA/UW)
            r'(<B[^>]*>\s*<SPAN[^>]*COLOR:\s*navy[^>]*>.*?</SPAN>\s*</B>)|'
            r'(<SPAN\s+class=bbc_color[^>]*COLOR:\s*navy[^>]*>\s*<STRONG[^>]*>.*?</STRONG>\s*</SPAN>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 3,
        'min_content_size': 1000,
        'file_filter': lambda f: 'spell' in f.lower() or 'page_625' in f or 'page_1219' in f,
    },
    {
        'name': 'spell_bold_only',
        'description': '法术标题 (仅粗体+12pt, ACG)',
        'pattern': re.compile(
            # ACG 格式: <B><SPAN style="FONT-SIZE: 12pt...">法术名（<SPAN lang=EN-US>English Name</SPAN>）</SPAN></B>
            # 注意：英文名可能在嵌套的 <SPAN lang=EN-US> 中
            r'(<B[^>]*>\s*<SPAN[^>]*FONT-SIZE:\s*12pt[^>]*>(?:(?!</B>).)*?</SPAN>\s*</B>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 10,  # ACG 有 134 个法术，需要更多匹配以避免误判
        'min_content_size': 50000,  # ACG 文件约 700KB
        'file_filter': lambda f: 'spell' in f.lower() and 'acg' in f.lower(),
    },
    # 注意: ISM 文件没有明显的法术标题标记，使用普通处理（按大小自动切分）
    # [2025-12-02 新增] UM 法术文件专用模式
    # UM 文件中法术名没有颜色标记，但每个法术都以"学派"属性行开头
    # 法术名在"学派"行的前一个<P>段落中，格式如：敏锐感官 (Acute Senses)
    # 匹配模式：法术名段落 + 学派段落
    {
        'name': 'spell_um_school',
        'description': '法术标题 (UM格式, 按法术名+学派分割)',
        'pattern': re.compile(
            # 匹配：
            # 1. 一个包含法术名的 <P> 段落（不含 <B> 标签，包含中文和英文括号）
            # 2. 紧跟着的 <P> 段落包含蓝色的"学派"
            r'(<P[^>]*>(?:(?!<B).)*?[\u4e00-\u9fff]+[^<]*\([A-Za-z][^)]*\)[^<]*</(?:SPAN|P)>.*?</P>\s*'
            r'<P[^>]*><B[^>]*>\s*<SPAN[^>]*COLOR:\s*#0033cc[^>]*>学派)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 10,  # UM 有大量法术
        'min_content_size': 100000,  # UM 文件很大
        'file_filter': lambda f: 'spell' in f.lower() and 'um' in f.lower(),
    },
    # --- 专长文件模式 ---
    {
        'name': 'feat',
        'description': '专长标题 (棕色或12pt)',
        'pattern': re.compile(
            r'(<SPAN\s+class=bbc_size\s+style="FONT-SIZE:\s*12pt">\s*<STRONG>.*?</STRONG>.*?</SPAN>)|'
            r'(<SPAN\s+class=bbc_color\s+style="COLOR:\s*brown">\s*<STRONG>.*?</STRONG>.*?</SPAN>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 2,
        'min_content_size': 1000,
        'file_filter': lambda f: '专长' in f.lower() or 'feat' in f.lower(),
    },
    # --- 通用模式 (无文件名限制) ---
    {
        'name': 'blue_title',
        'description': '蓝色小标题 (#0033cc)',
        'pattern': re.compile(
            # [2025-12-02 修复] 匹配一个或多个连续的蓝色标题块作为一个整体
            # 避免"大胆特技（"、"Ex"、"）："被分割成三个独立的切分点
            r'((?:<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#0033cc[^>]*>.*?</SPAN>\s*</B>\s*)+)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 2,
        'min_content_size': 5000,
        # [2025-12-02 修复] 排除法术文件，因为法术文件中蓝色 #0033cc 用于属性标签（学派、环位等）
        # 而不是法术名标题，错误切分会导致每个法术被拆成多个独立的属性块
        'file_filter': lambda f: 'spell' not in f.lower(),
    },
    {
        'name': 'navy_title',
        'description': '科技指南条目 (navy 色)',
        'pattern': re.compile(
            # 匹配 navy 色 + 大字体(12pt/14pt/18pt) 的标题
            r'(<SPAN\s+class=bbc_color\s+style="COLOR:\s*navy">\s*<SPAN\s+class=bbc_size\s+style="[^"]*FONT-SIZE:\s*1[248]pt[^"]*"[^>]*>.*?</SPAN>\s*</SPAN>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 2,
        'min_content_size': 10000,
        'file_filter': lambda f: '科技' in f or 'page_76' in f,
    },
    # --- 职业子页面模式 (骑士团、血脉等) ---
    {
        'name': 'bloodline_blue',
        'description': '血脉标题 (蓝色 #0000cc)',
        'pattern': re.compile(
            # 血脉格式: <B><SPAN style="...COLOR: #0000cc...">血脉名</SPAN></B>
            r'(<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#0000cc[^>]*>.*?</SPAN>\s*</B>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 5,
        'min_content_size': 50000,
        'file_filter': lambda f: '血脉' in f or 'page_99' in f,
    },
    {
        'name': 'order_brown',
        'description': '骑士团标题 (棕色 bbc_color)',
        'pattern': re.compile(
            # 骑士团格式: <SPAN class=bbc_color style="COLOR: brown">骑士团名(Ex)</SPAN>
            r'(<SPAN\s+class=bbc_color\s+style="COLOR:\s*brown"[^>]*>.*?</SPAN>)',
            re.IGNORECASE | re.DOTALL
        ),
        'min_matches': 10,
        'min_content_size': 50000,
        'file_filter': lambda f: '骑士团' in f or 'page_1212' in f,
    },
]


def detect_title_pattern(source_file: str, content_html: str) -> Optional[Dict[str, Any]]:
    """
    [2025-12-01 重构] 通用标题模式检测器。
    
    遍历所有已定义的标题模式，返回第一个匹配的模式配置。
    
    Args:
        source_file: 源文件名
        content_html: HTML 内容字符串
        
    Returns:
        匹配的模式配置字典，或 None
    """
    if not source_file:
        source_file = ""
    
    for pattern_config in TITLE_PATTERNS:
        # 检查文件名过滤器
        file_filter = pattern_config.get('file_filter')
        if file_filter and not file_filter(source_file):
            continue
        
        # 检查内容大小
        min_size = pattern_config.get('min_content_size', 0)
        if len(content_html) < min_size:
            continue
        
        # 检查匹配数量
        pattern = pattern_config['pattern']
        matches = pattern.findall(content_html)
        # findall 可能返回元组列表（如果有多个捕获组），需要计算实际匹配数
        if isinstance(matches, list) and matches and isinstance(matches[0], tuple):
            # 多个捕获组的情况，计算非空匹配
            actual_matches = sum(1 for m in matches if any(m))
        else:
            actual_matches = len(matches)
        
        min_matches = pattern_config.get('min_matches', 2)
        if actual_matches >= min_matches:
            logging.debug(f"  -> 检测到 {pattern_config['name']}: {actual_matches} 个匹配 ({pattern_config['description']})")
            return pattern_config
    
    return None


def split_by_generic_pattern(
    content_html: str, 
    pattern_config: Dict[str, Any],
    cleaned_node_name: str, 
    full_path_str: str, 
    source_file: str,
    validate_size: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    [2025-12-01 重构] 通用标题模式切分函数。
    
    根据给定的模式配置切分 HTML 内容。
    
    [2025-12-02 新增] 添加切分块大小验证，如果不满足约束则返回 None。
    
    Args:
        content_html: HTML 内容字符串
        pattern_config: 模式配置字典
        cleaned_node_name: 清理后的节点名称
        full_path_str: 完整路径字符串
        source_file: 源文件名
        validate_size: 是否验证切分块大小（默认 True）
        
    Returns:
        数据条目列表，如果验证失败则返回 None
    """
    pattern = pattern_config['pattern']
    pattern_name = pattern_config['name']
    
    # 使用模式进行切分
    html_chunks = pattern.split(content_html)
    
    if len(html_chunks) <= 1:
        logging.debug(f"  -> {pattern_name} 模式未找到匹配")
        return None
    
    # 计算实际标题数量
    # 注意：如果模式有多个捕获组，split 会产生多个元素
    num_groups = pattern.groups
    if num_groups > 1:
        # 多捕获组模式，每 (num_groups + 1) 个元素为一组
        step = num_groups + 1
        num_titles = (len(html_chunks) - 1) // step
    else:
        # 单捕获组模式，每 2 个元素为一组
        step = 2
        num_titles = len(html_chunks) // 2
    
    if num_titles < 2:
        logging.debug(f"  -> {pattern_name} 标题数量不足 ({num_titles}个)，跳过切分")
        return None
    
    # [2025-12-02 新增] 预计算所有块的大小，用于验证
    chunk_sizes = []
    temp_results = []
    
    # 处理概述部分 (第一个标题之前的内容)
    chunk_0_html = html_chunks[0]
    chunk_0_md = convert_html_to_mixed_format(chunk_0_html)
    
    if chunk_0_md and len(chunk_0_md.strip()) > 50:
        chunk_0_title = f"{cleaned_node_name} (概述)"
        chunk_0_full_path = f"{full_path_str}/{chunk_0_title}"
        entry = create_data_entry(chunk_0_md, chunk_0_full_path, chunk_0_title, f"{source_file}#overview")
        temp_results.append(entry)
        chunk_sizes.append(len(chunk_0_md))
    
    # 处理各个标题块
    i = 1
    entry_count = 0
    while i < len(html_chunks):
        # 获取标题标签（可能是多个捕获组中的一个）
        if num_groups > 1:
            # 多捕获组：找到非空的那个
            tag_parts = html_chunks[i:i + num_groups]
            tag_html = next((p for p in tag_parts if p), "")
            content_idx = i + num_groups
        else:
            tag_html = html_chunks[i]
            content_idx = i + 1
        
        content_after_tag_html = html_chunks[content_idx] if content_idx < len(html_chunks) else ""
        
        # 完整的块 = 标题 + 标题后的内容
        full_chunk_html = tag_html + content_after_tag_html
        chunk_md = convert_html_to_mixed_format(full_chunk_html)
        
        # 从标题中提取名称
        chunk_title = extract_title_from_tag(tag_html, entry_count + 1)
        
        if chunk_md:
            chunk_full_path = f"{full_path_str}/{chunk_title}"
            chunk_source_file = f"{source_file}#{chunk_title.replace(' ', '_').replace('/', '_')}"
            
            entry = create_data_entry(chunk_md, chunk_full_path, chunk_title, chunk_source_file)
            temp_results.append(entry)
            chunk_sizes.append(len(chunk_md))
        
        entry_count += 1
        i = content_idx + 1
    
    # [2025-12-02 新增] 验证切分块大小
    if validate_size and chunk_sizes:
        is_valid, reason = validate_chunk_sizes(chunk_sizes)
        if not is_valid:
            logging.warning(f"  -> {pattern_name} 切分验证失败: {reason}，回退到固定大小切分")
            return None
    
    logging.info(f"  -> 使用 {pattern_config['description']} 模式切分 (找到 {len(temp_results)} 个条目)")
    for entry in temp_results:
        logging.debug(f"    -> 已分割: {entry['metadata']['source_title']}")
    
    return temp_results


def extract_title_from_tag(tag_html: str, fallback_index: int) -> str:
    """
    从 HTML 标签中提取标题文本。
    
    尝试多种方式提取标题：
    1. 从 STRONG 标签获取
    2. 从 SPAN 标签获取
    3. 从整个标签的文本获取
    4. 使用回退索引
    """
    if not tag_html:
        return f"条目 {fallback_index}"
    
    try:
        soup = BeautifulSoup(tag_html, 'html.parser')
        
        # 优先从 STRONG 标签获取
        strong_tag = soup.find('strong')
        if strong_tag:
            title = strong_tag.get_text(strip=True)
            if title:
                return clean_title(title)
        
        # 从最内层的 SPAN 标签获取
        spans = soup.find_all('span')
        if spans:
            # 取最内层的 span（通常是最后一个）
            for span in reversed(spans):
                title = span.get_text(strip=True)
                if title:
                    return clean_title(title)
        
        # 从整个标签获取文本
        title = soup.get_text(strip=True)
        if title:
            return clean_title(title)
        
    except Exception as e:
        logging.warning(f"  -> 解析标题失败: {e}")
    
    return f"条目 {fallback_index}"


def clean_title(title: str) -> str:
    """清理标题文本"""
    # 只取第一行
    title = title.split('\n')[0].strip()
    # 移除多余空白
    title = ' '.join(title.split())
    # 截断过长的标题
    if len(title) > 100:
        title = title[:100] + "..."
    return title if title else "未命名条目"


def validate_chunk_sizes(chunk_sizes: List[int]) -> tuple[bool, str]:
    """
    [2025-12-02 新增] 验证切分块大小是否符合约束。
    
    约束：每个块必须在 CHUNK_SIZE_MIN (500) ~ CHUNK_SIZE_MAX (10000) 之间。
    
    Args:
        chunk_sizes: 各块的字符数列表
        
    Returns:
        (是否通过验证, 失败原因)
    """
    if not chunk_sizes:
        return False, "没有切分块"
    
    for i, size in enumerate(chunk_sizes):
        if size < CHUNK_SIZE_MIN:
            return False, f"块 {i+1} 过小 ({size} < {CHUNK_SIZE_MIN})"
        if size > CHUNK_SIZE_MAX:
            return False, f"块 {i+1} 过大 ({size} > {CHUNK_SIZE_MAX})"
    
    return True, "通过验证"


def split_by_fixed_size(
    page_content_md: str,
    cleaned_node_name: str,
    full_path_str: str,
    source_file: str
) -> List[Dict[str, Any]]:
    """
    [2025-12-02 新增] 按固定大小切分内容。
    
    当其他切分方式不满足约束时使用此函数作为回退。
    
    Args:
        page_content_md: Markdown 格式的内容
        cleaned_node_name: 清理后的节点名称
        full_path_str: 完整路径字符串
        source_file: 源文件名
        
    Returns:
        数据条目列表
    """
    results = []
    chunk_size = CHUNK_SIZE_MAX  # 使用最大允许大小
    num_chunks = (len(page_content_md) + chunk_size - 1) // chunk_size
    
    logging.info(f"  -> 按固定大小切分为 {num_chunks} 个部分 (每部分最大 {chunk_size} 字符)")
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(page_content_md))
        chunk_content = page_content_md[start:end]
        
        chunk_title = f"{cleaned_node_name} (Part {chunk_idx + 1}/{num_chunks})"
        chunk_full_path = f"{full_path_str}/Part{chunk_idx + 1}"
        entry = create_data_entry(
            chunk_content,
            chunk_full_path,
            chunk_title,
            f"{source_file}#part{chunk_idx + 1}"
        )
        results.append(entry)
    
    return results


# ============================================================================
# [2025-12-02 新增] 通用 HTML 标题层级自适应切分系统
# ============================================================================

def analyze_heading_structure(content_html: str) -> Dict[str, Any]:
    """
    分析 HTML 内容中的标题结构。
    
    返回每个标题级别的统计信息：数量、平均块大小、是否适合切分等。
    """
    results = {}
    
    for tag in HEADING_TAGS:
        # 构建匹配该标题的正则
        pattern = re.compile(
            rf'(<{tag}[^>]*>.*?</{tag}>)',
            re.IGNORECASE | re.DOTALL
        )
        
        # 切分并分析
        chunks = pattern.split(content_html)
        
        if len(chunks) <= 1:
            # 没有找到该级别的标题
            results[tag] = {
                'count': 0,
                'chunks': [],
                'suitable': False,
                'reason': '未找到标题'
            }
            continue
        
        # 计算每个块的大小
        chunk_sizes = []
        
        # 第一块（概述）
        if chunks[0].strip():
            overview_md = convert_html_to_mixed_format(chunks[0])
            chunk_sizes.append(len(overview_md))
        
        # 后续块（标题 + 内容）
        for i in range(1, len(chunks), 2):
            tag_html = chunks[i]
            content_html_after = chunks[i + 1] if (i + 1) < len(chunks) else ""
            full_chunk = tag_html + content_html_after
            chunk_md = convert_html_to_mixed_format(full_chunk)
            chunk_sizes.append(len(chunk_md))
        
        num_headings = len(chunks) // 2
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        
        # 判断是否适合切分
        suitable = True
        reason = "适合切分"
        
        if num_headings < 2:
            suitable = False
            reason = f"标题数量不足 ({num_headings}个)"
        elif max_size > MAX_DOC_SIZE * 1.5:
            suitable = False
            reason = f"最大块过大 ({max_size:,} 字符)"
        elif min_size < MIN_CHUNK_SIZE and num_headings > 3:
            suitable = False
            reason = f"最小块过小 ({min_size} 字符)，切分过细"
        elif avg_size < MIN_CHUNK_SIZE:
            suitable = False
            reason = f"平均块大小过小 ({avg_size:.0f} 字符)"
        
        results[tag] = {
            'count': num_headings,
            'chunks': chunks,
            'chunk_sizes': chunk_sizes,
            'avg_size': avg_size,
            'max_size': max_size,
            'min_size': min_size,
            'suitable': suitable,
            'reason': reason
        }
    
    return results


def find_best_heading_level(content_html: str) -> Optional[str]:
    """
    自动选择最佳的标题级别进行切分。
    
    选择策略：
    1. 优先选择高级别标题（h1 > h2 > h3...）
    2. 要求至少 2 个标题
    3. 每块大小应在 MIN_CHUNK_SIZE ~ MAX_DOC_SIZE 之间
    4. 避免切分过细（平均块大小不能太小）
    
    Returns:
        最佳标题标签名（如 'h2'），或 None（如果没有合适的标题）
    """
    analysis = analyze_heading_structure(content_html)
    
    # 按标题级别从高到低遍历
    for tag in HEADING_TAGS:
        info = analysis.get(tag, {})
        if info.get('suitable', False):
            logging.debug(f"  -> 选择 <{tag}> 作为切分标签: {info['count']} 个标题, "
                         f"平均 {info['avg_size']:.0f} 字符/块")
            return tag
    
    # 没有找到合适的标题级别
    return None


def split_by_heading_level(
    content_html: str,
    heading_tag: str,
    cleaned_node_name: str,
    full_path_str: str,
    source_file: str
) -> Optional[List[Dict[str, Any]]]:
    """
    按指定的标题级别切分 HTML 内容。
    
    [2025-12-02 新增] 添加切分块大小验证，如果不满足约束则返回 None。
    
    Args:
        content_html: HTML 内容字符串
        heading_tag: 标题标签（如 'h2', 'h3'）
        cleaned_node_name: 清理后的节点名称
        full_path_str: 完整路径字符串
        source_file: 源文件名
        
    Returns:
        数据条目列表，如果验证失败则返回 None
    """
    results = []
    chunk_sizes = []
    
    # 构建切分正则
    pattern = re.compile(
        rf'(<{heading_tag}[^>]*>.*?</{heading_tag}>)',
        re.IGNORECASE | re.DOTALL
    )
    
    chunks = pattern.split(content_html)
    
    if len(chunks) <= 1:
        return None
    
    num_headings = len(chunks) // 2
    
    # 处理概述部分（第一个标题之前的内容）
    chunk_0_html = chunks[0]
    chunk_0_md = convert_html_to_mixed_format(chunk_0_html)
    
    if chunk_0_md and len(chunk_0_md.strip()) > 50:
        chunk_0_title = f"{cleaned_node_name} (概述)"
        chunk_0_full_path = f"{full_path_str}/{chunk_0_title}"
        entry = create_data_entry(chunk_0_md, chunk_0_full_path, chunk_0_title, f"{source_file}#overview")
        results.append(entry)
        chunk_sizes.append(len(chunk_0_md))
    
    # 处理各个标题块
    for i in range(1, len(chunks), 2):
        tag_html = chunks[i]
        content_after_tag_html = chunks[i + 1] if (i + 1) < len(chunks) else ""
        
        # 完整的块 = 标题 + 标题后的内容
        full_chunk_html = tag_html + content_after_tag_html
        chunk_md = convert_html_to_mixed_format(full_chunk_html)
        
        # 从标题标签中提取标题文本
        try:
            chunk_soup = BeautifulSoup(tag_html, 'html.parser')
            title_tag = chunk_soup.find(heading_tag)
            chunk_title = title_tag.get_text(strip=True) if title_tag else f"Section {i // 2 + 1}"
            chunk_title = clean_title(chunk_title)
        except Exception:
            chunk_title = f"Section {i // 2 + 1}"
        
        if chunk_md:
            chunk_full_path = f"{full_path_str}/{chunk_title}"
            chunk_source_file = f"{source_file}#{chunk_title.replace(' ', '_').replace('/', '_')}"
            
            entry = create_data_entry(chunk_md, chunk_full_path, chunk_title, chunk_source_file)
            results.append(entry)
            chunk_sizes.append(len(chunk_md))
    
    # [2025-12-02 新增] 验证切分块大小
    if chunk_sizes:
        is_valid, reason = validate_chunk_sizes(chunk_sizes)
        if not is_valid:
            logging.warning(f"  -> <{heading_tag}> 切分验证失败: {reason}")
            return None
    
    logging.info(f"  -> 使用 <{heading_tag}> 自适应切分 (找到 {num_headings} 个章节)")
    for entry in results:
        logging.debug(f"    -> 已分割: {entry['metadata']['source_title']}")
    
    return results


def auto_split_by_heading(
    content_html: str,
    cleaned_node_name: str,
    full_path_str: str,
    source_file: str
) -> Optional[List[Dict[str, Any]]]:
    """
    自动选择最佳标题级别并切分内容。
    
    这是对外的主入口函数，整合了分析和切分逻辑。
    
    切分优先级：
    1. 标准 HTML 标题标签 (h1-h6)
    2. PF 蓝色标题模式 (#0033cc) - 仅当没有标准标题时
    
    Args:
        content_html: HTML 内容字符串
        cleaned_node_name: 清理后的节点名称
        full_path_str: 完整路径字符串
        source_file: 源文件名
        
    Returns:
        数据条目列表，如果没有合适的切分方案则返回 None
    """
    # 1. 首先尝试标准 HTML 标题级别
    best_tag = find_best_heading_level(content_html)
    
    if best_tag:
        return split_by_heading_level(
            content_html,
            best_tag,
            cleaned_node_name,
            full_path_str,
            source_file
        )
    
    # 2. 如果没有标准 HTML 标题，尝试 PF 蓝色标题模式
    # 注意：这里使用宽松的匹配，只要求 2 个以上蓝色标题
    blue_pattern = re.compile(
        r'((?:<B[^>]*>\s*<SPAN[^>]*COLOR:\s*#0033cc[^>]*>.*?</SPAN>\s*</B>\s*)+)',
        re.IGNORECASE | re.DOTALL
    )
    blue_matches = blue_pattern.findall(content_html)
    
    if len(blue_matches) >= 2:
        logging.info(f"  -> 使用蓝色标题 (#0033cc) 回退切分 (找到 {len(blue_matches)} 个)")
        
        # 构建临时的模式配置
        temp_config = {
            'name': 'blue_title_fallback',
            'description': '蓝色小标题回退 (#0033cc)',
            'pattern': blue_pattern,
        }
        
        return split_by_generic_pattern(
            content_html,
            temp_config,
            cleaned_node_name,
            full_path_str,
            source_file
        )
    
    logging.debug(f"  -> 未找到合适的标题进行切分")
    return None


# --- 日志配置 ---
# 确保 logs 目录存在
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # 日志文件保存到 logs 目录
        logging.FileHandler(LOG_DIR / "package_json.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# --- 核心辅助函数 ---

def load_config(config_path: Path) -> Dict[str, Any]:
    """加载 JSON 配置文件"""
    if not config_path.exists():
        logging.error(f"配置文件未找到: {config_path}")
        logging.error("请先运行 'python tools/analyze_chm.py' 生成配置文件。")
        raise FileNotFoundError(f"配置文件 {config_path} 未找到。")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"配置文件 {config_path} 格式错误: {e}")
        raise
    except Exception as e:
        logging.error(f"加载配置文件时出错: {e}")
        raise


def get_html_soup(html_path: Path) -> Optional[BeautifulSoup]:
    """加载并解析HTML文件，使用 chardet 自动检测编码"""
    if not html_path.exists():
        logging.warning(f"HTML 文件未找到: {html_path}")
        return None

    try:
        # 使用 chardet 检测编码
        with open(html_path, 'rb') as f:
            raw_data = f.read()
        
        # 检测编码
        detected = chardet.detect(raw_data)
        encoding = detected.get('encoding', 'utf-8')
        confidence = detected.get('confidence', 0)
        
        logging.debug(f"检测到编码: {encoding} (置信度: {confidence:.2f}) for {html_path}")
        
        # 如果置信度太低，使用默认编码列表
        if confidence < 0.7:
            encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1']
        else:
            encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb18030']
        
        # 尝试解码
        for enc in encodings_to_try:
            try:
                content = raw_data.decode(enc)
                return BeautifulSoup(content, 'html.parser')
            except (UnicodeDecodeError, LookupError):
                continue
        
        # 如果所有编码都失败，使用 errors='ignore'
        logging.warning(f"所有编码尝试失败 {html_path}，使用 UTF-8 with errors='ignore'")
        content = raw_data.decode('utf-8', errors='ignore')
        return BeautifulSoup(content, 'html.parser')
        
    except Exception as e:
        logging.error(f"解析HTML文件时出错 {html_path}: {e}")
        return None


def convert_html_to_mixed_format(html_content: str) -> str:
    """
    [Gemini 优化版本]
    将 HTML 转换为混合格式：
    - 表格 (<table>) 保留原始 HTML（Gemini 对 HTML 表格理解更好）
    - 其他内容转换为 Markdown
    
    [2025-12-01 修复] 使用 html2text 直接处理整个 HTML 内容，而不是只遍历顶级元素。
    之前的实现会丢失嵌套在深层元素中的内容。
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 检查是否包含表格
    tables = soup.find_all('table')
    
    if not tables:
        # 没有表格，直接用 html2text 转换整个内容
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.protect_links = True
        h.mark_code = True
        
        try:
            markdown = h.handle(html_content)
            # 清理空行但保留段落结构
            lines = markdown.splitlines()
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                stripped = line.rstrip()
                if not stripped:
                    if not prev_empty:
                        cleaned_lines.append('')
                    prev_empty = True
                else:
                    cleaned_lines.append(stripped)
                    prev_empty = False
            return '\n'.join(cleaned_lines).strip()
        except Exception as e:
            logging.warning(f"转换 HTML 失败: {e}，回退到纯文本提取")
            return soup.get_text(separator='\n', strip=True)
    
    # 有表格的情况：需要特殊处理
    # 策略：用占位符替换表格，转换 Markdown，再替换回来
    result_parts = []
    table_placeholders = {}
    
    # 为每个表格创建唯一占位符
    for i, table in enumerate(tables):
        placeholder = f"__TABLE_PLACEHOLDER_{i}__"
        table_placeholders[placeholder] = str(table)
        # 用占位符替换表格
        table.replace_with(soup.new_string(placeholder))
    
    # 转换剩余内容为 Markdown
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_emphasis = False
    h.protect_links = True
    h.mark_code = True
    
    try:
        markdown = h.handle(str(soup))
        # 清理空行
        lines = markdown.splitlines()
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(stripped)
                prev_empty = False
        markdown = '\n'.join(cleaned_lines).strip()
        
        # 将占位符替换回表格 HTML
        for placeholder, table_html in table_placeholders.items():
            markdown = markdown.replace(placeholder, f"\n\n{table_html}\n\n")
        
        return markdown.strip()
    except Exception as e:
        logging.warning(f"转换 HTML 失败: {e}，回退到纯文本提取")
        # 重新解析原始内容（因为上面可能修改了 soup）
        original_soup = BeautifulSoup(html_content, 'html.parser')
        return original_soup.get_text(separator='\n', strip=True)


def create_data_entry(
        page_content: str,
        full_path: str,
        source_title: str,
        source_file: Optional[str]
) -> Dict[str, Any]:
    """
    根据设计文档规范创建单个JSON对象
    """
    return {
        "page_content": page_content.strip(),
        "metadata": {
            "full_path": full_path,
            "source_title": source_title.strip(),
            "source_file": source_file or "N/A"
        }
    }


# --- [V3: 删除了 get_content_between_tags 和 get_html_before_tag] ---
# --- (这两个函数基于错误的“理想化”假设) ---


def clean_node_name(node_name: str) -> str:
    """
    智能清理 node_name, 解决路径重复问题。

    1. 移除 .html / .htm 扩展名
    2. 处理 "path/like/key" -> "key" (取最后一部分)
    3. 处理 "key::parent" -> "key" (取第一部分)
    """
    # 1. 移除文件扩展名
    name = node_name.replace(".html", "").replace(".htm", "")

    # 2. 如果 node_name 像一个文件路径, 只取最后的文件名部分
    if '/' in name:
        name = name.split('/')[-1]

    # 3. 如果 node_name 包含 `::` (来自 CHM 目录), 只取 `::` 之前的部分
    if '::' in name:
        name = name.split('::')[0]

    return name.strip()


def detect_content_type(source_file: str, content_html: str) -> Optional[Dict[str, Any]]:
    """
    [2025-12-01 重构] 自动检测内容类型，返回合适的切分模式配置。
    注意：此功能仅在 PF (Pathfinder) 版本中启用。
    
    Args:
        source_file: 源文件名
        content_html: HTML 内容字符串
        
    Returns:
        匹配的模式配置字典，或 None
    """
    # DND 版本不使用特殊切分
    if is_dnd():
        return None
    
    # 使用通用标题模式检测器
    return detect_title_pattern(source_file, content_html)



# --- 核心递归处理函数 ---

def process_node(
        node_name: str,
        node_config: Dict[str, Any],
        current_path_parts: List[str],
        chm_source_dir: Path,
        common_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    递归处理配置树中的单个节点，并返回一个包含所有数据条目的列表。
    """

    results = []

    # --- 1. 检查 Action ---
    action = node_config.get("action", "process")
    if action == "skip_all":
        logging.info(f"跳过 (skip_all): {'/'.join(current_path_parts + [node_name])}")
        return []

    # 使用原始 node_name 记录日志, 保持与 config.json 一致
    logging.info(f"处理: {'/'.join(current_path_parts + [node_name])}")

    # 使用清理函数处理 node_name
    cleaned_node_name = clean_node_name(node_name)

    # --- 2. 准备元数据和路径 ---
    new_path_parts = current_path_parts + [cleaned_node_name]
    full_path_str = "/".join(new_path_parts)
    source_file = node_config.get("source_file")
    
    # [2025-12-01 修复] 对文件名进行 URL 解码，处理 %20 等编码字符
    if source_file:
        source_file_decoded = unquote(source_file)
    else:
        source_file_decoded = None
    
    html_path = (chm_source_dir / source_file_decoded).resolve() if source_file_decoded else None

    # --- 3. 处理本节点 (如果 Action 不是 'skip_children_only') ---
    if action == "process" and html_path:
        soup = get_html_soup(html_path)
        if soup:
            # 确定内容选择器
            selector = node_config.get("selector") or common_config.get("default_content_selector", "body")

            # [2025-12-01 修复] PF 版本始终使用 body，避免 select_one 只返回第一个匹配元素的问题
            # 例如 select_one("p.MsoNormal") 只会返回第一个 <p class="MsoNormal">，导致大量内容丢失
            # DND 版本保持原有逻辑
            if is_pf():
                # PF 版本：始终使用 body 获取完整内容
                content_soup = soup.body
                if not content_soup:
                    logging.error(f"在 {html_path} 中未找到 'body'。跳过此文件。")
                    content_soup = None
            elif selector == "body" or selector is None:
                content_soup = soup.body
                if not content_soup:
                    logging.error(f"在 {html_path} 中未找到 'body'。跳过此文件。")
                    content_soup = None
            else:
                # DND 版本：对于指定的选择器，尝试使用 select_one
                # 但如果文件很大（>1MB），直接使用 body 以包含所有内容
                file_size = html_path.stat().st_size if html_path.exists() else 0
                if file_size > 1024 * 1024:  # 文件大于 1MB
                    logging.info(f"文件 {html_path.name} 较大 ({file_size/1024/1024:.2f}MB)，使用 body 而不是选择器 '{selector}'")
                    content_soup = soup.body
                else:
                    content_soup = soup.select_one(selector)
                    if not content_soup:
                        logging.warning(f"在 {html_path} 中未找到选择器 '{selector}'。将回退到 'body'。")
                        content_soup = soup.body

            if content_soup:
                split_by = node_config.get("split_by")

                # 将选中的内容转为字符串, 准备用Regex处理
                content_html_str = str(content_soup)

                # [2025-12-01 新增] 自动检测并使用通用模式切分
                # [2025-01 重构] 统一使用 detect_content_type 和 split_by_generic_pattern
                pattern_config = detect_content_type(source_file_decoded or "", content_html_str)
                
                if pattern_config:
                    pattern_name = pattern_config['name']
                    logging.info(f"  -> 检测到 {pattern_name} 模式，尝试通用切分")
                    pattern_results = split_by_generic_pattern(
                        content_html_str,
                        pattern_config,
                        cleaned_node_name, 
                        full_path_str, 
                        source_file_decoded or ""
                    )
                    if pattern_results:
                        results.extend(pattern_results)
                    else:
                        # [2025-12-02 修复] 切分失败（验证不通过），回退到固定大小切分
                        logging.warning(f"  -> {pattern_name} 模式切分不满足约束，使用固定大小切分")
                        page_content_md = convert_html_to_mixed_format(content_html_str)
                        if page_content_md:
                            fixed_results = split_by_fixed_size(
                                page_content_md,
                                cleaned_node_name,
                                full_path_str,
                                source_file_decoded or ""
                            )
                            results.extend(fixed_results)
                        pattern_config = None  # 重置以便继续普通处理（如果需要）
                
                # 如果已经通过模式处理完成，跳过普通处理
                if pattern_config:
                    pass  # 已处理完成，不需要额外操作
                
                # --- 情况A: 不分割 (split_by: null) ---
                elif not split_by:
                    logging.debug(f"  -> 作为单个文档处理 (No Split)")
                    page_content_md = convert_html_to_mixed_format(content_html_str)

                    if page_content_md:
                        # 检查文档是否过大
                        if len(page_content_md) > MAX_DOC_SIZE:
                            logging.warning(f"  -> 文档过大 ({len(page_content_md):,} 字符)，尝试自动切分...")
                            
                            # [2025-12-02 重构] 使用新的自适应标题级别切分系统
                            auto_results = auto_split_by_heading(
                                content_html_str,
                                cleaned_node_name,
                                full_path_str,
                                source_file_decoded or ""
                            )
                            
                            if auto_results:
                                # 验证自动切分结果
                                chunk_sizes = [len(r['page_content']) for r in auto_results]
                                is_valid, reason = validate_chunk_sizes(chunk_sizes)
                                if is_valid:
                                    results.extend(auto_results)
                                else:
                                    logging.warning(f"  -> 自动切分验证失败: {reason}，使用固定大小切分")
                                    fixed_results = split_by_fixed_size(
                                        page_content_md,
                                        cleaned_node_name,
                                        full_path_str,
                                        source_file_decoded or ""
                                    )
                                    results.extend(fixed_results)
                            else:
                                # 如果 HTML 标题切分失败，尝试按 <hr> 切分
                                hr_pattern = re.compile(r'(<hr[^>]*?/?>)', re.IGNORECASE)
                                hr_chunks = hr_pattern.split(content_html_str)
                                
                                if len(hr_chunks) >= 3:  # 至少有 1 个 <hr>
                                    logging.info(f"  -> 使用 <hr> 切分 (找到 {len(hr_chunks) // 2} 个分隔符)")
                                    
                                    hr_results = []
                                    hr_chunk_sizes = []
                                    
                                    # 处理概述部分
                                    chunk_0_md = convert_html_to_mixed_format(hr_chunks[0])
                                    if chunk_0_md and len(chunk_0_md.strip()) > 50:
                                        entry = create_data_entry(
                                            chunk_0_md,
                                            f"{full_path_str}/{cleaned_node_name} (概述)",
                                            f"{cleaned_node_name} (概述)",
                                            f"{source_file_decoded}#overview"
                                        )
                                        hr_results.append(entry)
                                        hr_chunk_sizes.append(len(chunk_0_md))
                                    
                                    # 处理各个部分
                                    for i in range(1, len(hr_chunks), 2):
                                        content_after_hr = hr_chunks[i + 1] if (i + 1) < len(hr_chunks) else ""
                                        chunk_md = convert_html_to_mixed_format(content_after_hr)
                                        if chunk_md:
                                            part_num = i // 2 + 1
                                            entry = create_data_entry(
                                                chunk_md,
                                                f"{full_path_str}/Part{part_num}",
                                                f"{cleaned_node_name} (Part {part_num})",
                                                f"{source_file_decoded}#part{part_num}"
                                            )
                                            hr_results.append(entry)
                                            hr_chunk_sizes.append(len(chunk_md))
                                    
                                    # 验证 hr 切分结果
                                    is_valid, reason = validate_chunk_sizes(hr_chunk_sizes)
                                    if is_valid:
                                        results.extend(hr_results)
                                    else:
                                        logging.warning(f"  -> HR 切分验证失败: {reason}，使用固定大小切分")
                                        fixed_results = split_by_fixed_size(
                                            page_content_md,
                                            cleaned_node_name,
                                            full_path_str,
                                            source_file_decoded or ""
                                        )
                                        results.extend(fixed_results)
                                else:
                                    # 最后的回退：按固定大小切分
                                    logging.warning(f"  -> 无法自动切分，强制按固定大小切分")
                                    fixed_results = split_by_fixed_size(
                                        page_content_md,
                                        cleaned_node_name,
                                        full_path_str,
                                        source_file_decoded or ""
                                    )
                                    results.extend(fixed_results)
                        else:
                            # 文档大小正常，直接添加
                            entry = create_data_entry(page_content_md, full_path_str, cleaned_node_name, source_file_decoded)
                            results.append(entry)
                    else:
                        logging.warning(f"  -> 内容为空 (Selector: {selector})")

                # --- 情况B: 按标题分割 (split_by: "hX") [V3: 使用Regex重写] ---
                else:
                    logging.debug(f"  -> 按 <{split_by}> 标签(Regex)分割")

                    # 1. 编译一个健壮的、不区分大小写的Regex来查找 <split_by>...</split_by> 块
                    # 关键: ( ... ) 是一个捕获组。
                    # re.split() 会保留捕获组的内容 (即<h4...>...</h4> 块) 在列表中。
                    # 它不关心<h4...>在HTML中的嵌套层级。
                    split_pattern_re = re.compile(
                        # e.g., (<h4[^>]*>.*?</h4>)
                        rf'(<{split_by}[^>]*>.*?</{split_by}>)',
                        re.IGNORECASE | re.DOTALL
                    )

                    html_chunks = split_pattern_re.split(content_html_str)

                    # 如果没有找到匹配项, html_chunks 列表将只有1个元素 (原始字符串)
                    if len(html_chunks) <= 1:
                        logging.warning(
                            f"  -> 在 {html_path} 中设置了 split_by='{split_by}', 但Regex未找到任何匹配。将作为单个文档处理。")
                        # 回退到 "不分割" 逻辑
                        page_content_md = convert_html_to_mixed_format(content_html_str)
                        if page_content_md:
                            entry = create_data_entry(
                                page_content_md,
                                full_path_str,
                                cleaned_node_name,
                                source_file_decoded
                            )
                            results.append(entry)

                    else:
                        # --- 1. 处理 "Chunk 0" (第一个 <split_by> 之前的内容) ---
                        # html_chunks[0] 总是第一个 <split_by> 之前的内容
                        chunk_0_html = html_chunks[0]
                        chunk_0_md = convert_html_to_mixed_format(chunk_0_html)

                        if chunk_0_md:
                            # 为这个 "概述" 块创建条目
                            chunk_0_title = f"{cleaned_node_name} (概述)"
                            chunk_0_full_path = f"{full_path_str}/{chunk_0_title}"
                            chunk_0_source_file = f"{source_file_decoded}#overview"

                            entry = create_data_entry(
                                chunk_0_md,
                                chunk_0_full_path,
                                chunk_0_title,
                                chunk_0_source_file
                            )
                            results.append(entry)
                            logging.debug(f"    -> 已分割: {chunk_0_full_path} (Chunk 0)")
                        else:
                            logging.debug(f"  -> 'Chunk 0' (第一个 {split_by} 之前) 为空，已跳过。")

                        # --- 2. 迭代处理 "Chunk 1...N" ---
                        # 列表是 [chunk_0, tag_1, chunk_1, tag_2, chunk_2, ...]
                        # 我们从索引 1 (tag_1) 开始, 每次跳 2
                        for i in range(1, len(html_chunks), 2):
                            tag_html = html_chunks[i]  # e.g., <h4>...</h4>
                            content_after_tag_html = html_chunks[i + 1] if (i + 1) < len(html_chunks) else ""

                            # 完整的块 = 标签 + 标签后的内容
                            full_chunk_html = tag_html + content_after_tag_html
                            chunk_md = convert_html_to_mixed_format(full_chunk_html)

                            # 为了获取标题, 我们单独用BS4解析 tag_html
                            try:
                                # 只解析标签本身来获取标题
                                chunk_soup = BeautifulSoup(tag_html, 'html.parser')
                                title_tag = chunk_soup.find(split_by)
                                chunk_title = title_tag.get_text(
                                    strip=True) if title_tag else f"Untitled Section {i // 2 + 1}"
                            except Exception as e:
                                logging.warning(f"  -> 解析标题失败: {e}. 使用默认标题。")
                                chunk_title = f"Untitled Section {i // 2 + 1}"

                            if not chunk_title.strip():
                                chunk_title = f"Untitled Section {i // 2 + 1}"
                                logging.warning(f"  -> 在 {html_path} 中找到一个空的 <{split_by}> 标签。")

                            if chunk_md:
                                chunk_full_path = f"{full_path_str}/{chunk_title}"
                                chunk_source_title = chunk_title
                                chunk_source_file = f"{source_file_decoded}#{chunk_title.replace(' ', '_')}"

                                entry = create_data_entry(
                                    chunk_md,
                                    chunk_full_path,
                                    chunk_source_title,
                                    chunk_source_file
                                )
                                results.append(entry)
                                logging.debug(f"    -> 已分割: {chunk_full_path}")

    # --- 4. 递归处理子节点 ---
    children_config = node_config.get("children", {})
    if children_config:
        for child_name, child_config in children_config.items():
            child_results = process_node(
                child_name,
                child_config,
                new_path_parts,  # 传递更新后的路径
                chm_source_dir,
                common_config
            )
            results.extend(child_results)

    return results


# --- 主执行函数 ---

def main():
    """
    阶段2的主执行函数。
    """
    logging.info("--- 阶段 2: 开始打包 JSON (Final Version) ---")

    try:
        config = load_config(CONFIG_FILE)
    except Exception as e:
        logging.error(f"无法启动打包程序: {e}")
        return

    # --- 1. 准备路径 ---
    chm_source_dir = Path(config.get("chm_source_path", "./chm_source"))
    output_file = Path(config.get("output_file", "./data/rules_data.json"))

    # 确保输出目录存在
    output_dir = output_file.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"确保输出目录存在: {output_dir}")
    except OSError as e:
        logging.error(f"创建输出目录 {output_dir} 失败: {e}")
        return

    common_config = config.get("common_config", {})
    tree_rules = config.get("tree_processing_rules", {})

    if not tree_rules:
        logging.error("配置文件中的 'tree_processing_rules' 为空。无法继续。")
        return

    # --- 2. 递归处理 ---
    logging.info("开始递归处理配置树...")
    all_data_entries = []

    for top_level_name, node_config in tree_rules.items():
        all_data_entries.extend(
            process_node(
                node_name=top_level_name,
                node_config=node_config,
                current_path_parts=[],  # 从根路径开始
                chm_source_dir=chm_source_dir,
                common_config=common_config
            )
        )

    logging.info(f"处理完成。总共生成 {len(all_data_entries)} 个数据条目。")

    # --- 3. 写入最终的 JSON 文件 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data_entries, f, ensure_ascii=False, indent=2)
        logging.info(f"成功写入最终数据文件: {output_file}")
    except IOError as e:
        logging.error(f"写入 {output_file} 失败: {e}")
    except Exception as e:
        logging.error(f"序列化JSON时发生未知错误: {e}")

    logging.info("--- 阶段 2: 打包完成 (Final Version) ---")


if __name__ == "__main__":
    main()