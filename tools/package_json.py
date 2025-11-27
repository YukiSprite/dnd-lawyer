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
"""

import os
import json
from pathlib import Path
from bs4 import BeautifulSoup, Tag
import html2text
import chardet  # 用于编码检测
import logging
import re  # 导入正则表达式
from typing import List, Dict, Any, Optional

# --- 全局配置 ---

# 数据处理配置文件（由 analyze_chm.py 自动生成）
CONFIG_FILE = Path("./config/data_processing.json")

# 单个文档的最大字符数阈值（超过此值将尝试自动切分）
# 降低到 20K 以避免上下文超限（4个文档 * 20K ≈ 50K tokens，远低于128K限制）
MAX_DOC_SIZE = 20000  # 20KB，约 2 万字符（≈13K tokens）

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
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')
    result_parts = []
    
    # 遍历顶级元素
    for element in soup.children:
        if not hasattr(element, 'name'):
            continue
            
        # 如果是表格，保留原始 HTML
        if element.name == 'table':
            result_parts.append(str(element))
        # 如果元素包含表格，也保留原始 HTML
        elif element.find('table'):
            result_parts.append(str(element))
        else:
            # 其他内容转 Markdown
            h = html2text.HTML2Text()
            h.body_width = 0
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_emphasis = False
            h.protect_links = True
            h.mark_code = True
            
            try:
                markdown = h.handle(str(element))
                cleaned = "\n".join(line.rstrip() for line in markdown.splitlines() if line.strip())
                if cleaned:
                    result_parts.append(cleaned)
            except Exception as e:
                logging.warning(f"转换元素失败: {e}，保留原始文本。")
                text = element.get_text(separator='\n', strip=True)
                if text:
                    result_parts.append(text)
    
    return "\n\n".join(result_parts)


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
    [V5 改进] 智能清理 node_name, 解决路径重复问题。

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

    # [V5] 使用原始 node_name 记录日志, 保持与 config.json 一致
    logging.info(f"处理: {'/'.join(current_path_parts + [node_name])}")

    # --- [V5 改进] 使用新的清理函数 ---
    cleaned_node_name = clean_node_name(node_name)

    # --- 2. 准备元数据和路径 ---
    new_path_parts = current_path_parts + [cleaned_node_name]  # <-- V5 change
    full_path_str = "/".join(new_path_parts)
    source_file = node_config.get("source_file")
    html_path = (chm_source_dir / source_file).resolve() if source_file else None

    # --- 3. 处理本节点 (如果 Action 不是 'skip_children_only') ---
    if action == "process" and html_path:
        soup = get_html_soup(html_path)
        if soup:
            # 确定内容选择器
            selector = node_config.get("selector") or common_config.get("default_content_selector", "body")

            # 对于大文件，直接使用 body 以避免 select_one 只选择第一个元素的问题
            if selector == "body" or selector is None:
                content_soup = soup.body
                if not content_soup:
                    logging.error(f"在 {html_path} 中未找到 'body'。跳过此文件。")
                    content_soup = None
            else:
                # 对于指定的选择器，尝试使用 select_one
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

                # [V3: 将选中的内容转为字符串, 准备用Regex处理]
                content_html_str = str(content_soup)

                # --- 情况A: 不分割 (split_by: null) ---
                if not split_by:
                    logging.debug(f"  -> 作为单个文档处理 (No Split)")
                    page_content_md = convert_html_to_mixed_format(content_html_str)

                    if page_content_md:
                        # 检查文档是否过大
                        if len(page_content_md) > MAX_DOC_SIZE:
                            logging.warning(f"  -> 文档过大 ({len(page_content_md):,} 字符)，尝试自动切分...")
                            
                            # 尝试按 h2 -> h3 -> h4 -> h5 -> h6 -> hr 的顺序自动切分
                            auto_split_success = False
                            for auto_tag in ['h2', 'h3', 'h4', 'h5', 'h6', 'hr']:
                                # 检查是否有足够的该标签
                                test_pattern = re.compile(rf'<{auto_tag}[^>]*>.*?</{auto_tag}>' if auto_tag != 'hr' else r'<hr[^>]*/?>', re.IGNORECASE | re.DOTALL)
                                matches = test_pattern.findall(content_html_str)
                                
                                if len(matches) >= 2:  # 至少要有 2 个标签才值得切分
                                    logging.info(f"  -> 使用 <{auto_tag}> 自动切分 (找到 {len(matches)} 个标签)")
                                    
                                    if auto_tag == 'hr':
                                        # hr 是自闭合标签，需要特殊处理
                                        split_pattern_re = re.compile(r'(<hr[^>]*?/?>)', re.IGNORECASE)
                                    else:
                                        # 复用现有的切分逻辑
                                        split_pattern_re = re.compile(
                                            rf'(<{auto_tag}[^>]*>.*?</{auto_tag}>)',
                                            re.IGNORECASE | re.DOTALL
                                        )
                                    html_chunks = split_pattern_re.split(content_html_str)
                                    
                                    # 处理概述部分
                                    chunk_0_html = html_chunks[0]
                                    chunk_0_md = convert_html_to_mixed_format(chunk_0_html)
                                    if chunk_0_md:
                                        chunk_0_title = f"{cleaned_node_name} (概述)"
                                        chunk_0_full_path = f"{full_path_str}/{chunk_0_title}"
                                        entry = create_data_entry(chunk_0_md, chunk_0_full_path, chunk_0_title, f"{source_file}#overview")
                                        results.append(entry)
                                    
                                    # 处理各个章节
                                    for i in range(1, len(html_chunks), 2):
                                        tag_html = html_chunks[i]
                                        content_after_tag_html = html_chunks[i + 1] if (i + 1) < len(html_chunks) else ""
                                        full_chunk_html = tag_html + content_after_tag_html
                                        chunk_md = convert_html_to_mixed_format(full_chunk_html)
                                        
                                        if chunk_md:
                                            # 提取章节标题
                                            if auto_tag == 'hr':
                                                # hr 标签没有文本，使用序号
                                                chunk_title = f"Part {i // 2 + 1}"
                                            else:
                                                chunk_soup = BeautifulSoup(tag_html, 'html.parser')
                                                title_tag = chunk_soup.find(auto_tag)
                                                chunk_title = title_tag.get_text(strip=True) if title_tag else f"Section {i // 2 + 1}"
                                            
                                            chunk_full_path = f"{full_path_str}/{chunk_title}"
                                            entry = create_data_entry(chunk_md, chunk_full_path, chunk_title, f"{source_file}#{chunk_title.replace(' ', '_')}")
                                            results.append(entry)
                                    
                                    auto_split_success = True
                                    break  # 成功切分，跳出循环
                            
                            if not auto_split_success:
                                logging.warning(f"  -> 无法自动切分（未找到合适的标题标签），强制按固定大小切分")
                                # 强制按 MAX_DOC_SIZE 大小切分
                                chunk_size = MAX_DOC_SIZE
                                num_chunks = (len(page_content_md) + chunk_size - 1) // chunk_size
                                logging.info(f"  -> 强制切分为 {num_chunks} 个部分")
                                
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
                        else:
                            # 文档大小正常，直接添加
                            entry = create_data_entry(page_content_md, full_path_str, cleaned_node_name, source_file)
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
                                cleaned_node_name,  # <-- V5 change
                                source_file
                            )
                            results.append(entry)

                    else:
                        # --- 1. 处理 "Chunk 0" (第一个 <split_by> 之前的内容) ---
                        # html_chunks[0] 总是第一个 <split_by> 之前的内容
                        chunk_0_html = html_chunks[0]
                        chunk_0_md = convert_html_to_mixed_format(chunk_0_html)

                        if chunk_0_md:
                            # 为这个 "概述" 块创建条目
                            chunk_0_title = f"{cleaned_node_name} (概述)"  # <-- V5 change
                            chunk_0_full_path = f"{full_path_str}/{chunk_0_title}"
                            chunk_0_source_file = f"{source_file}#overview"

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
                                chunk_source_file = f"{source_file}#{chunk_title.replace(' ', '_')}"

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