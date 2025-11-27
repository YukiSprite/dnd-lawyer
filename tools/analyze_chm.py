import os
import json
import sys
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import unquote

# --- 常量定义 ---

# CHM 解包后的源文件夹
CHM_SOURCE_DIR = "chm_source/extracted"

# 阶段 1 的输出文件（存放在 config 目录）
CONFIG_OUTPUT_FILE = "config/data_processing.json"
PREVIEW_OUTPUT_FILE = "config/chm_structure_preview.txt"

# 启发式规则：当一个页面中 H2 或 H3 标签数量超过此阈值时，猜测需要分割
SPLIT_HEURISTIC_THRESHOLD = 9


# --- 基础工具函数 ---

def read_file_with_encodings(filepath: str) -> str | None:
    """
    尝试使用多种编码读取文件。优先识别 BOM，其次尝试常见编码。
    你当前的 .hhc 已转 UTF-8，此函数仍保留健壮性，便于复用到其他 CHM。
    """
    try:
        with open(filepath, "rb") as f:
            raw = f.read()

        # BOM 识别
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig")
        if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            # 常见 UTF-16 BOM
            try:
                return raw.decode("utf-16")
            except UnicodeDecodeError:
                pass

        # 常见编码尝试（utf-8 优先）
        for enc in ["utf-8", "gbk", "gb18030", "utf-16", "utf-16le", "utf-16be", "latin-1"]:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue

        print(f"严重警告: 无法用常见编码解码 {filepath}", file=sys.stderr)
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"警告: 读取 {filepath} 时发生未知错误 {e}", file=sys.stderr)
        return None


def find_hhc_file(source_dir: str) -> str | None:
    """
    在源目录中查找第一个 .hhc 文件。
    """
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".hhc"):
                return os.path.join(root, file)
    return None


def get_soup(doc_text: str) -> BeautifulSoup:
    """
    返回容错更强的 BeautifulSoup。
    优先 html5lib，其次 lxml，最后退回 html.parser。
    """
    try:
        return BeautifulSoup(doc_text, "html5lib")
    except Exception:
        try:
            return BeautifulSoup(doc_text, "lxml")
        except Exception:
            return BeautifulSoup(doc_text, "html.parser")


def ensure_unique_key(target_dict: dict, key: str) -> str:
    """
    在 target_dict 中为 key 生成唯一键。
    若已存在 'key'，则依次尝试 'key__2', 'key__3', ...
    返回最终使用的键名。
    """
    if key not in target_dict:
        return key
    idx = 2
    while True:
        candidate = f"{key}__{idx}"
        if candidate not in target_dict:
            return candidate
        idx += 1


def normalize_local(value: str | None) -> str | None:
    """
    将空字符串或仅空白的 Local 视为 None。
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


# --- 选择器统计 ---

def update_selector_counter(soup: BeautifulSoup, counter: Counter):
    """
    遍历文档中所有带 id / class 的元素，统计潜在 CSS 选择器。
    以 tag#id / tag.class 计数，更广覆盖（不仅限 div）。
    """
    for el in soup.find_all(True):  # True => 所有标签
        tag = el.name
        if not tag:
            continue
        el_id = el.get("id")
        if el_id:
            # id 正常为 str，但 robust 一点
            if isinstance(el_id, list) and el_id:
                el_id = el_id[0]
            if isinstance(el_id, str):
                counter[f"{tag}#{el_id}"] += 1

        classes = el.get("class")
        if classes:
            for cls in classes:
                if isinstance(cls, str) and cls.strip():
                    counter[f"{tag}.{cls}"] += 1


def format_potential_selectors(counter: Counter, top_n: int = 20) -> list[str]:
    """
    将 Counter 转换为排序后的字符串列表，用于 config.json 展示。
    """
    return [f"{selector} (出现 {count} 次)" for selector, count in counter.most_common(top_n)]


# --- HHC 解析 ---

def parse_hhc_li(li_element: BeautifulSoup) -> dict | None:
    """
    解析单个 <LI> 节点，抽取 <object type="text/sitemap"> -> Name/Local，
    并解析其子树 <UL>。
    对于不规范 HHC：如果直属没有 <UL>，尝试把“紧邻的兄弟 <UL>”当作它的子树。
    """
    obj = li_element.find("object", attrs={"type": "text/sitemap"}, recursive=False)
    if not obj:
        return None

    name_param = obj.find("param", attrs={"name": "Name"})
    local_param = obj.find("param", attrs={"name": "Local"})

    name = name_param.get("value") if name_param else None
    local_file = normalize_local(local_param.get("value") if local_param else None)

    if not name:
        return None

    node_data = {"name": name, "local": local_file, "children": []}

    # 优先：直属 <ul>
    nested_ul = li_element.find("ul", recursive=False)

    # 兼容：<LI> 后面的紧邻兄弟是 <UL>（不闭合 LI 的常见写法）
    if not nested_ul:
        sib = li_element.next_sibling
        # 跳过空白文本节点
        while sib is not None and not getattr(sib, "name", None):
            sib = sib.next_sibling
        if getattr(sib, "name", None) == "ul":
            nested_ul = sib

    if nested_ul:
        node_data["children"] = parse_hhc_ul(nested_ul)

    return node_data


def parse_hhc_ul(ul_element: BeautifulSoup) -> list[dict]:
    """
    递归解析 <UL>。为了对付破损的 (X)HTML，不使用递归查找；
    只遍历“看起来像直接孩子”的 li。
    """
    nodes = []
    if not hasattr(ul_element, "contents"):
        return nodes

    for child in ul_element.contents:
        if getattr(child, "name", None) == "li":
            node = parse_hhc_li(child)
            if node:
                nodes.append(node)
    return nodes


def parse_hhc_file(hhc_file_path: str) -> list[dict]:
    """
    解析 .hhc 的入口：定位根 UL/LI，并展开为节点列表（森林）。
    """
    print(f"正在解析 HHC 文件: {hhc_file_path}")
    content = read_file_with_encodings(hhc_file_path)
    if not content:
        raise Exception(f"无法读取 HHC 文件: {hhc_file_path}")

    soup = get_soup(content)
    body = soup.find("body")
    if not body:
        raise Exception("HHC 文件中未找到 <BODY> 标签")

    # 首选：body 下的直系 <ul> 作为根
    root_uls = body.find_all("ul", recursive=False)

    # 回退：若没抓到，抓所有 <ul>，过滤掉那些是某个 <li> 的后代（近似根）
    if not root_uls:
        all_uls = body.find_all("ul")
        root_uls = [u for u in all_uls if not u.find_parent("li")]

    all_nodes = []
    for ul in root_uls:
        all_nodes.extend(parse_hhc_ul(ul))

    # 极端破损：顶层 <li> 直接挂在 body
    for li in body.find_all("li", recursive=False):
        node = parse_hhc_li(li)
        if node:
            all_nodes.append(node)

    return all_nodes


# --- 分析与配置生成 ---

def analyze_node_recursive(
    raw_nodes: list[dict],
    chm_dir: str,
    processing_rules_output: dict,
    selector_counter: Counter,
    depth: int = 0,
) -> list[str]:
    """
    深度优先遍历 HHC 节点，分析对应 HTML，写出规则，并生成预览行。
    """
    preview_lines = []

    for idx, node in enumerate(raw_nodes, 1):
        title = node["name"]
        local_file = node["local"]

        node_config = {
            "action": None,
            "split_by": None,
            "selector": None,  # 默认为 null，将继承 default_content_selector
            "source_file": local_file,  # 方便调试
            "children": {}
        }

        analysis_info = ""

        if local_file:
            node_config["action"] = "process"
            
            # 尝试原始文件名和 URL 解码后的文件名
            html_path = os.path.join(chm_dir, local_file)
            if not os.path.exists(html_path):
                # 如果原始路径不存在，尝试 URL 解码（%20 -> 空格等）
                decoded_file = unquote(local_file)
                html_path = os.path.join(chm_dir, decoded_file)
            
            content = read_file_with_encodings(html_path)

            if content:
                soup = get_soup(content)

                # 1) 收集选择器统计
                update_selector_counter(soup, selector_counter)

                # 2) 启发式分割
                h_counts = [len(soup.find_all(f"h{i}")) for i in range(1, 7)]
                analysis_info = f"[h_counts: {h_counts}]"

                if h_counts[0] > SPLIT_HEURISTIC_THRESHOLD:  # h1
                    node_config["split_by"] = "h1"
                elif h_counts[1] > SPLIT_HEURISTIC_THRESHOLD:  # h2
                    node_config["split_by"] = "h2"
                elif h_counts[2] > SPLIT_HEURISTIC_THRESHOLD:  # h3
                    node_config["split_by"] = "h3"
                elif h_counts[3] > SPLIT_HEURISTIC_THRESHOLD:  # h4
                    node_config["split_by"] = "h4"
                else:
                    node_config["split_by"] = None
            else:
                analysis_info = "[文件未找到!]"
                node_config["action"] = "skip_all"

        else:
            node_config["action"] = "skip_children_only"
            node_config["split_by"] = None
            analysis_info = "[目录节点]"

        # 唯一键：标题 + （有则）local，避免覆盖
        base_key = title if not local_file else f"{title}::{local_file}"
        unique_key = ensure_unique_key(processing_rules_output, base_key)
        processing_rules_output[unique_key] = node_config

        # 预览行
        indent = "  " * depth
        preview_line = (
            f"{indent}L {title} (file: {local_file}, "
            f"action: {node_config['action']}, split: {node_config['split_by']}) {analysis_info}"
        )
        preview_lines.append(preview_line)

        # 递归子节点
        if node["children"]:
            child_lines = analyze_node_recursive(
                node["children"],
                chm_dir,
                node_config["children"],
                selector_counter,
                depth=depth + 1,
            )
            preview_lines.extend(child_lines)

    return preview_lines


# --- Main 执行 ---

def main():
    """
    阶段 1 - 解析 HHC + 走 HTML 启发式，输出 default_config.json 与 structure_preview.txt
    """
    print(f"--- 阶段 1: 分析 CHM 结构 ---")
    print(f"源目录: {CHM_SOURCE_DIR}")

    # 1) 定位 .hhc
    hhc_file_path = find_hhc_file(CHM_SOURCE_DIR)
    if not hhc_file_path:
        print(f"错误: 在 {CHM_SOURCE_DIR} 中未找到 .hhc 文件。", file=sys.stderr)
        return
    print(f"找到 HHC 文件: {hhc_file_path}")

    # 2) 解析 HHC -> 森林
    try:
        raw_root_nodes = parse_hhc_file(hhc_file_path)
    except Exception as e:
        print(f"解析 HHC 文件时出错: {e}", file=sys.stderr)
        return

    # 3) 递归分析 HTML
    print("正在递归分析 HHC 树和 HTML 文件...")
    selector_counter = Counter()
    tree_processing_rules = {}

    preview_lines = analyze_node_recursive(
        raw_root_nodes,
        CHM_SOURCE_DIR,
        tree_processing_rules,
        selector_counter,
        depth=0,
    )
    print("分析完成。")

    # 4) 生成 common_config
    potential_selectors = format_potential_selectors(selector_counter)
    default_selector = (
        selector_counter.most_common(1)[0][0] if selector_counter else "body"
    )

    common_config = {
        "default_content_selector": default_selector,
        "potential_selectors_found": potential_selectors,
    }

    # 5) 组装最终配置
    final_config = {
        "chm_source_path": CHM_SOURCE_DIR,
        "hhc_file": os.path.basename(hhc_file_path),
        "output_file": "./data/rules_data.json",
        "common_config": common_config,
        "tree_processing_rules": tree_processing_rules,
    }

    # 6) 写入 default_config.json
    try:
        output_dir = os.path.dirname(CONFIG_OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(CONFIG_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_config, f, ensure_ascii=False, indent=2)
        print(f"成功生成配置文件初稿: {CONFIG_OUTPUT_FILE}")
    except Exception as e:
        print(f"写入 {CONFIG_OUTPUT_FILE} 时出错: {e}", file=sys.stderr)

    # 7) 写入 structure_preview.txt
    try:
        output_dir = os.path.dirname(PREVIEW_OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(PREVIEW_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(f"# CHM 结构预览 (源: {hhc_file_path})\n")
            f.write("# (这是分析脚本的猜测，请基于此审查并修改 config.json)\n")
            f.write("-" * 40 + "\n")
            for line in preview_lines:
                f.write(line + "\n")
        print(f"成功生成结构预览文件: {PREVIEW_OUTPUT_FILE}")
    except Exception as e:
        print(f"写入 {PREVIEW_OUTPUT_FILE} 时出错: {e}", file=sys.stderr)

    print(f"--- 阶段 1 完成 ---")
    print(
        "下一步: \n"
        "配置文件已生成到 config/ 目录:\n"
        f"  - {CONFIG_OUTPUT_FILE}\n"
        f"  - {PREVIEW_OUTPUT_FILE}\n"
        "可以直接运行 package_json.py，或根据需要手动调整配置。"
    )


if __name__ == "__main__":
    # 运行此脚本前，请确保已安装 beautifulsoup4 以及 html5lib（或 lxml）
    # pip install beautifulsoup4 html5lib
    main()
