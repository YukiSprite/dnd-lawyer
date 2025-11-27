"""
模块 1: 数据加载器 (src/data_loader.py)
负责加载和分割来自 rules_data.json 的数据。
"""

import sys
from typing import List, Dict, Any

try:
    from langchain_community.document_loaders import JSONLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError as e:
    print(f"ImportError: 捕获到导入错误，真实原因如下：")
    print("==================== 真实错误 ====================")
    print(e)
    print("==================================================")
    print("\n你原有的提示（可能不准确）: pip install langchain langchain-community langchain-text-splitters")
    sys.exit(1)

# 添加父目录到 sys.path 以便导入 config
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目配置
from config import config


def _extract_metadata_from_record(
    record: Dict[str, Any],
    default_metadata: Dict[str, Any] # (我们忽略这个参数，转而使用 'record')
) -> Dict[str, Any]:
    """
    [!! 关键修复 !!]

    此函数用于 JSONLoader 的 'metadata_func' 参数。
    'record' 参数是原始 JSON 文件中的单个对象，例如：
    {
        "page_content": "...",
        "metadata": {
            "full_path": "...",
            "source_title": "..."
        }
    }

    我们的目标是提取 'metadata' 键中的字典，并将其作为 Document 的元数据返回。
    """

    # 1. 尝试从原始记录(record)中获取 'metadata' 键
    metadata_content = record.get('metadata')

    # 2. 如果它是一个字典，就返回它
    if isinstance(metadata_content, dict):
        # 返回: {'full_path': ..., 'source_title': ...}
        return metadata_content

    # 3. 如果 'metadata' 键不存在或不是字典，返回一个空字典
    print(f"!!! [Data Loader] 警告: 在记录中未找到 'metadata' 键或其格式不是字典。")
    print(f"!!! 问题的记录 (Record): {record}")
    return {}


def load_rules_documents(json_path: str = config.DATA_JSON_PATH) -> List[Document]:
    """
    根据 PDD 3.x 规范，从 rules_data.json 加载文档。
    """
    print(f"--- [Data Loader] 开始加载: {json_path} ---")

    loader = JSONLoader(
        file_path=json_path,
        jq_schema=config.JQ_SCHEMA,           # ".[]"
        content_key=config.CONTENT_KEY_NAME, # "page_content"

        # --- [!! 关键修复 !!] ---
        # 使用新的、正确的 metadata_func，它直接从 'record' 中提取
        metadata_func=_extract_metadata_from_record
        # --- [!! 修复结束 !!] ---
    )

    try:
        documents = loader.load()
        print(f"--- [Data Loader] 成功加载 {len(documents)} 篇原始文档。 ---")

        # 验证加载的数据是否符合PDD预期
        if documents:
            sample_doc = documents[0]

            # --- [!! 调试打印 !!] ---
            # 满足你的要求：打印出第一个文档的元数据结构
            print("==================================================")
            print("--- [Data Loader] 调试：打印第一个文档的元数据 (sample_doc.metadata) ---")
            print(f"    类型 (Type): {type(sample_doc.metadata)}")
            print(f"    内容 (Content): {sample_doc.metadata}")
            print("==================================================")
            # --- [!! 调试结束 !!] ---


            # [!! 验证 !!]
            # 经过 _extract_metadata_from_record 处理后,
            # sample_doc.metadata 现在应该是:
            # {'full_path': ..., 'source_title': ...}
            # 'full_path' 现在应该位于顶层。
            if 'full_path' not in sample_doc.metadata:
                print("!!! [Data Loader] 警告: 加载的文档 metadata 中缺少 'full_path'。")
                print("!!! 请检查 rules_data.json 的格式是否符合 PDD 3.2 规范。")
            if not sample_doc.page_content:
                print("!!! [Data Loader] 警告: 加载的文档 page_content 为空。")

        return documents

    except Exception as e:
        print(f"!!! [Data Loader] 加载 JSON 文件失败: {e}")
        print("!!! 请确保文件存在且格式正确。")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """
    将加载的 N 个长文档分割成 M 个小块 (Chunks)。
    """
    print(f"--- [Data Loader] 开始分割 {len(documents)} 篇文档... ---")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True  # 增加块的起始索引，有助于调试
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"--- [Data Loader] 分割完成：共 {len(split_docs)} 个文档块。 ---")
    return split_docs