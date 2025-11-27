"""
模块: 父文档检索器 (src/parent_retriever.py)
实现 Parent Document Retrieval 策略
"""

import os
import sys
import pickle
from typing import List

# 添加父目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
    from langchain_core.stores import InMemoryStore
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers.tfidf import TFIDFRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"ImportError: {e}")
    print("请确保安装: pip install langchain langchain-community langchain-huggingface chromadb scikit-learn")
    sys.exit(1)

# 导入项目配置
from config import config


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """
    加载 Embedding 模型
    
    使用 config/api_config.py 中的统一配置
    """
    print("--- [Parent Retriever] 使用 config/api_config.py 统一配置 ---")
    
    # 导入统一的 API 配置
    try:
        from config import api_config
        return api_config.create_embedding_model()
    except ImportError:
        print("警告: 无法导入 config/api_config，使用备用配置")
        # 备用方案：使用 config.py
        from config import config
        device = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("--- [Parent Retriever] 使用 CUDA GPU 加速")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print("--- [Parent Retriever] 使用 Apple MPS GPU 加速")
        except ImportError:
            print("--- [Parent Retriever] PyTorch 未安装，使用 CPU")

        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device}
        )


# --- 1. 索引构建 (离线) ---

def create_and_save_parent_retriever(parent_documents: List[Document]):
    """
    [离线执行] 创建父文档检索器并保存索引。
    
    Args:
        parent_documents: 完整的父文档列表（未切分）
    """
    if not parent_documents:
        print("!!! [Parent Retriever] 没有可用于索引的文档。")
        return

    print(f"--- [Parent Retriever] 开始构建父文档索引... ---")
    print(f"--- [Parent Retriever] 父文档数量: {len(parent_documents)} ---")

    # 初始化 Embedding 模型
    embeddings = _get_embedding_model()

    # 创建父文档存储（内存）
    docstore = InMemoryStore()
    
    # 创建子文档切分器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    # 确保目录存在
    os.makedirs(config.DB_PATH, exist_ok=True)
    print(f"--- [Parent Retriever] 向量数据库目录: {config.DB_PATH} ---")

    # 创建 Chroma 向量存储（用于子文档）
    print("--- [Parent Retriever] 正在创建 Chroma 向量存储... ---")
    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=config.DB_PATH
    )
    print("--- [Parent Retriever] Chroma 向量存储创建成功 ---")

    # 创建父文档检索器
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
    )

    # 添加文档（自动完成父子映射）- 分批处理以避免 ChromaDB 批处理限制
    print("--- [Parent Retriever] 正在添加文档并构建索引... ---")
    
    # 预处理：检测超大文档并手动切分
    print("--- [Parent Retriever] 预处理：检测超大文档... ---")
    MAX_SINGLE_DOC_SIZE = 2_000_000  # 单个文档最大 200万字符
    processed_docs = []
    
    for doc in parent_documents:
        doc_size = len(doc.page_content)
        if doc_size > MAX_SINGLE_DOC_SIZE:
            # 超大文档，手动切分成多个小文档
            title = doc.metadata.get('source_title', 'Untitled')
            full_path = doc.metadata.get('full_path', 'Unknown')
            print(f"⚠️  发现超大文档: {title} ({doc_size:,} 字符)，手动切分...")
            
            # 按固定大小切分（确保不会生成过多子文档）
            chunk_size = 500_000  # 每个切片 50万字符
            num_chunks = (doc_size + chunk_size - 1) // chunk_size
            print(f"   将切分为 {num_chunks} 个部分")
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, doc_size)
                chunk_content = doc.page_content[start:end]
                
                # 创建子文档
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        'source_title': f"{title} (Part {i+1}/{num_chunks})",
                        'full_path': f"{full_path}/Part{i+1}",
                        'source_file': doc.metadata.get('source_file', 'N/A'),
                        'is_split': True,
                        'original_doc': title
                    }
                )
                processed_docs.append(chunk_doc)
        else:
            processed_docs.append(doc)
    
    print(f"--- [Parent Retriever] 预处理完成: {len(parent_documents)} 个原始文档 -> {len(processed_docs)} 个处理后文档 ---")
    
    # ChromaDB 的子文档批处理限制约为 5461
    # 父文档会被切分成多个子文档，需要使用更小的父文档批次
    batch_size = 30  # 初始批次大小（减小以避免超限）
    total_docs = len(processed_docs)
    
    i = 0
    while i < total_docs:
        batch = processed_docs[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        print(f"--- [Parent Retriever] 正在处理批次 {batch_num}/{total_batches} (父文档 {i+1}-{min(i+batch_size, total_docs)}, 批次大小={batch_size}) ---")
        
        try:
            parent_retriever.add_documents(batch, ids=None)
            i += batch_size  # 成功，处理下一批
        except Exception as e:
            if "Batch size" in str(e) and "greater than max batch size" in str(e):
                # 批次太大，减半重试
                if batch_size <= 5:
                    # 已经很小了，单个文档处理
                    print(f"⚠️  警告: 批次大小已降至最小 ({batch_size})，尝试逐个处理...")
                    for single_doc in batch:
                        try:
                            parent_retriever.add_documents([single_doc], ids=None)
                        except Exception as single_error:
                            print(f"⚠️  跳过超大文档: {single_doc.metadata.get('source_title', 'unknown')} - {single_error}")
                    i += batch_size
                else:
                    # 减半批次大小重试
                    batch_size = max(5, batch_size // 2)
                    print(f"⚠️  批次过大，减小批次大小为 {batch_size}，重试当前批次...")
                    # 不增加 i，重试当前位置
            else:
                # 其他错误，直接抛出
                raise

    # 持久化向量存储
    print("--- [Parent Retriever] 正在持久化向量存储... ---")
    vectorstore.persist()
    print(f"--- [Parent Retriever] 向量库已保存到 {config.DB_PATH} ---")

    # 保存父文档存储
    docstore_path = os.path.join(config.DB_PATH, "parent_docstore.pkl")
    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore.store, f)
    print(f"--- [Parent Retriever] 父文档存储已保存到 {docstore_path} ---")

    print("--- [Parent Retriever] 索引构建完成 ---")


# --- 2. 应用运行时加载 (在线) ---

def get_parent_retriever() -> ParentDocumentRetriever:
    """
    [在线执行] 加载父文档检索器。
    
    Returns:
        ParentDocumentRetriever: 配置好的父文档检索器
    """
    print("--- [Parent Retriever] 正在加载父文档检索器... ---")

    # 检查文件是否存在
    if not os.path.exists(config.DB_PATH) or not os.listdir(config.DB_PATH):
        raise FileNotFoundError(f"向量数据库未找到: {config.DB_PATH}")
    
    docstore_path = os.path.join(config.DB_PATH, "parent_docstore.pkl")
    if not os.path.exists(docstore_path):
        raise FileNotFoundError(f"父文档存储未找到: {docstore_path}")

    # 加载 Embedding 模型
    embeddings = _get_embedding_model()

    # 加载 Chroma 向量存储
    vectorstore = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=config.DB_PATH
    )

    # 加载父文档存储
    docstore = InMemoryStore()
    with open(docstore_path, 'rb') as f:
        docstore.store = pickle.load(f)
    
    print(f"--- [Parent Retriever] 已加载 {len(docstore.store)} 个父文档 ---")

    # 创建子文档切分器（必须与索引时一致）
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    # 创建父文档检索器
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": config.PARENT_RETRIEVER_TOP_K}
    )

    print("--- [Parent Retriever] 加载完成 ---")
    return parent_retriever
