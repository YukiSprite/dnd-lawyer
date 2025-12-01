"""
模块: 父文档检索器 (src/parent_retriever.py)
实现 Parent Document Retrieval 策略
支持路径加权检索（在搜索阶段直接应用加权）
支持混合检索（关键词优先 + 语义补充）
"""

import os
import sys
import pickle
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# 添加父目录到 sys.path 以便导入 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 进度条支持
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("ℹ️  提示: 安装 tqdm 可显示进度条 (pip install tqdm)")

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
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
    
    # ========================================
    # 动态批处理策略：
    # 1. 首次运行时从配置的 MAX_BATCH_SIZE 开始
    # 2. 如果失败，减半重试当前批次
    # 3. 成功后，记住当前可用的 batch_size，后续批次使用该值
    # 4. 这样可以自动找到最优 batch_size，避免反复 OOM
    # ========================================
    # 从配置读取批处理大小
    try:
        from config import api_config
        MAX_BATCH_SIZE = api_config.EMBEDDING_BATCH_SIZE
    except (ImportError, AttributeError):
        MAX_BATCH_SIZE = 512  # 默认值
    
    MIN_BATCH_SIZE = 5  # 最小批次大小
    print(f"--- [Parent Retriever] 初始批处理大小: {MAX_BATCH_SIZE} (自动调整) ---")
    
    total_docs = len(processed_docs)
    i = 0
    success_count = 0
    
    # 记录已验证可用的 batch_size（首次成功后固定使用）
    proven_batch_size = None
    
    # 创建进度条
    if TQDM_AVAILABLE:
        pbar = tqdm(total=total_docs, desc="构建索引", unit="doc", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = None
    
    while i < total_docs:
        # 如果已经找到可用的 batch_size，直接使用；否则从最大值开始探测
        current_batch_size = proven_batch_size if proven_batch_size else MAX_BATCH_SIZE
        batch_processed = False
        
        while not batch_processed:
            # 确保不超过剩余文档数
            actual_batch_size = min(current_batch_size, total_docs - i)
            batch = processed_docs[i:i + actual_batch_size]
            
            batch_num = success_count + 1
            
            # 只在首次探测或减小 batch 时打印详细信息
            if proven_batch_size is None or current_batch_size != proven_batch_size:
                if pbar:
                    pbar.write(f"尝试 batch_size={actual_batch_size}...")
                else:
                    print(f"--- [Parent Retriever] 批次 {batch_num} (文档 {i+1}-{i+len(batch)}/{total_docs}, batch_size={actual_batch_size}) ---")
            
            try:
                parent_retriever.add_documents(batch, ids=None)
                # 成功！
                i += actual_batch_size
                success_count += 1
                batch_processed = True
                
                # 首次成功，记住这个 batch_size
                if proven_batch_size is None:
                    proven_batch_size = current_batch_size
                    if pbar:
                        pbar.write(f"✓ 确定最优 batch_size={proven_batch_size}")
                    else:
                        print(f"    ✓ 确定最优 batch_size={proven_batch_size}")
                
                # 更新进度条
                if pbar:
                    pbar.update(actual_batch_size)
                    pbar.set_postfix({'batch': proven_batch_size})
                else:
                    print(f"    ✓ 批次 {batch_num} 处理成功 ({i}/{total_docs})")
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是批次过大的错误
                if "Batch size" in error_msg or "batch" in error_msg.lower() or "memory" in error_msg.lower() or "OOM" in error_msg:
                    if current_batch_size <= MIN_BATCH_SIZE:
                        # 已经是最小批次了，逐个处理
                        if pbar:
                            pbar.set_description(f"逐个处理 (batch={current_batch_size})")
                        else:
                            print(f"    ⚠️ 批次大小已最小 ({current_batch_size})，逐个处理...")
                        
                        for idx, single_doc in enumerate(batch):
                            try:
                                parent_retriever.add_documents([single_doc], ids=None)
                                if pbar:
                                    pbar.update(1)
                            except Exception as single_error:
                                if pbar:
                                    pbar.write(f"⚠️ 跳过: {single_doc.metadata.get('source_title', 'unknown')[:30]}")
                                else:
                                    print(f"        ✗ 跳过文档: {single_doc.metadata.get('source_title', 'unknown')[:30]} - {single_error}")
                        
                        # 注意：进度条已在循环中更新，这里不再更新 i（已在循环中处理）
                        i += actual_batch_size
                        success_count += 1
                        batch_processed = True
                    else:
                        # 减半重试
                        current_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                        if pbar:
                            pbar.set_description(f"重试 (batch={current_batch_size})")
                        else:
                            print(f"    ⚠️ 批次过大，减小到 {current_batch_size}，重试...")
                        # 不增加 i，继续循环重试
                else:
                    # 其他未知错误，直接抛出
                    if pbar:
                        pbar.close()
                    print(f"    ✗ 未知错误: {error_msg}")
                    raise

    # 关闭进度条
    if pbar:
        pbar.set_description("索引构建完成")
        pbar.close()
    
    # 持久化向量存储
    print("\n--- [Parent Retriever] 正在持久化向量存储... ---")
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


# --- 3. 路径加权检索器 ---

class PathBoostedRetriever(BaseRetriever):
    """
    路径加权检索器 - 在搜索阶段直接应用路径加权
    
    策略：
    1. 从向量数据库检索更多候选子文档（带相似度分数）
    2. 根据路径规则对相似度分数进行加权调整
    3. 按加权后的分数排序
    4. 映射回父文档并去重
    5. 返回最终结果
    """
    
    vectorstore: Any = None
    docstore: Any = None
    embedding_model: Any = None
    child_splitter: Any = None
    
    # 检索参数
    top_k: int = 40           # 最终返回的文档数
    search_k: int = 200       # 初始搜索的子文档数（更多候选用于加权筛选）
    
    # 路径加权配置
    path_boost_rules: Dict[str, float] = {}
    path_exclusion_rules: List[str] = []
    enable_path_boosting: bool = True
    enable_path_exclusion: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def _apply_path_boost(self, full_path: str, base_score: float) -> float:
        """
        应用路径加权
        
        Args:
            full_path: 文档的完整路径
            base_score: 原始相似度分数
            
        Returns:
            加权后的分数，如果文档应被排除则返回 -1
        """
        # 路径排除检查
        if self.enable_path_exclusion:
            for exclusion_keyword in self.path_exclusion_rules:
                if exclusion_keyword in full_path:
                    return -1.0  # 标记为排除
        
        # 路径加权
        if self.enable_path_boosting and self.path_boost_rules:
            for path_keyword, boost_value in self.path_boost_rules.items():
                if path_keyword in full_path:
                    boosted_score = base_score + boost_value
                    # 确保分数在合理范围内（Chroma 的距离分数可能需要反转）
                    return boosted_score
        
        return base_score
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        执行路径加权检索
        """
        print(f"\n[PathBoostedRetriever] 开始路径加权检索...")
        print(f"[PathBoostedRetriever] 初始检索 {self.search_k} 个子文档，目标返回 {self.top_k} 个父文档")
        
        # 1. 从向量数据库检索子文档（带相似度分数）
        # similarity_search_with_score 返回 (Document, score) 元组
        # 注意：Chroma 返回的是距离分数（越小越好），需要转换
        try:
            child_docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.search_k
            )
        except Exception as e:
            print(f"[PathBoostedRetriever] 检索出错: {e}")
            return []
        
        print(f"[PathBoostedRetriever] 检索到 {len(child_docs_with_scores)} 个子文档")
        
        # 2. 收集所有父文档及其最佳加权分数
        # 一个父文档可能有多个子文档命中，取最高分
        parent_scores: Dict[str, tuple] = {}  # {parent_id: (best_score, parent_doc)}
        excluded_count = 0
        boosted_count = 0
        
        for child_doc, distance_score in child_docs_with_scores:
            parent_doc_id = child_doc.metadata.get("doc_id")
            if not parent_doc_id or parent_doc_id not in self.docstore.store:
                continue
            
            parent_doc = self.docstore.store[parent_doc_id]
            full_path = parent_doc.metadata.get('full_path', '')
            
            # 将距离分数转换为相似度分数（Chroma 使用 L2 距离，越小越好）
            # 转换公式：similarity = 1 / (1 + distance)
            similarity_score = 1.0 / (1.0 + distance_score)
            
            # 应用路径加权
            boosted_score = self._apply_path_boost(full_path, similarity_score)
            
            if boosted_score < 0:
                # 文档被排除
                excluded_count += 1
                continue
            
            if boosted_score != similarity_score:
                boosted_count += 1
            
            # 更新父文档的最佳分数
            if parent_doc_id not in parent_scores:
                parent_scores[parent_doc_id] = (boosted_score, parent_doc, similarity_score)
            else:
                current_best = parent_scores[parent_doc_id][0]
                if boosted_score > current_best:
                    parent_scores[parent_doc_id] = (boosted_score, parent_doc, similarity_score)
        
        print(f"[PathBoostedRetriever] 映射到 {len(parent_scores)} 个唯一父文档")
        if excluded_count > 0:
            print(f"[PathBoostedRetriever] 路径排除: {excluded_count} 个子文档")
        if boosted_count > 0:
            print(f"[PathBoostedRetriever] 路径加权: {boosted_count} 个子文档受到加权调整")
        
        # 3. 按加权分数排序
        sorted_parents = sorted(
            parent_scores.values(),
            key=lambda x: x[0],  # 按加权后的分数排序
            reverse=True
        )
        
        # 4. 显示排序结果（前 15 个）
        print(f"\n[PathBoostedRetriever] 加权后文档排名（前 {min(15, len(sorted_parents))} 个）：")
        for rank, (boosted_score, parent_doc, original_score) in enumerate(sorted_parents[:15], 1):
            title = parent_doc.metadata.get('source_title', '未知')
            full_path = parent_doc.metadata.get('full_path', '未知')
            
            # 显示加权变化
            if abs(boosted_score - original_score) > 0.001:
                boost_delta = boosted_score - original_score
                boost_str = f" ({'+' if boost_delta > 0 else ''}{boost_delta:.3f})"
            else:
                boost_str = ""
            
            print(f"  {rank}. [{full_path}] {title} 分数={boosted_score:.4f}{boost_str}")
        
        # 5. 返回前 top_k 个父文档
        result_docs = [parent_doc for (_, parent_doc, _) in sorted_parents[:self.top_k]]
        print(f"\n[PathBoostedRetriever] 返回 {len(result_docs)} 个文档")
        
        return result_docs


def get_path_boosted_retriever() -> PathBoostedRetriever:
    """
    [在线执行] 加载路径加权检索器。
    
    这是推荐的检索器，在搜索阶段就应用路径加权，
    而不是先搜索再加权再重新排序。
    
    Returns:
        PathBoostedRetriever: 配置好的路径加权检索器
    """
    print("--- [Path Boosted Retriever] 正在加载路径加权检索器... ---")

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
    
    print(f"--- [Path Boosted Retriever] 已加载 {len(docstore.store)} 个父文档 ---")

    # 创建子文档切分器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    # 获取路径加权规则
    path_boost_rules = config.get_path_boost_rules()
    path_exclusion_rules = getattr(config, 'PATH_EXCLUSION_RULES', [])
    
    print(f"--- [Path Boosted Retriever] 路径加权规则: {path_boost_rules} ---")

    # 创建路径加权检索器
    search_k = getattr(config, 'PATH_BOOSTED_SEARCH_K', config.PARENT_RETRIEVER_TOP_K * 5)
    
    retriever = PathBoostedRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        embedding_model=embeddings,
        child_splitter=child_splitter,
        top_k=config.PARENT_RETRIEVER_TOP_K,
        search_k=search_k,  # 使用配置中的值
        path_boost_rules=path_boost_rules,
        path_exclusion_rules=path_exclusion_rules,
        enable_path_boosting=getattr(config, 'ENABLE_PATH_BOOSTING', True),
        enable_path_exclusion=getattr(config, 'ENABLE_PATH_EXCLUSION', False),
    )

    print("--- [Path Boosted Retriever] 加载完成 ---")
    return retriever


# --- 4. 统一的检索器获取函数 ---

def get_retriever() -> BaseRetriever:
    """
    获取配置的检索器
    
    根据配置自动选择使用哪种检索器：
    1. ENABLE_PATH_BOOSTING=True -> PathBoostedRetriever (路径加权)
    2. 否则 -> ParentDocumentRetriever (基础版)
    
    注意：混合检索（关键词优先）现在在 Agent 层实现，
    通过 ENABLE_HYBRID_RETRIEVAL 配置在 agent_gemini.py 中启用
    
    Returns:
        配置好的检索器实例
    """
    # 检查是否启用路径加权
    if getattr(config, 'ENABLE_PATH_BOOSTING', True):
        print("[Retriever] 使用路径加权检索器")
        return get_path_boosted_retriever()
    
    # 默认使用基础检索器
    print("[Retriever] 使用基础父文档检索器")
    return get_parent_retriever()
