"""
模块 3: 检索器 (src/retriever.py)
负责创建、保存和加载检索器。

(v1.1 更新: 增强了 GPU 检测日志)
"""

import os
import sys
import pickle
from typing import List

print("--- 检查当前的 Python 解释器 ---")
print(f"Python 可执行文件路径: {sys.executable}")
print("--- 检查完毕 ---")

try:
    # 1. 核心数据结构 (从 langchain_core 导入)
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    # 2. 经典实现 (从 langchain_classic 导入)
    from langchain_classic.retrievers import (
        EnsembleRetriever,
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline

    # 3. 社区集成包 (从 langchain_community 导入)
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers.tfidf import TFIDFRetriever

    # --- [!! 弃用警告修复 !!] ---
    # 警告提示 HuggingFaceEmbeddings 已移至新包
    # 旧的导入 (已弃用):
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    #
    # 新的导入 (需要 pip install langchain-huggingface):
    from langchain_huggingface import HuggingFaceEmbeddings
    # --- [!! 修复结束 !!] ---


except ImportError as e:
    print(f"ImportError: 捕获到导入错误，真实原因如下：")
    print("==================== 真实错误 ====================")
    print(e)
    print("==================================================")
    print("\n!!! [Retriever] 依赖安装可能不完整。")
    print("!!! 请尝试运行: pip install -U langchain-huggingface langchain-community langchain-core langchain-classic langchain-text-splitters chromadb scikit-learn sentence-transformers")
    sys.exit(1)

# 导入项目配置
import config


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """
    私有辅助函数：加载 Embedding 模型。
    """
    print(f"--- [Retriever] 加载 Embedding 模型: {config.EMBEDDING_MODEL_NAME} ---")

    # (v1.1) 增强的设备检测
    device = 'cpu'
    try:
        if _check_cuda():
            device = 'cuda'
        elif _check_mps():
            device = 'mps'
        else:
            print("--- [Retriever] 未检测到 CUDA (NVIDIA GPU) 或 MPS (Apple GPU)。")
            print("--- [Retriever] Embedding 模型将回退到 CPU 运行。")
    except Exception as e:
        print(f"--- [Retriever] GPU 检测失败: {e}。将回退到 CPU。")

    print(f"--- [Retriever] Embedding 模型将使用: {device} ---")

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )


def _check_cuda():
    """(v1.1) 检查 CUDA (NVIDIA GPU)"""
    try:
        import torch
        is_available = torch.cuda.is_available()
        if not is_available:
            print("--- [Retriever Check] 'torch' 已导入, 但 torch.cuda.is_available() 返回 False。")
            print("--- [Retriever Check] 请确保您安装了支持 CUDA 的 PyTorch 版本。")
            print("--- [Retriever Check] (例如: pip install torch --index-url https://download.pytorch.org/whl/cu121)")
        return is_available
    except ImportError:
        print("--- [Retriever Check] 'torch' 模块未找到。")
        print("--- [Retriever Check] HuggingFace Embeddings 需要 'torch' 才能使用 GPU。")
        return False


def _check_mps():
    """(v1.1) 检查 MPS (Apple Silicon GPU)"""
    try:
        import torch
        # 检查 PyTorch 版本是否支持 MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("--- [Retriever Check] 检测到 Apple MPS。")
            return True
        else:
            print("--- [Retriever Check] 未检测到 Apple MPS 支持。")
            return False
    except ImportError:
        # 如果 'torch' 没装, _check_cuda() 已经提示过了
        return False


# --- 1. 索引构建时调用的函数 (Called by build_index.py) ---

def create_and_save_retriever(documents: List[Document]):
    """
    [离线执行] 由 build_index.py 调用。
    """

    if not documents:
        print("!!! [Retriever] 没有可用于索引的文档，已跳过。")
        return

    # 步骤 1: 初始化 Embedding 模型
    embeddings = _get_embedding_model()

    # 步骤 2: 创建并持久化 ChromaDB (向量检索)
    print(f"--- [Retriever] 正在创建并保存 Chroma 向量库... ---")
    print(f"--- [Retriever] 目标目录: {config.DB_PATH} ---")

    # 确保目录存在
    os.makedirs(config.DB_PATH, exist_ok=True)

    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=config.DB_PATH
        )
        vectorstore.persist()
        print(f"--- [Retriever] Chroma 向量库已成功保存到 {config.DB_PATH} ---")
    except Exception as e:
        print(f"!!! [Retriever] 创建 Chroma 向量库失败: {e}")
        print("!!! 这可能是由于网络问题（无法下载模型）或磁盘权限问题。")
        return

    # 步骤 3: 创建并持久化 TfidfRetriever (关键词检索)
    print(f"--- [Retriever] 正在创建并保存 TF-IDF 检索器... ---")
    print(f"--- [Retriever] 目标文件: {config.TFIDF_PATH} ---")

    # 确保 TFIDF 模型的目录存在
    os.makedirs(os.path.dirname(config.TFIDF_PATH), exist_ok=True)

    try:
        tfidf_retriever = TFIDFRetriever.from_documents(documents=documents)

        # 使用 pickle 保存 TF-IDF 检索器
        with open(config.TFIDF_PATH, 'wb') as f:
            pickle.dump(tfidf_retriever, f)

        print(f"--- [Retriever] TF-IDF 检索器已成功保存到 {config.TFIDF_PATH} ---")
    except Exception as e:
        print(f"!!! [Retriever] 创建 TF-IDF 检索器失败: {e}")


# --- 2. 应用运行时调用的函数 (Called by app.py / agent.py) ---

def get_hybrid_retriever() -> BaseRetriever:
    """
    [在线执行] 由 agent.py 或 console.py 调用。
    """

    print("--- [Retriever] 正在加载混合检索器... ---")

    # 步骤 1: 检查文件是否存在
    if not os.path.exists(config.DB_PATH) or not os.listdir(config.DB_PATH):
        raise FileNotFoundError(f"Chroma 数据库目录未找到或为空: {config.DB_PATH}")
    if not os.path.exists(config.TFIDF_PATH):
        raise FileNotFoundError(f"TF-IDF 索引文件未找到: {config.TFIDF_PATH}")

    # 步骤 2: 加载 Embedding 模型
    embeddings = _get_embedding_model()

    # 步骤 3: 加载 ChromaDB (向量检索器)
    print(f"--- [Retriever] 从 {config.DB_PATH} 加载 Chroma 库... ---")
    vectorstore = Chroma(
        persist_directory=config.DB_PATH,
        embedding_function=embeddings
    )
    chroma_retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVER_TOP_K}
    )
    print("--- [Retriever] Chroma 加载成功。 ---")

    # 步骤 4: 加载 TfidfRetriever (关键词检索器)
    print(f"--- [Retriever] 从 {config.TFIDF_PATH} 加载 TF-IDF 库... ---")
    with open(config.TFIDF_PATH, 'rb') as f:
        tfidf_retriever = pickle.load(f)

    tfidf_retriever.k = config.RETRIEVER_TOP_K
    print("--- [Retriever] TF-IDF 加载成功。 ---")

    # 步骤 5: 组合成 EnsembleRetriever (混合检索器)
    print("--- [Retriever] 正在组合 EnsembleRetriever... ---")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, tfidf_retriever],
        # (v1.2 修复) 修正 v1.1 更新中引入的拼写错误
        # 应该是 RETRIEVER_WEIGHTS (来自 config.py), 而不是 RETRIEVING_WEIGHTS
        weights=config.RETRIEVER_WEIGHTS
    )

    print("--- [Retriever] 混合检索器已准备就绪。 ---")
    return ensemble_retriever