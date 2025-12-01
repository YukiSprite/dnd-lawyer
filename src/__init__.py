"""
Source package - 统一模块加载

Agent 已合并，内部根据 config.settings 中的 CURRENT_VERSION 自动选择对应的 prompt

检索器策略：
1. ENABLE_PATH_BOOSTING=True -> PathBoostedRetriever (路径加权)
2. 否则 -> ParentDocumentRetriever (基础版)

混合重排序（关键词优先）在 Agent 层实现，通过 ENABLE_HYBRID_RETRIEVAL 配置启用

使用 get_retriever() 统一获取检索器
"""

from config.settings import CURRENT_VERSION, is_pf, is_dnd

# 统一导入 Agent（Agent 内部会根据版本选择不同的 prompt）
from .agent_gemini import GeminiAgentExecutor, create_gemini_agent_executor

# 导出通用模块
from .llm_gemini import get_gemini_llm
from .parent_retriever import (
    get_parent_retriever, 
    get_path_boosted_retriever, 
    get_retriever,  # 统一的检索器获取函数
    PathBoostedRetriever,
)
from .data_loader import load_rules_documents, split_documents
from .retriever import get_hybrid_retriever as get_tfidf_hybrid_retriever, create_and_save_retriever

__all__ = [
    'GeminiAgentExecutor',
    'create_gemini_agent_executor',
    'get_gemini_llm',
    'get_parent_retriever',
    'get_path_boosted_retriever',
    'get_retriever',  # 推荐使用此函数
    'PathBoostedRetriever',
    'load_rules_documents',
    'split_documents',
    'get_tfidf_hybrid_retriever',
    'create_and_save_retriever',
    'is_pf',
    'is_dnd',
]

