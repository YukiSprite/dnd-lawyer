"""
模块: Agent (src/agent_gemini.py)
针对 Gemini 1.5 Flash 超大上下文窗口优化的简化 Agent
支持 PF (Pathfinder) 和 DND (Dungeons & Dragons) 两种规则版本

注意：路径加权逻辑已移至 parent_retriever.py 中的 PathBoostedRetriever
在搜索阶段直接应用加权，而不是后处理

混合检索策略（关键词优先 + 语义补充）在本模块内实现
"""

from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict
import re
import math
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from config.settings import get_current_version, get_version_info

# ============================================
# 版本特定的 Prompt 模板
# ============================================

# Pathfinder 规则专用 Prompt
PF_AGENT_TEMPLATE = """你是一个专业的 Pathfinder 规则专家助手。

你的任务是基于提供的规则文档，准确、详细地回答用户的问题。

## 重要指引：
1. **引用来源**：在回答中明确指出信息来源（使用文档的 full_path）
2. **保持准确**：严格基于提供的规则文档，不要编造信息
3. **结构感知**：注意文档的层级关系（通过 full_path 判断）
4. **表格理解**：文档中可能包含 HTML 表格，请正确解析
5. **完整回答**：如果问题涉及多个方面，请综合所有相关文档
6. **未找到时**：如果文档中没有相关信息，请明确告知

## 检索到的规则文档：

{context}

## 用户问题：

{input}

## 你的回答：
"""

# DND 规则专用 Prompt
DND_AGENT_TEMPLATE = """你是一个专业的 DND 规则专家助手。

你的任务是基于提供的规则文档，准确、详细地回答用户的问题。

## 重要指引：
1. **引用来源**：在回答中明确指出信息来源（使用文档的 full_path）
2. **保持准确**：严格基于提供的规则文档，不要编造信息
3. **结构感知**：注意文档的层级关系（通过 full_path 判断）
4. **表格理解**：文档中可能包含 HTML 表格，请正确解析
5. **完整回答**：如果问题涉及多个方面，请综合所有相关文档
6. **未找到时**：如果文档中没有相关信息，请明确告知

## 检索到的规则文档：

{context}

## 用户问题：

{input}

## 你的回答：
"""

# 根据版本选择 Prompt
def get_agent_template() -> str:
    """根据当前版本获取对应的 Prompt 模板"""
    version = get_current_version()
    if version == "dnd":
        return DND_AGENT_TEMPLATE
    else:
        return PF_AGENT_TEMPLATE

def get_agent_prompt() -> ChatPromptTemplate:
    """获取当前版本的 ChatPromptTemplate"""
    return ChatPromptTemplate.from_template(get_agent_template())

# 默认使用动态获取的 Prompt
GEMINI_PROMPT = get_agent_prompt()


class GeminiAgentExecutor:

    def __init__(self, llm: ChatGoogleGenerativeAI, retriever: BaseRetriever, embedding_model=None):
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model  # 用于语义相似度过滤
        
        # 聊天历史（保持最近 K 轮对话）
        self.chat_history: List[tuple] = []
        self.history_k = 5
        
        # 动态文档数量控制
        self.current_doc_count = config.PARENT_RETRIEVER_MAX_K  # 从最大值开始
        self.max_doc_count = config.PARENT_RETRIEVER_MAX_K
        self.min_doc_count = config.PARENT_RETRIEVER_MIN_K
        
        # 创建 chain
        self.chain = GEMINI_PROMPT | self.llm | StrOutputParser()
        
        # ============================================
        # 关键词检索索引（延迟构建）
        # ============================================
        self._keyword_index: Dict[str, List[Tuple[str, float]]] = {}  # {term: [(doc_id, score), ...]}
        self._doc_term_matrix: Dict[str, Dict[str, float]] = {}  # {doc_id: {term: score}}
        self._idf_scores: Dict[str, float] = {}  # {term: idf_score}
        self._doc_cache: Dict[str, Document] = {}  # {doc_id: Document}
        self._keyword_index_built: bool = False

    # ============================================
    # 关键词检索相关方法
    # ============================================
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词：中文 n-gram + 英文单词
        
        Args:
            text: 待分词文本
            
        Returns:
            分词结果列表
        """
        tokens = []
        
        # 提取英文单词和数字（转小写）
        english_pattern = r'[a-zA-Z0-9]+'
        english_tokens = re.findall(english_pattern, text.lower())
        tokens.extend(english_tokens)
        
        # 提取中文词组
        chinese_pattern = r'[\u4e00-\u9fff]+'
        chinese_segments = re.findall(chinese_pattern, text)
        
        for segment in chinese_segments:
            # 添加完整词（2-10字）
            if 2 <= len(segment) <= 10:
                tokens.append(segment)
            # 添加 2-gram 到 4-gram
            for n in range(2, min(5, len(segment) + 1)):
                for i in range(len(segment) - n + 1):
                    tokens.append(segment[i:i+n])
        
        return tokens
    
    def _build_keyword_index(self, docs: List[Document]):
        """
        为文档列表构建关键词索引
        
        Args:
            docs: 文档列表
        """
        if self._keyword_index_built and len(self._doc_cache) >= len(docs):
            return
        
        print(f"[Agent] 正在构建关键词索引 ({len(docs)} 个文档)...")
        
        # 清空旧索引
        self._keyword_index.clear()
        self._doc_term_matrix.clear()
        self._idf_scores.clear()
        self._doc_cache.clear()
        
        # 计算词频
        doc_term_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        doc_lengths: Dict[str, int] = {}
        term_doc_count: Dict[str, int] = defaultdict(int)
        
        for doc in docs:
            # 生成文档 ID
            doc_id = f"{doc.metadata.get('full_path', 'unknown')}::{hash(doc.page_content[:200])}"
            self._doc_cache[doc_id] = doc
            
            # 获取文档内容
            content = doc.page_content
            title = doc.metadata.get('source_title', '')
            full_path = doc.metadata.get('full_path', '')
            
            # 标题加权（出现3次）
            full_text = f"{title} {title} {title} {full_path} {content}"
            tokens = self._tokenize(full_text)
            
            doc_lengths[doc_id] = len(tokens)
            seen_terms: Set[str] = set()
            
            for token in tokens:
                doc_term_freq[doc_id][token] += 1
                if token not in seen_terms:
                    term_doc_count[token] += 1
                    seen_terms.add(token)
        
        # 计算 IDF
        total_docs = len(docs)
        for term, doc_count in term_doc_count.items():
            self._idf_scores[term] = math.log((total_docs + 1) / (doc_count + 1)) + 1
        
        # 计算 TF-IDF 并构建倒排索引
        avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 1
        
        for doc_id, term_freq in doc_term_freq.items():
            doc_len = doc_lengths[doc_id]
            self._doc_term_matrix[doc_id] = {}
            
            for term, freq in term_freq.items():
                # BM25 风格的 TF 归一化
                k1, b = 1.5, 0.75
                tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_doc_length))
                
                # TF-IDF 分数
                tf_idf = tf_norm * self._idf_scores[term]
                self._doc_term_matrix[doc_id][term] = tf_idf
                
                # 更新倒排索引
                if term not in self._keyword_index:
                    self._keyword_index[term] = []
                self._keyword_index[term].append((doc_id, tf_idf))
        
        # 对倒排索引中的文档按分数排序
        for term in self._keyword_index:
            self._keyword_index[term].sort(key=lambda x: x[1], reverse=True)
        
        self._keyword_index_built = True
        print(f"[Agent] 关键词索引构建完成: {len(self._keyword_index)} 个词项")
    
    def _keyword_search(self, query: str, docs: List[Document]) -> List[Tuple[Document, float, int]]:
        """
        对文档进行关键词检索排序
        
        Args:
            query: 用户查询
            docs: 待排序的文档列表
            
        Returns:
            [(Document, keyword_score, match_count), ...] 按分数降序
        """
        # 构建索引
        self._build_keyword_index(docs)
        
        # 分词查询
        query_tokens = list(set(self._tokenize(query)))  # 去重
        
        # 计算每个文档的匹配分数
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_match_count: Dict[str, int] = defaultdict(int)
        
        for token in query_tokens:
            if token in self._keyword_index:
                for doc_id, score in self._keyword_index[token]:
                    doc_scores[doc_id] += score
                    doc_match_count[doc_id] += 1
        
        # 计算最终分数并排序
        results = []
        for doc_id, base_score in doc_scores.items():
            if doc_id in self._doc_cache:
                # 匹配词数奖励
                match_bonus = 1 + 0.3 * (doc_match_count[doc_id] - 1)
                final_score = base_score * match_bonus
                results.append((self._doc_cache[doc_id], final_score, doc_match_count[doc_id]))
        
        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _hybrid_rerank(
        self, 
        query: str, 
        docs: List[Document],
        keyword_boost: float = 0.5,
        keyword_min_score: float = 0.1
    ) -> List[Document]:
        """
        混合重排序：关键词优先 + 语义排序
        
        策略：
        1. 对检索结果进行关键词匹配打分
        2. 关键词匹配的文档获得固定加分（只加一次，不累加）
        3. 没有关键词匹配的文档保持原语义排序
        
        Args:
            query: 用户查询
            docs: 语义检索返回的文档列表（已按语义相似度排序）
            keyword_boost: 关键词匹配的固定加分值（只加一次）
            keyword_min_score: 关键词匹配的最低分数阈值（归一化后）
            
        Returns:
            重排序后的文档列表
        """
        if not docs:
            return docs
        
        print(f"\n[Agent] 执行混合重排序（关键词优先）...")
        print(f"[Agent] 参数: keyword_boost={keyword_boost}, min_score={keyword_min_score}")
        
        # 1. 关键词检索打分
        keyword_results = self._keyword_search(query, docs)
        
        # 归一化关键词分数到 0-1
        if keyword_results:
            max_score = max(score for _, score, _ in keyword_results)
            min_score = min(score for _, score, _ in keyword_results)
            score_range = max_score - min_score if max_score > min_score else 1
            
            keyword_scores = {}
            for doc, score, match_count in keyword_results:
                doc_id = id(doc)
                norm_score = (score - min_score) / score_range
                keyword_scores[doc_id] = (norm_score, match_count)
        else:
            keyword_scores = {}
        
        # 2. 结合语义排序和关键词打分
        # 原始语义排序的位置分数（越靠前分数越高）
        results = []
        keyword_matched_count = 0
        
        for rank, doc in enumerate(docs):
            doc_id = id(doc)
            
            # 语义排序的位置分数（归一化到 0-1）
            semantic_position_score = 1.0 - (rank / len(docs))
            
            # 关键词匹配分数
            if doc_id in keyword_scores:
                kw_score, match_count = keyword_scores[doc_id]
                
                if kw_score >= keyword_min_score:
                    # 关键词匹配：固定加分（只加一次，不管匹配几个关键词）
                    final_score = semantic_position_score + keyword_boost
                    source = f"keyword({match_count}词)"
                    keyword_matched_count += 1
                    is_boosted = True
                else:
                    final_score = semantic_position_score
                    source = "semantic"
                    is_boosted = False
            else:
                final_score = semantic_position_score
                source = "semantic"
                is_boosted = False
            
            results.append((doc, final_score, source, is_boosted, rank))
        
        # 3. 按最终分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 显示所有文档的重排序结果
        print(f"\n[Agent] 混合重排序结果（关键词匹配: {keyword_matched_count}/{len(docs)} 个文档）：")
        print(f"{'排名':>4} | {'原排名':>6} | {'来源':^15} | {'分数':>6} | {'标题'}")
        print("-" * 80)
        
        for new_rank, (doc, score, source, is_boosted, old_rank) in enumerate(results, 1):
            title = doc.metadata.get('source_title', '未知')[:40]
            icon = "🔑" if is_boosted else "🧠"
            rank_change = old_rank + 1 - new_rank
            
            if rank_change > 0:
                change_str = f"↑{rank_change}"
            elif rank_change < 0:
                change_str = f"↓{-rank_change}"
            else:
                change_str = "="
            
            print(f"{new_rank:>4} | {old_rank+1:>4}{change_str:>2} | {icon} {source:<12} | {score:.3f} | {title}")
        
        # 返回排序后的文档
        reranked_docs = [doc for doc, _, _, _, _ in results]
        print(f"\n[Agent] 混合重排序完成")
        
        return reranked_docs

    def _calculate_semantic_similarity(self, query: str, doc: Document) -> float:
        """
        使用 embedding 模型计算查询与文档的语义相似度
        
        注意：路径加权已移至 PathBoostedRetriever，在搜索阶段直接应用
        此方法现在仅用于文档去重时的相似度计算
        
        Args:
            query: 用户查询
            doc: 文档
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.embedding_model:
            return 1.0  # 如果没有 embedding 模型，默认全部通过
        
        try:
            # 获取查询和文档的 embedding
            query_embedding = self.embedding_model.embed_query(query)

            doc_text = doc.page_content[:]
            doc_embedding = self.embedding_model.embed_query(doc_text)
            
            # 计算余弦相似度
            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)
            
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            return float(similarity)
            
        except Exception as e:
            print(f"[Agent] 计算相似度时出错: {e}")
            return 1.0  # 出错时默认通过
    
    def _calculate_doc_to_doc_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        计算两个文档之间的相似度
        
        Args:
            doc1: 文档1
            doc2: 文档2
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            # 获取两个文档的 embedding
            doc1_text = doc1.page_content[:1000]  # 限制长度以加快计算
            doc2_text = doc2.page_content[:1000]
            
            doc1_embedding = self.embedding_model.embed_query(doc1_text)
            doc2_embedding = self.embedding_model.embed_query(doc2_text)
            
            # 计算余弦相似度
            vec1 = np.array(doc1_embedding)
            vec2 = np.array(doc2_embedding)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            return float(similarity)
        except Exception as e:
            print(f"[Agent] 计算文档间相似度时出错: {e}")
            return 0.0
    
    def _deduplicate_and_refill_documents(
        self, 
        query: str, 
        initial_docs: List[Document], 
        target_count: int = 30,
        similarity_threshold: float = 0.80,
        max_attempts: int = 3
    ) -> List[Document]:
        """
        去重相似文档并动态补充，确保返回足够数量的独特文档
        
        策略：
        1. 对初始文档按相似度去重
        2. 如果去重后不足目标数量，增加检索数量重新检索
        3. 对新检索结果去重（包括与已保留文档比较）
        4. 重复直到达到目标数量或达到最大尝试次数
        
        Args:
            query: 用户查询
            initial_docs: 初始检索到的文档列表（已排序）
            target_count: 目标文档数量
            similarity_threshold: 文档间相似度阈值
            max_attempts: 最大尝试次数
            
        Returns:
            去重并补充后的文档列表
        """
        if not self.embedding_model:
            print("[Agent] ⚠️  未提供 embedding 模型，跳过文档去重")
            return initial_docs[:target_count]
        
        if not initial_docs:
            return initial_docs
        
        print(f"[Agent] 正在进行文档去重与动态补充（目标: {target_count} 个独特文档，阈值: {similarity_threshold}）...")
        
        unique_docs = []
        all_checked_docs = set()  # 使用set存储已检查文档的ID，避免重复检查
        skipped_count = 0
        
        # 第一轮：处理初始文档
        print(f"\n[Agent] 第 1 轮：处理初始 {len(initial_docs)} 个文档...")
        for i, current_doc in enumerate(initial_docs):
            # 生成文档唯一标识
            doc_id = f"{current_doc.metadata.get('full_path', 'unknown')}::{current_doc.page_content[:100]}"
            
            if doc_id in all_checked_docs:
                continue
            all_checked_docs.add(doc_id)
            
            is_duplicate = False
            current_title = current_doc.metadata.get('source_title', '未知')[:40]
            
            # 与已保留的文档比较
            for kept_doc in unique_docs:
                doc_similarity = self._calculate_doc_to_doc_similarity(current_doc, kept_doc)
                
                if doc_similarity >= similarity_threshold:
                    # 发现重复文档
                    kept_title = kept_doc.metadata.get('source_title', '未知')[:40]
                    if i < 10:  # 只显示前10个，避免输出过多
                        print(f"  ✗ 跳过: {current_title}... (相似度={doc_similarity:.3f}, 重复)")
                    is_duplicate = True
                    skipped_count += 1
                    break
            
            if not is_duplicate:
                unique_docs.append(current_doc)
                if i < 10 or len(unique_docs) <= 5:
                    print(f"  ✓ 保留: {current_title}...")
        
        print(f"[Agent] 第 1 轮完成: {len(initial_docs)} 个文档 → {len(unique_docs)} 个独特文档")
        
        # 如果已经足够，直接返回
        if len(unique_docs) >= target_count:
            print(f"[Agent] ✓ 已达到目标数量 ({target_count})，无需补充")
            return unique_docs[:target_count]
        
        # 需要补充的轮次
        attempt = 1
        retrieve_multiplier = 2  # 每次增加检索数量的倍数
        
        while len(unique_docs) < target_count and attempt < max_attempts:
            attempt += 1
            needed = target_count - len(unique_docs)
            
            # 计算新的检索数量（指数增长）
            new_retrieve_count = len(initial_docs) * (retrieve_multiplier ** attempt)
            new_retrieve_count = min(new_retrieve_count, 200)  # 提高上限
            
            print(f"\n[Agent] 第 {attempt} 轮：还需 {needed} 个文档，正在检索 {new_retrieve_count} 个候选...")
            
            try:
                # 🔧 关键修复：直接从 ParentDocumentRetriever 的 vectorstore 检索更多子文档
                # 使用 getattr 避免类型检查错误
                if not hasattr(self.retriever, 'vectorstore') or not hasattr(self.retriever, 'docstore'):
                    print(f"[Agent] ⚠️  检索器不支持动态数量检索，停止补充")
                    break
                
                vectorstore = getattr(self.retriever, 'vectorstore')
                docstore = getattr(self.retriever, 'docstore')
                
                # 从向量数据库检索更多子文档
                child_docs = vectorstore.similarity_search(query, k=new_retrieve_count)
                
                # 从子文档 ID 获取父文档
                more_docs = []
                for child_doc in child_docs:
                    # ParentDocumentRetriever 在子文档的 metadata 中存储父文档 ID
                    parent_doc_id = child_doc.metadata.get("doc_id")
                    if parent_doc_id and parent_doc_id in docstore.store:
                        parent_doc = docstore.store[parent_doc_id]
                        more_docs.append(parent_doc)
                
                print(f"[Agent] 从 {len(child_docs)} 个子文档检索到 {len(more_docs)} 个父文档")
                
                if not more_docs or len(more_docs) <= len(all_checked_docs):
                    print(f"[Agent] ⚠️  没有获取到新文档，停止补充")
                    break
                
                # 计算新文档与查询的相似度并排序
                new_candidates = []
                for doc in more_docs:
                    doc_id = f"{doc.metadata.get('full_path', 'unknown')}::{doc.page_content[:100]}"
                    
                    # 跳过已检查过的文档
                    if doc_id in all_checked_docs:
                        continue
                    
                    sim = self._calculate_semantic_similarity(query, doc)
                    new_candidates.append((doc, sim, doc_id))
                
                # 按相似度排序
                new_candidates.sort(key=lambda x: x[1], reverse=True)
                
                print(f"[Agent] 获取到 {len(new_candidates)} 个新候选文档")
                
                # 处理新候选文档
                added_in_round = 0
                for doc, sim, doc_id in new_candidates:
                    if len(unique_docs) >= target_count:
                        break
                    
                    all_checked_docs.add(doc_id)
                    
                    is_duplicate = False
                    current_title = doc.metadata.get('source_title', '未知')[:40]
                    
                    # 与已保留的文档比较
                    for kept_doc in unique_docs:
                        doc_similarity = self._calculate_doc_to_doc_similarity(doc, kept_doc)
                        
                        if doc_similarity >= similarity_threshold:
                            if added_in_round < 5:  # 只显示前几个
                                print(f"  ✗ 跳过: {current_title}... (相似度={doc_similarity:.3f})")
                            is_duplicate = True
                            skipped_count += 1
                            break
                    
                    if not is_duplicate:
                        unique_docs.append(doc)
                        added_in_round += 1
                        if added_in_round <= 5:
                            print(f"  ✓ 新增: {current_title}... (与查询相似度={sim:.3f})")
                
                print(f"[Agent] 第 {attempt} 轮完成: 新增 {added_in_round} 个独特文档，当前共 {len(unique_docs)} 个")
                
                if added_in_round == 0:
                    print(f"[Agent] ⚠️  本轮未找到新的独特文档，停止补充")
                    break
                    
            except Exception as e:
                print(f"[Agent] ⚠️  第 {attempt} 轮检索时出错: {e}")
                break
        
        final_count = len(unique_docs)
        total_checked = len(all_checked_docs)
        
        print(f"\n[Agent] ✓ 去重与补充完成:")
        print(f"    - 检查了 {total_checked} 个文档")
        print(f"    - 保留了 {final_count} 个独特文档")
        print(f"    - 移除了 {skipped_count} 个重复文档")
        print(f"    - 达成率: {final_count}/{target_count} ({final_count/target_count*100:.1f}%)")
        
        if final_count < target_count:
            print(f"[Agent] ⚠️  未能达到目标数量，可能需要：")
            print(f"    1. 降低相似度阈值 (当前: {similarity_threshold})")
            print(f"    2. 增加初始检索数量 (当前: {len(initial_docs)})")
        
        return unique_docs
    
    def _filter_docs_by_similarity(self, query: str, docs: List[Document], threshold: float = 0.5, mode: str = "rank") -> List[Document]:
        """
        基于语义相似度过滤或排序文档
        
        ⚠️ 警告：此方法会重新计算原始语义相似度，会覆盖 PathBoostedRetriever 的路径加权！
        如果使用 PathBoostedRetriever，建议禁用此功能 (ENABLE_SEMANTIC_FILTER = False)
        
        Args:
            query: 用户查询
            docs: 检索到的文档列表（已经过路径加权排序）
            threshold: 相似度阈值 (0-1)，仅在 mode="threshold" 时使用
            mode: "rank" (按相似度排序) 或 "threshold" (过滤低于阈值的文档)
            
        Returns:
            过滤/排序后的文档列表
        """
        if not self.embedding_model:
            print("[Agent] ⚠️  未提供 embedding 模型，跳过语义过滤")
            return docs
        
        if not docs:
            return docs
        
        if mode == "rank":
            print(f"[Agent] 正在计算语义相似度并排序...")
        else:
            print(f"[Agent] 正在计算语义相似度（阈值: {threshold}）...")
        
        # 计算所有文档与查询的相似度
        doc_scores = []
        
        for i, doc in enumerate(docs):
            similarity = self._calculate_semantic_similarity(query, doc)
            doc_scores.append({
                'doc': doc,
                'similarity': similarity,
                'index': i
            })
        
        if mode == "rank":
            # 按相似度降序排序
            doc_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 显示排序结果（前15个，因为后面会去重）
            print(f"\n[Agent] 文档相似度排名（前 {min(15, len(doc_scores))} 个）：")
            for rank, item in enumerate(doc_scores[:15], 1):
                title = item['doc'].metadata.get('source_title', '未知')
                full_path = item['doc'].metadata.get('full_path', '未知')
                category = full_path.split('/')[0] if '/' in full_path else '未知'
                
                print(f"  {rank}. [{category}] {title[:]}... 相似度={item['similarity']:.3f}")
            
            # 返回排序后的文档
            filtered_docs = [item['doc'] for item in doc_scores]
            print(f"\n[Agent] 语义排序完成: {len(docs)} 个文档 → {len(filtered_docs)} 个文档")
            
        else:  # mode == "threshold"
            # 按阈值过滤
            filtered_docs = []
            for item in doc_scores:
                title = item['doc'].metadata.get('source_title', '未知')
                print(f"  文档 {item['index']+1}: {title[:]}... 相似度={item['similarity']:.3f}", end="")
                
                if item['similarity'] >= threshold:
                    filtered_docs.append(item['doc'])
                    print(" ✓ 保留")
                else:
                    print(" ✗ 过滤")
            
            print(f"[Agent] 语义过滤: {len(docs)} 个文档 → {len(filtered_docs)} 个文档")
        
        return filtered_docs
    
    def _format_documents(self, docs: List[Document]) -> str:
        """格式化文档为上下文字符串"""
        if not docs:
            return "未检索到相关规则文档。"
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            full_path = doc.metadata.get('full_path', '未知来源')
            source_title = doc.metadata.get('source_title', '未知标题')
            content = doc.page_content
            
            formatted.append(f"""
--- 文档 {i} ---
**来源路径**: {full_path}
**标题**: {source_title}

{content}
---
""")
        
        return "\n".join(formatted)

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理用户输入并返回响应。
        支持动态调整文档数量以应对上下文长度限制。
        
        Args:
            input_dict: 包含 'input' 键的字典
            
        Returns:
            包含 'output' 键的字典
        """
        user_input = input_dict.get("input", "未知输入")
        
        try:
            print(f"\n[Agent] 用户输入: {user_input}")
            
            # 1. 检索相关文档（尽可能多）
            initial_retrieve_count = config.PARENT_RETRIEVER_TOP_K
            print(f"[Agent] 正在检索文档（初始检索: {initial_retrieve_count} 个）...")
            retrieved_docs = self.retriever.invoke(user_input)
            print(f"[Agent] 检索到 {len(retrieved_docs)} 个候选文档")
            
            # 2. 混合重排序：关键词优先 + 语义排序（可配置）
            # 这会对语义检索的结果进行二次排序，让精确匹配关键词的文档排在前面
            if getattr(config, 'ENABLE_HYBRID_RETRIEVAL', False):
                keyword_boost = getattr(config, 'KEYWORD_MATCH_BOOST', 0.5)
                keyword_min_score = getattr(config, 'KEYWORD_MIN_SCORE_THRESHOLD', 0.1)
                
                retrieved_docs = self._hybrid_rerank(
                    user_input,
                    retrieved_docs,
                    keyword_boost=keyword_boost,
                    keyword_min_score=keyword_min_score
                )
            
            # 3. 语义相似度排序/过滤（可配置）
            # ⚠️ 注意：PathBoostedRetriever 已经在检索阶段完成了路径加权排序
            # 如果启用此选项，会重新计算原始相似度，覆盖掉路径加权的效果！
            # 建议：如果使用 PathBoostedRetriever，应禁用此选项 (ENABLE_SEMANTIC_FILTER = False)
            elif hasattr(config, 'ENABLE_SEMANTIC_FILTER') and config.ENABLE_SEMANTIC_FILTER:
                filter_mode = getattr(config, 'SEMANTIC_FILTER_MODE', 'rank')
                similarity_threshold = getattr(config, 'SEMANTIC_SIMILARITY_THRESHOLD', 0.4)
                
                print(f"[Agent] ⚠️  警告：启用语义重排序会覆盖 PathBoostedRetriever 的路径加权！")
                retrieved_docs = self._filter_docs_by_similarity(
                    user_input, 
                    retrieved_docs, 
                    similarity_threshold,
                    mode=filter_mode
                )
            else:
                print(f"[Agent] 重排序已禁用（保留检索器的原始排序）")
            
            # 4. 文档去重并补充：移除内容相似的重复文档，并动态补充（可配置）
            if hasattr(config, 'ENABLE_DOCUMENT_DEDUPLICATION') and config.ENABLE_DOCUMENT_DEDUPLICATION:
                dedup_threshold = getattr(config, 'DOCUMENT_SIMILARITY_THRESHOLD', 0.80)
                target_doc_count = getattr(config, 'PARENT_RETRIEVER_TOP_K', 30)
                max_attempts = getattr(config, 'MAX_DEDUP_ATTEMPTS', 3)
                
                retrieved_docs = self._deduplicate_and_refill_documents(
                    user_input,
                    retrieved_docs,
                    target_count=target_doc_count,
                    similarity_threshold=dedup_threshold,
                    max_attempts=max_attempts
                )
            else:
                print(f"[Agent] 文档去重已禁用，跳过去重")
            
            # 4. 限制文档数量到当前设定值（取前N个最相关的）
            final_doc_count = min(self.current_doc_count, len(retrieved_docs))
            retrieved_docs = retrieved_docs[:final_doc_count]
            print(f"[Agent] 最终使用 {len(retrieved_docs)} 个文档（最相关的前 {final_doc_count} 个）")
            
            # 5. 格式化上下文
            context = self._format_documents(retrieved_docs)
            
            # 6. 调用 Gemini（不传递历史，只传递当前问题和规则文档）
            print("[Agent] 正在生成回答...")
            response = self.chain.invoke({
                "context": context,
                "input": user_input
            })
            
            # 7. 拼接文档来源路径到回答末尾
            doc_sources = []
            for doc in retrieved_docs:
                full_path = doc.metadata.get('full_path', '未知来源')
                if full_path not in doc_sources:  # 去重
                    doc_sources.append(full_path)
            
            if doc_sources:
                sources_text = "\n\n" + "="*50 + "\n"
                sources_text += "📚 **参考的规则文档来源**：\n\n"
                for i, source in enumerate(doc_sources, 1):
                    sources_text += f"{i}. {source}\n"
                response_with_sources = response + sources_text
            else:
                response_with_sources = response
            
            # 成功！尝试增加文档数量（渐进式增加）
            if self.current_doc_count < self.max_doc_count:
                self.current_doc_count = min(self.current_doc_count + 1, self.max_doc_count)
                print(f"[Agent] ✓ 响应成功，下次将尝试使用 {self.current_doc_count} 个文档")
            
            # 8. 保存对话历史（仅用于本地记录，不传递给模型）
            self.chat_history.append((user_input, response_with_sources))
            if len(self.chat_history) > self.history_k:
                self.chat_history = self.chat_history[-self.history_k:]
            
            print("[Agent] 回答生成完成")
            
            return {"output": response_with_sources}
            
        except Exception as e:
            error_str = str(e)
            
            # 检测是否是上下文长度超限错误
            if "context length" in error_str.lower() or "max" in error_str.lower() and "token" in error_str.lower():
                print(f"[Agent] ⚠️  上下文长度超限")
                
                # 减少文档数量并重试
                if self.current_doc_count > self.min_doc_count:
                    self.current_doc_count = max(self.current_doc_count - 4, self.min_doc_count)
                    print(f"[Agent] 📉 减少文档数量到 {self.current_doc_count}，正在重试...")
                    
                    # 递归重试
                    return self.invoke(input_dict)
                else:
                    error_msg = f"抱歉，即使使用最少文档数量（{self.min_doc_count}）仍然超出上下文限制。请尝试更简短的问题。"
                    print(f"[Agent] ❌ {error_msg}")
                    self.chat_history.append((user_input, error_msg))
                    return {"output": error_msg}
            
            # 其他错误
            error_msg = f"抱歉，处理您的请求时发生错误: {error_str}"
            print(f"[Agent] 错误: {e}")
            
            # 保存错误到历史
            self.chat_history.append((user_input, error_msg))
            
            return {"output": error_msg}


def create_gemini_agent_executor(
    llm: ChatGoogleGenerativeAI,
    retriever: BaseRetriever,
    embedding_model=None
) -> GeminiAgentExecutor:
    """
    创建 Agent 执行器。
    
    Args:
        llm: Gemini LLM 实例
        retriever: 父文档检索器实例
        embedding_model: Embedding 模型实例（用于语义相似度过滤）
        
    Returns:
        GeminiAgentExecutor 实例
    """
    print("[Agent] 正在创建 Agent 执行器...")
    if embedding_model:
        print("[Agent] ✓ 已启用语义相似度过滤（基于 embedding 模型）")
    return GeminiAgentExecutor(llm=llm, retriever=retriever, embedding_model=embedding_model)
