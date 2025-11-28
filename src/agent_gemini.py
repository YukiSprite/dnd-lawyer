"""
模块: Agent (src/agent_gemini.py)
针对 Gemini 1.5 Flash 超大上下文窗口优化的简化 Agent
"""

from typing import Dict, Any, List
import re
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config

# Agent 提示词模板
GEMINI_AGENT_TEMPLATE = """你是一个专业的 Pathfinder 规则专家助手。

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

GEMINI_PROMPT = ChatPromptTemplate.from_template(GEMINI_AGENT_TEMPLATE)


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

    def _calculate_semantic_similarity(self, query: str, doc: Document) -> float:
        """
        使用 embedding 模型计算查询与文档的语义相似度
        支持基于路径的相似度加权（正向加权和负向降权）
        
        Args:
            query: 用户查询
            doc: 文档
            
        Returns:
            相似度分数 (0-1)，可能经过路径加权调整
            如果文档被排除，返回 -1.0
        """
        if not self.embedding_model:
            return 1.0  # 如果没有 embedding 模型，默认全部通过
        
        full_path = doc.metadata.get('full_path', '')
        source_title = doc.metadata.get('source_title', '')
        
        # 🆕 路径排除：如果启用且文档路径匹配排除规则，直接返回 -1
        if getattr(config, 'ENABLE_PATH_EXCLUSION', False):
            exclusion_rules = getattr(config, 'PATH_EXCLUSION_RULES', [])
            for exclusion_keyword in exclusion_rules:
                if exclusion_keyword in full_path:
                    # print(f"[Agent] 路径排除: {full_path} 匹配 '{exclusion_keyword}'，跳过")
                    return -1.0  # 标记为排除
        
        try:
            # 获取查询的 embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # 🔧 改进：综合计算标题、路径和内容的相似度
            # 1. 计算与标题的相似度（权重最高，因为标题最能代表文档主题）
            title_similarity = 0.0
            if source_title:
                title_embedding = self.embedding_model.embed_query(source_title)
                title_vec = np.array(title_embedding)
                query_vec = np.array(query_embedding)
                title_similarity = float(np.dot(query_vec, title_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(title_vec)))
            
            # 2. 计算与路径最后部分的相似度（包含具体条目名称）
            path_similarity = 0.0
            if full_path:
                # 取路径的最后一个部分（通常是具体条目名称）
                path_last_part = full_path.split('/')[-1] if '/' in full_path else full_path
                path_embedding = self.embedding_model.embed_query(path_last_part)
                path_vec = np.array(path_embedding)
                query_vec = np.array(query_embedding)
                path_similarity = float(np.dot(query_vec, path_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(path_vec)))
            
            # 3. 计算与内容摘要的相似度（取前500字符，避免噪音）
            content_similarity = 0.0
            doc_text = doc.page_content[:500]
            if doc_text:
                doc_embedding = self.embedding_model.embed_query(doc_text)
                doc_vec = np.array(doc_embedding)
                query_vec = np.array(query_embedding)
                content_similarity = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)))
            
            # 4. 综合相似度：标题 > 路径 > 内容
            # 使用加权平均，优先考虑标题匹配
            base_similarity = max(
                title_similarity * 1.5,      # 标题完全匹配最重要
                path_similarity * 0.8,      # 路径匹配
                content_similarity * 0.8,    # 内容匹配权重稍低
                (title_similarity * 0.5 + path_similarity * 0.3 + content_similarity * 0.2)  # 加权平均
            )
            
            # 路径加权：支持正向加权（提升）和负向加权（降低）
            if getattr(config, 'ENABLE_PATH_BOOSTING', False):
                boost_rules = getattr(config, 'PATH_BOOST_RULES', {})
                
                for path_keyword, boost_value in boost_rules.items():
                    if path_keyword in full_path:
                        # 应用加权（正值提升，负值降低）
                        boosted_similarity = base_similarity + boost_value
                        # 确保相似度在 [0, 1] 范围内
                        boosted_similarity = max(0.0, min(1.0, boosted_similarity))
                        
                        # 调试信息（可选）
                        if boost_value >= 0:
                            print(f"[Agent] 路径加权↑: {full_path[:50]}... 匹配 '{path_keyword}', {base_similarity:.3f} → {boosted_similarity:.3f} (+{boost_value})")
                            pass
                        else:
                            print(f"[Agent] 路径降权↓: {full_path[:50]}... 匹配 '{path_keyword}', {base_similarity:.3f} → {boosted_similarity:.3f} ({boost_value})")
                            pass
                        
                        return boosted_similarity
            
            return base_similarity
        except Exception as e:
            print(f"[Agent] 计算相似度时出错: {e}")
            return 1.0  # 出错时默认通过
    
    def _calculate_doc_to_doc_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        计算两个文档之间的相似度
        综合考虑标题和内容，避免结构相似但主题不同的文档被误判为重复
        
        Args:
            doc1: 文档1
            doc2: 文档2
            
        Returns:
            相似度分数 (0-1)
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            # 获取两个文档的标题
            title1 = doc1.metadata.get('source_title', '')
            title2 = doc2.metadata.get('source_title', '')
            
            # 🔧 改进：首先比较标题相似度
            # 如果标题明显不同，则认为不是重复文档
            title_similarity = 0.0
            if title1 and title2:
                title1_embedding = self.embedding_model.embed_query(title1)
                title2_embedding = self.embedding_model.embed_query(title2)
                
                vec1 = np.array(title1_embedding)
                vec2 = np.array(title2_embedding)
                
                title_similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                
                # 如果标题相似度低于阈值，直接判定为不重复
                # 避免 "气化形体" 和 "黑暗视觉" 这种标题完全不同的法术被误判
                if title_similarity < 0.75:
                    return title_similarity  # 返回较低的标题相似度
            
            # 计算内容相似度（取更多内容以获取效果描述）
            doc1_text = doc1.page_content[:1500]
            doc2_text = doc2.page_content[:1500]
            
            doc1_embedding = self.embedding_model.embed_query(doc1_text)
            doc2_embedding = self.embedding_model.embed_query(doc2_text)
            
            vec1 = np.array(doc1_embedding)
            vec2 = np.array(doc2_embedding)
            
            content_similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            
            # 综合相似度：标题权重更高
            # 只有标题和内容都相似时，才判定为重复文档
            combined_similarity = min(
                title_similarity * 0.6 + content_similarity * 0.4,  # 加权平均
                title_similarity + 0.15  # 标题不同时，限制最高相似度
            )
            
            return combined_similarity
            
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
                    print(f"  ✗ 跳过: {current_title}... 与 {kept_title} (相似度={doc_similarity:.3f}, 重复)")
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
            
            # print(f"\n[Agent] 第 {attempt} 轮：还需 {needed} 个文档，正在检索 {new_retrieve_count} 个候选...")
            
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
                
                # print(f"[Agent] 获取到 {len(new_candidates)} 个新候选文档")
                
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
                            kept_title = kept_doc.metadata.get('source_title', '未知')[:40]
                            # print(f"  ✗ 跳过: {current_title}...与{kept_title} (相似度={doc_similarity:.3f})")
                            is_duplicate = True
                            skipped_count += 1
                            break
                    
                    if not is_duplicate:
                        unique_docs.append(doc)
                        added_in_round += 1
                        if added_in_round <= 5:
                            # print(f"  ✓ 新增: {current_title}... (与查询相似度={sim:.3f})")
                            pass
                
                print(f"[Agent] 第 {attempt} 轮完成: 新增 {added_in_round} 个独特文档，当前共 {len(unique_docs)} 个")
                
                if added_in_round == 0:
                    print(f"[Agent] ⚠️  本轮未找到新的独特文档，停止补充")
                    break
                    
            except Exception as e:
                print(f"[Agent] ⚠️  第 {attempt} 轮检索时出错: {e}")
                break
        
        final_count = len(unique_docs)
        total_checked = len(all_checked_docs)
        
        # print(f"\n[Agent] ✓ 去重与补充完成:")
        # print(f"    - 检查了 {total_checked} 个文档")
        # print(f"    - 保留了 {final_count} 个独特文档")
        # print(f"    - 移除了 {skipped_count} 个重复文档")
        # print(f"    - 达成率: {final_count}/{target_count} ({final_count/target_count*100:.1f}%)")
        
        if final_count < target_count:
            # print(f"[Agent] ⚠️  未能达到目标数量，可能需要：")
            # print(f"    1. 降低相似度阈值 (当前: {similarity_threshold})")
            # print(f"    2. 增加初始检索数量 (当前: {len(initial_docs)})")
            pass
        
        return unique_docs
    
    def _filter_docs_by_similarity(self, query: str, docs: List[Document], threshold: float = 0.5, mode: str = "rank") -> List[Document]:
        """
        基于语义相似度过滤或排序文档
        同时处理路径排除规则
        
        Args:
            query: 用户查询
            docs: 检索到的文档列表
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
        excluded_count = 0
        
        for i, doc in enumerate(docs):
            similarity = self._calculate_semantic_similarity(query, doc)
            
            # 相似度为 -1 表示被排除
            if similarity < 0:
                excluded_count += 1
                continue
            
            doc_scores.append({
                'doc': doc,
                'similarity': similarity,
                'index': i
            })
        
        if excluded_count > 0:
            print(f"[Agent] 路径排除: 已过滤 {excluded_count} 个不符合条件的文档")
        
        if mode == "rank":
            # 按相似度降序排序
            doc_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 显示排序结果（前15个，因为后面会去重）
            print(f"\n[Agent] 文档相似度排名（前 {min(15, len(doc_scores))} 个）：")
            for rank, item in enumerate(doc_scores[:15], 1):
                title = item['doc'].metadata.get('source_title', '未知')
                full_path = item['doc'].metadata.get('full_path', '未知')
                category = full_path.split('/')[0] if '/' in full_path else '未知'
                
                # 标注版本信息
                version_tag = ""
                if "2024" in full_path or "2025" in full_path:
                    version_tag = " 🆕"
                elif any(old in full_path for old in ["玩家手册/", "城主指南/", "怪物图鉴/"]):
                    version_tag = " 📜"
                
                print(f"  {rank}. [{category}]{version_tag} {title[:35]}... 相似度={item['similarity']:.3f}")
            
            # 返回排序后的文档
            filtered_docs = [item['doc'] for item in doc_scores]
            print(f"\n[Agent] 语义排序完成: {len(docs)} 个文档 → {len(filtered_docs)} 个文档（已排除 {excluded_count} 个）")
            
        else:  # mode == "threshold"
            # 按阈值过滤
            filtered_docs = []
            for item in doc_scores:
                title = item['doc'].metadata.get('source_title', '未知')
                print(f"  文档 {item['index']+1}: {title[:30]}... 相似度={item['similarity']:.3f}", end="")
                
                if item['similarity'] >= threshold:
                    filtered_docs.append(item['doc'])
                    print(" ✓ 保留")
                else:
                    print(" ✗ 过滤")
            
            print(f"[Agent] 语义过滤: {len(docs)} 个文档 → {len(filtered_docs)} 个文档（已排除 {excluded_count} 个）")
        
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
            
            # 2. 语义相似度排序/过滤：使用 embedding 自动判断文档相关性（可配置）
            if hasattr(config, 'ENABLE_SEMANTIC_FILTER') and config.ENABLE_SEMANTIC_FILTER:
                filter_mode = getattr(config, 'SEMANTIC_FILTER_MODE', 'rank')
                similarity_threshold = getattr(config, 'SEMANTIC_SIMILARITY_THRESHOLD', 0.4)
                
                retrieved_docs = self._filter_docs_by_similarity(
                    user_input, 
                    retrieved_docs, 
                    similarity_threshold,
                    mode=filter_mode
                )
            else:
                print(f"[Agent] 语义过滤已禁用，跳过过滤")
            
            # 3. 文档去重并补充：移除内容相似的重复文档，并动态补充（可配置）
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
