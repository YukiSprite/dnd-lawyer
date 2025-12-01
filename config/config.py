"""
项目配置文件 (config.py) - Gemini 版本
存储所有硬编码的配置，如路径、模型名称和超参数。
支持 PF (Pathfinder) 和 DND (Dungeons & Dragons) 两种规则版本
"""
from .api_config import MODEL_NAME
from .settings import get_current_version

# 1. 文件路径
# ----------------
# CHM解包器输出的标准化JSON数据文件
DATA_JSON_PATH = "data/rules_data.json"

# ChromaDB 持久化存储目录
DB_PATH = "vector_store/"

# TF-IDF 检索器持久化路径 (使用 pickle)
TFIDF_PATH = "vector_store/tfidf_retriever.pkl"


# 2. 模型名称
# ----------------
# 用于向量化的 Embedding 模型
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# Google Gemini 模型
MODEL_NAME = MODEL_NAME


# 3. 父文档检索 (Parent Document Retrieval) 参数
# ----------------
# 子文档（Child Chunk）切分大小
CHILD_CHUNK_SIZE = 500

# 子文档重叠
CHILD_CHUNK_OVERLAP = 50

# 父文档检索器返回的文档数量
# 初始检索数量：尽可能多地检索文档（用于后续语义排序）
PARENT_RETRIEVER_TOP_K = 40  # 增加初始检索数量

# 路径加权检索器的初始搜索数量（搜索更多候选用于加权筛选）
# 建议设置为 PARENT_RETRIEVER_TOP_K 的 3-5 倍
PATH_BOOSTED_SEARCH_K = 200

# 最大文档数量（动态调整的上限，最终使用的文档数）
PARENT_RETRIEVER_MAX_K = 10  # 最终使用的文档数上限

# 最小文档数量（动态调整的下限）
PARENT_RETRIEVER_MIN_K = 2

# 是否启用语义相似度重排序（使用 embedding 模型重新计算相似度）
# ⚠️ 警告：如果使用 PathBoostedRetriever（默认），应设为 False！
# 因为 PathBoostedRetriever 已经在检索阶段完成了路径加权排序，
# 启用此选项会重新计算原始相似度，覆盖掉路径加权的效果。
ENABLE_SEMANTIC_FILTER = False  # 默认禁用，避免覆盖路径加权

# 语义相似度过滤模式
# "rank": 按相似度降序排序，取前N个（推荐）
# "threshold": 过滤低于阈值的文档
SEMANTIC_FILTER_MODE = "rank"

# 语义相似度阈值（0-1），仅在 mode="threshold" 时使用
# 推荐值：0.3-0.5（太高会过滤掉相关文档，太低则无效）
SEMANTIC_SIMILARITY_THRESHOLD = 0.4

# 启用文档去重与动态补充（移除内容相似的重复文档）
# 去重后如果文档不足，会自动补充更多文档
ENABLE_DOCUMENT_DEDUPLICATION = True

# ============================================
# 混合检索配置（关键词优先 + 语义补充）
# ============================================
# 启用混合检索（关键词优先，语义补充）
# 设为 False 则使用纯语义检索
ENABLE_HYBRID_RETRIEVAL = False

# 混合检索模式：
# "keyword_first": 关键词完全匹配优先，语义检索补充不足部分
# "weighted_fusion": 加权融合关键词和语义分数
# "cascade": 先关键词，没结果时再语义
HYBRID_RETRIEVAL_MODE = "keyword_first"

# 关键词检索返回的最大文档数
KEYWORD_RETRIEVAL_TOP_K = 20

# 关键词匹配的最低分数阈值（TF-IDF 分数，0-1）
# 低于此分数的关键词结果不会被优先
KEYWORD_MIN_SCORE_THRESHOLD = 0.1

# 混合检索中语义结果的权重（0-1）
# 仅在 mode="weighted_fusion" 时生效
SEMANTIC_WEIGHT_IN_HYBRID = 0.3

# 关键词匹配加权值（对精确匹配的文档额外加分）
# 在 "keyword_first" 模式下，关键词匹配的文档会获得此加分
KEYWORD_MATCH_BOOST = 0.1

# 文档间相似度阈值（0-1）
# 如果两个文档的相似度 > 此阈值，则认为它们是重复的
# 推荐值：0.75-0.85（太低会误删不同文档，太高则去重效果不明显）
DOCUMENT_SIMILARITY_THRESHOLD = 0.80

# 去重补充的最大尝试轮数
# 如果去重后文档不足，最多尝试几轮补充
MAX_DEDUP_ATTEMPTS = 3

# 路径加权配置
# 启用基于路径的相似度加权（优先某些来源的文档）
ENABLE_PATH_BOOSTING = True

# ============================================
# 版本特定的路径加权规则
# ============================================

# Pathfinder 版本的路径加权规则
PF_PATH_BOOST_RULES = {
    # 优先规则（正向加权）
    "CRB-核心规则书": 0.1,
    "APG":0.05,
    "ACG":0.05,
    "ARG":0.05,
    "UM":0.05,
    "UC":0.05,
    "MA":0.05,
    "MC":0.05,
    "OA":0.05,
    "UI":0.05,
    "VC":0.05,
    "HA":0.05,
    "AG":0.05,
    "BotD":0.05,
    "UW":0.05,
    "PA":0.05,
    "未整理":-0.05,
    # 降权规则（负向加权）
    # 可根据需要添加
}

# DND 版本的路径加权规则
DND_PATH_BOOST_RULES = {
    # 优先规则（正向加权）
    "Credits": 0.1,           # Credits 来源的文档相似度 +0.1
    "2024": 0.15,             # 2024 版本规则 +0.15（最高优先级）
    "玩家手册2024": 0.15,
    "城主指南2024": 0.15,
    "怪物图鉴2025": 0.12,
    "贤者谏言2025": 0.12,
    
    # 降权规则（负向加权）
    # 注意：降权值不要太大，否则可能完全过滤掉这些文档
    "玩家手册/": -0.15,       # 2014 版玩家手册 -0.15（与 2024 版抵消）
    "城主指南/": -0.15,       # 2014 版城主指南 -0.15
    "怪物图鉴/": -0.12,       # 2014 版怪物图鉴 -0.12
}

# 根据版本动态选择路径加权规则
def get_path_boost_rules() -> dict:
    """根据当前版本获取对应的路径加权规则"""
    version = get_current_version()
    if version == "dnd":
        return DND_PATH_BOOST_RULES
    else:
        return PF_PATH_BOOST_RULES

# 默认使用动态获取的规则
PATH_BOOST_RULES = get_path_boost_rules()

# 路径完全排除规则（可选）
# 如果文档路径包含这些关键词，则完全不返回该文档
# 推荐：先使用降权，只有在确实不需要时才使用排除
ENABLE_PATH_EXCLUSION = False  # 默认禁用，使用降权即可
PATH_EXCLUSION_RULES = [
    # 示例（如果启用）：
    # "旧版说明",
    # "更新日志",
]


# 4. JSON 加载器配置
# ----------------
# JQ Schema: 遍历 JSON 数组
JQ_SCHEMA = ".[]"

# JSON 对象中包含文本内容的键名
CONTENT_KEY_NAME = "page_content"


# 5. Agent 核心逻辑配置（保留用于向后兼容）
# ----------------
# 短输入阈值
SHORT_INPUT_THRESHOLD = 30

# 最大检索循环次数
MAX_RETRIEVAL_LOOPS = 2

# 文档池限制
DOC_POOL_LIMIT = 8
