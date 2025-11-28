"""
项目配置文件 (config.py) - Gemini 版本
存储所有硬编码的配置，如路径、模型名称和超参数。
"""
from .api_config import MODEL_NAME
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

# 最大文档数量（动态调整的上限，最终使用的文档数）
PARENT_RETRIEVER_MAX_K = 10  # 最终使用的文档数上限

# 最小文档数量（动态调整的下限）
PARENT_RETRIEVER_MIN_K = 2

# 是否启用语义相似度排序（使用 embedding 模型自动判断文档相关性）
ENABLE_SEMANTIC_FILTER = True

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

# 路径加权规则：{路径关键词: 加权值}
# 如果文档的 full_path 包含关键词，则相似度增加对应值
# 推荐值：0.05-0.15（太高会导致无关文档排名靠前）
# 负值表示降权：-0.1 会降低相似度
PATH_BOOST_RULES = {
    # 优先规则（正向加权）
    "CRB":0.1,
    "FAQ":0.1
    # 降权规则（负向加权）
    # 下面是dnd的例子，pf我还没想好要用哪些
    # 注意：降权值不要太大，否则可能完全过滤掉这些文档
    # "玩家手册/": -0.15,       # 2014 版玩家手册 -0.15（与 2024 版抵消）
    # "城主指南/": -0.15,       # 2014 版城主指南 -0.15
    # "怪物图鉴/": -0.12,       # 2014 版怪物图鉴 -0.12
}

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
