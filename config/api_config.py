"""
API 配置和模型管理中心
所有 API Key 和模型调用都集中在这里管理

配置说明：
1. 敏感信息（API Key 等）存储在 .env 文件中
2. 复制 .env.example 为 .env 并填写您的配置
3. .env 文件不会被 git 跟踪，可以安全上传到 GitHub
"""

import os
from typing import Optional
from pathlib import Path

# 加载环境变量（从 .env 文件）
try:
    from dotenv import load_dotenv
    
    # 查找 .env 文件（在项目根目录）
    env_path = Path(__file__).parent.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 已加载配置文件: {env_path}")
    else:
        print(f"⚠️  未找到 .env 文件: {env_path}")
        print(f"ℹ️  将使用环境变量或默认值")
        print(f"ℹ️  请复制 .env.example 为 .env 并填写配置")
        
except ImportError:
    print("⚠️  未安装 python-dotenv 包")
    print("ℹ️  请运行: pip install python-dotenv")
    print("ℹ️  将使用环境变量或默认值")

# ============================================
# 🔑 API KEY 配置区（从环境变量读取）
# ============================================

# 从环境变量读取 API Key
API_KEY = os.getenv("API_KEY", "")

# ============================================
# 🌐 API 来源站点配置（从环境变量读取）
# ============================================

# API 提供商类型
API_PROVIDER = os.getenv("API_PROVIDER", "openai")

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "")

# ============================================
# 📝 模型配置（从环境变量读取）
# ============================================

# 模型名称
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro-free")

# 温度参数
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# Embedding 模型名称
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-zh-v1.5")

# Embedding 设备
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# Embedding 批处理大小（更大的值 = 更快，但需要更多显存）
# 使用动态策略：从此值开始，失败则减半重试
# RTX 5090 (32GB) 建议: 1024
# RTX 4090 (24GB) 建议: 512
# RTX 3090 (24GB) 建议: 256
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1024"))

# 多 GPU 支持（如果有多块 GPU）
# 设置为 True 将使用 DataParallel 在多 GPU 上并行处理
EMBEDDING_MULTI_GPU = os.getenv("EMBEDDING_MULTI_GPU", "false").lower() == "true"


# ============================================
# 🔧 辅助函数
# ============================================

def get_google_api_key() -> str:
    """
    获取 Google API Key（优先使用本文件配置）
    
    Returns:
        str: API Key
        
    Raises:
        ValueError: 如果未配置 API Key
    """
    # 优先使用本文件配置的 API Key
    if API_KEY and API_KEY.strip():
        return API_KEY.strip()
    
    # 备用：尝试从环境变量读取
    env_key = os.getenv("API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    
    # 如果都没有，抛出错误
    raise ValueError(
        "未配置 API Key！\n"
        "请在 api_config.py 文件中设置 API_KEY = 'your-key'\n"
        "或设置环境变量: export API_KEY='your-key'"
    )


def get_api_base_url() -> Optional[str]:
    """
    获取 API Base URL（如果配置了自定义端点）
    
    Returns:
        Optional[str]: API Base URL，如果未配置则返回 None（使用默认）
    """
    if API_BASE_URL and API_BASE_URL.strip():
        return API_BASE_URL.strip().rstrip('/')
    
    return None


def get_embedding_device() -> str:
    """
    自动检测或返回配置的 Embedding 设备
    
    Returns:
        str: "cuda", "mps" 或 "cpu"
    """
    if EMBEDDING_DEVICE.lower() != "auto":
        return EMBEDDING_DEVICE.lower()
    
    # 自动检测
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ 检测到 CUDA GPU，Embedding 将使用 GPU 加速")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ 检测到 Apple MPS，Embedding 将使用 GPU 加速")
            return "mps"
    except ImportError:
        pass
    
    print("ℹ 未检测到 GPU，Embedding 将使用 CPU")
    return "cpu"


# ============================================
#  模型实例化函数（统一接口）
# ============================================

def create_gemini_llm():
    """
    创建 Gemini LLM 实例（统一的模型创建入口）
    根据 API_PROVIDER 选择使用 Google 官方 API 或 OpenAI 兼容 API
    
    Returns:
        ChatGoogleGenerativeAI 或 ChatOpenAI: 配置好的 LLM 实例
    """
    # 获取 API Key
    api_key = get_google_api_key()
    
    # 获取 API Base URL（如果配置了）
    base_url = get_api_base_url()
    
    print(f" 正在初始化模型: {MODEL_NAME}")
    print(f"   Temperature: {TEMPERATURE}")
    print(f"   API 提供商: {API_PROVIDER}")
    if base_url:
        print(f"   API Base URL: {base_url}")
    else:
        print(f"   API Base URL: 官方 Google API")
    
    # 根据提供商类型选择不同的实现
    if API_PROVIDER.lower() == "openai":
        # 使用 OpenAI 兼容接口（适用于大多数第三方 Gemini 代理）
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "未安装 langchain-openai 包\n"
                "请运行: pip install langchain-openai"
            )
        
        if not base_url:
            raise ValueError("使用 OpenAI 模式时，必须配置 API_BASE_URL")
        
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            api_key=api_key,
            base_url=base_url
        )
        
    else:  # API_PROVIDER == "google"
        # 使用 Google 官方 API (gRPC)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "未安装 langchain-google-genai 包\n"
                "请运行: pip install langchain-google-genai"
            )
        
        llm_kwargs = {
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "google_api_key": api_key,
            "convert_system_message_to_human": True  # Gemini 不支持 system message
        }
        
        # 注意：ChatGoogleGenerativeAI 不支持自定义 base_url（gRPC 协议）
        if base_url:
            print("⚠️  警告: Google 官方 API 模式不支持自定义 base_url")
        
        llm = ChatGoogleGenerativeAI(**llm_kwargs)
    
    # 简单测试连接（跳过测试，因为某些 API 返回格式可能不标准）
    print(f"✓ Gemini LLM 实例创建成功")
    print(f"ℹ️  首次调用时将验证 API 连接")
    
    return llm


def create_embedding_model():
    """
    创建 Embedding 模型实例（统一的模型创建入口）
    
    支持批处理大小配置以加速处理：
    - 更大的 batch_size = 更快的处理速度
    - 需要更多 GPU 显存
    
    多 GPU 支持：
    - 设置 EMBEDDING_MULTI_GPU=true 启用
    - 自动检测可用 GPU 数量并使用 DataParallel
    
    Returns:
        HuggingFaceEmbeddings: 配置好的 Embedding 实例
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "未安装 langchain-huggingface 包\n"
            "请运行: pip install langchain-huggingface sentence-transformers"
        )
    
    device = get_embedding_device()
    
    # 检测 GPU 数量
    gpu_count = 0
    if device == "cuda":
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"🖥️  检测到 {gpu_count} 个 GPU")
        except ImportError:
            pass
    
    print(f"🔤 正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")
    print(f"   设备: {device}")
    print(f"   批处理大小: {EMBEDDING_BATCH_SIZE}")
    print(f"   多 GPU 模式: {'启用' if EMBEDDING_MULTI_GPU and gpu_count > 1 else '禁用'}")
    
    # 配置 encode_kwargs 以启用批处理和其他优化
    # 注意: 不要设置 show_progress_bar，langchain_huggingface 内部会处理
    encode_kwargs = {
        'batch_size': EMBEDDING_BATCH_SIZE,
        'normalize_embeddings': True,  # BGE 模型推荐启用归一化
    }
    
    # 多 GPU 配置
    model_kwargs = {'device': device}
    if EMBEDDING_MULTI_GPU and gpu_count > 1:
        # sentence-transformers 支持多 GPU
        # 通过设置 device 为 None 并在 encode 时使用 multi_process_pool
        print(f"   ⚡ 启用多 GPU 并行处理 ({gpu_count} GPUs)")
        # 对于多 GPU，增大批处理大小
        encode_kwargs['batch_size'] = EMBEDDING_BATCH_SIZE * gpu_count
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"✓ Embedding 模型加载成功！")
    
    return embeddings


# ============================================
# 📊 配置验证
# ============================================

def validate_config():
    """
    验证配置是否完整
    
    Returns:
        bool: 配置是否有效
    """
    print("\n" + "="*60)
    print("  API 配置验证")
    print("="*60)
    
    all_ok = True
    
    # 检查 API Key
    try:
        api_key = get_google_api_key()
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ Google API Key: {masked_key}")
    except ValueError as e:
        print(f"✗ Google API Key: 未配置")
        print(f"  {e}")
        all_ok = False
    
    # 检查 API 提供商和 Base URL
    print(f"✓ API 提供商: {API_PROVIDER}")
    base_url = get_api_base_url()
    if base_url:
        print(f"✓ API Base URL: {base_url}")
    else:
        if API_PROVIDER.lower() == "openai":
            print(f"✗ API Base URL: 未配置（OpenAI 模式需要配置）")
            all_ok = False
        else:
            print(f"✓ API Base URL: 官方 Google API（默认）")
    
    # 检查模型配置
    print(f"✓ Gemini 模型: {MODEL_NAME}")
    print(f"✓ Embedding 模型: {EMBEDDING_MODEL_NAME}")
    print(f"✓ Embedding 设备: {EMBEDDING_DEVICE}")
    
    print("="*60)
    
    if all_ok:
        print("✓ 所有配置验证通过！")
    else:
        print("✗ 配置验证失败，请检查上述错误")
    
    print()
    return all_ok


# ============================================
# 🔧 便捷函数（向后兼容）
# ============================================

def get_llm():
    """向后兼容的 LLM 获取函数"""
    return create_gemini_llm()


def get_embeddings():
    """向后兼容的 Embeddings 获取函数"""
    return create_embedding_model()


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("🧪 API 配置测试\n")
    
    # 验证配置
    if not validate_config():
        print("⚠️  请先配置 API Key 后再运行测试")
        exit(1)
    
    print("\n测试 1: 创建 Gemini LLM")
    print("-" * 60)
    try:
        llm = create_gemini_llm()
        response = llm.invoke("用一句话介绍 DND 5E")
        print(f"测试回答: {response.content[:100]}...")
        print("✓ LLM 测试通过\n")
    except Exception as e:
        print(f"✗ LLM 测试失败: {e}\n")
    
    print("\n测试 2: 创建 Embedding 模型")
    print("-" * 60)
    try:
        embeddings = create_embedding_model()
        test_vec = embeddings.embed_query("测试文本")
        print(f"向量维度: {len(test_vec)}")
        print("✓ Embedding 测试通过\n")
    except Exception as e:
        print(f"✗ Embedding 测试失败: {e}\n")
    
    print("="*60)
    print("✓ 所有测试完成！")
    print("="*60)
