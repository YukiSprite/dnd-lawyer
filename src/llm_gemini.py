"""
模块: Gemini LLM (src/llm_gemini.py)
根据新设计文档，使用 Google Gemini 1.5 Flash 替代 Ollama

注意: 此模块现在使用 api_config.py 统一管理 API 配置
"""
import sys

try:
    from langchain_core.language_models import BaseChatModel
except ImportError:
    print("ImportError: 必要的 langchain-core 包未安装。")
    print("请运行: pip install langchain-core")
    sys.exit(1)

# 导入统一的 API 配置管理
try:
    from config import api_config
except ImportError:
    print("错误: 无法导入 config/api_config.py")
    print("请确保 config/api_config.py 文件存在")
    sys.exit(1)


def get_gemini_llm() -> BaseChatModel:
    print("--- [LLM] 使用 config/api_config.py 统一配置 ---")
    
    # 使用 api_config 的统一接口创建 LLM
    return api_config.create_gemini_llm()
