#!/usr/bin/env python3
"""
索引构建脚本
使用父文档检索策        print("="*60)
        print("\n下一步:")
        print("  1. 编辑 api_config.py，设置 API_KEY")
        print("  2. 运行应用: python console_gemini.py")
        print("\n提示: 也可以设置环境变量: export API_KEY='your-key'")

注意: API 配置在 api_config.py 中统一管理
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_rules_documents
from src.parent_retriever import create_and_save_parent_retriever


def main():
    """
    主函数：构建父文档检索索引
    """
    print("=" * 60)
    print("DND 规则 AI 助手 - 索引构建工具")
    print("=" * 60)
    
    # 验证 API 配置（虽然构建索引不需要 Gemini API，但需要 Embedding）
    print("\n[配置检查] 验证 API 配置...")
    try:
        from config import api_config
        # 只验证 Embedding 相关配置
        print(f"✓ Embedding 模型: {api_config.EMBEDDING_MODEL_NAME}")
        print(f"✓ Embedding 设备: {api_config.EMBEDDING_DEVICE}")
    except ImportError:
        print("⚠️  警告: 无法导入 config/api_config.py，使用默认配置")
    
    try:
        # 步骤 1: 加载原始文档（不进行切分）
        print("\n[1/2] 正在加载规则文档...")
        parent_documents = load_rules_documents()
        
        if not parent_documents:
            print("!!! 错误: 没有加载到任何文档。")
            print("!!! 请检查 data/rules_data.json 文件是否存在且格式正确。")
            sys.exit(1)
        
        print(f"成功加载 {len(parent_documents)} 个父文档（完整页面）")
        
        # 步骤 2: 创建并保存父文档检索器
        print("\n[2/2] 正在构建父文档检索索引...")
        create_and_save_parent_retriever(parent_documents)
        
        print("\n" + "=" * 60)
        print("索引构建完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 编辑 api_config.py，设置 GOOGLE_API_KEY")
        print("  2. 运行应用: python console_gemini.py")
        print("\n提示: 也可以设置环境变量: export GOOGLE_API_KEY='your-key'")
        
    except FileNotFoundError as e:
        print(f"\n!!! 错误: 文件未找到")
        print(f"!!! 详情: {e}")
        print("!!! 请确保已运行 package_json.py 生成 rules_data.json")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n!!! 严重错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
