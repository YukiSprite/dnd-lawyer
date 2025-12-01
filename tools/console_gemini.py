"""
注意: API Key 配置在 api_config.py 中统一管理
支持 PF (Pathfinder) 和 DND (Dungeons & Dragons) 两种规则版本
版本设置在 config/settings.py 中修改

检索器策略：
1. ENABLE_HYBRID_RETRIEVAL=True -> HybridRetriever (关键词优先 + 语义补充)
2. ENABLE_PATH_BOOSTING=True -> PathBoostedRetriever (路径加权)
3. 否则 -> ParentDocumentRetriever (基础版)
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.llm_gemini import get_gemini_llm
    from src.parent_retriever import get_retriever  # 使用统一的检索器获取函数
    # 使用动态导入，根据 settings.py 中的版本自动选择对应的 Agent
    from src import create_gemini_agent_executor
    from config.settings import get_current_version, get_version_info
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装所需依赖:")
    print("  pip install langchain langchain-google-genai langchain-community")
    print("  pip install langchain-huggingface chromadb scikit-learn")
    sys.exit(1)


def main():
    """主运行函数"""
    # 获取当前版本信息
    version = get_current_version()
    version_info = get_version_info()
    version_name = version_info.get("name", "Unknown")
    
    print("=" * 60)
    print(f"{version_name} 规则 AI 助手")
    print(f"当前版本: {version.upper()}")
    print("=" * 60)
    print("\n正在初始化...")


    try:
        # 1. 验证 API 配置（使用 config/api_config.py）
        print("\n[配置检查] 验证 API 配置...")
        try:
            from config import api_config
            if not api_config.validate_config():
                print("\n!!! 配置验证失败")
                print("请编辑 config/api_config.py 文件，设置 API_KEY")
                sys.exit(1)
        except ImportError:
            print("\n!!! 错误: 无法导入 config/api_config.py")
            print("请确保 config/api_config.py 文件存在")
            sys.exit(1)
        except ValueError as e:
            print(f"\n!!! 配置错误: {e}")
            print("请编辑 config/api_config.py 文件，设置 API_KEY")
            sys.exit(1)

        # 2. 加载 LLM
        print("\n[1/4] 正在加载 LLM...")
        llm = get_gemini_llm()

        # 2. 加载 Embedding 模型（用于语义相似度过滤和文档去重）
        print("\n[2/4] 正在加载 Embedding 模型...")
        try:
            from config import api_config, config
            embedding_model = None
            
            # 如果启用了语义过滤或文档去重，都需要加载 embedding 模型
            need_embedding = config.ENABLE_SEMANTIC_FILTER or config.ENABLE_DOCUMENT_DEDUPLICATION
            
            if need_embedding:
                embedding_model = api_config.create_embedding_model()
                if config.ENABLE_SEMANTIC_FILTER:
                    print(f"[配置] 语义相似度过滤已启用（阈值: {config.SEMANTIC_SIMILARITY_THRESHOLD}）")
                else:
                    print("[配置] 语义相似度重排序已禁用（保留路径加权排序）")
                if config.ENABLE_DOCUMENT_DEDUPLICATION:
                    print(f"[配置] 文档去重已启用（阈值: {config.DOCUMENT_SIMILARITY_THRESHOLD}）")
            else:
                print("[配置] 语义过滤和文档去重均已禁用，跳过加载 Embedding 模型")
        except Exception as e:
            print(f"[警告] 加载 Embedding 模型失败: {e}")
            print("[警告] 将跳过语义相似度过滤和文档去重")
            embedding_model = None

        # 4. 加载检索器（根据配置自动选择）
        print("\n[3/4] 正在加载检索器...")
        try:
            retriever = get_retriever()  # 统一的检索器获取函数
        except Exception as e:
            print(f"[错误] 加载检索器失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # 5. 创建 Agent 执行器
        print("\n[4/4] 正在创建 Gemini Agent...")
        agent_executor = create_gemini_agent_executor(llm, retriever, embedding_model)

    except FileNotFoundError as e:
        print(f"\n!!! 错误: 索引文件未找到")
        print(f"详情: {e}")
        print("\n请先运行索引构建脚本:")
        print("  python scripts/build_index_gemini.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n!!! 严重错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("初始化完成！")
    print("=" * 60)
    print("\n使用说明:")
    print(f"  - 输入您的 {version_name} 规则问题")
    print("  - 系统会返回完整规则文档，保持表格和结构完整性")
    print(f"  - 当前版本: {version.upper()} ({version_name})")
    print("  - 修改 config/settings.py 中的 CURRENT_VERSION 切换版本")
    print()

    # 交互循环
    while True:
        try:
            user_input = input("\n请输入问题 >> ")

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！")
                break

            if not user_input.strip():
                continue

            print("\n助手 (正在思考...):")
            print("-" * 60)
            
            response = agent_executor.invoke({"input": user_input})
            
            print(response['output'])
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n检测到中断，正在退出...")
            break
            
        except EOFError:
            print("\n\n输入流结束，正在退出...")
            break
            
        except Exception as e:
            print(f"\n!!! 运行时错误: {e}")
            print("尝试继续...")

    print("\n感谢使用！")


if __name__ == "__main__":
    main()
