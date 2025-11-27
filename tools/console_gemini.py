"""
注意: API Key 配置在 api_config.py 中统一管理
"""

import sys

try:
    from src.llm_gemini import get_gemini_llm
    from src.parent_retriever import get_parent_retriever
    from src.agent_gemini import create_gemini_agent_executor
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装所需依赖:")
    print("  pip install langchain langchain-google-genai langchain-community")
    print("  pip install langchain-huggingface chromadb scikit-learn")
    sys.exit(1)


def main():
    """主运行函数"""
    print("=" * 60)
    print("DND 规则 AI 助手")
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

        # 3. 加载 Embedding 模型（用于语义相似度过滤）
        print("\n[2/4] 正在加载 Embedding 模型...")
        try:
            from config import api_config, config
            embedding_model = None
            if config.ENABLE_SEMANTIC_FILTER:
                embedding_model = api_config.create_embedding_model()
                print(f"[配置] 语义相似度过滤已启用（阈值: {config.SEMANTIC_SIMILARITY_THRESHOLD}）")
            else:
                print("[配置] 语义相似度过滤已禁用")
        except Exception as e:
            print(f"[警告] 加载 Embedding 模型失败: {e}")
            print("[警告] 将跳过语义相似度过滤")
            embedding_model = None

        # 4. 加载父文档检索器
        print("\n[3/4] 正在加载父文档检索器...")
        retriever = get_parent_retriever()

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
    print("  - 输入您的 DND 规则问题")
    print("  - 系统会返回完整规则文档，保持表格和结构完整性")
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
