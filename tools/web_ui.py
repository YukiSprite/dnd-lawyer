"""
DND 规则助手 - Streamlit Web 界面

启动方式:
  1. 在项目根目录运行: conda activate dnd && python -m streamlit run tools/web_ui.py --server.port 6008 --server.address 0.0.0.0
  2. 或使用启动脚本: ./run_webui.sh
"""

import streamlit as st
import sys
import os

# 添加项目根目录到 path（tools 目录的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录

# 页面配置（必须在最开始）
st.set_page_config(
    page_title="DND 规则助手 🎲",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    /* 主容器 */
    .main {
        padding: 1rem;
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    
    /* 来源引用样式 */
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    
    /* 加载动画 */
    .stSpinner > div {
        text-align: center;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    
    /* 标题样式 */
    h1 {
        color: #1a237e;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    """加载 Agent（使用缓存避免重复加载）"""
    from src.llm_gemini import get_gemini_llm
    from src.parent_retriever import get_retriever
    from src.agent_gemini import create_gemini_agent_executor
    from config import api_config, config
    
    # 1. 验证 API 配置
    if not api_config.validate_config():
        raise ValueError("API 配置验证失败，请检查 config/api_config.py")
    
    # 2. 加载 LLM
    llm = get_gemini_llm()
    
    # 3. 加载 Embedding 模型（用于语义过滤和文档去重）
    embedding_model = None
    need_embedding = config.ENABLE_SEMANTIC_FILTER or config.ENABLE_DOCUMENT_DEDUPLICATION
    if need_embedding:
        embedding_model = api_config.create_embedding_model()
    
    # 4. 加载检索器
    retriever = get_retriever()
    
    # 5. 创建 Agent
    agent = create_gemini_agent_executor(
        llm=llm,
        retriever=retriever,
        embedding_model=embedding_model
    )
    
    return agent


def format_sources(response: str) -> tuple:
    """
    从响应中分离主要内容和来源引用
    
    Returns:
        (main_content, sources_list)
    """
    if "参考的规则文档来源" in response:
        parts = response.split("=" * 50)
        main_content = parts[0].strip()
        
        # 解析来源
        sources = []
        if len(parts) > 1:
            source_section = parts[1]
            lines = source_section.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and "." in line:
                    # 移除序号
                    source = line.split(".", 1)[1].strip() if "." in line else line
                    sources.append(source)
        
        return main_content, sources
    
    return response, []


def display_message(role: str, content: str, sources: list = None):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <strong>🧑 你的问题：</strong>
            <p>{content}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <strong>🤖 规则助手：</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用 Streamlit 原生 markdown 渲染（支持格式）
        st.markdown(content)
        
        # 显示来源
        if sources:
            with st.expander("📚 参考的规则文档来源", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"`{i}.` {source}")


def main():
    """主函数"""
    
    # 标题
    st.title("🎲 DND 规则助手")
    st.markdown("*基于 AI 的 D&D 5E 规则问答系统*")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 版本信息
        st.markdown("---")
        st.markdown("### 📖 关于")
        st.markdown("""
        **DND 规则助手** 是一个基于大语言模型的规则查询工具。
        
        - 🔍 智能检索规则文档
        - 📚 支持完整的 5E 规则库
        - 🎯 精准引用规则来源
        """)
        
        st.markdown("---")
        
        # 清空对话按钮
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # 示例问题
        st.markdown("### 💡 示例问题")
        example_questions = [
            "圣武士能否投掷近战武器触发至圣斩？",
            "法师在几级时能选择奥术学派？",
            "偷袭的触发条件是什么？",
            "战士的动作如潮如何使用？",
            "什么是借机攻击？",
        ]
        
        for q in example_questions:
            if st.button(q, use_container_width=True, key=f"example_{q}"):
                st.session_state.pending_question = q
                st.rerun()
    
    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # 加载 Agent
    if st.session_state.agent is None:
        try:
            st.session_state.agent = load_agent()
            st.success("✅ 规则数据库加载完成！")
        except Exception as e:
            st.error(f"❌ 加载失败: {e}")
            st.stop()
    
    # 显示历史消息
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"], msg.get("sources"))
    
    # 处理待处理的问题（来自示例按钮）
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # 显示用户消息
        display_message("user", question)
        
        # 生成回答
        with st.spinner("🤔 正在查询规则文档并生成回答..."):
            try:
                result = st.session_state.agent.invoke({"input": question})
                response = result.get("output", "无法获取回答")
                main_content, sources = format_sources(response)
                
                # 添加助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": main_content,
                    "sources": sources
                })
                
                # 显示回答
                display_message("assistant", main_content, sources)
                
            except Exception as e:
                st.error(f"❌ 生成回答时出错: {e}")
        
        st.rerun()
    
    # 用户输入
    st.markdown("---")
    
    # 使用表单防止重复提交
    with st.form(key="question_form", clear_on_submit=True):
        user_input = st.text_input(
            "请输入你的规则问题：",
            placeholder="例如：圣武士的至圣斩如何使用？",
            key="user_input"
        )
        
        col1, col2 = st.columns([6, 1])
        with col2:
            submit_button = st.form_submit_button("发送 📤", use_container_width=True)
    
    # 处理用户输入
    if submit_button and user_input.strip():
        question = user_input.strip()
        
        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # 显示用户消息
        display_message("user", question)
        
        # 生成回答
        with st.spinner("🤔 正在查询规则文档并生成回答..."):
            try:
                result = st.session_state.agent.invoke({"input": question})
                response = result.get("output", "无法获取回答")
                main_content, sources = format_sources(response)
                
                # 添加助手消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": main_content,
                    "sources": sources
                })
                
                # 显示回答
                display_message("assistant", main_content, sources)
                
            except Exception as e:
                st.error(f"❌ 生成回答时出错: {e}")
        
        st.rerun()
    
    # 底部信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "🎲 DND 规则助手 | 基于 Gemini AI | "
        "规则数据来源于 DND 5E 不全书"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
