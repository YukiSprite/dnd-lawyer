#!/usr/bin/env python3
"""
主启动脚本 - 规则 AI 助手 (Gemini 版本)
支持 PF (Pathfinder) 和 DND (Dungeons & Dragons) 两种规则版本

启动模式：
  python run.py          # 默认启动 Web UI（端口 6008）
  python run.py web      # 启动 Web UI
  python run.py web 8080 # 启动 Web UI（指定端口）
  python run.py console  # 启动命令行控制台

版本切换说明：
修改 config/settings.py 中的 CURRENT_VERSION 变量：
- "pf"  : 使用 Pathfinder 规则配置
- "dnd" : 使用 DND 规则配置
"""

import sys
import os
import subprocess

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def start_webui(port: int = 6008):
    """启动 Streamlit Web UI"""
    print("=" * 60)
    print("🎲 规则 AI 助手 - Web UI")
    print("=" * 60)
    print(f"端口: {port}")
    print(f"访问地址: http://localhost:{port}")
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    print()
    
    # 使用 subprocess 启动 streamlit
    webui_path = os.path.join(PROJECT_ROOT, "webui", "app.py")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", webui_path,
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\n\n👋 Web UI 已停止")


def start_console():
    """启动命令行控制台"""
    from tools.console_gemini import main
    main()


def print_help():
    """打印帮助信息"""
    print("""
规则 AI 助手 - 启动脚本

用法:
  python run.py [模式] [参数]

模式:
  web [端口]    启动 Web UI（默认端口 6008）
  console       启动命令行控制台
  help          显示此帮助信息

示例:
  python run.py              # 启动 Web UI（默认）
  python run.py web          # 启动 Web UI
  python run.py web 8080     # 启动 Web UI，端口 8080
  python run.py console      # 启动命令行控制台
""")


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if not args:
        # 默认启动 Web UI
        start_webui()
    elif args[0] in ["web", "webui", "ui"]:
        # 启动 Web UI
        port = int(args[1]) if len(args) > 1 else 6008
        start_webui(port)
    elif args[0] in ["console", "cli", "cmd"]:
        # 启动命令行控制台
        start_console()
    elif args[0] in ["help", "-h", "--help"]:
        print_help()
    else:
        print(f"未知参数: {args[0]}")
        print_help()
        sys.exit(1)
