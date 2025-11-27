#!/usr/bin/env python3
"""
主启动脚本 - DND 规则 AI 助手 (Gemini 版本)
从项目根目录方便地启动控制台程序
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行控制台
from tools.console_gemini import main

if __name__ == "__main__":
    main()
