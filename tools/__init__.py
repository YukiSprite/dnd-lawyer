"""
Tools package - 统一模块加载

package_json 已合并，内部根据 config.settings 中的 CURRENT_VERSION 自动选择处理逻辑
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CURRENT_VERSION, is_pf, is_dnd

# 统一导入 package_json（内部会根据版本选择不同的处理逻辑）
from tools import package_json

def get_package_json_module():
    """
    获取 package_json 模块（为了向后兼容保留此函数）
    
    Returns:
        module: package_json 模块
    """
    return package_json

__all__ = [
    'package_json',
    'get_package_json_module',
    'is_pf',
    'is_dnd',
]
