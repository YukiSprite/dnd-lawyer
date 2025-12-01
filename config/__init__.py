"""
Configuration package - 统一配置加载

配置文件已合并，根据 settings.py 中的 CURRENT_VERSION 自动选择对应的参数
"""

from .settings import CURRENT_VERSION, get_current_version, get_version_info, is_pf, is_dnd

# 统一导入配置（配置文件内部会根据版本选择不同的参数）
from .config import *
from . import config

# 导出版本控制函数
__all__ = [
    'config',
    'CURRENT_VERSION',
    'get_current_version',
    'get_version_info',
    'is_pf',
    'is_dnd',
]
