"""
版本设置文件 - 用于切换 PF (Pathfinder) 和 DND (Dungeons & Dragons) 规则版本

使用方法：
修改 CURRENT_VERSION 变量来切换版本：
- "pf"  : 使用 Pathfinder 规则配置
- "dnd" : 使用 DND 规则配置
"""

# ============================================
# 🎮 版本设置
# ============================================

# 当前使用的规则版本
# 可选值: "pf" (Pathfinder) 或 "dnd" (Dungeons & Dragons)
CURRENT_VERSION = "pf"

# ============================================
# 版本信息
# ============================================

VERSION_INFO = {
    "pf": {
        "name": "Pathfinder",
        "description": "Pathfinder 规则系统",
        "config_module": "config.config",
        "agent_module": "src.agent_gemini",
        "agent_template_name": "Pathfinder",
    },
    "dnd": {
        "name": "Dungeons & Dragons",
        "description": "DND 规则系统",
        "config_module": "config.config",
        "agent_module": "src.agent_gemini",
        "agent_template_name": "DND",
    }
}

def get_current_version() -> str:
    """获取当前版本"""
    return CURRENT_VERSION

def get_version_info() -> dict:
    """获取当前版本的详细信息"""
    return VERSION_INFO.get(CURRENT_VERSION, VERSION_INFO["pf"])

def set_version(version: str) -> bool:
    """
    设置当前版本（运行时修改，仅在当前会话有效）
    
    Args:
        version: "pf" 或 "dnd"
        
    Returns:
        bool: 是否设置成功
    """
    global CURRENT_VERSION
    if version in VERSION_INFO:
        CURRENT_VERSION = version
        return True
    return False

def is_pf() -> bool:
    """检查当前是否是 Pathfinder 版本"""
    return CURRENT_VERSION == "pf"

def is_dnd() -> bool:
    """检查当前是否是 DND 版本"""
    return CURRENT_VERSION == "dnd"
