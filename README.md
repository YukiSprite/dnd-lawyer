# DND汴京小助手 🎲

基于 Google Gemini 的 DND 5E 规则问答系统，支持官方 API 和 OpenAI 兼容代理服务。
还在为某个行为是否符合规则发愁吗？还在为不是律政先锋而苦恼吗？试试这款汴京小助手，帮你在面对群友超强超劲的疑似灰色构筑与脑洞大开的神奇操作时能有理有据地进行讨论

---

## 目录

- [快速开始](#-快速开始)
- [核心特性](#-核心特性)
- [API 配置](#-api-配置)
- [安装配置](#-安装配置)
- [使用指南](#-使用指南)
- [常见问题](#-常见问题)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)
- [To Do List](#to-do-list)
- [更新日志](#更新日志)
- [贡献](#贡献)
- [许可证](#许可证)

---

## ⚡ 快速开始

### 首次使用（3 分钟）

#### 步骤 1：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤 2：配置 API
将.env.example重命名为.env，
```bash
cp .env.example .env
```
然后按照如下步骤进行配置

```
# 选择 API 提供商
API_PROVIDER = "openai"  # 推荐使用 OpenAI 兼容代理
API_BASE_URL = "你的代理服务地址"  # 例如：https://api.your-proxy.com/v1
API_KEY = "你的API密钥"
```

#### 步骤 3：初始化数据

**选项 A：自动下载并构建（推荐）**
```bash
bash init.sh
```

**选项 B：使用已有的不全书文件**
```bash
# 1. 清空旧数据
bash clear.sh

# 2. 将你的不全书文件复制到 chm_source/ 目录

# 3. 构建索引（不重新下载）
bash noDownload.sh
```

#### 步骤 4：启动应用
```bash
python run.py
```

等待加载完成后，即可开始提问！

---

## ✨ 核心特性

### 🌐 灵活的 API 支持
- ✅ **Google 官方 API** - 直接使用 Google AI Studio 的 API（需科学上网）
- ✅ **OpenAI 兼容模式** - 支持任何 OpenAI 格式的第三方代理服务（推荐国内用户）
- 🔄 **一键切换** - 只需修改配置文件，无需改动代码

### 🧠 超大上下文窗口
- 利用现代大模型的 1M+ token 上下文窗口
- 一次性处理多个完整的规则文档
- 无需复杂的记忆管理和多轮检索

### 📊 表格完整保留
- HTML 表格保持原始格式不变
- 无 Markdown 转换导致的表格错乱
- 大模型直接理解 HTML 结构，保证准确性

### 🔍 智能父文档检索
- **检索阶段**：使用 500 字小块进行精确匹配
- **生成阶段**：返回 2000-5000 字的完整父文档
- **语义过滤**：可选的 embedding 模型进行相似度排序
- 平衡检索精度与上下文完整性

### 🌏 中文优化
- `chardet` 自动检测编码（GBK/UTF-8/GB2312）
- 专为中文 CHM 文件优化的解析器
- 保留中文特殊字符和排版

### 💰 低成本运行
- **无需本地 GPU**：推理完全使用云端 API
- **可选 GPU 加速**：Embedding 模型可使用 GPU 加速检索
- **灵活部署**：CPU 即可运行，降低硬件门槛

---

## 🔑 API 配置

### 配置文件位置

所有 API 配置集中在 `config/api_config.py` 文件中：

```python
# ============================================
# 🌐 API 提供商选择
# ============================================

# API 提供商类型：google (官方) 或 openai (兼容代理)
API_PROVIDER = "openai"

# API Base URL（仅 OpenAI 兼容模式需要填写）
API_BASE_URL = "https://api.your-proxy.com/v1"

# API Key
API_KEY = "YOUR_API_KEY"

# ============================================
# 🤖 模型配置
# ============================================

# Gemini 模型名称
MODEL_NAME = "gemini-2.5-pro"  # 或 gemini-1.5-flash, gemini-1.5-pro 等

# 温度参数（0-1，越低越确定性强）
TEMPERATURE = 0.1

# ============================================
# 🔍 Embedding 模型配置（用于语义检索）
# ============================================

# Embedding 模型名称（推荐中文模型）
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"

# Embedding 计算设备
EMBEDDING_DEVICE = "auto"  # "cuda", "mps", "cpu" 或 "auto"
```

### 模式 1：Google 官方 API（需科学上网）

适用于能直接访问 Google 服务的用户：

```python
API_PROVIDER = "google"
API_BASE_URL = ""  # 留空
API_KEY = "YOUR_API_KEY"
```

**获取 API Key**：访问 [Google AI Studio](https://aistudio.google.com/app/apikey)

### 模式 2：OpenAI 兼容代理（推荐国内用户）

适用于使用第三方 Gemini API 代理服务的用户：

```python
API_PROVIDER = "openai"
API_BASE_URL = "https://your-proxy.com/v1"  # 你的代理服务地址
API_KEY = "YOUR_API_KEY"  # 代理服务提供的 API Key
```

**优势**：
- ✅ 国内直连，无需科学上网
- ✅ 通常提供更稳定的服务质量
- ✅ 统一的 OpenAI 格式接口

### 验证配置

配置完成后，可以运行以下命令测试 API 连接：

```bash
python config/api_config.py
```

如果配置正确，你会看到：
```
✅ API 配置验证成功！
✅ LLM 测试通过
✅ Embedding 模型加载成功
```

---

## 🛠️ 安装配置

### 系统要求

| 项目 | 要求 | 说明 |
|------|------|------|
| **Python 版本** | 3.12 | 仅在3.12版本进行过测试，没有测试过其它版本 |
| **网络连接** | 必需 | 用于调用API |
| **磁盘空间** | ~4GB | 包含 Embedding 模型和向量库 |
| **内存** | 4GB+ | 8GB 更佳 |
| **GPU** | 可选 | 仅用于 Embedding 加速（非必需） |

### 依赖安装

```bash
pip install -r requirements.txt
```

**主要依赖**：
- `langchain` / `langchain-core` - LLM 应用框架
- `langchain-google-genai` - Google Gemini API 支持
- `langchain-openai` - OpenAI 兼容模式支持
- `langchain-huggingface` - Embedding 模型支持
- `chromadb` - 向量数据库
- `beautifulsoup4` - HTML 解析
- `chardet` - 编码检测

## 📖 使用指南

### 启动应用

```bash
python run.py
```

### 示例对话

```
请输入问题 >> 圣武士能否使用带有投掷词条的近战武器，投掷攻击敌人触发至圣斩？

助手 (正在思考...):
------------------------------------------------------------

#中间的调试输出略

根据提供的规则文档，**可以**。圣武士使用带有投掷词条的近战武器（例如标枪或手斧）投掷并命中敌人时，可以触发**至圣斩 (Divine Smite)**。

以下是详细的规则解释：

1.  **至圣斩的触发条件**：
    根据规则，至圣斩的施法时间为“附赠动作，当你使用**近战武器**或徒手打击命中一个生物后立即执行”。这里的关键在于“使用近战武器命中”，它规定了造成伤害的**物品类型**，而没有限定攻击的**方式**（即近战攻击或远程攻击）。
    *   **来源**：`Credits/第七章：法术/法术详述/1环/至圣斩｜Divine Smite`

2.  **投掷词条的定义**：
    规则中对“投掷 (Thrown)”词条的描述是：“如果一把武器具有投掷词条，你可以把它投掷出去来发动一次**远程攻击**。……如果投掷出去的武器是一把**近战武器**，则你在攻击检定和伤害掷骰中使用与你用其发动近战攻击时相同的属性调整值。”
    *   **来源**：`Credits/第六章：装备/武器/词条`

**结论**：
当你投掷一把具有“投掷”词条的近战武器（如标枪）时，你是在进行一次“远程攻击”，但你使用的武器本身依然被归类为“近战武器”。因此，这次攻击满足了至圣斩“使用近战武器命中”的触发条件。

**补充说明**：
魔法物品 **救世主的眷顾 (Redeemer's Regards)** 的描述也间接支持了这一结论。该物品是一把长弓（属于**远程武器**），其特殊效果是“任何以此武器发动的远程攻击，在使用至圣斩Divine Smite……时均视为**近战武器攻击**”。
这件物品的存在意义在于，它为本身不符合条件的**远程武器**创造了触发至圣斩的特例。而对于本身就是**近战武器**的投掷武器来说，则不需要这样的特例，因为它们天然就符合至圣斩的触发条件。
*   **来源**：`狮鹫的鞍中珍宝Ⅱ/魔法物品/武器目录/武器-珍稀/救世主的眷顾 Redeemer's Regards`

==================================================
📚 **参考的规则文档来源**：

1. Credits/第七章：法术/法术详述/1环/至圣斩｜Divine Smite
2. Credits/第七章：法术/法术详述/2环/魔化武器｜Magic Weapon
3. Credits/第七章：法术/法术详述/1环/强令对决｜Compelled Duel
4. Credits/第三章：角色职业/圣武士/荣耀之誓
5. Credits/第六章：装备/武器/词条
6. 狮鹫的鞍中珍宝Ⅱ/魔法物品/武器目录/武器-珍稀/救世主的眷顾 Redeemer's Regards
7. 简介/法术/法术详述/5环/神圣武器｜Holy Weapon
8. Credits/第三章：角色职业/圣武士
9. 前言/法术/5环/放逐斩｜Banishing Smite
10. 资源简写/法术速查/法术/疾风跃袭｜Wind Sprint
11. 前言/法术/5环/湮灭波｜Destructive Wave
12. 狮鹫的鞍中珍宝Ⅱ/魔法物品/武器目录/武器-极珍稀
13. 狮鹫的鞍中珍宝Ⅱ/魔法物品/武器目录/武器-珍稀/裂界鞘棍 Rift Sheath
14. 资源简写/法术速查/法术
15. 莫提的位面游记/印记城派系特工/和谐营·和平卫士

------------------------------------------------------------
```
示例对话2:
```
请输入问题 >> 法师在几级时能选择自己的奥术学派？
助手 (正在思考...):
------------------------------------------------------------
#调试输出略

根据提供的规则文档，法师在 **3级** 时选择他们的奥术学派（即法师子职）。

详细信息如下：

1.  **核心规则**：
    在《法师》职业描述的“法师特性”表格中，明确列出了3级时获得的职业特性是“法师子职”。
    > **来源路径**: Credits/第三章：角色职业/法师

    紧随其后的特性描述中也对此进行了说明：
    > **3级：法师子职 Wizard Subclass**
    > 你选择获得一项法师子职。防护师，预言师，塑能师以及幻术师。子职是一种特化，在特定的法师等级给予你对应的独特能力。
    > **来源路径**: Credits/第三章：角色职业/法师

2.  **学派概念**：
    文档中解释了“奥术学派”是法师子职（或称奥术传承）的基础。法师们将法术分为八大学派，而这些学派也构成了不同的研究传统和学院。
    > **来源路径**: 资源简写/第三章：角色职业/法师/咒法学派

3.  **子职示例**：
    提供的具体法师子职文档，如“塑能师”和“咒法师”，其第一个特性都是在3级时获得的，这也印证了法师在3级选择子职的规则。
    *   **塑能师**在3级获得“塑能学者”和“强力戏法”特性。
        > **来源路径**: Credits/第三章：角色职业/法师/塑能师
    *   **咒法师**在3级获得“王车易位”和“咒法学者”特性。
        > **来源路径**: UA/奥术子职II/法师-咒法师二期

**总结**：法师在达到 **3级** 时，可以选择一个法师子职，这通常是基于一个特定的魔法学派，例如塑能师或咒法师。

==================================================
📚 **参考的规则文档来源**：

1. Credits/第三章：角色职业/法师
2. 资源简写/第三章：角色职业/法师/咒法学派
3. Credits/第三章：角色职业/吟游诗人/逸闻学院
4. Credits/第三章：角色职业/法师/塑能师
5. 封面/角色选项/子职/法师（EGW）
6. 前言/职业/武僧/散打宗
7. 斯翠海文：混沌研习（无模组）/伙伴与敌人/一年级生
8. 斯翠海文：混沌研习（无模组）/伙伴与敌人/粹丽/粹丽学院
9. UA/奥术子职II/法师-咒法师二期
10. Credits/第二章：创建角色/兼职

------------------------------------------------------------
```
### 高级功能

#### 1. 版本优先级控制 🆕

**功能**：优先返回 2024/2025 版本的规则，降低 2014 版本的优先级。

**配置** (`config/config.py`)：

```python
# 启用路径加权
ENABLE_PATH_BOOSTING = True

# 路径加权规则：{路径关键词: 加权值}
PATH_BOOST_RULES = {
    # 新版本优先（正向加权）
    "2024": 0.15,              # 2024 版本 +0.15
    "2025": 0.12,              # 2025 版本 +0.12
    "玩家手册2024": 0.15,
    "城主指南2024": 0.15,
    "怪物图鉴2025": 0.12,
    
    # 旧版本降权（负向加权）
    "玩家手册/": -0.15,        # 2014 版玩家手册 -0.15
    "城主指南/": -0.15,        # 2014 版城主指南 -0.15
    "怪物图鉴/": -0.12,        # 2014 版怪物图鉴 -0.12
    
    # 其他优先来源
    "Credits": 0.1,            # 官方说明/鸣谢 +0.1
}
```

**效果**：
```
查询："法师的法术位如何计算？"

无优先级:
  玩家手册/法师 (相似度=0.85) 📜 → 排名第1
  玩家手册2024/法师 (相似度=0.82) 🆕 → 排名第2

启用优先级后:
  玩家手册2024/法师 (0.82+0.15=0.97) 🆕 → 排名第1 ✨
  玩家手册/法师 (0.85-0.15=0.70) 📜 → 排名第2
```


#### 2. 路径加权与排除

**功能**：支持正向加权（提升）和负向加权（降低），以及完全排除特定路径。

**说明**：Credits 路径包含官方说明、鸣谢、规则总览等重要内容。

#### 3. 启用语义相似度过滤

编辑 `config/config.py`：

```python
# 启用语义过滤（使用 embedding 模型进行二次筛选）
ENABLE_SEMANTIC_FILTER = True

# 过滤模式："rank" (排序) 或 "threshold" (阈值过滤)
SEMANTIC_FILTER_MODE = "rank"

# 相似度阈值（仅在 threshold 模式下使用）
SEMANTIC_SIMILARITY_THRESHOLD = 0.4
```

#### 4. 查看检索日志

控制台会实时显示：
- 检索到的文档数量
- 语义过滤结果（如果启用）
- 最终使用的文档列表
- 文档来源路径

---

## ❓ 常见问题

### Q1: API 连接失败？

**症状**：
```
ServiceUnavailable: 503 failed to connect...
或
Error: API request failed
```

**解决方案**：

1. **检查 API 配置**
   ```bash
   python config/api_config.py  # 运行配置验证
   ```

2. **验证配置项**
    配置项可以不写在api_config.py中，也可以写在你的.env文件中
   - `API_PROVIDER` 是否正确（"google" 或 "openai"）
   - `API_BASE_URL` 是否正确（OpenAI 模式必填）
   - `API_KEY` 是否有效

3. **网络问题**
   - Google 模式：确保可以访问 Google 服务
   - OpenAI 模式：测试代理服务是否可用

4. **查看详细日志**
   ```bash
   # 查看最近的日志
   tail -f logs/*.log
   ```

### Q2: 如何切换 API 提供商？

编辑 `config/api_config.py`，修改 `API_PROVIDER` 即可：

```python
# 方案 1：切换到 Google 官方
API_PROVIDER = "google"
API_BASE_URL = ""  # 留空
API_KEY = "YOUR_API_KEY"

# 方案 2：切换到 OpenAI 兼容代理
API_PROVIDER = "openai"
API_BASE_URL = "https://your-proxy.com/v1"
API_KEY = "YOUR_API_KEY"
```

保存后重启应用即可生效，**无需修改任何代码**。

### Q3: 索引构建失败？

**常见原因**：
- ❌ ChromaDB 批处理限制
- ❌ 内存不足
- ❌ 编码错误

**解决方案**：

```bash
# 1. 清空旧索引
rm -rf vector_store/*

# 2. 检查依赖
pip install -r requirements.txt

# 3. 重新构建索引
python scripts/build_index_gemini.py
```

**调整参数**（如果内存不足）：

编辑 `config/config.py`：
```python
# 减小分块大小
CHILD_CHUNK_SIZE = 300  # 默认 500
PARENT_CHUNK_SIZE = 1500  # 默认 2000
```

### Q4: 如何更新规则数据？

当有新版本的 DND 不全书时：

```bash
# 步骤 1：更新 CHM 源文件
# 将新的 CHM 文件复制到 chm_source/ 目录

# 步骤 2：重新生成 JSON 数据
python tools/package_json.py

# 步骤 3：重建索引
python scripts/build_index_gemini.py

# 步骤 4：验证
python run.py
```

### Q5: 为什么回答质量不稳定？

**可能原因与解决方案**：

1. **检索文档不够相关**
   - 启用语义过滤：`ENABLE_SEMANTIC_FILTER = True`
   - 增加检索数量：`PARENT_RETRIEVER_TOP_K = 10`

2. **模型温度过高**
   - 降低温度参数：`TEMPERATURE = 0.1`（更确定性）

3. **上下文太长导致截断**
   - 减少文档数量：`PARENT_RETRIEVER_MAX_K = 8`
   - 使用更大上下文的模型：`gemini-3-pro`

### Q6: 如何查看检索到了哪些文档？

在每次回答末尾，系统会自动显示：

```
==================================================
📚 **参考的规则文档来源**：

1. 玩家手册2024/职业/法师 Wizard/总览 Overview
2. 城主指南/魔法物品/魔法物品表格
...
```

你也可以在控制台日志中看到更详细的检索过程。


---

## 📁 项目结构

```
dnd-lawyer/
├── run.py                          # ⭐ 主启动脚本
├── init.sh                         # 🚀 一键初始化脚本（含下载）
├── noDownload.sh                   # 🔧 初始化脚本（不下载不全书）
├── clear.sh                        # 🗑️  清理脚本
├── requirements.txt                # 📦 Python 依赖
│
├── config/                         # ⚙️ 配置目录
│   ├── api_config.py               #   🔑 API 配置（重要！）
│   └── config.py                   #   📋 项目参数配置
│
├── src/                            # 🧠 核心源代码
│   ├── llm_gemini.py               #   💬 Gemini LLM 初始化
│   ├── agent_gemini.py             #   🤖 Agent 逻辑（问答协调）
│   ├── parent_retriever.py         #   🔍 父文档检索器
│   └── data_loader.py              #   📖 数据加载与分割
│
├── tools/                          # 🛠️ 工具脚本
│   ├── console_gemini.py           #   💻 交互控制台
│   ├── package_json.py             #   📝 数据生成工具
│   ├── analyze_chm.py              #   🔬 CHM 结构分析
│   └── check_system.py             #   ✅ 系统检查工具
│
├── scripts/                        # 📜 离线构建脚本
│   ├── build_index_gemini.py       #   🏗️  索引构建（Gemini 版本）
│   └── extract_chm.sh              #   📂 CHM 解压脚本
│
├── data/                           # 💾 规则数据
│   └── rules_data.json             #   📄 处理后的规则 JSON
│
├── vector_store/                   # 🗄️ 向量数据库
│   ├── chroma.sqlite3              #   数据库文件
│   ├── parent_docstore.pkl         #   父文档存储
│   └── [其他 ChromaDB 文件]
│
├── chm_source/                     # 📚 CHM 源文件
│   ├── xxx.chm                     #   原始 CHM 文件
│   └── extracted/                  #   解压后的文件
│
├── logs/                           # 📋 日志文件
└── models_cache/                   # 🤗 Embedding 模型缓存
```

### 核心模块说明

| 模块 | 功能 | 说明 |
|------|------|------|
| **config/api_config.py** | API 配置中心 | 统一管理 API Key、模型选择和 Embedding 配置(当前版本中，这个功能被集成到了/.env文件中) |
| **src/llm_gemini.py** | LLM 初始化 | 创建 Gemini LLM 实例（支持 Google 和 OpenAI 模式） |
| **src/agent_gemini.py** | Agent 逻辑 | 实现检索-生成流程，支持语义过滤和动态文档数量 |
| **src/parent_retriever.py** | 父文档检索器 | 小块检索 + 完整父文档返回策略 |
| **tools/console_gemini.py** | 交互控制台 | 启动应用的入口，提供命令行交互界面 |
| **scripts/build_index_gemini.py** | 索引构建 | 离线构建向量数据库（只需运行一次） |

### 数据流程

```
CHM 源文件 (chm_source/)
    ↓
[extract_chm.sh] 解压
    ↓
HTML 文件 (chm_source/extracted/)
    ↓
[package_json.py] 解析 + 清洗
    ↓
JSON 数据 (data/rules_data.json)
    ↓
[build_index_gemini.py] 分块 + 向量化
    ↓
向量数据库 (vector_store/)
    ↓
[run.py → console_gemini.py] 应用运行
    ↓
用户问答 🎯
```

---

## 🔧 技术栈

- **框架**：LangChain
- **向量数据库**：ChromaDB
- **Embedding**：BAAI/bge-base-zh-v1.5（中文语义向量）
- **语言**：Python 3.12

---

## To Do List
[] 支持更多规则
## 📝 更新日志

### v1.0.20251127
- ✅ 支持调用API（替代 Ollama）
- ✅ 支持 Google 官方和 OpenAI 兼容双模式
- ✅ 实现语义相似度过滤功能
- ✅ 优化父文档检索策略

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目仅供学习和个人使用。所使用的规则数据来源为https://github.com/DND5eChm/DND5e_chm

**免责声明**：
- DND 规则内容版权归 Wizards of the Coast 所有
---

<div align="center">

### ⭐ 如果这个项目对你有帮助，请给个星标！⭐

Made with ❤️ for DND 5E players and DMs

[⬆ 回到顶部](#dnd-规则-ai-助手-)

</div>
