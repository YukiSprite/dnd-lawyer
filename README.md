# DND汴京小助手 🎲

基于 Google Gemini 的 DND 5E 规则问答系统，支持官方 API 和 OpenAI 兼容代理服务。现已兼容pathfinder 1e。
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
收集了一些被问到过的问题：
```markdown
请输入问题 >> 一个种族特性选择了取代了剑类训练的爪击的天狗，获得了两次爪抓天武攻击，那他能通过1级变形者的变形者利爪能力再“任意”长出两对爪击来进行四次天武爪抓攻击吗

助手 (正在思考...):
------------------------------------------------------------

中间的调试输出略

基于提供的规则文档，**不能**。

作为一名 1 级天狗变形者（Shifter），你**无法**通过叠加种族特性和职业能力来获得 4 次爪抓攻击。

以下是基于规则文档的详细解释：

### 1. 肢体限制原则
*   **规则来源**：文档 9（职业/极限荒野/变形者/Part 18）
*   **关键文本**：在“变形者利爪（Shifter Claws，Su）”的描述中明确指出：“**每只手**的爪抓攻击都是主要天生武器……如果她用一只利爪发动攻击的同时**另一只手**持握武器……”
*   **解析**：变形者利爪并非凭空长出新的肢体，而是将你现有的双手转化为利爪。
    *   **天狗种族特性**：你的“利爪”取代了“剑类训练”，使用的是你的**双手**。
    *   **变形者利爪**：使用的是你的**双手**。
    *   **冲突**：你只有一双手。在同一轮攻击中，同一个肢体通常只能作为天生武器进行一次攻击。这两种能力作用于相同的肢体部位，因此效果重叠，而不是数量叠加。

### 2. 效果重叠与取优
虽然攻击次数不能叠加（仍然是 2 次爪抓），但这并非没有好处。

*   **伤害骰取高**：
    *   天狗的种族爪抓通常造成 **1d3** 伤害（中型）。
    *   1 级变形者的利爪造成 **1d4** 伤害（中型）。
    *   你可以使用变形者利爪的伤害骰，因为它更高。
*   **特殊材质与DR穿透**：
    *   随着等级提升（如 3 级时），变形者利爪可以忽略特定的伤害减免（DR/寒铁、DR/魔法等），这是普通种族爪抓不具备的。

### 3. 唯一的例外可能性（不同部位）
*   **规则来源**：文档 9（边栏：替换天生攻击 / 荒野起源）
*   **解析**：如果你想获得超过 2 次的攻击，你不能选择“爪抓”，但可以利用《荒野起源》中的变体规则，将变形者利爪替换为**不需要双手**的攻击形式。
    *   规则提到：“每种替换的天生攻击都会取代其中一个化形者的爪抓攻击。”
    *   如果你选择的拟态（Aspect）支持（例如蝙蝠、熊、鳄鱼、狼等），你可以将变形者利爪的能力替换为**啮咬（Bite）**。
    *   **理论构建**：在这种情况下，你可以使用天狗种族的 2 次爪抓（手部），同时使用变形者职业能力生成 1 次啮咬（头部）。这将给你带来 **3 次天生攻击**（2 爪抓 + 1 啮咬），而不是 4 次爪抓。

### 总结
你无法获得 4 次爪抓攻击，因为你只有两只手。在 1 级时，你拥有 **2 次爪抓攻击**，你可以选择使用变形者利爪的数据（1d4 伤害）来替代种族数据（1d3 伤害）进行攻击。

==================================================
📚 **参考的规则文档来源**：

1. 规则/Unchained规则/游戏进行/移除多次攻击
2. 未整理/炼狱血脉/魔裔能力变体/Part1
3. 职业/极限荒野（Ultimate Wilderness）/变形者/职业变体
4. 种族/种族构建/种族特性/Part35
5. 官方FAQ/极限魔法(Ultimate Magic)
6. 未整理/荒野源始WO/选项/专长
7. 官方FAQ/极限荒野(Ulimate Wildness)
8. 专长/ARG 进阶种族手册/Part78
9. 职业/极限荒野（Ultimate Wilderness）/变形者/Part 18
10. 未整理/极限荒野UW/专长/Part4
```
示例对话2:
```markdown
请输入问题 >> 顺势斩能和要害打击一起吗
助手 (正在思考...):
------------------------------------------------------------
调试输出略

基于 Pathfinder 规则文档，**顺势斩（Cleave）不能和要害打击（Vital Strike）一起使用**。

以下是详细的规则解释：

### 1. 动作类型的冲突
根据 **《未整理/近战战术工具箱/规则详解》** 中的定义，这两个专长属于不同的动作类别，无法兼容：

*   **要害打击（Vital Strike）**：必须配合**攻击动作（Attack Action）**使用。攻击动作是标准动作的一种特定子类。
*   **顺势斩（Cleave）**：本身是一个特定的**标准动作（Standard Action）**。

### 2. 规则明确禁止
在 **《未整理/近战战术工具箱/规则详解》** 文档中，关于“标准动作”的章节有明确的文字说明，直接回答了你的问题：

> **标准动作 Standard Action：** 一些选择（比如**顺势斩**）是标准动作，允许你进行一次攻击，这并不视为攻击动作。这些选择无法与其它标准动作合并，**也无法与只影响攻击动作的选择合并（比如要害打击）**。

### 总结
当你使用“顺势斩”时，你正在进行一个特定的标准动作，而不是通用的“攻击动作”。因此，你无法获得那些必须依附于“攻击动作”才能生效的专长（如要害打击）的好处。

==================================================
📚 **参考的规则文档来源**：

1. 未整理/武器大师手册/专长/武器掌握
2. 专长/流派专长一览/专长详述/Part 42
3. 魔法物品/魔法物品/魔法武器防具/武器附魔/Part49
4. 未整理/近战战术工具箱/规则详解
5. 专长/ARG 进阶种族手册/Part48
6. 专长/全专长列表/Part4
7. 职业/进阶职业/PoP进阶之路/义洛理的勇士/Part3
8. 官方FAQ/CRB(战斗)
9. 官方FAQ/CRB(专长与技能、物品)/Part 1
10. 专长/UC 极限战斗/Part 117

------------------------------------------------------------
```
### 高级功能

#### 1. 版本优先级控制 🆕

**功能**：优先查询 CRB与FAQ中的内容。

**配置** (`config/config.py`)：

```python
PATH_BOOST_RULES = {
    # 优先规则（正向加权）
    "CRB":0.1,
    "FAQ":0.1
    # 降权规则（负向加权）
    # 下面是dnd的例子，pf我还没想好要用哪些
    # 注意：降权值不要太大，否则可能完全过滤掉这些文档
    # "玩家手册/": -0.15,       # 2014 版玩家手册 -0.15（与 2024 版抵消）
    # "城主指南/": -0.15,       # 2014 版城主指南 -0.15
    # "怪物图鉴/": -0.12,       # 2014 版怪物图鉴 -0.12
}
```

#### 2. 路径加权与排除

**功能**：支持正向加权（提升）和负向加权（降低），以及完全排除特定路径。

**说明**：CRB和FAQ 路径和规则最相关


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
[☑️] 支持更多规则，本版本为pathfinder 1e版本的规则助手
## 📝 更新日志

### v1.0.20251127
- ☑️ 支持调用API（替代 Ollama）
- ☑️ 支持 Google 官方和 OpenAI 兼容双模式
- ☑️ 实现语义相似度过滤功能
- ☑️ 优化父文档检索策略

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目仅供学习和个人使用。所使用的规则数据来源为[纯美苹果园的pf不全书](https://www.goddessfantasy.net/bbs/index.php?topic=141119.0)

**免责声明**：
- PF 规则内容版权归原作者所有
---

<div align="center">

### ⭐ 如果这个项目对你有帮助，请给个星标！⭐

Made with ❤️ for pathfinder 1e players and DMs

[⬆ 回到顶部](#dnd-规则-ai-助手-)

</div>
