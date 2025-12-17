echo ""
echo "[1/6] 下载最新 CHM 文件..."
echo "使用本地已有的 CHM 文件，跳过下载步骤。"

# 2. 解压 CHM 文件
echo ""
echo "[2/6] 解压 CHM 文件..."
# 检查是否安装了 libchm-bin
if ! command -v extract_chmLib &> /dev/null; then
    echo "⚠️  未找到 extract_chmLib，正在安装 libchm-bin..."
    sudo apt-get update && sudo apt-get install -y libchm-bin
fi

# 找到 CHM 文件并解压
CHM_FILE=$(find chm_source -maxdepth 1 -name "*.chm" | head -n 1)
if [ -z "$CHM_FILE" ]; then
    echo "❌ 错误: 未找到 CHM 文件"
    exit 1
fi

# 清理旧的解压目录和配置文件，确保使用最新的 CHM 文件
if [ -d "chm_source/extracted" ]; then
    echo "   清理旧的解压文件..."
    rm -rf chm_source/extracted
fi
if [ -f "config/data_processing.json" ]; then
    echo "   清理旧的配置文件..."
    rm -f config/data_processing.json
fi

echo "   解压: $CHM_FILE"
extract_chmLib "$CHM_FILE" chm_source/extracted/
echo "✓ CHM 文件解压完成"

# 3. 安装 Python 依赖
echo ""
echo "[3/6] 安装 Python 依赖..."
pip install -r requirements.txt

# 4. 分析 CHM 结构
echo ""
echo "[4/6] 分析 CHM 结构..."
python tools/analyze_chm.py

# 5. 生成规则数据
echo ""
echo "[5/6] 生成规则数据..."
python tools/package_json.py

# 6. 构建向量索引
echo ""
echo "[6/6] 构建向量索引"
python scripts/build_index_gemini.py

echo ""
echo "=========================================="
echo "✓ 初始化完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 编辑 config/api_config.py 配置你的 API Key"
echo "2. 运行: python run.py"
echo ""