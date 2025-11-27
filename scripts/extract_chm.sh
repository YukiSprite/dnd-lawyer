#!/bin/bash
set -e

echo "=========================================="
echo "CHM 文件解压工具"
echo "=========================================="

# 检查是否安装了 libchm-bin
if ! command -v extract_chmLib &> /dev/null; then
    echo "⚠️  未找到 extract_chmLib，正在安装 libchm-bin..."
    sudo apt-get update && sudo apt-get install -y libchm-bin
fi

# 查找 CHM 文件
CHM_FILE=$(find chm_source -maxdepth 1 -name "*.chm" | head -n 1)

if [ -z "$CHM_FILE" ]; then
    echo "❌ 错误: 在 chm_source/ 目录中未找到 .chm 文件"
    echo ""
    echo "请确保 CHM 文件位于 chm_source/ 目录中"
    exit 1
fi

echo "找到 CHM 文件: $CHM_FILE"
echo ""
echo "正在解压..."

# 解压 CHM 文件到同一目录
extract_chmLib "$CHM_FILE" chm_source/

# 统计解压的文件
HTML_COUNT=$(find chm_source -type f \( -name "*.html" -o -name "*.htm" \) | wc -l)
echo ""
echo "✓ 解压完成！"
echo "  共解压 $HTML_COUNT 个 HTML 文件"
echo ""
echo "下一步："
echo "1. 运行: python tools/analyze_chm.py"
echo "2. 运行: python tools/package_json.py"
echo ""
