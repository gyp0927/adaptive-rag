@echo off
chcp 65001 >nul
echo ========================================
echo  Adaptive RAG 启动脚本
echo ========================================
echo.

REM 确保数据目录存在
if not exist "data" mkdir data
if not exist "data\documents" mkdir data\documents
if not exist "data\qdrant_storage" mkdir data\qdrant_storage

echo [1/2] 数据目录检查完成
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保 Python 已安装并加入 PATH
    pause
    exit /b 1
)

echo [2/2] Python 版本:
python --version
echo.

echo ========================================
echo  正在启动服务...
echo  地址: http://localhost:8000
echo  API 文档: http://localhost:8000/docs
echo  健康检查: http://localhost:8000/health
echo ========================================
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 启动 uvicorn（开发模式，Windows 下 --reload 不稳定，手动重启即可）
python -m uvicorn adaptive_rag.api.main:app --host 127.0.0.1 --port 8000

pause
