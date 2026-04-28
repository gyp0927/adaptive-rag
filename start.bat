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

echo [1/4] 数据目录检查完成
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保 Python 已安装并加入 PATH
    pause
    exit /b 1
)

echo [2/4] Python 版本:
python --version
echo.

REM 检查 .env 是否存在
if not exist ".env" (
    echo [警告] 未找到 .env 文件，请复制 .env.example 并配置你的 API Key
    echo.
)

REM 检查 alembic 迁移
echo [3/4] 检查数据库迁移...
python -m alembic current >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行，执行数据库初始化...
    python -m alembic upgrade head
) else (
    echo [3/4] 数据库迁移检查完成
)
echo.

echo [4/4] 所有检查通过
echo.

echo ========================================
echo  正在启动服务...
echo.
echo  地址:           http://localhost:8000
echo  API 文档:       http://localhost:8000/docs
echo  前端界面:       http://localhost:8000
echo  健康检查:       http://localhost:8000/health
echo  就绪检查:       http://localhost:8000/ready
echo  Prometheus:     http://localhost:8000/metrics
echo.
echo  功能:
echo    - 文档上传:    POST /api/v1/documents/upload
echo    - 文本上传:    POST /api/v1/documents/text
echo    - 智能查询:    POST /api/v1/query
echo    - 系统统计:    GET  /api/v1/admin/stats
echo    - 手动迁移:    POST /api/v1/admin/migrate
echo.
echo  按 Ctrl+C 停止服务
echo ========================================
echo.

REM 启动 uvicorn（Windows 下 --reload 不稳定，手动重启即可）
python -m uvicorn adaptive_rag.api.main:app --host 127.0.0.1 --port 8000

pause
