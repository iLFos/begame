@echo off
chcp 65001 >nul
echo ========================================
echo   热血乌龙球 - 一键启动脚本
echo ========================================
echo.

REM 检查 Node.js 是否安装
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未检测到 Node.js！
    echo.
    echo 请先安装 Node.js: https://nodejs.org/
    echo 安装后重新运行此脚本。
    echo.
    pause
    exit /b 1
)

echo [1/3] 检查信号文件...
if not exist "movda.txt" (
    echo 1 > movda.txt
    echo      已创建 movda.txt (初始值: 1)
) else (
    echo      已存在 ✓
)

echo.
echo [2/3] 启动本地服务器...
echo.
echo ========================================
echo  服务器启动后，请打开浏览器访问:
echo  http://localhost:8080
echo ========================================
echo.
echo  控制信号: 修改 movda.txt 内容为 1 或 0
echo  按 Ctrl+C 停止服务器
echo ========================================
echo.

node server.js

echo.
echo 服务器已停止。
pause
