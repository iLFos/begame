@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === eeggamev1.0 · 单独启动游戏服务器 ===
echo.
node server.js
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请确认已安装 Node.js 并加入 PATH。
    pause
)