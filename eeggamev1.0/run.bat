@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === eeggamev1.0 · 启动主程序 ===
echo.
python main.py
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请确认已安装 Python 3.9+ 与 requirements.txt 中的依赖。
    pause
)