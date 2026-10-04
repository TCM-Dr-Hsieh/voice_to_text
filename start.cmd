@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The Python environment .venv was not found. Double-click install.cmd first ^(see README.md^).
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py --open
if errorlevel 1 pause
