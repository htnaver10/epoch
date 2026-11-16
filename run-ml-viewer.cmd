@echo off
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" (
  echo Backend environment missing. Follow MLP-INTEGRATION.md to install it.
  pause
  exit /b 1
)
"backend\.venv\Scripts\python.exe" "backend\run_viewer.py" %*
if errorlevel 1 pause
