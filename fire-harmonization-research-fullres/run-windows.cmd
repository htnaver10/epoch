@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Training environment missing. See WINDOWS_TRAINING_READINESS.md.
  exit /b 2
)
if "%~1"=="" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" --help
) else (
  "%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
)
exit /b %errorlevel%
