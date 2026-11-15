@echo off
setlocal
if not exist "%~dp0.data-env\python.exe" (
  echo Native-reader environment missing. See WINDOWS_TRAINING_READINESS.md.
  exit /b 2
)
set "MAMBA_ROOT_PREFIX=%~dp0.cache\mamba"
"%~dp0.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0.data-env" python "%~dp0run.py" %*
exit /b %errorlevel%
