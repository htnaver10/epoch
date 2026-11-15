@echo off
setlocal
if not exist "%~dp0.data-env\python.exe" (
  echo Native reader environment missing.
  exit /b 2
)
set "MAMBA_ROOT_PREFIX=%~dp0.cache\mamba"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
"%~dp0.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0.data-env" python -u "%~dp0tools\run_substantial_round.py" %*
set "ROUND_RESULT=%errorlevel%"
if "%~1"=="" pause
exit /b %ROUND_RESULT%
