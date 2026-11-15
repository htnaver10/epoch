@echo off
setlocal
set "MAMBA_ROOT_PREFIX=%~dp0.cache\mamba"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
"%~dp0.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0.data-env" python -u "%~dp0tools\run_teacher_student.py" %*
set "ROUND_RESULT=%errorlevel%"
if "%~1"=="" pause
exit /b %ROUND_RESULT%
