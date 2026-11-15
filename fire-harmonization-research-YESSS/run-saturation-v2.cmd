@echo off
setlocal
cd /d "%~dp0"
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
"..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" "saturation_aware_v2\run.py" %*
set "result=%errorlevel%"
if "%~1"=="" pause
exit /b %result%
