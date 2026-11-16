@echo off
setlocal
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if "%~1"=="" goto all
if /I "%~1"=="all" goto all
if /I "%~1"=="expand" goto expand
"%~dp0..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" -u "%~dp0wide_diverse_v2\run.py" %*
exit /b %errorlevel%
:expand
"%~dp0..\fire-harmonization-research-fullres\.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0..\fire-harmonization-research-fullres\.data-env" python -u "%~dp0wide_diverse_v2\expand_data.py"
exit /b %errorlevel%
:all
if exist "%~dp0wide_diverse_v2\results\expansion_receipt.json" goto trainall
"%~dp0..\fire-harmonization-research-fullres\.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0..\fire-harmonization-research-fullres\.data-env" python -u "%~dp0wide_diverse_v2\expand_data.py"
if errorlevel 1 exit /b %errorlevel%
:trainall
"%~dp0..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" -u "%~dp0wide_diverse_v2\run.py" all
exit /b %errorlevel%
