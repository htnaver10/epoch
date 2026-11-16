@echo off
setlocal
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if "%~1"=="" goto all
if /I "%~1"=="all" goto all
if /I "%~1"=="expand" goto expand
"%~dp0..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" -u "%~dp0wide_diverse_v1\run.py" %*
exit /b %errorlevel%
:expand
"%~dp0..\fire-harmonization-research-fullres\.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0..\fire-harmonization-research-fullres\.data-env" python -u "%~dp0wide_diverse_v1\expand_data.py"
exit /b %errorlevel%
:all
if exist "%~dp0wide_diverse_v1\data\manifests\extra_train.jsonl" goto prepared
"%~dp0..\fire-harmonization-research-fullres\.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0..\fire-harmonization-research-fullres\.data-env" python -u "%~dp0wide_diverse_v1\expand_data.py"
if errorlevel 1 exit /b %errorlevel%
:prepared
if exist "%~dp0wide_diverse_v1\results\expansion_receipt.json" goto checkaudit
goto finalize
:checkaudit
findstr /C:"reference_qa_audit_by_group" "%~dp0wide_diverse_v1\results\expansion_receipt.json" >nul
if not errorlevel 1 goto trainall
:finalize
"%~dp0..\fire-harmonization-research-fullres\.cache\micromamba\micromamba.exe" --no-rc run --prefix "%~dp0..\fire-harmonization-research-fullres\.data-env" python -u "%~dp0wide_diverse_v1\expand_data.py" --finalize-existing
if errorlevel 1 exit /b %errorlevel%
:trainall
"%~dp0..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" -u "%~dp0wide_diverse_v1\run.py" all
exit /b %errorlevel%
