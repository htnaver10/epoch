@echo off
setlocal
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
if /I "%~1"=="train-tree" goto tree
if /I "%~1"=="predict-tree" goto tree
"%~dp0..\fire-harmonization-research-fullres\.venv\Scripts\python.exe" -u "%~dp0run.py" %*
goto finished
:tree
"%~dp0..\fire-harmonization-research\.venv\Scripts\python.exe" -u "%~dp0run.py" %*
:finished
set "TRAIN_RESULT=%errorlevel%"
if "%~1"=="" pause
exit /b %TRAIN_RESULT%
