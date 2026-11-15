@echo off
setlocal
if not exist "%~dp0data\manifests\real_training_pilot_v1_stats.json" (
  echo The combined real pilot is not prepared yet. See WINDOWS_TRAINING_READINESS.md.
  exit /b 2
)
set "RESUME_PILOT="
if exist "%~dp0models\real_training_pilot_v1\last.pt" set "RESUME_PILOT=--resume"
echo Training the real NASA all-land pilot: 54 training tiles, 71 validation tiles.
echo Keep this window open. A summary appears after each completed epoch.
call "%~dp0run-windows.cmd" train --manifest data/manifests/real_training_pilot_v1.jsonl --stats data/manifests/real_training_pilot_v1_stats.json --config configs/train_rtx5070.json --run real_training_pilot_v1 --epochs 2 %RESUME_PILOT% %*
set "TRAINING_RESULT=%errorlevel%"
if "%~1"=="" pause
exit /b %TRAINING_RESULT%
