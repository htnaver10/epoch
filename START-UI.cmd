@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8080/ in your browser.
echo Keep this window open while using EPOCH. Press Ctrl+C to stop.
python -m http.server 8080 --bind 127.0.0.1 --directory ready-to-run
if errorlevel 1 pause
