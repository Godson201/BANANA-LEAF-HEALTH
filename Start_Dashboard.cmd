@echo off
setlocal
echo Starting LeafLens at http://127.0.0.1:5050
echo Keep this window open. Press Ctrl+C to stop.
"%~dp0.venv\Scripts\python.exe" "%~dp0dashboard\app.py"
pause
